#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba_research.common.v3_utils import (
    append_stem_suffix,
    resolve_path,
    save_json,
)
from keiba_research.features.registry import (
    STACKER_TASK_CHOICES,
    get_stacker_feature_columns,
)
from keiba_research.training.binary_common import (
    coerce_feature_matrix,
    compute_binary_metrics,
    fold_integrity,
    prepare_binary_frame,
)
from keiba_research.training.cv_policy import (
    DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    attach_cv_policy_columns,
    build_capped_expanding_year_folds,
    build_cv_policy_payload,
    make_capped_expanding_window_definition,
    select_recent_window_years,
)
from keiba_research.training.stacker_common import (
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HOLDOUT_YEAR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_SEED,
    STACKER_PARAM_CLI_FLAGS,
    _default_base_holdout_paths,
    _default_base_oof_paths,
    _feature_manifest_payload,
    _fit_lgbm_final,
    _fit_lgbm_fold,
    _label_col,
    _load_valid_years,
    _merge_prediction_features,
    _meta_payload,
    _oof_frame,
    _pred_col,
    _predict_lgbm,
    _prediction_paths,
    _required_pred_cols,
    _save_lgbm_model,
    _summary,
)

logger = logging.getLogger(__name__)


def _meta_code_hash_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path(resolve_path("src/keiba_research/training/stacker_common.py")),
        Path(resolve_path("src/keiba_research/features/registry.py")),
        Path(resolve_path("src/keiba_research/training/cv_policy.py")),
    ]


def parse_args(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    task = str(default_task or "win")
    oof_defaults = _default_base_oof_paths(task)
    holdout_defaults = _default_base_holdout_paths(task)

    parser = argparse.ArgumentParser(
        description=(
            "Train strict temporal v3 stacker with capped expanding yearly CV "
            f"(min{DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS}, "
            f"max{DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS} by default)."
        )
    )
    parser.add_argument("--task", choices=list(STACKER_TASK_CHOICES), default=task)
    parser.add_argument(
        "--params-json",
        default="",
        help=(
            "Optional JSON file to set default stack hyperparameters. "
            "If omitted, automatically loads data/oof/stack_v3_{task}_best_params.json "
            "when present. Explicit CLI flags take precedence."
        ),
    )
    parser.add_argument(
        "--disable-default-params-json",
        action="store_true",
        help="Disable implicit fallback to data/oof/stack_v3_{task}_best_params.json.",
    )
    parser.add_argument("--features-input", default="data/features_v3.parquet")
    parser.add_argument("--holdout-input", default="")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    )
    parser.add_argument(
        "--max-train-years",
        type=int,
        default=DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    )
    parser.add_argument("--lgbm-oof", default=oof_defaults["lgbm"])
    parser.add_argument("--xgb-oof", default=oof_defaults["xgb"])
    parser.add_argument("--cat-oof", default=oof_defaults["cat"])
    parser.add_argument("--lgbm-holdout", default=holdout_defaults["lgbm"])
    parser.add_argument("--xgb-holdout", default=holdout_defaults["xgb"])
    parser.add_argument("--cat-holdout", default=holdout_defaults["cat"])
    parser.add_argument("--oof-output", default="")
    parser.add_argument("--holdout-output", default="")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--all-years-model-output", default="")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--feature-manifest-output", default="")
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix appended to default artifact/output filenames.",
    )
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-freq", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv_list)
    _apply_default_prediction_paths(args, argv_list)
    return args


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    for token in argv:
        if token == flag or token.startswith(prefix):
            return True
    return False


def _default_params_json_path(task: str) -> Path:
    return resolve_path(f"data/oof/stack_v3_{task}_best_params.json")


def _resolve_params_json_path(args: argparse.Namespace) -> Path | None:
    if str(args.params_json).strip():
        params_path = resolve_path(str(args.params_json))
        if not params_path.exists():
            raise SystemExit(f"params-json not found: {params_path}")
        return params_path

    if bool(getattr(args, "disable_default_params_json", False)):
        return None

    default_path = _default_params_json_path(str(args.task))
    if default_path.exists():
        return default_path
    return None


def _apply_params_json(
    args: argparse.Namespace,
    params: dict[str, Any],
    *,
    argv: list[str],
    params_path: Path,
) -> None:
    if not hasattr(args, "_tuned_final_iterations"):
        args._tuned_final_iterations = None  # type: ignore[attr-defined]
    if not hasattr(args, "_applied_params_json"):
        args._applied_params_json = None  # type: ignore[attr-defined]

    params_task = params.get("task")
    if params_task is not None and str(params_task) != str(args.task):
        raise SystemExit(f"params-json task mismatch: expected={args.task} actual={params_task}")

    if "min_train_years" in params and not _argv_has_flag(argv, "--min-train-years"):
        args.min_train_years = int(params["min_train_years"])
    if "max_train_years" in params and not _argv_has_flag(argv, "--max-train-years"):
        args.max_train_years = int(params["max_train_years"])

    model_params = params.get("lgbm_params", {})
    if model_params is not None and not isinstance(model_params, dict):
        raise SystemExit("params-json field lgbm_params must be an object.")
    for key, flag in STACKER_PARAM_CLI_FLAGS.items():
        if key not in model_params or _argv_has_flag(argv, flag):
            continue
        setattr(args, key, model_params[key])

    if "final_num_boost_round" in params and not _argv_has_flag(argv, "--num-boost-round"):
        args._tuned_final_iterations = int(params["final_num_boost_round"])  # type: ignore[attr-defined]

    args._applied_params_json = str(params_path)  # type: ignore[attr-defined]


def _binary_oof_path(task: str, model: str, artifact_suffix: str) -> str:
    prefix = "win" if str(task) == "win" else "place"
    return append_stem_suffix(f"data/oof/{prefix}_{model}_oof.parquet", artifact_suffix)


def _binary_holdout_path(task: str, model: str, holdout_year: int, artifact_suffix: str) -> str:
    prefix = "win" if str(task) == "win" else "place"
    return append_stem_suffix(
        f"data/holdout/{prefix}_{model}_holdout_{int(holdout_year)}_pred_v3.parquet",
        artifact_suffix,
    )


def _apply_default_prediction_paths(args: argparse.Namespace, argv: list[str]) -> None:
    defaults = {
        "lgbm_oof": _binary_oof_path(args.task, "lgbm", str(args.artifact_suffix)),
        "xgb_oof": _binary_oof_path(args.task, "xgb", str(args.artifact_suffix)),
        "cat_oof": _binary_oof_path(args.task, "cat", str(args.artifact_suffix)),
        "lgbm_holdout": _binary_holdout_path(
            args.task,
            "lgbm",
            int(args.holdout_year),
            str(args.artifact_suffix),
        ),
        "xgb_holdout": _binary_holdout_path(
            args.task,
            "xgb",
            int(args.holdout_year),
            str(args.artifact_suffix),
        ),
        "cat_holdout": _binary_holdout_path(
            args.task,
            "cat",
            int(args.holdout_year),
            str(args.artifact_suffix),
        ),
    }
    flag_map = {
        "lgbm_oof": "--lgbm-oof",
        "xgb_oof": "--xgb-oof",
        "cat_oof": "--cat-oof",
        "lgbm_holdout": "--lgbm-holdout",
        "xgb_holdout": "--xgb-holdout",
        "cat_holdout": "--cat-holdout",
    }
    for attr, default_value in defaults.items():
        if _argv_has_flag(argv, flag_map[attr]):
            continue
        setattr(args, attr, default_value)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.holdout_year) <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if int(args.min_train_years) <= 0:
        raise SystemExit("--min-train-years must be > 0")
    if int(args.max_train_years) < int(args.min_train_years):
        raise SystemExit("--max-train-years must be >= --min-train-years")
    if int(args.num_boost_round) <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if int(args.early_stopping_rounds) <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if int(args.n_bins) <= 1:
        raise SystemExit("--n-bins must be > 1")


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    task = str(args.task)
    defaults = {
        "oof": append_stem_suffix(f"data/oof/{task}_stack_oof.parquet", args.artifact_suffix),
        "holdout": append_stem_suffix(
            f"data/holdout/{task}_stack_holdout_{int(args.holdout_year)}_pred_v3.parquet",
            args.artifact_suffix,
        ),
        "metrics": append_stem_suffix(
            f"data/oof/{task}_stack_cv_metrics.json",
            args.artifact_suffix,
        ),
        "model": append_stem_suffix(f"models/{task}_stack_v3.txt", args.artifact_suffix),
        "all_years_model": append_stem_suffix(
            f"models/{task}_stack_all_years_v3.txt",
            args.artifact_suffix,
        ),
        "meta": append_stem_suffix(
            f"models/{task}_stack_bundle_meta_v3.json",
            args.artifact_suffix,
        ),
        "feature_manifest": append_stem_suffix(
            f"models/{task}_stack_feature_manifest_v3.json",
            args.artifact_suffix,
        ),
    }
    return {
        "oof": resolve_path(args.oof_output or defaults["oof"]),
        "holdout": resolve_path(args.holdout_output or defaults["holdout"]),
        "metrics": resolve_path(args.metrics_output or defaults["metrics"]),
        "model": resolve_path(args.model_output or defaults["model"]),
        "all_years_model": resolve_path(args.all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(args.meta_output or defaults["meta"]),
        "feature_manifest": resolve_path(
            args.feature_manifest_output or defaults["feature_manifest"]
        ),
    }


def run_stacker_training(
    *,
    task: str = "win",
    features_input: str = "data/features_v3.parquet",
    holdout_input: str | None = None,
    lgbm_oof: str = "",
    xgb_oof: str = "",
    cat_oof: str = "",
    lgbm_holdout: str | None = None,
    xgb_holdout: str | None = None,
    cat_holdout: str | None = None,
    oof_output: str = "",
    holdout_output: str | None = None,
    metrics_output: str = "",
    model_output: str = "",
    all_years_model_output: str | None = None,
    meta_output: str | None = None,
    feature_manifest_output: str | None = None,
    holdout_year: int = DEFAULT_HOLDOUT_YEAR,
    min_train_years: int | None = None,
    max_train_years: int | None = None,
    log_level: str = "INFO",
    disable_default_params_json: bool = False,
    params_json: str | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    min_data_in_leaf: int | None = None,
    lambda_l1: float | None = None,
    lambda_l2: float | None = None,
    feature_fraction: float | None = None,
    bagging_fraction: float | None = None,
    bagging_freq: int | None = None,
    num_boost_round: int | None = None,
) -> int:
    argv_list: list[str] = [
        "--task", str(task),
        "--features-input", str(features_input),
        "--holdout-year", str(int(holdout_year)),
    ]
    if min_train_years is not None:
        argv_list.extend(["--min-train-years", str(int(min_train_years))])
    if max_train_years is not None:
        argv_list.extend(["--max-train-years", str(int(max_train_years))])
    if holdout_input:
        argv_list.extend(["--holdout-input", str(holdout_input)])
    if lgbm_oof:
        argv_list.extend(["--lgbm-oof", str(lgbm_oof)])
    if xgb_oof:
        argv_list.extend(["--xgb-oof", str(xgb_oof)])
    if cat_oof:
        argv_list.extend(["--cat-oof", str(cat_oof)])
    if lgbm_holdout:
        argv_list.extend(["--lgbm-holdout", str(lgbm_holdout)])
    if xgb_holdout:
        argv_list.extend(["--xgb-holdout", str(xgb_holdout)])
    if cat_holdout:
        argv_list.extend(["--cat-holdout", str(cat_holdout)])
    if oof_output:
        argv_list.extend(["--oof-output", str(oof_output)])
    if holdout_output:
        argv_list.extend(["--holdout-output", str(holdout_output)])
    if metrics_output:
        argv_list.extend(["--metrics-output", str(metrics_output)])
    if model_output:
        argv_list.extend(["--model-output", str(model_output)])
    if all_years_model_output:
        argv_list.extend(["--all-years-model-output", str(all_years_model_output)])
    if meta_output:
        argv_list.extend(["--meta-output", str(meta_output)])
    if feature_manifest_output:
        argv_list.extend(["--feature-manifest-output", str(feature_manifest_output)])
    if disable_default_params_json:
        argv_list.append("--disable-default-params-json")
    if params_json:
        argv_list.extend(["--params-json", str(params_json)])
    if learning_rate is not None:
        argv_list.extend(["--learning-rate", str(float(learning_rate))])
    if num_leaves is not None:
        argv_list.extend(["--num-leaves", str(int(num_leaves))])
    if min_data_in_leaf is not None:
        argv_list.extend(["--min-data-in-leaf", str(int(min_data_in_leaf))])
    if lambda_l1 is not None:
        argv_list.extend(["--lambda-l1", str(float(lambda_l1))])
    if lambda_l2 is not None:
        argv_list.extend(["--lambda-l2", str(float(lambda_l2))])
    if feature_fraction is not None:
        argv_list.extend(["--feature-fraction", str(float(feature_fraction))])
    if bagging_fraction is not None:
        argv_list.extend(["--bagging-fraction", str(float(bagging_fraction))])
    if bagging_freq is not None:
        argv_list.extend(["--bagging-freq", str(int(bagging_freq))])
    if num_boost_round is not None:
        argv_list.extend(["--num-boost-round", str(int(num_boost_round))])
    argv_list.extend(["--log-level", str(log_level)])

    args = parse_args(argv_list)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    args._tuned_final_iterations = None  # type: ignore[attr-defined]
    args._applied_params_json = None  # type: ignore[attr-defined]
    if params_json:
        params_path = resolve_path(str(params_json))
        if not params_path.exists():
            raise SystemExit(f"params-json not found: {params_path}")
    else:
        params_path = _resolve_params_json_path(args)
    if params_path is not None:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise SystemExit("params-json must be a JSON object.")
        _apply_params_json(args, params, argv=argv_list, params_path=params_path)
        logger.info("applied params-json=%s", getattr(args, "_applied_params_json", None))
    _validate_args(args)

    label_col = _label_col(str(args.task))
    pred_col = _pred_col(str(args.task))
    required_pred_cols = _required_pred_cols(str(args.task))
    outputs = _resolve_output_paths(args)
    window_definition = make_capped_expanding_window_definition(
        int(args.min_train_years),
        int(args.max_train_years),
    )

    features_path = resolve_path(args.features_input)
    if not features_path.exists():
        raise SystemExit(f"features input not found: {features_path}")
    raw = pd.read_parquet(features_path)
    base = prepare_binary_frame(raw, label_col=label_col)
    merged = _merge_prediction_features(base, _prediction_paths(args, holdout=False))
    feature_cols = get_stacker_feature_columns(merged, task=str(args.task))
    base_valid_years = sorted(
        set.intersection(
            *[
                set(_load_valid_years(path))
                for path in _prediction_paths(args, holdout=False).values()
            ]
        )
    )

    eligible = merged.copy()
    for col in required_pred_cols:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    eligible = eligible[eligible["year"] < int(args.holdout_year)].copy()
    eligible = eligible[eligible[required_pred_cols].notna().all(axis=1)].copy()
    if eligible.empty:
        raise SystemExit("No eligible rows for stacker training after OOF filtering.")

    years = sorted(eligible["year"].unique().tolist())
    folds = build_capped_expanding_year_folds(
        years,
        min_window_years=int(args.min_train_years),
        max_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        raise SystemExit(
            "No stacker folds available under capped_expanding "
            f"(available_years={years}, min={args.min_train_years}, max={args.max_train_years})"
        )

    oof_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold in folds:
        train_df = eligible[eligible["year"].isin(fold.train_years)].copy()
        valid_df = eligible[eligible["year"] == fold.valid_year].copy()
        if train_df.empty or valid_df.empty:
            raise SystemExit(
                f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
            )
        fold_integrity(train_df, valid_df, int(fold.valid_year))

        x_train = coerce_feature_matrix(train_df, feature_cols)
        y_train = train_df[label_col].astype(int)
        x_valid = coerce_feature_matrix(valid_df, feature_cols)
        y_valid = valid_df[label_col].astype(int)

        _, pred_valid, best_iteration = _fit_lgbm_fold(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=int(args.seed) + int(fold.fold_id),
        )
        best_iterations.append(int(best_iteration))

        metrics = compute_binary_metrics(
            y_valid.to_numpy(dtype=int),
            pred_valid,
            n_bins=int(args.n_bins),
        )
        fold_metrics.append(
            {
                "fold_id": int(fold.fold_id),
                "train_years": list(map(int, fold.train_years)),
                "valid_year": int(fold.valid_year),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_races": int(train_df["race_id"].nunique()),
                "valid_races": int(valid_df["race_id"].nunique()),
                "best_iteration": int(best_iteration),
                "logloss": metrics["logloss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "ece": metrics["ece"],
                "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
                "min_train_window_years": int(args.min_train_years),
                "train_window_years": int(args.max_train_years),
                "holdout_year": int(args.holdout_year),
                "window_definition": window_definition,
            }
        )
        oof_frames.append(
            _oof_frame(
                valid_df,
                label_col=label_col,
                pred_col=pred_col,
                pred_values=pred_valid,
                fold_id=int(fold.fold_id),
                valid_year=int(fold.valid_year),
                args=args,
                window_definition=window_definition,
            )
        )

    oof = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(
        ["race_id", "horse_no"],
        kind="mergesort",
    )
    outputs["oof"].parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(outputs["oof"], index=False)

    cv_best_iteration_median = (
        int(round(float(np.median(best_iterations))))
        if best_iterations
        else int(args.num_boost_round)
    )
    cv_best_iteration_median = max(1, min(cv_best_iteration_median, int(args.num_boost_round)))
    final_iterations = cv_best_iteration_median
    final_iteration_source = "cv_best_iteration_median"
    tuned_final_iterations = getattr(args, "_tuned_final_iterations", None)
    if tuned_final_iterations is not None:
        final_iterations = max(1, min(int(tuned_final_iterations), int(args.num_boost_round)))
        final_iteration_source = "params_json"
    recent_years = select_recent_window_years(
        years,
        train_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
    )
    recent_df = eligible[eligible["year"].isin(recent_years)].copy()
    all_df = eligible.copy()

    main_model = _fit_lgbm_final(
        x_train=coerce_feature_matrix(recent_df, feature_cols),
        y_train=recent_df[label_col].astype(int),
        args=args,
        seed=int(args.seed) + 1000,
        n_estimators=int(final_iterations),
    )
    all_years_model = _fit_lgbm_final(
        x_train=coerce_feature_matrix(all_df, feature_cols),
        y_train=all_df[label_col].astype(int),
        args=args,
        seed=int(args.seed) + 2000,
        n_estimators=int(final_iterations),
    )
    _save_lgbm_model(main_model, outputs["model"])
    _save_lgbm_model(all_years_model, outputs["all_years_model"])

    holdout_rows = 0
    holdout_races = 0
    holdout_input = resolve_path(args.holdout_input) if args.holdout_input else None
    if holdout_input is not None and holdout_input.exists():
        holdout_raw = pd.read_parquet(holdout_input)
        if label_col not in holdout_raw.columns:
            holdout_raw[label_col] = 0
        holdout_base = prepare_binary_frame(holdout_raw, label_col=label_col)
        holdout_merged = _merge_prediction_features(
            holdout_base,
            _prediction_paths(args, holdout=True),
        )
        holdout_merged = holdout_merged[holdout_merged["year"] >= int(args.holdout_year)].copy()
        holdout_x = coerce_feature_matrix(holdout_merged, feature_cols)
        holdout_pred = holdout_merged[
            [
                c
                for c in [
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "race_date",
                    "field_size",
                    label_col,
                ]
                if c in holdout_merged.columns
            ]
        ].copy()
        holdout_pred[pred_col] = _predict_lgbm(
            main_model,
            holdout_x,
            num_iteration=int(final_iterations),
        )
        holdout_pred["valid_year"] = holdout_merged["year"].astype(int).to_numpy()
        holdout_pred = attach_cv_policy_columns(
            holdout_pred,
            train_window_years=int(args.max_train_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        )
        holdout_pred["min_train_window_years"] = int(args.min_train_years)
        holdout_pred = holdout_pred.sort_values(["race_id", "horse_no"], kind="mergesort")
        outputs["holdout"].parent.mkdir(parents=True, exist_ok=True)
        holdout_pred.to_parquet(outputs["holdout"], index=False)
        holdout_rows = int(len(holdout_pred))
        holdout_races = int(holdout_pred["race_id"].nunique())

    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": pred_col,
        "required_pred_cols": required_pred_cols,
        "feature_columns": feature_cols,
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.max_train_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        ),
        "config": {
            "holdout_year": int(args.holdout_year),
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "params_json": getattr(args, "_applied_params_json", None),
            "final_iteration_source": final_iteration_source,
        },
        "data_summary": {
            "rows": int(len(eligible)),
            "races": int(eligible["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
            "holdout_rows": int(holdout_rows),
            "holdout_races": int(holdout_races),
        },
        "folds": fold_metrics,
        "summary": {
            "logloss": _summary([m.get("logloss") for m in fold_metrics]),
            "brier": _summary([m.get("brier") for m in fold_metrics]),
            "auc": _summary([m.get("auc") for m in fold_metrics]),
            "ece": _summary([m.get("ece") for m in fold_metrics]),
            "best_iteration": _summary([float(x) for x in best_iterations]),
        },
        "cv_best_iteration_median": int(cv_best_iteration_median),
        "final_iterations": int(final_iterations),
    }
    save_json(outputs["metrics"], metrics_payload)

    input_paths = {
        "features_v3": str(features_path),
        **{f"base_oof_{k}": str(v) for k, v in _prediction_paths(args, holdout=False).items()},
    }
    if holdout_input is not None:
        input_paths["holdout_input"] = str(holdout_input)
        input_paths.update(
            {f"base_holdout_{k}": str(v) for k, v in _prediction_paths(args, holdout=True).items()}
        )

    save_json(
        outputs["feature_manifest"],
        _feature_manifest_payload(
            args=args,
            feature_cols=feature_cols,
            valid_years=[int(fold.valid_year) for fold in folds],
        ),
    )
    save_json(
        outputs["meta"],
        _meta_payload(
            args=args,
            feature_cols=feature_cols,
            base_valid_years=base_valid_years,
            valid_years=[int(fold.valid_year) for fold in folds],
            recent_years=recent_years,
            input_paths=input_paths,
            output_paths=outputs,
            holdout_rows=holdout_rows,
            holdout_races=holdout_races,
            code_hash_paths=_meta_code_hash_paths(),
        ),
    )

    logger.info(
        "stacker task=%s years=%s folds=%s recent_years=%s holdout_rows=%s",
        args.task,
        years,
        len(folds),
        recent_years,
        holdout_rows,
    )
    logger.info("wrote %s", outputs["oof"])
    logger.info("wrote %s", outputs["metrics"])
    logger.info("wrote %s", outputs["model"])
    logger.info("wrote %s", outputs["all_years_model"])
    logger.info("wrote %s", outputs["feature_manifest"])
    logger.info("wrote %s", outputs["meta"])
    return 0

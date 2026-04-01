#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba_research.common.v3_utils import resolve_path, save_json
from keiba_research.features.registry import get_stacker_feature_columns
from keiba_research.training.binary_common import (
    coerce_feature_matrix,
    compute_binary_metrics,
    fold_integrity,
    prepare_binary_frame,
)
from keiba_research.training.cv_policy import (
    DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    build_capped_expanding_year_folds,
)
from keiba_research.training.stacker_common import (
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HOLDOUT_YEAR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_SEED,
    _fit_lgbm_fold,
    _label_col,
    _required_pred_cols,
)

logger = logging.getLogger(__name__)

LGBM_LR_RANGE = (0.005, 0.10)
LGBM_NUM_LEAVES_RANGE = (15, 255)
LGBM_MIN_DATA_IN_LEAF_RANGE = (20, 1000)
LGBM_LAMBDA_L1_RANGE = (1e-3, 10.0)
LGBM_LAMBDA_L2_RANGE = (1e-3, 1000.0)
LGBM_FEATURE_FRACTION_RANGE = (0.6, 1.0)
LGBM_BAGGING_FRACTION_RANGE = (0.5, 1.0)
LGBM_BAGGING_FREQ_RANGE = (0, 10)


@dataclass(frozen=True)
class CvEvalResult:
    value_mean_logloss: float
    fold_logloss: dict[int, float]
    fold_best_iteration: dict[int, int]
    fold_metrics: list[dict[str, Any]]


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    state: str
    value_mean_logloss: float | None
    params: dict[str, Any]
    fold_logloss: dict[int, float]
    fold_best_iteration: dict[int, int]


def _default_base_oof_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/oof/{prefix}_lgbm_oof.parquet",
        "xgb": f"data/oof/{prefix}_xgb_oof.parquet",
        "cat": f"data/oof/{prefix}_cat_oof.parquet",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune v3 strict temporal stacker with Optuna on capped-expanding yearly CV. "
            "Default window is min=2, max=3. Current learner is LightGBM."
        )
    )
    parser.add_argument("--task", choices=["win", "place"], default="win")
    parser.add_argument("--features-input", default="data/features_v3.parquet")
    parser.add_argument("--lgbm-oof", default="")
    parser.add_argument("--xgb-oof", default="")
    parser.add_argument("--cat-oof", default="")
    parser.add_argument("--trials-output", default="")
    parser.add_argument("--best-output", default="")
    parser.add_argument("--best-params-output", default="")
    parser.add_argument("--storage", default="")
    parser.add_argument("--study-name", default="")
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix appended to the default study name and output paths.",
    )
    parser.add_argument("--n-trials", type=int, default=300, help="Target total trial count.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout seconds (0=disabled).",
    )
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
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    defaults = _default_base_oof_paths(str(args.task))
    if not str(args.lgbm_oof).strip():
        args.lgbm_oof = defaults["lgbm"]
    if not str(args.xgb_oof).strip():
        args.xgb_oof = defaults["xgb"]
    if not str(args.cat_oof).strip():
        args.cat_oof = defaults["cat"]
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.n_trials) <= 0:
        raise SystemExit("--n-trials must be > 0")
    if int(args.timeout) < 0:
        raise SystemExit("--timeout must be >= 0")
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


def _default_study_name(task: str, artifact_suffix: str = "") -> str:
    base = f"stack_v3_{task}"
    suffix = str(artifact_suffix).strip().replace(" ", "_")
    return f"{base}_{suffix}" if suffix else base


def _resolve_output_paths(args: argparse.Namespace, *, study_name: str) -> dict[str, Path]:
    defaults = {
        "trials": f"data/oof/{study_name}_trials.parquet",
        "best": f"data/oof/{study_name}_best.json",
        "best_params": f"data/oof/{study_name}_best_params.json",
        "storage": f"data/optuna/{study_name}.sqlite3",
    }
    return {
        "trials": resolve_path(args.trials_output or defaults["trials"]),
        "best": resolve_path(args.best_output or defaults["best"]),
        "best_params": resolve_path(args.best_params_output or defaults["best_params"]),
        "storage": resolve_path(args.storage or defaults["storage"]),
    }


def _fixed_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
        "min_train_years": int(args.min_train_years),
        "max_train_years": int(args.max_train_years),
        "num_boost_round": int(args.num_boost_round),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "n_bins": int(args.n_bins),
        "seed": int(args.seed),
        "artifact_suffix": str(args.artifact_suffix).strip(),
    }


def _study_config_payload(
    args: argparse.Namespace,
    *,
    features_input: Path,
    base_oof_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "task": str(args.task),
        "holdout_year": int(args.holdout_year),
        "features_input": str(features_input),
        "base_oof_paths": {key: str(value) for key, value in base_oof_paths.items()},
        "artifact_suffix": str(args.artifact_suffix).strip(),
        "fixed_config": _fixed_config_payload(args),
    }


def _assert_study_compatible(study, *, expected_config: dict[str, Any]) -> None:
    existing = study.user_attrs.get("study_config")
    if existing is None:
        if study.trials:
            raise SystemExit(
                "Existing study has trials but no stored study_config. "
                "Use a new study name or clear the storage."
            )
        study.set_user_attr("study_config", expected_config)
        return
    if existing != expected_config:
        raise SystemExit(
            "Study configuration mismatch for resume. "
            f"expected={expected_config} existing={existing}"
        )


def suggest_model_params(trial) -> dict[str, Any]:
    return {
        "learning_rate": float(trial.suggest_float("learning_rate", *LGBM_LR_RANGE, log=True)),
        "num_leaves": int(trial.suggest_int("num_leaves", *LGBM_NUM_LEAVES_RANGE)),
        "min_data_in_leaf": int(
            trial.suggest_int("min_data_in_leaf", *LGBM_MIN_DATA_IN_LEAF_RANGE)
        ),
        "lambda_l1": float(trial.suggest_float("lambda_l1", *LGBM_LAMBDA_L1_RANGE, log=True)),
        "lambda_l2": float(trial.suggest_float("lambda_l2", *LGBM_LAMBDA_L2_RANGE, log=True)),
        "feature_fraction": float(
            trial.suggest_float("feature_fraction", *LGBM_FEATURE_FRACTION_RANGE)
        ),
        "bagging_fraction": float(
            trial.suggest_float("bagging_fraction", *LGBM_BAGGING_FRACTION_RANGE)
        ),
        "bagging_freq": int(trial.suggest_int("bagging_freq", *LGBM_BAGGING_FREQ_RANGE)),
    }


def _summary_with_median(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _set_trial_user_attrs(
    trial,
    *,
    fold_logloss: dict[int, float],
    fold_best_iteration: dict[int, int],
) -> None:
    trial.set_user_attr(
        "fold_logloss",
        {int(fold_id): float(value) for fold_id, value in fold_logloss.items()},
    )
    trial.set_user_attr(
        "fold_best_iteration",
        {int(fold_id): int(value) for fold_id, value in fold_best_iteration.items()},
    )


def _build_train_args(
    *,
    tune_args: argparse.Namespace,
    params: dict[str, Any],
) -> argparse.Namespace:
    out = argparse.Namespace()
    out.task = str(tune_args.task)
    out.holdout_year = int(tune_args.holdout_year)
    out.min_train_years = int(tune_args.min_train_years)
    out.max_train_years = int(tune_args.max_train_years)
    out.num_boost_round = int(tune_args.num_boost_round)
    out.early_stopping_rounds = int(tune_args.early_stopping_rounds)
    out.seed = int(tune_args.seed)
    out.n_bins = int(tune_args.n_bins)
    for key, value in params.items():
        setattr(out, key, value)
    return out


def _load_prediction_frame(path: Path, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction input not found: {path}")
    frame = pd.read_parquet(path)
    required = {"race_id", "horse_no", pred_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")
    out = frame[["race_id", "horse_no", pred_col]].copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if out.duplicated(["race_id", "horse_no"]).any():
        raise SystemExit(f"Duplicate keys in prediction input: {path}")
    return out


def _merge_prediction_features(frame: pd.DataFrame, pred_paths: dict[str, Path]) -> pd.DataFrame:
    merged = frame.copy()
    for pred_col, path in pred_paths.items():
        pred_df = _load_prediction_frame(path, pred_col)
        merged = merged.merge(pred_df, on=["race_id", "horse_no"], how="left")
    return merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _load_valid_years(path: Path) -> list[int]:
    if not path.exists():
        raise SystemExit(f"prediction input not found: {path}")
    frame = pd.read_parquet(path, columns=["valid_year"])
    if "valid_year" not in frame.columns:
        raise SystemExit(f"Missing valid_year column in {path}")
    years = (
        pd.to_numeric(frame["valid_year"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    return [int(year) for year in years]


def _evaluate_stacker_cv(
    *,
    frame: pd.DataFrame,
    folds: list,
    feat_cols: list[str],
    label_col: str,
    train_args: argparse.Namespace,
    progress_callback: Callable[[int, float, dict[int, float], dict[int, int]], None] | None = None,
) -> CvEvalResult:
    fold_metrics: list[dict[str, Any]] = []
    fold_logloss: dict[int, float] = {}
    fold_best_iteration: dict[int, int] = {}
    running_logloss: list[float] = []

    for fold in folds:
        train_df = frame[frame["year"].isin(fold.train_years)].copy()
        valid_df = frame[frame["year"] == fold.valid_year].copy()
        if train_df.empty or valid_df.empty:
            raise RuntimeError(
                f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
            )
        fold_integrity(train_df, valid_df, int(fold.valid_year))

        x_train = coerce_feature_matrix(train_df, feat_cols)
        y_train = train_df[label_col].astype(int)
        x_valid = coerce_feature_matrix(valid_df, feat_cols)
        y_valid = valid_df[label_col].astype(int)

        _, pred_valid, best_iteration = _fit_lgbm_fold(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=train_args,
            seed=int(train_args.seed) + int(fold.fold_id),
        )
        metrics = compute_binary_metrics(
            y_valid.to_numpy(dtype=int),
            pred_valid,
            n_bins=int(train_args.n_bins),
        )

        fold_id = int(fold.fold_id)
        logloss_value = metrics.get("logloss")
        if logloss_value is None or not np.isfinite(logloss_value):
            raise RuntimeError(
                f"fold={fold.fold_id} produced invalid logloss for task={train_args.task}"
            )

        fold_logloss[fold_id] = float(logloss_value)
        fold_best_iteration[fold_id] = int(best_iteration)
        running_logloss.append(float(logloss_value))
        fold_metrics.append(
            {
                "fold_id": fold_id,
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
                "min_train_years": int(train_args.min_train_years),
                "max_train_years": int(train_args.max_train_years),
                "holdout_year": int(train_args.holdout_year),
            }
        )

        if progress_callback is not None:
            progress_callback(
                fold_id,
                float(np.mean(np.asarray(running_logloss, dtype=float))),
                dict(fold_logloss),
                dict(fold_best_iteration),
            )

    if not running_logloss:
        raise RuntimeError("No fold results were produced during CV evaluation.")

    return CvEvalResult(
        value_mean_logloss=float(np.mean(np.asarray(running_logloss, dtype=float))),
        fold_logloss=fold_logloss,
        fold_best_iteration=fold_best_iteration,
        fold_metrics=fold_metrics,
    )


def _trial_results_from_study(study) -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial in study.trials:
        fold_logloss = trial.user_attrs.get("fold_logloss", {})
        fold_best_iteration = trial.user_attrs.get("fold_best_iteration", {})
        results.append(
            TrialResult(
                trial_number=int(trial.number),
                state=str(trial.state.name),
                value_mean_logloss=None if trial.value is None else float(trial.value),
                params=dict(trial.params),
                fold_logloss={
                    int(key): float(value) for key, value in (fold_logloss or {}).items()
                },
                fold_best_iteration={
                    int(key): int(value) for key, value in (fold_best_iteration or {}).items()
                },
            )
        )
    return results


def _trial_results_to_frame(results: list[TrialResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "trial_number": int(result.trial_number),
            "state": result.state,
            "value_mean_logloss": result.value_mean_logloss,
        }
        for key, value in result.params.items():
            row[f"param/{key}"] = value
        for fold_id, value in result.fold_logloss.items():
            row[f"fold/{fold_id}/logloss"] = value
        for fold_id, value in result.fold_best_iteration.items():
            row[f"fold/{fold_id}/best_iteration"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trial_number"], kind="mergesort")


def _complete_trial_results(results: list[TrialResult]) -> list[TrialResult]:
    return [
        result
        for result in results
        if result.state == "COMPLETE" and result.value_mean_logloss is not None
    ]


def select_best_trial_result(*, results: list[TrialResult]) -> tuple[TrialResult, str]:
    complete = _complete_trial_results(results)
    if not complete:
        raise ValueError("No COMPLETE trials available.")
    selected = min(
        complete,
        key=lambda result: (float(result.value_mean_logloss), int(result.trial_number)),
    )
    return selected, "min_complete"


def _median_best_iteration(fold_best_iteration: dict[int, int], *, upper_bound: int) -> int:
    if not fold_best_iteration:
        raise ValueError("fold_best_iteration must not be empty")
    values = np.asarray(list(fold_best_iteration.values()), dtype=float)
    median_value = int(np.median(values))
    return max(1, min(median_value, int(upper_bound)))


def _build_baseline_summary(
    *,
    args: argparse.Namespace,
    eval_result: CvEvalResult,
) -> dict[str, Any]:
    return {
        "value_mean_logloss": float(eval_result.value_mean_logloss),
        "fold_logloss": eval_result.fold_logloss,
        "fold_best_iteration": eval_result.fold_best_iteration,
        "logloss_summary": _summary_with_median(list(eval_result.fold_logloss.values())),
        "fixed_config": _fixed_config_payload(args),
    }


def _build_best_output(
    *,
    args: argparse.Namespace,
    outputs: dict[str, Path],
    study_name: str,
    total_trials: int,
    selected_trial: TrialResult,
    baseline_summary: dict[str, Any],
    selection_mode: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "study_name": study_name,
        "storage": str(outputs["storage"]),
        "direction": "minimize",
        "total_trials": int(total_trials),
        "best_trial_number": int(selected_trial.trial_number),
        "best_value_mean_logloss": float(selected_trial.value_mean_logloss),
        "best_fold_logloss": selected_trial.fold_logloss,
        "best_fold_best_iteration": selected_trial.fold_best_iteration,
        "best_params": dict(selected_trial.params),
        "task": str(args.task),
        "model": "lgbm",
        "selection_mode": selection_mode,
        "fixed_config": _fixed_config_payload(args),
        "artifact_suffix": str(args.artifact_suffix).strip(),
        "baseline_summary": baseline_summary,
    }


def _build_best_params_output(
    *,
    args: argparse.Namespace,
    selected_trial: TrialResult,
    valid_years: list[int],
) -> dict[str, Any]:
    final_iterations = _median_best_iteration(
        selected_trial.fold_best_iteration,
        upper_bound=int(args.num_boost_round),
    )
    return {
        "task": str(args.task),
        "model": "lgbm",
        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
        "min_train_years": int(args.min_train_years),
        "max_train_years": int(args.max_train_years),
        "tuned_under_holdout_year": int(args.holdout_year),
        "valid_years": list(map(int, valid_years)),
        "selected_trial_number": int(selected_trial.trial_number),
        "selection_metric": "mean_cv_logloss",
        "artifact_suffix": str(args.artifact_suffix).strip(),
        "lgbm_params": dict(selected_trial.params),
        "final_num_boost_round": int(final_iterations),
    }


def run_tune_stacker(
    *,
    task: str = "win",
    features_input: str = "data/features_v3.parquet",
    lgbm_oof: str = "",
    xgb_oof: str = "",
    cat_oof: str = "",
    study_name: str = "",
    storage: str = "",
    trials_output: str = "",
    best_output: str = "",
    best_params_output: str = "",
    holdout_year: int = DEFAULT_HOLDOUT_YEAR,
    min_train_years: int = DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    max_train_years: int = DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    n_trials: int = 300,
    timeout: int = 0,
    seed: int = DEFAULT_SEED,
    log_level: str = "INFO",
) -> int:
    argv_list: list[str] = [
        "--task", str(task),
        "--features-input", str(features_input),
        "--lgbm-oof", str(lgbm_oof),
        "--xgb-oof", str(xgb_oof),
        "--cat-oof", str(cat_oof),
        "--study-name", str(study_name),
        "--storage", str(storage),
        "--trials-output", str(trials_output),
        "--best-output", str(best_output),
        "--best-params-output", str(best_params_output),
        "--holdout-year", str(int(holdout_year)),
        "--min-train-years", str(int(min_train_years)),
        "--max-train-years", str(int(max_train_years)),
        "--n-trials", str(int(n_trials)),
        "--timeout", str(int(timeout)),
        "--seed", str(int(seed)),
        "--log-level", str(log_level),
    ]
    args = _parse_args(argv_list)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    _validate_args(args)

    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("optuna is not installed. Run `uv sync --extra optuna`.") from exc

    study_name = str(
        args.study_name or _default_study_name(str(args.task), str(args.artifact_suffix))
    )
    outputs = _resolve_output_paths(args, study_name=study_name)

    features_input = resolve_path(args.features_input)
    if not features_input.exists():
        raise SystemExit(f"features input not found: {features_input}")

    pred_paths = {
        _required_pred_cols(str(args.task))[0]: resolve_path(str(args.lgbm_oof)),
        _required_pred_cols(str(args.task))[1]: resolve_path(str(args.xgb_oof)),
        _required_pred_cols(str(args.task))[2]: resolve_path(str(args.cat_oof)),
    }

    label_col = _label_col(str(args.task))
    raw = pd.read_parquet(features_input)
    base = prepare_binary_frame(raw, label_col=label_col)
    merged = _merge_prediction_features(base, pred_paths)
    feature_cols = get_stacker_feature_columns(merged, task=str(args.task))

    required_pred_cols = _required_pred_cols(str(args.task))
    eligible = merged.copy()
    for col in required_pred_cols:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    eligible = eligible[eligible["year"] < int(args.holdout_year)].copy()
    eligible = eligible[eligible[required_pred_cols].notna().all(axis=1)].copy()
    if eligible.empty:
        raise SystemExit("No eligible rows for stacker tuning after OOF/holdout filtering.")

    years = sorted(eligible["year"].unique().tolist())
    folds = build_capped_expanding_year_folds(
        years,
        min_window_years=int(args.min_train_years),
        max_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        raise SystemExit(
            "No stacker tuning folds available under capped_expanding "
            f"(available_years={years}, min={args.min_train_years}, max={args.max_train_years})"
        )

    base_valid_years = sorted(
        set.intersection(*[set(_load_valid_years(path)) for path in pred_paths.values()])
    )
    logger.info(
        "stack optuna task=%s years=%s folds=%s valid_years=%s feature_count=%s",
        args.task,
        years,
        len(folds),
        [int(fold.valid_year) for fold in folds],
        len(feature_cols),
    )
    logger.info("upstream valid_year intersection=%s", base_valid_years)

    baseline_params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
    }
    baseline_eval = _evaluate_stacker_cv(
        frame=eligible,
        folds=folds,
        feat_cols=feature_cols,
        label_col=label_col,
        train_args=_build_train_args(tune_args=args, params=baseline_params),
    )
    baseline_summary = _build_baseline_summary(args=args, eval_result=baseline_eval)

    outputs["storage"].parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{outputs['storage']}"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )
    _assert_study_compatible(
        study,
        expected_config=_study_config_payload(
            args,
            features_input=features_input,
            base_oof_paths={
                "lgbm": resolve_path(str(args.lgbm_oof)),
                "xgb": resolve_path(str(args.xgb_oof)),
                "cat": resolve_path(str(args.cat_oof)),
            },
        ),
    )

    def objective(trial) -> float:
        model_params = suggest_model_params(trial)
        train_args = _build_train_args(tune_args=args, params=model_params)

        def on_progress(
            fold_id: int,
            running_mean: float,
            fold_logloss: dict[int, float],
            fold_best_iteration: dict[int, int],
        ) -> None:
            trial.report(float(running_mean), step=int(fold_id))
            _set_trial_user_attrs(
                trial,
                fold_logloss=fold_logloss,
                fold_best_iteration=fold_best_iteration,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        eval_result = _evaluate_stacker_cv(
            frame=eligible,
            folds=folds,
            feat_cols=feature_cols,
            label_col=label_col,
            train_args=train_args,
            progress_callback=on_progress,
        )
        _set_trial_user_attrs(
            trial,
            fold_logloss=eval_result.fold_logloss,
            fold_best_iteration=eval_result.fold_best_iteration,
        )
        return float(eval_result.value_mean_logloss)

    existing_trials = len(study.trials)
    additional_trials = max(0, int(args.n_trials) - existing_trials)
    logger.info(
        "study=%s storage=%s existing_trials=%s target_trials=%s additional=%s",
        study.study_name,
        outputs["storage"],
        existing_trials,
        args.n_trials,
        additional_trials,
    )
    timeout = None if int(args.timeout) == 0 else int(args.timeout)
    if additional_trials > 0:
        study.optimize(objective, n_trials=additional_trials, timeout=timeout)

    results = _trial_results_from_study(study)
    trials_frame = _trial_results_to_frame(results)
    outputs["trials"].parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(outputs["trials"], index=False)
    logger.info("wrote %s rows=%s", outputs["trials"], len(trials_frame))

    selected_trial, selection_mode = select_best_trial_result(results=results)
    best_output = _build_best_output(
        args=args,
        outputs=outputs,
        study_name=study_name,
        total_trials=len(study.trials),
        selected_trial=selected_trial,
        baseline_summary=baseline_summary,
        selection_mode=selection_mode,
    )
    save_json(outputs["best"], best_output)
    logger.info("wrote %s", outputs["best"])

    best_params_output = _build_best_params_output(
        args=args,
        selected_trial=selected_trial,
        valid_years=[int(fold.valid_year) for fold in folds],
    )
    save_json(outputs["best_params"], best_params_output)
    logger.info("wrote %s", outputs["best_params"])
    return 0

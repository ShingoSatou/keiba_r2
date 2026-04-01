#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba_research.common.v3_utils import (
    hash_files,
    resolve_path,
)
from keiba_research.features.registry import (
    FEATURE_MANIFEST_VERSION,
    STACKER_REQUIRED_PRED_FEATURES_PLACE,
    STACKER_REQUIRED_PRED_FEATURES_WIN,
)
from keiba_research.training.cv_policy import (
    attach_cv_policy_columns,
    make_capped_expanding_window_definition,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEED = 42
DEFAULT_CV_WINDOW_POLICY = "capped_expanding"


def _label_col(task: str) -> str:
    return "y_win" if str(task) == "win" else "y_place"


def _pred_col(task: str) -> str:
    return "p_win_stack" if str(task) == "win" else "p_place_stack"


def _required_pred_cols(task: str) -> list[str]:
    if str(task) == "win":
        return list(STACKER_REQUIRED_PRED_FEATURES_WIN)
    return list(STACKER_REQUIRED_PRED_FEATURES_PLACE)


def _default_base_oof_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/oof/{prefix}_lgbm_oof.parquet",
        "xgb": f"data/oof/{prefix}_xgb_oof.parquet",
        "cat": f"data/oof/{prefix}_cat_oof.parquet",
    }


def _default_base_holdout_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/holdout/{prefix}_lgbm_holdout_2025_pred_v3.parquet",
        "xgb": f"data/holdout/{prefix}_xgb_holdout_2025_pred_v3.parquet",
        "cat": f"data/holdout/{prefix}_cat_holdout_2025_pred_v3.parquet",
    }


STACKER_PARAM_CLI_FLAGS: dict[str, str] = {
    "learning_rate": "--learning-rate",
    "num_leaves": "--num-leaves",
    "min_data_in_leaf": "--min-data-in-leaf",
    "lambda_l1": "--lambda-l1",
    "lambda_l2": "--lambda-l2",
    "feature_fraction": "--feature-fraction",
    "bagging_fraction": "--bagging-fraction",
    "bagging_freq": "--bagging-freq",
}


def _model_thread_count() -> int:
    raw = str(os.environ.get("V3_MODEL_THREADS", "")).strip()
    if not raw:
        return -1
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid V3_MODEL_THREADS=%s; fallback to -1", raw)
        return -1
    if value == 0:
        logger.warning("Ignoring V3_MODEL_THREADS=0; fallback to -1")
        return -1
    return value


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


def _prediction_paths(args: argparse.Namespace, *, holdout: bool) -> dict[str, Path]:
    if str(args.task) == "win":
        mapping = {
            "p_win_lgbm": args.lgbm_holdout if holdout else args.lgbm_oof,
            "p_win_xgb": args.xgb_holdout if holdout else args.xgb_oof,
            "p_win_cat": args.cat_holdout if holdout else args.cat_oof,
        }
    else:
        mapping = {
            "p_place_lgbm": args.lgbm_holdout if holdout else args.lgbm_oof,
            "p_place_xgb": args.xgb_holdout if holdout else args.xgb_oof,
            "p_place_cat": args.cat_holdout if holdout else args.cat_oof,
        }
    return {pred_col: resolve_path(path) for pred_col, path in mapping.items()}


def _merge_prediction_features(frame: pd.DataFrame, pred_paths: dict[str, Path]) -> pd.DataFrame:
    merged = frame.copy()
    for pred_col, path in pred_paths.items():
        pred_df = _load_prediction_frame(path, pred_col)
        merged = merged.merge(pred_df, on=["race_id", "horse_no"], how="left")
    return merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _fit_lgbm_fold(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, np.ndarray, int]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_data_in_leaf),
        reg_alpha=float(args.lambda_l1),
        reg_lambda=float(args.lambda_l2),
        colsample_bytree=float(args.feature_fraction),
        subsample=float(args.bagging_fraction),
        subsample_freq=int(args.bagging_freq),
        random_state=int(seed),
        n_jobs=_model_thread_count(),
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_names=["valid"],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(int(args.early_stopping_rounds), verbose=False)],
    )
    best_iteration = int(getattr(model, "best_iteration_", 0) or int(args.num_boost_round))
    pred = np.clip(
        np.asarray(model.predict_proba(x_valid, num_iteration=best_iteration)[:, 1], dtype=float),
        1e-8,
        1.0 - 1e-8,
    )
    return model, pred, best_iteration


def _fit_lgbm_final(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
    seed: int,
    n_estimators: int,
) -> Any:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_data_in_leaf),
        reg_alpha=float(args.lambda_l1),
        reg_lambda=float(args.lambda_l2),
        colsample_bytree=float(args.feature_fraction),
        subsample=float(args.bagging_fraction),
        subsample_freq=int(args.bagging_freq),
        random_state=int(seed),
        n_jobs=_model_thread_count(),
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


def _predict_lgbm(model: Any, x: pd.DataFrame, *, num_iteration: int | None = None) -> np.ndarray:
    pred = model.predict_proba(x, num_iteration=num_iteration)[:, 1]
    return np.clip(np.asarray(pred, dtype=float), 1e-8, 1.0 - 1e-8)


def _save_lgbm_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def _summary(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _oof_frame(
    valid_df: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    pred_values: np.ndarray,
    fold_id: int,
    valid_year: int,
    args: argparse.Namespace,
    window_definition: str,
) -> pd.DataFrame:
    out = valid_df[
        [
            c
            for c in [
                "race_id",
                "horse_id",
                "horse_no",
                "t_race",
                "race_date",
                "field_size",
                "target_label",
                label_col,
            ]
            if c in valid_df.columns
        ]
    ].copy()
    out[pred_col] = np.asarray(pred_values, dtype=float)
    out["fold_id"] = int(fold_id)
    out["valid_year"] = int(valid_year)
    out = attach_cv_policy_columns(
        out,
        train_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
        cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        window_definition=window_definition,
    )
    out["min_train_window_years"] = int(args.min_train_years)
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort")
    return out


def _feature_manifest_payload(
    *,
    args: argparse.Namespace,
    feature_cols: list[str],
    valid_years: list[int],
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": _pred_col(str(args.task)),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(args.holdout_year),
            "window_definition": make_capped_expanding_window_definition(
                int(args.min_train_years),
                int(args.max_train_years),
            ),
        },
        "feature_columns": list(feature_cols),
        "forbidden_feature_check_passed": True,
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
        "artifact_suffix": str(args.artifact_suffix).strip(),
    }


def _meta_payload(
    *,
    args: argparse.Namespace,
    feature_cols: list[str],
    base_valid_years: list[int],
    valid_years: list[int],
    recent_years: list[int],
    input_paths: dict[str, str],
    output_paths: dict[str, Path],
    holdout_rows: int,
    holdout_races: int,
    code_hash_paths: list[Path],
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": _pred_col(str(args.task)),
        "feature_columns": list(feature_cols),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(args.holdout_year),
            "window_definition": make_capped_expanding_window_definition(
                int(args.min_train_years),
                int(args.max_train_years),
            ),
        },
        "base_oof_valid_years": list(map(int, base_valid_years)),
        "stacker_oof_valid_years": list(map(int, valid_years)),
        "recent_window_years": list(map(int, recent_years)),
        "input_paths": input_paths,
        "output_paths": {
            "oof": str(output_paths["oof"]),
            "holdout": str(output_paths["holdout"]),
            "metrics": str(output_paths["metrics"]),
            "main_model": str(output_paths["model"]),
            "all_years_model": str(output_paths["all_years_model"]),
            "feature_manifest": str(output_paths["feature_manifest"]),
        },
        "holdout_summary": {
            "rows": int(holdout_rows),
            "races": int(holdout_races),
        },
        "forbidden_feature_check_passed": True,
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
        "artifact_suffix": str(args.artifact_suffix).strip(),
        "tuned_defaults": {
            "params_json": getattr(args, "_applied_params_json", None),
        },
        "code_hash": hash_files(code_hash_paths),
    }


from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts_v3.cv_policy_v3 import (
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_TRAIN_WINDOW_YEARS,
    attach_cv_policy_columns,
)
from scripts_v3.v3_common import (
    append_stem_suffix,
    assert_fold_integrity,
    resolve_path,
)


def prepare_binary_frame(frame: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    required = {"race_id", "horse_id", "horse_no", "race_date", "field_size", label_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = frame.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year.astype(int)

    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)

    out["horse_id"] = out["horse_id"].astype(str)
    out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    out["t_race"] = pd.to_datetime(out.get("t_race", out["race_date"]), errors="coerce")

    out = out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    return out


def coerce_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    mat = frame[feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(mat[col]):
            mat[col] = pd.to_numeric(mat[col], errors="coerce")
    return mat


def binary_output_paths(
    *,
    task: str,
    model: str,
    holdout_year: int,
    artifact_suffix: str,
    ext: str,
    oof_output: str,
    metrics_output: str,
    model_output: str,
    all_years_model_output: str,
    meta_output: str,
    feature_manifest_output: str,
    holdout_output: str,
) -> dict[str, Path]:
    defaults = {
        "oof": append_stem_suffix(f"data/oof/{task}_{model}_oof.parquet", artifact_suffix),
        "metrics": append_stem_suffix(
            f"data/oof/{task}_{model}_cv_metrics.json",
            artifact_suffix,
        ),
        "model": append_stem_suffix(f"models/{task}_{model}_v3.{ext}", artifact_suffix),
        "all_years_model": append_stem_suffix(
            f"models/{task}_{model}_all_years_v3.{ext}",
            artifact_suffix,
        ),
        "meta": append_stem_suffix(
            f"models/{task}_{model}_bundle_meta_v3.json",
            artifact_suffix,
        ),
        "feature_manifest": append_stem_suffix(
            f"models/{task}_{model}_feature_manifest_v3.json",
            artifact_suffix,
        ),
        "holdout": append_stem_suffix(
            f"data/holdout/{task}_{model}_holdout_{int(holdout_year)}_pred_v3.parquet",
            artifact_suffix,
        ),
    }
    return {
        "oof": resolve_path(oof_output or defaults["oof"]),
        "metrics": resolve_path(metrics_output or defaults["metrics"]),
        "model": resolve_path(model_output or defaults["model"]),
        "all_years_model": resolve_path(all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(meta_output or defaults["meta"]),
        "feature_manifest": resolve_path(feature_manifest_output or defaults["feature_manifest"]),
        "holdout": resolve_path(holdout_output or defaults["holdout"]),
    }


def reliability_bins(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int,
) -> tuple[list[dict[str, Any]], float]:
    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_ids = np.digitize(p_pred, edges[1:-1], right=False)
    n_total = float(len(p_pred))
    bins: list[dict[str, Any]] = []
    ece_value = 0.0

    for idx in range(int(n_bins)):
        mask = bin_ids == idx
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_id": int(idx),
                    "left": float(edges[idx]),
                    "right": float(edges[idx + 1]),
                    "count": 0,
                    "mean_pred": None,
                    "frac_pos": None,
                }
            )
            continue
        mean_pred = float(np.mean(p_pred[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        ece_value += abs(mean_pred - frac_pos) * (count / n_total)
        bins.append(
            {
                "bin_id": int(idx),
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": int(count),
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
            }
        )
    return bins, float(ece_value)


def compute_binary_metrics(
    y_true: np.ndarray | pd.Series,
    p_pred: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and p_pred length mismatch")
    if y.shape[0] == 0:
        raise ValueError("Cannot compute metrics for empty inputs")

    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    try:
        logloss_value: float | None = float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        logloss_value = None

    try:
        auc_value: float | None = float(roc_auc_score(y, p))
    except ValueError:
        auc_value = None

    brier_value = float(brier_score_loss(y, p))
    reliability, ece_value = reliability_bins(y, p, n_bins=n_bins)
    return {
        "n_rows": int(y.shape[0]),
        "base_rate": float(np.mean(y)),
        "logloss": logloss_value,
        "brier": brier_value,
        "auc": auc_value,
        "ece": ece_value,
        "reliability": reliability,
    }


def fold_integrity(train_df: pd.DataFrame, valid_df: pd.DataFrame, valid_year: int) -> None:
    assert_fold_integrity(train_df, valid_df, int(valid_year))


def build_oof_frame(
    valid_df: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    pred_values: np.ndarray,
    fold_id: int,
    valid_year: int,
    train_window_years: int = DEFAULT_TRAIN_WINDOW_YEARS,
    holdout_year: int,
    cv_window_policy: str = DEFAULT_CV_WINDOW_POLICY,
    window_definition: str | None = None,
) -> pd.DataFrame:
    out_cols = [
        "race_id",
        "horse_id",
        "horse_no",
        "t_race",
        "race_date",
        "field_size",
        "target_label",
        label_col,
    ]
    existing_cols = [c for c in out_cols if c in valid_df.columns]
    out = valid_df[existing_cols].copy()
    out[pred_col] = np.asarray(pred_values, dtype=float)
    out["fold_id"] = int(fold_id)
    out["valid_year"] = int(valid_year)
    out = attach_cv_policy_columns(
        out,
        train_window_years=int(train_window_years),
        holdout_year=int(holdout_year),
        cv_window_policy=str(cv_window_policy),
        window_definition=window_definition,
    )
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort")
    return out


__all__ = [
    "binary_output_paths",
    "build_oof_frame",
    "coerce_feature_matrix",
    "compute_binary_metrics",
    "fold_integrity",
    "prepare_binary_frame",
    "reliability_bins",
]

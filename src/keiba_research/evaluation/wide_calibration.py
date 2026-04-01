#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from keiba_research.common.v3_utils import resolve_path, save_json
from keiba_research.evaluation.backtest_common import parse_years
from keiba_research.training.binary_common import reliability_bins

logger = logging.getLogger(__name__)

DEFAULT_BIN_EDGES = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0])




def parse_bin_edges(raw: str) -> np.ndarray:
    text = str(raw).strip()
    if not text:
        return DEFAULT_BIN_EDGES.copy()

    try:
        edges = np.array([float(token.strip()) for token in text.split(",") if token.strip()])
    except ValueError as exc:
        raise SystemExit(f"Failed to parse --bins: {raw}") from exc

    if edges.size < 2:
        raise SystemExit("--bins must contain at least two edges")
    if not np.isclose(edges[0], 0.0):
        raise SystemExit("--bins must start at 0.0")
    if not np.isclose(edges[-1], 1.0):
        raise SystemExit("--bins must end at 1.0")
    if np.any(~np.isfinite(edges)):
        raise SystemExit("--bins must be finite")
    if np.any(np.diff(edges) <= 0.0):
        raise SystemExit("--bins must be strictly increasing")
    return edges


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _coerce_binary_labels(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.astype("string").str.strip().str.lower()
        mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.notna().all():
            return mapped.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    invalid = ~(numeric.isin([0, 1]) | numeric.isna())
    if bool(invalid.any()):
        raise SystemExit(
            f"{column} must contain only binary values in {{0,1}}. "
            f"invalid_rows={int(invalid.sum())}"
        )
    return numeric.astype("Int64")


def _normalize_surface_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "芝", "shiba", "turf"}:
        return "shiba"
    if text in {"2", "ダート", "dirt"}:
        return "dirt"
    return None


def _slugify(text: object) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower()).strip("_")
    return slug or "group"


def _prepare_input_frame(
    frame: pd.DataFrame,
    *,
    race_col: str,
    p_col: str,
    y_col: str,
    group_col: str,
) -> pd.DataFrame:
    required = {race_col, p_col, y_col}
    if group_col:
        required.add(group_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    out = frame.copy()
    if group_col:
        before_missing = int(out[group_col].isna().sum())
        logger.info(
            "group column '%s' has missing rows before filtering: %s",
            group_col,
            before_missing,
        )

    out[race_col] = pd.to_numeric(out[race_col], errors="coerce").astype("Int64")
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out[y_col] = _coerce_binary_labels(out[y_col], column=y_col)

    required_for_drop = [race_col, p_col, y_col]
    dropped = int(out[required_for_drop].isna().any(axis=1).sum())
    if dropped > 0:
        logger.info("dropping rows with missing required values: %s", dropped)
    out = out.dropna(subset=required_for_drop).copy()

    out[race_col] = out[race_col].astype(int)
    out[y_col] = out[y_col].astype(int)

    if ((out[p_col] < 0.0) | (out[p_col] > 1.0)).any():
        bad = int(((out[p_col] < 0.0) | (out[p_col] > 1.0)).sum())
        raise SystemExit(f"{p_col} must be in [0, 1]. invalid_rows={bad}")

    if group_col == "surface":
        out["__surface_group__"] = out[group_col].map(_normalize_surface_value)
        unknown = int(out["__surface_group__"].isna().sum())
        if unknown > 0:
            logger.info(
                "surface rows outside shiba/dirt mapping remain in all only: %s",
                unknown,
            )

    return out.reset_index(drop=True)


def _select_years(
    frame: pd.DataFrame,
    *,
    years_arg: str,
    require_years_arg: str,
) -> tuple[pd.DataFrame, list[int], list[int], list[int]]:
    years_requested = str(years_arg).strip()
    years_required = str(require_years_arg).strip()

    out = frame.copy()
    if "valid_year" not in out.columns and "race_date" in out.columns:
        race_year = pd.to_datetime(out["race_date"], errors="coerce").dt.year
        if race_year.notna().all():
            out["valid_year"] = race_year.astype(int)

    if "valid_year" not in out.columns:
        if years_requested or years_required:
            raise SystemExit(
                "--years/--require-years requires 'valid_year' column or derivable race_date"
            )
        return out, [], [], []

    out["valid_year"] = pd.to_numeric(out["valid_year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["valid_year"]).copy()
    out["valid_year"] = out["valid_year"].astype(int)
    available = sorted(out["valid_year"].unique().tolist())

    required_years = parse_years(years_required)
    if required_years:
        missing = sorted(set(required_years) - set(available))
        if missing:
            raise SystemExit(
                f"required years are missing after filtering: {missing}, available={available}"
            )

    selected_years = parse_years(years_requested) if years_requested else available
    if selected_years:
        missing_selected = sorted(set(selected_years) - set(available))
        if missing_selected:
            raise SystemExit(
                f"selected years not found in input: {missing_selected}, available={available}"
            )
        out = out[out["valid_year"].isin(selected_years)].copy()
    if out.empty:
        raise SystemExit("No rows left after year selection.")

    selected_available = sorted(out["valid_year"].unique().tolist())
    if required_years:
        missing_after = sorted(set(required_years) - set(selected_available))
        if missing_after:
            raise SystemExit(
                "required years are missing after year selection: "
                f"{missing_after}, selected_available={selected_available}"
            )
    return out.reset_index(drop=True), selected_years, selected_available, available


def assign_bins(probabilities: np.ndarray, edges: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1:
        raise ValueError("probabilities must be 1-D")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    bin_ids = np.searchsorted(edges, p, side="right") - 1
    return np.clip(bin_ids, 0, len(edges) - 2).astype(int)


def _build_empty_calibration_table(edges: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx in range(len(edges) - 1):
        rows.append(
            {
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "n_pairs": 0,
                "exp_count": 0.0,
                "act_count": 0.0,
                "pred_rate": math.nan,
                "obs_rate": math.nan,
                "diff": math.nan,
                "z_indep": math.nan,
                "diff_ci_low": math.nan,
                "diff_ci_high": math.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_race_bin_stats(
    frame: pd.DataFrame,
    *,
    race_col: str,
    p_col: str,
    y_col: str,
    bin_ids: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    race_codes, inverse = np.unique(frame[race_col].to_numpy(dtype=np.int64), return_inverse=True)
    n_races = race_codes.shape[0]
    counts = np.zeros((n_races, n_bins), dtype=float)
    sum_p = np.zeros((n_races, n_bins), dtype=float)
    sum_y = np.zeros((n_races, n_bins), dtype=float)

    np.add.at(counts, (inverse, bin_ids), 1.0)
    np.add.at(sum_p, (inverse, bin_ids), frame[p_col].to_numpy(dtype=float))
    np.add.at(sum_y, (inverse, bin_ids), frame[y_col].to_numpy(dtype=float))
    return counts, sum_p, sum_y


def bootstrap_bin_diff_ci(
    frame: pd.DataFrame,
    *,
    race_col: str,
    p_col: str,
    y_col: str,
    bin_ids: np.ndarray,
    n_bins: int,
    bootstrap_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        nan_arr = np.full(n_bins, np.nan, dtype=float)
        return nan_arr, nan_arr

    counts, sum_p, sum_y = _build_race_bin_stats(
        frame,
        race_col=race_col,
        p_col=p_col,
        y_col=y_col,
        bin_ids=bin_ids,
        n_bins=n_bins,
    )
    n_races = counts.shape[0]
    rng = np.random.default_rng(int(seed))
    diff_samples = np.full((int(bootstrap_n), n_bins), np.nan, dtype=float)

    for idx in range(int(bootstrap_n)):
        sampled = rng.integers(0, n_races, size=n_races)
        sample_counts = counts[sampled].sum(axis=0)
        sample_sum_p = sum_p[sampled].sum(axis=0)
        sample_sum_y = sum_y[sampled].sum(axis=0)
        valid = sample_counts > 0.0
        diff_samples[idx, valid] = (
            sample_sum_y[valid] / sample_counts[valid] - sample_sum_p[valid] / sample_counts[valid]
        )

    low = np.full(n_bins, np.nan, dtype=float)
    high = np.full(n_bins, np.nan, dtype=float)
    for idx in range(n_bins):
        finite = diff_samples[:, idx][np.isfinite(diff_samples[:, idx])]
        if finite.size == 0:
            continue
        low[idx], high[idx] = np.percentile(finite, [2.5, 97.5])
    return low, high


def summarize_calibration_bins(
    frame: pd.DataFrame,
    *,
    p_col: str,
    y_col: str,
    race_col: str,
    edges: np.ndarray,
    bootstrap_n: int,
    seed: int,
) -> pd.DataFrame:
    n_bins = len(edges) - 1
    if frame.empty:
        return _build_empty_calibration_table(edges)

    probs = frame[p_col].to_numpy(dtype=float)
    labels = frame[y_col].to_numpy(dtype=float)
    bin_ids = assign_bins(probs, edges)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(int)
    exp_counts = np.bincount(bin_ids, weights=probs, minlength=n_bins)
    act_counts = np.bincount(bin_ids, weights=labels, minlength=n_bins)
    z_den = np.bincount(bin_ids, weights=probs * (1.0 - probs), minlength=n_bins)
    ci_low, ci_high = bootstrap_bin_diff_ci(
        frame,
        race_col=race_col,
        p_col=p_col,
        y_col=y_col,
        bin_ids=bin_ids,
        n_bins=n_bins,
        bootstrap_n=int(bootstrap_n),
        seed=int(seed),
    )

    rows: list[dict[str, Any]] = []
    for idx in range(n_bins):
        n_pairs = int(counts[idx])
        pred_rate = float(exp_counts[idx] / n_pairs) if n_pairs > 0 else math.nan
        obs_rate = float(act_counts[idx] / n_pairs) if n_pairs > 0 else math.nan
        diff = float(obs_rate - pred_rate) if n_pairs > 0 else math.nan
        if z_den[idx] > 0.0:
            z_indep = float((act_counts[idx] - exp_counts[idx]) / np.sqrt(z_den[idx]))
        else:
            z_indep = math.nan
        rows.append(
            {
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "n_pairs": n_pairs,
                "exp_count": float(exp_counts[idx]),
                "act_count": float(act_counts[idx]),
                "pred_rate": pred_rate,
                "obs_rate": obs_rate,
                "diff": diff,
                "z_indep": z_indep,
                "diff_ci_low": float(ci_low[idx]) if np.isfinite(ci_low[idx]) else math.nan,
                "diff_ci_high": float(ci_high[idx]) if np.isfinite(ci_high[idx]) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_overall_metrics(
    frame: pd.DataFrame,
    *,
    p_col: str,
    y_col: str,
    race_col: str,
    ece_bins: int,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "brier": None,
            "logloss": None,
            "ece": None,
            "n_pairs": 0,
            "n_races": 0,
        }

    y_true = frame[y_col].to_numpy(dtype=int)
    p_pred = frame[p_col].to_numpy(dtype=float)
    _, ece_value = reliability_bins(y_true, p_pred, n_bins=int(ece_bins))
    return {
        "brier": float(brier_score_loss(y_true, p_pred)),
        "logloss": float(log_loss(y_true, np.clip(p_pred, 1e-12, 1.0 - 1e-12), labels=[0, 1])),
        "ece": float(ece_value),
        "n_pairs": int(len(frame)),
        "n_races": int(frame[race_col].nunique()),
    }


def _bin_label(left: float, right: float) -> str:
    return f"{left:.2f}-{right:.2f}"


def save_reliability_curve(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0)

    valid = table["n_pairs"].gt(0) & table["pred_rate"].notna() & table["obs_rate"].notna()
    if bool(valid.any()):
        subset = table.loc[valid].copy()
        max_pairs = float(subset["n_pairs"].max())
        sizes = 40.0 + 180.0 * (subset["n_pairs"].to_numpy(dtype=float) / max_pairs)
        ax.scatter(
            subset["pred_rate"].to_numpy(dtype=float),
            subset["obs_rate"].to_numpy(dtype=float),
            s=sizes,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("pred_rate")
    ax.set_ylabel("obs_rate")
    ax.set_title("Reliability Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_bin_diff_plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table), dtype=float)
    valid = table["diff"].notna()

    if bool(valid.any()):
        subset = table.loc[valid].copy()
        x_valid = x[valid.to_numpy(dtype=bool)]
        y = subset["diff"].to_numpy(dtype=float)
        low = subset["diff_ci_low"].to_numpy(dtype=float)
        high = subset["diff_ci_high"].to_numpy(dtype=float)
        lower_err = np.where(np.isfinite(low), y - low, np.nan)
        upper_err = np.where(np.isfinite(high), high - y, np.nan)
        yerr = np.vstack([np.nan_to_num(lower_err, nan=0.0), np.nan_to_num(upper_err, nan=0.0)])
        ax.errorbar(x_valid, y, yerr=yerr, fmt="o")

    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_bin_label(row.bin_left, row.bin_right) for row in table.itertuples()],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("obs_rate - pred_rate")
    ax.set_xlabel("bin")
    ax.set_title("Bin Diff")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_group_specs(
    frame: pd.DataFrame,
    *,
    group_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    if not group_col:
        return [("", frame)]

    specs: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    if group_col == "surface":
        specs.append(("shiba", frame[frame["__surface_group__"] == "shiba"].copy()))
        specs.append(("dirt", frame[frame["__surface_group__"] == "dirt"].copy()))
        return specs

    non_missing = frame[frame[group_col].notna()].copy()
    if non_missing.empty:
        return specs

    seen_slugs: dict[str, int] = {}
    for group_value in non_missing[group_col].drop_duplicates().tolist():
        subset = non_missing[non_missing[group_col] == group_value].copy()
        base_slug = _slugify(group_value)
        count = seen_slugs.get(base_slug, 0)
        seen_slugs[base_slug] = count + 1
        slug = base_slug if count == 0 else f"{base_slug}_{count + 1}"
        specs.append((slug, subset))
    return specs


def _write_outputs(
    out_dir: Path,
    *,
    suffix: str,
    grouped: bool,
    table: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    if grouped:
        table_path = out_dir / f"calibration_table_{suffix}.parquet"
        metrics_path = out_dir / f"metrics_{suffix}.json"
        reliability_path = out_dir / f"reliability_curve_{suffix}.png"
        diff_path = out_dir / f"bin_diff_{suffix}.png"
    else:
        table_path = out_dir / "calibration_table.parquet"
        metrics_path = out_dir / "metrics.json"
        reliability_path = out_dir / "reliability_curve.png"
        diff_path = out_dir / "bin_diff.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(table_path, index=False)
    save_json(metrics_path, metrics)
    save_reliability_curve(table, reliability_path)
    save_bin_diff_plot(table, diff_path)
    logger.info("wrote %s", table_path)
    logger.info("wrote %s", metrics_path)
    logger.info("wrote %s", reliability_path)
    logger.info("wrote %s", diff_path)


def evaluate_group(
    frame: pd.DataFrame,
    *,
    race_col: str,
    p_col: str,
    y_col: str,
    edges: np.ndarray,
    ece_bins: int,
    bootstrap_n: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = summarize_calibration_bins(
        frame,
        p_col=p_col,
        y_col=y_col,
        race_col=race_col,
        edges=edges,
        bootstrap_n=int(bootstrap_n),
        seed=int(seed),
    )
    metrics = compute_overall_metrics(
        frame,
        p_col=p_col,
        y_col=y_col,
        race_col=race_col,
        ece_bins=int(ece_bins),
    )
    return table, metrics


def run_evaluate_wide_calibration(
    *,
    input: str,
    out_dir: str,
    p_col: str = "p_wide",
    y_col: str = "y_wide",
    race_col: str = "race_id",
    group_col: str = "",
    years: str = "",
    require_years: str = "",
    bins: str = "",
    ece_bins: int = 10,
    bootstrap_n: int = 1000,
    seed: int = 42,
    log_level: str = "INFO",
) -> int:
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    if int(bootstrap_n) < 1:
        raise SystemExit("--bootstrap-n must be >= 1")
    if int(ece_bins) <= 1:
        raise SystemExit("--ece-bins must be > 1")

    input_path = resolve_path(input)
    out_dir_path = resolve_path(out_dir)
    group_col_stripped = str(group_col).strip()
    edges = parse_bin_edges(str(bins))

    raw = _load_input(input_path)
    prepared = _prepare_input_frame(
        raw,
        race_col=str(race_col),
        p_col=str(p_col),
        y_col=str(y_col),
        group_col=group_col_stripped,
    )
    prepared, selected_years, selected_available_years, available_years = _select_years(
        prepared,
        years_arg=str(years),
        require_years_arg=str(require_years),
    )
    logger.info(
        "loaded rows=%s races=%s from %s",
        len(prepared),
        prepared[str(race_col)].nunique() if not prepared.empty else 0,
        input_path,
    )

    grouped = bool(group_col_stripped)
    specs = _build_group_specs(prepared, group_col=group_col_stripped)
    for idx, (suffix, subset) in enumerate(specs):
        table, metrics = evaluate_group(
            subset,
            race_col=str(race_col),
            p_col=str(p_col),
            y_col=str(y_col),
            edges=edges,
            ece_bins=int(ece_bins),
            bootstrap_n=int(bootstrap_n),
            seed=int(seed) + idx,
        )
        metrics["selected_years"] = list(map(int, selected_years))
        metrics["selected_available_years"] = list(map(int, selected_available_years))
        metrics["available_years"] = list(map(int, available_years))
        metrics["required_years"] = parse_years(str(require_years))
        if grouped:
            metrics["group_col"] = group_col_stripped
            metrics["group_value"] = suffix
        _write_outputs(
            out_dir_path,
            suffix=suffix,
            grouped=grouped,
            table=table,
            metrics=metrics,
        )
        logger.info(
            "evaluated group=%s n_pairs=%s n_races=%s",
            suffix or "all",
            metrics["n_pairs"],
            metrics["n_races"],
        )

    return 0

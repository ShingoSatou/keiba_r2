#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from keiba_research.common.v3_utils import (
    append_stem_suffix,
    hash_files,
    resolve_database_url,
    resolve_path,
    save_json,
)
from keiba_research.db.database import Database
from keiba_research.evaluation.backtest_common import parse_years
from keiba_research.evaluation.pl_common import estimate_p_wide_by_race
from keiba_research.evaluation.wide_pair_calibration import (
    apply_wide_pair_calibrator,
    fit_wide_pair_calibrator,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/oof/pl_v3_holdout_2025_pred.parquet"
DEFAULT_MODEL_OUTPUT = "models/wide_pair_calibrator_v3.joblib"
DEFAULT_META_OUTPUT = "models/wide_pair_calibrator_bundle_meta_v3.json"
DEFAULT_PRED_OUTPUT = "data/oof/wide_pair_calibration_2025_pred.parquet"
DEFAULT_METRICS_OUTPUT = "data/oof/wide_pair_calibration_2025_metrics.json"


def _combined_suffix(*parts: str) -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    return "_".join(cleaned)




def _resolve_output_path(raw: str, default_path: str, *, method: str, artifact_suffix: str) -> Path:
    if str(raw).strip() and str(raw).strip() != default_path:
        return resolve_path(raw)

    suffix = _combined_suffix(
        str(artifact_suffix),
        "" if str(method) == "isotonic" else str(method),
    )
    return resolve_path(append_stem_suffix(default_path, suffix))


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _pair_from_horse_input(frame: pd.DataFrame, *, mc_samples: int, seed: int) -> pd.DataFrame:
    required = {"race_id", "horse_no", "pl_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"horse-level input is missing required columns: {missing}")

    work = frame.copy()
    pair = estimate_p_wide_by_race(
        work[["race_id", "horse_no", "pl_score"]],
        score_col="pl_score",
        mc_samples=int(mc_samples),
        seed=int(seed),
        top_k=3,
    )
    pair = pair.rename(columns={"p_wide": "p_wide_raw"})

    carry_cols = [
        "valid_year",
        "race_date",
        "fold_id",
        "cv_window_policy",
        "train_window_years",
        "holdout_year",
        "window_definition",
    ]
    for column in carry_cols:
        if column not in work.columns:
            continue
        mapping = (
            work[["race_id", column]]
            .drop_duplicates(["race_id"])
            .set_index("race_id")[column]
            .to_dict()
        )
        pair[column] = pair["race_id"].map(mapping)
    return pair


def _pair_from_pair_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_no_1", "horse_no_2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"pair-level input is missing required columns: {missing}")

    work = frame.copy()
    if "p_wide_raw" in work.columns:
        work["p_wide_raw"] = pd.to_numeric(work["p_wide_raw"], errors="coerce")
    elif "p_wide" in work.columns:
        work["p_wide_raw"] = pd.to_numeric(work["p_wide"], errors="coerce")
    else:
        raise SystemExit("pair-level input must contain p_wide or p_wide_raw")
    return work


def _prepare_pair_frame(
    frame: pd.DataFrame, *, mc_samples: int, seed: int
) -> tuple[pd.DataFrame, str]:
    columns = set(frame.columns)
    if {"horse_no_1", "horse_no_2"} <= columns and ({"p_wide", "p_wide_raw"} & columns):
        return _pair_from_pair_input(frame), "pair"
    if {"horse_no", "pl_score"} <= columns:
        return _pair_from_horse_input(frame, mc_samples=mc_samples, seed=seed), "horse"
    raise SystemExit("input must be horse-level (pl_score) or pair-level (p_wide/p_wide_raw)")


def _normalize_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no_1"] = pd.to_numeric(out["horse_no_1"], errors="coerce").astype("Int64")
    out["horse_no_2"] = pd.to_numeric(out["horse_no_2"], errors="coerce").astype("Int64")
    out["p_wide_raw"] = pd.to_numeric(out["p_wide_raw"], errors="coerce")
    out = out.dropna(subset=["race_id", "horse_no_1", "horse_no_2", "p_wide_raw"]).copy()
    if out.empty:
        raise SystemExit("No rows left after normalizing wide calibrator pair input.")
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no_1"] = out["horse_no_1"].astype(int)
    out["horse_no_2"] = out["horse_no_2"].astype(int)
    return out


def _select_years(
    frame: pd.DataFrame,
    *,
    years_arg: str,
    require_years_arg: str,
) -> tuple[pd.DataFrame, list[int], list[int], list[int]]:
    years_requested = str(years_arg).strip()
    years_required = str(require_years_arg).strip()

    if "valid_year" not in frame.columns:
        if years_requested or years_required:
            raise SystemExit("--years/--require-years requires 'valid_year' column in input")
        return frame.copy(), [], [], []

    out = frame.copy()
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
    return out, selected_years, selected_available, available


def _fetch_top3_horse_nos(db: Database, race_ids: list[int]) -> dict[int, set[int]]:
    rows = db.fetch_all(
        """
        SELECT
            run.race_id,
            run.horse_no
        FROM core.runner AS run
        JOIN core.result AS res
          ON res.race_id = run.race_id
         AND res.horse_id = run.horse_id
        WHERE run.race_id = ANY(%(race_ids)s)
          AND res.finish_pos <= 3
        """,
        {"race_ids": race_ids},
    )
    mapping: dict[int, set[int]] = {}
    for row in rows:
        mapping.setdefault(int(row["race_id"]), set()).add(int(row["horse_no"]))
    return mapping


def _attach_labels(pair: pd.DataFrame, top3_map: dict[int, set[int]]) -> pd.DataFrame:
    labeled = pair.copy()
    labels: list[int] = []
    for row in labeled.itertuples():
        winners = top3_map.get(int(row.race_id), set())
        labels.append(int(int(row.horse_no_1) in winners and int(row.horse_no_2) in winners))
    labeled["y_wide"] = labels
    return labeled


def _metric_payload(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"logloss": None, "brier": None, "auc": None}
    payload: dict[str, float | None] = {}
    try:
        payload["logloss"] = float(log_loss(y_true, np.clip(score, 1e-12, 1.0 - 1e-12)))
    except ValueError:
        payload["logloss"] = None
    try:
        payload["brier"] = float(brier_score_loss(y_true, score))
    except ValueError:
        payload["brier"] = None
    try:
        payload["auc"] = float(roc_auc_score(y_true, score))
    except ValueError:
        payload["auc"] = None
    return payload


def _prepare_dataset(
    *,
    path: Path,
    mc_samples: int,
    seed: int,
    years_arg: str = "",
    require_years_arg: str = "",
    filter_years: bool,
) -> tuple[pd.DataFrame, str, dict[str, object]]:
    raw_input = _load_input(path)
    pair_input, input_mode = _prepare_pair_frame(
        raw_input,
        mc_samples=int(mc_samples),
        seed=int(seed),
    )
    required_years = parse_years(str(require_years_arg))
    if filter_years:
        pair_input, _, selected_years, available_years = _select_years(
            pair_input,
            years_arg=str(years_arg),
            require_years_arg=str(require_years_arg),
        )
    else:
        if "valid_year" in pair_input.columns:
            valid_year = pd.to_numeric(pair_input["valid_year"], errors="coerce").astype("Int64")
            pair_input = pair_input.loc[valid_year.notna()].copy()
            pair_input["valid_year"] = valid_year.loc[pair_input.index].astype(int)
            available_years = sorted(pair_input["valid_year"].unique().tolist())
        else:
            available_years = []
        selected_years = list(available_years)
    pair_input = _normalize_pair_frame(pair_input)
    metadata = {
        "selected_years": list(map(int, selected_years)),
        "available_years": list(map(int, available_years)),
        "required_years": list(map(int, required_years)),
    }
    return pair_input, input_mode, metadata


def _dataset_metrics(frame: pd.DataFrame, *, year_meta: dict[str, object]) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "selected_years": list(map(int, year_meta.get("selected_years", []))),
        "available_years": list(map(int, year_meta.get("available_years", []))),
        "required_years": list(map(int, year_meta.get("required_years", []))),
        "raw": _metric_payload(
            frame["y_wide"].to_numpy(dtype=int),
            frame["p_wide_raw"].to_numpy(dtype=float),
        ),
        "calibrated": _metric_payload(
            frame["y_wide"].to_numpy(dtype=int),
            frame["p_wide"].to_numpy(dtype=float),
        ),
    }


def run_wide_calibrator(
    *,
    fit_input: str,
    apply_input: str | None = None,
    method: str = "isotonic",
    model_output: str = DEFAULT_MODEL_OUTPUT,
    meta_output: str = DEFAULT_META_OUTPUT,
    pred_output: str = DEFAULT_PRED_OUTPUT,
    metrics_output: str = DEFAULT_METRICS_OUTPUT,
    database_url: str = "",
    years: str = "",
    require_years: str = "",
    log_level: str = "INFO",
) -> int:
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    fit_input_raw = str(fit_input).strip()
    if not fit_input_raw:
        raise SystemExit("--fit-input is required")
    fit_input_path = resolve_path(fit_input_raw)
    apply_input_path = resolve_path(str(apply_input).strip() if apply_input else fit_input_raw)
    _m = str(method)
    model_output_path = _resolve_output_path(
        model_output, DEFAULT_MODEL_OUTPUT, method=_m, artifact_suffix=""
    )
    meta_output_path = _resolve_output_path(
        meta_output, DEFAULT_META_OUTPUT, method=_m, artifact_suffix=""
    )
    pred_output_path = _resolve_output_path(
        pred_output, DEFAULT_PRED_OUTPUT, method=_m, artifact_suffix=""
    )
    metrics_output_path = _resolve_output_path(
        metrics_output, DEFAULT_METRICS_OUTPUT, method=_m, artifact_suffix=""
    )
    database_url_resolved = resolve_database_url(database_url)

    fit_input_data, fit_input_mode, fit_year_meta = _prepare_dataset(
        path=fit_input_path,
        mc_samples=10_000,
        seed=42,
        years_arg=str(years),
        require_years_arg=str(require_years),
        filter_years=True,
    )
    apply_input_data, apply_input_mode, apply_year_meta = _prepare_dataset(
        path=apply_input_path,
        mc_samples=10_000,
        seed=42,
        filter_years=False,
    )
    race_ids = sorted(
        set(fit_input_data["race_id"].unique().tolist())
        | set(apply_input_data["race_id"].unique().tolist())
    )

    with Database(connection_string=database_url_resolved) as db:
        top3_map = _fetch_top3_horse_nos(db, race_ids)

    fit_labeled = _attach_labels(fit_input_data, top3_map)
    apply_labeled = _attach_labels(apply_input_data, top3_map)
    bundle = fit_wide_pair_calibrator(
        fit_labeled["p_wide_raw"].to_numpy(dtype=float),
        fit_labeled["y_wide"].to_numpy(dtype=float),
        method=str(method),
    )
    fit_calibrated = apply_wide_pair_calibrator(fit_labeled, bundle)
    holdout_eval = apply_wide_pair_calibrator(apply_labeled, bundle)
    if "p_wide" not in fit_calibrated.columns or "p_wide" not in holdout_eval.columns:
        raise SystemExit("wide pair calibrator did not produce p_wide")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output_path)
    holdout_eval.to_parquet(pred_output_path, index=False)

    metrics = {
        "method": str(method),
        "fit_input_mode": fit_input_mode,
        "apply_input_mode": apply_input_mode,
        "fit": _dataset_metrics(fit_calibrated, year_meta=fit_year_meta),
        "holdout_eval": _dataset_metrics(holdout_eval, year_meta=apply_year_meta),
        "artifact_suffix": "",
    }
    save_json(metrics_output_path, metrics)

    meta = {
        "created_at": bundle["created_at"],
        "input_path": str(fit_input_path),
        "fit_input_path": str(fit_input_path),
        "apply_input_path": str(apply_input_path),
        "input_mode": fit_input_mode,
        "fit_input_mode": fit_input_mode,
        "apply_input_mode": apply_input_mode,
        "method": str(method),
        "artifact_suffix": "",
        "database_url_env_priority": ["V3_DATABASE_URL"],
        "model_output": str(model_output_path),
        "pred_output": str(pred_output_path),
        "metrics_output": str(metrics_output_path),
        "fit": {
            **fit_year_meta,
            "rows": int(len(fit_calibrated)),
            "races": int(fit_calibrated["race_id"].nunique()) if not fit_calibrated.empty else 0,
        },
        "holdout_eval": {
            **apply_year_meta,
            "rows": int(len(holdout_eval)),
            "races": int(holdout_eval["race_id"].nunique()) if not holdout_eval.empty else 0,
        },
        "metrics": metrics,
        "code_hash": hash_files(
            [
                Path(__file__),
                Path(resolve_path("src/keiba_research/evaluation/wide_pair_calibration.py")),
                Path(resolve_path("src/keiba_research/evaluation/pl_common.py")),
            ]
        ),
    }
    save_json(meta_output_path, meta)

    logger.info(
        "trained wide pair calibrator fit_rows=%s fit_races=%s holdout_rows=%s holdout_races=%s",
        len(fit_calibrated),
        metrics["fit"]["races"],
        len(holdout_eval),
        metrics["holdout_eval"]["races"],
    )
    logger.info("wrote %s", model_output_path)
    logger.info("wrote %s", pred_output_path)
    logger.info("wrote %s", metrics_output_path)
    logger.info("wrote %s", meta_output_path)
    return 0

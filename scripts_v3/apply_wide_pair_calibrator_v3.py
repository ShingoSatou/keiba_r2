#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.pl_v3_common import estimate_p_wide_by_race  # noqa: E402
from scripts_v3.train_wide_pair_calibrator_v3 import (  # noqa: E402
    _load_input,
    _pair_from_pair_input,
)
from scripts_v3.v3_common import resolve_path  # noqa: E402
from scripts_v3.wide_pair_calibration_v3 import apply_wide_pair_calibrator  # noqa: E402

logger = logging.getLogger(__name__)




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


def run_apply_wide_calibrator(
    *,
    input: str,
    model: str = "models/wide_pair_calibrator_v3.joblib",
    output: str = "data/predictions/race_v3_wide_calibrated.parquet",
    mc_samples: int = 10_000,
    seed: int = 42,
    log_level: str = "INFO",
) -> int:
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    input_path = resolve_path(input)
    model_path = resolve_path(model)
    output_path = resolve_path(output)
    if not model_path.exists():
        raise SystemExit(f"calibrator model not found: {model_path}")

    frame = _load_input(input_path)
    if {"horse_no_1", "horse_no_2"} <= set(frame.columns) and (
        {"p_wide", "p_wide_raw"} & set(frame.columns)
    ):
        pair = _pair_from_pair_input(frame)
    elif {"horse_no", "pl_score"} <= set(frame.columns):
        pair = _pair_from_horse_input(
            frame,
            mc_samples=int(mc_samples),
            seed=int(seed),
        )
    else:
        raise SystemExit("input must be horse-level (pl_score) or pair-level (p_wide/p_wide_raw)")

    bundle = joblib.load(model_path)
    calibrated = apply_wide_pair_calibrator(pair, bundle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(output_path, index=False)
    logger.info("wrote %s", output_path)
    return 0

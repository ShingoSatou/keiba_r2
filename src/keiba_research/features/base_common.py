from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from keiba_research.common.v3_utils import hash_files, save_json
from keiba_research.db.database import Database

DEFAULT_ALLOWED_CONDITION_CODES = {10, 16, 999}
DEFAULT_ALLOWED_RACE_TYPE_CODES = {13, 14}


def apply_segment_filter(
    df: pd.DataFrame,
    *,
    numeric_code_fn,
) -> pd.DataFrame:
    race_type = numeric_code_fn(df["race_type_code"])
    condition = numeric_code_fn(df["condition_code_min_age"])
    field_size = numeric_code_fn(df["field_size"])
    distance = numeric_code_fn(df["distance_m"])
    horse_no = numeric_code_fn(df["horse_no"])
    finish_pos = numeric_code_fn(df["finish_pos"])

    mask = (
        (numeric_code_fn(df["track_code"]).between(1, 10))
        & (numeric_code_fn(df["surface"]) == 2)
        & (race_type.isin(sorted(DEFAULT_ALLOWED_RACE_TYPE_CODES)))
        & (~condition.isin([701, 702, 703]))
        & (condition.isin(DEFAULT_ALLOWED_CONDITION_CODES))
        & (distance > 0)
        & (field_size > 0)
        & (horse_no.between(1, 18))
        & (finish_pos.notna())
    )
    return df.loc[mask].copy()


def load_base_data(
    db: Database,
    *,
    history_from: date,
    to_date: date,
    build_race_datetime_fn,
    distance_to_bucket_fn,
    going_to_bucket_fn,
) -> pd.DataFrame:
    query = """
    SELECT
        r.race_id,
        r.race_date,
        r.track_code,
        r.race_no,
        r.surface,
        r.distance_m,
        r.going,
        r.weather,
        r.field_size,
        r.start_time,
        r.class_code,
        r.grade_code,
        r.race_type_code,
        r.weight_type_code,
        r.condition_code_min_age,
        r.condition_code_2yo,
        r.condition_code_3yo,
        r.condition_code_4yo,
        r.condition_code_5up,
        r.condition_code_min_age_raw,
        run.horse_id,
        run.horse_no,
        run.gate,
        run.jockey_id,
        run.trainer_id,
        run.carried_weight,
        run.body_weight,
        run.body_weight_diff,
        run.jockey_code_raw,
        run.trainer_code_raw,
        run.sex,
        res.finish_pos,
        res.time_sec,
        res.final3f_sec
    FROM core.race r
    JOIN core.runner run
      ON r.race_id = run.race_id
    JOIN core.result res
      ON run.race_id = res.race_id
     AND run.horse_id = res.horse_id
    WHERE r.track_code BETWEEN 1 AND 10
      AND r.race_date BETWEEN %(history_from)s AND %(to_date)s
    ORDER BY r.race_date, r.race_id, run.horse_no
    """
    rows = db.fetch_all(query, {"history_from": history_from, "to_date": to_date})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["race_datetime"] = build_race_datetime_fn(frame["race_date"], frame["start_time"])
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
    frame["distance_bucket"] = frame["distance_m"].map(distance_to_bucket_fn)
    frame["going_bucket"] = frame["going"].map(going_to_bucket_fn)
    return frame


def build_segment_filter_payload() -> dict[str, Any]:
    return {
        "track_code": "01-10",
        "surface": 2,
        "race_type_code": sorted(DEFAULT_ALLOWED_RACE_TYPE_CODES),
        "condition_code_excluded": [701, 702, 703],
        "condition_code_allowed": sorted(DEFAULT_ALLOWED_CONDITION_CODES),
    }


def build_features_base_meta(
    *,
    features: pd.DataFrame,
    database_url: str,
    from_date: date,
    to_date: date,
    history_days: int,
    with_te: bool,
    missing_rate_columns: list[str],
    base_distance_m: float,
    window_2y_days: int,
    shrinkage_k: float,
    code_hash_paths: list[Path],
) -> dict[str, Any]:
    missing_rate = {
        col: float(features[col].isna().mean())
        for col in missing_rate_columns
        if col in features.columns
    }
    return {
        "from_date": str(from_date),
        "to_date": str(to_date),
        "history_days": history_days,
        "with_te": bool(with_te),
        "database_url_env_priority": ["V3_DATABASE_URL"],
        "rows": int(len(features)),
        "races": int(features["race_id"].nunique()) if not features.empty else 0,
        "columns": features.columns.tolist(),
        "history_scope": {
            "track_code": "01-10",
            "history_from_date": str(from_date - timedelta(days=history_days)),
            "history_to_date": str(to_date),
            "source_tables": ["core.race", "core.runner", "core.result"],
        },
        "segment_filter": build_segment_filter_payload(),
        "speed_index": {
            "base_distance_m": base_distance_m,
            "window_days": window_2y_days,
            "shrinkage_k": shrinkage_k,
        },
        "missing_rate": missing_rate,
        "code_hash": hash_files(code_hash_paths),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def write_features_base_outputs(
    *,
    features: pd.DataFrame,
    output_path: Path,
    meta_path: Path,
    meta_payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    save_json(meta_path, meta_payload)

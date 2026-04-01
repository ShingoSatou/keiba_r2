#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3 import rebuild_v3_jsonl_common as rebuild_common  # noqa: E402
from scripts_v3.v3_common import resolve_database_url, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_FROM_DATE = "2016-01-01"

DEFAULT_COMMIT_INTERVAL = rebuild_common.DEFAULT_COMMIT_INTERVAL
DEFAULT_INPUT_DIR = rebuild_common.DEFAULT_INPUT_DIR
DEFAULT_TARGET_CONDITION_CODES = rebuild_common.DEFAULT_TARGET_CONDITION_CODES
RA_COND_CODE_2YO_LEN = rebuild_common.RA_COND_CODE_2YO_LEN
RA_COND_CODE_2YO_START = rebuild_common.RA_COND_CODE_2YO_START
RA_COND_CODE_3YO_LEN = rebuild_common.RA_COND_CODE_3YO_LEN
RA_COND_CODE_3YO_START = rebuild_common.RA_COND_CODE_3YO_START
RA_COND_CODE_4YO_LEN = rebuild_common.RA_COND_CODE_4YO_LEN
RA_COND_CODE_4YO_START = rebuild_common.RA_COND_CODE_4YO_START
RA_COND_CODE_5UP_LEN = rebuild_common.RA_COND_CODE_5UP_LEN
RA_COND_CODE_5UP_START = rebuild_common.RA_COND_CODE_5UP_START
RA_COND_CODE_MIN_AGE_LEN = rebuild_common.RA_COND_CODE_MIN_AGE_LEN
RA_COND_CODE_MIN_AGE_START = rebuild_common.RA_COND_CODE_MIN_AGE_START
RA_MONTHDAY_LEN = rebuild_common.RA_MONTHDAY_LEN
RA_MONTHDAY_START = rebuild_common.RA_MONTHDAY_START
RA_RACE_NO_LEN = rebuild_common.RA_RACE_NO_LEN
RA_RACE_NO_START = rebuild_common.RA_RACE_NO_START
RA_TRACK_CODE_LEN = rebuild_common.RA_TRACK_CODE_LEN
RA_TRACK_CODE_START = rebuild_common.RA_TRACK_CODE_START
RA_YEAR_LEN = rebuild_common.RA_YEAR_LEN
RA_YEAR_START = rebuild_common.RA_YEAR_START
_parse_condition_codes = rebuild_common._parse_condition_codes
_parse_ra_age_condition_payload = rebuild_common.parse_ra_age_condition_payload
_race_id_in_history_scope = rebuild_common.race_id_in_history_scope
rebuild_v3_database = rebuild_common.rebuild_v3_database


def _parse_iso_date(raw: str, *, option_name: str) -> date:
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"invalid {option_name}: {raw}") from exc


def run_rebuild(
    *,
    database_url: str = "",
    input_pattern: str | None = None,
    input_dir: str = DEFAULT_INPUT_DIR,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = "",
    o1_consolidated_date: str = "",
    condition_codes: str = ",".join(str(code) for code in DEFAULT_TARGET_CONDITION_CODES),
    summary_output: str = "data/reports/keiba_v3_rebuild_summary.json",
    commit_interval: int = DEFAULT_COMMIT_INTERVAL,
    log_level: str = "INFO",
) -> int:
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    from_date_parsed = _parse_iso_date(from_date, option_name="--from-date")
    to_date_parsed = (
        _parse_iso_date(to_date, option_name="--to-date")
        if str(to_date).strip()
        else None
    )
    if to_date_parsed is not None and to_date_parsed < from_date_parsed:
        raise SystemExit("--to-date must be greater than or equal to --from-date")

    database_url_resolved = resolve_database_url(database_url)
    summary_output_path = resolve_path(summary_output)
    input_pattern_resolved = str(input_pattern).strip() if input_pattern else None
    input_dir_resolved = resolve_path(input_dir) if str(input_dir).strip() else None
    condition_codes_parsed = _parse_condition_codes(condition_codes)

    rebuild_v3_database(
        database_url=database_url_resolved,
        input_pattern=input_pattern_resolved,
        input_dir=input_dir_resolved,
        o1_consolidated_date=str(o1_consolidated_date).strip() or None,
        from_date=from_date_parsed,
        to_date=to_date_parsed,
        condition_codes=condition_codes_parsed,
        summary_output=summary_output_path,
        commit_interval=int(commit_interval),
    )
    return 0

#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild keiba_v3 directly from JSONL inputs with broad central history tables "
            "and target-segment-only odds/mining tables."
        )
    )
    parser.add_argument("--database-url", default="", help="Target keiba_v3 database URL.")
    parser.add_argument(
        "--target-db-url",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--input", default="", help="Optional JSONL input path or glob.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing canonical v3 rebuild JSONL inputs.",
    )
    parser.add_argument("--from-date", default=DEFAULT_FROM_DATE)
    parser.add_argument("--to-date", default="")
    parser.add_argument(
        "--o1-date",
        default="",
        help="Pinned consolidated O1 snapshot date (YYYYMMDD for 0B41_ALL_<date>.jsonl).",
    )
    parser.add_argument(
        "--condition-codes",
        default=",".join(str(code) for code in DEFAULT_TARGET_CONDITION_CODES),
        help="Comma-separated condition_code_min_age values for target-segment-only tables.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/reports/keiba_v3_rebuild_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=DEFAULT_COMMIT_INTERVAL,
        help="Commit interval while ingesting JSONL records.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if str(args.source_db_url).strip():
        raise SystemExit(
            "--source-db-url is no longer supported. Rebuild v3 from JSONL inputs via "
            "--input/--input-dir instead."
        )
    if str(args.target_db_url).strip() and not str(args.database_url).strip():
        args.database_url = args.target_db_url
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    from_date = _parse_iso_date(args.from_date, option_name="--from-date")
    to_date = (
        _parse_iso_date(args.to_date, option_name="--to-date")
        if str(args.to_date).strip()
        else None
    )
    if to_date is not None and to_date < from_date:
        raise SystemExit("--to-date must be greater than or equal to --from-date")

    database_url = resolve_database_url(args.database_url)
    summary_output = resolve_path(args.summary_output)
    input_pattern = str(args.input).strip() or None
    input_dir = resolve_path(args.input_dir) if str(args.input_dir).strip() else None
    condition_codes = _parse_condition_codes(args.condition_codes)

    rebuild_v3_database(
        database_url=database_url,
        input_pattern=input_pattern,
        input_dir=input_dir,
        o1_consolidated_date=str(args.o1_date).strip() or None,
        from_date=from_date,
        to_date=to_date,
        condition_codes=condition_codes,
        summary_output=summary_output,
        commit_interval=int(args.commit_interval),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

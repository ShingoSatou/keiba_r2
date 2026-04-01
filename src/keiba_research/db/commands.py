from __future__ import annotations

import argparse
from pathlib import Path

from keiba_research.common.assets import (
    cache_root,
    ensure_json_has_no_absolute_paths,
    jsonl_root,
    rewrite_json_asset_paths,
)
from keiba_research.db.migrate import run_migrate
from keiba_research.db.rebuild import run_rebuild


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="db_command", required=True)

    migrate = subparsers.add_parser("migrate", help="Apply keiba_v3 migrations.")
    migrate.add_argument("--database-url", default="")
    migrate.set_defaults(handler=handle_migrate)

    rebuild = subparsers.add_parser("rebuild", help="Rebuild keiba_v3 from pinned JSONL inputs.")
    rebuild.add_argument("--database-url", default="")
    rebuild.add_argument("--input-dir", default="")
    rebuild.add_argument("--from-date", default="2016-01-01")
    rebuild.add_argument("--to-date", default="")
    rebuild.add_argument("--condition-codes", default="10,16,999")
    rebuild.add_argument("--summary-output", default="")
    rebuild.add_argument(
        "--o1-date",
        required=True,
        help="Pinned consolidated O1 snapshot date (YYYYMMDD for 0B41_ALL_<date>.jsonl).",
    )
    rebuild.set_defaults(handler=handle_rebuild)


def handle_migrate(args: argparse.Namespace) -> int:
    return int(run_migrate(database_url=str(args.database_url).strip()))


def handle_rebuild(args: argparse.Namespace) -> int:
    input_dir = (
        Path(str(args.input_dir).strip()).resolve() if str(args.input_dir).strip() else jsonl_root()
    )
    summary_output = (
        Path(str(args.summary_output).strip()).resolve()
        if str(args.summary_output).strip()
        else cache_root() / f"keiba_v3_rebuild_summary_{str(args.o1_date).strip()}.json"
    )
    rc = run_rebuild(
        database_url=str(args.database_url).strip(),
        input_dir=str(input_dir),
        summary_output=str(summary_output),
        from_date=str(args.from_date),
        to_date=str(args.to_date).strip(),
        condition_codes=str(args.condition_codes),
        o1_consolidated_date=str(args.o1_date).strip(),
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(summary_output)
    ensure_json_has_no_absolute_paths(summary_output)
    return 0

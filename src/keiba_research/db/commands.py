from __future__ import annotations

import argparse
from pathlib import Path

from keiba_research.common.assets import (
    cache_root,
    ensure_json_has_no_absolute_paths,
    jsonl_root,
    rewrite_json_asset_paths,
)
from scripts_v3.migrate_v3 import main as migrate_v3_main
from scripts_v3.rebuild_v3_db import main as rebuild_v3_main


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
    argv: list[str] = []
    if str(args.database_url).strip():
        argv.extend(["--database-url", str(args.database_url).strip()])
    return int(migrate_v3_main(argv))


def handle_rebuild(args: argparse.Namespace) -> int:
    input_dir = (
        Path(str(args.input_dir).strip()).resolve() if str(args.input_dir).strip() else jsonl_root()
    )
    summary_output = (
        Path(str(args.summary_output).strip()).resolve()
        if str(args.summary_output).strip()
        else cache_root() / f"keiba_v3_rebuild_summary_{str(args.o1_date).strip()}.json"
    )
    argv = [
        "--input-dir",
        str(input_dir),
        "--summary-output",
        str(summary_output),
        "--from-date",
        str(args.from_date),
        "--condition-codes",
        str(args.condition_codes),
        "--o1-date",
        str(args.o1_date),
    ]
    if str(args.database_url).strip():
        argv.extend(["--database-url", str(args.database_url).strip()])
    if str(args.to_date).strip():
        argv.extend(["--to-date", str(args.to_date).strip()])
    rc = int(rebuild_v3_main(argv))
    if rc != 0:
        return rc
    rewrite_json_asset_paths(summary_output)
    ensure_json_has_no_absolute_paths(summary_output)
    return 0

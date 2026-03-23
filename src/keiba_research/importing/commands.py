from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from keiba_research.common.assets import rewrite_json_asset_paths, study_paths
from keiba_research.common.state import update_study_config


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="import_command", required=True)

    legacy = subparsers.add_parser(
        "legacy-tuning", help="Import a legacy tuning result as read-only seed."
    )
    legacy.add_argument("--study-id", required=True)
    legacy.add_argument("--kind", choices=["binary", "stack"], required=True)
    legacy.add_argument("--source-storage", default="")
    legacy.add_argument("--source-best", default="")
    legacy.add_argument("--source-best-params", required=True)
    legacy.add_argument("--task", default="")
    legacy.add_argument("--model", default="")
    legacy.set_defaults(handler=handle_legacy_tuning)


def _copy_if_present(source: str, destination: Path) -> None:
    raw = str(source).strip()
    if not raw:
        return
    src = Path(raw).resolve()
    if not src.exists():
        raise SystemExit(f"legacy import source not found: {src}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)


def _normalize_imported_study_id(study_id: str) -> str:
    study = str(study_id).strip()
    if not study:
        raise SystemExit("study_id is required")
    return study if study.startswith("imported.") else f"imported.{study}"


def handle_legacy_tuning(args: argparse.Namespace) -> int:
    normalized_study_id = _normalize_imported_study_id(str(args.study_id))
    study = study_paths(normalized_study_id)
    _copy_if_present(str(args.source_storage), study["storage"])
    _copy_if_present(str(args.source_best), study["best"])
    _copy_if_present(str(args.source_best_params), study["selected_trial"])
    rewrite_json_asset_paths(study["best"])
    rewrite_json_asset_paths(study["selected_trial"])
    update_study_config(
        normalized_study_id,
        {
            "study_id": normalized_study_id,
            "kind": str(args.kind),
            "task": str(args.task).strip() or None,
            "model": str(args.model).strip() or None,
            "imported": True,
            "read_only_seed": True,
            "origin_type": "legacy_seed",
        },
    )
    return 0

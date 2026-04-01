from __future__ import annotations

import argparse

from keiba_research.common.run_config import generate_config_from_study, load_run_config
from keiba_research.common.state import write_toml


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="config_command", required=True)

    from_study = subparsers.add_parser(
        "from-study",
        help="Generate a run_config.toml section from a study's selected_trial.json.",
    )
    from_study.add_argument("--study-id", required=True)
    from_study.add_argument(
        "--output",
        default="",
        help="Output path for the TOML file. Prints to stdout if omitted.",
    )
    from_study.add_argument(
        "--merge-into",
        default="",
        help="Existing TOML config to merge the study params into.",
    )
    from_study.set_defaults(handler=handle_from_study)

    show = subparsers.add_parser("show", help="Display a run_config.toml.")
    show.add_argument("path", help="Path to a run_config.toml file.")
    show.set_defaults(handler=handle_show)


def handle_from_study(args: argparse.Namespace) -> int:
    config = generate_config_from_study(str(args.study_id))

    if str(args.merge_into).strip():
        base = load_run_config(str(args.merge_into))
        _deep_merge(base, config)
        config = base

    if str(args.output).strip():
        from pathlib import Path

        write_toml(Path(str(args.output)), config)
        print(f"wrote {args.output}")
    else:
        from keiba_research.common.state import _dump_toml_sections

        for line in _dump_toml_sections(config):
            print(line)
    return 0


def handle_show(args: argparse.Namespace) -> int:
    config = load_run_config(str(args.path))
    from keiba_research.common.state import _dump_toml_sections

    for line in _dump_toml_sections(config):
        print(line)
    return 0


def _deep_merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

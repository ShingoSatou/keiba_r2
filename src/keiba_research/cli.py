from __future__ import annotations

import argparse

from keiba_research.config import commands as config_commands
from keiba_research.db import commands as db_commands
from keiba_research.evaluation import commands as evaluation_commands
from keiba_research.features import commands as features_commands
from keiba_research.importing import commands as importing_commands
from keiba_research.training import commands as training_commands
from keiba_research.tuning import commands as tuning_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m keiba_research",
        description=(
            "Research-first CLI for v3 rebuild, features, training, tuning, "
            "and evaluation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db", help="DB commands")
    db_commands.register(db_parser)

    features_parser = subparsers.add_parser("features", help="Feature build commands")
    features_commands.register(features_parser)

    train_parser = subparsers.add_parser("train", help="Training commands")
    training_commands.register(train_parser)

    tune_parser = subparsers.add_parser("tune", help="Tuning commands")
    tuning_commands.register(tune_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    evaluation_commands.register(eval_parser)

    config_parser = subparsers.add_parser("config", help="Config management commands")
    config_commands.register(config_parser)

    import_parser = subparsers.add_parser("import", help="Legacy import commands")
    importing_commands.register(import_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))

from __future__ import annotations

import argparse

from keiba_research.common.assets import (
    ensure_json_has_no_absolute_paths,
    feature_build_paths,
    rewrite_json_asset_paths,
    run_paths,
    study_paths,
)
from keiba_research.common.state import load_study_config, update_study_config
from keiba_research.training.cv_policy import (
    DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    DEFAULT_TRAIN_WINDOW_YEARS,
)
from keiba_research.tuning.binary import run_tune_binary
from keiba_research.tuning.stacker import run_tune_stacker


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="tune_command", required=True)

    binary = subparsers.add_parser("binary", help="Run or resume a binary Optuna study.")
    binary.add_argument("--study-id", required=True)
    binary.add_argument("--task", choices=["win", "place"], default="win")
    binary.add_argument("--model", choices=["lgbm", "xgb", "cat"], default="lgbm")
    binary.add_argument("--feature-profile", required=True)
    binary.add_argument("--feature-build-id", required=True)
    binary.add_argument("--holdout-year", type=int, default=2025)
    binary.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    binary.add_argument("--n-trials", type=int, default=300)
    binary.add_argument("--timeout", type=int, default=0)
    binary.add_argument("--seed", type=int, default=42)
    binary.add_argument("--log-level", default="INFO")
    binary.set_defaults(handler=handle_binary)

    stack = subparsers.add_parser("stack", help="Run or resume a stacker Optuna study.")
    stack.add_argument("--study-id", required=True)
    stack.add_argument("--task", choices=["win", "place"], default="win")
    stack.add_argument("--source-run-id", required=True)
    stack.add_argument("--feature-profile", required=True)
    stack.add_argument("--feature-build-id", required=True)
    stack.add_argument("--holdout-year", type=int, default=2025)
    stack.add_argument(
        "--min-train-years",
        type=int,
        default=DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    )
    stack.add_argument(
        "--max-train-years",
        type=int,
        default=DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    )
    stack.add_argument("--n-trials", type=int, default=300)
    stack.add_argument("--timeout", type=int, default=0)
    stack.add_argument("--seed", type=int, default=42)
    stack.add_argument("--log-level", default="INFO")
    stack.set_defaults(handler=handle_stack)


def _assert_study_writable(study_id: str) -> None:
    config = load_study_config(study_id)
    if bool(config.get("read_only_seed")):
        raise SystemExit(f"study {study_id} is read_only_seed and cannot be resumed")


def handle_binary(args: argparse.Namespace) -> int:
    _assert_study_writable(args.study_id)
    feature_paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    study = study_paths(args.study_id)
    update_study_config(
        args.study_id,
        {
            "study_id": str(args.study_id),
            "kind": "binary",
            "task": str(args.task),
            "model": str(args.model),
            "feature_profile": str(args.feature_profile),
            "feature_build_id": str(args.feature_build_id),
            "read_only_seed": False,
            "imported": False,
        },
    )
    rc = int(
        run_tune_binary(
            task=str(args.task),
            model=str(args.model),
            input_base=str(feature_paths["features"]),
            input_te=str(feature_paths["features_te"]),
            study_name=str(args.study_id),
            storage=str(study["storage"]),
            trials_output=str(study["trials"]),
            best_output=str(study["best"]),
            best_params_output=str(study["selected_trial"]),
            holdout_year=int(args.holdout_year),
            train_window_years=int(args.train_window_years),
            n_trials=int(args.n_trials),
            timeout=int(args.timeout),
            seed=int(args.seed),
            log_level=str(args.log_level),
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(study["best"])
    rewrite_json_asset_paths(study["selected_trial"])
    ensure_json_has_no_absolute_paths(study["best"])
    ensure_json_has_no_absolute_paths(study["selected_trial"])
    return 0


def handle_stack(args: argparse.Namespace) -> int:
    _assert_study_writable(args.study_id)
    feature_paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    source = run_paths(args.source_run_id)
    study = study_paths(args.study_id)
    task = str(args.task)
    update_study_config(
        args.study_id,
        {
            "study_id": str(args.study_id),
            "kind": "stack",
            "task": task,
            "source_run_id": str(args.source_run_id),
            "feature_profile": str(args.feature_profile),
            "feature_build_id": str(args.feature_build_id),
            "read_only_seed": False,
            "imported": False,
        },
    )
    rc = int(
        run_tune_stacker(
            task=task,
            features_input=str(feature_paths["features"]),
            lgbm_oof=str(source["oof"] / f"{task}_lgbm_oof.parquet"),
            xgb_oof=str(source["oof"] / f"{task}_xgb_oof.parquet"),
            cat_oof=str(source["oof"] / f"{task}_cat_oof.parquet"),
            study_name=str(args.study_id),
            storage=str(study["storage"]),
            trials_output=str(study["trials"]),
            best_output=str(study["best"]),
            best_params_output=str(study["selected_trial"]),
            holdout_year=int(args.holdout_year),
            min_train_years=int(args.min_train_years),
            max_train_years=int(args.max_train_years),
            n_trials=int(args.n_trials),
            timeout=int(args.timeout),
            seed=int(args.seed),
            log_level=str(args.log_level),
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(study["best"])
    rewrite_json_asset_paths(study["selected_trial"])
    ensure_json_has_no_absolute_paths(study["best"])
    ensure_json_has_no_absolute_paths(study["selected_trial"])
    return 0

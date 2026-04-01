from __future__ import annotations

import argparse

from keiba_research.common.assets import (
    asset_relative,
    ensure_json_has_no_absolute_paths,
    feature_build_paths,
    read_json,
    rewrite_json_asset_paths,
    run_paths,
    study_paths,
)
from keiba_research.common.run_config import load_run_config, save_resolved_params
from keiba_research.common.state import (
    asset_payload,
    update_run_bundle,
    update_run_config,
    update_run_metrics,
)
from scripts_v3.cv_policy_v3 import (
    DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    DEFAULT_TRAIN_WINDOW_YEARS,
)
from scripts_v3.feature_registry_v3 import STACK_LIKE_PL_FEATURE_PROFILES
from scripts_v3.train_binary_model_v3 import main as train_binary_main
from scripts_v3.train_pl_v3 import main as train_pl_main
from scripts_v3.train_stacker_v3 import main as train_stacker_main
from scripts_v3.train_wide_pair_calibrator_v3 import main as train_wide_calibrator_main


class _StoreAndMarkSpecified(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def _binary_model_ext(model: str) -> str:
    if str(model) == "lgbm":
        return "txt"
    if str(model) == "xgb":
        return "json"
    if str(model) == "cat":
        return "cbm"
    raise SystemExit(f"unsupported binary model: {model}")


def _load_run_config(run_id: str) -> dict[str, object]:
    run = run_paths(run_id)
    if not run["config"].exists():
        return {}
    import tomllib

    return tomllib.loads(run["config"].read_text(encoding="utf-8"))


def _finalize_metadata(*paths) -> None:
    for path in paths:
        rewrite_json_asset_paths(path)
        ensure_json_has_no_absolute_paths(path)


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="train_command", required=True)

    binary = subparsers.add_parser("binary", help="Train one binary model into a run bundle.")
    binary.add_argument("--run-id", required=True)
    binary.add_argument("--task", choices=["win", "place"], default="win")
    binary.add_argument("--model", choices=["lgbm", "xgb", "cat"], default="lgbm")
    binary.add_argument("--feature-profile", required=True)
    binary.add_argument("--feature-build-id", required=True)
    binary.add_argument("--feature-set", choices=["base", "te"], default="base")
    binary.add_argument("--config", default="", help="run_config.toml (exclusive with --study-id)")
    binary.add_argument("--study-id", default="")
    binary.add_argument("--holdout-year", type=int, default=2025)
    binary.set_defaults(train_window_years_explicit=False)
    binary.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        action=_StoreAndMarkSpecified,
    )
    binary.add_argument("--database-url", default="")
    binary.add_argument("--log-level", default="INFO")
    binary.set_defaults(handler=handle_binary)

    stack = subparsers.add_parser("stack", help="Train one stacker into a run bundle.")
    stack.add_argument("--run-id", required=True)
    stack.add_argument("--task", choices=["win", "place"], default="win")
    stack.add_argument("--feature-profile", required=True)
    stack.add_argument("--feature-build-id", required=True)
    stack.add_argument("--source-run-id", default="")
    stack.add_argument("--config", default="", help="run_config.toml (exclusive with --study-id)")
    stack.add_argument("--study-id", default="")
    stack.add_argument("--holdout-year", type=int, default=2025)
    stack.set_defaults(min_train_years_explicit=False, max_train_years_explicit=False)
    stack.add_argument(
        "--min-train-years",
        type=int,
        default=DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
        action=_StoreAndMarkSpecified,
    )
    stack.add_argument(
        "--max-train-years",
        type=int,
        default=DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
        action=_StoreAndMarkSpecified,
    )
    stack.add_argument("--log-level", default="INFO")
    stack.set_defaults(handler=handle_stack)

    pl = subparsers.add_parser("pl", help="Train the PL layer into a run bundle.")
    pl.add_argument("--run-id", required=True)
    pl.add_argument("--feature-profile", required=True)
    pl.add_argument("--feature-build-id", required=True)
    pl.add_argument("--source-run-id", default="")
    pl.add_argument(
        "--pl-feature-profile",
        choices=list(STACK_LIKE_PL_FEATURE_PROFILES),
        default="stack_default",
    )
    pl.add_argument("--holdout-year", type=int, default=2025)
    pl.set_defaults(train_window_years_explicit=False)
    pl.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        action=_StoreAndMarkSpecified,
    )
    pl.add_argument("--log-level", default="INFO")
    pl.set_defaults(handler=handle_pl)

    wide = subparsers.add_parser(
        "wide-calibrator", help="Train wide pair calibrator into a run bundle."
    )
    wide.add_argument("--run-id", required=True)
    wide.add_argument("--source-run-id", default="")
    wide.add_argument("--method", choices=["isotonic", "logreg"], default="isotonic")
    wide.add_argument("--years", default="")
    wide.add_argument("--require-years", default="")
    wide.add_argument("--database-url", default="")
    wide.add_argument("--log-level", default="INFO")
    wide.set_defaults(handler=handle_wide_calibrator)


def _study_params_path(study_id: str) -> str:
    return str(study_paths(study_id)["selected_trial"])


def _validate_config_exclusivity(args: argparse.Namespace) -> None:
    if str(getattr(args, "config", "")).strip() and str(getattr(args, "study_id", "")).strip():
        raise SystemExit("--config and --study-id are mutually exclusive")


def _load_config_section(config_path: str, *keys: str) -> dict[str, object]:
    """Load a nested section from a run_config.toml."""
    config = load_run_config(config_path)
    section: object = config
    for key in keys:
        if not isinstance(section, dict):
            return {}
        section = section.get(key, {})
    return dict(section) if isinstance(section, dict) else {}


def _apply_config_to_argv(
    argv: list[str],
    section: dict[str, object],
    *,
    flag_map: dict[str, str],
) -> list[str]:
    """Append config values as CLI flags (only if not already present)."""
    existing_flags = set(argv)
    for param, flag in flag_map.items():
        if param in section and flag not in existing_flags:
            argv.extend([flag, str(section[param])])
    return argv


_BINARY_CONFIG_FLAGS: dict[str, str] = {
    "learning_rate": "--learning-rate",
    "num_leaves": "--num-leaves",
    "min_data_in_leaf": "--min-data-in-leaf",
    "lambda_l1": "--lambda-l1",
    "lambda_l2": "--lambda-l2",
    "feature_fraction": "--feature-fraction",
    "bagging_fraction": "--bagging-fraction",
    "bagging_freq": "--bagging-freq",
    "final_num_boost_round": "--num-boost-round",
    "train_window_years": "--train-window-years",
    "max_depth": "--max-depth",
    "depth": "--depth",
}


_STACKER_CONFIG_FLAGS: dict[str, str] = {
    "learning_rate": "--learning-rate",
    "num_leaves": "--num-leaves",
    "min_data_in_leaf": "--min-data-in-leaf",
    "lambda_l1": "--lambda-l1",
    "lambda_l2": "--lambda-l2",
    "feature_fraction": "--feature-fraction",
    "bagging_fraction": "--bagging-fraction",
    "bagging_freq": "--bagging-freq",
    "final_num_boost_round": "--num-boost-round",
    "min_train_years": "--min-train-years",
    "max_train_years": "--max-train-years",
}


def handle_binary(args: argparse.Namespace) -> int:
    _validate_config_exclusivity(args)
    feature_paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    run = run_paths(args.run_id)
    task = str(args.task)
    model = str(args.model)
    feature_set = str(args.feature_set)

    config_path = str(getattr(args, "config", "") or "").strip()
    config_section: dict[str, object] = {}
    if config_path:
        config_section = _load_config_section(config_path, "binary", task, model)
        if "feature_set" in config_section:
            feature_set = str(config_section["feature_set"])

    feature_input = (
        feature_paths["features_te"] if feature_set == "te" else feature_paths["features"]
    )

    update_run_config(
        args.run_id,
        {
            "run_id": str(args.run_id),
            "feature_profile": str(args.feature_profile),
            "feature_build_id": str(args.feature_build_id),
            "holdout_year": int(args.holdout_year),
        },
    )

    model_ext = _binary_model_ext(model)
    argv = [
        "--task",
        task,
        "--model",
        model,
        "--input",
        str(feature_input),
        "--holdout-input",
        str(feature_input),
        "--oof-output",
        str(run["oof"] / f"{task}_{model}_oof.parquet"),
        "--holdout-output",
        str(run["holdout"] / f"{task}_{model}_holdout_{int(args.holdout_year)}.parquet"),
        "--metrics-output",
        str(run["reports"] / f"{task}_{model}_cv_metrics.json"),
        "--model-output",
        str(run["models"] / f"{task}_{model}_v3.{model_ext}"),
        "--all-years-model-output",
        str(run["models"] / f"{task}_{model}_all_years_v3.{model_ext}"),
        "--meta-output",
        str(run["models"] / f"{task}_{model}_bundle_meta_v3.json"),
        "--feature-manifest-output",
        str(run["models"] / f"{task}_{model}_feature_manifest_v3.json"),
        "--holdout-year",
        str(int(args.holdout_year)),
        "--log-level",
        str(args.log_level),
        "--disable-default-params-json",
    ]
    if config_path:
        _apply_config_to_argv(argv, config_section, flag_map=_BINARY_CONFIG_FLAGS)
    elif str(args.study_id).strip():
        argv.extend(["--params-json", _study_params_path(str(args.study_id).strip())])
    if bool(getattr(args, "train_window_years_explicit", False)) or (
        not str(args.study_id).strip() and not config_path
    ):
        argv.extend(["--train-window-years", str(int(args.train_window_years))])
    rc = int(train_binary_main(argv))
    if rc != 0:
        return rc
    _finalize_metadata(
        run["models"] / f"{task}_{model}_bundle_meta_v3.json",
        run["models"] / f"{task}_{model}_feature_manifest_v3.json",
    )
    save_resolved_params(
        args.run_id,
        f"binary.{task}.{model}",
        {
            "feature_set": feature_set,
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            **(dict(config_section) if config_section else {}),
        },
    )

    metrics_path = run["reports"] / f"{task}_{model}_cv_metrics.json"
    update_run_bundle(
        args.run_id,
        f"binary.{task}.{model}",
        {
            "feature_set": feature_set,
            "study_id": str(args.study_id).strip() or None,
            **asset_payload(
                input=feature_input,
                oof=run["oof"] / f"{task}_{model}_oof.parquet",
                holdout=run["holdout"] / f"{task}_{model}_holdout_{int(args.holdout_year)}.parquet",
                metrics=metrics_path,
                model=run["models"] / f"{task}_{model}_v3.{model_ext}",
                all_years_model=run["models"] / f"{task}_{model}_all_years_v3.{model_ext}",
                meta=run["models"] / f"{task}_{model}_bundle_meta_v3.json",
                feature_manifest=run["models"] / f"{task}_{model}_feature_manifest_v3.json",
            ),
        },
    )
    update_run_metrics(
        args.run_id,
        f"binary.{task}.{model}",
        {
            "path": asset_relative(metrics_path),
            "report": read_json(metrics_path),
        },
    )
    return 0


def handle_stack(args: argparse.Namespace) -> int:
    _validate_config_exclusivity(args)
    feature_paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    run = run_paths(args.run_id)
    source = run_paths(str(args.source_run_id).strip() or str(args.run_id))
    task = str(args.task)

    config_path = str(getattr(args, "config", "") or "").strip()
    config_section: dict[str, object] = {}
    if config_path:
        config_section = _load_config_section(config_path, "stacker", task)

    update_run_config(
        args.run_id,
        {
            "run_id": str(args.run_id),
            "feature_profile": str(args.feature_profile),
            "feature_build_id": str(args.feature_build_id),
            "holdout_year": int(args.holdout_year),
        },
    )

    argv = [
        "--task",
        task,
        "--features-input",
        str(feature_paths["features"]),
        "--holdout-input",
        str(feature_paths["features"]),
        "--lgbm-oof",
        str(source["oof"] / f"{task}_lgbm_oof.parquet"),
        "--xgb-oof",
        str(source["oof"] / f"{task}_xgb_oof.parquet"),
        "--cat-oof",
        str(source["oof"] / f"{task}_cat_oof.parquet"),
        "--lgbm-holdout",
        str(source["holdout"] / f"{task}_lgbm_holdout_{int(args.holdout_year)}.parquet"),
        "--xgb-holdout",
        str(source["holdout"] / f"{task}_xgb_holdout_{int(args.holdout_year)}.parquet"),
        "--cat-holdout",
        str(source["holdout"] / f"{task}_cat_holdout_{int(args.holdout_year)}.parquet"),
        "--oof-output",
        str(run["oof"] / f"{task}_stack_oof.parquet"),
        "--holdout-output",
        str(run["holdout"] / f"{task}_stack_holdout_{int(args.holdout_year)}.parquet"),
        "--metrics-output",
        str(run["reports"] / f"{task}_stack_cv_metrics.json"),
        "--model-output",
        str(run["models"] / f"{task}_stack_v3.txt"),
        "--all-years-model-output",
        str(run["models"] / f"{task}_stack_all_years_v3.txt"),
        "--meta-output",
        str(run["models"] / f"{task}_stack_bundle_meta_v3.json"),
        "--feature-manifest-output",
        str(run["models"] / f"{task}_stack_feature_manifest_v3.json"),
        "--holdout-year",
        str(int(args.holdout_year)),
        "--log-level",
        str(args.log_level),
        "--disable-default-params-json",
    ]
    if config_path:
        _apply_config_to_argv(argv, config_section, flag_map=_STACKER_CONFIG_FLAGS)
    elif str(args.study_id).strip():
        argv.extend(["--params-json", _study_params_path(str(args.study_id).strip())])
    if bool(getattr(args, "min_train_years_explicit", False)) or (
        not str(args.study_id).strip() and not config_path
    ):
        argv.extend(["--min-train-years", str(int(args.min_train_years))])
    if bool(getattr(args, "max_train_years_explicit", False)) or (
        not str(args.study_id).strip() and not config_path
    ):
        argv.extend(["--max-train-years", str(int(args.max_train_years))])
    rc = int(train_stacker_main(argv))
    if rc != 0:
        return rc
    _finalize_metadata(
        run["models"] / f"{task}_stack_bundle_meta_v3.json",
        run["models"] / f"{task}_stack_feature_manifest_v3.json",
    )
    save_resolved_params(
        args.run_id,
        f"stack.{task}",
        {
            "holdout_year": int(args.holdout_year),
            "min_train_years": int(args.min_train_years),
            "max_train_years": int(args.max_train_years),
            **(dict(config_section) if config_section else {}),
        },
    )

    metrics_path = run["reports"] / f"{task}_stack_cv_metrics.json"
    update_run_bundle(
        args.run_id,
        f"stack.{task}",
        {
            "study_id": str(args.study_id).strip() or None,
            "source_run_id": str(args.source_run_id).strip() or str(args.run_id),
            **asset_payload(
                input=feature_paths["features"],
                oof=run["oof"] / f"{task}_stack_oof.parquet",
                holdout=run["holdout"] / f"{task}_stack_holdout_{int(args.holdout_year)}.parquet",
                metrics=metrics_path,
                model=run["models"] / f"{task}_stack_v3.txt",
                all_years_model=run["models"] / f"{task}_stack_all_years_v3.txt",
                meta=run["models"] / f"{task}_stack_bundle_meta_v3.json",
                feature_manifest=run["models"] / f"{task}_stack_feature_manifest_v3.json",
            ),
        },
    )
    update_run_metrics(
        args.run_id,
        f"stack.{task}",
        {
            "path": asset_relative(metrics_path),
            "report": read_json(metrics_path),
        },
    )
    return 0


def handle_pl(args: argparse.Namespace) -> int:
    feature_paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    run = run_paths(args.run_id)
    source = run_paths(str(args.source_run_id).strip() or str(args.run_id))
    profile = str(args.pl_feature_profile)

    update_run_config(
        args.run_id,
        {
            "run_id": str(args.run_id),
            "feature_profile": str(args.feature_profile),
            "feature_build_id": str(args.feature_build_id),
            "pl_feature_profile": profile,
            "holdout_year": int(args.holdout_year),
        },
    )

    argv = [
        "--features-input",
        str(feature_paths["features"]),
        "--holdout-input",
        str(feature_paths["features"]),
        "--pl-feature-profile",
        profile,
        "--win-lgbm-oof",
        str(source["oof"] / "win_lgbm_oof.parquet"),
        "--win-xgb-oof",
        str(source["oof"] / "win_xgb_oof.parquet"),
        "--win-cat-oof",
        str(source["oof"] / "win_cat_oof.parquet"),
        "--place-lgbm-oof",
        str(source["oof"] / "place_lgbm_oof.parquet"),
        "--place-xgb-oof",
        str(source["oof"] / "place_xgb_oof.parquet"),
        "--place-cat-oof",
        str(source["oof"] / "place_cat_oof.parquet"),
        "--win-stack-oof",
        str(source["oof"] / "win_stack_oof.parquet"),
        "--place-stack-oof",
        str(source["oof"] / "place_stack_oof.parquet"),
        "--win-lgbm-holdout",
        str(source["holdout"] / f"win_lgbm_holdout_{int(args.holdout_year)}.parquet"),
        "--win-xgb-holdout",
        str(source["holdout"] / f"win_xgb_holdout_{int(args.holdout_year)}.parquet"),
        "--win-cat-holdout",
        str(source["holdout"] / f"win_cat_holdout_{int(args.holdout_year)}.parquet"),
        "--place-lgbm-holdout",
        str(source["holdout"] / f"place_lgbm_holdout_{int(args.holdout_year)}.parquet"),
        "--place-xgb-holdout",
        str(source["holdout"] / f"place_xgb_holdout_{int(args.holdout_year)}.parquet"),
        "--place-cat-holdout",
        str(source["holdout"] / f"place_cat_holdout_{int(args.holdout_year)}.parquet"),
        "--win-stack-holdout",
        str(source["holdout"] / f"win_stack_holdout_{int(args.holdout_year)}.parquet"),
        "--place-stack-holdout",
        str(source["holdout"] / f"place_stack_holdout_{int(args.holdout_year)}.parquet"),
        "--oof-output",
        str(run["oof"] / f"pl_{profile}_oof.parquet"),
        "--wide-oof-output",
        str(run["oof"] / f"pl_{profile}_wide_oof.parquet"),
        "--emit-wide-oof",
        "--metrics-output",
        str(run["reports"] / f"pl_{profile}_cv_metrics.json"),
        "--model-output",
        str(run["models"] / f"pl_{profile}_recent_window.joblib"),
        "--all-years-model-output",
        str(run["models"] / f"pl_{profile}_all_years.joblib"),
        "--meta-output",
        str(run["models"] / f"pl_{profile}_bundle_meta.json"),
        "--holdout-output",
        str(run["holdout"] / f"pl_{profile}_holdout_{int(args.holdout_year)}.parquet"),
        "--year-coverage-output",
        str(run["reports"] / f"pl_{profile}_year_coverage.json"),
        "--holdout-year",
        str(int(args.holdout_year)),
        "--train-window-years",
        str(int(args.train_window_years)),
        "--log-level",
        str(args.log_level),
    ]
    rc = int(train_pl_main(argv))
    if rc != 0:
        return rc
    _finalize_metadata(
        run["models"] / f"pl_{profile}_bundle_meta.json",
    )
    save_resolved_params(
        args.run_id,
        f"pl.{profile}",
        {
            "pl_feature_profile": profile,
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
        },
    )

    metrics_path = run["reports"] / f"pl_{profile}_cv_metrics.json"
    update_run_bundle(
        args.run_id,
        f"pl.{profile}",
        {
            "source_run_id": str(args.source_run_id).strip() or str(args.run_id),
            **asset_payload(
                input=feature_paths["features"],
                oof=run["oof"] / f"pl_{profile}_oof.parquet",
                wide_oof=run["oof"] / f"pl_{profile}_wide_oof.parquet",
                holdout=run["holdout"] / f"pl_{profile}_holdout_{int(args.holdout_year)}.parquet",
                metrics=metrics_path,
                model=run["models"] / f"pl_{profile}_recent_window.joblib",
                all_years_model=run["models"] / f"pl_{profile}_all_years.joblib",
                meta=run["models"] / f"pl_{profile}_bundle_meta.json",
                year_coverage=run["reports"] / f"pl_{profile}_year_coverage.json",
            ),
        },
    )
    update_run_metrics(
        args.run_id,
        f"pl.{profile}",
        {
            "path": asset_relative(metrics_path),
            "report": read_json(metrics_path),
        },
    )
    return 0


def handle_wide_calibrator(args: argparse.Namespace) -> int:
    run = run_paths(args.run_id)
    source_run_id = str(args.source_run_id).strip() or str(args.run_id)
    source = run_paths(source_run_id)
    source_config = _load_run_config(source_run_id)
    profile = str(source_config.get("pl_feature_profile") or "stack_default")
    holdout_year = int(source_config.get("holdout_year") or 2025)
    fit_input_path = source["oof"] / f"pl_{profile}_wide_oof.parquet"
    if not fit_input_path.exists():
        fit_input_path = source["oof"] / f"pl_{profile}_oof.parquet"
    if not fit_input_path.exists():
        raise SystemExit(
            "wide calibrator fit input not found. expected either "
            f"{source['oof'] / f'pl_{profile}_wide_oof.parquet'} or "
            f"{source['oof'] / f'pl_{profile}_oof.parquet'}"
        )
    apply_input_path = source["holdout"] / f"pl_{profile}_holdout_{holdout_year}.parquet"
    metrics_path = run["reports"] / f"wide_pair_calibration_{str(args.method)}_metrics.json"

    run_config_update: dict[str, object] = {
        "run_id": str(args.run_id),
        "pl_feature_profile": profile,
        "holdout_year": holdout_year,
    }
    for key in ("feature_profile", "feature_build_id"):
        value = source_config.get(key)
        if value:
            run_config_update[key] = value
    update_run_config(args.run_id, run_config_update)

    argv = [
        "--fit-input",
        str(fit_input_path),
        "--apply-input",
        str(apply_input_path),
        "--method",
        str(args.method),
        "--model-output",
        str(run["models"] / f"wide_pair_calibrator_{str(args.method)}.joblib"),
        "--meta-output",
        str(run["models"] / f"wide_pair_calibrator_{str(args.method)}_bundle_meta.json"),
        "--pred-output",
        str(run["predictions"] / f"wide_pair_calibration_{str(args.method)}_pred.parquet"),
        "--metrics-output",
        str(metrics_path),
        "--log-level",
        str(args.log_level),
    ]
    if str(args.years).strip():
        argv.extend(["--years", str(args.years)])
    if str(args.require_years).strip():
        argv.extend(["--require-years", str(args.require_years)])
    if str(args.database_url).strip():
        argv.extend(["--database-url", str(args.database_url).strip()])
    rc = int(train_wide_calibrator_main(argv))
    if rc != 0:
        return rc
    _finalize_metadata(
        run["models"] / f"wide_pair_calibrator_{str(args.method)}_bundle_meta.json",
    )
    save_resolved_params(
        args.run_id,
        f"wide_calibrator.{str(args.method)}",
        {
            "method": str(args.method),
            "source_run_id": source_run_id,
            "holdout_year": holdout_year,
        },
    )

    update_run_bundle(
        args.run_id,
        f"wide_calibrator.{str(args.method)}",
        {
            "source_run_id": source_run_id,
            **asset_payload(
                input=fit_input_path,
                apply_input=apply_input_path,
                model=run["models"] / f"wide_pair_calibrator_{str(args.method)}.joblib",
                meta=run["models"] / f"wide_pair_calibrator_{str(args.method)}_bundle_meta.json",
                predictions=run["predictions"]
                / f"wide_pair_calibration_{str(args.method)}_pred.parquet",
                metrics=metrics_path,
            ),
        },
    )
    update_run_metrics(
        args.run_id,
        f"wide_calibrator.{str(args.method)}",
        {
            "path": asset_relative(metrics_path),
            "report": read_json(metrics_path),
        },
    )
    return 0

from __future__ import annotations

import argparse

from keiba_research.common.assets import (
    ensure_json_has_no_absolute_paths,
    feature_build_paths,
    rewrite_json_asset_paths,
)
from keiba_research.common.state import write_toml
from scripts_v3.build_features_base_v3 import run_build_features_base
from scripts_v3.build_features_v3 import run_build_features
from scripts_v3.build_features_v3_te import run_build_features_te


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="features_command", required=True)

    build_base = subparsers.add_parser(
        "build-base", help="Build base features for a named profile/build."
    )
    build_base.add_argument("--feature-profile", required=True)
    build_base.add_argument("--feature-build-id", required=True)
    build_base.add_argument("--from-date", required=True)
    build_base.add_argument("--to-date", required=True)
    build_base.add_argument("--history-days", type=int, default=730)
    build_base.add_argument("--database-url", default="")
    build_base.add_argument("--log-level", default="INFO")
    build_base.set_defaults(handler=handle_build_base)

    build = subparsers.add_parser("build", help="Build features_v3 from base features.")
    build.add_argument("--feature-profile", required=True)
    build.add_argument("--feature-build-id", required=True)
    build.add_argument("--database-url", default="")
    build.add_argument("--log-level", default="INFO")
    build.set_defaults(handler=handle_build)

    build_te = subparsers.add_parser("build-te", help="Build TE-augmented features_v3.")
    build_te.add_argument("--feature-profile", required=True)
    build_te.add_argument("--feature-build-id", required=True)
    build_te.add_argument("--log-level", default="INFO")
    build_te.set_defaults(handler=handle_build_te)


def _write_feature_build_config(
    *,
    feature_profile: str,
    feature_build_id: str,
    payload: dict[str, object],
) -> None:
    paths = feature_build_paths(feature_profile, feature_build_id)
    write_toml(
        paths["config"],
        {
            "feature_profile": str(feature_profile),
            "feature_build_id": str(feature_build_id),
            **payload,
        },
    )


def handle_build_base(args: argparse.Namespace) -> int:
    paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    _write_feature_build_config(
        feature_profile=args.feature_profile,
        feature_build_id=args.feature_build_id,
        payload={
            "from_date": str(args.from_date),
            "to_date": str(args.to_date),
            "history_days": int(args.history_days),
        },
    )

    rc = int(
        run_build_features_base(
            from_date=str(args.from_date),
            to_date=str(args.to_date),
            history_days=int(args.history_days),
            output=str(paths["base"]),
            meta_output=str(paths["base_meta"]),
            database_url=str(args.database_url).strip(),
            log_level=str(args.log_level),
            with_te=False,
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(paths["base_meta"])
    ensure_json_has_no_absolute_paths(paths["base_meta"])

    rc = int(
        run_build_features_base(
            from_date=str(args.from_date),
            to_date=str(args.to_date),
            history_days=int(args.history_days),
            output=str(paths["base_te"]),
            meta_output=str(paths["base_te_meta"]),
            database_url=str(args.database_url).strip(),
            log_level=str(args.log_level),
            with_te=True,
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(paths["base_te_meta"])
    ensure_json_has_no_absolute_paths(paths["base_te_meta"])
    return 0


def handle_build(args: argparse.Namespace) -> int:
    paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    rc = int(
        run_build_features(
            input=str(paths["base"]),
            output=str(paths["features"]),
            meta_output=str(paths["features_meta"]),
            database_url=str(args.database_url).strip(),
            log_level=str(args.log_level),
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(paths["features_meta"])
    ensure_json_has_no_absolute_paths(paths["features_meta"])
    return 0


def handle_build_te(args: argparse.Namespace) -> int:
    paths = feature_build_paths(args.feature_profile, args.feature_build_id)
    rc = int(
        run_build_features_te(
            base_input=str(paths["features"]),
            te_source_input=str(paths["base_te"]),
            output=str(paths["features_te"]),
            meta_output=str(paths["features_te_meta"]),
            log_level=str(args.log_level),
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(paths["features_te_meta"])
    ensure_json_has_no_absolute_paths(paths["features_te_meta"])
    return 0

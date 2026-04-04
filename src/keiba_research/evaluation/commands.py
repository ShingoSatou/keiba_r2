from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from keiba_research.common.assets import (
    asset_relative,
    cache_root,
    ensure_json_has_no_absolute_paths,
    read_json,
    rewrite_json_asset_paths,
    run_paths,
    write_json,
)
from keiba_research.common.state import update_run_bundle, update_run_metrics
from keiba_research.evaluation.backtest_wide import run_backtest_wide
from keiba_research.evaluation.execution_report import write_execution_report
from keiba_research.evaluation.report_view import (
    DEFAULT_VIEW_HOST,
    DEFAULT_VIEW_PORT,
    build_report_view,
    serve_report_view,
)


def register(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="eval_command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run backtest for a run bundle.")
    backtest.add_argument("--run-id", required=True)
    backtest.add_argument(
        "--input-kind", choices=["pl_holdout", "pl_oof", "wide_calibrated"], default="pl_holdout"
    )
    backtest.add_argument("--pl-feature-profile", default="")
    backtest.add_argument("--years", default="")
    backtest.add_argument("--require-years", default="")
    backtest.add_argument("--database-url", default="")
    backtest.add_argument("--log-level", default="INFO")
    backtest.set_defaults(handler=handle_backtest)

    compare = subparsers.add_parser("compare", help="Compare two run bundles.")
    compare.add_argument("--left-run-id", required=True)
    compare.add_argument("--right-run-id", required=True)
    compare.add_argument("--output", default="")
    compare.set_defaults(handler=handle_compare)

    report = subparsers.add_parser(
        "report",
        help="Generate execution report summary/detail from an existing run bundle.",
    )
    report.add_argument("--run-id", required=True)
    report.add_argument("--summary-output", default="")
    report.add_argument("--detail-output", default="")
    report.add_argument("--annotation", default="")
    report.set_defaults(handler=handle_report)

    report_view = subparsers.add_parser(
        "report-view",
        help="Launch a local viewer for one or two execution reports.",
    )
    report_view.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="run_id to view. Specify twice to open compare mode.",
    )
    report_view.add_argument("--output-html", default="")
    report_view.add_argument("--host", default=DEFAULT_VIEW_HOST)
    report_view.add_argument("--port", type=int, default=DEFAULT_VIEW_PORT)
    report_view.add_argument("--refresh", action="store_true")
    report_view.add_argument("--no-serve", action="store_true")
    report_view.add_argument("--open-browser", action="store_true")
    report_view.set_defaults(handler=handle_report_view)


def _load_run_config(run_id: str) -> dict[str, object]:
    run = run_paths(run_id)
    if not run["config"].exists():
        return {}
    import tomllib

    return tomllib.loads(run["config"].read_text(encoding="utf-8"))


def handle_backtest(args: argparse.Namespace) -> int:
    run = run_paths(args.run_id)
    run_config = _load_run_config(args.run_id)
    profile = str(args.pl_feature_profile).strip() or str(
        run_config.get("pl_feature_profile") or "stack_default"
    )
    holdout_year = int(run_config.get("holdout_year") or 2025)
    effective_holdout_year = (
        holdout_year + 1
        if str(args.input_kind) in {"pl_holdout", "wide_calibrated"}
        else holdout_year
    )

    if str(args.input_kind) == "pl_holdout":
        input_path = run["holdout"] / f"pl_{profile}_holdout_{holdout_year}.parquet"
    elif str(args.input_kind) == "pl_oof":
        input_path = run["oof"] / f"pl_{profile}_oof.parquet"
    else:
        input_path = run["predictions"] / "wide_pair_calibration_isotonic_pred.parquet"

    output_path = run["reports"] / f"backtest_{str(args.input_kind)}.json"
    meta_path = run["reports"] / f"backtest_{str(args.input_kind)}_meta.json"
    rc = int(
        run_backtest_wide(
            input=str(input_path),
            output=str(output_path),
            meta_output=str(meta_path),
            holdout_year=int(effective_holdout_year),
            database_url=str(args.database_url).strip(),
            years=str(args.years),
            require_years=str(args.require_years),
            log_level=str(args.log_level),
        )
    )
    if rc != 0:
        return rc
    rewrite_json_asset_paths(meta_path)
    ensure_json_has_no_absolute_paths(meta_path)

    report = read_json(output_path)
    update_run_bundle(
        args.run_id,
        f"backtest.{str(args.input_kind)}",
        {
            **{
                "input_kind": str(args.input_kind),
                "pl_feature_profile": profile,
            },
            **{
                "input": asset_relative(input_path),
                "report": asset_relative(output_path),
                "meta": asset_relative(meta_path),
            },
        },
    )
    update_run_metrics(
        args.run_id,
        f"backtest.{str(args.input_kind)}",
        {
            "path": asset_relative(output_path),
            "report": report,
        },
    )
    return 0


def _flatten_numeric(payload: Any, *, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(value, prefix=child))
    elif isinstance(payload, list):
        return out
    elif isinstance(payload, bool):
        return out
    elif isinstance(payload, (int, float)):
        out[prefix] = float(payload)
    return out


def handle_compare(args: argparse.Namespace) -> int:
    left = run_paths(args.left_run_id)
    right = run_paths(args.right_run_id)
    left_metrics = read_json(left["metrics"])
    right_metrics = read_json(right["metrics"])

    left_flat = _flatten_numeric(left_metrics.get("sections", {}))
    right_flat = _flatten_numeric(right_metrics.get("sections", {}))
    common = sorted(set(left_flat) & set(right_flat))
    deltas = {
        key: {
            "left": left_flat[key],
            "right": right_flat[key],
            "delta": right_flat[key] - left_flat[key],
        }
        for key in common
    }

    output_path = (
        Path(str(args.output).strip()).resolve()
        if str(args.output).strip()
        else cache_root()
        / "compare"
        / f"{str(args.left_run_id)}__vs__{str(args.right_run_id)}.json"
    )
    payload = {
        "comparison_version": 1,
        "left_run_id": str(args.left_run_id),
        "right_run_id": str(args.right_run_id),
        "left_metrics": asset_relative(left["metrics"]),
        "right_metrics": asset_relative(right["metrics"]),
        "common_numeric_deltas": deltas,
        "left_sections": sorted((left_metrics.get("sections") or {}).keys()),
        "right_sections": sorted((right_metrics.get("sections") or {}).keys()),
    }
    write_json(output_path, payload)
    return 0


def handle_report(args: argparse.Namespace) -> int:
    write_execution_report(
        str(args.run_id),
        summary_output=str(args.summary_output).strip() or None,
        detail_output=str(args.detail_output).strip() or None,
        annotation_path=str(args.annotation).strip() or None,
    )
    return 0


def handle_report_view(args: argparse.Namespace) -> int:
    run_ids = [str(run_id).strip() for run_id in list(getattr(args, "run_id", []) or []) if str(run_id).strip()]
    html_path = build_report_view(
        run_ids,
        output_html=str(args.output_html).strip() or None,
        refresh=bool(args.refresh),
    )
    if bool(args.no_serve):
        print(html_path)
        return 0
    return int(
        serve_report_view(
            html_path=html_path,
            host=str(args.host).strip() or DEFAULT_VIEW_HOST,
            port=int(args.port),
            open_browser=bool(args.open_browser),
        )
    )

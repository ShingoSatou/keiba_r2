from __future__ import annotations

import tomllib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from keiba_research.common.assets import (
    asset_relative,
    asset_root,
    read_json,
    resolve_asset_path,
    run_paths,
    study_paths,
    write_json,
)
from keiba_research.common.run_config import generate_config_from_study
from keiba_research.common.state import utc_now

SUMMARY_SCHEMA_VERSION = 1
DETAIL_SCHEMA_VERSION = 1


def build_execution_report(
    run_id: str,
    *,
    annotation_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = run_paths(run_id)
    run_config = _read_toml(run["config"])
    bundle = read_json(run["bundle"])
    metrics = read_json(run["metrics"])
    resolved_params = _read_toml(run["resolved_params"])
    annotation = _read_toml(
        Path(annotation_path).resolve()
        if str(annotation_path or "").strip()
        else run["execution_report_annotation"]
    )

    issues = _collect_issues(
        run=run,
        bundle=bundle,
        metrics=metrics,
        resolved_params=resolved_params,
    )
    sections = _collect_sections(bundle=bundle, metrics=metrics)
    feature_config = _load_feature_build_config(run_config)
    source_run_ids = _collect_source_run_ids(sections)
    primary_source_run_id = source_run_ids[0] if source_run_ids else None
    pipeline = _build_pipeline(sections)

    quality_summary, quality_detail = _build_quality(sections)
    coverage_summary, coverage_detail = _build_coverage(
        run_config=run_config,
        sections=sections,
    )
    backtest_summary, roi_detail = _build_backtest(sections)
    layer_settings, setting_sources = _build_layer_settings(
        run_id=run_id,
        run_config=run_config,
        sections=sections,
        resolved_params=resolved_params,
    )
    diagnostics = _build_diagnostics(
        sections=sections,
        quality_detail=quality_detail,
        roi_detail=roi_detail,
    )
    code_revision, code_fingerprint = _build_code_fingerprint(annotation=annotation, sections=sections)

    created_at = utc_now()
    conditions = _build_conditions(
        run_config=run_config,
        feature_config=feature_config,
        source_run_ids=source_run_ids,
        primary_source_run_id=primary_source_run_id,
    )
    computed_status = _compute_status(
        issues=issues,
        pipeline=pipeline,
        backtest_summary=backtest_summary,
    )
    status = str(annotation.get("status") or computed_status)
    title = str(annotation.get("title") or _auto_title(run_id=run_id, conditions=conditions))
    description = str(
        annotation.get("description")
        or _auto_description(
            run_id=run_id,
            conditions=conditions,
            pipeline=pipeline,
            source_run_ids=source_run_ids,
        )
    )

    paths = {
        "run_root": asset_relative(run["root"]),
        "config": asset_relative(run["config"]),
        "bundle": asset_relative(run["bundle"]),
        "metrics": asset_relative(run["metrics"]),
        "resolved_params": asset_relative(run["resolved_params"]),
        "summary": asset_relative(run["execution_report_summary"]),
        "detail": asset_relative(run["execution_report_detail"]),
        "annotation": (
            asset_relative(run["execution_report_annotation"])
            if run["execution_report_annotation"].exists()
            else None
        ),
        "feature_build_config": (
            asset_relative(_feature_build_config_path(run_config))
            if _feature_build_config_path(run_config) is not None
            and _feature_build_config_path(run_config).exists()
            else None
        ),
    }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "report_id": str(run_id),
        "run_id": str(run_id),
        "created_at": created_at,
        "status": status,
        "title": title,
        "description": description,
        "code_revision": code_revision,
        "conditions": conditions,
        "pipeline": pipeline,
        "quality_summary": quality_summary,
        "coverage_summary": coverage_summary,
        "backtest_summary": backtest_summary,
        "paths": paths,
    }

    detail = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        **summary,
        "source_of_truth": _build_source_of_truth(
            run=run,
            feature_config=feature_config,
            sections=sections,
        ),
        "lineage": {
            "source_run_ids": source_run_ids,
            "source_report_ids": list(source_run_ids),
            "primary_source_run_id": primary_source_run_id,
            "primary_source_report_id": primary_source_run_id,
            "section_source_runs": {
                name: section["bundle"].get("source_run_id")
                for name, section in sections.items()
                if section["bundle"].get("source_run_id") is not None
            },
        },
        "layer_settings": layer_settings,
        "setting_sources": setting_sources,
        "quality_detail": quality_detail,
        "coverage_detail": coverage_detail,
        "roi_detail": roi_detail,
        "diagnostics": diagnostics,
        "artifacts": {
            "bundle_sections": {name: section["bundle"] for name, section in sections.items()},
            "metrics_sections": {
                name: {"path": section["metrics_path"]} for name, section in sections.items()
            },
        },
        "code_fingerprint": code_fingerprint,
        "issues": issues,
    }
    return summary, detail


def write_execution_report(
    run_id: str,
    *,
    summary_output: str | None = None,
    detail_output: str | None = None,
    annotation_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = run_paths(run_id)
    summary_path = _resolve_report_output(
        default_path=run["execution_report_summary"],
        override=summary_output,
    )
    detail_path = _resolve_report_output(
        default_path=run["execution_report_detail"],
        override=detail_output,
    )
    summary, detail = build_execution_report(run_id, annotation_path=annotation_path)
    summary["paths"]["summary"] = asset_relative(summary_path)
    summary["paths"]["detail"] = asset_relative(detail_path)
    detail["paths"]["summary"] = asset_relative(summary_path)
    detail["paths"]["detail"] = asset_relative(detail_path)
    write_json(summary_path, summary)
    write_json(detail_path, detail)
    return summary, detail


def _resolve_report_output(*, default_path: Path, override: str | None) -> Path:
    path = Path(str(override).strip()).resolve() if str(override or "").strip() else default_path
    asset_relative(path)
    return path


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _feature_build_config_path(run_config: dict[str, Any]) -> Path | None:
    feature_profile = str(run_config.get("feature_profile") or "").strip()
    feature_build_id = str(run_config.get("feature_build_id") or "").strip()
    if not feature_profile or not feature_build_id:
        return None
    return (
        asset_root()
        / "data"
        / "features"
        / feature_profile
        / feature_build_id
        / "config.toml"
    )


def _load_feature_build_config(run_config: dict[str, Any]) -> dict[str, Any]:
    config_path = _feature_build_config_path(run_config)
    return _read_toml(config_path) if config_path is not None else {}


def _collect_issues(
    *,
    run: dict[str, Path],
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    resolved_params: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for name, path in (
        ("config", run["config"]),
        ("bundle", run["bundle"]),
    ):
        if not path.exists():
            issues.append({"kind": "missing_file", "path": str(path), "message": f"{name} is missing"})
    if not bundle.get("sections"):
        issues.append({"kind": "empty_bundle", "path": asset_relative(run["bundle"]), "message": "bundle.json has no sections"})

    if run["metrics"].exists() and not metrics.get("sections"):
        issues.append(
            {
                "kind": "empty_metrics",
                "path": asset_relative(run["metrics"]),
                "message": "metrics.json exists but has no sections",
            }
        )
    if run["resolved_params"].exists() and not resolved_params:
        issues.append(
            {
                "kind": "empty_resolved_params",
                "path": asset_relative(run["resolved_params"]),
                "message": "resolved_params.toml exists but is empty",
            }
        )
    for section_name, section_payload in (bundle.get("sections") or {}).items():
        if not isinstance(section_payload, dict):
            continue
        for key, value in section_payload.items():
            if key in {"study_id", "source_run_id", "input_kind", "feature_set", "pl_feature_profile"}:
                continue
            if not isinstance(value, str) or not value:
                continue
            if "/" not in value:
                continue
            try:
                resolved = resolve_asset_path(value)
            except SystemExit:
                continue
            if not resolved.exists():
                issues.append(
                    {
                        "kind": "missing_artifact",
                        "path": value,
                        "message": f"{section_name}.{key} does not exist",
                    }
                )
    return issues


def _collect_sections(
    *,
    bundle: dict[str, Any],
    metrics: dict[str, Any],
) -> OrderedDict[str, dict[str, Any]]:
    names = list(
        OrderedDict.fromkeys(
            [
                *list((bundle.get("sections") or {}).keys()),
                *list((metrics.get("sections") or {}).keys()),
            ]
        )
    )
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for name in names:
        bundle_section = dict((bundle.get("sections") or {}).get(name) or {})
        metrics_section = dict((metrics.get("sections") or {}).get(name) or {})
        out[name] = {
            "kind": _section_kind(name),
            "parts": _section_parts(name),
            "bundle": bundle_section,
            "metrics": metrics_section,
            "report_path": _resolve_report_path(name=name, bundle_section=bundle_section, metrics_section=metrics_section),
            "metrics_path": str(metrics_section.get("path") or bundle_section.get("metrics") or ""),
            "report": _load_section_report(name=name, bundle_section=bundle_section, metrics_section=metrics_section),
        }
    return out


def _resolve_report_path(
    *,
    name: str,
    bundle_section: dict[str, Any],
    metrics_section: dict[str, Any],
) -> str | None:
    if _section_kind(name) == "backtest":
        return str(bundle_section.get("report") or metrics_section.get("path") or "").strip() or None
    return str(metrics_section.get("path") or bundle_section.get("metrics") or "").strip() or None


def _load_section_report(
    *,
    name: str,
    bundle_section: dict[str, Any],
    metrics_section: dict[str, Any],
) -> dict[str, Any]:
    embedded = metrics_section.get("report")
    if isinstance(embedded, dict):
        return embedded
    report_key = "report" if _section_kind(name) == "backtest" else "metrics"
    report_path = str(bundle_section.get(report_key) or metrics_section.get("path") or "").strip()
    if not report_path:
        return {}
    resolved = resolve_asset_path(report_path)
    return read_json(resolved)


def _section_kind(section_name: str) -> str:
    return str(section_name).split(".", 1)[0]


def _section_parts(section_name: str) -> dict[str, str]:
    parts = str(section_name).split(".")
    if not parts:
        return {}
    kind = parts[0]
    if kind == "binary" and len(parts) >= 3:
        return {"task": parts[1], "model": parts[2]}
    if kind == "stack" and len(parts) >= 2:
        return {"task": parts[1]}
    if kind == "pl" and len(parts) >= 2:
        return {"profile": parts[1]}
    if kind == "wide_calibrator" and len(parts) >= 2:
        return {"method": parts[1]}
    if kind == "backtest" and len(parts) >= 2:
        return {"input_kind": parts[1]}
    return {}


def _collect_source_run_ids(sections: OrderedDict[str, dict[str, Any]]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for section in sections.values():
        source_run_id = str(section["bundle"].get("source_run_id") or "").strip()
        if source_run_id:
            seen[source_run_id] = None
    return list(seen.keys())


def _build_pipeline(sections: OrderedDict[str, dict[str, Any]]) -> dict[str, Any]:
    binary = {
        "win": {"lgbm": False, "xgb": False, "cat": False},
        "place": {"lgbm": False, "xgb": False, "cat": False},
    }
    stack = {"win": False, "place": False}
    pl_profiles: dict[str, bool] = {}
    wide_methods: dict[str, dict[str, Any]] = {}
    backtest = {"pl_holdout": False, "pl_oof": False, "wide_calibrated": False}

    for name, section in sections.items():
        kind = section["kind"]
        parts = section["parts"]
        if kind == "binary":
            binary[parts["task"]][parts["model"]] = True
        elif kind == "stack":
            stack[parts["task"]] = True
        elif kind == "pl":
            pl_profiles[parts["profile"]] = True
        elif kind == "wide_calibrator":
            wide_methods[parts["method"]] = {
                "enabled": True,
                "method": parts["method"],
            }
        elif kind == "backtest":
            backtest[parts["input_kind"]] = True

    return {
        "binary": binary,
        "stack": stack,
        "pl": {
            "enabled": bool(pl_profiles),
            "profiles": pl_profiles,
        },
        "wide_calibrator": {
            "enabled": bool(wide_methods),
            "methods": wide_methods,
        },
        "backtest": backtest,
    }


def _build_conditions(
    *,
    run_config: dict[str, Any],
    feature_config: dict[str, Any],
    source_run_ids: list[str],
    primary_source_run_id: str | None,
) -> dict[str, Any]:
    return {
        "feature_profile": run_config.get("feature_profile"),
        "feature_build_id": run_config.get("feature_build_id"),
        "holdout_year": run_config.get("holdout_year"),
        "pl_feature_profile": run_config.get("pl_feature_profile"),
        "source_run_ids": source_run_ids,
        "primary_source_run_id": primary_source_run_id,
        "source_report_ids": list(source_run_ids),
        "primary_source_report_id": primary_source_run_id,
        "target_segment": None,
        "from_date": feature_config.get("from_date"),
        "to_date": feature_config.get("to_date"),
        "history_days": feature_config.get("history_days"),
    }


def _build_quality(
    sections: OrderedDict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {
        "binary": {},
        "stack": {},
        "pl": {},
        "wide_calibrator": {},
        "backtest": {},
    }
    detail = {"sections": {}}
    for name, section in sections.items():
        report = section["report"]
        curated = _curate_quality(name=name, report=report)
        detail["sections"][name] = {
            "kind": section["kind"],
            "parts": section["parts"],
            "path": section["report_path"],
            "curated": curated,
        }
        if section["kind"] == "binary":
            summary["binary"][f"{section['parts']['task']}.{section['parts']['model']}"] = curated
        elif section["kind"] == "stack":
            summary["stack"][section["parts"]["task"]] = curated
        elif section["kind"] == "pl":
            summary["pl"][section["parts"]["profile"]] = curated
        elif section["kind"] == "wide_calibrator":
            summary["wide_calibrator"][section["parts"]["method"]] = curated
        elif section["kind"] == "backtest":
            summary["backtest"][section["parts"]["input_kind"]] = curated
    return _drop_empty(summary), detail


def _curate_quality(*, name: str, report: dict[str, Any]) -> dict[str, Any]:
    kind = _section_kind(name)
    if kind == "binary":
        summary = report.get("summary") or {}
        curated = {
            "logloss": _metric_mean(summary.get("logloss")),
            "brier": _metric_mean(summary.get("brier")),
            "auc": _metric_mean(summary.get("auc")),
            "ece": _metric_mean(summary.get("ece")),
        }
        benter = summary.get("benter_r2_valid")
        if isinstance(benter, dict):
            curated["benter_r2_valid"] = _metric_mean(benter)
        return _drop_none(curated)
    if kind == "stack":
        summary = report.get("summary") or {}
        return _drop_none(
            {
                "logloss": _metric_mean(summary.get("logloss")),
                "brier": _metric_mean(summary.get("brier")),
                "auc": _metric_mean(summary.get("auc")),
                "ece": _metric_mean(summary.get("ece")),
            }
        )
    if kind == "pl":
        summary = report.get("summary") or {}
        return _drop_none(
            {
                "pl_nll_valid": _metric_mean(summary.get("pl_nll_valid")),
                "top3_logloss": _metric_mean(summary.get("top3_logloss")),
                "top3_brier": _metric_mean(summary.get("top3_brier")),
                "top3_auc": _metric_mean(summary.get("top3_auc")),
                "top3_ece": _metric_mean(summary.get("top3_ece")),
            }
        )
    if kind == "wide_calibrator":
        return _drop_none(
            {
                "fit": _drop_none(_calibration_metrics(report.get("fit") or {})),
                "holdout_eval": _drop_none(_calibration_metrics(report.get("holdout_eval") or {})),
            }
        )
    if kind == "backtest":
        summary = report.get("summary") or {}
        return _drop_none(
            {
                "period_from": summary.get("period_from"),
                "period_to": summary.get("period_to"),
                "n_races": summary.get("n_races"),
                "n_bets": summary.get("n_bets"),
                "n_hits": summary.get("n_hits"),
                "hit_rate": summary.get("hit_rate"),
                "total_bet": summary.get("total_bet"),
                "total_return": summary.get("total_return"),
                "roi": summary.get("roi"),
                "max_drawdown": summary.get("max_drawdown"),
                "logloss": summary.get("logloss"),
                "auc": summary.get("auc"),
            }
        )
    return {}


def _calibration_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    calibrated = payload.get("calibrated") or {}
    return {
        "rows": payload.get("rows"),
        "races": payload.get("races"),
        "selected_years": payload.get("selected_years"),
        "logloss": calibrated.get("logloss"),
        "brier": calibrated.get("brier"),
        "auc": calibrated.get("auc"),
        "ece": calibrated.get("ece"),
    }


def _build_coverage(
    *,
    run_config: dict[str, Any],
    sections: OrderedDict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {
        "binary": {},
        "stack": {},
        "pl": {},
        "wide_calibrator": {},
        "backtest": {},
    }
    detail = {"sections": {}}
    for name, section in sections.items():
        report = section["report"]
        bundle_section = section["bundle"]
        curated = _curate_coverage(
            name=name,
            report=report,
            bundle_section=bundle_section,
            run_config=run_config,
        )
        detail["sections"][name] = {
            "kind": section["kind"],
            "parts": section["parts"],
            "path": section["report_path"],
            "curated": curated,
        }
        if section["kind"] == "binary":
            summary["binary"][f"{section['parts']['task']}.{section['parts']['model']}"] = curated
        elif section["kind"] == "stack":
            summary["stack"][section["parts"]["task"]] = curated
        elif section["kind"] == "pl":
            summary["pl"][section["parts"]["profile"]] = curated
        elif section["kind"] == "wide_calibrator":
            summary["wide_calibrator"][section["parts"]["method"]] = curated
        elif section["kind"] == "backtest":
            summary["backtest"][section["parts"]["input_kind"]] = curated
    return _drop_empty(summary), detail


def _curate_coverage(
    *,
    name: str,
    report: dict[str, Any],
    bundle_section: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    kind = _section_kind(name)
    if kind in {"binary", "stack"}:
        data_summary = report.get("data_summary") or {}
        cv_policy = report.get("cv_policy") or {}
        return _drop_none(
            {
                "rows": data_summary.get("rows"),
                "races": data_summary.get("races"),
                "years": data_summary.get("years"),
                "oof_valid_years": cv_policy.get("valid_years"),
                "holdout_years": _parquet_years(bundle_section.get("holdout")),
                "holdout_year": run_config.get("holdout_year"),
            }
        )
    if kind == "pl":
        data_summary = report.get("data_summary") or {}
        year_coverage = _load_json_path(bundle_section.get("year_coverage"))
        return _drop_none(
            {
                "rows": data_summary.get("rows"),
                "races": data_summary.get("races"),
                "years": data_summary.get("years"),
                "oof_valid_years": year_coverage.get("pl_oof_valid_years"),
                "holdout_years": _parquet_years(bundle_section.get("holdout")),
                "pl_oof_valid_years": year_coverage.get("pl_oof_valid_years"),
                "pl_holdout_train_years": year_coverage.get("pl_holdout_train_years"),
                "base_oof_years": year_coverage.get("base_oof_years"),
                "stacker_oof_years": year_coverage.get("stacker_oof_years"),
            }
        )
    if kind == "wide_calibrator":
        fit = report.get("fit") or {}
        holdout_eval = report.get("holdout_eval") or {}
        return _drop_none(
            {
                "fit_rows": fit.get("rows"),
                "fit_races": fit.get("races"),
                "fit_years": fit.get("selected_years"),
                "holdout_eval_rows": holdout_eval.get("rows"),
                "holdout_eval_races": holdout_eval.get("races"),
                "holdout_eval_years": holdout_eval.get("selected_years"),
            }
        )
    if kind == "backtest":
        meta = _load_json_path(bundle_section.get("meta"))
        input_meta = meta.get("input") or {}
        return _drop_none(
            {
                "rows": input_meta.get("rows"),
                "pair_rows_for_backtest": input_meta.get("pair_rows_for_backtest"),
                "selected_races": input_meta.get("selected_races"),
                "selected_years": input_meta.get("selected_years"),
                "available_years_after_holdout_filter": input_meta.get(
                    "available_years_after_holdout_filter"
                ),
            }
        )
    return {}


def _build_backtest(
    sections: OrderedDict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {}
    detail = {"backtests": {}}
    for name, section in sections.items():
        if section["kind"] != "backtest":
            continue
        input_kind = section["parts"]["input_kind"]
        report = section["report"]
        meta = _load_json_path(section["bundle"].get("meta"))
        backtest_summary = _curate_quality(name=name, report=report)
        purchase_rule = _extract_purchase_rule(meta)
        summary[input_kind] = backtest_summary
        detail["backtests"][input_kind] = {
            "report_path": section["bundle"].get("report"),
            "meta_path": section["bundle"].get("meta"),
            "summary": backtest_summary,
            "purchase_rule": purchase_rule,
            "input": _drop_none(
                {
                    "input_kind": input_kind,
                    "input_mode": (meta.get("input") or {}).get("input_mode"),
                    "p_wide_source": (meta.get("input") or {}).get("p_wide_source"),
                    "selected_years": (meta.get("input") or {}).get("selected_years"),
                    "input_filter_holdout_year": (meta.get("input") or {}).get(
                        "input_filter_holdout_year"
                    ),
                }
            ),
        }
    return summary, detail


def _extract_purchase_rule(meta: dict[str, Any]) -> dict[str, Any]:
    config = meta.get("config") or {}
    selection = config.get("selection") or {}
    bankroll = config.get("bankroll") or {}
    input_meta = meta.get("input") or {}
    return _drop_none(
        {
            "input_kind": input_meta.get("input_mode"),
            "p_wide_source": input_meta.get("p_wide_source"),
            "min_p_wide": selection.get("min_p_wide"),
            "min_p_wide_stage": selection.get("min_p_wide_stage"),
            "ev_threshold": selection.get("ev_threshold"),
            "max_bets_per_race": selection.get("max_bets_per_race"),
            "kelly_fraction": bankroll.get("kelly_fraction_scale"),
            "race_cap_fraction": bankroll.get("race_cap_fraction"),
            "daily_cap_fraction": bankroll.get("daily_cap_fraction"),
            "bankroll_init_yen": bankroll.get("bankroll_init_yen"),
            "bet_unit_yen": bankroll.get("bet_unit_yen"),
            "min_bet_yen": bankroll.get("min_bet_yen"),
            "max_bet_yen": bankroll.get("max_bet_yen"),
        }
    )


def _build_layer_settings(
    *,
    run_id: str,
    run_config: dict[str, Any],
    sections: OrderedDict[str, dict[str, Any]],
    resolved_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer_settings: dict[str, Any] = {}
    setting_sources: dict[str, Any] = {}
    for name, section in sections.items():
        resolved_section = _nested_get(resolved_params, name.split("."))
        report_config = dict((section["report"].get("config") or {}))
        bundle_settings = {
            key: value
            for key, value in section["bundle"].items()
            if key in {"study_id", "source_run_id", "feature_set", "input_kind"}
        }
        year_coverage = (
            _load_json_path(section["bundle"].get("year_coverage"))
            if section["kind"] == "pl"
            else {}
        )
        combined_settings = {
            "bundle": bundle_settings,
            "resolved_params": resolved_section,
            "report_config": report_config,
        }
        if year_coverage:
            combined_settings["year_coverage"] = year_coverage
        layer_settings[name] = _drop_empty(combined_settings)
        setting_sources[name] = _classify_setting_sources(
            run_id=run_id,
            run_config=run_config,
            section_name=name,
            section=section,
            resolved_section=resolved_section,
            report_config=report_config,
        )
    return layer_settings, setting_sources


def _classify_setting_sources(
    *,
    run_id: str,
    run_config: dict[str, Any],
    section_name: str,
    section: dict[str, Any],
    resolved_section: dict[str, Any],
    report_config: dict[str, Any],
) -> dict[str, str]:
    out: dict[str, str] = {}
    bundle_section = section["bundle"]
    defaults = _section_defaults(section_name)
    study_id = str(bundle_section.get("study_id") or "").strip()
    study_values = _load_study_selected_values(study_id=study_id, section_name=section_name)
    combined: dict[str, Any] = {}
    combined.update(report_config)
    combined.update(resolved_section)
    combined.update(
        {key: value for key, value in bundle_section.items() if key in {"feature_set", "study_id", "source_run_id"}}
    )

    source_run_id = str(bundle_section.get("source_run_id") or "").strip()
    source_run_config = _read_toml(run_paths(source_run_id)["config"]) if source_run_id else {}
    for key, value in combined.items():
        if key == "study_id":
            out[key] = "study_selected" if study_id else "unknown"
        elif key == "source_run_id":
            out[key] = "inherited_from_source_run" if value and str(value) != run_id else "unknown"
        elif key in study_values and study_values[key] == value:
            out[key] = "study_selected"
        elif key in defaults and defaults[key] == value:
            out[key] = "default"
        elif source_run_id and key in source_run_config and source_run_config[key] == value:
            out[key] = "inherited_from_source_run"
        elif key in resolved_section:
            out[key] = "manual_override" if key not in defaults or defaults.get(key) != value else "default"
        else:
            out[key] = "unknown"
    return out


def _section_defaults(section_name: str) -> dict[str, Any]:
    kind = _section_kind(section_name)
    if kind == "binary":
        return {
            "feature_set": "base",
            "train_window_years": 3,
            "holdout_year": 2025,
            "operational_mode": "t10_only",
            "include_entity_id_features": False,
        }
    if kind == "stack":
        return {
            "min_train_years": 2,
            "max_train_years": 3,
            "holdout_year": 2025,
        }
    if kind == "pl":
        return {
            "train_window_years": 3,
            "holdout_year": 2025,
            "pl_feature_profile": "stack_default",
            "operational_mode": "t10_only",
            "include_final_odds_features": False,
        }
    if kind == "wide_calibrator":
        return {"method": "isotonic"}
    return {}


def _load_study_selected_values(*, study_id: str, section_name: str) -> dict[str, Any]:
    if not study_id:
        return {}
    raw = read_json(study_paths(study_id)["selected_trial"])
    values = {key: value for key, value in raw.items() if not isinstance(value, dict)}
    try:
        normalized = generate_config_from_study(study_id)
    except SystemExit:
        normalized = {}
    values.update(_nested_get(normalized, _study_config_path(section_name).split(".")))
    return values


def _study_config_path(section_name: str) -> str:
    kind = _section_kind(section_name)
    parts = _section_parts(section_name)
    if kind == "binary":
        return f"binary.{parts['task']}.{parts['model']}"
    if kind == "stack":
        return f"stacker.{parts['task']}"
    return section_name


def _build_diagnostics(
    *,
    sections: OrderedDict[str, dict[str, Any]],
    quality_detail: dict[str, Any],
    roi_detail: dict[str, Any],
) -> dict[str, Any]:
    layer_health: dict[str, Any] = {}
    pl_segments: dict[str, Any] = {}
    for name, section in sections.items():
        curated = (((quality_detail.get("sections") or {}).get(name) or {}).get("curated") or {})
        primary_metric_name, primary_metric_value = _primary_metric(curated)
        layer_health[name] = _drop_none(
            {
                "kind": section["kind"],
                "primary_metric": primary_metric_name,
                "primary_value": primary_metric_value,
            }
        )
        if section["kind"] == "pl":
            holdout_summary = section["report"].get("holdout_summary") or {}
            if holdout_summary.get("segments"):
                pl_segments[section["parts"]["profile"]] = holdout_summary.get("segments")

    return {
        "layer_health": layer_health,
        "pl_top3_segments": pl_segments,
        "slice_candidates": [
            "pl.holdout_summary.segments.all",
            "pl.holdout_summary.segments.3yo",
            "pl.holdout_summary.segments.4up",
            "backtest.input.selected_years",
            "backtest.summary.period_from",
            "wide_calibrator.fit.selected_years",
            "wide_calibrator.holdout_eval.selected_years",
        ],
        "purchase_funnel_candidates": [
            "backtest.input.rows",
            "backtest.input.pair_rows_for_backtest",
            "backtest.input.selected_races",
            "backtest.summary.n_bets",
            "backtest.summary.n_hits",
            "backtest.summary.total_bet",
            "backtest.summary.total_return",
            "backtest.summary.roi",
        ],
        "backtest_inputs": list((roi_detail.get("backtests") or {}).keys()),
    }


def _primary_metric(curated: dict[str, Any]) -> tuple[str | None, Any]:
    for key in (
        "roi",
        "logloss",
        "top3_logloss",
        "pl_nll_valid",
        "benter_r2_valid",
        "auc",
    ):
        if key in curated:
            return key, curated[key]
    if "holdout_eval" in curated and isinstance(curated["holdout_eval"], dict):
        holdout_eval = curated["holdout_eval"]
        for key in ("logloss", "brier", "auc", "ece"):
            if key in holdout_eval:
                return f"holdout_eval.{key}", holdout_eval[key]
    return None, None


def _build_code_fingerprint(
    *,
    annotation: dict[str, Any],
    sections: OrderedDict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    code_revision = dict(annotation.get("code_revision") or {})
    if "git_commit" not in code_revision:
        code_revision["git_commit"] = None

    layer_hashes: dict[str, Any] = {}
    for name, section in sections.items():
        meta = _load_json_path(section["bundle"].get("meta"))
        if meta.get("code_hash") is not None:
            layer_hashes[name] = meta.get("code_hash")
    return code_revision, {
        "git_commit": code_revision.get("git_commit"),
        "layer_code_hashes": layer_hashes,
    }


def _build_source_of_truth(
    *,
    run: dict[str, Path],
    feature_config: dict[str, Any],
    sections: OrderedDict[str, dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run": _drop_none(
            {
                "config": asset_relative(run["config"]) if run["config"].exists() else None,
                "bundle": asset_relative(run["bundle"]) if run["bundle"].exists() else None,
                "metrics": asset_relative(run["metrics"]) if run["metrics"].exists() else None,
                "resolved_params": (
                    asset_relative(run["resolved_params"]) if run["resolved_params"].exists() else None
                ),
                "annotation": (
                    asset_relative(run["execution_report_annotation"])
                    if run["execution_report_annotation"].exists()
                    else None
                ),
            }
        ),
        "feature_build": {},
        "sections": {},
    }
    feature_config_path = _feature_build_config_path(_read_toml(run["config"]))
    if feature_config_path is not None and feature_config and feature_config_path.exists():
        out["feature_build"] = {"config": asset_relative(feature_config_path)}
    for name, section in sections.items():
        out["sections"][name] = _drop_none(
            {
                "bundle_metrics_path": section["bundle"].get("metrics"),
                "report_path": section["report_path"],
                "meta_path": section["bundle"].get("meta"),
                "year_coverage_path": section["bundle"].get("year_coverage"),
            }
        )
    return out


def _parquet_years(relative_path: Any) -> list[int] | None:
    text = str(relative_path or "").strip()
    if not text:
        return None
    resolved = resolve_asset_path(text)
    if not resolved.exists():
        return None
    try:
        frame = pd.read_parquet(resolved, columns=["valid_year"])
    except Exception:
        return None
    if "valid_year" not in frame.columns:
        return None
    years = (
        pd.to_numeric(frame["valid_year"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    return [int(year) for year in years]


def _load_json_path(relative_path: Any) -> dict[str, Any]:
    text = str(relative_path or "").strip()
    if not text:
        return {}
    resolved = resolve_asset_path(text)
    return read_json(resolved)


def _auto_title(*, run_id: str, conditions: dict[str, Any]) -> str:
    feature_profile = conditions.get("feature_profile")
    feature_build_id = conditions.get("feature_build_id")
    holdout_year = conditions.get("holdout_year")
    if feature_profile and feature_build_id and holdout_year:
        return f"{run_id} · {feature_profile}/{feature_build_id} · holdout {holdout_year}"
    return str(run_id)


def _auto_description(
    *,
    run_id: str,
    conditions: dict[str, Any],
    pipeline: dict[str, Any],
    source_run_ids: list[str],
) -> str:
    active_binary = [
        f"{task}.{model}"
        for task, models in (pipeline.get("binary") or {}).items()
        for model, enabled in models.items()
        if enabled
    ]
    active_stack = [
        task for task, enabled in (pipeline.get("stack") or {}).items() if enabled
    ]
    active_pl = list(((pipeline.get("pl") or {}).get("profiles") or {}).keys())
    active_wide = list(((pipeline.get("wide_calibrator") or {}).get("methods") or {}).keys())
    active_backtest = [
        input_kind for input_kind, enabled in (pipeline.get("backtest") or {}).items() if enabled
    ]
    return (
        f"Run {run_id} derived from feature build "
        f"{conditions.get('feature_profile')}/{conditions.get('feature_build_id')} "
        f"with holdout_year={conditions.get('holdout_year')}. "
        f"binary={active_binary or ['none']}, stack={active_stack or ['none']}, "
        f"pl={active_pl or ['none']}, wide={active_wide or ['none']}, "
        f"backtest={active_backtest or ['none']}, source_runs={source_run_ids or ['self']}."
    )


def _compute_status(
    *,
    issues: list[dict[str, str]],
    pipeline: dict[str, Any],
    backtest_summary: dict[str, Any],
) -> str:
    if issues:
        return "invalid"
    has_train_section = any(
        [
            any(enabled for models in (pipeline.get("binary") or {}).values() for enabled in models.values()),
            any((pipeline.get("stack") or {}).values()),
            bool((pipeline.get("pl") or {}).get("enabled")),
            bool((pipeline.get("wide_calibrator") or {}).get("enabled")),
        ]
    )
    if not has_train_section:
        return "invalid"
    if not backtest_summary:
        return "partial"
    return "complete"


def _metric_mean(value: Any) -> float | None:
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if _is_number(mean) else None
    return float(value) if _is_number(value) else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nested_get(payload: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict):
            return {}
        current = current.get(part, {})
    return dict(current) if isinstance(current, dict) else {}


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            child = _drop_empty(value)
            if child:
                out[key] = child
            continue
        if value in (None, [], {}):
            continue
        out[key] = value
    return out

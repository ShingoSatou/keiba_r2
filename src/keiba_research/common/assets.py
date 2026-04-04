from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ASSET_ENV = "V3_ASSET_ROOT"
VALID_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_ROOT = PROJECT_ROOT / ".local" / "v3_assets"


def _require_id(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise SystemExit(f"{label} is required")
    if not VALID_ID_RE.fullmatch(text):
        raise SystemExit(
            f"{label} must match {VALID_ID_RE.pattern} and may not contain path separators: {text}"
        )
    return text


def asset_root() -> Path:
    raw = os.getenv(ASSET_ENV, "").strip()
    root = Path(raw).expanduser().resolve() if raw else DEFAULT_ASSET_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def jsonl_root() -> Path:
    return asset_root() / "data" / "jsonl"


def cache_root() -> Path:
    root = asset_root() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def feature_build_root(feature_profile: str, feature_build_id: str) -> Path:
    profile = _require_id(feature_profile, label="feature_profile")
    build_id = _require_id(feature_build_id, label="feature_build_id")
    root = asset_root() / "data" / "features" / profile / build_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def feature_build_paths(feature_profile: str, feature_build_id: str) -> dict[str, Path]:
    root = feature_build_root(feature_profile, feature_build_id)
    return {
        "root": root,
        "config": root / "config.toml",
        "base": root / "features_base.parquet",
        "base_meta": root / "features_base_meta.json",
        "base_te": root / "features_base_te.parquet",
        "base_te_meta": root / "features_base_te_meta.json",
        "features": root / "features_v3.parquet",
        "features_meta": root / "features_v3_meta.json",
        "features_te": root / "features_v3_te.parquet",
        "features_te_meta": root / "features_v3_te_meta.json",
    }


def study_root(study_id: str) -> Path:
    study = _require_id(study_id, label="study_id")
    if study.startswith("imported."):
        imported_id = _require_id(study.removeprefix("imported."), label="imported_study_id")
        root = asset_root() / "studies" / "imported" / imported_id
    else:
        root = asset_root() / "studies" / study
    root.mkdir(parents=True, exist_ok=True)
    return root


def study_paths(study_id: str) -> dict[str, Path]:
    root = study_root(study_id)
    return {
        "root": root,
        "config": root / "config.toml",
        "storage": root / "study.sqlite3",
        "selected_trial": root / "selected_trial.json",
        "best": root / "best.json",
        "trials": root / "trials.parquet",
    }


def run_root(run_id: str) -> Path:
    run = _require_id(run_id, label="run_id")
    root = asset_root() / "runs" / run
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_paths(run_id: str) -> dict[str, Path]:
    root = run_root(run_id)
    artifacts = root / "artifacts"
    models = artifacts / "models"
    oof = artifacts / "oof"
    holdout = artifacts / "holdout"
    predictions = artifacts / "predictions"
    reports = artifacts / "reports"
    for path in (artifacts, models, oof, holdout, predictions, reports):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "config": root / "config.toml",
        "bundle": root / "bundle.json",
        "metrics": root / "metrics.json",
        "resolved_params": root / "resolved_params.toml",
        "execution_report_summary": root / "execution_report_summary.json",
        "execution_report_detail": root / "execution_report_detail.json",
        "execution_report_annotation": root / "execution_report_annotation.toml",
        "artifacts": artifacts,
        "models": models,
        "oof": oof,
        "holdout": holdout,
        "predictions": predictions,
        "reports": reports,
    }


def asset_relative(path: Path | str) -> str:
    root = asset_root()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise SystemExit(f"path is outside {ASSET_ENV}: {resolved}") from exc


def resolve_asset_path(relative_path: str) -> Path:
    rel = str(relative_path).strip()
    if not rel:
        raise SystemExit("relative asset path is required")
    path = Path(rel)
    if path.is_absolute():
        raise SystemExit("absolute paths are not allowed in run/study metadata")
    return (asset_root() / path).resolve()


def _absolute_path_string(value: str) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("sqlite:///"):
        candidate = Path(text.removeprefix("sqlite:///"))
        return text if candidate.is_absolute() else None
    candidate = Path(text)
    return text if candidate.is_absolute() else None


def relativize_asset_value(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: relativize_asset_value(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [relativize_asset_value(value) for value in payload]
    if not isinstance(payload, str):
        return payload

    absolute = _absolute_path_string(payload)
    if absolute is None:
        return payload

    if absolute.startswith("sqlite:///"):
        raw_path = absolute.removeprefix("sqlite:///")
        try:
            return f"sqlite:///{asset_relative(raw_path)}"
        except SystemExit:
            return payload

    try:
        return asset_relative(absolute)
    except SystemExit:
        candidate = None
        resolved = Path(absolute).resolve()
        if resolved.suffix == ".jsonl":
            jsonl_candidate = jsonl_root() / resolved.name
            if jsonl_candidate.exists():
                candidate = jsonl_candidate
        if candidate is None:
            return payload
        try:
            return candidate.relative_to(asset_root()).as_posix()
        except ValueError:
            return payload


def collect_absolute_path_strings(payload: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            hits.extend(collect_absolute_path_strings(value, prefix=child_prefix))
        return hits
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}[{index}]"
            hits.extend(collect_absolute_path_strings(value, prefix=child_prefix))
        return hits
    if isinstance(payload, str):
        absolute = _absolute_path_string(payload)
        if absolute is not None:
            hits.append((prefix, absolute))
    return hits


def rewrite_json_asset_paths(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = relativize_asset_value(payload)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_json_has_no_absolute_paths(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits = collect_absolute_path_strings(payload)
    if not hits:
        return
    sample = ", ".join(f"{label}={value}" for label, value in hits[:5])
    raise SystemExit(f"absolute paths are not allowed in metadata {path}: {sample}")


def read_json(path: Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keiba_research.common.assets import (
    asset_relative,
    read_json,
    run_paths,
    study_paths,
    write_json,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value)!r}")


def _dump_toml_sections(payload: dict[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    scalars: dict[str, Any] = {}
    tables: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            tables[key] = value
        else:
            scalars[key] = value

    if prefix:
        lines.append(f"[{prefix}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")

    for key, value in tables.items():
        if lines:
            lines.append("")
        child_prefix = f"{prefix}.{key}" if prefix else key
        lines.extend(_dump_toml_sections(value, child_prefix))
    return lines


def write_toml(path: Path, payload: dict[str, Any]) -> None:
    lines = _dump_toml_sections(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _merge_strict(
    existing: dict[str, Any], updates: dict[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        label = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            current = merged.get(key, {})
            if current is None:
                current = {}
            if not isinstance(current, dict):
                raise SystemExit(f"conflicting config value for {label}")
            merged[key] = _merge_strict(current, value, prefix=label)
            continue
        if key in merged and merged[key] != value:
            raise SystemExit(f"conflicting config value for {label}: {merged[key]!r} != {value!r}")
        merged[key] = value
    return merged


def update_run_config(run_id: str, patch: dict[str, Any]) -> Path:
    paths = run_paths(run_id)
    current = {}
    if paths["config"].exists():
        import tomllib

        current = tomllib.loads(paths["config"].read_text(encoding="utf-8"))
    merged = _merge_strict(current, patch)
    write_toml(paths["config"], merged)
    return paths["config"]


def update_study_config(study_id: str, patch: dict[str, Any]) -> Path:
    paths = study_paths(study_id)
    current = {}
    if paths["config"].exists():
        import tomllib

        current = tomllib.loads(paths["config"].read_text(encoding="utf-8"))
    merged = _merge_strict(current, patch)
    write_toml(paths["config"], merged)
    return paths["config"]


def load_study_config(study_id: str) -> dict[str, Any]:
    paths = study_paths(study_id)
    if not paths["config"].exists():
        return {}
    import tomllib

    return tomllib.loads(paths["config"].read_text(encoding="utf-8"))


def update_run_bundle(run_id: str, section: str, payload: dict[str, Any]) -> Path:
    paths = run_paths(run_id)
    bundle = read_json(
        paths["bundle"],
        default={
            "bundle_version": 1,
            "run_id": run_id,
            "generated_at": utc_now(),
            "sections": {},
        },
    )
    sections = bundle.setdefault("sections", {})
    sections[str(section)] = payload
    bundle["generated_at"] = utc_now()
    write_json(paths["bundle"], bundle)
    return paths["bundle"]


def update_run_metrics(run_id: str, section: str, payload: dict[str, Any]) -> Path:
    paths = run_paths(run_id)
    metrics = read_json(
        paths["metrics"],
        default={
            "metrics_version": 1,
            "run_id": run_id,
            "generated_at": utc_now(),
            "sections": {},
        },
    )
    sections = metrics.setdefault("sections", {})
    sections[str(section)] = payload
    metrics["generated_at"] = utc_now()
    write_json(paths["metrics"], metrics)
    return paths["metrics"]


def asset_payload(**paths: Path) -> dict[str, str]:
    return {key: asset_relative(path) for key, path in paths.items()}

"""Declarative run configuration via TOML.

Provides helpers for loading, validating, and generating run_config.toml files
that consolidate parameter sources (CLI, study, defaults) into a single record.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from keiba_research.common.assets import run_paths, study_paths
from keiba_research.common.state import load_study_config, write_toml


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load and return a run_config.toml as a nested dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _study_params_to_binary_section(
    params: dict[str, Any],
    *,
    task: str,
    model: str,
) -> dict[str, Any]:
    """Convert a selected_trial.json params dict to a binary config section."""
    section: dict[str, Any] = {}
    model_params = params.get("lgbm_params") or params.get("xgb_params") or params.get("cat_params")
    if isinstance(model_params, dict):
        for key, value in model_params.items():
            section[key] = value
    if "final_num_boost_round" in params:
        section["final_num_boost_round"] = int(params["final_num_boost_round"])
    if "final_iterations" in params:
        section["final_iterations"] = int(params["final_iterations"])
    if "feature_set" in params:
        section["feature_set"] = str(params["feature_set"])
    if "train_window_years" in params:
        section["train_window_years"] = int(params["train_window_years"])
    return section


def _study_params_to_stacker_section(
    params: dict[str, Any],
    *,
    task: str,
) -> dict[str, Any]:
    """Convert a selected_trial.json params dict to a stacker config section."""
    section: dict[str, Any] = {}
    model_params = params.get("lgbm_params", {})
    if isinstance(model_params, dict):
        for key, value in model_params.items():
            section[key] = value
    if "final_num_boost_round" in params:
        section["final_num_boost_round"] = int(params["final_num_boost_round"])
    if "min_train_years" in params:
        section["min_train_years"] = int(params["min_train_years"])
    if "max_train_years" in params:
        section["max_train_years"] = int(params["max_train_years"])
    return section


def generate_config_from_study(study_id: str) -> dict[str, Any]:
    """Read a study's selected_trial.json and generate a run_config dict.

    The generated dict follows the TOML schema:
    [binary.<task>.<model>], [stacker.<task>], etc.
    """
    paths = study_paths(study_id)
    trial_path = paths["selected_trial"]
    if not trial_path.exists():
        raise SystemExit(f"selected_trial.json not found for study {study_id}: {trial_path}")

    params = json.loads(trial_path.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise SystemExit("selected_trial.json must be a JSON object")

    config: dict[str, Any] = {}

    study_cfg = load_study_config(study_id)
    kind = str(study_cfg.get("kind", params.get("study_type", ""))).strip()
    task = str(params.get("task", "win"))
    model = str(params.get("model", "lgbm"))

    has_model_params = (
        "lgbm_params" in params or "xgb_params" in params or "cat_params" in params
    )
    if has_model_params:
        if kind == "stack":
            config.setdefault("stacker", {})
            config["stacker"][task] = _study_params_to_stacker_section(params, task=task)
        else:
            config.setdefault("binary", {}).setdefault(task, {})
            config["binary"][task][model] = _study_params_to_binary_section(
                params, task=task, model=model
            )

    return config


def save_resolved_params(run_id: str, section: str, params: dict[str, Any]) -> Path:
    """Save the resolved parameters used for a training step.

    Writes to runs/<run_id>/resolved_params.toml, merging with existing sections.
    """
    paths = run_paths(run_id)
    resolved_path = paths["root"] / "resolved_params.toml"

    existing: dict[str, Any] = {}
    if resolved_path.exists():
        existing = tomllib.loads(resolved_path.read_text(encoding="utf-8"))

    parts = section.split(".")
    target = existing
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = params

    write_toml(resolved_path, existing)
    return resolved_path

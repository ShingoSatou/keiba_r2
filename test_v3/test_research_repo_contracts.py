from __future__ import annotations

import inspect
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import keiba_research.common.assets as assets_module
import keiba_research.training.commands as training_commands
from keiba_research.cli import build_parser
from keiba_research.common.assets import (
    asset_relative,
    collect_absolute_path_strings,
    ensure_json_has_no_absolute_paths,
    resolve_asset_path,
    rewrite_json_asset_paths,
    run_paths,
    study_paths,
)
from keiba_research.common.state import (
    asset_payload,
    load_study_config,
    update_run_bundle,
    update_run_config,
    update_run_metrics,
    update_study_config,
)
from keiba_research.common.v3_utils import hash_files, resolve_path
from keiba_research.db.commands import handle_rebuild
from keiba_research.evaluation.commands import handle_backtest, handle_compare, handle_report
from keiba_research.training.binary import run_binary_training
from keiba_research.training.commands import (
    handle_binary,
    handle_pl,
    handle_stack,
    handle_wide_calibrator,
)
from keiba_research.training.cv_policy import (
    build_capped_expanding_year_folds,
    build_fixed_window_year_folds,
    select_recent_window_years,
)
from keiba_research.training.stacker import (
    _meta_code_hash_paths,
    run_stacker_training,
)
from keiba_research.training.stacker_common import _meta_payload
from keiba_research.training.wide_calibrator import (
    run_wide_calibrator,
    wide_calibrator_artifact_paths,
)
from keiba_research.tuning.commands import (
    _assert_study_writable,
)
from keiba_research.tuning.commands import (
    handle_binary as handle_tune_binary,
)


@pytest.fixture()
def asset_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "assets"
    monkeypatch.setenv("V3_ASSET_ROOT", str(root))
    return root


def _argv_to_dict(argv: list[str]) -> dict[str, str | bool]:
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            parsed[token] = argv[index + 1]
            index += 2
        else:
            parsed[token] = True
            index += 1
    return parsed


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_execution_report_run(
    asset_root_env: Path,
    *,
    run_id: str,
    source_run_id: str = "",
    with_backtest: bool = True,
    annotation: dict[str, object] | None = None,
    study_id: str = "",
) -> Path:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "config.toml").write_text(
        '\n'.join(
            [
                'feature_profile = "baseline_v3"',
                'feature_build_id = "build_001"',
                'from_date = "2016-01-01"',
                'to_date = "2025-12-31"',
                "history_days = 730",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (feature_root / "features_v3.parquet").write_text("features", encoding="utf-8")

    if source_run_id:
        update_run_config(
            source_run_id,
            {
                "run_id": source_run_id,
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "holdout_year": 2025,
            },
        )

    update_run_config(
        run_id,
        {
            "run_id": run_id,
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
            "holdout_year": 2025,
            "pl_feature_profile": "stack_default",
        },
    )
    run = run_paths(run_id)

    binary_metrics = {
        "summary": {
            "logloss": {"mean": 0.41},
            "brier": {"mean": 0.14},
            "auc": {"mean": 0.77},
            "ece": {"mean": 0.03},
            "benter_r2_valid": {"mean": 0.12},
        },
        "data_summary": {
            "rows": 1200,
            "races": 120,
            "years": [2022, 2023, 2024],
            "oof_rows": 900,
            "oof_races": 90,
            "holdout_rows": 300,
            "holdout_races": 30,
        },
        "cv_policy": {"valid_years": [2022, 2023, 2024]},
        "config": {
            "holdout_year": 2025,
            "train_window_years": 3,
            "operational_mode": "t10_only",
            "include_entity_id_features": False,
            "params_json": "studies/study_001/selected_trial.json" if study_id else None,
        },
    }
    binary_metrics_path = run["reports"] / "win_lgbm_cv_metrics.json"
    _write_json_file(binary_metrics_path, binary_metrics)
    pd.DataFrame({"valid_year": [2025, 2026]}).to_parquet(
        run["holdout"] / "win_lgbm_holdout_2025.parquet", index=False
    )
    (run["oof"] / "win_lgbm_oof.parquet").write_text("oof", encoding="utf-8")
    (run["models"] / "win_lgbm_v3.txt").write_text("model", encoding="utf-8")
    _write_json_file(run["models"] / "win_lgbm_bundle_meta_v3.json", {"code_hash": "binary-hash"})
    update_run_bundle(
        run_id,
        "binary.win.lgbm",
        {
            "feature_set": "te",
            "study_id": study_id or None,
            "input": asset_relative(feature_root / "features_v3.parquet"),
            "oof": asset_relative(run["oof"] / "win_lgbm_oof.parquet"),
            "holdout": asset_relative(run["holdout"] / "win_lgbm_holdout_2025.parquet"),
            "metrics": asset_relative(binary_metrics_path),
            "model": asset_relative(run["models"] / "win_lgbm_v3.txt"),
            "meta": asset_relative(run["models"] / "win_lgbm_bundle_meta_v3.json"),
        },
    )
    update_run_metrics(
        run_id,
        "binary.win.lgbm",
        {"path": asset_relative(binary_metrics_path), "report": binary_metrics},
    )

    stack_metrics = {
        "summary": {
            "logloss": {"mean": 0.39},
            "brier": {"mean": 0.13},
            "auc": {"mean": 0.79},
            "ece": {"mean": 0.02},
        },
        "data_summary": {"rows": 900, "races": 90, "years": [2023, 2024]},
        "cv_policy": {"valid_years": [2023, 2024]},
        "config": {
            "holdout_year": 2025,
            "min_train_window_years": 2,
            "train_window_years": 3,
            "params_json": "studies/stack_study/selected_trial.json",
        },
    }
    stack_metrics_path = run["reports"] / "win_stack_cv_metrics.json"
    _write_json_file(stack_metrics_path, stack_metrics)
    pd.DataFrame({"valid_year": [2025]}).to_parquet(
        run["holdout"] / "win_stack_holdout_2025.parquet", index=False
    )
    (run["oof"] / "win_stack_oof.parquet").write_text("oof", encoding="utf-8")
    (run["models"] / "win_stack_v3.txt").write_text("model", encoding="utf-8")
    _write_json_file(run["models"] / "win_stack_bundle_meta_v3.json", {"code_hash": "stack-hash"})
    update_run_bundle(
        run_id,
        "stack.win",
        {
            "source_run_id": source_run_id or run_id,
            "oof": asset_relative(run["oof"] / "win_stack_oof.parquet"),
            "holdout": asset_relative(run["holdout"] / "win_stack_holdout_2025.parquet"),
            "metrics": asset_relative(stack_metrics_path),
            "model": asset_relative(run["models"] / "win_stack_v3.txt"),
            "meta": asset_relative(run["models"] / "win_stack_bundle_meta_v3.json"),
        },
    )
    update_run_metrics(
        run_id,
        "stack.win",
        {"path": asset_relative(stack_metrics_path), "report": stack_metrics},
    )

    pl_metrics = {
        "summary": {
            "pl_nll_valid": {"mean": 1.11},
            "top3_logloss": {"mean": 0.58},
            "top3_brier": {"mean": 0.18},
            "top3_auc": {"mean": 0.72},
            "top3_ece": {"mean": 0.04},
        },
        "data_summary": {"rows": 600, "races": 60, "years": [2024]},
        "holdout_summary": {
            "rows": 200,
            "races": 20,
            "years": [2025],
            "top3_logloss": 0.57,
            "top3_brier": 0.17,
            "top3_auc": 0.73,
            "top3_ece": 0.03,
            "segments": {
                "all": {"rows": 200, "logloss": 0.57},
                "3yo": {"rows": 80, "logloss": 0.61},
                "4up": {"rows": 120, "logloss": 0.54},
            },
        },
        "config": {
            "holdout_year": 2025,
            "train_window_years": 3,
            "operational_mode": "t10_only",
            "pl_feature_profile": "stack_default",
        },
    }
    pl_metrics_path = run["reports"] / "pl_stack_default_cv_metrics.json"
    _write_json_file(pl_metrics_path, pl_metrics)
    _write_json_file(
        run["reports"] / "pl_stack_default_year_coverage.json",
        {
            "base_oof_years": [2022, 2023, 2024],
            "stacker_oof_years": [2023, 2024],
            "pl_oof_valid_years": [2024],
            "pl_holdout_train_years": [2022, 2023, 2024],
        },
    )
    pd.DataFrame({"valid_year": [2025]}).to_parquet(
        run["holdout"] / "pl_stack_default_holdout_2025.parquet", index=False
    )
    (run["oof"] / "pl_stack_default_oof.parquet").write_text("oof", encoding="utf-8")
    _write_json_file(run["models"] / "pl_stack_default_bundle_meta.json", {"code_hash": "pl-hash"})
    update_run_bundle(
        run_id,
        "pl.stack_default",
        {
            "source_run_id": source_run_id or run_id,
            "oof": asset_relative(run["oof"] / "pl_stack_default_oof.parquet"),
            "holdout": asset_relative(run["holdout"] / "pl_stack_default_holdout_2025.parquet"),
            "metrics": asset_relative(pl_metrics_path),
            "meta": asset_relative(run["models"] / "pl_stack_default_bundle_meta.json"),
            "year_coverage": asset_relative(run["reports"] / "pl_stack_default_year_coverage.json"),
        },
    )
    update_run_metrics(
        run_id,
        "pl.stack_default",
        {"path": asset_relative(pl_metrics_path), "report": pl_metrics},
    )

    wide_metrics = {
        "fit": {
            "rows": 80,
            "races": 8,
            "selected_years": [2024],
            "calibrated": {"logloss": 0.48, "brier": 0.16, "auc": 0.71, "ece": 0.02},
        },
        "holdout_eval": {
            "rows": 20,
            "races": 2,
            "selected_years": [2025],
            "calibrated": {"logloss": 0.46, "brier": 0.15, "auc": 0.74, "ece": 0.01},
        },
    }
    wide_metrics_path = run["reports"] / "wide_pair_calibration_isotonic_metrics.json"
    _write_json_file(wide_metrics_path, wide_metrics)
    (run["predictions"] / "wide_pair_calibration_isotonic_pred.parquet").write_text(
        "pred", encoding="utf-8"
    )
    _write_json_file(
        run["models"] / "wide_pair_calibrator_isotonic_bundle_meta.json",
        {"code_hash": "wide-hash"},
    )
    update_run_bundle(
        run_id,
        "wide_calibrator.isotonic",
        {
            "source_run_id": source_run_id or run_id,
            "predictions": asset_relative(
                run["predictions"] / "wide_pair_calibration_isotonic_pred.parquet"
            ),
            "metrics": asset_relative(wide_metrics_path),
            "meta": asset_relative(run["models"] / "wide_pair_calibrator_isotonic_bundle_meta.json"),
        },
    )
    update_run_metrics(
        run_id,
        "wide_calibrator.isotonic",
        {"path": asset_relative(wide_metrics_path), "report": wide_metrics},
    )

    if with_backtest:
        backtest_report = {
            "summary": {
                "period_from": "2025-01-01",
                "period_to": "2025-12-31",
                "n_races": 20,
                "n_bets": 30,
                "n_hits": 10,
                "hit_rate": 0.3333,
                "total_bet": 3000,
                "total_return": 4200,
                "roi": 1.4,
                "max_drawdown": -500,
                "logloss": 0.56,
                "auc": 0.71,
            }
        }
        backtest_meta = {
            "input": {
                "input_mode": "horse",
                "p_wide_source": "v3_pl_score_mc",
                "rows": 200,
                "pair_rows_for_backtest": 600,
                "selected_races": 20,
                "selected_years": [2025],
                "available_years_after_holdout_filter": [2025],
                "input_filter_holdout_year": 2026,
            },
            "config": {
                "selection": {
                    "min_p_wide": 0.0,
                    "min_p_wide_stage": "candidate",
                    "ev_threshold": 0.0,
                    "max_bets_per_race": 5,
                },
                "bankroll": {
                    "bankroll_init_yen": 1_000_000,
                    "kelly_fraction_scale": 0.25,
                    "race_cap_fraction": 0.05,
                    "daily_cap_fraction": 0.2,
                    "bet_unit_yen": 100,
                    "min_bet_yen": 100,
                    "max_bet_yen": None,
                },
            },
        }
        backtest_report_path = run["reports"] / "backtest_pl_holdout.json"
        backtest_meta_path = run["reports"] / "backtest_pl_holdout_meta.json"
        _write_json_file(backtest_report_path, backtest_report)
        _write_json_file(backtest_meta_path, backtest_meta)
        update_run_bundle(
            run_id,
            "backtest.pl_holdout",
            {
                "input_kind": "pl_holdout",
                "pl_feature_profile": "stack_default",
                "report": asset_relative(backtest_report_path),
                "meta": asset_relative(backtest_meta_path),
            },
        )
        update_run_metrics(
            run_id,
            "backtest.pl_holdout",
            {"path": asset_relative(backtest_report_path), "report": backtest_report},
        )

    (run["resolved_params"]).write_text(
        '\n'.join(
            [
                "[binary.win.lgbm]",
                'feature_set = "te"',
                "holdout_year = 2025",
                "train_window_years = 3",
                "num_leaves = 63",
                "final_num_boost_round = 123",
                "",
                "[stack.win]",
                f'source_run_id = "{source_run_id or run_id}"',
                "holdout_year = 2025",
                "min_train_years = 2",
                "max_train_years = 3",
                "",
                "[pl.stack_default]",
                'pl_feature_profile = "stack_default"',
                "holdout_year = 2025",
                "train_window_years = 3",
                "",
                "[wide_calibrator.isotonic]",
                'method = "isotonic"',
                f'source_run_id = "{source_run_id or run_id}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    if study_id:
        study = study_paths(study_id)
        update_study_config(
            study_id,
            {
                "study_id": study_id,
                "kind": "binary",
                "task": "win",
                "model": "lgbm",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "imported": False,
                "read_only_seed": False,
            },
        )
        _write_json_file(
            study["selected_trial"],
            {
                "task": "win",
                "model": "lgbm",
                "feature_set": "te",
                "train_window_years": 3,
                "lgbm_params": {"num_leaves": 63},
                "final_num_boost_round": 123,
            },
        )

    if annotation:
        lines: list[str] = []
        for key in ("title", "description", "status"):
            if key in annotation:
                lines.append(f'{key} = "{annotation[key]}"')
        if "code_revision" in annotation:
            lines.extend(
                [
                    "",
                    "[code_revision]",
                    f'git_commit = "{annotation["code_revision"]}"',
                ]
            )
        (run["execution_report_annotation"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run["root"]


def test_binary_config_mapping_matches_run_binary_signature() -> None:
    signature = inspect.signature(run_binary_training)
    assert set(training_commands._BINARY_CONFIG_KWARGS.values()) <= set(signature.parameters)


def test_stacker_config_mapping_matches_run_stacker_signature() -> None:
    signature = inspect.signature(run_stacker_training)
    assert set(training_commands._STACKER_CONFIG_KWARGS.values()) <= set(signature.parameters)


def test_train_binary_wrapper_normalizes_num_boost_round_config_alias(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_v3.parquet").write_text("placeholder", encoding="utf-8")
    config_path = asset_root_env.parent / "binary_alias.toml"
    config_path.write_text(
        "[binary.win.lgbm]\nlearning_rate = 0.03\nnum_boost_round = 123\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_binary_training(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["all_years_model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["oof_output"])).write_text("oof", encoding="utf-8")
        Path(str(kwargs["holdout_output"])).write_text("holdout", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["feature_manifest_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_binary_training", fake_run_binary_training
    )

    rc = handle_binary(
        type(
            "Args",
            (),
            {
                "run_id": "binary_config_alias",
                "task": "win",
                "model": "lgbm",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "feature_set": "base",
                "config": str(config_path),
                "study_id": "",
                "holdout_year": 2025,
                "train_window_years": 3,
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["num_boost_round"] == 123
    resolved = tomllib.loads(
        (run_paths("binary_config_alias")["root"] / "resolved_params.toml").read_text(
            encoding="utf-8"
        )
    )
    params = resolved["binary"]["win"]["lgbm"]
    assert params["final_num_boost_round"] == 123
    assert "num_boost_round" not in params


def test_train_binary_wrapper_rejects_duplicate_boost_round_config_keys(
    asset_root_env: Path,
) -> None:
    config_path = asset_root_env.parent / "binary_duplicate.toml"
    config_path.write_text(
        "[binary.win.lgbm]\nfinal_num_boost_round = 120\nnum_boost_round = 121\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot contain both"):
        handle_binary(
            type(
                "Args",
                (),
                {
                    "run_id": "binary_config_duplicate",
                    "task": "win",
                    "model": "lgbm",
                    "feature_profile": "baseline_v3",
                    "feature_build_id": "build_001",
                    "feature_set": "base",
                    "config": str(config_path),
                    "study_id": "",
                    "holdout_year": 2025,
                    "train_window_years": 3,
                    "database_url": "",
                    "log_level": "INFO",
                },
            )()
        )


def test_config_section_to_kwargs_rejects_multiple_sources_for_same_kwarg() -> None:
    with pytest.raises(SystemExit, match="multiple keys that map to num_boost_round"):
        training_commands._config_section_to_kwargs(
            {
                "final_num_boost_round": 120,
                "final_iterations": 121,
            },
            key_map={
                "final_num_boost_round": "num_boost_round",
                "final_iterations": "num_boost_round",
            },
            context="binary.win.cat",
        )


def test_train_binary_wrapper_normalizes_num_boost_round_config_alias_for_cat(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_v3.parquet").write_text("placeholder", encoding="utf-8")
    config_path = asset_root_env.parent / "binary_cat_alias.toml"
    config_path.write_text(
        "[binary.win.cat]\ndepth = 6\nnum_boost_round = 55\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_binary_training(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["all_years_model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["oof_output"])).write_text("oof", encoding="utf-8")
        Path(str(kwargs["holdout_output"])).write_text("holdout", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["feature_manifest_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_binary_training", fake_run_binary_training
    )

    rc = handle_binary(
        type(
            "Args",
            (),
            {
                "run_id": "binary_cat_config_alias",
                "task": "win",
                "model": "cat",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "feature_set": "base",
                "config": str(config_path),
                "study_id": "",
                "holdout_year": 2025,
                "train_window_years": 3,
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["num_boost_round"] == 55
    resolved = tomllib.loads(
        (run_paths("binary_cat_config_alias")["root"] / "resolved_params.toml").read_text(
            encoding="utf-8"
        )
    )
    params = resolved["binary"]["win"]["cat"]
    assert params["final_iterations"] == 55
    assert "num_boost_round" not in params


def test_train_binary_wrapper_rejects_cat_alias_and_canonical_iteration_keys(
    asset_root_env: Path,
) -> None:
    config_path = asset_root_env.parent / "binary_cat_duplicate.toml"
    config_path.write_text(
        "[binary.win.cat]\nfinal_iterations = 54\nnum_boost_round = 55\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot contain both"):
        handle_binary(
            type(
                "Args",
                (),
                {
                    "run_id": "binary_cat_config_duplicate",
                    "task": "win",
                    "model": "cat",
                    "feature_profile": "baseline_v3",
                    "feature_build_id": "build_001",
                    "feature_set": "base",
                    "config": str(config_path),
                    "study_id": "",
                    "holdout_year": 2025,
                    "train_window_years": 3,
                    "database_url": "",
                    "log_level": "INFO",
                },
            )()
        )


def test_train_stack_wrapper_normalizes_num_boost_round_config_alias(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_v3.parquet").write_text("placeholder", encoding="utf-8")
    source = run_paths("stack_source_alias")
    source["oof"].mkdir(parents=True, exist_ok=True)
    source["holdout"].mkdir(parents=True, exist_ok=True)
    for model in ("lgbm", "xgb", "cat"):
        (source["oof"] / f"win_{model}_oof.parquet").write_text("oof", encoding="utf-8")
        (source["holdout"] / f"win_{model}_holdout_2025.parquet").write_text(
            "holdout",
            encoding="utf-8",
        )
    config_path = asset_root_env.parent / "stack_alias.toml"
    config_path.write_text(
        "[stacker.win]\nnum_boost_round = 77\nmin_train_years = 1\nmax_train_years = 2\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_stacker_training(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["all_years_model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["oof_output"])).write_text("oof", encoding="utf-8")
        Path(str(kwargs["holdout_output"])).write_text("holdout", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["feature_manifest_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_stacker_training", fake_run_stacker_training
    )

    rc = handle_stack(
        type(
            "Args",
            (),
            {
                "run_id": "stack_config_alias",
                "task": "win",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "source_run_id": "stack_source_alias",
                "config": str(config_path),
                "study_id": "",
                "holdout_year": 2025,
                "min_train_years": 2,
                "max_train_years": 3,
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["num_boost_round"] == 77
    resolved = tomllib.loads(
        (run_paths("stack_config_alias")["root"] / "resolved_params.toml").read_text(
            encoding="utf-8"
        )
    )
    params = resolved["stack"]["win"]
    assert params["final_num_boost_round"] == 77
    assert "num_boost_round" not in params


def test_train_stack_wrapper_rejects_duplicate_boost_round_config_keys(
    asset_root_env: Path,
) -> None:
    config_path = asset_root_env.parent / "stack_duplicate.toml"
    config_path.write_text(
        "[stacker.win]\nfinal_num_boost_round = 80\nnum_boost_round = 81\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot contain both"):
        handle_stack(
            type(
                "Args",
                (),
                {
                    "run_id": "stack_config_duplicate",
                    "task": "win",
                    "feature_profile": "baseline_v3",
                    "feature_build_id": "build_001",
                    "source_run_id": "stack_source_duplicate",
                    "config": str(config_path),
                    "study_id": "",
                    "holdout_year": 2025,
                    "min_train_years": 2,
                    "max_train_years": 3,
                    "log_level": "INFO",
                },
            )()
        )


def test_wide_calibrator_artifact_paths_are_method_aware(asset_root_env: Path) -> None:
    run = run_paths("wide_paths")
    isotonic_paths = wide_calibrator_artifact_paths(run, method="isotonic")
    logreg_paths = wide_calibrator_artifact_paths(run, method="logreg")

    assert isotonic_paths["model"].name == "wide_pair_calibrator_isotonic.joblib"
    assert isotonic_paths["predictions"].name == "wide_pair_calibration_isotonic_pred.parquet"
    assert logreg_paths["model"].name == "wide_pair_calibrator_logreg.joblib"
    assert logreg_paths["meta"].name == "wide_pair_calibrator_logreg_bundle_meta.json"
    assert logreg_paths["predictions"].name == "wide_pair_calibration_logreg_pred.parquet"
    assert logreg_paths["metrics"].name == "wide_pair_calibration_logreg_metrics.json"


def test_run_bundle_uses_asset_relative_paths(asset_root_env: Path) -> None:
    run = run_paths("baseline_run")
    payload_path = run["reports"] / "metrics.json"
    payload_path.write_text("{}", encoding="utf-8")

    update_run_config(
        "baseline_run",
        {
            "run_id": "baseline_run",
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
        },
    )
    update_run_bundle(
        "baseline_run",
        "binary.win.lgbm",
        asset_payload(metrics=payload_path),
    )
    update_run_metrics(
        "baseline_run",
        "binary.win.lgbm",
        {"path": asset_relative(payload_path), "report": {"summary": {"logloss": {"mean": 0.12}}}},
    )

    bundle = json.loads(run["bundle"].read_text(encoding="utf-8"))
    metrics = json.loads(run["metrics"].read_text(encoding="utf-8"))

    assert (
        bundle["sections"]["binary.win.lgbm"]["metrics"]
        == "runs/baseline_run/artifacts/reports/metrics.json"
    )
    assert (
        metrics["sections"]["binary.win.lgbm"]["path"]
        == "runs/baseline_run/artifacts/reports/metrics.json"
    )
    assert (
        resolve_asset_path(bundle["sections"]["binary.win.lgbm"]["metrics"])
        == payload_path.resolve()
    )


def test_asset_relative_rejects_paths_outside_asset_root(
    asset_root_env: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        asset_relative(outside)


def test_rewrite_json_asset_paths_rewrites_nested_asset_paths(asset_root_env: Path) -> None:
    sample_json = asset_root_env / "cache" / "sample.json"
    study_storage = asset_root_env / "studies" / "study_001" / "study.sqlite3"
    feature_input = asset_root_env / "data" / "features" / "baseline_v3" / "build_001" / "f.parquet"
    sample_json.parent.mkdir(parents=True, exist_ok=True)
    study_storage.parent.mkdir(parents=True, exist_ok=True)
    feature_input.parent.mkdir(parents=True, exist_ok=True)
    sample_json.write_text(
        json.dumps(
            {
                "storage": f"sqlite:///{study_storage}",
                "nested": {
                    "input": str(feature_input),
                },
            }
        ),
        encoding="utf-8",
    )

    rewrite_json_asset_paths(sample_json)
    ensure_json_has_no_absolute_paths(sample_json)

    payload = json.loads(sample_json.read_text(encoding="utf-8"))
    assert payload["storage"] == "sqlite:///studies/study_001/study.sqlite3"
    assert payload["nested"]["input"] == "data/features/baseline_v3/build_001/f.parquet"


def test_imported_study_is_read_only_seed(asset_root_env: Path) -> None:
    study = study_paths("imported.binary_seed")
    study["selected_trial"].write_text("{}", encoding="utf-8")
    update_study_config(
        "imported.binary_seed",
        {
            "study_id": "imported.binary_seed",
            "kind": "binary",
            "origin_type": "legacy_seed",
            "imported": True,
            "read_only_seed": True,
        },
    )

    config = load_study_config("imported.binary_seed")
    assert config["read_only_seed"] is True
    with pytest.raises(SystemExit):
        _assert_study_writable("imported.binary_seed")


def test_compare_payload_uses_run_ids_and_metrics_paths(asset_root_env: Path) -> None:
    left = run_paths("left_run")
    right = run_paths("right_run")
    update_run_metrics(
        "left_run",
        "backtest.pl_holdout",
        {
            "path": asset_relative(left["reports"] / "left.json"),
            "report": {"summary": {"roi": 1.02, "bets": 100}},
        },
    )
    update_run_metrics(
        "right_run",
        "backtest.pl_holdout",
        {
            "path": asset_relative(right["reports"] / "right.json"),
            "report": {"summary": {"roi": 1.08, "bets": 120}},
        },
    )

    out = asset_root_env / "cache" / "compare.json"
    rc = handle_compare(
        type(
            "Args",
            (),
            {
                "left_run_id": "left_run",
                "right_run_id": "right_run",
                "output": str(out),
            },
        )()
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["left_run_id"] == "left_run"
    assert payload["right_run_id"] == "right_run"
    assert payload["left_metrics"] == "runs/left_run/metrics.json"
    assert payload["right_metrics"] == "runs/right_run/metrics.json"
    assert payload["common_numeric_deltas"]["backtest.pl_holdout.report.summary.roi"][
        "delta"
    ] == pytest.approx(0.06)


def test_backtest_wrapper_uses_effective_holdout_year_for_holdout_inputs(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = run_paths("pl_run")
    update_run_config(
        "pl_run",
        {
            "run_id": "pl_run",
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
            "pl_feature_profile": "stack_default",
            "holdout_year": 2025,
        },
    )
    holdout_input = run["holdout"] / "pl_stack_default_holdout_2025.parquet"
    holdout_input.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_backtest_wide(**kwargs: object) -> int:
        captured.update(kwargs)
        output = Path(str(kwargs["output"]))
        meta = Path(str(kwargs["meta_output"]))
        output.write_text("{}", encoding="utf-8")
        meta.write_text(
            json.dumps(
                {
                    "input": {"path": str(kwargs["input"])},
                    "report": {"path": str(output)},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "keiba_research.evaluation.commands.run_backtest_wide", fake_run_backtest_wide
    )

    rc = handle_backtest(
        type(
            "Args",
            (),
            {
                "run_id": "pl_run",
                "input_kind": "pl_holdout",
                "pl_feature_profile": "",
                "years": "",
                "require_years": "",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["holdout_year"] == 2026
    meta = json.loads(
        (run["reports"] / "backtest_pl_holdout_meta.json").read_text(encoding="utf-8")
    )
    assert (
        meta["input"]["path"]
        == "runs/pl_run/artifacts/holdout/pl_stack_default_holdout_2025.parquet"
    )
    assert meta["report"]["path"] == "runs/pl_run/artifacts/reports/backtest_pl_holdout.json"
    assert not collect_absolute_path_strings(meta)


def test_tune_binary_wrapper_rewrites_study_metadata(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    features_base = feature_root / "features_v3.parquet"
    features_te = feature_root / "features_v3_te.parquet"
    features_base.write_text("placeholder", encoding="utf-8")
    features_te.write_text("placeholder", encoding="utf-8")

    def fake_run_tune_binary(**kwargs: object) -> int:
        Path(str(kwargs["trials_output"])).write_text("trial", encoding="utf-8")
        Path(str(kwargs["best_output"])).write_text(
            json.dumps(
                {
                    "storage": f"sqlite:///{kwargs['storage']}",
                    "best_input": str(kwargs["input_te"]),
                    "train_window_years": int(kwargs["train_window_years"]),  # type: ignore[arg-type]
                }
            ),
            encoding="utf-8",
        )
        Path(str(kwargs["best_params_output"])).write_text(
            json.dumps(
                {
                    "feature_set": "te",
                    "input": str(kwargs["input_te"]),
                    "train_window_years": int(kwargs["train_window_years"]),  # type: ignore[arg-type]
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.tuning.commands.run_tune_binary", fake_run_tune_binary)

    rc = handle_tune_binary(
        type(
            "Args",
            (),
            {
                "study_id": "study_001",
                "task": "win",
                "model": "lgbm",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "holdout_year": 2025,
                "train_window_years": 3,
                "n_trials": 1,
                "timeout": 0,
                "seed": 42,
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    study = study_paths("study_001")
    best = json.loads(study["best"].read_text(encoding="utf-8"))
    selected = json.loads(study["selected_trial"].read_text(encoding="utf-8"))
    assert best["storage"] == "sqlite:///studies/study_001/study.sqlite3"
    assert best["best_input"] == "data/features/baseline_v3/build_001/features_v3_te.parquet"
    assert best["train_window_years"] == 3
    assert selected["input"] == "data/features/baseline_v3/build_001/features_v3_te.parquet"
    assert selected["train_window_years"] == 3
    assert not collect_absolute_path_strings(best)
    assert not collect_absolute_path_strings(selected)


def test_train_pl_public_surface_rejects_meta_default() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "pl",
                "--run-id",
                "run_001",
                "--feature-profile",
                "baseline_v3",
                "--feature-build-id",
                "build_001",
                "--pl-feature-profile",
                "meta_default",
            ]
        )


def test_train_binary_wrapper_redirects_feature_manifest_to_run_bundle(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_v3.parquet").write_text("placeholder", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_binary_training(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["all_years_model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["oof_output"])).write_text("oof", encoding="utf-8")
        Path(str(kwargs["holdout_output"])).write_text("holdout", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text(
            json.dumps(
                {
                    "input_path": str(kwargs["input"]),
                    "output_paths": {
                        "feature_manifest": str(kwargs["feature_manifest_output"]),
                    },
                }
            ),
            encoding="utf-8",
        )
        Path(str(kwargs["feature_manifest_output"])).write_text(
            json.dumps({"input_path": str(kwargs["input"])}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_binary_training", fake_run_binary_training
    )

    rc = handle_binary(
        type(
            "Args",
            (),
            {
                "run_id": "run_001",
                "task": "win",
                "model": "lgbm",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "feature_set": "base",
                "study_id": "",
                "holdout_year": 2025,
                "train_window_years": 0,
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert "feature_manifest_output" in captured
    manifest_path = Path(str(captured["feature_manifest_output"]))
    assert manifest_path == run_paths("run_001")["models"] / "win_lgbm_feature_manifest_v3.json"
    meta_path = run_paths("run_001")["models"] / "win_lgbm_bundle_meta_v3.json"
    meta = json.loads(meta_path.read_text())
    assert meta["input_path"] == "data/features/baseline_v3/build_001/features_v3.parquet"
    assert (
        meta["output_paths"]["feature_manifest"]
        == "runs/run_001/artifacts/models/win_lgbm_feature_manifest_v3.json"
    )
    assert not collect_absolute_path_strings(meta)


def test_db_rebuild_wrapper_rewrites_summary_paths(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_file = asset_root_env / "data" / "jsonl" / "RACE_2023.jsonl"
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file.write_text("{}", encoding="utf-8")

    def fake_run_rebuild(**kwargs: object) -> int:
        summary_path = Path(str(kwargs["summary_output"]))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "input_files": [str(jsonl_file)],
                    "summary_output": str(summary_path),
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.db.commands.run_rebuild", fake_run_rebuild)

    summary_output = asset_root_env / "cache" / "summary.json"
    rc = handle_rebuild(
        type(
            "Args",
            (),
            {
                "database_url": "",
                "input_dir": str(jsonl_file.parent),
                "from_date": "2023-01-01",
                "to_date": "2023-12-31",
                "condition_codes": "10,16,999",
                "summary_output": str(summary_output),
                "o1_date": "20260215",
            },
        )()
    )
    assert rc == 0
    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["input_files"] == ["data/jsonl/RACE_2023.jsonl"]
    assert payload["summary_output"] == "cache/summary.json"
    assert not collect_absolute_path_strings(payload)


def test_db_rebuild_wrapper_rewrites_symlink_resolved_jsonl_paths(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    external_jsonl = tmp_path / "external" / "RACE_2023.jsonl"
    external_jsonl.parent.mkdir(parents=True, exist_ok=True)
    external_jsonl.write_text("{}", encoding="utf-8")
    linked_jsonl = asset_root_env / "data" / "jsonl" / external_jsonl.name
    linked_jsonl.parent.mkdir(parents=True, exist_ok=True)
    linked_jsonl.symlink_to(external_jsonl)

    def fake_run_rebuild(**kwargs: object) -> int:
        summary_path = Path(str(kwargs["summary_output"]))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "input_files": [str(external_jsonl.resolve())],
                    "summary_output": str(summary_path),
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.db.commands.run_rebuild", fake_run_rebuild)

    summary_output = asset_root_env / "cache" / "summary_symlink.json"
    rc = handle_rebuild(
        type(
            "Args",
            (),
            {
                "database_url": "",
                "input_dir": str(linked_jsonl.parent),
                "from_date": "2023-01-01",
                "to_date": "2023-12-31",
                "condition_codes": "10,16,999",
                "summary_output": str(summary_output),
                "o1_date": "20260215",
            },
        )()
    )
    assert rc == 0
    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["input_files"] == ["data/jsonl/RACE_2023.jsonl"]
    assert payload["summary_output"] == "cache/summary_symlink.json"
    assert not collect_absolute_path_strings(payload)


def test_public_surface_defaults_match_three_year_contract() -> None:
    parser = build_parser()
    train_binary = parser.parse_args(
        [
            "train",
            "binary",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
        ]
    )
    train_stack = parser.parse_args(
        [
            "train",
            "stack",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
        ]
    )
    train_pl = parser.parse_args(
        [
            "train",
            "pl",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
        ]
    )
    tune_binary = parser.parse_args(
        [
            "tune",
            "binary",
            "--study-id",
            "study_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
        ]
    )
    tune_stack = parser.parse_args(
        [
            "tune",
            "stack",
            "--study-id",
            "study_001",
            "--source-run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
        ]
    )

    assert train_binary.train_window_years == 3
    assert train_stack.min_train_years == 2
    assert train_stack.max_train_years == 3
    assert train_pl.train_window_years == 3
    assert tune_binary.train_window_years == 3
    assert tune_stack.min_train_years == 2
    assert tune_stack.max_train_years == 3


def test_training_public_surface_accepts_window_overrides() -> None:
    parser = build_parser()
    binary_args = parser.parse_args(
        [
            "train",
            "binary",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
            "--train-window-years",
            "1",
        ]
    )
    pl_args = parser.parse_args(
        [
            "train",
            "pl",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
            "--train-window-years",
            "1",
        ]
    )
    stack_args = parser.parse_args(
        [
            "train",
            "stack",
            "--run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
            "--min-train-years",
            "1",
            "--max-train-years",
            "1",
        ]
    )
    tune_binary_args = parser.parse_args(
        [
            "tune",
            "binary",
            "--study-id",
            "study_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
            "--train-window-years",
            "1",
        ]
    )
    tune_stack_args = parser.parse_args(
        [
            "tune",
            "stack",
            "--study-id",
            "study_001",
            "--source-run-id",
            "run_001",
            "--feature-profile",
            "baseline_v3",
            "--feature-build-id",
            "build_001",
            "--min-train-years",
            "1",
            "--max-train-years",
            "1",
        ]
    )

    assert binary_args.train_window_years == 1
    assert pl_args.train_window_years == 1
    assert stack_args.min_train_years == 1
    assert stack_args.max_train_years == 1
    assert tune_binary_args.train_window_years == 1
    assert tune_stack_args.min_train_years == 1
    assert tune_stack_args.max_train_years == 1


def test_train_pl_wrapper_forwards_train_window_years(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_root = asset_root_env / "data" / "features" / "baseline_v3" / "build_001"
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_v3.parquet").write_text("placeholder", encoding="utf-8")
    source = run_paths("source_run")
    source["oof"].mkdir(parents=True, exist_ok=True)
    source["holdout"].mkdir(parents=True, exist_ok=True)
    for name in (
        "win_lgbm_oof.parquet",
        "win_xgb_oof.parquet",
        "win_cat_oof.parquet",
        "place_lgbm_oof.parquet",
        "place_xgb_oof.parquet",
        "place_cat_oof.parquet",
        "win_stack_oof.parquet",
        "place_stack_oof.parquet",
    ):
        (source["oof"] / name).write_text("placeholder", encoding="utf-8")
    for name in (
        "win_lgbm_holdout_2025.parquet",
        "win_xgb_holdout_2025.parquet",
        "win_cat_holdout_2025.parquet",
        "place_lgbm_holdout_2025.parquet",
        "place_xgb_holdout_2025.parquet",
        "place_cat_holdout_2025.parquet",
        "win_stack_holdout_2025.parquet",
        "place_stack_holdout_2025.parquet",
    ):
        (source["holdout"] / name).write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_pl_training(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["oof_output"])).write_text("oof", encoding="utf-8")
        Path(str(kwargs["wide_oof_output"])).write_text("wide_oof", encoding="utf-8")
        Path(str(kwargs["holdout_output"])).write_text("holdout", encoding="utf-8")
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["all_years_model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["year_coverage_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr("keiba_research.training.commands.run_pl_training", fake_run_pl_training)

    rc = handle_pl(
        type(
            "Args",
            (),
            {
                "run_id": "pl_run",
                "feature_profile": "baseline_v3",
                "feature_build_id": "build_001",
                "source_run_id": "source_run",
                "pl_feature_profile": "stack_default",
                "holdout_year": 2025,
                "train_window_years": 3,
                "train_window_years_explicit": False,
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["train_window_years"] == 3


def test_wide_calibrator_wrapper_prefers_wide_oof_and_sets_apply_input(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = run_paths("pl_source")
    update_run_config(
        "pl_source",
        {
            "run_id": "pl_source",
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
            "pl_feature_profile": "stack_default",
            "holdout_year": 2025,
        },
    )
    fit_input = source["oof"] / "pl_stack_default_wide_oof.parquet"
    fallback_input = source["oof"] / "pl_stack_default_oof.parquet"
    apply_input = source["holdout"] / "pl_stack_default_holdout_2025.parquet"
    fit_input.write_text("wide_oof", encoding="utf-8")
    fallback_input.write_text("horse_oof", encoding="utf-8")
    apply_input.write_text("holdout", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_wide_calibrator(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["pred_output"])).write_text("pred", encoding="utf-8")
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_wide_calibrator", fake_run_wide_calibrator
    )

    rc = handle_wide_calibrator(
        type(
            "Args",
            (),
            {
                "run_id": "wide_run",
                "source_run_id": "pl_source",
                "method": "isotonic",
                "years": "2024",
                "require_years": "2024",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["fit_input"] == str(fit_input)
    assert captured["apply_input"] == str(apply_input)
    config = tomllib.loads(run_paths("wide_run")["config"].read_text(encoding="utf-8"))
    assert config["feature_profile"] == "baseline_v3"
    assert config["feature_build_id"] == "build_001"
    assert config["pl_feature_profile"] == "stack_default"
    assert config["holdout_year"] == 2025


def test_wide_calibrator_wrapper_falls_back_to_horse_oof(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = run_paths("pl_source_fallback")
    update_run_config(
        "pl_source_fallback",
        {
            "run_id": "pl_source_fallback",
            "pl_feature_profile": "stack_default",
            "holdout_year": 2025,
        },
    )
    fit_input = source["oof"] / "pl_stack_default_oof.parquet"
    apply_input = source["holdout"] / "pl_stack_default_holdout_2025.parquet"
    fit_input.write_text("horse_oof", encoding="utf-8")
    apply_input.write_text("holdout", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_wide_calibrator(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["pred_output"])).write_text("pred", encoding="utf-8")
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_wide_calibrator", fake_run_wide_calibrator
    )

    rc = handle_wide_calibrator(
        type(
            "Args",
            (),
            {
                "run_id": "wide_run_fallback",
                "source_run_id": "pl_source_fallback",
                "method": "isotonic",
                "years": "",
                "require_years": "",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["fit_input"] == str(fit_input)
    assert captured["apply_input"] == str(apply_input)


def test_wide_calibrator_wrapper_uses_helper_artifact_paths_for_logreg(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = run_paths("pl_source_logreg")
    update_run_config(
        "pl_source_logreg",
        {
            "run_id": "pl_source_logreg",
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
            "pl_feature_profile": "stack_default",
            "holdout_year": 2025,
        },
    )
    fit_input = source["oof"] / "pl_stack_default_wide_oof.parquet"
    apply_input = source["holdout"] / "pl_stack_default_holdout_2025.parquet"
    fit_input.write_text("wide_oof", encoding="utf-8")
    apply_input.write_text("holdout", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_wide_calibrator(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["pred_output"])).write_text("pred", encoding="utf-8")
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_wide_calibrator", fake_run_wide_calibrator
    )

    rc = handle_wide_calibrator(
        type(
            "Args",
            (),
            {
                "run_id": "wide_run_logreg",
                "source_run_id": "pl_source_logreg",
                "method": "logreg",
                "years": "",
                "require_years": "",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0

    outputs = wide_calibrator_artifact_paths(run_paths("wide_run_logreg"), method="logreg")
    assert captured["model_output"] == str(outputs["model"])
    assert captured["meta_output"] == str(outputs["meta"])
    assert captured["pred_output"] == str(outputs["predictions"])
    assert captured["metrics_output"] == str(outputs["metrics"])

    bundle = json.loads(run_paths("wide_run_logreg")["bundle"].read_text(encoding="utf-8"))
    section = bundle["sections"]["wide_calibrator.logreg"]
    assert section["model"] == asset_relative(outputs["model"])
    assert section["meta"] == asset_relative(outputs["meta"])
    assert section["predictions"] == asset_relative(outputs["predictions"])
    assert section["metrics"] == asset_relative(outputs["metrics"])


def test_wide_calibrator_cross_run_config_supports_backtest(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = run_paths("pl_source_backtest")
    update_run_config(
        "pl_source_backtest",
        {
            "run_id": "pl_source_backtest",
            "feature_profile": "baseline_v3",
            "feature_build_id": "build_001",
            "pl_feature_profile": "stack_default",
            "holdout_year": 2025,
        },
    )
    (source["oof"] / "pl_stack_default_wide_oof.parquet").write_text("wide_oof", encoding="utf-8")
    (source["holdout"] / "pl_stack_default_holdout_2025.parquet").write_text(
        "holdout",
        encoding="utf-8",
    )

    def fake_run_wide_calibrator(**kwargs: object) -> int:
        Path(str(kwargs["model_output"])).write_text("model", encoding="utf-8")
        Path(str(kwargs["pred_output"])).write_text("pred", encoding="utf-8")
        Path(str(kwargs["metrics_output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.run_wide_calibrator", fake_run_wide_calibrator
    )

    rc = handle_wide_calibrator(
        type(
            "Args",
            (),
            {
                "run_id": "wide_run_backtest",
                "source_run_id": "pl_source_backtest",
                "method": "isotonic",
                "years": "2024",
                "require_years": "2024",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0

    captured: dict[str, object] = {}

    def fake_run_backtest_wide(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["output"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["meta_output"])).write_text(
            json.dumps(
                {"input": {"path": str(kwargs["input"])}, "report": {"path": str(kwargs["output"])}}
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "keiba_research.evaluation.commands.run_backtest_wide", fake_run_backtest_wide
    )

    rc = handle_backtest(
        type(
            "Args",
            (),
            {
                "run_id": "wide_run_backtest",
                "input_kind": "wide_calibrated",
                "pl_feature_profile": "",
                "years": "",
                "require_years": "",
                "database_url": "",
                "log_level": "INFO",
            },
        )()
    )
    assert rc == 0
    assert captured["holdout_year"] == 2026
    expected_input = (
        run_paths("wide_run_backtest")["predictions"]
        / "wide_pair_calibration_isotonic_pred.parquet"
    )
    assert captured["input"] == str(expected_input)


def test_three_year_contract_year_coverage_example() -> None:
    years = list(range(2016, 2026))
    holdout_year = 2025

    binary_folds = build_fixed_window_year_folds(
        years,
        window_years=3,
        holdout_year=holdout_year,
    )
    binary_valid_years = [fold.valid_year for fold in binary_folds]
    stack_folds = build_capped_expanding_year_folds(
        binary_valid_years,
        min_window_years=2,
        max_window_years=3,
        holdout_year=holdout_year,
    )
    stack_valid_years = [fold.valid_year for fold in stack_folds]
    pl_folds = build_fixed_window_year_folds(
        stack_valid_years,
        window_years=3,
        holdout_year=holdout_year,
    )
    pl_valid_years = [fold.valid_year for fold in pl_folds]

    assert binary_valid_years == [2019, 2020, 2021, 2022, 2023, 2024]
    assert stack_valid_years == [2021, 2022, 2023, 2024]
    assert pl_valid_years == [2024]
    assert select_recent_window_years(years, train_window_years=3, holdout_year=holdout_year) == [
        2022,
        2023,
        2024,
    ]
    assert select_recent_window_years(
        binary_valid_years,
        train_window_years=3,
        holdout_year=holdout_year,
    ) == [2022, 2023, 2024]
    assert select_recent_window_years(
        stack_valid_years,
        train_window_years=3,
        holdout_year=holdout_year,
    ) == [2022, 2023, 2024]
    assert pl_valid_years == [2024]


def test_wide_calibrator_report_separates_fit_and_holdout_eval(
    asset_root_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root_env.mkdir(parents=True, exist_ok=True)
    fit_input = asset_root_env / "fit.parquet"
    apply_input = asset_root_env / "apply.parquet"
    fit_frame = pd.DataFrame(
        [
            {
                "race_id": 1,
                "horse_no_1": 1,
                "horse_no_2": 2,
                "p_wide_raw": 0.35,
                "valid_year": 2024,
            },
            {
                "race_id": 1,
                "horse_no_1": 1,
                "horse_no_2": 3,
                "p_wide_raw": 0.20,
                "valid_year": 2024,
            },
        ]
    )
    apply_frame = pd.DataFrame(
        [
            {
                "race_id": 2,
                "horse_no_1": 1,
                "horse_no_2": 2,
                "p_wide_raw": 0.40,
                "valid_year": 2025,
            },
            {
                "race_id": 2,
                "horse_no_1": 1,
                "horse_no_2": 3,
                "p_wide_raw": 0.10,
                "valid_year": 2025,
            },
        ]
    )
    fit_frame.to_parquet(fit_input, index=False)
    apply_frame.to_parquet(apply_input, index=False)

    class FakeDatabase:
        def __init__(self, connection_string: str):
            self.connection_string = connection_string

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch_all(self, query: str, params: dict[str, list[int]]):
            rows = []
            for race_id in params["race_ids"]:
                rows.extend(
                    [
                        {"race_id": race_id, "horse_no": 1},
                        {"race_id": race_id, "horse_no": 2},
                        {"race_id": race_id, "horse_no": 3},
                    ]
                )
            return rows

    monkeypatch.setattr("keiba_research.training.wide_calibrator.Database", FakeDatabase)

    model_output = asset_root_env / "model.joblib"
    meta_output = asset_root_env / "meta.json"
    pred_output = asset_root_env / "pred.parquet"
    metrics_output = asset_root_env / "metrics.json"

    rc = run_wide_calibrator(
        fit_input=str(fit_input),
        apply_input=str(apply_input),
        method="isotonic",
        model_output=str(model_output),
        meta_output=str(meta_output),
        pred_output=str(pred_output),
        metrics_output=str(metrics_output),
        database_url="postgresql://example/test",
        years="2024",
        require_years="2024",
    )
    assert rc == 0
    metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
    assert metrics["fit"]["selected_years"] == [2024]
    assert metrics["holdout_eval"]["selected_years"] == [2025]
    assert "raw" in metrics["fit"]
    assert "calibrated" in metrics["fit"]
    assert "raw" in metrics["holdout_eval"]
    assert "calibrated" in metrics["holdout_eval"]
    pred = pd.read_parquet(pred_output)
    assert pred["race_id"].unique().tolist() == [2]


def test_asset_root_defaults_to_fixed_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / ".local" / "v3_assets"
    monkeypatch.delenv("V3_ASSET_ROOT", raising=False)
    monkeypatch.setattr(assets_module, "DEFAULT_ASSET_ROOT", expected)

    assert assets_module.asset_root() == expected
    assert assets_module.asset_root().exists()


def test_eval_report_generates_summary_and_detail_for_full_run(
    asset_root_env: Path,
) -> None:
    _prepare_execution_report_run(
        asset_root_env,
        run_id="report_full",
        source_run_id="upstream_run",
        study_id="study_001",
        annotation={
            "title": "Annotated full report",
            "description": "full execution report",
            "status": "complete",
            "code_revision": "abc1234",
        },
    )

    rc = handle_report(
        type(
            "Args",
            (),
            {
                "run_id": "report_full",
                "summary_output": "",
                "detail_output": "",
                "annotation": "",
            },
        )()
    )
    assert rc == 0

    run = run_paths("report_full")
    summary = json.loads(run["execution_report_summary"].read_text(encoding="utf-8"))
    detail = json.loads(run["execution_report_detail"].read_text(encoding="utf-8"))

    assert summary["report_id"] == "report_full"
    assert summary["run_id"] == "report_full"
    assert summary["title"] == "Annotated full report"
    assert summary["description"] == "full execution report"
    assert summary["status"] == "complete"
    assert summary["conditions"]["from_date"] == "2016-01-01"
    assert summary["quality_summary"]["binary"]["win.lgbm"]["logloss"] == 0.41
    assert summary["quality_summary"]["pl"]["stack_default"]["top3_logloss"] == 0.58
    assert summary["quality_summary"]["wide_calibrator"]["isotonic"]["holdout_eval"]["auc"] == 0.74
    assert summary["backtest_summary"]["pl_holdout"]["roi"] == 1.4
    assert summary["coverage_summary"]["pl"]["stack_default"]["pl_oof_valid_years"] == [2024]
    assert summary["paths"]["summary"] == "runs/report_full/execution_report_summary.json"
    assert detail["lineage"]["primary_source_run_id"] == "upstream_run"
    assert detail["setting_sources"]["binary.win.lgbm"]["feature_set"] == "study_selected"
    assert detail["setting_sources"]["binary.win.lgbm"]["operational_mode"] == "default"
    assert detail["setting_sources"]["binary.win.lgbm"]["params_json"] == "unknown"
    assert detail["diagnostics"]["pl_top3_segments"]["stack_default"]["3yo"]["logloss"] == 0.61
    assert detail["roi_detail"]["backtests"]["pl_holdout"]["purchase_rule"]["kelly_fraction"] == 0.25
    assert detail["code_fingerprint"]["git_commit"] == "abc1234"
    assert detail["code_fingerprint"]["layer_code_hashes"]["pl.stack_default"] == "pl-hash"


def test_eval_report_marks_binary_only_run_partial(asset_root_env: Path) -> None:
    _prepare_execution_report_run(
        asset_root_env,
        run_id="report_partial",
        with_backtest=False,
    )

    # Keep only the binary section to simulate an incomplete run without backtest.
    run = run_paths("report_partial")
    bundle = json.loads(run["bundle"].read_text(encoding="utf-8"))
    metrics = json.loads(run["metrics"].read_text(encoding="utf-8"))
    bundle["sections"] = {"binary.win.lgbm": bundle["sections"]["binary.win.lgbm"]}
    metrics["sections"] = {"binary.win.lgbm": metrics["sections"]["binary.win.lgbm"]}
    run["bundle"].write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    run["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run["resolved_params"].write_text(
        '[binary.win.lgbm]\nfeature_set = "te"\nholdout_year = 2025\ntrain_window_years = 3\n',
        encoding="utf-8",
    )

    rc = handle_report(
        type(
            "Args",
            (),
            {
                "run_id": "report_partial",
                "summary_output": "",
                "detail_output": "",
                "annotation": "",
            },
        )()
    )
    assert rc == 0

    summary = json.loads(run["execution_report_summary"].read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    assert "backtest_summary" in summary
    assert summary["backtest_summary"] == {}


def test_eval_report_preserves_cross_run_lineage(asset_root_env: Path) -> None:
    _prepare_execution_report_run(
        asset_root_env,
        run_id="report_lineage",
        source_run_id="baseline_upstream",
        with_backtest=False,
    )

    rc = handle_report(
        type(
            "Args",
            (),
            {
                "run_id": "report_lineage",
                "summary_output": "",
                "detail_output": "",
                "annotation": "",
            },
        )()
    )
    assert rc == 0

    detail = json.loads(
        run_paths("report_lineage")["execution_report_detail"].read_text(encoding="utf-8")
    )
    assert detail["lineage"]["source_run_ids"] == ["baseline_upstream"]
    assert detail["lineage"]["section_source_runs"]["stack.win"] == "baseline_upstream"
    assert detail["setting_sources"]["stack.win"]["source_run_id"] == "inherited_from_source_run"


def test_eval_report_cli_and_docs_consistency() -> None:
    parser = build_parser()
    args = parser.parse_args(["eval", "report", "--run-id", "run_001"])
    assert args.run_id == "run_001"

    readme = Path("README.md").read_text(encoding="utf-8")
    active_asset_root = Path("docs/history/active-asset-root.md").read_text(encoding="utf-8")
    assert "/home/sato/projects/REPO-v3-research/.local/v3_assets" in readme
    assert "/home/sato/projects/REPO-v3-research/.local/v3_assets" in active_asset_root


def test_eval_report_public_cli_generates_outputs(asset_root_env: Path) -> None:
    _prepare_execution_report_run(
        asset_root_env,
        run_id="report_cli",
        source_run_id="upstream_run",
        with_backtest=True,
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "python", "-m", "keiba_research", "eval", "report", "--run-id", "report_cli"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr

    run = run_paths("report_cli")
    assert run["execution_report_summary"].exists()
    assert run["execution_report_detail"].exists()

    summary = json.loads(run["execution_report_summary"].read_text(encoding="utf-8"))
    assert summary["report_id"] == "report_cli"
    assert summary["paths"]["summary"] == "runs/report_cli/execution_report_summary.json"


def test_python_module_help_smoke() -> None:
    env = dict(os.environ)
    env.setdefault("V3_ASSET_ROOT", "/tmp/keiba_research_help")
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "python", "-m", "keiba_research", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_rebuild_cli_accepts_pinned_o1_date() -> None:
    args = build_parser().parse_args(["db", "rebuild", "--o1-date", "20260215"])
    assert args.o1_date == "20260215"


def test_training_cli_defaults() -> None:
    parser = build_parser()
    binary_args = parser.parse_args(
        ["train", "binary", "--run-id", "r", "--feature-profile", "fp", "--feature-build-id", "fb"]
    )
    stack_args = parser.parse_args(
        ["train", "stack", "--run-id", "r", "--feature-profile", "fp", "--feature-build-id", "fb"]
    )
    pl_args = parser.parse_args(
        ["train", "pl", "--run-id", "r", "--feature-profile", "fp", "--feature-build-id", "fb"]
    )
    assert binary_args.train_window_years == 3
    assert stack_args.min_train_years == 2
    assert stack_args.max_train_years == 3
    assert pl_args.train_window_years == 3


def test_stacker_meta_code_hash_includes_entrypoint() -> None:
    import argparse

    args = argparse.Namespace(
        task="win",
        min_train_years=2,
        max_train_years=3,
        holdout_year=2025,
        artifact_suffix="",
    )
    code_hash_paths = _meta_code_hash_paths()
    payload = _meta_payload(
        args=args,
        feature_cols=["p_win_lgbm", "p_win_xgb", "p_win_cat"],
        base_valid_years=[2022, 2023],
        valid_years=[2024],
        recent_years=[2022, 2023],
        input_paths={"features_v3": "data/features_v3.parquet"},
        output_paths={
            "oof": Path("data/oof/stack.parquet"),
            "holdout": Path("data/holdout/stack.parquet"),
            "metrics": Path("reports/metrics.json"),
            "model": Path("models/main.txt"),
            "all_years_model": Path("models/all_years.txt"),
            "feature_manifest": Path("models/feature_manifest.json"),
        },
        holdout_rows=10,
        holdout_races=2,
        code_hash_paths=code_hash_paths,
    )

    assert code_hash_paths[0] == Path(resolve_path("src/keiba_research/training/stacker.py"))
    assert Path(resolve_path("src/keiba_research/training/stacker.py")) in code_hash_paths
    assert Path(resolve_path("src/keiba_research/training/stacker_common.py")) in code_hash_paths
    assert all(path.is_absolute() for path in code_hash_paths)
    assert payload["code_hash"] == hash_files(code_hash_paths)


def test_stacker_main_preserves_cli_flags_over_params_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params_path = tmp_path / "stack_params.json"
    params_path.write_text(
        json.dumps(
            {
                "min_train_years": 2,
                "max_train_years": 8,
                "lgbm_params": {},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, int] = {}

    class StopAfterValidate(Exception):
        pass

    def fake_validate(args: object) -> None:
        namespace = args  # keep mypy/ruff happy without changing runtime behavior
        captured["min_train_years"] = int(namespace.min_train_years)
        captured["max_train_years"] = int(namespace.max_train_years)
        raise StopAfterValidate

    monkeypatch.setattr("keiba_research.training.stacker._validate_args", fake_validate)

    with pytest.raises(StopAfterValidate):
        run_stacker_training(
            params_json=str(params_path),
            min_train_years=9,
        )

    assert captured["min_train_years"] == 9
    assert captured["max_train_years"] == 8

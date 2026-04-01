from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pandas as pd
import pytest

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
from keiba_research.db.commands import handle_rebuild
from keiba_research.evaluation.commands import handle_backtest, handle_compare
from keiba_research.training.commands import (
    handle_binary,
    handle_pl,
    handle_wide_calibrator,
)
from keiba_research.tuning.commands import (
    _assert_study_writable,
)
from keiba_research.tuning.commands import (
    handle_binary as handle_tune_binary,
)
from scripts_v3.cv_policy_v3 import (
    build_capped_expanding_year_folds,
    build_fixed_window_year_folds,
    select_recent_window_years,
)
from scripts_v3.rebuild_v3_db import parse_args as parse_rebuild_args
from scripts_v3.train_binary_model_v3 import parse_args as parse_binary_args
from scripts_v3.train_pl_v3 import parse_args as parse_pl_args
from scripts_v3.train_stacker_v3 import (
    _meta_code_hash_paths,
)
from scripts_v3.train_stacker_v3 import (
    main as train_stack_main,
)
from scripts_v3.train_stacker_v3 import (
    parse_args as parse_stack_args,
)
from scripts_v3.train_stacker_v3_common import _meta_payload
from scripts_v3.train_wide_pair_calibrator_v3 import main as train_wide_calibrator_script_main
from scripts_v3.v3_common import hash_files, resolve_path


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

    captured: dict[str, list[str]] = {}

    def fake_backtest_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        output = Path(argv[argv.index("--output") + 1])
        meta = Path(argv[argv.index("--meta-output") + 1])
        output.write_text("{}", encoding="utf-8")
        meta.write_text(
            json.dumps(
                {
                    "input": {"path": argv[argv.index("--input") + 1]},
                    "report": {"path": str(output)},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.evaluation.commands.backtest_wide_main", fake_backtest_main)

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
    argv = captured["argv"]
    assert argv[argv.index("--holdout-year") + 1] == "2026"
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

    def fake_tune_binary_main(argv: list[str]) -> int:
        args = _argv_to_dict(argv)
        Path(args["--trials-output"]).write_text("trial", encoding="utf-8")
        Path(args["--best-output"]).write_text(
            json.dumps(
                {
                    "storage": f"sqlite:///{args['--storage']}",
                    "best_input": args["--input-te"],
                    "train_window_years": int(args["--train-window-years"]),
                }
            ),
            encoding="utf-8",
        )
        Path(args["--best-params-output"]).write_text(
            json.dumps(
                {
                    "feature_set": "te",
                    "input": args["--input-te"],
                    "train_window_years": int(args["--train-window-years"]),
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.tuning.commands.tune_binary_main", fake_tune_binary_main)

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
    captured: dict[str, list[str]] = {}

    def fake_train_binary_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        args: dict[str, str] = {}
        index = 0
        while index < len(argv):
            token = argv[index]
            if not token.startswith("--"):
                index += 1
                continue
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                args[token] = argv[index + 1]
                index += 2
            else:
                index += 1
        Path(args["--metrics-output"]).write_text("{}", encoding="utf-8")
        Path(args["--model-output"]).write_text("model", encoding="utf-8")
        Path(args["--all-years-model-output"]).write_text("model", encoding="utf-8")
        Path(args["--oof-output"]).write_text("oof", encoding="utf-8")
        Path(args["--holdout-output"]).write_text("holdout", encoding="utf-8")
        Path(args["--meta-output"]).write_text(
            json.dumps(
                {
                    "input_path": args["--input"],
                    "output_paths": {
                        "feature_manifest": args["--feature-manifest-output"],
                    },
                }
            ),
            encoding="utf-8",
        )
        Path(args["--feature-manifest-output"]).write_text(
            json.dumps({"input_path": args["--input"]}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.train_binary_main", fake_train_binary_main
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
    argv = captured["argv"]
    assert "--feature-manifest-output" in argv
    manifest_path = Path(argv[argv.index("--feature-manifest-output") + 1])
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

    def fake_rebuild_main(argv: list[str]) -> int:
        summary_path = Path(argv[argv.index("--summary-output") + 1])
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

    monkeypatch.setattr("keiba_research.db.commands.rebuild_v3_main", fake_rebuild_main)

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

    def fake_rebuild_main(argv: list[str]) -> int:
        summary_path = Path(argv[argv.index("--summary-output") + 1])
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

    monkeypatch.setattr("keiba_research.db.commands.rebuild_v3_main", fake_rebuild_main)

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

    captured: dict[str, list[str]] = {}

    def fake_train_pl_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        args = _argv_to_dict(argv)
        Path(args["--oof-output"]).write_text("oof", encoding="utf-8")
        Path(args["--wide-oof-output"]).write_text("wide_oof", encoding="utf-8")
        Path(args["--holdout-output"]).write_text("holdout", encoding="utf-8")
        Path(args["--metrics-output"]).write_text("{}", encoding="utf-8")
        Path(args["--model-output"]).write_text("model", encoding="utf-8")
        Path(args["--all-years-model-output"]).write_text("model", encoding="utf-8")
        Path(args["--year-coverage-output"]).write_text("{}", encoding="utf-8")
        Path(args["--meta-output"]).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr("keiba_research.training.commands.train_pl_main", fake_train_pl_main)

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
    argv = captured["argv"]
    assert argv[argv.index("--train-window-years") + 1] == "3"


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

    captured: dict[str, list[str]] = {}

    def fake_train_wide_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        args = _argv_to_dict(argv)
        Path(args["--model-output"]).write_text("model", encoding="utf-8")
        Path(args["--pred-output"]).write_text("pred", encoding="utf-8")
        Path(args["--metrics-output"]).write_text("{}", encoding="utf-8")
        Path(args["--meta-output"]).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.train_wide_calibrator_main", fake_train_wide_main
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
    argv = captured["argv"]
    assert argv[argv.index("--fit-input") + 1] == str(fit_input)
    assert argv[argv.index("--apply-input") + 1] == str(apply_input)
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

    captured: dict[str, list[str]] = {}

    def fake_train_wide_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        args = _argv_to_dict(argv)
        Path(args["--model-output"]).write_text("model", encoding="utf-8")
        Path(args["--pred-output"]).write_text("pred", encoding="utf-8")
        Path(args["--metrics-output"]).write_text("{}", encoding="utf-8")
        Path(args["--meta-output"]).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.train_wide_calibrator_main", fake_train_wide_main
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
    argv = captured["argv"]
    assert argv[argv.index("--fit-input") + 1] == str(fit_input)
    assert argv[argv.index("--apply-input") + 1] == str(apply_input)


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

    def fake_train_wide_main(argv: list[str]) -> int:
        args = _argv_to_dict(argv)
        Path(args["--model-output"]).write_text("model", encoding="utf-8")
        Path(args["--pred-output"]).write_text("pred", encoding="utf-8")
        Path(args["--metrics-output"]).write_text("{}", encoding="utf-8")
        Path(args["--meta-output"]).write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "keiba_research.training.commands.train_wide_calibrator_main", fake_train_wide_main
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

    captured: dict[str, list[str]] = {}

    def fake_backtest_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        args = _argv_to_dict(argv)
        Path(args["--output"]).write_text("{}", encoding="utf-8")
        Path(args["--meta-output"]).write_text(
            json.dumps({"input": {"path": args["--input"]}, "report": {"path": args["--output"]}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("keiba_research.evaluation.commands.backtest_wide_main", fake_backtest_main)

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
    argv = captured["argv"]
    assert argv[argv.index("--holdout-year") + 1] == "2026"
    expected_input = (
        run_paths("wide_run_backtest")["predictions"]
        / "wide_pair_calibration_isotonic_pred.parquet"
    )
    assert (
        argv[argv.index("--input") + 1]
        == str(expected_input)
    )


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

    monkeypatch.setattr("scripts_v3.train_wide_pair_calibrator_v3.Database", FakeDatabase)

    model_output = asset_root_env / "model.joblib"
    meta_output = asset_root_env / "meta.json"
    pred_output = asset_root_env / "pred.parquet"
    metrics_output = asset_root_env / "metrics.json"

    rc = train_wide_calibrator_script_main(
        [
            "--fit-input",
            str(fit_input),
            "--apply-input",
            str(apply_input),
            "--method",
            "isotonic",
            "--model-output",
            str(model_output),
            "--meta-output",
            str(meta_output),
            "--pred-output",
            str(pred_output),
            "--metrics-output",
            str(metrics_output),
            "--database-url",
            "postgresql://example/test",
            "--years",
            "2024",
            "--require-years",
            "2024",
        ]
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
    args = parse_rebuild_args(["--o1-date", "20260215"])
    assert args.o1_date == "20260215"


def test_training_scripts_accept_disable_default_params_json_flag() -> None:
    binary_args = parse_binary_args(["--disable-default-params-json"])
    stack_args = parse_stack_args(["--disable-default-params-json"])
    pl_args = parse_pl_args([])
    assert binary_args.disable_default_params_json is True
    assert stack_args.disable_default_params_json is True
    assert binary_args.train_window_years == 3
    assert stack_args.min_train_years == 2
    assert stack_args.max_train_years == 3
    assert pl_args.train_window_years == 3
    binary_manifest_args = parse_binary_args(["--feature-manifest-output", "models/test.json"])
    assert binary_manifest_args.feature_manifest_output == "models/test.json"


def test_stacker_meta_code_hash_includes_entrypoint() -> None:
    args = parse_stack_args([])
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

    assert code_hash_paths[0] == Path(resolve_path("scripts_v3/train_stacker_v3.py"))
    assert Path(resolve_path("scripts_v3/train_stacker_v3.py")) in code_hash_paths
    assert Path(resolve_path("scripts_v3/train_stacker_v3_common.py")) in code_hash_paths
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

    monkeypatch.setattr("scripts_v3.train_stacker_v3._validate_args", fake_validate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_stacker_v3.py",
            "--params-json",
            str(params_path),
            "--min-train-years",
            "9",
        ],
    )

    with pytest.raises(StopAfterValidate):
        train_stack_main()

    assert captured["min_train_years"] == 9
    assert captured["max_train_years"] == 8

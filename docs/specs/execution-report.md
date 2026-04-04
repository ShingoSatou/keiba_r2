# Execution Report

## Purpose
- 1 回の仮説検証結果を user-facing な「実行レポート」として固定する
- compare 用の raw numeric flatten とは別に、一覧 / 可視化 / 比較しやすい curated surface を持つ
- source of truth は引き続き `run` とし、execution report は派生 read model に留める

## Unit
- v1 は `1 report = 1 run`
- `report_id == run_id`
- UI 用語として `execution report` を使ってよいが、内部では `run_id` を必ず保持する

## Current workspace asset root
- current workspace の `V3_ASSET_ROOT` は `/home/sato/projects/REPO-v3-research/.local/v3_assets`
- `V3_ASSET_ROOT` 未設定時も同じ path を default として使う
- Git 管理しない

## Source of truth
- `runs/<run_id>/config.toml`
- `runs/<run_id>/bundle.json`
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/resolved_params.toml`
- `runs/<run_id>/artifacts/reports/*`
- `runs/<run_id>/artifacts/oof/*`
- `runs/<run_id>/artifacts/holdout/*`
- `runs/<run_id>/artifacts/predictions/*`
- `data/features/<feature_profile>/<feature_build_id>/config.toml`
- optional `runs/<run_id>/execution_report_annotation.toml`

## Outputs
- `runs/<run_id>/execution_report_summary.json`
- `runs/<run_id>/execution_report_detail.json`

## Summary schema
必須 top-level field:
- `schema_version`
- `report_id`
- `run_id`
- `created_at`
- `status`
- `title`
- `description`
- `code_revision`
- `conditions`
- `pipeline`
- `quality_summary`
- `coverage_summary`
- `backtest_summary`
- `paths`

`conditions`:
- `feature_profile`
- `feature_build_id`
- `holdout_year`
- `pl_feature_profile`
- `source_run_ids`
- `primary_source_run_id`
- `source_report_ids`
- `primary_source_report_id`
- `target_segment`
- `from_date`
- `to_date`
- `history_days`

`pipeline`:
- `binary`
- `stack`
- `pl`
- `wide_calibrator`
- `backtest`

`quality_summary`:
- `binary.<task>.<model>`: `logloss`, `brier`, `auc`, `ece`, optional `benter_r2_valid`
- `stack.<task>`: `logloss`, `brier`, `auc`, `ece`
- `pl.<profile>`: `pl_nll_valid`, `top3_logloss`, `top3_brier`, `top3_auc`, `top3_ece`
- `wide_calibrator.<method>`: `fit`, `holdout_eval`
- `backtest.<input_kind>`: `period_from`, `period_to`, `n_races`, `n_bets`, `n_hits`, `hit_rate`, `total_bet`, `total_return`, `roi`, `max_drawdown`, optional `logloss`, `auc`

`coverage_summary`:
- `binary` / `stack`: `rows`, `races`, `years`, `oof_valid_years`, `holdout_years`
- `pl`: `rows`, `races`, `years`, `pl_oof_valid_years`, `pl_holdout_train_years`, `base_oof_years`, `stacker_oof_years`
- `wide_calibrator`: `fit_rows`, `fit_races`, `fit_years`, `holdout_eval_rows`, `holdout_eval_races`, `holdout_eval_years`
- `backtest`: `rows`, `pair_rows_for_backtest`, `selected_races`, `selected_years`

## Detail schema
detail は summary を含み、次を追加する。
- `source_of_truth`
- `lineage`
- `layer_settings`
- `setting_sources`
- `quality_detail`
- `coverage_detail`
- `roi_detail`
- `diagnostics`
- `artifacts`
- `code_fingerprint`
- `issues`

`setting_sources` の値:
- `study_selected`
- `default`
- `manual_override`
- `inherited_from_source_run`
- `unknown`

## Annotation file
optional `execution_report_annotation.toml`:

```toml
title = "Annotated full report"
description = "full execution report"
status = "complete"

[code_revision]
git_commit = "abc1234"
```

annotation が無いとき:
- `title` は run 条件から自動補完する
- `description` は active section と条件から自動補完する
- `status` は `complete | partial | invalid` を自動判定する

## Source mapping
| report field | source |
|---|---|
| `conditions.feature_profile` | run `config.toml` |
| `conditions.from_date` / `to_date` / `history_days` | feature build `config.toml` |
| `pipeline.*` | `bundle.json.sections` |
| `layer_settings.*.resolved_params` | `resolved_params.toml` |
| `layer_settings.*.report_config` | layer metrics report `config` |
| `quality_summary.*` | layer report `summary` / calibration report / backtest report |
| `coverage_summary.*` | layer report `data_summary`, `cv_policy`, PL `year_coverage.json`, backtest meta |
| `roi_detail.*.purchase_rule` | `backtest_<input_kind>_meta.json` |
| `code_fingerprint.layer_code_hashes` | layer `*_bundle_meta*.json` |

## Required vs optional
初版で必須:
- summary/detail generation
- curated quality / coverage / backtest blocks
- lineage
- annotation override
- setting provenance with `unknown` fallback

初版で任意:
- `code_revision.git_commit`
- `target_segment`
- full raw artifact duplication

後回し:
- strict git SHA capture for old runs
- top-level execution report registry
- new diagnostic slice 計算
- compare を execution report summary ベースへ置き換えること

## Samples
- [execution_report_summary.sample.json](examples/execution_report_summary.sample.json)
- [execution_report_detail.sample.json](examples/execution_report_detail.sample.json)

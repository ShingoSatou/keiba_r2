# CLI Reference

このファイルは、repo-level CLI の現在の public surface をまとめたものです。  
ここに書くのは `python -m keiba_research ...` の契約です。内部実装の全引数一覧ではありません。

## Entry point
```bash
uv run python -m keiba_research <group> <command> [options]
```

current workspace default:
- `V3_ASSET_ROOT` 未設定時は `/home/sato/projects/REPO-v3-research/.local/v3_assets` を使います
- 明示 override したいときだけ `V3_ASSET_ROOT` を設定します

group:
- `db`
- `features`
- `train`
- `tune`
- `eval`
- `import`

## `db`
### `db migrate`
用途:
- `keiba_v3` migration を適用する

主な引数:
- optional `--database-url`

出力:
- DB schema 更新
- file 出力なし

### `db rebuild`
用途:
- canonical JSONL から `keiba_v3` を再構築する

必須引数:
- `--o1-date YYYYMMDD`

主な引数:
- optional `--database-url`
- optional `--input-dir`
- optional `--from-date`
- optional `--to-date`
- optional `--condition-codes`
- optional `--summary-output`

default:
- `from-date=2016-01-01`
- `condition-codes=10,16,999`

出力:
- DB table 再構築
- rebuild summary JSON

注意:
- repo-level wrapper は rebuild summary JSON の asset path を `V3_ASSET_ROOT` 相対へ正規化します

## `features`
### `features build-base`
用途:
- base features と TE source 用 base features を作る

必須引数:
- `--feature-profile`
- `--feature-build-id`
- `--from-date`
- `--to-date`

主な引数:
- optional `--history-days`
- optional `--database-url`
- optional `--log-level`

出力:
- `features_base.parquet`
- `features_base_meta.json`
- `features_base_te.parquet`
- `features_base_te_meta.json`
- `config.toml`

### `features build`
用途:
- `features_base.parquet` から `features_v3.parquet` を作る

必須引数:
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--database-url`
- optional `--log-level`

出力:
- `features_v3.parquet`
- `features_v3_meta.json`

### `features build-te`
用途:
- `features_v3.parquet` と `features_base_te.parquet` から `features_v3_te.parquet` を作る

必須引数:
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--log-level`

出力:
- `features_v3_te.parquet`
- `features_v3_te_meta.json`

## `tune`
### `tune binary`
用途:
- binary model 用 Optuna study を作成または resume する

必須引数:
- `--study-id`
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--task {win,place}`
- optional `--model {lgbm,xgb,cat}`
- optional `--holdout-year`
- optional `--train-window-years`
- optional `--n-trials`
- optional `--timeout`
- optional `--seed`

出力:
- `studies/<study_id>/config.toml`
- `studies/<study_id>/study.sqlite3`
- `studies/<study_id>/best.json`
- `studies/<study_id>/selected_trial.json`
- `studies/<study_id>/trials.parquet`

注意:
- imported study は resume 不可
- selected trial の `feature_set` は train 時に手で合わせる
- public default は `fixed_sliding`, `train_window_years=3`
- repo-level wrapper が生成した `best.json` / `selected_trial.json` の path field は
  `V3_ASSET_ROOT` 相対です
- imported study は legacy provenance を残すため、完全 path-clean は要求しません

### `tune stack`
用途:
- stacker 用 Optuna study を作成または resume する

必須引数:
- `--study-id`
- `--source-run-id`
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--task {win,place}`
- optional `--holdout-year`
- optional `--min-train-years`
- optional `--max-train-years`
- optional `--n-trials`
- optional `--timeout`
- optional `--seed`

出力:
- `studies/<study_id>/...`

## `train`
### `train binary`
用途:
- 1 task x 1 model の binary model を run bundle に保存する

必須引数:
- `--run-id`
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--task {win,place}`
- optional `--model {lgbm,xgb,cat}`
- optional `--feature-set {base,te}`
- optional `--study-id`
- optional `--holdout-year`
- optional `--train-window-years`
- optional `--database-url`

出力:
- `runs/<run_id>/config.toml`
- `runs/<run_id>/bundle.json`
- `runs/<run_id>/metrics.json`
- `artifacts/oof/<task>_<model>_oof.parquet`
- `artifacts/holdout/<task>_<model>_holdout_<holdout_year>.parquet`
- `artifacts/reports/<task>_<model>_cv_metrics.json`
- `artifacts/models/<task>_<model>_v3.<ext>`
- `artifacts/models/<task>_<model>_all_years_v3.<ext>`
- `artifacts/models/<task>_<model>_bundle_meta_v3.json`
- `artifacts/models/<task>_<model>_feature_manifest_v3.json`

注意:
- `--database-url` は現在 interface 上だけ存在し、binary learner 自体は DB を読みません
- public default は `fixed_sliding`, `train_window_years=3`
- `--study-id` を使い、window override を明示しない場合は `selected_trial.json` 側の window が使われます
- `selected_trial.json` の `feature_set` は train 時に手で一致させます
- `*_bundle_meta_v3.json` と `*_feature_manifest_v3.json` の path field は
  `V3_ASSET_ROOT` 相対へ正規化されます

### `train stack`
用途:
- stacker を run bundle に保存する

必須引数:
- `--run-id`
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--task {win,place}`
- optional `--source-run-id`
- optional `--study-id`
- optional `--holdout-year`
- optional `--min-train-years`
- optional `--max-train-years`

出力:
- `stack.<task>` section
- stack OOF / holdout / metrics / model / feature manifest

注意:
- public default は `capped_expanding`, `min_train_years=2`, `max_train_years=3`
- `--study-id` を使い、window override を明示しない場合は `selected_trial.json` 側の window が使われます

### `train pl`
用途:
- PL layer を run bundle に保存する

必須引数:
- `--run-id`
- `--feature-profile`
- `--feature-build-id`

主な引数:
- optional `--source-run-id`
- optional `--pl-feature-profile`
- optional `--holdout-year`
- optional `--train-window-years`

出力:
- `pl.<pl_feature_profile>` section
- PL OOF / wide OOF / holdout / metrics / model / year coverage

注意:
- current public path は stack 系 profile 前提
- CLI で選べるのは `stack_default` と `stack_default_age_v1` のみ
- public default は `fixed_sliding`, `train_window_years=3`

### `train wide-calibrator`
用途:
- pair-level wide calibrator を run bundle に保存する

必須引数:
- `--run-id`

主な引数:
- optional `--source-run-id`
- optional `--method {isotonic,logreg}`
- optional `--years`
- optional `--require-years`
- optional `--database-url`

出力:
- `wide_calibrator.<method>` section
- calibrator model / predictions / metrics

注意:
- fit input は `pl_<profile>_wide_oof.parquet` を第一候補、`pl_<profile>_oof.parquet` を fallback にします
- apply/eval input は `pl_<profile>_holdout_<holdout_year>.parquet` です
- `--years` / `--require-years` は fit dataset selection として扱います
- metrics report は `fit` と `holdout_eval` を分けます

## `eval`
### `eval backtest`
用途:
- run 内の予測を使って wide backtest を行う

必須引数:
- `--run-id`

主な引数:
- optional `--input-kind {pl_holdout,pl_oof,wide_calibrated}`
- optional `--pl-feature-profile`
- optional `--years`
- optional `--require-years`
- optional `--database-url`

出力:
- `backtest.<input_kind>` section
- `artifacts/reports/backtest_<input_kind>.json`
- `artifacts/reports/backtest_<input_kind>_meta.json`

注意:
- `pl_holdout` と `wide_calibrated` は holdout-only artifact なので、wrapper は内部的に `holdout_year + 1` を downstream filter に渡します
- `wide_calibrated` は現在 isotonic prediction file 固定です
- `wide_calibrated` が読む prediction は `train wide-calibrator` の holdout apply output です
- `backtest_<input_kind>_meta.json` の path field は `V3_ASSET_ROOT` 相対へ正規化されます

### `eval compare`
用途:
- 2 run の `metrics.json` を比較する

必須引数:
- `--left-run-id`
- `--right-run-id`

主な引数:
- optional `--output`

出力:
- default: `cache/compare/<left>__vs__<right>.json`

### `eval report`
用途:
- 既存 run bundle から execution report summary/detail を生成する

必須引数:
- `--run-id`

主な引数:
- optional `--summary-output`
- optional `--detail-output`
- optional `--annotation`

出力:
- default: `runs/<run_id>/execution_report_summary.json`
- default: `runs/<run_id>/execution_report_detail.json`

注意:
- v1 では `1 report = 1 run`
- source of truth は引き続き run bundle です
- title / description / status / code revision override は `execution_report_annotation.toml` から読めます
- compare surface は変わりません

## `import`
### `import legacy-tuning`
用途:
- legacy tuning asset を read-only study seed として取り込む

必須引数:
- `--study-id`
- `--kind {binary,stack}`
- `--source-best-params`

主な引数:
- optional `--source-storage`
- optional `--source-best`
- optional `--task`
- optional `--model`

出力:
- `studies/imported/<study_id>/config.toml`
- optional copied `study.sqlite3`
- optional copied `best.json`
- copied `selected_trial.json`

注意:
- CLI では `imported.<study_id>` として参照する
- `--source-*` は user-managed external file を渡す
- imported study は `read_only_seed=true` で、repo-level `tune` から resume できない
- path rewrite は best-effort
- imported study は legacy provenance を残す seed なので、fully self-contained / path-clean は要求しない

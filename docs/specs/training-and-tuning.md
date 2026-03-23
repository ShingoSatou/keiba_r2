# Training And Tuning

このファイルは train / tune surface の overview です。  
binary / stack / calibrator の詳細契約は `binary-stacker-and-calibration.md`、PL / backtest / 購入ルールは `pl-inference-and-wide-backtest.md` を優先します。
図で追いたい場合は `architecture-and-cv.md` を見てください。

## Training surface
この repo の train surface は次です。
- `train binary`
  - `task={win,place}` x `model={lgbm,xgb,cat}` を 1 回ずつ学習
- `train stack`
  - `task={win,place}` ごとに stacker を 1 本学習
- `train pl`
  - upstream binary / stack の OOF と holdout を入力に PL を学習
- `train wide-calibrator`
  - PL OOF / wide OOF で fit し、PL holdout に apply/eval する

各 train コマンドは `--run-id` を必須にし、成果物を `runs/<run_id>/` に保存します。

repo-level CLI で管理するのは「どこに保存するか」と「どの feature build / source run を使うか」です。  
実際の learner 実装は `scripts_v3/` に残っています。

## What a run means
`run` は「比較のために固定した 1 実験結果」です。

最低限含むもの:
- 固定条件を記録した `config.toml`
- train で生成した artifact を指す `bundle.json`
- compare に使う数値指標を集約した `metrics.json`
- 少なくとも 1 つの evaluation 結果

feature build だけ終わった状態や、study だけある状態は `run` とは呼びません。

## Run assembly rules
同じ `run_id` に積み上げてよいもの:
- 同じ `feature_profile`
- 同じ `feature_build_id`
- 同じ `holdout_year`
- 整合する `source_run_id`
- 同じ run を完成させるための binary / stack / PL / wide / backtest

別 `run_id` に分けるべきもの:
- feature build が違う
- holdout year が違う
- PL profile を比較したい
- upstream prediction source を切り替えたい
- baseline と candidate を並行保持したい

official compare のために run を「完成」とみなす最低条件:
1. train 系 section が必要数ある
2. `metrics.json` に比較したい数値が入っている
3. backtest 比較をしたいなら `backtest.*` section がある

## Binary training
`train binary` は 1 回の実行で 1 task x 1 model を学習します。

入力:
- `feature_profile`
- `feature_build_id`
- `feature_set`
  - `base` なら `features_v3.parquet`
  - `te` なら `features_v3_te.parquet`
- optional `study_id`

出力:
- `artifacts/oof/<task>_<model>_oof.parquet`
- `artifacts/holdout/<task>_<model>_holdout_<holdout_year>.parquet`
- `artifacts/reports/<task>_<model>_cv_metrics.json`
- `artifacts/models/<task>_<model>_v3.<ext>`
- `artifacts/models/<task>_<model>_all_years_v3.<ext>`
- `artifacts/models/<task>_<model>_bundle_meta_v3.json`
- `artifacts/models/<task>_<model>_feature_manifest_v3.json`

重要な点:
- repo-level wrapper は `--disable-default-params-json` を常に渡します。
- binary の feature manifest も run 配下へ保存します。repo 直下 `models/` への副作用出力は
  public contract に含めません。
- つまり旧 `data/oof/binary_v3_*_best_params.json` の暗黙 fallback は使いません。
- tuning 済みパラメータを使うなら `--study-id` を明示します。
- `selected_trial.json` に `feature_set` が入っていても、入力 parquet の選択は wrapper の `--feature-set` に従います。
- そのため、study で選ばれた `feature_set` と train 時の `--feature-set` は手動で一致させる必要があります。
- wrapper の `--database-url` は現在 interface 上だけ存在し、binary learner 自体は DB を読みません。
- wrapper が生成する `*_bundle_meta_v3.json` と `*_feature_manifest_v3.json` の path field は
  `V3_ASSET_ROOT` 相対へ正規化されます。

## Binary / stack / PL dependency model
- `train binary`
  - feature build を直接入力に使います
- `train stack`
  - feature build に加えて、指定 `source_run_id` の binary OOF / holdout を使います
- `train pl`
  - feature build と `source_run_id` の binary / stack OOF / holdout を使います
- `train wide-calibrator`
  - `source_run_id` の PL wide OOF を第一候補、PL OOF を fallback として fit に使います
  - apply/eval は `source_run_id` の PL holdout を使います

同じ repo / worktree でも、`run_id` を分ければ model 変更と PL 変更を切り分けて比較できます。

## Stack training
`train stack` は current implementation では LightGBM stacker です。

入力:
- `feature_profile`
- `feature_build_id`
- `source_run_id`
  - 未指定なら自分自身の `run_id`
  - ここから binary OOF / holdout を読む
- optional `study_id`

出力:
- `artifacts/oof/<task>_stack_oof.parquet`
- `artifacts/holdout/<task>_stack_holdout_<holdout_year>.parquet`
- `artifacts/reports/<task>_stack_cv_metrics.json`
- `artifacts/models/<task>_stack_v3.txt`
- `artifacts/models/<task>_stack_all_years_v3.txt`
- `artifacts/models/<task>_stack_bundle_meta_v3.json`
- `artifacts/models/<task>_stack_feature_manifest_v3.json`

重要な点:
- ここでも implicit params-json fallback は wrapper が止めています。
- `source_run_id` を分ければ、binary 改善だけを別 run に切り出して stack を検証できます。

## PL training
`train pl` は PL ranking layer を学習します。

repo-level wrapper が clean に扱う profile:
- `stack_default`
- `stack_default_age_v1`

入力:
- `feature_profile`
- `feature_build_id`
- `source_run_id`
  - binary / stack の OOF / holdout を読む
- `pl_feature_profile`

出力:
- `artifacts/oof/pl_<profile>_oof.parquet`
- `artifacts/oof/pl_<profile>_wide_oof.parquet`
- `artifacts/holdout/pl_<profile>_holdout_<holdout_year>.parquet`
- `artifacts/reports/pl_<profile>_cv_metrics.json`
- `artifacts/reports/pl_<profile>_year_coverage.json`
- `artifacts/models/pl_<profile>_recent_window.joblib`
- `artifacts/models/pl_<profile>_all_years.joblib`
- `artifacts/models/pl_<profile>_bundle_meta.json`

注意:
- underlying script には `meta_default` profile もあります。
- ただし repo-level wrapper は meta OOF / meta holdout を管理していません。
- そのため、この repo の public surface としては stack 系 profile を前提に扱います。
- CLI で選べる profile も `stack_default` と `stack_default_age_v1` のみです。
- public default は `fixed_sliding`, `train_window_years=3` です
- repo-level wrapper は `--train-window-years` を expose し、binary / stack と同様に override できます

## Wide calibrator training
`train wide-calibrator` は PL wide OOF / OOF で pair-level calibrator を fit し、PL holdout に apply/eval します。

入力:
- `source_run_id`
- `method`
  - `isotonic` または `logreg`
- optional `years`, `require_years`
  - fit dataset selection として扱います

出力:
- `artifacts/models/wide_pair_calibrator_<method>.joblib`
- `artifacts/models/wide_pair_calibrator_<method>_bundle_meta.json`
- `artifacts/predictions/wide_pair_calibration_<method>_pred.parquet`
  - holdout apply の out-of-sample calibrated prediction
- `artifacts/reports/wide_pair_calibration_<method>_metrics.json`
  - `fit` と `holdout_eval` を分けた nested report

## Tuning surface
- Optuna は `study` として扱います。
- 新規 study は resume 可です。
- imported legacy study は read-only seed です。
- imported legacy study は seed として読むだけで、portable / path-clean な canonical bundle にはしません。
- 採択した trial は `selected_trial.json` に固定します。

### `tune binary`
- feature build を入力に使います
- 生成物:
  - `study.sqlite3`
  - `best.json`
  - `selected_trial.json`
  - `trials.parquet`
- `best.json` と `selected_trial.json` の path field は repo-level wrapper が
  `V3_ASSET_ROOT` 相対へ正規化します。
- default contract:
  - fixed-sliding yearly CV
  - `train_window_years=3`
  - `operational_mode=t10_only`
  - `include_entity_id_features=false`
- 変動する主な要素:
  - `train_window_years`
  - `feature_set={base,te}`
  - model hyperparameters
- current learner:
  - `lgbm`, `xgb`, `cat`

### `tune stack`
- feature build と `source_run_id` の binary OOF を入力に使います
- 生成物は binary study と同じです
- current learner:
  - LightGBM
- default tunable window:
  - `min_train_years=2`
  - `max_train_years=3`

## Same-run vs cross-run composition
- same run で自然な合成
  - binary -> stack -> pl -> wide -> backtest
- cross-run で自然な合成
  - `train stack --source-run-id <binary_run>`
  - `train pl --source-run-id <upstream_run>`
  - `train wide-calibrator --source-run-id <pl_run>`

cross-run を使う理由:
- model 側だけの変更と PL 側だけの変更を分離したい
- imported seed から作った upstream run を再利用したい
- upstream を固定したまま PL profile を比較したい

## Legacy import
- 旧 v3 の sqlite, `best.json`, `best_params.json` は seed として import します。
- import 後の study は `imported.<study_id>` として参照します。
- import 済み seed は再開せず、そこから clean な baseline run を作る用途に限ります。
- imported study 自体に fully self-contained / path-clean は求めません。
- 比較や継続学習の主語には imported study そのものではなく、そこから作る fresh run を使います。
- bulk OOF や大量の派生物は原則 import しません。

例:
```bash
uv run python -m keiba_research import legacy-tuning \
  --study-id imported_win_lgbm_seed \
  --kind binary \
  --task win \
  --model lgbm \
  --source-storage /path/to/legacy/study.sqlite3 \
  --source-best /path/to/legacy/best.json \
  --source-best-params /path/to/legacy/best_params.json
```

`--source-*` は user-supplied external artifact です。  
この repo や sibling checkout に存在する前提ではありません。

# Binary, Stacker, And Calibration

このファイルは、binary / stacker / wide pair calibrator の学習契約をまとめた詳細仕様です。  
run/study の保存先は `data-contract.md`、CLI の exact args は `cli-reference.md` を見てください。

## 1. Scope
- `train binary`
- `tune binary`
- `train stack`
- `tune stack`
- `train wide-calibrator`

ここでは learner の責務、時系列境界、OOF/holdout 契約、校正の位置づけを扱います。

## 2. Temporal policy
### 2.1 OOF and holdout boundary
current 実装では binary / stack / PL のすべてで次を採用します。

- OOF 学習対象
  - `year < holdout_year`
- holdout 推論対象
  - `year >= holdout_year`

例:
- `holdout_year = 2025`
  - OOF は 2024 年以前
  - holdout は 2025 年以降

### 2.2 CV policy per layer
- binary
  - `fixed_sliding`
  - `train_window_years = 3`
- stacker
  - `capped_expanding`
  - `min_train_years` と `max_train_years` を使用
- PL
  - `fixed_sliding`
  - 詳細は `pl-inference-and-wide-backtest.md`

public default:
- binary / PL
  - `train_window_years = 3`
- stacker
  - `min_train_years = 2`
  - `max_train_years = 3`

OOF / holdout parquet には少なくとも次が入ります。

- `fold_id`
- `valid_year`
- `cv_window_policy`
- `train_window_years`
- `holdout_year`
- `window_definition`

## 3. Binary
### 3.1 Role
- `win` と `place` を個別に学習する horse-level 二値分類層です
- model choice は `lgbm`, `xgb`, `cat`
- label 列は `y_win` または `y_place`

### 3.2 Input contract
- `feature_set=base`
  - `features_v3.parquet`
- `feature_set=te`
  - `features_v3_te.parquet`
- actual feature 列は `feature_registry_v3.py` が決めます

### 3.3 Outputs
- OOF
  - `artifacts/oof/<task>_<model>_oof.parquet`
- holdout
  - `artifacts/holdout/<task>_<model>_holdout_<holdout_year>.parquet`
- metrics
  - `artifacts/reports/<task>_<model>_cv_metrics.json`
- models
  - recent-window model
  - all-years model
  - bundle meta

bundle / metrics の section 名は `binary.<task>.<model>` です。

### 3.4 Binary CV and metrics
- yearly fold は fixed 3 年窓です
- fold metric は少なくとも次を持ちます
  - `logloss`
  - `brier`
  - `auc`
  - `ece`
  - `base_rate`
  - `reliability`
- `win` だけは Benter 系指標も計算します
  - `benter_beta_hat`
  - `benter_r2_valid`
  - `benter_r2_valid_beta1`

### 3.5 Binary tuning
`tune binary` の default contract は次です。

- `cv_window_policy = fixed_sliding`
- `train_window_years = 3`
- `operational_mode = t10_only`
- `include_entity_id_features = false`

変動する主な要素:
- `train_window_years`
- `feature_set = {base, te}`
- model hyperparameters

trial selection:
- objective は mean logloss の最小化です
- `win` では baseline の Benter 指標に対する制約を先に見て、通過 trial の中で最低 logloss を優先します

## 4. Stacker
### 4.1 Role
- binary の horse-level prediction と市場情報を統合する strict temporal stacker です
- current learner は LightGBM です

### 4.2 Input contract
- feature build
  - `features_v3.parquet`
- upstream binary predictions
  - `source_run_id` の OOF / holdout を使用
- required prediction 列
  - `win`: `p_win_lgbm`, `p_win_xgb`, `p_win_cat`
  - `place`: `p_place_lgbm`, `p_place_xgb`, `p_place_cat`
- additional feature 列
  - stacker 用 context
  - t20 / t15 / t10 odds 系

`source_run_id` 未指定時は自分自身の `run_id` を upstream source とみなします。

### 4.3 Stacker CV
- `capped_expanding` を使います
- current tuning / training surface では
  - `min_train_years` default = 2
  - `max_train_years` default = 3
- `window_definition` も binary とは別の capped-expanding 文言になります

### 4.4 Outputs
- OOF
  - `artifacts/oof/<task>_stack_oof.parquet`
- holdout
  - `artifacts/holdout/<task>_stack_holdout_<holdout_year>.parquet`
- metrics
  - `artifacts/reports/<task>_stack_cv_metrics.json`
- models
  - recent-window model
  - all-years model
  - feature manifest

bundle / metrics の section 名は `stack.<task>` です。

### 4.5 Stacker tuning
- objective は mean logloss の最小化です
- fixed learner は LightGBM
- tunable window は `min_train_years`, `max_train_years`

## 5. Wide Pair Calibrator
### 5.1 Role
- PL が出した pair-level wide probability の raw 値を、pair-level label に対して校正する層です
- public surface は `train wide-calibrator`

### 5.2 Input modes
下位 script は 2 種類の入力を扱えます。

- horse-level
  - `race_id`, `horse_no`, `pl_score`
  - script 側で Monte Carlo により `p_wide_raw` を再構成します
- pair-level
  - `race_id`, `horse_no_1`, `horse_no_2`, `p_wide` または `p_wide_raw`

repo-level wrapper は current 実装で source run の PL holdout horse-level parquet を入力にします。
repo-level wrapper は次の順で fit input を解決します。

1. `artifacts/oof/pl_<profile>_wide_oof.parquet`
2. `artifacts/oof/pl_<profile>_oof.parquet`

apply / evaluation input は常に `artifacts/holdout/pl_<profile>_holdout_<holdout_year>.parquet` です。
`--years` / `--require-years` は fit dataset selection として扱います。

### 5.3 Label and methods
- label は DB の実着順から作る `y_wide`
  - 2 頭とも top3 に入った pair を 1
- method choice
  - `isotonic`
  - `logreg`

raw と calibrated の列名契約:
- input
  - `p_wide_raw`
- output
  - `p_wide`

### 5.4 Outputs
- model
  - `artifacts/models/wide_pair_calibrator_<method>.joblib`
- predictions
  - `artifacts/predictions/wide_pair_calibration_<method>_pred.parquet`
  - holdout apply の out-of-sample artifact
- metrics
  - `artifacts/reports/wide_pair_calibration_<method>_metrics.json`
  - `fit` と `holdout_eval` を分けた nested report

bundle / metrics の section 名は `wide_calibrator.<method>` です。

### 5.5 Current public limitation
- `eval backtest --input-kind wide_calibrated` は isotonic output 固定です
- `logreg` の prediction file を public wrapper から切り替える引数はまだありません

## 6. Run composition and caveats
### 6.1 Natural composition
- same run
  - binary -> stack -> pl -> wide-calibrator -> backtest
- cross run
  - `train stack --source-run-id <binary_run>`
  - `train pl --source-run-id <upstream_run>`
  - `train wide-calibrator --source-run-id <pl_run>`

### 6.2 Wide-calibrator run config
- `train wide-calibrator` は source run の `pl_feature_profile` と `holdout_year` を読んで入力 parquet を決めます
- repo-level wrapper は calibrator run 側にもその profile/year を写します
- source run に `feature_profile` / `feature_build_id` がある場合は、それも calibrator run config に写します
- そのため cross-run でも `eval backtest --input-kind wide_calibrated` を自己完結で実行できます

## 7. Metrics and compare surface
- repo-level wrapper は report JSON 全体を `metrics.json.sections.*.report` に保存します
- `eval compare` はこの report 配下の numeric 値を平坦化して delta を作ります
- つまり compare 対象は純粋な quality metric だけではありません
  - rows
  - years
  - iteration count
  - cap 設定由来の数値
  なども numeric なら delta 化されます

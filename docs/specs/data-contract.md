# Data Contract

## Environment
- `V3_ASSET_ROOT`
  - 必須
  - この repo が生成する外部資産の保存先
- `V3_DATABASE_URL`
  - DB 接続先
- `V3_MODEL_THREADS`
  - 必要に応じて下位スクリプトへ渡す

この repo は、repo 相対の共有出力を使いません。  
feature、study、run、compare 出力はすべて `V3_ASSET_ROOT` 配下に置きます。

## Identifier rules
`run_id`, `study_id`, `feature_profile`, `feature_build_id` には次の制約があります。
- 空文字禁止
- path separator 禁止
- 正規表現: `^[A-Za-z0-9][A-Za-z0-9._-]*$`

意図:
- asset root 配下の directory 名として安全に使う
- bundle metadata に path を埋め込まない
- compare / import の参照名を機械的に扱えるようにする

## Asset layout
```text
$V3_ASSET_ROOT/
├── data/
│   ├── jsonl/
│   └── features/<feature_profile>/<feature_build_id>/
├── studies/
│   ├── <study_id>/
│   └── imported/<study_id>/
├── runs/<run_id>/
└── cache/
```

## Canonical JSONL input
rebuild の前提 input は次です。
- `RACE`
- `DIFF`
- `MING`
- `0B13`
- `0B17`
- `0B41_ALL_<date>.jsonl`

`0B41` は「最新ファイル自動採用」をしません。  
`db rebuild` では `--o1-date YYYYMMDD` を必須にし、使った consolidated snapshot を明示します。  
baseline を再現する場合は、まず `0B41_ALL_20260215.jsonl` を基準に扱います。

rebuild の scope:
- history scope table
  - `race`, `horse`, `jockey`, `trainer`, `runner`, `result`
- target scope table
  - `payout`, `o1_header`, `o1_win`, `o1_place`, `o3_header`, `o3_wide`, `mining_dm`, `mining_tm`, `rt_mining_dm`, `rt_mining_tm`

`db rebuild` は summary JSON を出力します。repo-level CLI の default は `cache/keiba_v3_rebuild_summary_<o1_date>.json` です。

## Feature build contract
feature build は `feature_profile` と `feature_build_id` で分離します。

保存先:
```text
$V3_ASSET_ROOT/data/features/<feature_profile>/<feature_build_id>/
├── config.toml
├── features_base.parquet
├── features_base_meta.json
├── features_base_te.parquet
├── features_base_te_meta.json
├── features_v3.parquet
├── features_v3_meta.json
├── features_v3_te.parquet
└── features_v3_te_meta.json
```

`config.toml` には少なくとも次が入ります。
- `feature_profile`
- `feature_build_id`
- `from_date`
- `to_date`
- `history_days`

最小例:
```toml
feature_profile = "baseline_v3"
feature_build_id = "baseline_20260321"
from_date = "2016-01-01"
to_date = "2025-12-31"
history_days = 730
```

## Study bundle contract
study は Optuna の mutable state です。

保存先:
```text
$V3_ASSET_ROOT/studies/<study_id>/
├── config.toml
├── study.sqlite3
├── best.json
├── selected_trial.json
└── trials.parquet
```

imported legacy seed は次へ置きます。
```text
$V3_ASSET_ROOT/studies/imported/<study_id>/
```

CLI では `imported.<study_id>` として参照します。  
imported study は `read_only_seed = true` を持ち、resume できません。
imported study は provenance-preserving seed です。  
新 repo の canonical study/run のような完全 self-contained bundle は要求しません。

repo-level `tune` / `import` wrapper が生成またはコピーした JSON metadata は、可能な限り
`V3_ASSET_ROOT` 相対 path に正規化されます。  
新規 study で生成される `best.json` / `selected_trial.json` に absolute path を残すことは許容しません。  
legacy import だけは source file 由来の外部 path を含み得ます。  
これは imported study を read-only seed と割り切り、旧環境の provenance を残すためです。

`config.toml` には次のような情報が入ります。
- `study_id`
- `kind`
  - `binary` または `stack`
- `task`
- `model`
  - binary のとき
- `source_run_id`
  - stack のとき
- `feature_profile`
- `feature_build_id`
- `imported`
- `read_only_seed`

最小例:
```toml
study_id = "win_lgbm_baseline"
kind = "binary"
task = "win"
model = "lgbm"
feature_profile = "baseline_v3"
feature_build_id = "baseline_20260321"
imported = false
read_only_seed = false
```

`selected_trial.json` の最小例:
```json
{
  "task": "win",
  "model": "lgbm",
  "feature_set": "te",
  "operational_mode": "t10_only",
  "include_entity_id_features": false,
  "train_window_years": 3,
  "lgbm_params": {
    "learning_rate": 0.03,
    "num_leaves": 63
  },
  "final_num_boost_round": 412
}
```

重要な不変条件:
- 新規 study の resume では `config.toml` と既存 study の user attrs が一致しないと失敗します。
- imported study は `read_only_seed=true` なので repo-level `tune` から再開できません。
- imported study は path-clean を保証しません。比較や学習の基準に使う場合は、そこから fresh run を作って固定します。

## Run bundle contract
run は train 結果と evaluation 結果を固定保存する比較単位です。

保存先:
```text
$V3_ASSET_ROOT/runs/<run_id>/
├── config.toml
├── bundle.json
├── metrics.json
├── resolved_params.toml
└── artifacts/
    ├── models/
    ├── oof/
    ├── holdout/
    ├── predictions/
    └── reports/
```

意味:
- `config.toml`
  - feature profile、feature build、holdout year、PL profile、source run などの固定条件
- `bundle.json`
  - artifact の索引
- `metrics.json`
  - compare 用に集約した数値指標
- `resolved_params.toml`
  - 各 train ステップで commands 層から実際に渡されたパラメータの記録
  - section key は `bundle.json` と同じ命名（例: `binary.win.lgbm`, `stack.win`）
  - commands 層で確定した引数と study/config 由来の値を保存する
  - training 側で適用される内部デフォルトは必ずしも含まれない
  - 再現確認や study → run の対応追跡に使う
- `artifacts/`
  - 実ファイル本体

repo-level `train` / `eval` / `db rebuild` wrapper が生成する JSON metadata は、
absolute path ではなく `V3_ASSET_ROOT` 相対 path を保存します。  
`bundle.json` と `metrics.json` だけでなく、`*_bundle_meta*.json`, `backtest_*_meta.json`,
feature meta, rebuild summary も同じ扱いです。

`bundle.json` の基本形:
```json
{
  "bundle_version": 1,
  "run_id": "<run_id>",
  "generated_at": "<timestamp>",
  "sections": {}
}
```

`metrics.json` の基本形:
```json
{
  "metrics_version": 1,
  "run_id": "<run_id>",
  "generated_at": "<timestamp>",
  "sections": {}
}
```

`config.toml` の更新ルール:
- 同じ run に対する設定追加は strict merge です。
- 既存値と違う値で同じ key を書こうとすると失敗します。
- つまり 1 run の `feature_profile`, `feature_build_id`, `holdout_year`, `pl_feature_profile` は一貫していなければなりません。

repo-level CLI が保存する代表 section:
- `binary.win.lgbm`, `binary.win.xgb`, `binary.win.cat`
- `binary.place.lgbm`, `binary.place.xgb`, `binary.place.cat`
- `stack.win`, `stack.place`
- `pl.stack_default`, `pl.stack_default_age_v1`
- `wide_calibrator.isotonic`, `wide_calibrator.logreg`
- `backtest.pl_holdout`, `backtest.pl_oof`, `backtest.wide_calibrated`

`config.toml` の最小例:
```toml
run_id = "baseline_run"
feature_profile = "baseline_v3"
feature_build_id = "baseline_20260321"
holdout_year = 2025
pl_feature_profile = "stack_default"
```

`resolved_params.toml` の最小例:
```toml
[binary.win.lgbm]
feature_set = "te"
holdout_year = 2025
train_window_years = 3
num_leaves = 63
learning_rate = 0.03
final_num_boost_round = 412

[stack.win]
holdout_year = 2025
min_train_years = 2
max_train_years = 3
num_leaves = 31
final_num_boost_round = 200
```

`bundle.json` の section 最小例:
```json
{
  "bundle_version": 1,
  "run_id": "baseline_run",
  "generated_at": "2026-03-21T10:00:00Z",
  "sections": {
    "binary.win.lgbm": {
      "feature_set": "base",
      "study_id": "win_lgbm_baseline",
      "input": "data/features/baseline_v3/baseline_20260321/features_v3.parquet",
      "oof": "runs/baseline_run/artifacts/oof/win_lgbm_oof.parquet",
      "holdout": "runs/baseline_run/artifacts/holdout/win_lgbm_holdout_2025.parquet",
      "metrics": "runs/baseline_run/artifacts/reports/win_lgbm_cv_metrics.json",
      "model": "runs/baseline_run/artifacts/models/win_lgbm_v3.txt"
    }
  }
}
```

`metrics.json` の section 最小例:
```json
{
  "metrics_version": 1,
  "run_id": "baseline_run",
  "generated_at": "2026-03-21T10:05:00Z",
  "sections": {
    "binary.win.lgbm": {
      "path": "runs/baseline_run/artifacts/reports/win_lgbm_cv_metrics.json",
      "report": {
        "summary": {
          "logloss": {
            "mean": 0.48
          }
        }
      }
    }
  }
}
```

compare 出力の default path:
```text
$V3_ASSET_ROOT/cache/compare/<left_run_id>__vs__<right_run_id>.json
```

compare 出力の最小例:
```json
{
  "comparison_version": 1,
  "left_run_id": "baseline_run",
  "right_run_id": "candidate_run",
  "left_metrics": "runs/baseline_run/metrics.json",
  "right_metrics": "runs/candidate_run/metrics.json",
  "common_numeric_deltas": {
    "backtest.pl_holdout.report.summary.roi": {
      "left": 1.02,
      "right": 1.08,
      "delta": 0.06
    }
  }
}
```

## Path rules
- repo-native metadata は `V3_ASSET_ROOT` 相対 path だけを持ちます。
  対象は `bundle.json`, `metrics.json`, study config, feature meta, rebuild summary,
  `*_bundle_meta*.json`, `backtest_*_meta.json` です。
- repo-native metadata に absolute path は保存しません。
- old repo の metadata を取り込むときだけ import 時に rewrite します。
- imported study は例外です。
  imported study は read-only seed なので、legacy provenance を残すため外部 absolute path を含み得ます。
- `run_id`, `study_id`, `feature_profile`, `feature_build_id` には path separator を含めません。

`asset_relative()` は `V3_ASSET_ROOT` 外の path を拒否します。  
`resolve_asset_path()` は absolute path を拒否します。  
この 2 つが永続 metadata の path 制約を担保します。

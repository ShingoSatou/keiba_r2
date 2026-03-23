# Architecture And CV

このファイルは、architecture と CV / OOF の流れを図で読むための visual guide です。  
細かい契約は `feature-and-odds.md`, `binary-stacker-and-calibration.md`, `pl-inference-and-wide-backtest.md` を見てください。

## 1. System map
```mermaid
flowchart LR
  JSONL["Canonical JSONL"] --> DB["db rebuild -> keiba_v3"]
  DB --> FB["features build-base/build/build-te"]
  FB --> TB["tune binary / tune stack (study)"]
  FB --> B["train binary (run)"]
  TB --> B
  B --> S["train stack (run)"]
  S --> P["train pl (run)"]
  P --> W["train wide-calibrator (run)"]
  P --> BT1["eval backtest: pl_oof / pl_holdout"]
  W --> BT2["eval backtest: wide_calibrated"]
  BT1 --> CMP["eval compare"]
  BT2 --> CMP
```

見方:
- `study` は tune の mutable state
- `run` は train / eval の固定比較単位
- `compare` は `metrics.json` を読む派生処理で、source of truth ではありません

## 2. Layer dependency and OOF usage
```mermaid
flowchart TD
  F["features_v3 / features_v3_te"] --> B1["binary train/CV"]
  B1 --> BOOF["binary OOF"]
  B1 --> BH["binary holdout"]

  F --> S1["stack train/CV"]
  BOOF --> S1
  BH --> SH["stack holdout scoring"]
  S1 --> SOOF["stack OOF"]

  F --> P1["PL train/CV"]
  SOOF --> P1
  SH --> PH["PL holdout scoring"]
  P1 --> POOF["PL OOF"]
 P1 --> PWOOF["PL wide_oof"]

  POOF --> BTO["backtest pl_oof"]
  PH --> BTH["backtest pl_holdout"]
  PWOOF --> WC["wide calibrator fit"]
  POOF --> WC2["wide calibrator fit fallback"]
  PH --> WA["wide calibrator apply/eval"]
  WC --> WA
  WC2 --> WA
  WA --> WP["wide calibrated predictions"]
  WP --> BTW["backtest wide_calibrated"]
```

重要:
- stacker は binary OOF を train/CV に使い、binary holdout を holdout scoring に使います
- PL は stack OOF を train/CV に使い、stack holdout を holdout scoring に使います
- wide calibrator は PL wide OOF を第一候補、horse-level PL OOF を fallback にして fit し、PL holdout に apply/eval します
- `source_run_id` を変えると、これら upstream OOF / holdout の参照元 run を切り替えられます

## 3. CV split matrix
| layer | train rows | valid rows | holdout rows | CV policy | OOF artifact | downstream use |
|---|---|---|---|---|---|---|
| binary | `year < holdout_year` | 各 fold の `valid_year` | `year >= holdout_year` | fixed-sliding 3y | `win/place_<model>_oof.parquet` | stacker |
| stacker | `year < holdout_year` かつ binary OOF が必要 | capped-expanding の `valid_year` | `year >= holdout_year` かつ binary holdout が必要 | capped-expanding `min=2 max=3` | `<task>_stack_oof.parquet` | PL |
| PL | `year < holdout_year` かつ stack OOF が必要 | fixed-sliding の `valid_year` | `year >= holdout_year` かつ stack holdout が必要 | fixed-sliding 3y | `pl_<profile>_oof.parquet` | backtest / inspection |
| wide calibrator | PL OOF / wide OOF | yearly fold CV なし | `pl_<profile>_holdout_<holdout_year>.parquet` | OOF fit / holdout apply | `pl_<profile>_wide_oof.parquet` or `pl_<profile>_oof.parquet` | `wide_calibrated` backtest |
| backtest | train/valid split を作らない | なし | `valid_year < filter_year` の選択 | CV ではない | なし | compare |

## 4. Binary and PL fixed-sliding example
例として available years が `2016-2025`, `holdout_year = 2025`, `train_window_years = 3` のとき:

```mermaid
flowchart TB
  F1["Fold 1<br>train: 2016-2018<br>valid: 2019"] --> O1["OOF rows: 2019"]
  F2["Fold 2<br>train: 2017-2019<br>valid: 2020"] --> O2["OOF rows: 2020"]
  F3["Fold 3<br>train: 2018-2020<br>valid: 2021"] --> O3["OOF rows: 2021"]
  F4["Fold 4<br>train: 2019-2021<br>valid: 2022"] --> O4["OOF rows: 2022"]
  F5["Fold 5<br>train: 2020-2022<br>valid: 2023"] --> O5["OOF rows: 2023"]
  F6["Fold 6<br>train: 2021-2023<br>valid: 2024"] --> O6["OOF rows: 2024"]
  FM["Final model<br>recent window: 2022-2024"] --> H["Holdout score: 2025+"]
```

この policy を使う層:
- binary
- PL

ただし PL は upstream の stacker OOF coverage に制約されます。  
`2016-2025`, `holdout_year=2025` の repo-level cascade では、PL OOF valid year は `2024` のみになります。

評価指標:
- 同じ fold split 上で `logloss`, `brier`, `auc`, `ece` を計算します
- `win` binary だけは同じ split 上で Benter 指標も追加します
- metric によって split が変わるわけではありません

## 5. Stacker capped-expanding example
例として `2016-2025`, `holdout_year = 2025` で binary OOF valid years が `2019-2024`,
`min_train_years = 2`, `max_train_years = 3` のとき:

```mermaid
flowchart TB
  S1["Fold 1<br>train: 2019-2020<br>valid: 2021"] --> V1["OOF rows: 2021"]
  S2["Fold 2<br>train: 2019-2021<br>valid: 2022"] --> V2["OOF rows: 2022"]
  S3["Fold 3<br>train: 2020-2022<br>valid: 2023"] --> V3["OOF rows: 2023"]
  S4["Fold 4<br>train: 2021-2023<br>valid: 2024"] --> V4["OOF rows: 2024"]
  SM["Final model<br>recent capped window: 2022-2024"] --> SH["Holdout score: 2025+"]
```

見方:
- 初期 fold では expanding
- 3 年に達した後は recent 3 years に cap されます
- stacker の `train_window_years` は実質 `max_train_years` を表します

## 6. Backtest is not CV
backtest は fold CV ではありません。

- `pl_oof`
  - すでに作った OOF prediction を後から評価する
- `pl_holdout`
  - holdout prediction を後から評価する
- `wide_calibrated`
  - calibrated pair prediction を後から評価する

repo-level wrapper の holdout-year rule:
- `pl_oof`
  - `holdout_year` をそのまま使う
- `pl_holdout`, `wide_calibrated`
  - `holdout_year + 1` を下位 script に渡す

これは backtest script が `valid_year < filter_year` で rows を選ぶためです。

## 7. What changes and what does not
### 7.1 What changes by layer
- CV policy
  - binary / PL は fixed-sliding
  - stacker は capped-expanding
- upstream dependency
  - stacker は binary OOF / holdout
  - PL は stack OOF / holdout
  - wide calibrator は PL OOF / wide OOF で fit し、PL holdout に apply します

### 7.2 What does not change by metric
- `logloss`
- `brier`
- `auc`
- `ece`
- Benter

これらは「同じ split の上で計算する metric」が違うだけです。  
split policy 自体は layer が決めます。

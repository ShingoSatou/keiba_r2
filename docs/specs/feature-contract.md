# Feature Contract

## Principle
- このファイルは feature contract の要約です。
- 列粒度の詳細、odds snapshot、safe TE merge は `feature-and-odds.md` を優先します。
- 移行時点では現行 v3 の baseline feature contract を保ちます。
- 新しい特徴量は baseline を直接上書きせず、別 `feature_profile` として追加します。
- すべての train run は `feature_profile` と `feature_build_id` を明示します。
- PL 側の feature 切替は `pl_feature_profile` として別軸で記録します。

## Naming
- baseline: `baseline_v3`
- 追加案: `baseline_v3_plus_<slug>_vN`
- 削除案: `baseline_v3_minus_<slug>_vN`

命名では「何を変えた profile か」を読めることを優先します。  
複数変更を混ぜる場合も、run 名ではなく feature profile 名で差分を追えるようにします。

## Segment and history scope
base features は次の segment filter を前提にします。
- `track_code` 01-10
- `surface = 2`
- `race_type_code in {13, 14}`
- `condition_code_min_age in {10, 16, 999}`
- `condition_code_min_age not in {701, 702, 703}`
- `distance_m > 0`
- `field_size > 0`
- `horse_no` は 1-18
- `finish_pos` not null

JV-Data 仕様書での読み替え:
- `surface` は raw JV code ではなく repo の正規化列です
  - `2009.トラックコード` の 23-29 を `surface = 2` に寄せています
  - したがって current target は平地ダートに加えてサンドも含みます
- `race_type_code 13`
  - `2005.競走種別コード` では `サラブレッド系3歳以上`
- `race_type_code 14`
  - `2005.競走種別コード` では `サラブレッド系4歳以上`
- `condition_code_min_age 10`
  - repo では整数で `10` ですが、JV `2007.競走条件コード` では `010`
  - 名称は `１０００万円以下 ２勝クラス`
- `condition_code_min_age 16`
  - repo では整数で `16` ですが、JV `2007.競走条件コード` では `016`
  - 名称は `１６００万円以下 ３勝クラス`
- `condition_code_min_age 999`
  - `2007.競走条件コード` では `オープン`
- 除外する `701 / 702 / 703`
  - `新馬`, `未出走`, `未勝利`

履歴参照の source table は `core.race`, `core.runner`, `core.result` です。  
`build-base` は `from_date - history_days` から `to_date` までの history scope を読んで rolling feature を作ります。

重要:
- lag / experience / TE の計算は広い history scope 上で行います。
- target segment filter は output を絞る段階で掛かります。
- つまり history scope 自体は target segment 限定ではありません。

## Build stages
### `features build-base`
- 入力: `keiba_v3`
- 出力:
  - `features_base.parquet`
  - `features_base_meta.json`
  - `features_base_te.parquet`
  - `features_base_te_meta.json`
- 備考:
  - repo-level wrapper は base build 実行時に TE source 用 base も同時に作ります
  - default `history_days` は 730
  - `--with-te` では jockey/trainer rolling target encoding 系列を加えます
  - meta JSON には rows, races, columns, history scope, segment filter, missing_rate, code_hash が入ります

### `features build`
- 入力: `features_base.parquet`
- 出力:
  - `features_v3.parquet`
  - `features_v3_meta.json`
- 追加される主要列:
  - `y_win`
  - `y_place`
  - `finish_pos`
  - t20 / t15 / t10 odds 系
  - final odds 系
  - stacker 用時系列 odds 差分列
- 不変条件:
  - `race_id`, `horse_no` で sort 済み
  - as-of future reference を検出したら失敗
  - required columns が欠けていると失敗

### `features build-te`
- 入力:
  - `features_v3.parquet`
  - `features_base_te.parquet`
- 出力:
  - `features_v3_te.parquet`
  - `features_v3_te_meta.json`
- join key:
  - `race_id`, `horse_id`, `horse_no`
- 不変条件:
  - 両入力で join key は一意
  - `te-source-input` に base 行が欠けると失敗
  - feature registry が認める safe TE extra columns だけを加える

## Stage semantics
- `features_base`
  - DB 由来の履歴特徴量と基本コンテキスト
  - odds snapshot や label 派生はまだ薄い
- `features_base_te`
  - `features_base` に TE 用安全列を加えたもの
  - `features build-te` の右側入力
- `features_v3`
  - binary / stack / PL が直接使う主 feature frame
  - `y_win`, `y_place`, odds snapshot, stacker 用時系列 odds 列を含む
- `features_v3_te`
  - `features_v3` に safe TE extra columns を one-to-one merge したもの

ここでの `TE` は target encoding です。  
この repo では、jockey / trainer などの rolling target encoding 系列を safe TE として扱います。

## Feature sets used by training
- `feature_set=base`
  - `features_v3.parquet` を使う
- `feature_set=te`
  - `features_v3_te.parquet` を使う

## Downstream compatibility matrix
| downstream command | consumes `features_base` | consumes `features_v3` | consumes `features_base_te` | consumes `features_v3_te` |
|---|---|---|---|---|
| `features build` | yes | no | no | no |
| `features build-te` | no | yes | yes | no |
| `tune binary` | no | yes | no | yes |
| `train binary --feature-set base` | no | yes | no | no |
| `train binary --feature-set te` | no | no | no | yes |
| `tune stack` | no | yes | no | no |
| `train stack` | no | yes | no | no |
| `train pl` | no | yes | no | no |

## Guardrails
- feature build は必ず named output にします。
- unsuffixed な共通出力は使いません。
- leakage guard と as-of 整合を維持します。
- baseline と candidate は別 `feature_profile` または別 `feature_build_id` として共存させます。

## Feature registry
実際に model が使う列集合は `scripts_v3/feature_registry_v3.py` が管理します。

現在の代表 surface:
- binary
  - base feature 群
  - optional final odds feature 群
  - optional safe TE feature 群
  - entity ID feature は default OFF
- stacker
  - binary prediction 列
  - 文脈 feature
  - t20 / t15 / t10 odds 系
- PL
  - profile ごとの feature set
  - underlying choices は `stack_default`, `stack_default_age_v1`, `meta_default`

研究 repo の public surface としては、まず `stack_default` を baseline にします。  
`meta_default` は feature registry には残っていますが、repo-level train wrapper の clean path には含めません。

重要:
- current 実装の `get_binary_feature_columns()` は `BINARY_T10_ODDS_FEATURES` を加えていません
- したがって、binary が t10 odds を default で使うという理解は現状 code と一致しません
- この spec は current code に合わせ、binary の default feature contract を「base + optional final + optional safe TE + optional entity ID」として扱います

# Feature And Odds

このファイルは、`features_base` / `features_v3` / `features_v3_te` の生成と、
odds snapshot の扱いを実装準拠でまとめた詳細仕様です。  
命名や change-unit の原則は `feature-contract.md`、保存先は `data-contract.md` を優先します。

## 1. Scope
- public entrypoint は `python -m keiba_research features ...`
- build logic の source of truth は `src/keiba_research/features/commands.py` と `scripts_v3/*`
- `feature_profile` は current 実装では build script を切り替える dispatch key ではなく、保存先と provenance の識別子です

## 2. Build stages
### 2.1 `features build-base`
- repo-level wrapper は 1 回の実行で次を連続生成します
  - `features_base.parquet`
  - `features_base_meta.json`
  - `features_base_te.parquet`
  - `features_base_te_meta.json`
- public CLI は `--with-te` を expose せず、base と TE source を常に両方作ります
- `config.toml` には `feature_profile`, `feature_build_id`, `from_date`, `to_date`, `history_days` を保存します

### 2.2 `features build`
- 入力は `features_base.parquet`
- 出力は `features_v3.parquet`
- ここで label/result 列と odds snapshot 列を追加します

### 2.3 `features build-te`
- 入力は `features_v3.parquet` と `features_base_te.parquet`
- 出力は `features_v3_te.parquet`
- one-to-one merge で safe TE extra columns だけを追加します

## 3. `features_base`
### 3.1 History scope and segment filter
- history scope は `from_date - history_days` から `to_date`
- source table は `core.race`, `core.runner`, `core.result`
- rolling feature の計算は広い history scope 上で行い、segment filter は output を切る最後の段で掛けます
- target segment の条件は current 実装で次です
  - `track_code` 01-10
  - `surface = 2`
  - `race_type_code in {13, 14}`
  - `condition_code_min_age in {10, 16, 999}`
  - `condition_code_min_age not in {701, 702, 703}`
  - `distance_m > 0`
  - `field_size > 0`
  - `horse_no` は 1-18
  - `finish_pos` not null

JV-Data 仕様書の code 表に沿った補足:
- `surface` は repo の正規化列で、raw JV field ではありません
  - `2009.トラックコード` の 23-29 を `surface = 2` として扱います
  - 23-26, 29 は平地ダート、27-28 は平地サンドです
- `race_type_code 13`
  - `2005.競走種別コード` の `サラブレッド系3歳以上`
- `race_type_code 14`
  - `2005.競走種別コード` の `サラブレッド系4歳以上`
- `condition_code_min_age 10`
  - repo では整数 `10` ですが、JV `2007.競走条件コード` では `010`
  - 名称は `１０００万円以下 ２勝クラス`
- `condition_code_min_age 16`
  - repo では整数 `16` ですが、JV `2007.競走条件コード` では `016`
  - 名称は `１６００万円以下 ３勝クラス`
- `condition_code_min_age 999`
  - `オープン`
- 除外する `701 / 702 / 703`
  - `新馬`, `未出走`, `未勝利`

### 3.2 Main column groups
`features_base` が持つ代表列群は次です。

- race context
  - `track_code`, `surface`, `distance_m`, `going`, `weather`, `field_size`, `grade_code`, `race_type_code`, `weight_type_code`, `condition_code_min_age`
- age / composition
  - `age`, `is_3yo`, `race_month`, `race_month_sin`, `race_month_cos`, `min_age_numeric`, `age_minus_min_age`, `is_min_age_runner`, `n_3yo_in_race`, `share_3yo_in_race`, `age_rank_pct_in_race`
- runner basics
  - `sex`, `carried_weight`, `body_weight`, `body_weight_diff`, `jockey_key`, `trainer_key`, `is_jockey_change`
- lag / performance
  - `days_since_lag1`, `lag1_distance_diff`, `lag1_course_type_match`, `lag1_finish_pos`, `lag2_finish_pos`, `lag3_finish_pos`
  - `lag1_speed_index`, `lag2_speed_index`, `lag3_speed_index`, `d_speed_index_1_2`, `d_speed_index_2_3`, `speed_index_slope_3r`
  - `lag1_up3_index`, `lag2_up3_index`, `lag3_up3_index`, `d_up3_index_1_2`, `d_up3_index_2_3`, `up3_index_slope_3r`
- experience / aptitude
  - `prior_starts_2y`, `days_since_first_seen_2y`, `apt_same_distance_top3_rate_2y`, `apt_same_going_top3_rate_2y`
- mining features
  - `meta_dm_time_x10`, `meta_dm_rank`, `meta_tm_score`, `meta_tm_rank`
- recent entity rate
  - `jockey_top3_rate_6m`, `trainer_top3_rate_6m`
- relative race-local features
  - `rel_lag1_speed_index_z`, `rel_lag1_speed_index_rank`, `rel_lag1_speed_index_pct`, `rel_carried_weight_z`, `rel_jockey_top3_rate_z`, `rel_meta_tm_score_z`

### 3.3 TE source columns
`features_base_te` は `features_base` に次を加えた TE source です。

- `jockey_target_label_mean_6m`
- `trainer_target_label_mean_6m`
- `rel_jockey_target_label_mean_z`

current public surface では、この 3 列が safe TE extra columns の中心です。

### 3.4 As-of and leakage guard
- lag 系は `lag1_race_datetime < race_datetime` を強制します
- `lag1_race_datetime` 自体は output に残さず、派生特徴だけを残します
- mining 系は `rt_mining_dm` / `rt_mining_tm` のうち `create_datetime <= race_datetime` を満たす最新行を優先し、欠けたら `mining_dm` / `mining_tm` に fallback します

## 4. `features_v3`
### 4.1 Label and result columns
`features_v3` では `features_base` に次を加えます。

- `y_win`
  - `target_label == 3`
- `y_place`
  - `target_label >= 1`
- `finish_pos`

### 4.2 Win odds snapshots
win odds は次の snapshot を持ちます。

- `final`
  - `odds_win_final`, `odds_final_data_kbn`, `p_win_odds_final_raw`, `p_win_odds_final_norm`
- `t20`
  - `odds_win_t20`, `odds_t20_data_kbn`, `p_win_odds_t20_raw`, `p_win_odds_t20_norm`
- `t15`
  - `odds_win_t15`, `odds_t15_data_kbn`, `p_win_odds_t15_raw`, `p_win_odds_t15_norm`
- `t10`
  - `odds_win_t10`, `odds_t10_data_kbn`, `p_win_odds_t10_raw`, `p_win_odds_t10_norm`

加えて stacker 用の時系列差分列を持ちます。

- `d_logit_win_15_20`
- `d_logit_win_10_15`
- `d_logit_win_10_20`

### 4.3 Place odds snapshots
place odds は `t20` / `t15` / `t10` を持ちます。

- raw band
  - `odds_place_t20_lower`, `odds_place_t20_upper`
  - `odds_place_t15_lower`, `odds_place_t15_upper`
  - `odds_place_t10_lower`, `odds_place_t10_upper`
- derived probability / width
  - `place_mid_prob_t20`, `place_mid_prob_t15`, `place_mid_prob_t10`
  - `place_width_log_ratio_t20`, `place_width_log_ratio_t15`, `place_width_log_ratio_t10`
  - `d_place_mid_10_20`, `d_place_width_10_20`
  - `place_width_log_ratio`

### 4.4 Snapshot selection rules
- `final`
  - `announce_datetime <= race_datetime` を満たす行だけを候補にし、その中で `data_kbn DESC`, `announce_dt DESC` の優先で選びます
- `t20` / `t15` / `t10`
  - 各 as-of 時刻より未来の announce は除外し、`announce_datetime <= asof` を満たす最新行だけを採用します
- snapshot が無い場合でも required 列は NaN で必ず出力します
- odds 側の leakage guard は `assert_asof_no_future_reference()` が再検証します

### 4.5 Physical columns and `operational_mode`
- current build は `t10_only` default でも final odds 列を物理的には保持します
- つまり `operational_mode` は build 出力を削る switch ではなく、主に downstream の feature selection 制約です
- `features_v3_meta.json` には次のような existence / coverage 情報を保存します
  - `contains_final_odds_columns`
  - `contains_t10_odds_columns`
  - `contains_stacker_timeseries_columns`
  - `coverage.*`

## 5. `features_v3_te`
### 5.1 Join contract
- join key は `race_id`, `horse_id`, `horse_no`
- `features_v3` 側も `features_base_te` 側も join key 一意が必須です
- `te-source-input` に base 側の行欠けがあると失敗します
- merge は `validate="one_to_one"` です

### 5.2 Safe TE rule
- safe TE extra columns の source of truth は `feature_registry_v3.py`
- current 判定は `get_binary_safe_te_feature_columns(frame, operational_mode=\"t10_only\", include_entity_ids=False)`
- 候補は `target` / `te` を含む列名ですが、次は除外されます
  - post-race 列
  - ID / key / date / time 系
  - odds 系
  - final odds 系
  - entity ID 列

## 6. Feature registry
### 6.1 Role
feature registry の役割は「どの learner がどの列を使うか」を固定し、禁止列を検証することです。

- 重複 feature 列を禁止
- post-race 列を禁止
- `t10_only` では final odds 列を禁止

### 6.2 Binary
binary の current 実装は次です。

- `BINARY_BASE_FEATURES`
- `operational_mode=includes_final` のときだけ `FINAL_ODDS_BASE_FEATURES`
- `feature_set=te` のときだけ safe TE extra columns
- `include_entity_id_features=true` のときだけ entity ID 列

重要:
- docs 上の古い説明では「binary は t10 odds feature 群を使う」と読める箇所がありました
- しかし current code の `get_binary_feature_columns()` は `BINARY_T10_ODDS_FEATURES` を加えていません
- 現行仕様書は code に合わせ、binary の default feature contract を上の定義で扱います

### 6.3 Stacker
stacker は次を使います。

- required upstream prediction 列
  - `p_win_lgbm`, `p_win_xgb`, `p_win_cat`
  - または `p_place_lgbm`, `p_place_xgb`, `p_place_cat`
- 文脈列
- t20 / t15 / t10 odds 系とその差分列

### 6.4 PL
PL の required upstream prediction 列は profile で変わります。

- `stack_default`, `stack_default_age_v1`
  - `p_win_stack`, `p_place_stack`
- `meta_default`
  - `p_win_meta`, `p_place_meta`, `p_win_odds_t10_norm`

repo-level public wrapper の clean path は stack-like profile のみです。

## 7. Metadata and guardrails
各 stage の meta JSON は少なくとも次を持ちます。

- rows / races / columns
- history scope または input path
- segment filter
- coverage / missing rate
- code hash

研究上の guardrails:
- unsuffixed 共通出力は使わず、`feature_profile` と `feature_build_id` を必ず明示します
- baseline と candidate は別 `feature_profile` または別 `feature_build_id` として共存させます
- current 実装では `feature_profile` は provenance の名前であり、script dispatch の switch ではありません

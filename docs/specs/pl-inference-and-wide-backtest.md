# PL Inference And Wide Backtest

このファイルは、PL layer、`p_top3` / `p_wide` 推定、wide backtest、購入ルールをまとめた詳細仕様です。  
public wrapper の exact args は `cli-reference.md`、run/study の保存先は `data-contract.md` を見てください。

## 1. PL role
- PL は ranking layer です
- upstream の stack prediction と一部文脈特徴量を使って、race 内順位分布を扱える score に変換します
- その score から Monte Carlo で `p_top3` と pair-level `p_wide_raw` を推定します

## 2. PL feature profiles
### 2.1 Public profiles
repo-level wrapper が clean path として扱う profile は次です。

- `stack_default`
- `stack_default_age_v1`

`meta_default` は registry と下位 script に残っていますが、repo-level train wrapper の public surface には出していません。

### 2.2 Required upstream prediction columns
- stack-like profile
  - `p_win_stack`
  - `p_place_stack`
- meta profile
  - `p_win_meta`
  - `p_place_meta`
  - `p_win_odds_t10_norm`

current public path は stack-like profile 前提です。

### 2.3 Materialized PL feature blocks
stack-like profile では `pl_v3_common.py` が次を materialize します。

- core
  - `z_win_stack`
  - `z_place_stack`
  - `place_width_log_ratio`
- interaction
  - `z_win_stack_x_z_place_stack`
  - `z_win_stack_x_place_width_log_ratio`
  - `z_place_stack_x_place_width_log_ratio`
  - `z_win_stack_x_field_size`
  - `z_place_stack_x_field_size`
  - `z_win_stack_x_distance_m`
  - `z_place_stack_x_distance_m`
  - `z_win_stack_race_centered`
  - `z_place_stack_race_centered`
  - `place_width_log_ratio_race_centered`
  - `z_win_stack_rank_pct`
  - `z_place_stack_rank_pct`
  - `place_width_log_ratio_rank_pct`
  - `z_win_stack_x_is_3yo`
  - `z_place_stack_x_is_3yo`
  - `place_width_log_ratio_x_is_3yo`
  - `z_win_stack_x_share_3yo_in_race`
  - `z_place_stack_x_share_3yo_in_race`
- age extension
  - `stack_default_age_v1` のときだけ `age_minus_min_age`, `is_min_age_runner`, `age_rank_pct_in_race`, `prior_starts_2y` とその interaction を追加

## 3. PL training contract
### 3.1 Temporal policy
- eligible rows は `year < holdout_year`
- yearly CV は `fixed_sliding`, `train_window_years = 3`
- holdout scoring 対象は `year >= holdout_year`

### 3.2 Outputs
- OOF horse-level
  - `artifacts/oof/pl_<profile>_oof.parquet`
- OOF pair-level
  - `artifacts/oof/pl_<profile>_wide_oof.parquet`
- holdout horse-level
  - `artifacts/holdout/pl_<profile>_holdout_<holdout_year>.parquet`
- metrics
  - `artifacts/reports/pl_<profile>_cv_metrics.json`
- year coverage
  - `artifacts/reports/pl_<profile>_year_coverage.json`
- models
  - recent-window model
  - all-years model
  - bundle meta

wrapper は current 実装で常に `--emit-wide-oof` を渡します。

### 3.3 Output columns
horse-level OOF / holdout の代表列:
- `race_id`, `horse_id`, `horse_no`
- `target_label`, `finish_pos`, `y_win`, `y_place`, `y_top3`
- required upstream prediction columns
- `pl_score`
- `p_top3`
- `fold_id`, `valid_year`, `cv_window_policy`, `train_window_years`, `holdout_year`, `window_definition`

pair-level OOF の代表列:
- `race_id`, `horse_no_1`, `horse_no_2`, `kumiban`
- `p_wide`
- `p_top3_1`, `p_top3_2`
- `fold_id`, `valid_year`, policy columns

### 3.4 Year coverage
`year_coverage.json` は current 実装の制約を可視化するために持ちます。

- `base_oof_years`
- `stacker_oof_years`
- `pl_eligible_years`
- `pl_oof_valid_years`
- `pl_holdout_train_years`
- `pl_fixed_window_oof_feasible`

OOF fold が 0 件でも、holdout scoring 自体は別に成立し得ます。

`2016-2025`, `holdout_year=2025` の例:
- `base_oof_years`
  - `2019-2024`
- `stacker_oof_years`
  - `2021-2024`
- `pl_oof_valid_years`
  - `2024`
- `pl_holdout_train_years`
  - `2022-2024`

## 4. Monte Carlo estimation
### 4.1 `p_top3`
- 入力は race 内の `pl_score`
- Gumbel sampling により top-k inclusion を近似します
- current default
  - `mc_samples = 10000`
  - `top_k = 3`

### 4.2 `p_wide_raw`
- 同じ sampled top-k inclusion から pair co-occurrence を計算します
- pair-level wide probability は `estimate_p_wide_by_race()` が作ります
- raw pair probability の列名は calibrator 文脈では `p_wide_raw`、PL OOF 文脈では `p_wide` です

### 4.3 Seed policy
- race ごとに deterministic な RNG seed を作ります
- fold OOF や holdout では base seed に fold offset / phase offset を加えます

## 5. Wide backtest input modes
`eval backtest` の下位 script は 2 種類の入力を扱います。

### 5.1 Horse-level input
- required columns
  - `race_id`
  - `horse_no`
  - `pl_score`
- script 側で Monte Carlo により pair-level `p_wide` を再構成します
- `p_wide_source = v3_pl_score_mc`

### 5.2 Pair-level input
- required columns
  - `race_id`
  - `horse_no_1`
  - `horse_no_2`
  - `kumiban`
  - `p_wide` または `p_wide_raw`
- `kumiban` と horse pair の整合、重複 key、確率の範囲を検証します

### 5.3 Repo-level `input-kind`
- `pl_holdout`
  - horse-level `pl_<profile>_holdout_<holdout_year>.parquet`
- `pl_oof`
  - horse-level `pl_<profile>_oof.parquet`
- `wide_calibrated`
  - pair-level `wide_pair_calibration_isotonic_pred.parquet`
  - `train wide-calibrator` が OOF で fit し、holdout に apply した out-of-sample pair prediction

wide calibrator の public workflow:
- fit
  - `pl_<profile>_wide_oof.parquet` を第一候補に使う
  - 無ければ horse-level `pl_<profile>_oof.parquet` から pair を再構成する
- apply / evaluation
  - `pl_<profile>_holdout_<holdout_year>.parquet`

`2016-2025`, `holdout_year=2025` の例では、fit years は `2024`、apply/eval years は `2025` です。

## 6. Holdout-year rule in backtest
backtest script 自体は `valid_year < --holdout-year` で入力を絞ります。  
そのため wrapper は次の調整を入れます。

- `pl_oof`
  - run config の `holdout_year` をそのまま渡す
- `pl_holdout`
  - `holdout_year + 1` を渡す
- `wide_calibrated`
  - `holdout_year + 1` を渡す

例:
- run config の `holdout_year = 2025`
  - `pl_oof` backtest は 2024 年以前
  - `pl_holdout` / `wide_calibrated` backtest は 2025 年を含める

## 7. Purchase rule
### 7.1 Candidate generation
1. pair-level `p_wide` を用意する
2. DB の `core.o3_wide` から race+kumiban ごとの latest odds を結合する
3. `ev_profit = p_wide * odds - 1.0` を計算する

odds source:
- `core.o3_wide`
- `min_odds_x10 / 10`
- `data_kbn DESC`, `announce_mmddhhmi DESC` で latest を選ぶ

payout source:
- `core.payout`
- `bet_type = 5`
- `selection = kumiban`

### 7.2 Filters
backtest script には次の filter があります。

- `min_p_wide`
  - 適用段階は `candidate` または `selected`
- `ev_threshold`
  - `ev_profit >= threshold`
- `max_bets_per_race`
  - EV 順に上位だけ残す

### 7.3 Kelly allocation
各候補の bet fraction は次です。

- full Kelly
  - `(p * odds - 1) / (odds - 1)`
- negative は 0 に丸める
- public default は fractional Kelly
  - `kelly_fraction = 0.25`

bet amount の流れ:
1. `kelly_f * bankroll`
2. optional `max_bet_yen`
3. `bet_unit_yen` 単位に切り捨て
4. `min_bet_yen` 未満を除外
5. race cap 超過時は全 bet を同比率で縮小
6. その後に daily cap を超える場合も同比率で縮小

### 7.4 Caps and defaults
public repo-level `eval backtest` はこれらを外に expose していません。  
したがって current public surface では次の default が事実上固定です。

- `mc_samples = 10000`
- `pl_top_k = 3`
- `min_p_wide = 0.0`
- `min_p_wide_stage = candidate`
- `ev_threshold = 0.0`
- `max_bets_per_race = 5`
- `kelly_fraction = 0.25`
- `race_cap_fraction = 0.05`
- `daily_cap_fraction = 0.20`
- `bankroll_init_yen = 1_000_000`
- `bet_unit_yen = 100`
- `min_bet_yen = 100`
- `max_bet_yen = none`

## 8. Backtest outputs
### 8.1 Report
`backtest_<input_kind>.json` は次を持ちます。

- `summary`
  - `period_from`, `period_to`
  - `n_races`, `n_bets`, `n_hits`
  - `hit_rate`, `total_bet`, `total_return`, `roi`, `max_drawdown`
- `monthly`
- `bets`

### 8.2 Meta report
`backtest_<input_kind>_meta.json` は次を持ちます。

- input mode と `p_wide_source`
- selected years / available years
- holdout-year filter
- CV policy
- selection config
- bankroll config
- DB sources

### 8.3 Quality metrics
- `target_label` と `p_top3` があるときだけ `logloss` と `auc` を追加します
- horse-level input では `p_top3` quality を見られます
- pair-level calibrated input は pair selection の成績が中心で、quality metric の意味は別物です

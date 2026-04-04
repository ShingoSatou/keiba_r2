# Evaluation And Comparison

## Scope
- official compare は `run` 同士の `metrics + backtest` 比較です。
- execution report は `run` から派生生成する read model です。
- selection-test suite は持ち込みません。
- production 用の single-race API は持ちません。
- 必要なら inspection 用の軽量 scoring surface を後から追加しますが、現状の中心は batch evaluation です。

## Backtest
`eval backtest` は run に保存済みの予測を入力に使います。

入力種別:
- `pl_holdout`
  - `artifacts/holdout/pl_<pl_feature_profile>_holdout_<holdout_year>.parquet`
- `pl_oof`
  - `artifacts/oof/pl_<pl_feature_profile>_oof.parquet`
- `wide_calibrated`
  - `artifacts/predictions/wide_pair_calibration_isotonic_pred.parquet`

出力:
- `artifacts/reports/backtest_<input_kind>.json`
- `artifacts/reports/backtest_<input_kind>_meta.json`
- `metrics.json` の `backtest.<input_kind>` section
- `bundle.json` の `backtest.<input_kind>` section

`backtest_<input_kind>_meta.json` の path field は repo-level wrapper が
`V3_ASSET_ROOT` 相対へ正規化します。

`metrics.json` に書かれるため、backtest 実行後の run は compare の材料になります。

注意:
- `wide_calibrated` は現在 `wide_pair_calibration_isotonic_pred.parquet` を前提にしています。
- この prediction file は `train wide-calibrator` が OOF で fit し、holdout に apply した out-of-sample artifact です。
- `logreg` 校正結果を使う compare surface は、まだ repo-level wrapper に出していません。
- `pl_holdout` と `wide_calibrated` は holdout-only artifact なので、wrapper は downstream に `holdout_year + 1` を渡して当年データが落ちないようにしています。
- `pl_oof` は run config の `holdout_year` をそのまま downstream に渡します。
- repo-level `eval backtest` は購入閾値や bankroll 引数を public に expose していません。
- したがって current public surface では、backtest の購入ルールは下位 script の default 値に固定されます。
- 購入ルールの詳細は `pl-inference-and-wide-backtest.md` を見てください。

## Compare
`eval compare` は 2 つの `run_id` の `metrics.json` を比較します。

比較ロジック:
- 両 run の `metrics.json` を読む
- `sections` 配下の数値項目を平坦化する
- 共通の numeric key だけを delta 化する
- compare summary を `cache/compare/<left>__vs__<right>.json` に書く

compare が読む対象:
- binary CV metric
- stack CV metric
- PL CV metric
- wide calibrator metric
- backtest metric
- rows や years や iteration count のような numeric な補助値

wide calibrator metric の注意:
- `fit` と `holdout_eval` の両方の numeric 値が compare に入ります
- in-sample と out-of-sample を混同しないため、section 名だけでなく key path も読む前提です

compare が読まないもの:
- bundle artifact の中身
- report JSON の文字列フィールド
- `metrics.json` 内の非数値項目
- list
- bool

compare 出力の主な field:
- `left_run_id`
- `right_run_id`
- `left_metrics`
- `right_metrics`
- `common_numeric_deltas`
- `left_sections`
- `right_sections`

解釈上の注意:
- delta の符号だけでは優劣を決められません
- `logloss` は低いほど良い
- `roi` は高いほど良い
- compare は metric の意味を自動解釈しません
- compare は quality metric だけを抽出していません
- そのため rows, holdout_year, final_iterations のような numeric key も delta に入ります
- したがって、最終判断では section 名と metric 名を人が読む必要があります

compare は `metrics.json` の数値だけを見ます。  
backtest を比較に含めたいなら、先に各 run で `eval backtest` を実行して metrics に書き込んでおく必要があります。

## Execution report
`eval report` は 1 run から `execution_report_summary.json` と `execution_report_detail.json` を生成します。

前提:
- source of truth は `run` のままです
- execution report は `bundle.json`, `metrics.json`, `resolved_params.toml`, run config, feature build config, artifact report を読む派生物です
- v1 では `1 report = 1 run` です

出力:
- `runs/<run_id>/execution_report_summary.json`
- `runs/<run_id>/execution_report_detail.json`
- optional `runs/<run_id>/execution_report_annotation.toml`

注意:
- compare は引き続き `metrics.json` だけを読みます
- execution report の追加で `eval compare` の入出力は変わりません
- detail は raw artifact 全文を複製する場所ではなく、artifact path と curated field を持つ read model です

## Rules
- 比較対象は常に `run_id` で明示します。
- global default や manifest snapshot に依存しません。
- compare の基準は「どの run が current か」ではなく、「どの条件の run 同士を比べるか」です。

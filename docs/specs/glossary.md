# Glossary

このファイルは、この repo の仕様で使う用語を定義します。  
曖昧な用語が出た場合は、この定義を正とします。

## Core terms
- `asset root`
  - `V3_ASSET_ROOT` で指定する外部資産の保存先
  - feature build, study, run, compare 出力はここに置く
- `feature_profile`
  - 特徴量契約の名前
  - どの列集合・どの特徴量差分を使うかを識別する
- `feature_build_id`
  - 同じ `feature_profile` 内で build 実行を区別する ID
  - 期間や build 日、調整条件の違いを分ける
- `feature build`
  - DB から生成した parquet 群
  - `features_base`, `features_v3`, `features_v3_te` を含む
- `study`
  - Optuna の mutable な探索単位
  - trial を追加できる
- `imported study`
  - legacy tuning 結果を seed として取り込んだ study
  - `read_only_seed=true` で resume 不可
  - provenance 保持を優先するため、完全 self-contained や path-clean は要求しない
- `run`
  - 比較のために固定した 1 実験結果
  - config, artifact, metrics, evaluation 結果を持つ
- `source_run_id`
  - 今回の train run が upstream prediction を読む元の run
- `compare report`
  - 2 run の `metrics.json` 差分をまとめた派生 JSON
- `execution report`
  - 1 run から派生生成する user-facing report JSON
  - v1 では `1 report = 1 run`
  - `run` 自体を置き換える source of truth ではない

## Prediction terms
- `OOF`
  - out-of-fold prediction
  - CV の validation 側予測を各行に戻したもの
- `holdout`
  - 指定 `holdout_year` の予測
- `holdout_year`
  - holdout 用に切り出す年
  - repo-level CLI の default は 2025
- `binary`
  - `win` / `place` を個別に学習する二値分類モデル群
- `stack`
  - binary OOF / holdout を入力にする strict temporal stacker
- `PL`
  - ranking layer
  - upstream の予測と一部文脈特徴量を入力にする
- `wide calibrator`
  - PL 出力から推定した pair-level wide probability を校正する層
- `p_top3`
  - PL score から Monte Carlo で推定した horse-level top3 probability
- `p_wide_raw`
  - PL score から Monte Carlo で推定した pair-level wide probability の未校正版
- `p_wide`
  - pair-level wide probability
  - 文脈により raw のことも calibrated のこともあるため、詳細 doc で出所を確認する
- `wide_oof`
  - PL OOF から生成した pair-level wide probability parquet

## Feature-set terms
- `base`
  - `features_v3.parquet`
  - safe TE extra columns を含まない
- `te`
  - `features_v3_te.parquet`
  - safe TE extra columns を含む
- `pl_feature_profile`
  - PL で使う feature recipe の識別子
  - current choices は `stack_default`, `stack_default_age_v1`, `meta_default`
- `operational_mode`
  - feature registry の odds contract 切替
  - current choices は `t10_only`, `includes_final`
  - build 出力そのものを減らす switch ではなく、主に downstream の feature selection 制約

## Persistence terms
- `config.toml`
  - run / study / feature build の固定条件
- `bundle.json`
  - run が生成した artifact の索引
- `metrics.json`
  - compare で使う数値指標の集約
- `execution_report_summary.json`
  - 一覧 / 比較向けに絞った execution report summary
- `execution_report_detail.json`
  - summary を含み、lineage / settings / diagnostics を足した詳細 report
- `execution_report_annotation.toml`
  - title / description / status / code revision override 用の任意 annotation
- `section`
  - `bundle.json` / `metrics.json` の中の名前付き領域
  - 例: `binary.win.lgbm`, `pl.stack_default`, `backtest.pl_holdout`

## Working-style terms
- `baseline`
  - 現時点の比較基準にする feature profile, study seed, run
- `read_only_seed`
  - imported study に付くフラグ
  - repo-level tuning から再開しない
- `worktree`
  - コード系統を分けるための git worktree
  - 実験結果の単位ではない
- `input_kind`
  - repo-level `eval backtest` が読む入力種別
  - `pl_holdout`, `pl_oof`, `wide_calibrated`

## Non-goals in this repo
- `current default`
  - この repo では持たない
- `selection-suite`
  - この repo では持たない
- `task/lane workflow`
  - この repo では持たない

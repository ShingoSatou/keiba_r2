---
name: research-run-ops
description: research repo の run を作成・比較・固定する。run config, bundle, metrics, artifacts の保存先と命名を扱うときに使う。
---

# Research Run Ops

## When to use
- `run` を新規作成するとき
- `run` 同士を比較するとき
- `bundle.json` や `metrics.json` の保存形式を扱うとき

## Read first
1. `docs/specs/architecture.md`
2. `docs/specs/data-contract.md`
3. `docs/specs/evaluation-and-comparison.md`

## Rules
- `run` は immutable にする
- output は `V3_ASSET_ROOT/runs/<run_id>/` 配下に集約する
- compare は `run_id` を明示し、global default に依存しない
- 共有名の unsuffixed 出力は使わない

## Workflow
1. run config を固定する
2. artifact 出力先を `run_id` で分離する
3. metrics と backtest を保存する
4. 比較対象の `run_id` を明示して差分を読む

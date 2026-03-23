---
name: legacy-asset-import
description: 旧 repo 由来の manifest, best_params, bundle meta, Optuna sqlite を新しい asset contract に移植する。path rewrite が必要なときに使う。
---

# Legacy Asset Import

## When to use
- 旧 manifest / bundle meta / best params を新 repo の asset contract に移すとき
- absolute path を relative path に書き換えるとき
- legacy tuning を seed 化するとき

## Read first
1. `docs/specs/data-contract.md`
2. `docs/specs/training-and-tuning.md`
3. `docs/history/migration-ledger.md`

## Rules
- 取り込む資産は seed として扱う
- `V3_ASSET_ROOT` 相対 path に正規化する
- `current default` を復活させない
- 再取得できないデータは user-managed external archive に残す

## Workflow
1. legacy asset の型を分ける
2. path を rewrite する
3. imported seed として保管する
4. fresh run / study へ接続する

## Example
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

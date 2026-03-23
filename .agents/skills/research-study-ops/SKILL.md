---
name: research-study-ops
description: research repo の Optuna study を作成・再開・seed import する。study.sqlite3, selected_trial.json, resume 可否を扱うときに使う。
---

# Research Study Ops

## When to use
- Optuna study を新規作成するとき
- study を resume するとき
- legacy study を import するとき

## Read first
1. `docs/specs/training-and-tuning.md`
2. `docs/specs/data-contract.md`
3. `docs/history/decision-log.md`

## Rules
- `study` は mutable
- imported legacy study は read-only seed として扱う
- selected trial は `selected_trial.json` に固定する
- `study_id` を必ず明示する

## Workflow
1. study config を固定する
2. study storage を `V3_ASSET_ROOT/studies/<study_id>/study.sqlite3` に置く
3. selected trial を export する
4. そこから fresh `run` を作る

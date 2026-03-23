---
trigger: always_on
---

# Project Rules

## Repo model
- この repo は production repo ではなく、research-only repo として扱う。
- source of truth は `run`, `study`, `feature_profile`。
- `current default`, UI/API surface, task/lane workflow は前提にしない。

## Tech
- Python 3.11 固定
- 依存管理は uv
- 入口は repo-level CLI を優先する

## Working style
- まず `docs/specs/` と `docs/history/` を確認する。
- 仕様変更は docs を先に更新するか、少なくとも同じ差分で更新する。
- worktree はコード系統が分岐するときだけ使う。
- run / study / feature profile を混同しない。

## Verification
- 振る舞いを変えたら、関係する smoke / pytest を追加または更新する。
- 変更の確認では、出力先が `V3_ASSET_ROOT` 相対で固定されているかを確認する。

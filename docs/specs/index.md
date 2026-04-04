# Specs Index

このディレクトリは、この repo の canonical specification です。  
最初にこのファイルを読み、次に glossary と各分野の specs を辿ってください。

まず「この repo は何を研究し、どのレースを対象にし、何を比較単位にするのか」を掴みたい場合は
`project-purpose-and-scope.md` を先に読んでください。

## Scope
この repo が扱うのは次です。
- `keiba_v3` migration / rebuild
- feature build
- binary / stack / PL / wide calibrator の training
- Optuna tuning
- backtest と run compare
- legacy tuning seed の import

この repo が扱わないのは次です。
- production default promotion
- FastAPI / frontend
- single-race operational API
- selection-suite
- task / lane workflow
- v1 / v2 surface

## Reading order
1. `project-purpose-and-scope.md`
2. `glossary.md`
3. `architecture.md`
4. `architecture-and-cv.md`
5. `data-contract.md`
6. `operations.md`
7. `feature-contract.md`
8. `feature-and-odds.md`
9. `training-and-tuning.md`
10. `binary-stacker-and-calibration.md`
11. `pl-inference-and-wide-backtest.md`
12. `evaluation-and-comparison.md`
13. `execution-report.md`
14. `cli-reference.md`

## Lifecycle
```text
canonical JSONL
  -> db migrate / db rebuild
  -> feature build
  -> optional study tuning
  -> run assembly
  -> backtest
  -> compare
```

## Term ownership
| term | owning doc |
|---|---|
| `run`, `study`, `feature_profile`, `worktree` | `glossary.md`, `architecture.md` |
| `feature_build_id`, `source_run_id`, `holdout_year` | `glossary.md`, `training-and-tuning.md` |
| `OOF`, `holdout`, `PL`, `stack`, `wide` | `glossary.md`, `training-and-tuning.md` |
| `p_top3`, `p_wide_raw`, `p_wide`, `wide_oof` | `glossary.md`, `pl-inference-and-wide-backtest.md` |
| `canonical JSONL`, `0B41`, `o1-date` | `glossary.md`, `data-contract.md` |
| `leakage guard`, `as-of`, `TE`, odds snapshots | `glossary.md`, `feature-contract.md`, `feature-and-odds.md` |
| `selected_trial`, `read_only_seed` | `glossary.md`, `data-contract.md`, `training-and-tuning.md` |
| `wide_calibrated`, purchase rule, compare limitations | `glossary.md`, `pl-inference-and-wide-backtest.md`, `evaluation-and-comparison.md` |
| `execution report`, `report_id`, report annotation | `glossary.md`, `execution-report.md`, `data-contract.md` |

## Public entrypoint
repo-level CLI:

```bash
uv run python -m keiba_research <group> <command> ...
```

詳細な subcommand 契約は `cli-reference.md` を見てください。

安全ルール:
- rebuild は `scripts_v3/rebuild_v3_db.py` 直呼びではなく、repo-level `db rebuild` を使います。
- wrapper は `--o1-date` を必須にして O1 snapshot を pin します。

## Current operating model
- source of truth は `run`, `study`, `feature_profile`
- global manifest は持たない
- compare は `run_id` を明示して行う
- repo-native metadata は `V3_ASSET_ROOT` 相対 path を持つ
- imported study は read-only seed の例外で、fully self-contained / path-clean は要求しない
- 通常の実験は 1 repo / 1 worktree / 複数 run

## If you are lost
- そもそもこの repo が何を目的にし、どのレースを対象にし、何を比較単位にしているか分からない
  - `project-purpose-and-scope.md`
- 用語が分からない
  - `glossary.md`
- どのファイルが何を担当しているか分からない
  - `architecture.md`
- アーキテクチャと CV / OOF を図で見たい
  - `architecture-and-cv.md`
- どこに何が保存されるか分からない
  - `data-contract.md`
- どの単位で差分を切るか分からない
  - `operations.md`
- CLI の正確な入口を知りたい
  - `cli-reference.md`
- 特徴量列や odds snapshot を詳しく追いたい
  - `feature-and-odds.md`
- binary / stack / wide calibrator の契約を追いたい
  - `binary-stacker-and-calibration.md`
- PL / `p_top3` / `p_wide` / 購入ルールを追いたい
  - `pl-inference-and-wide-backtest.md`
- 実行レポートの schema / source of truth / sample を見たい
  - `execution-report.md`

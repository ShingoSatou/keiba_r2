# Keiba Research

この repo は、競馬 v3 の研究を把握しやすく、再現しやすく回すための独立した研究 repo です。  
旧 repo の sibling directory や shared runtime を前提にせず、この repo 単体で構造と運用を読めるように整理しています。

repo 全体の目的、対象レース、比較単位は `docs/specs/project-purpose-and-scope.md` を起点に読むのが最短です。

## 何を扱う repo か
- `run`: 比較可能な 1 実験結果の固定 bundle
- `study`: Optuna の mutable な探索状態
- `feature_profile`: 特徴量契約の識別子

この repo は production の `current default` を持ちません。  
比較は常に `run_id` を明示して行います。

## Repo map
- `keiba_research/`, `src/keiba_research/`
  - 研究用の canonical CLI と保存契約
- `scripts_v3/`
  - CLI から呼ぶ curated subset の既存 v3 実装
- `migrations_v3/`
  - `keiba_v3` 用 migration
- `test_v3/`
  - 研究 repo 向け contract / smoke test
- `docs/specs/`
  - この repo 自体の仕様
- `docs/history/`
  - 研究継続に必要な要約履歴

## 最初に読む順番
1. `docs/specs/index.md`
2. `docs/specs/project-purpose-and-scope.md`
3. `docs/specs/glossary.md`
4. `docs/specs/architecture.md`
5. `docs/specs/data-contract.md`
6. `docs/specs/operations.md`
7. `docs/specs/cli-reference.md`
8. 必要に応じて `docs/specs/feature-contract.md`, `docs/specs/training-and-tuning.md`, `docs/specs/evaluation-and-comparison.md`
9. 歴史的背景が必要なときだけ `docs/history/README.md`

## Quick start
```bash
uv sync
export V3_ASSET_ROOT=/path/to/v3_assets
export V3_DATABASE_URL=postgresql://...
uv run python -m keiba_research --help
```

この workspace で現在使っている `V3_ASSET_ROOT` は
[`docs/history/active-asset-root.md`](docs/history/active-asset-root.md)
に公開用テンプレートを置いています。実際の absolute path は repo には commit せず、
ローカル環境変数や private note で管理します。

必要な JSONL は `$V3_ASSET_ROOT/data/jsonl/` 配下に置きます。

代表的な流れ:
```bash
uv run python -m keiba_research db migrate
uv run python -m keiba_research db rebuild --o1-date 20260215
uv run python -m keiba_research features build-base --feature-profile baseline_v3 --feature-build-id baseline_20260320 --from-date 2016-01-01 --to-date 2025-12-31
uv run python -m keiba_research features build --feature-profile baseline_v3 --feature-build-id baseline_20260320
uv run python -m keiba_research tune binary --study-id win_lgbm_baseline --task win --model lgbm --feature-profile baseline_v3 --feature-build-id baseline_20260320
uv run python -m keiba_research train binary --run-id baseline_run --task win --model lgbm --feature-profile baseline_v3 --feature-build-id baseline_20260320 --study-id win_lgbm_baseline
uv run python -m keiba_research eval compare --left-run-id baseline_run --right-run-id candidate_run
```

## 詳細 docs
- project purpose / scope: `docs/specs/project-purpose-and-scope.md`
- architecture: `docs/specs/architecture.md`
- specs index: `docs/specs/index.md`
- glossary: `docs/specs/glossary.md`
- data contract: `docs/specs/data-contract.md`
- CLI reference: `docs/specs/cli-reference.md`
- feature contract: `docs/specs/feature-contract.md`
- training / tuning: `docs/specs/training-and-tuning.md`
- evaluation / comparison: `docs/specs/evaluation-and-comparison.md`
- operations: `docs/specs/operations.md`
- history: `docs/history/README.md`

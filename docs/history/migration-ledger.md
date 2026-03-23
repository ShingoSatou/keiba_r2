# Migration Ledger

## 新 repo に持ち込んだもの
- `python -m keiba_research` を中心にした research CLI
- `V3_ASSET_ROOT` ベースの run / study / feature build 契約
- `keiba_v3` rebuild, feature build, training, tuning, backtest に必要な curated `scripts_v3/`
- `database.py`, `parsers.py` の必要責務
- reverse-engineered specs と要約履歴
- research repo 向け `AGENTS.md` と `.agents`

## 新 repo に持ち込まなかったもの
- FastAPI / frontend
- production default promotion
- selection-test suite
- task / lane workflow
- v1 / v2 surface
- dated run reports や raw audit 群

## 移行方針
- 旧 repo の構成をそのまま複製するのではなく、研究に必要な導線だけへ再編した
- 旧 provenance は archive として外出しし、この repo には要約だけを残した
- 既存 v3 実装は、まず curated subset の `scripts_v3/` として残し、その上に repo-level CLI をかぶせた

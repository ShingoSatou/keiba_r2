# Phase 1: scripts_v3 内部整理

**実施日:** 2026-04-01
**Issue:** #3

## 変更内容

### 1-1. train_stacker_v3_common.py の分割
- `train_stacker_v3.py`（新規）に `parse_args()`, `main()`, CLI ヘルパーを移動
- `train_stacker_v3_common.py` にはユーティリティ関数のみ残存
- stacker meta の `code_hash` は `train_stacker_v3.py` と common / CV / feature registry をまとめて hash するように補強
- params-json 適用時も CLI 明示フラグ優先を維持し、`code_hash` 入力 path は absolute に正規化
- `commands.py`, `test_research_repo_contracts.py` の import を更新

### 1-2. v3_common.py から bankroll 関数を分離
- `bankroll_v3.py`（新規）に `BankrollConfig`, Kelly 関連関数, `compute_max_drawdown` を移動
- `backtest_wide_v3.py`, `backtest_v3_common.py` を直接 import に変更
- `v3_common.py` から bankroll 関連コードを削除

### 1-3. 再エクスポートチェーンの解消
- `train_binary_v3_common.py` の `__all__` を自身で定義するシンボルのみに縮小
- `train_binary_model_v3.py`, `train_pl_v3.py`, `tune_binary_optuna_v3.py` を直接 import に変更
- `v3_common.py` から `cv_policy_v3` の re-export を削除
- `backtest_v3_common.py` を `cv_policy_v3` から直接 import に変更

## 影響
- ロジック変更なし
- 全 22 テスト通過

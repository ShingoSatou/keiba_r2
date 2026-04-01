# Phase 4: scripts_v3 を src/keiba_research に統合し scripts_v3/ を削除

**実施日:** 2026-04-02
**Issue:** #6
**PR:** #10

## 変更内容

### 4-1. ファイル移動（28 ファイル）
scripts_v3/ 配下の全ファイルを src/keiba_research/ 各サブパッケージへ移動（git rename として追跡）。

| 旧パス | 新パス |
|---|---|
| `scripts_v3/v3_common.py` | `src/keiba_research/common/v3_utils.py` |
| `scripts_v3/cv_policy_v3.py` | `src/keiba_research/training/cv_policy.py` |
| `scripts_v3/bankroll_v3.py` | `src/keiba_research/evaluation/bankroll.py` |
| `scripts_v3/train_binary_v3_common.py` | `src/keiba_research/training/binary_common.py` |
| `scripts_v3/train_pl_v3_common.py` | `src/keiba_research/training/pl_common.py` |
| `scripts_v3/train_stacker_v3_common.py` | `src/keiba_research/training/stacker_common.py` |
| `scripts_v3/train_binary_model_v3.py` | `src/keiba_research/training/binary.py` |
| `scripts_v3/train_pl_v3.py` | `src/keiba_research/training/pl.py` |
| `scripts_v3/train_stacker_v3.py` | `src/keiba_research/training/stacker.py` |
| `scripts_v3/train_wide_pair_calibrator_v3.py` | `src/keiba_research/training/wide_calibrator.py` |
| `scripts_v3/tune_binary_optuna_v3.py` | `src/keiba_research/tuning/binary.py` |
| `scripts_v3/tune_stacker_optuna_v3.py` | `src/keiba_research/tuning/stacker.py` |
| `scripts_v3/feature_registry_v3.py` | `src/keiba_research/features/registry.py` |
| `scripts_v3/build_features_base_v3_common.py` | `src/keiba_research/features/base_common.py` |
| `scripts_v3/build_features_base_v3.py` | `src/keiba_research/features/base.py` |
| `scripts_v3/build_features_v3.py` | `src/keiba_research/features/main.py` |
| `scripts_v3/build_features_v3_te.py` | `src/keiba_research/features/te.py` |
| `scripts_v3/backtest_v3_common.py` | `src/keiba_research/evaluation/backtest_common.py` |
| `scripts_v3/backtest_wide_v3.py` | `src/keiba_research/evaluation/backtest_wide.py` |
| `scripts_v3/apply_wide_pair_calibrator_v3.py` | `src/keiba_research/evaluation/apply_wide_calibrator.py` |
| `scripts_v3/evaluate_wide_calibration_v3.py` | `src/keiba_research/evaluation/wide_calibration.py` |
| `scripts_v3/odds_v3_common.py` | `src/keiba_research/evaluation/odds_common.py` |
| `scripts_v3/pl_v3_common.py` | `src/keiba_research/evaluation/pl_common.py` |
| `scripts_v3/metrics_benter_v3_common.py` | `src/keiba_research/evaluation/metrics_benter.py` |
| `scripts_v3/wide_pair_calibration_v3.py` | `src/keiba_research/evaluation/wide_pair_calibration.py` |
| `scripts_v3/rebuild_v3_db.py` | `src/keiba_research/db/rebuild.py` |
| `scripts_v3/rebuild_v3_jsonl_common.py` | `src/keiba_research/db/jsonl_common.py` |
| `scripts_v3/migrate_v3.py` | `src/keiba_research/db/migrate.py` |

### 4-2. import パスの統一
- 全ファイル内の `from scripts_v3.X import` → `from keiba_research.X import` に変換
- `commands.py` (5 ファイル) およびテストの import / `monkeypatch.setattr` 文字列を更新
- `# noqa: E402` コメントおよびファイル先頭の `sys.path.insert` 操作を除去

### 4-3. ruff 設定の整理
- `pyproject.toml` の `scripts_v3/*.py = ["E402"]` 除外設定を削除

### 4-4. Copilot レビュー指摘の修正（PR #9 対応分）
- `handle_binary` / `handle_stack`: `final_num_boost_round` → `num_boost_round` 変換
- `run_wide_calibrator`: `_resolve_output_path()` でメソッド別サフィックス付与を復元

## 影響
- `scripts_v3/` ディレクトリを削除
- import パスが `keiba_research.*` に統一
- 全 24 テスト通過

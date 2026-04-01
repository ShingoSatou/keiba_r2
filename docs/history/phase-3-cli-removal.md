# Phase 3: scripts_v3 からの CLI 除去とライブラリ専用化

**実施日:** 2026-04-02
**Issue:** #5
**PR:** #9

## 変更内容

### 3-1. scripts_v3 の CLI エントリポイント撤去
- 全 14 スクリプトから `parse_args()` / `main()` / `if __name__ == "__main__":` を除去
- 代わりに `run_X(**kwargs)` 形式のライブラリ関数を追加

| スクリプト | 追加した run 関数 |
|---|---|
| `rebuild_v3_db.py` | `run_rebuild` |
| `migrate_v3.py` | `run_migrate` |
| `build_features_base_v3.py` | `run_build_features_base` |
| `build_features_v3.py` | `run_build_features` |
| `build_features_v3_te.py` | `run_build_features_te` |
| `train_binary_model_v3.py` | `run_binary_training` |
| `train_stacker_v3.py` | `run_stacker_training` |
| `train_pl_v3.py` | `run_pl_training` |
| `train_wide_pair_calibrator_v3.py` | `run_wide_calibrator` |
| `tune_binary_optuna_v3.py` | `run_tune_binary` |
| `tune_stacker_optuna_v3.py` | `run_tune_stacker` |
| `backtest_wide_v3.py` | `run_backtest_wide` |
| `apply_wide_pair_calibrator_v3.py` | `run_apply_wide_calibrator` |
| `evaluate_wide_calibration_v3.py` | `run_evaluate_wide_calibration` |

### 3-2. commands.py の argv 生成を廃止
- `src/keiba_research/*/commands.py` で argv リスト組み立て → `run_X(**kwargs)` 直接呼び出しに変更
- `db`, `features`, `training`, `tuning`, `evaluation` の 5 コマンドファイルを更新

### 3-3. テストの kwargs ベース移行
- `test_v3/test_research_repo_contracts.py` の 10 monkeypatch テストを更新
  - monkeypatch 対象を旧 main 関数 → `run_X` 関数に変更
  - fake 実装を `argv: list[str]` 受取 → `**kwargs` 受取に変更

### 3-4. run_stacker_training の sentinel 化
- `min_train_years` / `max_train_years` のデフォルトを定数 → `None` に変更
- `_apply_params_json` が params_json の値で上書きできるよう、引数が未指定の場合は argv_list に追加しない設計

## 影響
- ロジック変更なし（argparse 内部で同じ値を解釈）
- 全 24 テスト通過

## Copilot レビュー指摘（PR #9、修正は Phase 4 ブランチで対応）
1. `handle_binary` / `handle_stack`: config_section の `final_num_boost_round` を `run_X` 関数へ渡す際に `num_boost_round` へ変換する必要があった
2. `run_wide_calibrator`: `resolve_path()` ではなく `_resolve_output_path()` を使わないと method 別サフィックスが付かない

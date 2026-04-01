# Phase 2: 宣言的 TOML 設定と学習パラメータ保存の導入

**実施日:** 2026-04-01
**Issue:** #4

## 変更内容

### 2-1. run_config.py の新規作成
- `src/keiba_research/common/run_config.py` に TOML 読み込み・バリデーション関数を追加
- `load_run_config()`: TOML ファイルを読み込み dict として返す
- `generate_config_from_study()`: `selected_trial.json` から run_config dict を生成
- `save_resolved_params()`: 確定パラメータを `resolved_params.toml` に保存

### 2-2. config コマンドの追加
- `src/keiba_research/config/commands.py` に `from-study` / `show` サブコマンドを実装
- `python -m keiba_research config from-study --study-id ...` で TOML 生成
- `python -m keiba_research config show <path>` で TOML 表示

### 2-3. --config 引数の追加
- `train binary` / `train stack` に `--config` 引数を追加（`--study-id` と排他）
- TOML のセクション（`binary.<task>.<model>` / `stacker.<task>`）からパラメータを読み込み

### 2-4. resolved params の保存
- 各 train コマンド完了後に `runs/<run_id>/resolved_params.toml` を自動保存
- binary, stack, pl, wide-calibrator すべてに対応

## 影響
- ロジック変更なし
- 既存の `--study-id` / `--params-json` フローは後方互換で残存
- 全 22 テスト通過

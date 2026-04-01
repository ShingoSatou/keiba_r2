# Operations

## Environment setup
```bash
uv sync
export V3_ASSET_ROOT=/path/to/v3_assets
export V3_DATABASE_URL=postgresql://...
```

この workspace で現在使っている `V3_ASSET_ROOT` は
[`docs/history/active-asset-root.md`](../history/active-asset-root.md)
の公開用テンプレートを見ます。operator ごとの absolute path は repo に commit しません。

optional:
```bash
uv sync --extra optuna --extra xgboost --extra catboost
```

## Daily flow
1. `db migrate`
2. 必要なら `db rebuild --o1-date <YYYYMMDD>`
3. `features build-base`
4. `features build`
5. 必要なら `features build-te`
6. `tune binary` / `tune stack`
7. `train binary` / `train stack` / `train pl` / `train wide-calibrator`
8. `eval backtest`
9. `eval compare`

日常運用では、同じ repo / worktree の中で複数 `run_id` を作って比較します。  
worktree を増やすのは、設定差ではなくコード系統差を持ちたいときだけです。

## GitHub operating model
- GitHub で管理する主語は `Issue`, `PR`, `docs`, `decision` です。
- `run`, `study`, `feature_profile` の source of truth は引き続き research repo と asset/report 側に置きます。
- branch は `run` ごとではなく、repo-tracked な変更ごとに切ります。
- parameter 変更だけで repo に差分が出ない実験は、PR を作らず `Issue + report link + run_id/study_id` で追います。
- `feature_profile` の差分でも既存コードで表現できる実験条件差なら PR は不要です。
- 新しい特徴量実装、feature registry 変更、train logic 変更、CLI/docs/CI 変更のように repo に差分が出るときは PR を作ります。
- 採用されなかった実験結果は原則 `Issue + report` で閉じます。再利用価値のある運用知見や判断を残す場合だけ docs-only PR を許容します。
- `main` は branch protection の対象にしたい branch です。GitHub plan が許す環境では PR 必須、required checks 必須、force push 禁止、branch deletion 禁止、conversation resolution 必須、approval 必須なしを基本にします。
- 現在の private repo + current GitHub plan では branch protection / ruleset API が 403 になるため、remote enforcement が使えない間は PR-only, squash-only, CI 必須を運用ルールとして維持します。

## Typical commands
### DB rebuild
```bash
uv run python -m keiba_research db rebuild --o1-date 20260215
```

### Feature build
```bash
uv run python -m keiba_research features build-base --feature-profile baseline_v3 --feature-build-id baseline_20260320 --from-date 2016-01-01 --to-date 2025-12-31
uv run python -m keiba_research features build --feature-profile baseline_v3 --feature-build-id baseline_20260320
uv run python -m keiba_research features build-te --feature-profile baseline_v3 --feature-build-id baseline_20260320
```

### Binary tuning and training
```bash
uv run python -m keiba_research tune binary --study-id win_lgbm_baseline --task win --model lgbm --feature-profile baseline_v3 --feature-build-id baseline_20260320
uv run python -m keiba_research train binary --run-id baseline_run --task win --model lgbm --feature-profile baseline_v3 --feature-build-id baseline_20260320 --study-id win_lgbm_baseline
```

### Legacy tuning import
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

`--source-*` はこの repo 内のファイルとは限りません。  
旧 archive や手元保管の asset から、明示的に path を渡します。

### Compare
```bash
uv run python -m keiba_research eval compare --left-run-id baseline_run --right-run-id candidate_run
```

## Common experiment patterns
### 1. baseline 再現
- `baseline_v3` の feature build を作る
- imported seed か fresh study を使って binary / stack / PL を学習する
- `eval backtest` まで行って baseline run を固定する

### 2. feature 追加の検証
- `baseline_v3_plus_<slug>_vN` を新しい `feature_profile` として作る
- feature build を別 ID で出す
- 同じ train / eval 手順で別 run を作る
- `eval compare` で baseline run と比べる

### 3. model と PL の寄与分離
- `baseline`
- `model_only`
- `pl_only`
- `model_plus_pl`

この 4 run に分けると、PL 側の寄与と upstream model 側の寄与を混同しにくくなります。

### 4. legacy seed を使う
- `import legacy-tuning` で study seed を取り込む
- imported study は read-only のまま残す
- imported study 自体を canonical artifact と見なさない
- 比較や継続研究の基準は imported study から作った fresh run に置く
- seed から fresh run を作って baseline とする

## Change-unit decision table
| 変えたいもの | 新しい `feature_profile` | 新しい `feature_build_id` | 新しい `study` | 新しい `run` | 新しい `worktree` |
|---|---|---|---|---|---|
| 特徴量セットを増やす | yes | usually yes | optional | yes | no |
| 同じ特徴量で build 日や期間だけ変える | no | yes | optional | yes | no |
| Optuna を追加で回す | no | no | yes | optional | no |
| selected trial を変えて学習し直す | no | no | no if same study | yes | no |
| PL profile を変える | no | no | no | yes | no |
| upstream binary/stack source を変える | no | no | optional | yes | no |
| feature registry 実装を変える | maybe | maybe | maybe | yes | yes |
| rebuild/parser/schema を変える | maybe | maybe | maybe | yes | yes |

判断の原則:
- 設定差だけなら `run` / `study` / `feature_build_id` で分ける
- コード系統差なら `worktree` を切る

## Working rules
- `run` は immutable にします。
- `study` は mutable にします。
- `feature_profile` と `feature_build_id` は必ず明示します。
- run の比較は同じ repo / worktree 内で複数 run を作って行います。
- worktree を増やすのは、feature registry や train logic そのものを分岐させたいときだけです。

## Docs maintenance
- 振る舞いを変えたら、この repo 内の docs を更新します。
- old archive 由来の長い provenance はこの repo に持ち込みません。
- archive から知見を引いたら、raw log を複製するのではなく `docs/history/` に要点を要約します。
- 用語の意味が曖昧になったら、まず `docs/specs/glossary.md` を更新します。
- CLI の入口や引数が変わったら、`docs/specs/cli-reference.md` も更新します。

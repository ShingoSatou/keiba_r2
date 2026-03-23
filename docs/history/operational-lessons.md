# Operational Lessons

- shared unsuffixed output は混乱の原因になる
  - feature, model, report は named output として分離する
- absolute path を metadata に残すと移行で壊れる
  - repo-native の永続 metadata は `V3_ASSET_ROOT` 相対 path に限定する
  - imported read-only seed だけは provenance 保持のため例外を許容する
- `current default` は比較を曖昧にする
  - `run_id` を明示して compare する
- feature と model と PL を分離して比較しないと寄与が読めない
  - 必要なら `baseline`, `model_only`, `pl_only`, `model_plus_pl` のように run を切る
- worktree を実験単位にすると repo が把握しづらくなる
  - 通常は 1 worktree 内で複数 run を回し、コード分岐が必要な時だけ worktree を増やす

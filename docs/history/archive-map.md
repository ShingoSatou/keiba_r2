# Archive Map

この repo は raw archive を同梱しません。  
過去の provenance を深掘りしたいときは、別保管している旧 repo archive を参照してください。

## Concept map
| old concept | status in this repo | new repo replacement | when to look outside |
|---|---|---|---|
| current operating model | removed as a live doc | `docs/specs/architecture.md`, `docs/specs/operations.md` | 過去の production 運用経緯を確認したいとき |
| global manifest / current default | removed | `run`, `study`, `feature_profile` | 旧 default 切替の履歴を確認したいとき |
| dated run reports | not vendored | `runs/<run_id>/metrics.json`, `bundle.json` | 旧 run の細かな判断経緯を読みたいとき |
| selection / compare playbook | removed | `docs/specs/evaluation-and-comparison.md` | selection-suite 運用の背景を調べたいとき |
| lane / task workflow | removed | `docs/history/open-hypotheses.md`, `decision-log.md` | 未完了研究の生ログを掘りたいとき |
| old compare / sweep scripts | removed from public surface | `python -m keiba_research eval compare` | 旧比較ロジックそのものを再確認したいとき |

## How to use the archive
- archive 内の raw path は、この repo の相対 path ではありません。
- archive の所在はこの repo に固定しません。ローカルや別保管先に置いて構いません。
- archive から得た知見を現在の研究判断に使う場合は、必要箇所だけを `decision-log.md` か `operational-lessons.md` に要約します。

## What stays out of this repo
- dated run report 一式
- selection-test suite の raw 運用ログ
- task / lane workflow の実体
- old compare / sweep 実行結果

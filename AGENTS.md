# AGENTS.md

## 前提
- このリポジトリは、競馬 v3 の研究を単純化して回すための独立した research repo です。

## コアモデル
- source of truth は `run`, `study`, `feature_profile` です。
- `run` は固定された比較単位、`study` は resume 可能な探索単位、`feature_profile` は特徴量契約の識別子です。
- `current default`, global manifest, selection-test suite, task/lane workflow は前提にしません。
- 通常運用は `1 repo / 1 worktree / 複数 run` です。
- worktree を増やすのは、コード系統そのものが分岐する変更だけに限ります。

## 読み方
- 最初に `docs/specs/index.md` を読み、そこから glossary / architecture / data-contract / operations を辿ります。
- その後、必要に応じて feature / training / evaluation の specs を読みます。
- CLI の正確な入口と引数は `docs/specs/cli-reference.md` を見ます。
- `scripts_v3/` は内部実装です。まず `uv run python -m keiba_research ...` を public entrypoint として扱います。
- 歴史的背景が必要なときだけ `docs/history/` を見ます。

## 進め方
- 非自明なタスクでは、実装前に短い計画と検証方針を置きます。
- まず spec で契約を確認し、その後に実装を見ます。
- 影響範囲は最小にします。
- 新しい抽象化や依存関係は、必要性が明確なときだけ導入します。
- 単なる応急処置ではなく、できるだけ根本原因を直します。
- 非自明な変更では、仕上げる前に「もっと単純で一貫したやり方がないか」を一度見直します。
- ただし明白な小修正では、過剰設計を避けます。
- バグ報告や失敗テストを受けたら、まずログ・エラー・失敗箇所を見て、可能な限り自走して直します。
- 前提が崩れた、想定より影響が広い、契約変更が必要になった場合は、惰性で押し切らずに立ち止まって再計画します。
- 安全で可逆な仮定が置けるなら明示して進めます。仮定次第で契約、asset layout、run comparability が変わる場合だけ、ユーザー確認を優先します。

## 差分の切り方
- 特徴量集合を変えるなら、まず新しい `feature_profile` を検討します。
- build 期間や build 日だけが違うなら、新しい `feature_build_id` を使います。
- Optuna の探索を増やすなら、新しい `study` を使います。
- 学習条件、PL profile、upstream source、評価条件を変えるなら、新しい `run` を使います。
- コード系統そのものを分けたいときだけ、新しい `worktree` を使います。

## 変更原則
- 破壊的な変更は避け、必要なら別 `run` / 別 `study` / 別 `feature_profile` として分けます。
- `V3_ASSET_ROOT` 配下のローカル資産は Git 管理しません。
- 振る舞いを変えるときは、コードだけでなくこの repo 内の docs を同じ差分で更新します。

## 完了前確認
- 動作を示せないまま完了扱いにしません。
- 関係する tests, CLI, logs, generated artifacts を確認して、何を検証したかを明示します。
- 必要なら既存契約や変更前の挙動との差分を確認します。
- 振る舞いが変わった場合は、対応する tests を追加または更新します。
- 契約変更や運用判断をしたら、その理由を `docs/history/decision-log.md` に残します。
- 最終確認では、変更した docs 同士が矛盾していないかを見ます。

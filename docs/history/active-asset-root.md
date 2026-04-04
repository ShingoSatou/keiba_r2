# Active Asset Root

この workspace で使う `V3_ASSET_ROOT` は次に固定します。

```bash
export V3_ASSET_ROOT=/home/sato/projects/REPO-v3-research/.local/v3_assets
```

repo root 相対で書くなら次と同値です。

```bash
export V3_ASSET_ROOT="$(git rev-parse --show-toplevel)/.local/v3_assets"
```

主な用途:

- `r2` research の run, study, feature build, compare, execution report 保存先

主な配下:

- `runs/`
- `studies/`
- `data/features/`
- `cache/`

更新ルール:

- current workspace ではこの path を基準にします。
- repo-level code も、`V3_ASSET_ROOT` 未設定時はこの path を default として使います。
- `.local/v3_assets` は Git 管理しません。
- path や用途を変えるときは、このファイルと `README.md` の案内を同じ差分で更新します。

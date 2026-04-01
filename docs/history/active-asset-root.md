# Active Asset Root

この repo では、asset root の置き方と命名だけを公開 docs に残します。
operator ごとの absolute path は public repo に commit しません。

公開用テンプレート:

```bash
export V3_ASSET_ROOT=/path/to/v3_assets_research_r2_wide_YYYYMMDD
```

主な用途:

- `r2 wide calibration / rule comparison` 実験の run, study, feature build, report 保存先

主な配下:

- `runs/`
- `studies/`
- `data/features/`
- `reports/`

更新ルール:

- 公開 repo には machine-specific absolute path を残しません。
- 実際の path は shell, direnv, `.env`, private note などローカルな手段で管理します。
- 命名規則や用途が変わるときだけ、このファイルと `README.md` の案内を同じ差分で更新します。

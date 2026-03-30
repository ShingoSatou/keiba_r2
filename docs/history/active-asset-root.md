# Active Asset Root

この repo では、別セッションでも現在の `V3_ASSET_ROOT` を repo 内から確認できるようにします。

現在の shared asset root:

```bash
export V3_ASSET_ROOT=/home/sato/projects/v3_assets_research_r2_wide_20260325
```

主な用途:

- `r2 wide calibration / rule comparison` 実験の run, study, feature build, report 保存先

主な配下:

- `runs/`
- `studies/`
- `data/features/`
- `reports/`

更新ルール:

- この repo の shared asset root を切り替えたら、このファイルと `README.md` の案内を同じ差分で更新します。

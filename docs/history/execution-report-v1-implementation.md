# Execution Report v1 Implementation

## Summary
- `eval report --run-id <run_id>` を追加する
- `execution_report_summary.json` と `execution_report_detail.json` を `runs/<run_id>/` に生成する
- current workspace の artifact root は `/home/sato/projects/REPO-v3-research/.local/v3_assets` に固定する

## What to read
- run config: `runs/<run_id>/config.toml`
- run bundle: `runs/<run_id>/bundle.json`
- run metrics: `runs/<run_id>/metrics.json`
- resolved params: `runs/<run_id>/resolved_params.toml`
- feature build config: `data/features/<feature_profile>/<feature_build_id>/config.toml`
- layer reports: `artifacts/reports/*`
- layer meta: `artifacts/models/*_bundle_meta*.json`
- optional annotation: `runs/<run_id>/execution_report_annotation.toml`

## Reused as-is
- section presence
- metrics report `summary`
- PL `year_coverage.json`
- wide calibrator `fit` / `holdout_eval`
- backtest `summary` と `meta.config`
- source_run_id / study_id / feature_set

## Derived
- `status`
- auto `title`
- auto `description`
- curated `quality_summary`
- curated `coverage_summary`
- lineage summary
- `setting_sources`
- diagnostics candidate list

## v1 task breakdown
1. low risk
   - fixed workspace asset root を docs / code / `.gitignore` に反映する
2. medium risk
   - `eval report` CLI を追加する
3. medium risk
   - run bundle から summary/detail を組み立てる read-only builder を追加する
4. medium risk
   - annotation override と setting provenance fallback を追加する
5. low risk
   - sample JSON と spec doc を追加する
6. medium risk
   - same-run / partial / lineage / provenance / docs consistency test を追加する

## Deferred
- strict git commit capture for every train/eval step
- target segment canonical owner
- global report index
- new diagnostic slice calculation beyond existing artifacts

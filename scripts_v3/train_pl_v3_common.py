from __future__ import annotations

from pathlib import Path

from scripts_v3.v3_common import append_stem_suffix, resolve_path


def profile_suffix(feature_profile: str) -> str:
    return "" if str(feature_profile) == "stack_default" else f"_{feature_profile}"


def pl_output_paths(
    *,
    pl_feature_profile: str,
    holdout_year: int,
    artifact_suffix: str,
    oof_output: str,
    wide_oof_output: str,
    metrics_output: str,
    model_output: str,
    all_years_model_output: str,
    meta_output: str,
    holdout_output: str,
    year_coverage_output: str,
) -> dict[str, Path]:
    suffix = profile_suffix(str(pl_feature_profile))
    defaults = {
        "oof": append_stem_suffix(f"data/oof/pl_v3_oof{suffix}.parquet", artifact_suffix),
        "wide_oof": append_stem_suffix(f"data/oof/pl_v3_wide_oof{suffix}.parquet", artifact_suffix),
        "metrics": append_stem_suffix(f"data/oof/pl_v3_cv_metrics{suffix}.json", artifact_suffix),
        "model": append_stem_suffix(f"models/pl_v3_recent_window{suffix}.joblib", artifact_suffix),
        "all_years_model": append_stem_suffix(
            f"models/pl_v3_all_years{suffix}.joblib",
            artifact_suffix,
        ),
        "meta": append_stem_suffix(f"models/pl_v3_bundle_meta{suffix}.json", artifact_suffix),
        "holdout": append_stem_suffix(
            f"data/oof/pl_v3_holdout_{int(holdout_year)}_pred{suffix}.parquet",
            artifact_suffix,
        ),
        "year_coverage": append_stem_suffix(
            f"data/oof/v3_pipeline_year_coverage{suffix}.json",
            artifact_suffix,
        ),
    }
    return {
        "oof": resolve_path(oof_output or defaults["oof"]),
        "wide_oof": resolve_path(wide_oof_output or defaults["wide_oof"]),
        "metrics": resolve_path(metrics_output or defaults["metrics"]),
        "model": resolve_path(model_output or defaults["model"]),
        "all_years_model": resolve_path(all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(meta_output or defaults["meta"]),
        "holdout": resolve_path(holdout_output or defaults["holdout"]),
        "year_coverage": resolve_path(year_coverage_output or defaults["year_coverage"]),
    }

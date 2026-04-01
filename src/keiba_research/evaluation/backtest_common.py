from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba_research.common.v3_utils import resolve_path
from keiba_research.evaluation.bankroll import BankrollConfig
from keiba_research.training.cv_policy import DEFAULT_CV_WINDOW_POLICY, make_window_definition


def parse_years(raw: str) -> list[int]:
    return sorted({int(token.strip()) for token in str(raw).split(",") if token.strip()})


def check_overwrite(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists. pass --force to overwrite: {joined}")


def round_or_none(value: Any, digits: int) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def build_backtest_report(
    *,
    args: Any,
    summary: dict[str, Any],
    monthly_rows: list[dict[str, Any]],
    bet_records: list[dict[str, Any]],
    bankroll_config: BankrollConfig,
    input_mode: str,
    p_wide_source: str,
    selected_years: list[int],
    available_years: list[int],
    loaded_input_len: int,
    pair_probs: pd.DataFrame,
    metric_frame: pd.DataFrame,
    quality_metrics_fn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logloss_value, auc_value = quality_metrics_fn(metric_frame)
    if "logloss" not in summary:
        summary["logloss"] = round(float(logloss_value), 4) if logloss_value is not None else None
        summary["auc"] = round(float(auc_value), 4) if auc_value is not None else None

    def extract_single_value(column: str) -> Any:
        for source in (pair_probs, metric_frame):
            if column not in source.columns:
                continue
            values = pd.Series(source[column]).dropna().unique().tolist()
            if not values:
                continue
            if len(values) > 1:
                raise SystemExit(
                    "Mixed cv policy metadata detected in backtest input for "
                    f"'{column}': {values[:5]}"
                )
            return values[0]
        return None

    train_window_years = extract_single_value("train_window_years")
    window_definition = extract_single_value("window_definition")
    cv_window_policy = extract_single_value("cv_window_policy")
    policy_holdout_year = extract_single_value("holdout_year")
    valid_years = (
        sorted(pair_probs["valid_year"].dropna().astype(int).unique().tolist())
        if "valid_year" in pair_probs.columns
        else selected_years
    )
    if cv_window_policy is None and train_window_years is not None:
        cv_window_policy = DEFAULT_CV_WINDOW_POLICY
    if window_definition is None and train_window_years is not None:
        window_definition = make_window_definition(int(train_window_years))

    payload = {
        "summary": summary,
        "monthly": monthly_rows,
        "bets": bet_records,
    }
    meta_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input": {
            "path": str(resolve_path(args.input)),
            "input_mode": input_mode,
            "p_wide_source": p_wide_source,
            "rows": int(loaded_input_len),
            "pair_rows_for_backtest": int(len(pair_probs)),
            "selected_races": int(pair_probs["race_id"].nunique()),
            "available_years_after_holdout_filter": available_years,
            "selected_years": selected_years,
            "input_filter_holdout_year": int(args.holdout_year),
        },
        "cv_policy": {
            "cv_window_policy": cv_window_policy,
            "train_window_years": (
                int(train_window_years) if train_window_years is not None else None
            ),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(policy_holdout_year) if policy_holdout_year is not None else None,
            "window_definition": window_definition,
        },
        "config": {
            "pl": {
                "mc_samples": int(args.mc_samples),
                "top_k": int(args.pl_top_k),
                "seed": int(args.seed),
            },
            "selection": {
                "min_p_wide": float(args.min_p_wide),
                "min_p_wide_stage": str(args.min_p_wide_stage),
                "ev_threshold": float(args.ev_threshold),
                "max_bets_per_race": int(args.max_bets_per_race),
            },
            "bankroll": {
                "bankroll_init_yen": int(bankroll_config.bankroll_init_yen),
                "kelly_fraction_scale": float(bankroll_config.kelly_fraction_scale),
                "race_cap_fraction": float(bankroll_config.race_cap_fraction),
                "daily_cap_fraction": float(bankroll_config.daily_cap_fraction),
                "bet_unit_yen": int(bankroll_config.bet_unit_yen),
                "min_bet_yen": int(bankroll_config.min_bet_yen),
                "max_bet_yen": bankroll_config.max_bet_yen,
            },
        },
        "db_sources": {
            "odds": "core.o3_wide(min_odds_x10, latest data_kbn/announce per race+kumiban)",
            "payout": "core.payout(bet_type=5, selection=kumiban)",
            "race_date": "core.race(race_date)",
            "horse_name": "core.runner(horse_name)",
        },
        "output": {
            "result_path": str(resolve_path(args.output)),
            "meta_path": str(resolve_path(args.meta_output)),
            "n_bets": int(len(bet_records)),
            "n_months": int(len(monthly_rows)),
        },
    }
    return payload, meta_payload

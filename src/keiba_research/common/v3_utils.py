"""v3 共通ユーティリティ.

v3 内で利用するヘルパーを集約する。
CLI引数・出力形式は v3 内で統一。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd

from keiba_research.training.cv_policy import (
    FoldSpec,
    build_fixed_window_year_folds,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# パス / ファイル ユーティリティ
# ---------------------------------------------------------------------------


def resolve_path(path_str: str) -> Path:
    """相対パスなら PROJECT_ROOT 基準で解決する。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """JSON をインデント付きで書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_files(paths: list[Path]) -> str:
    """複数ファイルの SHA-256 ダイジェストを返す。"""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_dotenv_value(name: str, *, dotenv_path: Path | None = None) -> str | None:
    """`.env` から単一キーを読み出す。環境変数が無いときの補助用途。"""
    path = dotenv_path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != str(name):
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None


def resolve_database_url(
    cli_value: str | None = None,
    *,
    primary_env: str = "V3_DATABASE_URL",
    fallback_env: str | None = None,
) -> str:
    """v3 系スクリプト用の DB URL を解決する。"""
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()

    env_names = [primary_env]
    if fallback_env and str(fallback_env).strip():
        env_names.append(str(fallback_env).strip())

    for env_name in env_names:
        value = os.getenv(env_name)
        if value and str(value).strip():
            return str(value).strip()

    for env_name in env_names:
        value = _load_dotenv_value(env_name)
        if value and str(value).strip():
            return str(value).strip()

    raise SystemExit(
        f"database url is not configured. Pass --database-url or set {primary_env} in env/.env."
    )


def artifact_suffix_fragment(value: str | None) -> str:
    """artifact suffix を `_suffix` 形式へ正規化する。"""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    normalized = raw.replace(" ", "_")
    return normalized if normalized.startswith("_") else f"_{normalized}"


def append_stem_suffix(path_str: str, suffix: str | None) -> str:
    """拡張子の前に suffix を付与したパス文字列を返す。"""
    fragment = artifact_suffix_fragment(suffix)
    if not fragment:
        return str(path_str)
    path = Path(str(path_str))
    if not path.suffix:
        return str(path.with_name(f"{path.name}{fragment}"))
    return str(path.with_name(f"{path.stem}{fragment}{path.suffix}"))


# ---------------------------------------------------------------------------
# Fixed-length sliding 年次 CV
# ---------------------------------------------------------------------------


def build_rolling_year_folds(
    years: list[int],
    *,
    train_window_years: int,
    holdout_year: int,
) -> list[FoldSpec]:
    """Backward-compatible alias for fixed-length sliding yearly folds."""
    return build_fixed_window_year_folds(
        years,
        window_years=int(train_window_years),
        holdout_year=int(holdout_year),
    )


def assert_fold_integrity(train_df: pd.DataFrame, valid_df: pd.DataFrame, valid_year: int) -> None:
    """train/valid 間の時系列リーク・レースID重複を検出する。"""
    train_year_max = int(train_df["year"].max())
    if train_year_max >= valid_year:
        raise ValueError(
            f"Temporal leakage detected: train max year={train_year_max}, valid year={valid_year}"
        )
    overlap = set(train_df["race_id"].unique()) & set(valid_df["race_id"].unique())
    if overlap:
        raise ValueError(f"Race leakage detected across train/valid: {len(overlap)} races")


# ---------------------------------------------------------------------------
# ソート検証
# ---------------------------------------------------------------------------


def assert_sorted(df: pd.DataFrame) -> None:
    """race_id asc / horse_no asc のソート順を検証する。"""
    sorted_df = df.sort_values(["race_id", "horse_no"], kind="mergesort")
    left = df[["race_id", "horse_no"]].reset_index(drop=True)
    right = sorted_df[["race_id", "horse_no"]].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Output sort violation: expected race_id asc / horse_no asc.")


# ---------------------------------------------------------------------------
# レース日時構築
# ---------------------------------------------------------------------------


def _time_to_seconds(value: object) -> int:
    """時刻を秒に変換する。パース不能な場合は 23:59:00 を返す。"""
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if value is None or pd.isna(value):
        return 23 * 3600 + 59 * 60
    text = str(value).strip()
    if not text:
        return 23 * 3600 + 59 * 60
    parts = text.split(":")
    if len(parts) >= 2:
        try:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) >= 3 else 0
            return hour * 3600 + minute * 60 + second
        except ValueError:
            return 23 * 3600 + 59 * 60
    return 23 * 3600 + 59 * 60


def build_race_datetime(
    race_date_series: pd.Series,
    start_time_series: pd.Series,
) -> pd.Series:
    """race_date + start_time から race_datetime を構築する。"""
    race_date = pd.to_datetime(race_date_series, errors="coerce")
    start_seconds = start_time_series.map(_time_to_seconds)
    return race_date + pd.to_timedelta(start_seconds, unit="s")


# ---------------------------------------------------------------------------
# 組番
# ---------------------------------------------------------------------------


def kumiban_from_horse_nos(horse_no_1: int, horse_no_2: int) -> str:
    """2頭の馬番からワイド組番文字列を生成する。"""
    left, right = sorted((int(horse_no_1), int(horse_no_2)))
    return f"{left:02d}{right:02d}"


__all__ = [
    "PROJECT_ROOT",
    # パス / ファイル
    "resolve_path",
    "save_json",
    "hash_files",
    # CV (local)
    "build_rolling_year_folds",
    "assert_fold_integrity",
    # ソート
    "assert_sorted",
    # 日時
    "build_race_datetime",
    # 組番
    "kumiban_from_horse_nos",
]

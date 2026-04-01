#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.v3_common import resolve_database_url  # noqa: E402


@dataclass(frozen=True)
class MigrationFile:
    version: str
    path: Path
    checksum: str


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _discover_migrations(migrations_dir: Path) -> list[MigrationFile]:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        raise SystemExit(f"migrations dir not found: {migrations_dir}")

    migration_files: list[MigrationFile] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        sql = path.read_text(encoding="utf-8", errors="replace")
        migration_files.append(
            MigrationFile(version=path.name, path=path, checksum=_sha256_hex(sql))
        )
    return migration_files


def _ensure_schema_migrations_table(conn: psycopg.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            version TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def _fetch_applied(conn: psycopg.Connection, table_name: str) -> dict[str, dict]:
    _ensure_schema_migrations_table(conn, table_name)
    rows = conn.execute(
        f"SELECT version, checksum, applied_at FROM {table_name} ORDER BY version"
    ).fetchall()
    return {r["version"]: {"checksum": r["checksum"], "applied_at": r["applied_at"]} for r in rows}


def _print_list(migrations: list[MigrationFile], applied: dict[str, dict]) -> None:
    for migration in migrations:
        status = "pending"
        note = ""
        if migration.version in applied:
            status = "applied"
            if migration.checksum != applied[migration.version]["checksum"]:
                note = " (checksum mismatch)"
        print(f"{status:7} {migration.version}{note}")


def _apply_one(
    conn: psycopg.Connection,
    migration: MigrationFile,
    *,
    baseline: bool,
    table_name: str,
) -> None:
    if not baseline:
        sql = migration.path.read_text(encoding="utf-8", errors="replace")
        conn.execute(sql)

    conn.execute(
        f"""
        INSERT INTO {table_name} (version, checksum)
        VALUES (%s, %s)
        ON CONFLICT (version) DO UPDATE
        SET checksum = EXCLUDED.checksum
        """,
        (migration.version, migration.checksum),
    )


def run_migrate(
    *,
    database_url: str = "",
    log_level: str = "INFO",
) -> int:
    import logging

    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    database_url_resolved = resolve_database_url(database_url)
    migrations_dir = Path("migrations_v3")
    migrations = _discover_migrations(migrations_dir)

    if not migrations:
        print(f"No migrations found in {migrations_dir}")
        return 0

    migrations_table = "public.schema_migrations_v3"
    with psycopg.connect(database_url_resolved, autocommit=True, row_factory=dict_row) as conn:
        applied = _fetch_applied(conn, migrations_table)

        target = None
        for migration in migrations:
            if migration.version in applied:
                if target and migration.version == target:
                    break
                continue

            mode = "APPLY"
            print(f"[{mode}] {migration.version}")
            _apply_one(
                conn,
                migration,
                baseline=False,
                table_name=migrations_table,
            )

            if target and migration.version == target:
                break

    return 0

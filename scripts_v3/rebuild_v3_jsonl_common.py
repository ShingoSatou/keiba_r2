from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from keiba_research.db.database import Database
from keiba_research.rebuild.parsers import (
    DMRecord,
    HorseMasterRecord,
    JockeyRecord,
    O3HeaderRecord,
    O3WideRecord,
    OddsTimeSeriesRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TMRecord,
    TrainerRecord,
)
from scripts_v3.v3_common import save_json

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "data"
DEFAULT_COMMIT_INTERVAL = 5000
DEFAULT_TARGET_CONDITION_CODES = (10, 16, 999)
DEFAULT_TARGET_RACE_TYPE_CODES = (13, 14)
SUPPORTED_DATASPECS = ("RACE", "DIFF", "MING", "0B41", "0B13", "0B17")

HISTORY_SCOPE_TABLES = [
    "race",
    "horse",
    "jockey",
    "trainer",
    "runner",
    "result",
]
TARGET_SCOPE_TABLES = [
    "payout",
    "o1_header",
    "o1_win",
    "o1_place",
    "o3_header",
    "o3_wide",
    "mining_dm",
    "mining_tm",
    "rt_mining_dm",
    "rt_mining_tm",
]
TABLE_COPY_ORDER = [*HISTORY_SCOPE_TABLES, *TARGET_SCOPE_TABLES]
TABLE_SCOPE = {
    **{table_name: "history" for table_name in HISTORY_SCOPE_TABLES},
    **{table_name: "target" for table_name in TARGET_SCOPE_TABLES},
}

RACE_AGE_CONDITION_COLUMNS = [
    "condition_code_2yo",
    "condition_code_3yo",
    "condition_code_4yo",
    "condition_code_5up",
    "condition_code_min_age_raw",
]

RA_YEAR_START = 11
RA_YEAR_LEN = 4
RA_MONTHDAY_START = 15
RA_MONTHDAY_LEN = 4
RA_TRACK_CODE_START = 19
RA_TRACK_CODE_LEN = 2
RA_RACE_NO_START = 25
RA_RACE_NO_LEN = 2
RA_COND_CODE_2YO_START = 622
RA_COND_CODE_2YO_LEN = 3
RA_COND_CODE_3YO_START = 625
RA_COND_CODE_3YO_LEN = 3
RA_COND_CODE_4YO_START = 628
RA_COND_CODE_4YO_LEN = 3
RA_COND_CODE_5UP_START = 631
RA_COND_CODE_5UP_LEN = 3
RA_COND_CODE_MIN_AGE_START = 634
RA_COND_CODE_MIN_AGE_LEN = 3


def _slice_byte_int(payload: object, start: int, length: int) -> int | None:
    if payload is None:
        return None
    try:
        data = str(payload).encode("cp932", errors="ignore")
        text = data[start : start + length].decode("cp932").strip().strip("\u3000")
    except UnicodeDecodeError:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def parse_ra_age_condition_payload(payload: object) -> dict[str, int | None] | None:
    year = _slice_byte_int(payload, RA_YEAR_START, RA_YEAR_LEN)
    monthday = _slice_byte_int(payload, RA_MONTHDAY_START, RA_MONTHDAY_LEN)
    track_code = _slice_byte_int(payload, RA_TRACK_CODE_START, RA_TRACK_CODE_LEN)
    race_no = _slice_byte_int(payload, RA_RACE_NO_START, RA_RACE_NO_LEN)
    if not year or not monthday or not track_code or not race_no:
        return None

    month = monthday // 100
    day = monthday % 100
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None

    race_id = (year * 10000 + month * 100 + day) * 10000 + track_code * 100 + race_no
    return {
        "race_id": race_id,
        "condition_code_2yo": _slice_byte_int(
            payload,
            RA_COND_CODE_2YO_START,
            RA_COND_CODE_2YO_LEN,
        ),
        "condition_code_3yo": _slice_byte_int(
            payload,
            RA_COND_CODE_3YO_START,
            RA_COND_CODE_3YO_LEN,
        ),
        "condition_code_4yo": _slice_byte_int(
            payload,
            RA_COND_CODE_4YO_START,
            RA_COND_CODE_4YO_LEN,
        ),
        "condition_code_5up": _slice_byte_int(
            payload,
            RA_COND_CODE_5UP_START,
            RA_COND_CODE_5UP_LEN,
        ),
        "condition_code_min_age_raw": _slice_byte_int(
            payload,
            RA_COND_CODE_MIN_AGE_START,
            RA_COND_CODE_MIN_AGE_LEN,
        ),
    }


def _parse_condition_codes(raw: str) -> list[int]:
    codes = sorted({int(token.strip()) for token in str(raw).split(",") if token.strip()})
    if not codes:
        raise ValueError("--condition-codes must not be empty")
    return codes


def race_id_track_code(race_id: int) -> int:
    return (int(race_id) // 100) % 100


def is_central_track(track_code: int) -> bool:
    return 1 <= int(track_code) <= 10


def is_central_race(race_id: int) -> bool:
    return is_central_track(race_id_track_code(race_id))


def race_id_date(race_id: int) -> date | None:
    race_id_int = int(race_id)
    if race_id_int <= 0:
        return None
    date_int = race_id_int // 10000
    year = date_int // 10000
    month = (date_int // 100) % 100
    day = date_int % 100
    try:
        return date(year, month, day)
    except ValueError:
        return None


def race_date_in_range(
    race_date: date | None,
    *,
    from_date: date,
    to_date: date | None,
) -> bool:
    if race_date is None:
        return False
    if race_date < from_date:
        return False
    if to_date is not None and race_date > to_date:
        return False
    return True


def race_id_in_history_scope(
    race_id: int,
    *,
    from_date: date,
    to_date: date | None,
) -> bool:
    if not is_central_race(race_id):
        return False
    return race_date_in_range(race_id_date(race_id), from_date=from_date, to_date=to_date)


def load_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _canonical_input_files(
    input_dir: Path, *, o1_consolidated_date: str | None = None
) -> list[Path]:
    files: list[Path] = []
    for prefix in ("DIFF", "MING", "RACE", "0B13", "0B17"):
        files.extend(sorted(input_dir.glob(f"{prefix}_*.jsonl")))

    consolidated_o1 = sorted(input_dir.glob("0B41_ALL*.jsonl"))
    dated_o1 = [
        path for path in consolidated_o1 if re.fullmatch(r"0B41_ALL_\d{8}", path.stem) is not None
    ]
    requested_date = str(o1_consolidated_date or "").strip()
    if requested_date:
        requested_path = input_dir / f"0B41_ALL_{requested_date}.jsonl"
        if not requested_path.exists():
            raise FileNotFoundError(
                f"requested consolidated O1 snapshot not found: {requested_path}"
            )
        files.append(requested_path)
    elif dated_o1:
        files.append(max(dated_o1, key=lambda path: path.stem.rsplit("_", 1)[-1]))
    elif consolidated_o1:
        files.append(max(consolidated_o1, key=lambda path: path.stat().st_mtime))
    else:
        files.extend(sorted(input_dir.glob("0B41_*.jsonl")))

    return files


def collect_input_files(
    input_pattern: str | None,
    input_dir: Path | None,
    *,
    o1_consolidated_date: str | None = None,
) -> list[Path]:
    files: list[Path] = []
    if input_pattern:
        input_path = Path(input_pattern)
        if "*" in input_pattern:
            parent = input_path.parent if input_path.parent.exists() else Path(".")
            files.extend(sorted(parent.glob(input_path.name)))
        else:
            files.append(input_path)
    if input_dir:
        files.extend(_canonical_input_files(input_dir, o1_consolidated_date=o1_consolidated_date))

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            continue
        dataspec = resolved.stem.split("_")[0]
        if dataspec not in SUPPORTED_DATASPECS:
            continue
        seen.add(resolved)
        unique_files.append(resolved)
    return unique_files


def ensure_race_stub(db: Database, race_id: int, cache: set[int] | None = None) -> None:
    if not race_id:
        return
    race_id_int = int(race_id)
    if race_id_int <= 0:
        return
    if cache is not None and race_id_int in cache:
        return

    race_no = race_id_int % 100
    track_code = (race_id_int // 100) % 100
    race_dt = race_id_date(race_id_int)
    if race_dt is None:
        return

    db.execute(
        """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface, distance_m
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, 0, 0
        )
        ON CONFLICT (race_id) DO NOTHING
        """,
        {
            "race_id": race_id_int,
            "race_date": race_dt,
            "track_code": track_code,
            "race_no": race_no,
        },
    )
    if cache is not None:
        cache.add(race_id_int)


def upsert_race(
    db: Database,
    race: RaceRecord,
    *,
    age_conditions: dict[str, int | None] | None = None,
) -> None:
    age = age_conditions or {}
    db.execute(
        """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface,
            distance_m, going, weather, class_code, field_size, start_time,
            turn_dir, course_inout, grade_code, race_type_code,
            weight_type_code, condition_code_min_age,
            condition_code_2yo, condition_code_3yo, condition_code_4yo,
            condition_code_5up, condition_code_min_age_raw
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, %(surface)s,
            %(distance_m)s, %(going)s, %(weather)s,
            %(class_code)s, %(field_size)s, %(start_time)s,
            %(turn_dir)s, %(course_inout)s, %(grade_code)s, %(race_type_code)s,
            %(weight_type_code)s, %(condition_code_min_age)s,
            %(condition_code_2yo)s, %(condition_code_3yo)s, %(condition_code_4yo)s,
            %(condition_code_5up)s, %(condition_code_min_age_raw)s
        )
        ON CONFLICT (race_id) DO UPDATE SET
            surface = CASE
                WHEN EXCLUDED.surface > 0
                    AND (
                        core.race.surface = 0
                        OR core.race.surface IS NULL
                        OR core.race.distance_m = 0
                        OR core.race.distance_m IS NULL
                        OR core.race.start_time IS NULL
                    )
                THEN EXCLUDED.surface
                ELSE core.race.surface
            END,
            distance_m = CASE
                WHEN EXCLUDED.distance_m > 0
                    AND (
                        core.race.distance_m = 0
                        OR core.race.distance_m IS NULL
                        OR core.race.distance_m < 800
                    )
                THEN EXCLUDED.distance_m
                ELSE core.race.distance_m
            END,
            going = COALESCE(EXCLUDED.going, NULLIF(core.race.going, 0)),
            weather = COALESCE(EXCLUDED.weather, NULLIF(core.race.weather, 0)),
            class_code = COALESCE(
                NULLIF(EXCLUDED.class_code, 0),
                NULLIF(core.race.class_code, 0),
                core.race.class_code,
                0
            ),
            field_size = COALESCE(EXCLUDED.field_size, NULLIF(core.race.field_size, 0)),
            start_time = COALESCE(EXCLUDED.start_time, core.race.start_time),
            turn_dir = COALESCE(core.race.turn_dir, EXCLUDED.turn_dir),
            course_inout = COALESCE(
                NULLIF(core.race.course_inout, 0),
                NULLIF(EXCLUDED.course_inout, 0),
                core.race.course_inout
            ),
            grade_code = COALESCE(core.race.grade_code, EXCLUDED.grade_code),
            race_type_code = COALESCE(
                NULLIF(EXCLUDED.race_type_code, 0),
                NULLIF(core.race.race_type_code, 0),
                core.race.race_type_code
            ),
            weight_type_code = COALESCE(
                NULLIF(EXCLUDED.weight_type_code, 0),
                NULLIF(core.race.weight_type_code, 0),
                core.race.weight_type_code
            ),
            condition_code_min_age = COALESCE(
                NULLIF(EXCLUDED.condition_code_min_age, 0),
                NULLIF(core.race.condition_code_min_age, 0),
                core.race.condition_code_min_age
            ),
            condition_code_2yo = COALESCE(
                EXCLUDED.condition_code_2yo,
                core.race.condition_code_2yo
            ),
            condition_code_3yo = COALESCE(
                EXCLUDED.condition_code_3yo,
                core.race.condition_code_3yo
            ),
            condition_code_4yo = COALESCE(
                EXCLUDED.condition_code_4yo,
                core.race.condition_code_4yo
            ),
            condition_code_5up = COALESCE(
                EXCLUDED.condition_code_5up,
                core.race.condition_code_5up
            ),
            condition_code_min_age_raw = COALESCE(
                EXCLUDED.condition_code_min_age_raw,
                core.race.condition_code_min_age_raw
            ),
            updated_at = now()
        """,
        {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "track_code": race.track_code,
            "race_no": race.race_no,
            "surface": race.surface,
            "distance_m": race.distance_m,
            "going": race.going,
            "weather": race.weather,
            "class_code": race.class_code,
            "field_size": race.field_size,
            "start_time": race.start_time,
            "turn_dir": race.turn_dir,
            "course_inout": race.course_inout,
            "grade_code": race.grade_code,
            "race_type_code": race.race_type_code,
            "weight_type_code": race.weight_type_code,
            "condition_code_min_age": race.condition_code_min_age,
            "condition_code_2yo": age.get("condition_code_2yo"),
            "condition_code_3yo": age.get("condition_code_3yo"),
            "condition_code_4yo": age.get("condition_code_4yo"),
            "condition_code_5up": age.get("condition_code_5up"),
            "condition_code_min_age_raw": age.get("condition_code_min_age_raw"),
        },
    )


def prepare_master_data_cache(db: Database) -> tuple[set[int], set[int]]:
    jockeys = {int(row["jockey_id"]) for row in db.fetch_all("SELECT jockey_id FROM core.jockey")}
    trainers = {
        int(row["trainer_id"]) for row in db.fetch_all("SELECT trainer_id FROM core.trainer")
    }
    return jockeys, trainers


def upsert_runner(
    db: Database,
    runner: RunnerRecord,
    master_jockeys: set[int],
    master_trainers: set[int],
    race_stub_cache: set[int],
) -> None:
    safe_jockey_id = runner.jockey_id if runner.jockey_id in master_jockeys else None
    safe_trainer_id = runner.trainer_id if runner.trainer_id in master_trainers else None

    if runner.data_kubun in ("A", "B2"):
        safe_jockey_id = None
        safe_trainer_id = None

    ensure_race_stub(db, runner.race_id, cache=race_stub_cache)

    db.execute(
        """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = COALESCE(EXCLUDED.horse_name, core.horse.horse_name),
            updated_at = now()
        """,
        {"horse_id": runner.horse_id, "horse_name": runner.horse_name},
    )
    db.execute(
        """
        INSERT INTO core.runner (
            race_id, horse_id, horse_no, gate, jockey_id,
            trainer_id, carried_weight, body_weight, body_weight_diff,
            sex, data_kubun, trainer_code_raw, trainer_name_abbr,
            jockey_code_raw, jockey_name_abbr
        ) VALUES (
            %(race_id)s,
            COALESCE(
                (
                    SELECT r.horse_id
                    FROM core.runner AS r
                    WHERE r.race_id = %(race_id)s
                      AND r.horse_no = %(horse_no)s
                ),
                %(horse_id)s
            ),
            %(horse_no)s, %(gate)s, %(jockey_id)s,
            %(trainer_id)s, %(carried_weight)s, %(body_weight)s, %(body_weight_diff)s,
            %(sex)s, %(data_kubun)s, %(trainer_code_raw)s, %(trainer_name_abbr)s,
            %(jockey_code_raw)s, %(jockey_name_abbr)s
        )
        ON CONFLICT (race_id, horse_id) DO UPDATE SET
            horse_no = EXCLUDED.horse_no,
            gate = EXCLUDED.gate,
            jockey_id = EXCLUDED.jockey_id,
            trainer_id = EXCLUDED.trainer_id,
            carried_weight = EXCLUDED.carried_weight,
            body_weight = EXCLUDED.body_weight,
            body_weight_diff = EXCLUDED.body_weight_diff,
            sex = COALESCE(EXCLUDED.sex, core.runner.sex),
            data_kubun = EXCLUDED.data_kubun,
            trainer_code_raw = EXCLUDED.trainer_code_raw,
            trainer_name_abbr = EXCLUDED.trainer_name_abbr,
            jockey_code_raw = EXCLUDED.jockey_code_raw,
            jockey_name_abbr = EXCLUDED.jockey_name_abbr,
            updated_at = now()
        """,
        {
            "race_id": runner.race_id,
            "horse_id": runner.horse_id,
            "horse_no": runner.horse_no,
            "gate": runner.gate,
            "jockey_id": safe_jockey_id,
            "trainer_id": safe_trainer_id,
            "carried_weight": runner.carried_weight,
            "body_weight": runner.body_weight,
            "body_weight_diff": runner.body_weight_diff,
            "sex": runner.sex,
            "data_kubun": runner.data_kubun,
            "trainer_code_raw": runner.trainer_code_raw,
            "trainer_name_abbr": runner.trainer_name_abbr,
            "jockey_code_raw": runner.jockey_code_raw,
            "jockey_name_abbr": runner.jockey_name_abbr,
        },
    )

    if runner.finish_pos and runner.finish_pos > 0:
        db.execute(
            """
            INSERT INTO core.result (
                race_id, horse_id, finish_pos, time_sec,
                margin, final3f_sec, corner1_pos, corner2_pos, corner3_pos, corner4_pos
            ) VALUES (
                %(race_id)s,
                COALESCE(
                    (
                        SELECT r.horse_id
                        FROM core.runner AS r
                        WHERE r.race_id = %(race_id)s
                          AND r.horse_no = %(horse_no)s
                    ),
                    %(horse_id)s
                ),
                %(finish_pos)s, %(time_sec)s,
                %(margin)s, %(final3f_sec)s, %(corner1_pos)s, %(corner2_pos)s,
                %(corner3_pos)s, %(corner4_pos)s
            )
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                finish_pos = EXCLUDED.finish_pos,
                time_sec = EXCLUDED.time_sec,
                margin = EXCLUDED.margin,
                final3f_sec = EXCLUDED.final3f_sec,
                corner1_pos = EXCLUDED.corner1_pos,
                corner2_pos = EXCLUDED.corner2_pos,
                corner3_pos = EXCLUDED.corner3_pos,
                corner4_pos = EXCLUDED.corner4_pos,
                updated_at = now()
            """,
            {
                "race_id": runner.race_id,
                "horse_id": runner.horse_id,
                "horse_no": runner.horse_no,
                "finish_pos": runner.finish_pos,
                "time_sec": runner.time_sec,
                "margin": runner.margin,
                "final3f_sec": runner.final3f_sec,
                "corner1_pos": runner.corner1_pos,
                "corner2_pos": runner.corner2_pos,
                "corner3_pos": runner.corner3_pos,
                "corner4_pos": runner.corner4_pos,
            },
        )


def upsert_payout_records(
    db: Database, payouts: list[PayoutRecord], race_stub_cache: set[int]
) -> int:
    if not payouts:
        return 0
    race_id = payouts[0].race_id
    ensure_race_stub(db, race_id, cache=race_stub_cache)
    rows = [
        {
            "race_id": payout.race_id,
            "bet_type": payout.bet_type,
            "selection": payout.selection,
            "payout_yen": payout.payout_yen,
            "popularity": payout.popularity,
        }
        for payout in payouts
    ]
    db.execute_many(
        """
        INSERT INTO core.payout (race_id, bet_type, selection, payout_yen, popularity)
        VALUES (%(race_id)s, %(bet_type)s, %(selection)s, %(payout_yen)s, %(popularity)s)
        ON CONFLICT (race_id, bet_type, selection) DO UPDATE SET
            payout_yen = EXCLUDED.payout_yen,
            popularity = EXCLUDED.popularity
        """,
        rows,
    )
    return len(rows)


def upsert_horse_master(db: Database, horse: HorseMasterRecord) -> None:
    db.execute(
        """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = COALESCE(EXCLUDED.horse_name, core.horse.horse_name),
            updated_at = now()
        """,
        {"horse_id": horse.horse_id, "horse_name": horse.horse_name},
    )


def upsert_jockey(db: Database, jockey: JockeyRecord) -> None:
    db.execute(
        """
        INSERT INTO core.jockey (jockey_id, jockey_name)
        VALUES (%(jockey_id)s, %(jockey_name)s)
        ON CONFLICT (jockey_id) DO UPDATE SET
            jockey_name = COALESCE(EXCLUDED.jockey_name, core.jockey.jockey_name),
            updated_at = now()
        """,
        {"jockey_id": jockey.jockey_id, "jockey_name": jockey.jockey_name},
    )


def upsert_trainer(db: Database, trainer: TrainerRecord) -> None:
    db.execute(
        """
        INSERT INTO core.trainer (trainer_id, trainer_name)
        VALUES (%(trainer_id)s, %(trainer_name)s)
        ON CONFLICT (trainer_id) DO UPDATE SET
            trainer_name = COALESCE(EXCLUDED.trainer_name, core.trainer.trainer_name),
            updated_at = now()
        """,
        {"trainer_id": trainer.trainer_id, "trainer_name": trainer.trainer_name},
    )


def upsert_o1_timeseries_bulk(
    db: Database,
    records: list[OddsTimeSeriesRecord],
    race_stub_cache: set[int],
) -> tuple[int, int]:
    if not records:
        return (0, 0)

    header_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record.race_id, record.data_kbn, record.announce_mmddhhmi)
        header_map[key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "win_pool_total_100yen": record.win_pool_total_100yen,
            "data_create_ymd": record.data_create_ymd,
            "sale_flag_place": record.sale_flag_place,
            "place_pay_key": record.place_pay_key,
            "place_pool_total_100yen": record.place_pool_total_100yen,
        }

    for row in header_map.values():
        ensure_race_stub(db, int(row["race_id"]), cache=race_stub_cache)

    db.execute_many(
        """
        INSERT INTO core.o1_header (
            race_id, data_kbn, announce_mmddhhmi, win_pool_total_100yen,
            data_create_ymd, sale_flag_place, place_pay_key, place_pool_total_100yen
        )
        VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(win_pool_total_100yen)s,
            %(data_create_ymd)s, %(sale_flag_place)s, %(place_pay_key)s, %(place_pool_total_100yen)s
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            win_pool_total_100yen = EXCLUDED.win_pool_total_100yen,
            data_create_ymd = COALESCE(EXCLUDED.data_create_ymd, core.o1_header.data_create_ymd),
            sale_flag_place = EXCLUDED.sale_flag_place,
            place_pay_key = EXCLUDED.place_pay_key,
            place_pool_total_100yen = EXCLUDED.place_pool_total_100yen
        """,
        list(header_map.values()),
    )

    win_map: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    place_map: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for record in records:
        if record.has_win_block:
            win_key = (
                record.race_id,
                record.data_kbn,
                record.announce_mmddhhmi,
                record.horse_no,
            )
            win_map[win_key] = {
                "race_id": record.race_id,
                "data_kbn": record.data_kbn,
                "announce_mmddhhmi": record.announce_mmddhhmi,
                "horse_no": record.horse_no,
                "win_odds_x10": record.win_odds_x10,
                "win_popularity": record.win_popularity,
            }
        if record.has_place_block:
            place_key = (
                record.race_id,
                record.data_kbn,
                record.announce_mmddhhmi,
                record.horse_no,
            )
            place_map[place_key] = {
                "race_id": record.race_id,
                "data_kbn": record.data_kbn,
                "announce_mmddhhmi": record.announce_mmddhhmi,
                "horse_no": record.horse_no,
                "min_odds_x10": record.place_min_odds_x10,
                "max_odds_x10": record.place_max_odds_x10,
                "place_popularity": record.place_popularity,
            }

    win_rows = list(win_map.values())
    if win_rows:
        db.execute(
            """
            INSERT INTO core.o1_win (
                race_id, data_kbn, announce_mmddhhmi,
                horse_no, win_odds_x10, win_popularity
            )
            SELECT
                x.race_id, x.data_kbn, x.announce_mmddhhmi,
                x.horse_no, x.win_odds_x10, x.win_popularity
            FROM jsonb_to_recordset(%(rows_json)s::jsonb) AS x(
                race_id bigint,
                data_kbn integer,
                announce_mmddhhmi text,
                horse_no integer,
                win_odds_x10 integer,
                win_popularity integer
            )
            ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, horse_no)
            DO UPDATE SET
                win_odds_x10 = EXCLUDED.win_odds_x10,
                win_popularity = EXCLUDED.win_popularity
            """,
            {"rows_json": json.dumps(win_rows, ensure_ascii=False)},
        )

    place_rows = list(place_map.values())
    if place_rows:
        db.execute(
            """
            INSERT INTO core.o1_place (
                race_id, data_kbn, announce_mmddhhmi,
                horse_no, min_odds_x10, max_odds_x10, place_popularity
            )
            SELECT
                x.race_id, x.data_kbn, x.announce_mmddhhmi,
                x.horse_no, x.min_odds_x10, x.max_odds_x10, x.place_popularity
            FROM jsonb_to_recordset(%(rows_json)s::jsonb) AS x(
                race_id bigint,
                data_kbn integer,
                announce_mmddhhmi text,
                horse_no integer,
                min_odds_x10 integer,
                max_odds_x10 integer,
                place_popularity integer
            )
            ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, horse_no)
            DO UPDATE SET
                min_odds_x10 = EXCLUDED.min_odds_x10,
                max_odds_x10 = EXCLUDED.max_odds_x10,
                place_popularity = EXCLUDED.place_popularity
            """,
            {"rows_json": json.dumps(place_rows, ensure_ascii=False)},
        )

    return (len(win_rows), len(place_rows))


def upsert_o3_wide_records_bulk(
    db: Database, records: list[O3WideRecord], race_stub_cache: set[int]
) -> int:
    if not records:
        return 0

    header_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record.race_id, record.data_kbn, record.announce_mmddhhmi)
        header_map[key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "wide_pool_total_100yen": record.wide_pool_total_100yen,
            "starters": record.starters,
            "sale_flag_wide": record.sale_flag_wide,
            "data_create_ymd": record.data_create_ymd,
        }
    for row in header_map.values():
        ensure_race_stub(db, int(row["race_id"]), cache=race_stub_cache)

    db.execute_many(
        """
        INSERT INTO core.o3_header (
            race_id, data_kbn, announce_mmddhhmi,
            wide_pool_total_100yen, starters, sale_flag_wide, data_create_ymd
        ) VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
            %(wide_pool_total_100yen)s, %(starters)s, %(sale_flag_wide)s, %(data_create_ymd)s
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            wide_pool_total_100yen = EXCLUDED.wide_pool_total_100yen,
            starters = EXCLUDED.starters,
            sale_flag_wide = EXCLUDED.sale_flag_wide,
            data_create_ymd = EXCLUDED.data_create_ymd
        """,
        list(header_map.values()),
    )

    detail_rows = [
        {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "kumiban": record.kumiban,
            "min_odds_x10": record.min_odds_x10,
            "max_odds_x10": record.max_odds_x10,
            "popularity": record.popularity,
        }
        for record in records
    ]
    db.execute(
        """
        INSERT INTO core.o3_wide (
            race_id, data_kbn, announce_mmddhhmi,
            kumiban, min_odds_x10, max_odds_x10, popularity
        )
        SELECT
            x.race_id, x.data_kbn, x.announce_mmddhhmi,
            x.kumiban, x.min_odds_x10, x.max_odds_x10, x.popularity
        FROM jsonb_to_recordset(%(rows_json)s::jsonb) AS x(
            race_id bigint,
            data_kbn integer,
            announce_mmddhhmi text,
            kumiban text,
            min_odds_x10 integer,
            max_odds_x10 integer,
            popularity integer
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, kumiban) DO UPDATE SET
            min_odds_x10 = EXCLUDED.min_odds_x10,
            max_odds_x10 = EXCLUDED.max_odds_x10,
            popularity = EXCLUDED.popularity
        """,
        {"rows_json": json.dumps(detail_rows, ensure_ascii=False)},
    )
    return len(detail_rows)


def _mining_params(record: DMRecord | TMRecord) -> dict[str, Any]:
    return {
        "race_id": record.race_id,
        "horse_no": record.horse_no,
        "data_kbn": record.data_kbn,
        "data_create_ymd": getattr(record, "data_create_ymd", "00000000"),
        "data_create_hm": getattr(record, "data_create_hm", "0000"),
        "dm_time_x10": getattr(record, "dm_time_x10", None),
        "dm_rank": getattr(record, "dm_rank", None),
        "tm_score": getattr(record, "tm_score", None),
        "tm_rank": getattr(record, "tm_rank", None),
        "payload_raw": record.payload_raw,
    }


def _ensure_race_stubs_for_mining_records(
    db: Database, records: list[DMRecord | TMRecord], race_stub_cache: set[int] | None
) -> None:
    race_ids = {int(record.race_id) for record in records if getattr(record, "race_id", 0)}
    for race_id in race_ids:
        ensure_race_stub(db, race_id, cache=race_stub_cache)


def insert_mining_records_batch(
    db: Database,
    rec_id: str,
    records: list[DMRecord | TMRecord],
    race_stub_cache: set[int] | None = None,
) -> int:
    if not records:
        return 0

    _ensure_race_stubs_for_mining_records(db, records, race_stub_cache)
    if rec_id == "DM":
        db.execute_many(
            """
            INSERT INTO core.mining_dm (
                race_id, horse_no, data_kbn, dm_time_x10, dm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(horse_no)s, %(data_kbn)s,
                %(dm_time_x10)s, %(dm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                dm_time_x10 = EXCLUDED.dm_time_x10,
                dm_rank = EXCLUDED.dm_rank,
                payload_raw = EXCLUDED.payload_raw
            """,
            [_mining_params(record) for record in records],
        )
    else:
        db.execute_many(
            """
            INSERT INTO core.mining_tm (
                race_id, horse_no, data_kbn, tm_score, tm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(horse_no)s, %(data_kbn)s, %(tm_score)s, %(tm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                tm_score = EXCLUDED.tm_score,
                tm_rank = EXCLUDED.tm_rank,
                payload_raw = EXCLUDED.payload_raw
            """,
            [_mining_params(record) for record in records],
        )
    return len(records)


def insert_rt_mining_records_batch(
    db: Database,
    rec_id: str,
    records: list[DMRecord | TMRecord],
    race_stub_cache: set[int] | None = None,
) -> int:
    if not records:
        return 0

    _ensure_race_stubs_for_mining_records(db, records, race_stub_cache)
    if rec_id == "DM":
        db.execute_many(
            """
            INSERT INTO core.rt_mining_dm (
                race_id, data_kbn, data_create_ymd, data_create_hm,
                horse_no, dm_time_x10, dm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(data_kbn)s, %(data_create_ymd)s, %(data_create_hm)s,
                %(horse_no)s, %(dm_time_x10)s, %(dm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no) DO UPDATE SET
                dm_time_x10 = EXCLUDED.dm_time_x10,
                dm_rank = EXCLUDED.dm_rank,
                payload_raw = EXCLUDED.payload_raw,
                ingested_at = now()
            """,
            [_mining_params(record) for record in records],
        )
    else:
        db.execute_many(
            """
            INSERT INTO core.rt_mining_tm (
                race_id, data_kbn, data_create_ymd, data_create_hm,
                horse_no, tm_score, tm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(data_kbn)s, %(data_create_ymd)s, %(data_create_hm)s,
                %(horse_no)s, %(tm_score)s, %(tm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no) DO UPDATE SET
                tm_score = EXCLUDED.tm_score,
                tm_rank = EXCLUDED.tm_rank,
                payload_raw = EXCLUDED.payload_raw,
                ingested_at = now()
            """,
            [_mining_params(record) for record in records],
        )
    return len(records)


def delete_rt_mining_records(
    db: Database, rec_id: str, race_id: int, data_create_ymd: str, data_create_hm: str
) -> int:
    sql = (
        """
        DELETE FROM core.rt_mining_dm
        WHERE race_id = %(race_id)s
          AND data_create_ymd = %(data_create_ymd)s
          AND data_create_hm = %(data_create_hm)s
        """
        if rec_id == "DM"
        else """
        DELETE FROM core.rt_mining_tm
        WHERE race_id = %(race_id)s
          AND data_create_ymd = %(data_create_ymd)s
          AND data_create_hm = %(data_create_hm)s
        """
    )
    db.execute(
        sql,
        {
            "race_id": race_id,
            "data_create_ymd": data_create_ymd,
            "data_create_hm": data_create_hm,
        },
    )
    return 1


def _extract_rt_mining_header(payload: str) -> dict[str, Any] | None:
    try:
        b_payload = payload.encode("cp932")
    except UnicodeEncodeError:
        b_payload = payload.encode("cp932", errors="replace")

    if len(b_payload) < 31:
        b_payload = b_payload.ljust(31, b" ")

    rec_type = b_payload[0:2].decode("cp932", errors="ignore").strip()
    if rec_type not in {"DM", "TM"}:
        return None

    data_kbn_raw = b_payload[2:3].decode("cp932", errors="ignore").strip()
    try:
        data_kbn = int(data_kbn_raw)
    except (TypeError, ValueError):
        data_kbn = -1

    data_create_ymd = b_payload[3:11].decode("cp932", errors="ignore").strip() or "00000000"
    race_key = b_payload[11:27].decode("cp932", errors="ignore")
    data_create_hm = b_payload[27:31].decode("cp932", errors="ignore").strip() or "0000"

    race_id = 0
    if len(race_key) >= 16:
        try:
            year = int(race_key[0:4])
            month = int(race_key[4:6])
            day = int(race_key[6:8])
            track = int(race_key[8:10])
            race_no = int(race_key[14:16])
            date_int = year * 10000 + month * 100 + day
            race_id = date_int * 10000 + track * 100 + race_no
        except ValueError:
            race_id = 0

    return {
        "rec_type": rec_type,
        "race_id": race_id,
        "data_kbn": data_kbn,
        "data_create_ymd": data_create_ymd,
        "data_create_hm": data_create_hm,
    }


def process_file(
    db: Database,
    file_path: Path,
    *,
    from_date: date,
    to_date: date | None,
    commit_interval: int = DEFAULT_COMMIT_INTERVAL,
) -> dict[str, int]:
    dataspec = file_path.stem.split("_")[0]
    is_rt_mining = dataspec in {"0B13", "0B17"}

    stats = {
        "race": 0,
        "runner": 0,
        "result": 0,
        "payout": 0,
        "o1": 0,
        "o1_place": 0,
        "o3": 0,
        "mining": 0,
        "rt_mining_delete": 0,
        "horse": 0,
        "jockey": 0,
        "trainer": 0,
        "skipped_out_of_scope": 0,
        "errors": 0,
    }

    master_jockeys, master_trainers = prepare_master_data_cache(db)
    race_stub_cache: set[int] = set()
    o1_batch: list[OddsTimeSeriesRecord] = []
    o3_batch: list[O3WideRecord] = []
    dm_batch: list[DMRecord] = []
    tm_batch: list[TMRecord] = []

    def flush_all(force: bool = False) -> None:
        nonlocal o1_batch, o3_batch, dm_batch, tm_batch
        if o1_batch and (force or len(o1_batch) >= 50000):
            win_rows, place_rows = upsert_o1_timeseries_bulk(db, o1_batch, race_stub_cache)
            stats["o1"] += win_rows
            stats["o1_place"] += place_rows
            o1_batch = []

        if o3_batch and (force or len(o3_batch) >= 30000):
            stats["o3"] += upsert_o3_wide_records_bulk(db, o3_batch, race_stub_cache)
            o3_batch = []

        if dm_batch and (force or len(dm_batch) >= 10000):
            if is_rt_mining:
                stats["mining"] += insert_rt_mining_records_batch(
                    db, "DM", dm_batch, race_stub_cache
                )
            else:
                stats["mining"] += insert_mining_records_batch(db, "DM", dm_batch, race_stub_cache)
            dm_batch = []

        if tm_batch and (force or len(tm_batch) >= 10000):
            if is_rt_mining:
                stats["mining"] += insert_rt_mining_records_batch(
                    db, "TM", tm_batch, race_stub_cache
                )
            else:
                stats["mining"] += insert_mining_records_batch(db, "TM", tm_batch, race_stub_cache)
            tm_batch = []

    for index, record in enumerate(load_jsonl(file_path), start=1):
        rec_id = str(record.get("rec_id", "")).strip()
        payload = str(record.get("payload", ""))
        try:
            if rec_id == "RA":
                race = RaceRecord.parse(payload)
                if not (
                    is_central_track(race.track_code)
                    and race_date_in_range(race.race_date, from_date=from_date, to_date=to_date)
                ):
                    stats["skipped_out_of_scope"] += 1
                else:
                    age_payload = parse_ra_age_condition_payload(payload) or {}
                    upsert_race(db, race, age_conditions=age_payload)
                    stats["race"] += 1

            elif rec_id == "SE":
                runner = RunnerRecord.parse(payload)
                if not race_id_in_history_scope(
                    runner.race_id,
                    from_date=from_date,
                    to_date=to_date,
                ):
                    stats["skipped_out_of_scope"] += 1
                else:
                    upsert_runner(db, runner, master_jockeys, master_trainers, race_stub_cache)
                    stats["runner"] += 1
                    if runner.finish_pos and runner.finish_pos > 0:
                        stats["result"] += 1

            elif rec_id == "HR":
                payouts = PayoutRecord.parse(payload)
                if not payouts:
                    continue
                if not race_id_in_history_scope(
                    payouts[0].race_id,
                    from_date=from_date,
                    to_date=to_date,
                ):
                    stats["skipped_out_of_scope"] += 1
                else:
                    stats["payout"] += upsert_payout_records(db, payouts, race_stub_cache)

            elif rec_id == "O1":
                records = OddsTimeSeriesRecord.parse(payload)
                if not records:
                    continue
                if not race_id_in_history_scope(
                    records[0].race_id,
                    from_date=from_date,
                    to_date=to_date,
                ):
                    stats["skipped_out_of_scope"] += 1
                else:
                    o1_batch.extend(records)

            elif rec_id == "O3":
                records = O3WideRecord.parse(payload)
                if records:
                    race_id = records[0].race_id
                    if not race_id_in_history_scope(race_id, from_date=from_date, to_date=to_date):
                        stats["skipped_out_of_scope"] += 1
                    else:
                        o3_batch.extend(records)
                else:
                    header = O3HeaderRecord.parse(payload)
                    if not race_id_in_history_scope(
                        header.race_id,
                        from_date=from_date,
                        to_date=to_date,
                    ):
                        stats["skipped_out_of_scope"] += 1
                    else:
                        ensure_race_stub(db, header.race_id, race_stub_cache)
                        db.execute(
                            """
                            INSERT INTO core.o3_header (
                                race_id, data_kbn, announce_mmddhhmi,
                                wide_pool_total_100yen, starters, sale_flag_wide, data_create_ymd
                            ) VALUES (
                                %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
                                %(wide_pool_total_100yen)s, %(starters)s, %(sale_flag_wide)s,
                                %(data_create_ymd)s
                            )
                            ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
                                wide_pool_total_100yen = EXCLUDED.wide_pool_total_100yen,
                                starters = EXCLUDED.starters,
                                sale_flag_wide = EXCLUDED.sale_flag_wide,
                                data_create_ymd = EXCLUDED.data_create_ymd
                            """,
                            {
                                "race_id": header.race_id,
                                "data_kbn": header.data_kbn,
                                "announce_mmddhhmi": header.announce_mmddhhmi,
                                "wide_pool_total_100yen": header.wide_pool_total_100yen,
                                "starters": header.starters,
                                "sale_flag_wide": header.sale_flag_wide,
                                "data_create_ymd": header.data_create_ymd,
                            },
                        )

            elif rec_id == "DM":
                if is_rt_mining:
                    header = _extract_rt_mining_header(payload)
                    if header and header["data_kbn"] == 0 and header["race_id"] > 0:
                        if not race_id_in_history_scope(
                            int(header["race_id"]), from_date=from_date, to_date=to_date
                        ):
                            stats["skipped_out_of_scope"] += 1
                        else:
                            flush_all(force=True)
                            stats["rt_mining_delete"] += delete_rt_mining_records(
                                db,
                                rec_id="DM",
                                race_id=int(header["race_id"]),
                                data_create_ymd=str(header["data_create_ymd"]),
                                data_create_hm=str(header["data_create_hm"]),
                            )
                    else:
                        records = DMRecord.parse(payload)
                        if not records:
                            continue
                        if not race_id_in_history_scope(
                            records[0].race_id, from_date=from_date, to_date=to_date
                        ):
                            stats["skipped_out_of_scope"] += 1
                        else:
                            dm_batch.extend(records)
                else:
                    records = DMRecord.parse(payload)
                    if not records:
                        continue
                    if not race_id_in_history_scope(
                        records[0].race_id, from_date=from_date, to_date=to_date
                    ):
                        stats["skipped_out_of_scope"] += 1
                    else:
                        dm_batch.extend(records)

            elif rec_id == "TM":
                if is_rt_mining:
                    header = _extract_rt_mining_header(payload)
                    if header and header["data_kbn"] == 0 and header["race_id"] > 0:
                        if not race_id_in_history_scope(
                            int(header["race_id"]), from_date=from_date, to_date=to_date
                        ):
                            stats["skipped_out_of_scope"] += 1
                        else:
                            flush_all(force=True)
                            stats["rt_mining_delete"] += delete_rt_mining_records(
                                db,
                                rec_id="TM",
                                race_id=int(header["race_id"]),
                                data_create_ymd=str(header["data_create_ymd"]),
                                data_create_hm=str(header["data_create_hm"]),
                            )
                    else:
                        records = TMRecord.parse(payload)
                        if not records:
                            continue
                        if not race_id_in_history_scope(
                            records[0].race_id, from_date=from_date, to_date=to_date
                        ):
                            stats["skipped_out_of_scope"] += 1
                        else:
                            tm_batch.extend(records)
                else:
                    records = TMRecord.parse(payload)
                    if not records:
                        continue
                    if not race_id_in_history_scope(
                        records[0].race_id, from_date=from_date, to_date=to_date
                    ):
                        stats["skipped_out_of_scope"] += 1
                    else:
                        tm_batch.extend(records)

            elif rec_id == "UM":
                horse = HorseMasterRecord.parse(payload)
                upsert_horse_master(db, horse)
                stats["horse"] += 1

            elif rec_id == "KS":
                jockey = JockeyRecord.parse(payload)
                upsert_jockey(db, jockey)
                if jockey.jockey_id > 0:
                    master_jockeys.add(jockey.jockey_id)
                stats["jockey"] += 1

            elif rec_id == "CH":
                trainer = TrainerRecord.parse(payload)
                upsert_trainer(db, trainer)
                if trainer.trainer_id > 0:
                    master_trainers.add(trainer.trainer_id)
                stats["trainer"] += 1

            flush_all(force=False)

            if index % commit_interval == 0:
                flush_all(force=True)
                db.connect().commit()
                race_stub_cache = set()
                logger.info(
                    "%s: processed=%s skipped_out_of_scope=%s errors=%s",
                    file_path.name,
                    f"{index:,}",
                    f"{stats['skipped_out_of_scope']:,}",
                    f"{stats['errors']:,}",
                )

        except Exception:
            db.connect().rollback()
            o1_batch = []
            o3_batch = []
            dm_batch = []
            tm_batch = []
            race_stub_cache = set()
            stats["errors"] += 1
            if stats["errors"] <= 20:
                logger.exception("process error file=%s rec_id=%s", file_path.name, rec_id)

    flush_all(force=True)
    db.connect().commit()
    return stats


def _target_race_where(*, from_date: date, to_date: date | None) -> tuple[str, dict[str, Any]]:
    date_clause = ""
    params: dict[str, Any] = {
        "from_date": from_date,
        "race_type_codes": list(DEFAULT_TARGET_RACE_TYPE_CODES),
        "condition_codes": list(DEFAULT_TARGET_CONDITION_CODES),
    }
    if to_date is not None:
        date_clause = "AND race_date <= %(to_date)s"
        params["to_date"] = to_date
    sql = f"""
    race_date >= %(from_date)s
      {date_clause}
      AND track_code BETWEEN 1 AND 10
      AND surface = 2
      AND race_type_code = ANY(%(race_type_codes)s)
      AND condition_code_min_age = ANY(%(condition_codes)s)
    """
    return sql, params


def truncate_target(db: Database) -> None:
    table_list = ", ".join(f"core.{table_name}" for table_name in TABLE_COPY_ORDER)
    db.execute(f"TRUNCATE TABLE {table_list} CASCADE")
    db.connect().commit()


def _count_table(db: Database, table_name: str) -> int:
    row = db.fetch_one(f"SELECT COUNT(*) AS count FROM core.{table_name}")
    return int(row["count"]) if row else 0


def prune_target_only_tables(
    db: Database,
    *,
    from_date: date,
    to_date: date | None,
    condition_codes: list[int],
) -> dict[str, dict[str, int]]:
    where_sql, base_params = _target_race_where(from_date=from_date, to_date=to_date)
    params = dict(base_params)
    params["condition_codes"] = condition_codes
    before = {table_name: _count_table(db, table_name) for table_name in TARGET_SCOPE_TABLES}

    db.execute(
        f"""
        DELETE FROM core.o1_header
        WHERE race_id NOT IN (
            SELECT race_id
            FROM core.race
            WHERE {where_sql}
        )
        """,
        params,
    )
    db.execute(
        f"""
        DELETE FROM core.o3_header
        WHERE race_id NOT IN (
            SELECT race_id
            FROM core.race
            WHERE {where_sql}
        )
        """,
        params,
    )
    for table_name in ("payout", "mining_dm", "mining_tm", "rt_mining_dm", "rt_mining_tm"):
        db.execute(
            f"""
            DELETE FROM core.{table_name}
            WHERE race_id NOT IN (
                SELECT race_id
                FROM core.race
                WHERE {where_sql}
            )
            """,
            params,
        )
    db.connect().commit()

    after = {table_name: _count_table(db, table_name) for table_name in TARGET_SCOPE_TABLES}
    return {
        table_name: {
            "before": before[table_name],
            "after": after[table_name],
            "deleted": max(before[table_name] - after[table_name], 0),
        }
        for table_name in TARGET_SCOPE_TABLES
    }


def _range_query(db: Database, where_sql: str, params: dict[str, Any]) -> dict[str, Any]:
    row = db.fetch_one(
        f"""
        SELECT
            MIN(race_date) AS min_race_date,
            MAX(race_date) AS max_race_date,
            COUNT(*) AS races
        FROM core.race
        WHERE {where_sql}
        """,
        params,
    )
    return {
        "min_race_date": None if row["min_race_date"] is None else str(row["min_race_date"]),
        "max_race_date": None if row["max_race_date"] is None else str(row["max_race_date"]),
        "races": int(row["races"]),
    }


def build_rebuild_summary(
    db: Database,
    *,
    input_files: list[Path],
    from_date: date,
    to_date: date | None,
    condition_codes: list[int],
    process_stats: dict[str, int],
    prune_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    counts = {table_name: _count_table(db, table_name) for table_name in TABLE_COPY_ORDER}
    history_where = "race_date >= %(from_date)s"
    history_params: dict[str, Any] = {"from_date": from_date}
    if to_date is not None:
        history_where += " AND race_date <= %(to_date)s"
        history_params["to_date"] = to_date

    target_where, target_params = _target_race_where(from_date=from_date, to_date=to_date)
    target_params["condition_codes"] = condition_codes
    master_row = db.fetch_one(
        """
        SELECT
            COUNT(DISTINCT horse_id) AS horses,
            COUNT(DISTINCT jockey_id) AS jockeys,
            COUNT(DISTINCT trainer_id) AS trainers
        FROM core.runner
        """
    )
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_kind": "jsonl",
        "input_files": [str(path) for path in input_files],
        "history_scope": {
            "from_date": str(from_date),
            "to_date": None if to_date is None else str(to_date),
            "track_code": "01-10",
        },
        "target_segment_filter": {
            "from_date": str(from_date),
            "to_date": None if to_date is None else str(to_date),
            "track_code": "01-10",
            "surface": 2,
            "race_type_code": list(DEFAULT_TARGET_RACE_TYPE_CODES),
            "condition_code_min_age": condition_codes,
        },
        "scope_by_table": TABLE_SCOPE,
        "processed_stats": process_stats,
        "prune_stats": prune_stats,
        "selected_history_race_ids": _range_query(db, history_where, history_params)["races"],
        "selected_target_race_ids": _range_query(db, target_where, target_params)["races"],
        "target_summary": {
            "counts": counts,
            "history_date_range": _range_query(db, history_where, history_params),
            "target_segment_date_range": _range_query(db, target_where, target_params),
            "masters": {
                "horses": int(master_row["horses"]) if master_row else 0,
                "jockeys": int(master_row["jockeys"]) if master_row else 0,
                "trainers": int(master_row["trainers"]) if master_row else 0,
            },
        },
    }


def rebuild_v3_database(
    *,
    database_url: str,
    input_pattern: str | None,
    input_dir: Path | None,
    o1_consolidated_date: str | None = None,
    from_date: date,
    to_date: date | None,
    condition_codes: list[int],
    summary_output: Path,
    commit_interval: int = DEFAULT_COMMIT_INTERVAL,
) -> dict[str, Any]:
    input_files = collect_input_files(
        input_pattern,
        input_dir,
        o1_consolidated_date=o1_consolidated_date,
    )
    if not input_files:
        raise SystemExit("No supported JSONL files were found for v3 rebuild.")

    totals: dict[str, int] = defaultdict(int)
    with Database(connection_string=database_url) as db:
        truncate_target(db)
        for file_path in input_files:
            logger.info("rebuild input: %s", file_path)
            stats = process_file(
                db,
                file_path,
                from_date=from_date,
                to_date=to_date,
                commit_interval=commit_interval,
            )
            for key, value in stats.items():
                totals[key] += value
        prune_stats = prune_target_only_tables(
            db,
            from_date=from_date,
            to_date=to_date,
            condition_codes=condition_codes,
        )
        summary = build_rebuild_summary(
            db,
            input_files=input_files,
            from_date=from_date,
            to_date=to_date,
            condition_codes=condition_codes,
            process_stats=dict(totals),
            prune_stats=prune_stats,
        )

    save_json(summary_output, summary)
    logger.info("wrote %s", summary_output)
    return summary


__all__ = [
    "DEFAULT_COMMIT_INTERVAL",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_TARGET_CONDITION_CODES",
    "DEFAULT_TARGET_RACE_TYPE_CODES",
    "HISTORY_SCOPE_TABLES",
    "RA_COND_CODE_2YO_LEN",
    "RA_COND_CODE_2YO_START",
    "RA_COND_CODE_3YO_LEN",
    "RA_COND_CODE_3YO_START",
    "RA_COND_CODE_4YO_LEN",
    "RA_COND_CODE_4YO_START",
    "RA_COND_CODE_5UP_LEN",
    "RA_COND_CODE_5UP_START",
    "RA_COND_CODE_MIN_AGE_LEN",
    "RA_COND_CODE_MIN_AGE_START",
    "RA_MONTHDAY_LEN",
    "RA_MONTHDAY_START",
    "RA_RACE_NO_LEN",
    "RA_RACE_NO_START",
    "RA_TRACK_CODE_LEN",
    "RA_TRACK_CODE_START",
    "RA_YEAR_LEN",
    "RA_YEAR_START",
    "RACE_AGE_CONDITION_COLUMNS",
    "TABLE_COPY_ORDER",
    "TABLE_SCOPE",
    "TARGET_SCOPE_TABLES",
    "_parse_condition_codes",
    "build_rebuild_summary",
    "collect_input_files",
    "parse_ra_age_condition_payload",
    "race_id_date",
    "race_id_in_history_scope",
    "rebuild_v3_database",
]

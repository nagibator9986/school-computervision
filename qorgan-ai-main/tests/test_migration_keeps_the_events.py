"""Migration 0009 rebuilds `cameras` three times. Measured: EIGHT tables survive it.

`migrations/env.py::_suspend_foreign_keys` says, in writing, that a batch rebuild
cascade-deletes every row that references the rebuilt table and reports success. Migration
0009 is exactly that shape -- it adds `school_id` to four root tables, makes each NOT NULL,
then swaps two unique constraints, and every one of those is a copy / move / **DROP** /
rename on SQLite. A single `create_foreign_key` is enough to force the rebuild.

`test_migrations.py::test_a_migration_leaves_no_broken_foreign_key_behind` already checks
one half: run the chain and the rows are still there. **That half alone cannot tell a
working guard from a migration that was never dangerous.** Both look like a passing test,
and only one of them is a reason to trust the next migration somebody writes.

So this measures the difference. Two databases, populated identically, migrated by the same
revision -- one through `migrations/`, the other through a copy of `migrations/` with the
single line `_suspend_foreign_keys(connection)` deleted. Every watched table must survive
in the first and be EMPTY in the second.

**WHY EVERY TABLE AND NOT JUST `events`.** This test measured only `events` until it was
run across the whole schema, and the answer was worse than assumed: without the guard,
**eight** tables lose every row -- not the one or two the previous two incidents found
(`cameras -> events`, then `events -> notifications`). That matters for what this test is
FOR. A future migration that spared `events` but not `notifications` would have passed the
one-table version green, and the school's alert history would have gone silently, with exit
code 0. The blast radius is the thing being characterised, so the whole blast radius is
what gets asserted -- and a table added to `WATCHED` later cannot quietly go unchecked,
because `_populate` and the before-count assertion both fail loudly if it is not filled.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from qorgan.db.engine import build_engine
from tests.conftest import REPO_ROOT

GUARD_CALL = "        _suspend_foreign_keys(connection)\n"

# Every table hanging off a root table that 0009 rebuilds. Each gets exactly one row, so
# "survived" is 1 and "cascade-deleted" is 0 for all of them.
WATCHED = (
    "events",
    "notifications",
    "canteen_sessions",
    "recognition_attempts",
    "person_photos",
    "face_embeddings",
    "lessons",
    "lesson_tracks",
)

# The four tables 0009 REBUILDS. They must survive both runs: if one of these were lost,
# the measurement below would be describing a broken migration rather than a cascade.
REBUILT = ("cameras", "persons", "users", "meal_windows")

# One row per table, as revision 0008 shapes them. Literal SQL rather than the ORM, because
# the ORM describes TODAY's schema and this has to describe 0008's.
ROWS: tuple[str, ...] = (
    """insert into cameras (name, display_name, location, camera_type, role, rtsp_host,
        rtsp_port, priority, enabled, created_at, updated_at) values
        ('hall_left','Холл слева','Холл','BULLYING','MAIN_HALL','10.0.0.1',554,1,1,
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into persons (external_id, external_id_source, full_name, person_type,
        class_name, is_active, created_at, updated_at) values
        ('7','ROSTER','Ученик Седьмой','PUPIL','5-А',1,
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into users (username, password_hash, role, is_active, created_at, updated_at)
        values ('director','x','DIRECTOR',1,'2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into meal_windows (kind, name, starts_at, ends_at, enabled, created_at,
        updated_at) values ('BREAKFAST','Завтрак','08:00','09:00',1,
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into events (camera_id, event_type, occurred_at, confidence,
        candidate_probability, validation_score, skeleton_confirmed, severity, summary_text,
        track_ids, status, created_at, updated_at) values
        (1,'BULLYING','2026-03-04 09:12:30',0.93,0.9,0.8,1,'ALERT','Агрессия','3,7','NEW',
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into notifications (event_id, channel, status, attempts, created_at,
        updated_at) values (1,'TELEGRAM','SENT',1,'2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into person_photos (person_id, path, sha256, created_at, updated_at)
        values (1,'photos/7.jpg','abc','2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into face_embeddings (person_id, photo_id, model_name, model_version, dim,
        normalized, vector, created_at, updated_at) values
        (1,1,'buffalo_l','1',512,1,X'00','2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into canteen_sessions (person_id, meal_window_id, entry_camera_id, state,
        opened_at, created_at, updated_at) values (1,1,1,'OPEN',
        '2026-03-04 09:00:00','2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into recognition_attempts (camera_id, occurred_at, accepted, reason)
        values (1,'2026-03-04 09:10:00',1,'MATCH')""",
    """insert into lessons (camera_id, state, started_at, min_presence_seconds,
        ambiguous_observations, unclaimed_observations, dropped_tracks, resumed_count,
        created_at, updated_at) values (1,'OPEN','2026-03-04 09:00:00',1.0,0,0,0,0,
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
    """insert into lesson_tracks (lesson_id, track_id, first_seen_at, last_seen_at,
        observed_seconds, observations, settled, hand_raises, stands, away_seconds,
        brief_excursions, created_at, updated_at) values
        (1,3,'2026-03-04 09:00:00','2026-03-04 09:00:00',10.0,5,1,0,0,0.0,0,
        '2026-03-04 09:00:00','2026-03-04 09:00:00')""",
)


def _config(script_location: Path, url: str) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(script_location))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _migrations_without_the_guard(tmp_path: Path) -> Path:
    """The same migrations, with the one protecting line removed. Nothing else changes."""
    target = tmp_path / "migrations_unguarded"
    shutil.copytree(
        REPO_ROOT / "migrations", target, ignore=shutil.ignore_patterns("__pycache__")
    )
    env = target / "env.py"
    original = env.read_text(encoding="utf-8")
    without = original.replace(GUARD_CALL, "")
    assert without != original, (
        "could not find the guard call in migrations/env.py to remove, so the control "
        "database would be migrated WITH the protection and this test would compare two "
        f"identical runs and pass. Looked for exactly: {GUARD_CALL!r}"
    )
    env.write_text(without, encoding="utf-8")
    return target


def _populate(url: str) -> None:
    """One row in every watched table, at revision 0008.

    Every statement is required. A silently skipped insert would leave its table reading
    zero both before and after, which is indistinguishable from a row that was
    cascade-deleted -- so the test would report the guard working on a table it never
    filled. The before-count assertion in `_survivors_after_0009` is the backstop.
    """
    engine = build_engine(url)
    with engine.begin() as connection:
        for statement in ROWS:
            connection.execute(text(statement))
    engine.dispose()


def _counts(url: str) -> dict[str, int]:
    engine = build_engine(url)
    with engine.connect() as connection:
        counts = {
            # Justified: `table` comes from the two module constants above and can never
            # be anything else. SQLite takes no bind parameter in a FROM clause.
            table: int(
                connection.execute(
                    text(f"select count(*) from {table}")  # noqa: S608
                ).scalar_one()
            )
            for table in WATCHED + REBUILT
        }
    engine.dispose()
    return counts


def _survivors_after_0009(script_location: Path, url: str) -> dict[str, int]:
    """Populate at 0008, run 0009, and count what is left of every watched table.

    **The two revision numbers move together and must stay adjacent.** This migration was
    numbered 0008 on its own branch and renumbered to 0009 at merge, because
    `feat/psychologist-cabinet` had already taken 0008. The database is therefore brought
    to the revision IMMEDIATELY BEFORE the one under test and no earlier: stopping at 0007
    would run the referral migration inside the measured window, and its own rebuild of
    `events` would be credited to -- or blamed on -- this one.
    """
    config = _config(script_location, url)
    command.upgrade(config, "0008")
    _populate(url)

    before = _counts(url)
    unfilled = sorted(table for table, count in before.items() if count != 1)
    assert not unfilled, (
        f"{unfilled} did not hold exactly one row BEFORE the migration, so nothing this "
        "test says about them afterwards means anything. Fix `ROWS`."
    )

    command.upgrade(config, "0009")
    return _counts(url)


@pytest.fixture(scope="module")
def measured(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, int], dict[str, int]]:
    """Both runs, once. Two full migration chains is the expensive part of this file."""
    tmp = tmp_path_factory.mktemp("migration-0009")
    guarded_url = f"sqlite+pysqlite:///{(tmp / 'guarded.sqlite3').as_posix()}"
    unguarded_url = f"sqlite+pysqlite:///{(tmp / 'unguarded.sqlite3').as_posix()}"

    guarded = _survivors_after_0009(REPO_ROOT / "migrations", guarded_url)
    unguarded = _survivors_after_0009(_migrations_without_the_guard(tmp), unguarded_url)
    return guarded, unguarded


@pytest.mark.parametrize("table", REBUILT)
def test_a_rebuilt_table_survives_its_own_rebuild(
    measured: tuple[dict[str, int], dict[str, int]], table: str
) -> None:
    """The control. If a root table were lost, the counts below would mean nothing."""
    guarded, unguarded = measured
    assert guarded[table] == 1, f"{table} did not survive its own rebuild WITH the guard"
    assert unguarded[table] == 1, (
        f"{table} did not survive its own rebuild without the guard either, so the "
        "cascade measurements below describe a broken migration rather than a cascade."
    )


@pytest.mark.parametrize("table", WATCHED)
def test_the_school_keeps_its_records_only_because_the_guard_is_there(
    measured: tuple[dict[str, int], dict[str, int]], table: str
) -> None:
    """One parameter per table hanging off a rebuilt root. Both halves are the point."""
    guarded, unguarded = measured

    assert guarded[table] == 1, (
        f"migration 0009 destroyed `{table}` WITH the protection in place. "
        "`migrations/env.py::_suspend_foreign_keys` is not doing what it says, or this "
        "migration found a way round it. On `events` or `notifications` that is the "
        "school's incident and alert history gone, with exit code 0."
    )
    assert unguarded[table] == 0, (
        f"`{table}` survived migration 0009 with `_suspend_foreign_keys` REMOVED, so it "
        "was never in danger and the assertion above proves nothing about the guard for "
        "it. Either it stopped referencing a rebuilt table with ON DELETE CASCADE, or "
        "0009 stopped rebuilding that table. Find out which before trusting the other "
        "half: a protection nobody has seen fail is a protection nobody has tested."
    )

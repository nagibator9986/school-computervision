"""The migration chain must build the schema the models describe, from nothing.

The system has to run end-to-end against an empty database with zero pupils.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from qorgan.db.engine import build_engine
from qorgan.db.models import Base
from qorgan.settings import Settings
from tests.conftest import REPO_ROOT

EXPECTED_TABLES = {
    "alembic_version",
    "app_settings",
    "cameras",
    "canteen_sessions",
    # The offline classroom analyses (migration 0010). Listed one by one rather than matched
    # by prefix: this set is a manifest a person signed, and a `startswith` would let a ninth
    # table appear without anyone noticing it had.
    "classvision_attestations",
    "classvision_frames",
    "classvision_lessons",
    "classvision_place_lessons",
    "classvision_places",
    "classvision_readings",
    "classvision_runs",
    "classvision_teacher_lessons",
    "events",
    "face_embeddings",
    "lesson_tracks",
    "lessons",
    "meal_windows",
    "mode_logs",
    "notifications",
    "person_photos",
    "persons",
    "psychologist_notes",
    "recognition_attempts",
    "schools",
    "users",
    "worker_heartbeats",
}


def _alembic_config(settings: Settings) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    return config


def test_upgrade_head_builds_the_whole_schema_from_empty(settings: Settings) -> None:
    command.upgrade(_alembic_config(settings), "head")

    engine = build_engine(settings.database_url)
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
    engine.dispose()


def test_the_migrated_database_starts_with_zero_pupils(settings: Settings) -> None:
    command.upgrade(_alembic_config(settings), "head")

    engine = build_engine(settings.database_url)
    with engine.connect() as connection:
        from sqlalchemy import text

        assert connection.execute(text("select count(*) from persons")).scalar() == 0
        assert connection.execute(text("select count(*) from canteen_sessions")).scalar() == 0
    engine.dispose()


def test_the_migration_matches_the_models(settings: Settings) -> None:
    """If someone edits a model and forgets the migration, this fails.

    The legacy answer to that problem was ALTER TABLE statements run from application
    code on every boot, which is how its schema truth ended up smeared across a .sql
    file and a Python function that disagreed with each other (audit M-18).
    """
    command.upgrade(_alembic_config(settings), "head")

    engine = build_engine(settings.database_url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        drift = compare_metadata(context, Base.metadata)
    engine.dispose()

    assert not drift, f"models and migrations disagree: {drift}"


def test_downgrade_removes_everything(settings: Settings) -> None:
    config = _alembic_config(settings)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = build_engine(settings.database_url)
    remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    engine.dispose()
    assert not remaining, f"downgrade left tables behind: {sorted(remaining)}"


def test_the_index_on_the_hottest_join_column_exists(settings: Settings) -> None:
    """Face recognition joins face_embeddings on person_id for every face in every
    frame. Legacy had no index on it."""
    command.upgrade(_alembic_config(settings), "head")

    engine = build_engine(settings.database_url)
    indexed = {
        column
        for index in inspect(engine).get_indexes("face_embeddings")
        for column in index["column_names"]
    }
    engine.dispose()
    assert "person_id" in indexed


def test_upgrading_a_database_that_already_has_events_keeps_them(settings: Settings) -> None:
    """The school's machine is not an empty database, and `events.reasons` (0004) is NOT
    NULL. A column added without a default fails outright on a populated table — and the
    one machine where that is discovered would be the one holding the school's incidents.

    The reasons of a past event are NOT recoverable and are not guessed at: they get ''.
    The legacy re-derived person_type on every boot from 24 LIKE patterns and silently
    reverted the school's own corrections (audit M-18); inventing evidence for an assault
    that was recorded before we stored evidence would be that mistake with worse stakes.
    """
    from sqlalchemy import text

    config = _alembic_config(settings)
    _a_school_database_at(config, "0003")

    command.upgrade(config, "head")

    engine = build_engine(settings.database_url)
    with engine.connect() as connection:
        row = connection.execute(
            text("select summary_text, confidence, reasons from events")
        ).one()
    engine.dispose()

    assert row.summary_text == "Зафиксирована агрессия", "the upgrade lost an event"
    assert row.confidence == 0.93
    assert row.reasons == "", "a pre-existing event was given reasons nobody recorded"


def _a_school_database_at(config: Config, revision: str) -> None:
    """A database at `revision` holding one camera and one bullying event.

    The school's machine is not an empty database, and every migration that touches
    `cameras` has to be tried against one that is not -- a batch rebuild of that table
    cascade-deletes everything pointing at it, and an empty database cannot show it.
    Shared by the two tests that need the same starting point, so they cannot drift into
    proving things about two different databases.
    """
    from sqlalchemy import text

    command.upgrade(config, revision)

    engine = build_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as connection:
        connection.execute(
            text(
                "insert into cameras (name, display_name, location, camera_type, role, "
                "rtsp_host, rtsp_port, priority, enabled, created_at, updated_at) values "
                "('hall_left', 'Холл слева', 'Холл', 'BULLYING', 'MAIN_HALL', '10.0.0.1', "
                "554, 1, 1, '2026-03-04 09:00:00', '2026-03-04 09:00:00')"
            )
        )
        connection.execute(
            text(
                "insert into events (camera_id, event_type, occurred_at, confidence, "
                "candidate_probability, validation_score, skeleton_confirmed, severity, "
                "summary_text, track_ids, status, created_at, updated_at) values "
                "(1, 'BULLYING', '2026-03-04 09:12:30', 0.93, 0.9, 0.8, 1, 'ALERT', "
                "'Зафиксирована агрессия', '3,7', 'NEW', '2026-03-04 09:12:30', "
                "'2026-03-04 09:12:30')"
            )
        )
    engine.dispose()


def test_a_migration_leaves_no_broken_foreign_key_behind(settings: Settings) -> None:
    """**The receipt for `migrations/env.py::_suspend_foreign_keys`.**

    Migrations run with SQLite's foreign key enforcement OFF, and they have to: batch mode
    rebuilds a table by DROPPING it, and dropping `cameras` cascade-deletes every event,
    meal session and recognition attempt that points at it. That is a real defect this
    suite caught (`test_upgrading_a_database_that_already_has_events_keeps_them`), and
    turning enforcement off is the standard fix.

    But the fix has a price: while it is off, a migration CAN leave a dangling reference
    and nothing complains. So the price is paid back here -- `PRAGMA foreign_key_check`
    scans the whole database and returns a row per violation. Empty is the only pass.
    """
    from sqlalchemy import text

    config = _alembic_config(settings)
    _a_school_database_at(config, "0003")

    command.upgrade(config, "head")

    engine = build_engine(settings.database_url)
    with engine.connect() as connection:
        violations = connection.execute(text("PRAGMA foreign_key_check")).fetchall()
        surviving = connection.execute(text("select count(*) from events")).scalar()
        enforced = connection.execute(text("PRAGMA foreign_keys")).scalar()
    engine.dispose()

    # The premise, so that "no violations" cannot pass by having nothing to check. The
    # event must still BE there (it is the row that references `cameras`), and the
    # suspension must not have escaped the migration onto application connections.
    assert surviving == 1, "there is no surviving reference for the check to examine"
    assert enforced == 1, "foreign key enforcement leaked out of the migration"

    assert not violations, (
        "the migration left dangling foreign keys behind. Enforcement is suspended while "
        f"migrations run, so nothing raised at the time: {violations}"
    )


def test_alembic_ini_holds_no_connection_string() -> None:
    text_content = (REPO_ROOT / "alembic.ini").read_text(encoding="utf-8")
    for line in text_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("sqlalchemy.url") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip()
            assert not value, "the database URL must come from DATABASE_URL, not from the repo"


def test_there_is_exactly_one_migration_head() -> None:
    versions = list((REPO_ROOT / "migrations" / "versions").glob("*.py"))
    assert versions, "no migrations found"
    assert isinstance(Path(versions[0]), Path)

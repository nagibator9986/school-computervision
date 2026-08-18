"""`qorgan backup` — client §11 item 14, which we never built.

This database is the school's record of who ate and of every incident anyone reviewed.
It is one file on one Windows box with no RAID and no replica, and until now nothing in
the system copied it anywhere, ever.

Two things this file pins down, because both are the difference between a backup and a
file that merely looks like one:

  * **The WAL is included.** The database runs in WAL mode, so the newest committed data
    is in `qorgan.sqlite3-wal` and NOT in `qorgan.sqlite3`. Copying the one file — which
    is what an engineer with a scheduled `xcopy` would do, and what every "backup script"
    on the internet does — silently loses the most recent transactions. `VACUUM INTO`
    reads through a connection and therefore sees them.

  * **The copy is opened and checked before we call it a backup.** A backup nobody has
    ever read is a hope. This is the same principle as `qorgan doctor` building a real
    ONNX session rather than trusting the advertised provider list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from qorgan.db.models import Camera
from qorgan.enums import CameraRole, CameraType
from qorgan.maintenance.backup import BackupError, backup_database, verify_backup
from qorgan.settings import Settings


def _camera(session: Session, name: str = "hall_left") -> None:
    session.add(
        Camera(
            name=name,
            display_name="Холл слева",
            camera_type=CameraType.BULLYING,
            role=CameraRole.MAIN_HALL,
            rtsp_host="10.0.0.1",
        )
    )
    session.commit()


def _rows_in(backup: Path, table: str) -> int:
    """Open the BACKUP itself — not the original — and read it."""
    import sqlite3

    with sqlite3.connect(backup) as connection:
        return connection.execute(f"select count(*) from {table}").fetchone()[0]  # noqa: S608


def test_the_backup_is_a_real_database_with_the_real_rows(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _camera(session)

    report = backup_database(tmp_path / "backup.sqlite3")

    assert report.destination.is_file()
    assert report.bytes_written > 0
    assert _rows_in(report.destination, "cameras") == 1


def test_the_backup_contains_data_that_is_still_in_the_WAL(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """THE test. The database runs in WAL mode: a committed row lives in the -wal file
    until a checkpoint folds it into the main one. A scheduled copy of `qorgan.sqlite3`
    alone would restore a database missing exactly the newest incidents — and would look
    perfectly successful doing it."""
    _camera(session, "hall_left")
    _camera(session, "hall_right")

    # Deliberately NO checkpoint here: this is the state a real backup finds the database
    # in, seconds after a worker wrote an event.
    report = backup_database(tmp_path / "backup.sqlite3")

    assert _rows_in(report.destination, "cameras") == 2, (
        "the backup lost committed rows that were still in the WAL"
    )


def test_the_backup_does_not_need_the_workers_to_stop(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """An open connection with an uncommitted write in flight must not make the backup
    fail: this runs on a schedule against a live system, at a moment nobody chose."""
    _camera(session, "hall_left")
    session.add(
        Camera(
            name="uncommitted",
            display_name="x",
            camera_type=CameraType.BULLYING,
            role=CameraRole.MAIN_HALL,
            rtsp_host="10.0.0.2",
        )
    )
    session.flush()  # written, not committed

    report = backup_database(tmp_path / "backup.sqlite3")

    # The committed row is there; the in-flight one is not. That is what a consistent
    # snapshot means.
    assert _rows_in(report.destination, "cameras") == 1


def test_an_existing_file_is_never_silently_overwritten(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Yesterday's backup is not scratch space. Overwriting it with today's is how a
    school ends up with exactly one backup, taken at the worst possible moment."""
    _camera(session)
    target = tmp_path / "backup.sqlite3"
    target.write_bytes(b"yesterday")

    with pytest.raises(BackupError, match="already exists"):
        backup_database(target)

    assert target.read_bytes() == b"yesterday", "the old backup was destroyed"


def test_a_backup_of_a_postgres_database_is_refused_not_faked(
    settings: Settings, tmp_path: Path
) -> None:
    """VACUUM INTO is SQLite's. The spec keeps the door open to PostgreSQL, and the day
    somebody walks through it this command must say so rather than write an empty file
    and report success."""
    settings.database_url = "postgresql+psycopg://qorgan@db/qorgan"

    with pytest.raises(BackupError, match="SQLite"):
        backup_database(tmp_path / "backup.sqlite3")


def test_the_destination_directory_is_created(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """SQLite will not create the directory, and the default destination is a `backups/`
    folder that does not exist on a fresh install."""
    _camera(session)

    report = backup_database(tmp_path / "does" / "not" / "exist" / "backup.sqlite3")

    assert report.destination.is_file()


def test_the_default_destination_is_dated_and_sits_beside_the_database(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """No argument is the case that will actually run on a schedule at 3am. It must not
    collide with yesterday's, and it must not land in whatever directory Task Scheduler
    felt like starting in — the scheduler sets none, so a relative path is a coin toss.

    "Beside the database" rather than a hardcoded `./data/backups`: DATABASE_URL is the
    one place that knows where the database really is, and an install that moved it is
    entitled to have its backups follow.
    """
    _camera(session)

    report = backup_database()

    assert report.destination.is_absolute()
    assert report.destination.parent == tmp_path / "backups", (
        "the backup did not land beside the database it is a backup of"
    )
    assert report.destination.suffix == ".sqlite3"
    assert "qorgan-" in report.destination.name
    assert _rows_in(report.destination, "cameras") == 1


def test_two_backups_on_the_same_day_do_not_collide(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Otherwise the second scheduled run of the day fails on "already exists" — or, if
    that guard were ever softened, quietly replaces the morning's copy."""
    _camera(session)

    first = backup_database(tmp_path / "a.sqlite3").destination
    second = backup_database(tmp_path / "b.sqlite3").destination

    assert first != second
    assert first.is_file() and second.is_file()


def test_an_in_memory_database_is_refused(settings: Settings, tmp_path: Path) -> None:
    """There is no file to copy and no school running on one. Better a clear refusal than
    a 4 KB file that reads as an empty but valid database — which is exactly what an
    unattended scheduled backup would then produce, forever, successfully."""
    settings.database_url = "sqlite+pysqlite:///:memory:"

    with pytest.raises(BackupError, match="memory"):
        backup_database(tmp_path / "backup.sqlite3")


def test_a_corrupt_backup_is_not_reported_as_a_backup(tmp_path: Path) -> None:
    """A backup nobody has ever opened is a hope, not a backup. If the copy cannot be
    read, this must fail loudly now — on a machine where somebody is watching — rather
    than in the hour the school needs to restore it."""
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)

    with pytest.raises(BackupError, match="unreadable|integrity"):
        verify_backup(corrupt)


def test_a_healthy_backup_passes_verification(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _camera(session)
    report = backup_database(tmp_path / "backup.sqlite3")

    verify_backup(report.destination)  # must not raise


# -- the command ------------------------------------------------------------


def test_the_command_writes_a_backup_and_says_where(
    settings: Settings, session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`qorgan backup` — the thing the client asked for by name."""
    from qorgan.cli import main

    target = tmp_path / "out" / "backup.sqlite3"
    code = main(["backup", "--to", str(target)])

    assert code == 0
    assert target.is_file()
    # The engineer must be told the path, or a scheduled backup is a file nobody can find.
    assert str(target) in capsys.readouterr().out


def test_the_command_fails_loudly_rather_than_with_a_traceback(
    settings: Settings, session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This runs unattended from Task Scheduler. A non-zero exit is the only thing the
    scheduler understands, and a traceback is not an error message."""
    from qorgan.cli import main

    target = tmp_path / "backup.sqlite3"
    target.write_bytes(b"yesterday")

    code = main(["backup", "--to", str(target)])

    assert code == 1
    assert "already exists" in capsys.readouterr().err


def test_the_backup_does_not_disturb_the_live_database(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The system keeps running. A backup that leaves the live database locked, drained
    or checkpointed into a different shape is worse than no backup."""
    _camera(session)

    backup_database(tmp_path / "backup.sqlite3")

    session.expire_all()
    assert session.execute(text("select count(*) from cameras")).scalar() == 1
    _camera(session, "hall_right")  # still writable
    assert session.execute(text("select count(*) from cameras")).scalar() == 2

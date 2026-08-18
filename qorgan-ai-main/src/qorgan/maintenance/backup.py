"""`qorgan backup` — a copy of the database that is actually a copy of the database.

The school's §11 asked for this and it was never built. What is at stake is one file on
one Windows machine: every incident anyone reviewed, and the record of who ate. There is
no RAID, no replica and no cloud, and until now nothing in this system copied it anywhere.

**Why `VACUUM INTO` and not a file copy.** The database runs in WAL mode, so the newest
committed transactions live in `qorgan.sqlite3-wal` and not in `qorgan.sqlite3`. A
scheduled `xcopy` of the one file — the obvious thing, and what most backup advice on the
internet amounts to — silently restores a database missing exactly the most recent
events, and looks completely successful while doing it. `VACUUM INTO` goes through a
connection, so it sees the WAL, takes a consistent snapshot of a live database without
stopping the workers, and writes one self-contained file with no sidecars to forget.

**The copy is opened and read before we call it a backup.** A backup nobody has ever
opened is a hope. This is `qorgan doctor` building a real ONNX session rather than
trusting the advertised provider list: the same lesson, in a place where being wrong is
discovered a year later by a head teacher who needed it.

**Where the backup goes is a decision, not a default.** This writes to the same disk as
the database, which is not a backup of that disk. Copy it off the machine — see
`docs/windows-autostart.md`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from qorgan.db.engine import get_engine
from qorgan.db.types import utcnow
from qorgan.logging_setup import get_logger
from qorgan.paths import PathOutsideRoot, ensure_within
from qorgan.settings import get_settings, resolve

logger = get_logger(__name__)

# Backups go in a `backups/` folder beside the database itself, wherever DATABASE_URL
# says that is. Not a hardcoded ./data/backups: the URL is the one place that knows where
# the database really lives, and an install that moved it is entitled to have its backups
# follow. Absolute either way — this runs from Task Scheduler, which sets no working
# directory, so a relative path lands somewhere nobody chose.
BACKUP_DIR_NAME = "backups"

# What we ourselves write, and therefore what counts as a backup when reading the folder
# back. Anything else in there was put there by a person and is not ours to describe.
BACKUP_SUFFIX = ".sqlite3"

_SQLITE_URL_MARKER = ":///"


class BackupError(Exception):
    """The backup did not happen. Never raised for a backup that merely looks odd."""


@dataclass(frozen=True, slots=True)
class BackupReport:
    destination: Path
    bytes_written: int

    def summary(self) -> str:
        return f"{self.destination} ({self.bytes_written / 1e6:.1f} MB)"


def backup_database(destination: Path | None = None) -> BackupReport:
    """Snapshot the live database. Safe to run while the workers are running.

    Returns where it went and how big it is. Raises `BackupError` — never returns a
    report for a file it could not write or could not read back.
    """
    # Resolves the URL first, so a database this cannot copy is refused before it has
    # created directories or written anything.
    source = source_database()
    target = resolve(destination) if destination else default_destination(source=source)

    if target.exists():
        # Yesterday's backup is not scratch space. (SQLite refuses this too; catching it
        # here means the message names the file rather than quoting a C library.)
        raise BackupError(
            f"{target} already exists and will not be overwritten. A backup that "
            "replaces yesterday's leaves the school with exactly one, taken at a moment "
            "nobody chose."
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    _vacuum_into(target)
    verify_backup(target)

    report = BackupReport(destination=target, bytes_written=target.stat().st_size)
    logger.info("database backed up", extra={"summary": report.summary()})
    return report


def default_destination(moment: datetime | None = None, source: Path | None = None) -> Path:
    """`<database dir>/backups/qorgan-<when>.sqlite3`, in the school's own time.

    Dated to the second, because the scheduled run must not collide with yesterday's —
    or with the engineer who ran it by hand ten minutes earlier. The school's timezone
    rather than UTC for the same reason the alert message uses it: the person reading the
    filename is looking for "the one from this morning".
    """
    when = (moment or utcnow()).astimezone(get_settings().tz)
    return backup_directory(source) / f"qorgan-{when.strftime('%Y-%m-%d-%H%M%S')}.sqlite3"


def backup_directory(source: Path | None = None) -> Path:
    """The folder backups live in: `backups/` beside the database itself.

    One function rather than two spellings of the same rule. The writer and the reader
    disagreeing about where backups go is the classic version of this project's signature
    fault -- the dashboard would list an empty folder while the scheduled task filled a
    different one, and both would look correct.
    """
    return (source or source_database()).parent / BACKUP_DIR_NAME


@dataclass(frozen=True, slots=True)
class BackupFile:
    """One file in `backups/`, exactly as the filesystem describes it."""

    path: Path
    size_bytes: int
    # TIMEZONE-AWARE UTC, always. `st_mtime` is a POSIX timestamp: an instant, carrying no
    # zone at all. `datetime.fromtimestamp(ts)` without `tz=` silently stamps the SERVER's
    # local zone onto it -- and `default_destination` above dates the file NAME in the
    # SCHOOL's zone, so the very same backup would then report two different moments, one
    # in its name and one in whatever renders this. Converted for a human exactly once, at
    # the surface that shows it; never here.
    modified_at: datetime


def existing_backups() -> list[BackupFile]:
    """Every backup on this disk, newest first. Reads the folder; opens nothing.

    **No caller input reaches this.** There is no name, no id and no path in any request
    that gets here -- the directory is derived from DATABASE_URL and scanned. That is the
    strongest form of "no path traversal" available: not a check that a supplied path is
    safe, but no supplied path. `ensure_within` still runs on every entry, because a
    symlink dropped into `backups/` is a path we did not build and would otherwise let the
    page report the size of a file outside the folder.

    **Deliberately does not open the files.** A row must be drawable for a half-written or
    corrupt copy, because that is precisely the state this page exists to reveal.
    `backup_database` integrity-checks what it writes, at the moment it writes it, where a
    failure can still be acted on; re-verifying every file on every page load would read
    the whole backup set off the disk each time somebody refreshes.
    """
    folder = backup_directory()
    if not folder.is_dir():
        return []  # nothing has ever been backed up. The page has to be able to say that.

    found = [_described(entry, folder) for entry in folder.iterdir()]
    return sorted(
        (file for file in found if file is not None),
        key=lambda file: file.modified_at,
        reverse=True,
    )


def _described(entry: Path, folder: Path) -> BackupFile | None:
    """One directory entry, or None if it is not ours to describe."""
    if entry.suffix.lower() != BACKUP_SUFFIX or not entry.is_file():
        return None
    try:
        confined = ensure_within(folder, entry)
    except PathOutsideRoot:
        logger.warning(
            "ignored an entry in backups/ that resolves outside the folder",
            extra={"name": entry.name},
        )
        return None

    stat = confined.stat()
    return BackupFile(
        path=confined,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def source_database() -> Path:
    """Where DATABASE_URL actually points, or a refusal naming why this cannot be copied.

    VACUUM INTO is SQLite's alone. The spec keeps the door open to PostgreSQL (§4.3), and
    the day somebody walks through it this command must say so rather than fail obscurely
    — or, worse, appear to work.
    """
    url = get_settings().database_url
    if "sqlite" not in url or _SQLITE_URL_MARKER not in url:
        scheme = url.split("://", 1)[0]
        raise BackupError(
            f"`qorgan backup` backs up SQLite, and DATABASE_URL is {scheme}://... Use that "
            "server's own tooling (pg_dump for PostgreSQL); VACUUM INTO cannot copy it."
        )

    raw = url.split(_SQLITE_URL_MARKER, 1)[1]
    if not raw or raw == ":memory:":
        raise BackupError(
            "DATABASE_URL is an in-memory database; there is nothing on disk to back up. "
            "A backup of it would be a valid, empty file — and an unattended scheduled "
            "backup would produce one every night, successfully, forever."
        )
    return resolve(Path(raw))


def verify_backup(path: Path) -> None:
    """Open the copy and make SQLite read it. A backup nobody opened is a hope."""
    # closing(), because sqlite3's own context manager commits and does NOT close. On
    # Windows a leaked handle keeps the file locked, which would leave `qorgan backup`
    # holding the backup it just took.
    try:
        with closing(sqlite3.connect(path)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.DatabaseError as exc:
        raise BackupError(f"the backup at {path} is unreadable: {exc}") from exc

    if not result or result[0] != "ok":
        raise BackupError(f"the backup at {path} failed its integrity check: {result}")


def _vacuum_into(target: Path) -> None:
    """The snapshot itself.

    AUTOCOMMIT because VACUUM cannot run inside a transaction, and SQLAlchemy opens one
    on the first statement otherwise. The path is a BOUND PARAMETER, so a directory name
    containing a quote — this school's paths are Cyrillic and hand-typed — cannot break
    or rewrite the statement.
    """
    engine = get_engine()
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text("VACUUM INTO :target"), {"target": str(target)})
    except Exception as exc:
        raise BackupError(f"could not write the backup to {target}: {exc}") from exc

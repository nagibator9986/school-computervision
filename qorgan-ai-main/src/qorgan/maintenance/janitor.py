"""Retention. Nothing grows forever, because disks do not.

The legacy had no janitor anywhere. Snapshots, clips, face-cache JPEGs, late-identity
JPEGs and a new dated debug folder every three seconds all accumulated for the life of the
installation, and the entry-watch loop wrote about eight JPEGs and a JSON file *every
tick, twice a second, in production* (audit H-15). The disk was a slow-motion outage.

But retention on a system like this is not merely a housekeeping problem, and the rules
below are about privacy as much as about space:

  * **Media is a photograph of a child.** Keeping it longer than it is useful is not
    thrift, it is a liability. Events expire.

  * **An event a human reviewed and confirmed is not deleted on a timer.** Somebody
    decided it mattered. Deleting the evidence out from under them would be worse than
    keeping it.

  * **Deleting the file and the row must not drift apart.** The row goes first: an event
    with a dangling media path is a broken link, while a file with no row is invisible
    forever. So the file is removed only once the row no longer points at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models import Event, RecognitionAttempt
from qorgan.db.types import utcnow
from qorgan.enums import EventStatus
from qorgan.events.recorder import media_root
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

# How long an ordinary event's media is kept. Long enough for a school to act on it;
# short enough that we are not warehousing photographs of children.
DEFAULT_MEDIA_DAYS = 90

# Recognition attempts are calibration data, and there are thousands per day. They are
# useful for weeks, not years.
DEFAULT_ATTEMPT_DAYS = 30

# Statuses that mean a human looked at this and decided it mattered. Never expired on a
# timer -- deleting evidence somebody is relying on is worse than keeping a file.
PROTECTED = (EventStatus.CONFIRMED, EventStatus.REVIEWED)


@dataclass
class SweepReport:
    media_deleted: int = 0
    bytes_freed: int = 0
    attempts_deleted: int = 0
    orphans_deleted: int = 0
    protected: int = 0

    def summary(self) -> str:
        return (
            f"{self.media_deleted} media file(s) deleted ({self.bytes_freed / 1e6:.1f} MB), "
            f"{self.orphans_deleted} orphan(s), {self.attempts_deleted} recognition "
            f"attempt(s) pruned, {self.protected} reviewed event(s) kept."
        )


def sweep(
    *,
    media_days: int = DEFAULT_MEDIA_DAYS,
    attempt_days: int = DEFAULT_ATTEMPT_DAYS,
    dry_run: bool = False,
) -> SweepReport:
    """One pass. Safe to run on a timer, and safe to run twice."""
    report = SweepReport()
    _expire_media(report, media_days, dry_run)
    _prune_attempts(report, attempt_days, dry_run)
    _remove_orphans(report, dry_run)

    logger.info("retention sweep complete", extra={"summary": report.summary()})
    return report


def _expire_media(report: SweepReport, days: int, dry_run: bool) -> None:
    """Drop the media of old events. The EVENT row survives — a school still wants to know
    that a fight happened in March, even once the picture of it is gone."""
    cutoff = utcnow() - timedelta(days=days)
    root = media_root()

    with session_scope() as session:
        old = session.scalars(select(Event).where(Event.occurred_at < cutoff)).all()

        for event in old:
            if event.status in PROTECTED:
                report.protected += 1
                continue

            paths = [p for p in (event.snapshot_path, event.clip_path) if p]
            if not paths:
                continue

            if dry_run:
                report.media_deleted += len(paths)
                continue

            # The ROW first. An event pointing at a file that is gone is a broken link a
            # user will meet; a file with no row is invisible and merely wastes disk.
            event.snapshot_path = None
            event.clip_path = None
            session.flush()

            for relative in paths:
                report.bytes_freed += _delete(root / relative)
                report.media_deleted += 1


def _prune_attempts(report: SweepReport, days: int, dry_run: bool) -> None:
    """Recognition attempts are calibration data: thousands a day, useful for weeks."""
    cutoff = utcnow() - timedelta(days=days)

    with session_scope() as session:
        stale = session.scalars(
            select(RecognitionAttempt).where(RecognitionAttempt.occurred_at < cutoff)
        ).all()

        report.attempts_deleted = len(stale)
        if dry_run:
            return

        for attempt in stale:
            session.delete(attempt)


def _remove_orphans(report: SweepReport, dry_run: bool) -> None:
    """Files no event points at any more.

    They accumulate: a crash between writing the file and writing the row, or a media
    path replaced by a burst upgrade. Each one is a photograph of a child that nothing in
    the system knows about, which is the worst kind to keep.
    """
    root = media_root()
    if not root.is_dir():
        return

    with session_scope() as session:
        referenced = {
            path
            for path in session.scalars(
                select(Event.snapshot_path).where(Event.snapshot_path.is_not(None))
            ).all()
        }
        referenced |= {
            path
            for path in session.scalars(
                select(Event.clip_path).where(Event.clip_path.is_not(None))
            ).all()
        }

    for folder in ("snapshots", "clips"):
        for file in (root / folder).rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(root).as_posix()
            if relative in referenced:
                continue

            report.orphans_deleted += 1
            if not dry_run:
                report.bytes_freed += _delete(file)


def _delete(path: Path) -> int:
    """Returns the bytes freed. A file that is already gone is not an error."""
    try:
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        return 0
    except OSError:
        logger.exception("could not delete media", extra={"path": str(path)})
        return 0
    return size

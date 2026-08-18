"""The lesson record: opening one, writing the ledgers through, and closing it.

Everything in `lesson.py` is arithmetic in RAM. This is the part that survives, and the
split exists because of a bug that cost the canteen real meal records: the legacy kept
sessions in a dict inside a module-global singleton, so a restart lost every open one
without a line in the log. A meal session is minutes long. A lesson is forty-five, so a
lesson is "open" for almost the whole of its life, and a worker restart in the middle of
one must cost `flush_interval_seconds` of counting rather than the lesson.

**Track ids do not survive a restart, and the record says so rather than hiding it.**
ByteTrack numbers from scratch when the process does, so after a resume every child in the
room is a new track and a second row. Merging the two would mean deciding that track 3 of
the first run and track 17 of the second are one child, which is identification, which §8
forbids in a classroom and the corridor measurement (14 970 faces, zero recognised) says
we could not do anyway. So the rows stay apart and `lessons.resumed_count` tells the
reader why the count of tracks jumped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from qorgan.classroom.ledger import TrackLedger
from qorgan.classroom.lesson import Doubt
from qorgan.config.classroom import LessonRules
from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import Lesson, LessonTrack
from qorgan.db.types import utcnow
from qorgan.enums import LessonCloseReason, LessonState
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Clock:
    """Turns the camera loop's monotonic timestamps into wall-clock instants.

    `Frame.captured_at` is `time.monotonic()` -- an arbitrary origin that resets with the
    process, deliberately, because it is used to measure AGE and a wall clock can step
    backwards. The database stores instants. Converting needs both readings taken at the
    same moment, which is what this pair is, and doing it at the WRITE boundary rather
    than carrying datetimes through the ledgers keeps the pure layer free of clocks.

    Take a fresh one per flush. A `Clock` built at worker start and used an hour later
    would carry an hour of accumulated drift between the two clocks straight into
    `first_seen_at`.
    """

    monotonic: float
    wall: datetime

    @classmethod
    def now(cls) -> Clock:
        return cls(monotonic=time.monotonic(), wall=utcnow())

    def at(self, monotonic: float) -> datetime:
        return self.wall - timedelta(seconds=self.monotonic - monotonic)


@dataclass(frozen=True, slots=True)
class LessonHandle:
    lesson_id: int
    resumed: bool
    # What the lesson had already failed to see before this worker attached. Carried out
    # so the accumulator can be SEEDED with it: `flush` assigns these counters rather than
    # adding to them (the accumulator holds the lesson's running total), so a resumed
    # lesson starting from zero would overwrite the pre-restart measurement with a smaller
    # number on its first flush -- and a lesson would silently look cleaner every time the
    # worker died. Zero for a fresh lesson.
    doubt: Doubt


def open_or_resume(camera_id: int, rules: LessonRules, now: datetime | None = None) -> LessonHandle:
    """The open lesson for this camera, or a new one.

    Resuming rather than opening a second lesson is what makes a worker restart cheap. It
    keys on the CAMERA and on `state=OPEN`, so two workers cannot be given the same
    lesson by accident -- a camera belongs to exactly one worker group (`load_workers`
    refuses an unassigned camera), so at most one process is ever writing here.
    """
    moment = now or utcnow()

    def _open() -> LessonHandle:
        with session_scope() as session:
            row = session.scalar(
                select(Lesson)
                .where(Lesson.camera_id == camera_id, Lesson.state == LessonState.OPEN)
                .order_by(Lesson.started_at.desc())
            )
            if row is not None:
                row.resumed_count += 1
                logger.info(
                    "re-attached to an open lesson",
                    extra={"lesson": row.id, "camera_id": camera_id, "resumes": row.resumed_count},
                )
                return LessonHandle(row.id, resumed=True, doubt=_doubt_so_far(row))

            fresh = Lesson(
                camera_id=camera_id,
                state=LessonState.OPEN,
                started_at=moment,
                # Copied, never re-read. See db/models/classroom.py.
                min_presence_seconds=rules.min_presence_seconds,
            )
            session.add(fresh)
            session.flush()
            return LessonHandle(fresh.id, resumed=False, doubt=Doubt())

    return with_retry(_open)


def _doubt_so_far(row: Lesson) -> Doubt:
    """What this lesson had already failed to see, so a resuming worker carries it on."""
    return Doubt(
        ambiguous=row.ambiguous_observations,
        unclaimed=row.unclaimed_observations,
        dropped_tracks=row.dropped_tracks,
    )


def flush(lesson_id: int, ledgers: list[TrackLedger], doubt: Doubt, clock: Clock) -> int:
    """Write these ledgers' totals through, and the lesson's doubt counters with them.

    An UPSERT keyed on (lesson, track): the same track is written on every flush, each
    time with its running totals, so the row is the truth as of the last flush rather
    than one row per flush. Totals are ASSIGNED, not added -- the ledger already holds
    the whole lesson's count for that track, and adding would double it every 30 seconds.

    The doubt counters ARE assigned too, from the accumulator's running totals, for the
    same reason. That is exactly why a resuming worker SEEDS its accumulator from the row
    (`LessonHandle.doubt`): starting from zero and then assigning would overwrite what the
    previous worker measured with a smaller number, and every crash would quietly make the
    lesson look like it had been watched better than it was.
    """

    def _write() -> int:
        with session_scope() as session:
            lesson = session.get(Lesson, lesson_id)
            if lesson is None or lesson.state is LessonState.CLOSED:
                # The janitor timed it out from under us. Refusing to write is right:
                # appending to a closed record would put tracks after `ended_at`.
                return 0

            lesson.ambiguous_observations = doubt.ambiguous
            lesson.unclaimed_observations = doubt.unclaimed
            lesson.dropped_tracks = doubt.dropped_tracks

            for ledger in ledgers:
                _upsert(session, lesson_id, ledger, clock)
            return len(ledgers)

    return with_retry(_write)


def _upsert(session, lesson_id: int, ledger: TrackLedger, clock: Clock) -> None:
    row = session.scalar(
        select(LessonTrack).where(
            LessonTrack.lesson_id == lesson_id, LessonTrack.track_id == ledger.track_id
        )
    )
    if row is None:
        row = LessonTrack(
            lesson_id=lesson_id,
            track_id=ledger.track_id,
            first_seen_at=clock.at(ledger.first_seen),
            last_seen_at=clock.at(ledger.last_seen),
        )
        session.add(row)

    row.last_seen_at = clock.at(ledger.last_seen)
    row.observed_seconds = ledger.observed_seconds
    row.observations = ledger.observations
    row.settled = ledger.settled
    row.hand_raises = ledger.hand_raises
    row.stands = ledger.stands
    row.away_seconds = ledger.away_seconds
    row.brief_excursions = ledger.brief_excursions


def close(lesson_id: int, reason: LessonCloseReason, now: datetime | None = None) -> bool:
    """End the lesson. Idempotent: a lesson the janitor already closed stays as it was."""
    moment = now or utcnow()

    def _close() -> bool:
        with session_scope() as session:
            row = session.get(Lesson, lesson_id)
            if row is None or row.state is LessonState.CLOSED:
                return False
            row.state = LessonState.CLOSED
            row.close_reason = reason
            row.ended_at = moment
            return True

    return with_retry(_close)


def close_stale_lessons(rules: LessonRules, now: datetime | None = None) -> int:
    """Close lessons nobody ended, the same janitor the canteen runs over meal sessions.

    A lesson left open does not merely look untidy: `open_or_resume` would re-attach to
    it tomorrow morning, and tomorrow's tracks would be counted into yesterday's report
    under a `started_at` from the day before. Better to record honestly that we lost the
    end of a lesson.

    It takes the RULES and not a pipeline, because the sweep reads `max_lesson_minutes`
    and nothing else -- the same correction `close_sessions_nobody_exited` needed after
    five releases of being reachable only through an object that owns a camera.
    """
    moment = now or utcnow()
    cutoff = moment - timedelta(minutes=rules.max_lesson_minutes)

    def _sweep() -> int:
        with session_scope() as session:
            stale = session.scalars(
                select(Lesson).where(
                    Lesson.state == LessonState.OPEN, Lesson.started_at < cutoff
                )
            ).all()
            for row in stale:
                row.state = LessonState.CLOSED
                row.close_reason = LessonCloseReason.TIMEOUT
                row.ended_at = moment
            if stale:
                logger.warning("force-closed lessons nobody ended", extra={"count": len(stale)})
            return len(stale)

    return with_retry(_sweep)

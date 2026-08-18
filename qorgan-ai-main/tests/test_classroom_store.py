"""The lesson record: opening, writing through, resuming, and the janitor.

The bug being defended against is a specific one with a price already paid. The legacy
kept canteen sessions in a RAM dict inside a module-global singleton, and a restart lost
every open one silently. A meal session is minutes long; **a lesson is forty-five, so it
is "open" for almost the whole of its life** and the window in which a restart destroys
it is nearly all of it.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.classroom.ledger import TrackLedger
from qorgan.classroom.lesson import Doubt
from qorgan.classroom.store import Clock, close, close_stale_lessons, flush, open_or_resume
from qorgan.cli import _classroom_lesson_rules, _cmd_supervisor
from qorgan.config.camera import CAMERA_ADAPTER
from qorgan.config.canteen import SessionRules
from qorgan.config.classroom import LessonRules
from qorgan.config.workers import WorkersConfig
from qorgan.db.models import Camera, Lesson, LessonTrack
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole, CameraType, LessonCloseReason, LessonState
from qorgan.supervisor.supervisor import Supervisor

RULES = LessonRules(max_lesson_minutes=60.0, min_presence_seconds=300.0)


def _camera(session: Session) -> Camera:
    row = Camera(
        name="class_7a",
        display_name="7-А",
        camera_type=CameraType.CLASSROOM,
        role=CameraRole.CLASSROOM,
        rtsp_host="10.0.0.9",
    )
    session.add(row)
    session.commit()
    return row


def _ledger(track_id: int, **kwargs) -> TrackLedger:
    ledger = TrackLedger(track_id=track_id, first_seen=100.0, last_seen=200.0)
    for key, value in kwargs.items():
        setattr(ledger, key, value)
    return ledger


def test_opening_a_lesson_writes_a_row_immediately(session: Session) -> None:
    """From the moment it opens, not when it ends. A lesson that only becomes a row at
    the bell is a lesson that a crash at minute 40 erases entirely."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    row = session.get(Lesson, handle.lesson_id)
    assert row is not None
    assert row.state is LessonState.OPEN
    assert not handle.resumed


def test_the_presence_threshold_is_stamped_on_the_row(session: Session) -> None:
    """Copied at open, never re-read. A report is a statement about a past hour, and
    re-reading today's YAML would silently restate it under a threshold nobody applied."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, LessonRules(min_presence_seconds=123.0))

    assert session.get(Lesson, handle.lesson_id).min_presence_seconds == 123.0


def test_a_second_worker_resumes_the_open_lesson_rather_than_starting_another(
    session: Session,
) -> None:
    """**The mid-lesson restart, which is the whole reason this is a table.**

    A second lesson row would split the lesson in two and report the first as having
    ended at the moment the worker died.
    """
    camera = _camera(session)
    first = open_or_resume(camera.id, RULES)
    second = open_or_resume(camera.id, RULES)

    assert second.lesson_id == first.lesson_id
    assert second.resumed
    assert session.get(Lesson, first.lesson_id).resumed_count == 1


def test_a_resume_is_counted_because_track_ids_restart_with_the_process(
    session: Session,
) -> None:
    """ByteTrack numbers from scratch when the process does, so after a resume every
    child in the room is a NEW track and a SECOND row. The count of tracks jumps without
    a child having moved, and `resumed_count` is what lets the report say why."""
    camera = _camera(session)
    open_or_resume(camera.id, RULES)
    for _ in range(3):
        open_or_resume(camera.id, RULES)

    lesson = session.scalar(select(Lesson))
    assert lesson.resumed_count == 3


def test_a_resuming_worker_inherits_what_the_lesson_already_failed_to_see(
    session: Session,
) -> None:
    """**Otherwise every crash makes the lesson look better watched than it was.**

    `flush` ASSIGNS the doubt counters from the accumulator's running total, because that
    total is the lesson's, not the interval's. A worker that resumed from zero would
    therefore overwrite 140 discarded poses with 3 on its first flush, and nothing on the
    row or the page would show that the earlier measurement had been thrown away.
    """
    camera = _camera(session)
    first = open_or_resume(camera.id, RULES)
    flush(first.lesson_id, [], Doubt(ambiguous=140, unclaimed=7, dropped_tracks=2), Clock.now())

    second = open_or_resume(camera.id, RULES)

    assert second.doubt.ambiguous == 140
    assert second.doubt.unclaimed == 7
    assert second.doubt.dropped_tracks == 2


def test_a_fresh_lesson_starts_with_no_doubt(session: Session) -> None:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    assert handle.doubt == Doubt()


def test_a_closed_lesson_is_not_resumed(session: Session) -> None:
    """Yesterday's lesson must not absorb this morning's tracks."""
    camera = _camera(session)
    first = open_or_resume(camera.id, RULES)
    close(first.lesson_id, LessonCloseReason.EMPTY_ROOM)

    second = open_or_resume(camera.id, RULES)

    assert second.lesson_id != first.lesson_id
    assert not second.resumed


def test_flush_writes_the_totals_through(session: Session) -> None:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    flush(
        handle.lesson_id,
        [_ledger(3, hand_raises=2, stands=1, away_seconds=42.0, settled=True, observations=90)],
        Doubt(ambiguous=5, unclaimed=6, dropped_tracks=1),
        Clock.now(),
    )

    row = session.scalar(select(LessonTrack))
    assert (row.track_id, row.hand_raises, row.stands) == (3, 2, 1)
    assert row.away_seconds == 42.0
    assert row.settled is True

    lesson = session.get(Lesson, handle.lesson_id)
    assert (lesson.ambiguous_observations, lesson.unclaimed_observations) == (5, 6)
    assert lesson.dropped_tracks == 1


def test_flushing_the_same_track_twice_updates_one_row(session: Session) -> None:
    """**An UPSERT, and the totals are ASSIGNED rather than added.**

    The ledger already holds the whole lesson's count for that track, so adding would
    double it every thirty seconds for forty-five minutes -- and one row per flush would
    make a 45-minute lesson 90 rows per child.
    """
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    flush(handle.lesson_id, [_ledger(3, hand_raises=1)], Doubt(), Clock.now())
    flush(handle.lesson_id, [_ledger(3, hand_raises=4)], Doubt(), Clock.now())

    rows = session.scalars(select(LessonTrack)).all()
    assert len(rows) == 1
    assert rows[0].hand_raises == 4


def test_flushing_into_a_closed_lesson_writes_nothing(session: Session) -> None:
    """The janitor may have timed it out from under the worker. Appending then would put
    tracks after the lesson's own `ended_at`."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)
    close(handle.lesson_id, LessonCloseReason.TIMEOUT)

    written = flush(handle.lesson_id, [_ledger(3, hand_raises=1)], Doubt(), Clock.now())

    assert written == 0
    assert session.scalars(select(LessonTrack)).all() == []


def test_the_clock_turns_monotonic_frame_times_into_instants() -> None:
    """`Frame.captured_at` is `time.monotonic()` -- an arbitrary origin that resets with
    the process. The database stores instants, and the conversion needs both readings
    taken at the same moment, which is what a `Clock` is."""
    wall = utcnow()
    clock = Clock(monotonic=1000.0, wall=wall)

    assert clock.at(1000.0) == wall
    assert clock.at(940.0) == wall - timedelta(seconds=60)


def test_closing_is_idempotent(session: Session) -> None:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    assert close(handle.lesson_id, LessonCloseReason.EMPTY_ROOM)
    assert not close(handle.lesson_id, LessonCloseReason.TIMEOUT)
    assert session.get(Lesson, handle.lesson_id).close_reason is LessonCloseReason.EMPTY_ROOM


def test_the_janitor_closes_a_lesson_nobody_ended(session: Session) -> None:
    """A lesson left OPEN is re-attached to tomorrow morning, and tomorrow's tracks are
    counted into yesterday's report under a `started_at` from the day before."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=3))

    assert close_stale_lessons(RULES) == 1

    row = session.get(Lesson, handle.lesson_id)
    assert row.state is LessonState.CLOSED
    assert row.close_reason is LessonCloseReason.TIMEOUT
    assert row.ended_at is not None


def test_the_janitor_leaves_a_lesson_that_is_still_running(session: Session) -> None:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)

    assert close_stale_lessons(RULES) == 0
    assert session.get(Lesson, handle.lesson_id).state is LessonState.OPEN


def test_the_janitor_needs_only_the_rules(session: Session) -> None:
    """It reads `max_lesson_minutes` and nothing else -- no camera, no pipeline.

    `close_sessions_nobody_exited` spent five releases reachable only through a
    `SessionManager` that needs an `entry_camera_id` the sweep never looks at, so its only
    callers were tests that already had one. The supervisor has no camera rows and no
    business making any.
    """
    camera = _camera(session)
    open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=3))

    assert close_stale_lessons(LessonRules(max_lesson_minutes=60.0)) == 1


# -- the sweep as PRODUCTION runs it ------------------------------------------


class _NeverSpawns:
    """A process factory that hands out nothing. The sweep is the subject, not the fleet."""

    def __call__(self, group):
        raise AssertionError("this test must not start a worker process")


def _workers() -> WorkersConfig:
    return WorkersConfig(
        groups=[{"name": "class_group", "cameras": ["class_7a"]}],
        restart_backoff_seconds=0.01,
        restart_backoff_max_seconds=0.08,
    )


def test_the_supervisor_tick_closes_an_abandoned_lesson(session: Session) -> None:
    """**Driven through `Supervisor.tick()`, not by calling the sweep by hand.**

    `close_stale_lessons` works whenever a test calls it directly, and that is exactly how
    the canteen's equivalent stayed broken for five releases: nothing in `src/` called it,
    every test was green, and the instrument it fed read zero by construction. So this
    goes through the production path.
    """
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=3))

    supervisor = Supervisor(_workers(), factory=_NeverSpawns(), lesson_rules=RULES)
    supervisor.workers = []  # nothing to supervise; the sweep is what is under test
    supervisor.tick()

    session.expire_all()
    row = session.get(Lesson, handle.lesson_id)
    assert row.state is LessonState.CLOSED, "nothing in production ends an abandoned lesson"
    assert row.close_reason is LessonCloseReason.TIMEOUT


def test_the_next_lesson_is_not_swallowed_by_the_abandoned_one(session: Session) -> None:
    """The consequence the sweep exists to prevent, stated as behaviour.

    Without it, `open_or_resume` re-attaches to yesterday's open lesson and this
    morning's tracks are counted into it, under a `started_at` from the day before.
    """
    camera = _camera(session)
    stale = open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=3))

    supervisor = Supervisor(_workers(), factory=_NeverSpawns(), lesson_rules=RULES)
    supervisor.workers = []
    supervisor.tick()

    fresh = open_or_resume(camera.id, RULES)
    assert fresh.lesson_id != stale.lesson_id
    assert not fresh.resumed


def test_a_supervisor_with_no_lesson_rules_sweeps_nothing(session: Session) -> None:
    """An installation with no classroom camera has no lessons, and must not crash trying
    to sweep them. This is the branch today's shipped config actually takes."""
    supervisor = Supervisor(_workers(), factory=_NeverSpawns())
    supervisor.workers = []
    supervisor.tick()  # must not raise


def test_both_sweeps_run_on_the_same_tick(session: Session) -> None:
    """`_sweep_due` CONSUMES the interval, so a supervisor that let each sweep ask for
    itself would starve whichever asked second -- for a whole minute, every minute."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=3))

    supervisor = Supervisor(
        _workers(),
        factory=_NeverSpawns(),
        session_rules=SessionRules(),
        lesson_rules=RULES,
    )
    supervisor.workers = []
    supervisor.tick()

    session.expire_all()
    assert session.get(Lesson, handle.lesson_id).state is LessonState.CLOSED


def test_the_supervisor_command_wires_the_lesson_sweep(
    settings, monkeypatch
) -> None:
    """`qorgan supervisor` is what actually runs, and a sweep it never passes rules to
    would return on its first line forever while every test above stayed green.

    **Today this legitimately passes `None`, and that is worth stating out loud**: the
    shipped `config/` has no classroom camera, because the school has not installed one
    (§12.4 asks for a separate camera that does not exist yet). What is asserted is the
    WIRING -- that the keyword is passed at all, and that `_classroom_lesson_rules` finds
    real rules the moment a classroom camera appears in the config.
    """
    built: dict[str, object] = {}

    class Recording:
        def __init__(self, _config, **kwargs: object) -> None:
            built.update(kwargs)

        def run(self) -> None:
            """The real one blocks until SIGTERM."""

    monkeypatch.setattr("qorgan.supervisor.Supervisor", Recording)
    _cmd_supervisor(argparse.Namespace())

    assert "lesson_rules" in built, "qorgan supervisor never wires the lesson sweep"
    assert built["lesson_rules"] is None, "the shipped config has no classroom camera yet"


def test_the_rules_are_found_as_soon_as_a_classroom_camera_exists() -> None:
    """The other half of the test above: the wiring is inert today only because there is
    no camera, not because the lookup does not work."""
    camera = CAMERA_ADAPTER.validate_python(
        {
            "name": "class_7a",
            "display_name": "7-А",
            "camera_type": "classroom",
            "role": "classroom",
            "rtsp": {"host": "10.0.0.9"},
            "classroom": {"lesson": {"max_lesson_minutes": 42.0}},
        }
    )

    found = _classroom_lesson_rules({"class_7a": camera})

    assert found is not None
    assert found.max_lesson_minutes == 42.0


def test_a_lesson_track_row_has_nowhere_to_put_a_person(session: Session) -> None:
    """**§8, enforced by the schema rather than by good intentions.**

    Every other observation table here points at `persons`. These must not: identification
    inside a classroom was ruled out in writing, and this school's own footage says it
    would not work anyway (14 970 corridor faces, median 11.5 px, zero recognised). A
    column that could only ever be filled by a method measured to produce nothing is a
    column every later report would join through as though it named a child.
    """
    columns = set(LessonTrack.__table__.columns.keys()) | set(Lesson.__table__.columns.keys())
    named = {c for c in columns if "person" in c or "student" in c or "pupil" in c}

    assert not named, f"a classroom table grew an identity column: {sorted(named)}"

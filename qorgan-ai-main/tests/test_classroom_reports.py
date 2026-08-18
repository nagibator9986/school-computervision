"""The lesson report, and the sentences that must travel with it.

The canteen taught this codebase the shape of the mistake here. Its «кто не ел» question
was unanswerable by the legacy because the canteen log only ever contained children who
had been SEEN -- so "who is missing from this log" had nothing to be asked against. The
classroom equivalent is listing only the tracks that did something: "nobody put their hand
up" and "we did not look" would render identically, and the second is the one that needs
saying.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from qorgan.classroom.ledger import TrackLedger
from qorgan.classroom.lesson import Doubt
from qorgan.classroom.reports import lesson_report, recent_lessons
from qorgan.classroom.store import Clock, close, flush, open_or_resume
from qorgan.config.classroom import LessonRules
from qorgan.db.models import Camera, Lesson
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole, CameraType, LessonCloseReason

RULES = LessonRules(min_presence_seconds=300.0)


def _camera(session: Session, name: str = "class_7a") -> Camera:
    row = Camera(
        name=name,
        display_name="7-А",
        camera_type=CameraType.CLASSROOM,
        role=CameraRole.CLASSROOM,
        rtsp_host="10.0.0.9",
    )
    session.add(row)
    session.commit()
    return row


def _ledger(track_id: int, seconds: float, **kwargs) -> TrackLedger:
    ledger = TrackLedger(track_id=track_id, first_seen=0.0, last_seen=seconds)
    ledger.settled = True
    for key, value in kwargs.items():
        setattr(ledger, key, value)
    return ledger


def _lesson(session: Session, ledgers, doubt: Doubt | None = None) -> int:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)
    flush(handle.lesson_id, ledgers, doubt or Doubt(), Clock.now())
    return handle.lesson_id


def test_a_track_that_did_nothing_still_appears(session: Session) -> None:
    """**The canteen's lesson, applied.** A report listing only tracks with a non-zero
    count makes "nobody raised a hand" and "we did not look" identical on the page."""
    lesson_id = _lesson(session, [_ledger(1, 600.0), _ledger(2, 600.0, hand_raises=3)])

    report = lesson_report(lesson_id)

    assert [row.track_id for row in report.present] == [1, 2]
    assert report.present[0].hand_raises == 0


def test_short_tracks_are_separated_not_dropped(session: Session) -> None:
    """Fragments are the direct measure of how badly the tracker is losing people.
    Dropping them makes a room the system kept losing look like a calm one."""
    lesson_id = _lesson(session, [_ledger(1, 600.0), _ledger(2, 30.0)])

    report = lesson_report(lesson_id)

    assert [row.track_id for row in report.present] == [1]
    assert [row.track_id for row in report.fragments] == [2]
    assert report.tracks == 2


def test_the_presence_threshold_comes_off_the_row_not_the_config(session: Session) -> None:
    """An old lesson must not be restated under today's YAML."""
    camera = _camera(session)
    handle = open_or_resume(camera.id, LessonRules(min_presence_seconds=60.0))
    flush(handle.lesson_id, [_ledger(1, 100.0)], Doubt(), Clock.now())

    report = lesson_report(handle.lesson_id)

    assert report.min_presence_seconds == 60.0
    assert [row.track_id for row in report.present] == [1]


def test_tracks_are_ordered_by_id_and_never_by_a_count(session: Session) -> None:
    """**A report sorted by "most hand raises" ranks children**, and ranking is exactly
    the judgement §8 told the school the system would not make."""
    lesson_id = _lesson(
        session,
        [
            _ledger(1, 600.0, hand_raises=0),
            _ledger(2, 600.0, hand_raises=9),
            _ledger(3, 600.0, hand_raises=4),
        ],
    )

    report = lesson_report(lesson_id)

    assert [row.track_id for row in report.present] == [1, 2, 3]


def test_place_metrics_on_an_unsettled_track_are_marked_unmeasured(session: Session) -> None:
    """Not zero. A zero that also means "unknown" is the defect migration 0005 exists
    about, and this is the last chance to keep the two apart before the page renders."""
    unsettled = _ledger(1, 600.0)
    unsettled.settled = False
    lesson_id = _lesson(session, [unsettled, _ledger(2, 600.0)])

    report = lesson_report(lesson_id)

    assert not report.present[0].measured_place
    assert report.present[1].measured_place
    assert report.unsettled == 1


def test_the_two_permanent_caveats_are_always_present(session: Session) -> None:
    """They hold on every lesson whatever the counts, and their absence on a quiet one is
    exactly how the claim gets lost. A track is not a child; nothing here is validated."""
    lesson_id = _lesson(session, [_ledger(1, 600.0)])

    caveat = lesson_report(lesson_id).caveat()

    assert len(caveat) >= 2
    assert "ТРЕКУ" in caveat[0], "the report stopped saying the numbers are per-track"
    assert "не проверен" in caveat[1], "the report stopped saying the thresholds are unvalidated"


def test_the_caveat_says_a_track_count_is_not_a_headcount(session: Session) -> None:
    lesson_id = _lesson(session, [_ledger(1, 600.0), _ledger(2, 600.0)])

    joined = " ".join(lesson_report(lesson_id).caveat())

    assert "НЕ" in joined and "учеников" in joined


def test_conditional_caveats_appear_only_when_there_is_a_number(session: Session) -> None:
    """An absent line says nothing, so a line that always appeared would be noise the
    reader learns to skip -- and it is the same list that has to carry the real warnings."""
    quiet = _lesson(session, [_ledger(1, 600.0)])
    assert len(lesson_report(quiet).caveat()) == 2


def test_ambiguity_and_dropped_tracks_reach_the_caveat(session: Session) -> None:
    lesson_id = _lesson(
        session,
        [_ledger(1, 600.0)],
        Doubt(ambiguous=140, unclaimed=3, dropped_tracks=2),
    )

    joined = " ".join(lesson_report(lesson_id).caveat())

    assert "140" in joined, "discarded observations never reached the reader"
    assert "2" in joined


def test_a_resumed_lesson_warns_that_one_child_may_be_several_tracks(
    session: Session,
) -> None:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)
    open_or_resume(camera.id, RULES)  # the restart
    flush(handle.lesson_id, [_ledger(1, 600.0)], Doubt(), Clock.now())

    joined = " ".join(lesson_report(handle.lesson_id).caveat())

    assert "перезапус" in joined


def test_a_missing_lesson_reports_nothing_rather_than_an_empty_one(session: Session) -> None:
    """An empty report for a lesson that does not exist would render as a room in which
    nothing happened."""
    assert lesson_report(999) is None


def test_the_summary_refuses_to_call_a_track_a_child(session: Session) -> None:
    lesson_id = _lesson(session, [_ledger(1, 600.0)])

    assert "not a child" in lesson_report(lesson_id).summary()


def test_recent_lessons_are_newest_first(session: Session) -> None:
    camera = _camera(session)
    older = open_or_resume(camera.id, RULES, now=utcnow() - timedelta(hours=2))
    close(older.lesson_id, LessonCloseReason.EMPTY_ROOM)
    newer = open_or_resume(camera.id, RULES, now=utcnow())

    reports = recent_lessons()

    assert [r.lesson_id for r in reports] == [newer.lesson_id, older.lesson_id]


def test_recent_lessons_names_the_camera(session: Session) -> None:
    camera = _camera(session, name="class_9b")
    open_or_resume(camera.id, RULES)

    assert recent_lessons()[0].camera_name == "class_9b"


def test_deleting_a_camera_takes_its_lessons_with_it(session: Session) -> None:
    """`ondelete="CASCADE"`, and the foreign key really is enforced at runtime.

    This is what makes `lesson_report`'s "camera {id}" fallback unreachable in normal
    operation -- a lesson cannot outlive its camera. The fallback stays anyway, because
    there IS one window where the guarantee is suspended: migrations run with
    `PRAGMA foreign_keys=OFF` (see `migrations/env.py`, and the reason is that a batch
    table rebuild would otherwise cascade-delete the school's events). A page that raised
    rather than printing an id would take the whole report down for one orphaned row.
    """
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)
    flush(handle.lesson_id, [_ledger(1, 600.0)], Doubt(), Clock.now())

    session.delete(camera)
    session.commit()

    assert session.get(Lesson, handle.lesson_id) is None
    assert lesson_report(handle.lesson_id) is None

"""**One child, one record.** The three defects that split a child into two, and the guards.

Six people in the school's own roster hold two IDs each, with their meals split across both
and neither record true. We found that in THEIR data. This file exists because the canteen
pipeline was about to manufacture the same thing in ours, and it is one causal chain:

  1. Person tracking ran only on "due" frames -- every 1.5 s inside, every 0.25 s at the
     door. ByteTrack associates by IOU and a motion model and needs frame-to-frame
     continuity, so at that cadence a child is simply issued a NEW track id. Two track ids,
     two Unknown sessions, one child.
  2. `SessionManager.open(person_id=None)` skips the dedup entirely -- it dedups by
     person_id, and an Unknown session has none -- so a split track opens a second session
     with nothing to stop it.
  3. The exit camera got ONE close attempt, and `close()` can refuse TRANSIENTLY. A spent
     attempt is never repeated, so the session never closes and force-closes as UNKNOWN 90
     minutes later.

The trade runs both ways, and the tests below hold both ends of it: **suppressing a real
child's session is exactly as bad as duplicating one.** A duplicate corrupts a record; a
suppression puts a child who ate on the "did not eat" report, which is the one report the
school actually asked for.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import CanteenSession
from qorgan.enums import CameraRole, SessionState
from qorgan.settings import Settings
from tests.canteen_fakes import (
    BEHIND_BOX,
    PERSON_BOX,
    TINY_BOX,
    FakePersonDetector,
    FakeRecognizer,
    at_fps,
    build_pipeline,
    canteen_rows,
    frame,
    vector,
)


@pytest.fixture
def rows(session: Session) -> dict:
    return canteen_rows(session)


def _sessions(session: Session) -> list[CanteenSession]:
    session.expire_all()
    return list(session.scalars(select(CanteenSession)).all())


# -- defect 1: ByteTrack cannot track at 0.67 Hz -------------------------------

# Two seconds of the real entry/exit cadence: 15 fps (the sub-stream's own rate),
# det_every 1. This was 16 frames, from `display_fps: 8` -- a field production never read,
# so "the real cadence" was 8 fps on no camera. The DURATION is what these tests are about
# (the inside camera's 1.5 s interval has to fall inside the window twice), so the frame
# count moves with the rate and the two seconds stay two seconds.
FRAMES = 30


@pytest.mark.parametrize("role", [CameraRole.CANTEEN_ENTRY, CameraRole.CANTEEN_EXIT])
def test_person_tracking_runs_on_every_frame_at_the_door(
    settings: Settings, session: Session, rows: dict, role: CameraRole
) -> None:
    """**A meal-session RECORD depends on track continuity, so tracking must be continuous.**

    ByteTrack associates by IOU and a motion model. Behind the recognition gate it was asked
    to associate across 0.25 s -- and at the inside cameras across 1.5 s, by which time a
    child has crossed the room. Association fails and they are issued a NEW track id. That is
    not an occasional switch, it is structural, and every new id is another meal session.
    """
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])
    person = FakePersonDetector()
    pipeline = build_pipeline(role, rows, recognizer, person)

    for tick in range(FRAMES):
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert person.calls == FRAMES, (
        "person tracking is still behind the recognition gate: ByteTrack is being asked to "
        "associate across gaps it cannot associate across, and a split track is a split "
        "meal record"
    )


def test_person_tracking_stays_on_the_interval_inside(
    settings: Settings, session: Session, rows: dict
) -> None:
    """An inside camera confirms presence; it never opens and never closes a session.

    A duplicate track there is a duplicate CONFIRMATION and creates no record at all, so
    continuity buys nothing, and running YOLO at the door's full rate here would be paying
    for it anyway. It stays on its 1.5 s interval: due at t=0.0 and t=1.5, and on no other
    frame in these two seconds.

    (The cost multiplier this docstring used to quote was denominated in the 8 fps that no
    camera runs at. It is dropped rather than rescaled -- the ratio it came from is not
    reproducible from anything in the tree, and the argument does not need it.)
    """
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])
    person = FakePersonDetector()
    pipeline = build_pipeline(CameraRole.CANTEEN_INSIDE, rows, recognizer, person)

    for tick in range(FRAMES):
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert person.calls == 2, (
        "the inside camera is paying for tracking continuity it has no record to protect"
    )


def test_face_detection_is_still_skipped_once_every_track_is_bound(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**Tracking every frame must not drag face detection along with it.**

    Detection is the expensive half -- 25.4 ms against the embedding's 10.0 -- and
    `IdentityService._needs_a_face` is what makes tracking on every frame affordable at all:
    once every track in shot is BOUND or EXHAUSTED there is no face left to look for, and the
    frame costs only the 17.3 ms of YOLO.

    This is the guard on that. If someone throttles the face work back with a blanket gate,
    or rips `_needs_a_face` out, the tracking fix stops being cheap and this test says so.
    """
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])
    person = FakePersonDetector()
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person)

    pipeline.on_frame(None, frame(at=at_fps(0)))
    assert recognizer.detect_calls == 1, "the first frame must look for a face"

    for tick in range(1, FRAMES):
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    row = _sessions(session)[0]
    assert row.person_id == rows["pupil"].id, "the fixture never bound the track; it proves nothing"
    assert recognizer.detect_calls == 1, (
        "face detection is running on a track whose identity is already final -- 25.4 ms a "
        "frame to re-learn a thing we know"
    )
    assert person.calls == FRAMES, "...and tracking must still have run on every frame"


# -- defect 2: an Unknown session has no dedup ---------------------------------


def test_a_child_tracked_continuously_opens_exactly_one_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The control. One child, one continuous track, one session -- even unrecognised."""
    recognizer = FakeRecognizer()
    recognizer.show(vector(99))  # a stranger: matches nobody in the gallery
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, FakePersonDetector())

    for tick in range(40):
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    rows_ = _sessions(session)
    assert len(rows_) == 1
    assert rows_[0].person_id is None


def test_a_split_track_does_not_open_a_second_unknown_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**The corruption, manufactured on purpose.**

    A child stands in the doorway and we never manage to name them: one Unknown session. The
    queue closes over them for three seconds -- longer than ByteTrack's buffer -- and when
    they re-emerge in the same doorway they are a NEW track id. That id resolves
    independently and opens a SECOND Unknown session.

    One child, two meal records, the meal split across both, neither of them true. That is
    exactly the shape of the six double-ID people in the school's own roster.
    """
    recognizer = FakeRecognizer()
    recognizer.show(vector(99))
    person = FakePersonDetector()
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person)

    # The child is at the door as track 1. Unrecognised -> exactly one Unknown session.
    person.sees({1: PERSON_BOX})
    for tick in range(8):  # 0.000 .. 0.875 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 1, "the fixture did not open the first Unknown session"

    # The queue closes over them. The track dies.
    person.walk_away()
    for tick in range(8, 28):  # 1.000 .. 3.375 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    # ...and the SAME child re-emerges in the SAME doorway under a NEW track id, well inside
    # `person_cooldown_seconds` (5 s) of the session their old id opened.
    person.sees({2: PERSON_BOX})
    for tick in range(28, 44):  # 3.500 .. 5.375 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 1, (
        "one child now holds two Unknown meal sessions -- the exact corruption we found in "
        "the school's roster, and we just built it ourselves"
    )


def test_a_person_box_too_small_to_be_a_child_at_the_door_opens_no_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """`min_person_box_area` -- declared for months, read by nothing, wired here.

    A person box of 3 000 px is a figure at the far end of the room, not a child at the
    door. A meal record made out of one is a hole in the register invented from nothing, and
    it goes on to be a child on the "did not eat" report who never existed.
    """
    recognizer = FakeRecognizer()
    recognizer.show(vector(99))
    person = FakePersonDetector()
    person.sees({1: TINY_BOX})
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person)

    for tick in range(8):
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert recognizer.embed_calls == 1, (
        "this fixture no longer reproduces what its name claims: the track never acquired a "
        "face, so it was never decided, and no session was suppressed -- it was never offered"
    )
    assert _sessions(session) == [], "a figure at the back of the room was given a meal record"


def test_a_second_child_at_the_door_still_gets_their_own_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**The other end of the trade, and the reason the rule is not a global cooldown.**

    Two unrecognised children queue at the door. The second one is in shot BEHIND the first,
    looking away, so we have no face for them and they bind nothing. The first walks in; the
    second steps forward into *exactly* the spot the first was standing in, half a second
    later -- inside the cooldown, and squarely in the same place.

    Every geometric test says "same child". They are not, and the proof is that ByteTrack
    had them BOTH in shot at once: it never gives one person two track ids at the same time.
    Two tracks ever seen in the same frame are two different children, always.

    A global cooldown -- or a place test without that check -- eats this child's session, and
    a child who ate lands on the "did not eat" report.
    """
    recognizer = FakeRecognizer()
    recognizer.show(vector(99))
    person = FakePersonDetector()
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person)

    # Child A in the doorway (and holding the only face); child B queuing behind them.
    person.sees({1: PERSON_BOX, 2: BEHIND_BOX})
    for tick in range(4):  # 0.000 .. 0.375 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 1, "child A's Unknown session did not open"

    # A walks in. B steps forward into A's spot and we finally get a look at them.
    person.sees({2: PERSON_BOX})
    for tick in range(4, 20):  # 0.500 .. 2.375 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 2, (
        "the dedup ate a real child's meal session. They stood where the last child stood, "
        "and soon after -- but they were in shot TOGETHER, so they cannot be one child under "
        "two track ids, and this one ate and will be reported as having not"
    )


def test_a_child_who_joins_the_queue_AFTER_the_last_session_opened_is_not_eaten(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The test above passes for the wrong reason, and this is the one that catches it.

    There, both children are in shot from tick 0. That is the SINGLE arrangement in which
    `UnknownGuard` was safe -- because `opened()` froze a snapshot of the previous track's
    Sighting, `note()` rebinds a fresh frozen Sighting every frame, and so the stored
    `last_seen` never advanced past the moment the session opened. The co-visibility test
    therefore only ever asked "was this track already in shot when the LAST session opened?"

    It was blind to co-visibility that begins one frame later.

    So: child A is alone at the door, unrecognised, and opens an Unknown session. Child B
    walks up BEHIND them a moment afterwards, stands beside them in plain sight, and steps
    into the doorway when A goes in. Every geometric test says "same child". ByteTrack had
    them in the same frame, so they cannot be -- but the frozen snapshot never saw it.

    B ate. B is reported as not having eaten.
    """
    recognizer = FakeRecognizer()
    recognizer.show(vector(99))
    person = FakePersonDetector()
    pipeline = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person)

    # Child A alone in the doorway. Their Unknown session opens with nobody else in shot.
    person.sees({1: PERSON_BOX})
    for tick in range(4):  # 0.000 .. 0.375 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 1, "child A's Unknown session did not open"

    # NOW child B joins the queue, in shot beside A. This is what the frozen snapshot missed.
    person.sees({1: PERSON_BOX, 2: BEHIND_BOX})
    for tick in range(4, 8):  # 0.500 .. 0.875 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    # A walks in; B steps into exactly A's spot, still inside the cooldown.
    person.sees({2: PERSON_BOX})
    for tick in range(8, 24):  # 1.000 .. 2.875 s
        pipeline.on_frame(None, frame(at=at_fps(tick)))

    assert len(_sessions(session)) == 2, (
        "the dedup ate a real child's meal session. B stood in shot BESIDE A -- ByteTrack "
        "never gives one person two track ids at the same moment -- so they cannot be one "
        "child under two ids. B ate, and will be reported as having not."
    )


# -- defect 3: the exit gets ONE close attempt, and close() can refuse ----------


def test_an_exit_close_refused_as_too_young_is_retried_until_it_succeeds(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**An act that is REFUSED is not an act that is DONE.**

    `should_act` fires once per track, and at the entry that is right: `open()` is terminal.
    At the exit it is not. `close()` refuses a session younger than 30 s, because the exit
    camera is pointed at the backs of heads and the back it most often sees is a child who
    has just walked IN.

    So a child recognised at the exit while their session is young spends their one attempt
    on a refusal. They then stay in frame well past 30 s -- and never get another go. Their
    session silently never closes, and is force-closed as UNKNOWN 90 minutes later.

    The retry must also stop the moment it succeeds: at most one CLOSE per track, ever.
    """
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    entry = build_pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer)
    exit_pipe = build_pipeline(CameraRole.CANTEEN_EXIT, rows, recognizer)

    entry.on_frame(None, frame(at=at_fps(0)))
    assert len(_sessions(session)) == 1, "the fixture never opened a session to close"

    # The pupil is recognised at the exit while their session is seconds old. close() refuses.
    alerts = [exit_pipe.on_frame(None, frame(at=at_fps(tick))) for tick in range(8)]
    assert alerts.count("alert") == 0
    row = _sessions(session)[0]
    assert row.state is SessionState.OPEN, "a session younger than 30 s must never close"

    # They linger in front of the exit camera, and the session ages past the 30 s guard. The
    # session's AGE is what changes here -- so the row is aged, not the clock. Nothing sleeps.
    row.opened_at = row.opened_at - timedelta(seconds=60)
    session.commit()

    alerts += [exit_pipe.on_frame(None, frame(at=1.0 + at_fps(tick))) for tick in range(16)]

    row = _sessions(session)[0]
    assert row.state is SessionState.CLOSED, (
        "the one close attempt was spent on a TRANSIENT refusal and never repeated. This "
        "child stood at the exit for two more seconds and walked out with an open session, "
        "which force-closes as UNKNOWN in 90 minutes"
    )
    assert alerts.count("alert") == 1, "the session was closed more than once"

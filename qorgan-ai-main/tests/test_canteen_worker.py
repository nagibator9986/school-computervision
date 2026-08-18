"""The canteen worker: entry opens, exit closes, inside only confirms.

Driven with a fake recogniser, so the ROLE logic — which is where the legacy's 6 330-line
function went wrong — is testable without a GPU.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import BindingSettings
from qorgan.db.models import CanteenSession, RecognitionAttempt
from qorgan.enums import CameraRole, SessionState
from qorgan.settings import Settings
from tests.canteen_fakes import (
    FakePersonDetector,
    FakeRecognizer,
    build_pipeline,
    canteen_rows,
    frame,
    vector,
)

# The shared canteen fakes, under the names this module's tests have always used.
_pipeline = build_pipeline
_frame = frame
_vector = vector


@pytest.fixture
def rows(session: Session) -> dict:
    return canteen_rows(session)


# -- entry -------------------------------------------------------------------


def test_the_entry_camera_opens_a_session(settings: Settings, session: Session, rows: dict) -> None:
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.person_id == rows["pupil"].id
    assert row.state is SessionState.OPEN


def test_a_cook_walking_in_does_not_open_a_meal_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """Counting staff as pupils is how a cook ends up on the 'did not eat' report."""
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["cook"])

    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    assert session.scalars(select(CanteenSession)).all() == []


def test_an_unrecognised_child_still_opens_an_unknown_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The canteen must keep working with an empty or failing gallery: it records Unknown
    sessions rather than nothing at all."""
    recognizer = FakeRecognizer()
    recognizer.show(_vector(99))  # a stranger

    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.person_id is None


# -- exit --------------------------------------------------------------------


def test_the_exit_camera_closes_a_session(settings: Settings, session: Session, rows: dict) -> None:
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    entry = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer)
    exit_pipe = _pipeline(CameraRole.CANTEEN_EXIT, rows, recognizer)

    entry.on_frame(None, _frame(at=0.0))
    # Reach past the 30-second guard by ageing the session itself.
    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    from datetime import timedelta

    row.opened_at = row.opened_at - timedelta(minutes=10)
    session.commit()

    exit_pipe.on_frame(None, _frame(at=100.0))

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.state is SessionState.CLOSED


def test_the_exit_camera_never_opens_a_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """Only the entry camera opens. If the exit camera could, a child walking out would
    be recorded as walking in."""
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    _pipeline(CameraRole.CANTEEN_EXIT, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    assert session.scalars(select(CanteenSession)).all() == []


# -- inside ------------------------------------------------------------------


def test_an_inside_camera_never_opens_a_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    _pipeline(CameraRole.CANTEEN_INSIDE, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    assert session.scalars(select(CanteenSession)).all() == []


def test_an_inside_camera_confirms_a_pupil_who_has_a_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])

    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame(at=0.0))
    _pipeline(CameraRole.CANTEEN_INSIDE, rows, recognizer).on_frame(None, _frame(at=100.0))

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.state is SessionState.INSIDE_CONFIRMED


def test_an_inside_camera_never_attaches_a_pupil_to_somebody_elses_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The legacy's resolve_exit_session attached a recognised pupil to somebody ELSE'S
    oldest Unknown session, handing them another child's dwell time and meal status.

    An inside camera sees a face, not a journey. Nothing links this child to any
    particular Unknown session, so picking one is a guess dressed up as data.
    """
    recognizer = FakeRecognizer()

    # A stranger walks in: an Unknown session is opened for them.
    recognizer.show(_vector(99))
    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame(at=0.0))

    # Now a KNOWN pupil is seen inside. They must NOT be bolted onto that Unknown session.
    recognizer.show(rows["faces"]["pupil"])
    _pipeline(CameraRole.CANTEEN_INSIDE, rows, recognizer).on_frame(None, _frame(at=100.0))

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.person_id is None, "a pupil was attached to a stranger's session"


# -- calibration data --------------------------------------------------------


def test_every_recognition_attempt_is_recorded(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The data the legacy never had. It tuned eighteen overlapping thresholds by feel,
    and 1816 of its 1820 canteen records ended up with student_id = NULL -- with no
    record of WHY any of them failed."""
    recognizer = FakeRecognizer()
    recognizer.show(_vector(99))  # a stranger: this will fail to match

    _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame())

    session.expire_all()
    attempt = session.scalars(select(RecognitionAttempt)).one()
    assert attempt.accepted is False
    assert attempt.reason == "low_score"
    assert attempt.top1_score is not None
    assert attempt.face_width == 90


def test_nothing_happens_when_there_is_no_face(
    settings: Settings, session: Session, rows: dict
) -> None:
    recognizer = FakeRecognizer()
    recognizer.show_nobody()

    status = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer).on_frame(None, _frame())

    assert status == "ok"
    session.expire_all()
    assert session.scalars(select(CanteenSession)).all() == []


# -- once per track, not once per frame --------------------------------------

# Three attempts, a one-second backoff between them, and a three-second TTL. Those numbers
# do not fit inside each other, and that is the point: a child can walk out of shot while
# still RETRYING. See `test_a_fast_walker...` below.
RETRYING = BindingSettings(
    min_face_frames=1,
    max_wait_seconds=1.5,
    max_attempts=3,
    retry_backoff_seconds=1.0,
    track_ttl_seconds=3.0,
)


def test_a_child_at_the_door_is_embedded_once_not_once_per_frame(
    settings: Settings, session: Session, rows: dict
) -> None:
    """The old worker called detect() -- detection AND the embedding -- on every due
    frame. For five children queuing over ten seconds that is ~200 embeddings (spec §4.4)."""
    recognizer = FakeRecognizer()
    recognizer.show(rows["faces"]["pupil"])
    pipeline = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer)

    for tick in range(40):
        pipeline.on_frame(None, _frame(at=tick * 0.5))

    assert recognizer.embed_calls == 1

    session.expire_all()
    assert len(session.scalars(select(CanteenSession)).all()) == 1


def test_a_child_we_never_recognise_opens_ONE_unknown_session_and_logs_every_attempt(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**The child the gallery cannot name still ate.**

    Every embedding is rejected, we run out of attempts, and the track is EXHAUSTED. That
    is a verdict, not a non-event: exactly one Unknown session, not one per frame and not
    none at all.

    And every one of those failed attempts is on the record. `RecognitionAttempt` is the
    instrument that measures `min_score`'s unmeasured ceiling, and a table that contains
    only its successes measures nothing.
    """
    recognizer = FakeRecognizer()
    recognizer.show(_vector(99))  # a stranger: matches nobody in the gallery
    pipeline = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, binding=RETRYING)

    for tick in range(40):
        pipeline.on_frame(None, _frame(at=tick * 0.5))

    assert recognizer.embed_calls == RETRYING.max_attempts, "the GPU was burned forever"

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.person_id is None, "an unrecognised child must still be recorded as having eaten"

    attempts = session.scalars(select(RecognitionAttempt)).all()
    assert len(attempts) == RETRYING.max_attempts, "the attempt log is once per EMBED"
    assert all(a.accepted is False for a in attempts)
    assert all(a.reason == "low_score" for a in attempts), (
        "the attempt log has become a table of successes only, and it can no longer say "
        "WHY anything failed -- which is the entire reason it exists"
    )


def test_a_fast_walker_lost_mid_retry_still_opens_an_unknown_session(
    settings: Settings, session: Session, rows: dict
) -> None:
    """**The hole that per-track binding opens, and this closes.**

    A child is at the door for a second and a half. We get two bad looks, reject both, and
    are waiting out a backoff before the third -- and then they are gone. Three attempts a
    second apart do not fit into the time this child was in shot.

    RETRYING is not a verdict. If the track is simply forgotten, this child walked in, ate,
    and left no record at all -- and is then absent from the 'did not eat' report, which is
    the one report the school actually asked for. A hole we can count beats a child who
    silently never ate.
    """
    recognizer = FakeRecognizer()
    recognizer.show(_vector(99))
    person = FakePersonDetector()
    pipeline = _pipeline(CameraRole.CANTEEN_ENTRY, rows, recognizer, person, binding=RETRYING)

    # 1.5 s at the door: two rejected embeddings, and a third that never comes.
    for tick in range(4):
        pipeline.on_frame(None, _frame(at=tick * 0.5))

    assert recognizer.embed_calls == 2
    assert recognizer.embed_calls < RETRYING.max_attempts, (
        "this fixture no longer reproduces the fast walker: the track was EXHAUSTED while "
        "still in shot, so it never dies mid-retry and the test proves nothing"
    )
    session.expire_all()
    assert session.scalars(select(CanteenSession)).all() == [], "nothing is decided yet"

    # ...and they walk on, and the track dies while still RETRYING.
    person.walk_away()
    for tick in range(20):
        pipeline.on_frame(None, _frame(at=2.0 + tick * 0.5))

    session.expire_all()
    row = session.scalars(select(CanteenSession)).one()
    assert row.person_id is None, (
        "the fast walker vanished without a trace: a child walked in and the system has no "
        "record they were ever there"
    )

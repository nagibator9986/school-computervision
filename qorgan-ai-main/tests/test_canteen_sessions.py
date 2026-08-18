"""The meal-session state machine. These rules were earned in a real canteen doorway."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.canteen.sessions import SessionManager
from qorgan.config.canteen import MealOutcomeRules, SessionRules
from qorgan.db.models import Camera, CanteenSession, Person
from qorgan.db.types import utcnow
from qorgan.enums import (
    CameraRole,
    CameraType,
    CloseReason,
    IdentitySource,
    PersonType,
    SessionOutcome,
    SessionState,
)

NOW = utcnow()


@pytest.fixture
def cameras(session: Session) -> tuple[int, int]:
    entry = Camera(
        name="canteen_entry",
        display_name="Вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    exit_camera = Camera(
        name="canteen_exit",
        display_name="Выход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_EXIT,
        rtsp_host="10.0.0.2",
    )
    session.add_all([entry, exit_camera])
    session.commit()
    return entry.id, exit_camera.id


@pytest.fixture
def manager(cameras: tuple[int, int]) -> SessionManager:
    entry_id, exit_id = cameras
    return SessionManager(SessionRules(), MealOutcomeRules(), entry_id, exit_id)


@pytest.fixture
def pupil(session: Session) -> Person:
    person = Person(
        external_id="gen-alice",
        full_name="Петрова Мария",
        person_type=PersonType.STUDENT,
        class_name="5А",
    )
    session.add(person)
    session.commit()
    return person


@pytest.fixture
def cook(session: Session) -> Person:
    person = Person(external_id="gen-cook", full_name="Азимов Атабек", person_type=PersonType.STAFF)
    session.add(person)
    session.commit()
    return person


# -- entry -------------------------------------------------------------------


def test_a_pupil_walking_in_opens_a_session(manager: SessionManager, pupil: Person) -> None:
    result = manager.open(pupil.id, NOW)

    assert result.opened
    assert result.session_id is not None


def test_a_session_is_a_row_that_survives_a_restart(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    """THE fix. Legacy sessions lived in a RAM dict inside a module-global singleton, so
    a process restart silently lost every open session -- every child who had walked in
    but not yet walked out simply ceased to exist."""
    result = manager.open(pupil.id, NOW)

    session.expire_all()
    row = session.get(CanteenSession, result.session_id)

    assert row is not None
    assert row.state is SessionState.OPEN
    assert row.person_id == pupil.id
    assert row.identity_source is IdentitySource.ENTRY


def test_staff_never_open_a_meal_session(manager: SessionManager, cook: Person) -> None:
    """Counting a cook as a pupil is how they end up on the 'did not eat' report."""
    result = manager.open(cook.id, NOW, is_staff=True)

    assert not result.opened
    assert result.reason == "staff_do_not_open_sessions"


def test_an_unrecognised_face_still_opens_a_session(manager: SessionManager) -> None:
    """The system must run with zero pupils imported: the canteen records Unknown
    sessions until a face registry exists."""
    result = manager.open(None, NOW)

    assert result.opened


def test_one_child_lingering_in_the_doorway_is_one_meal_not_forty(
    manager: SessionManager, pupil: Person
) -> None:
    first = manager.open(pupil.id, NOW)
    second = manager.open(pupil.id, NOW + timedelta(seconds=5))

    assert first.opened
    assert not second.opened
    assert second.reason == "cooldown"


def test_a_pupil_already_inside_does_not_open_a_second_session(
    manager: SessionManager, pupil: Person
) -> None:
    manager.open(pupil.id, NOW)
    again = manager.open(pupil.id, NOW + timedelta(minutes=5))  # past the cooldown

    assert not again.opened
    assert again.reason == "already_inside"


# -- the config must not advertise a cooldown no code applies ----------------


def test_there_is_no_exit_cooldown_knob_because_nothing_applies_one() -> None:
    """`SessionRules.exit_cooldown_seconds` was read only inside
    `_cooldown_block(entering=False)`, whose one caller (`open`) always passes
    `entering=True`; `close()` never called it, so the knob applied to nothing.

    An exit cooldown has no work to do that the state machine does not already do: a
    person has at most one open session, a closed session never reopens, and a close with
    no open session is a no-op -- so an exit cannot be double-counted. Worse, the dead
    branch filtered `opened_at`, so had it ever run an 'exit cooldown' would have blocked a
    CLOSE because a session was recently OPENED (the wrong column). A knob nothing applies
    is a lie the config file tells whoever tunes next.
    """
    import inspect

    assert "exit_cooldown_seconds" not in SessionRules.model_fields
    # `close()` is simpler: the cooldown helper no longer carries the dead entry/exit split.
    assert "entering" not in inspect.signature(SessionManager._cooldown_block).parameters


# -- the exit camera looks at the backs of heads -----------------------------


def test_the_exit_camera_will_not_close_a_session_that_just_opened(
    manager: SessionManager, pupil: Person
) -> None:
    """The exit camera sees the BACK of a child who has just walked IN. Without this rule
    a pupil is marked as having left the moment they arrive -- and that single confusion
    would corrupt the entire meal record."""
    manager.open(pupil.id, NOW)

    result = manager.close(pupil.id, NOW + timedelta(seconds=2))

    assert not result.closed
    assert result.reason == "session_too_young"


def test_a_pupil_who_really_leaves_is_closed(manager: SessionManager, pupil: Person) -> None:
    manager.open(pupil.id, NOW)

    result = manager.close(pupil.id, NOW + timedelta(minutes=15))

    assert result.closed
    assert result.outcome is SessionOutcome.ATE


def test_closing_a_pupil_who_never_entered_does_nothing(
    manager: SessionManager, pupil: Person
) -> None:
    result = manager.close(pupil.id, NOW)

    assert not result.closed
    assert result.reason == "no_open_session"


# -- ate / did not eat -------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (35, SessionOutcome.NOT_ATE),  # in the door, out again: no time to eat
        (59, SessionOutcome.NOT_ATE),
        (60, SessionOutcome.ATE),
        (900, SessionOutcome.ATE),
    ],
)
def test_the_outcome_follows_the_dwell_time(
    manager: SessionManager, pupil: Person, seconds: int, expected: SessionOutcome
) -> None:
    """The ladder the state machine can reach: under 60s did not eat, 60s+ ate. The exit
    guard means no session ever closes younger than 30s, so those two bands are the whole
    of it -- the old '<20s came in and left' band was never reachable."""
    manager.open(pupil.id, NOW)
    result = manager.close(pupil.id, NOW + timedelta(seconds=seconds))

    assert result.closed
    assert result.outcome is expected


def test_the_meal_ladder_itself() -> None:
    rules = MealOutcomeRules()
    assert rules.classify(30) == "not_ate"
    assert rules.classify(59) == "not_ate"
    assert rules.classify(120) == "ate"


# -- the exit's 30-second guard is absolute: nothing supplies a bypass --------


def test_a_young_session_stays_open_and_there_is_no_flag_to_force_it(
    manager: SessionManager, pupil: Person
) -> None:
    """The safe default is the only default. The exit camera catching a just-entered
    child's face must not close their session, and there is no quick-return flag to force
    it: the strong evidence such a flag would demand is a departure signal this pipeline
    does not produce, so the guard holds until the session is old enough."""
    manager.open(pupil.id, NOW)

    result = manager.close(pupil.id, NOW + timedelta(seconds=12))

    assert not result.closed
    assert result.reason == "session_too_young"


def test_the_came_in_and_left_outcome_is_gone_because_production_can_never_reach_it() -> None:
    """`close(quick_return=True)` was the ONLY path to closing a session younger than the
    30-second guard, and therefore the ONLY path to `SessionOutcome.LEFT_IMMEDIATELY`,
    `CloseReason.QUICK_RETURN`, and the `< left_immediately_below_seconds` "came in and
    left" band.

    Production closes with `close(person_id, utcnow())` (worker/canteen.py::_on_exit) and
    never passes the flag, so every one of those was unreachable: an outcome, a reason, and
    two config knobs that could never fire. Wiring it needs a genuine-departure signal --
    "this recognised face really is leaving, not a just-entered child glancing back" -- and
    the exit pipeline produces no such signal; the only derivable trigger is the exact
    ambiguous case the guard exists to reject. So the honest fix deletes the vocabulary
    rather than fake a trigger. This test is the headstone.
    """
    import inspect

    assert not hasattr(SessionOutcome, "LEFT_IMMEDIATELY")
    assert not hasattr(CloseReason, "QUICK_RETURN")
    assert "quick_return" not in inspect.signature(SessionManager.close).parameters
    assert "left_immediately_below_seconds" not in MealOutcomeRules.model_fields
    assert "quick_return_enabled" not in SessionRules.model_fields
    assert "quick_return_max_age_seconds" not in SessionRules.model_fields


# -- inside cameras ----------------------------------------------------------


def test_an_inside_camera_confirms_presence(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    opened = manager.open(pupil.id, NOW)

    assert manager.confirm_inside(opened.session_id)

    session.expire_all()
    assert session.get(CanteenSession, opened.session_id).state is SessionState.INSIDE_CONFIRMED


def test_an_inside_camera_can_name_an_unknown_session(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    """A session opened as Unknown, resolved later by a camera that got a better look."""
    opened = manager.open(None, NOW)

    assert manager.late_bind(opened.session_id, pupil.id, score=0.71)

    session.expire_all()
    row = session.get(CanteenSession, opened.session_id)
    assert row.person_id == pupil.id
    assert row.identity_source is IdentitySource.INSIDE_LATE_BIND


def test_a_late_bind_never_overwrites_an_identity_we_already_had(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    """The legacy's resolve_exit_session attached a recognised pupil to somebody ELSE'S
    oldest Unknown session, handing them another child's dwell time and meal status."""
    other = Person(external_id="gen-other", full_name="Другой", person_type=PersonType.STUDENT)
    session.add(other)
    session.commit()

    opened = manager.open(pupil.id, NOW)

    assert not manager.late_bind(opened.session_id, other.id, score=0.9)

    session.expire_all()
    assert session.get(CanteenSession, opened.session_id).person_id == pupil.id


# -- the janitor -------------------------------------------------------------


def test_a_session_nobody_exited_is_force_closed(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    """Better to record honestly that we lost track of a child than to leave a session
    open for the rest of the year, quietly blocking every future meal for them."""
    opened = manager.open(pupil.id, NOW)

    closed = manager.force_close_stale(NOW + timedelta(minutes=120))

    assert closed == 1
    session.expire_all()
    row = session.get(CanteenSession, opened.session_id)
    assert row.state is SessionState.CLOSED
    assert row.outcome is SessionOutcome.UNKNOWN
    assert row.close_reason is CloseReason.TIMEOUT


def test_a_fresh_session_is_not_force_closed(manager: SessionManager, pupil: Person) -> None:
    manager.open(pupil.id, NOW)

    assert manager.force_close_stale(NOW + timedelta(minutes=10)) == 0


def test_a_force_closed_pupil_can_eat_again_tomorrow(
    manager: SessionManager, pupil: Person, session: Session
) -> None:
    """The point of the force-close: a stuck session must not block the child forever."""
    manager.open(pupil.id, NOW)
    manager.force_close_stale(NOW + timedelta(minutes=120))

    tomorrow = manager.open(pupil.id, NOW + timedelta(days=1))

    assert tomorrow.opened
    session.expire_all()
    assert session.scalars(select(CanteenSession)).all().__len__() == 2

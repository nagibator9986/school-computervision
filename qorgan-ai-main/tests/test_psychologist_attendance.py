"""One named child's canteen attendance, week by week — the only longitudinal signal in
this system that is honestly about a named child.

It is honest because of where the identity comes from: the canteen recognises a face at a
door, at conversational distance, against a roster the school issued. It needs none of the
classroom identification §8 forbids, and none of the corridor recognition this school's
own footage measured at zero.

**The assertions here are about counting and about silence.** Counting, because
`days_present` and `sessions` are different numbers and a page that confused them would
turn a recognition glitch into an extra meal. Silence, because the module must not draw a
conclusion from the counts however suggestive they are — the last test builds the exact
shape a person would call alarming («ходил каждый день и перестал») and asserts that the
page says nothing about it at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Person
from qorgan.enums import SessionOutcome, UserRole
from qorgan.psychologist.attendance import WEEKS, attendance_trend
from qorgan.psychologist.signals import SignalState
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.psychologist_fakes import (
    ClientFor,
    a_camera,
    a_meal,
    a_pupil,
    an_unattributed_meal,
    client_factory,
)


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    yield from client_factory(app, session)


@pytest.fixture
def camera(session: Session) -> Camera:
    return a_camera(session)


@pytest.fixture
def pupil(session: Session) -> Person:
    return a_pupil(session)


# -- the counting ------------------------------------------------------------


def test_a_child_with_no_record_gets_every_week_and_a_reason(
    pupil: Person, session: Session
) -> None:
    """**EMPTY here is a statement about the CAMERA, not about the child**, and the page
    has to say so: a table of zeroes with no explanation reads as "this child does not
    eat", which is a claim nobody measured."""
    del session
    trend = attendance_trend(pupil.id)

    assert trend is not None
    assert len(trend.weeks) == WEEKS
    assert trend.total_sessions == 0
    assert trend.state is SignalState.EMPTY
    assert any("камеру столовой ещё не перевесили" in line for line in trend.caveat())


def test_two_sessions_on_one_day_are_one_day_present(
    pupil: Person, camera: Camera, session: Session
) -> None:
    """**The distinction `days_present` exists for.** A child recognised twice at one lunch
    has not attended twice, and counting sessions as days would make a recognition glitch
    look like an extra meal."""
    a_meal(session, pupil, camera, days_ago=0.0)
    a_meal(session, pupil, camera, days_ago=0.01)

    this_week = attendance_trend(pupil.id).weeks[-1]
    assert this_week.sessions == 2
    assert this_week.days_present == 1


def test_meals_recorded_counts_only_the_sessions_that_closed_as_eaten(
    pupil: Person, camera: Camera, session: Session
) -> None:
    """Shown BESIDE `sessions` and never instead of it: `UNKNOWN` outcomes are ordinary,
    and presenting only the confirmed meals would understate the record."""
    a_meal(session, pupil, camera, outcome=SessionOutcome.ATE)
    a_meal(session, pupil, camera, days_ago=0.02, outcome=SessionOutcome.UNKNOWN)

    this_week = attendance_trend(pupil.id).weeks[-1]
    assert this_week.sessions == 2
    assert this_week.meals_recorded == 1


def test_another_childs_meals_do_not_appear_in_this_childs_trend(
    pupil: Person, camera: Camera, session: Session
) -> None:
    """The whole value of this page is that it is about ONE named child."""
    other = a_pupil(session, "student_471")
    a_meal(session, other, camera)
    an_unattributed_meal(session, camera)

    assert attendance_trend(pupil.id).total_sessions == 0
    assert attendance_trend(other.id).total_sessions == 1


def test_the_weeks_run_oldest_first_and_end_with_this_one(
    pupil: Person, camera: Camera, session: Session
) -> None:
    """A person reads a trend left to right. Newest-first would invert every impression the
    table gives without changing a single number in it."""
    a_meal(session, pupil, camera, days_ago=0.0)

    trend = attendance_trend(pupil.id)
    starts = [week.starts_on for week in trend.weeks]

    assert starts == sorted(starts), "the weeks are not in order"
    assert trend.weeks[-1].sessions == 1, "today landed outside the last bucket"
    assert trend.weeks[0].sessions == 0


def test_a_session_older_than_the_window_is_outside_the_table_but_still_counted(
    pupil: Person, camera: Camera, session: Session
) -> None:
    """`total_sessions` and `first_record` are the page's way of saying "there is more
    history than this table shows". Without them an eight-week window silently truncates a
    child's record into a claim about their whole time at the school."""
    a_meal(session, pupil, camera, days_ago=WEEKS * 7 + 30)

    trend = attendance_trend(pupil.id)
    assert sum(week.sessions for week in trend.weeks) == 0
    assert trend.total_sessions == 1
    assert trend.first_record is not None
    assert trend.state is SignalState.LIVE


def test_a_narrower_window_is_honoured_and_never_collapses_to_nothing(
    pupil: Person,
) -> None:
    """`weeks=0` would be a table with no rows, which renders as a page that has nothing to
    say rather than as a page asked the wrong question."""
    assert len(attendance_trend(pupil.id, weeks=2).weeks) == 2
    assert len(attendance_trend(pupil.id, weeks=0).weeks) == 1


# -- the page ----------------------------------------------------------------


def test_an_unknown_child_is_a_404_and_not_an_empty_trend(
    client_for: ClientFor,
) -> None:
    """Eight empty weeks under an id nobody holds reads as "this child stopped coming",
    which is a claim about a child who does not exist."""
    assert attendance_trend(4242) is None
    assert client_for(UserRole.PSYCHOLOGIST).get("/psychologist/pupils/4242").status_code == 404


def test_the_trend_asks_for_a_superset_of_the_sessions_page(
    client_for: ClientFor, pupil: Person, camera: Camera, session: Session
) -> None:
    """It shows the same child's meal record in a different shape, so it must never be a
    second door into it: anybody who can open the trend can already open the sessions. The
    psychologist ROLE was widened for this rather than the PAGE narrowed."""
    a_meal(session, pupil, camera)
    psychologist = client_for(UserRole.PSYCHOLOGIST)

    assert psychologist.get(f"/psychologist/pupils/{pupil.id}").status_code == 200
    assert psychologist.get(f"/pupils/{pupil.id}/canteen").status_code == 200

    canteen_staff = client_for(UserRole.CANTEEN_STAFF)
    assert canteen_staff.get(f"/psychologist/pupils/{pupil.id}").status_code == 403


def test_the_empty_weeks_are_drawn_rather_than_skipped(
    client_for: ClientFor, pupil: Person, camera: Camera, session: Session
) -> None:
    """A table that omitted its zero weeks would hide exactly the thing a person is looking
    for -- and the system must not be the one that points at it either."""
    a_meal(session, pupil, camera, days_ago=0.0)

    body = client_for(UserRole.PSYCHOLOGIST).get(f"/psychologist/pupils/{pupil.id}").text
    trend = attendance_trend(pupil.id)

    for week in trend.weeks:
        assert str(week.starts_on) in body, f"week of {week.starts_on} is missing"


def _weeks_of_meals(session: Session, pupil: Person, camera: Camera, weeks: range) -> None:
    for week in weeks:
        for day in range(5):
            a_meal(session, pupil, camera, days_ago=week * 7 + day)


def _markup_without_the_numbers(body: str) -> str:
    """The page with every digit removed, so two renderings differ only where the PAGE
    decided something -- a class, a tag, a word -- rather than where the data did."""
    return "".join(character for character in body if not character.isdigit())


def test_the_shape_a_person_would_call_alarming_renders_the_same_page_as_a_flat_one(
    client_for: ClientFor, pupil: Person, camera: Camera, session: Session
) -> None:
    """**«Ходил обедать каждый день и перестал», built exactly** -- the pattern §13 asks
    the psychologist to notice, and the one the system must not notice for them.

    Asserted by rendering it beside a child who ate all eight weeks and comparing the two
    pages with the digits stripped out. If anything on this page were conditional on the
    numbers -- a colour, a tag, an arrow, a word like «падение» -- the two would differ.
    They do not, so the page states counts and the reader draws the conclusion.

    This is stronger than grepping for forbidden words, and it is why it replaced that:
    a word list only catches the vocabulary somebody thought of, and `caveat()` legitimately
    contains «нормой» and «падение» in the sentence that says the system does neither.
    """
    other = a_pupil(session, "student_471")
    _weeks_of_meals(session, pupil, camera, range(3, 7))  # four weeks on, then silence
    _weeks_of_meals(session, other, camera, range(0, 7))  # every week, unremarkable

    # Asserted against the WINDOW rather than against fixed indices: the buckets are
    # Mondays, so which index a "21 days ago" meal lands in depends on what day the suite
    # is run. What does not depend on it is that twenty meals are inside the eight weeks
    # and that the most recent fortnight is empty -- which is the shape under test.
    trend = attendance_trend(pupil.id)
    assert trend.total_sessions == 20
    assert sum(week.days_present for week in trend.weeks) == 20
    assert [week.days_present for week in trend.weeks[-2:]] == [0, 0]

    psychologist = client_for(UserRole.PSYCHOLOGIST)
    alarming = psychologist.get(f"/psychologist/pupils/{pupil.id}").text
    flat = psychologist.get(f"/psychologist/pupils/{other.id}").text

    assert _markup_without_the_numbers(alarming) == _markup_without_the_numbers(flat), (
        "the page renders differently for a child whose attendance stopped -- something "
        "here is deciding, and §8 promised the school that nothing would"
    )

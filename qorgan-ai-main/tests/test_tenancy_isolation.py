"""Two schools in one database, and every attempt this suite can make to read across.

**This test proves isolation by TRYING, which is the half `test_tenancy_guard` cannot do.**
The guard is an AST scan: it proves a statement NAMED `school_id`, never that the join it
named was the right one. An event scoped through `persons` instead of through `cameras`
satisfies the guard completely and still hands one school's incidents to another. So the
two tests fail on opposite mistakes and neither is sufficient alone -- one catches the
query nobody wrote a filter for, this one catches the filter that does not filter.

**IT GOES IN THROUGH THE FRONT DOOR.** Every read below is an HTTP request made by a
logged-in `TestClient` against the real app, with the real middleware, the real capability
checks and a real session cookie -- not a call to the read-model function underneath. That
is deliberate and it is this repository's most expensive recurring lesson: a `/media`
traversal test passed with the defence sabotaged because httpx normalised the path before
sending it; every login test passed while logging in from a browser was impossible; and a
"the event can be reviewed" test was green while the page had stopped drawing the button.
Three times in two days, a test asserted something no human could reach. A tenancy test
that called `pupil_page(school_id=...)` directly would assert that a function honours an
argument -- which is not the question. The question is what a person sees.

**THE FIXTURE CANNOT BE LENIENT, BY CONSTRUCTION.** `school_id` falls back to "the only
school there is" and RAISES when there are several (`db/models/school.py`). This test
creates a second school in `setup`, so from that line on every insert must name its school
and every un-plumbed read path raises instead of guessing. The default that makes the rest
of the suite readable is switched off here, and it is switched off by the same mechanism
production would use.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import (
    Camera,
    CanteenSession,
    Event,
    Lesson,
    LessonTrack,
    MealWindow,
    Notification,
    Person,
    School,
    User,
)
from qorgan.db.types import utcnow
from qorgan.enums import (
    CameraRole,
    CameraType,
    EventStatus,
    EventType,
    LessonState,
    MealKind,
    NotificationChannel,
    NotificationStatus,
    PersonType,
    SessionOutcome,
    SessionState,
    Severity,
    UserRole,
)
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.conftest import DEFAULT_SCHOOL_SLUG
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

# **Both schools enrol a pupil numbered 7, and that is the point of this constant.**
# `persons.external_id` was globally UNIQUE until migration 0009. Under one school that is
# correct and invisible; under two it is wrong in a way that produces no error a school can
# act on -- the second school imports its roster and half of it is rejected as duplicates of
# children it has never met. If the composite unique regressed to a global one, the second
# `_enrol` below raises IntegrityError and every test in this file dies at setup.
SHARED_EXTERNAL_ID = "7"

MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)

OURS, THEIRS = "ALPHA", "BETA"


@dataclass(frozen=True, slots=True)
class Tenant:
    """One school and one of everything it owns, with the ids needed to ask for them."""

    school_id: int
    marker: str
    camera_id: int
    person_id: int
    event_id: int
    lesson_id: int
    username: str


def _camera(session: Session, school_id: int, marker: str) -> Camera:
    row = Camera(
        school_id=school_id,
        # The SAME name in both schools. `cameras.name` was globally unique too, and a
        # school naming its own hall camera `hall_left` must not depend on whether another
        # school got there first.
        name="hall_left",
        display_name=f"Холл {marker}",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.flush()
    return row


def _enrol(session: Session, school_id: int, marker: str) -> Person:
    row = Person(
        school_id=school_id,
        external_id=SHARED_EXTERNAL_ID,
        full_name=f"Ученик {marker}",
        person_type=PersonType.STUDENT,
        class_name=f"5-{marker[0]}",
    )
    session.add(row)
    session.flush()
    return row


def _incident(session: Session, camera: Camera, marker: str) -> Event:
    row = Event(
        camera_id=camera.id,
        event_type=EventType.BULLYING,
        occurred_at=MOMENT,
        confidence=0.93,
        candidate_probability=0.9,
        validation_score=0.8,
        skeleton_confirmed=True,
        severity=Severity.ALERT,
        summary_text=f"Зафиксирована агрессия {marker}",
        track_ids="3,7",
        status=EventStatus.NEW,
    )
    session.add(row)
    session.flush()
    session.add(
        Notification(
            event_id=row.id,
            channel=NotificationChannel.TELEGRAM,
            status=NotificationStatus.FAILED,
            attempts=6,
            last_error=f"не доставлено {marker}",
        )
    )
    return row


def _meal(session: Session, school_id: int, camera: Camera, person: Person) -> None:
    """A meal eaten TODAY, so it lands inside the /canteen page's local day."""
    window = MealWindow(
        school_id=school_id,
        kind=MealKind.BREAKFAST,
        name="Завтрак",
        starts_at=time(8, 0),
        ends_at=time(9, 0),
    )
    session.add(window)
    session.flush()
    session.add(
        CanteenSession(
            person_id=person.id,
            meal_window_id=window.id,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=SessionOutcome.ATE,
            opened_at=utcnow(),
            dwell_seconds=420.0,
        )
    )


def _lesson(session: Session, school_id: int, marker: str) -> Lesson:
    """A classroom camera and one lesson on it, for `/lessons`.

    A SECOND camera, and its name carries the marker on purpose: `LessonReport` renders
    `Camera.name`, and the hall camera above is deliberately called `hall_left` in BOTH
    schools to prove the composite unique -- so it could never tell the two pages apart.
    """
    camera = Camera(
        school_id=school_id,
        name=f"class-{marker}",
        display_name=f"Кабинет {marker}",
        camera_type=CameraType.CLASSROOM,
        role=CameraRole.CLASSROOM,
        rtsp_host="10.0.0.2",
    )
    session.add(camera)
    session.flush()
    lesson = Lesson(
        camera_id=camera.id,
        state=LessonState.CLOSED,
        started_at=MOMENT,
        min_presence_seconds=1.0,
    )
    session.add(lesson)
    session.flush()
    session.add(
        LessonTrack(
            lesson_id=lesson.id,
            track_id=3,
            first_seen_at=MOMENT,
            last_seen_at=MOMENT,
            observed_seconds=120.0,
            observations=60,
            settled=True,
        )
    )
    return lesson


def _populate(session: Session, school_id: int, marker: str) -> Tenant:
    """One school, with one of every root kind and every kind derived from one."""
    camera = _camera(session, school_id, marker)
    person = _enrol(session, school_id, marker)
    event = _incident(session, camera, marker)
    _meal(session, school_id, camera, person)
    lesson = _lesson(session, school_id, marker)

    username = f"head-{marker.lower()}"
    session.add(
        User(
            school_id=school_id,
            username=username,
            password_hash=hash_password(PASSWORD),
            # ADMIN holds VIEW_BULLYING, VIEW_PUPILS, VIEW_CANTEEN and MANAGE_USERS, so one
            # account can try every page below. It does NOT hold MANAGE_SCHOOLS: a school's
            # headteacher is not a superadmin, and this test would be worth much less if the
            # account doing the reading were the one account meant to see everything.
            role=UserRole.ADMIN,
        )
    )
    session.flush()
    return Tenant(
        school_id=school_id,
        marker=marker,
        camera_id=camera.id,
        person_id=person.id,
        event_id=event.id,
        lesson_id=lesson.id,
        username=username,
    )


@pytest.fixture
def two_schools(session: Session) -> tuple[Tenant, Tenant]:
    """The installation `conftest` builds, plus a second school on the same database."""
    ours_id = session.scalar(select(School.id).where(School.slug == DEFAULT_SCHOOL_SLUG))
    assert ours_id is not None, "conftest is expected to create the first school"

    other = School(slug="gymnasium-4", name="Гимназия №4")
    session.add(other)
    session.flush()

    ours = _populate(session, int(ours_id), OURS)
    theirs = _populate(session, int(other.id), THEIRS)
    session.commit()
    return ours, theirs


@pytest.fixture
def client(
    settings: Settings, two_schools: tuple[Tenant, Tenant]
) -> Iterator[tuple[TestClient, Tenant, Tenant]]:
    """Logged in as OUR school's headteacher, through the real login form."""
    ours, theirs = two_schools
    with TestClient(create_app(), follow_redirects=False) as test_client:
        response = test_client.post(
            "/login",
            data=with_token(test_client, {"username": ours.username, "password": PASSWORD}),
        )
        assert response.status_code == 303, (
            f"could not log in as {ours.username}; every assertion below would then be "
            "passing because the page was a redirect to /login and contained nobody's data"
        )
        yield test_client, ours, theirs


# Every page a school's headteacher can open that renders another school's data if the
# query behind it is unscoped. The marker strings are what a human would actually read on
# the page -- a camera's display name, a child's name, an incident summary.
PAGES = ("/events", "/notifications", "/pupils", "/canteen", "/lessons")


@pytest.mark.parametrize("path", PAGES)
def test_a_page_shows_this_school_and_never_the_other(
    client: tuple[TestClient, Tenant, Tenant], path: str
) -> None:
    """The whole module in one assertion, repeated per page.

    Both halves matter. Asserting only that BETA is absent would pass on a page that is
    broken, empty, or a redirect to /login -- and a test that passes when the feature is
    gone is the failure mode this repository has paid for three times. So each page must
    also still show OUR school's row.
    """
    test_client, ours, theirs = client
    page = test_client.get(path)

    assert page.status_code == 200, f"{path} did not render for a headteacher of its school"
    assert ours.marker in page.text, (
        f"{path} does not show this school's own data, so its silence about the other "
        "school proves nothing -- an empty page hides everything equally."
    )
    assert theirs.marker not in page.text, (
        f"{path} rendered another school's data to {ours.username}. On this installation "
        "that is one school's children -- their names, their classes, or an incident "
        "involving them -- displayed to staff of a different school."
    )


def test_another_schools_pupil_is_not_reachable_by_guessing_the_id(
    client: tuple[TestClient, Tenant, Tenant],
) -> None:
    """The id is a small integer in a URL, so it is guessable by typing.

    404 and not 403: whether that id exists in some other school is not this school's
    business, and an answer that could be told apart would turn this URL into a directory
    of every child on the installation, one number at a time.
    """
    test_client, _, theirs = client
    page = test_client.get(f"/pupils/{theirs.person_id}/canteen")

    assert page.status_code == 404, (
        f"a headteacher reached another school's pupil (id {theirs.person_id}) and their "
        f"whole meal history by typing a number: HTTP {page.status_code}"
    )


def test_our_own_pupil_is_still_reachable_by_id(
    client: tuple[TestClient, Tenant, Tenant],
) -> None:
    """The control for the test above: a 404 for everything would also pass it."""
    test_client, ours, _ = client
    page = test_client.get(f"/pupils/{ours.person_id}/canteen")

    assert page.status_code == 200, (
        "this school's own pupil is no longer reachable, so the 404 above is not isolation "
        "working -- it is the page being broken for everybody."
    )
    assert ours.marker in page.text


def test_another_schools_lesson_is_not_reachable_by_guessing_the_id(
    client: tuple[TestClient, Tenant, Tenant],
) -> None:
    """`/lessons/{id}` takes a small integer, so it is reachable by typing.

    **This test exists because a sabotage found it missing.** The `/lessons` case in
    `PAGES` above exercises only the INDEX, and the index scopes its own id list -- so a
    filter broken inside `lesson_report` left every assertion in this file green while a
    lesson report of another school's classroom was one URL away. Covering the list page
    is not covering the page it links to.
    """
    test_client, _, theirs = client
    page = test_client.get(f"/lessons/{theirs.lesson_id}")

    assert page.status_code == 404, (
        f"a headteacher opened another school's lesson report (id {theirs.lesson_id}) by "
        f"typing a number: HTTP {page.status_code}"
    )


def test_our_own_lesson_is_still_reachable_by_id(
    client: tuple[TestClient, Tenant, Tenant],
) -> None:
    """The control: a 404 for every lesson would also pass the test above."""
    test_client, ours, _ = client
    page = test_client.get(f"/lessons/{ours.lesson_id}")

    assert page.status_code == 200, (
        "this school's own lesson is no longer reachable, so the 404 above is not "
        "isolation working -- it is the page being broken for everybody."
    )
    assert ours.marker in page.text


def test_another_schools_incident_cannot_be_ruled_on(
    client: tuple[TestClient, Tenant, Tenant], session: Session
) -> None:
    """A WRITE across the boundary, which is worse than a read.

    Reviewing stamps `reviewed_by_id` with the acting user, so a successful cross-tenant
    review writes one school's member of staff into another school's judgement about a
    named child -- and the detector's correction signal is fed from these verdicts.
    """
    test_client, _, theirs = client
    response = test_client.post(
        f"/events/{theirs.event_id}/review",
        data=with_token(test_client, {"verdict": EventStatus.CONFIRMED.value}),
    )

    assert response.status_code == 404, (
        f"a headteacher passed judgement on another school's incident: HTTP "
        f"{response.status_code}"
    )
    after = session.get(Event, theirs.event_id)
    session.refresh(after)
    assert after.status is EventStatus.NEW, "the other school's event was modified anyway"
    assert after.reviewed_by_id is None


def test_our_own_incident_can_still_be_ruled_on(
    client: tuple[TestClient, Tenant, Tenant], session: Session
) -> None:
    """The control: a review that 404s for everybody would pass the test above."""
    test_client, ours, _ = client
    response = test_client.post(
        f"/events/{ours.event_id}/review",
        data=with_token(test_client, {"verdict": EventStatus.CONFIRMED.value}),
    )

    assert response.status_code == 303, (
        "this school's own event could not be reviewed, so the 404 above proves nothing"
    )
    after = session.get(Event, ours.event_id)
    session.refresh(after)
    assert after.status is EventStatus.CONFIRMED


def test_the_accounts_page_lists_one_schools_staff(
    client: tuple[TestClient, Tenant, Tenant],
) -> None:
    """`users` is a root table, and the account list is how one school reaches another's.

    An account is not merely data about a person: it is the thing MANAGE_USERS can retire
    or demote. A headteacher who can see another school's administrators is one form
    submission away from locking that school out of its own system.
    """
    test_client, ours, theirs = client
    page = test_client.get("/users")

    assert page.status_code == 200
    assert ours.username in page.text, "the accounts page does not list this school's own staff"
    assert theirs.username not in page.text, (
        f"{ours.username} can see another school's account ({theirs.username}), and holds "
        "MANAGE_USERS over the page that lists it"
    )


def test_two_schools_may_both_enrol_a_pupil_numbered_seven(
    two_schools: tuple[Tenant, Tenant], session: Session
) -> None:
    """The constraint multi-school breaks quietly, asserted as a fact about the database.

    `two_schools` already proved it by inserting both without an IntegrityError. This says
    so out loud, and checks the two rows really are two people rather than one row read
    twice -- which is what a surviving global unique plus a silent upsert would look like.
    """
    ours, theirs = two_schools
    rows = session.scalars(
        select(Person).where(Person.external_id == SHARED_EXTERNAL_ID).order_by(Person.id)
    ).all()

    assert len(rows) == 2, (
        f"expected both schools to hold a pupil numbered {SHARED_EXTERNAL_ID}, found "
        f"{len(rows)}. If this is 1, `persons.external_id` is globally unique again and "
        "the second school's roster import silently rejects children it has never met."
    )
    assert {row.school_id for row in rows} == {ours.school_id, theirs.school_id}
    assert rows[0].id != rows[1].id


def test_two_schools_may_both_name_a_camera_hall_left(
    two_schools: tuple[Tenant, Tenant], session: Session
) -> None:
    """`cameras.name` was globally unique for the same reason and breaks the same way."""
    ours, theirs = two_schools
    rows = session.scalars(select(Camera).where(Camera.name == "hall_left")).all()

    assert len(rows) == 2, (
        f"expected both schools to have a camera named `hall_left`, found {len(rows)}"
    )
    assert {row.school_id for row in rows} == {ours.school_id, theirs.school_id}

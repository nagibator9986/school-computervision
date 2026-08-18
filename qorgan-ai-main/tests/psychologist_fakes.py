"""Shared setup for the §13 cabinet tests: real rows, real logins, no mocks of our own.

Four test modules need the same handful of things -- an app, a logged-in client per role, a
camera with an event on it, and a pupil with meal sessions. They live here rather than
being copied four ways for the reason `classroom_fakes` and `canteen_fakes` exist: four
copies of a builder is four chances for one to drift into describing a different world.

**Plain functions and one generator, not pytest fixtures.** Importing a fixture into
another module re-binds the name and ruff calls it a redefinition (F811), which is fair:
the import reads as a definition and is not one. So the modules here declare their own two
fixtures over `client_factory` and call the builders directly.

Everything writes through the functions production writes through
(`events.store.record_event`, the ORM models). Nothing fakes a capability: a test that
patched `ROLE_CAPABILITIES` to reach a page would be asserting about its own patch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, CanteenSession, Lesson, Person, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import (
    CameraRole,
    CameraType,
    LessonState,
    PersonType,
    SessionOutcome,
    SessionState,
    Severity,
    UserRole,
)
from qorgan.events.store import record_event
from qorgan.passwords import hash_password
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


def client_factory(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about. Used with `yield from`.

    The account is created and the password is genuinely posted to /login, rather than the
    session cookie being planted: a test that installs its own session proves nothing about
    whether logging in works, and this suite has already been caught by that shape once --
    every login test passed while a browser could not log in at all.
    """
    # ONE client per role, kept and handed back on the second ask. Not an optimisation: the
    # account is `user_<role>` and `users.username` is unique, so a test that asked for the
    # same role twice -- «the psychologist writes a note, then the psychologist reads it» --
    # died on an IntegrityError from the fixture rather than on anything it was asserting.
    # Memoising also makes the second ask the SAME logged-in session, which is what a person
    # coming back to a page actually is.
    clients: dict[UserRole, TestClient] = {}

    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            if role in clients:
                return clients[role]

            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login", data=with_token(client, {"username": username, "password": PASSWORD})
            )
            assert response.status_code == 303, "login failed"
            clients[role] = client
            return client

        yield make


def a_camera(session: Session) -> Camera:
    row = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.commit()
    return row


def a_pupil(session: Session, external_id: str = "student_470") -> Person:
    row = Person(
        external_id=external_id,
        full_name="Иванов Иван",
        person_type=PersonType.STUDENT,
        class_name="7-А",
    )
    session.add(row)
    session.commit()
    return row


def an_event(camera: Camera, summary: str = "Зафиксирована агрессия") -> int:
    """One recorded incident, written the way the detector writes one."""
    return record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text=summary,
        track_ids="3,7",
    )


def a_meal(
    session: Session,
    person: Person,
    camera: Camera,
    *,
    days_ago: float = 0.0,
    outcome: SessionOutcome | None = SessionOutcome.ATE,
) -> CanteenSession:
    """One closed meal session for this child, `days_ago` before now.

    Relative to `utcnow()` rather than to a fixed date, so the week buckets a test asserts
    on are the buckets the page would really draw today. A fixture pinned to 2026-03-04
    would drift out of the eight-week window and start passing vacuously.
    """
    opened: datetime = utcnow() - timedelta(days=days_ago)
    row = CanteenSession(
        person_id=person.id,
        entry_camera_id=camera.id,
        state=SessionState.CLOSED,
        outcome=outcome,
        opened_at=opened,
        closed_at=opened + timedelta(minutes=20),
        dwell_seconds=1200.0,
    )
    session.add(row)
    session.commit()
    return row


def a_lesson(session: Session, camera: Camera) -> Lesson:
    """One observation period in a classroom. **It carries no `person_id` and cannot.**

    `lessons` and `lesson_tracks` are the two tables in this schema that do not point at
    `persons` (see `db/models/classroom.py`), which is why the cabinet's classroom block is
    ANONYMOUS however many of these exist -- and why a test can build ten of them and still
    not make that block say anything about a child.
    """
    row = Lesson(
        camera_id=camera.id,
        state=LessonState.CLOSED,
        started_at=utcnow() - timedelta(hours=1),
        ended_at=utcnow(),
        min_presence_seconds=300.0,
    )
    session.add(row)
    session.commit()
    return row


def an_unattributed_meal(session: Session, camera: Camera) -> CanteenSession:
    """An entry the camera could not put a name to -- `person_id IS NULL`.

    Almost every session in the school's database is this today, which is exactly why the
    cabinet counts ATTRIBUTED sessions: counting all of them would make the canteen signal
    look alive while naming nobody.
    """
    row = CanteenSession(
        person_id=None,
        entry_camera_id=camera.id,
        state=SessionState.CLOSED,
        outcome=SessionOutcome.UNKNOWN,
        opened_at=utcnow(),
    )
    session.add(row)
    session.commit()
    return row

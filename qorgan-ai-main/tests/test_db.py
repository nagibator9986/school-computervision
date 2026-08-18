"""The database layer must refuse the two things that broke the legacy system:
absolute paths, and naive local-time datetimes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from qorgan.db.engine import BUSY_TIMEOUT_MS, build_engine
from qorgan.db.models import Camera, Event, Person, PersonPhoto
from qorgan.enums import CameraRole, CameraType, EventType, PersonType, Severity
from qorgan.settings import Settings


def _camera(session: Session) -> Camera:
    camera = Camera(
        name="hall_left",
        display_name="Hall left",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.flush()
    return camera


def _person(session: Session) -> Person:
    person = Person(external_id="p-001", full_name="Иванов Иван", person_type=PersonType.STUDENT)
    session.add(person)
    session.flush()
    return person


def test_sqlite_runs_in_wal_with_a_busy_timeout(settings: Settings) -> None:
    """Legacy ran journal_mode=delete with no busy_timeout, so one writer blocked
    every reader and the workers saw `database is locked` under load."""
    engine = build_engine(settings.database_url)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert connection.execute(text("PRAGMA busy_timeout")).scalar() == BUSY_TIMEOUT_MS
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_an_absolute_media_path_is_rejected(session: Session) -> None:
    """R6. Legacy stored absolute paths; the project moved twice and each move
    broke 100% of the photo and clip rows."""
    person = _person(session)
    session.add(
        PersonPhoto(person_id=person.id, path=r"C:\qorgan_ai\media\students\1.jpg", sha256="x" * 64)
    )
    with pytest.raises(StatementError, match="absolute path rejected"):
        session.flush()


def test_a_traversal_path_is_rejected(session: Session) -> None:
    person = _person(session)
    session.add(PersonPhoto(person_id=person.id, path="../../etc/passwd", sha256="x" * 64))
    with pytest.raises(StatementError, match="traversal"):
        session.flush()


def test_a_relative_media_path_round_trips(session: Session) -> None:
    person = _person(session)
    session.add(PersonPhoto(person_id=person.id, path=r"students\5a\ivanov.jpg", sha256="a" * 64))
    session.flush()
    stored = session.query(PersonPhoto).one()
    # Backslashes are normalised, so the same row reads the same on any platform.
    assert stored.path == "students/5a/ivanov.jpg"


def test_a_naive_datetime_is_rejected(session: Session) -> None:
    """Legacy stored naive local-time ISO strings and compared them by string prefix."""
    camera = _camera(session)
    session.add(
        Event(
            camera_id=camera.id,
            event_type=EventType.BULLYING,
            occurred_at=datetime(2026, 7, 12, 10, 30),
            confidence=0.9,
            candidate_probability=0.8,
            severity=Severity.ALERT,
        )
    )
    with pytest.raises(StatementError, match="naive datetime rejected"):
        session.flush()


def test_an_aware_datetime_round_trips_as_utc(session: Session) -> None:
    camera = _camera(session)
    moment = datetime(2026, 7, 12, 10, 30, tzinfo=UTC)
    session.add(
        Event(
            camera_id=camera.id,
            event_type=EventType.BULLYING,
            occurred_at=moment,
            confidence=0.9,
            candidate_probability=0.8,
            severity=Severity.ALERT,
        )
    )
    session.commit()

    session.expire_all()
    stored = session.query(Event).one()
    assert stored.occurred_at.tzinfo is not None
    assert stored.occurred_at == moment


def test_person_type_is_stored_not_guessed(session: Session) -> None:
    """Legacy re-derived person_type on every boot from 24 substring patterns,
    reverting manual corrections (audit H-02)."""
    person = _person(session)
    session.commit()
    assert session.get(Person, person.id).person_type is PersonType.STUDENT

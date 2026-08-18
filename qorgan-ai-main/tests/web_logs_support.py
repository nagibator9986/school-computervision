"""Fixtures and helpers shared by the /logs tests.

Split out rather than duplicated: `test_web_logs.py` and `test_web_logs_leaks.py` ask the
same page two different questions -- does it tell the truth, and does it keep a secret --
and a second copy of `client_for` is a second copy that can quietly stop logging in.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, Severity, UserRole
from qorgan.events.store import record_event
from qorgan.passwords import hash_password
from qorgan.settings import Settings, resolve
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about."""
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login",
                data=with_token(client, {"username": username, "password": PASSWORD}),
            )
            assert response.status_code == 303, "login failed"
            return client

        yield make


@pytest.fixture
def json_logs(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Structured logs on (the shipped default), and the directory they live in.

    The `settings` fixture turns JSON off, which is exactly the state one test asserts is
    reported as UNAVAILABLE -- so every test that wants entries has to ask for this.
    """
    monkeypatch.setattr(settings, "log_json", True)
    directory = resolve(settings.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def camera(session: Session) -> Camera:
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


def record(message: str, **fields: object) -> dict[str, object]:
    """One record in the shape `logging_setup.JsonFormatter` emits."""
    return {
        "ts": "2026-07-25T10:00:00+0500",
        "level": "ERROR",
        "logger": "qorgan.capture.stream",
        "process": "worker-hall",
        "message": message,
        **fields,
    }


def write(directory: Path, name: str, *records: dict[str, object]) -> Path:
    target = directory / f"{name}.log"
    with target.open("a", encoding="utf-8") as handle:
        for entry in records:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return target


def an_event(camera: Camera) -> int:
    return record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия",
        track_ids="3,7",
    )

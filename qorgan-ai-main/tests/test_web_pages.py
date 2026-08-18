"""The events log, and the authenticated /media handler.

The canteen journal lives in test_web_canteen.py: this file hit the 500-line cap (R1) and
the canteen surface was the natural seam.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import (
    CameraRole,
    CameraType,
    EventStatus,
    Severity,
    UserRole,
)
from qorgan.events.recorder import media_root, write_snapshot
from qorgan.events.store import record_event
from qorgan.notify.message import local_time
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

# 09:12:30 UTC is 14:12:30 in Almaty. Deliberately the same moment as
# `tests/test_alert_message.py::MOMENT`: the phone and the page describe one incident and
# must not disagree about when it happened.
MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)


@pytest.fixture
def client(settings: Settings, session: Session) -> Iterator[TestClient]:
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as test_client:
        test_client.post(
            "/login",
            data=with_token(test_client, {"username": "op", "password": PASSWORD}),
        )
        yield test_client


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


def _snapshot() -> str:
    frame = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    return write_snapshot(frame, "hall_left", utcnow())


def _event(
    camera: Camera,
    *,
    summary: str = "Зафиксирована агрессия",
    media: bool = True,
    occurred_at: datetime | None = None,
) -> int:
    return record_event(
        camera_id=camera.id,
        occurred_at=occurred_at or utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text=summary,
        track_ids="3,7",
        snapshot_path=_snapshot() if media else None,
    )


# -- the events log ----------------------------------------------------------


def test_the_events_page_needs_a_session(settings: Settings, session: Session) -> None:
    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get("/events").status_code == 303


def test_an_event_appears_in_the_log(client: TestClient, camera: Camera) -> None:
    _event(camera)

    response = client.get("/events")

    assert response.status_code == 200
    assert "Зафиксирована агрессия" in response.text
    assert "Холл слева" in response.text


def test_the_events_page_shows_the_school_s_own_time_not_UTC(
    client: TestClient, camera: Camera
) -> None:
    """The same defect the Telegram caption was fixed for, one layer over.

    A teacher opens `/events` beside the alert on their phone. The alert says 14:12:30 and
    the page said 09:12:30 — same row, same column, five hours apart — and a teacher sent
    to the wrong five minutes of CCTV finds an empty corridor and concludes the system is
    lying to them (`HANDOVER.md` §6a).

    Asserting the UTC string is ABSENT is half the test: "a timestamp is rendered" was
    already true before the fix and would pass just as happily against the raw column.
    """
    _event(camera, media=False, occurred_at=MOMENT)

    response = client.get("/events")

    assert "14:12:30" in response.text, "the page is not in the school's time"
    assert "09:12:30" not in response.text, "the raw UTC clock is still on the page"


def test_the_page_and_the_alert_are_converted_by_ONE_function(
    client: TestClient, camera: Camera
) -> None:
    """Not just "both are local" — both are `notify.message.local_time`.

    Two converters that agree today are two converters, and the defect this whole branch
    fixes is a value that is true in one layer and quietly wrong in the next. A second
    formatter here would drift on the first day someone changes a format string, so the
    test pins the output of the one function rather than a date shape written out by hand.
    """
    _event(camera, media=False, occurred_at=MOMENT)

    response = client.get("/events")

    assert local_time(MOMENT) in response.text


def test_the_clip_is_a_video_element_not_a_link(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Legacy's clips were unplayable in the UI: it sent a link, not a player."""
    event_id = _event(camera)
    from qorgan.events.store import attach_media

    attach_media(event_id, clip_path="clips/2026-03-04/hall.mp4")

    response = client.get("/events")

    assert "<video" in response.text
    assert "/media/clips/2026-03-04/hall.mp4" in response.text


def test_a_pupils_name_cannot_inject_script(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The legacy built the DOM with innerHTML from server JSON, so a pupil named
    `<img src=x onerror=...>` gave stored XSS in the operator's browser (audit H-05)."""
    _event(camera, summary='<img src=x onerror="alert(1)">')

    response = client.get("/events")

    assert "<img src=x onerror=" not in response.text
    assert "&lt;img src=x onerror=" in response.text, "the name was not escaped"


def test_the_log_is_paginated(client: TestClient, camera: Camera) -> None:
    """The legacy loaded the ENTIRE events table on every render, every 2.5 seconds, per
    client. Fine with 400 rows; fatal with 40 000, and it gets worse every day."""
    for _ in range(30):
        _event(camera, media=False)

    first = client.get("/events?page=1")
    second = client.get("/events?page=2")

    assert first.status_code == 200
    assert "1 / 2" in first.text
    assert second.status_code == 200


def test_an_event_can_be_marked_a_false_positive(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The only channel through which the school tells us we are wrong. A detector nobody
    corrects never improves."""
    event_id = _event(camera)

    response = client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "false_positive"}),
    )

    assert response.status_code == 303
    session.expire_all()
    event = session.get(Event, event_id)
    assert event.status is EventStatus.FALSE_POSITIVE
    assert event.reviewed_by_id is not None
    assert event.reviewed_at is not None


def test_an_event_can_be_confirmed(client: TestClient, session: Session, camera: Camera) -> None:
    event_id = _event(camera)

    client.post(f"/events/{event_id}/review", data=with_token(client, {"verdict": "confirmed"}))

    session.expire_all()
    assert session.get(Event, event_id).status is EventStatus.CONFIRMED


def test_a_nonsense_verdict_is_refused(client: TestClient, camera: Camera) -> None:
    event_id = _event(camera)

    assert client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "lol"}),
    ).status_code == 400


def test_events_can_be_filtered_by_status(client: TestClient, camera: Camera) -> None:
    event_id = _event(camera, summary="ЛОЖНОЕ")
    client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "false_positive"}),
    )
    _event(camera, summary="НОВОЕ")

    response = client.get("/events?status_filter=new")

    assert "НОВОЕ" in response.text
    assert "ЛОЖНОЕ" not in response.text


# -- /media ------------------------------------------------------------------


def test_media_is_not_served_to_anonymous_callers(settings: Settings, session: Session) -> None:
    """These are photographs of children. The legacy mounted them with StaticFiles on a
    server bound to 0.0.0.0 with no auth anywhere."""
    path = _snapshot()

    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get(f"/media/{path}").status_code == 303


def test_media_is_served_to_a_logged_in_operator(client: TestClient) -> None:
    path = _snapshot()

    response = client.get(f"/media/{path}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "no-store" in response.headers["cache-control"]


def test_a_path_traversal_is_refused(client: TestClient) -> None:
    """A path is a string an attacker controls, and `..` is a valid string.

    Note the URL-encoding. An HTTP client normalises a literal `../` out of the path
    before it ever leaves the machine, so a test written the obvious way gets a 404 from
    the router and never reaches the handler at all — it would pass while testing
    nothing. Percent-encoding survives normalisation and lands in the path parameter,
    which is exactly how a real attacker would send it.
    """
    secret = media_root().parent / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("the telegram token", encoding="utf-8")

    response = client.get("/media/%2e%2e%2fsecret.txt")

    assert response.status_code == 403, "the traversal reached the filesystem"
    assert "telegram" not in response.text


def test_a_deep_path_traversal_is_refused(client: TestClient) -> None:
    """Nested, so that even a naive prefix check on the string would pass it."""
    response = client.get("/media/snapshots%2f..%2f..%2f..%2fsecret.txt")

    assert response.status_code in (403, 404)


def test_an_unsupported_file_type_is_refused(client: TestClient) -> None:
    """Serving arbitrary types out of a media directory is how a stray .html file becomes
    stored XSS on your own origin."""
    evil = media_root() / "evil.html"
    evil.parent.mkdir(parents=True, exist_ok=True)
    evil.write_text("<script>alert(1)</script>", encoding="utf-8")

    assert client.get("/media/evil.html").status_code == 403


def test_a_missing_file_is_a_404(client: TestClient) -> None:
    assert client.get("/media/snapshots/nope.jpg").status_code == 404

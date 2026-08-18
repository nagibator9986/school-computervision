"""The notification history: proof that an alert did, or did not, arrive.

The legacy spawned a raw thread per alert with a bare `requests.post` inside it and
swallowed every exception (audit M-14). A bullying alert that never reached anyone left
no row, no log line, and no trace of any kind. The teacher who was never told believed
they had been told nothing because nothing had happened.

So the page under test here exists to answer one question: **did this alert reach a
human?** Every test below is a way of getting that answer wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Notification, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import (
    CameraRole,
    CameraType,
    NotificationChannel,
    NotificationStatus,
    Severity,
    UserRole,
)
from qorgan.events.store import record_event
from qorgan.notify.telegram import TelegramClient
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture(autouse=True)
def no_live_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the app's own notifier from rewriting this file's fixture data.

    `create_app()`'s lifespan starts a `NotificationWorker`, and the `settings` fixture
    supplies a bot token, so the worker starts for real. Every QUEUED row a test below
    creates would then be picked up by that background thread, which makes a live HTTPS
    request to api.telegram.org and overwrites `status`, `attempts` and `last_error` --
    the exact columns under assertion. A test racing its own fixture data is not a test.

    Disabled the way production disables it: `from_settings()` returning None is the
    supported "Telegram is not configured" state, so the worker declines to start rather
    than being reached into. It also means that if the PAGE ever sends anything, it has
    to build its own client to do it, which is precisely what
    `test_opening_the_page_neither_sends_nor_retries_anything` is about.
    """
    monkeypatch.setattr(TelegramClient, "from_settings", classmethod(lambda cls: None))


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
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
def client(client_for: ClientFor) -> TestClient:
    return client_for(UserRole.OPERATOR)


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


def _event(
    camera: Camera,
    *,
    summary: str = "Зафиксирована агрессия",
    occurred_at: datetime | None = None,
) -> int:
    return record_event(
        camera_id=camera.id,
        occurred_at=occurred_at or utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text=summary,
        track_ids="3,7",
    )


def _notification(
    event_id: int,
    *,
    status: NotificationStatus = NotificationStatus.SENT,
    attempts: int = 1,
    error: str | None = None,
    sent: bool = False,
) -> int:
    with session_scope() as db:
        row = Notification(
            event_id=event_id,
            channel=NotificationChannel.TELEGRAM,
            status=status,
            attempts=attempts,
            last_error=error,
            sent_at=utcnow() if sent else None,
        )
        db.add(row)
        db.flush()
        return row.id


# -- who may read it ---------------------------------------------------------


def test_the_history_needs_a_session(settings: Settings, session: Session) -> None:
    """The legacy had ~50 endpoints and no authentication anywhere (audit C-01)."""
    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get("/notifications").status_code == 303


def test_a_canteen_worker_is_refused_the_notification_history(client_for: ClientFor) -> None:
    """§14: столовая — БЕЗ доступа к буллингу.

    Every row on this page is an alert about a bullying incident: it carries the event's
    summary, its camera and the minute it happened. A canteen worker reading "НЕ
    ДОСТАВЛЕНО — Зафиксирована агрессия, Холл слева, 14:12" has read the bullying log,
    whatever the URL is called. This is the test that says the page chose its capability
    by what it DISCLOSES rather than by what it is named after.
    """
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get("/notifications").status_code == 403, (
        "§14 violated: the canteen worker read bullying alerts off the notification history"
    )


def test_an_operator_reads_the_history(client: TestClient, camera: Camera) -> None:
    _notification(_event(camera), sent=True)

    assert client.get("/notifications").status_code == 200


def test_the_link_is_drawn_only_for_a_role_that_may_follow_it(client_for: ClientFor) -> None:
    """The nav comes from the same table the route is gated on. Two sources of truth would
    mean a canteen worker clicking a link into a 403 and reporting the system as broken."""
    operator = client_for(UserRole.OPERATOR)
    canteen = client_for(UserRole.CANTEEN_STAFF)

    assert '"/notifications"' in operator.get("/events").text
    assert "/notifications" not in canteen.get("/canteen").text


# -- the point of the page ---------------------------------------------------


def test_an_alert_that_never_arrived_says_so_with_its_error(
    client: TestClient, camera: Camera
) -> None:
    """The whole reason this page exists.

    In the legacy, this alert simply ceased to exist: the thread died, the exception was
    swallowed, and nothing anywhere recorded that a teacher had not been told a child was
    being hurt. Here it is a row, and the row says what went wrong.
    """
    event_id = _event(camera, summary="Зафиксирована агрессия")
    _notification(
        event_id,
        status=NotificationStatus.FAILED,
        attempts=6,
        error="HTTP 400: Bad Request: chat not found",
    )

    page = client.get("/notifications")

    assert page.status_code == 200
    assert "Зафиксирована агрессия" in page.text
    assert "chat not found" in page.text, "the reason the alert failed is not on the page"
    assert "Холл слева" in page.text


def test_an_alert_still_waiting_in_the_queue_is_listed_too(
    client: TestClient, camera: Camera
) -> None:
    """Sent AND unsent. A history that lists only what succeeded answers the easy question."""
    _notification(_event(camera, summary="ЕЩЁ В ОЧЕРЕДИ"), status=NotificationStatus.QUEUED)

    assert "ЕЩЁ В ОЧЕРЕДИ" in client.get("/notifications").text


def test_a_delivered_alert_is_listed_with_its_channel(client: TestClient, camera: Camera) -> None:
    _notification(_event(camera, summary="ДОШЛО"), sent=True)

    page = client.get("/notifications")

    assert "ДОШЛО" in page.text
    assert "telegram" in page.text, "the channel an alert went out on is not on the page"


def test_the_undelivered_can_be_filtered_out_of_the_noise(
    client: TestClient, camera: Camera
) -> None:
    """A teacher looking for a specific silence should not have to read a thousand successes."""
    _notification(_event(camera, summary="ДОШЛО"), sent=True)
    _notification(
        _event(camera, summary="НЕ ДОШЛО"), status=NotificationStatus.FAILED, error="timeout"
    )

    page = client.get("/notifications?status_filter=failed")

    assert "НЕ ДОШЛО" in page.text
    assert "ДОШЛО" not in page.text.replace("НЕ ДОШЛО", "")


def test_a_filter_the_page_did_not_apply_is_not_echoed_back(
    client: TestClient, camera: Camera
) -> None:
    """A page showing a filter it did not apply is lying about what it is showing.

    It is also the only route by which raw query-string text would reach the markup, so
    the value that comes back is the one that was actually used, never what was typed.
    """
    _notification(_event(camera, summary="ВИДНО ВСЕГДА"), sent=True)

    page = client.get("/notifications?status_filter=<script>whatever</script>")

    assert "ВИДНО ВСЕГДА" in page.text, "an unusable filter hid rows instead of being ignored"
    assert "whatever" not in page.text


# -- pagination --------------------------------------------------------------


def test_the_history_is_paginated(client: TestClient, camera: Camera) -> None:
    """The legacy loaded its ENTIRE event table on every render, every 2.5 seconds, per
    client (audit M-19). Fine at 400 rows, fatal at 40 000 -- and the notification table
    grows by a row per alert forever, so it is the table that gets there first.
    """
    for number in range(1, 31):
        _notification(
            _event(camera),
            status=NotificationStatus.FAILED,
            error=f"сбой №{number:02d}",
        )

    first = client.get("/notifications?page=1")
    second = client.get("/notifications?page=2")

    assert first.status_code == 200
    assert second.status_code == 200
    assert "1 / 2" in first.text, "the page does not say where in the history it is"
    assert "сбой №30" in first.text, "the newest attempt is not on the first page"
    assert "сбой №01" not in first.text, "the whole table was rendered on one page"
    assert "сбой №01" in second.text, "the second page did not move"


def test_a_page_past_the_end_is_empty_rather_than_an_error(
    client: TestClient, camera: Camera
) -> None:
    """A guessed page number is a URL, and a URL is user input."""
    _notification(_event(camera), sent=True)

    assert client.get("/notifications?page=9999").status_code == 200
    assert client.get("/notifications?page=0").status_code == 200
    assert client.get("/notifications?page=-4").status_code == 200


# -- never render a secret ---------------------------------------------------


def test_a_bot_token_in_an_error_message_never_reaches_the_page(
    client: TestClient, camera: Camera, settings: Settings
) -> None:
    """R4, and audit H-04: the legacy served its live bot token to anyone who opened its
    settings page.

    `notifications.last_error` is free text written from a third party's response body and
    from exception strings. The log path is scrubbed by `RedactingFormatter`; the DATABASE
    column is not, so the value is safe in one layer and raw in the next -- and this page
    is the layer that renders it to a browser.
    """
    token = settings.telegram_bot_token.get_secret_value()
    _notification(
        _event(camera),
        status=NotificationStatus.FAILED,
        error=f"ConnectError posting to https://api.telegram.org/bot{token}/sendPhoto",
    )

    page = client.get("/notifications")

    assert page.status_code == 200
    assert token not in page.text, "the Telegram bot token was rendered to the browser"
    assert "***REDACTED***" in page.text, "the error was dropped instead of being redacted"


def test_a_chat_id_in_an_error_message_never_reaches_the_page(
    client: TestClient, camera: Camera, settings: Settings
) -> None:
    """The chat id says which group the school's alerts go to. It is env-only for the same
    reason the token is, and it is not token-shaped, so only being a REGISTERED secret
    keeps it off this page."""
    chat_id = settings.telegram_chat_id
    _notification(
        _event(camera),
        status=NotificationStatus.FAILED,
        error=f"HTTP 400: Bad Request: chat {chat_id} not found",
    )

    assert chat_id not in client.get("/notifications").text


def test_the_page_carries_no_configuration_at_all(
    client: TestClient, camera: Camera, settings: Settings
) -> None:
    """The standing guard against H-04 coming back as a debugging convenience: this page
    reports history, and history is not configuration."""
    _notification(_event(camera), sent=True)

    page = client.get("/notifications").text

    assert settings.telegram_bot_token.get_secret_value() not in page
    assert settings.telegram_chat_id not in page
    assert settings.rtsp_password.get_secret_value() not in page


# -- escaping ----------------------------------------------------------------


def test_an_error_message_cannot_inject_script(client: TestClient, camera: Camera) -> None:
    """An error string is attacker-influenced text: it comes back from Telegram, and it can
    quote the caption we sent -- which contains a summary naming children. The legacy built
    its DOM with `innerHTML` from server JSON, so exactly this was stored XSS (audit H-05).
    """
    _notification(
        _event(camera),
        status=NotificationStatus.FAILED,
        error='<img src=x onerror="alert(1)">',
    )

    page = client.get("/notifications")

    assert "<img src=x onerror=" not in page.text
    assert "&lt;img src=x onerror=" in page.text, "the error text was not escaped"


def test_a_pupils_name_in_the_summary_cannot_inject_script(
    client: TestClient, camera: Camera
) -> None:
    _notification(_event(camera, summary='<svg onload="alert(1)">'), sent=True)

    page = client.get("/notifications")

    assert "<svg onload=" not in page.text
    assert "&lt;svg onload=" in page.text


# -- zero side effects -------------------------------------------------------


def test_opening_the_page_neither_sends_nor_retries_anything(
    client: TestClient, camera: Camera, session: Session
) -> None:
    """Opening a tab must not make the system do work.

    The legacy's `POST /page-activate/{page}` restarted the AI workers -- with a five
    second `thread.join()` inside the HTTP handler -- every time somebody opened a tab.
    Reading the delivery history is reading. Re-sending an alert about a child is a
    decision, and a GET is not how a person makes one.
    """
    notification_id = _notification(
        _event(camera),
        status=NotificationStatus.FAILED,
        attempts=6,
        error="HTTP 401: Unauthorized",
    )
    before = _state_of(session, notification_id)

    client.get("/notifications")
    client.get("/notifications?status_filter=failed")
    client.get("/notifications?page=1")

    assert _state_of(session, notification_id) == before, (
        "loading the history changed a notification row"
    )


def _state_of(session: Session, notification_id: int) -> tuple:
    session.expire_all()
    row = session.get(Notification, notification_id)
    assert row is not None
    return (row.status, row.attempts, row.last_error, row.sent_at)


# -- the clock ---------------------------------------------------------------


def test_the_time_shown_is_the_schools_clock_not_utc(client: TestClient, camera: Camera) -> None:
    """The alert on the teacher's phone says 14:12, because `notify.message.local_time`
    puts the school's wall clock on it (§7). Every timestamp in the database is UTC by
    column type, and the school is UTC+5.

    If this page renders the column raw it says 09:12 for the same alert, and the teacher
    comparing the two concludes they are two different incidents -- or goes to the wrong
    five minutes of CCTV, finds an empty corridor, and concludes the system is lying. The
    same value, true in the database and wrong on the page.
    """
    _notification(
        _event(camera, occurred_at=datetime(2026, 3, 4, 9, 12, 0, tzinfo=UTC)),
        status=NotificationStatus.FAILED,
        error="timeout",
    )

    page = client.get("/notifications")

    assert "04.03.2026 14:12:00" in page.text, "the page is showing UTC, not the school's clock"

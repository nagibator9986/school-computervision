"""Telegram: it uploads the evidence, and it does not lose the alert."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import unquote_plus

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, Notification
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, NotificationStatus, Severity
from qorgan.events.recorder import media_root, write_snapshot
from qorgan.events.store import record_event
from qorgan.notify.queue import MAX_ATTEMPTS, NotificationWorker, enqueue
from qorgan.notify.telegram import TelegramClient
from qorgan.settings import Settings

MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)


class FakeTelegram:
    """Stands in for api.telegram.org. Records what was actually uploaded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str]] = []
        self.responses: list[httpx.Response] = []

    def queue(self, response: httpx.Response) -> None:
        self.responses.append(response)

    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        # The BODY, not just the method. What the message actually says to a teacher is
        # the whole of client §7, and it travels in here -- multipart for sendPhoto, form
        # for sendMessage. A fake that records only the method cannot tell an alert
        # carrying its reasons from one carrying nothing.
        body = request.content.decode("utf-8", "replace")
        self.calls.append((method, dict(request.headers), body))
        if self.responses:
            return self.responses.pop(0)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    @property
    def methods(self) -> list[str]:
        return [method for method, _, _ in self.calls]

    @property
    def bodies(self) -> list[str]:
        return [body for _, _, body in self.calls]


@pytest.fixture
def telegram(monkeypatch: pytest.MonkeyPatch) -> FakeTelegram:
    fake = FakeTelegram()
    transport = httpx.MockTransport(fake.handler)

    real_client = httpx.Client

    def _client(**kwargs):
        kwargs["transport"] = transport
        return real_client(**kwargs)

    monkeypatch.setattr("qorgan.notify.telegram.httpx.Client", _client)
    return fake


@pytest.fixture
def client() -> TelegramClient:
    return TelegramClient("123456:token", "-100999")


def _event(session: Session, settings: Settings, *, with_snapshot: bool = True) -> int:
    import numpy as np

    camera = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()

    snapshot = None
    if with_snapshot:
        frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        snapshot = write_snapshot(frame, "hall_left", MOMENT)

    return record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=Verdict(0.93, 0.9, 0.8, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия — Холл слева (93%)",
        track_ids="3,7",
        snapshot_path=snapshot,
    )


# -- the client --------------------------------------------------------------


def test_a_snapshot_is_UPLOADED_not_linked(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """THE fix. The legacy sent a text message containing a link to
    http://127.0.0.1:8000/... -- the loopback address OF THE SERVER. A teacher reading
    that alert on their phone could never open it. Not once, not ever. The system built
    to show staff what happened sent them a URL to their own phone's localhost."""
    import numpy as np

    frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    path = media_root() / write_snapshot(frame, "hall_left", MOMENT)

    result = client.send_photo(path, "Зафиксирована агрессия")

    assert result.ok
    assert telegram.methods == ["sendPhoto"], "the picture was not uploaded"


def test_a_rate_limit_is_not_a_failure(
    settings: Settings, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """Telegram tells us EXACTLY how long to wait. The legacy ignored it and dropped the
    message."""
    telegram.queue(
        httpx.Response(
            429,
            json={
                "ok": False,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 42},
            },
        )
    )

    result = client.send_message("x")

    assert not result.ok
    assert result.retry_after == 42
    assert not result.permanent, "a rate limit must be retried, not given up on"


def test_a_bad_token_is_permanent(
    settings: Settings, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """It will still be bad in an hour. Retrying poison forever delays every real alert
    queued behind it."""
    telegram.queue(httpx.Response(401, json={"ok": False, "description": "Unauthorized"}))

    result = client.send_message("x")

    assert not result.ok
    assert result.permanent


def test_a_network_failure_is_temporary(
    settings: Settings, client: TelegramClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("the router is unplugged")

    transport = httpx.MockTransport(_explode)
    real = httpx.Client
    monkeypatch.setattr(
        "qorgan.notify.telegram.httpx.Client",
        lambda **kw: real(**{**kw, "transport": transport}),
    )

    result = client.send_message("x")

    assert not result.ok
    assert not result.permanent, "a network blip must be retried"


def test_a_file_too_large_for_telegram_is_permanent(
    settings: Settings, client: TelegramClient, tmp_path
) -> None:
    huge = tmp_path / "huge.jpg"
    huge.write_bytes(b"\0" * (11 * 1024 * 1024))

    result = client.send_photo(huge, "x")

    assert result.permanent
    assert "limit" in result.error


def test_a_missing_file_is_permanent(settings: Settings, client: TelegramClient, tmp_path) -> None:
    result = client.send_photo(tmp_path / "gone.jpg", "x")
    assert result.permanent


# -- the queue ---------------------------------------------------------------


def test_an_alert_is_a_row_before_it_is_a_message(
    settings: Settings, session: Session, telegram: FakeTelegram
) -> None:
    """The legacy spawned a raw thread per alert with a bare requests.post inside it and
    swallowed every exception. If the network was down for the ten seconds the alert
    existed, the alert simply ceased to exist -- no retry, no record, no log line."""
    event_id = _event(session, settings)

    enqueue(event_id)

    session.expire_all()
    row = session.scalars(select(Notification)).one()
    assert row.status is NotificationStatus.QUEUED
    assert row.event_id == event_id


def test_the_queue_sends_the_snapshot(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    enqueue(_event(session, settings))

    sent = NotificationWorker(client).drain_once()

    assert sent == 1
    assert telegram.methods == ["sendPhoto"]

    session.expire_all()
    row = session.scalars(select(Notification)).one()
    assert row.status is NotificationStatus.SENT
    assert row.sent_at is not None


def test_the_uploaded_alert_says_why_when_and_which_event(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """Client §7, end to end: from the Verdict, through the database, onto the phone.

    The message used to be severity + camera + confidence and nothing else. A teacher
    could not tell whether a child was on the floor, could not tell when it happened, and
    could not look the event up afterwards.
    """
    event_id = _event(session, settings)
    enqueue(event_id)

    NotificationWorker(client).drain_once()

    caption = telegram.bodies[0]
    assert "падение или низкая поза" in caption, "the alert does not say WHY"
    # MOMENT is 09:12:30 UTC and the school is UTC+5.
    assert "04.03.2026 14:12:30" in caption, "the alert does not say WHEN, in school time"
    assert str(event_id) in caption, "the alert cannot be looked up afterwards"


def test_the_text_fallback_carries_the_same_facts_as_the_photo(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """An event with no picture is still an event, and it is not a lesser message: it is
    the ONLY message, so it must carry everything §7 asks for."""
    event_id = _event(session, settings, with_snapshot=False)
    enqueue(event_id)

    NotificationWorker(client).drain_once()

    assert telegram.methods == ["sendMessage"]
    # sendMessage is form-encoded rather than multipart, so the text arrives percent-
    # escaped on the wire.
    caption = unquote_plus(telegram.bodies[0])
    assert "падение или низкая поза" in caption
    assert "04.03.2026 14:12:30" in caption
    assert str(event_id) in caption


def test_an_event_with_no_snapshot_still_notifies_somebody(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """A fight with no picture is still a fight."""
    enqueue(_event(session, settings, with_snapshot=False))

    NotificationWorker(client).drain_once()

    assert telegram.methods == ["sendMessage"]


def test_a_transient_failure_is_retried_not_lost(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    enqueue(_event(session, settings))
    telegram.queue(httpx.Response(500, json={"ok": False, "description": "server error"}))

    worker = NotificationWorker(client)
    worker.drain_once()

    session.expire_all()
    row = session.scalars(select(Notification)).one()
    assert row.status is NotificationStatus.QUEUED, "the alert was thrown away"
    assert row.attempts == 1
    assert row.last_error is not None


def test_a_retried_alert_eventually_gets_through(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    enqueue(_event(session, settings))
    telegram.queue(httpx.Response(500, json={"ok": False, "description": "server error"}))

    worker = NotificationWorker(client)
    worker.drain_once()  # fails
    worker._backoff_until.clear()  # pretend the backoff elapsed
    worker.drain_once()  # succeeds

    session.expire_all()
    assert session.scalars(select(Notification)).one().status is NotificationStatus.SENT


def test_an_alert_in_backoff_is_not_hammered(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """A network blip must not become a hot loop."""
    enqueue(_event(session, settings))
    telegram.queue(httpx.Response(500, json={"ok": False, "description": "server error"}))

    worker = NotificationWorker(client)
    worker.drain_once()
    attempted_again = worker.drain_once()

    assert attempted_again == 0, "the queue hammered a failing endpoint"


def test_a_permanent_failure_is_given_up_on_LOUDLY(
    settings: Settings,
    session: Session,
    telegram: FakeTelegram,
    client: TelegramClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silently giving up on an assault alert is not acceptable. Giving up audibly is the
    least we can do."""
    import logging

    enqueue(_event(session, settings, with_snapshot=False))
    telegram.queue(httpx.Response(403, json={"ok": False, "description": "bot was blocked"}))

    with caplog.at_level(logging.ERROR):
        NotificationWorker(client).drain_once()

    session.expire_all()
    assert session.scalars(select(Notification)).one().status is NotificationStatus.FAILED
    assert any("ALERT NOT DELIVERED" in record.message for record in caplog.records)


def test_the_queue_gives_up_after_enough_attempts(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    enqueue(_event(session, settings, with_snapshot=False))
    worker = NotificationWorker(client)

    for _ in range(MAX_ATTEMPTS):
        telegram.queue(httpx.Response(500, json={"ok": False, "description": "boom"}))
        worker._backoff_until.clear()
        worker.drain_once()

    session.expire_all()
    row = session.scalars(select(Notification)).one()
    assert row.status is NotificationStatus.FAILED
    assert row.attempts == MAX_ATTEMPTS


def test_a_rate_limited_alert_waits_exactly_as_long_as_telegram_asked(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    enqueue(_event(session, settings, with_snapshot=False))
    telegram.queue(
        httpx.Response(
            429,
            json={"ok": False, "description": "Too Many", "parameters": {"retry_after": 300}},
        )
    )

    worker = NotificationWorker(client)
    worker.drain_once()

    notification_id = session.scalars(select(Notification.id)).one()
    wait_until = worker._backoff_until[notification_id]
    assert wait_until > datetime.now(UTC) + timedelta(seconds=250)


def test_every_attempt_is_recorded_including_the_successes(
    settings: Settings, session: Session, telegram: FakeTelegram, client: TelegramClient
) -> None:
    """The legacy logged only the failures of a WhatsApp PLACEHOLDER provider that always
    failed, so the notifications table was a list of things that never happened."""
    enqueue(_event(session, settings))

    NotificationWorker(client).drain_once()

    session.expire_all()
    row = session.scalars(select(Notification)).one()
    assert row.status is NotificationStatus.SENT
    assert row.provider_message_id is not None


def test_the_system_runs_without_telegram_configured(settings: Settings, session: Session) -> None:
    """Not an error. A school without a bot still gets a dashboard and an event log."""
    enqueue(_event(session, settings))

    assert NotificationWorker(None).drain_once() == 0

    session.expire_all()
    assert session.scalars(select(Notification)).one().status is NotificationStatus.QUEUED


def test_an_unconfigured_client_is_none(settings: Settings) -> None:
    settings.telegram_chat_id = ""
    assert TelegramClient.from_settings() is None


def test_events_that_do_not_notify_leave_no_notification(
    settings: Settings, session: Session
) -> None:
    """Only alerts above the notify threshold are queued at all."""
    _event(session, settings)

    session.expire_all()
    assert session.scalars(select(Notification)).all() == []
    assert session.scalars(select(Event)).one().confidence == 0.93

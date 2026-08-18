"""Rule R4 at the database boundary: a secret must not survive into a stored column.

R4 names the database explicitly -- "No secret in the repo, in a config file, in the
database, in a log line, or burnt into a debug image" -- because the legacy leaked the
same bot token six different ways (audit C-02/C-03). The sqlite file is the artefact
that gets copied onto a USB stick and emailed to whoever is debugging this week.

Two columns hold free-form error text that nobody wrote by hand:

  * `notifications.last_error` is `SendResult.error`, and `TelegramClient._post` builds
    every request against `https://api.telegram.org/bot<TOKEN>/...`. Any `httpx.HTTPError`
    that quotes the URL it was working on carries the bot token into that string, and
    `NotificationWorker._settle` copied the string into the column unaltered.
  * `worker_heartbeats.last_error` is `f"{type(exc).__name__}: {exc}"` off a crashing
    worker (`worker/entrypoint.py`), and a worker's exceptions are about
    `rtsp://user:password@host/...` -- the fleet camera password of audit C-03.

**Log lines were already safe and the database was not.** `RedactingFormatter` scrubs
every record on its way to a log file, so the very same error text is masked in the log
and printed in full in the row beside it.

Each test here fails if, and only if, the secret itself survives into the column.
"Some error text was recorded" is precisely the assertion that does not catch this.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.common import RtspSettings
from qorgan.db.models import Camera, Event, Notification, WorkerHeartbeat
from qorgan.enums import (
    CameraRole,
    CameraType,
    EventType,
    Severity,
    WorkerState,
)
from qorgan.notify.queue import NotificationWorker, enqueue
from qorgan.notify.telegram import TelegramClient
from qorgan.redaction import MASK
from qorgan.rtsp import build_url
from qorgan.settings import Settings
from qorgan.supervisor.heartbeat import write_heartbeat

MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)


@pytest.fixture
def telegram_that_fails_quoting_its_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport whose failure message is authored by httpx, not by this test.

    `raise_for_status()` formats "Client error '401 Unauthorized' for url '<url>'" --
    the whole URL, `/bot<TOKEN>/` included. `httpx.HTTPStatusError` is an
    `httpx.HTTPError`, so `TelegramClient._post`'s `except` clause catches it and folds
    it into `SendResult.error`. Nothing here writes the token into a string by hand;
    httpx does, which is the entire point of the defect.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        httpx.Response(401, json={"ok": False}, request=request).raise_for_status()
        raise AssertionError("unreachable")  # pragma: no cover

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        "qorgan.notify.telegram.httpx.Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": transport}),
    )


def _queue_one_alert(session: Session) -> None:
    """The smallest thing the notification worker will pick up: an event, then a row."""
    camera = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()

    event = Event(
        camera_id=camera.id,
        event_type=EventType.BULLYING,
        occurred_at=MOMENT,
        confidence=0.93,
        candidate_probability=0.9,
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия — Холл слева (93%)",
    )
    session.add(event)
    session.commit()
    enqueue(event.id)


def _stored_notification_error(session: Session) -> str:
    session.expire_all()
    return session.scalars(select(Notification)).one().last_error or ""


def _drain_one_failing_attempt(settings: Settings, session: Session) -> str:
    _queue_one_alert(session)
    client = TelegramClient(
        settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id
    )
    NotificationWorker(client).drain_once()
    return _stored_notification_error(session)


# -- notifications.last_error -------------------------------------------------


def test_the_bot_token_does_not_reach_notifications_last_error(
    settings: Settings, session: Session, telegram_that_fails_quoting_its_url: None
) -> None:
    """The defect itself. Anyone holding the sqlite file held the bot token."""
    token = settings.telegram_bot_token.get_secret_value()

    stored = _drain_one_failing_attempt(settings, session)

    assert token not in stored, (
        "R4: the Telegram bot token was written into notifications.last_error.\n"
        f"stored: {stored!r}"
    )


def test_no_fragment_of_the_bot_token_reaches_notifications_last_error(
    settings: Settings, session: Session, telegram_that_fails_quoting_its_url: None
) -> None:
    """Masking the numeric bot id and leaving the rest is not redaction.

    The half after the colon is the part that authenticates; a check for the whole
    token would pass on a string that still hands the reader everything they need.
    """
    secret_half = settings.telegram_bot_token.get_secret_value().split(":", 1)[1]

    stored = _drain_one_failing_attempt(settings, session)

    assert secret_half not in stored, (
        "R4: the authenticating half of the bot token survived into "
        f"notifications.last_error.\nstored: {stored!r}"
    )


def test_the_redacted_notification_row_still_says_what_went_wrong(
    settings: Settings, session: Session, telegram_that_fails_quoting_its_url: None
) -> None:
    """Blanking the column would pass the two tests above and destroy the audit trail.

    The queue exists because the legacy dropped alerts silently; a row that records
    nothing about why an assault alert failed is the same failure in a new place.
    """
    stored = _drain_one_failing_attempt(settings, session)

    assert stored, "the column was blanked rather than redacted"
    assert MASK in stored, f"the secret was dropped without a trace of it: {stored!r}"
    assert "401" in stored, f"the reason the send failed was lost: {stored!r}"


# -- worker_heartbeats.last_error ---------------------------------------------


def test_the_camera_password_does_not_reach_worker_heartbeats_last_error(
    settings: Settings, session: Session
) -> None:
    """The sibling column, exposed the same way.

    `worker/entrypoint.py` stores `f"{type(exc).__name__}: {exc}"` from a crashed
    worker, and what a worker's exceptions are about is the RTSP URL built by
    `qorgan.rtsp.build_url` -- credentials and all. That is audit C-03's one shared
    fleet password, in a column the dashboard renders.
    """
    password = settings.rtsp_password.get_secret_value()
    url = build_url("hall_left", RtspSettings(host="10.0.0.1"))
    assert password in url, "the fixture no longer builds a URL that carries a secret"
    exc = ConnectionError(f"could not open {url}")

    write_heartbeat(
        "bullying_hall",
        WorkerState.CRASHED,
        last_error=f"{type(exc).__name__}: {exc}",
    )

    session.expire_all()
    stored = session.scalars(select(WorkerHeartbeat)).one().last_error or ""
    assert password not in stored, (
        "R4: the camera password was written into worker_heartbeats.last_error.\n"
        f"stored: {stored!r}"
    )
    assert "10.0.0.1:554" in stored, f"the host was lost along with the password: {stored!r}"

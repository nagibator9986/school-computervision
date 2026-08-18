"""The notification queue. An alert about a child being hurt must not be lost.

**The queue is a table, not a list in memory.** The legacy spawned a raw thread per alert
with a bare `requests.post` inside it and swallowed every exception. If the network was
down for the ten seconds the alert existed, the alert simply ceased to exist — no retry,
no record, no log line. Here a notification is a row from the moment the event is written,
so it survives a crash, a restart, and a broken router.

Retry policy, and why each part is there:

  * **Exponential backoff** on a transient failure. A network blip should not become a
    hot loop.
  * **Telegram's own `retry_after`** is honoured on a 429. It knows better than we do.
  * **Permanent failures are not retried.** A bad token will still be bad in an hour, and
    a queue that retries poison forever delays every real alert behind it.
  * **A dead letter after N attempts**, logged loudly. Silently giving up on an assault
    alert is not acceptable; giving up *audibly* is.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import Event, Notification
from qorgan.db.types import utcnow
from qorgan.enums import NotificationChannel, NotificationStatus
from qorgan.events.reasons import unpack_reasons
from qorgan.events.recorder import media_root
from qorgan.logging_setup import get_logger
from qorgan.notify.message import compose_caption
from qorgan.notify.telegram import SendResult, TelegramClient
from qorgan.redaction import redact

logger = get_logger(__name__)

MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 900.0  # fifteen minutes: beyond that the alert is stale anyway
POLL_SECONDS = 2.0


def enqueue(event_id: int) -> int:
    """Register the intent to notify. Written in the same breath as the event."""

    def _add() -> int:
        with session_scope() as session:
            notification = Notification(
                event_id=event_id,
                channel=NotificationChannel.TELEGRAM,
                status=NotificationStatus.QUEUED,
            )
            session.add(notification)
            session.flush()
            return notification.id

    return with_retry(_add)


@dataclass(frozen=True, slots=True)
class Pending:
    notification_id: int
    event_id: int
    attempts: int
    summary: str
    confidence: float
    snapshot: str | None
    clip: str | None
    # What the message must actually say, beyond how sure we were (client §7). Read back
    # out of the event row rather than passed in at enqueue time: the row is the truth,
    # and on the merge path it is deliberately raised to the incident's worst moment
    # between the enqueue and the send.
    occurred_at: datetime
    reasons: tuple[str, ...]

    @property
    def caption(self) -> str:
        return compose_caption(
            summary=self.summary,
            occurred_at=self.occurred_at,
            event_id=self.event_id,
            reasons=self.reasons,
        )


class NotificationWorker:
    """Drains the queue on a background thread. Runs in the web process, not a worker.

    Deliberately: the detection workers must never block on a network call, and the legacy
    called a blocking `requests.post(timeout=10)` *from inside the frame loop* — so an
    unreachable Telegram stalled the detector for ten seconds per alert (audit M-14).
    """

    def __init__(
        self, client: TelegramClient | None, *, poll_seconds: float = POLL_SECONDS
    ) -> None:
        self._client = client
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backoff_until: dict[int, datetime] = {}

    def start(self) -> NotificationWorker:
        if self._client is None:
            logger.warning("Telegram is not configured; alerts will be recorded but not sent")
            return self

        self._thread = threading.Thread(target=self._run, name="notifier", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -- the loop ------------------------------------------------------------

    def _run(self) -> None:
        """Never dies (rule R7). A failure here is a delayed alert, not a silent one."""
        while not self._stop.is_set():
            try:
                sent = self.drain_once()
            except Exception:
                logger.exception("the notification queue failed a pass")
                sent = 0

            if sent == 0:
                self._stop.wait(self._poll)

    def drain_once(self) -> int:
        """Send whatever is due. Returns how many were attempted; testable directly."""
        if self._client is None:
            return 0

        pending = self._due(utcnow())
        for item in pending:
            self._attempt(item)
        return len(pending)

    def _due(self, now: datetime) -> list[Pending]:
        with session_scope() as session:
            rows = session.execute(
                select(
                    Notification.id,
                    Notification.event_id,
                    Notification.attempts,
                    Event.summary_text,
                    Event.confidence,
                    Event.snapshot_path,
                    Event.clip_path,
                    Event.occurred_at,
                    Event.reasons,
                )
                .join(Event, Event.id == Notification.event_id)
                .where(Notification.status == NotificationStatus.QUEUED)
                .order_by(Notification.id)
                .limit(20)
            ).all()

        return [
            Pending(
                notification_id=row.id,
                event_id=row.event_id,
                attempts=row.attempts,
                summary=row.summary_text,
                confidence=row.confidence,
                snapshot=row.snapshot_path,
                clip=row.clip_path,
                occurred_at=row.occurred_at,
                reasons=unpack_reasons(row.reasons),
            )
            for row in rows
            if self._backoff_until.get(row.id, now) <= now
        ]

    # -- one notification ----------------------------------------------------

    def _attempt(self, item: Pending) -> None:
        assert self._client is not None
        result = self._send(item)

        if result.ok:
            self._settle(item, NotificationStatus.SENT, result)
            logger.info("alert sent", extra={"event_id": item.event_id})
            return

        attempts = item.attempts + 1
        if result.permanent or attempts >= MAX_ATTEMPTS:
            self._settle(item, NotificationStatus.FAILED, result, attempts=attempts)
            # Loud. Giving up on an assault alert silently is not acceptable; giving up
            # audibly is the least we can do.
            logger.error(
                "ALERT NOT DELIVERED — a bullying alert could not be sent and has been given up on",
                extra={
                    "event_id": item.event_id,
                    "attempts": attempts,
                    "error": result.error,
                    "permanent": result.permanent,
                },
            )
            return

        wait = result.retry_after or _backoff(attempts)
        self._backoff_until[item.notification_id] = utcnow() + timedelta(seconds=wait)
        self._settle(item, NotificationStatus.QUEUED, result, attempts=attempts)
        logger.warning(
            "alert delivery failed, will retry",
            extra={
                "event_id": item.event_id,
                "attempt": attempts,
                "retry_in_seconds": round(wait),
                "error": result.error,
            },
        )

    def _send(self, item: Pending) -> SendResult:
        """Deliver the evidence itself, and fall back to words if we must.

        The legacy sent a text message containing a link to `http://127.0.0.1:8000/...` —
        the loopback address of the SERVER. A teacher reading that alert on their phone
        could never open it. Not once. Media delivery was broken by design.

        The caption is the same on every path. The text fallback is not a lesser message:
        when there is no picture it is the ONLY message, so it carries everything.
        """
        assert self._client is not None
        client, caption, root = self._client, item.caption, media_root()

        if not item.snapshot:
            return client.send_message(caption)

        result = client.send_photo(root / item.snapshot, caption)

        if result.permanent:
            # The picture is gone, or too big for Telegram. Retrying will not conjure it
            # back — but the alert itself still matters, so send the words.
            logger.warning(
                "could not send the snapshot; sending the alert as text",
                extra={"event_id": item.event_id, "error": result.error},
            )
            return client.send_message(caption)

        if result.ok and item.clip:
            # The clip is a bonus. Its failure must not undo an alert already delivered.
            client.send_video(root / item.clip, caption)

        return result

    def _settle(
        self,
        item: Pending,
        status: NotificationStatus,
        result: SendResult,
        attempts: int | None = None,
    ) -> None:
        def _update() -> None:
            with session_scope() as session:
                row = session.get(Notification, item.notification_id)
                if row is None:
                    return
                row.status = status
                row.attempts = attempts if attempts is not None else row.attempts + 1
                # R4: `result.error` is exception text out of httpx, and every request
                # this client makes goes to `https://api.telegram.org/bot<TOKEN>/...`.
                # httpx quotes the URL it was working on verbatim — `HTTPStatusError`
                # always does — so the bot token was landing in a column that anyone
                # holding the sqlite file can read. The log line written beside it was
                # already scrubbed by RedactingFormatter; the row was not. Redact BEFORE
                # the write, using the one redactor the log path uses: the legacy leaked
                # this same token six ways (audit C-02/C-03), and two redactors that
                # disagree is how that happens again.
                row.last_error = redact(result.error) if result.error else None
                row.provider_message_id = result.message_id
                if status is NotificationStatus.SENT:
                    row.sent_at = utcnow()

        with_retry(_update)


def _backoff(attempt: int) -> float:
    return min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)

"""Persisting events. The worker writes here directly; the web process only reads."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select

from qorgan.config.camera import CameraConfig
from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import Camera, Event
from qorgan.db.tenancy import owned_by, resolve_school_id
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import EventType, Severity, TelegramSkipReason
from qorgan.events.reasons import pack_reasons
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


def camera_id_for(name: str, school_id: int | None = None) -> int | None:
    """This school's camera with that name.

    **`cameras.name` stopped being globally unique at migration 0009**, so two schools may
    each have a `hall_left` -- and unscoped this returns whichever row the database reached
    first. A worker would then write its events against another school's camera row, which
    files a fight in the wrong school's log and sends the alert to the wrong people.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        return session.scalar(
            select(Camera.id).where(Camera.name == name, owned_by(Camera, school))
        )


def record_event(
    *,
    camera_id: int,
    occurred_at: datetime,
    verdict: Verdict,
    severity: Severity,
    summary_text: str,
    track_ids: str,
    snapshot_path: str | None = None,
    clip_path: str | None = None,
) -> int:
    """Insert one event and return its id.

    Wrapped in `with_retry`: under WAL a concurrent writer can still produce
    `database is locked`, and in the legacy that exception killed the event-writing
    thread outright and forever. The system then looked perfectly healthy while
    recording nothing at all (audit H-10). A locked database is a retry, never a death.
    """

    def _insert() -> int:
        with session_scope() as session:
            event = Event(
                camera_id=camera_id,
                event_type=EventType.BULLYING,
                occurred_at=occurred_at,
                confidence=verdict.confidence,
                candidate_probability=verdict.candidate_probability,
                validation_score=verdict.validation_score,
                skeleton_confirmed=verdict.skeleton_confirmed,
                severity=severity,
                # Written at its FINAL severity, before the insert. The legacy computed
                # this afterwards, so every row in its database claims to be a mere
                # suspicion no matter how certain the event was.
                summary_text=summary_text,
                # WHY, not just how sure. The reasons used to reach the log line and stop
                # there, so the alert a teacher read could say it was 93% certain and
                # never say what it had seen (client §7).
                reasons=pack_reasons(verdict.reasons),
                snapshot_path=snapshot_path,
                clip_path=clip_path,
                track_ids=track_ids,
            )
            session.add(event)
            session.flush()
            return event.id

    return with_retry(_insert)


def attach_media(
    event_id: int, *, snapshot_path: str | None = None, clip_path: str | None = None
) -> None:
    """Attach media that arrived after the event was written.

    The burst capture opens the HD stream and finishes *after* the event has usually
    already been recorded, so this back-fills it. COALESCE semantics: updating only the
    clip must not wipe the snapshot — the legacy's `update_media` overwrote both fields
    unconditionally and silently erased whichever one it was not given (audit M-20).
    """

    def _update() -> None:
        with session_scope() as session:
            event = session.get(Event, event_id)
            if event is None:
                logger.warning("no such event to attach media to", extra={"event_id": event_id})
                return
            if snapshot_path is not None:
                event.snapshot_path = snapshot_path
            if clip_path is not None:
                event.clip_path = clip_path

    with_retry(_update)


def raise_confidence(
    event_id: int,
    confidence: float,
    severity: Severity,
    summary: str,
    reasons: tuple[str, ...] = (),
) -> None:
    """A merged event turned out to be worse than the one already recorded.

    Only ever raises. A four-second fight produces many candidates; the row should end up
    describing its worst moment, not whichever frame happened to be written first.

    **The reasons move with the confidence, and for the same reason.** The first look at a
    fight is usually judged before the skeleton can confirm, so it writes a row with
    little or nothing to say for itself; the look that confirms is the one that raises the
    row *and* sends the Telegram. Since the notifier reads the reasons back out of the
    row, leaving the first look's reasons here would send a message explaining an incident
    other than the one being reported.
    """

    def _update() -> None:
        with session_scope() as session:
            event = session.get(Event, event_id)
            if event is None or event.confidence >= confidence:
                return
            event.confidence = confidence
            event.severity = severity
            event.summary_text = summary
            event.reasons = pack_reasons(reasons)

    with_retry(_update)


def record_telegram_decision(event_id: int, reason: TelegramSkipReason | None) -> None:
    """Write down whether a human was told about this event, and if not, why (client §7).

    Called on EVERY event that reaches the notification decision, including the ones that
    are sent — `reason=None` clears the column. Clearing matters as much as setting: a
    fight's first look is usually judged before the skeleton can confirm and is withheld
    with a reason, and the look that finally confirms raises the same row and sends it. A
    row left saying "не отправлено — нет подтверждения скелета" beside a message already
    on a teacher's phone is worse than saying nothing, because it would be read.

    Unconditional rather than conditional, for the reason `attach_media` is COALESCE and
    not an overwrite: the two halves of "was it sent" must be written by one statement, or
    they disagree the first time somebody adds a branch.

    **R4 has nothing to redact here and that is by construction.** This column takes a
    member of a closed enum or nothing at all — never an exception string, never a response
    body, never anything a third party authored. `notifications.last_error` is the column
    that had to learn this lesson: it carried the live bot token into the database because
    it stores text httpx wrote. Redaction belongs at the write, so the way to not need it
    at the write is to store a token from a set we defined.
    """

    def _update() -> None:
        with session_scope() as session:
            event = session.get(Event, event_id)
            if event is None:
                logger.warning(
                    "no such event to record a notification decision for",
                    extra={"event_id": event_id},
                )
                return
            event.telegram_skip_reason = reason

    with_retry(_update)


def ensure_cameras(
    cameras: Mapping[str, CameraConfig], school_id: int | None = None
) -> dict[str, int]:
    """Make sure every configured camera has a row, and return name -> id.

    The CONFIG is the source of truth for which cameras exist; this table is a projection
    of it, so that events have something to point at and the dashboard has something to
    list. Note what is deliberately NOT stored: no stream URL, and therefore no password.
    The legacy kept `rtsp://admin:<password>@host` in this table in plaintext, for all
    fifteen rows, and served the column to anyone who asked (audit C-01, C-03).
    """

    def _sync() -> dict[str, int]:
        ids: dict[str, int] = {}
        with session_scope() as session:
            school = resolve_school_id(session, school_id)
            for name, config in cameras.items():
                # Scoped for the same reason as `camera_id_for`, with a worse failure:
                # unscoped, this would find ANOTHER school's `hall_left` and overwrite its
                # host, port and role with this school's config -- silently repointing a
                # camera row that another school's events already hang off.
                row = session.scalar(
                    select(Camera).where(Camera.name == name, owned_by(Camera, school))
                )
                if row is None:
                    row = Camera(name=name, school_id=school)
                    session.add(row)

                row.display_name = config.display_name
                row.location = config.location
                row.camera_type = config.camera_type
                row.role = config.role
                row.rtsp_host = config.rtsp.host
                row.rtsp_port = config.rtsp.port
                row.priority = config.priority
                row.enabled = config.enabled
                row.updated_at = utcnow()

                session.flush()
                ids[name] = row.id
        return ids

    return with_retry(_sync)

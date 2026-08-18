"""Why an alert did not arrive, out of the notification rows themselves.

The queue is a table precisely so that this question has an answer (`notify.queue`): the
legacy spawned a raw thread per alert with a bare `requests.post` in it and swallowed
every exception, so an alert that failed left no trace anywhere -- no retry, no record,
no log line. Here every attempt updates `status`, `attempts` and `last_error`, and this
module reads them back.

**Scope, stated because the gap matters.** This answers "an alert was QUEUED and did not
arrive". It does not answer "no alert was ever queued for this event", and it must not try
to: an event nobody attempted to deliver has no notification row here to read. That
question is now answered on the event itself -- `events.telegram_skip_reason`, written by
the bullying worker at the moment it decides (`qorgan.enums.TelegramSkipReason`) and shown
on /events beside the incident it belongs to.

The alternative was to *recompute* it here from the camera config, and that is exactly the
temptation this project keeps losing to: the threshold may have been retuned since, and a
merged duplicate legitimately has no notification, so the recomputed answer would be
confidently wrong for the two cases anyone would look this up for. Recording the decision
where it is taken is what removed the temptation rather than resisting it.

**WHY THIS PANEL TAKES A SCHOOL, AND WHY `None` MEANS NOTHING RATHER THAN EVERYTHING.**
This module is reached from `/logs`, gated on `VIEW_DIAGNOSTICS` -- which `UserRole.
SUPERADMIN` holds, because §14 gives them "управление серверами". But this panel is not
the server. Every row carries a camera's name and the minute an incident involving a child
happened, which is one school's data told from the delivery side; the rest of `/logs` (the
journal, the worker table) genuinely is the installation's.

So the capability is right for the page and wrong for this one panel, and **R5 cannot see
that**: it walks the route table and checks every route is GUARDED, which this one is. It
has no way to ask whether the guard chosen was the right one. `roles.py` argues that the
only account able to reach every school must be the account that reaches the fewest
children; a superadmin belongs to no school, so no alert here is theirs, and an empty
panel is the honest answer. Refusing them the whole page instead would take away the log
viewer §14 gives them by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Event, Notification
from qorgan.db.tenancy import owned_by
from qorgan.enums import NotificationStatus
from qorgan.redaction import redact
from qorgan.settings import get_settings

ALERTS_PAGE_SIZE = 20


@dataclass(frozen=True, slots=True)
class UndeliveredAlert:
    """One queued alert that has not reached anybody.

    Note what is NOT on here: `Event.summary_text`. It is on /events, where the question
    is what happened to a child; the question here is why a message did not send, and the
    camera and the time answer it. "Never a child's name unnecessarily" is a rule about
    the word *unnecessarily* -- the summary is generated text today and carries no name,
    but a page that renders whatever is nearby is the page that renders the name the day
    somebody puts one in the column.
    """

    event_id: int
    occurred_at: str
    camera: str
    status: str
    attempts: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class AlertPage:
    """One page of undelivered alerts, and the numbers that describe it.

    The page number travels WITH the rows rather than beside them: the route asked for a
    page, this is the page it actually got, and a "7 / 3" over page-3 rows is the shape of
    lie this project keeps paying for.
    """

    alerts: tuple[UndeliveredAlert, ...]
    page: int
    pages: int
    total: int


def telegram_configured() -> bool:
    """The commonest answer to "why was no alert sent", and a live fact rather than a guess.

    With no token or no chat id, `TelegramClient.from_settings()` returns None and
    `NotificationWorker.start` logs "alerts will be recorded but not sent" and starts no
    thread at all. Every alert then sits at QUEUED with zero attempts forever, which reads
    like a stuck queue unless the page says why.
    """
    return get_settings().telegram_enabled


def undelivered(
    page: int, school_id: int | None, page_size: int = ALERTS_PAGE_SIZE
) -> AlertPage:
    """One school's queued-but-unsent alerts, newest first, one page at a time.

    `school_id=None` returns an EMPTY page rather than everybody's -- see the module
    docstring, which is where that argument lives because it is about §14 and not about
    this function.

    **The page number is clamped against the count, and the clamped value is what comes
    back.** An OFFSET past the last row returns nothing, and nothing on this panel is
    rendered as "все поставленные в очередь тревоги доставлены" -- so `?alerts_page=99`
    would answer "did my alerts go out?" with a confident yes. The same value has to be
    true in the query and in the "3 / 3" the reader sees, so only one of them computes it.
    """
    if school_id is None:
        return AlertPage((), 1, 1, 0)

    with session_scope() as session:
        unsent = Notification.status != NotificationStatus.SENT
        total = _total(session, unsent, school_id)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)

        rows = session.execute(
            _joined(
                select(
                    Notification.event_id,
                    Notification.status,
                    Notification.attempts,
                    Notification.last_error,
                    Event.occurred_at,
                    Camera.display_name,
                ),
                unsent,
                school_id,
            )
            .order_by(Notification.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        ).all()

    return AlertPage(tuple(_row(row) for row in rows), page, pages, total)


def _total(session, unsent, school_id: int) -> int:
    """Counted through the SAME joins the rows are selected through, not over
    `notifications` alone.

    An inner join drops a notification whose event or camera has gone, so a count over the
    wider population would put "17" in the heading above twelve rows and page as if there
    were seventeen. The cascades make that unlikely rather than impossible, and "unlikely"
    is how this class of bug reads right up until it does not. The school filter rides the
    same helper for the same reason.
    """
    return int(
        session.scalar(_joined(select(func.count(Notification.id)), unsent, school_id)) or 0
    )


def _joined(query, unsent, school_id: int):
    """The population this panel is about, written once so the count cannot describe a
    different set of rows from the ones underneath it.

    The school filter belongs here for the same reason: the count and the rows must be
    about one population, and "one school's undelivered alerts" is that population.
    """
    return (
        query.join(Event, Event.id == Notification.event_id)
        .join(Camera, Camera.id == Event.camera_id)
        .where(unsent, owned_by(Camera, school_id))
    )


def _row(row) -> UndeliveredAlert:
    return UndeliveredAlert(
        event_id=row.event_id,
        occurred_at=row.occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
        camera=row.display_name,
        status=row.status.value,
        attempts=row.attempts,
        # REDACTED HERE, and this one is not theoretical. `last_error` is
        # `f"{type(exc).__name__}: {exc}"` from an httpx failure, and every request this
        # client makes goes to `https://api.telegram.org/bot<TOKEN>/sendPhoto` -- httpx
        # puts the URL it was calling into the exception text. So the bot token is sitting
        # in this column in plaintext, and it never passed the log formatter on the way in.
        # Redaction is at the FORMATTER for logs and it has to be here for the database,
        # or the same secret is masked in one layer and rendered in the next.
        last_error=redact(row.last_error) if row.last_error else None,
    )

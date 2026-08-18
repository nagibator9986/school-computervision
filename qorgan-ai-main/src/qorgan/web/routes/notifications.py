"""The delivery history: every alert this system tried to send, and what became of it.

**Why this page exists at all.** The legacy spawned a raw thread per alert with a bare
`requests.post` inside it and swallowed every exception (audit M-14). When Telegram was
unreachable for the ten seconds the alert existed, the alert simply ceased to exist: no
retry, no row, no log line. A bullying alert that never arrived was indistinguishable
from a quiet afternoon, and the teacher who was never told had no way to find out they
had not been told. `notify.queue` fixed the losing; this page fixes the *not knowing*.

So the page lists SENT and UNSENT alike. A history that shows only what worked answers
the question nobody needs answered.

Three constraints shape the code below:

  * **Pagination is not optional.** The legacy loaded its entire event table on every
    render, every 2.5 seconds, per client (audit M-19). The notification table gains a row
    per alert and is never pruned, so it is the table that reaches 40 000 rows first.

  * **Nothing here sends anything.** Rendering is reading. The legacy restarted its AI
    workers from an HTTP handler -- with a five-second `thread.join()` in it -- whenever
    somebody opened a tab. `notify.queue.NotificationWorker` owns delivery; a second
    sender racing it is how you get an alert delivered twice, or not at all.

  * **`last_error` is untrusted text.** It is written from a third party's response body
    and from exception strings. It is escaped (Jinja, never `|safe`) and redacted before
    it is rendered -- see `_row`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, Request
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session
from starlette.responses import Response

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Event, Notification
from qorgan.db.tenancy import owned_by
from qorgan.enums import NotificationStatus
from qorgan.notify.message import local_time
from qorgan.redaction import redact
from qorgan.roles import Capability
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

router = APIRouter()

# **VIEW_BULLYING, and the reasoning is worth writing down because the page's NAME argues
# for something else.** "Notifications" sounds like plumbing -- a delivery log, an
# operational health screen, the sort of thing you would hand to whoever keeps the server
# running. It is not. Every row here carries a bullying event's summary, its camera and
# the minute it happened; a row reading "НЕ ДОСТАВЛЕНО — Зафиксирована агрессия, Холл
# слева, 04.03.2026 14:12" IS the bullying log, told from the delivery side. §14 says
# canteen staff work "БЕЗ доступа к буллингу", and §14 does not care what the URL is
# called. Gate on what a page DISCLOSES, never on what it is named after.
#
# A separate `VIEW_NOTIFICATIONS` capability was considered and rejected. `VIEW_MEDIA` was
# split in two because one grant covered two genuinely different disclosures -- the
# evidence for one incident, and a photographic register of every pupil. There is no such
# seam here: strip the summary and the camera and you still disclose that an incident
# occurred, when, and on which camera, which is the same secret. A second capability over
# the same facts would only create a second door into §14 -- one that could be granted to
# a role explicitly denied the first, by someone reading the capability's name.
#
# It follows that anyone who can open /events can open this, and nobody else. That is the
# intent: the person who is meant to receive these alerts is the person who needs to know
# one did not arrive.
viewer = Depends(require_capability(Capability.VIEW_BULLYING))

# Same as /events, deliberately. Two pages of the same incidents disagreeing about how
# much is a page is the sort of small inconsistency that gets "fixed" by removing one.
PAGE_SIZE = 25

# The status column holds a machine token; a human needs a sentence. Mapped here rather
# than in the template because a template that decides anything is a template that nobody
# tests. `НЕ ДОСТАВЛЕНО` is shouted on purpose: it is the only word on this page that
# means a child's alert reached nobody.
STATUS_LABELS: dict[NotificationStatus, str] = {
    NotificationStatus.QUEUED: "в очереди",
    NotificationStatus.SENT: "доставлено",
    NotificationStatus.FAILED: "НЕ ДОСТАВЛЕНО",
}

# An empty cell reads as a missing value; this reads as "there wasn't one".
NOTHING = "—"


@dataclass(frozen=True, slots=True)
class NotificationRow:
    """One delivery attempt, already formatted, already redacted. The template only prints.

    Every string here is final: the escaping is Jinja's, and everything else -- the clock,
    the redaction, the label -- has happened before the template sees it.
    """

    id: int
    event_id: int
    summary: str
    camera: str
    occurred_at: str
    channel: str
    status: str
    status_label: str
    attempts: int
    sent_at: str
    error: str


@router.get("/notifications")
def notifications_page(
    request: Request,
    page: int = 1,
    status_filter: str | None = None,
    user=viewer,
) -> Response:
    """Read-only, and that is a design decision rather than an omission.

    There is no "retry" button here. Re-sending an alert about a child is a decision with
    consequences -- it re-notifies a group of adults about an incident they may have
    already handled -- and it belongs where it can be reasoned about, not one click away
    from a page a teacher opens to check something. `notify.queue` already retries with
    backoff and dead-letters loudly; a hand-rolled second sender competing with it is the
    legacy's thread-per-alert with better manners.
    """
    page = max(1, page)
    wanted = _wanted(status_filter)
    rows, total, undelivered = _page(page, wanted, school_of(user))

    return render(
        request,
        "notifications.html",
        notifications=rows,
        page=page,
        pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
        total=total,
        undelivered=undelivered,
        # The filter that was APPLIED, never the string that was typed. A page showing a
        # filter it silently ignored is lying about what it is showing -- and this is the
        # one value on the page that comes straight from the query string, so normalising
        # it here is also what keeps raw user text out of the markup entirely.
        status_filter=wanted.value if wanted else "",
        statuses=[(status.value, _label(status)) for status in NotificationStatus],
    )


def _label(status: NotificationStatus) -> str:
    """Through here from both the rows and the filter, so a status cannot be one thing in
    the table and another in the dropdown -- and so that a status nobody has written a
    label for degrades to its own name instead of raising. A `KeyError` here would 500 the
    page whose entire job is to tell a teacher an alert did not arrive."""
    return STATUS_LABELS.get(status, status.value)


def _wanted(status_filter: str | None) -> NotificationStatus | None:
    """A filter nobody defined shows everything, rather than 400-ing at a reader."""
    if not status_filter:
        return None
    try:
        return NotificationStatus(status_filter)
    except ValueError:
        return None


def _page(
    page: int, wanted: NotificationStatus | None, school_id: int
) -> tuple[list[NotificationRow], int, int]:
    """One page of history, the size of the whole history, and how much of it never arrived."""
    with session_scope() as session:
        counts = _counts(session, school_id)
        total = counts.get(wanted, 0) if wanted is not None else sum(counts.values())

        query = _through_the_join(select(Notification, Event, Camera.display_name), school_id)
        if wanted is not None:
            query = query.where(Notification.status == wanted)

        rows = session.execute(
            # Newest first: the alert somebody is looking for is almost always the last
            # one. By primary key rather than by `created_at`, which ties -- SQLite writes
            # several of these inside the same millisecond during a burst, and a page whose
            # order is undefined can show the same row twice and skip another.
            query.order_by(Notification.id.desc()).limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)
        ).all()

        return (
            [_row(notification, event, camera) for notification, event, camera in rows],
            total,
            counts.get(NotificationStatus.FAILED, 0),
        )


def _through_the_join(query: Select, school_id: int) -> Select:
    """A notification, its event, and that event's camera, for ONE school. Written once and
    applied to BOTH the rows and the counts.

    **The school filter lives here rather than at the two call sites, and that is the same
    argument the join is here for, one floor down.** `total` and the rows have to agree or
    the header lies about what the table shows; if the scope were applied separately, the
    two would agree only for as long as nobody edited one of them. A count that included
    another school's undelivered alerts would put a number on this page that no row
    explains -- and this is the page a teacher opens precisely to find out whether they
    were told about something.

    Counting one set of rows and rendering another is how `total` -- and therefore the page
    count, and therefore whether "вперёд" is drawn at all -- goes quietly wrong. A count
    taken straight off `notifications` includes rows this inner join drops, so the header
    would claim 30, the last page would render 28, and the reader would be told there is a
    page that comes back empty. Nothing in that is loud.

    Today the two agree, because `db.engine` sets `PRAGMA foreign_keys=ON` and both foreign
    keys cascade, so an event or a camera cannot be deleted out from under a notification.
    That is an invariant held in a different module: true, and not this page's to assume.
    Sharing the join makes the agreement structural instead of circumstantial.
    """
    return (
        query.join(Event, Event.id == Notification.event_id)
        .join(Camera, Camera.id == Event.camera_id)
        .where(owned_by(Camera, school_id))
    )


def _counts(session: Session, school_id: int) -> dict[NotificationStatus, int]:
    """Every total this page needs, in ONE pass.

    The page wants two numbers: how many rows the current filter matches, and how many
    alerts have failed outright. Asked separately that is two full scans per render on top
    of the page itself, and scanning the whole table on every render is the exact defect
    (M-19) the pagination above exists to undo. Grouping gets both from one.
    """
    rows = session.execute(
        _through_the_join(
            select(Notification.status, func.count(Notification.id)), school_id
        ).group_by(Notification.status)
    ).all()
    return {status: int(count) for status, count in rows}


def _row(notification: Notification, event: Event, camera: str) -> NotificationRow:
    return NotificationRow(
        id=notification.id,
        event_id=notification.event_id,
        summary=event.summary_text,
        camera=camera,
        # The school's wall clock, through the SAME function that stamps the alert on the
        # teacher's phone (`notify.message.local_time`). The column is UTC and the school
        # is UTC+5, so rendering it raw would put 09:12 on the page beside the 14:12 the
        # phone shows for that very alert -- and the person reading both is trying to work
        # out whether a message arrived. One function, so the two cannot drift apart.
        occurred_at=local_time(event.occurred_at),
        channel=notification.channel.value,
        status=notification.status.value,
        status_label=_label(notification.status),
        attempts=notification.attempts,
        sent_at=local_time(notification.sent_at) if notification.sent_at else NOTHING,
        # **Redacted here, at the boundary that renders it.** `last_error` is free text
        # built from a third party's response body and from exception strings, and it is
        # stored exactly as it arrived. The log path is scrubbed by `RedactingFormatter`
        # and the database column is not, so the same value is safe in one layer and raw
        # in the next -- and this page is the layer with a browser on the other end.
        # Doing it on the way in would not help either: rows written before any such fix
        # would still be raw, and this page has to be safe about what is already stored.
        # Audit H-04 is the legacy serving its live bot token to anyone who opened a page.
        error=redact(notification.last_error) if notification.last_error else NOTHING,
    )

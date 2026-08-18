"""Incidents a named person handed to the psychologist, and the act of handing one over.

**One writer.** `refer()` is the only function anywhere that writes `events.referred_at`
and `events.referred_by_id`, and it cannot write one without the other — they are two
halves of one statement, assigned together inside one transaction. A referral nobody
signed is the system referring a child, which §8 promised the school would never happen.

**A referral is NOT a status, and this module learned that the expensive way.** `refer()`
also set `status = REFERRED_TO_PSYCHOLOGIST` for one day. That token was already redundant
— every query below selects on `referred_at IS NOT NULL`, and none has ever looked at the
status — but writing it was not harmless. The review controls are drawn only for an
unreviewed event, so referring an incident **removed the school's ability to tell us the
detector was wrong about it**, from a browser, permanently: «This is the only channel
through which the school tells us we were wrong, and a detector nobody corrects never
improves» (`web.routes.events.review_event`). In the other order it destroyed a verdict
outright — a confirmed event, once referred, stopped saying it had been confirmed, and
`reviewed_at`/`reviewed_by_id` record who and when but never WHAT was decided.

So the two facts stay in two columns. `status` answers «был ли детектор прав»; these two
answer «куда это ушло и кто так решил». Neither can overwrite the other now, in either
order, and an event can be both referred and confirmed — which is what §14 gives the
psychologist («подтверждённые случаи») and what the old shape made unreachable.

Nothing here ranks. The list is by referral time, newest first, because that is the order
a person works through a list — not by severity, not by confidence, and not by how many
times a child appears. An order of interest is a judgement, and the judgement is the
reader's.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Event, User
from qorgan.db.tenancy import owned_by, resolve_school_id, scope
from qorgan.db.types import utcnow
from qorgan.logging_setup import get_logger

# The same converter the phone alert and the events page use, imported rather than
# reimplemented: a second formatter agrees on the day it is written and drifts on the day
# either is edited, and the two guesses about a naive datetime are five hours apart.
from qorgan.notify.message import local_time

logger = get_logger(__name__)

# How many referrals the cabinet shows. Bounded for the reason /lessons is: a page that
# renders every referral the school has ever made gets slower every term until somebody
# calls it broken. CHOSEN, not measured -- no school has used this page yet.
RECENT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class Referral:
    """One handover, already formatted for the page.

    `status` is the event's CURRENT status and may no longer say «передано психологу» --
    see the module docstring. It is shown rather than hidden: "this was referred, and has
    since been confirmed as bullying" is a different situation from "this was referred and
    nobody has looked at it since", and the psychologist is entitled to tell them apart.
    """

    event_id: int
    occurred_at: str
    referred_at: str
    # The username of the person who referred it, or "—" if that account has since been
    # deleted (the FK is ON DELETE SET NULL). Never blank and never guessed.
    referred_by: str
    camera: str
    severity: str
    summary: str
    status: str
    snapshot: str | None
    clip: str | None


class UnknownEvent(LookupError):
    """No event holds that id. Raised rather than returning None so that a route cannot
    turn a missing incident into a silent success."""


def refer(event_id: int, *, user_id: int, username: str, school_id: int | None = None) -> bool:
    """Hand this incident to the psychologist, in the name of the person doing it.

    Returns whether this was the FIRST referral of this event. Re-referring is allowed and
    is not an error -- an operator who presses the button twice meant it -- but it does not
    rewrite `referred_at` or `referred_by_id`: the handover happened when it happened, and
    the second press is not a new fact about the child.

    **`status` IS NOT TOUCHED HERE, and that is the whole correction of 2026-07-29.**
    Handing an incident to a person says nothing about whether the detector was right, and
    writing both facts into one column meant each silently erased the other -- see the
    module docstring. An incident stays reviewable after it is referred, and a verdict
    already reached survives being referred.

    **`school_id` scopes the lookup, and it is NOT `session.get`.** The id reaches this
    function from a URL, so a primary-key fetch would hand back whatever row that number
    names on the whole installation -- letting one school's operator write their own name
    onto another school's incident. An event belonging to another school raises
    `UnknownEvent`, exactly as a missing one does: which ids exist elsewhere on the
    installation is not this school's business. `None` means "the only school there is"
    and raises if there are several (`db.tenancy.resolve_school_id`), so a caller that
    forgets to say cannot silently reach across.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        event = session.scalar(
            scope(select(Event), Event, school).where(Event.id == event_id)
        )
        if event is None:
            raise UnknownEvent(f"no event with id {event_id}")

        first_time = event.referred_at is None
        if first_time:
            event.referred_by_id = user_id
            event.referred_at = utcnow()

    # The event id and the person, never the summary: a log line outlives the request, is
    # pasted into tickets, and is read by whoever is on call.
    logger.info(
        "event referred to the psychologist by a person",
        extra={"event_id": event_id, "by": username, "first_time": first_time},
    )
    return first_time


def referred_incidents(
    limit: int = RECENT_LIMIT, *, school_id: int | None = None
) -> tuple[Referral, ...]:
    """Every incident somebody handed over IN ONE SCHOOL, newest handover first."""
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        rows = session.execute(_query(limit, school)).all()
        return tuple(_row(event, camera, who) for event, camera, who in rows)


def referral_count(session: Session, school_id: int) -> int:
    """How many handovers exist in this school. Taken on a caller's session so the cabinet
    index does not open a second transaction per block.

    **Scoped separately from the list beside it, and that is the point.** The count and
    `_query` are two statements built out of the same table; a filter on one that vouched
    for the other would leave the cabinet reporting every school's handovers above a list
    showing one school's -- a number too big and nothing anywhere saying so.
    """
    from sqlalchemy import func

    return int(
        session.scalar(
            select(func.count(Event.id))
            .join(Camera, Camera.id == Event.camera_id)
            .where(Event.referred_at.is_not(None), owned_by(Camera, school_id))
        )
        or 0
    )


def _query(limit: int, school_id: int):
    """`referred_at IS NOT NULL`, within one school. There has never been a status token to
    select on, and for one day there was one that this query still, correctly, ignored.

    The status moves on when the same incident is later confirmed or dismissed. The
    handover does not, so the two are read from two columns.

    The school comes off the event's CAMERA. The referring account is left unfiltered on
    purpose: it is reached through `referred_by_id` on a row already confined to this
    school, and filtering the outer join too would blank the name on any referral whose
    author has since moved -- which is the one thing that join exists to prevent.
    """
    return (
        select(Event, Camera.display_name, User.username)
        .join(Camera, Camera.id == Event.camera_id)
        # OUTER, because `referred_by_id` is ON DELETE SET NULL: an inner join would drop
        # the referral entirely the day the account that made it is removed, which would
        # quietly shorten this list rather than show a referral with no name on it.
        .outerjoin(User, User.id == Event.referred_by_id)
        .where(Event.referred_at.is_not(None), owned_by(Camera, school_id))
        .order_by(Event.referred_at.desc())
        .limit(limit)
    )


def _row(event: Event, camera: str, username: str | None) -> Referral:
    return Referral(
        event_id=event.id,
        occurred_at=local_time(event.occurred_at),
        referred_at=local_time(event.referred_at) if event.referred_at else "—",
        referred_by=username or "—",
        camera=camera,
        severity=event.severity.value,
        summary=event.summary_text,
        status=event.status.value,
        snapshot=event.snapshot_path,
        clip=event.clip_path,
    )

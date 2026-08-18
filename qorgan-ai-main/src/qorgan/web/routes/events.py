"""The events log: what the detector found, and what a human made of it.

Two legacy defects shape this file:

  * **Pagination.** The legacy loaded the entire events table and the entire canteen log
    on every render, every 2.5 seconds, per client (audit M-19). That is fine with 400
    rows and fatal with 40 000, and it gets worse every single day the system runs.

  * **Reviewing.** A detector nobody corrects never improves. Marking an event as a false
    positive is not a cosmetic feature: it is the only channel through which the school
    tells us we are wrong, and it feeds the eval harness.

And one this file carried until 2026-07-26:

  * **The school's time, not UTC.** This page printed `occurred_at` straight off the
    column, which is UTC by column type, while the Telegram alert about the same row
    printed Almaty (UTC+5). One incident, two answers, five hours apart — and a teacher
    sent to the wrong five minutes of CCTV finds an empty corridor and concludes the
    system is lying to them (`HANDOVER.md` §6a). The conversion is not repeated here; the
    alert's `local_time` is imported. See the import for why that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from sqlalchemy import func, select
from starlette.responses import RedirectResponse, Response

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Event, User
from qorgan.db.tenancy import owned_by, scope
from qorgan.db.types import utcnow
from qorgan.enums import EventStatus, EventType, Severity, TelegramSkipReason
from qorgan.logging_setup import get_logger

# The SAME converter the phone alert uses, imported rather than reimplemented. A second
# formatter here would agree with the first on the day it was written and drift on the day
# either one is edited — which is precisely the shape of the defect being fixed, a value
# true in one layer and quietly wrong in the next. It also refuses a naive datetime rather
# than guessing at it, and the two guesses are five hours apart.
from qorgan.notify.message import local_time

# The ONE writer of a referral. Imported rather than reimplemented here so that the status,
# the name and the minute cannot come apart: see `psychologist.referrals`.
from qorgan.psychologist.referrals import UnknownEvent, refer
from qorgan.roles import Capability
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()

# Reading the log and passing judgement on a child are two different things, and §14's
# roles divide on exactly that line: reviewing is separate so that a role can one day
# read an event without being able to rule on it.
viewer = Depends(require_capability(Capability.VIEW_BULLYING))
reviewer = Depends(require_capability(Capability.REVIEW_BULLYING))
# §9's «передано психологу». A separate right from REVIEW_BULLYING because it is a
# separate act: a verdict says whether this was an assault, a referral hands a named child
# to a specialist, and the second outlives the first. See `roles.Capability`.
referrer = Depends(require_capability(Capability.REFER_TO_PSYCHOLOGIST))

PAGE_SIZE = 25

# The one filter value on the events page that is NOT an `EventStatus`. §9 asks that an
# operator be able to find «передано психологу», and after 2026-07-29 that is not a status
# -- it is `events.referred_at`. Named once here so the route, the template and the tests
# cannot disagree about the string.
REFERRED_FILTER = "referred"

# `TelegramSkipReason` in the language of the person asking "почему мне не пришло
# уведомление о драке?". Mapped here and not in the template, for the reason
# /notifications maps its statuses here: a template that decides anything is a template
# nobody tests. Missing entries degrade to the token rather than raising -- a KeyError on
# the page whose job is to explain a silence would replace the explanation with a 500.
SKIP_REASON_LABELS: dict[TelegramSkipReason, str] = {
    TelegramSkipReason.SKELETON_NOT_RUN: "поза не анализировалась",
    TelegramSkipReason.NO_SKELETON_CONFIRMATION: "нет подтверждения скелета",
    TelegramSkipReason.WEAK_EVIDENCE_ONLY: "только слабые признаки (социальный контакт)",
    TelegramSkipReason.LOW_CONFIDENCE: "недостаточная уверенность",
    TelegramSkipReason.ALREADY_NOTIFIED: "уже сообщено по этому инциденту",
}


def skip_reason_label(reason: TelegramSkipReason) -> str:
    return SKIP_REASON_LABELS.get(reason, reason.value)


@dataclass(frozen=True, slots=True)
class EventRow:
    id: int
    # Already in the school's wall clock by the time it reaches the template. Formatting
    # in `_row` rather than in Jinja keeps the one conversion in one place; a filter in
    # the template would be a second place for it to be forgotten.
    occurred_at: str
    camera: str
    severity: str
    confidence: float
    summary: str
    status: str
    skeleton_confirmed: bool
    snapshot: str | None
    clip: str | None
    # Empty when a Telegram was raised for this event, or when it predates the column.
    # The template prints it and nothing else; it never infers a silence from its absence,
    # because "" cannot tell "was sent" from "was written before we recorded this".
    telegram_skip_reason: str
    # Both empty when this incident was never handed to the psychologist. Read off
    # `referred_at`/`referred_by_id` and NOT off `status`, so an event referred and later
    # confirmed still shows who referred it -- confirming an assault does not un-refer the
    # child. `referred_by` is "—" rather than blank when the account has since been
    # removed, because blank would read as "nobody did this".
    referred_at: str
    referred_by: str


@router.get("/events")
def events_page(
    request: Request,
    page: int = 1,
    status_filter: str | None = None,
    user=viewer,
) -> Response:
    page = max(1, page)
    rows, total = _page(page, status_filter, school_of(user))

    return render(
        request,
        "events.html",
        events=rows,
        page=page,
        pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
        total=total,
        status_filter=status_filter or "",
        statuses=[s.value for s in EventStatus],
        referred_filter=REFERRED_FILTER,
    )


@router.post("/events/{event_id}/review")
def review_event(
    event_id: int,
    request: Request,
    verdict: str = Form(...),
    user=reviewer,
) -> Response:
    """A human's judgement on a BULLYING event. This is the only channel through which the
    school tells us we are wrong, and a detector nobody corrects never improves.

    **This school's bullying events only** — see `_reviewable_row_or_404`, which is where
    both of those are enforced and why they are one function rather than two.
    """
    try:
        new_status = EventStatus(verdict)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unknown verdict") from None

    if new_status is EventStatus.NEW:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot un-review an event")

    # **A referral cannot arrive here at all, and it is the ENUM that guarantees it.**
    # `EventStatus` carries no referral member (see its docstring), so
    # "referred_to_psychologist" fails the parse above and this handler answers 400 without
    # having to know what it refused. That is stronger than the explicit rejection which
    # stood here until 2026-07-29: an explicit branch has to be remembered again for the
    # next member somebody adds, and the parse does not.
    #
    # This handler never touches `referred_at`, and `refer()` never touches `status`. So
    # ruling on an incident cannot un-refer a child, and handing a child on cannot erase
    # the school's verdict -- or its ability to reach one, which is what it used to do.
    #
    # `user` is the account the capability dependency already loaded, NOT a second
    # `current_user(request)` lookup: the referral work wrote one and the tenancy work
    # removed it, and keeping both would issue two queries for one answer.
    with session_scope() as session:
        event = _reviewable_row_or_404(session, event_id, school_of(user))
        event.status = new_status
        event.reviewed_by_id = user.id
        event.reviewed_at = utcnow()

    logger.info(
        "event reviewed",
        extra={"event_id": event_id, "verdict": new_status.value, "by": user.username},
    )
    return RedirectResponse("/events", status_code=status.HTTP_303_SEE_OTHER)


def _reviewable_row_or_404(session, event_id: int, school_id: int) -> Event:
    """This id, if it is a BULLYING event THIS SCHOOL may rule on.

    **Two checks, one door, and neither may be dropped.** The SCHOOL comes from the
    tenancy work and the TYPE from §12.1's weapons work; they were written on branches
    that never saw each other and they answer different questions -- "whose row is this"
    and "what kind of row is this". Both were separately the only thing standing between
    a caller and a row they must not touch, so this function asks both. Merging the two
    helpers rather than keeping one each is what stops the next route picking a door.

    **The type is checked on the ROW.** This route is guarded by `REVIEW_BULLYING`;
    §12.1's weapon alerts are answered through `POST /weapons/{id}/rule`, which is guarded
    by `CONFIRM_WEAPON_ALERT` and accepts only the two verdicts that ARE a ruling. Until
    this check existed, a weapon alert could be moved off `NEW` from HERE by guessing an
    id -- including to `reviewed`, which takes the row out of «ожидает подтверждения
    человека» while asserting nothing at all, so the panel stops offering the buttons and
    no human ever answers the question. That is exactly the state
    `weapons.store.WEAPON_VERDICTS` excludes, and this route knew nothing about it.
    Measured before the check: POST with `verdict=confirmed` returned 303 and set
    `reviewed_by_id`; so did `verdict=reviewed`.

    Two consequences, neither obvious from the diff:

      * `weapons/store.py`'s claim that nothing in `src/` moves a weapon row off NEW except
        `rule_on_weapon_alert` is true again. It was false.
      * whether a DEVELOPER may declare a child armed is now decided by ONE thing, the grant
        of `CONFIRM_WEAPON_ALERT`. Narrowing that grant would have changed nothing while
        this door was open, because `REVIEW_BULLYING` stays either way.

    **The school is checked with a SCOPED SELECT, not `session.get`.** A primary key fetch
    asks the database for whatever row that number names, so one school's operator could
    pass judgement on another school's incident about another school's children, writing
    their name into `reviewed_by_id`. Scoped through the event's camera, which knows the
    school.

    404 for every refusal, and always the same 404: the answer `/weapons/{id}/rule` gives
    for a bullying id, and the answer a real event belonging to another school gets.
    Neither route tells a caller which kind of row an id it may not touch happens to be,
    and a 404 that could be told apart from a 403 is a directory of the others.
    """
    mine = scope(select(Event), Event, school_id)
    event = session.scalar(mine.where(Event.id == event_id))
    if event is None or event.event_type is not EventType.BULLYING:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such bullying event")
    return event


@router.post("/events/{event_id}/refer")
def refer_event(event_id: int, user=referrer) -> Response:
    """§9: «передано психологу». **A person's act, recorded with that person's name.**

    The system never does this to a child on its own -- there is no measurement anywhere
    that reaches this handler, and `qorgan.classroom` deliberately produces no score that
    could. What happens here is that a member of staff decided, and the row says who.

    **Scoped to the caller's school**, for the same reason `review_event` above is: the id
    arrives out of a URL, and an unscoped handover would let one school's operator write
    their own name onto another school's incident about another school's child. `refer`
    answers `UnknownEvent` for an id belonging elsewhere, which is the same 404 a caller
    gets for an id that does not exist -- whether it exists on this installation is not
    this school's business.
    """
    try:
        # Idempotent on purpose: a second press does not rewrite the first handover's time
        # or its name. `refer` reports which it was to the log, not to the page.
        refer(event_id, user_id=user.id, username=user.username, school_id=school_of(user))
    except UnknownEvent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event") from None

    return RedirectResponse("/events", status_code=status.HTTP_303_SEE_OTHER)


def _filter_clause(status_filter: str):
    """The WHERE behind the page's filter box, or `None` if it named nothing we know.

    **`REFERRED_FILTER` is not an `EventStatus` and cannot be one.** A referral is not a
    verdict on the detector (see `EventStatus`), so it lives in its own column -- and the
    filter the psychologist's work needs therefore selects on `referred_at IS NOT NULL`
    rather than on a status token. It sits beside the four verdicts in one box because
    that is one question to the reader («покажи мне только эти»), not because the values
    behind it are the same kind of thing.

    An unknown string returns `None` and the page shows everything, unchanged: a filter
    somebody typed by hand is not an error worth a 400, and silently showing nothing would
    be worse than showing all.
    """
    if status_filter == REFERRED_FILTER:
        return Event.referred_at.is_not(None)
    try:
        return Event.status == EventStatus(status_filter)
    except ValueError:
        return None


def _page(page: int, status_filter: str | None, school_id: int) -> tuple[list[EventRow], int]:
    """One school's BULLYING events. §12.1's weapon alerts live on /weapons, not here.

    TWO filters, from two branches, and both are load-bearing: **the school**, or one
    school's operator reads another school's corridor; and **the type**, because this page
    would MISREPORT a weapon row -- it draws `skeleton_confirmed` as «скелет ✓/✗», and on
    a weapon row that is False only because there is no pose tier in that pipeline at all.
    The grants are separate too (`VIEW_BULLYING` here, `VIEW_WEAPONS` there), so leaving
    weapon rows would make the first a second door onto the second -- the §14 mistake
    `roles.py` exists about.

    `total` carries both filters too, and `counter` joins nothing of its own: a filter on
    `query` alone would leave the header counting rows the page never shows.
    """
    with session_scope() as session:
        mine = owned_by(Camera, school_id) & (Event.event_type == EventType.BULLYING)
        query = (
            select(Event, Camera.display_name, User.username)
            .join(Camera, Camera.id == Event.camera_id)
            # OUTER, and outer twice over: most events were never referred, and the account
            # that referred one may since have been removed (the FK is ON DELETE SET NULL).
            # An inner join would silently drop every event from the log in both cases.
            #
            # The referring account is NOT scoped to `school_id` and must not be: it is
            # joined through `events.referred_by_id`, and the event it hangs off has
            # already been confined to this school by `mine`. Filtering the join as well
            # would blank the name on any referral made by an account since moved or
            # removed, which is the one thing this outer join exists to prevent.
            .outerjoin(User, User.id == Event.referred_by_id)
            .where(mine)
        )
        counter = (
            select(func.count(Event.id)).join(Camera, Camera.id == Event.camera_id).where(mine)
        )

        clause = _filter_clause(status_filter) if status_filter else None
        if clause is not None:
            query = query.where(clause)
            counter = counter.where(clause)

        total = int(session.scalar(counter) or 0)

        rows = session.execute(
            query.order_by(Event.occurred_at.desc()).limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)
        ).all()

        return [_row(event, camera, who) for event, camera, who in rows], total


def _row(event: Event, camera: str, referred_by: str | None) -> EventRow:
    return EventRow(
        id=event.id,
        occurred_at=local_time(event.occurred_at),
        camera=camera,
        severity=event.severity.value,
        confidence=round(event.confidence, 2),
        summary=event.summary_text,
        status=event.status.value,
        skeleton_confirmed=event.skeleton_confirmed,
        snapshot=event.snapshot_path,
        clip=event.clip_path,
        # The school's first question about this system is "why was I not told?", and
        # /notifications cannot answer it: an alert that was never attempted has no row
        # there. It is answered on the row it is a property of.
        telegram_skip_reason=(
            skip_reason_label(event.telegram_skip_reason) if event.telegram_skip_reason else ""
        ),
        # From the referral columns, never from the status: see `EventRow`.
        referred_at=local_time(event.referred_at) if event.referred_at else "",
        referred_by=(referred_by or "—") if event.referred_at else "",
    )


SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.ALERT: 1, Severity.SUSPICION: 2}

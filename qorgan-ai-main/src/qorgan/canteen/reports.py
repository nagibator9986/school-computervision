"""Canteen reports. Chiefly: **who has no meal record today.**

That is the school's "who did not eat" question, answered as far as it honestly can be —
and the gap between the two phrasings is the whole of `DayReport.caveat()`. We know who
we have no record for. Whether that child ate is something we know only when the camera
recognised them, so the number is never presented, on any surface, without saying so.

The legacy could not answer this question at all, and the reason is worth understanding.
Its canteen log only ever contained pupils who had been *seen*. Asking "who did not eat"
means asking about children who are, by definition, absent from that log — so the answer
had to come from joining against the full roster, and there was no roster to join against.
The school's most important question about the canteen was unanswerable by the system
built to answer it.

Here the join is trivial, because `persons` is the roster.

**EVERY STATEMENT IN `day_report` IS SCOPED SEPARATELY, AND NONE STANDS IN FOR ANOTHER.**
`never_came` is the roster minus the meals. If the roster were one school's and the meals
were the installation's, a child of THIS school who ate would be cancelled by nothing and
a child of another school who ate would cancel a name that is not theirs -- so "who did
not eat", the one number on this page the school acts on, would be wrong in both
directions at once while looking entirely ordinary. The two counts of unattributed and
force-closed sessions cannot use the roster join at all: they are defined by `person_id IS
NULL`, so they reach their school through the entry camera instead (`db.tenancy.route`).

`school_id=None` means "the only school there is" and RAISES once there are several
(`db.tenancy.resolve_school_id`) -- the loud failure a caller who forgot to plumb the
school deserves, rather than a report about whoever happened to sort first. The web routes
never rely on it: they pass the school of the account making the request, because on the
day the fallback stops working they must already be right.

A note on days. Sessions are stored in UTC, but "today" is a wall-clock question asked by
a person standing in a school in Almaty. A day therefore runs from local midnight to local
midnight, converted to UTC for the query — not from UTC midnight, which in this timezone
falls at six in the morning and would split every school day in half.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.engine import session_scope
from qorgan.db.models import CanteenSession, Person
from qorgan.db.tenancy import owned_by, resolve_school_id, scope
from qorgan.enums import CloseReason, PersonType, SessionOutcome
from qorgan.identity.naming import display_name
from qorgan.settings import get_settings

# The one fact every surface that counts meals owes its reader, in one wording.
#
# `DayReport.caveat()` below does NOT use this constant, and that is deliberate rather than
# an oversight: its own first sentence quotes the labels a reader can SEE on the /canteen
# page and in the CSV («Нет записи о питании», `never_came`), and a caveat that explains a
# term nobody is looking at explains nothing. This constant is for the surfaces that count
# meals with no such label to quote -- `psychologist.attendance`, where the number is a
# count of days across weeks. Same fact, two readers; one place each, and this comment is
# the link between them so neither can drift alone.
NO_RECORD_IS_NOT_A_MISSED_MEAL = (
    "Отсутствие записи не означает, что ребёнок не ел. Это записи камер: ребёнка, "
    "которого не узнали на входе, здесь нет вовсе — и пока камеру столовой не перевесили, "
    "не узнают почти никого."
)


@dataclass(frozen=True, slots=True)
class Meal:
    person_id: int
    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None
    outcome: SessionOutcome | None
    dwell_seconds: float | None
    opened_at: datetime

    @property
    def display(self) -> str:
        return display_name(self)


@dataclass(frozen=True, slots=True)
class DayReport:
    day: date
    ate: tuple[Meal, ...]
    came_but_did_not_eat: tuple[Meal, ...]
    never_came: tuple[Person, ...]
    unknown_sessions: int
    # Sessions the janitor force-closed as UNKNOWN by TIMEOUT: nobody was ever recognised
    # at the exit. This is the measured price of a strict exit `min_score` (spec A §2.2,
    # config/identity.py) -- a hole we can count, chosen over a false meal record we
    # cannot detect. It does NOT count sessions closed normally.
    #
    # It measures a DIFFERENT failure from `unknown_sessions`, on a different camera:
    # `unknown_sessions` is an entry never attributed to anybody (person_id IS NULL),
    # this is a pupil who entered and was never recognised on the way OUT. The two
    # predicates are not disjoint -- a NULL-person session that also times out is counted
    # by BOTH -- so they must never be summed, and no surface may present them as parts of
    # a total. `qorgan pupils report` and the /canteen page each show them as two separate
    # counts for exactly that reason, and each shows both ALWAYS, including zero: a hidden
    # zero cannot be told apart from a value that was never computed, or dropped on the way
    # to the surface. (`summary()` below is the one-line headline, not the instrument
    # panel: it carries `unknown_sessions` only.)
    forced_unknown: int

    @property
    def did_not_eat(self) -> int:
        """Pupils with NO MEAL RECORD today. Not "pupils who did not eat" -- see `caveat`.

        The school does care about this number, and that is exactly why it must not be
        read as more than it is. It counts pupils we have no record for, and a missing
        record has three causes, only two of which mean the child went hungry:

        1. the child really did not eat;
        2. the child ate but was never recognised -- an unattributed entry opens a session
           with `person_id` NULL, which yields no `Meal` for any named pupil, so the child
           is absent from `seen` and lands in `never_came` under their real name;
        3. the child was never detected at all -- no session, named or unknown.

        The value is honest and stays as it is: correcting it would mean guessing WHICH
        pupils case 2 covers, which is an invented number, the disease we are curing.
        `unknown_sessions` bounds case 2 from above (UP TO N, never N). Case 3 is
        UNMEASURED and cannot be bounded -- we cannot count what we never detected -- so
        no surface may present this number as certain, on any day. Never render it without
        `caveat()`.
        """
        return len(self.came_but_did_not_eat) + len(self.never_came)

    def caveat(self) -> tuple[str, ...]:
        """The sentences that must travel with `never_came` / `did_not_eat`, everywhere.

        One wording, rendered by every surface (the /canteen page, its CSV export, and
        `qorgan pupils report`), because four hand-written copies of a caveat is four
        chances for one of them to quietly stop being true -- which is this codebase's
        signature disease, and the reason this caveat exists at all.

        Sentence 1 holds on every day, whatever the counts. Sentence 2 exists only when
        there is an N to talk about, and says UP TO N: an unattributed session may be
        staff, a visitor, or a child who was also recognised elsewhere in the day. Its
        absence says nothing -- see `did_not_eat` on the third, unmeasured failure mode.
        """
        # Quote words the reader can actually SEE. This used to gloss «Не приходили», a
        # label that survives on no surface: after the tile relabel the page says «Нет
        # записи о питании» (tile and heading) and the CSV says `never_came`. A caveat
        # that explains a term nobody is looking at explains nothing -- and the CSV's
        # opaque token is the one that most needs saying out loud, because that file is
        # the one the school forwards.
        lines = [
            "«Нет записи о питании» на странице и `never_came` в выгрузке значат одно: "
            "за этот день нет записи о питании этого ученика. "
            "Это не означает, что ученик не ел."
        ]
        if self.unknown_sessions:
            lines.append(
                f"Сегодня {self.unknown_sessions} сессий не удалось привязать к ученику: "
                f"до {self.unknown_sessions} из перечисленных могли поесть, но остаться "
                "неузнанными. Точнее сказать нельзя — непривязанная сессия может быть "
                "и сотрудником, и гостем, и учеником, которого узнали в другой раз."
            )
        return tuple(lines)

    def summary(self) -> str:
        return (
            f"{self.day}: {len(self.ate)} ate, "
            f"{len(self.came_but_did_not_eat)} came but did not eat, "
            f"{len(self.never_came)} with no meal record. "
            f"{self.unknown_sessions} session(s) could not be attributed to a pupil."
        )


def local_day_bounds(day: date) -> tuple[datetime, datetime]:
    """A school day runs from local midnight to local midnight, expressed in UTC.

    In Almaty, UTC midnight falls at six in the morning — using it would split every
    school day in half and put breakfast on the wrong date.
    """
    tz = get_settings().tz
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def day_report(day: date, school_id: int | None = None) -> DayReport:
    """Who ate, who came but did not eat, and who has no meal record at all, for ONE school.

    The third group is not "who never came": an unattributed entry yields no `Meal` for
    any named pupil, so a child who ate unrecognised lands there. See
    `DayReport.did_not_eat`, and never render that group without `DayReport.caveat()`.

    Each of the four statements below is scoped separately, and `school_id=None` means
    "the only school there is" and raises once there are several. The module docstring
    argues both, because both are about the report rather than about this function.
    """
    start, end = local_day_bounds(day)

    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        meals = _meals_between(session, start, end, school)
        roster = session.scalars(
            select(Person).where(
                Person.person_type == PersonType.STUDENT,
                Person.is_active.is_(True),
                owned_by(Person, school),
            )
        ).all()

        seen = {meal.person_id for meal in meals}
        never_came = tuple(_detach(session, person) for person in roster if person.id not in seen)

        unknown_count = _count_unknown(session, start, end, school)
        forced_unknown_count = _count_forced_unknown(session, start, end, school)

    ate = tuple(m for m in meals if m.outcome is SessionOutcome.ATE)
    did_not = tuple(m for m in meals if m.outcome is not SessionOutcome.ATE)

    return DayReport(
        day=day,
        ate=ate,
        came_but_did_not_eat=did_not,
        never_came=never_came,
        unknown_sessions=unknown_count,
        forced_unknown=forced_unknown_count,
    )


def _meals_between(
    session: Session, start: datetime, end: datetime, school_id: int
) -> list[Meal]:
    rows = session.execute(
        select(
            CanteenSession.person_id,
            CanteenSession.outcome,
            CanteenSession.dwell_seconds,
            CanteenSession.opened_at,
            Person.external_id,
            Person.full_name,
            Person.person_type,
            Person.class_name,
            Person.position,
        )
        .join(Person, Person.id == CanteenSession.person_id)
        .where(
            CanteenSession.opened_at >= start,
            CanteenSession.opened_at < end,
            # Through the PERSON, which this statement already joins for the name and the
            # class. The session's own route to a school is its entry camera, and both
            # answers are the same school -- a pupil cannot open a session on another
            # school's camera, because the gallery a camera matches against is its own
            # school's roster. Using the join that is already here avoids a second one.
            owned_by(Person, school_id),
        )
        .order_by(CanteenSession.opened_at)
    ).all()

    # One meal per pupil per day: the best outcome wins. A child who came twice, and ate
    # once, ate.
    best: dict[int, Meal] = {}
    for row in rows:
        meal = Meal(
            person_id=row.person_id,
            external_id=row.external_id,
            full_name=row.full_name,
            person_type=row.person_type,
            class_name=row.class_name,
            position=row.position,
            outcome=row.outcome,
            dwell_seconds=row.dwell_seconds,
            opened_at=row.opened_at,
        )
        current = best.get(row.person_id)
        if current is None or _rank(meal.outcome) > _rank(current.outcome):
            best[row.person_id] = meal

    return list(best.values())


def _count_unknown(session: Session, start: datetime, end: datetime, school_id: int) -> int:
    """Scoped through the ENTRY CAMERA, and it has to be: these sessions have no person.

    `person_id IS NULL` is the definition of the number, so the join `_meals_between` uses
    is not available here -- there is nobody to join to. `db.tenancy.route` sends
    `CanteenSession` through `entry_camera_id`, which cannot be null, for exactly this
    case.
    """
    from sqlalchemy import func

    return int(
        session.scalar(
            scope(select(func.count(CanteenSession.id)), CanteenSession, school_id).where(
                CanteenSession.opened_at >= start,
                CanteenSession.opened_at < end,
                CanteenSession.person_id.is_(None),
            )
        )
        or 0
    )


def _count_forced_unknown(
    session: Session, start: datetime, end: datetime, school_id: int
) -> int:
    """Sessions the janitor closed by TIMEOUT: nobody was ever recognised at the exit.

    This is the measured price of the exit camera's strict `min_score` (spec A §2.2) --
    counted, not assumed, so that if the price spikes we see it rather than guess.
    """
    from sqlalchemy import func

    return int(
        session.scalar(
            scope(select(func.count(CanteenSession.id)), CanteenSession, school_id).where(
                CanteenSession.opened_at >= start,
                CanteenSession.opened_at < end,
                CanteenSession.close_reason == CloseReason.TIMEOUT,
            )
        )
        or 0
    )


def _rank(outcome: SessionOutcome | None) -> int:
    """A child who came twice and ate once, ate."""
    order = {
        SessionOutcome.ATE: 3,
        SessionOutcome.NOT_ATE: 2,
        SessionOutcome.UNKNOWN: 0,
    }
    return order.get(outcome, 0) if outcome else 0


def _detach(session: Session, person: Person) -> Person:
    session.expunge(person)
    return person

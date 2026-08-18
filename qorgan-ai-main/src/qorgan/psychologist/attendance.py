"""One named child's canteen attendance, week by week.

**This is the only genuinely longitudinal signal in the system that is about a named
child, and the reason is where the identity comes from.** The canteen recognises a face at
a door, at conversational distance, against a roster the school itself issued — so
«ходил обедать каждый день и перестал» is a statement about a person, not about a track.
It needs none of the classroom identification §8 forbids, and none of the corridor
recognition this school's own footage measured at zero (14 970 faces, median 11.5 px, none
recognised).

**It is EMPTY today, and the table exists anyway.** The canteen camera is still pointed at
the wrong place, so almost nobody is recognised and almost no session is attributed. The
accumulation has to exist before that is fixed, or the count starts at zero on the morning
of the pilot and the school is told to wait another eight weeks. The page says which of
those two situations it is in, every time — see `SignalState`.

**NOTHING HERE COMPUTES A CONCLUSION.** No baseline, no trend line, no "падение", no
threshold, no colour that means "worrying". Counts per week, oldest week first, and a
person reads them. §8 promised the school «Никаких диагнозов и никаких направлений к
психологу от системы», and an arrow drawn on a chart is a diagnosis with better graphics.

The four-week personal norm §8 also promises is NOT built here, and this module is where
it would be easiest to sneak in. It is left out on purpose: §8's own text forbids the
identification that a per-child norm requires inside a classroom, so the promise is in
tension with itself and resolving it is the school's decision, not ours
(`qorgan.psychologist.__init__`, `docs/questions-for-school.md` §10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.canteen.reports import NO_RECORD_IS_NOT_A_MISSED_MEAL
from qorgan.db.engine import session_scope
from qorgan.db.models import CanteenSession, Person
from qorgan.db.tenancy import owned_by, resolve_school_id
from qorgan.enums import SessionOutcome
from qorgan.psychologist.signals import SignalState, signal_label
from qorgan.settings import get_settings

# How many weeks the page shows. **CHOSEN, NOT MEASURED** -- no school has used this page,
# and there is no recording to tune it against.
#
# It is deliberately NOT 4. §8 promises the school a comparison against «его собственной
# нормой за предыдущие 4 недели`, and a window of exactly four weeks on this page would
# read as that norm having been implemented. It has not been, it cannot be for the
# classroom half, and a window that quietly implies otherwise is the whole disease.
WEEKS = 8

# Monday. `date.weekday()` is 0 for Monday, which is what the arithmetic below assumes.
_WEEK_STARTS_ON = 0


@dataclass(frozen=True, slots=True)
class Week:
    """One week of one child's meal record. Counts only."""

    starts_on: date
    # Distinct LOCAL dates in this week on which the cameras recorded a session for this
    # child. Days, not sessions: a child who is recognised twice at one lunch has not
    # attended twice, and a count of sessions would make a recognition glitch look like an
    # extra meal.
    days_present: int
    sessions: int
    # Sessions that closed with `ATE`. Always shown beside `sessions` and never instead of
    # it: they measure different things, and `UNKNOWN` outcomes are common enough that
    # presenting only this one would understate the record.
    meals_recorded: int


@dataclass(frozen=True, slots=True)
class AttendanceTrend:
    person_id: int
    external_id: str
    display: str
    class_name: str | None
    is_active: bool
    weeks: tuple[Week, ...]
    total_sessions: int
    first_record: date | None

    @property
    def state(self) -> SignalState:
        """LIVE only when this child actually has a record. Otherwise EMPTY -- which here
        means "the camera has not been moved yet", not "this child does not eat"."""
        return SignalState.LIVE if self.total_sessions else SignalState.EMPTY

    @property
    def label(self) -> str:
        """The state in words, from the one table that holds them (`signals.py`).

        The template used to spell these two strings out itself, which meant the cabinet
        index and this page could start describing the same state differently after an edit
        to either. `Block.label` reads the same table; a template that decides anything is
        a template nobody tests.
        """
        return signal_label(self.state)

    def caveat(self) -> tuple[str, ...]:
        """The sentences that travel with these counts, on every surface that shows them.

        Sentence 1 holds whatever the numbers are. The rest exist only when there is
        something to say; an absent line asserts nothing.
        """
        lines = [NO_RECORD_IS_NOT_A_MISSED_MEAL]
        if self.total_sessions:
            lines.append(
                f"Всего записей по этому ученику: {self.total_sessions}, самая ранняя — "
                f"{self.first_record}. Это счётчики, а не вывод: система не сравнивает "
                "ребёнка с нормой и не отмечает падение. Что означают эти числа, решает "
                "человек."
            )
        else:
            lines.append(
                "По этому ученику нет ни одной записи. Это НЕ значит, что он не ходит в "
                "столовую: камеру столовой ещё не перевесили, поэтому узнаётся почти "
                "никто. Таблица ведётся уже сейчас, чтобы к моменту переноса камеры "
                "накопление не начиналось с нуля."
            )
        return tuple(lines)


def attendance_trend(
    person_id: int, *, weeks: int = WEEKS, school_id: int | None = None
) -> AttendanceTrend | None:
    """This child's meal record, one row per week, oldest first. `None` if no such person.

    `None` rather than an empty trend, for the reason `person_history` returns it: an empty
    page under an id nobody holds reads as "this child never ate", which is a claim about a
    child who does not exist. Another school's child is the same `None`, for the same
    reason and one more: whether that id exists elsewhere on the installation is not this
    school's business.

    **The school is carried into every helper below rather than left to the id.** The
    lookup here already confines `person_id` to one school, so each `.where` under it is
    belt-and-braces today -- but these helpers take a bare integer, and the day one is
    called from a path that did not check, a filter on the STATEMENT still holds and an
    invariant on the ARGUMENT does not. That is the shape of defect this codebase keeps
    paying for: a value true in one layer and quietly wrong in the next.
    """
    weeks = max(1, weeks)
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        person = session.scalar(
            select(Person).where(Person.id == person_id, owned_by(Person, school))
        )
        if person is None:
            return None

        buckets = _week_starts(_today_local(), weeks)
        rows = _sessions_since(session, person_id, buckets[0], school)
        return AttendanceTrend(
            person_id=person.id,
            external_id=person.external_id,
            display=person.display,
            class_name=person.class_name,
            is_active=person.is_active,
            weeks=tuple(_week(start, rows) for start in buckets),
            total_sessions=_total(session, person_id, school),
            first_record=_first_record(session, person_id, school),
        )


def _today_local() -> date:
    """A school week is a wall-clock fact in Almaty. UTC midnight falls there at six in
    the morning, so bucketing on UTC dates would put Monday breakfast in Sunday."""
    return datetime.now(tz=get_settings().tz).date()


def _week_starts(today: date, weeks: int) -> list[date]:
    """The Monday of each of the last `weeks` weeks, oldest first, ending with this one."""
    this_monday = today - timedelta(days=(today.weekday() - _WEEK_STARTS_ON) % 7)
    return [this_monday - timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]


def _sessions_since(
    session: Session, person_id: int, first_monday: date, school_id: int
) -> list[tuple[date, SessionOutcome | None]]:
    """Every session for this child since `first_monday`, as (local date, outcome).

    Bucketed in Python rather than by a SQL date function on purpose: the conversion to
    Almaty is a property of `Settings`, and a database-side `date()` would silently bucket
    on UTC on one backend and on the server's locale on another.
    """
    tz = get_settings().tz
    start = datetime.combine(first_monday, time.min, tzinfo=tz).astimezone(UTC)
    rows = session.execute(
        select(CanteenSession.opened_at, CanteenSession.outcome)
        # Joined to the child, not to the entry camera. `CanteenSession` routes to a school
        # through `entry_camera_id` (see `db.tenancy`), and that route is right for the
        # canteen's own pages -- a session is opened by a camera and `person_id` may be
        # NULL when the face was not identified. Here the question is already about ONE
        # NAMED CHILD, so the school that matters is the child's.
        .join(Person, Person.id == CanteenSession.person_id)
        .where(
            CanteenSession.person_id == person_id,
            CanteenSession.opened_at >= start,
            owned_by(Person, school_id),
        )
        .order_by(CanteenSession.opened_at)
    ).all()
    return [(opened.astimezone(tz).date(), outcome) for opened, outcome in rows]


def _week(start: date, rows: list[tuple[date, SessionOutcome | None]]) -> Week:
    end = start + timedelta(days=7)
    inside = [row for row in rows if start <= row[0] < end]
    return Week(
        starts_on=start,
        days_present=len({day for day, _ in inside}),
        sessions=len(inside),
        meals_recorded=sum(1 for _, outcome in inside if outcome is SessionOutcome.ATE),
    )


def _total(session: Session, person_id: int, school_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(CanteenSession.id))
            .join(Person, Person.id == CanteenSession.person_id)
            .where(CanteenSession.person_id == person_id, owned_by(Person, school_id))
        )
        or 0
    )


def _first_record(session: Session, person_id: int, school_id: int) -> date | None:
    earliest = session.scalar(
        select(func.min(CanteenSession.opened_at))
        .join(Person, Person.id == CanteenSession.person_id)
        .where(CanteenSession.person_id == person_id, owned_by(Person, school_id))
    )
    return earliest.astimezone(get_settings().tz).date() if earliest else None

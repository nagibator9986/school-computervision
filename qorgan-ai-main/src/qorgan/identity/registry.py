"""Who is enrolled, one page at a time -- and whether we can recognise any of them.

**One page at a time is the whole design.** The legacy read the entire persons table (and
the entire canteen log, and the entire events table) on every render, every 2.5 seconds,
per connected client (audit M-19). That is survivable at this school's 142 people and
fatal at the 800 the system is sold into, and it degrades every day the thing runs. So the
read model is a PAGE, and there is no code path here that can return "all of them".

**`recognisable` is not a nicety.** Four of this school's staff photographs contain no
detectable face at all (spec §1.1): those people are on the roster, they have a
photograph, and the system can never recognise them. A registry that shows a row and says
nothing implies the opposite. The flag is computed the way `faces.gallery.load_gallery`
computes its input -- embeddings for ONE model name and version -- because a vector this
page counted and the gallery will not load is a promise of a recognition that cannot
happen. The legacy shipped a rebuild script that wrote a different model's 512-d vectors
into the same column (audit M-29); the dimensions match and nothing complains.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.engine import session_scope
from qorgan.db.models import FaceEmbedding, Person
from qorgan.db.tenancy import owned_by, scope
from qorgan.enums import PersonType
from qorgan.identity.naming import display_name

PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class PupilRow:
    person_id: int
    external_id: str
    display: str
    class_name: str | None
    person_type: PersonType
    recognisable: bool


@dataclass(frozen=True, slots=True)
class PupilPage:
    rows: tuple[PupilRow, ...]
    page: int
    pages: int
    total: int


def pupil_page(
    school_id: int,
    page: int = 1,
    *,
    page_size: int = PAGE_SIZE,
    model: FaceModelSettings | None = None,
) -> PupilPage:
    """One page of ONE school's register. Active people only -- see `_active_only`.

    `school_id` is the first argument and has no default on purpose. A default here would
    be a register that renders somebody's children when the caller forgot to say whose,
    and the forgetting is silent: the page looks right, it is merely longer than the
    school has pupils.
    """
    settings = model or FaceModelSettings()
    page = max(1, page)

    with session_scope() as session:
        total = int(
            session.scalar(
                select(func.count(Person.id)).where(_active_only(), owned_by(Person, school_id))
            )
            or 0
        )
        rows = session.execute(_query(settings, page, page_size, school_id)).all()

    return PupilPage(
        rows=tuple(_row(row) for row in rows),
        page=page,
        pages=max(1, (total + page_size - 1) // page_size),
        total=total,
    )


def _active_only():
    """Retired ids are not on the register, and they are not lost either.

    A merge deactivates the dropped id rather than deleting it, because the school issued
    that id and a record saying so is worth keeping (`identity/merge.py`). Those rows have
    a home: `identity/duplicates.py` lists them as resolved merges, with the undo. Showing
    them here as well would put an id the system has retired next to one it has not, with
    nothing on the row explaining the difference.
    """
    return Person.is_active.is_(True)


def _query(settings: FaceModelSettings, page: int, page_size: int, school_id: int):
    """Ordered by class then by the school's own id, which is how a register is read.

    `recognisable` is a correlated EXISTS, not a join: a person with three photographs must
    not become three rows, and a COUNT would make the page pay for rows nobody displays.
    """
    usable = (
        select(FaceEmbedding.id)
        .where(
            FaceEmbedding.person_id == Person.id,
            FaceEmbedding.model_name == settings.model_name,
            FaceEmbedding.model_version == settings.model_version,
        )
        .exists()
    )

    return (
        select(
            Person.id,
            Person.external_id,
            Person.full_name,
            Person.person_type,
            Person.class_name,
            Person.position,
            usable.label("recognisable"),
        )
        .where(_active_only(), owned_by(Person, school_id))
        .order_by(Person.class_name, Person.external_id)
        .limit(page_size)
        .offset((page - 1) * page_size)
    )


def _row(row) -> PupilRow:
    return PupilRow(
        person_id=row.id,
        external_id=row.external_id,
        display=display_name(row),
        class_name=row.class_name,
        person_type=row.person_type,
        recognisable=bool(row.recognisable),
    )


@dataclass(frozen=True, slots=True)
class MealRow:
    """One meal session, as the history page shows it."""

    opened_at: str
    closed_at: str | None
    state: str
    outcome: str | None
    dwell_seconds: float | None


@dataclass(frozen=True, slots=True)
class PersonHistory:
    person_id: int
    external_id: str
    display: str
    class_name: str | None
    person_type: PersonType
    is_active: bool
    meals: tuple[MealRow, ...]
    page: int
    pages: int
    total: int


def person_history(
    person_id: int, school_id: int, page: int = 1, *, page_size: int = PAGE_SIZE
) -> PersonHistory | None:
    """This child's meal record, newest first. `None` if nobody holds that id.

    Paginated for the same reason the register is: a session per meal per day is a row
    that arrives twice a day for eleven years, and this page must not get slower every
    term. Keyed on `person_id` and never on a name -- the legacy keyed identity on
    "Surname Firstname" plus a class parsed out of a photo filename, so two children with
    the same name in the same class became one person (audit H-02).
    """
    page = max(1, page)

    with session_scope() as session:
        # NOT `session.get(Person, person_id)`. The id comes off a URL, and a primary key
        # fetch would hand one school's operator another school's child -- their name,
        # their class, and every meal they have eaten. `None` is the same answer this
        # function already gives for an id nobody holds, which is right: whether that
        # child exists in some other school is not this school's business.
        person = session.scalar(
            select(Person).where(Person.id == person_id, owned_by(Person, school_id))
        )
        if person is None:
            return None
        total, meals = _meals(session, person_id, page, page_size, school_id)

        return PersonHistory(
            person_id=person.id,
            external_id=person.external_id,
            display=person.display,
            class_name=person.class_name,
            person_type=person.person_type,
            is_active=person.is_active,
            meals=meals,
            page=page,
            pages=max(1, (total + page_size - 1) // page_size),
            total=total,
        )


def _meals(
    session: Session, person_id: int, page: int, page_size: int, school_id: int
) -> tuple[int, tuple[MealRow, ...]]:
    """This child's meals, scoped through the camera that opened each session.

    Scoped even though `person_id` was already checked to be this school's, because the
    check and the read are two statements and only one of them is in front of you when
    you edit this. `CanteenSession` is a derived table -- it reaches a school through its
    entry camera -- so the route is `db.tenancy.route`'s and not written by hand here.
    """
    from qorgan.db.models import CanteenSession

    total = int(
        session.scalar(
            scope(select(func.count(CanteenSession.id)), CanteenSession, school_id).where(
                CanteenSession.person_id == person_id
            )
        )
        or 0
    )
    rows = session.scalars(
        scope(select(CanteenSession), CanteenSession, school_id)
        .where(CanteenSession.person_id == person_id)
        .order_by(CanteenSession.opened_at.desc())
        .limit(page_size)
        .offset((page - 1) * page_size)
    ).all()

    return total, tuple(
        MealRow(
            opened_at=row.opened_at.strftime("%Y-%m-%d %H:%M"),
            closed_at=row.closed_at.strftime("%Y-%m-%d %H:%M") if row.closed_at else None,
            state=row.state.value,
            outcome=row.outcome.value if row.outcome else None,
            dwell_seconds=row.dwell_seconds,
        )
        for row in rows
    )

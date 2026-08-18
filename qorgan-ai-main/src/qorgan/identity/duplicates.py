"""The six pairs the school owes us a decision on, and the merges already made.

`identity/report.py` measures them: the cross-person cosine band is EMPTY from 0.48 to
0.77, so the six pairs above 0.60 are not the tail of an impostor distribution, they are a
different population -- six people holding two school ids each, their meals split across
both. `identity/merge.py` executes a decision about them and refuses to make one.

This module is the read model between those two: it takes the pairs the report names and
adds what a human needs in order to DECIDE -- the photograph, the class and the person
type on each side. Nothing here merges anybody.

**Resolved merges are listed too, and that is not a log.** It is the undo. A merge across
the pupil/staff line drops a child out of the meal record entirely and nothing reports it,
so `MergeResult.summary()` says "merge back the other way" -- and until `--reactivate`
existed, merging back was refused, because the id you would merge into is the one the
merge retired. A one-way door on a decision made by looking at a photograph. The retired
ids are therefore kept in front of the operator, with their consequence, for as long as
the merge stands.

**A merge is reversed by direction, not undone.** Reversing hands EVERYTHING to the
formerly-dropped id, including rows the kept id owned before the merge. That is the
semantics `merge_persons(..., reactivate=True)` has; the page says so in words rather than
implying a tidy rollback that nothing in the domain performs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.engine import session_scope
from qorgan.db.models import Person, PersonPhoto
from qorgan.db.tenancy import owned_by, resolve_school_id, scope
from qorgan.enums import PersonType
from qorgan.identity.report import gallery_report


@dataclass(frozen=True, slots=True)
class Side:
    """One of the two ids. Everything the decision needs, and nothing else."""

    person_id: int
    external_id: str
    display: str
    class_name: str | None
    person_type: PersonType
    is_active: bool
    # Relative to MEDIA_ROOT, served by the authenticated `/media` handler behind
    # VIEW_PUPIL_PHOTOS -- never by StaticFiles, which is how the legacy served a
    # photographic register of every child in the school to anyone who guessed a URL.
    photo: str | None


@dataclass(frozen=True, slots=True)
class Pair:
    a: Side
    b: Side
    similarity: float

    @property
    def crosses_person_type(self) -> bool:
        """Same rule as `MergeResult.crosses_person_type`, asked one step earlier.

        On this school's `student_470 / staff_334` it decides whether that person is FED:
        staff never open a meal session, so the answer has to be visible BEFORE the button
        is pressed, not only in the summary afterwards.
        """
        return self.a.person_type is not self.b.person_type


@dataclass(frozen=True, slots=True)
class ResolvedMerge:
    """One merge that has already happened, and the two ids it joined."""

    kept: Side
    dropped: Side

    @property
    def crosses_person_type(self) -> bool:
        return self.kept.person_type is not self.dropped.person_type


@dataclass(frozen=True, slots=True)
class DuplicatesView:
    outstanding: tuple[Pair, ...]
    resolved: tuple[ResolvedMerge, ...]


def duplicates_view(
    model: FaceModelSettings | None = None, school_id: int | None = None
) -> DuplicatesView:
    """Everything the page shows, for ONE school. One read, no writes, no model on a GPU.

    `gallery_report` reads embeddings out of the database and multiplies a matrix; it does
    not open the face model. It is deliberately computed per request and NOT polled: the
    legacy re-read whole tables every 2.5 seconds per client (audit M-19), and this page is
    opened by a human, a handful of times, to make six decisions.

    **The school is carried into every read below, and the reason is sharper here than on
    a page that merely displays.** A "duplicate" is a claim that two school ids are ONE
    human. Across schools that claim is simply false -- two different children who resemble
    each other -- and this page puts a merge button under it. Acting on it would re-point
    one child's meals and photographs onto another child in another school and retire an
    id its own school still uses. `_sides` and `_first_photos` are scoped as well as the
    pair list, so a photograph cannot arrive from outside the school even if a stale id
    reaches them.
    """
    school = school_id
    report = gallery_report(model or FaceModelSettings(), school)

    with session_scope() as session:
        school = resolve_school_id(session, school)
        sides = _sides(
            session,
            {person for pair in report.duplicates for person in (pair.person_a, pair.person_b)},
            school,
        )
        return DuplicatesView(
            outstanding=tuple(
                Pair(
                    a=sides[pair.person_a],
                    b=sides[pair.person_b],
                    similarity=pair.similarity,
                )
                for pair in report.duplicates
                if pair.person_a in sides and pair.person_b in sides
            ),
            resolved=_resolved(session, school),
        )


def _resolved(session: Session, school_id: int) -> tuple[ResolvedMerge, ...]:
    """Every id a merge retired, newest first.

    Read off `merged_into_id`, never off `is_active`: that boolean answers "merged into
    another id" and "left the school" with one "no", and offering to revive somebody who
    left is the mistake the column was added to stop.
    """
    rows = session.scalars(
        select(Person)
        .where(Person.merged_into_id.is_not(None), owned_by(Person, school_id))
        .order_by(Person.updated_at.desc())
    ).all()

    sides = _sides(
        session,
        {person.id for person in rows} | {person.merged_into_id for person in rows},
        school_id,
    )
    return tuple(
        ResolvedMerge(kept=sides[row.merged_into_id], dropped=sides[row.id])
        for row in rows
        if row.merged_into_id in sides and row.id in sides
    )


def _sides(session: Session, person_ids: Iterable[int], school_id: int) -> dict[int, Side]:
    """The ids resolve to people only if they are THIS school's.

    An id from anywhere else simply does not come back, and the two callers already drop
    a pair whose sides are not both present -- so an out-of-school id removes the pair
    from the page rather than rendering half of it.
    """
    wanted = {int(person_id) for person_id in person_ids if person_id is not None}
    if not wanted:
        return {}

    photos = _first_photos(session, wanted, school_id)
    people = session.scalars(
        select(Person).where(Person.id.in_(wanted), owned_by(Person, school_id))
    ).all()

    return {
        person.id: Side(
            person_id=person.id,
            external_id=person.external_id,
            display=person.display,
            class_name=person.class_name,
            person_type=person.person_type,
            is_active=person.is_active,
            photo=photos.get(person.id),
        )
        for person in people
    }


def _first_photos(session: Session, person_ids: set[int], school_id: int) -> dict[int, str]:
    """One photograph per id -- the lowest-numbered, so two operators looking at the same
    pair on different days are looking at the same two faces.

    Scoped through `persons`, because this is the statement that returns a PATH to a
    photograph of a child and the path is then rendered into an `<img>` behind
    VIEW_PUPIL_PHOTOS. `PersonPhoto` is a derived table; `db.tenancy.route` sends it to
    its school through `person_id`.
    """
    rows = session.execute(
        scope(select(PersonPhoto.person_id, PersonPhoto.path), PersonPhoto, school_id)
        .where(PersonPhoto.person_id.in_(person_ids))
        .order_by(PersonPhoto.person_id, PersonPhoto.id)
    ).all()

    photos: dict[int, str] = {}
    for person_id, path in rows:
        photos.setdefault(person_id, path)
    return photos

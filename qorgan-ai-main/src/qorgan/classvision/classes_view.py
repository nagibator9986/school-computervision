"""Классы: the level that was missing between «здесь есть уроки» and «вот этот ребёнок».

**Why this module exists.** The cabinet used to open on a flat list of lessons, and the only
way to a child was to notice which place number they sat in. That is the wrong shape for the
person who reads these pages: a psychologist thinks in CLASSES — 8-А has a seating plan, 3-Б
does not — and only then in the children inside one. So the walk is now
`Классы → Класс → Ученик`, and each step answers the question the next one depends on.

**A class is not a room, and this is where that stops being invisible.** 8-А was recorded in
two rooms (`camera_02` and `D14`), and a place is discovered per camera: «место 1» in one room
and «место 1» in the other are different chairs with different histories, and only one room's
plan is signed. A class page that merged them would silently merge two children. So pupils are
grouped BY ROOM, each group says whether its plan is signed, and nothing is ever summed across
the two.

**Nothing here is a new measurement.** Every number on these pages is read from rows the
importer already wrote; this module groups them and states what is missing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models.classvision import (ClassvisionAttestation, ClassvisionLesson,
                                          ClassvisionPlace, ClassvisionPlaceLesson,
                                          ClassvisionRun)
from qorgan.db.models.person import Person

UNSIGNED_RU = "плана рассадки нет"


@dataclass(frozen=True, slots=True)
class ClassRow:
    """One class in the index: what was recorded, and how much of it is named."""

    class_key: str
    rooms: tuple[str, ...]
    lessons: int
    first_day: str
    last_day: str
    pupil_places: int
    attested_places: int
    coverage_median: float | None

    @property
    def fully_attested(self) -> bool:
        return self.pupil_places > 0 and self.attested_places == self.pupil_places


@dataclass(frozen=True, slots=True)
class PupilRow:
    """One place of one class: the chair, the name if a plan is signed, and its last figures."""

    place_id: int
    label_ru: str
    person_name: str
    role: str
    lessons: int
    coverage_median: float | None
    latest_index: float | None
    first_index: float | None

    @property
    def direction_ru(self) -> str:
        """The word first. A direction drawn only in colour is a direction nobody prints.

        Deliberately NOT a verdict: `metrics/trend.py` decides whether a change is a trend,
        against the child's own median and MAD, and that calculation does not live here. This
        is the plain difference between the first and last lesson in the list, so the label
        says «ниже», not «спад».
        """
        if self.latest_index is None or self.first_index is None:
            return "не установлено"
        delta = self.latest_index - self.first_index
        if abs(delta) < 5.0:
            return "в пределах разброса"
        return "ниже" if delta < 0 else "выше"


@dataclass(frozen=True, slots=True)
class RoomGroup:
    """The pupils of one class in ONE room, and whether that room's plan is signed."""

    camera_key: str
    signed_by: str
    decision_ref: str
    valid_from_ru: str
    pupils: list[PupilRow] = field(default_factory=list)

    @property
    def attested(self) -> bool:
        return bool(self.signed_by)


def _rooms_by_class(session: Session, *, school_id: int) -> dict[str, set[str]]:
    """Which rooms each class was recorded in. A class is not a room, and can span several."""
    rooms: dict[str, set[str]] = {}
    for class_key, camera_key in session.execute(
        select(ClassvisionLesson.class_key, ClassvisionLesson.camera_key)
        .where(ClassvisionLesson.school_id == school_id).distinct()
    ).all():
        rooms.setdefault(class_key, set()).add(camera_key)
    return rooms


def _place_counts(session: Session, *, school_id: int) -> tuple[dict[str, int], dict[str, int]]:
    """Pupil places per class, and how many of them a live seating plan names."""
    places = dict(session.execute(
        select(ClassvisionPlace.class_key, func.count(ClassvisionPlace.id))
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.role == "pupil")
        .group_by(ClassvisionPlace.class_key)
    ).all())
    attested = dict(session.execute(
        select(ClassvisionPlace.class_key,
               func.count(func.distinct(ClassvisionAttestation.place_id)))
        .join(ClassvisionAttestation, ClassvisionAttestation.place_id == ClassvisionPlace.id)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionAttestation.valid_to.is_(None))
        .group_by(ClassvisionPlace.class_key)
    ).all())
    return places, attested


def _class_coverages(session: Session, *, school_id: int, class_key: str) -> list[float]:
    """Every pupil place-lesson coverage in one class, as percentages."""
    return [round(value * 100, 1) for value in session.scalars(
        select(ClassvisionPlaceLesson.coverage)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionLesson.class_key == class_key)
        .where(ClassvisionPlaceLesson.role == "pupil")
    )]


def classes_index(session: Session, *, school_id: int) -> list[ClassRow]:
    """Every class that has an imported lesson, most recently recorded first."""
    lessons = session.execute(
        select(ClassvisionLesson.class_key,
               func.count(func.distinct(ClassvisionLesson.id)),
               func.min(ClassvisionLesson.date_local),
               func.max(ClassvisionLesson.date_local))
        .where(ClassvisionLesson.school_id == school_id)
        .group_by(ClassvisionLesson.class_key)
    ).all()

    rooms = _rooms_by_class(session, school_id=school_id)
    places, attested = _place_counts(session, school_id=school_id)

    rows = []
    for class_key, lesson_count, first_day, last_day in lessons:
        covers = _class_coverages(session, school_id=school_id, class_key=class_key)
        rows.append(ClassRow(
            class_key=class_key,
            rooms=tuple(sorted(rooms.get(class_key, set()))),
            lessons=int(lesson_count or 0),
            first_day=first_day.isoformat() if first_day else "дата не прочитана",
            last_day=last_day.isoformat() if last_day else "дата не прочитана",
            pupil_places=int(places.get(class_key, 0)),
            attested_places=int(attested.get(class_key, 0)),
            coverage_median=round(statistics.median(covers), 1) if covers else None,
        ))
    rows.sort(key=lambda row: row.last_day, reverse=True)
    return rows


def _pupils_of_room(session: Session, *, school_id: int, class_key: str,
                    camera_key: str) -> list[PupilRow]:
    """Every place in one room of one class, with the figures its own lessons carry."""
    places = session.scalars(
        select(ClassvisionPlace)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.class_key == class_key)
        .where(ClassvisionPlace.camera_key == camera_key)
        .order_by(ClassvisionPlace.role.desc(), ClassvisionPlace.ordinal)
    ).all()

    rows: list[PupilRow] = []
    for place in places:
        found = session.execute(
            select(ClassvisionPlaceLesson, ClassvisionLesson.date_local, Person)
            .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
            .join(ClassvisionRun, ClassvisionPlaceLesson.run_id == ClassvisionRun.id)
            # The school is named on BOTH ends: the place above and the person here. A row
            # binding one school's chair to another school's child is the leak worth never
            # rendering, and the join is where it would happen.
            .outerjoin(Person, (ClassvisionPlaceLesson.person_id == Person.id)
                       & (Person.school_id == school_id))
            .where(ClassvisionLesson.school_id == school_id)
            .where(ClassvisionPlaceLesson.place_id == place.id)
            .where(ClassvisionRun.run_id == ClassvisionLesson.selected_run_id)
            .order_by(ClassvisionLesson.date_local, ClassvisionLesson.id)
        ).all()

        indices = [row.activity_index for row, _, _ in found if row.activity_index is not None]
        covers = [round(row.coverage * 100, 1) for row, _, _ in found]
        person = next((p for _, _, p in found if p is not None), None)
        rows.append(PupilRow(
            place_id=place.id,
            label_ru=place.label_ru or f"место {place.ordinal}",
            person_name=(person.full_name or person.external_id) if person else "",
            role=place.role,
            lessons=len(found),
            coverage_median=round(statistics.median(covers), 1) if covers else None,
            latest_index=round(indices[-1], 1) if indices else None,
            first_index=round(indices[0], 1) if indices else None,
        ))
    return rows


def _room_signature(session: Session, *, school_id: int, class_key: str,
                    camera_key: str) -> ClassvisionAttestation | None:
    """The live seating plan for one room, or None.

    ONE signature stands for the room: the plan is signed for the class's seating and every
    place in it carries the same document. Read from `classvision_attestations` rather than
    from the name already copied onto the observations, because the two answer different
    questions — what the school stands behind NOW, versus what was true when the lesson was
    imported. If they ever disagree, this page should be able to show both, and it cannot do
    that from one of them.
    """
    return session.execute(
        select(ClassvisionAttestation)
        .join(ClassvisionPlace, ClassvisionPlace.id == ClassvisionAttestation.place_id)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.class_key == class_key)
        .where(ClassvisionPlace.camera_key == camera_key)
        .where(ClassvisionAttestation.valid_to.is_(None))
        .order_by(ClassvisionAttestation.valid_from.desc())
    ).scalars().first()


def class_view(session: Session, *, school_id: int, class_key: str) -> dict[str, object] | None:
    """One class: its rooms, the pupils in each, and what each room's plan says.

    Returns None when the class has no imported lesson, so the route can 404 rather than
    render an empty page that looks like a class with nothing in it.
    """
    lessons = session.scalars(
        select(ClassvisionLesson)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionLesson.class_key == class_key)
        .order_by(ClassvisionLesson.date_local.desc().nulls_last())
    ).all()
    if not lessons:
        return None

    groups: list[RoomGroup] = []
    for camera_key in sorted({lesson.camera_key for lesson in lessons}):
        pupils = _pupils_of_room(session, school_id=school_id, class_key=class_key,
                                 camera_key=camera_key)
        signature = _room_signature(session, school_id=school_id, class_key=class_key,
                                    camera_key=camera_key)
        groups.append(RoomGroup(
            camera_key=camera_key,
            signed_by=signature.attested_by if signature else "",
            decision_ref=signature.decision_ref if signature else "",
            valid_from_ru=signature.valid_from.strftime("%d.%m.%Y") if signature else "",
            pupils=pupils,
        ))

    pupil_places = sum(1 for group in groups for row in group.pupils if row.role == "pupil")
    named = sum(1 for group in groups for row in group.pupils if row.person_name)
    covers = [row.coverage_median for group in groups for row in group.pupils
              if row.coverage_median is not None]
    return {
        "class_key": class_key,
        "groups": groups,
        "lessons": lessons,
        "pupil_places": pupil_places,
        "named_places": named,
        "coverage_median": round(statistics.median(covers), 1) if covers else None,
        "first_day": min((l.date_local for l in lessons if l.date_local), default=None),
        "last_day": max((l.date_local for l in lessons if l.date_local), default=None),
    }

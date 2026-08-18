"""The list of lessons: one row each, and the two totals that must never become one.

Split out of `cabinet.py` when that file crossed the repo's 500-line limit. The seam is not
arbitrary: everything here answers «which lessons exist and how much of each was actually
seen», while what stayed behind answers «what happened inside one lesson».

**The four names imported from `cabinet` are the seam, and getting them wrong cost two 500s.**
The first cut of this split left `selected_run`, `median_or_none` and `frame_counts_by_lesson`
on THIS side while `cabinet.lesson_view` still called them, and both pages crashed the moment
a lesson existed — invisibly, because no test in the suite had a lesson in it. They now live
with the lesson they are about, and `tests/test_classvision_pages.py` builds a world before it
looks at anything.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.classvision.cabinet import (frame_counts_by_lesson, local_clock,
                                        median_or_none, selected_run)
from qorgan.db.models.classvision import (ClassvisionFrame, ClassvisionLesson,
                                          ClassvisionPlaceLesson, ClassvisionReading,
                                          ClassvisionRun)


@dataclass(frozen=True, slots=True)
class LessonRow:
    """One line of the lessons list, with the demo flag it may never be shown without."""

    id: int
    is_demo: bool
    date_ru: str
    iso_week: str
    camera_key: str
    class_key: str
    duration_minutes: float
    pupil_places: int
    coverage_median: float | None
    coverage_min: float | None
    clock_source: str
    part_count: int
    frames: int
    has_reading: bool
    overlap_allowed: bool


@dataclass(frozen=True, slots=True)
class GroupTotals:
    """Totals for ONE kind of row. There is deliberately no combined constructor."""

    lessons: int = 0
    places: int = 0
    place_minutes: int = 0
    coverage_median: float | None = None


@dataclass(frozen=True, slots=True)
class LessonsIndex:
    """One list and one set of totals.

    **`is_demo` survives in the database and is deliberately not shown.** The operator asked
    for the demonstration rows to carry no visible marker, which is their call to make about
    their own presentation; the column stays so that the synthetic term can be deleted in one
    statement when real recordings arrive, instead of being separated by hand from dates.
    Nothing here reads it, so nothing here can render it.
    """

    lessons: list[LessonRow] = field(default_factory=list)
    totals: GroupTotals = GroupTotals()
    undated: int = 0
    attested_places: int = 0


def lessons_index(session: Session, *, school_id: int) -> LessonsIndex:
    """Every imported lesson, newest first."""
    lessons = list(session.scalars(
        select(ClassvisionLesson)
        .where(ClassvisionLesson.school_id == school_id)
        .order_by(ClassvisionLesson.date_local.desc().nulls_last(), ClassvisionLesson.id.desc())
    ))
    frames = frame_counts_by_lesson(session, school_id=school_id)
    readings = _lessons_with_a_reading(session, school_id=school_id)
    rows: list[LessonRow] = []
    for lesson in lessons:
        run = selected_run(session, school_id=school_id, lesson=lesson)
        coverages = _coverages(session, school_id=school_id, run=run)
        row = LessonRow(
            id=lesson.id, is_demo=bool(lesson.is_demo),
            date_ru=lesson.date_local.isoformat() if lesson.date_local else "дата не прочитана",
            iso_week=(f"{lesson.iso_year}-W{lesson.iso_week:02d}"
                      if lesson.iso_week else "нет недели"),
            camera_key=lesson.camera_key, class_key=lesson.class_key,
            duration_minutes=round(lesson.duration_minutes, 1),
            pupil_places=run.pupil_places if run else 0,
            coverage_median=median_or_none(coverages),
            coverage_min=min(coverages) if coverages else None,
            clock_source=run.clock_source if run else "неизвестно",
            part_count=lesson.part_count, frames=frames.get(lesson.id, 0),
            has_reading=lesson.id in readings, overlap_allowed=bool(lesson.overlap_allowed),
        )
        rows.append(row)
    return LessonsIndex(
        lessons=rows,
        totals=_totals(session, school_id=school_id, rows=rows),
        undated=sum(1 for lesson in lessons if lesson.date_local is None),
        attested_places=_attested_places(session, school_id=school_id),
    )


def _totals(session: Session, *, school_id: int, rows: list[LessonRow]) -> GroupTotals:
    if not rows:
        return GroupTotals()
    ids = [row.id for row in rows]
    minutes = session.scalar(
        select(func.sum(ClassvisionPlaceLesson.observed_seconds))
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.lesson_id.in_(ids))
    ) or 0.0
    covers = [row.coverage_median for row in rows if row.coverage_median is not None]
    return GroupTotals(
        lessons=len(rows), places=sum(row.pupil_places for row in rows),
        place_minutes=int(minutes / 60.0), coverage_median=median_or_none(covers))


def _coverages(session: Session, *, school_id: int,
               run: ClassvisionRun | None) -> list[float]:
    if run is None:
        return []
    return [round(value * 100, 1) for value in session.scalars(
        select(ClassvisionPlaceLesson.coverage)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.run_id == run.id)
        .where(ClassvisionPlaceLesson.role == "pupil")
    )]


def _lessons_with_a_reading(session: Session, *, school_id: int) -> set[int]:
    """Which lessons have a note that PASSED the guard. A withheld note is not a note."""
    return set(session.scalars(
        select(ClassvisionRun.lesson_id)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .join(ClassvisionReading, ClassvisionReading.run_id == ClassvisionRun.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionReading.guard_passed.is_(True))
    ))


def _attested_places(session: Session, *, school_id: int) -> int:
    return int(session.scalar(
        select(func.count(func.distinct(ClassvisionPlaceLesson.place_id)))
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.person_id.is_not(None))
    ) or 0)

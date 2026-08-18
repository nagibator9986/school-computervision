"""Which PLACE a discovered seat is, and which LESSON an imported run belongs to.

Both questions are invisible in one artefact and only exist once a second recording arrives,
which is why they live here rather than upstream. Both were defects before they were design,
and the write-ups are `classvision/INTEGRATION.md` §2a and §6a.

**A seat number is not a key across lessons.** `room/seats.py` numbers discovered clusters
in reading order, so ONE PUPIL ABSENT ON TUESDAY shifts every id behind the gap. Appending
`seat_3` to `seat_3` across a term therefore builds a history out of two or three children
and produces a textbook sustained decline that is pure bookkeeping — and nothing can notice,
because both rows genuinely say `seat_3`. So a seat is matched to a place by GEOMETRY, with
a gate and a margin, and a seat that fails either is stored with `place_id = NULL` rather
than joined to the nearest history.

**Three unmatched outcomes, named separately because they need different actions**: `new`
(nobody has seen this place), `ambiguous` (the room was re-arranged and two places are
equally close — the honest answer is a question for the teacher), and a run in which EVERY
seat is `new`, which is what a moved camera looks like and is visible as exactly that.

**A lesson is not a file.** A re-analysis of the same recording is a new RUN on the same
lesson and must not move `selected_run_id`; a DVR that cut one hour into two files is one
lesson chained by `continues_lesson_id`; two files that cover the same wall-clock hour are
that hour counted twice in every weekly total and are refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models.classvision import (ClassvisionAttestation, ClassvisionLesson,
                                          ClassvisionPlace, ClassvisionRun)

# How close a seat must be to a known place's anchor, in shoulder widths, to BE that place.
# `room/seats.py::assign` uses 1.5 WITHIN a lesson, where a miss costs one observation; here
# a miss costs a term of one child's history welded onto another's, so this is tighter.
# CHOSEN below the 1.5-2.5 shoulder widths that separate two pupils sharing a desk, so the
# gate cannot span a neighbour. Measured on the two real recordings of camera_01: 9 places
# created from the first, 9 of 9 recognised in the second, 0 unmatched, 0 ambiguous.
PLACE_MATCH_GATE_SCALES = 1.0

# ...and the nearest place must be this many times closer than the runner-up. MEASUREMENTS
# §4 applied to geometry rather than to faces: a best score with no margin over the second
# best is a preference, not an identification. CHOSEN.
PLACE_MATCH_MARGIN = 2.0

# A recorder rolling over to a new file. Generous for a DVR, far too tight to join two real
# lessons: the shortest break in this school's timetable is 5 minutes and a class changeover
# takes longer than the break.
CONTINUATION_GAP_SECONDS = 180.0

# How far a continuation may appear to start BEFORE its predecessor ended. File one starts
# 10:17:59 and runs 818.00 s, ending 10:31:37; file two starts 10:31:36. A genuine seam
# therefore reads as a one-second OVERLAP because both clocks are known only to the second,
# so a window of [0, gap] would miss the exact pair this check exists for.
CONTINUATION_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class PlaceMatch:
    """Which place this seat is, and — when it is none of them — which of the three cases."""

    place: ClassvisionPlace | None
    match: str  # matched | new | ambiguous
    reason_ru: str
    distance: float | None  # shoulder widths, None when there was nothing to compare


def known_places(session: Session, *, school_id: int, camera_key: str,
                 class_key: str) -> list[ClassvisionPlace]:
    return list(
        session.scalars(
            select(ClassvisionPlace)
            .where(ClassvisionPlace.school_id == school_id)
            .where(ClassvisionPlace.camera_key == camera_key)
            .where(ClassvisionPlace.class_key == class_key)
            .order_by(ClassvisionPlace.ordinal)
        )
    )


def match_place(centre: tuple[float, float], scale: float,
                candidates: list[ClassvisionPlace], role: str) -> PlaceMatch:
    """Nearest known place of the same role, normalised by the mean of the two widths.

    Normalised rather than absolute because the room is viewed at an angle: a fixed pixel
    gate spans two back-row desks while failing to cover one front-row pupil leaning forward.

    The adult is matched only against adult places and pupils only against pupil places. The
    adult's «место» is where a person stands and sits, not a chair, and letting it compete
    with a desk would attach a lesson of teaching to a child's history.
    """
    same_role = [p for p in candidates if p.role == role]
    if not same_role:
        return PlaceMatch(None, "new", "место в этой комнате встречено впервые.", None)

    ranked = sorted(
        (
            (
                math.dist(centre, (p.anchor_x, p.anchor_y))
                / max((scale + p.anchor_scale) / 2.0, 1e-6),
                p,
            )
            for p in same_role
        ),
        key=lambda pair: pair[0],
    )
    best_distance, best = ranked[0]

    if best_distance > PLACE_MATCH_GATE_SCALES:
        return PlaceMatch(
            None,
            "new",
            f"ближайшее известное место — {best.label_ru} на расстоянии "
            f"{best_distance:.2f} ширины плеч, за порогом {PLACE_MATCH_GATE_SCALES}: "
            "считается новым местом.",
            best_distance,
        )
    if len(ranked) > 1 and ranked[1][0] < best_distance * PLACE_MATCH_MARGIN:
        return PlaceMatch(
            None,
            "ambiguous",
            f"два известных места одинаково близки ({best.label_ru} — {best_distance:.2f}, "
            f"{ranked[1][1].label_ru} — {ranked[1][0]:.2f} ширины плеч). Место оставлено без "
            "привязки: угадывать, чья это история, нельзя.",
            best_distance,
        )
    return PlaceMatch(
        best, "matched", f"{best.label_ru}, расстояние {best_distance:.2f} ширины плеч.",
        best_distance,
    )


def create_place(session: Session, *, school_id: int, camera_key: str, class_key: str,
                 centre: tuple[float, float], scale: float, role: str, run_id: str,
                 first_seen_at: datetime | None, is_demo: bool) -> ClassvisionPlace:
    """A place nobody has seen before. Its anchor is this run's geometry and never moves.

    A running mean would let a slowly-drifting camera carry a place across a desk over a term
    with nothing ever failing. A fixed anchor makes a moved camera fail loudly instead, on a
    particular date, which is a fact a human can act on.
    """
    taken = [p.ordinal for p in known_places(
        session, school_id=school_id, camera_key=camera_key, class_key=class_key)]
    ordinal = (max(taken) + 1) if taken else 1
    place = ClassvisionPlace(
        school_id=school_id,
        is_demo=is_demo,
        camera_key=camera_key,
        class_key=class_key,
        ordinal=ordinal,
        label_ru="место взрослого" if role == "adult" else f"место {ordinal}",
        role=role,
        anchor_x=float(centre[0]),
        anchor_y=float(centre[1]),
        anchor_scale=float(scale),
        first_run_id=run_id,
        first_seen_at=first_seen_at,
    )
    session.add(place)
    session.flush()
    return place


def attested_person(session: Session, *, school_id: int, place_id: int,
                    on: date | None) -> tuple[int | None, str, str]:
    """The one route from a place to a name: `(person_id, identity_method, reason_ru)`.

    `on is None` — a run whose clock could not be read — is never named. A recording that
    cannot be placed on a calendar cannot be placed on either side of a re-seating, and the
    honest answer to "which seating plan applied" is then "we do not know", not "today's".

    The school is named in the statement even though `place_id` already implies it: this query
    is the one that turns a place into a CHILD'S NAME, so it is the last query in this package
    that should be reached through an id somebody else resolved (`tests/test_tenancy_guard.py`).

    **WHAT THIS DOES NOT DO, stated here because it will surprise somebody.** It is consulted
    AT IMPORT, so `ClassvisionPlaceLesson.person_id` records what the signed plan said on the
    day the artefact was imported. A plan signed afterwards does not retro-name the lessons
    already stored, and nothing here silently back-fills them: a name appearing on last term's
    numbers because a form was submitted this morning is a change to a stored observation, and
    that has to be a named act by a person, not a side effect. There is no writer for
    `classvision_attestations` in `qorgan` yet — deliberately, because the school's written
    answer to `docs/questions-for-school.md` §10.1 does not exist, and `decision_ref` NOT NULL
    is what stops one being invented. Until then every place is «место N» and the pages work.
    """
    if on is None:
        return None, "not_established", (
            "дата записи не установлена, поэтому неизвестно, какой план рассадки действовал."
        )
    row = session.scalars(
        select(ClassvisionAttestation)
        .join(ClassvisionPlace, ClassvisionAttestation.place_id == ClassvisionPlace.id)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionAttestation.place_id == place_id)
        .where(ClassvisionAttestation.valid_from <= on)
        .order_by(ClassvisionAttestation.valid_from.desc())
    ).first()
    if row is None or (row.valid_to is not None and row.valid_to < on):
        return None, "not_established", (
            "подписанного плана рассадки на эту дату нет; учёт ведётся по месту. "
            "Распознавание лица на этой камере измерено (0,30 при отрыве 0,10) и имя "
            "создать не может."
        )
    return row.person_id, "seat_map_attested", (
        f"план рассадки от {row.attested_at}, подписал: {row.attested_by}. "
        f"Основание: {row.decision_ref}."
    )


@dataclass(slots=True)
class LessonLink:
    """Which lesson this run joins, and what the store had to decide to say so."""

    lesson: ClassvisionLesson | None
    created: bool
    reanalysis_of: str | None = None
    # The stored lesson this one continues...
    continues: ClassvisionLesson | None = None
    # ...and the stored lesson that continues THIS one, which is the same seam met from the
    # other side. Both exist because a directory back-fill takes whatever order the glob
    # produced, and the chain has to point the same way whichever file arrived first.
    precedes: ClassvisionLesson | None = None
    overlaps: ClassvisionLesson | None = None
    notes: list[str] | None = None


def lesson_for_run(session: Session, *, school_id: int, camera_key: str, class_key: str,
                   video_sha256: str, started_at: datetime | None,
                   duration_seconds: float) -> LessonLink:
    """The lesson an incoming run belongs to. Never invents a second row for one hour.

    Asked in this order, because the answers are not independent: a re-analysis of a
    recording already stored is the same lesson; a file that begins where another ended is a
    DVR seam and is chained; only then is a wall-clock intersection an overlap, and the
    matched neighbour is excluded from that test so that a genuine seam cannot be refused as
    a duplicate. The continuation is looked for in BOTH directions — a directory back-fill
    takes whatever order the glob produced, and asking only «what ended just before this»
    stored the real D14 pair as two lessons when the long file arrived first.
    """
    same_video = session.scalars(
        select(ClassvisionRun)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionLesson.camera_key == camera_key)
        .where(ClassvisionLesson.class_key == class_key)
        .where(ClassvisionRun.video_sha256 == video_sha256)
    ).first()
    if same_video is not None:
        return LessonLink(
            lesson=same_video.lesson,
            created=False,
            reanalysis_of=same_video.run_id,
            notes=[
                f"это повторный разбор той же записи (прежний прогон {same_video.run_id}). "
                "Урок остался один, выбранный прогон НЕ переключён: переключение — "
                "отдельное действие человека."
            ],
        )
    if started_at is None:
        return LessonLink(lesson=None, created=True, notes=[])

    ends_at = started_at + timedelta(seconds=duration_seconds)
    stored = _dated_lessons(session, school_id=school_id, camera_key=camera_key,
                            class_key=class_key)
    seam, direction = _seam(stored, started_at, ends_at)
    overlap = _overlap(stored, started_at, ends_at, exclude=seam)
    return LessonLink(
        lesson=None, created=True,
        continues=seam if direction == "after" else None,
        precedes=seam if direction == "before" else None,
        overlaps=overlap, notes=[])


def _dated_lessons(session: Session, *, school_id: int, camera_key: str,
                   class_key: str) -> list[ClassvisionLesson]:
    return list(
        session.scalars(
            select(ClassvisionLesson)
            .where(ClassvisionLesson.school_id == school_id)
            .where(ClassvisionLesson.camera_key == camera_key)
            .where(ClassvisionLesson.class_key == class_key)
            .where(ClassvisionLesson.started_at.is_not(None))
            .order_by(ClassvisionLesson.started_at)
        )
    )


def _seam(stored: list[ClassvisionLesson], starts: datetime,
          ends: datetime) -> tuple[ClassvisionLesson | None, str]:
    """A stored lesson on either side of this one, and WHICH side it is on.

    `"after"` means the stored lesson ended where this one begins, so this one continues it;
    `"before"` means the stored lesson begins where this one ends, so it continues THIS one and
    the caller must re-point the stored row. Returning the side rather than just the neighbour
    is what stops the chain pointing backwards when the long file is imported first — a chain
    in which an 11-minute tail claims to precede the hour it followed reads as two lessons to
    anything that walks it.
    """
    for lesson in stored:
        if lesson.ended_at is None:
            continue
        after = (starts - lesson.ended_at).total_seconds()
        if -CONTINUATION_TOLERANCE_SECONDS <= after <= CONTINUATION_GAP_SECONDS:
            return lesson, "after"
        before = (lesson.started_at - ends).total_seconds()
        if -CONTINUATION_TOLERANCE_SECONDS <= before <= CONTINUATION_GAP_SECONDS:
            return lesson, "before"
    return None, ""


def _overlap(stored: list[ClassvisionLesson], starts: datetime, ends: datetime,
             exclude: ClassvisionLesson | None) -> ClassvisionLesson | None:
    for lesson in stored:
        if lesson is exclude or lesson.ended_at is None:
            continue
        if lesson.started_at < ends and starts < lesson.ended_at:
            return lesson
    return None


def iso_week_of(moment: datetime | None, zone: Any) -> tuple[date | None, int | None, int | None]:
    """The school's local day and its ISO week, or three NULLs. Never a guessed Monday."""
    if moment is None:
        return None, None, None
    local = moment.astimezone(zone).date()
    iso = local.isocalendar()
    return local, iso.year, iso.week

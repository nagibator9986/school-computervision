"""The video-classification view: the stills, the rectangles over them, and what was measured.

This is the client's «вот эти вот квадратики, метрики и т.д.» — and it is the page where the
system is easiest to misread, so almost all of this module is about what the rectangles ARE.

**The boxes are drawn in the browser, not into the JPEG.** A rectangle burned into the image
cannot be labelled, cannot be linked to the place's own page, and cannot be told apart from a
detection at a glance. Here each box is HTML positioned in PERCENT of the picture, so the
overlay stays on the pupil at any width, the label is selectable text, and `box_source` decides
what the label is allowed to claim.

**`box_source` is the honest half, and it is printed under every frame.** With
`place_geometry` the rectangle is the region a count was accumulated in — the place's own
centre and shoulder width measured over the whole lesson — and NOT the detector's output for
this frame, which the artefact does not carry. That distinction is invisible to a reader
looking at a box around a child's head, so it travels with the picture as a sentence rather
than as a column somebody may forget to render.

**A state is shown only where the second was actually observed.** The timeline carries a
`measured` flag per run of state; where it is false the box says «не наблюдалось» and carries
no state at all. A frame on which the follower had lost a place would otherwise be labelled
«сидит на месте», which is the same sentence a measurement would produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.classvision.cabinet import UNMEASURED_RU, local_clock, lesson_and_run, reading_of
from qorgan.classvision.frames import STATE_RU
from qorgan.db.models.classvision import ClassvisionFrame, ClassvisionLesson, ClassvisionPlace

# Which states get their own colour on the overlay. CHOSEN to separate the three kinds of thing
# a reader is looking for -- a visible action (hand, standing, board), a posture worth a second
# look (head down, turned), and "we did not see this place" -- and NOT to rank them. Nothing
# here is good or bad; `not_observed` is grey because it is an absence, not a warning.
STATE_KIND = {
    "hand_raised": "action",
    "stood_up": "action",
    "at_board": "action",
    "away_from_place": "moved",
    "head_down": "posture",
    "turned_away": "posture",
    "seated": "seated",
    "unknown": "unknown",
    "not_observed": "unknown",
}


@dataclass(frozen=True, slots=True)
class PlaceLabel:
    """What a box may be CALLED, taken from the place and not from the seat.

    A stored box carries `seat_label` turned into «место 2», and a seat number is not a place
    number: `room/seats.py` numbers discovered clusters in reading order, so camera 01's «место 1»
    is `seat_2`. Rendering the seat number left the same child called «место 1» on the lesson
    page and «место 2» on this one — the exact confusion the place table exists to prevent. So the
    label comes from `classvision_places`, and a box with no place keeps its seat label with a
    marker saying that is what it is.
    """

    label_ru: str
    short_ru: str


@dataclass(frozen=True, slots=True)
class Box:
    """One rectangle over one still, in PERCENT of the picture, with what it may claim.

    Percent rather than pixels because the same overlay has to sit on a 2560-px frame scaled
    into whatever width the page got. `clamped` records that the rectangle ran off the edge of
    the image and was cut to fit: a place at the edge of the frame is exactly the place whose
    coverage is worst, so the fact is shown rather than quietly corrected.
    """

    label_ru: str
    short_ru: str
    role: str
    place_id: int | None
    left: float
    top: float
    width: float
    height: float
    state_ru: str
    kind: str
    measured: bool
    coverage_percent: float | None
    index_text: str
    clamped: bool


@dataclass(frozen=True, slots=True)
class Frame:
    """One still: where it is in the recording, its boxes, and the sentence it must carry."""

    id: int
    video_seconds: float
    minute_ru: str
    wall_clock_ru: str
    image_url: str
    box_source: str
    caveat_ru: str
    boxes: tuple[Box, ...]
    happening_ru: str


def _box(raw: dict[str, Any], *, width: int, height: int,
         labels: dict[int, PlaceLabel]) -> Box:
    """One stored box, scaled into percent, clamped to the picture, named from its place."""
    left = float(raw.get("x") or 0.0) / max(width, 1) * 100
    top = float(raw.get("y") or 0.0) / max(height, 1) * 100
    box_width = float(raw.get("width") or 0.0) / max(width, 1) * 100
    box_height = float(raw.get("height") or 0.0) / max(height, 1) * 100
    clamped = left < 0 or top < 0 or left + box_width > 100 or top + box_height > 100
    left, top = max(left, 0.0), max(top, 0.0)
    coverage = raw.get("coverage")
    index = raw.get("activity_index")
    state = str(raw.get("state") or "not_observed")
    named = labels.get(raw.get("place_id")) or PlaceLabel(
        f"{raw.get('label_ru') or 'место'} (без привязки к месту комнаты)", "?")
    return Box(
        label_ru=named.label_ru, short_ru=named.short_ru, role=str(raw.get("role") or "pupil"),
        place_id=raw.get("place_id"),
        left=round(left, 2), top=round(top, 2),
        width=round(min(box_width, 100 - left), 2), height=round(min(box_height, 100 - top), 2),
        state_ru=str(raw.get("state_ru") or STATE_RU.get(state, state)),
        kind=STATE_KIND.get(state, "unknown"), measured=bool(raw.get("measured")),
        coverage_percent=None if coverage is None else round(float(coverage) * 100, 1),
        index_text=UNMEASURED_RU if index is None else f"{float(index):.1f}",
        clamped=clamped,
    )


def _happening(boxes: tuple[Box, ...]) -> str:
    """One line naming what is visible on this still. Counts of BOXES, never of children.

    Assembled here rather than in the template because it is the only sentence on the page that
    aggregates, and «мест в этом состоянии» has to be its literal wording: a place is not a
    child, and «трое отвернулись» would be a claim about people.
    """
    seen: dict[str, int] = {}
    for box in boxes:
        if box.measured:
            seen[box.state_ru] = seen.get(box.state_ru, 0) + 1
    unmeasured = sum(1 for box in boxes if not box.measured)
    parts = [f"{state} — мест {count}" for state, count in sorted(seen.items())]
    if unmeasured:
        parts.append(f"не наблюдалось — мест {unmeasured}")
    return "; ".join(parts) if parts else "на этом кадре ни одно место не наблюдалось"


def _minute(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def frames_view(session: Session, *, school_id: int, lesson_id: int) -> dict[str, Any] | None:
    """The frames of one lesson, or the stated reason there are none.

    An empty list is a normal outcome and gets a sentence rather than an empty page: the
    demonstration term has no recording, so it can have no stills, and «кадров нет» must not
    look like a page that failed to load.
    """
    found = lesson_and_run(session, school_id=school_id, lesson_id=lesson_id)
    if found is None:
        return None
    lesson, run = found
    stored = list(session.scalars(
        select(ClassvisionFrame)
        .join(ClassvisionLesson, ClassvisionFrame.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionFrame.run_id == run.id)
        .order_by(ClassvisionFrame.video_seconds)
    ))
    labels = _labels(session, school_id=school_id, lesson=lesson)
    frames = tuple(_frame(row, lesson=lesson, labels=labels) for row in stored)
    return {
        "lesson": lesson, "run": run, "frames": frames,
        "reading": reading_of(session, school_id=school_id, run=run),
        "empty_reason_ru": _empty_reason(lesson) if not frames else "",
        "legend": tuple((label, STATE_KIND.get(state, "unknown"))
                        for state, label in STATE_RU.items()),
    }


def _labels(session: Session, *, school_id: int,
            lesson: ClassvisionLesson) -> dict[int, PlaceLabel]:
    """Every place of this room and class, by id, with its stable label and a short form."""
    found = session.execute(
        select(ClassvisionPlace.id, ClassvisionPlace.label_ru, ClassvisionPlace.ordinal,
               ClassvisionPlace.role)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.camera_key == lesson.camera_key)
        .where(ClassvisionPlace.class_key == lesson.class_key)
    ).all()
    return {place_id: PlaceLabel(label, "взр." if role == "adult" else str(ordinal))
            for place_id, label, ordinal, role in found}


def _frame(row: ClassvisionFrame, *, lesson: ClassvisionLesson,
           labels: dict[int, PlaceLabel]) -> Frame:
    boxes = tuple(_box(raw, width=row.image_width, height=row.image_height, labels=labels)
                  for raw in (row.boxes or []))
    return Frame(
        id=row.id, video_seconds=row.video_seconds, minute_ru=_minute(row.video_seconds),
        # In the zone the recording was read in, because the camera's own timestamp is burned
        # into the picture this line sits beside -- see `cabinet.local_clock`.
        wall_clock_ru=local_clock(row.wall_clock, lesson.timezone, fmt="%H:%M:%S"),
        # Served by this cabinet's own handler, not by /media: that handler classifies the media
        # tree by top-level directory and refuses anything it has not been told about, and
        # `classvision/` is not in its map. Deny-by-default did its job; the route that knows
        # what these pictures are is the one that must serve them.
        image_url=f"/psychologist/lessons/{lesson.id}/frames/{row.id}/image",
        box_source=row.box_source, caveat_ru=row.caveat_ru, boxes=boxes,
        happening_ru=_happening(boxes),
    )


def _empty_reason(lesson: ClassvisionLesson) -> str:
    if lesson.is_demo:
        return (
            # Neutral wording, on the operator's instruction that the demonstration rows
            # carry no visible marker. The sentence still has to be TRUE: for these lessons
            # there is no recording, so there is nothing to cut and nothing to draw on. It
            # says that without naming why, and it does not claim the stills are pending.
            "Кадров для этого урока нет: разметка кадров делается по видеозаписи, а записи "
            "к этому уроку не приложено."
        )
    return (
        "Кадры для этого урока ещё не нарезаны. Их режет отдельная команда рядом с записью "
        "(`qorgan classvision frames --run … --video … --at …`): веб-процесс намеренно не "
        "умеет открывать видео и не должен уметь."
    )


def frame_image_path(session: Session, *, school_id: int, lesson_id: int,
                     frame_id: int) -> str | None:
    """The stored RELATIVE path of one frame, or None. The caller confines it to MEDIA_ROOT.

    Looked up through the lesson and the school rather than by `frame_id` alone: the id is a
    number in a URL, and «покажи картинку номер 7» must not be a way out of one school's data.
    """
    return session.scalars(
        select(ClassvisionFrame.image_path)
        .join(ClassvisionLesson, ClassvisionFrame.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionFrame.lesson_id == lesson_id)
        .where(ClassvisionFrame.id == frame_id)
    ).first()

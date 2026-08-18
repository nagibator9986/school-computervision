"""The video-classification view: stills on disk, and the boxes and states over them.

The client asked to see «вот эти вот квадратики, метрики и т.д.» — the detector's view of the
room — and the honest way to give a web process that, with no torch and no cv2 anywhere near
it, is a JPEG plus a row of coordinates. Two routes in, and the difference between them is a
column rather than a footnote:

* `from_manifest` — a classvision-side render supplies the boxes it actually detected.
  `box_source = "detector"`.
* `from_video` — ffmpeg (a subprocess, not a Python dependency) cuts a still out of the
  recording, and the rectangles are computed HERE from each place's own geometry.
  `box_source = "place_geometry"`, and `caveat_ru` says so on every row.

**Why the second route is allowed to exist.** The artefact carries no per-frame detections —
it carries per-place geometry and a state timeline — so a rectangle drawn from it is the
REGION A COUNT WAS ACCUMULATED IN, not the detector's output for that frame. That is a
genuinely useful thing to show a psychologist (it answers «где это место и что оно делало в
эту минуту»), and it is a genuinely different claim from «вот что нашла модель». A rectangle
drawn on a child reads as a detection whatever a column says, so the sentence travels with
the row and the page is expected to print it.

**The state comes from the place's own timeline and keeps its `measured` flag**, so «не
наблюдалось» never renders as «сидит». A box whose second falls in an unmeasured run is
labelled «не наблюдалось» and carries no state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models.classvision import (ClassvisionFrame, ClassvisionLesson,
                                          ClassvisionPlace, ClassvisionPlaceLesson,
                                          ClassvisionRun, ClassvisionTeacherLesson)
from qorgan.db.models.school import sole_school_id
from qorgan.settings import get_settings, resolve

# A box drawn around a place, in shoulder widths. CHOSEN as a DRAWING CONVENTION and named as
# one: the analyser measures a centre and a shoulder width, so a plausible upper body is half
# a shoulder either side and about two above/below. Nothing downstream may read these as a
# measured bounding box — that is what `box_source` is for.
BOX_HALF_WIDTH_SCALES = 0.95
BOX_ABOVE_SCALES = 1.35
BOX_BELOW_SCALES = 0.95

GEOMETRY_CAVEAT_RU = (
    "Рамки построены по геометрии МЕСТА (центр и ширина плеч, измеренные на этом уроке), а не "
    "по выходу детектора на этом кадре: артефакт не хранит покадровые рамки. Это область, в "
    "которой копились счётчики места, и подпись — состояние места в эту секунду по его "
    "собственной временной шкале."
)
DETECTOR_CAVEAT_RU = (
    "Рамки и подписи получены из разбора записи (classvision) и относятся именно к этому кадру."
)

MANIFEST_SCHEMA = "classvision-frames/1"


class FramesRefused(Exception):
    """Nothing was written, and this says which precondition was missing."""


@dataclass(slots=True)
class FramesResult:
    run_id: str
    written: int = 0
    skipped: int = 0
    box_source: str = ""
    directory: str = ""
    notes: list[str] = field(default_factory=list)

    def report_ru(self) -> list[str]:
        lines = [
            f"{self.run_id}: кадров записано {self.written}, пропущено (уже были) "
            f"{self.skipped}; рамки — {self.box_source}; каталог {self.directory or '—'}"
        ]
        return lines + [f"  примечание: {note}" for note in self.notes]


def _run(session: Any, run_id: str | None, school: int) -> ClassvisionRun:
    if not run_id:
        raise FramesRefused("не указан --run: кадры принадлежат конкретному прогону разбора")
    run = session.scalars(
        select(ClassvisionRun)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionRun.run_id == run_id)
    ).first()
    if run is None:
        raise FramesRefused(
            f"прогон {run_id} в базе не найден. Сначала `qorgan classvision import`: кадр без "
            "разбора — картинка без провенанса."
        )
    return run


def from_video(*, run_id: str | None, video: Path | None, seconds: list[float],
               school_id: int | None = None) -> FramesResult:
    """Cut stills with ffmpeg and label them from the places' own geometry and timelines."""
    if video is None or not Path(video).exists():
        raise FramesRefused(f"файл записи не найден: {video}")
    if not seconds:
        raise FramesRefused("не указано --at: нужны секунды записи, например --at 300,900,1800")
    if shutil.which("ffmpeg") is None:
        raise FramesRefused(
            "ffmpeg не найден. Кадры режет он, а не Python: в этот процесс не должны попасть "
            "ни torch, ни cv2 (INTEGRATION.md §1)."
        )
    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        run = _run(session, run_id, school)
        rows = _place_rows(session, run, school)
        result = FramesResult(run_id=run.run_id, box_source="place_geometry")
        target = resolve(get_settings().media_root) / "classvision" / run.run_id
        target.mkdir(parents=True, exist_ok=True)
        result.directory = str(target)
        for second in seconds:
            _one_frame(session, run, rows, video=Path(video), second=second, target=target,
                       school=school, result=result)
        return result


def _one_frame(session: Any, run: ClassvisionRun, rows: list[ClassvisionPlaceLesson], *,
               video: Path, second: float, target: Path, school: int,
               result: FramesResult) -> None:
    local, note = _local_seconds(run, second, video)
    if note:
        result.notes.append(note)
        return
    if _already(session, run, second, school):
        result.skipped += 1
        return
    # Named by the SESSION second, not by the second inside the part. An assembled lesson has
    # several files, and two parts can perfectly well both contain a second 382 -- under a
    # local-second name the later extraction would overwrite the earlier one's image while both
    # rows stayed in the database pointing at it.
    name = f"sec_{int(second):06d}.jpg"
    image = target / name
    completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, paths from the caller
        ["ffmpeg", "-nostdin", "-y", "-ss", f"{local:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(image)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not image.exists():
        result.notes.append(f"секунда {second:.0f}: ffmpeg не смог достать кадр")
        return
    session.add(ClassvisionFrame(
        lesson_id=run.lesson_id, run_id=run.id, video_seconds=float(second),
        wall_clock=None if run.started_at is None else run.started_at + timedelta(seconds=second),
        image_path=f"classvision/{run.run_id}/{name}",
        image_width=int(run.provenance.get("width") or 0),
        image_height=int(run.provenance.get("height") or 0),
        source_video=str(video),
        box_source="place_geometry",
        boxes=[_box(row, second) for row in rows] + _adult_box(session, run, school),
        caveat_ru=GEOMETRY_CAVEAT_RU,
    ))
    result.written += 1


def _local_seconds(run: ClassvisionRun, second: float, video: Path) -> tuple[float, str | None]:
    """Map a SESSION second to a second inside the file the caller supplied.

    An assembled artefact's timeline spans several files (`provenance.session.parts`), so a
    second past the first part's end is not in this file at all. Naming the part is the useful
    refusal; silently cutting the wrong minute is the one to avoid.
    """
    parts = (run.session or {}).get("parts") or []
    if not parts:
        return second, None
    offset = 0.0
    for index, part in enumerate(parts):
        length = float(part.get("duration_seconds") or 0.0)
        if second < offset + length:
            if index == 0 or Path(str(part.get("path") or "")).name == video.name:
                return second - offset, None
            return second, (
                f"секунда {second:.0f} лежит в части {index + 1} "
                f"({Path(str(part.get('path'))).name}), а передан другой файл — кадр не взят"
            )
        offset += length
    return second, f"секунда {second:.0f} за пределами склеенной записи"


def _already(session: Any, run: ClassvisionRun, second: float, school: int) -> bool:
    return session.scalars(
        select(ClassvisionFrame)
        .join(ClassvisionLesson, ClassvisionFrame.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionFrame.run_id == run.id)
        .where(ClassvisionFrame.video_seconds == float(second))
    ).first() is not None


def _place_rows(session: Any, run: ClassvisionRun,
                school: int) -> list[ClassvisionPlaceLesson]:
    return list(session.scalars(
        select(ClassvisionPlaceLesson)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionPlaceLesson.run_id == run.id)
        .order_by(ClassvisionPlaceLesson.seat_id)
    ))


def _box(row: ClassvisionPlaceLesson, second: float) -> dict[str, Any]:
    """One rectangle, its label, and — always — how well this place was seen.

    `coverage` rides in the box because this view is the one place a reader meets a count
    without a table around it, and a count without its coverage is forbidden everywhere else.
    """
    state, measured = _state_at(row.timeline, second)
    half = row.scale_px * BOX_HALF_WIDTH_SCALES
    return {
        "seat_id": row.seat_id,
        "label_ru": row.seat_label.replace("seat_", "место "),
        "role": row.role,
        "place_id": row.place_id,
        "x": round(row.centre_x - half, 1),
        "y": round(row.centre_y - row.scale_px * BOX_ABOVE_SCALES, 1),
        "width": round(half * 2, 1),
        "height": round(row.scale_px * (BOX_ABOVE_SCALES + BOX_BELOW_SCALES), 1),
        "state": state if measured else None,
        "state_ru": STATE_RU.get(state, state) if measured else "не наблюдалось",
        "measured": measured,
        "coverage": round(row.coverage, 3),
        "activity_index": row.activity_index,
    }


def _adult_box(session: Any, run: ClassvisionRun, school: int) -> list[dict[str, Any]]:
    """The adult's place, drawn but never given a state.

    His `place` is a zone he starts from, and the whole point of the follower is that he walks
    away from it — so a rectangle here says «вот где его место», not «вот где он сейчас». It is
    drawn anyway, because a frame in which the adult is the only person unlabelled reads as
    though the system did not see him; `measured: false` and the sentence say which it is.
    """
    teacher = session.scalars(
        select(ClassvisionTeacherLesson)
        .join(ClassvisionLesson, ClassvisionTeacherLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionTeacherLesson.run_id == run.id)
    ).first()
    if teacher is None or teacher.place_id is None:
        return []
    place = session.scalars(
        select(ClassvisionPlace)
        .where(ClassvisionPlace.school_id == school)
        .where(ClassvisionPlace.id == teacher.place_id)
    ).first()
    if place is None:
        return []
    half = place.anchor_scale * BOX_HALF_WIDTH_SCALES
    return [{
        "seat_id": teacher.seat_id,
        "label_ru": "место взрослого",
        "role": "adult",
        "place_id": place.id,
        "x": round(place.anchor_x - half, 1),
        "y": round(place.anchor_y - place.anchor_scale * BOX_ABOVE_SCALES, 1),
        "width": round(half * 2, 1),
        "height": round(place.anchor_scale * (BOX_ABOVE_SCALES + BOX_BELOW_SCALES), 1),
        "state": None,
        "state_ru": "место взрослого: положение в этот кадр не привязано",
        "measured": False,
        "coverage": teacher.pose_coverage,
        "activity_index": None,
    }]


def _state_at(timeline: list[dict[str, Any]], second: float) -> tuple[str, bool]:
    for run_of_state in timeline or []:
        if float(run_of_state.get("start_s", 0.0)) <= second < float(
                run_of_state.get("end_s", 0.0)):
            return str(run_of_state.get("state") or "unknown"), bool(run_of_state.get("measured"))
    return "not_observed", False


# The analyser's own vocabulary, translated once. Every one of these is a POSTURE or a
# POSITION: nothing here says anything about attention, interest or engagement, and no word
# below may be replaced by one that does.
STATE_RU = {
    "seated": "сидит на месте",
    "hand_raised": "поднята рука",
    "stood_up": "встал",
    "away_from_place": "вне своего места",
    "at_board": "у доски",
    "head_down": "голова опущена к парте",
    "turned_away": "голова повёрнута в сторону",
    "unknown": "поза не разобрана",
    "not_observed": "не наблюдалось",
}


def from_manifest(path: Path, *, school_id: int | None = None) -> FramesResult:
    """Store frames a classvision-side render produced, boxes included.

    The manifest names images RELATIVE to MEDIA_ROOT and this refuses one that is not there:
    a row pointing at a missing file is a broken picture on a page about children, and the
    person who can fix it is the person running this command.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(document.get("schema")) != MANIFEST_SCHEMA:
        raise FramesRefused(f"манифест не {MANIFEST_SCHEMA}: {document.get('schema')!r}")
    media = resolve(get_settings().media_root)
    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        run = _run(session, document.get("run_id"), school)
        result = FramesResult(run_id=run.run_id, box_source="detector")
        for frame in document.get("frames") or []:
            relative = str(frame["image"])
            if not (media / relative).exists():
                raise FramesRefused(f"кадра нет на диске: MEDIA_ROOT/{relative}")
            second = float(frame["video_seconds"])
            if _already(session, run, second, school):
                result.skipped += 1
                continue
            session.add(ClassvisionFrame(
                lesson_id=run.lesson_id, run_id=run.id, video_seconds=second,
                wall_clock=None if run.started_at is None
                else run.started_at + timedelta(seconds=second),
                image_path=relative,
                image_width=int(frame.get("width") or run.provenance.get("width") or 0),
                image_height=int(frame.get("height") or run.provenance.get("height") or 0),
                source_video=frame.get("source_video"),
                box_source="detector",
                boxes=frame.get("boxes") or [],
                caveat_ru=str(frame.get("caveat_ru") or DETECTOR_CAVEAT_RU),
            ))
            result.written += 1
        result.directory = str(media)
        return result

"""The lesson list and one lesson, assembled for a page. Every count arrives with its coverage.

**Why the assembling is here and not in `web/routes/classvision.py`.** The same reason
`qorgan.psychologist.cabinet` sits beside `web/routes/psychologist.py`: a route's job is the
capability gate and a template name, and every rule about what a number MEANS has to be
readable — and testable — without an HTTP client. A template that decides whether a zero is a
zero is a rule nothing can check.

**Three rules live here, each in exactly one function, because each was a defect first:**

1. **No bare count.** `Cell` cannot be built without saying whether it was measured, and
   `PlaceRow` carries `coverage_percent` beside every cell. `classvision/DESIGN.md` forbids a
   count printed without the share of the lesson it was counted in: on these recordings the
   observed coverages run 61.5 – 99.6 %, so «4 отхода» means two different things at the two
   ends of that range.

2. **Zero and «не измерялось» never look alike.** `board_visits` on a camera with no board
   polygon is not zero — the state cannot arise in a single frame — and `stands`/`away` on a
   place that never settled are not zero either, because with no baseline posture there is
   nothing to compare against. Both come back as `Cell(measured=False)`, and there is one
   more case between them: a count that is zero because every episode was shorter than the
   threshold. That zero is true and still reads as «этого не было», so `brief_only` marks it.

3. **A demonstration is never added to a measurement.** `is_demo` rides on every row from the
   database to the page. The operator asked for it to carry no visible marker — their call
   about their own presentation — so nothing here renders it; the column stays so the
   synthetic term can be deleted in one statement rather than picked apart by date.

**The lesson LIST lives in `lessons_index.py`**, split off when this file crossed the repo's
500-line limit. The four names at the bottom of this module are what it imports.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models.classvision import (ClassvisionFrame, ClassvisionLesson, ClassvisionPlace,
                                          ClassvisionPlaceLesson, ClassvisionReading,
                                          ClassvisionRun, ClassvisionTeacherLesson)
from qorgan.db.models.person import Person

UNMEASURED_RU = "не измерялось"

# The sentence a `brief_only` zero needs. Written once: the reference cabinet's own measurement
# is that a zero beside a state whose short runs were discarded reads as «такого не было», and
# it is the most common misreading of this table.
BRIEF_ONLY_RU = (
    "ноль означает «длительных эпизодов не набралось»: короткие включения этого состояния "
    "были, но они короче порога и в счётчик не попали."
)

DEMO_BANNER_RU = (
    "ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ. Этой записи не существует, разбора не было: счётчики "
    "синтезированы по распределениям реальных записей, чтобы показать, как выглядит семестр. "
    "Ни одно число на этой странице не является измерением, и складывать эти уроки с "
    "реальными нельзя."
)


@dataclass(frozen=True, slots=True)
class Cell:
    """One quantity, or the stated absence of one. There is no third state.

    `measured=False` means the page prints «не измерялось» and never a digit. `brief_only`
    means the digit is a true zero that needs a sentence beside it.
    """

    text: str
    measured: bool
    brief_only: bool = False
    note_ru: str = ""


@dataclass(frozen=True, slots=True)
class CountColumn:
    """A counter, its Russian column head, and the analyser state its short runs land in."""

    key: str
    label_ru: str
    state: str


# The six counters, in the order the reference cabinet prints them. Every one of them is a
# POSTURE or a POSITION: nothing here is about attention, interest or engagement, and no label
# below may be replaced by one that is.
COUNT_COLUMNS: tuple[CountColumn, ...] = (
    CountColumn("hand_raises", "поднимал руку", "hand_raised"),
    CountColumn("board_visits", "выходил к доске", "at_board"),
    CountColumn("stands", "вставал на месте", "stood_up"),
    CountColumn("away_episodes", "уходил с места", "away_from_place"),
    CountColumn("head_down_episodes", "клал голову на парту", "head_down"),
    CountColumn("turned_away_episodes", "отворачивался назад", "turned_away"),
)


def board_is_measurable(run: ClassvisionRun) -> bool:
    """Is «у доски» a measurement on this camera at all?

    Decided from the room layout, not from the count: with no board polygon the state cannot
    arise in a single frame, so every board figure is an absence of measurement and zero would
    be a claim nobody made.
    """
    return (run.room_layout or {}).get("board_zone") is not None


def count_cell(row: ClassvisionPlaceLesson, column: CountColumn, *,
               board_measurable: bool) -> Cell:
    """One counter of one place in one lesson, with the reason when it is not a number."""
    if column.key == "board_visits" and not board_measurable:
        return Cell(UNMEASURED_RU, False, note_ru=(
            "зона доски не задана для этой камеры, поэтому состояние «у доски» не может "
            "возникнуть ни в одном кадре; такие выходы попали в «уходил с места»."))
    if column.key in ("stands", "away_episodes") and not row.settled:
        return Cell(UNMEASURED_RU, False, note_ru=(
            "место не установило базовую позу, а «встал» и «ушёл» измеряются только "
            "относительно неё: " + (row.settle_refusal or "базовая поза не набрана.")))
    value = int(getattr(row, column.key) or 0)
    if value == 0 and _discarded(row, column.state):
        return Cell("0", True, brief_only=True, note_ru=BRIEF_ONLY_RU)
    if column.key == "hand_raises" and row.hand_unmeasurable_observations:
        return Cell(str(value), True, note_ru=(
            "нижняя граница: в части наблюдений кисти не было видно, и поднятая рука в них "
            "не могла быть ни подтверждена, ни опровергнута."))
    return Cell(str(value), True)


def _discarded(row: ClassvisionPlaceLesson, state: str) -> bool:
    """Were there short runs of this state that the episode threshold threw away?

    A demo row has no `discarded_short_runs` at all, and the safe reading of a missing key is
    «нет данных о коротких включениях» — i.e. do not add the caveat, because inventing one
    would be a claim about a recording that does not exist.
    """
    discarded = (row.ledger or {}).get("discarded_short_runs") or {}
    return int(discarded.get(state, 0) or 0) > 0


@dataclass(frozen=True, slots=True)
class PlaceRow:
    """One place in one lesson: the identity question, the coverage, the index and the cells."""

    place_lesson_id: int
    place_id: int | None
    label_ru: str
    # The child's name, but ONLY where a signed seating plan put it on this observation. Empty
    # otherwise, and the template prints the place instead. It is a separate field from
    # `label_ru` on purpose: a name and a chair are different claims, and a page that stores
    # them in one string can no longer show which of the two it is holding.
    person_name: str
    role: str
    coverage_percent: float
    observed_minutes: float
    settled: bool
    place_match: str
    place_match_reason: str
    identity_method: str
    identity_reason: str
    index_text: str
    index_measured: bool
    index_reason: str
    cells: tuple[Cell, ...]


def _index_of(row: ClassvisionPlaceLesson) -> tuple[str, bool, str]:
    """The activity index, or the reason there is none. NULL is never printed as a number.

    The adult is a separate refusal rather than a missing value: a pupil's counters do not
    apply to him at all (a raised hand at the board is pointing, not participation), so the
    page says which of the two it is.
    """
    if row.role == "adult":
        return ("не считается", False,
                "счётчики ученика к взрослому не применяются: поднятая рука взрослого — это "
                "указание на доску, а не участие в уроке.")
    if row.activity_index is None:
        return UNMEASURED_RU, False, row.activity_reason or (
            "наблюдений не хватило, чтобы показатель имел смысл.")
    return f"{row.activity_index:.1f}", True, ""


def place_rows(session: Session, *, school_id: int, run: ClassvisionRun) -> list[PlaceRow]:
    """Every place of one run, in room order, pupils first and the adult last."""
    board = board_is_measurable(run)
    rows = session.execute(
        select(ClassvisionPlaceLesson, ClassvisionPlace, Person)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .outerjoin(ClassvisionPlace, ClassvisionPlaceLesson.place_id == ClassvisionPlace.id)
        # The school goes in the ON clause, not in a WHERE: as a WHERE it would also delete
        # every row that has no person at all, which is most of them.
        .outerjoin(Person, (ClassvisionPlaceLesson.person_id == Person.id)
                   & (Person.school_id == school_id))
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.run_id == run.id)
        .order_by(ClassvisionPlaceLesson.role.desc(), ClassvisionPlaceLesson.seat_id)
    ).all()
    out = []
    for row, place, person in rows:
        text, measured, reason = _index_of(row)
        out.append(PlaceRow(
            place_lesson_id=row.id, place_id=row.place_id,
            # The stable place's own label when there is one, and this run's seat label when
            # there is not: an unmatched seat must not borrow a place's name, because the name
            # is what a term of history hangs on.
            label_ru=place.label_ru if place is not None else row.seat_label.replace(
                "seat_", "место (без привязки) "),
            person_name=(person.full_name or person.external_id) if person is not None else "",
            role=row.role, coverage_percent=round(row.coverage * 100, 1),
            observed_minutes=round(row.observed_seconds / 60.0, 1),
            settled=bool(row.settled), place_match=row.place_match,
            place_match_reason=row.place_match_reason or "", identity_method=row.identity_method,
            identity_reason=row.identity_reason or "", index_text=text, index_measured=measured,
            index_reason=reason,
            cells=tuple(count_cell(row, column, board_measurable=board)
                        for column in COUNT_COLUMNS),
        ))
    return out


def room_totals(rows: list[PlaceRow]) -> tuple[Cell, ...]:
    """The whole room's counters. A sum is only printed where every place was measured.

    One unmeasured place makes the column unmeasured for the room, not smaller: a total that
    silently omits a place is a smaller number that looks complete, which is worse than a
    stated absence.
    """
    pupils = [row for row in rows if row.role == "pupil"]
    out = []
    for index in range(len(COUNT_COLUMNS)):
        cells = [row.cells[index] for row in pupils]
        if not cells or not all(cell.measured for cell in cells):
            out.append(Cell(UNMEASURED_RU, False, note_ru=(
                "хотя бы на одном месте эта величина не измерялась, поэтому суммы по комнате "
                "нет: сумма без одного места — это меньшее число, выглядящее полным.")))
            continue
        out.append(Cell(str(sum(int(cell.text) for cell in cells)), True,
                        brief_only=any(cell.brief_only for cell in cells),
                        note_ru=BRIEF_ONLY_RU if any(c.brief_only for c in cells) else ""))
    return tuple(out)


def local_clock(moment: datetime | None, zone_name: str | None, *,
                fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """A stored UTC moment in the zone the recording was READ in, or a stated absence.

    **This is not cosmetic.** The frames page prints a time beside a still that has the camera's
    own timestamp burned into the picture. Rendering the stored UTC gave «04:58:43» under a
    photograph reading «09:58:43», which invites the reader to conclude that the analysis is
    about a different recording. The zone comes from the LESSON, stored once at import: deriving
    it from today's `SCHOOL_TIMEZONE` would silently move a stored lesson across a day boundary
    the first time an installation is reconfigured.
    """
    if moment is None:
        return "время съёмки не установлено"
    if not zone_name:
        return f"{moment.strftime(fmt)} UTC (часовой пояс записи не сохранён)"
    try:
        return moment.astimezone(ZoneInfo(zone_name)).strftime(fmt)
    except (ZoneInfoNotFoundError, ValueError):
        return f"{moment.strftime(fmt)} UTC (пояс {zone_name} неизвестен этой системе)"


def footnotes(*groups: Any) -> tuple[str, ...]:
    """Every distinct sentence the cells on one page need, in first-appearance order.

    Collected for the page rather than attached to each cell as a tooltip: a reason that
    explains why a column says «не измерялось» is the sentence that changes how the whole
    column reads, and a `title=` attribute is invisible on a printout and on a phone.
    """
    seen: list[str] = []
    for group in groups:
        for cell in group:
            note = getattr(cell, "note_ru", "")
            if note and note not in seen:
                seen.append(note)
    return tuple(seen)


def teacher_of(session: Session, *, school_id: int,
               run: ClassvisionRun) -> ClassvisionTeacherLesson | None:
    return session.scalars(
        select(ClassvisionTeacherLesson)
        .join(ClassvisionLesson, ClassvisionTeacherLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionTeacherLesson.run_id == run.id)
    ).first()


def reading_of(session: Session, *, school_id: int, run: ClassvisionRun,
               section: str = "lesson", target_key: str = "") -> ClassvisionReading | None:
    """The stored note for one run, guard verdict included. The caller decides what to show.

    Returned even when the guard failed, deliberately: the page shows the figures and the
    audit block shows why the prose was withheld. Silence would teach nobody anything.
    """
    return session.scalars(
        select(ClassvisionReading)
        .join(ClassvisionRun, ClassvisionReading.run_id == ClassvisionRun.id)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionReading.run_id == run.id)
        .where(ClassvisionReading.section == section)
        .where(ClassvisionReading.target_key == target_key)
    ).first()


def lesson_and_run(session: Session, *, school_id: int,
                   lesson_id: int) -> tuple[ClassvisionLesson, ClassvisionRun] | None:
    """One lesson and the run its pages show, or None when this school has no such lesson."""
    lesson = session.scalars(
        select(ClassvisionLesson)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionLesson.id == lesson_id)
    ).first()
    if lesson is None:
        return None
    run = selected_run(session, school_id=school_id, lesson=lesson)
    return None if run is None else (lesson, run)


def lesson_view(session: Session, *, school_id: int, lesson_id: int) -> dict[str, Any] | None:
    """Everything one lesson page prints, with nothing computed in the template."""
    found = lesson_and_run(session, school_id=school_id, lesson_id=lesson_id)
    if found is None:
        return None
    lesson, run = found
    rows = place_rows(session, school_id=school_id, run=run)
    pupils = [row for row in rows if row.role == "pupil"]
    coverages = [row.coverage_percent for row in pupils]
    totals = room_totals(rows)
    return {
        "lesson": lesson, "run": run, "rows": rows, "pupils": pupils,
        "columns": COUNT_COLUMNS, "totals": totals,
        "footnotes": footnotes(*[row.cells for row in rows], totals),
        "board_measurable": board_is_measurable(run),
        "coverage_median": median_or_none(coverages), "coverage_min": min(coverages, default=None),
        "place_minutes": int(sum(row.observed_minutes for row in pupils)),
        "teacher": teacher_of(session, school_id=school_id, run=run),
        "reading": reading_of(session, school_id=school_id, run=run),
        "frames": frame_counts_by_lesson(session, school_id=school_id).get(lesson.id, 0),
        "unassigned_percent": _unassigned_percent(run),
        "started_ru": local_clock(lesson.started_at, lesson.timezone),
        "ended_ru": local_clock(lesson.ended_at, lesson.timezone, fmt="%H:%M:%S"),
        "demo_banner_ru": DEMO_BANNER_RU if lesson.is_demo else "",
    }


def _unassigned_percent(run: ClassvisionRun) -> float | None:
    """What share of the people seen landed on no known place. A denominator of zero is None."""
    if not run.observations_total:
        return None
    return round(run.observations_unassigned / run.observations_total * 100, 1)


# ---------------------------------------------------------------------------
# Shared with `lessons_index.py`. They live HERE, on the "one lesson" side, because that is
# where the lesson and its chosen run are the subject; the index calls them once per row.
# Public names, since a leading underscore reaching across a module boundary says the split
# was drawn in the wrong place -- and the first version of this split was.
# ---------------------------------------------------------------------------


def median_or_none(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def selected_run(session: Session, *, school_id: int,
                  lesson: ClassvisionLesson) -> ClassvisionRun | None:
    """The run this lesson's pages show. Never «the newest»: moving that pointer is an act."""
    return session.scalars(
        select(ClassvisionRun)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionRun.lesson_id == lesson.id)
        .where(ClassvisionRun.run_id == lesson.selected_run_id)
    ).first()


def frame_counts_by_lesson(session: Session, *, school_id: int) -> dict[int, int]:
    return {lesson_id: count for lesson_id, count in session.execute(
        select(ClassvisionFrame.lesson_id, func.count(ClassvisionFrame.id))
        .join(ClassvisionLesson, ClassvisionFrame.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .group_by(ClassvisionFrame.lesson_id)
    ).all()}

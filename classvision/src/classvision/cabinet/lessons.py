"""One lesson as a psychologist opens it, and the only thing two lessons may be compared on.

--------------------------------------------------------------------------------
**WHY A LESSON PAGE EXISTS AT ALL, WHEN THERE IS ALREADY A PAGE PER PLACE.**

`cabinet/report.py` was built downwards from the client's sentence («первая неделя хорошо,
вторая не очень»), so its pages are a class overview and then one page per child-shaped
place. That is the right shape for a term. It is the wrong shape for the first five minutes
after a recording is analysed, which is when somebody actually opens this: the question then
is «что было на этом уроке и чему в нём можно верить», and answering it from nine per-place
pages means holding nine coverage figures in your head.

So the unit here is the SESSION — one lesson as a human means it, which is sometimes two
recordings (`weekly.py`'s first paragraph). Everything on the page is scoped to that one
hour, and every figure carries the fraction of the hour it was computed from.

--------------------------------------------------------------------------------
**THE COMPARISON THIS FILE MAKES, AND THE ONE IT REFUSES TO MAKE.**

The store this was built against holds exactly two analysed lessons and they are **not the
same class**:

| | `full_lesson.analysis.json` | `d14_session.analysis.json` |
|---|---|---|
| camera | camera01, board BEHIND the lens | D14, board in frame |
| date | 2026-08-07 (2026-W32) | 2026-08-15 (2026-W33) |
| pupil places | **8** | **5** |
| attested seat map | none | none |

Two rooms, two different groups of children, no signed seating plan anywhere. A
week-over-week trend across that pair is not a hard problem, it is a **category error**:
there is no child who appears in both recordings, so there is nothing for «стало хуже» to
be about. `metrics/trend.py` already refuses on two independent grounds (five lessons of
one pupil's own history, and an attested identity), and nothing in this file softens either.

**What genuinely IS comparable across those two recordings is the recordings.** How much of
the hour the room was visible, how many places the analysis found, how many times each
observable event happened anywhere in the room, where the adult was. Those are properties of
a measurement, they are the numbers an operator uses to decide whether a recording is worth
reading at all, and comparing them tells you about the camera and the room rather than about
a child. `COMPARABLE_COLUMNS` therefore carries, per column, both what it is and what it is
NOT — because the one sentence that matters on that table is «эта строка меняется, когда
меняется число детей в кадре, а не когда меняется поведение кого-то одного», and a heading
alone cannot carry it.

--------------------------------------------------------------------------------
**THE ZERO THAT IS NOT A ZERO, MEASURED ON THE TWO ARTEFACTS IN THE STORE.**

`weekly.COUNTER_KEYS` renders «выходил к доске» for every place. On camera01 that counter is
**0 for all eight pupils**, and it is 0 by construction: the board hangs behind the lens,
`configs/camera_01.yaml` records `board_zone: null` as a measured decision, and
`ledger.classify` cannot emit `AT_BOARD` without a polygon. The artefact says so in
`lesson.unmeasured` — «стоял у доски: зона доски не задана для этой камеры» — and the
teacher half of the same artefact already honours it (`presence.board.minutes_of_lesson` is
`null`, with `zone_configured: false` beside it).

The pupil half did not. The class page and the weekly table printed a confident `0` in a
column headed «выходил к доске», directly under a heading claiming to list observable
events, for a camera on which that event **cannot be observed**. That is rule 3 of this
project («ноль и «не увидели» — разные значения») broken in the same way `MEASUREMENTS.md`
§12.6 records for the board zone and §9.5 records for three other fields, one consumer
further along.

`voided_counters()` reads the run's own `unmeasured` list and turns exactly those counters
into a stated refusal. Note what it does NOT do: it does not delete the summed zero from
`weekly.WeekRow.counts`. That sum is a true statement about the ledger — the ledger really
did count no `AT_BOARD` episodes — and rewriting stored arithmetic to carry a display
decision is how a number acquires two meanings. The refusal lives at the rendering boundary,
where the reader is, and `tests/test_cabinet.py` asserts that no generated page prints a
digit in a voided column rather than trusting anyone to remember.

**The mapping is keyed on the analyser's own sentence, and that is a fragility.** If
`pipeline.py` reworded «стоял у доски», this would silently stop firing and the confident
zero would come back. `test_the_unmeasurable_counter_map_matches_what_the_analyser_emits`
fails on that wording change, which is the cheapest available way to make a string join
reviewable.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Any

from classvision.cabinet import store
from classvision.cabinet.weekly import COUNTER_KEYS, _chain_heads, lessons_word

# Which of `weekly.COUNTER_KEYS` a given `lesson.unmeasured` entry makes unmeasurable, keyed
# on the artefact's own `what` string. Written out rather than pattern-matched: a substring
# test on «доск» would also swallow «подтверждение разметки зон», which voids nothing and
# qualifies everything, and the two need opposite treatment.
COUNTER_VOIDED_BY: dict[str, tuple[str, ...]] = {
    "стоял у доски": ("board_visits",),
}

# ...and the same question for the adult's position split. `at_board` is voided by the same
# missing polygon; `at_desk` by the teacher's own zone being unmarked, which on camera01 is
# why `presence.state_share_of_lesson_percent["at_desk"]` reads 0.0 while the seat-ledger
# half of the very same artefact reports the adult at his place for 96.5 % of observations.
# Two true numbers, two different questions, and one of them is a structural zero.
TEACHER_STATE_VOIDED_BY: dict[str, tuple[str, ...]] = {
    "стоял у доски": ("at_board",),
    "«за своим столом» для взрослого": ("at_desk",),
}

# The adult's position states, in the order they are displayed. Order is the order of the
# question «где он был», not of magnitude: sorting by size would make two lessons' columns
# swap places and invite a reader to compare the first column of one with the first of the
# other.
TEACHER_STATES: tuple[tuple[str, str], ...] = (
    ("at_board", "у доски"),
    ("at_desk", "за своим столом"),
    ("among_pupils", "среди учеников"),
    ("moving", "перемещается"),
    ("out_of_frame", "не найден в кадре"),
)

# Every column of the cross-lesson table, with what it is and — the load-bearing half — what
# it is NOT. Kept as data rather than as prose in the template so that the table and its
# definitions cannot drift apart, which is the failure mode a legend always has.
COMPARABLE_COLUMNS: tuple[dict[str, str], ...] = (
    {"key": "window_minutes", "label": "минут разбора",
     "is": "длина окна, которое анализировалось: от появления класса до конца записи.",
     "is_not": "не длина урока по расписанию и не время, которое дети провели в классе."},
    {"key": "pupil_places", "label": "ученических мест",
     "is": "сколько устойчивых мест нашла кластеризация в этой комнате.",
     "is_not": "не численность класса: отсутствовавший ребёнок не создаёт места, а двое за "
               "одной партой могут дать одно. Разное число мест в двух комнатах — это две "
               "разные комнаты, а не «класс стал меньше»."},
    {"key": "coverage_median", "label": "видимость места (медиана)",
     "is": "медиана по местам: какую долю разобранных кадров место было видно.",
     "is_not": "не доля урока, в которую ребёнок присутствовал: место бывает не видно "
               "из-за соседа, а не из-за отсутствия."},
    {"key": "coverage_min", "label": "видимость места (минимум)",
     "is": "самое плохо видимое место этой записи.",
     "is_not": "не свойство ребёнка. Ниже 50 % индекс по такому месту не считается вовсе."},
    {"key": "unassigned_share", "label": "вне мест, %",
     "is": "доля наблюдений человека, не попавших ни на одно найденное место.",
     "is_not": "не «дети ходили по классу»: сюда же попадают вошедшие взрослые, "
               "проходящие мимо и любые фигуры, которых кластеризация не признала местом."},
)


# ---------------------------------------------------------------------------
# Voiding.
# ---------------------------------------------------------------------------


def voided_counters(unmeasured: list[dict[str, Any]] | None,
                    table: dict[str, tuple[str, ...]] | None = None) -> dict[str, str]:
    """Counter key -> the artefact's own sentence saying why it cannot be measured here.

    The value is the artefact's `what` and `why` joined, never a sentence written here: the
    reason a camera cannot see its board is a fact about that room, and the analyser is the
    only thing that knows which of the two reasons applies (`MEASUREMENTS.md` §12.6).
    """
    table = COUNTER_VOIDED_BY if table is None else table
    out: dict[str, str] = {}
    for item in unmeasured or []:
        what = str((item or {}).get("what") or "")
        for key in table.get(what, ()):
            why = str((item or {}).get("why") or "").strip()
            # The artefact's `what` is kept beside our own column label on purpose: the two
            # name the same event («выходил к доске» here, «стоял у доски» there), and a
            # reader checking this page against the artefact needs the artefact's own word
            # to find it there.
            # No quotation marks added around `what`: one of the analyser's own strings is
            # «за своим столом» для взрослого, which already carries a pair, and wrapping it
            # produced ««за своим столом» для взрослого» on the page.
            out[key] = f"{what}: {why}" if why else what
    return out


def _unmeasured_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    # `.get`, not `row["..."]`: the rows here are plain dicts built by `lesson_views`, and a
    # lesson with no selected run has no run columns at all rather than null ones.
    raw = row.get("unmeasured_json")
    try:
        return json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# One place, in one lesson.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlaceInLesson:
    """One place for one lesson, with its coverage in front of its counters.

    Deliberately built from `seat_lessons` rather than re-read from the artefact: the
    cabinet's copy is what every other page on this surface is computed from, and a lesson
    card that quietly went back to the JSON would be a second reader that can disagree with
    the first.
    """

    place_id: int | None
    ordinal: int | None
    label_ru: str
    role: str
    seat_label: str
    place_match: str
    place_match_reason_ru: str
    settled: bool | None
    coverage: float | None
    observations: int
    absent_observations: int
    observed_minutes: float
    counts: dict[str, int]
    index: float | None
    index_available: bool
    index_reason_ru: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place_id, "ordinal": self.ordinal,
            "label_ru": self.label_ru, "role": self.role,
            "seat_label": self.seat_label, "place_match": self.place_match,
            "settled": self.settled, "coverage": self.coverage,
            "observations": self.observations,
            "absent_observations": self.absent_observations,
            "observed_minutes": round(self.observed_minutes, 1),
            "counts": dict(self.counts), "index": self.index,
            "index_available": self.index_available,
            "index_reason_ru": self.index_reason_ru,
        }


# ---------------------------------------------------------------------------
# One lesson.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LessonView:
    """One SESSION — everything the lesson card renders, computed once."""

    room_key: str
    class_key: str
    session_id: int
    parts: list[dict[str, Any]]            # the `lessons` rows, in clock order
    date_local: str | None
    iso_year: int | None
    iso_week: int | None
    started_at: str | None
    ended_at: str | None
    clock_source: str
    window_minutes: float | None
    analysed_frames: int
    sample_fps: float | None
    thresholds_shas: list[str]
    run_ids: list[str]
    schemas: list[str]
    places: list[PlaceInLesson]
    counts: dict[str, int]                 # class-level, PUPIL places only
    voided: dict[str, str]
    coverage_median: float | None
    coverage_min: float | None
    place_minutes: float
    observations_total: int
    observations_unassigned: int
    seats_never_settled: int
    frames_with_no_person: int
    unmeasured: list[dict[str, Any]]
    teacher: dict[str, Any] | None
    adult_place_missing_reason_ru: str
    notes_ru: list[str] = field(default_factory=list)
    # A zero the SAME artefact contradicts. See `_board_conflict`.
    board_conflict_ru: str = ""

    @property
    def week_label(self) -> str:
        if self.iso_year is None or self.iso_week is None:
            return "без даты"
        return f"{self.iso_year}-W{self.iso_week:02d}"

    @property
    def pupil_places(self) -> list[PlaceInLesson]:
        return [p for p in self.places if p.role == "pupil"]

    @property
    def unassigned_share(self) -> float | None:
        """Share of person-observations that landed on no place at all.

        `None` and not `0.0` when the run stored no observation total: a share with an
        empty denominator is not a small share.
        """
        if not self.observations_total:
            return None
        return self.observations_unassigned / self.observations_total

    @property
    def title_ru(self) -> str:
        return (f"Урок {self.date_local or 'без даты'} · комната {self.room_key}"
                if self.date_local else f"Урок без даты · комната {self.room_key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_key": self.room_key, "class_key": self.class_key,
            "session_id": self.session_id, "date_local": self.date_local,
            "week": self.week_label, "started_at": self.started_at,
            "ended_at": self.ended_at, "clock_source": self.clock_source,
            "parts": [int(p["lesson_id"]) for p in self.parts],
            "window_minutes": self.window_minutes,
            "analysed_frames": self.analysed_frames,
            "run_ids": self.run_ids, "thresholds": self.thresholds_shas,
            "pupil_places": len(self.pupil_places),
            "places": [p.to_dict() for p in self.places],
            "counts": dict(self.counts), "voided_counters": self.voided,
            "coverage_median": self.coverage_median, "coverage_min": self.coverage_min,
            "place_minutes": round(self.place_minutes, 1),
            "observations_total": self.observations_total,
            "observations_unassigned": self.observations_unassigned,
            "unassigned_share": self.unassigned_share,
            "seats_never_settled": self.seats_never_settled,
            "unmeasured": self.unmeasured,
            "teacher": self.teacher,
            "adult_place_missing_reason_ru": self.adult_place_missing_reason_ru,
            "notes_ru": self.notes_ru,
            "board_conflict_ru": self.board_conflict_ru,
        }


def _board_conflict(counts: dict[str, int], voided: dict[str, str],
                    teacher: dict[str, Any] | None,
                    unassigned_share: float | None) -> str:
    """A zero in «выходил к доске» that this artefact's OWN board counter contradicts.

    Found on the real D14 session rather than reasoned about. That camera has the board in
    frame, so «выходил к доске» is measurable and the column is not voided — and it comes out
    **0 for all five pupil places**, while the same artefact's `board_occupancy` reports
    somebody standing at that board for **30.4 minutes, 52.3 % of the lesson**, of which the
    adult accounts for 4.0. Roughly twenty-six minutes of pupils at a board, under a column
    saying no pupil went to it.

    The cause is not a broken zone; it is what the counter is ABOUT. `AT_BOARD` is a state of
    a PLACE, and a child who walks to the board has left their place — the observation lands
    in «уходил с места» or, more often, in the 18.0 % of this recording's observations that
    belong to no place at all. So the zero is true of the ledger and false of the room, which
    is the exact difference this project spends its docstrings on.

    It is not repaired here: repairing it means changing `ledger.classify`, and that is a new
    measurement of every lesson ever analysed, not a display decision. What is repaired is
    the reader's exposure to it — the number is printed with the contradicting number beside
    it, from the same artefact, so that nobody quotes one without the other.
    """
    if "board_visits" in voided or counts.get("board_visits"):
        return ""
    occupancy = ((teacher or {}).get("board_occupancy") or {})
    if not occupancy.get("available"):
        return ""
    board = ((teacher or {}).get("states") or [])
    adult_minutes = next((s.get("minutes_of_lesson") for s in board
                          if s.get("key") == "at_board"), None)
    return (
        f"«Выходил к доске» по ученическим местам — 0, при том что по ЭТОЙ ЖЕ записи у "
        f"доски кто-то находился {occupancy.get('minutes')} мин "
        f"({occupancy.get('share_of_lesson_percent')} % урока)"
        + (f", а на взрослого из них приходится {adult_minutes} мин"
           if adult_minutes is not None else "")
        + ". Ноль здесь не означает, что ученики к доске не выходили: «у доски» — это "
          "состояние МЕСТА, а ребёнок, отошедший от своей парты, попадает в «уходил с "
          "места» или в наблюдения вне мест"
        + (f" ({unassigned_share * 100:.1f} % этой записи)"
           if unassigned_share is not None else "")
        + ". Это ограничение разбора, а не наблюдение о детях, и сравнивать этот столбец "
          "между записями нельзя.")


def board_conflict_for_run(connection: sqlite3.Connection, run_id: str) -> str:
    """The same contradiction, asked of ONE run rather than of a rendered lesson. "" if none.

    `lesson_views` answers this only for SELECTED runs, because that is all a page shows.
    Anything that has to decide what a run means — `cabinet/notes.py` builds a bundle for
    any run id, including the losing side of a re-analysis — needs the answer for the run it
    was handed, and falling back to "" for an unselected run would hand back the confident
    zero precisely where nobody is looking. So the question is asked of the run's own row,
    through the same `_board_conflict` the page uses; there is one rule about this zero in
    this package and it is written above.

    The pupil sum is recomputed here from `seat_lessons` rather than taken from a folded
    `LessonView`: a session assembled from two files has two runs, and each one's board
    column has to answer for its own file.
    """
    run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return ""
    row = dict(run)
    unmeasured = json.loads(row["unmeasured_json"] or "[]")
    # Column names come from the module constant `COUNTER_KEYS`, never from a caller, so the
    # interpolation is a literal list written three lines above and not an injection surface.
    columns = ", ".join(f"COALESCE(SUM({key}), 0)" for key, _label in COUNTER_KEYS)
    summed = connection.execute(
        f"SELECT {columns} FROM seat_lessons WHERE run_id = ? AND role = 'pupil'",
        (run_id,)).fetchone()
    counts = {key: int(summed[index] or 0)
              for index, (key, _label) in enumerate(COUNTER_KEYS)}
    total = row.get("observations_total") or 0
    share = (row.get("observations_unassigned") or 0) / total if total else None
    return _board_conflict(counts, voided_counters(unmeasured),
                           _teacher_block(row, unmeasured), share)


def _teacher_block(run_row: dict[str, Any],
                   unmeasured: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The adult's position split for one run, with the structurally-absent states voided.

    Two shares of the SAME adult in the SAME artefact answer two different questions, and
    this block is the one that answers «где в комнате он был» —
    `presence.state_share_of_lesson_percent`. The other one,
    `metrics.at_desk_share_of_observed`, answers «что делало его тело за его собственным
    местом» and is 96.5 % on the very camera whose position split reports `at_desk: 0.0`.
    Putting them in one column would be `MEASUREMENTS.md` §8's name collision arriving for
    the fourth time, so this block never mixes them and the adult's own page keeps the
    ledger half under its own heading.
    """
    raw = run_row.get("teacher_json")
    if not raw or raw in ("null", ""):
        return None
    record = json.loads(raw) or {}
    metrics = record.get("metrics") or {}
    if not metrics:
        return None
    if not metrics.get("available"):
        return {"available": False,
                "reason_ru": str(metrics.get("reason") or "").strip(),
                "not_an_assessment_ru": metrics.get("not_an_assessment_ru") or "",
                "states": [], "voided": {}, "attributed_percent": None,
                "transitions_between_episodes": None, "lesson_minutes": None}

    presence = metrics.get("presence") or {}
    shares = presence.get("state_share_of_lesson_percent") or {}
    minutes = presence.get("state_minutes_of_lesson") or {}
    episodes = presence.get("episode_counts") or {}
    voided = voided_counters(unmeasured, TEACHER_STATE_VOIDED_BY)
    if not (presence.get("board") or {}).get("zone_configured", False):
        voided.setdefault("at_board",
                          "зона доски для этой камеры не задана — состояние «у доски» не "
                          "может возникнуть ни в одном кадре")

    states = [{
        "key": key, "label_ru": label,
        "share_of_lesson_percent": None if key in voided else shares.get(key),
        "minutes_of_lesson": None if key in voided else minutes.get(key),
        "episodes": None if key in voided else episodes.get(key),
        "voided_why_ru": voided.get(key, ""),
    } for key, label in TEACHER_STATES]

    return {
        "available": True,
        "reason_ru": "",
        "not_an_assessment_ru": metrics.get("not_an_assessment_ru") or "",
        "states": states,
        "voided": voided,
        "attributed_percent": presence.get("attributed_share_of_lesson_percent"),
        "lesson_minutes": presence.get("lesson_minutes"),
        # The POSITION-episode boundary count, under its own unambiguous name. Never
        # `metrics["transitions"]`, which counts POSE changes at his own place and is `None`
        # on the branch where the adult never settles anywhere — see `metrics/teacher.py`.
        "transitions_between_episodes": presence.get("transitions_between_episodes"),
        "transitions_excluding_out_of_frame":
            presence.get("transitions_between_episodes_excluding_out_of_frame"),
        "board_direction_of_error_ru":
            ((presence.get("board") or {}).get("direction_of_error_ru") or ""),
        "board_occupancy": presence.get("board_occupancy"),
    }


def _places_of_run(connection: sqlite3.Connection, run_id: str) -> list[PlaceInLesson]:
    rows = connection.execute(
        """SELECT s.*, p.ordinal, p.label_ru AS place_label
           FROM seat_lessons s LEFT JOIN places p ON p.place_id = s.place_id
           WHERE s.run_id = ?
           ORDER BY s.role = 'adult', p.ordinal IS NULL, p.ordinal, s.seat_id""",
        (run_id,))
    out: list[PlaceInLesson] = []
    for row in rows:
        counts = {key: int(row[key] or 0) for key, _label in COUNTER_KEYS}
        out.append(PlaceInLesson(
            place_id=None if row["place_id"] is None else int(row["place_id"]),
            ordinal=None if row["ordinal"] is None else int(row["ordinal"]),
            label_ru=str(row["place_label"] or row["seat_label"] or "место"),
            role=str(row["role"]),
            seat_label=str(row["seat_label"] or ""),
            place_match=str(row["place_match"]),
            place_match_reason_ru=str(row["place_match_reason_ru"] or ""),
            settled=None if row["settled"] is None else bool(row["settled"]),
            coverage=None if row["coverage"] is None else float(row["coverage"]),
            observations=int(row["observations"] or 0),
            absent_observations=int(row["absent_observations"] or 0),
            observed_minutes=float(row["observed_seconds"] or 0.0) / 60.0,
            counts=counts,
            index=None if row["activity_index"] is None else float(row["activity_index"]),
            index_available=bool(row["activity_available"]),
            index_reason_ru=str(row["activity_reason_ru"] or "")))
    return out


def lesson_views(connection: sqlite3.Connection, *, room_key: str | None = None,
                 class_key: str | None = None) -> list[LessonView]:
    """Every SESSION in scope, oldest first, undated ones last.

    Undated lessons are present here and absent from `weekly.py`, and the difference is not
    an inconsistency: a recording whose clock could not be read cannot be placed on a
    calendar and therefore cannot join anything longitudinal, but it was still watched for
    an hour and its own card is the one place that hour is readable at all. The card says
    so in its own words rather than looking like a lesson that simply has no date column.
    """
    where, args = store._scope(room_key, class_key, prefix="l.")
    rows = [dict(row) for row in connection.execute(
        f"""SELECT l.*, r.run_id AS selected_run, r.unmeasured_json, r.teacher_json,
                   r.window_start_seconds, r.window_end_seconds, r.analysed_frames,
                   r.sample_fps, r.observations_total, r.observations_unassigned,
                   r.observations_unreadable, r.seats_never_settled,
                   r.frames_with_no_person, r.thresholds_sha, r.schema AS run_schema
            FROM lessons l LEFT JOIN runs r ON r.run_id = l.selected_run_id{where}
            ORDER BY l.started_at IS NULL, l.started_at""", args)]

    heads = _chain_heads(connection)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        head = heads.get(int(row["lesson_id"]), int(row["lesson_id"]))
        grouped.setdefault(head, []).append(row)

    views = [_fold_lesson(connection, head, parts) for head, parts in grouped.items()]
    views.sort(key=lambda v: (v.started_at is None, v.started_at or ""))
    return views


def _fold_lesson(connection: sqlite3.Connection, head: int,
                 parts: list[dict[str, Any]]) -> LessonView:
    parts = sorted(parts, key=lambda r: (r["started_at"] is None, r["started_at"] or ""))
    first = parts[0]
    last = parts[-1]

    places: list[PlaceInLesson] = []
    unmeasured: list[dict[str, Any]] = []
    teacher: dict[str, Any] | None = None
    frames = 0
    observations = 0
    unassigned = 0
    never_settled = 0
    no_person = 0
    window_minutes: float | None = None
    notes: list[str] = []

    for part in parts:
        run_id = part.get("selected_run")
        if run_id is None:
            notes.append(
                f"у записи №{part['lesson_id']} нет выбранного измерения — её содержимое в "
                f"эту карточку не входит. Это не пустой урок: прогоны, если они есть, "
                f"перечислены в `cabinet show`, и один из них нужно выбрать командой "
                f"`cabinet select-run`.")
            continue
        part_unmeasured = _unmeasured_of(part)
        for item in part_unmeasured:
            if item not in unmeasured:
                unmeasured.append(item)
        frames += int(part["analysed_frames"] or 0)
        observations += int(part["observations_total"] or 0)
        unassigned += int(part["observations_unassigned"] or 0)
        never_settled += int(part["seats_never_settled"] or 0)
        no_person += int(part["frames_with_no_person"] or 0)
        span = part["window_end_seconds"], part["window_start_seconds"]
        if span[0] is not None and span[1] is not None:
            window_minutes = (window_minutes or 0.0) + (float(span[0]) - float(span[1])) / 60.0
        if teacher is None:
            teacher = _teacher_block(part, part_unmeasured)
        places.extend(_places_of_run(connection, str(run_id)))

    if len(parts) > 1:
        notes.append(
            f"занятие записано {len(parts)} файлами подряд: счётчики по местам сложены, а "
            f"доли положения взрослого — нет, потому что доля от целого не складывается с "
            f"долей от другого целого. Положение взрослого показано по первой части, и это "
            f"названо здесь, а не подсчитано молча.")

    # Two seats of one place across two parts must not become two rows on the card.
    merged: dict[Any, PlaceInLesson] = {}
    for place in places:
        key = place.place_id if place.place_id is not None else ("unplaced", place.seat_label)
        if key in merged:
            existing = merged[key]
            existing.observations += place.observations
            existing.absent_observations += place.absent_observations
            existing.observed_minutes += place.observed_minutes
            for counter, value in place.counts.items():
                existing.counts[counter] = existing.counts.get(counter, 0) + value
            total = existing.observations + existing.absent_observations
            existing.coverage = (existing.observations / total) if total else None
            # An index per PART is not an index of the session. `weekly._fold` recomputes it
            # from the summed ledger with `metrics/activity.py`; this card is not the place
            # to produce a second, quietly different answer, so it declines to show one.
            existing.index = None
            existing.index_available = False
            existing.index_reason_ru = (
                "занятие склеено из нескольких файлов: индекс за него считается по сумме "
                "наблюдений на странице места, а не складывается из индексов частей.")
        else:
            merged[key] = place
    places = list(merged.values())

    pupils = [p for p in places if p.role == "pupil"]
    coverages = [p.coverage for p in pupils if p.coverage is not None]
    counts = {key: sum(p.counts.get(key, 0) for p in pupils) for key, _ in COUNTER_KEYS}

    adult_missing = ""
    if not any(p.role == "adult" for p in places):
        raw = first.get("teacher_json")
        record = json.loads(raw) if raw and raw not in ("null", "") else None
        if record:
            why = str((record.get("metrics") or {}).get("reason") or "").strip()
            adult_missing = (
                "разбор взрослого в этой записи есть, но его положения в кадре в артефакте "
                "нет (centre/scale_px пусты)" + (f": {why}" if why else "")
                + ". Места взрослого по этому уроку не заведено — это ОТСУТСТВИЕ "
                  "ИЗМЕРЕНИЯ, а не отсутствие взрослого в классе.")
        else:
            adult_missing = (
                "в артефакте нет блока о взрослом вообще. Это тоже не вывод о том, что "
                "взрослого в классе не было: это значит, что его никто не искал.")

    voided = voided_counters(unmeasured)
    view = LessonView(
        room_key=str(first["room_key"]), class_key=str(first["class_key"]),
        session_id=head, parts=parts,
        date_local=first["date_local"],
        iso_year=first["iso_year"], iso_week=first["iso_week"],
        started_at=first["started_at"], ended_at=last["ended_at"],
        clock_source=str(first["clock_source"]),
        window_minutes=window_minutes,
        analysed_frames=frames,
        sample_fps=first["sample_fps"],
        thresholds_shas=sorted({str(p["thresholds_sha"]) for p in parts
                                if p["thresholds_sha"]}),
        run_ids=[str(p["selected_run"]) for p in parts if p["selected_run"]],
        schemas=sorted({str(p["run_schema"]) for p in parts if p["run_schema"]}),
        places=places, counts=counts,
        voided=voided,
        coverage_median=statistics.median(coverages) if coverages else None,
        coverage_min=min(coverages) if coverages else None,
        place_minutes=sum(p.observed_minutes for p in pupils),
        observations_total=observations, observations_unassigned=unassigned,
        seats_never_settled=never_settled, frames_with_no_person=no_person,
        unmeasured=unmeasured, teacher=teacher,
        adult_place_missing_reason_ru=adult_missing,
        notes_ru=notes + [p["overlap_note"] for p in parts if p.get("overlap_note")])
    view.board_conflict_ru = _board_conflict(counts, voided, teacher,
                                             view.unassigned_share)
    return view


# ---------------------------------------------------------------------------
# Places whose numbers this lesson does not support. NOT a list of children.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reservation:
    """One place, and the reason a figure of this lesson does not stand for it.

    Every entry here is a statement about a MEASUREMENT — the place was not visible enough,
    or its posture baseline never settled. None of them is a statement about a child, none
    of them is ordered by severity, and there is deliberately no counterpart list of places
    that «стоит посмотреть»: that list would be a flag, and rule 1 of this project has no
    exception for a gently-worded one.
    """

    label_ru: str
    what_ru: str
    detail_ru: str


def reservations(view: LessonView) -> list[Reservation]:
    from classvision.metrics.activity import MIN_COVERAGE

    out: list[Reservation] = []
    for place in view.pupil_places:
        if place.coverage is not None and place.coverage < MIN_COVERAGE:
            out.append(Reservation(
                place.label_ru, "видно меньше половины разобранных кадров",
                f"видимость {place.coverage:.0%} при пороге {MIN_COVERAGE:.0%}. Индекс по "
                f"этому месту не считается вовсе, а счётчики относятся только к тем "
                f"кадрам, где место было видно: «ноль раз» здесь означает «ноль раз из "
                f"того, что мы видели»."))
        elif place.settled is False:
            out.append(Reservation(
                place.label_ru, "базовая поза не установлена",
                "за первые наблюдения этого места не набралось согласованных по масштабу "
                "показаний, поэтому «голова опущена» и «отвернулся» для него не "
                "определены: сравнивать позу не с чем."))
    return out


# ---------------------------------------------------------------------------
# Across lessons: rooms, classes, and the comparison that is allowed.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassAcrossRooms:
    """One class KEY and every room its lessons were recorded in.

    The class key is an operator assertion (`store._class_key`), and two lessons sharing one
    is therefore evidence about what somebody typed, not about who was in the room. When the
    same key spans two rooms, that is the single most dangerous state this cabinet can be
    in: the pages beneath it look like one class's term, and the arithmetic that would join
    them is one careless glance away. `warning_ru` exists so that the glance lands on a
    sentence instead.
    """

    class_key: str
    rooms: list[dict[str, Any]]

    @property
    def multi_room(self) -> bool:
        return len(self.rooms) > 1

    @property
    def stated(self) -> bool:
        """Whether a human actually typed this class key.

        `store._class_key` falls back to `"-"` with source `not_stated`, so `"-"` is not a
        class, it is the absence of one — and two lessons under it have nothing in common at
        all. A page that renders it as a class name invents an entity.
        """
        return self.class_key not in ("-", "", None)

    @property
    def display_ru(self) -> str:
        return f"класс {self.class_key}" if self.stated else "класс не указан"

    def to_dict(self) -> dict[str, Any]:
        return {"class_key": self.class_key, "stated": self.stated,
                "multi_room": self.multi_room, "rooms": self.rooms,
                "warning_ru": self.warning_ru()}

    def warning_ru(self) -> list[str]:
        """The cross-room paragraph, as a list of sentences the page renders as bullets.

        Assembled from the store's own numbers — room keys, dates, pupil-place counts — so
        that it cannot describe a configuration the database does not hold. A frozen warning
        would still be printed after somebody fixed the class keys, and a warning that is
        wrong is worse than none because it is still believed.
        """
        if not self.multi_room:
            return []
        listed = "; ".join(
            f"{room['room_key']} — {room['lessons']} "
            f"{lessons_word(int(room['lessons']))}, "
            f"{room['first_date'] or 'без даты'}"
            + (f"…{room['last_date']}" if room["last_date"] != room["first_date"] else "")
            + (f", ученических мест {room['pupil_places']}"
               if room["pupil_places"] is not None else "")
            for room in self.rooms)
        # The two halves of this sentence are built separately because the un-stated case is
        # not «класс называется “-”»: it is «класса никто не называл», and a sentence that
        # slots the second into the grammar of the first stops being Russian — which on a
        # page whose whole job is to be believed is not a cosmetic defect.
        key = ("то, что класс при загрузке НЕ указывали: «-» — это не имя класса, а его "
               "отсутствие" if not self.stated else
               f"строка «{self.class_key}», введённая оператором при загрузке")
        return [
            f"Записи под этим ключом сняты разными камерами: {listed}.",
            f"Это разные комнаты. Общее у них — только {key}; ни один артефакт не "
            f"утверждает, что в них один и тот же класс, и число найденных ученических "
            f"мест у них разное.",
            "«Место 3» одной камеры и «место 3» другой — два разных стула в двух разных "
            "комнатах. Номер места — это порядок чтения кадра (`room/seats.py` сортирует "
            "найденные кластеры сверху вниз), а не человек: совпадение номеров не значит "
            "ничего.",
            "Поэтому разница между неделями этих записей НЕ является динамикой и нигде "
            "здесь не выводится: между ними нет ни одного общего ребёнка, о котором такое "
            "утверждение можно было бы сделать. Комнаты не суммируются: у каждой свои "
            "места, своя история и свои страницы.",
            "Сравнимы между такими записями только свойства САМИХ ЗАПИСЕЙ — сколько "
            "времени места были видны, сколько мест найдено, сколько раз в комнате "
            "произошло каждое наблюдаемое событие. Это на странице «Что сравнимо между "
            "уроками».",
            "Чтобы этого предупреждения не было: загружайте разборы с явно указанным "
            "классом (`classvision cabinet import … --room-key D14 --class 8-А`), и тогда "
            "разные классы перестанут делить один ключ.",
        ]


def classes_across_rooms(connection: sqlite3.Connection) -> list[ClassAcrossRooms]:
    """Every class key with the rooms it was filed under, and how much each room holds."""
    rows = connection.execute(
        """SELECT l.class_key, l.room_key, COUNT(DISTINCT l.lesson_id) AS lessons,
                  SUM(CASE WHEN l.continues_lesson_id IS NULL THEN 1 ELSE 0 END) AS sessions,
                  MIN(l.date_local) AS first_date, MAX(l.date_local) AS last_date
           FROM lessons l GROUP BY l.class_key, l.room_key
           ORDER BY l.class_key, l.room_key""")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entry = dict(row)
        entry["pupil_places"] = int(connection.execute(
            "SELECT COUNT(*) FROM places WHERE room_key = ? AND class_key = ? "
            "AND role = 'pupil'",
            (row["room_key"], row["class_key"])).fetchone()[0] or 0)
        grouped.setdefault(str(row["class_key"]), []).append(entry)
    return [ClassAcrossRooms(class_key=key, rooms=value)
            for key, value in sorted(grouped.items())]


@dataclass(slots=True)
class ComparableView:
    """Every lesson in the cabinet as a row about the RECORDING, plus what that is not."""

    lessons: list[LessonView]
    classes: list[ClassAcrossRooms]
    columns: tuple[dict[str, str], ...] = COMPARABLE_COLUMNS

    @property
    def cross_room(self) -> list[ClassAcrossRooms]:
        return [c for c in self.classes if c.multi_room]

    @property
    def rooms(self) -> list[str]:
        out: list[str] = []
        for lesson in self.lessons:
            if lesson.room_key not in out:
                out.append(lesson.room_key)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "lessons": [lesson.to_dict() for lesson in self.lessons],
            "classes": [c.to_dict() for c in self.classes],
            "rooms": self.rooms,
            "columns": list(self.columns),
            "what_this_is_ru": WHAT_THIS_IS_RU,
            "what_this_is_not_ru": WHAT_THIS_IS_NOT_RU,
        }


WHAT_THIS_IS_RU = (
    "Каждая строка — свойство ОДНОЙ ЗАПИСИ целиком: сколько времени места были видны "
    "камере, сколько мест нашла кластеризация, сколько раз в комнате произошло каждое "
    "наблюдаемое событие, где был взрослый. Это то, что можно сопоставить между двумя "
    "записями, не зная, кто в них сидел."
)

WHAT_THIS_IS_NOT_RU = (
    "Ни одна строка здесь не является утверждением о ребёнке. Строка — это сумма по всей "
    "комнате, и она меняется, когда меняется число детей в кадре, качество обзора или "
    "разметка камеры, а не когда меняется поведение кого-то одного. Ни одна пара строк "
    "здесь не является динамикой: динамика строится только по конкретному месту одной "
    "комнаты и только при выполнении четырёх условий, перечисленных на странице этого места."
)


def comparable(connection: sqlite3.Connection) -> ComparableView:
    """Every lesson in the cabinet, in clock order, across all rooms and classes."""
    return ComparableView(lessons=lesson_views(connection),
                          classes=classes_across_rooms(connection))

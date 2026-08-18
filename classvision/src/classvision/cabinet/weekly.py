"""«Первая неделя хорошо, вторая не очень» — assembled from stored lessons, honestly.

--------------------------------------------------------------------------------
**THE SESSION, NOT THE FILE, IS THE UNIT.**

A lesson is a recording. A *session* is a lesson as a human means it, which is sometimes
two recordings: the D14 pair is one hour cut in two by the DVR at `10:31:35 / 10:31:36`.
`store.py` records that as `continues_lesson_id`; this file chains it. Counting the two
files as two lessons would put «два урока» in a week that had one, and would halve every
per-lesson figure of that hour.

**Chaining is a sum of ledgers, not a sum of results.** For a two-part session the
counters add (a hand raised in part two is a hand raised), the observations add, and the
activity index is **recomputed from the summed ledger by `metrics/activity.py` itself** —
the same function, on a bigger ledger. Averaging two indices would be a second, quietly
different implementation of a weighted sum, and the whole reason `activity.py` publishes
`parts` is that a number nobody can take apart is a verdict in costume. A test asserts
that a one-part session reproduces the artefact's own index exactly, which is what makes
the two-part case trustworthy.

--------------------------------------------------------------------------------
**THE WEEK IS A DISPLAY GRAIN. THE TREND IS PER LESSON.**

The client's sentence is about weeks, so the table is weekly. `metrics/trend.py` counts in
LESSONS — `MIN_HISTORY_LESSONS = 4` and `BASELINE_LESSONS = 8` are lessons of a pupil's own
history, chosen against «примерно один разобранный урок в неделю». Feeding it week medians
would silently redefine both constants, so it is fed the session series and the week table
is what the page shows beside it. Two grains, stated, rather than one grain quietly
changed.

**A week with no lessons and a week with zero hand-raises are different rows.** Rule 3 of
this project. `WeekRow.observed` is False for the first and the counters are `None`; for
the second they are `0`. A table that renders both as `0` invents a lesson.

--------------------------------------------------------------------------------
**THE REFUSAL IS A STATE, NOT AN ERROR.**

With one lesson in the store, «динамики нет» IS the answer, and a page that renders an
empty chart instead looks broken and gets debugged rather than read. So `TrendView` has
named states, each with its own Russian sentence and its own explanation of what would
change it, and `trend_gates` lists every precondition with met/unmet — the same shape
`identity/assign.py` already uses for names, for the same reason: a reader needs to know
which of four things is missing, not merely that something is.

The refusals that will fire in a real school, for genuinely different reasons:

  * `insufficient_lessons` — «данных пока мало: 1 урок из 5». Fixed by time.
  * `identity_not_established` — the place has no signed seating plan, so a trend would be
    a statement about a chair. Fixed by a human, never by more data. `metrics/trend.py`
    refuses on this outright and this file does not soften it.
  * `no_dated_lessons` — the recordings are in the store but their clock could not be read.
    Fixed by neither waiting nor signing, but by a date.
  * `not_applicable_adult` — refused by ROLE. `metrics/activity.py` will happily compute an
    index from the adult's ledger, because it has the same shape as a pupil's; the first
    build of this file let it, and printed «индекс 91,2 · поднимал руку 60» for a teacher
    who was pointing at a board. That is a teaching-quality score assembled out of a
    pupil's vocabulary, which `metrics/teacher.py` and INTEGRATION.md §7 both refuse to
    produce, so the adult gets no index, no weeks and no trend — only position per lesson.

--------------------------------------------------------------------------------
**NOTHING HERE RANKS ANYBODY.** No sort by index, no colour scale, no «below average»,
no class mean to be below. Places come out in room order, which is geometry and not
judgement. The comparison is always a pupil against themselves — that is
`metrics/trend.py`'s first paragraph and this file adds no second opinion.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from classvision.cabinet import store
from classvision.metrics import trend as trend_mod
from classvision.metrics.activity import Activity, activity
from classvision.states import RU_LABELS, PupilState

# Counters carried per session and per week. Each is an EPISODE count from the artefact's
# ledger, which is the number a teacher would recognise («три раза вставал»), not a frame
# count. Order is display order and is deliberately not by magnitude.
COUNTER_KEYS: tuple[tuple[str, str], ...] = (
    ("hand_raises", "поднимал руку"),
    ("board_visits", "выходил к доске"),
    ("stands", "вставал на месте"),
    ("away_episodes", "уходил с места"),
    ("head_down_episodes", "клал голову на парту"),
    ("turned_away_episodes", "отворачивался назад"),
)

# Place-match outcomes that are good enough to carry a history forward. "new" is included
# for the FIRST lesson of a place only — handled below — because every place is new once.
GOOD_MATCH = frozenset({"matched", "new"})


# ---------------------------------------------------------------------------
# Re-computing the index over a summed ledger, with `metrics/activity.py`.
# ---------------------------------------------------------------------------


class _Baselines:
    __slots__ = ("settled",)

    def __init__(self, settled: bool) -> None:
        self.settled = settled


class _SummedLedger:
    """The smallest object `metrics/activity.py` will accept, built from stored rows.

    Deliberately a duck-typed shim rather than a rebuilt `SeatLedger`: a real ledger owns
    episodes, baselines and a classifier's history, none of which survives into the
    artefact and none of which the index reads. Reconstructing a fake one would invite
    someone to compute an episode from it later, and episodes cannot be recovered from a
    sum — two parts of a split lesson can cut one episode in half, so the episode COUNTS
    are taken from the artefact and added, and no episode object is invented here.

    `count()` raises on a state that has no stored counter instead of returning 0, because
    a silently-zero participation term is exactly the sort of quiet wrongness this package
    keeps being written against.
    """

    __slots__ = ("_counts", "baselines", "coverage", "state_observations")

    def __init__(self, coverage: float, settled: bool,
                 state_observations: dict[str, int], counts: dict[str, int]) -> None:
        self.coverage = coverage
        self.baselines = _Baselines(settled)
        self.state_observations = state_observations
        self._counts = counts

    def count(self, state: PupilState) -> int:
        key = store.COUNT_KEYS.get(state.value, "MISSING")
        if key in (None, "MISSING"):
            raise KeyError(
                f"состояние {state.value!r} не имеет счётчика эпизодов в артефакте; "
                f"вернуть 0 здесь означало бы посчитать «не знаем» как «ноль».")
        return int(self._counts.get(key) or 0)


# ---------------------------------------------------------------------------
# Sessions.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionSeat:
    """One place, one session (one or two recordings), with everything qualifying it."""

    session_id: int                 # lesson_id of the first recording of the chain
    date_local: str | None
    iso_year: int | None
    iso_week: int | None
    parts: list[int]                # lesson_ids, in order
    observations: int
    absent_observations: int
    unreadable_observations: int
    hand_unmeasurable_observations: int
    observed_seconds: float
    coverage: float
    settled: bool
    counts: dict[str, int]
    activity: Activity
    stored_index: float | None      # what the artefact said, for cross-checking
    # "stable" | "unstable" — whether EVERY recording of this session put this seat on this
    # place unambiguously. Deliberately NOT called `place_match` and deliberately not
    # sharing that column's vocabulary: `seat_lessons.place_match` is per recording and its
    # values are "new" | "matched" | "ambiguous" | "moved". A session assembled entirely out
    # of "new" rows is perfectly stable, and under the old name it reported
    # `place_match: "matched"` — one field name carrying two meanings, and the one on the
    # outside was a claim ("this seat was recognised as a known place") that the rows
    # underneath had never made. That exact shape has already produced one false sentence
    # in a generated report in this project; it does not get a second.
    place_stability: str
    match_notes_ru: list[str] = field(default_factory=list)
    lesson_notes_ru: list[str] = field(default_factory=list)

    @property
    def week_label(self) -> str:
        if self.iso_year is None:
            return "без даты"
        return f"{self.iso_year}-W{self.iso_week:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "date_local": self.date_local,
            "week": self.week_label, "parts": self.parts,
            "observations": self.observations,
            "absent_observations": self.absent_observations,
            "observed_minutes": round(self.observed_seconds / 60.0, 1),
            "coverage": round(self.coverage, 3),
            "counts": dict(self.counts),
            "activity": self.activity.to_dict(),
            "stored_index": self.stored_index,
            "place_stability": self.place_stability,
            "notes_ru": self.match_notes_ru + self.lesson_notes_ru,
        }


def sessions_for_place(connection: sqlite3.Connection, place_id: int,
                       rows: list[dict[str, Any]] | None = None) -> list[SessionSeat]:
    """Every session this place was observed in, OLDEST FIRST.

    Undated lessons are excluded, not silently ordered last: `metrics/trend.py` compares a
    latest value against previous ones, and "previous" is meaningless for a recording that
    could not be placed on a calendar. They remain visible in the store and on the lesson
    list; they are simply not part of anything longitudinal, which is what
    `clock_source == "unknown"` was always supposed to mean.
    """
    if rows is None:
        rows = [dict(r) for r in connection.execute(
            """SELECT s.*, l.date_local, l.started_at, l.iso_year, l.iso_week,
                      l.continues_lesson_id, l.overlap_note, l.clock_source
               FROM seat_lessons s JOIN lessons l ON l.lesson_id = s.lesson_id
               WHERE s.place_id = ? AND s.run_id = l.selected_run_id
               ORDER BY l.started_at""", (place_id,))]

    chains = _chain_heads(connection)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("started_at") is None:
            continue
        head = chains.get(int(row["lesson_id"]), int(row["lesson_id"]))
        grouped.setdefault(head, []).append(row)

    out: list[SessionSeat] = []
    for head, parts in sorted(grouped.items(),
                              key=lambda pair: pair[1][0]["started_at"] or ""):
        out.append(_fold(head, sorted(parts, key=lambda r: r["started_at"] or "")))
    out.sort(key=lambda s: s.date_local or "")
    return out


# The adult's ledger has the same SHAPE as a pupil's, so `metrics/activity.py` will happily
# compute an index from it. It must not be allowed to. «Индекс наблюдаемой активности 91»
# beside the word «учитель» is a teaching-quality score, which is the one thing
# `metrics/teacher.py` and INTEGRATION.md §7 both refuse to produce — and its components
# would be nonsense anyway: the adult's raised-arm episodes are pointing and writing, and
# «отвернулся назад» for a person who faces the class is the normal posture, not a
# deviation from it. So the index is refused by role, with the reason carried in the object
# exactly as `activity()` carries its own refusals.
ADULT_NO_INDEX_RU = (
    "индекс активности для взрослого не считается: это была бы оценка работы учителя, "
    "построенная на признаках, которые для взрослого означают не то же, что для ученика. "
    "Положение взрослого по каждому уроку — в карточке урока."
)


def _chain_heads(connection: sqlite3.Connection) -> dict[int, int]:
    """lesson_id -> the lesson that starts its session. Resolves multi-file chains."""
    parents = {int(row["lesson_id"]): row["continues_lesson_id"]
               for row in connection.execute(
                   "SELECT lesson_id, continues_lesson_id FROM lessons")}
    heads: dict[int, int] = {}
    for lesson_id in parents:
        seen, current = set(), lesson_id
        while parents.get(current) is not None and current not in seen:
            seen.add(current)
            current = int(parents[current])
        heads[lesson_id] = current
    return heads


def _fold(head: int, parts: list[dict[str, Any]]) -> SessionSeat:
    """Sum the parts of one session and re-run the index over the sum."""
    observations = sum(int(p["observations"] or 0) for p in parts)
    absent = sum(int(p["absent_observations"] or 0) for p in parts)
    unreadable = sum(int(p["unreadable_observations"] or 0) for p in parts)
    hand_unmeasurable = sum(int(p["hand_unmeasurable_observations"] or 0) for p in parts)
    seconds = sum(float(p["observed_seconds"] or 0.0) for p in parts)

    states: dict[str, int] = {}
    for part in parts:
        for state, value in json.loads(part["state_observations_json"] or "{}").items():
            states[state] = states.get(state, 0) + int(value)

    counts: dict[str, int] = {}
    for key, _label in COUNTER_KEYS:
        counts[key] = sum(int(p[key] or 0) for p in parts)

    # Coverage over the whole session, from its own parts. Recomputed rather than averaged:
    # a 40-minute part at 0.99 and a 3-minute part at 0.20 do not average to the truth.
    total = observations + absent
    coverage = observations / total if total else 0.0
    settled = all(bool(p["settled"]) for p in parts)

    first = parts[0]
    if total == 0:
        # An empty denominator is not a coverage of zero. `observations / total` had to be
        # guarded against ZeroDivisionError, and the guard's value — 0.0 — then flowed into
        # `activity()`, which refused with «наблюдений слишком мало: 0 % кадров»: a measured
        # share, quoted about a seat that produced no samples at all. Rule 3, in the one
        # line where the guard was easier to write than the distinction.
        result = Activity(available=False, index=None, coverage=0.0,
                          reason="по этому месту в этом занятии нет ни одного наблюдения "
                                 "— ни удачного, ни неудачного. 0 % здесь не измеренная "
                                 "видимость, а отсутствие данных.")
    elif str(first["role"]) == "pupil":
        result = activity(_SummedLedger(coverage, settled, states, counts))
    else:
        result = Activity(available=False, index=None, coverage=coverage,
                          reason=ADULT_NO_INDEX_RU)

    stored = [p["activity_index"] for p in parts if p["activity_index"] is not None]
    notes = [p["place_match_reason_ru"] for p in parts
             if p["place_match"] not in GOOD_MATCH]
    lesson_notes = [p["overlap_note"] for p in parts if p.get("overlap_note")]
    if len(parts) > 1:
        lesson_notes.append(
            f"занятие записано {len(parts)} файлами подряд (регистратор разорвал запись); "
            f"счётчики и индекс посчитаны по сумме, а не по одной из частей.")

    return SessionSeat(
        session_id=head,
        date_local=first["date_local"],
        iso_year=first["iso_year"], iso_week=first["iso_week"],
        parts=[int(p["lesson_id"]) for p in parts],
        observations=observations, absent_observations=absent,
        unreadable_observations=unreadable,
        hand_unmeasurable_observations=hand_unmeasurable,
        observed_seconds=seconds, coverage=coverage, settled=settled,
        counts=counts, activity=result,
        stored_index=stored[0] if len(parts) == 1 and stored else None,
        place_stability=("stable" if all(p["place_match"] in GOOD_MATCH for p in parts)
                         else "unstable"),
        match_notes_ru=notes, lesson_notes_ru=lesson_notes)


# ---------------------------------------------------------------------------
# Weeks.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WeekRow:
    """One ISO week of one place. `observed` False means NO LESSON, not a zero."""

    iso_year: int
    iso_week: int
    observed: bool
    sessions: int = 0
    sessions_with_index: int = 0
    observed_minutes: float = 0.0
    coverage_min: float | None = None
    coverage_median: float | None = None
    counts: dict[str, int | None] = field(default_factory=dict)
    index_median: float | None = None
    index_values: list[float] = field(default_factory=list)
    unavailable_reasons_ru: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.iso_year}-W{self.iso_week:02d}"

    @property
    def monday(self) -> date:
        return date.fromisocalendar(self.iso_year, self.iso_week, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.label, "monday": self.monday.isoformat(),
            "observed": self.observed, "sessions": self.sessions,
            "sessions_with_index": self.sessions_with_index,
            "observed_minutes": round(self.observed_minutes, 1),
            "coverage_min": None if self.coverage_min is None else round(self.coverage_min, 3),
            "coverage_median": None if self.coverage_median is None
            else round(self.coverage_median, 3),
            "counts": self.counts,
            "index_median": None if self.index_median is None else round(self.index_median, 1),
            "index_values": [round(v, 1) for v in self.index_values],
            "unavailable_reasons_ru": self.unavailable_reasons_ru,
        }


def weeks_for(sessions: list[SessionSeat],
              span: list[tuple[int, int]] | None = None) -> list[WeekRow]:
    """Sessions folded into ISO weeks, with the empty weeks of `span` present and empty.

    The empty weeks are included on purpose. A term's page that lists only the weeks with
    data reads as an unbroken run of observation; the same page with the gaps in it reads
    as what actually happened, which is that a school has holidays, a camera has outages
    and a child has an illness. Their counters are `None`, never `0`.
    """
    buckets: dict[tuple[int, int], list[SessionSeat]] = {}
    for session in sessions:
        if session.iso_year is None or session.iso_week is None:
            continue
        buckets.setdefault((session.iso_year, session.iso_week), []).append(session)

    keys = sorted(set(buckets) | set(span or []))
    rows: list[WeekRow] = []
    for year, week in keys:
        group = buckets.get((year, week), [])
        if not group:
            rows.append(WeekRow(iso_year=year, iso_week=week, observed=False,
                                counts={key: None for key, _ in COUNTER_KEYS}))
            continue
        coverages = [s.coverage for s in group]
        indices = [s.activity.index for s in group if s.activity.available
                   and s.activity.index is not None]
        rows.append(WeekRow(
            iso_year=year, iso_week=week, observed=True, sessions=len(group),
            sessions_with_index=len(indices),
            observed_minutes=sum(s.observed_seconds for s in group) / 60.0,
            coverage_min=min(coverages), coverage_median=statistics.median(coverages),
            counts={key: sum(s.counts.get(key, 0) for s in group)
                    for key, _ in COUNTER_KEYS},
            # The MEDIAN of a week's lessons, not the mean: `metrics/trend.py` is built on
            # medians and MADs because a term has few lessons and one of them is a school
            # play, and a week's display statistic must not disagree with the statistic the
            # trend is computed from.
            index_median=statistics.median(indices) if indices else None,
            index_values=indices,
            unavailable_reasons_ru=[s.activity.reason for s in group
                                    if not s.activity.available and s.activity.reason]))
    return rows


def span_of(connection: sqlite3.Connection, *, room_key: str | None = None,
            class_key: str | None = None) -> list[tuple[int, int]]:
    """Every ISO week between the first and last dated lesson of this class, inclusive."""
    where, args = store._scope(room_key, class_key)
    row = connection.execute(
        f"SELECT MIN(date_local) AS a, MAX(date_local) AS b FROM lessons"
        f"{where}{' AND' if where else ' WHERE'} date_local IS NOT NULL", args).fetchone()
    if row is None or not row["a"]:
        return []
    first, last = date.fromisoformat(row["a"]), date.fromisoformat(row["b"])
    out, cursor = [], date.fromisocalendar(*first.isocalendar()[:2], 1)
    while cursor <= last:
        iso = cursor.isocalendar()
        out.append((iso.year, iso.week))
        cursor = date.fromordinal(cursor.toordinal() + 7)
    return out


# ---------------------------------------------------------------------------
# The trend, and the states that are not a trend.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrendView:
    """The trend, or the named reason there is none. Both are first-class results."""

    state: str                       # see STATE_RU
    headline_ru: str
    detail_ru: str
    gates: list[dict[str, Any]]
    lessons_have: int
    lessons_needed: int
    trend: dict[str, Any] | None = None
    series: list[float] = field(default_factory=list)
    series_dates: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.state == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "headline_ru": self.headline_ru,
            "detail_ru": self.detail_ru, "gates": self.gates,
            "lessons_have": self.lessons_have, "lessons_needed": self.lessons_needed,
            "trend": self.trend, "series": [round(v, 1) for v in self.series],
            "series_dates": self.series_dates,
        }


REQUIRED_LESSONS = trend_mod.MIN_HISTORY_LESSONS + 1


def trend_for(sessions: list[SessionSeat], *, attested: dict[str, Any] | None,
              attestation_covers_all: bool, role: str = "pupil",
              undated_lessons: int = 0) -> TrendView:
    """Build a `TrendView` for one place. Delegates the statistics to `metrics/trend.py`.

    The gates are evaluated in full and reported in full even after one has failed, because
    a psychologist looking at «динамики нет» needs to know whether the answer is «подождите
    четыре урока» or «подпишите план рассадки» — two entirely different actions, and the
    page must not make them look like the same shrug.
    """
    if role != "pupil":
        return TrendView(
            "not_applicable_adult", "динамика по взрослому не строится",
            ADULT_NO_INDEX_RU + " Накопленный по неделям показатель по взрослому был бы "
            "поверхностью для сравнения сотрудников, построенной на измерении, которое "
            "такого сравнения не выдерживает.",
            [], 0, REQUIRED_LESSONS)

    usable = [s for s in sessions if s.activity.available and s.activity.index is not None]
    stable_place = all(s.place_stability == "stable" for s in sessions) if sessions else False

    gates = [
        _gate("place_matched_in_every_lesson", stable_place,
              measured=f"{sum(1 for s in sessions if s.place_stability == 'stable')} из "
                       f"{len(sessions)} {_lessons_genitive(len(sessions))}",
              required="во всех уроках место опознано однозначно",
              detail_ru="Место сопоставляется между уроками по геометрии. Если в каком-то "
                        "уроке два места оказались одинаково близко, история за этот урок "
                        "не присоединяется — иначе к ней приписался бы соседний ребёнок."),
        _gate("identity_attested_for_the_whole_period",
              bool(attested) and attestation_covers_all,
              measured=(f"{attested['full_name'] or attested['external_id']} "
                        f"(утвердил: {attested['attested_by']}, {attested['attested_at']})"
                        if attested else "плана рассадки нет"),
              required="подписанный план рассадки, действующий на все уроки периода",
              detail_ru="Единица учёта — место, а не ребёнок. Динамика — это утверждение "
                        "о ребёнке, поэтому она строится только там, где человек письменно "
                        "подтвердил, кто на этом месте сидит, и подтверждение "
                        "действовало на протяжении всего периода."),
        _gate("enough_lessons", len(usable) >= REQUIRED_LESSONS,
              measured=f"{len(usable)} {lessons_word(len(usable))} с посчитанным индексом",
              required=f"не менее {REQUIRED_LESSONS}",
              detail_ru=f"metrics/trend.py требует {trend_mod.MIN_HISTORY_LESSONS} уроков "
                        f"собственной нормы плюс текущий. По медиане из трёх уроков "
                        f"«норма» — это не норма, а три числа."),
        _gate("index_available_in_every_lesson",
              bool(sessions) and len(usable) == len(sessions),
              measured=f"{len(usable)} из {len(sessions)}",
              required="во всех уроках хватило наблюдений для индекса",
              detail_ru="Урок, в котором место было видно меньше половины времени, в ряд "
                        "не попадает: индекс по нему — уверенное число об ученике, "
                        "которого мы в основном не видели."),
    ]

    if not sessions:
        # Two different emptinesses, and they need different actions from a human. «Уроков
        # нет» is solved by recording one; «уроки есть, но без даты» is solved by fixing
        # the camera's clock overlay or supplying the date, and a page that says the first
        # when the second is true sends the reader to look for a missing recording that is
        # actually sitting in the store.
        if undated_lessons:
            return TrendView(
                "no_dated_lessons",
                f"есть {undated_lessons} {lessons_word(undated_lessons)} без даты",
                "Запись(и) загружены, но время съёмки прочитать не удалось "
                "(clock_source = unknown), поэтому их нельзя расположить по порядку и "
                "нельзя отнести к неделе. Счётчики по уроку доступны в разборе самого "
                "урока; в кабинете они не накапливаются. Исправляется не ожиданием, а "
                "датой: перезапустите разбор с известным временем начала записи.",
                gates, 0, REQUIRED_LESSONS)
        return TrendView("no_data", "уроков ещё нет",
                         "Для этого места не загружено ни одного урока с датой.",
                         gates, 0, REQUIRED_LESSONS)

    if not (bool(attested) and attestation_covers_all):
        return TrendView(
            "identity_not_established",
            "динамика не строится: место не подписано",
            "Накоплено уроков: " + str(len(sessions)) + ". Счётчики ниже относятся к "
            "МЕСТУ и верны сами по себе. Динамика — это утверждение о конкретном "
            "ребёнке, и она появится только после подписанного плана рассадки: без него "
            "снижение показателей может означать, что на этом месте с ноября сидит другой "
            "ученик, и отличить одно от другого может только человек.",
            gates, len(usable), REQUIRED_LESSONS,
            series=[s.activity.index for s in usable if s.activity.index is not None],
            series_dates=[s.date_local or "" for s in usable])

    if not stable_place:
        return TrendView(
            "place_unstable", "динамика не строится: место опознано не во всех уроках",
            "В части уроков это место не удалось однозначно сопоставить с прежним "
            "(перестановка мебели или сдвиг камеры). Сравнивать такой ряд — значит "
            "сравнивать разные места.",
            gates, len(usable), REQUIRED_LESSONS)

    if len(usable) < REQUIRED_LESSONS:
        return TrendView(
            "insufficient_lessons",
            f"данных пока мало: {len(usable)} "
            f"{lessons_word(len(usable))} из {REQUIRED_LESSONS}",
            f"Это не ошибка и не пустой отчёт: при {len(usable)} "
            f"{_lessons_prepositional(len(usable))} собственная норма ученика ещё не "
            f"определена, "
            f"и любое «стало хуже» было бы сравнением с шумом. Счётчики и недельная "
            f"таблица ниже уже осмысленны — не хватает именно ДИНАМИКИ. "
            f"Осталось разобрать уроков: {REQUIRED_LESSONS - len(usable)}.",
            gates, len(usable), REQUIRED_LESSONS,
            series=[s.activity.index for s in usable if s.activity.index is not None],
            series_dates=[s.date_local or "" for s in usable])

    history = [s.activity.index for s in usable if s.activity.index is not None]
    computed = trend_mod.trend(history, identity_stable=True)
    return TrendView(
        state="available" if computed.available else "refused_by_trend",
        headline_ru=trend_mod.RU_DIRECTION[computed.direction],
        detail_ru=computed.reason or (
            "Сравнение ученика с самим собой: медиана его же предыдущих уроков и MAD как "
            "мера разброса. Ни с классом, ни с другим ребёнком этот показатель не "
            "сравнивается."),
        gates=gates, lessons_have=len(usable), lessons_needed=REQUIRED_LESSONS,
        trend=computed.to_dict(), series=history,
        series_dates=[s.date_local or "" for s in usable])


# ---------------------------------------------------------------------------
# The refusal, turned into a purchase order.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Request:
    """One thing to ask the school for, with the number already worked out.

    `trend_gates` says WHICH precondition is missing, which is the right thing for a page
    that has to be trusted. It is not enough for the person holding the page: «место не
    подписано» is a state, and what a psychologist has to do next is send an e-mail asking
    for specific documents and specific recordings. Between those two there was a step the
    reader had to take themselves — how many more lessons, in which room, signed by whom,
    valid from when — and every one of those is derivable from what the store already holds.

    So each request carries three fields and all three are load-bearing:

      * `what_ru` — the thing to ask for, WITH ITS NUMBER. «Ещё 4 урока», never «больше
        уроков»; the number is `REQUIRED_LESSONS` minus what is in the store, which is the
        same arithmetic the gate reports and is deliberately not re-derived by the reader.
      * `why_ru` — what it unlocks and why nothing else will do instead. A request whose
        reason is missing gets negotiated down: «а можно по трём урокам?» has an answer, and
        it is here rather than in a docstring nobody in the school will read.
      * `how_ru` — the concrete step, including the exact command and, where a human must
        supply something, the list of fields they must supply.

    Refusals in this project are required to be ACTIONABLE (rule 8). This type is that rule
    given a shape, so that a page cannot satisfy it with a sentence that merely sounds
    helpful.
    """

    key: str
    what_ru: str
    why_ru: str
    how_ru: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "what_ru": self.what_ru, "why_ru": self.why_ru,
                "how_ru": self.how_ru}


NOTHING_TO_REQUEST_RU = (
    "Всё, что нужно для динамики по этому месту, в кабинете уже есть."
)


def requests_for(history: PlaceHistory, *, room_key: str, class_key: str,
                 class_rooms: list[dict[str, Any]] | None = None,
                 threshold_sets: int = 1,
                 undated_lessons: int = 0) -> list[Request]:
    """Exactly what to ask the school for, in the order it has to arrive.

    Order is dependency order, not importance order: more recordings are useless without a
    plan naming who sat there, and a plan is useless if the recordings were made in two
    different rooms. A list sorted by «важность» would invite the reader to start with item
    one and stop.
    """
    if str(history.place["role"]) != "pupil":
        return []

    out: list[Request] = []
    usable = [s for s in history.sessions
              if s.activity.available and s.activity.index is not None]
    missing = REQUIRED_LESSONS - len(usable)
    place_id = int(history.place["place_id"])
    label = str(history.place["label_ru"])
    class_flag = f" --class {class_key}" if class_key not in ("-", "") else ""

    if missing > 0:
        out.append(Request(
            "more_lessons_same_room",
            f"Ещё {missing} разобранн{'ый' if missing == 1 else 'ых'} "
            f"{lessons_word(missing)} ЭТОГО ЖЕ класса в ЭТОЙ ЖЕ комнате "
            f"({room_key}). Сейчас в кабинете {len(usable)} "
            f"{lessons_word(len(usable))} с посчитанным индексом из "
            f"{REQUIRED_LESSONS} необходимых.",
            f"Норма, с которой сравнивается последний урок, — это медиана "
            f"{trend_mod.MIN_HISTORY_LESSONS} собственных предыдущих уроков ЭТОГО ребёнка "
            f"плюс текущий. По медиане из трёх уроков «норма» — это не норма, а три числа, "
            f"и любое «стало хуже» было бы сравнением с шумом. Уроки другой комнаты сюда "
            f"не годятся: там другие места и, скорее всего, другие дети.",
            f"По каждой новой записи: `classvision analyse <файл>.mp4 --room "
            f"<профиль_камеры>.yaml`, затем `classvision cabinet import "
            f"out/<файл>.analysis.json --room-key {room_key}{class_flag}`. Ключ комнаты и "
            f"класса должен совпадать буквально — другой ключ заводит другую историю."))

    if not history.attested:
        out.append(Request(
            "signed_seating_plan",
            # «место 3 (место №12)» — two numbers, both called «место», is the collision this
            # project spends its docstrings on, arriving in a sentence a school is meant to
            # act on. The display label is the one a reader sees on every page; the row id is
            # a key the command needs, and it says so rather than sitting there as a second
            # position number.
            f"Подписанный план рассадки на «{label}» в комнате {room_key}: кто сидит на "
            f"этом месте, с какой и по какую дату. (Внутренний номер этого места в базе "
            f"кабинета — {place_id}; он нужен только для команды ниже и не является "
            f"номером места в комнате.)",
            "Единица учёта здесь — МЕСТО, а не ребёнок, и динамика — это утверждение о "
            "ребёнке. Без письменного подтверждения снижение показателя может означать, "
            "что с какого-то дня на этом месте сидит другой ученик, и отличить одно от "
            "другого может только человек с планом рассадки. Ни лицо, ни трекер имени "
            "здесь не создают: распознавание на этих камерах измерено (лучшее совпадение "
            "0,30 при отрыве 0,10 от следующего кандидата) и такого утверждения не "
            "выдерживает.",
            "От школы нужны шесть вещей: (1) идентификатор ученика в её собственном "
            "реестре, (2) ФИО, (3) ФИО и должность того, кто это подтверждает, (4) дата "
            "подтверждения, (5) срок действия — обязательно С ДАТОЙ ОКОНЧАНИЯ, иначе план "
            "продолжит называть прошлогодних детей после пересадки, (6) ссылка на решение "
            "школы, разрешающее пофамильное накопление. Затем: `classvision cabinet attest "
            f"--place-id {place_id} --external-id … --full-name … --by … --at ГГГГ-ММ-ДД "
            f"--valid-from ГГГГ-ММ-ДД --valid-to ГГГГ-ММ-ДД --decision …`"))
    elif not history.attestation_covers_all:
        dates = [s.date_local for s in history.sessions if s.date_local]
        out.append(Request(
            "seating_plan_for_the_whole_period",
            f"План рассадки на {label}, действующий на ВСЕ уроки периода "
            f"({dates[0] if dates else '—'} … {dates[-1] if dates else '—'}). Сейчас "
            f"подтверждение покрывает только часть периода.",
            "Ряд, часть которого подписана одним ребёнком, а часть не подписана никем, — "
            "это ряд из двух разных утверждений. Сравнивать его с самим собой нельзя.",
            "Либо продлить действующее подтверждение (`--valid-from` / `--valid-to`), либо "
            "добавить второе на оставшиеся даты — и тогда динамика будет строиться "
            "отдельно по каждому отрезку, а не поверх пересадки."))

    if undated_lessons:
        out.append(Request(
            "dates_for_undated_recordings",
            f"Время начала для {undated_lessons} "
            f"{'записи' if undated_lessons == 1 else 'записей'} этого места, у "
            f"котор{'ой' if undated_lessons == 1 else 'ых'} часы прочитать не удалось.",
            "Запись без даты нельзя расположить по порядку и нельзя отнести к неделе, "
            "поэтому она не участвует ни в одной накопительной сводке — при том что "
            "наблюдение по ней есть и оно полное.",
            "Исправляется не ожиданием, а датой: узнать у школы время начала записи и "
            "перезапустить разбор с ним, либо настроить наложение часов на камере так, "
            "чтобы `video/clock.py` мог его прочитать."))

    rooms = [r for r in (class_rooms or []) if r.get("room_key") != room_key]
    if rooms:
        listed = ", ".join(str(r["room_key"]) for r in rooms)
        word = "комнаты" if len(rooms) == 1 else "комнат"
        out.append(Request(
            "one_class_one_room",
            f"Подтверждение от школы, какой класс снят под ключом «{class_key}» из "
            f"{word} {listed}, — и разделение ключей, если это не тот же класс.",
            "Сейчас один и тот же ключ класса стоит на записях из разных комнат. Пока это "
            "так, обзор кабинета показывает их рядом, и рядом стоящие числа из двух разных "
            "комнат читаются как две недели одного класса. Здесь они не складываются "
            "нигде, но следующий читатель об этом не знает.",
            "Загружать разборы с явно указанным классом: `classvision cabinet import … "
            "--room-key <камера> --class <класс>`. Переклеить ключи у уже загруженного "
            "урока нельзя — вместе с ними пришлось бы переносить историю мест, — поэтому "
            "такие уроки заводятся заново в чистой базе."))

    if threshold_sets > 1:
        out.append(Request(
            "one_set_of_thresholds",
            f"Пересчёт уроков этого класса одним набором порогов: сейчас их "
            f"{threshold_sets}.",
            "«Ребёнок стал менее активен» имеет смысл, только если обе недели мерили "
            "одинаково. Сравнение уроков, разобранных разными порогами, — это сравнение "
            "настроек, а не детей.",
            "Разобрать все записи класса одними настройками (`classvision analyse` с теми "
            "же порогами и тем же профилем комнаты) и выбрать эти прогоны расчётными: "
            "`classvision cabinet select-run <run_id>`. Прежние измерения не удаляются."))

    unavailable = [s for s in history.sessions if not s.activity.available]
    if unavailable and len(history.sessions) > len(usable):
        out.append(Request(
            "recordings_that_show_this_place",
            f"Из {len(history.sessions)} {lessons_word(len(history.sessions))} этого места "
            f"в ряд не попал{'о' if len(unavailable) != 1 else ''} "
            f"{len(unavailable)}: место было видно слишком малую часть времени.",
            "Урок, в котором место видно меньше половины разобранных кадров, даёт индекс, "
            "который является уверенным числом об ученике, которого мы в основном не "
            "видели. Такой урок в ряд не берётся — но и не исчезает: его счётчики есть в "
            "таблице занятий ниже.",
            "Это чинится не данными, а камерой: сместить объектив или переставить парту "
            "так, чтобы это место не заслонялось. Что именно мешало — видно на "
            "проверочном видео: `classvision verify <артефакт> --out out/verify.mp4`."))

    return out


def lessons_word(n: int) -> str:
    """Russian plural for «урок». A page that says «1 уроков» stops being read carefully.

    Not decoration. This surface exists to be read by a psychologist rather than by a
    developer, and a sentence with broken agreement in the very place it is reporting a
    refusal reads as a broken page — which is precisely the impression `TrendView`'s named
    states exist to avoid.
    """
    if 11 <= n % 100 <= 14:
        return "уроков"
    return {1: "урок", 2: "урока", 3: "урока", 4: "урока"}.get(n % 10, "уроков")


def _lessons_genitive(n: int) -> str:
    """The form «урок» takes after «из»: «из 1 урока», «из 5 уроков»."""
    return "урока" if n % 10 == 1 and n % 100 != 11 else "уроков"


def _lessons_prepositional(n: int) -> str:
    """The form «урок» takes after «при»: «при 1 уроке», «при 2 уроках»."""
    return "уроке" if n % 10 == 1 and n % 100 != 11 else "уроках"


def _gate(name: str, passed: bool, *, measured: str, required: str,
          detail_ru: str) -> dict[str, Any]:
    return {"gate": name, "passed": passed, "measured": measured, "required": required,
            "detail_ru": detail_ru}


# ---------------------------------------------------------------------------
# Assembling one place and one class.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlaceHistory:
    """Everything the per-pupil page renders, computed once."""

    place: dict[str, Any]
    attested: dict[str, Any] | None
    attestation_covers_all: bool
    display_name_ru: str
    sessions: list[SessionSeat]
    weeks: list[WeekRow]
    trend_view: TrendView
    # `None` per key means «не накоплено» (no dated session to sum), never «ноль».
    totals: dict[str, int | None]
    coverage_min: float | None
    unplaced_sessions: int
    # Recordings of this place whose clock could not be read. Carried on the object rather
    # than re-queried by every caller, because it is one of the numbers `requests_for` has
    # to put into a sentence, and a second query is a second chance to scope it differently.
    undated_lessons: int = 0
    # Filled by `class_view`, which is the only scope that knows about sibling rooms and
    # about how many threshold sets this class was measured with. Empty on a `PlaceHistory`
    # built directly, and empty is honest there: those two requests cannot be derived from
    # one place.
    requests: list[Request] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "place": self.place, "attested": self.attested,
            "attestation_covers_all": self.attestation_covers_all,
            "display_name_ru": self.display_name_ru,
            "sessions": [s.to_dict() for s in self.sessions],
            "weeks": [w.to_dict() for w in self.weeks],
            "trend": self.trend_view.to_dict(),
            "totals": self.totals, "coverage_min": self.coverage_min,
            "unplaced_sessions": self.unplaced_sessions,
            "undated_lessons": self.undated_lessons,
            "requests": [r.to_dict() for r in self.requests],
        }


def place_history(connection: sqlite3.Connection, place: dict[str, Any],
                  span: list[tuple[int, int]] | None = None) -> PlaceHistory:
    sessions = sessions_for_place(connection, int(place["place_id"]))
    dates = [s.date_local for s in sessions if s.date_local]

    attestations = store.attestations_for(connection, int(place["place_id"]))
    in_force = [store.attestation_in_force(connection, int(place["place_id"]), d)
                for d in dates]
    covers_all = bool(dates) and all(a is not None for a in in_force) and len(
        {a["external_id"] for a in in_force if a}) == 1
    attested = in_force[-1] if in_force and in_force[-1] else (
        attestations[-1] if attestations else None)

    name = place["label_ru"]
    if attested and covers_all:
        name = attested["full_name"] or attested["external_id"]
    elif attested:
        # An attestation that does not cover the whole period names the child on the page
        # but must not be allowed to imply the history is theirs throughout.
        name = f"{attested['full_name'] or attested['external_id']} ({place['label_ru']})"

    role = str(place["role"])
    # The adult carries no accumulated counters at all. Not «поднимал руку 60» — for a
    # person who points at a board and writes on it, that number is a measurement of
    # teaching described in a pupil's vocabulary, and a week table of it is a staff
    # comparison surface. See `ADULT_NO_INDEX_RU` and INTEGRATION.md §7.
    # `None`, not `0`, when there is nothing to sum. A place whose only recordings are
    # undated has been observed — sometimes for a whole hour — and simply cannot be placed
    # on a calendar, so its accumulated counter is UNKNOWN. Summing an empty list gave 0 and
    # the class page printed «поднимал руку — 0» for a pupil the same run had watched raise
    # his hand twice, one line below a badge saying the lesson was not accumulated. Rule 3
    # of this project, and `WeekRow` already had it right for empty weeks.
    if role != "pupil":
        totals: dict[str, int | None] = {}
    elif not sessions:
        totals = {key: None for key, _ in COUNTER_KEYS}
    else:
        totals = {key: sum(s.counts.get(key, 0) for s in sessions)
                  for key, _ in COUNTER_KEYS}
    undated = int(connection.execute(
        "SELECT COUNT(DISTINCT s.lesson_id) AS n FROM seat_lessons s "
        "JOIN lessons l ON l.lesson_id = s.lesson_id "
        "WHERE s.place_id = ? AND l.started_at IS NULL AND s.run_id = l.selected_run_id",
        (int(place["place_id"]),)).fetchone()["n"] or 0)

    return PlaceHistory(
        place=place, attested=attested, attestation_covers_all=covers_all,
        display_name_ru=name, sessions=sessions,
        weeks=[] if role != "pupil" else weeks_for(sessions, span),
        trend_view=trend_for(sessions, attested=attested,
                             attestation_covers_all=covers_all, role=role,
                             undated_lessons=undated),
        totals=totals,
        coverage_min=min((s.coverage for s in sessions), default=None),
        unplaced_sessions=sum(1 for s in sessions if s.place_stability != "stable"),
        undated_lessons=undated)


@dataclass(slots=True)
class ClassView:
    """One (room, class): the overview page's whole content."""

    room_key: str
    class_key: str
    lessons: list[dict[str, Any]]
    histories: list[PlaceHistory]
    span: list[tuple[int, int]]
    caveats_ru: list[str]
    unmeasured: list[dict[str, Any]]
    store_summary: dict[str, Any]
    extra_runs: list[dict[str, Any]]
    unplaced_seat_lessons: list[dict[str, Any]]
    chance_note: dict[str, Any]
    # Scoped to THIS class, and separate from `store_summary` on purpose. The overview used
    # `store_summary["threshold_sets"]`, which counts distinct threshold sets in the WHOLE
    # cabinet, to decide whether to print «уроки этого класса измерены разными наборами
    # порогов» — so a second class re-analysed once would have hung that warning on every
    # other class's page. The same scoping mistake `store.unmeasured()` already carries a
    # paragraph about.
    runs_in_class: int = 0
    threshold_sets_in_class: int = 1
    adults_without_place: list[dict[str, Any]] = field(default_factory=list)
    # EVERY room this class KEY was filed under, this one included. More than one entry is
    # the state this whole surface has to shout about: the pages beneath a class key look
    # like one class's term, so a key spanning two rooms puts two different groups of
    # children under one heading, and the arithmetic joining them is one glance away.
    # Nothing here ever joins them; `class_rooms` exists so that the page can say so.
    class_rooms: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pupil_histories(self) -> list[PlaceHistory]:
        return [h for h in self.histories if h.place["role"] == "pupil"]

    @property
    def other_rooms(self) -> list[dict[str, Any]]:
        return [r for r in self.class_rooms if str(r.get("room_key")) != self.room_key]

    @property
    def class_stated(self) -> bool:
        """Whether a human actually typed this class key.

        `store._class_key` falls back to `"-"` with source `not_stated`, so `"-"` is the
        ABSENCE of a class rather than the name of one — and two lessons sharing it have
        nothing in common but that absence. Rendering it as a class name invents an entity,
        which is the same mistake as rendering an unmeasured counter as a zero.
        """
        return self.class_key not in ("-", "", None)

    @property
    def class_display_ru(self) -> str:
        return f"класс {self.class_key}" if self.class_stated else "класс не указан"

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_key": self.room_key, "class_key": self.class_key,
            "lessons": self.lessons,
            "histories": [h.to_dict() for h in self.histories],
            "weeks": [f"{y}-W{w:02d}" for y, w in self.span],
            "caveats_ru": self.caveats_ru, "unmeasured": self.unmeasured,
            "store": self.store_summary, "extra_runs": self.extra_runs,
            "unplaced_seat_lessons": self.unplaced_seat_lessons,
            "expected_by_chance": self.chance_note,
            "runs_in_class": self.runs_in_class,
            "threshold_sets_in_class": self.threshold_sets_in_class,
            "adults_without_place": self.adults_without_place,
            "class_rooms": self.class_rooms,
            "class_stated": self.class_stated,
        }


def class_view(connection: sqlite3.Connection, *, room_key: str | None = None,
               class_key: str | None = None) -> ClassView:
    """Everything one class's pages need, in one pass over the store."""
    lesson_rows = store.lessons(connection, room_key=room_key, class_key=class_key)
    place_rows = store.places(connection, room_key=room_key, class_key=class_key)
    span = span_of(connection, room_key=room_key, class_key=class_key)
    histories = [place_history(connection, place, span) for place in place_rows]

    lesson_ids = {int(row["lesson_id"]) for row in lesson_rows}
    extra_runs = [
        dict(row) for row in connection.execute(
            "SELECT r.run_id, r.lesson_id, r.thresholds_sha, r.imported_at, "
            "       r.artefact_path, l.selected_run_id, l.date_local "
            "FROM runs r JOIN lessons l ON l.lesson_id = r.lesson_id "
            "WHERE l.selected_run_id IS NOT r.run_id")
        if int(row["lesson_id"]) in lesson_ids]

    unplaced = [dict(row) for row in connection.execute(
        "SELECT s.seat_label, s.place_match, s.place_match_reason_ru, l.date_local, "
        "       l.lesson_id FROM seat_lessons s JOIN lessons l ON l.lesson_id = s.lesson_id "
        "WHERE s.place_id IS NULL AND s.run_id = l.selected_run_id")
        if int(row["lesson_id"]) in lesson_ids]

    scope_where, scope_args = store._scope(room_key, class_key, prefix="l.")
    counted = connection.execute(
        f"SELECT COUNT(*) AS runs, COUNT(DISTINCT r.thresholds_sha) AS sets FROM runs r "
        f"JOIN lessons l ON l.lesson_id = r.lesson_id{scope_where}", scope_args).fetchone()

    pupils = sum(1 for p in place_rows if p["role"] == "pupil")
    room = room_key or (place_rows[0]["room_key"] if place_rows else
                        (lesson_rows[0]["room_key"] if lesson_rows else "—"))
    klass = class_key or (place_rows[0]["class_key"] if place_rows else
                          (lesson_rows[0]["class_key"] if lesson_rows else "—"))

    # Which rooms this CLASS KEY appears in — asked of the whole store, deliberately not of
    # the current scope. The question is «делит ли этот ключ ещё какая-то комната», and a
    # scoped query can only ever answer «нет».
    class_rooms = [dict(row) for row in connection.execute(
        """SELECT room_key, class_key, COUNT(*) AS lessons,
                  MIN(date_local) AS first_date, MAX(date_local) AS last_date
           FROM lessons WHERE class_key = ? GROUP BY room_key ORDER BY room_key""",
        (klass,))]
    for entry in class_rooms:
        entry["pupil_places"] = int(connection.execute(
            "SELECT COUNT(*) FROM places WHERE room_key = ? AND class_key = ? "
            "AND role = 'pupil'", (entry["room_key"], klass)).fetchone()[0] or 0)

    view = ClassView(
        room_key=room, class_key=klass,
        lessons=lesson_rows, histories=histories, span=span,
        caveats_ru=store.caveats(connection, room_key=room_key, class_key=class_key),
        unmeasured=store.unmeasured(connection, room_key=room_key, class_key=class_key),
        store_summary=store.summary(connection), extra_runs=extra_runs,
        unplaced_seat_lessons=unplaced,
        chance_note=trend_mod.expected_by_chance(max(pupils, 1)),
        runs_in_class=int(counted["runs"] or 0),
        threshold_sets_in_class=int(counted["sets"] or 0),
        adults_without_place=store.adults_without_place(
            connection, room_key=room_key, class_key=class_key),
        class_rooms=class_rooms)

    # The request list needs both halves: what one place is missing, and what the class as a
    # whole is missing. Assembled here because this is the only scope that has both.
    for history in view.histories:
        history.requests = requests_for(
            history, room_key=view.room_key, class_key=view.class_key,
            class_rooms=view.class_rooms,
            threshold_sets=view.threshold_sets_in_class,
            undated_lessons=history.undated_lessons)
    return view


def state_label_ru(state: str) -> str:
    """The RU label of a `PupilState` value, for rendering stored histograms."""
    try:
        return RU_LABELS[PupilState(state)]
    except ValueError:
        return state

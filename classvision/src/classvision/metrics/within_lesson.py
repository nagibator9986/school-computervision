"""«Динамика снижения активности» — the part of it this recording can actually answer.

--------------------------------------------------------------------------------
**WHY THIS FILE EXISTS AND WHY IT IS NOT `trend.py`.**

The school asked to see a child's «динамика снижения активности». Across lessons that
question needs an identity: the seat that is compared in October must hold the child it
held in September. `metrics/trend.py` refuses to answer without one, and that refusal is
correct and stays — the two lessons this package has been measured on are two different
rooms with two different numbers of children and no attested seating plan, so a
week-over-week per-child comparison here would be arithmetic performed on strangers.

Inside ONE lesson the precondition is free. A place is a place for the whole 50–58
minutes; the camera does not move; the seat map is irrelevant because nobody is being
named. «Было ли на этом месте в последней трети урока меньше наблюдаемой активности, чем
в первой» is answerable today, on both existing recordings, with nothing added.

**Everything here compares a seat with ITSELF.** Never with another seat, never with a
class mean, never with another lesson. The comparison is between one place's own segments,
which is the same principle `states.Baselines` arrives at by geometry and `trend.py` by
policy.

--------------------------------------------------------------------------------
**THE INDEX IS NOT RECOMPUTED HERE. `metrics/activity.py` IS CALLED.**

A segment is a `SeatLedger` fed only that segment's observations, and `activity()` is
applied to it unchanged — same weights, same `MIN_COVERAGE`, same refusals. There is no
second formula in this package and no place to add one. When a segment's coverage is too
low the index is REFUSED for that segment, and the refusal travels into the artefact as a
refusal with its reason, not as a zero and not as a missing key. A segment we could not
see is the single most important thing this module has to say about a lesson, and it is
the thing a plotted line would quietly interpolate away.

**The per-segment ledger shares the whole-lesson baselines, frozen.** `states.Baselines`
settles on a place's first twenty consistent observations and then never changes; if each
segment learned its own baseline, segment 4 would be measured against how the child sat
during segment 4, and a child who spent the last ten minutes slumped would be scored as
sitting normally — the decline would erase itself by construction. So segments are a
partition of the SAME classification, not four independent analyses. `_frozen()` copies
the settled baseline and blocks further learning; a place that never settled at all
propagates that refusal into every one of its segments, via `activity()`'s own «базовая
поза не установлена».

--------------------------------------------------------------------------------
**THE SEGMENTATION, AND WHY IT IS EQUAL FRACTIONS OF THE DETECTED WINDOW.**

Measured on both recordings (2026-08-16), segmenting the detected lesson window — never
the file, which on camera 01 opens with 110 seconds of empty room:

| scheme | segment length | segments refused for coverage | seats left with < 4 usable |
|---|---|---|---|
| thirds | 16.7 / 19.4 min | 0/24 cam01, 2/15 D14 | **8 of 13** |
| quarters | 12.5 / 14.5 min | 0/32, 4/20 | 3 of 13 |
| fifths | 10.0 / 11.6 min | 0/40, 4/25 | 1 of 13 (D14 place 3) |
| **sixths** | **8.3 / 9.7 min** | **0/48, 4/30** | **0 of 13** |
| eighths | 6.3 / 7.3 min | 1/64, 6/40 | 0 of 13, but one segment at coverage 0.02 |

**CHOSEN: six equal segments of the detected lesson window.** It is the coarsest
segmentation at which no place in either real lesson falls below the four usable segments
the direction test needs (see `MIN_USABLE_SEGMENTS`), and the finest at which no segment of
either lesson degenerates: at eighths, place 3 of D14 has a segment observed in 2 % of its
frames. Six segments of a 50-minute lesson are 8.3 minutes each, which is 25 times the
longest minimum episode duration the ledger enforces (`head_down_min_seconds` = 20 s), so
what a boundary can do to an episode is bounded and small — that bound is computed per run
and used as the magnitude floor below.

**Equal fractions, not fixed ten-minute bins, and the reason is measured rather than
aesthetic.** A fixed grid leaves a ragged final bin, and the final bin is exactly the part
of the lesson this whole module is about. On camera 01 the detected window is 3 002.8 s, so
a 600 s grid leaves a sixth bin of **2.8 seconds — six observations** — and that bin
produced an activity index of **40.0** with a coverage of **1.00** beside it. A confident
number with a clean coverage figure, computed on three seconds of a fifty-minute lesson.
Equal fractions cannot produce that, because every segment is the same size by
construction.

Equal fractions do a second, less obvious job. The index's participation term counts
EPISODES against a fixed normaliser (`EVENTS_FOR_FULL_CREDIT` = 3 for a whole lesson), so
its value depends on how long the window is. Segments of equal length have equal exposure
and their event terms are comparable to each other; segments of unequal length are not.
Which leads directly to the loudest warning in this file:

**A SEGMENT INDEX AND THE WHOLE-LESSON INDEX ARE NOT THE SAME QUANTITY AND MUST NEVER BE
COMPARED.** Measured, camera 01, place 3: the whole-lesson index is **97.7** and its six
segment indices are 88, 68, 66, 67, 69, — . Nothing declined; the child raised a hand twice
in fifty minutes, which is `EVENTS_FOR_FULL_CREDIT` credit over the lesson and at most one
event's worth in any one segment. The segment series is internally valid and comparable
across the lesson; against the lesson total it is nonsense. Every surface that draws both
has to say so, and `report/charts.py` does.

--------------------------------------------------------------------------------
**THE DECLINE STATISTIC: Mann–Kendall for the direction, Theil–Sen for the size.**

Both are rank/median based, both are defined for the four-to-six points we have, and
neither assumes a distribution we have no data to check. The pairing is the standard one
for a short monotone-trend question and it buys three things this domain needs:

* the direction is decided by the ORDER of the segments, so it cannot be produced by one
  extreme segment — which matters because the loudest single-segment moves in this data are
  the event term flipping between 0 and 1;
* the magnitude is a median of pairwise slopes, so one bad segment moves it by little;
* the null distribution of Mann–Kendall's S is exact and enumerable at these n, so «is this
  ordering more than chance» is computed rather than asserted. `_kendall_p` enumerates it.

**`MIN_USABLE_SEGMENTS = 4` is derived, not chosen.** With three points the best possible
evidence — a perfectly monotone series — has an exact two-sided p of **1/3**, so no
three-segment place can support a statement about direction at any honest threshold. The
same n = 3 is degenerate for the fit's own scatter: measured over 5 000 simulated draws,
the median absolute deviation of the Theil–Sen residuals is **identically zero in 100 % of
three-point series** (it is zero by construction — two of the three residuals are always
0), so a three-point series also cannot report how noisy it was. Four points give a minimum
two-sided p of 0.083 and a residual scatter that exists.

`DIRECTION_ALPHA = 0.10` is CHOSEN, and reported rather than hidden: `expected_by_chance()`
prints how many places in a class of this size would show a direction from noise alone —
0.8 places in a class of eight — because a psychologist shown one place needs to know
whether one is more than nothing. It is not 0.05, and the reason is a bias rather than a
convention: at four usable segments the smallest attainable p is 0.083, so an alpha of 0.05
would make every place that lost two segments to occlusion structurally unable to be
described. Losing segments is the normal case on camera D14 — four of thirty at sixths —
and those are the worst-seen places in the room, so a 0.05 threshold would silently
restrict the whole statistic to the children the camera happens to see well.

--------------------------------------------------------------------------------
**THE CONFOUND THAT MATTERS MOST HERE, AND WHAT WAS ACTUALLY MEASURED ABOUT IT.**

The expected failure is that coverage falls as a lesson goes on — people move, occlude one
another, the room warms up — so an index that falls is partly a picture that got worse.
Two things were done about it, and the second is the one that turned out to carry the load.

*First, the bound.* `_visibility_bound` computes, in index points, the largest fall the
observed coverage change alone could account for. The share terms get the ADVERSARIAL
treatment: if the place was seen a fraction `r` as well at the end as at the start, the
observations we lost could all have been the good ones, so each share may understate the
truth by `(1 − s)(1 − r)` and the index by `Σ w (1 − s)(1 − r)`. The participation term gets
a PROPORTIONAL treatment instead — events are assumed to occur at the same rate in the
unseen time as in the seen — and the difference is deliberate: the adversarial bound on a
rare countable thing is vacuous, since any unseen minute could in principle hold a
hand-raise, and a bound that says «we know nothing» is not a usable statement. Instead every
segment carries `events_are_a_lower_bound` whenever its coverage is below 1, and the floor
below discounts changes that rest on a single event.

*Second, and this is the finding.* On the two lessons we hold, the confound is mostly
handled UPSTREAM, by `activity.MIN_COVERAGE` refusing the badly-seen segments outright. Once
those are gone, the surviving segments' coverage is nearly flat and the bound is small:
across all thirteen places of both lessons at sixths, `visibility_bound_index_points` runs
**0.00 to 1.45**, against index changes of up to 25. The confound did not need correcting
because the segments in which it bites were refused rather than corrected — which is the
whole argument for refusing.

*And the premise itself did not survive measurement.* Coverage does NOT generally fall over
these lessons. On camera 01 it RISES at every one of the eight places (pupils are still
arriving in the first segment: place 3 runs 0.78 → 0.95, place 9 runs 0.93 → 1.00). On
camera D14 it is erratic rather than declining — two places collapse in the final segment
(place 6: 0.63, 0.91, 0.89, 0.71, 0.11) while three stay flat. So «падение видимости к концу
урока» is a real risk that must be bounded, and on this footage it is not a general fact.
Saying so is the point: the bound is machinery for a thing we did not find, kept because
the next camera may find it.

--------------------------------------------------------------------------------
**NO FLAG, NO RANKING, NO «NEEDS ATTENTION».**

There is no boolean here that says a place is a problem, no sort order that implies one, and
no colour anywhere downstream that means bad. `Change` reports a direction, a size in the
place's own index points, the p that direction was established at, how much of the change
was the event count rather than posture, and what would have to be true for the number to be
believed. Whether any of that matters about a particular child is a judgement that belongs
to a person who knows the child.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from classvision.ledger import SeatLedger
from classvision.metrics.activity import (
    EVENTS_FOR_FULL_CREDIT,
    WEIGHTS,
    Activity,
    activity,
)
from classvision.metrics.trend import Direction
from classvision.states import Baselines, PupilState, Reading, Thresholds

# How many equal parts the detected lesson window is cut into. CHOSEN at 6 — the table in
# the module docstring is the measurement it comes from: the coarsest cut at which neither
# real lesson leaves a place below `MIN_USABLE_SEGMENTS`, and the finest at which no segment
# degenerates.
SEGMENTS = 6

# Below this many segments with an available index, no direction is reported at all.
# DERIVED, not chosen: with three points a perfectly monotone series still has an exact
# two-sided Mann–Kendall p of 1/3, and the Theil–Sen residual scatter is identically zero.
# Three segments cannot carry either half of the statement.
MIN_USABLE_SEGMENTS = 4

# The two-sided Mann–Kendall p at or below which the ordering of the segments is called a
# direction. CHOSEN at 0.10; `expected_by_chance()` reports what that costs for a class of a
# given size. See the module docstring for why it is not 0.05.
DIRECTION_ALPHA = 0.10

# One participation EVENT is worth this many index points, exactly:
# 100 x weight / normaliser. Not a threshold — an arithmetic property of
# `metrics/activity.py`, restated here because the magnitude floor is built from it.
EVENT_QUANTUM_INDEX_POINTS = 100.0 * WEIGHTS["participation"] / EVENTS_FOR_FULL_CREDIT

# The three components of the index that are shares of observations. The fourth,
# `participation`, is a count of episodes and is handled apart from these everywhere in this
# file, because a share and a count do not degrade the same way when observations go
# missing.
SHARE_KEYS: tuple[str, ...] = ("head_up_share", "facing_front_share", "at_place_share")

RU_DIRECTION: dict[Direction, str] = {
    Direction.LOWER: "к концу урока ниже, чем в начале",
    Direction.HIGHER: "к концу урока выше, чем в начале",
    Direction.STEADY: "в пределах собственного разброса за урок",
    Direction.UNKNOWN: "направление не установлено",
}

# Why a direction was not stated. A separate field from `reason` so that a consumer can
# distinguish the cases without parsing Russian, and so the three are never merged: they
# call for three different actions from whoever reads the report.
REFUSED_TOO_FEW = "too_few_usable_segments"
REFUSED_VISIBILITY = "coverage_change_can_explain_it"
REFUSED_INCONSISTENT = "ordering_not_beyond_chance"


@dataclass(frozen=True, slots=True)
class Segment:
    """One equal slice of the lesson at one place, with its index and its uncertainty.

    `coverage` and `effective_coverage` are two different numbers and both are reported.
    `coverage` is the ledger's own — was anybody seen at this place — and it is what
    `activity()` gates on. `effective_coverage` additionally drops the observations whose
    pose did not resolve, because an observation classified `UNKNOWN` contributes nothing
    to any share and is therefore exactly as invisible to the index as an absent one. The
    visibility bound is computed from the second; the refusal comes from the first, so that
    the gate in this module and the gate in `activity.py` cannot drift apart.
    """

    ordinal: int                    # 1-based, in lesson order
    start_seconds: float
    end_seconds: float
    start_minute: float             # minutes from the start of the lesson window
    end_minute: float
    # Two counts that are USUALLY equal and must not be assumed to be. `analysed_frames` is
    # how many analysed frames of the recording fall in this segment; `coverage_denominator`
    # is `observations + absent_observations`, which is what `SeatLedger.coverage` divides
    # by. They differ when two people were assigned to this place in one frame — the ledger
    # then records two observations and no absence. On camera 01 place 7 that is 1 067
    # against 1 001 in segment 4, i.e. 66 frames in which somebody stood close enough to
    # this desk to be counted at it. Reporting one number would either hide that or make the
    # coverage in this block disagree with the coverage in `activity`.
    analysed_frames: int
    coverage_denominator: int
    observations: int
    absent_observations: int
    measured_observations: int      # observations whose state is not UNKNOWN
    coverage: float
    effective_coverage: float
    events: int                     # hand raises + stands + board visits, as EPISODES
    events_are_a_lower_bound: bool
    activity: Activity

    @property
    def midpoint_minute(self) -> float:
        return (self.start_minute + self.end_minute) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "start_minute": round(self.start_minute, 2),
            "end_minute": round(self.end_minute, 2),
            "analysed_frames": self.analysed_frames,
            "coverage_denominator": self.coverage_denominator,
            "observations": self.observations,
            "absent_observations": self.absent_observations,
            "measured_observations": self.measured_observations,
            "coverage": round(self.coverage, 3),
            "effective_coverage": round(self.effective_coverage, 3),
            "events": self.events,
            # True whenever the place was not seen in every frame of this segment. An
            # episode can only be counted where somebody was looking, so a count taken at
            # 70 % coverage is a floor and never a total. Stated per segment rather than
            # once at the bottom of the page, because it is the segments with the lowest
            # coverage whose counts a reader is most likely to compare.
            "events_are_a_lower_bound": self.events_are_a_lower_bound,
            "activity": self.activity.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Change:
    """The within-lesson decline statistic for one place, or a refusal that says what is
    missing.

    Every field that is a number carries the units it is in, because the two natural units
    here — index points across the covered span, and index points per ten minutes — differ
    by a factor that depends on how many segments survived, and a reader who takes one for
    the other is out by that factor.
    """

    available: bool
    direction: Direction
    reason: str
    refused_because: str | None = None

    # The size of the change, in this place's OWN index points, over the span the usable
    # segments actually cover. NOT over the lesson: when the last segment was refused, the
    # statistic says nothing about the last ten minutes and `span_minutes` is how much of
    # the lesson it does speak for.
    delta_index_points: float | None = None
    slope_index_points_per_10_minutes: float | None = None
    # THREE different durations, kept apart because two of them were one field until the
    # generated Russian said «описывает 39 минут урока из 58» beside a
    # `covers_share_of_lesson` of 0.83, which is 48 minutes. Same block, same lesson, two
    # answers to «сколько урока это описывает» — the exact name-collision failure
    # `MEASUREMENTS.md` §8 records at the level of a whole schema.
    #
    #   `span_minutes`     — midpoint of the first usable segment to midpoint of the last.
    #                        This is the base `delta = slope x span` is computed on and it
    #                        is a property of the FIT, not of the coverage.
    #   `covered_minutes`  — the total length of the usable segments themselves. This is
    #                        how much of the lesson the statement actually rests on, and it
    #                        excludes any refused segment in the middle.
    #   `covers_share_of_lesson` — `covered_minutes` over the whole detected window.
    span_minutes: float | None = None
    covered_minutes: float | None = None
    covers_share_of_lesson: float | None = None

    usable_segments: int = 0
    refused_segments: int = 0
    refused_ordinals: tuple[int, ...] = ()
    kendall_s: int | None = None
    kendall_p: float | None = None

    # The two floors the size had to clear, and which one bound it.
    floor_index_points: float | None = None
    floor_from: str | None = None
    boundary_floor_index_points: float | None = None
    event_floor_index_points: float | None = None
    visibility_bound_index_points: float | None = None

    # Visibility, in the terms the bound was computed from.
    effective_coverage_first: float | None = None
    effective_coverage_last: float | None = None
    coverage_retention: float | None = None      # r = last / first, capped at 1

    # Which component moved, in index points, first usable segment -> last usable segment.
    # The index is never shown without its parts, and a CHANGE in the index is not either.
    component_change_index_points: dict[str, float] = field(default_factory=dict)
    # ...and the SAME direction test run on each component separately. This is not
    # decoration and it is not a fishing expedition; it is the only way this block can
    # describe what is on camera 01 place 7, where the pupil's head-up share falls from
    # 0.90 to 0.19 across the lesson — the largest single movement in either recording —
    # while the total index is called «direction not established», because two hand-raise
    # episodes in segments 1 and 3 put two 30-point spikes into an otherwise falling series
    # and destroyed its ordering. Both statements are true of the same six numbers, and a
    # report that printed only the first would be hiding the finding behind its own
    # composite. Multiplicity is real and is stated: four extra tests per place, and
    # `expected_by_chance` reports what that costs for a class of a given size.
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    events_first: int | None = None
    events_last: int | None = None
    driven_by_event_count: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "direction": self.direction.value,
            "direction_ru": RU_DIRECTION[self.direction],
            "reason": self.reason,
            "refused_because": self.refused_because,
            "delta_index_points": _r(self.delta_index_points),
            "slope_index_points_per_10_minutes": _r(self.slope_index_points_per_10_minutes, 2),
            "span_minutes": _r(self.span_minutes),
            "covered_minutes": _r(self.covered_minutes),
            "covers_share_of_lesson": _r(self.covers_share_of_lesson, 3),
            "usable_segments": self.usable_segments,
            "refused_segments": self.refused_segments,
            "refused_ordinals": list(self.refused_ordinals),
            "kendall_s": self.kendall_s,
            "kendall_p": _r(self.kendall_p, 4),
            "floor_index_points": _r(self.floor_index_points, 2),
            "floor_from": self.floor_from,
            "boundary_floor_index_points": _r(self.boundary_floor_index_points, 2),
            "event_floor_index_points": _r(self.event_floor_index_points, 2),
            "visibility_bound_index_points": _r(self.visibility_bound_index_points, 2),
            "effective_coverage_first": _r(self.effective_coverage_first, 3),
            "effective_coverage_last": _r(self.effective_coverage_last, 3),
            "coverage_retention": _r(self.coverage_retention, 3),
            "component_change_index_points": {k: round(v, 1) for k, v in
                                              self.component_change_index_points.items()},
            "components": self.components,
            "events_first": self.events_first,
            "events_last": self.events_last,
            "driven_by_event_count": self.driven_by_event_count,
        }


@dataclass(frozen=True, slots=True)
class WithinLesson:
    """One place's lesson, cut into segments, plus the change across them."""

    segments: tuple[Segment, ...]
    change: Change
    segment_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments_requested": SEGMENTS,
            "segment_seconds": round(self.segment_seconds, 2),
            "segment_minutes": round(self.segment_seconds / 60.0, 2),
            "segments": [s.to_dict() for s in self.segments],
            "change": self.change.to_dict(),
            # Printed into the artefact rather than left to the reader's memory, because the
            # single most available misreading of this block is to compare one of these
            # indices with `metrics.activity.index` above it. See the module docstring.
            "not_comparable_to_ru": (
                "Индексы сегментов сравнимы ТОЛЬКО друг с другом внутри этого урока. "
                "С индексом за весь урок их сравнивать нельзя: событийная составляющая "
                f"нормирована на {EVENTS_FOR_FULL_CREDIT:.0f} события за урок, а сегмент "
                "короче урока, поэтому индекс сегмента систематически ниже. Пример из "
                "записи: место 3 камеры 01 — индекс за урок 97,7, по сегментам 88 / 68 / "
                "66 / 67 / 69; ничего не снижалось, ученик поднял руку дважды за 50 минут."
            ),
        }


def _r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


# ----------------------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------------------


def segment_edges(window_start: float, window_end: float,
                  segments: int = SEGMENTS) -> list[float]:
    """`segments + 1` boundaries over the DETECTED lesson window.

    The window and not the file: on camera 01 the recording opens with 110 seconds of empty
    room, and cutting the file into sixths would spend the first segment measuring
    furniture. `pipeline.lesson_window` already answers this and the answer is in the
    artefact as `lesson.window_seconds`.

    The last edge is nudged past the end so that the final analysed frame, which sits
    exactly on `window_end`, lands in the last segment instead of in none. The same class of
    off-by-one cost this project a whole seat histogram once already (`MEASUREMENTS.md`
    §5b), and it is cheaper to be explicit than to be surprised.
    """
    if segments < 1:
        raise ValueError("segments must be >= 1")
    step = (window_end - window_start) / segments
    edges = [window_start + i * step for i in range(segments)]
    edges.append(math.nextafter(window_end, math.inf))
    return edges


def _frozen(baselines: Baselines) -> Baselines:
    """A copy of a place's settled baseline that cannot learn anything further.

    Segments are fed the same readings the whole-lesson ledger already saw, so a live
    `Baselines` would settle a second time on a doubled window and every state of the
    segment could differ from the state the artefact reports for the same second. Copying
    and blocking makes the segments a strict partition of one classification.

    A place that never settled keeps `settled=False`, so `classify` returns `UNKNOWN` for
    every observation of every segment and `activity()` refuses each one with «базовая поза
    не установлена» — the whole-lesson refusal reproduced per segment rather than papered
    over. `refusal` is set when it was empty only to stop the copy from learning; the
    original's own wording is preserved whenever it had one.
    """
    copy = Baselines(
        settled=baselines.settled,
        seat_position=baselines.seat_position,
        seat_shoulder_y=baselines.seat_shoulder_y,
        upright_head=baselines.upright_head,
        habitual_direction=baselines.habitual_direction,
        scale=baselines.scale,
        refusal=baselines.refusal,
    )
    if not copy.settled and copy.refusal is None:
        copy.refusal = ("базовая поза этого места не установилась за весь урок — "
                        "по сегментам её тем более не с чего строить")
    return copy


def _segment_ledger(seat_id: int, thresholds: Thresholds, sample_interval: float,
                    baselines: Baselines,
                    observations: list[tuple[float, Reading, bool]],
                    absences: list[float], low: float, high: float) -> SeatLedger:
    """The ordinary `SeatLedger`, fed one segment's observations in time order.

    Episodes are therefore re-derived inside the segment: a hand-raise straddling a boundary
    becomes two candidate runs and each must clear the minimum hold on its own, so some
    boundary events are lost. That loss is not repaired — inventing a partial episode would
    break the rule the ledger exists for — it is BOUNDED, in `_boundary_floor`, and the
    bound is used as the smallest change worth reporting.
    """
    ledger = SeatLedger(seat_id=seat_id, thresholds=thresholds,
                        baselines=_frozen(baselines), sample_interval=sample_interval)
    stream: list[tuple[float, int, Any]] = []
    stream += [(t, 0, (reading, in_board)) for t, reading, in_board in observations
               if low <= t < high]
    stream += [(t, 1, None) for t in absences if low <= t < high]
    stream.sort(key=lambda item: (item[0], item[1]))
    for when, kind, payload in stream:
        if kind == 0:
            reading, in_board = payload
            ledger.observe(reading, in_board_zone=in_board)
        else:
            ledger.note_absent(when)
    ledger.finish()
    return ledger


# ----------------------------------------------------------------------------------
# the statistics: Theil-Sen and Mann-Kendall, both by hand and both small
# ----------------------------------------------------------------------------------


def _theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Median of pairwise slopes, and the median intercept that goes with it.

    Written out rather than imported: scipy is not a dependency of this package and would
    be a large one to add for eleven lines. With six points this is fifteen slopes.
    """
    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + 1, len(xs))
              if xs[j] != xs[i]]
    if not slopes:
        return 0.0, 0.0
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys, strict=True)])
    return intercept, slope


def _kendall_s(values: list[float]) -> int:
    """Concordant minus discordant pairs. Ties contribute zero, which is the honest value."""
    total = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] > values[i]:
                total += 1
            elif values[j] < values[i]:
                total -= 1
    return total


def _kendall_p(n: int, s: int) -> float:
    """Exact two-sided p for Mann–Kendall's S under the no-trend null, by enumeration.

    The null distribution of S over the `n!` orderings is the Mahonian distribution of
    inversions, and it is built here with the standard O(n^3) recursion rather than
    approximated by a normal. At the n this module works with — four to six, occasionally
    eight — the normal approximation is poor exactly where the decision is made: for n = 4
    it puts the two-sided p of a perfectly monotone series near 0.04 when the exact value
    is 0.083, i.e. it would report a direction that four points cannot establish.

    Ties make this an approximation and an ANTI-conservative one; the values here are
    activity indices computed to full float precision, so exact ties are a curiosity rather
    than a case, and `_kendall_s` counts them as zero either way.
    """
    if n < 2:
        return 1.0
    pairs = n * (n - 1) // 2
    # counts[k] = number of orderings with k inversions.
    counts = [1]
    for size in range(2, n + 1):
        prefix = [0.0]
        for value in counts:
            prefix.append(prefix[-1] + value)
        length = len(counts) + size - 1
        nxt = [0.0] * length
        for k in range(length):
            hi = min(k, len(counts) - 1)
            lo = max(0, k - size + 1)
            nxt[k] = prefix[hi + 1] - prefix[lo]
        counts = nxt
    total = sum(counts)
    # S = pairs - 2 * inversions, so |S| >= |s| is a pair of tails in inversion space.
    target = abs(s)
    tail = sum(count for k, count in enumerate(counts)
               if abs(pairs - 2 * k) >= target)
    return min(1.0, tail / total)


# ----------------------------------------------------------------------------------
# the confound
# ----------------------------------------------------------------------------------


def _visibility_bound(last: Segment, retention: float) -> float:
    """The largest index FALL that the coverage change alone could account for, in points.

    Derivation, for one share term. Let the place be seen a fraction `r` as well in the last
    segment as in the first. The observations missing from the last segment, relative to the
    first, number `m (1/r - 1)` where `m` is what we measured. Adversarially they were all
    the GOOD ones — that is the direction that MANUFACTURES a decline; losing the bad ones
    would hide a real decline instead, which is not the error this gate exists to catch. Had
    they been seen, the share would have been `s r + (1 - r)`, so the observed `s`
    understates by `(1 - s)(1 - r)` and the index by `w (1 - s)(1 - r)`.

    The participation term is treated PROPORTIONALLY — the true count is at most `k / r` —
    rather than adversarially, and that asymmetry is deliberate. The adversarial bound on a
    rare countable event is vacuous: any unseen half-minute could hold a hand-raise, so the
    honest adversarial answer is the whole 30 points of the term, every time coverage moves
    at all, which would refuse every place on every lesson and say nothing. The proportional
    assumption is stated, `Segment.events_are_a_lower_bound` carries it to the reader per
    segment, and the magnitude floor separately discounts any change that rests on one
    event.

    Returns 0.0 when coverage did not fall (`r >= 1`): the bound is one-sided by
    construction, because a place seen BETTER at the end cannot have had its decline
    manufactured by being seen worse.
    """
    if retention >= 1.0 or not last.activity.available:
        return 0.0
    parts = {p.key: p.value for p in last.activity.parts}
    shares = sum(WEIGHTS[key] * (1.0 - parts.get(key, 0.0)) * (1.0 - retention)
                 for key in SHARE_KEYS)
    participation = parts.get("participation", 0.0)
    event = min(1.0, participation / retention) - participation if retention > 0 else 0.0
    return 100.0 * (shares + WEIGHTS["participation"] * event)


def _boundary_floor(segment_seconds: float, thresholds: Thresholds, *,
                    weight: float | None = None) -> float:
    """The smallest index change that the CUT ITSELF can produce, in points.

    A segment boundary lands in the middle of episodes. The longest minimum duration the
    ledger enforces is `head_down_min_seconds` (20 s), and a run of that length falling
    across a boundary instead of inside a segment moves that segment's head-up share by
    `head_down_min_seconds / segment_seconds`. An interior segment has two boundaries, so:

        floor = 100 x max(share weight) x 2 x head_down_min_seconds / segment_seconds

    On camera 01 at six segments (500.5 s) that is **2.40 index points**; on D14 (581.3 s),
    **2.06**. A reported change smaller than this is a fact about where the cuts fell.

    Computed from the run's own thresholds and its own segment length rather than fixed, so
    a lesson analysed at a different `head_down_min_seconds`, or cut into a different number
    of parts, gets the floor that belongs to it.

    `weight` selects whose floor this is: omitted, it is the widest share weight, which is
    the right bound for the TOTAL because the total can move by the worst of its terms.
    Passed, it is that one component's own floor — a change in `at_place_share` (weight
    0.15) cannot be produced by a boundary to the same extent that a change in
    `head_up_share` (0.30) can, and using the total's floor for it would suppress real
    movement in the lighter components.
    """
    if segment_seconds <= 0:
        return 0.0
    widest = max(WEIGHTS[key] for key in SHARE_KEYS) if weight is None else weight
    return 100.0 * widest * 2.0 * thresholds.head_down_min_seconds / segment_seconds


# ----------------------------------------------------------------------------------
# the whole thing
# ----------------------------------------------------------------------------------


def analyse_seat(seat_id: int, observations: list[tuple[float, Reading, bool]],
                 absences: list[float], *, window: tuple[float, float],
                 thresholds: Thresholds, sample_interval: float,
                 baselines: Baselines, segments: int = SEGMENTS) -> WithinLesson:
    """One place's within-lesson dynamics. `observations` is `(seconds, Reading, in_board)`.

    The caller supplies exactly what the whole-lesson ledger was fed — same readings, same
    board-zone verdicts, same absences — because anything else would make this a second
    analysis of the lesson rather than a partition of the one in the artefact.
    """
    start, end = window
    edges = segment_edges(start, end, segments)
    segment_seconds = (end - start) / segments

    built: list[Segment] = []
    for ordinal, (low, high) in enumerate(itertools.pairwise(edges), start=1):
        ledger = _segment_ledger(seat_id, thresholds, sample_interval, baselines,
                                 observations, absences, low, high)
        measured = sum(count for state, count in ledger.state_observations.items()
                       if state != PupilState.UNKNOWN.value)
        denominator = ledger.observations + ledger.absent_observations
        frames = (len({t for t, _reading, _board in observations if low <= t < high})
                  + sum(1 for t in absences if low <= t < high))
        events = (ledger.count(PupilState.HAND_RAISED)
                  + ledger.count(PupilState.STOOD_UP)
                  + ledger.count(PupilState.AT_BOARD))
        built.append(Segment(
            ordinal=ordinal,
            start_seconds=low, end_seconds=min(high, end),
            start_minute=(low - start) / 60.0,
            end_minute=(min(high, end) - start) / 60.0,
            analysed_frames=frames,
            coverage_denominator=denominator,
            observations=ledger.observations,
            absent_observations=ledger.absent_observations,
            measured_observations=measured,
            coverage=ledger.coverage,
            # Against the ledger's own denominator, so that this number and the coverage
            # `activity()` refused or accepted on are the same arithmetic.
            effective_coverage=measured / denominator if denominator else 0.0,
            events=events,
            events_are_a_lower_bound=ledger.coverage < 1.0,
            activity=activity(ledger),
        ))

    change = _change(built, segment_seconds=segment_seconds, thresholds=thresholds,
                     lesson_minutes=(end - start) / 60.0)
    return WithinLesson(segments=tuple(built), change=change,
                        segment_seconds=segment_seconds)


def _component_directions(usable: list[Segment], xs: list[float], span: float, *,
                          segment_seconds: float,
                          thresholds: Thresholds) -> dict[str, dict[str, Any]]:
    """The same Theil–Sen + Mann–Kendall pair, run on each component of the index.

    Each component is measured in the SAME unit as the total — its contribution in index
    points — so the four sub-changes are on one scale and add up to the change in the total.
    The ordering test is unaffected by that scaling, since a weight is a positive constant.

    **Each component carries its OWN magnitude floor, and the first version of this function
    did not.** It ran the ordering test alone, and on the real D14 lesson it printed, about
    two real children:

        место 2 · «находится на своём месте»: ниже к концу урока (−0,1 балла, p = 0,06)
        место 4 · «голова поднята»:           ниже к концу урока (−0,0 балла, p = 0,06)

    Mann–Kendall tests an ORDER, and an order can be perfectly monotone while the movement
    is four decimal places below anything the measurement resolves. A p of 0.06 beside a
    change of −0.0 index points is the most confident-looking sentence in the whole block
    and it is about nothing. So the same derivation as the total's `_boundary_floor` is
    applied per component, with that component's own weight; the participation term instead
    gets `EVENT_QUANTUM_INDEX_POINTS`, because its contribution can only take the values
    0, 10, 20, 30 and a one-step move is one episode, which the total refuses for the same
    reason.

    The visibility bound is deliberately NOT applied here: it is derived for the index as a
    whole and splitting it across components would double-count the same lost observations
    four times. `Change.visibility_bound_index_points` qualifies every row of this block at
    once, and the report prints it in the same table.
    """
    out: dict[str, dict[str, Any]] = {}
    labels = {p.key: p.label_ru for p in usable[0].activity.parts}
    for key in (*SHARE_KEYS, "participation"):
        values = [next((p.value * p.weight * 100.0 for p in s.activity.parts
                        if p.key == key), 0.0) for s in usable]
        _intercept, slope = _theil_sen(xs, values)
        s_statistic = _kendall_s(values)
        p_value = _kendall_p(len(values), s_statistic)
        delta = slope * span
        floor = (EVENT_QUANTUM_INDEX_POINTS if key == "participation"
                 else _boundary_floor(segment_seconds, thresholds, weight=WEIGHTS[key]))
        if p_value > DIRECTION_ALPHA or abs(delta) <= floor:
            direction = Direction.UNKNOWN
        elif delta < 0:
            direction = Direction.LOWER
        else:
            direction = Direction.HIGHER
        out[key] = {
            "label_ru": labels.get(key, key),
            "weight": WEIGHTS[key],
            "values_index_points": [round(v, 1) for v in values],
            "delta_index_points": round(delta, 1),
            "first_minus_last_index_points": round(values[-1] - values[0], 1),
            "kendall_s": s_statistic,
            "kendall_p": round(p_value, 4),
            "floor_index_points": round(floor, 2),
            "direction": direction.value,
            "direction_ru": RU_DIRECTION[direction],
        }
    return out


def _change(segments: list[Segment], *, segment_seconds: float, thresholds: Thresholds,
            lesson_minutes: float) -> Change:
    """The decline statistic over the segments that have an index, or a refusal.

    Read the gates in order; each is a different sentence to a different reader.
    """
    usable = [s for s in segments if s.activity.available and s.activity.index is not None]
    refused_list = [s for s in segments if s not in usable]
    refused = len(refused_list)
    refused_ordinals = tuple(s.ordinal for s in refused_list)

    if len(usable) < MIN_USABLE_SEGMENTS:
        # ACTIONABLE: name the count, the requirement, and the thing that would fix it. The
        # fix here is never «run it again» — the segments were refused because the camera
        # could not see the place, and only the camera or the seating can change that.
        why = "; ".join(
            f"сегмент {s.ordinal} ({s.start_minute:.0f}–{s.end_minute:.0f} мин): "
            f"{s.activity.reason}"
            for s in segments if not s.activity.available) or "нет данных по сегментам"
        return Change(
            available=False, direction=Direction.UNKNOWN,
            refused_because=REFUSED_TOO_FEW,
            usable_segments=len(usable), refused_segments=refused,
            refused_ordinals=refused_ordinals,
            reason=(f"пригодных сегментов {len(usable)} из {len(segments)}, нужно "
                    f"{MIN_USABLE_SEGMENTS}: при трёх точках даже строго монотонный ряд "
                    f"объясняется случайностью с вероятностью 1/3. Причины отказов — {why}. "
                    "Помогло бы одно из двух: точка съёмки, с которой это место не "
                    "перекрывается соседями, или вторая камера на тот же класс."),
        )

    xs = [s.midpoint_minute for s in usable]
    ys = [float(s.activity.index) for s in usable]      # type: ignore[arg-type]
    _intercept, slope = _theil_sen(xs, ys)
    span = xs[-1] - xs[0]
    delta = slope * span
    s_statistic = _kendall_s(ys)
    p_value = _kendall_p(len(ys), s_statistic)

    first, last = usable[0], usable[-1]
    retention = (min(1.0, last.effective_coverage / first.effective_coverage)
                 if first.effective_coverage > 0 else 0.0)
    bound = _visibility_bound(last, retention)

    first_parts = {p.key: p.value * p.weight * 100.0 for p in first.activity.parts}
    last_parts = {p.key: p.value * p.weight * 100.0 for p in last.activity.parts}
    component_change = {key: last_parts.get(key, 0.0) - first_parts.get(key, 0.0)
                        for key in (*SHARE_KEYS, "participation")}
    components = _component_directions(usable, xs, span,
                                       segment_seconds=segment_seconds,
                                       thresholds=thresholds)
    event_delta = last.events - first.events

    boundary_floor = _boundary_floor(segment_seconds, thresholds)
    # A change that rests on ONE more or one fewer visible episode is discounted in full.
    # Measured justification: on camera 01 a pupil produces 0–4 hand-raise episodes in
    # fifty minutes (`MEASUREMENTS.md` §5c), so the difference between one and none in an
    # eight-minute window is inside the ordinary variation of a lesson, and it is worth
    # exactly `EVENT_QUANTUM_INDEX_POINTS` = 10.0 of the index's 100. Without this floor,
    # camera 01 place 2 (whose whole decline is one hand-raise in segment 1 and none after)
    # would be reported as a 7.4-point fall in observable activity.
    event_floor = (abs(component_change["participation"]) if abs(event_delta) <= 1 else 0.0)
    floor = max(boundary_floor, event_floor)
    floor_from = "boundary" if boundary_floor >= event_floor else "single_event"

    covered = sum(s.end_minute - s.start_minute for s in usable)
    common = {
        "usable_segments": len(usable), "refused_segments": refused,
        "refused_ordinals": refused_ordinals,
        "delta_index_points": delta,
        "slope_index_points_per_10_minutes": slope * 10.0,
        "span_minutes": span,
        "covered_minutes": covered,
        "covers_share_of_lesson": (min(1.0, covered / lesson_minutes)
                                   if lesson_minutes else None),
        "kendall_s": s_statistic, "kendall_p": p_value,
        "floor_index_points": floor, "floor_from": floor_from,
        "boundary_floor_index_points": boundary_floor,
        "event_floor_index_points": event_floor,
        "visibility_bound_index_points": bound,
        "effective_coverage_first": first.effective_coverage,
        "effective_coverage_last": last.effective_coverage,
        "coverage_retention": retention,
        "component_change_index_points": component_change,
        "components": components,
        "events_first": first.events, "events_last": last.events,
        "driven_by_event_count": (abs(component_change["participation"])
                                  > abs(delta) / 2.0 if delta else False),
    }
    # Which minutes are NOT in the statement, by name. A count of refused segments tells a
    # reader that something is missing; only the minutes tell them whether it is the part
    # of the lesson they care about — and on camera D14 the refused segment is the LAST one
    # at two places, which is precisely the part a question about «снижение к концу урока»
    # is about.
    missing = ""
    if refused_list:
        which = ", ".join(f"{s.start_minute:.0f}–{s.end_minute:.0f} мин"
                          for s in refused_list)
        missing = (f" Утверждение опирается на {covered:.0f} минут урока из "
                   f"{lesson_minutes:.0f}: сегмент(ы) {which} отказаны по видимости и в "
                   "расчёт не вошли.")

    if abs(delta) <= floor:
        note = ("меньше того, что даёт сама нарезка на сегменты"
                if floor_from == "boundary"
                else "укладывается в один-единственный видимый эпизод")
        return Change(available=True, direction=Direction.STEADY, **common,
                      reason=(f"изменение {delta:+.1f} балла за {span:.0f} мин — {note} "
                              f"(порог {floor:.1f}). Это не «стабильно вовлечён»: это "
                              "«наблюдаемых различий между началом и концом урока "
                              "меньше, чем разрешение самого показателя»." + missing))

    if abs(delta) <= bound:
        return Change(available=False, direction=Direction.UNKNOWN,
                      refused_because=REFUSED_VISIBILITY, **common,
                      reason=(f"изменение {delta:+.1f} балла целиком объяснимо тем, что к "
                              f"концу урока это место стало хуже видно: доля читаемых "
                              f"наблюдений {first.effective_coverage:.0%} → "
                              f"{last.effective_coverage:.0%}, а это одно способно дать до "
                              f"{bound:.1f} балла. Разделить «стал менее активен» и «стало "
                              "хуже видно» на этой записи нельзя. Помогло бы: точка съёмки, "
                              "с которой это место видно одинаково в начале и в конце "
                              "урока." + missing))

    if p_value > DIRECTION_ALPHA:
        return Change(available=False, direction=Direction.UNKNOWN,
                      refused_because=REFUSED_INCONSISTENT, **common,
                      reason=(f"показатель менялся ({delta:+.1f} балла по наклону), но не в "
                              f"одну сторону: p = {p_value:.2f} при пороге "
                              f"{DIRECTION_ALPHA:.2f} для {len(usable)} сегментов. Это "
                              "значит «то выше, то ниже», а не «ровно»." + missing))

    direction = Direction.LOWER if delta < 0 else Direction.HIGHER
    events_note = ""
    if common["driven_by_event_count"]:
        events_note = (f" Большая часть изменения — счёт видимых эпизодов "
                       f"({first.events} → {last.events}), а не поза; счёт эпизодов в "
                       "сегменте с неполной видимостью — оценка снизу.")
    return Change(available=True, direction=direction, **common,
                  reason=(f"{delta:+.1f} балла за {span:.0f} мин "
                          f"({slope * 10.0:+.1f} за 10 мин), p = {p_value:.3f}. Порог "
                          f"значимости величины {floor:.1f}, вклад падения видимости — до "
                          f"{bound:.1f}." + events_note + missing))


def expected_by_chance(pupil_seats: int, alpha: float = DIRECTION_ALPHA) -> dict[str, Any]:
    """How many places in a class this size would show a direction from noise alone.

    Reported BESIDE any list of places whose index moved, for the reason `trend.py` gives at
    the level of a term: a psychologist shown one place needs to know whether one is more
    than the room would produce anyway. Here the number is exact rather than approximate —
    `DIRECTION_ALPHA` IS the per-place false-positive rate of the test by construction, so
    the expectation is simply the product.
    """
    components = len(SHARE_KEYS) + 1
    return {
        "pupil_seats": pupil_seats,
        "alpha": alpha,
        "expected_places_by_chance": round(pupil_seats * alpha, 2),
        # The component tests are FOUR MORE tests per place, and they are reported with the
        # same word «ниже». Four times as many tests is four times as many chance results,
        # and a page that lists component directions without this number reads as a page
        # about children. Counted here rather than left to the renderer for the same reason
        # `caveats` lives in the artefact.
        "component_tests_per_place": components,
        "expected_component_results_by_chance": round(pupil_seats * components * alpha, 2),
        # Decimal COMMAS: this sentence is printed verbatim on a Russian page next to
        # «58,1 минуты», and a stray «0.10» is how a reader learns that this paragraph was
        # not written for them. Same defect and same fix as `pipeline._ru_number`.
        "note_ru": (f"Порог направления — p ≤ {alpha:.2f} на проверку".replace(".", ",")
                    + f". При {pupil_seats} местах примерно "
                    + f"{pupil_seats * alpha:.1f}".replace(".", ",")
                    + " места показали бы направление по индексу просто из-за "
                      "случайности, и примерно "
                    + f"{pupil_seats * components * alpha:.1f}".replace(".", ",")
                    + f" — по отдельным составляющим (их {components} на место). Список "
                      "мест с направлением имеет смысл читать только вместе с этими "
                      "числами."),
    }

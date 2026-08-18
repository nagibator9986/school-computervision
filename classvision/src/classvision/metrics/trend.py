"""«Первая неделя хорошо, вторая не очень» — the decline statistic, and its honest limits.

--------------------------------------------------------------------------------
**THE COMPARISON IS ALWAYS A PUPIL AGAINST THEMSELVES.** Never against the class, never
against a class average, never against another child. This is what the school was promised
in writing, and it is also the only comparison the data supports: a pupil at the back of
the room is 70 px of shoulder and one at the front is 220 px, they are occluded at
different rates, and their absolute indices are not commensurable. A ranking of children
by this index would mostly be a ranking of seats by how well the camera sees them.

**Robust statistics, because a term has few lessons and one of them is a school play.**
The baseline is the MEDIAN of a pupil's own previous lessons and the spread is the MAD
(median absolute deviation), not mean and standard deviation. With ten observations a
single strange lesson moves a mean by a third of a standard deviation and moves a median
by nothing, and strange lessons are not rare in a school.

**A single low lesson is not a signal, and the code will not report one as such.** With a
class of 9 and a threshold of one lesson below −1 MAD, roughly a third of the class trips
it every week by chance alone. `SUSTAINED_LESSONS` requires the fall to persist, and
`expected_by_chance()` computes and REPORTS how many pupils this class would be expected
to trip anyway — because a psychologist looking at three highlighted children needs to
know whether three is more than nothing.

**Nothing here decides anything.** There is no alert, no flag, no colour, no ranking and
no sort order that implies one. `Trend` reports a direction, a magnitude in the pupil's own
units, and how confident the comparison is; whether that matters is a judgement belonging
to a person who knows the child. The moment this file grows a `needs_attention` boolean it
has become the diagnosis the school was told this system would not make.

--------------------------------------------------------------------------------
**THE PRECONDITION THAT IS NOT NEGOTIABLE.** A trend requires knowing that this lesson's
seat holds the same child as three weeks ago. That is `identity/`'s job, and where it
cannot establish it, `trend()` refuses rather than comparing a place to itself and calling
the result a child. A seat whose occupant changed mid-term will otherwise produce a
textbook "sustained decline" that is two different children.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Lessons of a pupil's own history before any comparison is offered. CHOSEN. Below four,
# a median is barely a median and the MAD is noise; the school was promised a four-week
# personal norm, and at roughly one observed lesson per week this is that promise.
MIN_HISTORY_LESSONS = 4

# How many of the most recent lessons form the baseline. CHOSEN to match the four weeks
# the school was told about. Older lessons drop out, so a pupil who changed in September
# is not compared to September for the rest of the year.
BASELINE_LESSONS = 8

# Consecutive recent lessons that must sit below the baseline before the change is
# described as sustained. CHOSEN at 3: with 9 pupils, one lesson below −1 MAD happens to
# about a third of the class every week, two to about a tenth, three to about one class
# in thirty. See `expected_by_chance`.
SUSTAINED_LESSONS = 3

# How far below the pupil's own baseline counts as low, in MADs. CHOSEN at 1.0.
LOW_MADS = 1.0

# Below this, MAD is treated as this instead: a pupil whose previous lessons were nearly
# identical has a MAD near zero, and dividing by it turns a one-point wobble into a
# twenty-sigma event. In index points. CHOSEN.
MIN_MAD = 3.0


class Direction(StrEnum):
    LOWER = "lower"
    HIGHER = "higher"
    STEADY = "steady"
    UNKNOWN = "unknown"


RU_DIRECTION = {
    Direction.LOWER: "ниже собственной нормы",
    Direction.HIGHER: "выше собственной нормы",
    Direction.STEADY: "в пределах собственной нормы",
    Direction.UNKNOWN: "сравнить не с чем",
}


@dataclass(frozen=True, slots=True)
class Trend:
    """One pupil's latest lesson against their own previous ones."""

    available: bool
    direction: Direction
    latest: float | None
    baseline_median: float | None
    baseline_mad: float | None
    deviation_mads: float | None
    sustained_lessons: int
    history_used: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "direction": self.direction.value,
            "direction_ru": RU_DIRECTION[self.direction],
            "latest": _round(self.latest),
            "baseline_median": _round(self.baseline_median),
            "baseline_mad": _round(self.baseline_mad),
            "deviation_mads": _round(self.deviation_mads, 2),
            "sustained_lessons": self.sustained_lessons,
            "history_used": self.history_used,
            "reason": self.reason,
        }


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def trend(history: list[float], *, identity_stable: bool = True) -> Trend:
    """`history` is this pupil's index per lesson, OLDEST first, latest last.

    `identity_stable` is the caller's assertion that every entry belongs to the same
    child. When it is False this refuses outright: comparing a seat to itself across a
    re-seating produces a textbook decline that is two different children, and there is
    no statistical repair for it.
    """
    if not identity_stable:
        return Trend(False, Direction.UNKNOWN, None, None, None, None, 0, len(history),
                     reason="личность на этом месте не подтверждена на всём периоде — "
                            "динамика не строится")
    if len(history) < MIN_HISTORY_LESSONS + 1:
        return Trend(False, Direction.UNKNOWN, None, None, None, None, 0, len(history),
                     reason=f"нужно минимум {MIN_HISTORY_LESSONS + 1} уроков этого "
                            f"ученика, есть {len(history)}")

    latest = history[-1]
    baseline = history[-(BASELINE_LESSONS + 1):-1]
    median = statistics.median(baseline)
    mad = max(statistics.median([abs(v - median) for v in baseline]), MIN_MAD)
    deviation = (latest - median) / mad

    # How many of the most recent lessons, counting back, sit below the low bar. The
    # baseline for each is the same one -- recomputing it per lesson would let a decline
    # drag its own reference down and hide itself.
    sustained = 0
    for value in reversed(history):
        if (value - median) / mad <= -LOW_MADS:
            sustained += 1
        else:
            break

    if deviation <= -LOW_MADS:
        direction = Direction.LOWER
    elif deviation >= LOW_MADS:
        direction = Direction.HIGHER
    else:
        direction = Direction.STEADY

    return Trend(True, direction, latest, median, mad, deviation, sustained,
                 len(baseline))


def expected_by_chance(class_size: int, sustained_required: int = SUSTAINED_LESSONS,
                       low_mads: float = LOW_MADS) -> dict[str, Any]:
    """How many pupils in a class of this size would look like decliners anyway.

    Reported ALONGSIDE any list of pupils below their own norm, because a psychologist
    shown three names needs to know whether three is more than the class would produce
    from noise. Without it, a page listing three children every week reads as three
    children needing attention every week.

    The per-lesson probability is derived from the definition, not measured: for a roughly
    symmetric distribution, "below the median by at least 1 MAD" is about a quarter of
    lessons, since the MAD is by construction the half-way point of the deviations. That
    is an approximation and it is labelled as one -- it is here to establish an order of
    magnitude, which is what the reader needs.
    """
    per_lesson = 0.25 if low_mads == 1.0 else max(0.0, 0.5 - 0.25 * low_mads)
    probability = per_lesson ** sustained_required
    return {
        "class_size": class_size,
        "sustained_lessons_required": sustained_required,
        "approx_probability_per_pupil": round(probability, 4),
        "expected_pupils_by_chance": round(class_size * probability, 2),
        "note": "приблизительная оценка: доля уроков ниже собственной медианы на 1 MAD "
                "принята за 0.25 по определению MAD. Нужна, чтобы понимать, сколько "
                "учеников попало бы в список просто из-за случайности.",
    }

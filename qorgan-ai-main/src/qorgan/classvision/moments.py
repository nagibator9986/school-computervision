"""WHICH seconds of a recording are worth a still. Selection only; the cutting is `frames.py`.

A frame taken every ten minutes is a page of eight people sitting down, which teaches a reader
nothing about what the system saw and — worse — suggests the system only ever sees people
sitting. The interesting seconds are the ones where a state CHANGES: a hand goes up, somebody
leaves their place, a head goes down, the adult stands. Those are exactly the seconds the
counters were built from, so a still at one of them is a picture of a count.

**Spread beats interest.** The candidates are bucketed across the analysed window and the best
one in each bucket wins, rather than the ten most interesting overall: on camera 01 the six
liveliest moments of the hour fall inside its first eight minutes, and ten stills of one minute
of one lesson would look like a summary of the lesson and not be one.

**A place is not weighted more heavily for being a child who moves more.** `STATE_INTEREST`
scores STATES by how rarely they occur in these recordings — a board visit is rarer than a turned
head, so it is likelier to be shown — and never places. Ranking children by how photogenic their
counters are would be a ranking, which this system does not do.

This module reads the timelines and returns numbers. It opens no video, imports no ffmpeg and
knows nothing about images: `classvision/frames.py` cuts the stills, and the web process only
ever reads the files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.classvision.cabinet import lesson_and_run
from qorgan.db.models.classvision import ClassvisionLesson, ClassvisionPlaceLesson

# How interesting each observable state is, as a SELECTION weight and nothing else. Ordered by
# how rare the state is across the two real recordings (39 counted episodes in total): a board
# visit or a stand is a handful of seconds in an hour, a turned head is dozens. Nothing here is
# an assessment of a state, a place or a child -- it decides which second gets photographed.
STATE_INTEREST = {
    "at_board": 6.0,
    "stood_up": 5.0,
    "hand_raised": 4.0,
    "away_from_place": 3.0,
    "head_down": 2.0,
    "turned_away": 1.0,
}

# Seconds after a state run begins. CHOSEN: the first frame of a run is the one the classifier is
# least sure about (it is the frame that crossed the threshold), and a still from the middle of a
# short episode shows the posture the count is about. Two seconds is inside even the shortest
# counted episode -- the median counted episode is 7.3 s.
INTO_THE_EPISODE_SECONDS = 2.0

# No two stills closer than this. CHOSEN so that two moments never show the same posture twice:
# the shortest break between counted episodes at one place in these recordings is about a minute.
MIN_SEPARATION_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class Moment:
    """One second worth a still, and what made it interesting."""

    second: float
    score: float
    states_ru: tuple[str, ...]


def _candidates(rows: list[ClassvisionPlaceLesson]) -> dict[float, tuple[float, list[str]]]:
    """Every state change worth a look, keyed by the second, with a score and the states there.

    Two places starting an interesting state within the same second is a better still than
    either alone, so the scores add: a frame where a hand is up while somebody else is at the
    board shows the reader more of the system than two frames would.
    """
    found: dict[float, tuple[float, list[str]]] = {}
    for row in rows:
        for episode in row.timeline or []:
            if not episode.get("measured"):
                continue
            state = str(episode.get("state") or "")
            weight = STATE_INTEREST.get(state)
            if weight is None:
                continue
            start = float(episode.get("start_s") or 0.0)
            end = float(episode.get("end_s") or 0.0)
            if end - start < INTO_THE_EPISODE_SECONDS:
                continue
            second = round(start + INTO_THE_EPISODE_SECONDS)
            score, states = found.get(second, (0.0, []))
            found[second] = (score + weight, [*states, f"{row.seat_label}: {state}"])
    return found


def pick_moments(session: Session, *, school_id: int, lesson_id: int,
                 wanted: int = 10) -> list[Moment]:
    """The seconds to photograph, spread across the analysed window, best-scoring per bucket.

    An empty list is a normal answer: a demonstration lesson has an empty timeline because there
    was never a recording, and inventing seconds for it would produce a page of stills of nothing.
    """
    found = lesson_and_run(session, school_id=school_id, lesson_id=lesson_id)
    if found is None:
        return []
    _, run = found
    rows = list(session.scalars(
        select(ClassvisionPlaceLesson)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.run_id == run.id)
    ))
    candidates = _candidates(rows)
    if not candidates:
        return []
    window = _window(run, candidates)
    return _spread(candidates, window=window, wanted=wanted)


def _window(run: Any, candidates: dict[float, Any]) -> tuple[float, float]:
    """The analysed window, from the run when it says so and from the candidates otherwise."""
    provenance = run.provenance or {}
    start = provenance.get("window_start_seconds")
    end = provenance.get("window_end_seconds")
    if start is None or end is None or float(end) <= float(start):
        return min(candidates), max(candidates) + 1.0
    return float(start), float(end)


def _spread(candidates: dict[float, tuple[float, list[str]]], *, window: tuple[float, float],
            wanted: int) -> list[Moment]:
    """One winner per equal slice of the lesson, then the leftovers by score, then deduplicated."""
    start, end = window
    width = max((end - start) / max(wanted, 1), 1.0)
    best: dict[int, tuple[float, float, list[str]]] = {}
    for second, (score, states) in candidates.items():
        bucket = int((second - start) // width)
        if bucket not in best or score > best[bucket][0]:
            best[bucket] = (score, second, states)
    chosen = sorted((second, score, states) for score, second, states in best.values())
    out: list[Moment] = []
    for second, score, states in chosen:
        if out and second - out[-1].second < MIN_SEPARATION_SECONDS:
            continue
        out.append(Moment(second=float(second), score=round(score, 1),
                          states_ru=tuple(sorted(states))))
    return out[:wanted]

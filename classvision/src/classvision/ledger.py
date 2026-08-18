"""Observations become episodes, and episodes become one lesson's totals for one place.

**Episodes, not frame counts, and the difference is the whole point of this file.** A
pupil whose hand flickers above the threshold for one sampled frame has not raised their
hand; a pupil whose hand is up for eight seconds has, once — not sixteen times because we
sampled at 2 fps. So every countable thing here is a *run* of consecutive observations
that survived a minimum hold, separated from the next run by a minimum gap. Without the
gap rule a wavering arm is counted repeatedly and the pupil with the least steady arm
tops the report, which is a defect that looks exactly like a finding.

**Three quantities that a careless ledger merges into one, kept apart.**

  * `observations` — how many times we actually looked at this place and saw somebody.
  * `absent_observations` — how many analysed frames had nobody at this place at all.
  * `unreadable_observations` — somebody was there and their pose did not resolve.

A place with 40 % `SEATED` might be a pupil sitting quietly for 40 % of the lesson, or a
pupil we could only see 40 % of the time. Those are different lessons and different
children, and only these three columns separate them. Every total this module emits is
accompanied by them, and `report/` is forbidden from rendering one without the others.

**Nothing here knows a child's name.** A ledger belongs to a `seat_id`. Names are attached
later, by `identity/`, under a stated evidence standard, and when that standard is not met
the seat keeps its number and the report stays useful.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from classvision.geometry import HeadDirection, motion
from classvision.states import Baselines, PupilState, Reading, Thresholds, classify


@dataclass(frozen=True, slots=True)
class Episode:
    """One continuous stretch in one state, after hold and gap rules were applied."""

    state: PupilState
    start_seconds: float
    end_seconds: float
    observations: int

    @property
    def duration(self) -> float:
        return max(self.end_seconds - self.start_seconds, 0.0)


@dataclass(slots=True)
class _Run:
    """A candidate episode still accumulating. Private: it may yet be discarded."""

    state: PupilState
    start: float
    end: float
    count: int


@dataclass(slots=True)
class SeatLedger:
    """Everything one place did during one lesson."""

    seat_id: int
    thresholds: Thresholds
    baselines: Baselines = field(default_factory=Baselines)

    observations: int = 0
    absent_observations: int = 0
    unreadable_observations: int = 0
    hand_unmeasurable_observations: int = 0

    first_seen: float | None = None
    last_seen: float | None = None

    state_observations: Counter = field(default_factory=Counter)
    state_seconds: Counter = field(default_factory=Counter)
    episodes: list[Episode] = field(default_factory=list)

    # Stretches of the lesson in which this place was analysed and nobody was found at
    # it. Kept as intervals, not just as the `absent_observations` count, because the
    # count answers "how much did we miss" and only the intervals answer "WHEN", which is
    # the question a strip chart asks.
    absences: list[tuple[float, float]] = field(default_factory=list)

    # Total anchor movement, in shoulder widths. Not "fidgeting" -- it is displacement,
    # and a pupil who writes energetically and one who rocks in their chair produce the
    # same number. Named for what it measures.
    total_motion: float = 0.0
    motion_samples: int = 0
    direction_observations: Counter = field(default_factory=Counter)

    # Runs thrown away for being shorter than their state's minimum. Reported, not
    # swallowed: hundreds of these means the sampling rate or the threshold is wrong,
    # and it is the only signal that says so.
    discarded: dict = field(default_factory=dict)

    # Seconds between analysed frames — the reciprocal of the sampling rate. Supplied by
    # the pipeline rather than assumed, because every `*_seconds` total below is a count
    # of observations multiplied by it, and a ledger built at 2 fps whose owner believes
    # it was 5 fps reports two and a half times too much of everything.
    sample_interval: float = 0.5

    _run: _Run | None = None
    _gap: int = 0
    _last_position: tuple[float, float] | None = None
    _absence: tuple[float, float] | None = None

    def note_absent(self, when: float | None = None) -> None:
        """An analysed frame in which nobody was at this place.

        Counted rather than ignored, because a pupil who left the room and a pupil hidden
        behind a neighbour both produce this, and the totals must not silently shrink the
        lesson to only the frames where things went well.

        `when` is optional only so that older callers keep working; supplying it is what
        lets `timeline()` draw absence as its own band instead of leaving a hole that a
        reader would charitably fill in with "sitting quietly".
        """
        self.absent_observations += 1
        self._close_run(force=True)
        if when is None:
            return
        if (self._absence is not None
                and when - self._absence[1] <= self.sample_interval * 1.5):
            self._absence = (self._absence[0], when)
        else:
            self._flush_absence()
            self._absence = (when, when)

    def _flush_absence(self) -> None:
        if self._absence is not None:
            self.absences.append(self._absence)
            self._absence = None

    def observe(self, reading: Reading, *, in_board_zone: bool = False) -> None:
        self._flush_absence()
        self.observations += 1
        if self.first_seen is None:
            self.first_seen = reading.video_seconds
        self.last_seen = reading.video_seconds

        self.baselines.observe(reading, self.thresholds)

        if reading.hand_up is None:
            self.hand_unmeasurable_observations += 1
        if reading.direction is not HeadDirection.UNKNOWN:
            self.direction_observations[reading.direction.value] += 1

        if reading.scale is not None and reading.position is not None:
            self.total_motion += motion(self._last_position, reading.position, reading.scale)
            self.motion_samples += 1
            self._last_position = reading.position
        else:
            self.unreadable_observations += 1
            self._last_position = None

        state = classify(reading, self.baselines, self.thresholds, in_board_zone=in_board_zone)
        self.state_observations[state.value] += 1
        self.state_seconds[state.value] += self.sample_interval
        self._advance(state, reading.video_seconds)

    def _advance(self, state: PupilState, when: float) -> None:
        """Run-length encode the state stream, applying the gap rule.

        The gap rule is applied by NOT closing a run the instant the state changes: a
        single stray observation inside an eight-second stretch of hand-raising is far
        more likely to be the pose model blinking than the pupil putting their arm down
        and up again.
        """
        if self._run is not None and state == self._run.state:
            self._run.end = when
            self._run.count += 1
            self._gap = 0
            return

        if self._run is not None:
            self._gap += 1
            if self._gap < self._min_gap(self._run.state):
                return   # tolerate the interruption; the run continues
            self._close_run()

        self._run = _Run(state=state, start=when, end=when, count=1)
        self._gap = 0

    def _min_gap(self, state: PupilState) -> int:
        return self.thresholds.hand_min_gap if state is PupilState.HAND_RAISED else 2

    def _min_hold(self, state: PupilState) -> int:
        if state is PupilState.HAND_RAISED:
            return self.thresholds.hand_min_hold
        return 2

    def _min_seconds(self, state: PupilState) -> float:
        if state is PupilState.HEAD_DOWN:
            return self.thresholds.head_down_min_seconds
        if state in (PupilState.AWAY_FROM_PLACE, PupilState.AT_BOARD):
            return self.thresholds.away_min_seconds
        if state is PupilState.TURNED_AWAY:
            return self.thresholds.turned_min_seconds
        return 0.0

    def _close_run(self, *, force: bool = False) -> None:
        run = self._run
        self._run = None
        self._gap = 0
        if run is None:
            return
        duration = max(run.end - run.start, 0.0)
        if run.count < self._min_hold(run.state) or duration < self._min_seconds(run.state):
            # Too short to be real. Deliberately NOT recorded as a shorter episode: a
            # half-second lean accumulating into "minutes away from place" across a
            # lesson is precisely the artefact the minimum exists to prevent.
            self.discarded[run.state.value] = self.discarded.get(run.state.value, 0) + 1
            return
        self.episodes.append(Episode(run.state, run.start, run.end, run.count))
        _ = force

    def finish(self) -> None:
        self._close_run(force=True)
        self._flush_absence()

    # -- what the report reads ------------------------------------------------------

    def count(self, state: PupilState) -> int:
        """How many separate episodes of this state. The number a teacher would recognise."""
        return sum(1 for e in self.episodes if e.state is state)

    def seconds(self, state: PupilState) -> float:
        return sum(e.duration for e in self.episodes if e.state is state)

    @property
    def observed_seconds(self) -> float:
        return self.observations * self.sample_interval

    @property
    def coverage(self) -> float:
        """Fraction of analysed frames in which this place was actually seen.

        The single number that says how much of the rest of this ledger to believe.
        Anything below ~0.5 and the totals are a sample, not a measurement.
        """
        total = self.observations + self.absent_observations
        return self.observations / total if total else 0.0

    @property
    def motion_per_observation(self) -> float:
        return self.total_motion / self.motion_samples if self.motion_samples else 0.0

    # The state a timeline entry carries when nobody was at this place. It is deliberately
    # NOT a `PupilState`: "we did not see anyone here" is not something a pupil did, and
    # putting it in the enum would let it be summed with the states that are. It is in the
    # timeline because a strip chart that renders absence exactly like quiet sitting tells
    # the reader the opposite of the truth about the minutes it covers.
    NOT_OBSERVED = "not_observed"

    def timeline(self) -> list[dict]:
        """The lesson as a sequence of spans, for a strip chart. Ordered by time.

        Three things can be true of a moment at this place and the timeline keeps them
        apart, because they are the whole reason a strip is worth drawing:

          * an EPISODE — a run of one state that survived its hold and minimum duration;
          * an ABSENCE — analysed frames in which nobody was found here at all;
          * neither — the residue of runs that were discarded for being too short. These
            are left as gaps on purpose. Filling them with the surrounding state would
            manufacture continuity the measurement does not have, and the count of what
            was discarded is already reported in `discarded_short_runs`.

        So the spans do NOT tile the lesson, and any renderer must draw the uncovered part
        as uncovered. `covered_seconds` on each entry is not offered; a reader who wants
        totals reads `seconds`, which is computed from the same episodes.
        """
        spans = [
            {"state": e.state.value, "start_s": round(e.start_seconds, 2),
             "end_s": round(e.end_seconds, 2), "observations": e.observations,
             "measured": e.state is not PupilState.UNKNOWN}
            for e in self.episodes
        ]
        spans += [
            # The end of an absence run is the last absent frame, so it is extended by one
            # sampling interval: the frame stands for the interval that follows it, and
            # without this every absence is drawn one sample short of what it was.
            {"state": self.NOT_OBSERVED, "start_s": round(start, 2),
             "end_s": round(stop + self.sample_interval, 2),
             "observations": 0, "measured": False}
            for start, stop in self.absences
        ]
        spans.sort(key=lambda s: s["start_s"])
        return spans

    def summary(self) -> dict:
        """The whole ledger as plain data, uncertainty included, ready for the artefact."""
        return {
            "seat_id": self.seat_id,
            "settled": self.baselines.settled,
            # WHY it never settled, when it never did — `null` otherwise. `settled: false`
            # alone makes every state at this place `UNKNOWN` without saying whether the
            # place was barely seen or seen badly, and those call for different actions
            # from whoever reads the report: the first is a camera angle, the second is a
            # detection this footage cannot support. See `states.Baselines`.
            "settle_refusal": self.baselines.refusal,
            "first_seen_s": self.first_seen,
            "last_seen_s": self.last_seen,
            "observations": self.observations,
            "observed_seconds": round(self.observed_seconds, 1),
            "absent_observations": self.absent_observations,
            "unreadable_observations": self.unreadable_observations,
            "hand_unmeasurable_observations": self.hand_unmeasurable_observations,
            "coverage": round(self.coverage, 3),
            "counts": {
                "hand_raises": self.count(PupilState.HAND_RAISED),
                "stands": self.count(PupilState.STOOD_UP),
                "away_episodes": self.count(PupilState.AWAY_FROM_PLACE),
                "board_visits": self.count(PupilState.AT_BOARD),
                "head_down_episodes": self.count(PupilState.HEAD_DOWN),
                "turned_away_episodes": self.count(PupilState.TURNED_AWAY),
            },
            # -- TWO different times per state, and they are NOT interchangeable. ------
            #
            # This pair used to be one key called `seconds`, and that ambiguity produced a
            # real, shipped error: an LLM asked to summarise the artefact wrote «сидел
            # 96,5 % времени (6,0 минуты)» about the adult, because `seated_share` was 96.5
            # and the only "seated seconds" on offer was 357.5 s of qualifying EPISODES.
            # Both numbers were real, both were in the document, and the sentence joining
            # them was false. The guard that checks every number in generated text against
            # the source could not catch it — it verifies that a number EXISTS, not that it
            # is attached to the right claim.
            #
            # The fix is to make the unit part of the name, so no consumer can pick the
            # wrong one by accident and no reader can join them by mistake.
            #
            # `observed_seconds_by_state` — every observation counted, at
            # `sample_interval` each. Sums to `observed_seconds`. This is the one that
            # answers "what share of the time was this pupil doing X".
            "observed_seconds_by_state": {
                state: round(self.state_seconds.get(state, 0.0), 1)
                for state in (s.value for s in PupilState)
            },
            # `episode_seconds` — only time inside runs that survived the minimum hold and
            # minimum duration for their state. ALWAYS smaller, sometimes by a lot: the
            # adult's SEATED time is chopped up by TURNED_AWAY, so 46 minutes of sitting
            # becomes 6 minutes of qualifying seated episodes. This is the one that answers
            # "how long did the episodes we are willing to call events last".
            "episode_seconds": {
                state: round(self.seconds(PupilState(state)), 1)
                for state in (s.value for s in PupilState)
            },
            "state_observations": dict(self.state_observations),
            "discarded_short_runs": self.discarded,
            "motion_per_observation": round(self.motion_per_observation, 4),
            "head_directions": dict(self.direction_observations),
        }

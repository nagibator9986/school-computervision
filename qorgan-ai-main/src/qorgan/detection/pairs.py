"""Pairs of people: the per-frame metrics, and the state that accumulates across frames.

**The counters are the domain knowledge here, and their oddity is deliberate.**

Almost every counter increments by 1 on a frame that supports it and *decays by 1* on a
frame that does not, rather than resetting to zero. That hysteresis is what lets the
detector survive a dropped frame, a missed detection, or one frame in which two children
are briefly occluded by a third. A reset-to-zero design throws away the whole struggle
every time YOLO blinks -- and on a 720p corridor camera YOLO blinks constantly.

Two counters are deliberately different:
  * `peak_close_frames` is a high-water mark and never goes down.
  * `gap_frames` resets to zero the moment the pair is close again.

The legacy earned all of this in a real hallway. Port the behaviour, not the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qorgan.config.bullying import Counters as CounterConfig
from qorgan.config.bullying import PairMetrics as MetricConfig
from qorgan.detection.constants import MIN_SAMPLES_FOR_MOTION
from qorgan.detection.geometry import (
    approach_speed,
    distance,
    dynamic_threshold,
    iou,
    relative_speed,
)
from qorgan.detection.tracking import Track

PairKey = tuple[int, int]

# Three different IoU bars. They really are different, and they are not interchangeable:
# the legacy tuned each one separately.
CONTACT_IOU = 0.05  # touching-ish
OVERLAP_COUNTER_IOU = 0.10  # what counts as an overlap FRAME
HARD_CONTACT_IOU = 0.14  # a grab, not a near-miss

HARD_CONTACT_DISTANCE_FACTOR = 0.9  # hard_contact is 10% tighter on distance
WINDOW_FRAMES = 6  # how far back `window_drop` looks
WINDOW_MIN_SAMPLES = 4


def pair_key(a: int, b: int) -> PairKey:
    """Order-independent: (3, 7) and (7, 3) are the same two children."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True, slots=True)
class PairFrame:
    """Everything measured about one pair on one frame. Physical units throughout."""

    key: PairKey
    gap: float  # px between centres
    threshold: float  # px; the proximity threshold FOR THIS PAIR, scaled by box size
    contact_ratio: float  # the camera's contact_distance_ratio
    overlap: float  # IoU, 0..1
    approach: float  # px/s, positive when closing
    relative_speed: float  # px/s
    max_acceleration: float  # px/s^2
    max_direction_change: float  # degrees
    max_speed: float  # px/s

    @property
    def is_close(self) -> bool:
        return self.gap < self.threshold

    @property
    def contact_like(self) -> bool:
        return self.gap < self.threshold * self.contact_ratio or self.overlap > CONTACT_IOU

    @property
    def hard_contact(self) -> bool:
        """Both halves are ORs, so a heavily-overlapping pair can be hard_contact
        without being distance-close, and vice versa."""
        tight = self.threshold * self.contact_ratio * HARD_CONTACT_DISTANCE_FACTOR
        return self.gap < tight or self.overlap > HARD_CONTACT_IOU

    @property
    def is_overlapping(self) -> bool:
        return self.overlap > OVERLAP_COUNTER_IOU


def measure(a: Track, b: Track, metrics: MetricConfig, proximity_scale: float = 1.0) -> PairFrame:
    """Everything we can say about this pair from the current frame alone.

    `proximity_scale` is the staircase multiplier: on stairs people legitimately pass
    within touching distance, so the threshold is widened (~1.6x). It scales the
    proximity FLOOR only, exactly as the legacy did — and because contact, overlap and
    closeness are all expressed relative to that one number, widening it loosens
    everything downstream. That is intended, and it is why there is no separate
    staircase score multiplier.
    """
    threshold = dynamic_threshold(
        a.box,
        b.box,
        base=metrics.proximity_threshold * proximity_scale,
        close_distance_ratio=metrics.close_distance_ratio,
    )

    return PairFrame(
        key=pair_key(a.track_id, b.track_id),
        gap=distance(a.center, b.center),
        threshold=threshold,
        contact_ratio=metrics.contact_distance_ratio,
        overlap=iou(a.box, b.box),
        approach=approach_speed(a.center, b.center, a.velocity, b.velocity),
        relative_speed=relative_speed(a.velocity, b.velocity),
        max_acceleration=max(a.acceleration, b.acceleration),
        max_direction_change=max(a.direction_change, b.direction_change),
        max_speed=max(a.speed, b.speed),
    )


@dataclass(slots=True)
class PairState:
    """What has accumulated about this pair over time."""

    key: PairKey

    # Per-frame evidence counters. Increment by 1, decay by 1.
    close_frames: int = 0
    contact_frames: int = 0
    overlap_frames: int = 0
    still_frames: int = 0

    # A high-water mark: the longest unbroken close stretch this pair ever had.
    peak_close_frames: int = 0
    # Consecutive frames apart. Resets to 0 the moment they are close again.
    gap_frames: int = 0

    # Escalation counters. These advance ONLY on a frame that both looks high-risk and
    # scores above the (zone-adjusted) threshold — they are the evidence that a pair is
    # actually escalating, as opposed to merely standing near each other.
    aggression_frames: int = 0
    persistence_frames: int = 0

    peak_score: float = 0.0
    last_score: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confirmed_until: float = 0.0

    alerted: bool = False
    awaiting_new_contact: bool = False

    gaps: list[float] = field(default_factory=list)  # recent centre distances

    # -- per-frame ---------------------------------------------------------

    def observe(
        self, frame: PairFrame, low_motion: bool, config: CounterConfig, now: float
    ) -> None:
        if self.first_seen == 0.0:
            self.first_seen = now
        self.last_seen = now

        up, down = config.increment, config.decay
        self.contact_frames = _step(self.contact_frames, frame.contact_like, up, down)
        self.overlap_frames = _step(self.overlap_frames, frame.is_overlapping, up, down)
        self.close_frames = _step(self.close_frames, frame.is_close, up, down)
        self.still_frames = _step(self.still_frames, low_motion and frame.is_close, up, down)

        self.peak_close_frames = max(self.peak_close_frames, self.close_frames)
        self.gap_frames = 0 if frame.is_close else self.gap_frames + 1

        self._guard_after_alert(frame)

        self.gaps.append(frame.gap)
        if len(self.gaps) > WINDOW_FRAMES:
            self.gaps.pop(0)

    def escalate(self, escalating: bool, config: CounterConfig, now: float) -> None:
        """Advance (or drain) the two counters that actually gate an alert.

        While a pair is inside its confirmation hold window the counters are FLOORED
        just below the confirm bar rather than decayed, so a single good frame re-confirms
        instantly. Without that, a two-second scuffle that YOLO half-loses in the middle
        never reaches the bar at all.
        """
        if escalating:
            self.aggression_frames += config.increment
            self.persistence_frames += config.increment
            self.confirmed_until = now + (config.temporal_window_seconds or 0.0)
            return

        if self.confirmed_until >= now:
            self.aggression_frames = max(
                self.aggression_frames, 1, config.min_aggression_frames - 1
            )
            self.persistence_frames = max(
                self.persistence_frames, 1, config.min_pair_persistence_frames - 1
            )
        else:
            self.aggression_frames = max(0, self.aggression_frames - config.decay)
            self.persistence_frames = max(0, self.persistence_frames - config.decay)

    def drain(self, config: CounterConfig) -> None:
        """A suppression gate fired. Bleed the escalation counters down.

        Deliberately bypasses the confirmation floor above: a gate can drain even a
        confirmed pair. A gate saying "this is two friends talking" outranks the
        detector's own momentum.
        """
        self.aggression_frames = max(0, self.aggression_frames - config.decay)
        self.persistence_frames = max(0, self.persistence_frames - config.decay)

    def record_score(self, score: float, peak_decay: float) -> None:
        """peak_score bleeds away when nothing is happening, so a pair that scored
        highly ten seconds ago does not stay armed indefinitely."""
        self.last_score = score
        self.peak_score = max(self.peak_score * peak_decay, score)

    # -- the post-alert separation guard -----------------------------------

    def _guard_after_alert(self, frame: PairFrame) -> None:
        """After a real alert, a NEW physical contact is required before alerting again.

        Without this, one shove in a corridor raises an alert on every single frame for
        as long as the two children stay in shot.
        """
        if self.alerted and not frame.is_close:
            self.awaiting_new_contact = True
            self.contact_frames = 0
            self.overlap_frames = 0
            self.aggression_frames = 0
            self.persistence_frames = 0

        if self.awaiting_new_contact and frame.hard_contact:
            self.awaiting_new_contact = False
            self.alerted = False

    def mark_alerted(self) -> None:
        self.alerted = True
        self.awaiting_new_contact = False

    # -- derived -----------------------------------------------------------

    @property
    def window_drop(self) -> float:
        """How far the pair has closed across the recent window, in PIXELS.

        Not a speed. It answers "did they come together?", which is a different question
        from "are they close now" — a pair that has stood together for a minute has a
        window_drop of zero. Falls back to the last single-frame drop until the window
        has filled.
        """
        if len(self.gaps) < MIN_SAMPLES_FOR_MOTION:
            return 0.0
        if len(self.gaps) < WINDOW_MIN_SAMPLES:
            return max(0.0, self.gaps[-2] - self.gaps[-1])
        return max(0.0, self.gaps[0] - self.gaps[-1])

    @property
    def max_window_gap(self) -> float:
        """The widest the pair got inside the window. Used by the social-reapproach gate."""
        return max(self.gaps) if self.gaps else 0.0

    def is_confirmed_hold(self, now: float) -> bool:
        return self.confirmed_until >= now


def _step(value: int, supported: bool, increment: int, decay: int) -> int:
    return value + increment if supported else max(0, value - decay)


class PairStore:
    """Live pair states for one camera. Bounded: pairs nobody has seen are evicted."""

    def __init__(self, *, ttl_seconds: float = 30.0) -> None:
        self._pairs: dict[PairKey, PairState] = {}
        self._ttl = ttl_seconds

    def get(self, key: PairKey, now: float) -> PairState:
        state = self._pairs.get(key)
        if state is None:
            state = PairState(key=key, first_seen=now)
            self._pairs[key] = state
        return state

    def evict(self, now: float) -> None:
        """Rule R8. Track ids only ever increase, so a dict keyed on PAIRS of them grows
        quadratically and forever. The legacy leaked exactly this, for the life of the
        process."""
        stale = [
            key
            for key, state in self._pairs.items()
            if now - state.last_seen > self._ttl and not state.is_confirmed_hold(now)
        ]
        for key in stale:
            del self._pairs[key]

    def __len__(self) -> int:
        return len(self._pairs)

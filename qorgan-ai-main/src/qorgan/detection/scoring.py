"""Turning a pair's metrics into a score, and a score into a probability.

**This module fixes the flow-zone probability bug, and it is a real behaviour change.**

The legacy dampened a pair's score inside a corridor flow zone (x0.2-0.3) and raised the
bar it had to clear (to 4.0-5.0), which is correct and well-tuned. But it then computed
`candidate_probability` by normalising that *dampened* score against the *undampened*
base threshold. In a hall (multiplier 0.2, flow threshold 5.0, base threshold 1.4) any
event that cleared the flow bar was normalised against 1.4 and came out at ~0.997 --
saturated.

That single number then fed the burst trigger, the validation gate (`>= 0.72` was
effectively always true), and 70 % of the final confidence. So the one zone specifically
designed to *suppress* corridor false positives was handing them a near-perfect
confidence score on the way out. It was a false-positive engine wearing the costume of
a false-positive filter.

Fixed here: the probability is normalised against the same threshold the score was
judged by. It means flow-zone events now land near 0.5 rather than 0.997, which is
a materially quieter pipeline -- and the eval harness is how we confirm that is right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorgan.config.bullying import BullyingConfig, Zones
from qorgan.detection.constants import (
    SAME_DIRECTION_ALIGNMENT,
    STANDING_SPEED,
    SUSTAINED_FRAMES,
)
from qorgan.detection.geometry import Box
from qorgan.detection.pairs import PairFrame, PairState
from qorgan.detection.tracking import Track


@dataclass(frozen=True, slots=True)
class ZoneContext:
    """Which zones this pair is standing in."""

    in_normal_flow: bool = False
    in_staircase: bool = False

    def proximity_scale(self, zones: Zones) -> float:
        """On stairs people legitimately pass within touching distance."""
        return zones.staircase_proximity_multiplier if self.in_staircase else 1.0

    def effective_threshold(self, zones: Zones, base: float) -> float:
        """A flow zone REPLACES the alert bar; it does not scale it."""
        return zones.normal_flow_alert_threshold if self.in_normal_flow else base

    def score_multiplier(self, zones: Zones) -> float:
        """A flow zone damps the score. Guarded at 1.0: a zone may quieten a camera,
        never amplify one."""
        if not self.in_normal_flow:
            return 1.0
        return min(1.0, zones.normal_flow_score_multiplier)


def zone_context(a: Box, b: Box, zones: Zones, width: int, height: int) -> ZoneContext:
    """Flow membership is deliberately BROAD (either person, or the midpoint, is enough:
    one child in the corridor lane means this is corridor traffic). Staircase membership
    is deliberately STRICT (both must be on the stairs)."""
    a_center = a.normalized_center(width, height)
    b_center = b.normalized_center(width, height)
    midpoint = ((a_center[0] + b_center[0]) / 2, (a_center[1] + b_center[1]) / 2)

    in_flow = any(
        rect.contains(*a_center) or rect.contains(*b_center) or rect.contains(*midpoint)
        for rect in zones.normal_flow
    )
    on_stairs = any(
        rect.contains(*a_center) and rect.contains(*b_center) for rect in zones.staircase
    )
    return ZoneContext(in_normal_flow=in_flow, in_staircase=on_stairs)


def in_mirror_zone(box: Box, zones: Zones, width: int, height: int) -> bool:
    """A reflective column in the main hall produced phantom people for months. Tracks
    inside a mirror zone are dropped before pairing -- they never exist."""
    center = box.normalized_center(width, height)
    return any(rect.contains(*center) for rect in zones.mirror_ignore)


def score_pair(
    frame: PairFrame, state: PairState, config: BullyingConfig, zone: ZoneContext
) -> float:
    """The additive score, then the zone damping. Order matters: damp AFTER summing."""
    raw = (
        _motion_terms(frame, state, config)
        + _contact_terms(frame, state, config)
        + (config.weights.chaotic_struggle if _chaotic_struggle(frame, state, config) else 0.0)
    )
    return raw * zone.score_multiplier(config.zones)


def _motion_terms(frame: PairFrame, state: PairState, config: BullyingConfig) -> float:
    """Are they coming together, and how violently?"""
    weights, metrics = config.weights, config.metrics
    terms = (
        (frame.is_close, weights.close),
        (state.window_drop > metrics.window_drop_threshold, weights.window_drop),
        (frame.approach > metrics.speed_threshold, weights.approach),
        (frame.relative_speed > metrics.min_relative_speed, weights.relative_speed),
        (frame.max_acceleration > metrics.acceleration_threshold, weights.acceleration),
        (
            frame.max_direction_change > metrics.direction_change_threshold_deg,
            weights.direction_change,
        ),
    )
    return sum(weight for fired, weight in terms if fired)


def _contact_terms(frame: PairFrame, state: PairState, config: BullyingConfig) -> float:
    """Are they touching, and have they been touching?"""
    weights = config.weights
    terms = (
        (frame.contact_like, weights.contact),
        (frame.hard_contact, weights.hard_contact),
        (state.contact_frames >= weights.hold_min_frames, weights.contact_hold),
        (state.overlap_frames >= weights.hold_min_frames, weights.overlap_hold),
    )
    return sum(weight for fired, weight in terms if fired)


def _chaotic_struggle(frame: PairFrame, state: PairState, config: BullyingConfig) -> bool:
    """Sustained closeness + sustained contact + a messy, not-quite-sharp acceleration.

    The acceleration bar here is LOWER than the one on the `acceleration` term, because
    a struggle is a scrappy, continuous thing rather than a single sharp lunge.
    """
    weights = config.weights
    return (
        state.close_frames >= config.counters.min_pair_persistence_frames
        and state.contact_frames >= weights.hold_min_frames
        and frame.max_acceleration
        > config.metrics.acceleration_threshold * weights.chaotic_acceleration_factor
    )


def candidate_probability(score: float, threshold: float) -> float:
    """Squash a score into 0..1 around the threshold it was actually judged against.

    A logistic centred on the threshold: exactly at the bar it reads 0.5, and `spread`
    controls how quickly it saturates either side.

    **Pass the EFFECTIVE (zone-adjusted) threshold.** Passing the base one is the legacy
    bug this module exists to fix -- see the module docstring.
    """
    spread = max(0.35, threshold * 0.45)
    probability = 1.0 / (1.0 + math.exp(-((score - threshold) / spread)))
    return max(0.0, min(1.0, probability))


# -- the composite predicates the gates and the confirmation test lean on ------


def same_direction(a: Track, b: Track, speed_threshold: float) -> bool:
    """Two people walking the same way at the same pace. That is a corridor, not a fight."""
    if a.speed <= STANDING_SPEED or b.speed <= STANDING_SPEED:
        return False

    unit_a = (a.velocity[0] / a.speed, a.velocity[1] / a.speed)
    unit_b = (b.velocity[0] / b.speed, b.velocity[1] / b.speed)
    alignment = unit_a[0] * unit_b[0] + unit_a[1] * unit_b[1]

    return alignment > SAME_DIRECTION_ALIGNMENT and abs(a.speed - b.speed) < speed_threshold * 0.6


def low_motion(frame: PairFrame, config: BullyingConfig) -> bool:
    """Neither person is doing anything. Used to detect two children simply standing
    together, which is the single most common thing that looks close but is not a fight."""
    gate = config.gates.static_close
    return (
        frame.approach < gate.speed_threshold
        and frame.relative_speed < gate.relative_speed_threshold
        and frame.max_acceleration < gate.acceleration_threshold
    )


def motion_present(frame: PairFrame, state: PairState, config: BullyingConfig) -> bool:
    """Is ANY action signal present at all? Proximity alone must never raise an alert."""
    metrics = config.metrics
    return (
        state.window_drop > metrics.window_drop_threshold
        or frame.approach > metrics.speed_threshold * 0.5
        or frame.max_acceleration > metrics.acceleration_threshold * 0.5
        or frame.max_direction_change > metrics.direction_change_threshold_deg * 0.6
        or frame.hard_contact
        or (
            state.contact_frames >= SUSTAINED_FRAMES
            and frame.relative_speed > metrics.min_relative_speed * 0.5
        )
        or state.overlap_frames >= SUSTAINED_FRAMES
    )


@dataclass(frozen=True, slots=True)
class Signals:
    """The 'is this strong enough' predicates, computed once and passed around."""

    strong_proximity_drop: bool
    strong_contact: bool
    strong_motion_spike: bool
    strong_direction_change: bool
    same_direction: bool
    low_motion: bool
    motion_present: bool

    @property
    def multi_signal(self) -> bool:
        return (
            self.strong_proximity_drop
            and self.strong_contact
            and self.strong_motion_spike
            and self.strong_direction_change
        )


def signals(
    frame: PairFrame, state: PairState, a: Track, b: Track, config: BullyingConfig
) -> Signals:
    metrics = config.metrics
    return Signals(
        strong_proximity_drop=state.window_drop > metrics.window_drop_threshold * 1.15,
        strong_contact=(
            frame.hard_contact
            or state.contact_frames >= SUSTAINED_FRAMES
            or state.overlap_frames >= SUSTAINED_FRAMES
        ),
        strong_motion_spike=frame.max_acceleration > metrics.acceleration_threshold * 0.8,
        strong_direction_change=(
            frame.max_direction_change > metrics.direction_change_threshold_deg * 0.75
        ),
        same_direction=same_direction(a, b, metrics.speed_threshold),
        low_motion=low_motion(frame, config),
        motion_present=motion_present(frame, state, config),
    )

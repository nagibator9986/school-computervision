"""What does the detector report when nothing is happening?

**A threshold that sits below the noise floor is not a threshold. It is a coin flip with
a bias, and no amount of tuning can rescue it.**

A YOLO box does not sit still on a person. It breathes by a few pixels every frame, and
the tracker turns that wobble into velocity, and velocity into acceleration. So a child
walking calmly down a corridor -- true acceleration exactly zero -- still reports a spray
of nonzero accelerations. That spray is the noise floor, and it is a property of the
camera and the detector, not of the children.

This module measures it. The rule it exists to enforce:

    acceleration_threshold  MUST sit above the noise floor of a calm corridor,
                            or ordinary walking crosses it and the detector is
                            reacting to the box, not to the people.

The legacy shipped 6 (px/frame)/s. v2 converted that honestly to 60 px/s^2 -- and an
honest conversion of a wrong number is still a wrong number. Measured here (10 fps,
smoothing 0.45, +/-3 px of box wobble), an ordinary walk crosses 60 px/s^2 on about
**12% of its frames**. That threshold was never detecting aggression. Meanwhile a real
shove peaks around 800 px/s^2, so the signal was never the problem: there is roughly a
6x gap between a walk and a shove, and the threshold had been placed underneath both.

Feed this a recording of a QUIET corridor -- ten minutes, nobody fighting -- and it tells
you where to put the threshold on THAT camera. Until such a recording exists,
`calm_corridor()` stands in for it with a synthetic walk, which is honest about being a
model: it is a lower bound on the real floor, never an upper one. A real camera has
motion blur, compression, occlusion and lighting change, and every one of those makes the
box wobble more, not less.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from qorgan.config.bullying import PairMetrics
from qorgan.detection.constants import analysis_fps as _analysis_fps
from qorgan.detection.geometry import Box
from qorgan.detection.tracking import TrackStore
from qorgan.evaluation.harness import FrameData

# A track's velocity needs a few frames to mean anything -- the first samples are the
# smoothing filter spinning up, not the person.
SETTLED_HITS = 6

# How far above the measured floor a threshold should sit. The floor is measured on the
# footage you happen to have; the camera will show you worse.
SAFETY_FACTOR = 2.0

# What we assume a YOLOv8n box wobbles by, absent a real recording to measure. Chosen to
# be unflattering rather than optimistic.
ASSUMED_JITTER_PX = 3.0


@dataclass(frozen=True, slots=True)
class Spread:
    """The shape of a distribution, in the only four numbers that matter here."""

    p50: float
    p95: float
    p99: float
    peak: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Spread:
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0)
        ordered = sorted(values)

        def at(fraction: float) -> float:
            index = min(int(len(ordered) * fraction), len(ordered) - 1)
            return ordered[index]

        return cls(p50=at(0.50), p95=at(0.95), p99=at(0.99), peak=ordered[-1])


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """What the detector reported on footage where nothing happened."""

    speed: Spread
    acceleration: Spread
    accelerations: tuple[float, ...]  # kept, so `crossings` can answer the real question

    @property
    def samples(self) -> int:
        return len(self.accelerations)

    @property
    def recommended_acceleration_threshold(self) -> float:
        """Where the threshold goes if you believe this measurement."""
        return self.acceleration.p99 * SAFETY_FACTOR

    def crossings(self, threshold: float) -> float:
        """The number that matters: what fraction of QUIET frames would this threshold
        call motion? Every one of them is a false positive waiting for a second person to
        stand near it."""
        if not self.accelerations:
            return 0.0
        return sum(1 for value in self.accelerations if value > threshold) / self.samples

    def summary(self, current_threshold: float | None = None) -> str:
        lines = [
            f"{self.samples} track-frames of quiet footage",
            "",
            f"{'':<16}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}",
            f"{'speed px/s':<16}{self.speed.p50:9.1f}{self.speed.p95:9.1f}"
            f"{self.speed.p99:9.1f}{self.speed.peak:9.1f}",
            f"{'accel px/s^2':<16}{self.acceleration.p50:9.1f}{self.acceleration.p95:9.1f}"
            f"{self.acceleration.p99:9.1f}{self.acceleration.peak:9.1f}",
            "",
            f"recommended acceleration_threshold: "
            f"{self.recommended_acceleration_threshold:.0f} px/s^2"
            f"  (p99 x {SAFETY_FACTOR:g})",
        ]
        if current_threshold is not None:
            crossed = self.crossings(current_threshold)
            verdict = (
                "OK -- above the floor"
                if current_threshold > self.acceleration.peak
                else f"BELOW THE NOISE FLOOR. {crossed:.1%} of quiet frames cross it."
            )
            lines.append(f"configured:                         {current_threshold:.0f} px/s^2")
            lines.append(f"verdict: {verdict}")
        return "\n".join(lines)


def measure(frames: Iterable[FrameData], metrics: PairMetrics) -> NoiseFloor:
    """Run quiet footage through the PRODUCTION tracker and see what it claims happened.

    The tracker is the one the worker uses, configured the way that camera configures it
    (rule R2). Measuring the floor with a different smoothing constant would measure a
    different camera.
    """
    store = TrackStore(
        smoothing=metrics.tracker_smoothing,
        max_lost=metrics.tracker_max_lost,
    )
    speeds: list[float] = []
    accelerations: list[float] = []

    for timestamp, detections in frames:
        store.update(detections, timestamp)
        for track in store.confirmed(SETTLED_HITS):
            speeds.append(track.speed)
            accelerations.append(track.acceleration)

    return NoiseFloor(
        speed=Spread.of(speeds),
        acceleration=Spread.of(accelerations),
        accelerations=tuple(accelerations),
    )


def calm_corridor(
    *,
    fps: float,
    jitter_px: float = ASSUMED_JITTER_PX,
    seconds: float = 120.0,
    walk_speed: float = 90.0,
    seed: int = 7,
) -> Iterator[FrameData]:
    """A stand-in for the recording the school has not made yet.

    One person, walking in a straight line, at a constant speed, for two minutes. Their
    true acceleration is zero on every single frame. The only thing that is not smooth is
    the box, which wobbles by +/-`jitter_px` the way a real detector's boxes do.

    Everything this produces is therefore noise, by construction. That is the point: it
    is a floor no threshold may sit below, derived without needing anybody to hold a
    camera. Replace it with `VideoSource` over real footage the moment there is any.
    """
    rng = random.Random(seed)  # noqa: S311 - modelling a shaky bounding box, not keying a cipher
    dt = 1.0 / fps
    x, y = 100.0, 400.0

    for index in range(int(seconds * fps)):
        x += walk_speed * dt  # constant velocity, forever. No turns, no stops.
        cx = x + rng.uniform(-jitter_px, jitter_px)
        cy = y + rng.uniform(-jitter_px, jitter_px)
        yield index * dt, {1: Box(x1=cx - 30, y1=cy - 80, x2=cx + 30, y2=cy + 80)}


def analysis_fps(stream_fps: float, det_every: int) -> float:
    """The rate the DETECTOR sees, which is the rate the noise floor depends on.

    Re-exported from `detection.constants` rather than re-implemented: this function used
    to take `display_fps`, a field production never read, so the floor was modelled at
    10 fps for a loop running at 15. The formula now has exactly one definition and both
    the harness and the production config go through it.
    """
    return _analysis_fps(stream_fps, det_every)


def synthetic_floor(metrics: PairMetrics, fps: float, *, jitter_px: float = ASSUMED_JITTER_PX):
    """The floor this camera would have, if its boxes wobbled by `jitter_px`."""
    return measure(calm_corridor(fps=fps, jitter_px=jitter_px), metrics)


def separation(floor: NoiseFloor, shove_peak: float) -> float:
    """How many times louder a real shove is than the loudest thing a quiet corridor did.

    Below ~2 there is nothing a threshold can do, and the honest answer is that this
    camera cannot support acceleration-based detection at all. We measure ~6.
    """
    return shove_peak / floor.acceleration.peak if floor.acceleration.peak > 0 else math.inf

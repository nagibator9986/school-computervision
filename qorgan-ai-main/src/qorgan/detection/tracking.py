"""Track state: where each person is, how fast, and how that is changing.

**This module is where the acceleration bug is fixed, and it matters more than it looks.**

The legacy computed speed in px/*frame* and then divided by elapsed seconds to get
"acceleration", producing units of (px/frame)/s. A comment claimed the result was
FPS-independent. It is not: a camera at 8 fps and a camera at 25 fps report different
numbers for the identical physical motion, off by a factor of three.

Every `acceleration_threshold` in every config was therefore tuned per camera against a
quantity nobody could interpret, and none of them transfer. Until this is fixed, no
recalibration of the detector means anything at all -- which is why the spec calls it
out as a blocker.

Here: velocity is px/s, acceleration is px/s^2. Both are physical, both transfer.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from qorgan.detection.constants import MIN_SAMPLES_FOR_MOTION
from qorgan.detection.geometry import Box, Point, direction_change_degrees, distance

# Velocity is measured over a window of TIME, not a window of FRAMES.
#
# This is the subtle half of the same bug. Averaging over "the last 3 samples" gives a
# smoothing time-constant of 0.375 s at 8 fps and 0.12 s at 25 fps, so even with speed
# correctly in px/s the *acceleration* still would not transfer between cameras. A
# window in seconds is the same window on every camera.
VELOCITY_WINDOW_SECONDS = 0.3

# Enough samples to cover the window at any sane frame rate, and bounded (rule R8).
MAX_SAMPLES = 64


@dataclass(slots=True)
class Sample:
    timestamp: float
    center: Point
    speed: float = 0.0


@dataclass(slots=True)
class Track:
    """One tracked person, in physical units."""

    track_id: int
    box: Box
    timestamp: float  # seconds, monotonic

    velocity: Point = (0.0, 0.0)  # px/s
    speed: float = 0.0  # px/s
    acceleration: float = 0.0  # px/s^2
    direction_change: float = 0.0  # degrees since the previous velocity
    hits: int = 0  # frames this track has been seen

    history: deque[Sample] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))

    @property
    def center(self) -> Point:
        return self.box.center

    def is_confirmed(self, min_hits: int) -> bool:
        """A track seen once or twice is usually a detector hiccup, not a person."""
        return self.hits >= min_hits


class TrackStore:
    """Holds the live tracks for one camera and updates them from detections."""

    def __init__(self, *, smoothing: float = 0.55, max_lost: int = 18) -> None:
        self._smoothing = min(max(smoothing, 0.0), 1.0)
        self._max_lost = max_lost
        self._tracks: dict[int, Track] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def confirmed(self, min_hits: int) -> list[Track]:
        return [t for t in self._tracks.values() if t.is_confirmed(min_hits)]

    def update(self, detections: dict[int, Box], timestamp: float) -> None:
        """Fold this frame's detections in, then drop tracks nobody has seen lately."""
        self._frame += 1
        for track_id, box in detections.items():
            existing = self._tracks.get(track_id)
            if existing is None:
                self._tracks[track_id] = _new_track(track_id, box, timestamp)
            else:
                _advance(existing, box, timestamp, self._smoothing)
            self._last_seen[track_id] = self._frame

        self._evict()

    def _evict(self) -> None:
        """Bounded by construction (rule R8). The legacy leaked several dicts keyed on
        track id, and track ids only ever go up, so they grew without limit for as long
        as the process lived."""
        stale = [
            track_id
            for track_id, frame in self._last_seen.items()
            if self._frame - frame > self._max_lost
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)
            self._last_seen.pop(track_id, None)


def _new_track(track_id: int, box: Box, timestamp: float) -> Track:
    track = Track(track_id=track_id, box=box, timestamp=timestamp, hits=1)
    track.history.append(Sample(timestamp, box.center))
    return track


def _advance(track: Track, box: Box, timestamp: float, smoothing: float) -> None:
    if timestamp <= track.timestamp:
        # Two detections at the same instant. Take the newer box, but do not divide by
        # zero and do not invent motion.
        track.box = box
        track.hits += 1
        return

    previous_velocity = track.velocity
    _prune(track, timestamp)

    oldest = track.history[0]
    span = timestamp - oldest.timestamp

    velocity = _velocity(oldest.center, box.center, span)
    # Exponential smoothing, so one jittery box does not read as a lunge.
    track.velocity = (
        smoothing * previous_velocity[0] + (1 - smoothing) * velocity[0],
        smoothing * previous_velocity[1] + (1 - smoothing) * velocity[1],
    )
    track.speed = math.hypot(*track.velocity)

    # px/s^2: a change of speed (px/s) across the time window (s). Both sides are
    # physical, and the window is the same duration on every camera, so this number
    # means the same thing at 8 fps and at 25 fps.
    track.acceleration = abs(track.speed - oldest.speed) / span if span > 0 else 0.0
    track.direction_change = direction_change_degrees(previous_velocity, track.velocity)

    track.history.append(Sample(timestamp, box.center, track.speed))
    track.box = box
    track.timestamp = timestamp
    track.hits += 1


def _prune(track: Track, timestamp: float) -> None:
    """Drop samples that have fallen out of the time window, keeping at least one."""
    while (
        len(track.history) > 1 and timestamp - track.history[0].timestamp > VELOCITY_WINDOW_SECONDS
    ):
        track.history.popleft()


def _velocity(start: Point, end: Point, span: float) -> Point:
    if span <= 0:
        return (0.0, 0.0)
    return ((end[0] - start[0]) / span, (end[1] - start[1]) / span)


def travelled(track: Track) -> float:
    """Distance covered across the time window, in px. Used by the gates that ask
    'did this person actually go anywhere, or are they just wobbling?'"""
    if len(track.history) < MIN_SAMPLES_FOR_MOTION:
        return 0.0
    return distance(track.history[0].center, track.history[-1].center)

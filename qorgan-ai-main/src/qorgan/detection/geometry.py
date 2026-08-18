"""Boxes, distances, overlaps. Pure functions, no state, no config, no I/O.

This module exists because of rule R2. The production worker and the eval harness call
*these* functions, not two copies of them. The legacy had `analyze_aggression` and its
helpers duplicated across `bullying_worker.py`, `eval_harness.py` and `eval_subset.py`,
and the three had already diverged -- so the harness measured code that did not run in
production, and months of threshold tuning were built on a fake benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]

# Below this, a vector is numerically zero and has no meaningful direction.
EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned bounding box in pixels. x2 > x1 and y2 > y1 always."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def diagonal(self) -> float:
        """Used as the scale of a person. A person twice as close is twice as big, so
        every distance threshold is expressed relative to this rather than in raw
        pixels -- that is what lets one config work at both ends of a corridor."""
        return math.hypot(self.width, self.height)

    def clamped(self, width: int, height: int) -> Box:
        """Clip to the frame. A box that pokes outside it crops to nothing later."""
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        return Box(x1, y1, max(x2, x1 + 1.0), max(y2, y1 + 1.0))

    def normalized_center(self, width: int, height: int) -> Point:
        """Centre in 0..1 frame coordinates, which is how zones are defined."""
        cx, cy = self.center
        return (cx / width, cy / height) if width and height else (0.0, 0.0)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(a: Box, b: Box) -> float:
    """Intersection over union. 0.0 when the boxes do not touch."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def dynamic_threshold(a: Box, b: Box, base: float, close_distance_ratio: float) -> float:
    """The proximity threshold for THIS pair, scaled to how big the two people look.

    Two children at the far end of a corridor are 40 px apart when touching; two in the
    foreground are 200 px apart while still at arm's length. A fixed pixel threshold
    cannot serve both ends of a camera with any depth at all.

    The scale term is the SUM of the two diagonals (not the mean), which is what the
    legacy used and what every shipped `close_distance_ratio` was tuned against. `base`
    acts as a floor for distant, small boxes.
    """
    scale = (a.diagonal + b.diagonal) * close_distance_ratio
    return max(base, scale)


def direction_change_degrees(previous: Point, current: Point) -> float:
    """Angle between two velocity vectors, in degrees. 0 when either is still.

    A sudden change of direction is one of the few motion signals that separates a
    shove from walking: people in a corridor travel in straight lines.
    """
    px, py = previous
    cx, cy = current
    previous_magnitude = math.hypot(px, py)
    current_magnitude = math.hypot(cx, cy)
    if previous_magnitude < EPSILON or current_magnitude < EPSILON:
        return 0.0

    cosine = (px * cx + py * cy) / (previous_magnitude * current_magnitude)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def approach_speed(a_center: Point, b_center: Point, a_velocity: Point, b_velocity: Point) -> float:
    """How fast the gap is closing, in px/s. Positive means approaching.

    This is the component of relative velocity along the line between the two people.
    Two people running side by side have a large relative speed but zero approach
    speed; that difference is exactly what distinguishes a chase from a charge.
    """
    dx = b_center[0] - a_center[0]
    dy = b_center[1] - a_center[1]
    separation = math.hypot(dx, dy)
    if separation < EPSILON:
        return 0.0

    relative_vx = a_velocity[0] - b_velocity[0]
    relative_vy = a_velocity[1] - b_velocity[1]
    # Project the relative velocity onto the unit vector from a towards b.
    return (relative_vx * dx + relative_vy * dy) / separation


def relative_speed(a_velocity: Point, b_velocity: Point) -> float:
    """Magnitude of the difference of the two velocities, in px/s."""
    return math.hypot(a_velocity[0] - b_velocity[0], a_velocity[1] - b_velocity[1])

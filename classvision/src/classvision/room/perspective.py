"""Undo the camera's perspective, so one distance threshold means one thing everywhere.

**The problem, in one measurement.** Across a single frame of this room the shoulder width
runs from **19 px at the back to 232 px at the front** — a factor of twelve. Two pupils
sharing a front desk are ~150 px apart; two sharing a back desk are ~60 px apart. They are
the same distance apart *in the only unit that matters*, which is the width of the people
involved. Any algorithm that works in pixels is measuring the seating plan.

`geometry.py` solves this pointwise by dividing each measurement by that person's own
shoulder width. Clustering cannot do that, because a cluster is a relation between *many*
points and there is no single scale to divide by. The usual workaround — a pairwise
distance matrix normalised by the mean of each pair's scales — is O(n²) in memory, and at
52 minutes and 2 fps this room produces ~28 000 anchors, so that matrix is 6 GB.

**The fix is to warp the coordinates once, then use ordinary Euclidean distance.**

For a camera looking down at a floor, apparent size grows linearly with image row: a
person at image row *y* has scale ``s(y) ≈ a·y + b``. That is not an assumption here, it is
fitted to this room's own anchors and the fit quality is reported (`Perspective.r_squared`)
so a room where it does not hold says so instead of quietly returning nonsense.

Given that, define a map whose local distances are already in shoulder widths:

    du/dx = 1/s(y)      ⟹   u = x / (a·y + b)
    dv/dy = 1/s(y)      ⟹   v = (1/a)·ln(a·y + b)          (a ≠ 0)

The vertical integral is a logarithm because the divisor itself changes as you move down
the frame; using ``y/s(y)`` instead would be right only at one row. In ``(u, v)`` space a
Euclidean distance of 1.0 means "one shoulder width" anywhere in the frame, so DBSCAN's
single `eps` is finally a single physical statement, and the O(n²) matrix disappears.

**When the room is flat to the camera** (``a ≈ 0``, everyone the same size — a camera
facing a room square-on) the logarithm degenerates, and the honest map is simply
``x/b, y/b``. `fit` detects that case and says which map it used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this |a| the frame has no usable perspective gradient and the affine map is used.
# In units of shoulder-width-pixels per image row: 1e-4 means the scale changes by less
# than a tenth of a pixel over a thousand rows. CHOSEN as "indistinguishable from flat".
FLAT_GRADIENT = 1e-4

# The fit must explain at least this much of the variance in scale before the warp is
# trusted. Below it, the room does not behave like a plane viewed obliquely -- a mirror, a
# staircase, two cameras stitched -- and the caller is told rather than served a warp that
# is confidently wrong. CHOSEN; on this footage the fit lands far above it.
MIN_R_SQUARED = 0.35


@dataclass(frozen=True, slots=True)
class Perspective:
    """A fitted scale-vs-row model and the warp it induces."""

    a: float           # px of shoulder width per image row
    b: float           # px of shoulder width at row 0
    r_squared: float
    samples: int
    mode: str          # "log" (oblique) | "affine" (flat) | "degenerate"

    @property
    def usable(self) -> bool:
        return self.mode != "degenerate" and self.r_squared >= MIN_R_SQUARED

    def scale_at(self, y: float | np.ndarray):
        """Expected shoulder width, in pixels, for a body whose anchor is at row `y`."""
        return np.maximum(self.a * np.asarray(y, dtype=float) + self.b, 1e-6)

    def warp(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) image points -> (N, 2) points whose unit is one shoulder width."""
        points = np.asarray(points, dtype=float)
        x, y = points[:, 0], points[:, 1]
        s = self.scale_at(y)
        u = x / s
        if self.mode == "log":
            v = np.log(s) / self.a
        else:
            v = y / max(self.b, 1e-6)
        return np.column_stack([u, v])

    def to_dict(self) -> dict:
        return {"a": round(self.a, 6), "b": round(self.b, 2),
                "r_squared": round(self.r_squared, 3), "samples": self.samples,
                "mode": self.mode, "usable": self.usable}


def fit(positions: np.ndarray, scales: np.ndarray) -> Perspective:
    """Fit shoulder width as a linear function of image row, robustly.

    Robustly, because the input contains the adult — three times a pupil's scale — and
    anyone who walked through. A least-squares line through those is dragged by them, and
    a dragged line warps the whole room. So the fit is done on the per-row MEDIAN of a
    handful of row bands rather than on the raw points: a band's median is unmoved by one
    large person standing in it, and there are enough bands to define a line.
    """
    positions = np.asarray(positions, dtype=float)
    scales = np.asarray(scales, dtype=float)
    ok = np.isfinite(scales) & (scales > 0)
    positions, scales = positions[ok], scales[ok]
    if positions.shape[0] < 8:
        return Perspective(0.0, float(np.median(scales)) if scales.size else 1.0,
                           0.0, int(scales.size), "degenerate")

    y = positions[:, 1]
    # Bands by quantile rather than by fixed height, so a room occupying a third of the
    # frame still yields bands that each hold points.
    edges = np.quantile(y, np.linspace(0.0, 1.0, 9))
    edges = np.unique(edges)
    if edges.size < 3:
        return Perspective(0.0, float(np.median(scales)), 0.0, int(scales.size), "degenerate")

    rows, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_band = (y >= lo) & (y <= hi)
        if in_band.sum() < 3:
            continue
        rows.append(float(np.median(y[in_band])))
        medians.append(float(np.median(scales[in_band])))
    if len(rows) < 3:
        return Perspective(0.0, float(np.median(scales)), 0.0, int(scales.size), "degenerate")

    rows_arr = np.array(rows)
    med_arr = np.array(medians)
    a, b = np.polyfit(rows_arr, med_arr, 1)
    predicted = a * rows_arr + b
    residual = float(np.sum((med_arr - predicted) ** 2))
    total = float(np.sum((med_arr - med_arr.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-12 else 0.0

    mode = "log" if abs(a) >= FLAT_GRADIENT else "affine"
    # A line that predicts a non-positive width anywhere in the observed rows is not a
    # perspective model, it is an extrapolation accident.
    if np.any(a * y + b <= 1.0):
        mode = "affine"
        a, b = 0.0, float(np.median(scales))
        r_squared = 0.0

    return Perspective(float(a), float(b), float(r_squared), int(scales.size), mode)


def summarise(model: Perspective, positions: np.ndarray, scales: np.ndarray) -> dict:
    """How well the warp actually equalises scale — the number that justifies using it.

    Reports the spread of (true scale / predicted scale). If the warp works this ratio is
    near 1 everywhere and its spread is small; if it does not, the spread stays wide and
    the caller can see that a single `eps` is still measuring the seating plan.
    """
    info = model.to_dict()
    positions = np.asarray(positions, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if positions.size == 0 or not model.usable:
        return info
    ratio = scales / model.scale_at(positions[:, 1])
    info["scale_ratio"] = {
        "p10": round(float(np.percentile(ratio, 10)), 3),
        "median": round(float(np.median(ratio)), 3),
        "p90": round(float(np.percentile(ratio, 90)), 3),
        # Before the warp, the same spread measured on raw pixels. The comparison is the
        # whole argument for warping at all.
        "raw_px_p10_p90": [round(float(np.percentile(scales, 10)), 1),
                           round(float(np.percentile(scales, 90)), 1)],
    }
    return info


def as_units(model: Perspective, pixels: float, at_row: float) -> float:
    """A pixel distance at a given image row, expressed in shoulder widths."""
    return pixels / float(model.scale_at(at_row))


def frame_diagonal_units(model: Perspective, width: int, height: int) -> float:
    """A sanity bound: how many shoulder widths across the frame. Used to reject an `eps`
    large enough to swallow the room, which is otherwise a silent way to find one seat."""
    corners = np.array([[0.0, 0.0], [float(width), 0.0],
                        [0.0, float(height)], [float(width), float(height)]])
    warped = model.warp(corners)
    return float(max(math.dist(warped[i], warped[j])
                     for i in range(4) for j in range(i + 1, 4)))

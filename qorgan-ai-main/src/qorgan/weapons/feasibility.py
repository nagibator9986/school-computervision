"""Can this camera see an object of that size, at that distance? Arithmetic, not a hope.

`qorgan identity camera-report` answers "can this camera recognise anybody at the
resolution the worker actually feeds it?" and its answer for this school's hall is no --
14 970 faces, median 11.5 px, zero recognised. This module asks the same question about a
knife, and it has to be asked BEFORE a camera is mounted, because the answer does not
move afterwards:

    A knife in a hand at the school entrance is a 100+ px object. It will work.
    The same knife down a corridor at 15 m is ~15 px. It will never work.

That is what we told the school (`docs/questions-for-school.md` §7) and it is not a
threshold problem. No confidence, no imgsz and no model changes it: the object stops
occupying enough pixels to have a shape.

**What this can and cannot do, precisely.** It is a pinhole projection -- object size,
distance, horizontal field of view, and the width of the frame the worker really analyses.
It is exact for what it models and it does not model the things that also matter: motion
blur, JPEG artefacts on an NVR substream, whether the blade is edge-on, or how good any
particular weights are. So a PASS here is a necessary condition and not a promise, and the
report says so in those words. A FAIL is decisive: the pixels are not there and nothing
downstream can put them back.

**The lens is asked, never assumed.** Horizontal field of view is a property of the
hardware and is not in this project's config -- no camera file carries it, and inventing a
key nothing reads would be one more dead knob in a schema that has been cleaned of thirty.
So it is a command-line argument with a stated default, and the report prints which value
it used, because the answer moves a long way with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A 4 mm lens on a 1/2.8" sensor, which is the commonest fixed-lens IP camera and what
# this school's Hikvision units are. CHOSEN as a default to argue with, not measured:
# read the real figure off the camera's own datasheet and pass --hfov-deg.
DEFAULT_HFOV_DEGREES = 78.0

# The visible length of a kitchen knife held in a hand, in centimetres. CHOSEN. Most of a
# blade plus a little handle; a folding knife is half this and a bat is three times it,
# which is exactly why it is an argument.
DEFAULT_OBJECT_CM = 20.0


@dataclass(frozen=True, slots=True)
class SizeAtDistance:
    """How big something is, in pixels, from here."""

    distance_m: float
    pixels: float
    clears_gate: bool


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """What this camera can see of an object this size. Printed by `weapons camera-report`."""

    camera: str
    frame_width: int
    hfov_degrees: float
    object_cm: float
    min_object_pixels: float
    max_useful_distance_m: float
    samples: tuple[SizeAtDistance, ...]

    @property
    def usable(self) -> bool:
        """Is there ANY sampled distance at which this object clears the gate?"""
        return any(sample.clears_gate for sample in self.samples)

    def summary(self) -> str:
        lines = [
            f"{self.camera}: an object {self.object_cm:g} cm across, seen at "
            f"{self.frame_width} px wide with a {self.hfov_degrees:g}° lens.",
            f"  the module refuses anything under {self.min_object_pixels:g} px "
            "(config: weapons.min_object_pixels)",
            "",
        ]
        for sample in self.samples:
            mark = "OK  " if sample.clears_gate else "NO  "
            lines.append(f"  {mark}{sample.distance_m:>5.1f} m -> {sample.pixels:6.1f} px")
        lines.append("")
        lines.append(
            f"  beyond {self.max_useful_distance_m:.1f} m this object cannot clear the "
            "gate at any confidence"
        )
        return "\n".join(lines)


def apparent_pixels(object_m: float, distance_m: float, frame_width: int, hfov_deg: float) -> float:
    """Pinhole projection: how wide the object is, in pixels of the analysed frame.

    The frame's whole width spans `2 * distance * tan(hfov/2)` metres at that distance, so
    the object's share of the width is its share of the pixels. Exact for what it models;
    see the module docstring for what it does not.
    """
    if distance_m <= 0:
        raise ValueError(f"distance must be positive, got {distance_m}")
    span_m = 2.0 * distance_m * math.tan(math.radians(hfov_deg) / 2.0)
    return frame_width * object_m / span_m if span_m > 0 else 0.0


def max_useful_distance(
    object_m: float, frame_width: int, hfov_deg: float, min_pixels: float
) -> float:
    """The distance at which this object falls to exactly `min_pixels`. Beyond it: never.

    The single number worth taking to a site survey. Everything about where a weapons
    camera goes follows from it, and it is the number that makes "put one in the corridor
    as well" answerable without a term of silence to prove it.
    """
    if min_pixels <= 0:
        raise ValueError(f"min_pixels must be positive, got {min_pixels}")
    span_per_metre = 2.0 * math.tan(math.radians(hfov_deg) / 2.0)
    if span_per_metre <= 0:
        return 0.0
    return frame_width * object_m / (min_pixels * span_per_metre)


def assess(
    *,
    camera: str,
    frame_width: int,
    min_object_pixels: float,
    object_cm: float = DEFAULT_OBJECT_CM,
    hfov_deg: float = DEFAULT_HFOV_DEGREES,
    distances_m: tuple[float, ...] = (1.5, 3.0, 5.0, 10.0, 15.0),
) -> FeasibilityReport:
    """The whole answer for one camera, at a handful of distances a school recognises.

    The default distances are the places in a school building, not a sweep: 1.5 m is a
    doorway or a turnstile, 3 m is a lobby, 15 m is the far end of the corridor this
    school's hall cameras look down -- the exact case §7 promised would never work.
    """
    object_m = object_cm / 100.0
    samples = tuple(
        SizeAtDistance(
            distance_m=distance,
            pixels=(px := apparent_pixels(object_m, distance, frame_width, hfov_deg)),
            clears_gate=px >= min_object_pixels,
        )
        for distance in distances_m
    )
    return FeasibilityReport(
        camera=camera,
        frame_width=frame_width,
        hfov_degrees=hfov_deg,
        object_cm=object_cm,
        min_object_pixels=min_object_pixels,
        max_useful_distance_m=max_useful_distance(
            object_m, frame_width, hfov_deg, min_object_pixels
        ),
        samples=samples,
    )

"""Can this camera ever recognise anybody? **A question about optics, not thresholds.**

**0 of 14 970** faces in 250 clips of the school hall clear the enrolment gate -- at the
resolution the hall worker actually analyses (**1280x720**), where the median face is
**11.5 px** and the largest in the entire corpus is **50 px**. The gate needs a 120 px face
in the 2560x1440 clip; no such face exists. Upscaled to ArcFace's 112-pixel input those
faces are mush, and no value of any threshold recovers them.

**And 1280x720 is not a number you may assume.** `base.yaml` holds a default; profiles
override it. There is no fleet-wide analysis resolution, and this module reads it from the
camera's own merged config.

Quoting the default as if there were one is how the figures above were wrong ONCE ALREADY
(they read "max 37.5 px, median 9 px" -- scaled by 0.375 when the hall's real scale is 0.5).

And this docstring then got it wrong a SECOND time, by naming which profiles override: it
said two, and there were three. The omitted one closes meal sessions. So it names none now.
A list of overriders is a second source of truth, and a second source of truth is the bug.

(On the 2560x1440 HD burst the same faces give 2.2%. That number is true and useless: it
describes a stream the analysis loop never touches. Reporting it per-CAMERA instead of
per-STREAM is the bug this module exists to prevent -- and it is worse than reporting
nothing, because 2.2% reads as low-but-non-zero and invites "drop min_width to 40 and
recover some". **That recovers NOTHING, and it has been checked on the numbers:** dropping
to the 38 px small-face gate admits 77 of 14 970 faces, and **not one of them is
recognised** -- the best score among all 77 is 0.350, against a min_score of 0.45. The
faces that are big enough are still too degraded to score.)

So a report here is always **of one stream**, and it always states the resolution it was
measured at -- see `FaceSizeReport.analysed_at`, which this module derives from the frames
it was actually handed rather than from anything a caller claims.

And it **REFUSES**. `refuse_if_hopeless()` turns "this camera recognises nobody" from a
discovery made after months of tuning into a fact asserted at startup.

The legacy spent eighteen overlapping thresholds looking for a number that would fix a
problem no number could fix. If this says ~0%: **move the camera.**
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qorgan.config.identity import FaceGate

# Below this fraction, tuning is not the answer and never will be.
HOPELESS = 0.10

_MOVE_THE_CAMERA = (
    "The answer is to MOVE THE CAMERA -- closer, or lower, or both. Do NOT lower the "
    "gate: there is nothing underneath it to recover. On the school's hall, dropping to "
    "the 38px small-face gate admits 77 of 14 970 faces and recognises NONE of them (best "
    "score 0.350). An 11-pixel face upscaled to ArcFace's 112-pixel input is mush, and the "
    "legacy spent eighteen thresholds looking for a number that could not exist."
)


class CameraCannotRecognise(RuntimeError):
    """This stream's faces never clear the gate, so it is not an identity camera.

    Raised at startup, in the face of the operator, rather than discovered in month four
    from an event log full of Unknown.
    """


class Sized(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


class Detector(Protocol):
    """`detect_faces`, not `detect`: this module wants BOXES, and never a vector.

    Measuring how big the faces are does not need the 512-d ArcFace embedding, and paying
    for one per face across 250 clips was the expensive way to answer a question about
    optics.
    """

    def detect_faces(self, frame: np.ndarray) -> list[Sized]: ...


@dataclass(frozen=True, slots=True)
class FaceSizeReport:
    """One STREAM, measured at the resolution that stream is really analysed at."""

    frames: int
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    gate: FaceGate
    source: str
    # Derived from the frames themselves, never asserted by the caller: (width, height).
    analysed_at: tuple[int, int] | None = None

    @property
    def faces(self) -> int:
        return len(self.widths)

    @property
    def clearing_gate(self) -> int:
        return sum(
            1
            for width, height in zip(self.widths, self.heights, strict=True)
            if self.gate.accepts(width, height)
        )

    @property
    def fraction_clearing(self) -> float:
        return self.clearing_gate / self.faces if self.faces else 0.0

    @property
    def conclusive(self) -> bool:
        """No faces seen is not evidence of a working camera; it is no evidence at all."""
        return self.faces > 0

    @property
    def usable(self) -> bool:
        """May this stream be trusted to recognise anybody?"""
        return self.conclusive and self.fraction_clearing >= HOPELESS

    def width_percentile(self, p: float) -> float:
        return float(np.percentile(self.widths, p)) if self.widths else 0.0

    def resolution(self) -> str:
        if self.analysed_at is None:
            return "no frames"
        return f"{self.analysed_at[0]}x{self.analysed_at[1]}"

    def gate_description(self) -> str:
        return (
            f"{self.gate.min_width}x{self.gate.min_height}px "
            f"(min area {self.gate.min_area}px^2)"
        )

    def headline(self) -> str:
        # "this STREAM", and always with the resolution. The same faces answer this
        # question differently on two streams of one camera, which is why a per-camera
        # headline would be a lie on at least one of them.
        return (
            f"{self.source} @ {self.resolution()} -- can this stream recognise anybody at "
            "the resolution it is actually analysed at?"
        )

    def summary(self) -> str:
        return "\n".join([self.headline(), "", *self._body(), "", self._verdict()])

    def _body(self) -> list[str]:
        if not self.faces:
            return [f"  {self.frames} frame(s), NO FACES AT ALL."]
        return [
            f"  {self.frames} frame(s), {self.faces} face(s).",
            "",
            "  face width",
            f"    p50 {self.width_percentile(50):.0f}px   "
            f"p90 {self.width_percentile(90):.0f}px   "
            f"max {self.width_percentile(100):.0f}px",
            "",
            f"  clearing the {self.gate_description()} gate: "
            f"{self.clearing_gate} / {self.faces}  ({self.fraction_clearing * 100:.1f}%)",
        ]

    def _verdict(self) -> str:
        if not self.faces:
            return (
                "  UNANSWERED: either nobody walked through, or this camera cannot see a "
                "face at all. Neither is fixed by a threshold, and neither is a pass."
            )
        if not self.usable:
            return f"  ANSWER: NO. {_MOVE_THE_CAMERA}"
        return (
            f"  ANSWER: yes -- {self.fraction_clearing * 100:.1f}% of the faces this "
            "stream sees are big enough to recognise."
        )


def measure_faces(
    frames: Iterable[np.ndarray],
    detector: Detector,
    gate: FaceGate,
    source: str,
) -> FaceSizeReport:
    """Pure over its inputs: frames in, distribution out. No camera, no clock, no DB.

    The frames must all be the size the stream is ANALYSED at -- scale them with
    `prepare_frame` before they get here (`identity.streams.sample` does). Mixing two
    sizes is refused rather than averaged, because averaging is how a measurement of the
    wrong stream disappears into a plausible number.
    """
    widths: list[int] = []
    heights: list[int] = []
    analysed_at: tuple[int, int] | None = None
    count = 0

    for frame in frames:
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if analysed_at is None:
            analysed_at = size
        elif size != analysed_at:
            raise ValueError(
                f"{source}: frames of two different sizes in one report "
                f"({analysed_at[0]}x{analysed_at[1]} and {size[0]}x{size[1]}). A face size "
                "is meaningless without the frame it was measured in."
            )

        count += 1
        for face in detector.detect_faces(frame):
            widths.append(face.width)
            heights.append(face.height)

    return FaceSizeReport(
        frames=count,
        widths=tuple(widths),
        heights=tuple(heights),
        gate=gate,
        source=source,
        analysed_at=analysed_at,
    )


def refusal(report: FaceSizeReport) -> str:
    """The refusal, naming the STREAM, the GATE and the MEASURED FRACTION."""
    return "\n".join(
        [
            f"REFUSED: {report.source} cannot be an identity camera.",
            f"  stream measured at: {report.resolution()}  "
            "(the resolution the worker actually analyses it at)",
            f"  gate:               {report.gate_description()}",
            f"  measured:           {report.clearing_gate} of {report.faces} faces clear it "
            f"({report.fraction_clearing * 100:.1f}%)",
            "",
            f"  {_MOVE_THE_CAMERA}",
        ]
    )


def refuse_if_hopeless(report: FaceSizeReport) -> None:
    """GATE. A camera whose faces never clear the gate on its ANALYSIS stream is not an
    identity camera, and must not be used as one.

    Silent on an inconclusive report (no faces seen): that is a measurement that did not
    happen, and this function refuses cameras, not silence. Callers that need a positive
    answer ask `report.usable`.
    """
    if report.conclusive and not report.usable:
        raise CameraCannotRecognise(refusal(report))

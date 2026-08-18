"""Can this camera ever recognise anybody? A question about optics, not thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from qorgan.config.identity import FaceGate
from qorgan.detection.geometry import Box
from qorgan.identity.camera import (
    CameraCannotRecognise,
    FaceSizeReport,
    measure_faces,
    refuse_if_hopeless,
)


class _Face:
    def __init__(self, width: int, height: int) -> None:
        self.box = Box(0.0, 0.0, float(width), float(height))

    @property
    def width(self) -> int:
        return int(self.box.width)

    @property
    def height(self) -> int:
        return int(self.box.height)


class ScriptedDetector:
    """One list of face sizes per frame."""

    def __init__(self, per_frame: list[list[tuple[int, int]]]) -> None:
        self._per_frame = per_frame
        self._index = 0

    def detect_faces(self, _frame: np.ndarray) -> list[_Face]:
        sizes = self._per_frame[self._index]
        self._index += 1
        return [_Face(width, height) for width, height in sizes]


def _frames(count: int, width: int = 2560, height: int = 1440) -> list[np.ndarray]:
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def test_the_report_counts_every_face_it_saw() -> None:
    detector = ScriptedDetector([[(20, 24), (80, 96)], [], [(45, 54)]])

    report = measure_faces(_frames(3), detector, FaceGate(), source="hall_left")

    assert report.frames == 3
    assert report.faces == 3


def test_a_camera_whose_faces_are_too_small_says_so_in_percent() -> None:
    """**The measurement that ends eighteen threshold-tuning attempts.**

    250 clips of the school hall, 14 970 faces. The numbers depend entirely on WHICH STREAM
    you measure AND AT WHAT RESOLUTION -- which is the whole reason this command exists, and
    why it reads the resolution from the camera's own config instead of assuming one:

      on the 2560x1440 HD burst:    p50 23 px,   p90 45 px,  2.2% clear the 60 px gate
      at the hall's real 1280x720:  p50 11.5px,  max 50 px,  **0 of 14 970** clear the
                                    strict 60 px gate

    (1280x720, not 960x540: `hall.yaml` overrides `base.yaml`'s default.)

    Lower the bar to the 38 px small-face gate and 77 of 14 970 faces get through -- and
    **not one of them is recognised**: the best score among all 77 is 0.350, against a
    min_score of 0.45. So the honest number of RECOGNITIONS is ZERO, an 11-pixel face
    upscaled to ArcFace's 112-pixel input is mush, and even the biggest faces that do clear
    the gate are too degraded to match anybody. No threshold recovers it. The answer is to
    MOVE THE CAMERA (spec §2.4).
    """
    tiny = [[(23, 28)] for _ in range(98)]
    big = [[(80, 96)] for _ in range(2)]
    detector = ScriptedDetector(tiny + big)

    report = measure_faces(_frames(100), detector, FaceGate(), source="hall_left")

    assert report.clearing_gate == 2
    assert report.fraction_clearing == 0.02
    assert report.width_percentile(50) < 60

    summary = report.summary()
    assert "2.0%" in summary
    assert "move the camera" in summary.lower()


def test_a_camera_that_can_actually_see_faces_is_not_told_to_move() -> None:
    detector = ScriptedDetector([[(120, 140)] for _ in range(20)])

    report = measure_faces(_frames(20), detector, FaceGate(), source="canteen_entry")

    assert report.fraction_clearing == 1.0
    assert "move the camera" not in report.summary().lower()


def test_a_camera_that_sees_nobody_does_not_divide_by_zero() -> None:
    detector = ScriptedDetector([[] for _ in range(5)])

    report = measure_faces(_frames(5), detector, FaceGate(), source="yard_entry")

    assert report.faces == 0
    assert report.fraction_clearing == 0.0
    assert "no faces" in report.summary().lower()


def test_the_gate_is_the_config_gate_not_a_number_in_this_module() -> None:
    """The legacy had SIX different minimum-face-size gates. There is one."""
    detector = ScriptedDetector([[(45, 54)]])

    strict = measure_faces(_frames(1), detector, FaceGate(), source="c")
    assert strict.clearing_gate == 0

    detector = ScriptedDetector([[(45, 54)]])
    small = measure_faces(
        _frames(1),
        detector,
        FaceGate(min_width=38, min_height=48, min_area=1800),
        source="c",
    )
    assert small.clearing_gate == 1
    assert isinstance(small, FaceSizeReport)


def test_the_report_says_which_resolution_it_actually_measured_at() -> None:
    """A face size means nothing without the frame it was measured in.

    2.2% and 0% are the SAME 14 970 faces. The only difference is the resolution the
    measurement was taken at, so a report that does not state it is not a report.
    """
    detector = ScriptedDetector([[(37, 45)]])

    report = measure_faces(_frames(1, 960, 540), detector, FaceGate(), source="hall_left/analysis")

    assert report.analysed_at == (960, 540)
    assert "960x540" in report.summary()


def test_two_different_frame_sizes_in_one_report_is_a_loud_error() -> None:
    """Half the frames scaled and half not would average the bug into invisibility."""
    detector = ScriptedDetector([[(37, 45)], [(100, 120)]])
    mixed = [*_frames(1, 960, 540), *_frames(1, 2560, 1440)]

    with pytest.raises(ValueError, match="two different sizes"):
        measure_faces(mixed, detector, FaceGate(), source="hall_left")


def test_the_refusal_names_the_stream_the_gate_and_the_measured_fraction() -> None:
    """The gate: a camera that recognises nobody must not be usable as an identity camera.

    Asserted at startup, not discovered after months of tuning.
    """
    detector = ScriptedDetector([[(37, 45)] for _ in range(40)])

    report = measure_faces(
        _frames(40, 960, 540), detector, FaceGate(), source="hall_left/analysis"
    )

    with pytest.raises(CameraCannotRecognise) as refusal:
        refuse_if_hopeless(report)

    message = str(refusal.value)
    assert "hall_left/analysis" in message  # the STREAM, not the camera
    assert "60" in message and "70" in message  # the gate it failed
    assert "0.0%" in message  # the measured fraction
    assert "0 of 40" in message
    assert "move the camera" in message.lower()
    assert "960x540" in message  # ...at the resolution the worker really feeds it


def test_a_camera_that_can_see_faces_is_not_refused() -> None:
    detector = ScriptedDetector([[(120, 140)] for _ in range(20)])

    report = measure_faces(_frames(20), detector, FaceGate(), source="canteen_entry/analysis")

    refuse_if_hopeless(report)  # does not raise


def test_a_stream_that_saw_no_faces_is_not_silently_approved() -> None:
    """Zero faces is not evidence of a working camera. It is the absence of evidence."""
    detector = ScriptedDetector([[] for _ in range(5)])

    report = measure_faces(_frames(5), detector, FaceGate(), source="yard_entry/analysis")

    assert not report.usable
    assert not report.conclusive

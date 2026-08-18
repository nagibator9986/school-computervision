"""One preprocessing function, called by the bench and by the field.

Speeds are px/SECOND, so a pixel is not a unit of length -- it is a unit of THIS frame.
The harness resized every frame to capture.frame_width x frame_height and production
resized nothing at all, so a threshold tuned on the bench was denominated in different
pixels from the one production compared against. It did not transfer, and nobody would
have noticed: both numbers look like speeds.
"""

from __future__ import annotations

import numpy as np

from qorgan.capture.frames import prepare_frame
from qorgan.config.common import CaptureSettings


def test_a_frame_is_resized_to_the_analysis_resolution() -> None:
    prepared = prepare_frame(np.zeros((1440, 2560, 3), dtype=np.uint8), CaptureSettings())

    assert prepared.shape[:2] == (540, 960), "the NVR's resolution reached the detector"


def test_a_frame_already_at_the_analysis_resolution_is_handed_back_untouched() -> None:
    """A resize per frame per camera is not free, and this is the common case."""
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    assert prepare_frame(image, CaptureSettings()) is image


def test_the_resolution_comes_from_the_camera_not_from_a_constant() -> None:
    capture = CaptureSettings(frame_width=1280, frame_height=720)
    prepared = prepare_frame(np.zeros((1440, 2560, 3), dtype=np.uint8), capture)

    assert prepared.shape[:2] == (720, 1280)


def test_the_worker_and_the_harness_call_the_SAME_function() -> None:
    """Rule R2, checked mechanically -- for PREPROCESSING this time.

    R2 was applied to scoring and stopped there. Two preprocessing paths is two
    detectors, and calibrating one of them tunes a number the other never sees.
    """
    import qorgan.evaluation.video as harness
    import qorgan.worker.camera_loop as worker

    assert worker.prepare_frame is prepare_frame
    assert harness.prepare_frame is prepare_frame

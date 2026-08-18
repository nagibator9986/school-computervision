"""The streams of one camera, and the resolution each one is ACTUALLY analysed at.

A camera is not one thing to measure. It is at least two streams:

  analysis  the low-res substream (channel 102), scaled by `prepare_frame` to
            `capture.frame_width x frame_height` -- **that camera's own**, which is 1280x720
            on `hall` and `canteen_entry` and 960x540 everywhere else. **This is the only
            stream the worker ever runs a detector on, so it is the only stream that decides
            whether recognition is possible.**
  burst     the 2560x1440 HD main stream (channel 101), opened in short bursts for
            evidence. Analysed as decoded, because that is how it is used.

Measure the burst and you get 2.2% of hall faces clearing the 60 px gate. Measure the
stream production analyses and you get **0 of 14 970** clearing it -- and **zero
recognitions** in 14 970 faces. Both numbers are true; only the second is about production.
Reporting one number per CAMERA means picking one of them, and the tempting one is the wrong
one.

**And there is a second way to get it wrong, which has also happened: the right stream at an
assumed resolution.** 960x540 is `base.yaml`'s DEFAULT, not the fleet's analysis resolution;
the hall overrides it to 1280x720. Every face-size figure derived from the default came out
scaled by 0.375 when the truth was 0.5. So `capture` below is not a constant and must never
be treated as one -- it comes from the camera's merged config, per stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from qorgan.capture.frames import prepare_frame
from qorgan.config.common import CaptureSettings

# `open_clip` MOVED to `qorgan.capture.clip` and is not re-exported from here. The worker's
# file-backed camera opens clips too, and a second implementation of "open a recorded file"
# is precisely how the production path and the bench path came to disagree about what a
# frame is -- which is what this module's own docstring is a monument to (R2). One function,
# in the layer that owns frame capture, imported by everyone who needs it.

if TYPE_CHECKING:  # pragma: no cover - import cycle-free typing only
    import cv2

    from qorgan.config.camera import CameraConfig

DEFAULT_FRAMES = 200
DEFAULT_STRIDE = 5


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """One stream, and how it is analysed.

    `capture` is the settings every frame is scaled to before a detector ever sees it --
    or None for a stream that is analysed exactly as it decodes (the HD burst).
    """

    name: str
    burst: bool
    capture: CaptureSettings | None

    @property
    def gates_identity(self) -> bool:
        """Only a stream the worker really runs a detector on may decide the question."""
        return self.capture is not None

    def note(self) -> str:
        """Which of these rows is about production. Exactly one of them is."""
        if self.gates_identity:
            return (
                "^ THIS is the stream the worker analyses. It alone decides whether this "
                "camera can recognise anybody."
            )
        return (
            "^ NOT the stream the worker analyses -- shown for contrast. A face that "
            "clears the gate here clears nothing in production, and reading this row as "
            "the camera's answer is how a misleading 2.2% got written down twice."
        )

    def prepare(self, image: np.ndarray) -> np.ndarray:
        if self.capture is None:
            return image
        return prepare_frame(image, self.capture)


def streams_for(camera: CameraConfig) -> tuple[StreamSpec, ...]:
    """Every stream of a camera worth measuring, analysis first."""
    specs = [StreamSpec(name="analysis", burst=False, capture=camera.capture)]
    if camera.rtsp.burst_path:
        specs.append(StreamSpec(name="burst", burst=True, capture=None))
    return tuple(specs)


def clip_streams(capture: CaptureSettings) -> tuple[StreamSpec, ...]:
    """A recorded clip, measured twice: as it was recorded, and as production would see it.

    The 250 hall clips ARE the HD burst. Measuring them as-recorded is exactly what
    produced the superseded 2.2%; the second row is the one that describes production --
    and it is scaled by the CAPTURE SETTINGS PASSED IN, not by an assumed default, because
    scaling the hall's clips by `base.yaml`'s 960x540 when `hall.yaml` says 1280x720 is the
    other way this measurement has been got wrong.
    """
    return (
        StreamSpec(name="as-recorded", burst=True, capture=None),
        StreamSpec(name="analysis", burst=False, capture=capture),
    )


def sample(
    handle: Any,
    spec: StreamSpec,
    *,
    frames: int = DEFAULT_FRAMES,
    stride: int = DEFAULT_STRIDE,
) -> Iterator[np.ndarray]:
    """Every `stride`-th frame, up to `frames` of them, **scaled to this stream's
    analysis resolution** -- so we span the source rather than measuring one second of it,
    and so nothing downstream can measure the raw decode by accident.
    """
    taken = 0
    index = 0
    while taken < frames:
        ok, image = handle.read()
        if not ok or image is None:
            return
        if index % stride == 0:
            taken += 1
            yield spec.prepare(image)
        index += 1


def open_camera_stream(camera: CameraConfig, spec: StreamSpec) -> cv2.VideoCapture:
    from qorgan.capture.stream import open_rtsp
    from qorgan.rtsp import build_url, safe_url

    # The camera's own rtsp block, so this one-shot pull is bounded by the same
    # open/read timeouts the continuous reader uses. Without it this call would sit on
    # FFmpeg's 30 s default while an operator waits at a web page.
    handle = open_rtsp(build_url(camera.name, camera.rtsp, burst=spec.burst), camera.rtsp)
    if not handle.isOpened():
        # safe_url, never build_url: the real one carries the password (rule R4).
        raise OSError(f"could not open {safe_url(camera.rtsp, burst=spec.burst)}")
    return handle

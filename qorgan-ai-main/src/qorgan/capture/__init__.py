"""Frame capture: RTSP readers, frame-quality checks, and the one preprocessing step."""

from qorgan.capture.frames import prepare_frame
from qorgan.capture.quality import QualityPolicy, is_corrupt
from qorgan.capture.stream import CameraStream, Frame, StreamStats

__all__ = [
    "CameraStream",
    "Frame",
    "QualityPolicy",
    "StreamStats",
    "is_corrupt",
    "prepare_frame",
]

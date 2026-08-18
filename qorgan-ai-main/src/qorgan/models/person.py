"""YOLO person detection and tracking.

Thin by design: everything that *decides* anything lives in `qorgan.detection`, which is
pure and testable without a GPU. This module turns pixels into boxes and nothing else.

**One detector per camera, not one per worker group.** Ultralytics keeps its tracker
state on the model object, so two cameras sharing a model would hand each other's people
the same track ids and the pair analysis would be comparing children from two different
corridors. The weights are ~6 MB, and every camera in a group already shares the
process's CUDA context — which is where the real memory goes (measured: ~155 MB for the
context, against ~6 MB for these weights). So this is cheap, and the alternative is wrong.
"""

from __future__ import annotations

import threading

import numpy as np

from qorgan.config.common import YoloSettings
from qorgan.detection.geometry import Box
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

PERSON_CLASS = 0  # COCO


class PersonDetector:
    """Detects and tracks the people on ONE camera."""

    def __init__(self, settings: YoloSettings, camera: str, device: str = "cuda:0") -> None:
        from ultralytics import YOLO

        self._settings = settings
        self._device = device
        self.camera = camera
        self._model = YOLO(settings.model)
        # The GPU serialises anyway; the lock is here because the camera loop and a burst
        # thread can both reach for the same model.
        self._lock = threading.Lock()
        logger.info(
            "person detector loaded",
            extra={"camera": camera, "model": settings.model, "device": device},
        )

    def detect(self, frame: np.ndarray) -> dict[int, Box]:
        with self._lock:
            results = self._model.track(
                frame,
                imgsz=self._settings.imgsz,
                conf=self._settings.conf,
                iou=self._settings.iou,
                tracker=self._settings.tracker,
                classes=[PERSON_CLASS],
                persist=True,
                verbose=False,
                device=self._device,
            )
        return boxes_from(results)


def boxes_from(results) -> dict[int, Box]:
    """Ultralytics results -> {track_id: Box}. Pure, so it is tested without a GPU."""
    if not results:
        return {}

    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return {}  # people may be present, but nothing is tracked yet

    detections: dict[int, Box] = {}
    for track_id, xyxy in zip(boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=False):
        x1, y1, x2, y2 = xyxy
        detections[int(track_id)] = Box(float(x1), float(y1), float(x2), float(y2))
    return detections

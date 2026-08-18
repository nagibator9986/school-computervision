"""A real frame source: a video file, decoded, with YOLO finding the people.

Kept apart from the harness so that everything *decision-making* stays testable without a
GPU or a video decoder. This module is the only part of the evaluation path that needs
either, and it is deliberately thin.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from qorgan.capture.frames import prepare_frame
from qorgan.config.camera import BullyingCamera
from qorgan.detection.geometry import Box
from qorgan.evaluation.harness import FrameData
from qorgan.logging_setup import get_logger
from qorgan.models.pose import CROP_BUFFER, build_crops

logger = get_logger(__name__)

PERSON_CLASS = 0  # COCO


class VideoSource:
    """Decodes a clip and runs the same YOLO the worker runs, with the same tracker."""

    def __init__(
        self,
        path: Path,
        camera: BullyingCamera,
        *,
        device: str = "cuda:0",
        model: object | None = None,
    ) -> None:
        self.path = path
        self.video_id = path.name
        self.camera = camera
        self.width = camera.capture.frame_width
        self.height = camera.capture.frame_height

        # The configured GPU, not whatever Ultralytics feels like. `PersonDetector`
        # already passes this; the harness did not, and 663 clips is a long time to
        # spend quietly on a CPU.
        self._device = device
        self._model = _load(camera.yolo.model) if model is None else model

        # The same crop buffer, of the same depth, that the worker keeps -- stamped with
        # the same timestamp that ends up on the `Candidate` (see build_crops). The pose
        # model must be handed the same kind of input on the bench as in the field.
        self._recent: deque[tuple[float, np.ndarray]] = deque(maxlen=CROP_BUFFER)

    def crops(self, boxes: tuple[Box, Box], candidate_timestamp: float) -> list[np.ndarray]:
        """The recent prepared frames, cut down to this pair -- with the same builder the
        worker uses (`qorgan.models.pose.build_crops`).

        Raises `CropProvenanceError` if `candidate_timestamp`'s own frame has already
        aged out of the buffer -- e.g. a caller that judged candidates in a batch, after
        the frame loop moved on, rather than in lockstep with it.
        """
        return build_crops(list(self._recent), boxes, candidate_timestamp)

    def __iter__(self) -> Iterator[FrameData]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise OSError(f"cannot open video: {self.path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        skip = self.camera.capture.det_every
        index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return

                index += 1
                if index % skip:
                    continue

                # The timestamp is the clip's OWN clock, not wall time -- the labels are
                # written against the video, so the detector must be too. It is also the
                # SAME value stamped onto the buffered frame below, so a candidate born
                # at this timestamp can find its own frame again later.
                timestamp = index / fps
                yield timestamp, self._detect(timestamp, frame)
        finally:
            capture.release()

    def _detect(self, timestamp: float, frame) -> dict[int, Box]:
        # The SAME preprocessing the worker does (qorgan.capture.frames). Not a resize
        # that happens to have the same numbers in it -- the same function object.
        prepared = prepare_frame(frame, self.camera.capture)
        self._recent.append((timestamp, prepared))
        results = self._model.track(
            prepared,
            imgsz=self.camera.yolo.imgsz,
            conf=self.camera.yolo.conf,
            iou=self.camera.yolo.iou,
            tracker=self.camera.yolo.tracker,
            classes=[PERSON_CLASS],
            persist=True,
            verbose=False,
            device=self._device,
        )
        return _boxes_from(results)


def frame_size(path: Path) -> tuple[int, int]:
    """(width, height) as the container declares it. No frame is decoded.

    `eval scan` asks this of every clip before it scores any of them, to keep a crop ROI
    out of the clips directory (`scan.refuse_crops`). It is a header read, so asking it of
    all 663 clips costs a second and saves a corpus.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def clip_duration(path: Path) -> float:
    """How long this clip is, in seconds. OpenCV only -- no model, no GPU.

    `qorgan eval label` asks this of a SILENT clip and of nothing else. The detector never
    fired on it, so there is no moment to centre a label on and the human judged the whole
    thing: the interval recorded is [0, duration], and this is the duration. It is a header
    read, so the clip is never decoded.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or fps <= 0 or not frames or frames <= 0:
            # Guessing a length here would write a label interval over a span of video
            # nobody watched. Refuse: a wrong interval is worse than a missing one.
            raise OSError(
                f"{path}: cannot read a duration (fps={fps}, frames={frames}). "
                "A whole-clip label needs the clip's real length."
            )
        return frames / fps
    finally:
        capture.release()


def _load(name: str):
    """The real YOLO. Injectable in `VideoSource` only so the device wiring above can be
    tested without a GPU -- there is nothing else to stub here."""
    from ultralytics import YOLO

    return YOLO(name)


def _boxes_from(results) -> dict[int, Box]:
    if not results:
        return {}

    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return {}  # nothing tracked on this frame

    detections: dict[int, Box] = {}
    for track_id, xyxy in zip(boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=False):
        x1, y1, x2, y2 = xyxy
        detections[int(track_id)] = Box(x1, y1, x2, y2)
    return detections

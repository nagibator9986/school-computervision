"""YOLOv8n-pose: pixels in, COCO-17 keypoints out.

The judgement built on those keypoints lives in `qorgan.detection.skeleton`, which is
pure. This module is the only part of the skeleton tier that needs a GPU, and it is
deliberately dull.

Unlike the person detector, ONE pose model is shared by every camera in a worker group:
it carries no tracker state, so there is nothing for two cameras to corrupt.

**Two callers now ask this model different questions, and there is still ONE of it.**
`PoseEstimator` asks "was there aggression in this ~320 px crop of two people?";
`qorgan.classroom` asks "where are the arms of everyone in this whole room?". They differ
only in the pixels handed over and the size the model is asked to look at them in, so
`PoseModel` below owns the weights, the lock and the conversion, and both go through it.

That separation is rule R2, and it is the exact defect the legacy paid for: it carried
`analyze_aggression` in three files that had already drifted, so the eval harness graded
code production did not run. A second pose path here would be the same mistake starting
over -- and it would start plausibly, because the classroom's needs really are different.
They are different INPUTS to one extraction, not a second extraction.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from qorgan.config.bullying import SkeletonSettings
from qorgan.detection.geometry import Box
from qorgan.detection.skeleton import Frame, Keypoints, extract_reasons
from qorgan.detection.validation import SkeletonResult, score_skeleton
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

# The crop is resized to this width before the pose model sees it. Every threshold in
# detection/skeleton.py is a pixel distance in a crop of this width, so changing it here
# silently retunes all five features.
CROP_WIDTH = 320


class PoseModel:
    """The weights, the lock, and one way to turn an image into keypoints.

    Held apart from `PoseEstimator` so that a second caller with a different question --
    the classroom pipeline, which needs every skeleton in a whole frame rather than a
    verdict on a pair -- reuses this rather than loading a second model and growing a
    second extraction beside it (rule R2, and see the module docstring).

    `imgsz` and `conf` are per CALL, not per instance, because they are the two things
    the two callers legitimately disagree about: 320 px over a two-person crop against
    ~960 px over a classroom. Everything that could DRIFT -- which weights, how results
    are unpacked, what a missing confidence array means -- is here and shared.
    """

    def __init__(self, model: str, device: str = "cuda:0") -> None:
        from ultralytics import YOLO

        self.name = model
        self._device = device
        self._model = YOLO(model)
        self._lock = threading.Lock()
        logger.info("pose model loaded", extra={"model": model, "device": device})

    def keypoints(self, image: np.ndarray, *, imgsz: int, conf: float) -> Frame:
        """One image's people, as COCO-17 keypoints. The lock is because the camera loop
        and a burst thread can both reach for the same weights."""
        with self._lock:
            results = self._model.predict(
                image,
                imgsz=imgsz,
                conf=conf,
                device=self._device,
                verbose=False,
            )
        return keypoints_from(results)


class PoseEstimator:
    """Runs YOLOv8n-pose over a sequence of crops and reports what the skeleton saw."""

    def __init__(
        self,
        settings: SkeletonSettings,
        device: str = "cuda:0",
        model: PoseModel | None = None,
    ) -> None:
        self._settings = settings
        # Accepting a ready-made model is how one worker process that watches both a
        # corridor and a classroom loads the weights once. Building one when none is
        # given keeps every existing caller and test unchanged.
        self._pose = model or PoseModel(settings.model, device)

    def validate(self, crops: list[np.ndarray]) -> SkeletonResult:
        """The skeleton's verdict on one candidate pair.

        `skipped` means the model could not look — too few frames, most often. That is
        NOT the same as looking and seeing nothing, and the confidence cap treats the two
        differently: a skipped skeleton is neutral, a disagreeing one is a penalty.
        """
        if not self._settings.enabled:
            return SkeletonResult(skipped=True, skip_reason="disabled")

        if len(crops) < self._settings.min_frames:
            return SkeletonResult(
                skipped=True,
                skip_reason=f"insufficient_frames({len(crops)}<{self._settings.min_frames})",
            )

        sampled = _subsample(crops, self._settings.max_frames)
        try:
            frames = [self._keypoints(crop) for crop in sampled]
        except Exception as exc:
            logger.exception("pose estimation failed")
            return SkeletonResult(skipped=True, skip_reason=f"error:{type(exc).__name__}")

        reasons = extract_reasons(frames)
        return SkeletonResult(
            reasons=tuple(reasons),
            score=score_skeleton(reasons),
            skipped=False,
            skip_reason="",
        )

    def _keypoints(self, crop: np.ndarray) -> Frame:
        return self._pose.keypoints(crop, imgsz=CROP_WIDTH, conf=self._settings.conf)


def keypoints_from(results) -> Frame:
    """Ultralytics pose results -> one frame of people. Pure; tested without a GPU."""
    if not results:
        return []

    pose = results[0].keypoints
    if pose is None or pose.xy is None:
        return []

    xy = pose.xy.cpu().numpy()  # (people, 17, 2)
    conf = (
        pose.conf.cpu().numpy()
        if pose.conf is not None
        else np.ones(xy.shape[:2], dtype=float)
    )

    return [Keypoints(xy=xy[index], conf=conf[index]) for index in range(xy.shape[0])]


def crop_pair(frame: np.ndarray, boxes: tuple[Box, Box], padding: float = 0.15) -> np.ndarray:
    """The region containing both people, padded, resized to CROP_WIDTH.

    Both, not each: `close_upper_body_contact` needs to see two people in one picture,
    and a crop of one child cannot show contact with another.
    """
    import cv2

    height, width = frame.shape[:2]
    a, b = boxes

    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    x2 = max(a.x2, b.x2)
    y2 = max(a.y2, b.y2)

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    region = Box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y).clamped(width, height)

    patch = frame[int(region.y1) : int(region.y2), int(region.x1) : int(region.x2)]
    if patch.size == 0:
        return patch

    scale = CROP_WIDTH / max(patch.shape[1], 1)
    return cv2.resize(patch, (CROP_WIDTH, max(1, int(patch.shape[0] * scale))))


# How many recent frames a candidate's pose crop is built from. Crops, not frames: the
# legacy pinned ~900 MB of 1280x720 frames per queued validation job (H-07).
CROP_BUFFER = 24

# One buffered frame: the timestamp it was captured/decoded at, and its pixels. Both the
# worker and the harness stamp every frame with the SAME timestamp that ends up on the
# `Candidate` it produces, so a candidate's own frame can be found again later by that
# timestamp alone.
StampedFrame = tuple[float, np.ndarray]


class CropProvenanceError(ValueError):
    """The candidate's own frame is no longer in the crop buffer.

    `deque(maxlen=CROP_BUFFER)` evicts its oldest frame every time a new one arrives, with
    no warning. Today every candidate is judged synchronously, in the same loop iteration
    that appended its frame, so the eviction never catches up with a candidate before its
    crops are built. Batch the candidates, judge them after the loop, add a queue that
    drains slowly -- and it will. Silently falling back to whatever frames ARE in the
    buffer would hand the pose model a different moment's pixels; it would find skeletons
    in them anyway and `judge()` would return a confident, silently wrong verdict. Raising
    here is what turns that into a loud failure instead of a plausible one.
    """

    def __init__(self, candidate_timestamp: float, buffered_stamps: list[float]) -> None:
        if buffered_stamps:
            window = f"{min(buffered_stamps):.3f}..{max(buffered_stamps):.3f}"
        else:
            window = "empty"
        super().__init__(
            f"crops requested for the candidate at t={candidate_timestamp:.3f}, but the "
            f"frame buffer no longer holds that frame (buffer covers t={window} over "
            f"{len(buffered_stamps)} frames). The candidate's own frame was evicted "
            "before its crops were built -- refusing to crop the wrong footage."
        )
        self.candidate_timestamp = candidate_timestamp
        self.buffered_stamps = buffered_stamps


def build_crops(
    frames: Sequence[StampedFrame],
    boxes: tuple[Box, Box],
    candidate_timestamp: float,
) -> list[np.ndarray]:
    """The recent frames, each cut down to this pair. Empty crops are dropped.

    Shared, so that the worker's crop stack and the harness's are built the same way. A
    crop built differently is a different input, and a different input is a different
    detector -- which is the whole reason the harness's numbers were worth nothing.

    `candidate_timestamp` ties the crops to the candidate that asked for them: it must
    still be among the buffered frames, or this raises `CropProvenanceError` rather than
    quietly cropping whatever the buffer happens to hold right now.
    """
    stamps = [stamp for stamp, _ in frames]
    if candidate_timestamp not in stamps:
        raise CropProvenanceError(candidate_timestamp, stamps)

    crops = (crop_pair(frame, boxes) for _, frame in frames)
    return [crop for crop in crops if crop.size]


def _subsample(crops: list[np.ndarray], limit: int) -> list[np.ndarray]:
    """Spread the sample across the whole window rather than taking the first N: an
    assault has a beginning and an end, and the fall is usually at the end."""
    if len(crops) <= limit:
        return crops
    step = len(crops) / limit
    return [crops[int(index * step)] for index in range(limit)]

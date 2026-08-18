"""The pose model: pixels in, people out. The only part of the stack that wants a GPU.

Deliberately dull. Everything that *decides* anything lives in `geometry.py` and
`states.py`, which are pure and testable without a graphics card — the judgement is the
part that ends up in front of a psychologist, so it must be arguable without hardware.

**Why YOLO11-pose at imgsz 1280, and not the obvious alternatives.** Measured on this
footage and this machine (Apple M2 Pro, MPS, no CUDA — `MEASUREMENTS.md` §3):

| imgsz | s/frame | people found |
|-------|---------|--------------|
| 960   | —       | 9 (conf tail down to 0.56) |
| 1280  | 0.109   | 9 (all confident) |
| 1920  | 0.172   | 9 |

1280 finds every person 1920 finds, at 1.6x the speed, and 960 starts losing confidence
on the pupil furthest from the lens. The neighbouring codebase's `imgsz: 960` was reasoned
from an imagined room; this is the same question answered by the actual one.

**Tracking is asked for but not relied on, and that is the central design decision.**
Ultralytics keeps tracker state on the model object, so `PoseModel` is one-per-camera and
must not be shared. But a ByteTrack id is not a child: it dies to the first long occlusion
and comes back as a new number, which is exactly why the neighbouring codebase concluded
that per-child classroom metrics were impossible. In a classroom that conclusion is
avoidable, because a classroom has something a corridor does not — **fixed seats**. Track
ids provide short-term continuity between samples; the persistent identity comes from
`room/seats.py`, which assigns observations to seat positions that outlive any track. See
that module for why a seat is a far better proxy for a child than a track.

**Sampling below the video's frame rate weakens the tracker, knowingly.** At 2 fps the
motion between consecutive inputs is 10x what the tracker's defaults assume. For seated
pupils that is harmless — they move a few pixels — and for the one adult who walks it is
not, which is one more reason the seat layer, not the tracker, carries identity.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from classvision.geometry import Keypoints

# COCO person class, in every YOLO detection head.
PERSON_CLASS = 0


@dataclass(frozen=True, slots=True)
class Observation:
    """One person, in one sampled frame."""

    track_id: int | None       # None when the tracker has not committed to an id yet
    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    keypoints: Keypoints
    score: float

    @property
    def box_height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def box_width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where this person meets the floor, roughly.

        Used for zone tests rather than the box centre, because a person's zone is
        decided by where they are STANDING, and the box centre of somebody walking
        toward the lens rises up the frame while their feet do not move.
        """
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])


@dataclass(frozen=True, slots=True)
class Frame:
    """Everything the model saw in one sampled frame."""

    index: int
    video_seconds: float
    people: tuple[Observation, ...]


class PoseModel:
    """One camera's detector+pose+tracker. Never share one between two cameras.

    Ultralytics stores tracker state on the model object, so two cameras sharing an
    instance would hand each other's people the same ids. The weights are small and every
    camera in a process shares the same accelerator context, so one per camera is cheap
    and the alternative is wrong.
    """

    def __init__(self, weights: str = "yolo11m-pose.pt", *, device: str = "mps",
                 imgsz: int = 1280, conf: float = 0.30, iou: float = 0.50,
                 tracker: str = "bytetrack.yaml") -> None:
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self._model = YOLO(weights)
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict:
        """Everything about this model that could change a number. Goes into the artefact
        verbatim, because a metric compared across two weeks under two different imgsz is
        a comparison of settings wearing the costume of a comparison of children."""
        return {
            "weights": self.weights, "device": self.device, "imgsz": self.imgsz,
            "conf": self.conf, "iou": self.iou, "tracker": self.tracker,
        }

    def look(self, image: np.ndarray, index: int, video_seconds: float, *,
             track: bool = True) -> Frame:
        with self._lock:
            if track:
                results = self._model.track(
                    image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    tracker=self.tracker, classes=[PERSON_CLASS], persist=True,
                    device=self.device, verbose=False,
                )
            else:
                results = self._model.predict(
                    image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    classes=[PERSON_CLASS], device=self.device, verbose=False,
                )
        return Frame(index=index, video_seconds=video_seconds,
                     people=people_from(results))

    def reset_tracker(self) -> None:
        """Forget every track id. Call between files — otherwise the second video's first
        pupil inherits the first video's last id, and two lessons silently share a track."""
        with self._lock:
            if getattr(self._model, "predictor", None) is not None:
                trackers = getattr(self._model.predictor, "trackers", None)
                for tracker in trackers or ():
                    tracker.reset()


def people_from(results) -> tuple[Observation, ...]:
    """Ultralytics results -> our own types. Pure, so it is tested without a GPU.

    Kept as a free function for exactly that reason, and because it is the single place
    the library's shape conventions are decoded: one place to fix when they change, and
    one place to read when a number looks wrong.
    """
    if not results:
        return ()
    result = results[0]
    boxes = getattr(result, "boxes", None)
    poses = getattr(result, "keypoints", None)
    if boxes is None or poses is None or poses.xy is None or len(boxes) == 0:
        return ()

    xy = poses.xy.cpu().numpy()
    conf = (poses.conf.cpu().numpy() if poses.conf is not None
            else np.ones(xy.shape[:2], dtype=np.float32))
    xyxy = boxes.xyxy.cpu().numpy()
    scores = (boxes.conf.cpu().numpy() if boxes.conf is not None
              else np.ones(len(xyxy), dtype=np.float32))
    # `boxes.id` is None until the tracker commits — every frame before that, and any
    # detection it has not matched. None is passed through rather than replaced with -1:
    # a sentinel integer id would silently merge every un-tracked person into one track.
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xyxy)

    people = []
    for i in range(min(len(xyxy), xy.shape[0])):
        people.append(Observation(
            track_id=None if ids[i] is None else int(ids[i]),
            box=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
            keypoints=Keypoints(xy=xy[i], conf=conf[i]),
            score=float(scores[i]),
        ))
    return tuple(people)


def weights_present(weights: str) -> bool:
    """Ultralytics downloads missing weights on first use. That is convenient locally and
    unacceptable inside a school with no outbound internet, so the CLI checks first and
    says so plainly rather than hanging on a silent fetch."""
    return Path(weights).exists() or Path.cwd().joinpath(weights).exists()

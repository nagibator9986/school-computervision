"""Getting frames out of a recording, at a chosen rate, with their times attached.

**Sequential decode with `grab()`, never seeking.** Seeking into HEVC costs a keyframe
search and a re-decode of everything since; doing it 15 000 times is far slower than
decoding the file once. Measured on the reference recording: one sequential pass over all
62 901 frames of 2560x1440 HEVC took **107.9 s (583 fps)**, so at any sampling rate we
would plausibly use, decode is not the bottleneck — the pose model is, at 0.109 s/frame.
`grab()` skips the colour conversion for frames we are going to throw away, and
`retrieve()` pays for it only on the ones we keep.

**A sample carries THREE clocks and confuses none of them.** `index` is the frame's
position in the file, `video_seconds` is that divided by the frame rate, and the wall
clock is a separate object (`clock.WallClock`) that the caller applies. They are kept
apart because the failure they prevent is silent: an analysis that stamps its output with
video seconds and a report that reads them as wall time differ by an hour, look
identical, and put a lesson in the wrong week.

**Sampling rate is a decision with a stated cost, not a default to inherit.** Nothing
here picks one. `plan()` exists so the caller can see, before starting, what a rate costs
in frames and roughly in minutes — and so a run's provenance can record the rate it
actually used rather than the rate someone believes it used.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Sample:
    """One decoded frame and where it sits in the file."""

    index: int             # frame number in the file, from 0
    video_seconds: float   # index / fps -- NOT a wall clock
    image: np.ndarray      # BGR


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=Path(path),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        capture.release()


@dataclass(frozen=True, slots=True)
class Plan:
    """What a sampling rate will cost, worked out before anything is decoded."""

    sample_fps: float
    step: int
    samples: int
    decode_seconds: float
    pose_seconds: float

    @property
    def total_minutes(self) -> float:
        return (self.decode_seconds + self.pose_seconds) / 60.0


# Both measured on this machine and this footage (`MEASUREMENTS.md` §3), not looked up:
# a full sequential decode ran at 583 fps, and YOLO11m-pose at imgsz 1280 on MPS took
# 0.109 s per frame. They are here so `plan()` gives a number with a provenance rather
# than an encouraging guess.
DECODE_FPS = 583.0
POSE_SECONDS_PER_FRAME = 0.109


def plan(info: VideoInfo, sample_fps: float) -> Plan:
    step = max(int(round(info.fps / sample_fps)), 1) if sample_fps > 0 else 1
    samples = info.frame_count // step
    return Plan(
        sample_fps=info.fps / step if step else 0.0,
        step=step,
        samples=samples,
        decode_seconds=info.frame_count / DECODE_FPS,
        pose_seconds=samples * POSE_SECONDS_PER_FRAME,
    )


def samples(path: str | Path, sample_fps: float, *,
            start_seconds: float = 0.0,
            end_seconds: float | None = None) -> Iterator[Sample]:
    """Decode once, front to back, yielding every `step`-th frame inside the window.

    `start_seconds` / `end_seconds` are applied by SKIPPING during the same single pass
    rather than by seeking. On a 52-minute file, skipping to the middle costs about a
    second of `grab()` calls; seeking there costs correctness, because OpenCV's HEVC seek
    lands on a keyframe and reports a position that is not the one it landed on. The
    module that decides which week a lesson belongs to should not be built on that.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise ValueError(f"{path} reports no frame rate; cannot sample by time")

    step = max(int(round(fps / sample_fps)), 1) if sample_fps > 0 else 1
    first = int(round(start_seconds * fps))
    last = int(round(end_seconds * fps)) if end_seconds is not None else None

    try:
        index = 0
        while True:
            if last is not None and index > last:
                break
            if not capture.grab():
                break
            if index >= first and (index - first) % step == 0:
                ok, image = capture.retrieve()
                if ok:
                    yield Sample(index=index, video_seconds=index / fps, image=image)
            index += 1
    finally:
        capture.release()


def reference_frame(path: str | Path, at_seconds: float = 0.0) -> np.ndarray:
    """One frame, for calibration. Decoded from the start rather than sought, for the
    reason above — a calibration frame that is not the frame you asked for produces a
    seat map that is quietly a few centimetres wrong for the rest of the term."""
    for sample in samples(path, sample_fps=0.0, start_seconds=at_seconds):
        return sample.image
    raise ValueError(f"{path} has no frame at {at_seconds}s")


def empty_room_frame(path: str | Path, search_seconds: float = 120.0,
                     sample_fps: float = 0.5) -> tuple[np.ndarray, float] | None:
    """The median of the opening minutes: the room with the people averaged out.

    On this recording the room is genuinely empty for the first ~110 s, so the median is
    simply the empty room. On a recording where it is not, a per-pixel median over enough
    spread-out frames still removes anybody who moved and keeps the furniture — which is
    what a seat map needs. It cannot remove somebody who sat still for the whole search
    window, and `room/calibrate.py` is where that gets checked rather than assumed.

    Returns the frame and the number of seconds it was built from, or None if the file
    yielded nothing.
    """
    frames = [s.image for s in samples(path, sample_fps=sample_fps, end_seconds=search_seconds)]
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8), search_seconds

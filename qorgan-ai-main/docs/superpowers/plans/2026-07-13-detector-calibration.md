# Detector Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bullying detector's precision, recall, F1 and — above all — **false alerts per hour** measurable numbers rather than beliefs. Today nobody knows how well it works: not approximately, at all. Every threshold in `config/profiles/*.yaml` is a converted estimate. This plan first makes the bench match the field (B0 — without which no measurement means anything), then builds the corpus and the labelling tools (B1/B2), then reports the curve per camera (B3) and writes down honestly what the data cannot tell us (B4).

**Architecture:** Rule R2 — *the harness runs the production code, not a copy of it* — is currently true of scoring and false of everything else. Three shared seams are created and both callers are made to go through them:

1. `qorgan.capture.frames.prepare_frame()` — the ONE preprocessing step. `worker/camera_loop.py` and `evaluation/video.py` both call it. Today the harness resizes to `capture.frame_width × frame_height` and production resizes not at all, so a px/s threshold tuned on the bench is denominated in different pixels from the one production compares against.
2. `qorgan.models.validate.validate_candidate()` — the ONE crop→pose→judge step, extracted from `worker/bullying.py::_handle`. Today the harness never runs the skeleton, so every verdict is capped at `cap_without_skeleton = 0.72` while Telegram fires at `0.85`: the PR curve is identically empty exactly where the decision lives. Writing a second crop→pose→judge path inside the harness would fix that bug by re-creating the legacy's three-diverged-copies bug, so we extract rather than duplicate.
3. `qorgan.evaluation.clips.parse_clip_name()` — the ONE place a clip's camera is decided, from the clip's own filename. `--camera` picked one config for a whole run; `hall_left` carries a `mirror_ignore` zone over a reflective column that is not in `hall_right`'s field of view, so a global flag silently blanks part of the frame for 344 of the 663 clips.

Then: `eval scan` (detector at threshold 0 → `eval/candidates.csv`), `eval label` (shows the small **crop** — the review view — records against the **full-frame** video_id), `eval sample` (the ~80 non-firing clips, the only recall signal this data can give), `eval run` (per camera, plus false alerts per hour).

**Tech Stack:** Python 3.11, Pydantic v2 (`extra="forbid"`), Ultralytics YOLOv8n + YOLOv8n-pose, OpenCV, pytest, ruff. Venv python is `.venv/Scripts/python.exe`.

---

## Global Constraints

- No file >500 lines.
- No function >50 lines.
- No secret outside env vars.
- No absolute path in the DB.
- Every web endpoint authenticated.
- `extra="forbid"` on config models.
- Baseline 757 tests pass + `ruff check .` clean, keep it that way.
- Venv python is `.venv/Scripts/python.exe`.
- NEVER `git add -A`, NEVER commit anything under `eval/clips/`, `eval/candidates.csv`, `original_student_photos/` or `student_photos/` — these are video and photographs of children; always `git status` first.

---

## Task order

B0 is blocking and strictly first: Tasks 1–6. Nothing measured on a harness that differs from production is worth measuring. Then the corpus and labelling (Tasks 7–9), then the curve (Task 10) and the honest report (Task 11).

---

### Task 1: One `prepare_frame()`, called by the bench and by the field

Spec §3.2. `evaluation/video.py:65` resizes every frame to `capture.frame_width × frame_height`; `worker/camera_loop.py:99` hands YOLO whatever the RTSP substream delivers, unresized. Speeds are px/s. A threshold tuned on the bench is therefore in `capture.frame_width × frame_height` pixels and production is in some-other-resolution pixels, and it does not transfer. This is the same *class* of bug as the `(px/frame)/s` unit bug: a quantity that silently changes meaning between the bench and the field.

**That resolution is PER PROFILE.** 960×540 is only `base.yaml`'s default; `hall.yaml` and `canteen_entry.yaml` override it to **1280×720**. Quote the key, never a blanket number — assuming one number is how the identity spec's face-size figures came out wrong (identity spec §2.4).

It also fixes a latent bug on its own: `BullyingPipeline` already constructs `BullyingDetector(camera.bullying, camera.capture.frame_width, camera.capture.frame_height)` — so the zone maths in production is *already* denominated in the camera's configured frame (1280×720 on the hall) while YOLO's boxes are in native-substream pixels.

**Files:**
- `src/qorgan/capture/frames.py` (new)
- `src/qorgan/capture/__init__.py`
- `src/qorgan/worker/camera_loop.py`
- `src/qorgan/evaluation/video.py`
- `config/base.yaml`
- `config/profiles/hall.yaml`, `config/profiles/stairs.yaml`, `config/profiles/outdoor.yaml`, `config/profiles/canteen_entry.yaml`, `config/profiles/canteen_exit.yaml`, `config/profiles/canteen_inside.yaml`
- `tests/test_prepare_frame.py` (new)
- `tests/test_camera_loop.py`

**Interfaces:**
- Consumes: `qorgan.config.common.CaptureSettings` (`frame_width: int`, `frame_height: int`)
- Produces: `qorgan.capture.frames.prepare_frame(image: np.ndarray, capture: CaptureSettings) -> np.ndarray`
- Produces: `qorgan.capture` re-exports `prepare_frame`
- Changes: `qorgan.worker.camera_loop.CameraLoop._tick` hands `on_frame` and the preview publisher a frame whose image is `prepare_frame(...)` output
- Changes: `qorgan.evaluation.video.VideoSource._detect` calls `prepare_frame(...)` instead of `cv2.resize(...)`

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_prepare_frame.py`:

```python
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
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_prepare_frame.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'qorgan.capture.frames'` (collection error, 0 tests run).

- [ ] **Step 3: Write `src/qorgan/capture/frames.py`.**

```python
"""The ONE preprocessing step between a decoded frame and the detector.

Production used to hand YOLO whatever the RTSP substream happened to deliver, while
`evaluation/video.py` resized every frame to `capture.frame_width x frame_height`.
Speeds are px/SECOND, so a threshold tuned on the bench was in THAT frame's pixels and
production was in some-other-resolution pixels. It did not transfer -- the same *class*
of bug as the (px/frame)/s unit bug: a quantity that silently changes meaning between
the bench and the field.

Note that the frame is PER PROFILE, not one number: 960x540 is `base.yaml`'s default and
`hall.yaml` / `canteen_entry.yaml` override it to 1280x720. So there is no single "analysis
resolution" to quote -- only `capture.frame_width x frame_height`, of the camera in hand.

Rule R2 was applied to *scoring* and stopped there. Preprocessing is the other half of
"the same functions".

**Consequence, and it is not small.** Every speed and acceleration threshold -- and the
measured noise floor -- is now pinned to `capture.frame_width x frame_height`. Changing
that key silently invalidates every threshold in every profile, because px/s means
something different in a different frame. The config files say so, at the key itself.
"""

from __future__ import annotations

import cv2
import numpy as np

from qorgan.config.common import CaptureSettings


def prepare_frame(image: np.ndarray, capture: CaptureSettings) -> np.ndarray:
    """A decoded frame, at the resolution the thresholds were tuned in.

    The worker calls this. The eval harness calls this. Nothing else preprocesses.
    """
    height, width = image.shape[:2]
    if (width, height) == (capture.frame_width, capture.frame_height):
        return image  # already there; do not pay for a copy
    return cv2.resize(image, (capture.frame_width, capture.frame_height))
```

- [ ] **Step 4: Export it from `src/qorgan/capture/__init__.py`.** Replace the whole file:

```python
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
```

- [ ] **Step 5: Make the worker call it.** In `src/qorgan/worker/camera_loop.py`, add to the imports:

```python
from dataclasses import replace
```

and change the `qorgan.capture` import line to:

```python
from qorgan.capture import CameraStream, Frame, prepare_frame
```

then replace `_tick` in full:

```python
    def _tick(self, skip: int) -> None:
        frame = self._stream.read(timeout=1.0)
        if frame is None:
            return

        self.frames_processed += 1

        # ONE preprocessing function, shared with the eval harness (qorgan.capture.frames).
        # Production used to hand YOLO whatever the NVR sent while the harness resized to
        # capture.frame_width x frame_height -- so a px/s threshold tuned on the bench was
        # denominated in different pixels from the one production compared against.
        # BullyingDetector was ALREADY constructed with frame_width/frame_height, so the
        # zone maths was in the camera's configured frame (1280x720 on the hall) while the
        # boxes were in substream pixels.
        prepared = replace(frame, image=prepare_frame(frame.image, self.camera.capture))

        # det_every is honoured here, for every camera type. The legacy shipped
        # det_every: 2 on the canteen cameras with a comment about halving GPU load,
        # and then never read the key.
        status = "ok"
        if prepared.seq % skip == 0:
            status = self._on_frame(self.camera, prepared)

        self._publisher.publish(
            self.camera.name, prepared.image, self.camera.preview, status=status
        )
```

- [ ] **Step 6: Make the harness call it.** In `src/qorgan/evaluation/video.py`, add to the imports:

```python
from qorgan.capture.frames import prepare_frame
```

and replace `_detect`:

```python
    def _detect(self, frame) -> dict[int, Box]:
        # The SAME preprocessing the worker does (qorgan.capture.frames). Not a resize
        # that happens to have the same numbers in it -- the same function object.
        prepared = prepare_frame(frame, self.camera.capture)
        results = self._model.track(
            prepared,
            imgsz=self.camera.yolo.imgsz,
            conf=self.camera.yolo.conf,
            iou=self.camera.yolo.iou,
            tracker=self.camera.yolo.tracker,
            classes=[PERSON_CLASS],
            persist=True,
            verbose=False,
        )
        return _boxes_from(results)
```

- [ ] **Step 7: Run the new test — it passes.**

```
.venv/Scripts/python.exe -m pytest tests/test_prepare_frame.py -q
```

Expected: `4 passed`.

- [ ] **Step 8: Pin the behaviour end-to-end in the camera loop.** Append to `tests/test_camera_loop.py`:

```python
def test_the_detector_sees_the_analysis_resolution_whatever_the_nvr_sends(
    settings: Settings, address: str
) -> None:
    """The fake camera delivers 64x64. The detector must still see the camera's OWN
    configured analysis frame -- 960x540 for this default-profile camera, though `hall.yaml`
    and `canteen_entry.yaml` override that to 1280x720.

    Every px/s threshold in every profile is denominated in THIS frame. If the loop
    hands the detector the substream's own resolution, the thresholds are being compared
    against speeds measured in a different pixel, and no amount of tuning fixes that.
    """
    shapes: list[tuple[int, int]] = []

    def detector(camera, frame) -> str:
        shapes.append(frame.image.shape[:2])
        return "ok"

    _loop, _preview, _ = _run_loop(_camera(), address, on_frame=detector)

    assert shapes, "the detector never ran"
    assert all(shape == (540, 960) for shape in shapes), (
        f"the detector was handed {shapes[0]} -- the NVR's resolution, not the analysis one"
    )
```

- [ ] **Step 9: Write the warning where the next person will find it.** In `config/base.yaml`, replace the `capture:` block:

```yaml
capture:
  # THE ANALYSIS RESOLUTION. Both the worker (worker/camera_loop.py) and the eval
  # harness (evaluation/video.py) resize every frame to exactly this, through the one
  # shared function qorgan.capture.frames.prepare_frame().
  #
  # !!  CHANGING frame_width / frame_height INVALIDATES EVERY SPEED AND ACCELERATION
  # !!  THRESHOLD IN EVERY PROFILE, AND THE MEASURED NOISE FLOOR WITH THEM.
  #
  # Speeds are px/SECOND. A pixel is not a unit of length; it is a unit of THIS frame.
  # Halve the width and every speed halves, while every threshold stays where it was.
  # If you change these, you must re-run `qorgan eval run` and `qorgan eval noise-floor`
  # and re-derive the profiles. There is no shortcut and no conversion factor.
  frame_width: 960
  frame_height: 540
  display_fps: 8
  det_every: 1
```

- [ ] **Step 10: Add the same warning to every profile header.** Insert this block immediately under the opening comment of `config/profiles/hall.yaml`, `stairs.yaml`, `outdoor.yaml`, `canteen_entry.yaml`, `canteen_exit.yaml`, `canteen_inside.yaml`:

```yaml
# ---------------------------------------------------------------------------
# ANALYSIS RESOLUTION. Every frame reaching the detector on this camera type is resized
# to capture.frame_width x frame_height (base.yaml, or the override below) by the one
# shared qorgan.capture.frames.prepare_frame(). Every px/s and px/s^2 threshold in this
# file -- and the noise floor they were checked against -- is expressed in pixels of THAT
# frame. Change the resolution and every one of them is void.
# ---------------------------------------------------------------------------
```

- [ ] **Step 11: Full suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `762 passed` (757 + 4 new in `test_prepare_frame.py` + 1 new in `test_camera_loop.py`), `All checks passed!`.

- [ ] **Step 12: Commit.**

```
git status
git add src/qorgan/capture/frames.py src/qorgan/capture/__init__.py src/qorgan/worker/camera_loop.py src/qorgan/evaluation/video.py config/base.yaml config/profiles/hall.yaml config/profiles/stairs.yaml config/profiles/outdoor.yaml config/profiles/canteen_entry.yaml config/profiles/canteen_exit.yaml config/profiles/canteen_inside.yaml tests/test_prepare_frame.py tests/test_camera_loop.py
git commit -m "One prepare_frame(), shared by the bench and the field

Production handed YOLO whatever the substream sent; the harness resized to
capture.frame_width x frame_height. Speeds are px/s, so a threshold tuned on one is
meaningless on the other -- the same class of bug as (px/frame)/s. Both now call one
function. (That frame is per-profile: 960x540 by default, 1280x720 on hall and
canteen_entry -- so there is no single number to hard-code.)

Every speed threshold and the noise floor are now pinned to capture.frame_width;
the config says so at the key and in every profile header."
```

---

### Task 2: The harness runs the skeleton — through ONE shared crop→pose→judge

Spec §3.1. `harness.run`'s `skeleton` parameter defaults to `no_skeleton` and `evaluation/cli.py:160` never passes one, so `validation_score` is always 0.0, every verdict is capped at `cap_without_skeleton = 0.72`, and `Alert.notified` is always `False`. The Telegram threshold is `0.85`. **The PR curve is identically empty above 0.72** — empty exactly where the decision lives.

The obvious fix is a trap: a fresh crop→pose→judge path inside the harness would re-create the legacy's three-diverged-copies bug *while fixing a bug caused by it*. So the step is **extracted** out of `worker/bullying.py::_handle` (line 151) into one function both callers import.

**Files:**
- `src/qorgan/models/validate.py` (new)
- `src/qorgan/models/pose.py`
- `src/qorgan/worker/bullying.py`
- `src/qorgan/evaluation/harness.py`
- `src/qorgan/evaluation/video.py`
- `src/qorgan/evaluation/cli.py`
- `src/qorgan/evaluation/__init__.py`
- `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `qorgan.detection.pipeline.Candidate` (`key`, `timestamp`, `score`, `probability`, `boxes`, `center`, …)
- Consumes: `qorgan.detection.validation.judge(candidate_probability: float, validation_score: float, skeleton: SkeletonResult, config: ConfidenceConfig) -> Verdict`
- Consumes: `qorgan.config.bullying.Confidence` (imported into `detection.validation` as `ConfidenceConfig`)
- Produces: `qorgan.models.pose.CROP_BUFFER: int = 24`
- Produces: `qorgan.models.pose.build_crops(frames: Sequence[np.ndarray], boxes: tuple[Box, Box]) -> list[np.ndarray]`
- Produces: `qorgan.models.validate.SkeletonView` — Protocol: `validate(self, crops: list[np.ndarray]) -> SkeletonResult`
- Produces: `qorgan.models.validate.NoPose` and the singleton `NO_POSE: NoPose`
- Produces: `qorgan.models.validate.validate_candidate(candidate: Candidate, crops: list[np.ndarray], pose: SkeletonView, config: Confidence) -> Verdict`
- Changes: `qorgan.evaluation.harness.FrameSource` gains `crops(self, boxes: tuple[Box, Box]) -> list[np.ndarray]`
- Changes: `qorgan.evaluation.harness.run(source: FrameSource, config: BullyingConfig, *, pose: SkeletonView = NO_POSE) -> RunResult` (replaces `skeleton: SkeletonFn = no_skeleton`; `SkeletonFn` and `no_skeleton` are deleted)
- Changes: `qorgan.evaluation.video.VideoSource` gains `crops(...)` and a `deque` of the last `CROP_BUFFER` prepared frames
- Changes: `qorgan.evaluation.cli` gains `--device` (default `cuda:0`) on `run`/`gate`/`save-baseline` and builds a real `PoseEstimator`

**Steps:**

- [ ] **Step 1: Write the failing tests.** In `tests/test_evaluation.py`, add `SpyPose` and the two new tests, and update the three existing call sites that pass `skeleton=`.

Replace the import line `from qorgan.evaluation.harness import no_skeleton` with:

```python
from qorgan.models.validate import NO_POSE, validate_candidate
```

Add, immediately below the `SyntheticClip` class:

```python
FALL = ["body_fall_or_low_posture", "close_upper_body_contact", "rapid_hand_motion"]


class SpyPose:
    """A pose model that records that it was asked, and says it saw a fall.

    No GPU, no crops, no video: the DECISION is what is under test.
    """

    def __init__(self) -> None:
        self.calls = 0

    def validate(self, crops) -> SkeletonResult:
        self.calls += 1
        return SkeletonResult(tuple(FALL), score_skeleton(FALL), skipped=False, skip_reason="")
```

Then replace `test_without_a_skeleton_the_event_is_logged_but_nobody_is_notified`, `test_with_a_confirming_skeleton_the_assault_is_notified`, `test_one_fight_produces_one_notification_not_forty` and `test_the_harness_scores_itself_against_labels` so that they pass `pose=` instead of `skeleton=`:

```python
def test_without_a_pose_model_the_event_is_logged_but_nobody_is_notified() -> None:
    """The cap, end to end.

    If the pose model could not look, the confidence is capped below the notify bar and
    nobody is woken up -- however sure the heuristics were. But the incident is still
    RECORDED, because a suspicious event a human should review is not the same thing as
    no event at all.
    """
    config = BullyingConfig()
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), config, pose=NO_POSE)

    assert result.alerts, "the incident was not even logged"
    assert all(not alert.notified for alert in result.alerts), "an unconfirmed event notified"
    assert all(
        alert.verdict.confidence <= config.confidence.cap_without_skeleton
        for alert in result.alerts
    )


def test_the_harness_actually_runs_the_skeleton() -> None:
    """**The bug that made the PR curve a fiction.**

    `harness.run`'s skeleton defaulted to `no_skeleton` and the CLI never passed one, so
    validation_score was always 0.0, every verdict was capped at cap_without_skeleton
    (0.72), and Alert.notified was always False. The Telegram threshold is 0.85. The
    curve was identically empty above 0.72 -- empty exactly where the decision lives.
    """
    config = BullyingConfig()
    spy = SpyPose()

    result = run(SyntheticClip("fight.mp4", scene_an_assault()), config, pose=spy)

    assert spy.calls > 0, "the harness never asked the pose model anything"
    assert any(
        alert.verdict.confidence > config.confidence.cap_without_skeleton
        for alert in result.alerts
    ), "every verdict is still capped at 0.72; nothing above the notify bar is reachable"
    assert any(alert.notified for alert in result.alerts), "no alert production would send"


def test_with_a_confirming_skeleton_the_assault_is_notified() -> None:
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig(), pose=SpyPose())

    assert result.alerts, "no alert at all"
    assert any(alert.notified for alert in result.alerts), "a confirmed fall did not notify anyone"


def test_one_fight_produces_one_notification_not_forty() -> None:
    """Merging, end to end. Dozens of Telegram messages for one shove trains staff to
    ignore the alerts, which is worse than having none."""
    result = run(
        SyntheticClip("fight.mp4", scene_an_assault(frames=60)),
        BullyingConfig(),
        pose=SpyPose(),
    )

    notified = [a for a in result.alerts if a.notified]
    assert len(notified) == 1, f"one incident produced {len(notified)} notifications"


def test_the_harness_scores_itself_against_labels(tmp_path: Path) -> None:
    """The whole loop: run the production detector over a labelled clip, and get a
    precision/recall/F1 back. This is the thing the legacy project never had."""
    labels = load_labels(_labels_file(tmp_path, "fight.mp4,0.5,3.5,bullying\n"))
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig(), pose=SpyPose())

    metrics = evaluate(labels, result.predictions)

    assert metrics.true_positives == 1, f"the labelled fight was not found: {metrics.summary()}"
    assert metrics.recall == 1.0


def test_the_worker_and_the_harness_judge_through_the_same_function() -> None:
    """Rule R2, checked mechanically -- for the SLOW TIER this time.

    The legacy had `analyze_aggression` in three files that had already diverged. Fixing
    "the harness never runs the skeleton" by writing a second crop->pose->judge path
    inside the harness would have fixed that bug by re-creating the one that caused it.
    """
    import qorgan.evaluation.harness as harness
    import qorgan.worker.bullying as worker

    assert worker.validate_candidate is validate_candidate
    assert harness.validate_candidate is validate_candidate
```

Also give `SyntheticClip` the new `crops` method (a scripted scene has no pixels):

```python
    def crops(self, _boxes):
        """A scripted scene has no pixels. The pose model is then handed nothing, which
        is exactly what `PoseEstimator` calls `skipped` -- and the cap applies."""
        return []
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_evaluation.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'qorgan.models.validate'` (collection error).

- [ ] **Step 3: Add the shared crop builder to `src/qorgan/models/pose.py`.** Change the imports at the top of the file:

```python
from collections.abc import Sequence
```

and append, after `crop_pair`:

```python
# How many recent frames a candidate's pose crop is built from. Crops, not frames: the
# legacy pinned ~900 MB of 1280x720 frames per queued validation job (H-07).
CROP_BUFFER = 24


def build_crops(frames: Sequence[np.ndarray], boxes: tuple[Box, Box]) -> list[np.ndarray]:
    """The recent frames, each cut down to this pair. Empty crops are dropped.

    Shared, so that the worker's crop stack and the harness's are built the same way. A
    crop built differently is a different input, and a different input is a different
    detector -- which is the whole reason the harness's numbers were worth nothing.
    """
    crops = (crop_pair(frame, boxes) for frame in frames)
    return [crop for crop in crops if crop.size]
```

- [ ] **Step 4: Write `src/qorgan/models/validate.py`.**

```python
"""crop -> pose -> judge. ONE function. The worker calls it; the harness calls it.

The harness defaulted its `skeleton` parameter to `no_skeleton` and the CLI never passed
one, so `validation_score` was always 0.0, every verdict was capped at
`cap_without_skeleton` (0.72), and `Alert.notified` was always False. The Telegram
threshold is 0.85, so the PR curve was identically empty above 0.72 -- empty exactly
where the decision lives, and every threshold "calibrated" on it would have been tuned
against a number that could never be reached.

**And the fix contains a trap.** Writing a fresh crop->pose->judge path inside the
harness would re-create the legacy's three-diverged-copies bug *while fixing a bug caused
by it*. So the step lives here, once, and both callers import it. Rule R2, enforced
rather than restated.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from qorgan.config.bullying import Confidence
from qorgan.detection.pipeline import Candidate
from qorgan.detection.validation import SkeletonResult, Verdict, judge


class SkeletonView(Protocol):
    """Anything that can look at a stack of crops and say what the skeleton saw.

    `qorgan.models.pose.PoseEstimator` in production; a stub in a test. That is the
    point: the DECISION is testable without a GPU, and the decision is what we tune.
    """

    def validate(self, crops: list[np.ndarray]) -> SkeletonResult: ...


class NoPose:
    """No pose model at all.

    Everything is then capped below the notify bar, which is the correct and conservative
    default -- and, until this module existed, the only thing the harness ever did.
    """

    def validate(self, _crops: list[np.ndarray]) -> SkeletonResult:
        return SkeletonResult(skipped=True, skip_reason="not_configured")


NO_POSE = NoPose()


def validate_candidate(
    candidate: Candidate,
    crops: list[np.ndarray],
    pose: SkeletonView,
    config: Confidence,
) -> Verdict:
    """The slow tier's judgement on one candidate: look, score, blend, cap.

    Note what is NOT decided here. A confirmed candidate becomes an event *whatever* its
    confidence: a suspicious incident the skeleton could not confirm still belongs in the
    log for a human to review. Only the NOTIFICATION is gated on confidence.
    """
    skeleton = pose.validate(crops)
    return judge(
        candidate_probability=candidate.probability,
        validation_score=skeleton.score if not skeleton.skipped else 0.0,
        skeleton=skeleton,
        config=config,
    )
```

- [ ] **Step 5: Make the worker call it.** In `src/qorgan/worker/bullying.py`:

Replace the two model imports:

```python
from qorgan.models.person import PersonDetector
from qorgan.models.pose import CROP_BUFFER, PoseEstimator, build_crops
from qorgan.models.validate import validate_candidate
```

Delete the now-duplicated module constant (`CROP_BUFFER = 24` and its comment) and the `judge` / `Verdict` import becomes `from qorgan.detection.validation import Verdict` (`judge` is no longer called here).

Replace the crop construction inside `_enqueue`:

```python
    def _enqueue(self, candidate: Candidate, frame: Frame) -> None:
        """Hand a candidate to the slow tier. A full queue is an incident, not a shrug."""
        job = ValidationJob(
            candidate=candidate,
            # The SAME crop builder the harness uses. Crops, not frames: the legacy
            # pinned ~900 MB of HD frames per queued job (H-07).
            crops=build_crops(list(self._recent), candidate.boxes),
            snapshot=frame.image.copy(),
        )
        try:
            self._queue.put_nowait(job)
        except Full:
            # The legacy did `except Full: pass` here and a confirmed assault vanished
            # without a trace. If we ever drop one, it will be in the log in capitals.
            logger.error(
                "VALIDATION QUEUE FULL — a confirmed candidate was DROPPED",
                extra={
                    "camera": self.camera.name,
                    "pair": candidate.key,
                    "score": round(candidate.score, 2),
                },
            )
```

Replace the first six lines of `_handle`:

```python
    def _handle(self, job: ValidationJob) -> None:
        candidate = job.candidate
        confidence_config = self.camera.bullying.confidence

        # crop -> pose -> judge, through the ONE function the eval harness also calls
        # (qorgan.models.validate). Not a copy of it. See that module's docstring.
        verdict = validate_candidate(candidate, job.crops, self._pose, confidence_config)

        severity = severity_for(
            verdict.confidence,
            confidence_config.alert_threshold,
            confidence_config.critical_threshold,
        )
        summary = summarise(severity, self.camera.display_name, verdict.confidence)

        decision = self._merger.decide(candidate.key, candidate.timestamp, candidate.center)
        if not decision.is_new:
            self._merge(decision.merged_into, verdict, severity, summary)
            return

        self._create(job, verdict, severity, summary)
```

- [ ] **Step 6: Make the harness call it.** In `src/qorgan/evaluation/harness.py`, replace the imports and everything from `class FrameSource` to the end of the file:

```python
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qorgan.config.bullying import BullyingConfig
from qorgan.detection.geometry import Box
from qorgan.detection.merging import EventMerger
from qorgan.detection.pipeline import BullyingDetector, Candidate
from qorgan.detection.validation import Verdict
from qorgan.evaluation.metrics import Prediction
from qorgan.models.validate import NO_POSE, SkeletonView, validate_candidate

# One frame of a clip: when it happened, and who was in it.
FrameData = tuple[float, dict[int, Box]]


class FrameSource(Protocol):
    """Something that yields detections over time. A video plus YOLO, or a fixture."""

    video_id: str
    width: int
    height: int

    def __iter__(self) -> Iterator[FrameData]: ...

    def crops(self, boxes: tuple[Box, Box]) -> list[np.ndarray]:
        """The recent frames, cut down to this pair, for the pose model.

        A fixture has no pixels and returns []. `PoseEstimator` then reports `skipped`,
        and the cap applies -- which is exactly what a GPU-free decision test wants.
        """
        ...


@dataclass(frozen=True, slots=True)
class Alert:
    """What the worker would actually have recorded and sent."""

    video_id: str
    timestamp: float
    key: tuple[int, int]
    verdict: Verdict
    merged: bool
    notified: bool

    def as_prediction(self) -> Prediction:
        return Prediction(
            video_id=self.video_id,
            timestamp=self.timestamp,
            confidence=self.verdict.confidence,
        )


@dataclass
class RunResult:
    video_id: str
    alerts: list[Alert]
    frames: int
    candidates: int
    suppressed_by: dict[str, int]

    @property
    def predictions(self) -> list[Prediction]:
        """Only unmerged alerts count: a merged one raised no notification, so it is not
        a thing the school ever saw."""
        return [a.as_prediction() for a in self.alerts if not a.merged]


def run(
    source: FrameSource,
    config: BullyingConfig,
    *,
    pose: SkeletonView = NO_POSE,
) -> RunResult:
    """Replay one clip through the real detector, the real validator, and the real merger.

    `pose` defaults to NO_POSE, which caps every verdict at 0.72. That default is safe
    but it is NOT a benchmark: `evaluation/cli.py` passes a real `PoseEstimator`, because
    the Telegram threshold is 0.85 and a curve that stops at 0.72 measures nothing.
    """
    detector = BullyingDetector(config, source.width, source.height)
    merger = EventMerger(config.event_merge)

    alerts: list[Alert] = []
    suppressed: dict[str, int] = {}
    frames = 0
    candidates = 0

    for timestamp, detections in source:
        frames += 1
        result = detector.process(detections, timestamp)

        for gate, count in result.suppressed_by.items():
            suppressed[gate] = suppressed.get(gate, 0) + count

        for candidate in result.candidates:
            candidates += 1
            crops = source.crops(candidate.boxes)
            alerts.append(_judge(source.video_id, candidate, config, merger, pose, crops))

    return RunResult(
        video_id=source.video_id,
        alerts=alerts,
        frames=frames,
        candidates=candidates,
        suppressed_by=suppressed,
    )


def _judge(
    video_id: str,
    candidate: Candidate,
    config: BullyingConfig,
    merger: EventMerger,
    pose: SkeletonView,
    crops: list[np.ndarray],
) -> Alert:
    """The slow tier, exactly as the worker runs it -- through the SAME function object
    the worker calls (`qorgan.models.validate.validate_candidate`), not a copy of it."""
    verdict = validate_candidate(candidate, crops, pose, config.confidence)

    decision = merger.decide(candidate.key, candidate.timestamp, candidate.center)
    merger.remember(candidate.key, candidate.timestamp, verdict.confidence, candidate.center)

    return Alert(
        video_id=video_id,
        timestamp=candidate.timestamp,
        key=candidate.key,
        verdict=verdict,
        merged=not decision.is_new,
        # A merged event raises no second notification: one incident, one message.
        notified=decision.is_new and verdict.should_notify(config.confidence),
    )
```

Keep the module's existing top docstring, but replace its third paragraph (the one beginning "It also runs the **full** pipeline") with:

```
It also runs the **full** pipeline, including the skeleton validation tier and the
confidence cap — through `qorgan.models.validate.validate_candidate`, the same function
the worker's slow tier calls. Until this existed the harness never ran the skeleton at
all, so every verdict was capped at 0.72 while Telegram fires at 0.85, and the PR curve
was empty exactly where the decision lives.
```

- [ ] **Step 7: Give `VideoSource` a crop buffer.** In `src/qorgan/evaluation/video.py`, add to the imports:

```python
from collections import deque

import numpy as np

from qorgan.models.pose import CROP_BUFFER, build_crops
```

then add the buffer to `__init__` (after `self._model = YOLO(camera.yolo.model)`):

```python
        # The same crop buffer, of the same depth, that the worker keeps. The pose model
        # must be handed the same kind of input on the bench as in the field.
        self._recent: deque[np.ndarray] = deque(maxlen=CROP_BUFFER)
```

append the buffer write at the end of `_detect`, and add `crops`:

```python
    def crops(self, boxes: tuple[Box, Box]) -> list[np.ndarray]:
        """The recent prepared frames, cut down to this pair -- with the same builder the
        worker uses (`qorgan.models.pose.build_crops`)."""
        return build_crops(list(self._recent), boxes)
```

`_detect` becomes:

```python
    def _detect(self, frame) -> dict[int, Box]:
        prepared = prepare_frame(frame, self.camera.capture)
        self._recent.append(prepared)
        results = self._model.track(
            prepared,
            imgsz=self.camera.yolo.imgsz,
            conf=self.camera.yolo.conf,
            iou=self.camera.yolo.iou,
            tracker=self.camera.yolo.tracker,
            classes=[PERSON_CLASS],
            persist=True,
            verbose=False,
        )
        return _boxes_from(results)
```

- [ ] **Step 8: Make the CLI pass a real pose model.** In `src/qorgan/evaluation/cli.py`, add `--device` to `_common`:

```python
def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--clips", type=Path, default=CLIPS_DIR)
    parser.add_argument(
        "--camera",
        default="hall_left",
        help="which camera's tuning to evaluate with (the clips came from one)",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="where YOLO and the pose model run. 663 clips is a long time on a CPU.",
    )
```

and replace `_evaluate`, adding `_pose`:

```python
def _evaluate(args: argparse.Namespace) -> tuple[LabelSet, list[RunResult]]:
    from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

    labels = load_labels(args.labels)
    camera = _camera(args.camera)
    pose = _pose(camera, args.device)

    results = []
    for video_id in labels.videos:
        clip = args.clips / video_id
        if not clip.is_file():
            print(f"  ! missing clip, skipping: {clip}")
            continue
        print(f"  running {video_id} ...")
        results.append(run(VideoSource(clip, camera), camera.bullying, pose=pose))

    if not results:
        raise SystemExit(f"no clips found in {args.clips}. Nothing to measure.")
    return labels, results


def _pose(camera: BullyingCamera, device: str) -> SkeletonView:
    """The REAL pose model.

    Without it, `validation_score` is 0.0, every verdict is capped at 0.72, and the PR
    curve is empty above it -- which is where the notify threshold (0.85) lives. A
    benchmark that cannot produce a single alert production would send is not a benchmark.
    """
    from qorgan.models.pose import PoseEstimator  # imports ultralytics; keep it lazy

    return PoseEstimator(camera.bullying.skeleton, device=device)
```

with the new import at the top of the file:

```python
from qorgan.models.validate import SkeletonView
```

- [ ] **Step 9: Export the new names.** In `src/qorgan/evaluation/__init__.py`, nothing to add (the harness's public surface is unchanged), but confirm `no_skeleton` is not exported anywhere:

```
.venv/Scripts/python.exe -m grep -h 2>/dev/null; git grep -n "no_skeleton\|SkeletonFn"
```

Expected: no output. If any remains, delete it.

- [ ] **Step 10: Run the suite.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `764 passed` (762 + 2 net new in `test_evaluation.py`), `All checks passed!`.

- [ ] **Step 11: Commit.**

```
git status
git add src/qorgan/models/validate.py src/qorgan/models/pose.py src/qorgan/worker/bullying.py src/qorgan/evaluation/harness.py src/qorgan/evaluation/video.py src/qorgan/evaluation/cli.py tests/test_evaluation.py
git commit -m "The harness runs the skeleton -- through the worker's own function

harness.run defaulted skeleton=no_skeleton and the CLI never passed one, so
every verdict capped at 0.72 while Telegram fires at 0.85: the PR curve was
identically empty exactly where the decision lives.

Fixed by EXTRACTING crop->pose->judge out of worker/bullying._handle into
models/validate.validate_candidate, which both the worker and the harness now
call. Writing a second path in the harness would have fixed the bug by
recreating the one that caused it."
```

---

### Task 3: The clip's camera comes from the clip's filename

Spec §3.3. `eval run --camera` picks **one** camera config for an entire run. The corpus is 344 `hall_right` + 299 `hall_left`, and `hall_left` carries a `mirror_ignore` zone over a **reflective column that does not exist in `hall_right`'s field of view** (`config/cameras/hall_left.yaml` masks `x 0.52–0.56`; `hall_right.yaml` masks `x 0.34–0.38`). Evaluating one camera's footage against the other's zones silently blanks out part of the frame — and a silently blanked frame is a lower recall number that looks like a tuning result.

An un-inferable filename is a **hard error**, not a default.

**Files:**
- `src/qorgan/evaluation/clips.py` (new)
- `src/qorgan/evaluation/cli.py`
- `src/qorgan/evaluation/__init__.py`
- `tests/test_eval_clips.py` (new)

**Interfaces:**
- Consumes: `qorgan.config.loader.load_cameras() -> dict[str, CameraConfig]`
- Consumes: `qorgan.config.camera.BullyingCamera`
- Produces: `qorgan.evaluation.clips.ClipNameError(Exception)`
- Produces: `qorgan.evaluation.clips.ClipName` — frozen dataclass: `filename: str`, `camera: str`, `track_a: int`, `track_b: int`, `recorded_at: datetime`, `is_burst: bool`; property `pair -> tuple[int, int]`
- Produces: `qorgan.evaluation.clips.parse_clip_name(filename: str) -> ClipName` (raises `ClipNameError`)
- Produces: `qorgan.evaluation.clips.camera_for(filename: str, cameras: Mapping[str, CameraConfig]) -> BullyingCamera` (raises `ClipNameError`)
- Changes: `evaluation/cli.py` — `--camera` is removed from `run`/`gate`/`save-baseline` (it stays on `noise-floor`, which has no clip to ask); `_evaluate` resolves a camera *per clip* and caches one `PoseEstimator` per camera

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_clips.py`:

```python
"""A clip knows which camera it came from. Ask the clip.

`eval run --camera` picked ONE camera config for a whole run, and the corpus is 344
hall_right clips and 299 hall_left ones. hall_left carries a mirror_ignore zone over a
reflective column that is not in hall_right's field of view, so a global flag blanks part
of the frame for whichever half of the corpus it is wrong about -- silently, and the only
symptom is a recall number that looks like a tuning result.
"""

from __future__ import annotations

import pytest

from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import ClipNameError, camera_for, parse_clip_name

CROP = "hall_left_main_1009_1019_20260702_144150_952947.mp4"
BURST = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
RIGHT = "hall_right_main_212_233_burst101_20260702_101530_101010.mp4"


def test_the_camera_comes_from_the_filename() -> None:
    assert parse_clip_name(CROP).camera == "hall_left"
    assert parse_clip_name(RIGHT).camera == "hall_right"


def test_a_crop_and_its_burst_name_the_same_incident() -> None:
    """The join that halves the labelling time: same camera, same track pair, seconds
    apart. The crop was cut out of the burst."""
    crop, burst = parse_clip_name(CROP), parse_clip_name(BURST)

    assert crop.camera == burst.camera
    assert crop.pair == burst.pair == (1009, 1019)
    assert not crop.is_burst
    assert burst.is_burst
    assert abs((burst.recorded_at - crop.recorded_at).total_seconds()) < 30


@pytest.mark.parametrize(
    "filename",
    ["IMG_2201.mp4", "драка_в_коридоре.mp4", "hall_left.mp4", "hall_left_main_1009.mp4"],
)
def test_an_uninferable_name_is_a_HARD_ERROR(filename: str) -> None:
    """Not a default. A clip scored against the wrong camera's zones is a lie that looks
    like a measurement, and the three human-named clips are precisely the ones whose
    camera nobody can prove."""
    with pytest.raises(ClipNameError, match="cannot infer the camera"):
        parse_clip_name(filename)


def test_a_camera_that_is_not_in_the_config_is_a_hard_error() -> None:
    with pytest.raises(ClipNameError, match="not in config/cameras"):
        camera_for("basement_main_1_2_20260702_144150_952947.mp4", load_cameras())


def test_the_real_hall_configs_resolve_and_do_not_share_their_zones() -> None:
    """WHY this exists. If the two halls had the same zones, a global --camera flag would
    have been harmless and none of this would be worth a module."""
    cameras = load_cameras()
    left = camera_for(CROP, cameras)
    right = camera_for(RIGHT, cameras)

    assert left.name == "hall_left"
    assert right.name == "hall_right"
    assert left.bullying.zones.mirror_ignore != right.bullying.zones.mirror_ignore
    assert left.bullying.zones.normal_flow != right.bullying.zones.normal_flow
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_clips.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'qorgan.evaluation.clips'`.

- [ ] **Step 3: Write `src/qorgan/evaluation/clips.py`.**

```python
"""What a clip is, according to its own name.

`eval run --camera` picked ONE camera config for an entire run. The corpus is 344
`hall_right` clips and 299 `hall_left` ones, and `hall_left` carries a `mirror_ignore`
zone over a reflective column that **does not exist in hall_right's field of view**.
Evaluating one camera's footage against the other's zones silently blanks out part of the
frame -- and a silently blanked frame produces a lower recall number that looks exactly
like a tuning result.

So the camera comes from the clip. The school's recorder names every machine-made clip:

    hall_left_main_1009_1019_20260702_144150_952947.mp4           <- the crop (ROI)
    hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4  <- the full frame

    <camera>_main_<track_a>_<track_b>[_burst<n>]_<YYYYMMDD>_<HHMMSS>_<micros>.<ext>

That is also the join between the two views of one incident: same camera, same track-ID
pair, seconds apart. 621 of 660 parsable full-frame clips (94%) have a crop partner.

A name this does not fit is a HARD ERROR, never a default. The three unparsable clips in
the corpus are the human-named ones -- which is to say, precisely the clips whose camera
nobody can prove.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from qorgan.config.camera import BullyingCamera, CameraConfig

CLIP_PATTERN = re.compile(
    r"^(?P<camera>[a-z][a-z0-9_]*?)"
    r"_main"
    r"_(?P<track_a>\d+)_(?P<track_b>\d+)"
    r"(?:_burst(?P<burst>\d+))?"
    r"_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micros>\d{6})"
    r"\.[A-Za-z0-9]+$"
)

EXPECTED = (
    "<camera>_main_<track_a>_<track_b>[_burstNNN]_<YYYYMMDD>_<HHMMSS>_<micros>.<ext>"
)


class ClipNameError(Exception):
    """This clip's camera cannot be proved from its name. An error, never a default."""


@dataclass(frozen=True, slots=True)
class ClipName:
    """One clip, taken apart."""

    filename: str
    camera: str
    track_a: int
    track_b: int
    recorded_at: datetime
    is_burst: bool

    @property
    def pair(self) -> tuple[int, int]:
        """The track-ID pair. Half of the crop <-> full-frame join key."""
        return (self.track_a, self.track_b)


def parse_clip_name(filename: str) -> ClipName:
    """Take a clip's name apart, or refuse."""
    match = CLIP_PATTERN.match(filename)
    if match is None:
        raise ClipNameError(
            f"cannot infer the camera from {filename!r}. Expected {EXPECTED}. "
            "Scoring a clip against another camera's zones silently blanks part of the "
            "frame, so an un-inferable name is an error and not a default."
        )
    return ClipName(
        filename=filename,
        camera=match["camera"],
        track_a=int(match["track_a"]),
        track_b=int(match["track_b"]),
        recorded_at=datetime.strptime(
            f"{match['date']}{match['time']}{match['micros']}", "%Y%m%d%H%M%S%f"
        ),
        is_burst=match["burst"] is not None,
    )


def camera_for(filename: str, cameras: Mapping[str, CameraConfig]) -> BullyingCamera:
    """The camera whose zones and thresholds this clip must be scored against."""
    name = parse_clip_name(filename).camera
    camera = cameras.get(name)
    if camera is None:
        raise ClipNameError(
            f"{filename}: names camera {name!r}, which is not in config/cameras/. "
            f"Known: {', '.join(sorted(cameras))}"
        )
    if not isinstance(camera, BullyingCamera):
        raise ClipNameError(f"{filename}: {name!r} is not a bullying camera")
    return camera
```

- [ ] **Step 4: Run the new test — it passes.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_clips.py -q
```

Expected: `8 passed`.

- [ ] **Step 5: Make the CLI use it.** In `src/qorgan/evaluation/cli.py`:

Add imports:

```python
from qorgan.evaluation.clips import ClipNameError, camera_for
```

Remove the `--camera` argument from `_common` entirely (it stays on the `noise-floor` subcommand, which has no clip to ask), and replace `_evaluate`:

```python
def _evaluate(args: argparse.Namespace) -> tuple[LabelSet, list[RunResult]]:
    from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

    labels = load_labels(args.labels)
    cameras = load_cameras()
    poses: dict[str, SkeletonView] = {}

    results = []
    for video_id in labels.videos:
        clip = args.clips / video_id
        if not clip.is_file():
            print(f"  ! missing clip, skipping: {clip}")
            continue

        # The camera comes from the CLIP, not from a flag. hall_left masks a reflective
        # column that hall_right cannot see; one flag for both blanks part of the frame.
        try:
            camera = camera_for(video_id, cameras)
        except ClipNameError as exc:
            raise SystemExit(str(exc)) from exc

        if camera.name not in poses:
            poses[camera.name] = _pose(camera, args.device)

        print(f"  running {video_id} as {camera.name} ...")
        results.append(
            run(VideoSource(clip, camera), camera.bullying, pose=poses[camera.name])
        )

    if not results:
        raise SystemExit(f"no clips found in {args.clips}. Nothing to measure.")
    return labels, results
```

- [ ] **Step 6: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `772 passed`, `All checks passed!`.

- [ ] **Step 7: Commit.**

```
git status
git add src/qorgan/evaluation/clips.py src/qorgan/evaluation/cli.py tests/test_eval_clips.py
git commit -m "The clip's camera comes from the clip, and an unknown name is an error

--camera picked one config for a whole run. hall_left masks a reflective column
that is not in hall_right's field of view, so half the corpus was being scored
against zones that blank the wrong part of the frame -- silently, and the only
symptom is a recall figure that looks like a tuning result."
```

---

### Task 4: `VideoSource` runs on the configured device

Spec §3.4. `VideoSource` calls `.track()` with no `device=`, taking the Ultralytics default rather than the configured GPU. One line — and it matters when 663 clips go through it. (`PersonDetector.detect` already passes `device=self._device`; the harness's copy does not.)

**Files:**
- `src/qorgan/evaluation/video.py`
- `src/qorgan/evaluation/cli.py`
- `tests/test_eval_clips.py`

**Interfaces:**
- Changes: `qorgan.evaluation.video.VideoSource.__init__(self, path: Path, camera: BullyingCamera, *, device: str = "cuda:0", model: object | None = None) -> None` — `model` is injectable *only* so the device wiring is testable without a GPU
- Produces: `qorgan.evaluation.video._load(name: str)` — the default YOLO factory
- Changes: `evaluation/cli.py::_evaluate` passes `device=args.device` to `VideoSource`

**Steps:**

- [ ] **Step 1: Write the failing test.** Append to `tests/test_eval_clips.py`:

```python
class RecordingModel:
    """Stands in for a YOLO. Records what it was asked to do, and does nothing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def track(self, _frame, **kwargs):
        self.calls.append(kwargs)
        return []


def test_the_video_source_tracks_on_the_configured_device() -> None:
    """One line, and it matters when 663 clips go through it.

    Without `device=`, Ultralytics picks its own default -- so the harness could score the
    whole corpus on the CPU, slowly, while production runs on the GPU. Same weights, but
    not the same measurement of anything that depends on wall-clock throughput, and a
    silent CPU fallback is exactly the kind of difference between bench and field this
    whole section exists to close.
    """
    import numpy as np

    from qorgan.config.loader import load_cameras
    from qorgan.evaluation.video import VideoSource

    model = RecordingModel()
    source = VideoSource(
        Path(CROP), camera_for(CROP, load_cameras()), device="cuda:1", model=model
    )
    source._detect(np.zeros((1440, 2560, 3), dtype=np.uint8))

    assert model.calls, "the model was never asked to track anything"
    assert model.calls[0]["device"] == "cuda:1", "Ultralytics chose the device, not us"
```

with `from pathlib import Path` added to that file's imports.

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_clips.py -q -k device
```

Expected failure: `TypeError: VideoSource.__init__() got an unexpected keyword argument 'device'`.

- [ ] **Step 3: Wire the device through `src/qorgan/evaluation/video.py`.** Replace `__init__` and add `_load`:

```python
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

        # The same crop buffer, of the same depth, that the worker keeps.
        self._recent: deque[np.ndarray] = deque(maxlen=CROP_BUFFER)


def _load(name: str):
    """The real YOLO. Injectable in `VideoSource` only so the device wiring above can be
    tested without a GPU -- there is nothing else to stub here."""
    from ultralytics import YOLO

    return YOLO(name)
```

and add `device=self._device` to the `.track(...)` call in `_detect`:

```python
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
```

- [ ] **Step 4: Pass it from the CLI.** In `src/qorgan/evaluation/cli.py::_evaluate`, change the `run(...)` line to:

```python
        results.append(
            run(
                VideoSource(clip, camera, device=args.device),
                camera.bullying,
                pose=poses[camera.name],
            )
        )
```

and in `cmd_noise_floor`, change `measure(VideoSource(args.clip, camera), metrics)` to `measure(VideoSource(args.clip, camera, device=args.device), metrics)` — adding `--device` to the `noise-floor` subparser too:

```python
    floor.add_argument("--device", default="cuda:0")
```

- [ ] **Step 5: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `773 passed`, `All checks passed!`.

- [ ] **Step 6: Commit.**

```
git status
git add src/qorgan/evaluation/video.py src/qorgan/evaluation/cli.py tests/test_eval_clips.py
git commit -m "VideoSource runs on the configured device

.track() with no device= takes the Ultralytics default. One line, and it is the
difference between scoring 663 clips on the GPU and scoring them, silently, on
the CPU."
```

---

### Task 5: The stairs profile speaks px/second

Spec §3.4. `config/profiles/stairs.yaml:49-50` still carries `static_speed_threshold: 4.0` / `moving_speed_threshold: 8.0` — legacy **px/frame** values, compared against `Track.speed` in **px/s**. The schema defaults are `40.0` / `80.0`. A person standing perfectly still reads well above 4 px/s from box jitter alone, so `a_static` in `gates.staircase_pass` is never true and **gate 8 is dead on the one camera type it exists for.** The units migration missed this file.

**Files:**
- `config/profiles/stairs.yaml`
- `tests/test_noise_floor.py`

**Interfaces:**
- Consumes: `qorgan.config.bullying.StaircasePassGate` (`static_speed_threshold: float`, `moving_speed_threshold: float` — px/s)
- Consumes: `qorgan.detection.tracking.TrackStore`, `qorgan.evaluation.noise_floor.analysis_fps`, `ASSUMED_JITTER_PX`, `SETTLED_HITS`
- Changes: `config/profiles/stairs.yaml` — `static_speed_threshold: 40.0`, `moving_speed_threshold: 80.0`

**Steps:**

- [ ] **Step 1: Write the failing test.** Append to `tests/test_noise_floor.py` (it belongs beside `test_a_motionless_person_is_still_seen_as_motionless`, which pins the same quantity from the other side), and add `SETTLED_HITS` to that file's `noise_floor` import list:

```python
def test_a_standing_person_reads_as_static_on_the_stairs() -> None:
    """**Gate 8 was dead on the only camera type it exists for.**

    `staircase_pass` suppresses "one person standing, one walking past" -- the pattern
    that makes a staircase a staircase, because there is nowhere else to go. It asks
    `a_speed < static_speed_threshold`, and the stairs profile shipped `4.0`: a legacy
    px/FRAME value compared against Track.speed in px/SECOND.

    A person standing perfectly still reads well above 4 px/s from box jitter alone. So
    `a_static` was never true, the gate never fired, and the units migration's single
    remaining miss took out the one suppression the stairs have.

    If this test fails, the VALUE in stairs.yaml is wrong, not the test. The unit is
    px/second and this is what a motionless person costs in it.
    """
    camera = _stairs()
    gate = camera.bullying.gates.staircase_pass
    metrics = camera.bullying.metrics
    fps = _fps(camera)

    store = TrackStore(smoothing=metrics.tracker_smoothing, max_lost=metrics.tracker_max_lost)
    rng = random.Random(11)  # noqa: S311 - a shaky bounding box, not a cipher
    seen = static = 0

    for index in range(int(120 * fps)):  # two minutes of standing perfectly still
        cx = 640.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        cy = 400.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        store.update({1: Box(x1=cx - 30, y1=cy - 80, x2=cx + 30, y2=cy + 80)}, index / fps)

        track = store.tracks[1]
        if track.hits <= SETTLED_HITS:
            continue
        seen += 1
        if track.speed < gate.static_speed_threshold:
            static += 1

    assert static / seen > 0.90, (
        f"a person standing still on the stairs reads as MOVING on "
        f"{(1 - static / seen) * 100:.0f}% of frames against "
        f"static_speed_threshold={gate.static_speed_threshold} px/s. `a_static` is then "
        f"never true and gate 8 (staircase_pass) cannot fire at all."
    )


def test_the_stairs_thresholds_are_px_per_second_not_px_per_frame() -> None:
    """The regression this pins: 4.0 / 8.0 are the legacy's px/FRAME numbers, and they
    are what shipped. Anything under ~20 px/s is a px/frame value wearing a px/s label."""
    gate = _stairs().bullying.gates.staircase_pass

    assert gate.static_speed_threshold >= 20.0, "px/frame leaked back into stairs.yaml"
    assert gate.moving_speed_threshold > gate.static_speed_threshold


def _stairs() -> BullyingCamera:
    camera = load_cameras()["stairs_floor1"]
    assert isinstance(camera, BullyingCamera)
    return camera
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_noise_floor.py -q -k stairs
```

Expected failure: both tests fail. `test_a_standing_person_reads_as_static_on_the_stairs` reports roughly *"a person standing still on the stairs reads as MOVING on ~100% of frames against static_speed_threshold=4.0 px/s"*; `test_the_stairs_thresholds_are_px_per_second_not_px_per_frame` fails on `assert 4.0 >= 20.0`.

- [ ] **Step 3: Fix `config/profiles/stairs.yaml`.** Replace the `staircase_pass` block (lines ~45–50):

```yaml
    staircase_pass:
      enabled: true
      max_contact_frames: 3
      max_overlap_frames: 3
      # px/SECOND, both of them. These shipped as 4.0 / 8.0 -- the legacy's px/FRAME
      # values -- and were compared against Track.speed in px/s. A person standing
      # perfectly still reads well above 4 px/s from box jitter alone, so `a_static` was
      # never true and this gate, the ONE suppression the stairs have, could not fire.
      # See tests/test_noise_floor.py::test_a_standing_person_reads_as_static_on_the_stairs.
      static_speed_threshold: 40.0    # px/s  (legacy 4.0 px/frame)
      moving_speed_threshold: 80.0    # px/s  (legacy 8.0 px/frame)
```

- [ ] **Step 4: Run the suite.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `775 passed`, `All checks passed!`.

- [ ] **Step 5: Commit.**

```
git status
git add config/profiles/stairs.yaml tests/test_noise_floor.py
git commit -m "Gate 8 was dead on the stairs: px/frame compared against px/second

stairs.yaml still carried static_speed_threshold: 4.0 / moving: 8.0 -- the
legacy's px/FRAME values -- against Track.speed in px/s. A motionless person
reads well above 4 px/s from box jitter, so a_static was never true and
staircase_pass, the one suppression the stairs have, never fired.

Now 40 / 80, with a test that pins what a standing person costs in px/second."
```

---

### Task 6: A test that fails when a declared config key is read nowhere

Spec §3.5. About twelve config knobs are parsed, validated, and consumed by nothing — `SeparationGuard`, `ViolenceSettings`, `min_group_size`, several gate thresholds — while the gates hardcode multipliers of `PairMetrics` values instead. Editing them in YAML does nothing at all. That is a trap laid for whoever tunes next, and it is exactly how the legacy's 225 keys got where they are.

So rather than a one-off cleanup: **a test that reflects over every Pydantic model in `config/`, and fails if a declared field is referenced nowhere in `src/`.** Dead keys are then either wired up or deleted to make it pass — and cannot rot back in. **The allowlist is empty.** A "declared plug-in point with no consumer" is a dead key wearing a hat; `ViolenceSettings` goes, and Spec C will design its own config when it needs one.

> **HARD SEQUENCING — this task runs LAST, after Spec A is fully merged. Do not start it
> earlier, and do not "coordinate" informally.**
>
> The reflection test also catches dead keys in `config/canteen.py` (`EntrySettings.face_roi`,
> `ExitSettings.watch_window_seconds`, …) and `config/workers.py`. Spec §9 says this spec does
> not touch `canteen/` — but that is about *behaviour*, and an unread config key has none.
>
> Those files are **rewritten by Spec A** (Task 2 moves `FaceGate` / `RecognitionPolicy` /
> `SoftAccumulator` / `FaceModelSettings` into a new `config/identity.py`; Task 11 rewrites
> `config/workers.py`). If this task runs first, the two plans conflict in `config/`, and the
> dead-key list it computes will be **stale the moment Spec A lands** — including keys that
> Spec A creates (`BindingSettings`) and keys Spec A deletes anyway.
>
> So: **Spec B Tasks 1–5 and 7–11 run in parallel with Spec A. This task alone waits.** It is
> a whole-repo invariant and it should be established once, over the final shape of `config/`,
> not twice over two moving ones.

> **The test must REPORT its own blind spot, not describe it in a comment.**
>
> A name-based reflection test cannot tell a config field from an identically-named
> attribute elsewhere: `ProximityOnlyGate.max_speed` looks alive only because
> `PairData.max_speed` shares the name. A prose comment about that is not a check — and the
> whole point of this test is that a dead key "cannot rot back in". A blind spot documented
> in a comment is a blind spot the next person will not read.
>
> So the test additionally **enumerates every config field name that is also an attribute
> name on a non-config object, and prints them as AMBIGUOUS — invisible to me**. It does not
> need to resolve them. It only has to say, in its own output, which fields it cannot vouch
> for. The blind spot then lives where the next person will actually see it.
>
> If that ambiguous list turns out to be large, STOP and report it — we will reconsider an
> AST-based version that resolves attribute owners properly. If it is three names, we are done.

**Files:**
- `tests/test_config_deadkeys.py` (new)
- `src/qorgan/config/bullying.py`
- `src/qorgan/config/common.py`
- `src/qorgan/config/camera.py`
- `src/qorgan/config/canteen.py`
- `src/qorgan/config/workers.py`
- `config/base.yaml`
- `config/workers.yaml`
- `config/profiles/hall.yaml`, `canteen_entry.yaml`, `canteen_exit.yaml`, `canteen_inside.yaml`
- `tests/test_config_schema.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel.model_fields`
- Consumes: `tests.conftest.SRC_DIR`
- Produces: `tests/test_config_deadkeys.py::ALLOWLIST: frozenset[str]` — **empty, and it stays empty**
- Produces: `tests/test_config_deadkeys.py::test_every_declared_config_key_has_a_consumer(model)` — parametrized over every `BaseModel` in `qorgan.config.*`
- Deletes: `qorgan.config.bullying.SeparationGuard`, `ViolenceSettings`, `PoseSettings`, `Burst`; `Gates.separation_guard`, `BullyingConfig.violence`, `BullyingConfig.pose`, `BullyingConfig.burst`
- Deletes: `qorgan.config.common.DebugSettings`; `CameraBase.debug`; `RtspSettings.stream_queue_size`
- Deletes (fields): `StaticCloseGate.sudden_action_speed_ratio`, `SocialGroupGate.min_group_size`, `SocialGroupGate.max_direction_spread_deg`, `SocialReapproachGate.requires_hard_action`, `ProximityOnlyGate.max_speed`, `NormalFlowMotionGate.min_action_score`, `CrossingPassGate.action_evidence_min_skeleton_score`, `HallConfirmationGate.min_skeleton_score`, `BenignConversationGate.min_victim_displacement_ratio`, `Counters.cooldown_seconds`, `SkeletonSettings.run_on_crop`, `SkeletonSettings.required_for_high_alert`
- Deletes (canteen/workers): `EntrySettings.face_roi/person_cooldown_seconds/min_person_box_area`, `ExitSettings.face_roi/watch_window_seconds/person_cooldown_seconds/min_person_box_area`, `InsideSettings.exit_missing_frames`, `SessionRules.staff_presence_ttl_seconds`, `WorkersConfig.heartbeat_interval_seconds`

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_config_deadkeys.py`:

```python
"""A declared config key with no consumer is a trap laid for whoever tunes next.

About a dozen knobs were parsed, validated, and read by nothing -- `SeparationGuard`,
`ViolenceSettings`, `min_group_size`, several gate thresholds -- while the gates hardcoded
multipliers of `PairMetrics` values instead. Editing them in YAML did nothing at all.
That is exactly how the legacy's 225 keys got where they are: not by anyone deciding to
ship dead config, but by nothing ever saying no.

So: not a one-off cleanup. A test. It reflects over every Pydantic model in
`qorgan.config` and fails if a declared field is referenced nowhere in `src/`.

**The allowlist is empty, and it stays empty.** A "declared plug-in point with no
consumer" is a dead key wearing a hat.

Two things this test honestly cannot see, both caught by grep during the cleanup that
made it pass:
  * a field only its own model's validator reads (that is a use, but it is a use by
    nobody);
  * a field whose NAME collides with a live attribute elsewhere -- `ProximityOnlyGate.
    max_speed` reads as "referenced" because `PairData.max_speed` exists.
Both were removed by hand. The test's job is to stop the NEXT one arriving.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import qorgan.config
from tests.conftest import SRC_DIR

# EMPTY. Keep it that way. If a key has no consumer, wire it up or delete it.
ALLOWLIST: frozenset[str] = frozenset()


def _referenced_names() -> set[str]:
    """Every identifier `src/` mentions, minus the field declarations themselves.

    `speed_threshold: float = Field(...)` inside a config model DECLARES the key; it does
    not use it. Everything else -- `config.speed_threshold`, `Gate(speed_threshold=...)`,
    a string key -- does.
    """
    names: set[str] = set()
    for path in SRC_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        declarations = {
            statement.target
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        }
        for node in ast.walk(tree):
            if node in declarations:
                continue
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return names


def _config_models() -> list[type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for module_info in pkgutil.iter_modules(qorgan.config.__path__):
        module = importlib.import_module(f"qorgan.config.{module_info.name}")
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__.startswith("qorgan.config.")
            ):
                models[obj.__name__] = obj
    return [models[name] for name in sorted(models)]


REFERENCED = _referenced_names()


@pytest.mark.parametrize("model", _config_models(), ids=lambda m: m.__name__)
def test_every_declared_config_key_has_a_consumer(model: type[BaseModel]) -> None:
    dead = sorted(
        field
        for field in model.model_fields
        if field not in REFERENCED and f"{model.__name__}.{field}" not in ALLOWLIST
    )

    assert not dead, (
        f"{model.__name__}: {', '.join(dead)} -- declared, parsed, validated, and read by "
        f"NOTHING in src/. Editing it in YAML does nothing at all, and a tuner who does "
        f"not know that will spend a day on it. Wire it up or delete it. The allowlist is "
        f"empty and stays empty."
    )
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_config_deadkeys.py -q
```

Expected failure: 24 failing parametrizations. Exactly:

```
BenignConversationGate: min_victim_displacement_ratio
BullyingCamera:         debug
BullyingConfig:         violence          (+ pose, burst -- see note)
Burst:                  duration_seconds, cooldown_seconds
CameraBase:             debug
CanteenCamera:          debug
Counters:               cooldown_seconds
CrossingPassGate:       action_evidence_min_skeleton_score
DebugSettings:          save_frames, save_face_crops, draw_overlays
EntrySettings:          face_roi, person_cooldown_seconds, min_person_box_area
ExitSettings:           face_roi, watch_window_seconds, person_cooldown_seconds, min_person_box_area
Gates:                  separation_guard
HallConfirmationGate:   min_skeleton_score
InsideSettings:         exit_missing_frames
NormalFlowMotionGate:   min_action_score
PoseSettings:           check_interval_seconds, buffer_size
RtspSettings:           stream_queue_size
SeparationGuard:        separation_distance_ratio, min_separation_frames
SessionRules:           staff_presence_ttl_seconds
SkeletonSettings:       run_on_crop, required_for_high_alert
SocialGroupGate:        min_group_size, max_direction_spread_deg
SocialReapproachGate:   requires_hard_action
StaticCloseGate:        sudden_action_speed_ratio
ViolenceSettings:       buffer_size, check_interval_seconds, cache_seconds
WorkersConfig:          heartbeat_interval_seconds
```

(`BullyingConfig.pose` and `BullyingConfig.burst` do not appear — `pose` and `burst` are live *identifiers* elsewhere in `src/`, so the name-based scan cannot see them. Both models are dead by grep; delete them in Step 4.)

- [ ] **Step 3: Verify by grep, before deleting anything.** Run each and confirm the only hits are inside `src/qorgan/config/`:

```
git grep -n "separation_guard\|SeparationGuard\|ViolenceSettings\|PoseSettings\|min_group_size\|max_direction_spread_deg\|min_action_score\|sudden_action_speed_ratio\|action_evidence_min_skeleton_score\|min_victim_displacement_ratio\|requires_hard_action\|run_on_crop\|required_for_high_alert\|stream_queue_size\|heartbeat_interval_seconds\|face_roi\|exit_missing_frames\|staff_presence_ttl_seconds" -- src
git grep -n "\.max_speed\|gates\.proximity_only" -- src
git grep -n "\.burst\b\|\.pose\b\|\.violence\b\|\.debug\b" -- src/qorgan/worker src/qorgan/events src/qorgan/canteen src/qorgan/web src/qorgan/preview
git grep -n "counters\.cooldown_seconds\|\.min_skeleton_score\b" -- src
```

Expected: `src/qorgan/detection/pairs.py` hits for `max_speed` are `PairData.max_speed`, a **different** attribute that merely shares the name; `src/qorgan/detection/validation.py`'s `min_skeleton_score_for_confirmation` is a **different** key on `Confidence` and stays. Every other hit is a declaration inside `config/`. Nothing consumes any of them.

- [ ] **Step 4: Delete the dead bullying config.** In `src/qorgan/config/bullying.py`:

Delete the classes `SeparationGuard`, `PoseSettings`, `ViolenceSettings` and `Burst` entirely, and these fields:

```python
class StaticCloseGate(Base):
    ...
    sudden_action_bypass_enabled: bool = True
    # sudden_action_speed_ratio DELETED -- `_sudden_action` compares against
    # PairMetrics.acceleration_threshold and window_drop_threshold, never against a ratio.


class SocialGroupGate(Base):
    """Gate 3: a group in a flow zone moving the same direction."""

    enabled: bool = True
    # min_group_size / max_direction_spread_deg DELETED -- `social_group` reads
    # signals.same_direction and hardcoded multipliers of PairMetrics, never these.


class SocialReapproachGate(Base):
    """Gate 4: close, drifted apart, close again. Friends talking."""

    enabled: bool = True
    min_prior_close_frames: int = Field(default=4, ge=1)
    max_gap_frames: int = Field(default=12, ge=1)
    distance_delta_ratio: float = Field(default=0.35, gt=0)
    # requires_hard_action DELETED -- never read.


class ProximityOnlyGate(Base):
    """Gate 5: close, but no motion at all."""

    enabled: bool = True
    # max_speed DELETED. `proximity_only` reads signals.motion_present, not a speed.
    # The reflection test could not see this one: PairData.max_speed shares the name.


class NormalFlowMotionGate(Base):
    """Gate 6: inside a flow zone, demand a strong action signal."""

    enabled: bool = True
    # min_action_score DELETED -- `normal_flow_motion_required` builds `has_action` from
    # hardcoded multipliers of PairMetrics, never from a score.


class CrossingPassGate(Base):
    """Gate 7: two people walking past each other."""

    enabled: bool = True
    max_contact_frames: int = Field(default=2, ge=0)
    max_overlap_frames: int = Field(default=2, ge=0)
    # Legacy plumbed a `requires_skeleton_aggression` flag the function body never
    # read. If we want the bypass, it has to actually do something:
    action_evidence_bypass_enabled: bool = True
    # action_evidence_min_skeleton_score DELETED -- `_action_evidence` never sees a
    # skeleton; the skeleton runs in the SLOW tier, after the gates.


class HallConfirmationGate(Base):
    """Gate 9: hall cameras require sustained contact or overlap before alerting."""

    enabled: bool = False
    sustained_contact_min: int = Field(default=10, ge=1)
    sustained_overlap_min: int = Field(default=5, ge=1)
    # min_skeleton_score DELETED -- same reason. The skeleton is not available here.
    # Confidence.min_skeleton_score_for_confirmation is the live one, and it is elsewhere.


class BenignConversationGate(Base):
    """Gate 10: a confirmed pair, but the victim never moved."""

    enabled: bool = True
    # min_victim_displacement_ratio DELETED -- `benign_conversation` measures
    # displacement against PairMetrics thresholds, never against a ratio of its own.
```

Remove `cooldown_seconds` from `Counters`, `run_on_crop` and `required_for_high_alert` from `SkeletonSettings`, `separation_guard` from `Gates`, and `pose`, `violence`, `burst` from `BullyingConfig`:

```python
class BullyingConfig(Base):
    metrics: PairMetrics = PairMetrics()
    weights: ScoreWeights = ScoreWeights()
    counters: Counters = Counters()
    zones: Zones = Zones()
    gates: Gates = Gates()
    skeleton: SkeletonSettings = SkeletonSettings()
    confidence: Confidence = Confidence()
    event_merge: EventMerge = EventMerge()

    # The score a pair must reach to become a candidate, before zone modifiers.
    alert_score_threshold: float = Field(default=1.4, gt=0)
```

- [ ] **Step 5: Delete the dead shared config.** In `src/qorgan/config/common.py`: delete `stream_queue_size` from `RtspSettings` and delete the whole `DebugSettings` class. In `src/qorgan/config/camera.py`: delete the `debug: DebugSettings = DebugSettings()` field from `CameraBase` and the `DebugSettings` import.

In `src/qorgan/config/canteen.py`: delete `SessionRules.staff_presence_ttl_seconds`, `EntrySettings.face_roi`, `EntrySettings.person_cooldown_seconds`, `EntrySettings.min_person_box_area`, `ExitSettings.face_roi`, `ExitSettings.watch_window_seconds`, `ExitSettings.person_cooldown_seconds`, `ExitSettings.min_person_box_area`, `InsideSettings.exit_missing_frames`.

In `src/qorgan/config/workers.py`: delete `WorkersConfig.heartbeat_interval_seconds`, leaving a comment where it was:

```python
    # heartbeat_interval_seconds DELETED: the worker's cadence is a fixed 1 s in
    # worker/entrypoint.py and this key never reached it. `heartbeat_timeout_seconds`
    # above IS live -- the supervisor kills a worker that misses it.
```

- [ ] **Step 6: Delete the YAML that set them.** `extra="forbid"` means any surviving key is now a startup error, so this is not optional.

- `config/base.yaml`: delete the `stream_queue_size: 1` line and the whole `debug:` block (with its comment).
- `config/profiles/hall.yaml`: delete `cooldown_seconds: 8.0` (under `counters:`), `min_skeleton_score: 0.55` (under `hall_confirmation:`), `required_for_high_alert: true` (under `skeleton:`).
- `config/profiles/canteen_entry.yaml`: delete `face_roi`, `min_person_box_area`, `person_cooldown_seconds`.
- `config/profiles/canteen_exit.yaml`: delete `face_roi`, `min_person_box_area`, `person_cooldown_seconds`, `watch_window_seconds`.
- `config/profiles/canteen_inside.yaml`: delete `exit_missing_frames`.
- `config/workers.yaml`: delete `heartbeat_interval_seconds: 5.0`.

- [ ] **Step 7: Delete the test whose subject is gone.** In `tests/test_config_schema.py`, delete `test_a_violence_model_without_positive_classes_is_rejected` and remove `ViolenceSettings` from the import on line 8.

- [ ] **Step 8: Run everything.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m qorgan config validate
```

Expected: all pass (`~797 passed` — 775, minus 1 deleted, plus ~23 new parametrized dead-key tests), `All checks passed!`, and `config validate` prints `OK`. If `config validate` fails with an "extra key" error, a YAML line in Step 6 was missed; the error names it.

- [ ] **Step 9: Commit.**

```
git status
git add tests/test_config_deadkeys.py src/qorgan/config/bullying.py src/qorgan/config/common.py src/qorgan/config/camera.py src/qorgan/config/canteen.py src/qorgan/config/workers.py config/base.yaml config/workers.yaml config/profiles/hall.yaml config/profiles/canteen_entry.yaml config/profiles/canteen_exit.yaml config/profiles/canteen_inside.yaml tests/test_config_schema.py
git commit -m "A config key that nothing reads now fails the build

~30 keys were parsed, validated, and consumed by nothing: SeparationGuard,
ViolenceSettings, PoseSettings, Burst, DebugSettings, six gate thresholds.
Editing them in YAML did nothing at all -- which is a trap for whoever tunes
next, and exactly how the legacy got to 225 keys.

The allowlist is empty. A declared plug-in point with no consumer is a dead key
wearing a hat: ViolenceSettings goes, and Spec C will design its own."
```

---

### Task 7: `qorgan eval scan` — the candidate list

Spec §4, §5 (pass 1). The 663 full-frame clips go to `eval/clips/`, **gitignored** — they are footage of children. `qorgan eval scan` runs the detector over every one of them at threshold 0 and emits `eval/candidates.csv` (`clip, timestamp, score, probability, confidence`). A human then watches only what the detector **fired on**. That yields precision, and it concentrates the human's attention exactly where the label changes the answer.

**Files:**
- `src/qorgan/evaluation/scan.py` (new)
- `src/qorgan/evaluation/harness.py`
- `src/qorgan/evaluation/cli.py`
- `.gitignore`
- `tests/test_eval_scan.py` (new)

**Interfaces:**
- Changes: `qorgan.evaluation.harness.Alert` gains `score: float` (the candidate's raw aggression score, which `Verdict` does not carry)
- Produces: `qorgan.evaluation.scan.ScanRow` — frozen dataclass: `clip: str`, `timestamp: float`, `score: float`, `probability: float`, `confidence: float`
- Produces: `qorgan.evaluation.scan.SCAN_COLUMNS: tuple[str, ...] = ("clip", "timestamp", "score", "probability", "confidence")`
- Produces: `qorgan.evaluation.scan.rows_for(result: RunResult) -> list[ScanRow]`
- Produces: `qorgan.evaluation.scan.write_candidates(rows: Iterable[ScanRow], path: Path) -> int`
- Produces: `qorgan.evaluation.scan.load_candidates(path: Path) -> list[ScanRow]`
- Produces: `qorgan.evaluation.cli.cmd_scan(args) -> int` — `qorgan eval scan [--clips DIR] [--out FILE] [--device D]`
- Produces: `qorgan.evaluation.cli.CANDIDATES_PATH = Path("eval/candidates.csv")`, `CLIP_SUFFIXES = {".mp4", ".avi", ".mkv"}`

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_scan.py`:

```python
"""`qorgan eval scan`: what did the detector fire on, and when?

Watching 97 minutes of corridor to find the few seconds that matter is not a good use of
a human. The detector runs at threshold 0 over every clip and writes down every moment it
would have raised anything at all; the human then watches only those.
"""

from __future__ import annotations

from pathlib import Path

from qorgan.config.bullying import BullyingConfig
from qorgan.evaluation import run
from qorgan.evaluation.scan import ScanRow, load_candidates, rows_for, write_candidates
from tests.test_evaluation import SpyPose, SyntheticClip
from tests.test_detection_pipeline import scene_an_assault, scene_two_children_talking

CLIP = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"


def test_a_scan_writes_one_row_per_incident_the_detector_fired_on() -> None:
    result = run(SyntheticClip(CLIP, scene_an_assault()), BullyingConfig(), pose=SpyPose())
    rows = rows_for(result)

    assert rows, "the detector found the assault but scan reported nothing"
    assert all(row.clip == CLIP for row in rows)
    assert all(row.score > 0 and 0.0 <= row.probability <= 1.0 for row in rows)
    assert all(0.0 <= row.confidence <= 1.0 for row in rows)


def test_a_merged_alert_is_not_a_second_row() -> None:
    """One physical incident, one thing for a human to watch. Forty rows for one scuffle
    would waste the labeller's time on the same four seconds forty times."""
    result = run(
        SyntheticClip(CLIP, scene_an_assault(frames=60)), BullyingConfig(), pose=SpyPose()
    )

    assert len(rows_for(result)) == 1


def test_a_quiet_clip_produces_no_candidates() -> None:
    result = run(SyntheticClip(CLIP, scene_two_children_talking()), BullyingConfig())

    assert rows_for(result) == []


def test_candidates_round_trip(tmp_path: Path) -> None:
    rows = [ScanRow(CLIP, 2.4, 1.83, 0.71, 0.66), ScanRow(CLIP, 9.1, 2.20, 0.90, 0.88)]
    path = tmp_path / "candidates.csv"

    assert write_candidates(rows, path) == 2
    assert load_candidates(path) == rows


def test_loading_a_candidates_file_that_does_not_exist_is_an_empty_list(tmp_path: Path) -> None:
    """`eval label` is resumable, and resuming from nothing is a normal state."""
    assert load_candidates(tmp_path / "nope.csv") == []
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_scan.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'qorgan.evaluation.scan'`.

- [ ] **Step 3: Put the candidate's score on the Alert.** In `src/qorgan/evaluation/harness.py`, add `score: float` to `Alert` (after `key`) and set it in `_judge`:

```python
@dataclass(frozen=True, slots=True)
class Alert:
    """What the worker would actually have recorded and sent."""

    video_id: str
    timestamp: float
    key: tuple[int, int]
    # The RAW aggression score, which Verdict does not carry: `eval scan` writes it out
    # so a human reading candidates.csv can see WHY the detector fired, not just how sure
    # it ended up.
    score: float
    verdict: Verdict
    merged: bool
    notified: bool
```

and in `_judge`'s return: `score=candidate.score,` immediately after `key=candidate.key,`.

- [ ] **Step 4: Write `src/qorgan/evaluation/scan.py`.**

```python
"""Pass 1 of labelling: by exception.

97 minutes of corridor, and the few seconds that matter are somewhere in it. Watching all
of it is not a good use of a human, so the detector goes first: it runs at **threshold 0**
over every clip and writes down every moment it would have raised anything at all.

The human then watches only what it fired on -- perhaps 50-150 clips of 8 seconds each.
That yields precision, and it puts the human's attention exactly where the label changes
the answer.

It is only half a detector, and `qorgan eval sample` is the other half.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import astuple, dataclass
from pathlib import Path

from qorgan.evaluation.harness import RunResult

SCAN_COLUMNS = ("clip", "timestamp", "score", "probability", "confidence")


@dataclass(frozen=True, slots=True)
class ScanRow:
    """One moment a human should look at."""

    clip: str
    timestamp: float
    score: float
    probability: float
    confidence: float


def rows_for(result: RunResult) -> list[ScanRow]:
    """Every incident this clip produced.

    Merged alerts are excluded: a merged alert is the SAME physical incident, and forty
    rows for one four-second scuffle would spend the labeller's attention forty times on
    the same four seconds. One incident, one thing to watch -- the same rule the merger
    applies to Telegram.
    """
    return [
        ScanRow(
            clip=alert.video_id,
            timestamp=round(alert.timestamp, 3),
            score=round(alert.score, 3),
            probability=round(alert.verdict.candidate_probability, 3),
            confidence=round(alert.verdict.confidence, 3),
        )
        for alert in result.alerts
        if not alert.merged
    ]


def write_candidates(rows: Iterable[ScanRow], path: Path) -> int:
    """Write eval/candidates.csv. GITIGNORED: every row names a clip of children."""
    ordered = sorted(rows, key=lambda row: (row.clip, row.timestamp))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCAN_COLUMNS)
        writer.writerows(astuple(row) for row in ordered)
    return len(ordered)


def load_candidates(path: Path) -> list[ScanRow]:
    """Read it back. A missing file is an empty list: `eval label` is resumable, and
    resuming from nothing is an ordinary state, not an error."""
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            ScanRow(
                clip=row["clip"],
                timestamp=float(row["timestamp"]),
                score=float(row["score"]),
                probability=float(row["probability"]),
                confidence=float(row["confidence"]),
            )
            for row in csv.DictReader(handle)
        ]
```

- [ ] **Step 5: Add the subcommand.** In `src/qorgan/evaluation/cli.py`, add near the other path constants:

```python
CANDIDATES_PATH = Path("eval/candidates.csv")
CLIP_SUFFIXES = {".mp4", ".avi", ".mkv"}
```

register the subparser inside `add_parser`:

```python
    scan_cmd = sub.add_parser(
        "scan",
        help="run the detector over every clip at threshold 0 and list what it fired on",
        description=(
            "Pass 1 of labelling, by exception. 97 minutes of corridor is too much to "
            "watch; the detector goes first and writes down every moment it would have "
            "raised anything at all. A human then watches only those."
        ),
    )
    scan_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    scan_cmd.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    scan_cmd.add_argument("--device", default="cuda:0")
    scan_cmd.set_defaults(func=cmd_scan)
```

and add the command plus its helper:

```python
def cmd_scan(args: argparse.Namespace) -> int:
    clips = _clips_in(args.clips)
    cameras = load_cameras()
    poses: dict[str, SkeletonView] = {}
    rows: list[ScanRow] = []

    for index, clip in enumerate(clips, start=1):
        try:
            camera = camera_for(clip.name, cameras)
        except ClipNameError as exc:
            raise SystemExit(str(exc)) from exc
        if camera.name not in poses:
            poses[camera.name] = _pose(camera, args.device)

        found = _scan_one(clip, camera, poses[camera.name], args.device)
        rows.extend(found)
        print(f"  [{index}/{len(clips)}] {clip.name} ({camera.name}): {len(found)} candidate(s)")

    written = write_candidates(rows, args.out)
    print(f"\n{written} candidate(s) from {len(clips)} clip(s) -> {args.out}")
    print("Now label them:  qorgan eval label")
    return 0


def _scan_one(clip: Path, camera: BullyingCamera, pose: SkeletonView, device: str):
    from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

    result = run(VideoSource(clip, camera, device=device), camera.bullying, pose=pose)
    return rows_for(result)


def _clips_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"no clips directory: {directory}")
    clips = sorted(p for p in directory.iterdir() if p.suffix.lower() in CLIP_SUFFIXES)
    if not clips:
        raise SystemExit(f"no clips in {directory}. Nothing to scan.")
    return clips
```

with the imports:

```python
from qorgan.evaluation.scan import ScanRow, rows_for, write_candidates
```

- [ ] **Step 6: Confirm the footage cannot be committed.** `.gitignore` already carries `eval/clips/` and `eval/candidates.csv`. Add the two files the next tasks introduce, under the same comment:

```
# Evaluation footage. `eval/labels.csv` and `eval/baseline.json` are the *results* and
# belong in git; the clips they refer to are video of children and never do. Nor do the
# crop ROIs, nor the candidate/sample lists, whose every row names a child's clip.
eval/clips/
eval/crops/
eval/candidates.csv
eval/sample.csv
```

Then prove it:

```
git check-ignore -v eval/clips eval/crops eval/candidates.csv eval/sample.csv
```

Expected: four lines, each naming `.gitignore`.

- [ ] **Step 7: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `~802 passed`, `All checks passed!`.

- [ ] **Step 8: Commit.**

```
git status
git add src/qorgan/evaluation/scan.py src/qorgan/evaluation/harness.py src/qorgan/evaluation/cli.py .gitignore tests/test_eval_scan.py
git commit -m "qorgan eval scan: the detector goes first

97 minutes of corridor is too much for a human to watch. The detector runs over
every clip at threshold 0 and writes down every moment it would have raised
anything; the human watches only those. That is precision, and it puts the
attention where the label changes the answer.

candidates.csv, crops/ and sample.csv are gitignored: every row names a clip of
children."
```

---

### Task 8: `qorgan eval label` — the crop is the lens, the full frame is the record

Spec §2.1, §5. A human deciding *"is this a fight?"* does not need the scene. They need to see **the two children, close up** — which is exactly what the old detector's crop ROI is, and it is far faster to watch than a 1440p wide shot in which the pair occupies 3% of the pixels. Joining on `(camera, track_a, track_b)` and nearest timestamp, **621 of 660 full-frame clips (94%) have a crop partner.**

So: `eval label` shows the crop; `eval run` scores the full frame. **Labels are always written against the full-frame `video_id`, whichever view the human watched.** The crop is a lens, never the record.

It is a dev tool. It gets no web route.

**Files:**
- `src/qorgan/evaluation/labelling.py` (new)
- `src/qorgan/evaluation/labels.py`
- `src/qorgan/evaluation/cli.py`
- `tests/test_eval_label.py` (new)

**Interfaces:**
- Consumes: `qorgan.evaluation.clips.parse_clip_name`, `ClipName`, `ClipNameError`
- Consumes: `qorgan.evaluation.scan.ScanRow`, `load_candidates`
- Consumes: `qorgan.evaluation.labels.LabelKind`, `REQUIRED_COLUMNS`
- Produces: `qorgan.evaluation.labelling.LABEL_PAD_SECONDS: float = 2.0` (matches `metrics.DEFAULT_TOLERANCE_SECONDS`)
- Produces: `qorgan.evaluation.labelling.crop_partner(clip: str, crops: Iterable[str]) -> str | None`
- Produces: `qorgan.evaluation.labelling.interval_for(row: ScanRow, kind: LabelKind) -> Interval`
- Produces: `qorgan.evaluation.labelling.append_label(path: Path, interval: Interval) -> None`
- Produces: `qorgan.evaluation.labelling.already_labelled(path: Path) -> set[tuple[str, float]]`
- Produces: `qorgan.evaluation.labelling.is_done(row: ScanRow, done: set[tuple[str, float]]) -> bool`
- Produces: `qorgan.evaluation.labelling.open_in_player(path: Path) -> None`
- Produces: `qorgan.evaluation.cli.cmd_label(args) -> int` — `qorgan eval label [--candidates FILE] [--clips DIR] [--crops DIR] [--labels FILE]`

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_label.py`:

```python
"""`qorgan eval label`: watch the crop, record against the full frame.

A human deciding "is this a fight?" does not need the scene. They need the two children,
close up -- which is what the old detector's crop ROI is, and it is far faster to watch
than a 1440p wide shot in which the pair occupies 3% of the pixels.

But the crop has no scene, no zones and no frame geometry, so it can never be detector
input. Two views of one incident, each used for what it is good at. The label is always
written against the FULL FRAME.
"""

from __future__ import annotations

from pathlib import Path

from qorgan.evaluation.labels import LabelKind, load_labels
from qorgan.evaluation.labelling import (
    already_labelled,
    append_label,
    crop_partner,
    interval_for,
    is_done,
)
from qorgan.evaluation.scan import ScanRow

BURST = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
CROP = "hall_left_main_1009_1019_20260702_144150_952947.mp4"
CROP_LATER = "hall_left_main_1009_1019_20260702_150000_000000.mp4"
OTHER_PAIR = "hall_left_main_77_88_20260702_144151_000000.mp4"
OTHER_CAMERA = "hall_right_main_1009_1019_20260702_144151_000000.mp4"


def test_the_crop_partner_is_found_on_camera_pair_and_nearest_time() -> None:
    partner = crop_partner(BURST, [OTHER_CAMERA, OTHER_PAIR, CROP_LATER, CROP])

    assert partner == CROP, "the join is (camera, track_a, track_b) + NEAREST timestamp"


def test_a_full_frame_with_no_crop_partner_falls_back_to_itself() -> None:
    """6% of the corpus. The labeller then watches the wide shot -- slower, but it is a
    label, and a missing label is worse than a slow one."""
    assert crop_partner(BURST, [OTHER_CAMERA, OTHER_PAIR]) is None


def test_an_unparsable_crop_name_is_skipped_not_crashed_on() -> None:
    """Three clips in the corpus are human-named. They cannot be joined; they must not
    take the whole labelling run down with them."""
    assert crop_partner(BURST, ["драка.mp4", CROP]) == CROP


def test_the_label_is_written_against_the_FULL_FRAME_whichever_view_was_watched() -> None:
    """The crop is a lens, never the record. `eval run` scores the full frame, so a label
    against a crop's video_id would match nothing and read as a missed fight."""
    row = ScanRow(BURST, 4.25, 1.9, 0.8, 0.88)
    interval = interval_for(row, LabelKind.BULLYING)

    assert interval.video_id == BURST
    assert interval.start == 2.25
    assert interval.end == 6.25
    assert interval.kind is LabelKind.BULLYING


def test_a_label_at_time_zero_does_not_start_before_the_clip_does() -> None:
    assert interval_for(ScanRow(BURST, 0.5, 1.9, 0.8, 0.88), LabelKind.NORMAL).start == 0.0


def test_labels_are_appended_and_the_file_stays_readable(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    append_label(path, interval_for(ScanRow(BURST, 4.0, 1.9, 0.8, 0.88), LabelKind.BULLYING))
    append_label(path, interval_for(ScanRow(BURST, 30.0, 1.5, 0.6, 0.60), LabelKind.NORMAL))

    labels = load_labels(path)

    assert len(labels) == 2
    assert len(labels.positives(BURST)) == 1


def test_labelling_is_resumable(tmp_path: Path) -> None:
    """97 minutes of footage is more than one sitting. A labeller who stops for lunch
    must not come back to the beginning."""
    path = tmp_path / "labels.csv"
    done_row = ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)
    todo_row = ScanRow(BURST, 30.0, 1.5, 0.6, 0.60)
    append_label(path, interval_for(done_row, LabelKind.BULLYING))

    done = already_labelled(path)

    assert is_done(done_row, done)
    assert not is_done(todo_row, done)


def test_resuming_from_nothing_is_not_an_error(tmp_path: Path) -> None:
    assert already_labelled(tmp_path / "nope.csv") == set()
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_label.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'qorgan.evaluation.labelling'`.

- [ ] **Step 3: Write `src/qorgan/evaluation/labelling.py`.**

```python
"""The labelling tool's parts. A dev tool: it gets no web route.

**Watch the crop; record the full frame.**

`hall_left_main_1009_1019_….mp4` (186x326) and
`hall_left_main_1009_1019_burst101_….mp4` (2560x1440) are the same incident: same camera,
same track-ID pair, seconds apart. The crop was cut out of the burst.

A human deciding "is this a fight?" does not need the scene. They need the two children,
close up -- which is exactly what the crop is, and it is far faster to watch than a 1440p
wide shot in which the pair occupies 3% of the pixels. Measured across the corpus: 621 of
660 parsable full-frame clips (94%) have a crop partner.

But the crop can never be DETECTOR input: 320x450 has no scene, no zones, no frame
geometry, and the scorer's box-diagonal scaling is meaningless inside it. It would
produce numbers, and every one of them would be a lie.

So the label is always written against the **full-frame** video_id, whichever view the
human watched. The crop is a lens, never the record.
"""

from __future__ import annotations

import csv
import os
import subprocess  # noqa: S404 - opening a video in the OS player, on a dev machine
import sys
from collections.abc import Iterable
from pathlib import Path

from qorgan.evaluation.clips import ClipNameError, parse_clip_name
from qorgan.evaluation.labels import REQUIRED_COLUMNS, Interval, LabelKind
from qorgan.evaluation.scan import ScanRow

# How wide an interval one candidate timestamp becomes. Matches
# metrics.DEFAULT_TOLERANCE_SECONDS: a human annotator's start time is itself approximate,
# and a label narrower than the tolerance would be a distinction the scorer cannot see.
LABEL_PAD_SECONDS = 2.0


def crop_partner(clip: str, crops: Iterable[str]) -> str | None:
    """The crop showing the same incident as this full-frame clip, or None.

    The join: same camera, same track-ID pair, nearest timestamp. Unparsable names are
    skipped rather than fatal -- the three human-named clips in the corpus must not take
    a labelling run down with them.
    """
    try:
        target = parse_clip_name(clip)
    except ClipNameError:
        return None

    best: str | None = None
    best_gap = float("inf")
    for name in crops:
        try:
            other = parse_clip_name(name)
        except ClipNameError:
            continue
        if other.camera != target.camera or other.pair != target.pair or other.is_burst:
            continue
        gap = abs((other.recorded_at - target.recorded_at).total_seconds())
        if gap < best_gap:
            best, best_gap = name, gap
    return best


def interval_for(row: ScanRow, kind: LabelKind) -> Interval:
    """One candidate moment, as a labelled interval -- against the FULL FRAME."""
    return Interval(
        video_id=row.clip,
        start=max(0.0, row.timestamp - LABEL_PAD_SECONDS),
        end=row.timestamp + LABEL_PAD_SECONDS,
        kind=kind,
    )


def append_label(path: Path, interval: Interval) -> None:
    """Append one row. Written immediately, so a crash costs the last decision and not
    the afternoon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(REQUIRED_COLUMNS)
        writer.writerow(
            [interval.video_id, f"{interval.start:.2f}", f"{interval.end:.2f}", interval.kind.value]
        )


def already_labelled(path: Path) -> set[tuple[str, float]]:
    """(video_id, start) for every row already written. Resuming from nothing is normal."""
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["video_id"], float(row["t_start"])) for row in csv.DictReader(handle)}


def is_done(row: ScanRow, done: set[tuple[str, float]]) -> bool:
    """97 minutes is more than one sitting. A labeller who stops for lunch must not come
    back to the beginning."""
    start = max(0.0, row.timestamp - LABEL_PAD_SECONDS)
    return (row.clip, round(start, 2)) in {(clip, round(t, 2)) for clip, t in done}


def open_in_player(path: Path) -> None:
    """Hand the clip to the OS default player. A dev tool on a dev machine."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a local video file, on a developer's desktop
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)])  # noqa: S603 - same
```

- [ ] **Step 4: Export `Interval` from labels.** `qorgan.evaluation.labels` already defines `Interval` and `REQUIRED_COLUMNS`; no change is needed. Confirm:

```
git grep -n "^REQUIRED_COLUMNS\|^class Interval" -- src/qorgan/evaluation/labels.py
```

Expected: both present.

- [ ] **Step 5: Run the new test — it passes.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_label.py -q
```

Expected: `8 passed`.

- [ ] **Step 6: Add the `label` subcommand.** In `src/qorgan/evaluation/cli.py`, add the constants:

```python
CROPS_DIR = Path("eval/crops")
PROMPT = "[b]ullying / [n]ormal / [i]gnore / [s]kip / [q]uit > "
CHOICES = {
    "b": LabelKind.BULLYING,
    "n": LabelKind.NORMAL,
    "i": LabelKind.IGNORE,
}
```

register the subparser:

```python
    label_cmd = sub.add_parser(
        "label",
        help="watch each candidate and label it",
        description=(
            "Opens the CROP -- the two children, close up -- in your default video "
            "player, falling back to the full frame when no crop partner exists. The "
            "label is always recorded against the FULL-FRAME clip, whichever view you "
            "watched: the crop is a lens, never the record. Resumable; a dev tool."
        ),
    )
    label_cmd.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    label_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    label_cmd.add_argument("--crops", type=Path, default=CROPS_DIR)
    label_cmd.add_argument("--labels", type=Path, default=LABELS_PATH)
    label_cmd.set_defaults(func=cmd_label)
```

and the command:

```python
def cmd_label(args: argparse.Namespace) -> int:
    rows = load_candidates(args.candidates)
    if not rows:
        raise SystemExit(f"no candidates in {args.candidates}. Run `qorgan eval scan` first.")

    done = already_labelled(args.labels)
    todo = [row for row in rows if not is_done(row, done)]
    crops = [p.name for p in args.crops.iterdir()] if args.crops.is_dir() else []

    print(f"{len(rows)} candidate(s), {len(rows) - len(todo)} already labelled.\n")
    for index, row in enumerate(todo, start=1):
        view = _view_for(row, args, crops)
        print(f"[{index}/{len(todo)}] {row.clip}  t={row.timestamp:.1f}s  "
              f"score={row.score:.2f}  confidence={row.confidence:.2f}")
        print(f"        watching: {view.name}")
        open_in_player(view)

        choice = input(PROMPT).strip().lower()[:1]
        if choice == "q":
            break
        if choice not in CHOICES:
            print("        skipped (it will come back next run)")
            continue
        append_label(args.labels, interval_for(row, CHOICES[choice]))
        print(f"        recorded {CHOICES[choice].value} against {row.clip}")

    print(f"\nlabels -> {args.labels}")
    return 0


def _view_for(row: ScanRow, args: argparse.Namespace, crops: list[str]) -> Path:
    """The crop if there is one, the full frame if there is not.

    94% of the corpus has a crop partner, and watching the crop is several times faster.
    The other 6% get the wide shot -- slower, but a slow label beats a missing one.
    """
    partner = crop_partner(row.clip, crops)
    if partner is not None and (args.crops / partner).is_file():
        return args.crops / partner
    return args.clips / row.clip
```

with the imports:

```python
from qorgan.evaluation.labelling import (
    already_labelled,
    append_label,
    crop_partner,
    interval_for,
    is_done,
    open_in_player,
)
from qorgan.evaluation.labels import LabelKind, LabelSet, load_labels, write_template
from qorgan.evaluation.scan import ScanRow, load_candidates, rows_for, write_candidates
```

- [ ] **Step 7: Check the file limits.** `cli.py` is growing; `tests/test_code_limits.py` enforces 500 lines/file and 50 lines/function.

```
.venv/Scripts/python.exe -m pytest tests/test_code_limits.py -q
```

If `cli.py` is over 500 lines, split the scan/label commands into `src/qorgan/evaluation/cli_label.py` and have `add_parser` call `add_label_parsers(sub)` from it. Do not let it grow past the limit.

- [ ] **Step 8: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `~810 passed`, `All checks passed!`.

- [ ] **Step 9: Commit.**

```
git status
git add src/qorgan/evaluation/labelling.py src/qorgan/evaluation/cli.py tests/test_eval_label.py
git commit -m "qorgan eval label: watch the crop, record the full frame

The old detector's cropped pair ROI is the review view a human actually wants:
two children, close up, instead of a 1440p wide shot in which they are 3% of the
pixels. Joining on (camera, track_a, track_b) + nearest timestamp, 621 of 660
full-frame clips (94%) have one.

The crop can never be detector input -- no scene, no zones, no frame geometry --
so the label is written against the FULL-FRAME video_id, whichever view was
watched. The crop is a lens, never the record. Resumable; a dev tool, no route."
```

---

### Task 9: `qorgan eval sample` — the ~80 clips the detector did *not* fire on

Spec §5 (pass 2). Precision alone is half a detector. So we also label a random sample of **~80 clips the detector did not fire on.**

Be precise about what that measures: not *"fights we miss in the world"* — that needs footage we do not have — but **"fights we miss among clips the old detector flagged"**. That is a real number, and it is the one that says whether we kept the true positives while dropping the false ones.

**Files:**
- `src/qorgan/evaluation/scan.py`
- `src/qorgan/evaluation/video.py`
- `src/qorgan/evaluation/cli.py`
- `tests/test_eval_scan.py`

**Interfaces:**
- Produces: `qorgan.evaluation.scan.sample_quiet(clips: Sequence[str], fired: Iterable[str], *, count: int, seed: int) -> list[str]`
- Produces: `qorgan.evaluation.video.clip_duration(path: Path) -> float` (OpenCV only; no YOLO, no GPU)
- Produces: `qorgan.evaluation.cli.cmd_sample(args) -> int` — `qorgan eval sample [--clips DIR] [--candidates FILE] [--out FILE] [--count 80] [--seed 7]`
- Produces: `qorgan.evaluation.cli.SAMPLE_PATH = Path("eval/sample.csv")`
- Reuses: `write_candidates` / `load_candidates` — a sample row is a `ScanRow` whose `score`/`probability`/`confidence` are 0.0 and whose `timestamp` is the clip's midpoint, so `qorgan eval label --candidates eval/sample.csv` labels it with the same tool

**Steps:**

- [ ] **Step 1: Write the failing test.** Append to `tests/test_eval_scan.py`:

```python
def test_the_sample_only_draws_from_clips_the_detector_did_NOT_fire_on() -> None:
    """**The only recall signal this corpus can give**, and it is worth being precise
    about what it is: not "fights we miss in the world" -- that needs footage of a fight
    the old detector missed, which by construction does not exist -- but "fights we miss
    among the clips the old detector flagged". That is a real number, and it is the one
    that says whether we kept the true positives while dropping the false ones.
    """
    clips = [f"hall_left_main_1_2_2026070{i}_120000_000000.mp4" for i in range(1, 9)]
    fired = {clips[0], clips[3]}

    sample = sample_quiet(clips, fired, count=4, seed=7)

    assert len(sample) == 4
    assert not set(sample) & fired, "a clip the detector fired on is not a recall probe"
    assert len(set(sample)) == 4, "the same clip was drawn twice"


def test_the_sample_is_reproducible() -> None:
    clips = [f"hall_left_main_1_2_2026070{i}_120000_000000.mp4" for i in range(1, 9)]

    assert sample_quiet(clips, [], count=3, seed=7) == sample_quiet(clips, [], count=3, seed=7)


def test_asking_for_more_quiet_clips_than_exist_gives_all_of_them() -> None:
    clips = [f"hall_left_main_1_2_2026070{i}_120000_000000.mp4" for i in range(1, 4)]

    assert len(sample_quiet(clips, [clips[0]], count=80, seed=7)) == 2
```

with `sample_quiet` added to the `qorgan.evaluation.scan` import line.

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_eval_scan.py -q -k sample
```

Expected failure: `ImportError: cannot import name 'sample_quiet' from 'qorgan.evaluation.scan'`.

- [ ] **Step 3: Add `sample_quiet` to `src/qorgan/evaluation/scan.py`.**

```python
def sample_quiet(
    clips: Sequence[str],
    fired: Iterable[str],
    *,
    count: int = 80,
    seed: int = 7,
) -> list[str]:
    """A random sample of the clips the detector said NOTHING about.

    Pass 1 (`rows_for`) yields precision. Precision alone is half a detector: it cannot
    tell you what you missed. So a human also watches ~80 clips the detector was silent
    on.

    **This is the only recall signal this corpus can give, and it is not recall against
    the world.** A clip only exists here because the OLD detector fired on it, so there is
    no footage of a fight anything missed. What this measures is "fights we miss among
    clips the old detector flagged" -- which is exactly the question "did we keep the true
    positives while dropping the false ones?", and that question is worth answering.
    """
    quiet = sorted(set(clips) - set(fired))
    rng = random.Random(seed)  # noqa: S311 - choosing clips to watch, not keying a cipher
    return rng.sample(quiet, min(count, len(quiet)))
```

with the imports extended: `import random` and `from collections.abc import Iterable, Sequence`.

- [ ] **Step 4: Add `clip_duration` to `src/qorgan/evaluation/video.py`.**

```python
def clip_duration(path: Path) -> float:
    """How long this clip is, in seconds. OpenCV only -- no model, no GPU.

    Needed twice: to put the sample's label interval over the WHOLE clip (the detector
    said nothing, so there is no moment to centre on), and to report false alerts per
    HOUR, which is the number the school actually feels.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return frames / fps if fps > 0 else 0.0
    finally:
        capture.release()
```

- [ ] **Step 5: Add the `sample` subcommand.** In `src/qorgan/evaluation/cli.py`, add `SAMPLE_PATH = Path("eval/sample.csv")`, register:

```python
    sample_cmd = sub.add_parser(
        "sample",
        help="draw ~80 clips the detector did NOT fire on, to label for recall",
        description=(
            "Pass 2. Precision alone is half a detector. This is the only recall signal "
            "this corpus can give -- and it is NOT recall against the world: a clip only "
            "exists here because the OLD detector fired on it. It measures 'fights we "
            "miss among clips the old detector flagged', which is the question of whether "
            "we kept the true positives while dropping the false ones."
        ),
    )
    sample_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    sample_cmd.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    sample_cmd.add_argument("--out", type=Path, default=SAMPLE_PATH)
    sample_cmd.add_argument("--count", type=int, default=80)
    sample_cmd.add_argument("--seed", type=int, default=7)
    sample_cmd.set_defaults(func=cmd_sample)
```

and add:

```python
def cmd_sample(args: argparse.Namespace) -> int:
    from qorgan.evaluation.video import clip_duration  # cv2 only; keep the import local

    clips = [clip.name for clip in _clips_in(args.clips)]
    fired = {row.clip for row in load_candidates(args.candidates)}
    chosen = sample_quiet(clips, fired, count=args.count, seed=args.seed)

    # The detector said nothing, so there is no moment to centre on: the human judges the
    # WHOLE clip. timestamp = midpoint, and `interval_for`'s +/-2 s pad is widened to the
    # clip by writing the duration into it below.
    rows = [
        ScanRow(
            clip=name,
            timestamp=round(clip_duration(args.clips / name) / 2, 3),
            score=0.0,
            probability=0.0,
            confidence=0.0,
        )
        for name in chosen
    ]
    written = write_candidates(rows, args.out)

    print(f"{len(clips)} clip(s), {len(fired)} fired on, {len(clips) - len(fired)} quiet.")
    print(f"{written} drawn (seed {args.seed}) -> {args.out}")
    print("\nThis measures recall among clips the OLD detector flagged. It is NOT recall")
    print("against the world -- no clip here is a fight anything missed. See docs/questions-for-school.md.")
    print(f"\nNow label them:  qorgan eval label --candidates {args.out}")
    return 0
```

with `sample_quiet` added to the `qorgan.evaluation.scan` import line.

- [ ] **Step 6: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `~813 passed`, `All checks passed!`.

- [ ] **Step 7: Commit.**

```
git status
git add src/qorgan/evaluation/scan.py src/qorgan/evaluation/video.py src/qorgan/evaluation/cli.py tests/test_eval_scan.py
git commit -m "qorgan eval sample: the ~80 clips the detector was silent on

Precision alone is half a detector. This is the only recall signal the corpus can
give, and it is not recall against the world: a clip only exists here because the
OLD detector fired on it, so there is no footage of a fight anything missed.

It measures 'fights we miss among clips the old detector flagged' -- which is
exactly whether we kept the true positives while dropping the false ones."
```

---

### Task 10: `qorgan eval run` — per camera, and **false alerts per hour**

Spec §6, §2.3. `eval run` reports precision, recall, F1 and a PR curve — **per camera**, now that Task 3 makes per-camera evaluation correct.

But the figure that decides whether staff leave the system switched on is not F1. It is **false alerts per hour**. 97 minutes of adversarially-selected corridor measures it directly, and it goes in the report in bold. *A detector with excellent F1 that wakes a teacher twice a night will be unplugged within a week, and it will not matter what its F1 was.*

And the report must say which cameras this corpus does **not** calibrate: stairs has 17 clips with no fights in them, and the yard has **zero**.

**Files:**
- `src/qorgan/evaluation/metrics.py`
- `src/qorgan/evaluation/harness.py`
- `src/qorgan/evaluation/cli.py`
- `src/qorgan/evaluation/__init__.py`
- `tests/test_evaluation.py`

**Interfaces:**
- Produces: `qorgan.evaluation.metrics.SECONDS_PER_HOUR: float = 3600.0`
- Produces: `qorgan.evaluation.metrics.false_alerts_per_hour(metrics: Metrics, footage_seconds: float) -> float`
- Changes: `qorgan.evaluation.harness.RunResult` gains `duration_seconds: float = 0.0`, set by `run()` to the last timestamp it saw
- Produces: `qorgan.evaluation.cli._print_per_camera(labels: LabelSet, results: list[RunResult], threshold: float) -> None`
- Produces: `qorgan.evaluation.cli._false_alerts_line(metrics: Metrics, footage_seconds: float) -> str`
- Produces: `qorgan.evaluation.cli._print_uncalibrated(seen: set[str]) -> None`
- Consumes: `qorgan.evaluation.clips.parse_clip_name` (to group results by camera)

**Steps:**

- [ ] **Step 1: Write the failing test.** Append to `tests/test_evaluation.py`:

```python
def test_false_alerts_per_hour_is_the_number_the_school_actually_feels(tmp_path: Path) -> None:
    """**THE number.** A detector with an excellent F1 that wakes a teacher twice a night
    will be unplugged within a week, and it will not matter what its F1 was.

    97 minutes of adversarially-selected corridor -- every clip a trigger of the OLD
    detector, and mostly a false positive -- measures it directly.
    """
    labels = _one_fight(tmp_path)
    noise = [Prediction("hall.mp4", 100.0 + i * 10, 0.9) for i in range(6)]
    metrics = evaluate(labels, noise)

    assert metrics.false_positives == 6
    assert false_alerts_per_hour(metrics, footage_seconds=1800.0) == 12.0


def test_no_footage_is_not_a_false_alert_rate_of_zero(tmp_path: Path) -> None:
    """Zero hours of footage and zero alerts is not "a perfectly quiet detector"."""
    metrics = evaluate(_one_fight(tmp_path), [])

    assert false_alerts_per_hour(metrics, footage_seconds=0.0) == math.inf


def test_the_run_result_knows_how_long_the_clip_was() -> None:
    """Without a duration there is no per-hour anything."""
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig())

    assert result.duration_seconds > 0
    assert result.duration_seconds == pytest.approx((result.frames - 1) * STEP, rel=0.01)
```

with `import math` and the import line extended:

```python
from qorgan.evaluation.metrics import Metrics, Prediction, best_threshold, false_alerts_per_hour
```

- [ ] **Step 2: Run it and see it fail.**

```
.venv/Scripts/python.exe -m pytest tests/test_evaluation.py -q -k "per_hour or how_long"
```

Expected failure: `ImportError: cannot import name 'false_alerts_per_hour' from 'qorgan.evaluation.metrics'`.

- [ ] **Step 3: Add `false_alerts_per_hour` to `src/qorgan/evaluation/metrics.py`.**

```python
SECONDS_PER_HOUR = 3600.0


def false_alerts_per_hour(metrics: Metrics, footage_seconds: float) -> float:
    """**The number the school actually feels.**

    Not F1. A detector with an excellent F1 that wakes a teacher twice a night will be
    unplugged within a week, and it will not matter what its F1 was. Precision is a ratio
    and a ratio hides the denominator; this is the count of times a human is interrupted
    for nothing, per hour of corridor.

    Zero footage returns infinity, not zero: no measurement is not a clean bill of health.
    """
    hours = footage_seconds / SECONDS_PER_HOUR
    if hours <= 0:
        return math.inf
    return metrics.false_positives / hours
```

with `import math` at the top of the file.

- [ ] **Step 4: Give `RunResult` a duration.** In `src/qorgan/evaluation/harness.py`, add to `RunResult`:

```python
    # The clip's own length, as the detector saw it: the last timestamp it was handed.
    # Without this there is no per-HOUR anything, and per-hour is the number the school
    # actually feels.
    duration_seconds: float = 0.0
```

and in `run()`, track it:

```python
    duration = 0.0

    for timestamp, detections in source:
        frames += 1
        duration = max(duration, timestamp)
        ...

    return RunResult(
        video_id=source.video_id,
        alerts=alerts,
        frames=frames,
        candidates=candidates,
        suppressed_by=suppressed,
        duration_seconds=duration,
    )
```

- [ ] **Step 5: Export it.** In `src/qorgan/evaluation/__init__.py`, add `false_alerts_per_hour` to the `metrics` import and to `__all__`.

- [ ] **Step 6: Report it.** In `src/qorgan/evaluation/cli.py`, replace `cmd_run` and add the three helpers:

```python
def cmd_run(args: argparse.Namespace) -> int:
    labels, results = _evaluate(args)
    predictions = [p for r in results for p in r.predictions]
    footage = sum(r.duration_seconds for r in results)

    metrics = evaluate(labels, predictions, threshold=args.threshold)
    print(f"\n{len(results)} clip(s), {footage / 60:.1f} min, {len(predictions)} alert(s)\n")
    print(metrics.summary())
    print(_false_alerts_line(metrics, footage))

    _print_per_camera(labels, results, args.threshold)
    _print_curve(labels, predictions)
    _print_suppressions(results)
    _print_uncalibrated({parse_clip_name(r.video_id).camera for r in results})
    return 0


def _false_alerts_line(metrics: Metrics, footage_seconds: float) -> str:
    """THE number, and it goes in the report in bold. A detector with an excellent F1
    that wakes a teacher twice a night will be unplugged within a week."""
    rate = false_alerts_per_hour(metrics, footage_seconds)
    return (
        f"\n**FALSE ALERTS PER HOUR: {rate:.2f}**"
        f"   ({metrics.false_positives} in {footage_seconds / 3600:.2f} h of footage)"
    )


def _print_per_camera(labels: LabelSet, results: list[RunResult], threshold: float) -> None:
    """Per camera, because Task 3 made per-camera evaluation correct: hall_left masks a
    reflective column that hall_right cannot see, and one number over both would average
    two different detectors."""
    by_camera: dict[str, list[RunResult]] = {}
    for result in results:
        by_camera.setdefault(parse_clip_name(result.video_id).camera, []).append(result)

    print("\nper camera")
    for name, group in sorted(by_camera.items()):
        videos = {r.video_id for r in group}
        subset = LabelSet(tuple(i for i in labels.intervals if i.video_id in videos))
        predictions = [p for r in group for p in r.predictions]
        footage = sum(r.duration_seconds for r in group)
        metrics = evaluate(subset, predictions, threshold=threshold)

        print(f"  {name:<14} {metrics.summary()}")
        print(
            f"  {'':<14} false alerts/hour {false_alerts_per_hour(metrics, footage):.2f}"
            f"   ({len(group)} clip(s), {footage / 60:.1f} min)"
        )


def _print_uncalibrated(seen: set[str]) -> None:
    """The honest half. Seventeen clips with no fights in them is not a calibration, and
    zero clips is not one either."""
    missing = [
        name
        for name, camera in sorted(load_cameras().items())
        if isinstance(camera, BullyingCamera) and name not in seen
    ]
    if not missing:
        return

    print("\nNOT CALIBRATED BY THIS CORPUS — every threshold on these is still a guess:")
    for name in missing:
        print(f"  {name}")
```

with the imports extended:

```python
from qorgan.evaluation.clips import ClipNameError, camera_for, parse_clip_name
from qorgan.evaluation.metrics import (
    Metrics,
    Prediction,
    best_threshold,
    evaluate,
    false_alerts_per_hour,
    pr_curve,
)
```

- [ ] **Step 7: Suite and lint.**

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Expected: `~816 passed`, `All checks passed!`.

- [ ] **Step 8: Commit.**

```
git status
git add src/qorgan/evaluation/metrics.py src/qorgan/evaluation/harness.py src/qorgan/evaluation/cli.py src/qorgan/evaluation/__init__.py tests/test_evaluation.py
git commit -m "eval run: per camera, and FALSE ALERTS PER HOUR

The figure that decides whether staff leave the system switched on is not F1. A
detector with an excellent F1 that wakes a teacher twice a night is unplugged
within a week, and it will not matter what its F1 was. 97 minutes of
adversarially-selected corridor measures it directly.

Per camera, now that the camera comes from the clip. And it names the cameras
this corpus does NOT calibrate -- the stairs (17 clips) and the yard (0)."
```

---

### Task 11: Choose the operating point, save the baseline, write the honest report

Spec §6 (the last two sentences) and §7. Choose the operating point → set `notify_threshold` → `qorgan eval save-baseline` → `qorgan eval gate` before a threshold change lands. Then the report: what we know, with numbers. **What we do not:**

- **Recall against the world is unmeasured** (§2.2). There is no footage of a fight the old detector missed, because a clip only exists if the old detector fired.
- **Stairs and yard are uncalibrated** (§2.3). Seventeen clips with no fights in them is not a calibration; zero clips is not one either.
- Every number is from hall cameras at 10 fps in daylight.

> **`eval gate` cannot run in CI as the spec imagines it.** The corpus is gitignored — it is video of children — so no CI runner can see `eval/clips/`. The gate is therefore a **local pre-merge command**, documented in the report and in `CLAUDE.md`, and the committed artefacts (`eval/baseline.json`, `eval/labels.csv`) are what a reviewer checks. Raise this with the human before assuming otherwise.

**Files:**
- `docs/eval-report.md` (new)
- `eval/baseline.json` (new — committed; it is a *result*, not footage)
- `eval/labels.csv` (updated by the labelling runs — committed)
- `config/profiles/hall.yaml` (the chosen `notify_threshold`, if it moves)

**Interfaces:**
- Consumes: `qorgan eval scan`, `qorgan eval sample`, `qorgan eval label`, `qorgan eval run`, `qorgan eval save-baseline`, `qorgan eval gate`
- Consumes: `docs/questions-for-school.md` (the written request that names the recall gap)
- Produces: `docs/eval-report.md`

**Steps:**

- [ ] **Step 1: Put the corpus in place.** The 663 full-frame clips go to `eval/clips/`, the 1 293 crop ROIs to `eval/crops/`. Both are gitignored. Verify before doing anything else:

```
git status --short
git check-ignore -v eval/clips eval/crops
```

Expected: `git status` shows **nothing** under `eval/clips/` or `eval/crops/`. If it shows anything, stop: that is footage of children and it must never reach a commit.

- [ ] **Step 2: Scan, sample, label.**

```
.venv/Scripts/python.exe -m qorgan eval scan
.venv/Scripts/python.exe -m qorgan eval sample --count 80
.venv/Scripts/python.exe -m qorgan eval label
.venv/Scripts/python.exe -m qorgan eval label --candidates eval/sample.csv
```

- [ ] **Step 3: Read the curve, and choose.**

```
.venv/Scripts/python.exe -m qorgan eval run
```

Read the per-camera block, the **FALSE ALERTS PER HOUR** line, and the PR curve. The best-F1 threshold is a starting point, not a verdict: **in a school a missed fight and a false alarm are not worth the same.** Choose the operating point deliberately, and if it differs from `0.85`, set `bullying.confidence.notify_threshold` in `config/profiles/hall.yaml` — remembering `cap_without_skeleton` must stay strictly below it (the schema enforces this; `Confidence._cap_below_notify` will refuse otherwise).

- [ ] **Step 4: Record the baseline.**

```
.venv/Scripts/python.exe -m qorgan eval save-baseline --note "first calibration against the school's 663-clip corpus"
.venv/Scripts/python.exe -m qorgan eval gate
```

Expected: `gate` passes against the baseline just written.

- [ ] **Step 5: Write `docs/eval-report.md`.** It must contain, in this order:

1. **What the corpus is** — 663 full-frame clips, ~97 minutes, 344 `hall_right` + 299 `hall_left` + 17 stairs + 0 yard. All of it is the *old* detector's trigger clips, and they were mostly false positives: 97 minutes of ordinary school corridor, **adversarially selected against a bullying detector**. That is the best possible negative set.
2. **The numbers** — precision, recall, F1, per camera, at the chosen threshold; the PR curve; and, in bold, **false alerts per hour**, overall and per camera.
3. **What we do not know**, with equal weight:
   - Recall against the world is unmeasured, and *cannot* be measured from this data. A clip only exists here because the old detector fired on it, so there is no footage of a fight anything missed. The recall figure above is *"fights we miss among clips the old detector flagged"* — a real number, and not the one a parent would ask for.
   - The stairs and the yard are **uncalibrated**. Seventeen clips with no fights in them is not a calibration; zero clips is not one either. Every threshold on those cameras is still a guess.
   - Every number is from hall cameras, at 10 fps, in daylight.
4. **The gate** — `qorgan eval gate` runs locally against `eval/baseline.json` before a threshold change lands. It cannot run in CI: the corpus is footage of children and is gitignored.
5. **The ask** — link `docs/questions-for-school.md`. The recall gap is the school's to fill, and this report names it rather than papering over it.

- [ ] **Step 6: Commit the results — and nothing else.**

```
git status
git check-ignore -v eval/clips eval/crops eval/candidates.csv eval/sample.csv
git add docs/eval-report.md eval/baseline.json eval/labels.csv config/profiles/hall.yaml
git commit -m "The first calibration: what the detector does, and what we still do not know

Precision, recall, F1 and the PR curve per camera against the school's 663-clip
corpus -- and the number that decides whether the system stays switched on, which
is false alerts per hour.

And the half nobody enjoys writing: recall against the world is unmeasured and
cannot be measured from this data (a clip only exists here because the OLD
detector fired), and the stairs and the yard are not calibrated at all."
```

---

## Self-review

**(a) Every spec section maps to a task.**

| Spec § | Task |
|---|---|
| §2 the corpus is 663 clips | 7 (scan), 11 (put it in `eval/clips/`) |
| §2.1 label from the crop, evaluate on the full frame | 8 |
| §2.2 the corpus cannot measure recall against the world | 9 (named in `sample_quiet`'s docstring and the CLI output), 11 (the report) |
| §2.3 the corpus calibrates the hall and only the hall | 10 (`_print_uncalibrated`), 11 (the report) |
| §3.1 the harness never runs the skeleton | 2 |
| §3.2 production does not resize; the harness does | 1 |
| §3.3 the clip's camera must come from the clip | 3 |
| §3.4 `VideoSource` device | 4 |
| §3.4 stairs px/frame vs px/second | 5 |
| §3.5 a test that fails when a config key is read nowhere | 6 |
| §4 the corpus, gitignored | 7 (`.gitignore`), 11 (verified before every commit) |
| §5 pass 1, `eval scan` | 7 |
| §5 pass 2, `eval sample` | 9 |
| §5 `eval label`, resumable, crop partner, no web route | 8 |
| §6 PR curve per camera; false alerts per hour | 10 |
| §6 operating point → `notify_threshold` → `save-baseline` → `gate` | 11 |
| §7 the honest report | 11 |
| §8 testing table | every row is a named test: `prepare_frame` shared (T1 S1), harness runs the skeleton (T2 S1), camera from filename (T3 S1), stairs units (T5 S1), no dead config keys (T6 S1), scan/label round-trip and resumable append (T7 S1, T8 S1) |
| §9 what this spec does not do | Nothing here touches `faces/`, `identity/` or `canteen/` **behaviour**. Task 6 deletes unread *config keys* in `config/canteen.py` — flagged in that task as needing coordination with Spec A. |

**(b) No placeholders.** Every code step gives complete, runnable code. The only step whose content is prose rather than code is Task 11 Step 5 (the report), which is a document whose numbers do not exist until Tasks 7–10 have been run against the real corpus — writing invented numbers into it here would be exactly the fiction this spec exists to end.

**(c) Name and type consistency.** `prepare_frame(image, capture)` is imported by name into `worker.camera_loop` and `evaluation.video`, and Task 1's test asserts the two are the same object. `validate_candidate(candidate, crops, pose, config)` is imported by name into `worker.bullying` and `evaluation.harness`, and Task 2's test asserts the same. `SkeletonView` is the one protocol; `PoseEstimator`, `NoPose` and `SpyPose` all satisfy it. `ScanRow` is produced by `scan.rows_for` and `cli.cmd_sample`, and consumed by `labelling.interval_for` and `cli.cmd_label` — one type, one shape, both passes. `ClipName.camera` is what `camera_for` resolves against `load_cameras()` and what `_print_per_camera` groups on. `duration_seconds` is set once in `harness.run` and read in `cli` for the per-hour figure; `clip_duration` (OpenCV) is the separate, file-level function `cmd_sample` needs before any detector has run.
</content>
</invoke>

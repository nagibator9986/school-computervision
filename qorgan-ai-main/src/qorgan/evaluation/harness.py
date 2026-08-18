"""Run the production detector over labelled footage.

**This is rule R2 made real.** The harness imports `BullyingDetector` — the same object
the worker runs, not a copy of it. The legacy had the detection logic in three files that
had already diverged, so its harness measured code that did not run in production and
every threshold decision made against it was made against a fiction.

It also runs the **full** pipeline, including the skeleton validation tier and the
confidence cap — through `qorgan.models.validate.validate_candidate`, the same function
the worker's slow tier calls. Until this existed the harness never ran the skeleton at
all, so every verdict was capped at 0.72 while Telegram fires at 0.85, and the PR curve
was empty exactly where the decision lives.

The frame source is injectable, so this module needs no video decoder and no GPU to be
tested — and a synthetic scene can prove the harness itself is correct before a single
real clip arrives.
"""

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

    def crops(self, boxes: tuple[Box, Box], candidate_timestamp: float) -> list[np.ndarray]:
        """The recent frames, cut down to this pair, for the pose model.

        A fixture has no pixels and returns []. `PoseEstimator` then reports `skipped`,
        and the cap applies -- which is exactly what a GPU-free decision test wants.

        `candidate_timestamp` is the candidate's own moment: a real source must refuse
        (raise) rather than crop whatever frames it currently holds if that moment has
        already fallen out of its buffer.
        """
        ...


@dataclass(frozen=True, slots=True)
class Alert:
    """What the worker would actually have recorded and sent."""

    video_id: str
    timestamp: float
    key: tuple[int, int]
    # The RAW aggression score, which Verdict does not carry: `eval scan` writes it out
    # so a human reading candidates.csv can see WHY the detector fired, not just how sure
    # it ended up after the skeleton tier had its say.
    score: float
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
            crops = source.crops(candidate.boxes, candidate.timestamp)
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

    # Remembered under the key of the INCIDENT, not of this judgement, exactly as the
    # worker does -- the worker only ever remembers a key it wrote an event for. Filing a
    # re-tracked pair under its own fresh ids would invent a second incident that the
    # worker does not have, and the notification below would then fire for it twice.
    incident = decision.merged_into or candidate.key
    merger.remember(incident, candidate.timestamp, verdict.confidence, candidate.center)

    return Alert(
        video_id=video_id,
        timestamp=candidate.timestamp,
        key=candidate.key,
        score=candidate.score,
        verdict=verdict,
        merged=not decision.is_new,
        # One incident, one message -- and the message is raised the first time the
        # incident crosses the bar, which is frequently a merged judgement rather than
        # the first. Claimed through the SAME merger method the worker notifies from, so
        # what the eval reports and what the worker sends cannot drift apart (rule R2).
        notified=merger.claim_notification(incident, verdict.should_notify(config.confidence)),
    )

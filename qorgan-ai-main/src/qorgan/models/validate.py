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

"""The judgements behind each `TelegramSkipReason`, shared by the two modules that test it.

`test_telegram_skip_reason.py` proves each reason is what `detection.validation.judge`
actually decides. `test_telegram_skip_reason_recorded.py` proves that decision survives to
the event row and onto the screen. Both halves must be talking about the same judgements,
so the table lives here once: two copies would let the second file go on passing about a
judgement the first one no longer makes — which is the shape of every bug this project's
rule R2 exists to prevent.
"""

from __future__ import annotations

from qorgan.config.bullying import Confidence
from qorgan.detection.validation import SkeletonResult, score_skeleton
from qorgan.enums import TelegramSkipReason

CONFIG = Confidence()

# The heuristics' own confidence in the pair, as the fast tier hands it over.
PROBABILITY = 0.9

# Hands, contact, a kick, a sharp displacement, and a child on the floor.
FULL_FIGHT = (
    "rapid_hand_motion",
    "close_upper_body_contact",
    "kick_like_leg_motion",
    "sudden_body_displacement",
    "body_fall_or_low_posture",
)
# Gesturing and standing close: §5.9's weak social contact, and nothing else.
PLAYGROUND = ("rapid_hand_motion", "close_upper_body_contact")
# Suggestive on its own and no more: a box also jumps when the detector re-anchors.
A_SHRUG = ("sudden_body_displacement",)


def looked(reasons: tuple[str, ...]) -> SkeletonResult:
    """A pose model that saw exactly these things.

    Scored by the SAME function the real model's result is scored by, so these fixtures
    cannot quietly stop meaning what their names say if the weights are retuned.
    """
    return SkeletonResult(reasons=reasons, score=score_skeleton(list(reasons)), skipped=False)


COULD_NOT_LOOK = SkeletonResult(skipped=True, skip_reason="insufficient_frames")
CONFIRMED = looked(FULL_FIGHT)

# Every reason `judge` decides, and a (candidate probability, skeleton) that produces it.
# ALREADY_NOTIFIED is absent because it is the merger's call rather than the confidence's;
# `test_no_reason_exists_that_nothing_can_produce` asserts that this table plus that one
# member is the whole enum, so a new member cannot be added without showing it can happen.
FROM_A_JUDGEMENT: dict[TelegramSkipReason, tuple[float, SkeletonResult]] = {
    # The pose model never got to look. 0.9 * 0.7 = 0.63, the heuristics' alone.
    TelegramSkipReason.SKELETON_NOT_RUN: (PROBABILITY, COULD_NOT_LOOK),
    # It looked and scored 0.15, under the 0.45 bar; the blend takes the disagreement hit.
    TelegramSkipReason.NO_SKELETON_CONFIRMATION: (PROBABILITY, looked(A_SHRUG)),
    # It looked, reached the bar exactly, and saw nothing but weak evidence — so the cap
    # holds the confidence at 0.72, below the 0.85 notify bar by the config validator.
    TelegramSkipReason.WEAK_EVIDENCE_ONLY: (PROBABILITY, looked(PLAYGROUND)),
    # Confirmed, and the heuristics still unconvinced at 0.3, so the blend lands at 0.56.
    # Nothing is capped here; the sum is simply short.
    TelegramSkipReason.LOW_CONFIDENCE: (0.3, CONFIRMED),
}

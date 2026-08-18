"""The slow tier: skeleton evidence, and turning it into a confidence.

**The victim-evidence hierarchy is the single most important anti-false-positive idea in
the system, and it is worth stating plainly.**

Not all evidence that "something happened" is evidence that *an assault* happened:

  * `body_fall_or_low_posture` is CLEAN. Somebody went down. Nothing in ordinary school
    life puts a child on the floor, so this is real victim evidence on its own.

  * `sudden_body_displacement` is MOTION-ONLY. Somebody moved sharply -- but a bounding
    box also jumps when the detector re-anchors, when a person is briefly occluded, or
    simply because they are walking towards the camera. It is suggestive, never
    sufficient.

  * `rapid_hand_motion`, `close_upper_body_contact` and `kick_like_leg_motion` are WEAK.
    Children gesture, stand close, and swing their legs constantly. On their own these
    mean nothing at all, and treating them as evidence is how you get a system that
    cries wolf at a game of tag.

The confidence cap enforces the hierarchy mechanically: an event the skeleton did not
confirm is capped below the notification threshold, so it *cannot* raise a Telegram
alert no matter how excited the heuristics got. That inequality is validated at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from qorgan.config.bullying import Confidence as ConfidenceConfig
from qorgan.enums import TelegramSkipReason


class Evidence(StrEnum):
    """How much a skeleton reason is worth on its own."""

    CLEAN = "clean"  # sufficient by itself
    MOTION_ONLY = "motion_only"  # suggestive; could be perspective or box jitter
    WEAK = "weak"  # never sufficient alone


# The five skeleton features, and what each is actually worth. These strings are the
# contract between the skeleton model and the gates: the legacy had them as bare
# literals in two files, so renaming one silently disabled a gate in the other.
REASON_EVIDENCE: dict[str, Evidence] = {
    "body_fall_or_low_posture": Evidence.CLEAN,
    "sudden_body_displacement": Evidence.MOTION_ONLY,
    "rapid_hand_motion": Evidence.WEAK,
    "close_upper_body_contact": Evidence.WEAK,
    "kick_like_leg_motion": Evidence.WEAK,
}

REASON_WEIGHT: dict[str, float] = {
    "rapid_hand_motion": 0.25,
    "body_fall_or_low_posture": 0.20,
    "close_upper_body_contact": 0.20,
    "kick_like_leg_motion": 0.20,
    "sudden_body_displacement": 0.15,
}

# "Aggressive" means the skeleton score reached the confirmation bar, and that bar is ONE
# config field (Confidence.min_skeleton_score_for_confirmation), passed in by the caller
# that holds the config. There is no module-level AGGRESSIVE_SCORE constant any more. It was
# 0.45, and `_skeleton_confirms` required it AND the config knob, so the config knob was
# inert across its whole lower half. A hardcoded bar that silently shadows a config bar is
# the exact lie the rest of this system is built to prevent.


@dataclass(frozen=True, slots=True)
class SkeletonResult:
    """What the pose model saw. `skipped` means it could not look, not that it saw nothing."""

    reasons: tuple[str, ...] = ()
    score: float = 0.0
    skipped: bool = True
    skip_reason: str = "not_run"

    def is_aggressive(self, threshold: float) -> bool:
        """Did the skeleton look, and score at or above the confirmation bar? The bar is a
        config field the caller passes in; nothing here hardcodes it, so nothing can
        silently shadow it."""
        return not self.skipped and self.score >= threshold

    @property
    def clean_evidence(self) -> tuple[str, ...]:
        """Reasons that stand on their own. In practice: did somebody go down?"""
        return tuple(r for r in self.reasons if REASON_EVIDENCE.get(r) is Evidence.CLEAN)

    @property
    def has_clean_evidence(self) -> bool:
        return bool(self.clean_evidence)

    @property
    def only_weak_evidence(self) -> bool:
        """Nothing here but gesturing, standing close and kicking about. That is a
        playground, not an assault."""
        if not self.reasons:
            return False
        return all(REASON_EVIDENCE.get(r) is Evidence.WEAK for r in self.reasons)


def score_skeleton(reasons: list[str]) -> float:
    """Sum the weights of whatever the pose model found, capped at 1.0.

    `kick_like_leg_motion` is deliberately NOT counted on its own: a child kicking a
    ball, or just walking with a long stride, produces it constantly. It only counts
    when there is other evidence that the two people are actually engaged.
    """
    found = set(reasons)
    corroborated = bool(
        found & {"close_upper_body_contact", "sudden_body_displacement", "body_fall_or_low_posture"}
    )

    total = 0.0
    for reason in found:
        if reason == "kick_like_leg_motion" and not corroborated:
            continue  # walking, not kicking someone
        total += REASON_WEIGHT.get(reason, 0.0)

    return min(1.0, total)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The final judgement on one candidate."""

    confidence: float
    candidate_probability: float
    validation_score: float
    skeleton_confirmed: bool
    capped: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    # Why a human will NOT be woken up, or None if this judgement wants to wake one
    # (client §7's `_telegram_skip_reason`). Set by `judge`, which is the only place that
    # holds every input the answer depends on: the skeleton, the evidence and the config.
    #
    # **Recorded here rather than recomputed downstream, and that is the point.** The
    # obvious alternative is to work the reason out later from the stored confidence and
    # the camera's thresholds. That is a second source of truth for a decision already
    # made, and it drifts the moment anyone retunes a threshold: an event withheld last
    # March would start explaining itself with this month's numbers. The school is asking
    # about a specific afternoon.
    telegram_skip_reason: TelegramSkipReason | None = None

    def is_alert(self, config: ConfidenceConfig) -> bool:
        return self.confidence >= config.alert_threshold

    def is_critical(self, config: ConfidenceConfig) -> bool:
        return self.confidence >= config.critical_threshold

    def should_notify(self, config: ConfidenceConfig) -> bool:
        """The ONE place that decides whether a human gets woken up.

        Legacy had this threshold hardcoded in the worker, hardcoded again in the
        Telegram service, and quoted as a third value in the config comments.

        `judge` asks this once, at the moment of judgement, and carries the answer on
        `telegram_skip_reason`. The bullying worker reads that answer instead of asking
        again — one comparison per verdict, so the message a school is shown and the
        decision that produced it cannot be two different numbers.
        """
        return self.confidence >= config.notify_threshold


def judge(
    candidate_probability: float,
    validation_score: float,
    skeleton: SkeletonResult,
    config: ConfidenceConfig,
) -> Verdict:
    """Blend the heuristics with the validators, then apply the cap.

    The cap is the mechanism, not a safety net: without skeleton confirmation the
    confidence cannot reach the notification threshold, so skeleton confirmation is a
    *hard requirement* for waking anyone up. The config validator refuses to start if
    somebody retunes the two numbers so that stops being true.
    """
    blended = (
        candidate_probability * config.candidate_weight
        + validation_score * config.validation_weight
    )
    adjusted = _apply_skeleton_agreement(blended, skeleton, config)

    confirmed = _skeleton_confirms(skeleton, config)
    capped = not confirmed and adjusted > config.cap_without_skeleton
    final = config.cap_without_skeleton if capped else adjusted

    verdict = Verdict(
        confidence=max(0.0, min(0.999, final)),
        candidate_probability=candidate_probability,
        validation_score=validation_score,
        skeleton_confirmed=confirmed,
        capped=capped,
        reasons=skeleton.reasons,
    )
    # Built from the finished verdict rather than alongside it, so the reason is decided
    # by the same `should_notify` the notification is, against the same confidence.
    return replace(verdict, telegram_skip_reason=_telegram_skip_reason(verdict, skeleton, config))


def _telegram_skip_reason(
    verdict: Verdict, skeleton: SkeletonResult, config: ConfidenceConfig
) -> TelegramSkipReason | None:
    """Why this judgement will not wake anybody, or None because it will (client §7).

    The order is `_skeleton_confirms` read backwards, and it has to be: those are the
    conditions in the order they are actually tested, so each branch here names the one
    that genuinely stopped the alert. Naming a later one would be true and useless — "the
    confidence was low" is not an answer when the pose model was switched off.

    Note what is NOT a branch. There is no `SEVERITY_BELOW_ALERT`: `should_notify` compares
    against `notify_threshold` and nothing else, and `alert_threshold` only chooses a word
    for the summary. A second reason keyed off a second number is how the legacy ended up
    with 0.85 in the worker, 0.85 in the service and 0.90 in the config comments.
    """
    if verdict.should_notify(config):
        return None
    if skeleton.skipped:
        # It could not look. Distinct from looking and disagreeing, because this one is
        # usually a fault we can fix — the model is off, or `min_frames` is too high.
        return TelegramSkipReason.SKELETON_NOT_RUN
    if not skeleton.is_aggressive(config.min_skeleton_score_for_confirmation):
        return TelegramSkipReason.NO_SKELETON_CONFIRMATION
    if skeleton.only_weak_evidence:
        return TelegramSkipReason.WEAK_EVIDENCE_ONLY
    # The skeleton confirmed, so the cap is not what held this back: the blend simply did
    # not reach the bar.
    return TelegramSkipReason.LOW_CONFIDENCE


def _apply_skeleton_agreement(
    confidence: float, skeleton: SkeletonResult, config: ConfidenceConfig
) -> float:
    """The skeleton agreeing is a small boost; the skeleton looking and disagreeing is a
    penalty. The skeleton being unable to look is neither.

    "Agreeing" is measured against the SAME bar as confirmation
    (`min_skeleton_score_for_confirmation`), so tuning the knob can never make one event
    both confirm and be penalised at once -- which is what a second, divergent threshold
    here would produce.
    """
    if skeleton.skipped:
        return confidence
    if skeleton.is_aggressive(config.min_skeleton_score_for_confirmation):
        return min(0.999, confidence + skeleton.score * 0.05)
    return max(0.0, confidence * 0.93)


def _skeleton_confirms(skeleton: SkeletonResult, config: ConfidenceConfig) -> bool:
    """Confirmation needs the model to have looked, to have scored at or above the
    aggression bar, **and for the evidence not to be entirely weak.**

    The bar is ONE config field. It used to be two conditions -- `skeleton.aggressive`
    (against a hardcoded 0.45) AND `score >= min_skeleton_score_for_confirmation` -- and the
    max of the two meant the config field did nothing for any value at or below 0.45. Now
    the field IS the bar, across its whole range.

    The weak-evidence clause is the hierarchy doing its job rather than merely describing
    itself. Without it, `rapid_hand_motion` + `close_upper_body_contact` sums to exactly
    0.45, clears the bar, and confirms — so two children gesturing at each other in a
    corridor would be enough to wake a teacher at two in the morning. Weak evidence is never
    sufficient alone; that is what "weak" means.
    """
    return (
        not skeleton.skipped
        and skeleton.is_aggressive(config.min_skeleton_score_for_confirmation)
        and not skeleton.only_weak_evidence
    )

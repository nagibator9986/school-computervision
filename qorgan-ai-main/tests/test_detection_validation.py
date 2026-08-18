"""The validation tier: the victim-evidence hierarchy and the confidence cap."""

from __future__ import annotations

import pytest

from qorgan.config.bullying import Confidence
from qorgan.detection.validation import (
    REASON_EVIDENCE,
    Evidence,
    SkeletonResult,
    judge,
    score_skeleton,
)

CONFIG = Confidence()
# The one skeleton-aggression bar, read from config -- there is no hardcoded constant to
# import any more. Tests measure against the knob, exactly as the code now does.
BAR = CONFIG.min_skeleton_score_for_confirmation


def _skeleton(reasons: list[str], skipped: bool = False) -> SkeletonResult:
    return SkeletonResult(
        reasons=tuple(reasons),
        score=score_skeleton(reasons),
        skipped=skipped,
        skip_reason="" if not skipped else "insufficient_frames",
    )


# -- the hierarchy ---------------------------------------------------------


def test_a_fall_is_clean_evidence() -> None:
    """Nothing in ordinary school life puts a child on the floor."""
    assert REASON_EVIDENCE["body_fall_or_low_posture"] is Evidence.CLEAN
    assert _skeleton(["body_fall_or_low_posture"]).has_clean_evidence


def test_sudden_displacement_is_motion_only_not_clean() -> None:
    """A bounding box also jumps when the detector re-anchors, or when a child simply
    walks towards the camera. Suggestive; never sufficient."""
    assert REASON_EVIDENCE["sudden_body_displacement"] is Evidence.MOTION_ONLY
    assert not _skeleton(["sudden_body_displacement"]).has_clean_evidence


@pytest.mark.parametrize(
    "reason", ["rapid_hand_motion", "close_upper_body_contact", "kick_like_leg_motion"]
)
def test_hands_contact_and_kicks_are_weak_evidence(reason: str) -> None:
    """Children gesture, stand close, and swing their legs constantly."""
    assert REASON_EVIDENCE[reason] is Evidence.WEAK


def test_a_playground_produces_only_weak_evidence() -> None:
    playground = _skeleton(["rapid_hand_motion", "close_upper_body_contact"])
    assert playground.only_weak_evidence
    assert not playground.has_clean_evidence


def test_a_fall_is_not_only_weak_evidence() -> None:
    assert not _skeleton(["rapid_hand_motion", "body_fall_or_low_posture"]).only_weak_evidence


# -- scoring ---------------------------------------------------------------


def test_a_kick_alone_scores_nothing() -> None:
    """A child kicking a ball, or just walking with a long stride, produces this
    constantly. It only counts when the two are demonstrably engaged."""
    assert score_skeleton(["kick_like_leg_motion"]) == 0.0


def test_a_kick_counts_once_the_two_are_actually_engaged() -> None:
    corroborated = score_skeleton(["kick_like_leg_motion", "close_upper_body_contact"])
    assert corroborated > score_skeleton(["close_upper_body_contact"])


def test_the_skeleton_score_is_capped_at_one() -> None:
    everything = score_skeleton(list(REASON_EVIDENCE))
    assert everything == 1.0


def test_an_empty_skeleton_scores_nothing_and_is_not_aggressive() -> None:
    assert score_skeleton([]) == 0.0
    assert not _skeleton([]).is_aggressive(BAR)


def test_a_fall_plus_contact_is_aggressive() -> None:
    result = _skeleton(
        ["body_fall_or_low_posture", "close_upper_body_contact", "rapid_hand_motion"]
    )
    assert result.score >= BAR
    assert result.is_aggressive(BAR)


# -- the confidence cap: the mechanism that makes skeleton confirmation mandatory


def test_an_unconfirmed_event_cannot_reach_the_notification_threshold() -> None:
    """THE test. The heuristics can be as excited as they like; without the skeleton
    agreeing, nobody gets woken up."""
    verdict = judge(
        candidate_probability=0.99,
        validation_score=0.99,
        skeleton=SkeletonResult(skipped=True),
        config=CONFIG,
    )

    assert verdict.capped
    assert verdict.confidence == CONFIG.cap_without_skeleton
    assert not verdict.should_notify(CONFIG), "an unconfirmed event raised a Telegram alert"


def test_a_skeleton_confirmed_event_can_notify() -> None:
    strong = _skeleton(
        ["body_fall_or_low_posture", "close_upper_body_contact", "rapid_hand_motion"]
    )
    verdict = judge(0.99, 0.9, strong, CONFIG)

    assert verdict.skeleton_confirmed
    assert not verdict.capped
    assert verdict.should_notify(CONFIG)


def test_a_skeleton_that_looked_and_disagreed_is_a_penalty() -> None:
    """Inputs below the cap, so the penalty is actually observable: above the cap both
    cases clamp to 0.72 and the difference is invisible (which is itself correct --
    the cap dominates)."""
    looked_and_saw_nothing = _skeleton([])
    disagreed = judge(0.5, 0.5, looked_and_saw_nothing, CONFIG)
    could_not_look = judge(0.5, 0.5, SkeletonResult(skipped=True), CONFIG)

    assert not disagreed.capped and not could_not_look.capped
    assert disagreed.confidence < could_not_look.confidence


def test_the_cap_sits_strictly_below_the_notify_threshold() -> None:
    """The invariant the whole mechanism rests on. The config validator refuses to start
    if anyone retunes these two so that this stops being true."""
    assert CONFIG.cap_without_skeleton < CONFIG.notify_threshold


def test_the_blend_is_seventy_thirty() -> None:
    verdict = judge(1.0, 0.0, SkeletonResult(skipped=True), CONFIG)
    assert verdict.confidence == pytest.approx(0.7, abs=0.01)


def test_confidence_never_escapes_zero_to_one() -> None:
    for probability in (0.0, 0.5, 1.0):
        for validation in (0.0, 0.5, 1.0):
            verdict = judge(
                probability, validation, _skeleton(["body_fall_or_low_posture"]), CONFIG
            )
            assert 0.0 <= verdict.confidence <= 0.999


def test_a_weak_only_skeleton_does_not_confirm() -> None:
    """Two children gesturing at each other in a corridor must not wake a teacher at 2am.

    This is the hierarchy actually biting. `rapid_hand_motion` + `close_upper_body_contact`
    sums to exactly 0.45 — it clears the aggression bar and the confirmation score bar.
    It confirms anyway *only* if you forget that weak evidence is never sufficient alone,
    which is exactly the mistake this assertion exists to catch.
    """
    weak = _skeleton(["rapid_hand_motion", "close_upper_body_contact"])
    assert weak.is_aggressive(BAR), "the setup is wrong: this should clear the score bar"
    assert weak.only_weak_evidence

    verdict = judge(0.99, 0.99, weak, CONFIG)

    assert not verdict.skeleton_confirmed, "weak evidence alone confirmed an assault"
    assert verdict.capped
    assert not verdict.should_notify(CONFIG)


def test_one_fall_among_weak_evidence_does_confirm() -> None:
    """The complement: weak evidence is not poison, it is merely insufficient. Add a
    single piece of clean evidence -- somebody went down -- and the event stands."""
    real = _skeleton(["rapid_hand_motion", "close_upper_body_contact", "body_fall_or_low_posture"])

    verdict = judge(0.99, 0.9, real, CONFIG)

    assert verdict.skeleton_confirmed
    assert verdict.should_notify(CONFIG)


# -- the confirmation bar is ONE honest knob, not a knob a constant overrides --


def test_the_one_knob_controls_the_confirmation_bar_across_its_whole_range() -> None:
    """`min_skeleton_score_for_confirmation` was inert for every value at or below 0.45.

    `_skeleton_confirms` demanded BOTH `skeleton.aggressive` (score >= a hardcoded
    AGGRESSIVE_SCORE of 0.45) AND `score >= min_skeleton_score_for_confirmation`, so the
    effective bar was max(0.45, config) -- and the config lost that max across its whole
    lower half, which is most of its plausible range. A school tuning skeleton sensitivity
    downward would have seen no effect at all. The one knob must now BE the bar: lower it
    and a mid-strength skeleton confirms; raise it and the same skeleton does not.
    """
    # A fall plus a displacement: score 0.35 -- squarely inside the formerly-inert band --
    # and NOT only-weak evidence, so the score bar is the only thing left to decide it.
    skeleton = _skeleton(["body_fall_or_low_posture", "sudden_body_displacement"])
    assert skeleton.score == pytest.approx(0.35)
    assert not skeleton.only_weak_evidence

    lenient = Confidence(min_skeleton_score_for_confirmation=0.30)
    strict = Confidence(min_skeleton_score_for_confirmation=0.45)

    assert judge(0.9, 0.9, skeleton, lenient).skeleton_confirmed, (
        "lowering the confirmation knob into its formerly-inert range still did not confirm "
        "a 0.35 skeleton -- a hardcoded bar is silently dominating the knob"
    )
    assert not judge(0.9, 0.9, skeleton, strict).skeleton_confirmed

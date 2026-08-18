"""The pure predicates, on synthetic bodies. No GPU, no video, no model.

This is the layer whose output a psychologist eventually reads, so it must be arguable
without hardware. Every fixture below is a body built by hand at a known pose, so a
failure names the geometry rather than the footage.

The cases that matter most are the **negative** ones: the two defects found by running on
real footage were both a predicate returning a confident answer where it should have
returned "cannot tell". Those are pinned here so they cannot come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from classvision import geometry as g
from classvision.states import (
    Baselines,
    PupilState,
    Thresholds,
    classify,
    hand_raised_shoulder,
    read,
)


def body(*, shoulder_px: float = 100.0, head_up: float | None = 0.3,
         left_wrist: tuple[float, float] | None = None,
         left_elbow: tuple[float, float] | None = None,
         centre: tuple[float, float] = (500.0, 500.0),
         ears: tuple[bool, bool] = (True, True)) -> g.Keypoints:
    """One synthetic person. Coordinates are image pixels; y grows downwards."""
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    cx, cy = centre
    xy[g.L_SHOULDER] = (cx - shoulder_px / 2, cy)
    xy[g.R_SHOULDER] = (cx + shoulder_px / 2, cy)
    conf[g.L_SHOULDER] = conf[g.R_SHOULDER] = 0.9

    if head_up is not None:
        xy[g.NOSE] = (cx, cy - head_up * shoulder_px)
        conf[g.NOSE] = 0.8
    for index, present in zip((g.L_EAR, g.R_EAR), ears, strict=True):
        if present and head_up is not None:
            xy[index] = (cx, cy - head_up * shoulder_px)
            conf[index] = 0.7

    if left_wrist is not None:
        xy[g.L_WRIST] = left_wrist
        conf[g.L_WRIST] = 0.9
    if left_elbow is not None:
        xy[g.L_ELBOW] = left_elbow
        conf[g.L_ELBOW] = 0.7
    return g.Keypoints(xy=xy, conf=conf)


# -- scale and datum -------------------------------------------------------------------

def test_shoulder_width_is_the_unit_of_length():
    assert g.shoulder_width(body(shoulder_px=100.0)) == pytest.approx(100.0)


def test_a_body_too_small_to_scale_returns_unknown_not_zero():
    """Below `MIN_USABLE_SHOULDER_PX` a ratio of that width is inside the model's jitter.
    `None` means unknown; zero would mean measured-and-nothing-happened."""
    assert g.shoulder_width(body(shoulder_px=4.0)) is None


def test_anchor_is_the_shoulder_midpoint():
    assert g.anchor(body(centre=(500.0, 400.0))) == pytest.approx((500.0, 400.0))


# -- the hand-raise defect -------------------------------------------------------------

THRESHOLDS = Thresholds()


def test_a_hand_at_desk_level_is_not_raised():
    person = body(left_wrist=(450.0, 560.0), left_elbow=(440.0, 580.0))
    assert hand_raised_shoulder(person, 100.0, THRESHOLDS) is False


def test_a_hand_clearly_above_the_shoulder_line_with_the_forearm_up_is_raised():
    # wrist 60 px above the shoulder line (0.6 widths > the 0.45 threshold),
    # elbow below the wrist so the forearm points upward.
    person = body(left_wrist=(450.0, 440.0), left_elbow=(450.0, 500.0))
    assert hand_raised_shoulder(person, 100.0, THRESHOLDS) is True


def test_an_arm_extended_sideways_to_pass_something_is_not_a_raised_hand():
    """Measured on real footage: reaching for a bottle and handing a book across a desk
    both clear a height threshold. The forearm must point UP."""
    person = body(left_wrist=(300.0, 440.0), left_elbow=(360.0, 430.0))
    assert hand_raised_shoulder(person, 100.0, THRESHOLDS) is False


def test_no_readable_arm_returns_cannot_tell_rather_than_no():
    """`None` and `False` must not aggregate the same way: a pupil whose arm is hidden
    behind the pupil in front has not been observed to keep their hand down."""
    assert hand_raised_shoulder(body(), 100.0, THRESHOLDS) is None


def test_the_teacher_defect_a_lowered_head_must_not_make_everything_a_raised_hand():
    """THE REGRESSION THIS FILE EXISTS FOR. An earlier version measured the wrist against
    the HEAD, so a person leaning over a laptop -- head down, hands on the keyboard --
    read as raising their hand. Three of eight inspected detections were one adult typing.
    The datum is the shoulder line, which does not move when a head drops."""
    typing = body(head_up=-0.2, left_wrist=(450.0, 495.0), left_elbow=(440.0, 520.0))
    assert hand_raised_shoulder(typing, 100.0, THRESHOLDS) is False


# -- head direction --------------------------------------------------------------------

def test_both_ears_visible_means_facing_the_camera():
    assert g.head_direction(body(ears=(True, True))) is g.HeadDirection.TOWARD_CAMERA


def test_no_face_and_no_ears_but_shoulders_present_means_turned_away():
    assert g.head_direction(body(head_up=None)) is g.HeadDirection.AWAY


def test_head_height_is_unknown_when_no_head_keypoint_survives():
    assert g.head_height(body(head_up=None), 100.0) is None


# -- the state machine -----------------------------------------------------------------

def settled_baselines(seat=(500.0, 500.0), upright=0.3) -> Baselines:
    base = Baselines()
    for _ in range(THRESHOLDS.settle_observations):
        base.observe(read(body(centre=seat, head_up=upright), 0.0, THRESHOLDS), THRESHOLDS)
    assert base.settled
    return base


def test_nothing_is_classified_before_a_baseline_exists():
    """Not SEATED. The opening minutes of a lesson are when pupils are arriving and
    moving, and defaulting them to 'sitting' fills that window with a state nobody
    measured."""
    base = Baselines()
    reading = read(body(), 0.0, THRESHOLDS)
    assert classify(reading, base, THRESHOLDS) is PupilState.UNKNOWN


def test_a_settled_upright_pupil_is_seated():
    base = settled_baselines()
    reading = read(body(centre=(500.0, 500.0)), 10.0, THRESHOLDS)
    assert classify(reading, base, THRESHOLDS) is PupilState.SEATED


def test_moving_two_shoulder_widths_from_the_settled_place_is_away():
    base = settled_baselines()
    reading = read(body(centre=(760.0, 500.0)), 10.0, THRESHOLDS)
    assert classify(reading, base, THRESHOLDS) is PupilState.AWAY_FROM_PLACE


def test_the_shoulder_line_rising_is_standing_and_dropping_is_not():
    base = settled_baselines()
    up = read(body(centre=(500.0, 400.0)), 10.0, THRESHOLDS)
    assert classify(up, base, THRESHOLDS) is PupilState.STOOD_UP


def test_head_far_below_the_pupils_own_upright_posture_is_head_down():
    """Against the pupil's OWN baseline. A raw 'nose below the shoulder line' test fired
    on 94 % of real observations, because looking down at a book is what pupils do."""
    base = settled_baselines(upright=0.4)
    slumped = read(body(centre=(500.0, 500.0), head_up=-0.3), 10.0, THRESHOLDS)
    assert classify(slumped, base, THRESHOLDS) is PupilState.HEAD_DOWN


def test_an_unreadable_observation_is_unknown_not_a_state():
    base = settled_baselines()
    blank = g.Keypoints(xy=np.zeros((17, 2)), conf=np.zeros(17))
    assert classify(read(blank, 10.0, THRESHOLDS), base, THRESHOLDS) is PupilState.UNKNOWN


# -- the settling window, and the defect that reached a psychologist's report ------------
#
# Camera D14, the place at (838, 481). Its first twenty observations carried these shoulder
# widths, in this order — six of them a third to a fifth of the place's real 57 px, a
# badly-resolved detection of somebody arriving:
D14_SEAT5_SETTLING_SCALES = (66, 55, 73, 16, 14, 17, 14, 11, 26, 45,
                             65, 64, 58, 64, 63, 64, 60, 60, 59, 59)


def test_a_mis_scaled_detection_cannot_set_a_pupils_upright_posture():
    """**The third instance of this project's signature defect, pinned.**

    `head_up` is head height DIVIDED BY shoulder width, so a detection with a broken scale
    does not add noise to the baseline — it multiplies it. On D14 the six readings above
    reported `head_up` of 2.41–3.56 where the same child upright reads 0.6, the p75 that
    defines «сидит прямо» came out at 1.635 instead of 0.67, and every subsequent
    observation of that child therefore fell `head_drop` below their own baseline:
    5 343 of 5 549 observations, rendered as «22 эпизода с опущенной головой (суммарно
    44,5 минуты)» and an activity index of 51 against 95–98 for every other place in the
    room. Nothing in the artefact said the baseline had been built on rubbish.

    The body here is upright in every frame — only the reported SCALE varies — so a
    baseline that survives this window is one that measured the child, and a baseline that
    does not is one that measured the detector.

    The six broken readings do not merely get out-voted: they are thrown out, and the place
    then waits for six MORE clean observations before it settles at all. On the real
    footage it settled on observation 26 instead of 20 — three seconds of a child's lesson
    spent as UNKNOWN, against forty-four minutes of «лежит на парте» that never happened.
    """
    base = Baselines()
    upright = 0.60
    # `head_up` is expressed in shoulder widths, so a body whose head sits a FIXED NUMBER
    # OF PIXELS above the shoulder line reads higher and higher as the reported scale
    # shrinks. That is exactly what the pose model did.
    pixels_above = upright * 57.0
    # The measured twenty, then the clean observations that followed them in the recording.
    for scale in (*D14_SEAT5_SETTLING_SCALES, 60, 61, 59, 60, 58, 62):
        base.observe(read(body(shoulder_px=scale, head_up=pixels_above / scale),
                          0.0, THRESHOLDS), THRESHOLDS)

    assert base.settled, "twenty clean observations are available — this must settle"
    assert base.upright_head == pytest.approx(upright, abs=0.05), (
        f"upright_head is {base.upright_head}, so the baseline was built from the broken "
        "detections; every upright observation after this one reads as head-down")

    # The consequence, stated as the thing a reader would have seen.
    sitting_up = read(body(shoulder_px=57.0, head_up=upright), 100.0, THRESHOLDS)
    assert classify(sitting_up, base, THRESHOLDS) is not PupilState.HEAD_DOWN


def test_a_place_the_detector_never_resolves_is_refused_rather_than_settled():
    """A baseline is the most consequential number in the package and nothing downstream
    can see it, so «мы не смогли установить норму этого места» has to be sayable. It is
    said by NOT settling: every state stays UNKNOWN, `uncertainty.seats_never_settled`
    counts the place, and `ledger.settle_refusal` carries the reason in Russian.

    Note what does NOT trigger this: a place seen at two consistent sizes — say half the
    frames at 90 px and half at 20 — settles on whichever group reaches
    `settle_observations` first and discards the other, which is right. Refusal needs a
    place the detector never resolves the same way twice, so the scales below step by a
    factor of two and no band around any of them can hold twenty.
    """
    base = Baselines()
    spread = (30.0, 60.0, 120.0, 240.0, 480.0, 960.0)
    for index in range(THRESHOLDS.settle_observations * THRESHOLDS.settle_window_limit):
        base.observe(read(body(shoulder_px=spread[index % len(spread)]), 0.0, THRESHOLDS),
                     THRESHOLDS)

    assert not base.settled
    assert base.refusal and "согласованной шириной плеч" in base.refusal
    assert classify(read(body(shoulder_px=90.0), 99.0, THRESHOLDS),
                    base, THRESHOLDS) is PupilState.UNKNOWN


def test_a_place_genuinely_detected_small_still_settles():
    """The gate is RELATIVE to the window's own median, not to an absolute pixel size, so
    a child at the back of the room — small in every frame — keeps all their observations.
    An absolute floor would refuse the back row and call it a measurement."""
    base = Baselines()
    for _ in range(THRESHOLDS.settle_observations):
        base.observe(read(body(shoulder_px=18.0, head_up=0.5), 0.0, THRESHOLDS), THRESHOLDS)
    assert base.settled
    assert base.refusal is None
    assert base.upright_head == pytest.approx(0.5, abs=0.02)

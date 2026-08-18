"""The pure body geometry, tested with numbers rather than with a lesson we do not have.

Every threshold in `qorgan.classroom` is an estimate, and these tests do not pretend
otherwise: none of them asserts that 0.35 shoulder widths is the right bar for a raised
hand, because nothing in this repository knows that. What they pin down is the part that
is not a guess -- that the rule is the SAME rule at the front and the back of the room,
that the sign of "above" is the screen's, and that a joint the model could not see is
never counted as evidence either way.
"""

from __future__ import annotations

from qorgan.classroom.posture import (
    MIN_USABLE_SHOULDER_PX,
    anchor,
    hand_raised,
    left_the_place,
    rose_from_seat,
    shoulder_width,
)
from tests.classroom_fakes import UNSEEN, seated, with_hand_up

RATIO = 0.35


def test_shoulder_width_is_the_distance_between_the_shoulders() -> None:
    assert shoulder_width(seated(shoulder_width=40.0)) == 40.0


def test_a_person_with_one_shoulder_has_no_scale() -> None:
    """`None`, not a fallback. A scale guessed from one shoulder would make every ratio
    below it a number about nothing, and the metrics would still all return answers."""
    lopsided = seated()
    lopsided.conf[5] = UNSEEN
    assert shoulder_width(lopsided) is None


def test_a_person_too_small_to_scale_by_has_no_scale() -> None:
    """Below `MIN_USABLE_SHOULDER_PX` every threshold is inside the model's own jitter.

    The honest answer is "unknown", which is what None means here. The temptation is to
    return the tiny width and let the ratios do their work -- and that produces a child at
    the back of the room whose every twitch clears a two-pixel bar.
    """
    assert shoulder_width(seated(shoulder_width=MIN_USABLE_SHOULDER_PX - 1)) is None
    assert shoulder_width(seated(shoulder_width=MIN_USABLE_SHOULDER_PX)) is not None


def test_a_wrist_above_the_shoulder_line_is_a_raised_hand() -> None:
    person = with_hand_up(shoulder_width=40.0, above=30.0)  # 0.75 shoulder widths up
    assert hand_raised(person, scale=40.0, above_ratio=RATIO)


def test_a_wrist_resting_below_the_shoulders_is_not() -> None:
    """y grows DOWNWARDS. A wrist at a bigger y is on the desk, not in the air."""
    resting = seated(shoulder_y=300.0, right_wrist=(220.0, 340.0))
    assert not hand_raised(resting, scale=40.0, above_ratio=RATIO)


def test_a_wrist_just_under_the_bar_is_not_a_raised_hand() -> None:
    """The margin is a ratio of the scale, and it is applied -- not rounded away."""
    just_under = with_hand_up(shoulder_width=40.0, above=RATIO * 40.0 - 1.0)
    just_over = with_hand_up(shoulder_width=40.0, above=RATIO * 40.0 + 1.0)

    assert not hand_raised(just_under, scale=40.0, above_ratio=RATIO)
    assert hand_raised(just_over, scale=40.0, above_ratio=RATIO)


def test_the_same_pose_reads_the_same_at_the_front_and_the_back_of_the_room() -> None:
    """**The test that earns the whole ratio-of-shoulder-width design.**

    A classroom camera looks down rows of desks, so the child in front is two or three
    times the size of the child at the back IN THE SAME FRAME. A pixel threshold would
    fire for one row and never for the other, and the report would look like a fact about
    children when it was a fact about seating.

    Here the identical gesture is built at 3x scale, at a different place in the frame,
    and must give the identical answer.
    """
    near = with_hand_up(centre_x=200.0, shoulder_y=300.0, shoulder_width=60.0, above=30.0)
    far = with_hand_up(centre_x=900.0, shoulder_y=100.0, shoulder_width=20.0, above=10.0)

    assert hand_raised(near, scale=60.0, above_ratio=RATIO)
    assert hand_raised(far, scale=20.0, above_ratio=RATIO)

    # And the near-miss stays a near-miss at both scales, which is the other half: a
    # scale-invariant rule that only ever says True is not scale-invariant, it is broken.
    near_miss_near = with_hand_up(shoulder_width=60.0, above=RATIO * 60.0 - 2.0)
    near_miss_far = with_hand_up(shoulder_width=20.0, above=RATIO * 20.0 - 0.7)
    assert not hand_raised(near_miss_near, scale=60.0, above_ratio=RATIO)
    assert not hand_raised(near_miss_far, scale=20.0, above_ratio=RATIO)


def test_a_wrist_the_model_is_unsure_of_is_not_a_raised_hand() -> None:
    """**The confidence asymmetry, and it is deliberate.**

    Shoulders are read at LOOSE_CONF (0.20), wrists at the stricter KEYPOINT_CONF (0.25).
    A low-confidence wrist is the model guessing where a hidden arm went, and its guess
    for an occluded arm lands high, near the head -- exactly the position being tested
    for. Counting those would manufacture raised hands out of hidden ones, in a report
    about how often children participate.
    """
    unsure = with_hand_up(shoulder_width=40.0, above=30.0)
    unsure.conf[10] = 0.22  # above LOOSE_CONF, below KEYPOINT_CONF

    assert not hand_raised(unsure, scale=40.0, above_ratio=RATIO)


def test_nothing_is_a_raised_hand_without_a_scale() -> None:
    """A zero or negative scale must refuse, not divide the room into raised hands."""
    assert not hand_raised(with_hand_up(above=30.0), scale=0.0, above_ratio=RATIO)


def test_the_anchor_y_is_the_shoulder_line() -> None:
    """One quantity, one function. There is deliberately no separate `shoulder_line_y`
    returning this same number: two names for one value is two places for it to drift."""
    assert anchor(seated(shoulder_y=250.0))[1] == 250.0


def test_rising_is_measured_against_this_track_s_own_baseline() -> None:
    """Not against the frame, and not against other children -- §8's «сравниваем ребёнка
    только с ним самим», which here is also the only thing the geometry supports."""
    # Baseline shoulder line at y=300, scale 40, ratio 0.8 -> must rise 32 px.
    assert rose_from_seat(300.0, 300.0 - 33.0, scale=40.0, rise_ratio=0.8)
    assert not rose_from_seat(300.0, 300.0 - 31.0, scale=40.0, rise_ratio=0.8)


def test_slumping_is_not_reported_as_anything() -> None:
    """A shoulder line that DROPS is not a stand, and it is not «лежит на парте» either.

    §12.4 lists lying on the desk; §8 did not promise it and this package does not compute
    it. A one-directional test is how that stays true rather than becoming a metric nobody
    decided to ship.
    """
    assert not rose_from_seat(300.0, 400.0, scale=40.0, rise_ratio=0.8)


def test_leaving_the_place_counts_movement_in_both_directions() -> None:
    """Sideways into the aisle and forwards towards the board are both «вне места»."""
    seat = (200.0, 300.0)
    sideways = (200.0 + 90.0, 300.0)
    forwards = (200.0, 300.0 - 90.0)
    leaning = (200.0 + 20.0, 300.0)

    assert left_the_place(seat, sideways, scale=40.0, away_ratio=2.0)
    assert left_the_place(seat, forwards, scale=40.0, away_ratio=2.0)
    assert not left_the_place(seat, leaning, scale=40.0, away_ratio=2.0)


def test_the_anchor_is_the_shoulder_midpoint() -> None:
    """Not the box centre: a seated child's box is cropped by the desk in front and the
    child behind, so its centre wanders while the child sits perfectly still."""
    assert anchor(seated(centre_x=200.0, shoulder_y=300.0)) == (200.0, 300.0)


def test_a_person_without_shoulders_has_no_anchor() -> None:
    hidden = seated(shoulder_conf=UNSEEN)
    assert anchor(hidden) is None

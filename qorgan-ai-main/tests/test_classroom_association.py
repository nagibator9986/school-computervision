"""Joining skeletons to track ids, and refusing to when the geometry is ambiguous.

This is the module where being wrong is invisible. A mis-assignment credits one child's
raised hand to the child at the next desk, the counter goes up, and no later stage can
detect it -- the report would be confidently, unfalsifiably wrong. So the tests below are
mostly about what `assign` REFUSES, and about the refusals being counted rather than
silently swallowed.
"""

from __future__ import annotations

from qorgan.classroom.association import assign
from qorgan.detection.geometry import Box
from tests.classroom_fakes import UNSEEN, seated

# One desk's worth of box around a person whose shoulders are at (200, 300).
NEAR_DESK = Box(160.0, 260.0, 240.0, 400.0)


def test_a_skeleton_inside_one_box_belongs_to_that_track() -> None:
    person = seated(centre_x=200.0, shoulder_y=300.0)
    result = assign([person], {7: NEAR_DESK})

    assert set(result.people) == {7}
    assert result.people[7] is person
    assert result.ambiguous == 0
    assert result.unclaimed == 0


def test_two_children_at_their_own_desks_are_matched_separately() -> None:
    """The ordinary case, and the one that must not be lost to over-caution."""
    left = seated(centre_x=200.0, shoulder_y=300.0)
    right = seated(centre_x=500.0, shoulder_y=300.0)
    boxes = {7: NEAR_DESK, 9: Box(460.0, 260.0, 540.0, 400.0)}

    result = assign([left, right], boxes)

    assert result.people == {7: left, 9: right}
    assert result.ambiguous == 0


def test_the_pose_model_reordering_its_output_changes_nothing() -> None:
    """**The defect `detection/skeleton.py` documents, one level up.**

    The pose model reorders its people between frames. An earlier version of the skeleton
    code compared slot 0 with slot 0 and called the result displacement -- it was
    measuring the distance between two different children. Here the equivalent would be
    handing a raised hand to the wrong track, so the matching must be by GEOMETRY and
    order must be irrelevant.
    """
    left = seated(centre_x=200.0, shoulder_y=300.0)
    right = seated(centre_x=500.0, shoulder_y=300.0)
    boxes = {7: NEAR_DESK, 9: Box(460.0, 260.0, 540.0, 400.0)}

    forwards = assign([left, right], boxes)
    backwards = assign([right, left], boxes)

    assert forwards.people == backwards.people


def test_a_skeleton_inside_two_boxes_is_refused_and_counted() -> None:
    """Rows of desks make overlapping boxes ordinary, not exceptional.

    Nearest-box matching is the tempting alternative and it is the dangerous one: at a
    desk the nearest other person is 60 cm away. Refusing produces a hole we can see;
    guessing produces a number nobody can check.
    """
    person = seated(centre_x=200.0, shoulder_y=300.0)
    overlapping = {7: NEAR_DESK, 8: Box(150.0, 250.0, 260.0, 420.0)}

    result = assign([person], overlapping)

    assert result.people == {}
    assert result.ambiguous == 1
    assert result.unclaimed == 0, "an ambiguous skeleton is not an unclaimed one"


def test_one_box_holding_two_skeletons_keeps_neither() -> None:
    """The second skeleton in a person's box is usually the child sitting BEHIND them,
    whose own box was lost to occlusion this frame. Keeping the nearer, or the larger, or
    the first, would be a guess dressed as a rule -- and the guess credits a raised hand
    to the wrong child."""
    front = seated(centre_x=200.0, shoulder_y=380.0)
    behind = seated(centre_x=205.0, shoulder_y=280.0)

    result = assign([front, behind], {7: NEAR_DESK})

    assert result.people == {}
    assert result.ambiguous == 2, "both skeletons are dropped, and both are counted"


def test_a_skeleton_in_no_box_is_unclaimed_not_ambiguous() -> None:
    """Usually the adult at the front of the room, whom the person detector tracks like
    anybody else. Kept apart from `ambiguous` because the two have different causes and
    different cures -- a counter that means both is the defect migration 0005 is about."""
    teacher = seated(centre_x=900.0, shoulder_y=200.0)

    result = assign([teacher], {7: NEAR_DESK})

    assert result.people == {}
    assert result.unclaimed == 1
    assert result.ambiguous == 0


def test_a_skeleton_with_no_shoulders_cannot_be_placed() -> None:
    """No anchor, no assignment. It is unclaimed rather than guessed at from the ankles."""
    faceless = seated(shoulder_conf=UNSEEN)

    result = assign([faceless], {7: NEAR_DESK})

    assert result.people == {}
    assert result.unclaimed == 1


def test_an_empty_slot_in_the_frame_is_skipped_silently() -> None:
    """`Frame` slots may be None -- that person was not detected this frame. It is not a
    skeleton that failed to be placed, so it is counted as nothing at all."""
    person = seated(centre_x=200.0, shoulder_y=300.0)

    result = assign([None, person, None], {7: NEAR_DESK})

    assert result.people == {7: person}
    assert result.unclaimed == 0
    assert result.ambiguous == 0


def test_no_tracks_at_all_leaves_every_skeleton_unclaimed() -> None:
    """The first frames after a stream opens: pose has people, ByteTrack has no ids yet."""
    result = assign([seated(), seated(centre_x=500.0)], {})

    assert result.people == {}
    assert result.unclaimed == 2

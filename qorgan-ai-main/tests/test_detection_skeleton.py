"""Skeleton features, from synthetic keypoints. No model, no GPU.

That the judgement is testable without a graphics card is the point: this is the code
that decides whether a child gets help, and it should not need a GPU to interrogate.
"""

from __future__ import annotations

import numpy as np

from qorgan.detection.skeleton import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Frame,
    Keypoints,
    extract_reasons,
)

KEYPOINTS = 17


def _person(**joints: tuple[float, float]) -> Keypoints:
    """A person with only the named joints visible. Everything else is unseen."""
    xy = np.zeros((KEYPOINTS, 2), dtype=float)
    conf = np.zeros(KEYPOINTS, dtype=float)

    names = {
        "nose": NOSE,
        "l_shoulder": LEFT_SHOULDER,
        "r_shoulder": RIGHT_SHOULDER,
        "l_wrist": LEFT_WRIST,
        "r_wrist": RIGHT_WRIST,
        "l_hip": LEFT_HIP,
        "r_hip": RIGHT_HIP,
        "l_ankle": LEFT_ANKLE,
        "r_ankle": RIGHT_ANKLE,
    }
    for name, point in joints.items():
        index = names[name]
        xy[index] = point
        conf[index] = 0.9
    return Keypoints(xy=xy, conf=conf)


def _standing(x: float = 100.0, nose_y: float = 50.0) -> Keypoints:
    """Upright: hips well below the nose."""
    return _person(
        nose=(x, nose_y),
        l_shoulder=(x - 20, nose_y + 30),
        r_shoulder=(x + 20, nose_y + 30),
        l_wrist=(x - 25, nose_y + 90),
        r_wrist=(x + 25, nose_y + 90),
        l_hip=(x - 15, nose_y + 100),
        r_hip=(x + 15, nose_y + 100),
        l_ankle=(x - 15, nose_y + 190),
        r_ankle=(x + 15, nose_y + 190),
    )


def _on_the_floor(x: float = 100.0) -> Keypoints:
    """Down: the hips have come up level with the nose."""
    return _person(
        nose=(x, 150.0),
        l_hip=(x + 30, 160.0),
        r_hip=(x + 40, 160.0),
        l_shoulder=(x + 5, 152.0),
        r_shoulder=(x + 10, 155.0),
    )


def _frames(*people_per_frame: list[Keypoints | None]) -> list[Frame]:
    return list(people_per_frame)


# -- nothing happening -------------------------------------------------------


def test_a_person_standing_still_produces_no_reasons() -> None:
    still = [[_standing()] for _ in range(6)]
    assert extract_reasons(still) == []


def test_a_single_frame_cannot_be_judged() -> None:
    """One frame has no motion in it, by definition."""
    assert extract_reasons([[_standing()]]) == []


def test_an_empty_sequence_produces_nothing() -> None:
    assert extract_reasons([]) == []


# -- the clean evidence: a fall ----------------------------------------------


def test_someone_on_the_floor_is_a_fall() -> None:
    """The one piece of clean evidence. Nothing in ordinary school life puts a child
    on the floor."""
    falling = _frames([_standing()], [_on_the_floor()], [_on_the_floor()], [_on_the_floor()])
    assert "body_fall_or_low_posture" in extract_reasons(falling)


def test_a_standing_person_is_not_a_fall() -> None:
    upright = [[_standing()] for _ in range(6)]
    assert "body_fall_or_low_posture" not in extract_reasons(upright)


def test_one_frame_on_the_floor_is_not_enough() -> None:
    """A single low frame is usually the pose model having a bad moment."""
    blip = _frames([_standing()], [_on_the_floor()], [_standing()], [_standing()])
    assert "body_fall_or_low_posture" not in extract_reasons(blip)


# -- rapid hand motion -------------------------------------------------------


def test_wrists_flying_about_are_rapid_hand_motion() -> None:
    frames = []
    for index in range(6):
        swing = 40 if index % 2 else -40
        frames.append([_person(l_wrist=(100 + swing, 100), r_wrist=(140 - swing, 100))])

    assert "rapid_hand_motion" in extract_reasons(frames)


def test_still_hands_are_not_rapid_hand_motion() -> None:
    frames = [[_person(l_wrist=(100, 100), r_wrist=(140, 100))] for _ in range(6)]
    assert "rapid_hand_motion" not in extract_reasons(frames)


# -- close upper-body contact ------------------------------------------------


def test_two_people_shoulder_to_shoulder_are_in_contact() -> None:
    close = [
        [_standing(x=100), _standing(x=115)],
        [_standing(x=100), _standing(x=115)],
        [_standing(x=100), _standing(x=115)],
    ]
    assert "close_upper_body_contact" in extract_reasons(close)


def test_two_people_across_the_room_are_not_in_contact() -> None:
    apart = [[_standing(x=50), _standing(x=280)] for _ in range(4)]
    assert "close_upper_body_contact" not in extract_reasons(apart)


def test_three_children_standing_together_in_one_frame_is_not_contact() -> None:
    """The legacy bug: it counted every near PAIR, so three children together scored
    three hits in a single frame and confirmed instantly. It must count FRAMES."""
    huddle = [
        [_standing(x=100), _standing(x=115), _standing(x=130)],  # 3 close pairs, 1 frame
        [_standing(x=50), _standing(x=300), _standing(x=500)],  # everyone apart
    ]
    assert "close_upper_body_contact" not in extract_reasons(huddle)


# -- kicks -------------------------------------------------------------------


def test_snapping_ankles_are_kick_like() -> None:
    frames = []
    for index in range(6):
        swing = 35 if index % 2 else -35
        frames.append([_person(l_ankle=(100 + swing, 200), r_ankle=(120, 200))])

    assert "kick_like_leg_motion" in extract_reasons(frames)


def test_a_person_standing_still_is_not_kicking() -> None:
    frames = [[_person(l_ankle=(100, 200), r_ankle=(120, 200))] for _ in range(6)]
    assert "kick_like_leg_motion" not in extract_reasons(frames)


# -- sudden displacement, and the same-person bug ---------------------------


def test_a_person_knocked_sideways_is_a_sudden_displacement() -> None:
    shoved = [
        [_person(l_hip=(100, 150), r_hip=(120, 150))],
        [_person(l_hip=(160, 150), r_hip=(180, 150))],  # +60 px in one frame
    ]
    assert "sudden_body_displacement" in extract_reasons(shoved)


def test_reordering_the_pose_model_output_does_not_invent_a_shove() -> None:
    """THE bug, and it is subtler than it looks.

    The pose model does not guarantee a stable order for the people it finds. Comparing
    person slot 0 in frame N against slot 0 in frame N+1 therefore measures the distance
    between two DIFFERENT children whenever the model happens to swap them -- so two
    children standing perfectly still at opposite ends of the crop produce a violent
    'sudden displacement' out of nothing at all.

    The legacy patched this by insisting on the same slot index, which is the same bug
    wearing a hat: the slot index IS the unreliable thing. People are matched by
    proximity instead, so the swap is harmless.
    """
    a = _person(l_hip=(50, 150), r_hip=(70, 150))
    b = _person(l_hip=(400, 150), r_hip=(420, 150))

    # Both stock still. The model reports them in a different order each frame.
    swapping = _frames([a, b], [b, a], [a, b], [b, a])

    assert "sudden_body_displacement" not in extract_reasons(swapping), (
        "a reordering of two motionless children was read as somebody being shoved"
    )


def test_a_genuine_shove_still_registers_despite_the_matching() -> None:
    """The complement: proximity matching must not make the detector blind to a real
    displacement. The person nearest to where you were is still you, even after a shove."""
    victim_before = _person(l_hip=(100, 150), r_hip=(120, 150))
    victim_after = _person(l_hip=(160, 150), r_hip=(180, 150))  # +60 px
    bystander = _person(l_hip=(500, 150), r_hip=(520, 150))

    shoved = _frames([victim_before, bystander], [bystander, victim_after])  # reordered too

    assert "sudden_body_displacement" in extract_reasons(shoved)


def test_a_person_who_vanishes_is_not_compared_to_whoever_took_their_place() -> None:
    """A gap is a gap, not a licence to compare two different children."""
    a = _person(l_hip=(50, 150), r_hip=(70, 150))

    frames = _frames([a], [], [a])

    assert "sudden_body_displacement" not in extract_reasons(frames)


# -- the whole picture -------------------------------------------------------


def test_a_real_assault_produces_clean_evidence_and_more() -> None:
    """Somebody is knocked to the floor while the other flails at them."""
    attacker_standing = _standing(x=150)
    frames = [
        [_standing(x=100), attacker_standing],
        [
            _on_the_floor(x=105),
            _person(
                l_wrist=(160, 60), r_wrist=(190, 140), l_shoulder=(140, 80), r_shoulder=(175, 80)
            ),
        ],
        [
            _on_the_floor(x=110),
            _person(
                l_wrist=(200, 140), r_wrist=(150, 60), l_shoulder=(140, 80), r_shoulder=(175, 80)
            ),
        ],
        [
            _on_the_floor(x=112),
            _person(
                l_wrist=(155, 65), r_wrist=(195, 145), l_shoulder=(140, 80), r_shoulder=(175, 80)
            ),
        ],
    ]

    reasons = extract_reasons(frames)

    assert "body_fall_or_low_posture" in reasons, "the fall was missed"
    assert "rapid_hand_motion" in reasons

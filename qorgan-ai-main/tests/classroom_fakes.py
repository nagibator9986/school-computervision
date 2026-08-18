"""Building COCO-17 keypoint arrays by hand, so the classroom tests need no video.

The whole point of `qorgan.classroom.posture` being pure is that a child at a desk can be
described by seventeen pairs of numbers. This is where those numbers get written, once,
so that six test modules do not each grow their own slightly different idea of what a
seated child looks like.
"""

from __future__ import annotations

import numpy as np

from qorgan.detection.skeleton import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoints,
)

KEYPOINT_COUNT = 17

# Comfortably above both KEYPOINT_CONF (0.25) and LOOSE_CONF (0.20).
SEEN = 0.9
# Below both: the model did not find this joint.
UNSEEN = 0.05


def seated(
    *,
    centre_x: float = 200.0,
    shoulder_y: float = 300.0,
    shoulder_width: float = 40.0,
    left_wrist: tuple[float, float] | None = None,
    right_wrist: tuple[float, float] | None = None,
    wrist_conf: float = SEEN,
    shoulder_conf: float = SEEN,
) -> Keypoints:
    """One person, described by where their shoulders and wrists are.

    Everything the classroom metrics read is here; the other thirteen keypoints are
    present (the array must be 17 long) and marked unseen, which is also the honest
    picture of a child behind a desk -- hips, knees and ankles are behind furniture.
    """
    xy = np.zeros((KEYPOINT_COUNT, 2), dtype=float)
    conf = np.full(KEYPOINT_COUNT, UNSEEN, dtype=float)

    half = shoulder_width / 2.0
    xy[LEFT_SHOULDER] = (centre_x - half, shoulder_y)
    xy[RIGHT_SHOULDER] = (centre_x + half, shoulder_y)
    conf[LEFT_SHOULDER] = conf[RIGHT_SHOULDER] = shoulder_conf

    if left_wrist is not None:
        xy[LEFT_WRIST] = left_wrist
        conf[LEFT_WRIST] = wrist_conf
    if right_wrist is not None:
        xy[RIGHT_WRIST] = right_wrist
        conf[RIGHT_WRIST] = wrist_conf

    return Keypoints(xy=xy, conf=conf)


def with_hand_up(
    *,
    centre_x: float = 200.0,
    shoulder_y: float = 300.0,
    shoulder_width: float = 40.0,
    above: float = 30.0,
) -> Keypoints:
    """The same person with one wrist `above` pixels higher than the shoulder line.

    Screen coordinates: y grows downwards, so "higher" is a SMALLER y. Getting that sign
    wrong is the easiest mistake in this whole package, which is why it is written down
    in one place and every test raises a hand through this function.
    """
    return seated(
        centre_x=centre_x,
        shoulder_y=shoulder_y,
        shoulder_width=shoulder_width,
        right_wrist=(centre_x + shoulder_width / 2.0, shoulder_y - above),
    )

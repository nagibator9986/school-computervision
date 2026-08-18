"""The five skeleton features, as pure functions over COCO-17 keypoints.

No model, no GPU, no frames — a sequence of keypoint arrays in, a list of reason strings
out. That split is deliberate: the *judgement* is the hard part and the part that decides
whether a child gets help, so it must be testable without a graphics card. Running
YOLOv8n-pose to produce the keypoints lives in `qorgan.models.pose`.

Two bugs the legacy fixed the hard way and documented, both preserved here:

  * **Track the same person across frames.** An earlier version compared person slot 0 in
    frame N against person slot 0 in frame N+1 — but the pose model reorders its outputs,
    so it was measuring the distance between two *different children* and calling it
    displacement. Two children standing perfectly still at opposite ends of the crop
    produced a violent "shove" every time the model happened to swap their order.

    The legacy patched this by insisting on the same slot index, which is the same bug
    wearing a hat: the slot index *is* the thing that is unreliable. Here people are
    matched between frames by **proximity** — the person nearest to where you were is
    you — so a reordering is harmless and a genuine shove still registers.

  * **Count distinct frames, not pairs.** `close_upper_body_contact` originally counted
    every person-pair that was near, so three children standing together scored three
    hits in a single frame and "confirmed" instantly. It now counts frames.

All thresholds assume the ~320 px-wide person crop the validator is fed at ~5 fps. They
are pixel distances in that crop, not in the source frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# COCO-17 keypoint indices.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# A keypoint below this confidence is not a keypoint.
KEYPOINT_CONF = 0.25
LOOSE_CONF = 0.20

# Per-feature thresholds, in pixels of the crop.
FAST_HAND_PX = 18.0
FAST_HAND_MEAN_PX = 13.0
FAST_HAND_MIN_FRAMES = 3

CLOSE_SHOULDER_PX = 42.0
CLOSE_CONTACT_MIN_FRAMES = 2

KICK_ANKLE_PX = 22.0
KICK_MIN_FRAMES = 2

DISPLACEMENT_PX = 28.0

# Below the hips being this high relative to the nose, the person is on the floor.
LOW_POSTURE_RATIO = 0.30
LOW_POSTURE_MIN_FRAMES = 2

MIN_FRAMES_TO_JUDGE = 2
MIN_PEOPLE_FOR_CONTACT = 2


@dataclass(frozen=True, slots=True)
class Keypoints:
    """One person in one frame. `xy` is (17, 2); `conf` is (17,)."""

    xy: np.ndarray
    conf: np.ndarray

    def has(self, *indices: int, threshold: float = KEYPOINT_CONF) -> bool:
        return all(self.conf[i] >= threshold for i in indices)

    def point(self, index: int) -> tuple[float, float]:
        return float(self.xy[index][0]), float(self.xy[index][1])


# One frame's worth of people. A slot may be None: that person was not detected in this
# frame, and comparing across a gap would compare two different children.
Frame = list[Keypoints | None]


def extract_reasons(frames: list[Frame]) -> list[str]:
    """Everything the skeleton saw. The caller weighs them (see detection/validation.py)."""
    if len(frames) < MIN_FRAMES_TO_JUDGE:
        return []

    reasons: list[str] = []
    if _rapid_hand_motion(frames):
        reasons.append("rapid_hand_motion")
    if _body_fall_or_low_posture(frames):
        reasons.append("body_fall_or_low_posture")
    if _close_upper_body_contact(frames):
        reasons.append("close_upper_body_contact")
    if _kick_like_leg_motion(frames):
        reasons.append("kick_like_leg_motion")
    if _sudden_body_displacement(frames):
        reasons.append("sudden_body_displacement")
    return reasons


# -- the five features -------------------------------------------------------


def _rapid_hand_motion(frames: list[Frame]) -> bool:
    """Wrists moving fast. Weak evidence: children gesture constantly."""
    deltas = _joint_deltas(frames, (LEFT_WRIST, RIGHT_WRIST), KEYPOINT_CONF)
    if not deltas:
        return False

    fast = sum(1 for d in deltas if d > FAST_HAND_PX)
    return fast >= FAST_HAND_MIN_FRAMES or float(np.mean(deltas)) > FAST_HAND_MEAN_PX


def _body_fall_or_low_posture(frames: list[Frame]) -> bool:
    """The hips have come up towards the nose: the person is down.

    The ONE piece of clean evidence in the whole system. Nothing in ordinary school life
    puts a child on the floor.
    """
    low = 0
    for frame in frames:
        for person in frame:
            if person is None or not person.has(NOSE):
                continue
            if not person.has(LEFT_HIP, RIGHT_HIP, threshold=LOOSE_CONF):
                continue

            _, nose_y = person.point(NOSE)
            hip_y = (person.point(LEFT_HIP)[1] + person.point(RIGHT_HIP)[1]) / 2

            # Upright: hips sit well below the nose. On the floor: they converge.
            if nose_y > 0 and (hip_y - nose_y) < nose_y * LOW_POSTURE_RATIO:
                low += 1
                break  # one fall per frame, not one per person

    return low >= LOW_POSTURE_MIN_FRAMES


def _close_upper_body_contact(frames: list[Frame]) -> bool:
    """Two people's shoulders nearly touching, in at least two distinct FRAMES.

    Counting frames rather than person-pairs is the fix: three children standing together
    produce three near pairs in a single frame, which used to 'confirm' instantly.
    """
    close_frames = 0
    for frame in frames:
        people = [p for p in frame if p is not None]
        if len(people) < MIN_PEOPLE_FOR_CONTACT:
            continue
        if _any_shoulders_close(people):
            close_frames += 1

    return close_frames >= CLOSE_CONTACT_MIN_FRAMES


def _any_shoulders_close(people: list[Keypoints]) -> bool:
    for index, first in enumerate(people):
        for second in people[index + 1 :]:
            for joint in (LEFT_SHOULDER, RIGHT_SHOULDER):
                if not (
                    first.has(joint, threshold=LOOSE_CONF)
                    and second.has(joint, threshold=LOOSE_CONF)
                ):
                    continue
                if _distance(first.point(joint), second.point(joint)) < CLOSE_SHOULDER_PX:
                    return True
    return False


def _kick_like_leg_motion(frames: list[Frame]) -> bool:
    """Ankles snapping. Weak on its own -- walking with a long stride looks identical,
    which is why `score_skeleton` refuses to count it without corroboration."""
    deltas = _joint_deltas(frames, (LEFT_ANKLE, RIGHT_ANKLE), LOOSE_CONF)
    return sum(1 for d in deltas if d > KICK_ANKLE_PX) >= KICK_MIN_FRAMES


def _sudden_body_displacement(frames: list[Frame]) -> bool:
    """A person's hip centre jumping. Motion-only evidence: a box also jumps when the
    detector re-anchors, or simply because the child is walking towards the camera."""
    for before, after in _tracked_pairs(frames):
        if not (
            before.has(LEFT_HIP, RIGHT_HIP, threshold=LOOSE_CONF)
            and after.has(LEFT_HIP, RIGHT_HIP, threshold=LOOSE_CONF)
        ):
            continue
        if _distance(_hip_center(before), _hip_center(after)) > DISPLACEMENT_PX:
            return True
    return False


# -- matching people between frames ------------------------------------------
#
# The person nearest to where you were is you. This is the whole trick, and it is what
# makes every temporal feature above safe against the pose model reordering its output.


def _tracked_pairs(frames: list[Frame]) -> list[tuple[Keypoints, Keypoints]]:
    """Every (person in frame N, the same person in frame N+1) pairing we are sure of."""
    pairs: list[tuple[Keypoints, Keypoints]] = []
    for previous, current in pairwise(frames):
        pairs.extend(_match(previous, current))
    return pairs


def _match(previous: Frame, current: Frame) -> list[tuple[Keypoints, Keypoints]]:
    """Greedy nearest-centroid matching. Unmatched people are simply dropped: a gap is a
    gap, not a licence to compare two different children."""
    before = [p for p in previous if p is not None]
    after = [p for p in current if p is not None]
    if not before or not after:
        return []

    taken: set[int] = set()
    matched: list[tuple[Keypoints, Keypoints]] = []

    for person in before:
        best_index, best_distance = None, float("inf")
        for index, candidate in enumerate(after):
            if index in taken:
                continue
            gap = _distance(_centroid(person), _centroid(candidate))
            if gap < best_distance:
                best_index, best_distance = index, gap

        if best_index is not None:
            taken.add(best_index)
            matched.append((person, after[best_index]))

    return matched


def _centroid(person: Keypoints) -> tuple[float, float]:
    """The mean of whatever the model is confident about. Used only for matching."""
    visible = person.conf >= LOOSE_CONF
    if not visible.any():
        return (0.0, 0.0)
    point = person.xy[visible].mean(axis=0)
    return float(point[0]), float(point[1])


# -- helpers -----------------------------------------------------------------


def _joint_deltas(frames: list[Frame], joints: tuple[int, ...], threshold: float) -> list[float]:
    """How far each joint moved between consecutive frames, for the same PERSON."""
    deltas: list[float] = []
    for before, after in _tracked_pairs(frames):
        for joint in joints:
            if before.has(joint, threshold=threshold) and after.has(joint, threshold=threshold):
                deltas.append(_distance(before.point(joint), after.point(joint)))
    return deltas


def _hip_center(person: Keypoints) -> tuple[float, float]:
    left, right = person.point(LEFT_HIP), person.point(RIGHT_HIP)
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

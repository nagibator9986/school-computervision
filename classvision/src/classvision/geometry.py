"""What one body is doing, as pure functions over COCO-17 keypoints.

Numbers in, numbers out. No model, no config, no database, no frames — so every
judgement that ends up in front of a psychologist can be tested without a graphics card,
and so the thresholds can be argued about in one place instead of five.

**Everything is measured in units of the person's own shoulder width.** A classroom
camera looks down rows of desks: on this footage the shoulder width across one frame runs
from 19 px to 232 px, a factor of twelve. A threshold in pixels would fire on the front
row and never on the back row, and the resulting report would look like a fact about
children when it was a fact about seating. The measured p25 is 70 px against a usable
floor of ~8 px (`MIN_USABLE_SHOULDER_PX`), so on this camera the normalisation is
comfortable rather than marginal — see `MEASUREMENTS.md` §3.

**The honest limit of a single frame, which shapes the whole module.** Two of the states
the client asked for cannot be decided from one frame by anybody, model or human:

  * *Standing.* A pupil standing at the back can occupy the same image height, at the
    same image y, as one seated at the front. So there is no `is_standing()` here.
    `rose_from_seat()` instead asks whether this person has risen relative to where
    **this same track** was sitting — the baseline is supplied by the caller, because
    holding it is state and this module has none.
  * *Lying on the desk.* Pupils look down at their books constantly: a raw "nose has
    dropped below the shoulder line" test fired on **94 %** of sampled frames on this
    footage (`MEASUREMENTS.md` §5). So `head_lowered()` reports a continuous *depth*
    against the track's own upright baseline, and the decision to call it lying is a
    duration, taken upstream in `states.py`. A predicate that fires on 94 % of frames is
    not a predicate, it is a constant.

**Head direction is reported coarsely, on purpose.** At a median face height of 64 px the
eyes are ~10 px apart, so a yaw in degrees derived from eye geometry would be quoted to a
precision the pixels do not contain. What the keypoint *confidences* carry at this scale
is which side of the head the camera can see, and that is enough for the question actually
asked — is this pupil facing the board, or turned away from it — because on this camera
the board is behind the lens. `HeadDirection` therefore has five members and no degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# COCO-17 keypoint indices.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

# Two confidence bars, and the asymmetry between them is deliberate.
#
# STRICT is required of any keypoint whose POSITION decides something — above all a
# wrist. A low-confidence wrist is the model guessing where an occluded arm went, and its
# guess lands near the head, which is exactly the position a hand-raise test looks for.
# Counting those manufactures raised hands out of hidden ones.
#
# LOOSE is enough for a keypoint used only as a REFERENCE (the shoulders, which set the
# scale and the datum). Demanding STRICT there would discard whole tracks — the measured
# median mean-keypoint confidence on this footage is 0.7 — and discarding a track is not
# a neutral act: it turns "we could not see" into "nothing happened".
STRICT_CONF = 0.50
LOOSE_CONF = 0.30

# Below this many pixels between the shoulders, no ratio of that width means anything:
# at 6 px a threshold of 0.35 shoulder widths is 2 px, inside the pose model's own
# jitter. Chosen, not measured — it is roughly where the keypoints stop being separable.
# Everything about such an observation is reported as unknown rather than as zero.
MIN_USABLE_SHOULDER_PX = 8.0

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Keypoints:
    """One person in one frame: 17 (x, y) and 17 confidences, straight from the model."""

    xy: np.ndarray    # (17, 2)
    conf: np.ndarray  # (17,)

    def has(self, *indices: int, threshold: float = STRICT_CONF) -> bool:
        return all(float(self.conf[i]) >= threshold for i in indices)

    def point(self, index: int) -> Point:
        return float(self.xy[index][0]), float(self.xy[index][1])

    def any_of(self, *indices: int, threshold: float = STRICT_CONF) -> bool:
        return any(float(self.conf[i]) >= threshold for i in indices)


class HeadDirection(StrEnum):
    """Which way the head is turned, at the resolution the pixels actually support.

    On this camera the board and the teacher are BEHIND the lens, so `TOWARD_CAMERA`
    is "facing the front of the room" and `AWAY` is «смотрит назад». That mapping is a
    property of where this camera is mounted, not a universal one, and the caller states
    it — nothing here assumes a room layout.
    """

    TOWARD_CAMERA = "toward_camera"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    AWAY = "away"
    UNKNOWN = "unknown"


def shoulder_width(person: Keypoints) -> float | None:
    """The distance between the shoulders: this person's unit of length.

    Shoulders rather than the bounding-box diagonal, because a seated pupil's box is
    cropped by the desk in front and by the pupil behind, so its height measures the
    furniture as much as the person. The shoulders stay visible while a child leans,
    writes or turns, and they sit above desk height.

    `None` — meaning *unknown*, never zero — when either shoulder is missing or the pair
    is too close together to scale anything by.
    """
    if not person.has(L_SHOULDER, R_SHOULDER, threshold=LOOSE_CONF):
        return None
    width = math.dist(person.point(L_SHOULDER), person.point(R_SHOULDER))
    return width if width >= MIN_USABLE_SHOULDER_PX else None


def anchor(person: Keypoints) -> Point | None:
    """The shoulder midpoint: this person's position AND their vertical datum.

    Its **y is the shoulder line**, the reference a raised wrist and a dropped head are
    measured against. The nose would be the more intuitive datum and is the wrong one: it
    swings through most of a head's height every time a child looks down at a book, which
    is the commonest thing a child does at a desk.

    Its **x is the person's position**, in place of the box centre, which wanders by a
    large fraction of its width as the desk and neighbours occlude different parts of a
    seated body while the body has not moved.
    """
    if not person.has(L_SHOULDER, R_SHOULDER, threshold=LOOSE_CONF):
        return None
    left, right = person.point(L_SHOULDER), person.point(R_SHOULDER)
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def head_direction(person: Keypoints) -> HeadDirection:
    """Which side of the head the camera can see.

    Read from CONFIDENCES rather than from eye geometry, because at a 64 px face the
    inter-eye distance is ~10 px and any angle computed from it would be quoted to a
    precision the pixels do not hold. What the confidences do carry reliably at that
    scale is presence: an ear that has rotated away from the lens stops being detected.

    The rule, and its one real confusion. Both ears visible means the face is toward the
    camera. One ear plus the nose means a profile, turned toward the side of the ear that
    is still visible. Neither ear and no nose, while the shoulders ARE visible, means the
    back of the head — the pupil has turned away from the front of the room. The
    confusion is that long hair hides an ear as effectively as rotation does, so a
    profile can read as `AWAY`; `states.py` blunts this by requiring a hold, and the
    residue is counted in `unknown_head_observations` rather than argued away.
    """
    left_ear = person.has(L_EAR, threshold=LOOSE_CONF)
    right_ear = person.has(R_EAR, threshold=LOOSE_CONF)
    face = person.any_of(NOSE, L_EYE, R_EYE, threshold=LOOSE_CONF)

    if left_ear and right_ear:
        return HeadDirection.TOWARD_CAMERA
    if face and (left_ear or right_ear):
        # A visible LEFT ear means the head has rotated so that the person's left side
        # faces the lens — from the camera's point of view, a profile to the left.
        return HeadDirection.PROFILE_LEFT if left_ear else HeadDirection.PROFILE_RIGHT
    if face:
        return HeadDirection.TOWARD_CAMERA
    if person.has(L_SHOULDER, R_SHOULDER, threshold=LOOSE_CONF):
        return HeadDirection.AWAY
    return HeadDirection.UNKNOWN


def head_height(person: Keypoints, scale: float) -> float | None:
    """How far the head sits ABOVE the shoulder line, in shoulder widths.

    Positive is upright. This is the continuous quantity behind both "sitting up" and
    "lying on the desk", returned as a number rather than a verdict because the verdict
    needs a baseline and a duration that this module has no business holding.

    `None` when the head cannot be seen at all, which must not be read as zero: a pupil
    whose head is hidden behind the pupil in front is not a pupil lying on a desk.
    """
    shoulders = anchor(person)
    if shoulders is None or scale <= 0:
        return None
    for index in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR):
        if person.has(index, threshold=LOOSE_CONF):
            # Screen coordinates: y grows downwards, so "above" is a smaller y.
            return (shoulders[1] - person.point(index)[1]) / scale
    return None


def hand_raised(person: Keypoints, scale: float, *, above_head_ratio: float,
                require_elbow: bool = True) -> bool | None:
    """Is a hand held clearly up — above this person's own HEAD, not merely their shoulder?

    **Three-valued, and the third value is the point.** `None` means *could not be
    measured*, and it is returned whenever this person's head cannot be located. That is
    not defensive coding, it is a defect fix: an earlier version substituted a guessed
    head position (0.6 shoulder widths up) when no head keypoint survived, and on this
    footage it counted **the teacher** as raising his hand while he typed. He sits with
    his back to the lens, so no head keypoint is confident; he leans over the laptop, so
    his shoulder line drops below his own hands; and the guessed datum then sat low
    enough for a keyboard to clear it. Three of the eight detections inspected by eye
    were that one man typing.

    A fallback that turns "we cannot see the head" into "assume the head is here" cannot
    fail loudly — it produces a plausible number for a real person and nothing about the
    output says it was invented. `None` propagates to `states.py`, which counts it in
    `hand_raise_unmeasurable` and reports it beside the totals.

    **Above the head, and that is a correction to the obvious design.** Measuring against
    the shoulder line is the natural choice and it was tried: on this footage a wrist
    more than 0.35 shoulder widths above the shoulder line was present in **25 % of all
    sampled frames** (`MEASUREMENTS.md` §5). Children sit with a hand propping their chin
    or cheek, and that pose clears a shoulder-line threshold all lesson long. The
    neighbouring codebase ships exactly that rule with `above_shoulder_ratio: 0.35`,
    self-labelled a guess; this is the measurement that says the guess is too permissive.

    So the datum is the top of the head, and `require_elbow` additionally demands the
    elbow be above the shoulder line — an arm genuinely raised to answer is extended
    upward, whereas a hand resting against a temple leaves the elbow down near the desk.
    Both terms belong to the same body, so no camera geometry enters and a pupil at the
    back is judged by the same rule as one at the front.

    **What this still cannot separate**: stretching, adjusting hair, reaching for a shelf.
    `states.py` demands the pose be HELD, which discards the brief ones. Neither is a
    classifier of intent, and this function does not claim to have one — it reports arm
    geometry, which is a physical, observable fact.
    """
    shoulders = anchor(person)
    if shoulders is None or scale <= 0:
        return None

    height = head_height(person, scale)
    if height is None:
        return None   # no head, no datum, no answer -- see the docstring

    datum_y = shoulders[1] - height * scale

    for wrist, elbow in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
        if not person.has(wrist, threshold=STRICT_CONF):
            continue
        if person.point(wrist)[1] >= datum_y - above_head_ratio * scale:
            continue
        if require_elbow:
            if not person.has(elbow, threshold=LOOSE_CONF):
                continue
            if person.point(elbow)[1] >= shoulders[1]:
                continue
        return True
    return False


def rose_from_seat(seated_shoulder_y: float, shoulder_y: float, scale: float,
                   rise_ratio: float) -> bool:
    """Has this person risen from where THIS TRACK was sitting?

    Not "is this person standing" — see the module docstring. The caller supplies the
    baseline it established while the track was still.

    Deliberately one-directional: a shoulder line that DROPS by the same amount is not
    reported here. That is a pupil slumping or lying on the desk, and it is
    `head_lowered`'s question, measured against a different baseline.
    """
    if scale <= 0:
        return False
    return (seated_shoulder_y - shoulder_y) >= rise_ratio * scale


def head_lowered(upright_head_height: float, person: Keypoints, scale: float,
                 drop_ratio: float) -> bool:
    """Has the head fallen well below this track's own upright posture?

    The raw single-frame version of this test — "nose below the shoulder line" — fired on
    94 % of sampled frames, because looking down at a book is what pupils do. Comparing
    against the track's OWN upright baseline removes the pupil who simply sits low, and
    `states.py` then requires a duration, which removes the pupil who is reading.

    Returns False when the head cannot be seen: an occluded head is not a lowered one.
    """
    height = head_height(person, scale)
    if height is None or scale <= 0:
        return False
    return (upright_head_height - height) >= drop_ratio


def left_the_place(seat: Point, position: Point, scale: float, away_ratio: float) -> bool:
    """Is this person away from the place this track occupied when it settled?

    Displacement from the track's own settled position, in that person's own shoulder
    widths. A metric about a chair would need a seating plan; a metric about a track's
    own history needs nothing but the track.

    Horizontal AND vertical, not horizontal alone: a pupil stepping into the aisle moves
    mostly sideways in the image, one walking to the board mostly up it, and both are
    «вне места».
    """
    if scale <= 0:
        return False
    return math.dist(seat, position) >= away_ratio * scale


def motion(previous: Point | None, current: Point | None, scale: float) -> float:
    """How far the anchor moved between two observations, in shoulder widths.

    The raw material for a fidget/stillness measure. Zero when either end is unknown,
    which biases a poorly-seen track toward looking STILL — stated here because the
    aggregation layer has to weigh it against `observations`, not read it as calm.
    """
    if previous is None or current is None or scale <= 0:
        return 0.0
    return math.dist(previous, current) / scale

"""What one body is doing, as pure functions over COCO-17 keypoints.

No model, no config, no database, no frames: an array of numbers in, a bool or a float
out. Same split, and for the same reason, as `qorgan.detection.skeleton` -- which is
where `Keypoints` and the confidence conventions come from, rather than from a second
copy here (rule R2). The judgement is the part that ends up in front of a headteacher,
so it must be testable without a graphics card.

**Everything here is measured in units of the person's own shoulder width, never in
pixels, and that is not a stylistic choice.** `detection/skeleton.py` can use raw pixel
thresholds because every crop it sees has been resized to exactly `CROP_WIDTH`. A
classroom camera has no such luxury: it looks down rows of desks, so the child on the
front row is two or three times the size of the child at the back, in the same frame. A
pixel threshold would fire on one row and never on the other, and the resulting report
would look like a fact about children when it was a fact about seating.

**The honest limit of a single frame, which shapes the whole module.** Whether a hand is
up is visible in one frame: it is the wrist's position relative to that person's own
shoulders. Whether somebody is STANDING is not. A child standing at the back of the room
can occupy the same image height, and sit at the same image y, as a child seated at the
front; nothing in one frame separates them. So there is no `is_standing(keypoints)` here
and there cannot honestly be one. What there is instead is `rose_from_seat`, which asks
whether this person has risen relative to **where this same track was sitting** -- the
same principle the school was promised in §8, «сравниваем ребёнка только с ним самим»,
arrived at here by geometry rather than by policy. The baseline is supplied by the
caller, because holding it is state and this module has none.

**Where the scale itself is a guess, said plainly.** Shoulder width foreshortens when a
child turns sideways: at 60 degrees off-axis the apparent width is halved, which halves
every threshold below and makes each of them easier to cross. The bias therefore runs
towards OVER-counting for a turned child, not under-counting. `lesson.py` blunts it by
feeding the largest shoulder width that track has shown rather than this frame's, but
that is a mitigation and not a fix, and no measurement backs the residue.
"""

from __future__ import annotations

import math

from qorgan.detection.skeleton import (
    KEYPOINT_CONF,
    LEFT_SHOULDER,
    LEFT_WRIST,
    LOOSE_CONF,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoints,
)

# Below this many pixels between the shoulders, the person is too small or too turned for
# any ratio of that width to mean anything: at 6 px, a threshold of 0.35 shoulder widths is
# 2 px, which is inside the pose model's own jitter. Chosen, NOT measured -- picked as
# roughly the width at which the keypoints stop being separable at all. Everything about
# such a track is reported as unknown rather than as zero.
MIN_USABLE_SHOULDER_PX = 8.0

Point = tuple[float, float]


def shoulder_width(person: Keypoints) -> float | None:
    """The distance between the two shoulders: this person's unit of length.

    `None` when either shoulder is missing or the pair is too close together to scale
    anything by. Shoulders rather than the bounding box diagonal (which is what
    `detection/geometry.py` uses for corridor work) because a seated child's box is
    cropped by the desk in front of them and by the child behind them, and its height
    therefore measures the furniture as much as the person. The shoulders are the two
    keypoints a classroom camera sees most reliably: they are above desk height, and
    they stay visible while a child leans, writes, or turns.
    """
    if not person.has(LEFT_SHOULDER, RIGHT_SHOULDER, threshold=LOOSE_CONF):
        return None

    width = math.dist(person.point(LEFT_SHOULDER), person.point(RIGHT_SHOULDER))
    return width if width >= MIN_USABLE_SHOULDER_PX else None


def anchor(person: Keypoints) -> Point | None:
    """The shoulder midpoint: this person's position AND their vertical reference.

    Its **y is the shoulder line** -- the datum a raised wrist is measured against and the
    quantity whose change says somebody stood up. There is deliberately no second
    `shoulder_line_y` function returning that same number: two names for one quantity is
    two places for it to drift, and this codebase has paid for that more than once.

    The nose would be the more intuitive datum and is the wrong one: it swings through
    most of a head's height every time a child looks down at their book, which is the
    single commonest thing a child does at a desk.

    Its **x is the person's position**, in place of the bounding-box centre. A seated
    child's box shrinks and grows as the desk, the chair back and the child in front
    occlude different parts of them, and its centre wanders by a large fraction of the box
    while the child has not moved at all. The midpoint between two shoulders moves when
    the child does.
    """
    if not person.has(LEFT_SHOULDER, RIGHT_SHOULDER, threshold=LOOSE_CONF):
        return None

    left, right = person.point(LEFT_SHOULDER), person.point(RIGHT_SHOULDER)
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def hand_raised(person: Keypoints, scale: float, above_ratio: float) -> bool:
    """Is either wrist held clearly above this person's own shoulder line?

    The one metric in this package that a single frame can honestly answer, because both
    terms belong to the same body: no camera geometry enters, and a child at the back of
    the room is judged by exactly the same rule as one at the front.

    Note the confidence asymmetry. The shoulders are read at `LOOSE_CONF` (via `scale`
    and `anchor`) but the WRIST is required at the stricter `KEYPOINT_CONF`. A
    low-confidence wrist is the model guessing where an arm went, and its guess for an
    occluded arm lands high, near the head -- which is precisely the position being
    tested for. Counting those would manufacture raised hands out of hidden ones.

    **What this cannot separate.** A child stretching, resting their head on a raised
    hand, adjusting their hair, or reaching for a shelf all put a wrist above the
    shoulder line and all count here. `above_ratio` sets how far above, and `lesson.py`
    additionally demands the pose be HELD (a stretch is brief) -- but neither is a
    classifier of intent, and this function does not claim to have one. It reports arm
    geometry, which is what the school was promised: a physical, observable fact.
    """
    shoulders = anchor(person)
    if shoulders is None or scale <= 0:
        return False

    # Screen coordinates: y grows downwards, so "above" is a SMALLER y.
    datum = shoulders[1]
    margin = above_ratio * scale
    return any(
        person.has(wrist, threshold=KEYPOINT_CONF) and person.point(wrist)[1] < datum - margin
        for wrist in (LEFT_WRIST, RIGHT_WRIST)
    )


def rose_from_seat(
    seated_shoulder_y: float, shoulder_y: float, scale: float, rise_ratio: float
) -> bool:
    """Has this person risen from where THIS TRACK was sitting?

    Not "is this person standing" -- see the module docstring for why that question has
    no honest single-frame answer. The caller supplies `seated_shoulder_y`, the baseline
    it established for this track while the track was still; this compares against it.

    Deliberately one-directional. A shoulder line that DROPS by the same amount is not
    reported as anything: a child who slumps, or lies on the desk, produces exactly that,
    and «длительно ли лежит на парте» is a §12.4 metric the school was not promised and
    this package does not compute. See the package docstring on §12.4.
    """
    if scale <= 0:
        return False
    return (seated_shoulder_y - shoulder_y) >= rise_ratio * scale


def left_the_place(seat: Point, position: Point, scale: float, away_ratio: float) -> bool:
    """Is this person away from the place this track occupied when it settled?

    "Вне места", measured as displacement from the track's own settled position, in that
    person's own shoulder widths. A metric about a chair the school never told us the
    position of would need a seating plan we do not have and cannot infer; a metric about
    a track's own history needs nothing but the track.

    Horizontal AND vertical displacement, not horizontal alone: a child who steps out
    into the aisle moves mostly sideways in the image, but one who walks to the board
    moves mostly up it, and both are «вне места».
    """
    if scale <= 0:
        return False
    return math.dist(seat, position) >= away_ratio * scale

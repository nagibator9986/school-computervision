"""Which skeleton belongs to which tracked person. Pure; tested with numbers.

Two models look at a classroom frame and neither answers the other's question. YOLO with
ByteTrack (`models/person.py`) produces boxes that KEEP THEIR IDENTITY across frames but
say nothing about arms. YOLOv8n-pose (`models/pose.py`) produces skeletons that describe
arms but arrive in whatever order the model felt like and carry no identity at all. A
lesson metric needs both: "track 12 raised a hand" is a sentence about a skeleton and a
track id, and this module is the only place the two are joined.

**Getting this wrong is the exact defect `detection/skeleton.py` documents at length**,
one level up. There, an earlier version compared person slot 0 in frame N with person
slot 0 in frame N+1 and called the result displacement -- but the pose model reorders its
output, so it was measuring the distance between two different children. Here the same
mistake would attribute one child's raised hand to the child at the next desk, and the
report would be confidently, unfalsifiably wrong: nobody looking at "track 12: 9 raised
hands" can tell that four of them belonged to track 13.

So the rule is deliberately strict, and it REFUSES rather than guesses:

  * a skeleton is claimed by a track only if its anchor lies inside that track's box;
  * a skeleton whose anchor lies inside TWO boxes is dropped -- in a room where children
    sit in rows, overlapping boxes are the normal case, not the exception;
  * a box that could claim TWO skeletons is dropped, both of them, for the same frame.

Nearest-box matching was rejected, and it is the tempting option. At a desk the nearest
other person is 60 cm away and the cost of being wrong is silent: an arm is attributed,
a counter goes up, and no later stage can detect it. The dropped observations are
counted instead (`Assignment.ambiguous`) and carried to the report, because a hole we can
see is worth more than a number we cannot check -- the same trade the canteen makes when
it force-closes a session as unknown rather than closing it on a weak face match.
"""

from __future__ import annotations

from dataclasses import dataclass

from qorgan.classroom.posture import anchor
from qorgan.detection.geometry import Box
from qorgan.detection.skeleton import Frame, Keypoints


@dataclass(frozen=True, slots=True)
class Assignment:
    """One frame's skeletons, matched to track ids, plus what could not be matched."""

    people: dict[int, Keypoints]
    # Skeletons dropped because the geometry was ambiguous: the anchor fell inside more
    # than one person's box, or one box held more than one anchor. Counted rather than
    # resolved, and surfaced on the lesson report. A room where this is most of the
    # frames is a room whose numbers mean very little, and the report must be able to
    # say so instead of presenting confident totals built from a third of the evidence.
    ambiguous: int
    # Skeletons that landed inside no tracked box at all. Usually the adult at the front,
    # whom the person detector tracks like anybody else, or a partial body at the frame
    # edge -- but also every skeleton in a frame where tracking has not yet started, so
    # it is kept apart from `ambiguous`: the two have different causes and different
    # cures, and a boolean that means both is the bug `migrations/0005` was written about.
    unclaimed: int


def assign(frame: Frame, boxes: dict[int, Box]) -> Assignment:
    """Match this frame's skeletons to this frame's tracks. Ambiguity is refused."""
    claims: dict[int, list[Keypoints]] = {}
    ambiguous = 0
    unclaimed = 0

    for person in frame:
        if person is None:
            continue
        point = anchor(person)
        if point is None:
            unclaimed += 1
            continue

        owners = [track_id for track_id, box in boxes.items() if _contains(box, point)]
        if not owners:
            unclaimed += 1
        elif len(owners) > 1:
            # Inside two boxes at once. Rows of desks make this ordinary, and there is no
            # signal in one frame that says which of the two overlapping children the
            # shoulders belong to.
            ambiguous += 1
        else:
            claims.setdefault(owners[0], []).append(person)

    return _resolve(claims, ambiguous, unclaimed)


def _resolve(
    claims: dict[int, list[Keypoints]], ambiguous: int, unclaimed: int
) -> Assignment:
    """One box, one skeleton. A box that attracted two keeps neither.

    The second skeleton inside a person's box is usually the child SITTING BEHIND them,
    whose head and shoulders appear over the near child's back and whose own box was
    lost to occlusion this frame. Keeping the nearer, or the larger, or the first, would
    be a guess dressed as a rule -- and the guess that goes wrong here credits a raised
    hand to the wrong child.
    """
    people: dict[int, Keypoints] = {}
    for track_id, found in claims.items():
        if len(found) == 1:
            people[track_id] = found[0]
        else:
            ambiguous += len(found)

    return Assignment(people=people, ambiguous=ambiguous, unclaimed=unclaimed)


def _contains(box: Box, point: tuple[float, float]) -> bool:
    return box.x1 <= point[0] <= box.x2 and box.y1 <= point[1] <= box.y2

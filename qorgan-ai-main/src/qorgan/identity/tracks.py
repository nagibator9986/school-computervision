"""Which face belongs to which person? **A pure function, and that is the point.**

The canteen worker used to see a list of faces and nothing else — no idea whether the face
in frame 40 was the same child as the face in frame 1. So it recognised every face, every
time, and the small-face accumulator corroborated hits across whole different children
(spec §4.5).

A person track answers that, and assigning a face to one is geometry. Geometry does not
need a GPU, so this is unit-testable with a handful of boxes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qorgan.detection.geometry import Box
from qorgan.faces.recognizer import FaceBox


def assign_faces_to_tracks(
    faces: Sequence[FaceBox], person_boxes: Mapping[int, Box]
) -> dict[int, FaceBox]:
    """The best face per person track. **One object per track, not a list** (rule R8).

    A face is assigned to the person box that CONTAINS its centre. Two children standing
    close means one face sits inside both boxes, so the tightest box wins — the person
    actually standing there, not the one behind them.

    A face inside nobody's box is DROPPED. It is a poster, a reflection, or a bug, and it
    never gets a track, so it never gets a meal session.
    """
    best: dict[int, FaceBox] = {}

    for face in faces:
        track_id = _owner(face, person_boxes)
        if track_id is None:
            continue
        current = best.get(track_id)
        if current is None or face.quality > current.quality:
            best[track_id] = face

    return best


def _owner(face: FaceBox, person_boxes: Mapping[int, Box]) -> int | None:
    """The tightest person box containing this face's centre."""
    cx, cy = face.box.center

    containing = [
        (box.area, track_id)
        for track_id, box in person_boxes.items()
        if box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2
    ]
    if not containing:
        return None
    return min(containing)[1]

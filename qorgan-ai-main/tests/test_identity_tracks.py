"""Faces to person tracks, by containment. A pure function, tested without a GPU."""

from __future__ import annotations

import numpy as np

from qorgan.detection.geometry import Box
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.tracks import assign_faces_to_tracks


def _face(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> FaceBox:
    return FaceBox(
        box=Box(x1, y1, x2, y2),
        detection_score=score,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )


def test_a_face_inside_a_person_belongs_to_that_person() -> None:
    faces = [_face(110, 110, 150, 160)]
    people = {7: Box(100, 100, 200, 400)}

    assert assign_faces_to_tracks(faces, people) == {7: faces[0]}


def test_a_face_in_nobodys_box_is_dropped() -> None:
    """A face with no person under it is a poster, a reflection, or a bug. It never gets a
    track, so it never gets a meal session."""
    assert assign_faces_to_tracks([_face(10, 10, 40, 50)], {7: Box(500, 500, 600, 800)}) == {}


def test_a_face_between_two_people_goes_to_the_tighter_box() -> None:
    """Two children stand close, so the face lands inside BOTH boxes. It belongs to the
    one whose box it fits most tightly — the person actually standing there, not the one
    behind them with the bigger box."""
    face = _face(110, 110, 150, 160)
    tight = Box(100, 100, 200, 400)
    loose = Box(50, 50, 400, 900)

    assert assign_faces_to_tracks([face], {7: loose, 9: tight}) == {9: face}


def test_one_track_keeps_only_its_BEST_face() -> None:
    """Rule R8: one object per track, not a list. Quality = area x detection score, so a
    big confident face beats a small hesitant one."""
    small = _face(110, 110, 130, 135, score=0.99)
    big = _face(110, 110, 170, 180, score=0.80)
    people = {7: Box(100, 100, 200, 400)}

    assert assign_faces_to_tracks([small, big], people) == {7: big}


def test_two_people_each_keep_their_own_face() -> None:
    left = _face(110, 110, 150, 160)
    right = _face(310, 110, 350, 160)
    people = {7: Box(100, 100, 200, 400), 9: Box(300, 100, 400, 400)}

    assert assign_faces_to_tracks([left, right], people) == {7: left, 9: right}


def test_no_people_means_no_assignments() -> None:
    assert assign_faces_to_tracks([_face(10, 10, 40, 50)], {}) == {}

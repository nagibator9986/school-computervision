"""Face matching: the pure decision, the gap rule, and the small-face path."""

from __future__ import annotations

import numpy as np
import pytest

from qorgan.config.identity import FaceGate, RecognitionPolicy, SoftAccumulator
from qorgan.faces.accumulator import TrackAccumulator, accept_small_face
from qorgan.faces.gallery import normalise
from qorgan.faces.matching import Reason, cosine_scores, identify

POLICY = RecognitionPolicy(min_score=0.45, min_gap=0.05)


def _embedding(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _gallery(*people: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.stack([vector for _, vector in people])
    ids = np.array([person_id for person_id, _ in people], dtype=np.int64)
    return normalise(vectors), ids


def _blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """A face that sits between two people, at a controllable distance."""
    mixed = a * (1 - weight) + b * weight
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


# -- the basics --------------------------------------------------------------


def test_a_known_face_is_recognised() -> None:
    alice = _embedding(1)
    matrix, ids = _gallery((10, alice), (20, _embedding(2)))

    result = identify(alice, matrix, ids, POLICY)

    assert result.accepted
    assert result.person_id == 10
    assert result.reason is Reason.ACCEPTED
    assert result.score == pytest.approx(1.0, abs=0.01)


def test_a_stranger_is_not_recognised() -> None:
    matrix, ids = _gallery((10, _embedding(1)), (20, _embedding(2)))

    result = identify(_embedding(99), matrix, ids, POLICY)

    assert not result.accepted
    assert result.reason is Reason.LOW_SCORE


def test_an_empty_gallery_recognises_nobody_and_says_so() -> None:
    """The system must run end to end with zero pupils imported. Every face is Unknown,
    and the reason says why rather than looking like a failed match."""
    empty = np.zeros((0, 512), dtype=np.float32)

    result = identify(_embedding(1), empty, np.zeros(0, dtype=np.int64), POLICY)

    assert not result.accepted
    assert result.reason is Reason.NO_GALLERY


# -- the gap rule: the ambiguity gate ---------------------------------------


def test_two_children_who_match_equally_well_are_recognised_as_neither() -> None:
    """A face that matches two children almost equally well matches NEITHER. Accepting
    here is how a meal gets recorded against the wrong child."""
    alice, bob = _embedding(1), _embedding(2)
    matrix, ids = _gallery((10, alice), (20, bob))

    ambiguous = _blend(alice, bob, 0.5)
    result = identify(ambiguous, matrix, ids, RecognitionPolicy(min_score=0.1, min_gap=0.15))

    assert not result.accepted
    assert result.reason is Reason.AMBIGUOUS
    assert result.gap < 0.15


def test_a_lone_candidate_gets_no_free_gap() -> None:
    """A lesser legacy bug, in the opposite direction: it set `gap` to a huge sentinel
    when only one candidate existed, disabling the ambiguity gate in exactly the case
    where a single weak match is most dangerous. One person in the gallery means there
    is no second opinion -- so there is no gap evidence either."""
    matrix, ids = _gallery((10, _embedding(1)))

    strict = RecognitionPolicy(min_score=0.1, min_gap=0.05, single_candidate_gap=0.0)
    result = identify(_embedding(1), matrix, ids, strict)

    assert result.gap == 0.0
    assert not result.accepted, "a lone candidate was waved through on a phantom gap"
    assert result.reason is Reason.AMBIGUOUS


def test_a_school_can_choose_to_trust_a_lone_candidate() -> None:
    """It is a policy decision, made explicitly in config, not a sentinel hidden in code."""
    matrix, ids = _gallery((10, _embedding(1)))

    trusting = RecognitionPolicy(min_score=0.1, min_gap=0.05, single_candidate_gap=0.99)
    assert identify(_embedding(1), matrix, ids, trusting).accepted


# -- ranking: THE bug, the one that emptied the canteen log --------------------


def _many_photos_of(person: np.ndarray, count: int) -> list[np.ndarray]:
    """One child, photographed `count` times. Every shot is nearly the same face."""
    return [_blend(person, _embedding(50 + i), 0.05) for i in range(count)]


def test_the_ranking_is_by_person_not_by_photo() -> None:
    """A child with eight photos would otherwise fill the whole top-5 with themselves,
    pushing the runner-up -- the very person the gap rule compares against -- out of
    sight."""
    alice = _embedding(1)
    alice_again = _blend(alice, _embedding(50), 0.05)  # a second photo of the same child
    bob = _embedding(2)

    matrix, ids = _gallery((10, alice), (10, alice_again), (20, bob))

    result = identify(alice, matrix, ids, POLICY)

    people = [r.person_id for r in result.ranked]
    assert people == [10, 20], f"the top-5 was polluted by duplicate photos: {people}"
    assert len(result.ranked) == 2


def test_a_child_with_many_photos_is_still_recognised() -> None:
    """**1 816 of the legacy's 1 820 canteen records have `student_id = NULL`.** This is
    why, and it is worth being exact about, because the obvious reading of the symptom is
    backwards.

    The legacy ranked matches by PHOTO. A child with eight photos in the gallery takes
    the whole top of the ranking with herself, so `top1` and `top2` are two shots of the
    SAME face -- and `gap = top1 - top2` is then a measure of how much one photo of Alice
    differs from another photo of Alice. That number is tiny by construction. It is below
    `min_gap` essentially always.

    So the ambiguity gate did not wave people through, as one might guess from a log full
    of Unknowns. It rejected EVERY child, every time, for being insufficiently different
    from herself. It was not a soft gate that needed tightening -- it was a gate that
    could never open, and every one of the 18 threshold-tuning attempts in the legacy
    configs was rearranging deckchairs, because no value of `min_gap > 0` can pass a gap
    that is structurally ~0.

    Ranking per PERSON is the entire fix. The gap now compares Alice to BOB.
    """
    alice, bob = _embedding(1), _embedding(2)
    gallery = [(10, photo) for photo in _many_photos_of(alice, 8)] + [(20, bob)]
    matrix, ids = _gallery(*gallery)

    result = identify(alice, matrix, ids, POLICY)

    assert result.accepted, "the child with the most photos became the hardest to recognise"
    assert result.person_id == 10
    assert result.gap > POLICY.min_gap

    # And the counterfactual, so nobody can reintroduce per-photo ranking and still be
    # green: rank by photo and the runner-up IS Alice, so the gap collapses to noise.
    per_photo = sorted(cosine_scores(alice, matrix).tolist(), reverse=True)
    photo_gap = per_photo[0] - per_photo[1]
    assert photo_gap < POLICY.min_gap, (
        "this test no longer reproduces the legacy failure it is guarding against"
    )
    assert result.gap > photo_gap * 3


def test_recognition_is_pure_and_carries_its_whole_result() -> None:
    """The legacy wrote its top-5 and its reason onto instance attributes of a shared
    singleton called from four threads, so one thread could read another thread's result
    for another child's face -- and that value closes meal sessions (audit C-06)."""
    matrix, ids = _gallery((10, _embedding(1)), (20, _embedding(2)))
    face = _embedding(1)

    first = identify(face, matrix, ids, POLICY)
    second = identify(face, matrix, ids, POLICY)

    assert first == second, "identify() is not a pure function"
    assert first.ranked, "the result must carry its own evidence, not leave it on `self`"


# -- the small-face path -----------------------------------------------------


def _soft() -> TrackAccumulator:
    return TrackAccumulator(
        SoftAccumulator(
            enabled=True,
            min_score=0.34,
            min_gap=0.12,
            min_hits=2,
            window_seconds=6.0,
            face_gate=FaceGate(min_width=38, min_height=48, min_area=1800),
        )
    )


def _weak_look(person_id: int, score: float = 0.40, gap: float = 0.20):
    from qorgan.faces.matching import Ranked, Recognition

    return Recognition(
        person_id=None,
        score=score,
        gap=gap,
        reason=Reason.LOW_SCORE,
        ranked=(Ranked(person_id, score), Ranked(999, score - gap)),
    )


def test_one_weak_look_is_not_enough() -> None:
    accumulator = _soft()
    result = accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)

    assert not result.accepted


def test_the_same_child_seen_twice_on_ONE_track_is_recognised() -> None:
    """The first-graders. Their faces are systematically below the size gate, so a strict
    single-shot threshold never recognises them AT ALL."""
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=1.0, track_id=1)

    assert result.accepted
    assert result.person_id == 10
    assert result.reason is Reason.ACCEPTED


def test_two_different_TRACKS_do_not_corroborate_each_other() -> None:
    """**THE bug (spec §4.5).**

    `hits` was keyed by person_id, and one accumulator was shared across the whole camera.
    So a weak, wrong top-1 from child A and a weak, wrong top-1 from child B -- two
    entirely different children, both matching pupil X badly -- ADDED UP, and voted a
    stranger into being pupil X.

    That closes a meal session. A crowd of unknowns at a canteen door is not corroboration;
    it is a crowd.
    """
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=1.0, track_id=2)

    assert not result.accepted, (
        "two different children corroborated each other into being one pupil, and that "
        "pupil is now recorded as having eaten"
    )
    assert accumulator.evidence(track_id=1, person_id=10) == 1
    assert accumulator.evidence(track_id=2, person_id=10) == 1


def test_two_different_children_on_one_track_do_not_corroborate_either() -> None:
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(20), accumulator, now=1.0, track_id=1)

    assert not result.accepted


def test_looks_that_are_too_far_apart_in_time_do_not_corroborate() -> None:
    """Two glances a minute apart are not corroboration; they are two separate children
    walking past the same door."""
    accumulator = _soft()

    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    result = accept_small_face(_weak_look(10), accumulator, now=60.0, track_id=1)

    assert not result.accepted


def test_an_ambiguous_face_is_never_rescued_by_repetition() -> None:
    """Seeing two children who look alike five times running does not tell you which one
    it was. The accumulator cannot cure ambiguity; only a better look can."""
    from qorgan.faces.matching import Ranked, Recognition

    accumulator = _soft()
    ambiguous = Recognition(
        None, 0.60, 0.01, Reason.AMBIGUOUS, (Ranked(10, 0.60), Ranked(20, 0.59))
    )

    for tick in range(5):
        result = accept_small_face(ambiguous, accumulator, now=float(tick), track_id=1)
        assert not result.accepted


def test_a_face_that_matched_nobody_never_accumulates() -> None:
    """Otherwise the accumulator eventually 'recognises' a child out of pure noise."""
    accumulator = _soft()
    noise = _weak_look(10, score=0.05, gap=0.001)

    for tick in range(10):
        assert not accept_small_face(noise, accumulator, now=float(tick), track_id=1).accepted


def test_an_accepted_face_is_left_alone() -> None:
    from qorgan.faces.matching import Ranked, Recognition

    accumulator = _soft()
    already = Recognition(10, 0.9, 0.4, Reason.ACCEPTED, (Ranked(10, 0.9),))

    assert accept_small_face(already, accumulator, now=0.0, track_id=1) is already


def test_a_finished_track_can_be_forgotten_in_one_call() -> None:
    accumulator = _soft()
    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=1)
    accept_small_face(_weak_look(20), accumulator, now=0.0, track_id=1)
    accept_small_face(_weak_look(10), accumulator, now=0.0, track_id=2)

    accumulator.clear(track_id=1)

    assert accumulator.evidence(track_id=1, person_id=10) == 0
    assert accumulator.evidence(track_id=2, person_id=10) == 1


def test_the_accumulator_is_bounded() -> None:
    """Rule R8. A canteen runs all year, and track ids only ever increase."""
    accumulator = _soft()

    for index in range(500):
        accept_small_face(_weak_look(index), accumulator, now=float(index), track_id=index)
        accumulator.prune(now=float(index))

    assert len(accumulator.hits) < 20

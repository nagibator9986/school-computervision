"""The bind / retry / evict state machine. Pure: a fake clock, no GPU, no DB."""

from __future__ import annotations

import numpy as np

from qorgan.config.identity import BindingSettings
from qorgan.detection.geometry import Box
from qorgan.faces.matching import Ranked, Reason, Recognition
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.binding import BindingTable, BindState

CONFIG = BindingSettings(
    min_face_frames=3,
    max_wait_seconds=1.5,
    max_attempts=2,
    retry_backoff_seconds=1.0,
    track_ttl_seconds=3.0,
)


def _face(score: float = 0.9, size: float = 100.0) -> FaceBox:
    return FaceBox(
        box=Box(0.0, 0.0, size, size * 1.2),
        detection_score=score,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )


def _accepted(person_id: int = 10) -> Recognition:
    return Recognition(person_id, 0.82, 0.31, Reason.ACCEPTED, (Ranked(person_id, 0.82),))


def _rejected() -> Recognition:
    return Recognition(None, 0.21, 0.02, Reason.LOW_SCORE, (Ranked(10, 0.21), Ranked(20, 0.19)))


# -- observing ----------------------------------------------------------------


def test_one_glance_is_not_enough_to_spend_an_embedding_on() -> None:
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)

    assert not table.should_embed(1, now=0.0)


def test_after_enough_frames_we_embed() -> None:
    table = BindingTable(CONFIG)
    for tick in range(CONFIG.min_face_frames):
        table.observe(1, _face(), now=float(tick) * 0.1)

    assert table.should_embed(1, now=0.3)


def test_a_child_who_keeps_turning_away_is_still_recognised_eventually() -> None:
    """`max_wait_seconds`. A track we have only seen once, but have been watching for a
    second and a half, is embedded anyway — otherwise the child who looks at the floor for
    the whole queue is never recognised at all."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)
    table.observe(1, None, now=1.0)  # no face this frame; the track is still there

    assert not table.should_embed(1, now=1.0)
    assert table.should_embed(1, now=1.6)


def test_a_track_with_no_face_at_all_is_never_embedded() -> None:
    """There is nothing to embed. `should_embed` must not promise a face we do not have."""
    table = BindingTable(CONFIG)
    for tick in range(5):
        table.observe(1, None, now=float(tick))

    assert not table.should_embed(1, now=10.0)


def test_the_best_face_seen_so_far_is_the_one_we_keep() -> None:
    """One object per track, not a list (rule R8)."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(score=0.99, size=40.0), now=0.0)
    binding = table.observe(1, _face(score=0.80, size=120.0), now=0.1)

    assert binding.best is not None
    assert binding.best.width == 120


# -- binding ------------------------------------------------------------------


def test_an_accepted_track_is_never_recognised_again() -> None:
    """**The whole point.** Five children queuing over ten seconds cost 5 embeddings, not
    200 (spec §4.4)."""
    table = BindingTable(CONFIG)
    for tick in range(3):
        table.observe(1, _face(), now=float(tick) * 0.1)
    table.bind(1, _accepted(person_id=42), now=0.3)

    assert table.person_for(1) == 42
    assert table.get(1).state is BindState.BOUND

    for tick in range(40):
        table.observe(1, _face(), now=1.0 + tick * 0.1)
        assert not table.should_embed(1, now=1.0 + tick * 0.1), "a bound track was re-embedded"


def test_a_rejected_track_is_retried_after_a_backoff() -> None:
    """This is where the small-face path lives: a weak look, then a better one."""
    table = BindingTable(CONFIG)
    for tick in range(3):
        table.observe(1, _face(), now=float(tick) * 0.1)
    table.bind(1, _rejected(), now=0.3)

    assert table.get(1).state is BindState.RETRYING
    assert not table.should_embed(1, now=0.5), "retried with no backoff at all"

    table.observe(1, _face(), now=1.4)
    assert table.should_embed(1, now=1.4)


def test_a_track_that_keeps_failing_gives_up_rather_than_burning_the_gpu_forever() -> None:
    table = BindingTable(CONFIG)
    for attempt in range(CONFIG.max_attempts):
        at = attempt * 2.0
        table.observe(1, _face(), now=at)
        table.bind(1, _rejected(), now=at)

    assert table.get(1).state is BindState.EXHAUSTED
    table.observe(1, _face(), now=100.0)
    assert not table.should_embed(1, now=100.0)


# -- eviction -----------------------------------------------------------------


def test_a_track_that_is_gone_is_evicted() -> None:
    """The next child to get this track id is a DIFFERENT child. A binding that outlives
    its track hands one pupil another pupil's identity."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)
    table.bind(1, _accepted(), now=0.0)

    assert table.evict(live=(1,), now=1.0) == []
    assert table.person_for(1) == 10

    assert table.evict(live=(), now=1.0) == []  # still inside the TTL; a flicker is not a loss
    assert table.evict(live=(), now=5.0) == [1]
    assert table.person_for(1) is None


def test_expired_names_the_doomed_without_killing_them_yet() -> None:
    """The caller must be able to look at a track BEFORE it is forgotten. A track that
    dies while it is still RETRYING is a child who walked in and was never recognised, and
    somebody still has to open an Unknown session for them."""
    table = BindingTable(CONFIG)
    table.observe(1, _face(), now=0.0)

    assert table.expired(live=(), now=1.0) == []
    assert table.expired(live=(), now=5.0) == [1]
    assert table.get(1) is not None, "expired() must not evict; that is evict()'s job"

    assert table.evict(live=(), now=5.0) == [1]
    assert table.get(1) is None


def test_the_table_is_bounded() -> None:
    """Rule R8. Track ids only ever increase and a canteen runs all year."""
    table = BindingTable(CONFIG)

    for track_id in range(500):
        now = float(track_id)
        table.observe(track_id, _face(), now=now)
        table.evict(live=(track_id,), now=now)

    assert len(table) < 10

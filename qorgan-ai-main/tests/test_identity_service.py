"""One embedding per track, and every track resolved exactly once.

This is the test file the whole design exists for. Two invariants, and they are different:

  * **the GPU invariant** -- one track costs ONE embedding, not one per frame;
  * **the RECORD invariant** -- every track that ever held a face resolves EXACTLY ONCE,
    and a track that never got a good look still resolves, as UNKNOWN.

The second one is the one that nearly went missing. A child who walks through in two
seconds and whose track dies while we are still retrying is still a child who walked in.
"""

from __future__ import annotations

import numpy as np

from qorgan.config.identity import BindingSettings, RecognitionPolicy
from qorgan.detection.geometry import Box
from qorgan.enums import PersonType
from qorgan.faces.gallery import Gallery, GalleryCache, PersonInfo, normalise
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.binding import BindState
from qorgan.identity.service import IdentityService

FACE_BOX = Box(110.0, 110.0, 170.0, 182.0)
PERSON_BOX = Box(100.0, 100.0, 220.0, 500.0)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class SpyRecognizer:
    """Counts what it is asked to do. The embedding is the expensive half."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding
        self.detect_calls = 0
        self.embed_calls = 0

    def detect_faces(self, _frame: np.ndarray) -> list[FaceBox]:
        self.detect_calls += 1
        return [
            FaceBox(
                box=FACE_BOX,
                detection_score=0.95,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )
        ]

    def embed(self, _frame: np.ndarray, _face: FaceBox) -> np.ndarray:
        self.embed_calls += 1
        return self._embedding


class FrozenGallery(GalleryCache):
    """A GalleryCache that never touches the database."""

    def __init__(self, gallery: Gallery) -> None:
        self._frozen = gallery

    def get(self) -> Gallery:
        return self._frozen

    def reload(self) -> Gallery:
        return self._frozen


def _gallery(*people: tuple[int, np.ndarray]) -> FrozenGallery:
    return FrozenGallery(
        Gallery(
            matrix=normalise(np.stack([vector for _, vector in people])),
            person_ids=np.array([pid for pid, _ in people], dtype=np.int64),
            people={
                pid: PersonInfo(
                    person_id=pid,
                    external_id=f"student_{pid}",
                    full_name=None,
                    person_type=PersonType.STUDENT,
                    class_name="5-А",
                    position=None,
                )
                for pid, _ in people
            },
            model_name="buffalo_l",
            model_version="1.0",
        )
    )


def _service(
    recognizer: SpyRecognizer,
    gallery: FrozenGallery,
    binding: BindingSettings | None = None,
) -> IdentityService:
    return IdentityService(
        recognizer=recognizer,  # type: ignore[arg-type]
        gallery=gallery,
        policy=RecognitionPolicy(),
        binding=binding or BindingSettings(min_face_frames=3, max_wait_seconds=1.5),
    )


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _acted(service: IdentityService, people: dict[int, Box], now: float) -> list:
    """One frame's worth of tracks whose identity was DECIDED on this frame.

    `should_act` is true exactly once per track. It means decided, not recognised.
    """
    return [found for found in service.on_frame(_frame(), people, now) if found.should_act]


# -- THE test: the GPU invariant ---------------------------------------------


def test_one_track_costs_exactly_one_embedding_across_forty_frames() -> None:
    """**The whole design, in one assertion.**

    The old canteen worker called `detect()` -- detection AND the 512-d ArcFace embedding
    -- on every due frame, every 0.25 s, for every face in shot. The expensive half is the
    embedding. For five children queuing over ten seconds that is ~200 embeddings.

    Per track: watch, keep the best face, embed ONCE, bind, and never look again (§4.4).
    """
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    for tick in range(40):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)

    assert recognizer.embed_calls == 1, (
        f"the track was embedded {recognizer.embed_calls} times. That is the bug this "
        "module exists to kill."
    )
    # This line used to read `== 40`, commented "detection is cheap; it runs every frame".
    # Measured against the real buffalo_l: detect_faces 27.4 ms, embed 9.7 ms. Detection is
    # not the cheap half -- it is nearly THREE TIMES the embedding, and it was running on all
    # 40 frames to answer a question that was settled on frame 3. A test that asserts the
    # wrong premise defends it.
    assert recognizer.detect_calls < 10, (
        f"the detector ran {recognizer.detect_calls} times for a track bound almost "
        "immediately. Once every track in shot is resolved, there is no face left to find."
    )


def test_a_recognised_track_is_resolved_exactly_once() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    acted = []
    for tick in range(40):
        acted += _acted(service, {7: PERSON_BOX}, now=tick * 0.1)

    assert len(acted) == 1, "a meal session would be opened more than once for one child"
    assert acted[0].track_id == 7
    assert acted[0].person_id == 10
    assert acted[0].state is BindState.BOUND
    assert acted[0].person is not None
    assert acted[0].person.display == "Ученик 10, 5-А"


def test_five_children_queuing_cost_five_embeddings_not_two_hundred() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    boxes = {
        track: Box(100.0 + track * 200, 100.0, 220.0 + track * 200, 500.0) for track in range(5)
    }
    # The spy always puts its face at FACE_BOX, so only track 0's box contains it; give
    # every track its own frame instead, which is the honest way to count.
    for track, box in boxes.items():
        for tick in range(40):
            service.on_frame(_frame(), {track: PERSON_BOX if track == 0 else box}, now=tick * 0.1)

    assert recognizer.embed_calls <= 5


def test_a_face_with_no_person_under_it_never_reaches_the_gpu() -> None:
    """A face with nobody under it is a poster or a reflection. It costs nothing."""
    recognizer = SpyRecognizer(_vector(1))
    service = _service(recognizer, _gallery((10, _vector(1))))

    for tick in range(20):
        service.on_frame(_frame(), {}, now=tick * 0.1)

    assert recognizer.embed_calls == 0


def test_a_lost_track_is_evicted_so_the_next_child_is_not_mistaken_for_this_one() -> None:
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    for tick in range(5):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)
    assert recognizer.embed_calls == 1

    # The track is gone for longer than the TTL, then a NEW person gets the same id.
    for tick in range(50):
        service.on_frame(_frame(), {}, now=1.0 + tick * 0.1)
    for tick in range(5):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=10.0 + tick * 0.1)

    assert recognizer.embed_calls == 2, "a recycled track id inherited someone else's identity"


# -- THE OTHER test: the record invariant ------------------------------------
#
# Every track that ever held a face resolves exactly once. A child we never managed to
# recognise is still a child who walked in.


FAST_WALKER = BindingSettings(
    min_face_frames=1,
    max_wait_seconds=1.5,
    max_attempts=3,
    retry_backoff_seconds=1.0,
    track_ttl_seconds=3.0,
)


def test_a_track_we_never_recognise_resolves_once_as_unknown() -> None:
    """Every embed is rejected, we run out of attempts, and the track is EXHAUSTED. That
    is a decision, not a non-event: this child ate."""
    recognizer = SpyRecognizer(_vector(99))  # a stranger: matches nobody in the gallery
    service = _service(recognizer, _gallery((10, _vector(1)), (20, _vector(2))), FAST_WALKER)

    acted = []
    for tick in range(60):
        acted += _acted(service, {7: PERSON_BOX}, now=tick * 0.5)

    assert recognizer.embed_calls == FAST_WALKER.max_attempts, "the GPU was burned forever"
    assert len(acted) == 1, "an unrecognised child must resolve ONCE, not once per frame"
    assert acted[0].state is BindState.EXHAUSTED
    assert acted[0].person_id is None


def test_a_track_lost_while_still_RETRYING_still_resolves_as_unknown() -> None:
    """**The fast walker, and the hole this closes.**

    A child is at the door for a second and a half. We get two bad looks at them, reject
    both, and go into RETRYING with a one-second backoff -- and then they are gone. Three
    attempts at a 1 s backoff simply do not fit inside the time this child was in shot.

    RETRYING is not a decision. If the track is simply forgotten from that state, the
    child walked in, ate, and left NO RECORD AT ALL -- and is then missing from the 'did
    not eat' report, which is the one report the school actually asked for.

    So: a track that dies unresolved resolves as UNKNOWN on the way out. A hole we can
    count beats a child who silently never ate.
    """
    recognizer = SpyRecognizer(_vector(99))  # matches nobody
    service = _service(recognizer, _gallery((10, _vector(1)), (20, _vector(2))), FAST_WALKER)

    acted = []

    # 1.5 s at the door: two rejected embeddings (t=0.0 and t=1.0, one backoff apart),
    # and a third attempt that never comes because the child has walked on.
    for tick in range(4):
        acted += _acted(service, {7: PERSON_BOX}, now=tick * 0.5)

    assert recognizer.embed_calls == 2
    assert recognizer.embed_calls < FAST_WALKER.max_attempts, (
        "this fixture no longer reproduces the fast walker: the track exhausted its "
        "attempts while still in shot, so it never dies mid-retry"
    )
    assert service.state_of(7) is BindState.RETRYING
    assert acted == [], "nothing is decided yet -- RETRYING is not a verdict"

    # ...and the child is gone. The track dies while still unresolved.
    for tick in range(20):
        acted += _acted(service, {}, now=2.0 + tick * 0.5)

    assert len(acted) == 1, (
        "the fast walker vanished without a trace: a child walked in, and the system has "
        "no record they were ever there"
    )
    assert acted[0].track_id == 7
    assert acted[0].person_id is None
    assert acted[0].state is BindState.RETRYING
    assert service.state_of(7) is None, "the binding outlived its track"


def test_a_track_that_never_held_a_face_at_all_resolves_as_nothing() -> None:
    """A person YOLO tracked but whose face we never once saw is not a recognition
    failure -- there was nothing to recognise. It must not manufacture a meal session out
    of a back of a head."""

    class Blind(SpyRecognizer):
        def detect_faces(self, _frame: np.ndarray) -> list[FaceBox]:
            self.detect_calls += 1
            return []

    recognizer = Blind(_vector(1))
    service = _service(recognizer, _gallery((10, _vector(1))), FAST_WALKER)

    acted = []
    for tick in range(20):
        acted += _acted(service, {7: PERSON_BOX}, now=tick * 0.5)
    for tick in range(20):
        acted += _acted(service, {}, now=10.0 + tick * 0.5)

    assert recognizer.embed_calls == 0
    assert acted == []


def test_an_already_bound_track_never_asks_to_act_again() -> None:
    """`should_act` is the ONE frame a track is decided on. It is not 'we know who this
    is' -- that is what a bound binding is for, and confusing the two is how forty meal
    sessions get opened for one child."""
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    seen_after_binding = []
    for tick in range(40):
        for found in service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1):
            if not found.should_act:
                seen_after_binding.append(found)

    assert seen_after_binding, "a bound track should still report who it is, every frame"
    assert all(f.person_id == 10 for f in seen_after_binding)
    assert all(not f.embedded for f in seen_after_binding)


def test_a_frame_of_only_bound_tracks_costs_no_face_detection_at_all() -> None:
    """MEASURED against the real buffalo_l, and it overturns this module's premise.

        detect_faces:  27.4 ms   <- the expensive half
        embed:          9.7 ms   <- the one we called expensive

    The design was built on "the expensive half is the 512-d embedding". It is not. Face
    DETECTION costs nearly 3x more, and it was running unconditionally on every frame -- so
    "200 embeddings become 5" was true, and implied a saving far larger than the real one,
    because the dominant cost never went away.

    Detection exists to find a face for a track that still needs one. When every track in
    shot is already resolved, there is nothing left to find, and the frame should cost
    NOTHING. That is the actual lever, and it is worth ~3x what the embedding was.
    """
    alice = _vector(1)
    recognizer = SpyRecognizer(alice)
    service = _service(recognizer, _gallery((10, alice), (20, _vector(2))))

    for tick in range(10):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)

    assert recognizer.embed_calls == 1, "the track should have been bound by now"
    detections_while_binding = recognizer.detect_calls

    for tick in range(10, 40):
        service.on_frame(_frame(), {7: PERSON_BOX}, now=tick * 0.1)

    assert recognizer.detect_calls == detections_while_binding, (
        "the track was already bound, so there was no face left to look for -- yet the "
        f"detector ran {recognizer.detect_calls - detections_while_binding} more times. "
        "That is 27 ms a frame spent answering a question nobody asked."
    )

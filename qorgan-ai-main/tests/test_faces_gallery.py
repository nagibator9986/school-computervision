"""The face gallery: one matrix in memory, not a SQLite scan per face per frame."""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings, RecognitionPolicy
from qorgan.db.models import FaceEmbedding, Person
from qorgan.enums import PersonType
from qorgan.faces.gallery import GalleryCache, load_gallery, normalise
from qorgan.faces.matching import Reason, identify
from qorgan.settings import Settings

MODEL = FaceModelSettings()


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _add_pupil(
    session: Session,
    name: str,
    vector: np.ndarray,
    *,
    person_type: PersonType = PersonType.STUDENT,
    model_name: str = MODEL.model_name,
    active: bool = True,
) -> Person:
    person = Person(
        external_id=f"gen-{name}",
        full_name=name,
        person_type=person_type,
        class_name="5А" if person_type is PersonType.STUDENT else None,
        is_active=active,
    )
    session.add(person)
    session.flush()
    session.add(
        FaceEmbedding(
            person_id=person.id,
            model_name=model_name,
            model_version=MODEL.model_version,
            dim=512,
            normalized=True,
            vector=vector.astype(np.float32).tobytes(),
        )
    )
    session.commit()
    return person


def test_normalising_makes_a_dot_product_a_cosine() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    unit = normalise(vectors)

    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)


def test_a_zero_vector_survives_normalisation() -> None:
    """It must not become NaN and poison every comparison in the gallery."""
    unit = normalise(np.zeros((1, 512), dtype=np.float32))

    assert not np.isnan(unit).any()


def test_an_empty_gallery_is_honest(settings: Settings, session: Session) -> None:
    """The system must run end to end with zero pupils imported."""
    gallery = load_gallery(MODEL.model_name, MODEL.model_version)

    assert gallery.is_empty
    assert gallery.size == 0


def test_the_gallery_loads_every_pupil(settings: Settings, session: Session) -> None:
    _add_pupil(session, "Петрова Мария", _vector(1))
    _add_pupil(session, "Иванов Иван", _vector(2))

    gallery = load_gallery(MODEL.model_name, MODEL.model_version)

    assert gallery.size == 2
    assert len(gallery.people) == 2


def test_a_pupil_is_found_through_the_gallery(settings: Settings, session: Session) -> None:
    face = _vector(1)
    alice = _add_pupil(session, "Петрова Мария", face)
    _add_pupil(session, "Иванов Иван", _vector(2))

    gallery = load_gallery(MODEL.model_name, MODEL.model_version)
    result = identify(face, gallery.matrix, gallery.person_ids, RecognitionPolicy())

    assert result.accepted
    assert result.person_id == alice.id
    assert gallery.info(alice.id).full_name == "Петрова Мария"


def test_embeddings_from_a_different_model_are_never_mixed_in(
    settings: Settings, session: Session
) -> None:
    """The legacy shipped a DeepFace/Facenet512 rebuild script that wrote 512-d vectors
    from a DIFFERENT model into the same column. The dimensions match, so nothing would
    have complained -- the gallery would simply have started returning nonsense, mixing
    two incompatible vector spaces (audit M-29)."""
    _add_pupil(session, "Петрова Мария", _vector(1))
    _add_pupil(session, "Чужая Модель", _vector(2), model_name="facenet512")

    gallery = load_gallery(MODEL.model_name, MODEL.model_version)

    assert gallery.size == 1, "a foreign model's embeddings were loaded into the gallery"


def test_a_deactivated_pupil_leaves_the_gallery(settings: Settings, session: Session) -> None:
    """A child who has left the school must stop being recognised."""
    _add_pupil(session, "Ушедший Ребёнок", _vector(1), active=False)

    assert load_gallery(MODEL.model_name, MODEL.model_version).is_empty


def test_staff_are_in_the_gallery_but_marked_as_staff(settings: Settings, session: Session) -> None:
    """Staff must be recognised -- otherwise the cook opens a meal session every lunchtime
    as an Unknown child -- but they never get one."""
    cook = _add_pupil(session, "Азимов Атабек", _vector(1), person_type=PersonType.STAFF)

    gallery = load_gallery(MODEL.model_name, MODEL.model_version)

    assert gallery.info(cook.id).is_staff


def test_an_unknown_face_against_a_real_gallery_is_low_score(
    settings: Settings, session: Session
) -> None:
    _add_pupil(session, "Петрова Мария", _vector(1))

    gallery = load_gallery(MODEL.model_name, MODEL.model_version)
    result = identify(_vector(99), gallery.matrix, gallery.person_ids, RecognitionPolicy())

    assert not result.accepted
    assert result.reason is Reason.LOW_SCORE


def test_the_cache_does_not_re_read_the_database_on_every_face(
    settings: Settings, session: Session
) -> None:
    """The legacy re-read every embedding BLOB out of SQLite for every face in every
    frame on every camera (audit H-11)."""
    _add_pupil(session, "Петрова Мария", _vector(1))
    cache = GalleryCache(MODEL.model_name, MODEL.model_version)

    first = cache.get()
    second = cache.get()

    assert first is second, "the gallery was re-read from the database"


def test_the_cache_can_be_reloaded_when_a_pupil_is_imported(
    settings: Settings, session: Session
) -> None:
    cache = GalleryCache(MODEL.model_name, MODEL.model_version)
    assert cache.get().is_empty

    _add_pupil(session, "Новенький", _vector(1))

    assert cache.reload().size == 1


class _FakeClock:
    """A monotonic clock a test can advance by hand, so a bounded cadence can be crossed
    without a real sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_a_pupil_enrolled_after_startup_is_recognised_without_a_restart(
    settings: Settings, session: Session
) -> None:
    """The canteen worker loads the gallery ONCE at startup. Before this, running
    `qorgan pupils import-roster` against a live system enrolled a child the running
    worker could never see -- they went unrecognised until someone restarted the process.
    The cache must notice the roster changed and reload, on a bounded cadence."""
    clock = _FakeClock()
    _add_pupil(session, "Уже В Школе", _vector(2))  # the roster the worker booted with
    cache = GalleryCache(
        MODEL.model_name, MODEL.model_version, staleness_check_seconds=30.0, clock=clock
    )
    assert cache.get().size == 1  # loaded once at startup

    face = _vector(1)
    newcomer = _add_pupil(session, "Новенький", face)

    # The very next frame must NOT re-read the table: the staleness check runs on a bounded
    # cadence, not once per face -- re-reading every embedding blob per frame is the exact
    # cost (audit H-11) this cache exists to avoid.
    assert cache.get().size == 1, "the gallery re-read the database before its cadence elapsed"

    # A bounded interval later, with no restart, the running worker notices and reloads.
    clock.advance(31.0)
    gallery = cache.get()

    assert gallery.size == 2, "a pupil enrolled after startup is still invisible to the worker"
    # Recognisable, not merely present: a runner-up now exists, so the match is unambiguous.
    result = identify(face, gallery.matrix, gallery.person_ids, RecognitionPolicy())
    assert result.accepted and result.person_id == newcomer.id

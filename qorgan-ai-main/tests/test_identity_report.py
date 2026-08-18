"""The gallery report. Not a diagnostic -- a shipped command, because the failure it finds
is live in the school's data and fires six times."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.enums import PersonType
from qorgan.faces.gallery import Gallery, PersonInfo, load_gallery, normalise
from qorgan.identity.report import (
    DUPLICATE_SIMILARITY,
    analyse,
    extrapolate,
    person_similarity,
    read_unenrolled,
)
from qorgan.settings import Settings

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _same_face(vector: np.ndarray, seed: int, strength: float) -> np.ndarray:
    """A second photo of the same person, at a controllable similarity.

    `strength` IS the resulting cosine, exactly -- and the arithmetic has to earn that.

    The obvious mix, `vector * s + jitter * (1 - s)`, does NOT give a cosine of `s`: the
    jitter carries its own length, so renormalising lands at `s / sqrt(s**2 + (1-s)**2)`.
    At s = 0.47 that is 0.670 -- which is over DUPLICATE_SIMILARITY, so the pair this
    helper was asked to make an IMPOSTOR would come back flagged as a duplicate, and the
    test would be asserting the opposite of what it reads as.

    So project the jitter onto the subspace orthogonal to `vector` first (Gram-Schmidt),
    then mix with weights (s, sqrt(1 - s**2)) -- a point on the unit circle. The two
    components are now orthonormal, so the cosine is `s` on the nose. Verified below by
    the tests that quote 0.40, 0.47 and 0.78 and mean them.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.normal(size=512).astype(np.float32)
    jitter = jitter - vector * float(vector @ jitter)
    jitter = jitter / np.linalg.norm(jitter)
    mixed = vector * strength + jitter * float(np.sqrt(1.0 - strength**2))
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def _gallery(*rows: tuple[int, str, np.ndarray]) -> Gallery:
    return Gallery(
        matrix=normalise(np.stack([vector for _, _, vector in rows])),
        person_ids=np.array([pid for pid, _, _ in rows], dtype=np.int64),
        people={
            pid: PersonInfo(
                person_id=pid,
                external_id=external,
                full_name=None,
                person_type=PersonType.STUDENT,
                class_name="5-А",
                position=None,
            )
            for pid, external, _ in rows
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


# -- the helper has to mean what it says --------------------------------------


@pytest.mark.parametrize("strength", [0.40, 0.47, 0.60, 0.78, 0.995])
def test_the_fixture_delivers_the_similarity_it_promises(strength: float) -> None:
    """If `_same_face(v, seed, 0.47)` quietly returned 0.67, every threshold test below
    would still be green while asserting the opposite of what it reads as. The fixture is
    load-bearing, so it is tested."""
    him = _face(1)
    assert float(him @ _same_face(him, 5, strength)) == pytest.approx(strength, abs=1e-3)


# -- the matrix ---------------------------------------------------------------


def test_the_matrix_is_per_person_not_per_photo() -> None:
    """A child with two photos is ONE row and ONE column. The gap rule is by person; so
    is this."""
    alice = _face(1)
    matrix, ids = person_similarity(
        _gallery(
            (10, "student_333", alice),
            (10, "student_333", _same_face(alice, 2, 0.98)),
            (20, "student_334", _face(3)),
        )
    )

    assert matrix.shape == (2, 2)
    assert ids == [10, 20]


def test_the_matrix_keeps_the_BEST_photo_pair_between_two_people() -> None:
    """Two people, two photos each. The cross-person cell is the best of the four
    cross-photo scores -- not the first, not the average. A duplicate enrolment that only
    ONE of its two photos exposes must still be caught."""
    him = _face(1)
    matrix, ids = person_similarity(
        _gallery(
            (10, "staff_464", him),
            (10, "staff_464", _face(4)),
            (20, "student_477", _face(5)),
            (20, "student_477", _same_face(him, 2, 0.99)),  # the shot that gives it away
        )
    )

    assert ids == [10, 20]
    assert float(matrix[0, 1]) == pytest.approx(0.99, abs=0.01)


# -- the duplicate enrolments: six people hold two school IDs each ------------


def test_two_ids_holding_one_face_are_flagged_as_a_duplicate_enrolment() -> None:
    """**Two different children do not score 0.999.**

    A second enrolment batch re-registered five people who already existed. Their meals
    split across both ids, so the school's existing canteen records are already wrong for
    them. And the duplicate sits in top-2 and kills the gap, so the system is not
    inaccurate about these six -- it is BLIND to them (spec §2.1, §2.5).
    """
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "staff_464", him),
            (20, "student_477", _same_face(him, 2, 0.995)),  # the same human, second batch
            (30, "student_333", _face(9)),
        )
    )

    assert len(report.duplicates) == 1
    pair = report.duplicates[0]
    assert {pair.external_a, pair.external_b} == {"staff_464", "student_477"}
    assert pair.similarity >= DUPLICATE_SIMILARITY


def test_two_different_children_are_not_a_duplicate() -> None:
    """The legacy carried SAME_PERSON_SIMILARITY = 0.35. Against this data that constant
    would call 55 pairs the same person. The measured band says 0.60 (spec §2.2)."""
    report = analyse(_gallery((10, "student_333", _face(1)), (20, "student_334", _face(2))))

    assert report.duplicates == ()


def test_a_pair_at_0_47_is_an_impostor_and_a_pair_at_0_78_is_a_duplicate() -> None:
    """The band is empty from 0.48 to 0.77. That emptiness IS the measurement: the pairs
    at the top are not a tail of the impostor distribution, they are a different
    population (spec §2)."""
    him = _face(1)
    near = _same_face(him, 5, 0.47)  # ~0.47: two different children who look alike
    twin = _same_face(him, 6, 0.78)  # ~0.78: one human, two ids

    report = analyse(
        _gallery((10, "student_333", him), (20, "student_334", near), (30, "student_472", twin))
    )

    flagged = {frozenset((p.external_a, p.external_b)) for p in report.duplicates}
    assert frozenset(("student_333", "student_472")) in flagged
    assert frozenset(("student_333", "student_334")) not in flagged


def test_the_report_never_merges_anybody() -> None:
    """Detection is not resolution. 7-А 438/439 may be identical TWINS, and arithmetic
    cannot settle that -- only the school can. The report names the pair and stops; Task 7
    builds `pupils merge`. A report that quietly merged would be the legacy's namesake bug
    rebuilt in a new place."""
    him = _face(1)
    gallery = _gallery(
        (438, "student_438", him),
        (439, "student_439", _same_face(him, 3, 0.774)),
    )
    report = analyse(gallery)

    assert len(report.duplicates) == 1
    # Both ids survive, both keep their own row, and the caller is told to ask the school.
    assert report.people == 2
    assert set(gallery.people) == {438, 439}
    assert "merge" in report.summary().lower()


# -- the impostor ceiling, and the floor it implies --------------------------


def test_the_report_names_the_worst_impostor_and_the_floor_it_implies() -> None:
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "student_333", him),
            (20, "student_334", _same_face(him, 5, 0.40)),
            (30, "student_335", _face(7)),
        )
    )

    assert report.impostor_max == pytest.approx(0.40, abs=0.03)
    summary = report.summary()
    assert "impostor" in summary.lower()
    assert "floor" in summary.lower()


def test_duplicates_are_excluded_from_the_impostor_statistics() -> None:
    """A duplicate is not an impostor -- it is the same human. Leaving it in the impostor
    distribution would put its 0.999 at the top and make the ceiling look catastrophic."""
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "staff_464", him),
            (20, "student_477", _same_face(him, 2, 0.995)),
            (30, "student_333", _face(9)),
        )
    )

    assert report.impostor_max < DUPLICATE_SIMILARITY
    assert report.impostor_pairs == 2  # 3 pairs, minus the one duplicate


def test_the_histogram_shows_the_empty_band() -> None:
    """The evidence is a HOLE. A histogram that could not show an empty bucket could not
    show the finding."""
    him = _face(1)
    report = analyse(
        _gallery(
            (10, "student_333", him),
            (20, "student_334", _same_face(him, 5, 0.47)),
            (30, "student_472", _same_face(him, 6, 0.78)),
        )
    )

    counts = {(round(b.low, 2), round(b.high, 2)): b.count for b in report.histogram}
    assert counts[(0.45, 0.50)] == 1  # the worst impostor
    assert counts[(0.75, 0.80)] == 1  # the duplicate
    assert counts[(0.55, 0.60)] == 0  # ... and nothing whatsoever in between
    assert counts[(0.60, 0.65)] == 0


# -- the extrapolation --------------------------------------------------------


def test_the_risk_grows_with_the_school() -> None:
    """9 447 impostor pairs give P(two different children >= 0.45) = 1.06e-4. A school of
    S gives each child S-1 impostors, so P(a child has >=1 impostor above the gate) is
    1 - (1-p)^(S-1). At 800 pupils that is 8.1%: roughly one child in twelve. This is the
    argument for a SECOND photo per child (spec §4.1)."""
    risks = extrapolate(1.06e-4, sizes=(142, 500, 800, 1200))

    by_size = {r.size: r for r in risks}
    assert by_size[142].risk_per_child == pytest.approx(0.015, abs=0.002)
    assert by_size[800].risk_per_child == pytest.approx(0.081, abs=0.003)
    assert by_size[1200].risk_per_child == pytest.approx(0.119, abs=0.004)
    assert by_size[800].children_affected == pytest.approx(65, abs=3)


def test_a_gallery_with_no_impostors_above_the_gate_extrapolates_to_zero_risk() -> None:
    assert all(risk.risk_per_child == 0.0 for risk in extrapolate(0.0))


# -- the photos that could not be enrolled -----------------------------------


def test_the_report_reads_the_unenrollable_photos_back_out_of_the_database(
    settings: Settings, session: Session
) -> None:
    """Four staff photographs contain no detectable face. The import itemised them; the
    gallery report finds them again, next term, without re-running the import."""
    person = Person(
        external_id="staff_465",
        person_type=PersonType.STAFF,
        full_name=None,
    )
    session.add(person)
    session.flush()
    session.add(
        PersonPhoto(
            person_id=person.id,
            path="people/staff/staff_465_1778595393105.jpg",
            sha256="0" * 64,
            quality_note="no face found",
        )
    )
    session.commit()

    unenrolled = read_unenrolled()

    assert len(unenrolled) == 1
    assert unenrolled[0].external_id == "staff_465"
    assert unenrolled[0].reason == "no face found"
    assert unenrolled[0].display == "Сотрудник 465"


def test_a_photo_that_enrolled_cleanly_is_not_reported_as_unenrollable(
    settings: Settings, session: Session
) -> None:
    """`quality_note` is NULL for the photos that worked. If the query forgot to filter on
    it, every photo in the school would be itemised as a failure."""
    person = Person(external_id="student_333", person_type=PersonType.STUDENT, class_name="5-А")
    session.add(person)
    session.flush()
    session.add(
        PersonPhoto(
            person_id=person.id,
            path="people/5-А/student_333_1.jpg",
            sha256="1" * 64,
            quality_note=None,  # it enrolled
        )
    )
    session.commit()

    assert read_unenrolled() == ()


def test_an_empty_gallery_reports_honestly_rather_than_dividing_by_zero(
    settings: Settings, session: Session
) -> None:
    """The system must run end to end with zero pupils imported."""
    report = analyse(load_gallery(MODEL_NAME, MODEL_VERSION))

    assert report.people == 0
    assert report.pairs == 0
    assert report.impostor_probability == 0.0
    assert "0 people" in report.summary()


def test_the_embeddings_are_not_needed_to_prove_the_report_runs(
    settings: Settings, session: Session
) -> None:
    """A sanity check that FaceEmbedding is still the table we read."""
    assert FaceEmbedding.__tablename__ == "face_embeddings"

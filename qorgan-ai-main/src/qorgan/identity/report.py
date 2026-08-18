"""Can this gallery work at all, and who is enrolled twice?

Not a diagnostic. A **shipped command**, because the failure it finds is live in the
school's data and it fires six times.

Measured before a line of the module was written, because if one photo per child cannot
separate 142 children it certainly cannot separate 800, and the module would be built on
sand. Every photo embedded with the production model, then the full 138x138 cosine matrix
(138, not 142: four staff photos have no face).

The band is EMPTY from 0.48 to 0.77. That emptiness is the measurement: the six pairs at
the top are not a tail of the impostor distribution -- they are a different population.
Two different children do not score 0.999. **Six people hold two school IDs each**, their
meals split across both, and the school's existing canteen records are already wrong for
them.

This is the exact mirror of the legacy's namesake bug. The legacy collapsed two children
into one identity; this data does the reverse. The machinery we deleted was aimed at the
wrong failure. This is aimed at the one that is actually present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qorgan.config.identity import FaceModelSettings
from qorgan.faces.gallery import Gallery, load_gallery

# MEASURED (spec §2.1). Above this, two different external_ids are one human. The legacy
# carried SAME_PERSON_SIMILARITY = 0.35, which against this data would have called 55
# pairs the same person.
DUPLICATE_SIMILARITY = 0.60

# The gate the extrapolation below is quoted against. It is the OLD min_score -- the whole
# point is to show what it would cost at a real school's size.
IMPOSTOR_GATE = 0.45

SCHOOL_SIZES = (142, 500, 800, 1200)

HIST_LOW, HIST_HIGH, HIST_WIDTH = 0.30, 1.00, 0.05


@dataclass(frozen=True, slots=True)
class Bucket:
    low: float
    high: float
    count: int


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two external_ids, one human. Detection is not resolution -- see `qorgan pupils
    merge`. Which id is canonical is a decision only the school can make."""

    person_a: int
    person_b: int
    external_a: str
    external_b: str
    display_a: str
    display_b: str
    similarity: float


@dataclass(frozen=True, slots=True)
class SchoolRisk:
    size: int
    risk_per_child: float
    children_affected: float


@dataclass(frozen=True, slots=True)
class Unenrolled:
    external_id: str
    display: str
    photo: str
    reason: str


@dataclass(frozen=True, slots=True)
class GalleryReport:
    people: int
    pairs: int
    histogram: tuple[Bucket, ...]
    below_histogram: int
    duplicates: tuple[DuplicatePair, ...]
    impostor_pairs: int
    impostor_p50: float
    impostor_p90: float
    impostor_p99: float
    impostor_max: float
    impostor_above_gate: int
    gate: float
    extrapolation: tuple[SchoolRisk, ...]
    unenrolled: tuple[Unenrolled, ...]

    @property
    def impostor_probability(self) -> float:
        """P(two different children score above the gate)."""
        if self.impostor_pairs == 0:
            return 0.0
        return self.impostor_above_gate / self.impostor_pairs

    def summary(self) -> str:
        from qorgan.identity.report_text import render

        return render(self)


def person_similarity(gallery: Gallery) -> tuple[np.ndarray, list[int]]:
    """The best cross-person cosine, per PERSON pair. Pure.

    A child with two photos is one row and one column, not two. The gap rule is by person
    (see `faces.matching._rank`, and the 1 816 NULL records that not doing so cost); so is
    this.

    The best of the cross-photo scores, not the first and not the mean: a duplicate
    enrolment that only one of its two photographs gives away must still be caught.
    """
    if gallery.is_empty:
        return np.zeros((0, 0), dtype=np.float32), []

    ids = sorted({int(pid) for pid in gallery.person_ids.tolist()})
    index = {pid: row for row, pid in enumerate(ids)}
    rows = np.array([index[int(pid)] for pid in gallery.person_ids.tolist()])

    scores = (gallery.matrix @ gallery.matrix.T).astype(np.float32)
    best = np.full((len(ids), len(ids)), -1.0, dtype=np.float32)

    # Scatter-max, vectorised over the photo axis: one np.maximum.at per person-row is
    # enough, and it keeps a 138x138 gallery honest without a Python double loop.
    for person, row in enumerate(rows):
        np.maximum.at(best[row], rows, scores[person])

    np.fill_diagonal(best, 1.0)
    return best, ids


def extrapolate(p: float, sizes: Sequence[int] = SCHOOL_SIZES) -> tuple[SchoolRisk, ...]:
    """A school of S gives each child S-1 impostors: 1 - (1-p)^(S-1).

    At 800 pupils and p = 1.06e-4 that is 8.1% -- roughly one child in twelve has an
    impostor above a 0.45 gate. This is the argument for a SECOND photo per child, and it
    belongs in the questions to the school (spec §4.1).
    """
    return tuple(
        SchoolRisk(
            size=size,
            risk_per_child=(risk := 1.0 - (1.0 - p) ** (size - 1)),
            children_affected=size * risk,
        )
        for size in sizes
    )


def analyse(
    gallery: Gallery,
    unenrolled: Sequence[Unenrolled] = (),
    *,
    duplicate_similarity: float = DUPLICATE_SIMILARITY,
    gate: float = IMPOSTOR_GATE,
) -> GalleryReport:
    """Pure: a Gallery in, the whole report out. No database, no GPU."""
    matrix, ids = person_similarity(gallery)
    scores = (
        matrix[np.triu_indices(len(ids), k=1)] if len(ids) > 1 else np.zeros(0, dtype=np.float32)
    )

    # A duplicate is not an impostor -- it is the same human. Leaving its 0.999 in the
    # impostor distribution would make the ceiling look catastrophic and hide the real one.
    impostors = scores[scores < duplicate_similarity]
    above = int((impostors >= gate).sum())
    probability = float(above / len(impostors)) if len(impostors) else 0.0

    return GalleryReport(
        people=len(ids),
        pairs=int(scores.size),
        histogram=_histogram(scores),
        below_histogram=int((scores < HIST_LOW).sum()),
        duplicates=_duplicates(matrix, ids, gallery, duplicate_similarity),
        impostor_pairs=int(impostors.size),
        impostor_p50=_percentile(impostors, 50),
        impostor_p90=_percentile(impostors, 90),
        impostor_p99=_percentile(impostors, 99),
        impostor_max=_percentile(impostors, 100),
        impostor_above_gate=above,
        gate=gate,
        extrapolation=extrapolate(probability),
        unenrolled=tuple(unenrolled),
    )


def _duplicates(
    matrix: np.ndarray, ids: list[int], gallery: Gallery, threshold: float
) -> tuple[DuplicatePair, ...]:
    """Names them. Merges nobody -- 7-А 438/439 may be identical twins, and arithmetic
    cannot settle that. Task 7 builds `pupils merge`; the school decides."""
    found = []
    for i, j in zip(*np.triu_indices(len(ids), k=1), strict=True):
        score = float(matrix[i, j])
        if score < threshold:
            continue
        a, b = gallery.people[ids[i]], gallery.people[ids[j]]
        found.append(
            DuplicatePair(
                person_a=ids[i],
                person_b=ids[j],
                external_a=a.external_id,
                external_b=b.external_id,
                display_a=a.display,
                display_b=b.display,
                similarity=score,
            )
        )
    return tuple(sorted(found, key=lambda pair: -pair.similarity))


def _histogram(scores: np.ndarray) -> tuple[Bucket, ...]:
    """The empty band from 0.48 to 0.77 is the evidence, so the buckets must be able to
    come out zero and be SEEN to."""
    buckets = []
    edge = HIST_LOW
    while edge < HIST_HIGH - 1e-9:
        high = edge + HIST_WIDTH
        buckets.append(
            Bucket(low=edge, high=high, count=int(((scores >= edge) & (scores < high)).sum()))
        )
        edge = high
    return tuple(buckets)


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if values.size else 0.0


def read_unenrolled(school_id: int | None = None) -> tuple[Unenrolled, ...]:
    """The photos the import could not turn into a face, read back out of the database.

    Four staff photographs contain no detectable face at all. The people are on the roster
    and the system can never recognise them. That fact belongs in the database, not in one
    run's stdout (spec §1.1).

    Scoped through `persons`, which this statement already joins for the name: "whose
    photograph could not be enrolled" is a question about one school's roster, and the
    answer names people. `school_id=None` means the only school there is, and raises once
    there are several.
    """
    from sqlalchemy import select

    from qorgan.db.engine import session_scope
    from qorgan.db.models import Person, PersonPhoto
    from qorgan.db.tenancy import owned_by, resolve_school_id
    from qorgan.identity.naming import display_name

    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        rows = session.execute(
            select(
                Person.external_id,
                Person.full_name,
                Person.person_type,
                Person.class_name,
                Person.position,
                PersonPhoto.path,
                PersonPhoto.quality_note,
            )
            # PersonPhoto is the left side -- it is the table with the failures in it. Say
            # so, rather than letting the leftmost COLUMN pick the FROM and joining
            # persons to persons.
            .select_from(PersonPhoto)
            .join(Person, Person.id == PersonPhoto.person_id)
            .where(PersonPhoto.quality_note.is_not(None), owned_by(Person, school))
            .order_by(Person.external_id)
        ).all()

    return tuple(
        Unenrolled(
            external_id=row.external_id,
            display=display_name(row),
            photo=row.path,
            reason=row.quality_note,
        )
        for row in rows
    )


def gallery_report(
    settings: FaceModelSettings | None = None, school_id: int | None = None
) -> GalleryReport:
    """One school's gallery, and the photographs of its people that never enrolled.

    Both halves take the same school. A duplicate pair is two ids belonging to ONE human,
    and a cross-school "duplicate" would be two different children who happen to look
    alike -- offered to an operator with a merge button under it.
    """
    model = settings or FaceModelSettings()
    gallery = load_gallery(model.model_name, model.model_version, school_id)
    return analyse(gallery, read_unenrolled(school_id))

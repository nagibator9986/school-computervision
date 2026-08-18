"""Deciding who a face belongs to. A pure function, and that is the whole point.

**The legacy's recognition was a race condition that could feed the wrong child's lunch
record.** `FaceRecognitionService` was a process-wide singleton, and `_search_db` wrote
its results onto `self.last_ranked_matches` and `self.last_reason` — instance attributes —
before the caller read them back. Four threads called it concurrently (entry watch, exit
watch, the feeder, and the burst handler), so a thread could read the top-5 computed by a
*different thread for a different face*. That value then fed the soft-exit accumulator,
which closes meal sessions. A child could be marked as having eaten because someone else
walked past a camera at the wrong moment (audit C-06).

Here `identify()` takes everything it needs and returns everything it found. There is no
shared mutable state to race on, because there is no state at all.

**And the gap rule is fixed — this is why 1 816 of the legacy's 1 820 canteen records
have `student_id = NULL`.** `gap = top1 − top2` is the ambiguity check: a face that
matches two children almost equally well matches neither of them.

The legacy ranked by PHOTO. A child with eight photos in the gallery fills the top of the
ranking with herself, so `top1` and `top2` are two shots of the SAME face, and the "gap"
measures how much one photo of Alice differs from another photo of Alice. That is ~0 by
construction, and no `min_gap > 0` can ever be cleared by it.

So the gate was not too permissive, which is the natural guess from a log full of
Unknowns. It was impassable: it rejected every child, every time, for failing to be
sufficiently different from herself. The 18 overlapping thresholds in the legacy configs
were months spent tuning a number that could not matter. `_rank` below ranks by PERSON,
which is the whole fix — the gap now compares Alice to BOB.

Separately and much more mildly, the legacy set `gap` to a huge sentinel when only ONE
candidate existed, disabling the gate in the one case where a single weak match is most
dangerous. A lone candidate has no gap evidence, so here it gets none.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from qorgan.config.identity import RecognitionPolicy

# With fewer than two candidates there is no runner-up, and therefore no gap evidence.
MIN_CANDIDATES_FOR_GAP = 2


class Reason(StrEnum):
    """Why we decided what we decided. Recorded, so calibration is possible at all."""

    ACCEPTED = "accepted"
    NO_GALLERY = "no_gallery"  # nobody has been imported yet
    LOW_SCORE = "low_score"  # the best match is not good enough
    AMBIGUOUS = "ambiguous"  # two children match almost equally well
    FACE_TOO_SMALL = "face_too_small"
    # The track ended before we could decide: a child who walked through too fast for us
    # to get a good look. Not a failed match -- a match we never got to make. It is still
    # a child who walked in, so it is still a (Unknown) meal session (spec §4.4).
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Ranked:
    """One candidate, with the score that put it there."""

    person_id: int
    score: float


@dataclass(frozen=True, slots=True)
class Recognition:
    """The complete outcome. Nothing is left on an instance for someone else to read."""

    person_id: int | None
    score: float
    gap: float
    reason: Reason
    ranked: tuple[Ranked, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.person_id is not None

    @property
    def top1(self) -> Ranked | None:
        return self.ranked[0] if self.ranked else None


def cosine_scores(embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similarity of one face against every known face, in a single matmul.

    The legacy re-read every embedding BLOB out of SQLite, decoded each one with
    np.frombuffer, and looped in Python — for every face, in every frame, on every
    camera (audit H-11). This is the same computation as one dot product.

    Both sides are L2-normalised, so the dot product IS the cosine.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ embedding


def identify(
    embedding: np.ndarray,
    matrix: np.ndarray,
    person_ids: np.ndarray,
    policy: RecognitionPolicy,
    *,
    top_k: int = 5,
) -> Recognition:
    """Who is this? Pure: same inputs, same answer, no matter who else is calling."""
    if matrix.size == 0 or person_ids.size == 0:
        return Recognition(None, 0.0, 0.0, Reason.NO_GALLERY)

    scores = cosine_scores(embedding, matrix)
    ranked = _rank(scores, person_ids, top_k)

    best = ranked[0]
    gap = _gap(ranked, policy)

    if best.score < policy.min_score:
        return Recognition(None, best.score, gap, Reason.LOW_SCORE, ranked)

    if gap < policy.min_gap:
        # Two children match almost equally well, so this face matches NEITHER of them.
        # Accepting here is how you record a meal against the wrong child.
        return Recognition(None, best.score, gap, Reason.AMBIGUOUS, ranked)

    return Recognition(best.person_id, best.score, gap, Reason.ACCEPTED, ranked)


def _rank(scores: np.ndarray, person_ids: np.ndarray, top_k: int) -> tuple[Ranked, ...]:
    """Best score per PERSON, not per photo. **This one line is the canteen fix.**

    A child with eight photos would otherwise fill the whole top-5 with herself, and the
    runner-up — the person the gap rule exists to compare against — would be pushed out
    of sight. `top1 - top2` would then be the distance between two photos of ONE child:
    near zero, always, for everyone. The ambiguity gate would fire on every face in the
    school and recognise nobody, which is exactly what the legacy did.
    """
    best_per_person: dict[int, float] = {}
    for person_id, score in zip(person_ids.tolist(), scores.tolist(), strict=False):
        current = best_per_person.get(person_id)
        if current is None or score > current:
            best_per_person[person_id] = score

    ordered = sorted(best_per_person.items(), key=lambda item: -item[1])
    return tuple(Ranked(person_id=pid, score=float(score)) for pid, score in ordered[:top_k])


def _gap(ranked: tuple[Ranked, ...], policy: RecognitionPolicy) -> float:
    """top1 - top2.

    With only one person in the gallery there IS no second opinion, so there is no gap
    evidence either. The legacy returned a huge sentinel here, which disabled the
    ambiguity gate exactly when a single weak match is most dangerous. It returns what
    the config says a lone candidate is worth — 0.0 by default, i.e. nothing.
    """
    if len(ranked) < MIN_CANDIDATES_FOR_GAP:
        return policy.single_candidate_gap
    return ranked[0].score - ranked[1].score

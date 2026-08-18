"""The face gallery: every pupil's embeddings, as one matrix in memory.

The legacy re-read every embedding blob out of SQLite, decoded each with `np.frombuffer`,
and compared them in a Python loop — **for every face, in every frame, on every camera**
(audit H-11). At 130 pupils with several photos each, that is tens of thousands of blob
reads a second, to compute what is one matrix multiply.

It is loaded once, refreshed when the roster changes, and matched with a single matmul.

The embeddings are L2-normalised on the way in, so the dot product IS the cosine
similarity — no division at match time, and no chance of forgetting it.

**THE GALLERY IS ONE SCHOOL'S ROSTER, AND THAT IS THE MOST CONSEQUENTIAL FILTER IN IT.**
This is what a camera matches a face against. Unscoped, a camera in one school compares
every child who walks past it with every child on the INSTALLATION -- so the nearest
neighbour can be a pupil of another school, and the system then writes that child's name
onto a meal session, a recognition attempt and a late identity binding. Nothing reports
it: the match has a high score and a real person on the other end, and the two schools
never compare notes. A wrong school is a wrong child, and it arrives looking exactly like
a correct answer.

`GalleryCache` is therefore one cache per school, fingerprint included: another school's
enrolment must neither force this school's worker to reload nor -- much worse -- mask the
fact that this school's own roster changed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select

from qorgan.db.engine import session_scope
from qorgan.db.models import FaceEmbedding, Person
from qorgan.db.tenancy import owned_by, resolve_school_id
from qorgan.enums import PersonType
from qorgan.identity.naming import display_name
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PersonInfo:
    """Enough to make a decision without going back to the database mid-frame."""

    person_id: int
    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None = None

    @property
    def is_staff(self) -> bool:
        """Staff never open a meal session. One definition, used everywhere.

        The legacy had this heuristic written out twice with different marker words, so
        the same person could be staff in the database and a pupil at runtime (M-28).
        """
        return self.person_type is PersonType.STAFF

    @property
    def display(self) -> str:
        return display_name(self)


@dataclass
class Gallery:
    """An immutable snapshot of who the system knows about."""

    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    person_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    people: dict[int, PersonInfo] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""

    @property
    def is_empty(self) -> bool:
        return self.matrix.size == 0

    @property
    def size(self) -> int:
        """How many embeddings. Several per person is normal and desirable."""
        return int(self.matrix.shape[0])

    def info(self, person_id: int) -> PersonInfo | None:
        return self.people.get(person_id)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise, so a dot product is a cosine. A zero vector stays zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def load_gallery(model_name: str, model_version: str, school_id: int | None = None) -> Gallery:
    """Read every embedding for ONE model version and ONE school.

    Filtering on the model matters: the legacy shipped a DeepFace/Facenet512 rebuild
    script that wrote 512-d vectors from a *different* model into the same column. The
    dimensions match, so nothing would have complained — the gallery would simply have
    started returning nonsense, mixing two incompatible vector spaces (audit M-29).

    Filtering on the school matters MORE, and differently -- see the module docstring.
    `school_id=None` means "the only school there is" and raises once there are several.
    """
    rows = _read_rows(model_name, model_version, school_id)

    if not rows:
        logger.warning(
            "the face gallery is empty; every face will be Unknown",
            extra={"model": model_name, "version": model_version},
        )
        return Gallery(model_name=model_name, model_version=model_version)

    vectors = np.stack([np.frombuffer(row.vector, dtype=np.float32) for row in rows])
    people = {
        row.person_id: PersonInfo(
            person_id=row.person_id,
            external_id=row.external_id,
            full_name=row.full_name,
            person_type=row.person_type,
            class_name=row.class_name,
            position=row.position,
        )
        for row in rows
    }

    logger.info(
        "face gallery loaded",
        extra={"embeddings": len(rows), "people": len(people), "model": model_name},
    )
    return Gallery(
        matrix=normalise(vectors),
        person_ids=np.array([row.person_id for row in rows], dtype=np.int64),
        people=people,
        model_name=model_name,
        model_version=model_version,
    )


def _read_rows(model_name: str, model_version: str, school_id: int | None):
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        return session.execute(
            select(
                FaceEmbedding.person_id,
                FaceEmbedding.vector,
                Person.external_id,
                Person.full_name,
                Person.person_type,
                Person.class_name,
                Person.position,
            )
            .join(Person, Person.id == FaceEmbedding.person_id)
            .where(
                FaceEmbedding.model_name == model_name,
                FaceEmbedding.model_version == model_version,
                Person.is_active.is_(True),
                owned_by(Person, school),
            )
        ).all()


# How often the cache may CHECK whether the roster changed -- not how often it reloads.
# A cost knob, not a domain one (cf. supervisor.SWEEP_SECONDS): the check is one cheap
# indexed aggregate, and a pupil enrolled mid-service becomes recognisable within this
# bound without a restart. Any value well under an operator's patience works; the reload
# itself still happens only when the fingerprint actually moves, so the whole-table read
# stays a per-enrolment event, never a per-frame one (audit H-11).
STALENESS_CHECK_SECONDS = 30.0


class GalleryCache:
    """Holds the gallery for one process and reloads it when the roster changes.

    A pupil is imported perhaps once a term, so the whole table is NOT re-read on a timer.
    Instead a cheap fingerprint (how many embeddings, and the newest id) is checked at most
    once per `staleness_check_seconds`, and the gallery is reloaded only when it moves. That
    is what lets `import-roster` against a live canteen reach the running worker without a
    restart, while a resolved frame still costs no database read at all.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        school_id: int | None = None,
        *,
        staleness_check_seconds: float = STALENESS_CHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        # One cache per school, because one gallery is one school's roster. The
        # fingerprint is scoped too: a pupil enrolled at another school must not make this
        # school's worker reload, and -- worse -- must not be able to keep it from
        # noticing that its OWN roster changed.
        self._school_id = school_id
        self._lock = threading.Lock()
        self._gallery: Gallery | None = None
        self._staleness_check_seconds = staleness_check_seconds
        self._clock = clock
        self._fingerprint: tuple[int, int | None] | None = None
        self._checked_at = 0.0

    def get(self) -> Gallery:
        with self._lock:
            now = self._clock()
            if self._gallery is None:
                self._load(now)
            elif now - self._checked_at >= self._staleness_check_seconds:
                # The cadence gate, not a reload: at most one cheap fingerprint per interval.
                # Without it this check would run per face per frame, which is the whole-table
                # scan (audit H-11) the cache was built to kill.
                self._checked_at = now
                if self._roster_fingerprint() != self._fingerprint:
                    self._load(now)
            return self._gallery

    def reload(self) -> Gallery:
        with self._lock:
            self._load(self._clock())
            return self._gallery

    def _load(self, now: float) -> None:
        # Fingerprint BEFORE the read: a roster write that lands between the two will not
        # match the stored value next time, so it costs at most one redundant reload rather
        # than going unseen until the following enrolment.
        self._fingerprint = self._roster_fingerprint()
        self._gallery = load_gallery(self._model_name, self._model_version, self._school_id)
        self._checked_at = now

    def _roster_fingerprint(self) -> tuple[int, int | None]:
        """A cheap stand-in for "has the gallery's input changed?": how many active
        embeddings this model has, and the newest id among them. A new pupil raises both;
        a deactivated one drops the count (its rows leave the `is_active` set the gallery
        itself filters on). One indexed aggregate, no blobs decoded."""
        with session_scope() as session:
            school = resolve_school_id(session, self._school_id)
            count, max_id = session.execute(
                select(func.count(FaceEmbedding.id), func.max(FaceEmbedding.id))
                .join(Person, Person.id == FaceEmbedding.person_id)
                .where(
                    FaceEmbedding.model_name == self._model_name,
                    FaceEmbedding.model_version == self._model_version,
                    Person.is_active.is_(True),
                    owned_by(Person, school),
                )
            ).one()
        return int(count), (int(max_id) if max_id is not None else None)

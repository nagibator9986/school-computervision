"""Fixtures for the merge tests: two ids that are one human.

Extracted when `test_identity_merge.py` crossed the 500-line cap
(`tests/test_code_limits.py`). The cap is enforced, and the rule is split rather than
loosen -- so the merge tests and the UNDO tests are now two files over one set of
fixtures, instead of one file that had grown a second subject.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole, CameraType, PersonType, SessionState

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"

def face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def same_face(vector: np.ndarray, seed: int, strength: float = 0.995) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = rng.normal(size=512).astype(np.float32)
    jitter = jitter / np.linalg.norm(jitter)
    mixed = vector * strength + jitter * (1.0 - strength)
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def person(
    session: Session,
    external_id: str,
    vector: np.ndarray,
    person_type: PersonType = PersonType.STUDENT,
) -> Person:
    person = Person(
        external_id=external_id,
        person_type=person_type,
        class_name="11-А",
        full_name=None,
    )
    session.add(person)
    session.flush()
    session.add(
        PersonPhoto(
            person_id=person.id,
            path=f"people/11-А/{external_id}.jpg",
            sha256="0" * 64,
        )
    )
    session.add(
        FaceEmbedding(
            person_id=person.id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            dim=512,
            normalized=True,
            vector=vector.astype(np.float32).tobytes(),
        )
    )
    session.commit()
    return person


def camera(session: Session) -> Camera:
    camera = Camera(
        name="canteen_entry",
        display_name="Вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.flush()
    return camera


def open_session(session: Session, person_id: int) -> None:
    entry = camera(session)
    session.add(
        CanteenSession(
            person_id=person_id,
            entry_camera_id=entry.id,
            state=SessionState.OPEN,
            opened_at=utcnow(),
        )
    )
    session.commit()


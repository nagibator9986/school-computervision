"""Fixtures for the pupils section of the dashboard: people, photographs, embeddings.

Shared by `test_web_pupils.py` (the registry) and `test_web_duplicates.py` (the six
pairs), rather than one file growing a second subject: `tests/test_code_limits.py` caps a
file at 500 lines and the rule is split, never loosen. The same split already happened to
`test_identity_merge.py` -- see `identity_merge_fakes.py`, whose `face`/`same_face` are
reused here so that "these two vectors are one human" means the same thing in both suites.

The client fixtures are deliberately NOT here. `test_web_capability_roles.py` and
`test_web_media_capabilities.py` each spell out their own `client_for`, and a fixture
imported across modules is a fixture that is hard to read at the point it is used.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.db.types import utcnow
from qorgan.enums import (
    CameraRole,
    CameraType,
    PersonType,
    SessionOutcome,
    SessionState,
)
from qorgan.events.recorder import media_root

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"

# Not a real JPEG. `/media` decides on the SUFFIX and the resolved path, never on the
# bytes, and no test here decodes an image -- these are photographs of children and the
# suite has no business opening one.
FAKE_JPEG = b"\xff\xd8\xff\xe0not-a-real-jpeg"


def enrol(
    session: Session,
    external_id: str,
    *,
    class_name: str | None = "5-А",
    person_type: PersonType = PersonType.STUDENT,
    full_name: str | None = None,
    position: str | None = None,
    vector: np.ndarray | None = None,
    model_version: str = MODEL_VERSION,
    photo: bool = True,
) -> Person:
    """One person, as `faces.importer` would leave them.

    `vector=None` is the case the registry page has to be able to SAY out loud: four of
    this school's staff photographs contain no detectable face, so those people are on the
    roster and can never be recognised (spec §1.1). A row with no embedding is not a
    fixture convenience, it is live data.
    """
    person = Person(
        external_id=external_id,
        person_type=person_type,
        class_name=class_name,
        position=position,
        full_name=full_name,
    )
    session.add(person)
    session.flush()

    if photo:
        relative = f"people/{class_name or 'staff'}/{external_id}.jpg"
        target = media_root() / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FAKE_JPEG)
        session.add(PersonPhoto(person_id=person.id, path=relative, sha256="0" * 64))

    if vector is not None:
        session.add(
            FaceEmbedding(
                person_id=person.id,
                model_name=MODEL_NAME,
                model_version=model_version,
                dim=512,
                normalized=True,
                vector=vector.astype(np.float32).tobytes(),
            )
        )

    session.commit()
    return person


def canteen_camera(session: Session) -> Camera:
    """The entry camera, created once. `Camera.name` is unique, so a second call reuses it
    rather than raising -- a test about meals should not fail on camera bookkeeping."""
    existing = session.scalar(select(Camera).where(Camera.name == "canteen_entry"))
    if existing is not None:
        return existing

    camera = Camera(
        name="canteen_entry",
        display_name="Столовая, вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()
    return camera


def meal(session: Session, person_id: int) -> CanteenSession:
    """One closed meal session, attributed to a person.

    A session is a ROW here. The legacy kept them in a RAM dict inside a module-global
    singleton, so a restart lost every open session and one code path dropped a pupil out
    of the record entirely.
    """
    row = CanteenSession(
        person_id=person_id,
        entry_camera_id=canteen_camera(session).id,
        state=SessionState.CLOSED,
        outcome=SessionOutcome.ATE,
        opened_at=utcnow(),
        closed_at=utcnow(),
        dwell_seconds=420.0,
    )
    session.add(row)
    session.commit()
    return row

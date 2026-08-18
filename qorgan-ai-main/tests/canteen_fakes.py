"""Test doubles for the canteen pipeline. **One definition, shared by every canteen test.**

`test_canteen_worker` and `test_canteen_records` drive the same pipeline from the same
fakes. A second copy of a fixture is a second thing that can quietly stop reproducing what
its name claims -- and a test whose fixture does not produce what its name says is worse
than no test at all, because it is a guard everybody believes in.

No GPU and no model: the recogniser returns whatever face it is told to, and the person
detector returns whatever boxes it is told to. What is under test here is the ROLE logic
and the RECORD logic, which is where the legacy's 6 330-line function went wrong.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from qorgan.canteen.sessions import SessionManager
from qorgan.capture import Frame
from qorgan.config.camera import CAMERA_ADAPTER, CanteenCamera
from qorgan.config.canteen import MealOutcomeRules, SessionRules
from qorgan.config.identity import BindingSettings
from qorgan.db.models import Camera, FaceEmbedding, Person
from qorgan.detection.geometry import Box
from qorgan.enums import CameraRole, CameraType, PersonType
from qorgan.faces.gallery import GalleryCache
from qorgan.faces.recognizer import FaceBox
from qorgan.identity.service import IdentityService
from qorgan.worker.canteen import CanteenPipeline, _blocks_for

MODEL_NAME, MODEL_VERSION = "buffalo_l", "1.0"

# A child standing in the doorway. Area 210 000 px, far above `min_person_box_area`.
PERSON_BOX = Box(0.0, 0.0, 300.0, 700.0)

# The next child in the queue, standing behind them. Disjoint from PERSON_BOX (IOU 0.0),
# so a face at the top-left of PERSON_BOX is never assigned to this track -- a child in a
# queue who is looking away is a child we have no face for, which is the ordinary case.
BEHIND_BOX = Box(400.0, 50.0, 640.0, 620.0)

# A figure at the far end of the room. Area 3 000 px, BELOW the 3 200 default of
# `min_person_box_area` -- and it still CONTAINS the fake face's centre (55, 65), so the
# track really does acquire a face and really is decided. Without that the box-area test
# would pass for the wrong reason: no face, no act, no session, and nothing proven.
TINY_BOX = Box(30.0, 40.0, 80.0, 100.0)

# One on_frame call resolves a track: one look is enough to embed, and one rejection is
# enough to give up. That is what every role test assumes -- it drives a single frame and
# expects a decision out of it.
ONE_SHOT = BindingSettings(min_face_frames=1, max_wait_seconds=0.0001, max_attempts=1)

# The canteen entry and exit cameras analyse at 15 fps: the sub-stream's own rate
# (capture.stream_fps) with det_every 1. This said 8, from `display_fps` -- a field the
# production loop never read, so the "real cameras' cadence" below was never theirs.
FPS = 15.0


def at_fps(tick: int) -> float:
    """The capture time of the Nth analysis frame, on the real cameras' cadence."""
    return tick / FPS


class FakeRecognizer:
    """Returns whatever face we tell it to. No model, no GPU.

    `detect_calls` is the spy for `_needs_a_face`: face detection costs 25.4 ms against the
    embedding's 10.0, and the pipeline is only affordable because it stops looking for a
    face once every track in shot is resolved.
    """

    def __init__(self) -> None:
        self.faces: list[FaceBox] = []
        self.embedding: np.ndarray = np.zeros(512, dtype=np.float32)
        self.embed_calls = 0
        self.detect_calls = 0

    def show(self, embedding: np.ndarray, width: int = 90, height: int = 110) -> None:
        """One face, at the top-left of the frame -- i.e. inside `PERSON_BOX` and nothing
        else. A face is assigned to the person box that CONTAINS its centre, so this face
        belongs to whichever track is standing in the doorway, and to no other."""
        self.embedding = embedding
        self.faces = [
            FaceBox(
                box=Box(10.0, 10.0, 10.0 + width, 10.0 + height),
                detection_score=0.9,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )
        ]

    def show_nobody(self) -> None:
        self.faces = []

    def detect_faces(self, _frame: np.ndarray) -> list[FaceBox]:
        self.detect_calls += 1
        return self.faces

    def embed(self, _frame: np.ndarray, _face: FaceBox) -> np.ndarray:
        self.embed_calls += 1
        return self.embedding


class FakePersonDetector:
    """One person, always in shot, always track 1 -- until told otherwise.

    `calls` is the spy for the tracking cadence. ByteTrack associates by IOU and a motion
    model, so it needs frame-to-frame continuity; counting these calls is how we know the
    entry and exit cameras are giving it that, and that the inside cameras are not paying
    for it.
    """

    def __init__(self) -> None:
        self.people: dict[int, Box] = {1: PERSON_BOX}
        self.calls = 0

    def sees(self, people: dict[int, Box]) -> None:
        self.people = people

    def walk_away(self) -> None:
        self.people = {}

    def detect(self, _frame: np.ndarray) -> dict[int, Box]:
        self.calls += 1
        return self.people


def vector(seed: int) -> np.ndarray:
    """A normalised 512-d embedding. Seed 99 is a stranger: it matches nobody."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=512).astype(np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


def camera(role: CameraRole) -> CanteenCamera:
    blocks = {
        CameraRole.CANTEEN_ENTRY: {"entry": {}},
        CameraRole.CANTEEN_EXIT: {"exit": {}},
        CameraRole.CANTEEN_INSIDE: {"inside": {}},
    }
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "canteen",
            "role": role.value,
            "name": f"canteen_{role.value.split('_')[-1]}",
            "display_name": role.value,
            "rtsp": {"host": "10.0.0.1", "burst_path": None},
            "canteen": blocks[role],
        }
    )


def frame(at: float = 1000.0) -> Frame:
    return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), seq=1, captured_at=at, camera="c")


def camera_row(session: Session, name: str, display: str, role: CameraRole, host: str) -> Camera:
    row = Camera(
        name=name,
        display_name=display,
        camera_type=CameraType.CANTEEN,
        role=role,
        rtsp_host=host,
    )
    session.add(row)
    return row


def _embed(session: Session, person: Person, raw: np.ndarray) -> None:
    session.add(
        FaceEmbedding(
            person_id=person.id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            dim=512,
            normalized=True,
            vector=raw.tobytes(),
        )
    )


def canteen_rows(session: Session) -> dict:
    """The three canteen cameras, one pupil, one cook, and a face for each.

    A plain function rather than a fixture, so both canteen test modules get the SAME
    database out of it by construction rather than by two people writing it out twice.
    """
    entry = camera_row(session, "canteen_entry", "Вход", CameraRole.CANTEEN_ENTRY, "10.0.0.1")
    exit_row = camera_row(session, "canteen_exit", "Выход", CameraRole.CANTEEN_EXIT, "10.0.0.2")
    inside = camera_row(session, "canteen_inside", "Внутри", CameraRole.CANTEEN_INSIDE, "10.0.0.3")
    session.flush()

    pupil = Person(
        external_id="gen-alice",
        full_name="Петрова Мария",
        person_type=PersonType.STUDENT,
        class_name="5А",
    )
    cook = Person(external_id="gen-cook", full_name="Азимов Атабек", person_type=PersonType.STAFF)
    session.add_all([pupil, cook])
    session.flush()

    faces = {"pupil": vector(1), "cook": vector(2)}
    _embed(session, pupil, faces["pupil"])
    _embed(session, cook, faces["cook"])
    session.commit()

    return {
        "entry_id": entry.id,
        "exit_id": exit_row.id,
        "inside_id": inside.id,
        "pupil": pupil,
        "cook": cook,
        "faces": faces,
    }


def build_pipeline(
    role: CameraRole,
    rows: dict,
    recognizer: FakeRecognizer,
    person: FakePersonDetector | None = None,
    binding: BindingSettings | None = None,
) -> CanteenPipeline:
    sessions = SessionManager(SessionRules(), MealOutcomeRules(), rows["entry_id"], rows["exit_id"])
    camera_id = {
        CameraRole.CANTEEN_ENTRY: rows["entry_id"],
        CameraRole.CANTEEN_EXIT: rows["exit_id"],
        CameraRole.CANTEEN_INSIDE: rows["inside_id"],
    }[role]
    cam = camera(role)

    identity = IdentityService(
        recognizer=recognizer,  # type: ignore[arg-type]
        gallery=GalleryCache(MODEL_NAME, MODEL_VERSION),
        policy=_blocks_for(cam)[0],
        binding=binding or ONE_SHOT,
        soft=None,
    )
    return CanteenPipeline(
        camera=cam,
        camera_id=camera_id,
        person=person or FakePersonDetector(),  # type: ignore[arg-type]
        identity=identity,
        sessions=sessions,
    )

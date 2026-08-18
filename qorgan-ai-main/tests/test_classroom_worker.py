"""The classroom pipeline: when a lesson opens, when it closes, and when it must NOT.

The pipeline itself is plumbing -- everything it decides is decided by pure functions --
so what is worth testing is the lifecycle, and specifically the asymmetry that makes a
mid-lesson restart survivable: **a worker stopping does not end a lesson.**
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.capture import Frame
from qorgan.config.camera import ClassroomCamera
from qorgan.config.classroom import ClassroomConfig, LessonRules, PlaceRules
from qorgan.config.common import RtspSettings
from qorgan.db.models import Camera, Lesson, LessonTrack
from qorgan.detection.geometry import Box
from qorgan.enums import CameraRole, CameraType, LessonCloseReason, LessonState
from qorgan.worker.classroom import ClassroomPipeline
from tests.classroom_fakes import seated, with_hand_up

EMPTY_MINUTES = 1.0
DESK = Box(160.0, 260.0, 240.0, 400.0)


class FakePerson:
    """Stands in for YOLO+ByteTrack. `boxes` is what the next `detect` returns."""

    def __init__(self) -> None:
        self.boxes: dict[int, Box] = {}

    def detect(self, _image: np.ndarray) -> dict[int, Box]:
        return dict(self.boxes)


class FakePose:
    """Stands in for `PoseModel`. Records the imgsz/conf it was asked for."""

    def __init__(self) -> None:
        self.people: list = []
        self.calls: list[tuple[int, float]] = []

    def keypoints(self, _image: np.ndarray, *, imgsz: int, conf: float):
        self.calls.append((imgsz, conf))
        return list(self.people)


def _camera_row(session: Session) -> Camera:
    row = Camera(
        name="class_7a",
        display_name="7-А",
        camera_type=CameraType.CLASSROOM,
        role=CameraRole.CLASSROOM,
        rtsp_host="10.0.0.9",
    )
    session.add(row)
    session.commit()
    return row


def _config() -> ClassroomCamera:
    return ClassroomCamera(
        name="class_7a",
        display_name="7-А",
        role=CameraRole.CLASSROOM,
        rtsp=RtspSettings(host="10.0.0.9"),
        classroom=ClassroomConfig(
            place=PlaceRules(settle_observations=2),
            lesson=LessonRules(
                end_after_empty_minutes=EMPTY_MINUTES,
                flush_interval_seconds=1.0,
                min_presence_seconds=5.0,
            ),
        ),
    )


@pytest.fixture
def pipeline(session: Session):
    camera_row = _camera_row(session)
    person, pose = FakePerson(), FakePose()
    built = ClassroomPipeline(
        camera=_config(), camera_id=camera_row.id, person=person, pose=pose
    )
    return built, person, pose


def _frame(at: float) -> Frame:
    return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), seq=1, captured_at=at, camera="c")


def _occupy(person: FakePerson, pose: FakePose, *, raised: bool = False) -> None:
    person.boxes = {1: DESK}
    pose.people = [with_hand_up() if raised else seated()]


def _empty(person: FakePerson, pose: FakePose) -> None:
    person.boxes = {}
    pose.people = []


def test_an_empty_room_opens_no_lesson(pipeline, session: Session) -> None:
    """A camera pointed at an empty classroom all weekend must not record a lesson."""
    built, person, pose = pipeline
    _empty(person, pose)

    for tick in range(10):
        built.on_frame(built.camera, _frame(tick * 0.1))

    assert session.scalars(select(Lesson)).all() == []


def test_a_lesson_opens_on_the_first_frame_with_a_person(pipeline, session: Session) -> None:
    built, person, pose = pipeline
    _occupy(person, pose)

    built.on_frame(built.camera, _frame(0.0))

    lesson = session.scalar(select(Lesson))
    assert lesson is not None
    assert lesson.state is LessonState.OPEN


def test_the_pose_model_is_asked_for_the_classroom_size_not_the_crop_size(
    pipeline,
) -> None:
    """320 px is the bullying CROP width, and at that size a class of 25 has shoulders
    ~12 px apart -- below `MIN_USABLE_SHOULDER_PX`, so every metric would return unknown
    and the module would produce an empty report while appearing to work."""
    built, person, pose = pipeline
    _occupy(person, pose)

    built.on_frame(built.camera, _frame(0.0))

    imgsz, conf = pose.calls[0]
    assert imgsz == built.camera.classroom.pose.imgsz
    assert conf == built.camera.classroom.pose.conf


def test_counts_are_written_through_during_the_lesson(pipeline, session: Session) -> None:
    """Not only at the end. This is what makes a crash at minute 40 cost half a minute of
    counting rather than the whole lesson."""
    built, person, pose = pipeline
    _occupy(person, pose, raised=True)

    at = 0.0
    for _ in range(40):
        built.on_frame(built.camera, _frame(at))
        at += 0.1

    row = session.scalar(select(LessonTrack))
    assert row is not None, "nothing was written while the lesson was still running"
    assert row.hand_raises == 1


def test_stopping_the_worker_leaves_the_lesson_open(pipeline, session: Session) -> None:
    """**The asymmetry that makes a restart survivable, and the point of the whole table.**

    A stop that closed the lesson would report it as having ended at the moment the worker
    died, and the next worker would open a second lesson at minute 20. A lesson ends when
    the ROOM empties, or when the janitor times it out -- never because a process did.
    """
    built, person, pose = pipeline
    _occupy(person, pose, raised=True)
    at = 0.0
    for _ in range(20):
        built.on_frame(built.camera, _frame(at))
        at += 0.1

    built.stop()

    lesson = session.scalar(select(Lesson))
    assert lesson.state is LessonState.OPEN
    assert lesson.ended_at is None


def test_stopping_writes_the_counts_through_first(pipeline, session: Session) -> None:
    """Leaving the lesson open is only half of it: the ledgers in RAM must reach the row,
    or the restart resumes a lesson whose last thirty seconds never happened."""
    built, person, pose = pipeline
    _occupy(person, pose, raised=True)
    at = 0.0
    for _ in range(8):  # fewer frames than one flush interval
        built.on_frame(built.camera, _frame(at))
        at += 0.1

    built.stop()

    row = session.scalar(select(LessonTrack))
    assert row is not None and row.hand_raises == 1


def test_the_lesson_closes_once_the_room_has_been_empty_long_enough(
    pipeline, session: Session
) -> None:
    built, person, pose = pipeline
    _occupy(person, pose)
    built.on_frame(built.camera, _frame(0.0))

    _empty(person, pose)
    built.on_frame(built.camera, _frame(EMPTY_MINUTES * 60 + 1.0))

    lesson = session.scalar(select(Lesson))
    assert lesson.state is LessonState.CLOSED
    assert lesson.close_reason is LessonCloseReason.EMPTY_ROOM
    assert lesson.ended_at is not None


def test_a_brief_gap_does_not_end_the_lesson(pipeline, session: Session) -> None:
    """Everyone looking down at once, or one long occlusion, is not the end of a lesson."""
    built, person, pose = pipeline
    _occupy(person, pose)
    built.on_frame(built.camera, _frame(0.0))

    _empty(person, pose)
    built.on_frame(built.camera, _frame(10.0))

    assert session.scalar(select(Lesson)).state is LessonState.OPEN


def test_a_new_lesson_starts_after_the_room_refills(pipeline, session: Session) -> None:
    """The next lesson is a NEW record, not a resumption of the one that ended."""
    built, person, pose = pipeline
    _occupy(person, pose)
    built.on_frame(built.camera, _frame(0.0))

    _empty(person, pose)
    built.on_frame(built.camera, _frame(EMPTY_MINUTES * 60 + 1.0))

    _occupy(person, pose)
    built.on_frame(built.camera, _frame(EMPTY_MINUTES * 60 + 30.0))

    lessons = session.scalars(select(Lesson)).all()
    assert len(lessons) == 2
    assert [lesson.state for lesson in lessons] == [LessonState.CLOSED, LessonState.OPEN]


def test_an_ambiguous_frame_is_counted_on_the_lesson(pipeline, session: Session) -> None:
    """Two skeletons in one box: both refused, both counted, and the count must survive
    all the way to the row -- a hole we can see is worth more than a number we cannot.

    Also pins WHAT THE ROW MEANS while a lesson is running: it is the truth as of the last
    flush, not as of the last frame. Twelve of these twenty frames had been flushed when
    the loop ended, so the row says 24 and not 40, and `stop()` is what brings it current.
    Reading a live lesson's row as "the total so far" would understate every number on it
    by up to `flush_interval_seconds` of counting.
    """
    built, person, pose = pipeline
    person.boxes = {1: DESK}
    pose.people = [
        seated(centre_x=200.0, shoulder_y=380.0),
        seated(centre_x=205.0, shoulder_y=280.0),
    ]

    at = 0.0
    for _ in range(20):
        built.on_frame(built.camera, _frame(at))
        at += 0.1

    flushed = session.scalar(select(Lesson)).ambiguous_observations
    assert 0 < flushed < 40, "the row should hold the totals as of the last flush"

    built.stop()
    session.expire_all()
    assert session.scalar(select(Lesson)).ambiguous_observations == 40

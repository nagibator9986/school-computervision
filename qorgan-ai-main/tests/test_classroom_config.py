"""The classroom camera's schema, and the one pose model per worker process.

Two things are being defended here. First, that the discriminated union makes an
identifying classroom camera UNREPRESENTABLE rather than merely discouraged -- the same
mechanism that stopped the legacy shipping 25 bullying keys onto two canteen cameras.
Second, rule R2 at the process level: one set of pose weights, or a loud refusal.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qorgan.config.camera import CAMERA_ADAPTER, BullyingCamera, ClassroomCamera
from qorgan.config.classroom import ClassroomConfig, LessonRules, PoseSettings
from qorgan.config.common import RtspSettings
from qorgan.config.workers import WorkerGroup
from qorgan.enums import CameraRole, CameraType
from qorgan.worker.builders import _pose_model, _pose_weights


def _classroom(**kwargs) -> dict:
    return {
        "name": "class_7a",
        "display_name": "7-А",
        "camera_type": "classroom",
        "role": "classroom",
        "rtsp": {"host": "10.0.0.9"},
        **kwargs,
    }


# -- the camera schema --------------------------------------------------------


def test_a_classroom_camera_loads_from_the_discriminated_union() -> None:
    camera = CAMERA_ADAPTER.validate_python(_classroom())

    assert isinstance(camera, ClassroomCamera)
    assert camera.camera_type is CameraType.CLASSROOM
    assert camera.role is CameraRole.CLASSROOM


def test_a_classroom_camera_with_a_corridor_role_is_refused() -> None:
    with pytest.raises(ValidationError, match="not a classroom role"):
        CAMERA_ADAPTER.validate_python(_classroom(role="main_hall"))


def test_a_classroom_camera_cannot_carry_a_canteen_block() -> None:
    """**§8 made unrepresentable rather than merely documented.**

    `canteen` is where every recognition threshold in this schema lives. A classroom
    camera that could carry one is a config file away from identifying children in a
    classroom, which the school was told in writing would not happen -- and which this
    school's own footage says would not work anyway (14 970 corridor faces, zero
    recognised). `extra="forbid"` turns that from a promise into a startup error.
    """
    with pytest.raises(ValidationError):
        CAMERA_ADAPTER.validate_python(_classroom(canteen={"session": {}}))


def test_a_classroom_camera_cannot_carry_a_bullying_block() -> None:
    with pytest.raises(ValidationError):
        CAMERA_ADAPTER.validate_python(_classroom(bullying={"metrics": {}}))


def test_nothing_in_the_classroom_schema_is_shaped_like_an_identity() -> None:
    """A field-level guard, so that adding one has to be a deliberate edit to this test.

    The camera-type union stops a `canteen` block arriving whole; this stops a
    `min_score`, a `person_id` or a `recognition` growing directly on `ClassroomConfig`.
    """
    banned = ("min_score", "person", "student", "pupil", "recognition", "face", "identity")

    def _fields(model, seen=frozenset()) -> set[str]:
        found: set[str] = set()
        for name, field in model.model_fields.items():
            found.add(name)
            nested = field.annotation
            if hasattr(nested, "model_fields") and nested not in seen:
                found |= _fields(nested, seen | {nested})
        return found

    offenders = {f for f in _fields(ClassroomConfig) if any(b in f for b in banned)}
    assert not offenders, f"the classroom schema grew an identity-shaped key: {offenders}"


# -- the rules ---------------------------------------------------------------


def test_a_presence_threshold_longer_than_a_lesson_is_refused() -> None:
    """Every track would be a fragment, so every lesson would report an empty room."""
    with pytest.raises(ValidationError, match="min_presence_seconds"):
        LessonRules(max_lesson_minutes=10.0, min_presence_seconds=700.0)


def test_an_empty_room_timeout_longer_than_a_lesson_is_refused() -> None:
    """Every lesson would be force-closed as TIMEOUT and the close reason would stop
    carrying any information at all."""
    with pytest.raises(ValidationError, match="end_after_empty_minutes"):
        LessonRules(max_lesson_minutes=10.0, end_after_empty_minutes=20.0)


def test_an_unknown_classroom_key_is_a_startup_error() -> None:
    """R10. A typo must not silently use the default -- the legacy had 225 unvalidated
    keys, so a misspelling did nothing and nobody found out for months."""
    with pytest.raises(ValidationError):
        CAMERA_ADAPTER.validate_python(_classroom(classroom={"hand_rais": {}}))


# -- one pose model per process (R2) -----------------------------------------


def _group() -> WorkerGroup:
    return WorkerGroup(name="room_group", cameras=["class_7a"])


def _bullying() -> BullyingCamera:
    return BullyingCamera(
        name="hall_left",
        display_name="Холл",
        role=CameraRole.MAIN_HALL,
        rtsp=RtspSettings(host="10.0.0.1"),
    )


def test_a_group_with_no_pose_camera_loads_no_pose_model() -> None:
    """A loader that resolves to nothing, and no ultralytics import: a canteen-only group
    must not pay for weights it never uses."""
    loader = _pose_model(_group(), {})

    assert loader.model is None
    assert loader() is None


def test_the_pose_model_is_not_loaded_until_something_asks_for_it() -> None:
    """Lazy, and the laziness is load-bearing rather than an optimisation.

    `_serve` cannot know whether anything will want pose until the builders have run. An
    eager load also made `tests/test_disabled_camera.py` pull in a real model despite
    having patched every builder out -- that is how a model-free test quietly becomes a
    GPU-dependent one. Building the loader must touch neither ultralytics nor the GPU.
    """
    classroom = CAMERA_ADAPTER.validate_python(_classroom())

    loader = _pose_model(_group(), {"class_7a": classroom})

    assert loader.model == "yolov8n-pose.pt"  # resolved...
    assert loader._loaded is None  # ...but not loaded


def test_the_weights_a_group_needs_are_collected_from_both_camera_kinds() -> None:
    classroom = CAMERA_ADAPTER.validate_python(_classroom())
    cameras = {"class_7a": classroom, "hall_left": _bullying()}

    assert _pose_weights(cameras) == {"yolov8n-pose.pt"}


def test_a_group_asking_for_two_different_pose_models_is_refused_at_startup() -> None:
    """**Rule R2 at the process level, and it fails loudly rather than costing 6 MB.**

    Two sets of pose weights in one process is two pose paths that will diverge, which is
    the defect that let the legacy's eval harness grade code production did not run. It
    is a configuration mistake with an obvious fix -- split the group, or make the cameras
    agree -- so it says so at startup, naming both, instead of quietly loading both.

    The refusal happens BEFORE `require_gpu()` and before ultralytics is imported, which
    is what lets this run on a machine with no GPU at all.
    """
    classroom = CAMERA_ADAPTER.validate_python(
        _classroom(classroom={"pose": {"model": "yolov8s-pose.pt"}})
    )
    cameras = {"class_7a": classroom, "hall_left": _bullying()}

    with pytest.raises(ValueError, match="more than one pose model"):
        _pose_model(_group(), cameras)


def test_the_classroom_looks_at_a_bigger_image_than_the_bullying_crop() -> None:
    """320 px is the two-person CROP width. Across a whole room it puts a child's
    shoulders ~12 px apart -- under `MIN_USABLE_SHOULDER_PX`, so every metric would return
    unknown and the module would produce empty reports while appearing to work."""
    from qorgan.models.pose import CROP_WIDTH

    assert PoseSettings().imgsz > CROP_WIDTH

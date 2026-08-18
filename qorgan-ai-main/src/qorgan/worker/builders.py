"""Which pipelines a worker group runs, and what each one loads.

Split out of `entrypoint.py` when the fourth camera type arrived (§12.1's weapons) and
that file passed 500 lines (rule R1: split, never loosen). The boundary is a real one
rather than a convenience: `entrypoint.py` is the PROCESS -- signals, heartbeat, the
shutdown budget, the loop that keeps it alive -- and this is what that process decides to
build out of a group's cameras. Nothing here knows about signals and nothing there knows
about a model.

Two rules survive the move and are the reason most of this file is comments:

  * **ONE pose model per process, loaded lazily.** Two callers ask it different questions
    -- a ~320 px crop of a pair, and a whole classroom -- and there is one set of weights
    between them (rule R2; see `qorgan/models/pose.py` on why a second pose path is the
    defect that made the legacy's eval harness worthless).
  * **A `PersonDetector` per CAMERA, never per group.** Ultralytics keeps its tracker
    state on the model object, so two cameras sharing one would hand each other's children
    the same track ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qorgan.config.camera import (
    BullyingCamera,
    CameraConfig,
    CanteenCamera,
    ClassroomCamera,
    WeaponsCamera,
)
from qorgan.config.workers import WorkerGroup
from qorgan.enums import CameraRole
from qorgan.logging_setup import get_logger

if TYPE_CHECKING:  # heavy: torch + ultralytics. Never imported at module load.
    from qorgan.models.pose import PoseModel
    from qorgan.worker.bullying import BullyingPipeline
    from qorgan.worker.classroom import ClassroomPipeline
    from qorgan.worker.weapons import WeaponsPipeline

logger = get_logger(__name__)

def build_all(group: WorkerGroup, cameras: dict[str, CameraConfig]) -> dict[str, object]:
    """Every pipeline this group runs, keyed by camera name.

    ONE pose model for the process, handed to whoever needs it. Two callers ask it
    different questions -- a ~320 px crop of a pair, and a whole classroom -- but there is
    one set of weights and one extraction. See `qorgan/models/pose.py` on why a second
    pose path is the defect that made the legacy's eval harness worthless.
    """
    pose = _pose_model(group, cameras)
    return {
        **_build_pipelines(group, cameras, pose),
        **_build_canteen(group, cameras),
        **_build_classroom(group, cameras, pose),
        **_build_weapons(group, cameras),
    }


def _pose_weights(cameras: dict[str, CameraConfig]) -> set[str]:
    """Which pose weights this group's cameras ask for. Usually one name, often none."""
    wanted = {c.bullying.skeleton.model for c in cameras.values() if isinstance(c, BullyingCamera)}
    wanted |= {c.classroom.pose.model for c in cameras.values() if isinstance(c, ClassroomCamera)}
    return wanted


class PoseLoader:
    """The group's one pose model, loaded on first use and never twice.

    **Lazy on purpose.** The weights and the ultralytics import are hundreds of megabytes
    of process, and `_serve` cannot know whether anything will ask for them until the
    builders have run -- a group of canteen cameras needs no pose at all. Eager loading
    also made `tests/test_disabled_camera.py` pull in a real model despite having patched
    every builder out, which is how a model-free test quietly becomes a GPU-dependent one.

    Memoised, because being asked twice is the normal case: a process watching both a
    corridor and a classroom has two callers with two different questions and must still
    have ONE set of weights between them (rule R2 -- see `qorgan/models/pose.py`).
    """

    def __init__(self, model: str | None, device: str) -> None:
        self.model = model
        self._device = device
        self._loaded: PoseModel | None = None

    def __call__(self) -> PoseModel | None:
        if self.model is None:
            return None
        if self._loaded is None:
            from qorgan.gpu import require_gpu
            from qorgan.models.pose import PoseModel

            # Refuse to run 40x too slow on the CPU rather than doing it silently.
            require_gpu()
            self._loaded = PoseModel(self.model, device=self._device)
        return self._loaded


def _pose_model(group: WorkerGroup, cameras: dict[str, CameraConfig]) -> PoseLoader:
    """Which pose weights this process will run, checked now and loaded later.

    **The CHECK is eager and the LOAD is not**, and the split is deliberate: naming two
    different pose models in one group is a configuration mistake, and a configuration
    mistake must be a startup error rather than something discovered on the first frame of
    a lesson. Loading is deferred to `PoseLoader`, which explains why.

    A group whose corridor camera and classroom camera name different pose models is a
    group in which the two would silently diverge -- rule R2's whole subject, and the shape
    of the legacy defect that let the eval harness grade code production did not run. It
    has an obvious fix (split the group, or agree on the model), so it fails naming both
    instead of costing 6 MB and a month of confusion.
    """
    wanted = _pose_weights(cameras)
    if len(wanted) > 1:
        raise ValueError(
            f"worker group {group.name!r} asks for more than one pose model: "
            f"{sorted(wanted)}. One process runs one set of pose weights, so that the "
            "corridor and the classroom cannot drift apart. Split the group, or make the "
            "cameras agree."
        )
    return PoseLoader(next(iter(wanted)) if wanted else None, group.device)


def _build_pipelines(
    group: WorkerGroup, cameras: dict[str, CameraConfig], pose: PoseLoader
) -> dict[str, BullyingPipeline]:
    """Load the models and build a detection pipeline per bullying camera.

    Deferred imports: torch and ultralytics are several hundred megabytes of process, and
    a group with no bullying cameras should not pay for them.
    """
    bullying = {n: c for n, c in cameras.items() if isinstance(c, BullyingCamera)}
    if not bullying:
        return {}

    from qorgan.events.store import ensure_cameras
    from qorgan.gpu import require_gpu
    from qorgan.models.person import PersonDetector
    from qorgan.models.pose import PoseEstimator
    from qorgan.worker.bullying import BullyingPipeline

    # Refuse to run 40x too slow on the CPU rather than doing it silently.
    require_gpu()

    camera_ids = ensure_cameras(cameras)
    # One pose model for the whole group: it has no per-camera state to corrupt. The
    # weights come from `_pose_model` so that a classroom camera in the same process
    # shares them rather than loading a second copy of the same file.
    estimator = PoseEstimator(
        next(iter(bullying.values())).bullying.skeleton, device=group.device, model=pose()
    )

    return {
        name: BullyingPipeline(
            camera=camera,
            camera_id=camera_ids[name],
            person=PersonDetector(camera.yolo, name, device=group.device),
            pose=estimator,
        )
        for name, camera in bullying.items()
    }


def _build_classroom(
    group: WorkerGroup, cameras: dict[str, CameraConfig], pose: PoseLoader
) -> dict[str, ClassroomPipeline]:
    """One lesson pipeline per classroom camera.

    A `PersonDetector` each, never one shared, for the reason `models/person.py` states:
    Ultralytics keeps its tracker state on the model object, so two classrooms sharing
    one would hand each other's children the same track ids -- and here a track id is the
    whole of a row's identity.
    """
    classroom = {n: c for n, c in cameras.items() if isinstance(c, ClassroomCamera)}
    if not classroom:
        return {}

    from qorgan.events.store import ensure_cameras
    from qorgan.models.person import PersonDetector
    from qorgan.worker.classroom import ClassroomPipeline

    # A classroom camera always contributes its weights to `_pose_weights`, so the loader
    # can only be empty here if this function was called with cameras the loader never
    # saw. That is a wiring mistake, not a runtime condition, and it must not degrade into
    # a camera that is watched by nobody -- which is the failure `_switched_on` exists to
    # make impossible for the enabled/disabled case.
    model = pose()
    if model is None:
        raise ValueError(
            f"worker group {group.name!r} has classroom camera(s) {sorted(classroom)} but "
            "no pose model was resolved for them. The pose loader was built from a "
            "different set of cameras than the pipelines."
        )

    camera_ids = ensure_cameras(cameras)
    return {
        name: ClassroomPipeline(
            camera=camera,
            camera_id=camera_ids[name],
            person=PersonDetector(camera.yolo, name, device=group.device),
            pose=model,
        )
        for name, camera in classroom.items()
    }


def _build_weapons(
    group: WorkerGroup, cameras: dict[str, CameraConfig]
) -> dict[str, WeaponsPipeline]:
    """One weapons pipeline per weapons camera — **or no worker at all.**

    `YoloWeaponModel` raises `WeaponWeightsUnusable` from its constructor and **nothing
    here catches it.** The exception propagates out of `build_all`, out of `_serve`, and
    into `run_group`, which logs "worker crashed" and re-raises; the supervisor restarts
    the group, it fails again, and the backoff turns it into a loud repeating failure with
    the missing file named in every line.

    **There is deliberately no `except` and no degraded mode.** A weapons camera whose
    weights are missing must not come up watching nothing -- that is exactly what the
    previous system did: warn, fall back to motion analysis, and report healthy for months
    with a 0-byte `best.pt`. A camera the school believes is looking for knives and is not
    is worse than a camera that is visibly down.
    """
    weapons = {n: c for n, c in cameras.items() if isinstance(c, WeaponsCamera)}
    if not weapons:
        return {}

    from qorgan.events.store import ensure_cameras

    # Refuse to run 40x too slow on the CPU rather than doing it silently. Two models per
    # frame here, and one of them is the reason the camera exists.
    _require_gpu_now()

    camera_ids = ensure_cameras(cameras)
    return {
        name: _one_weapons_pipeline(camera, camera_ids[name], group.device)
        for name, camera in weapons.items()
    }


def _require_gpu_now() -> None:
    from qorgan.gpu import require_gpu

    require_gpu()


def _one_weapons_pipeline(
    camera: WeaponsCamera, camera_id: int, device: str
) -> WeaponsPipeline:
    """One camera's two models and the pipeline over them.

    A `PersonDetector` each, never one shared, for the reason `models/person.py` gives:
    Ultralytics keeps its tracker state on the model object, and here a person's track id
    is what «рядом с человеком» is measured against.
    """
    from qorgan.models.person import PersonDetector
    from qorgan.weapons.model import YoloWeaponModel
    from qorgan.worker.weapons import WeaponsPipeline

    return WeaponsPipeline(
        camera=camera,
        camera_id=camera_id,
        person=PersonDetector(camera.yolo, camera.name, device=device),
        weapons=YoloWeaponModel(
            camera.weapons.model,
            camera.name,
            tuple(camera.weapons.target_classes),
            device=device,
            # Passed so the model can say, at every start, whether screen 3 is inert on
            # these weights. It cannot refuse over it -- see `weights.inert_confusables`.
            confusable_classes=tuple(camera.weapons.confusable_classes),
        ),
    )


def _build_canteen(group: WorkerGroup, cameras: dict[str, CameraConfig]) -> dict[str, object]:
    """One InsightFace instance and one gallery for the whole group — **and a YOLO each.**

    Measured: InsightFace costs ~700 MB of VRAM per instance, which on this 4 GB card is
    the difference between running and not running. The legacy created up to FIVE of them
    in one process (audit H-12).

    The canteen cameras now carry a `PersonDetector` as well, because identity is bound
    per TRACK and a track is what YOLO+ByteTrack produces. That is a real cost on a card
    this size — the weights are only ~6 MB each, but they are not free — and it is
    `qorgan plan-workers` (Task 11) that sizes the fleet for it, not a guess here.
    """
    canteen = {n: c for n, c in cameras.items() if isinstance(c, CanteenCamera)}
    if not canteen:
        return {}

    from qorgan.events.store import ensure_cameras
    from qorgan.faces.gallery import GalleryCache
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.gpu import require_gpu
    from qorgan.models.person import PersonDetector
    from qorgan.worker.canteen import CanteenPipeline, build_identity

    # Refuse to run 40x too slow on the CPU rather than doing it silently. The canteen
    # cameras now load YOLO as well as InsightFace, and both fail silently to the CPU.
    require_gpu()

    camera_ids = ensure_cameras(cameras)
    first = next(iter(canteen.values()))
    model = first.canteen.face_model

    recognizer = FaceRecognizer.shared(model)
    gallery = GalleryCache(model.model_name, model.model_version)
    sessions = _sessions_for(cameras, camera_ids, first)

    return {
        name: CanteenPipeline(
            camera=camera,
            camera_id=camera_ids[name],
            # One detector per CAMERA, never per group: Ultralytics keeps its tracker
            # state on the model object, so two cameras sharing one would hand each
            # other's children the same track ids -- and a track id is now an identity.
            person=PersonDetector(camera.yolo, name, device=group.device),
            identity=build_identity(camera, recognizer, gallery),
            sessions=sessions,
        )
        for name, camera in canteen.items()
    }


def _sessions_for(
    cameras: dict[str, CameraConfig], camera_ids: dict[str, int], first: CanteenCamera
):
    """The one session machine the whole canteen shares. Roles, never camera names."""
    from qorgan.canteen.sessions import SessionManager

    entry_id = _role_id(cameras, camera_ids, CameraRole.CANTEEN_ENTRY)
    exit_id = _role_id(cameras, camera_ids, CameraRole.CANTEEN_EXIT)

    return SessionManager(
        first.canteen.session,
        first.canteen.meal_outcome,
        entry_camera_id=entry_id or camera_ids[first.name],
        exit_camera_id=exit_id,
    )


def _role_id(cameras: dict[str, CameraConfig], ids: dict[str, int], role: CameraRole) -> int | None:
    """Find a camera by ROLE, not by a hardcoded name.

    The legacy had camera names and IP addresses written into the logic in a dozen
    places — including one staff member's personal name (audit M-27).
    """
    for name, camera in cameras.items():
        if camera.role is role:
            return ids.get(name)
    return None


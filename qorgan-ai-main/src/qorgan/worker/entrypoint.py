"""What a worker process runs. Spawned by the supervisor, one per worker group.

A worker knows nothing about the web layer (rule R3). It reads config, opens its
cameras, publishes previews, writes events, and beats. Which cameras it covers comes
from `config/workers.yaml` and never from which page an operator has open — in the
legacy, nobody looking at the "Stairs" tab meant the stairs were not monitored at all.

**What each camera type BUILDS lives in `worker/builders.py`.** It moved there when the
fourth type arrived and this file passed 500 lines (R1: split, never loosen). The line is
process against payload: signals, the heartbeat, the shutdown budget and the loop that
keeps this alive are here; which models a group loads is there.
"""

from __future__ import annotations

import contextlib
import signal
import threading

from qorgan.config.camera import CameraConfig
from qorgan.config.loader import load_cameras, load_workers
from qorgan.config.workers import WorkerGroup
from qorgan.db.engine import reset_engine
from qorgan.enums import WorkerState
from qorgan.logging_setup import get_logger, setup_logging
from qorgan.preview import PreviewPublisher
from qorgan.settings import get_settings
from qorgan.supervisor.heartbeat import Heartbeat

# The grace the PARENT allows this process to shut down in. Imported rather than copied:
# the worker's stop budget only means anything against the number the supervisor actually
# enforces, and two copies of it is how a worker comes to believe it fits while being
# killed for not fitting. (No import cycle: `supervisor.managed` imports this module
# lazily, inside `spawn_worker_process`.)
from qorgan.supervisor.managed import TERMINATE_GRACE_SECONDS
from qorgan.worker.builders import build_all
from qorgan.worker.camera_loop import CameraLoop, stop_all, worst_case_stop_seconds

logger = get_logger(__name__)


def run_group(group_name: str) -> None:
    """Process entry point. Must stay importable and picklable: Windows has no fork."""
    setup_logging(f"worker-{group_name}")
    # The engine was created in the parent before the spawn; its connections do not
    # belong to us. Start clean.
    reset_engine()

    stop = threading.Event()
    _install_signal_handlers(stop)

    try:
        group, cameras = _resolve_group(group_name)
    except Exception:
        logger.exception("worker cannot start", extra={"group": group_name})
        raise

    heartbeat = Heartbeat(group_name, interval_seconds=5.0)
    try:
        with heartbeat:
            _serve(group, cameras, heartbeat, stop)
    except Exception as exc:
        logger.exception("worker crashed", extra={"group": group_name})
        heartbeat.record(error=f"{type(exc).__name__}: {exc}")
        heartbeat.stop(WorkerState.CRASHED)
        raise


def _resolve_group(group_name: str) -> tuple[WorkerGroup, dict[str, CameraConfig]]:
    cameras = load_cameras()
    workers = load_workers(cameras)
    group = next((g for g in workers.groups if g.name == group_name), None)
    if group is None:
        raise ValueError(f"no worker group named {group_name!r} in workers.yaml")
    return group, {name: cameras[name] for name in group.cameras}


def _serve(
    group: WorkerGroup,
    cameras: dict[str, CameraConfig],
    heartbeat: Heartbeat,
    stop: threading.Event,
) -> None:
    """Run this group's cameras until told to stop.

    One process, N cameras — see config/workers.yaml for why. The pose model is loaded
    once and shared (it carries no state); the person detectors are per-camera, because
    Ultralytics keeps its TRACKER state on the model and two cameras sharing one would
    hand each other's children the same track ids.
    """
    cameras = _switched_on(cameras)
    _check_shutdown_budget(group, cameras)

    settings = get_settings()
    publisher = PreviewPublisher(settings.preview_address)
    pipelines = build_all(group, cameras)

    loops = [
        CameraLoop(
            camera,
            publisher,
            on_frame=pipelines[name].on_frame if name in pipelines else _no_detection,
        ).start()
        for name, camera in cameras.items()
    ]
    logger.info(
        "worker running",
        extra={"group": group.name, "cameras": list(cameras), "detecting": list(pipelines)},
    )

    try:
        while not stop.wait(1.0):
            heartbeat.record(frames=sum(loop.frames_processed for loop in loops))
            if _every_source_finished(loops):
                logger.info(
                    "every frame source in this group has finished",
                    extra={"group": group.name, "cameras": list(cameras)},
                )
                break
    finally:
        logger.info("worker stopping", extra={"group": group.name})
        _shut_down(group, loops, pipelines, publisher)


def _shut_down(
    group: WorkerGroup,
    loops: list[CameraLoop],
    pipelines: dict[str, object],
    publisher: PreviewPublisher,
) -> None:
    """Cameras together, then the pipelines, then the bus.

    The cameras go together because one after another does not fit the supervisor's grace
    -- `stop_all` carries the arithmetic. They go FIRST because a pipeline should stop
    being fed before it is stopped.

    The pipelines are still stopped one after another. Each one's thread blocks 0.5 s at
    most (`queue.get(timeout=0.5)`) plus one validation job, so four of them are a second
    or two -- but their `join(timeout=5.0)` caps allow twenty, and that cap has never been
    measured against a wedged validator on this machine. It is left alone and said out
    loud rather than changed on a guess: this shutdown's measured defect was the cameras.
    """
    unstopped = stop_all(loops)
    if unstopped:
        logger.error(
            "shutdown is late: these cameras never answered",
            extra={
                "group": group.name,
                # Names, not URLs. safe_url is per-camera and already logged by the loop
                # that failed; repeating a URL here would be a second place to get R4
                # wrong.
                "cameras": unstopped,
                "grace_seconds": TERMINATE_GRACE_SECONDS,
                "consequence": (
                    "the supervisor may now log 'worker ignored terminate' and kill this "
                    "process. That line is about this worker; the cause is the cameras "
                    "named here."
                ),
            },
        )
    for pipeline in pipelines.values():
        pipeline.stop()
    publisher.close()


def _check_shutdown_budget(group: WorkerGroup, cameras: dict[str, CameraConfig]) -> None:
    """Say at STARTUP whether this group can stop inside the grace its parent allows.

    Checked here, in the worker, rather than left to a test, because both halves are
    per-group and one of them comes from YAML: a school that raises
    `rtsp.read_timeout_seconds` on one camera, or an engineer who adds a fifth camera to a
    group, moves this number. The consequence -- a killed process, no clean heartbeat, no
    `publisher.close()` -- would otherwise surface months later as an unexplained "worker
    ignored terminate", which is a true log line pointing at the wrong thing.

    A warning, not a refusal: taking a group of cameras off the air over a timing budget
    would be a worse outcome than a slow shutdown.
    """
    budget = worst_case_stop_seconds(cameras.values())
    if budget < TERMINATE_GRACE_SECONDS:
        return
    logger.warning(
        "this group cannot be sure of stopping inside the supervisor's grace",
        extra={
            "group": group.name,
            "worst_case_stop_seconds": round(budget, 1),
            "grace_seconds": TERMINATE_GRACE_SECONDS,
            "consequence": (
                "if a camera stops answering, this process will be killed rather than "
                "stopped: no final heartbeat and no clean preview shutdown. Lower "
                "rtsp.open_timeout_seconds / rtsp.read_timeout_seconds for this group's "
                "cameras."
            ),
        },
    )


def _every_source_finished(loops: list[CameraLoop]) -> bool:
    """Has every camera in this group run out of frames for good?

    Only a finite source -- a recorded clip with `at_end: stop` -- can ever say yes, so on
    the real fleet this is always False and the worker runs until it is told to stop. It
    exists because a run over the school's own recordings has an END, and a worker that
    sat spinning on an exhausted file afterwards would make "the run is done" something a
    person has to guess at from a log.

    `loops` empty means a group with no cameras at all, which is a configuration problem
    and must not read as "finished" -- `all([])` is True, so it is excluded explicitly.
    """
    return bool(loops) and all(loop.finished for loop in loops)


def _switched_on(cameras: dict[str, CameraConfig]) -> dict[str, CameraConfig]:
    """Only the cameras the school has left switched on.

    **A camera with `enabled: false` must never open its stream**, and until this existed
    it did. `enabled` is the one setting that reads like an off switch; `load_workers`
    consults it only to REFUSE an *enabled* camera that no group claims, and nothing
    filtered. So a school that switched a camera off still had it captured, analysed and
    previewed -- and because `events/store.py` writes the same flag to `cameras.enabled`,
    the dashboard displayed "disabled" beside a camera that was running. Both layers said
    off while the stream was open.

    Called at the TOP of `_serve`, before anything is built, so the guarantee covers the
    CameraLoop that opens RTSP, the detection pipelines and the preview publisher alike --
    one place to get right rather than three to forget. See `tests/test_disabled_camera.py`.
    """
    return {name: camera for name, camera in cameras.items() if camera.enabled}

def _no_detection(_camera: CameraConfig, _frame) -> str:
    """A camera in a group with no pipeline of its own. It still publishes a preview."""
    return "ok"


def _install_signal_handlers(stop: threading.Event) -> None:
    def handle(signum: int, _frame: object) -> None:
        logger.info("worker received signal", extra={"signal": signum})
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError):  # not the main thread; nothing to install
            signal.signal(sig, handle)

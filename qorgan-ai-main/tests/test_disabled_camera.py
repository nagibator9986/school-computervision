"""A camera the school has switched off must never open its stream.

**This is worse than a leaked password, and it was silent.** `enabled: false` on a camera
read like an off switch — it is the only thing in the configuration that looks like one —
and it did not stop anything. `load_workers` consults the flag exactly once, to REFUSE an
*enabled* camera that no worker group claims ("a camera nobody runs is a camera nobody is
watching"). It never filtered. `_resolve_group` then handed `_serve` every camera named in
`workers.yaml`, and `_serve` built a `CameraLoop` for each one with no check at all.

So a school that switched a camera off in the one place that offers to switch it off still
had it captured, analysed, and previewed. Nobody would find out: the flag is written to
`cameras.enabled` on every worker start (`events/store.py:160`), so the database agrees
with the YAML and both are wrong together — the dashboard would show "disabled" beside a
camera that is running.

That is this project's signature defect (true in one layer, silently false in the next)
pointed at the thing the whole system is careful about: whether children are being watched.

**The fix is deliberately one behaviour, in one place.** A disabled camera is dropped at
the top of `_serve` — the only function that builds the things which do the watching: the
`CameraLoop` that opens RTSP, the detection pipelines, the preview publisher. Filtering
there rather than where the group is resolved means the guarantee holds for every caller,
instead of depending on somebody having filtered earlier. Nothing else in worker startup is
touched — not the heartbeat, not the ordering — because a wide change to the code that
decides what watches children is not one to make while proving a narrow one.

Deliberately NOT changed here: `load_workers` still refuses an *enabled* unassigned
camera, which is the opposite guard and still right; and `cameras.enabled` is still written
by `events/store.py` and read by nobody, which stays a separate question.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.config.workers import WorkerGroup
from qorgan.settings import Settings
from qorgan.worker import builders, entrypoint


def _camera(name: str, *, enabled: bool) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": name,
            "display_name": name,
            "enabled": enabled,
            "rtsp": {"host": "10.0.0.1"},
        }
    )


def _group(*names: str) -> WorkerGroup:
    return WorkerGroup(name="bullying_hall", device="cpu", cameras=list(names))


def _loops_started(cameras: dict[str, CameraConfig]) -> list[str]:
    """Which cameras `_serve` actually builds a CameraLoop for.

    Patched at the name each module calls, not at the definition: this asserts what the
    worker really constructs, which is the thing that opens an RTSP stream. `CameraLoop`
    and `PreviewPublisher` are `entrypoint`'s; the per-type builders moved to
    `worker/builders.py` when the weapons pipeline pushed `entrypoint` past 500 lines,
    and `build_all` resolves them there at call time -- so that is where they are patched.
    """
    _Recorded.built.clear()
    stop = _ImmediateStop()
    with (
        patch.object(entrypoint, "CameraLoop", _Recorded),
        patch.object(entrypoint, "PreviewPublisher", _FakePublisher),
        patch.object(builders, "_build_pipelines", lambda *_a, **_k: {}),
        patch.object(builders, "_build_canteen", lambda *_a, **_k: {}),
    ):
        entrypoint._serve(_group(*cameras), cameras, _FakeHeartbeat(), stop)
    return list(_Recorded.built)


class _Recorded:
    """Stands in for `CameraLoop`, and records every camera one was built for. Building a
    CameraLoop is what opens the RTSP stream, so this list IS "who is being watched"."""

    built: ClassVar[list[str]] = []

    def __init__(self, camera, _publisher, **_kwargs) -> None:
        _Recorded.built.append(camera.name)
        # The real CameraLoop exposes the camera it owns, and `stop_all` reads it to name
        # the cameras that would not stop. A fake without it fails this file for a reason
        # that has nothing to do with what this file is about.
        self.camera = camera
        self.frames_processed = 0

    def start(self) -> _Recorded:
        return self

    def stop(self) -> bool:
        """True means "this camera really stopped", which is what `CameraLoop.stop`
        returns now. A fake returning None would have `_serve` report a shutdown failure
        that did not happen."""
        return True


class _FakePublisher:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        return None


class _FakeHeartbeat:
    def record(self, **_kwargs) -> None:
        return None


class _ImmediateStop:
    """Ends `_serve`'s loop on the first tick, so the test does not sleep."""

    def wait(self, _timeout: float) -> bool:
        return True


# -- the requirement ---------------------------------------------------------


def test_a_disabled_camera_never_becomes_a_camera_loop(settings: Settings) -> None:
    """**The test the whole branch exists for.** A `CameraLoop` is the thing that opens
    the RTSP stream, so "no loop was built" is the same statement as "this camera was
    never watched"."""
    cameras = {
        "hall_left": _camera("hall_left", enabled=True),
        "hall_right": _camera("hall_right", enabled=False),
    }

    built = _loops_started(cameras)

    assert "hall_right" not in built, (
        "a camera the school switched OFF was opened and analysed anyway"
    )


def test_an_enabled_camera_still_runs(settings: Settings) -> None:
    """The other half, and not a formality: a filter that stops everything would satisfy
    the test above perfectly. Silence is the failure mode this system is least able to
    notice."""
    cameras = {
        "hall_left": _camera("hall_left", enabled=True),
        "hall_right": _camera("hall_right", enabled=False),
    }

    assert "hall_left" in _loops_started(cameras)


def test_a_group_of_only_disabled_cameras_starts_nothing(settings: Settings) -> None:
    cameras = {"hall_right": _camera("hall_right", enabled=False)}

    assert _loops_started(cameras) == []


# -- the same decision, one layer earlier ------------------------------------


def test_a_disabled_camera_never_reaches_the_pipeline_builder(settings: Settings) -> None:
    """The filter runs before anything is built, not just before the loops.

    A detection pipeline is the other thing that consumes a camera's frames, and it loads
    the models. Asserting the pipeline builder never sees the disabled camera pins that the
    filter sits at the TOP of `_serve` -- if it slid down to the loop comprehension, this
    goes red while the loop tests stay green.
    """
    cameras = {
        "hall_left": _camera("hall_left", enabled=True),
        "hall_right": _camera("hall_right", enabled=False),
    }
    seen: list[str] = []

    # `*_rest` absorbs the pose loader `_serve` now threads through the builders. The
    # loader is LAZY, so patching the builders out still means no weights are loaded here
    # -- an eager one made this model-free test pull in a real pose model.
    def _record(_group, given: dict[str, CameraConfig], *_rest: object) -> dict:
        seen.extend(given)
        return {}

    with (
        patch.object(entrypoint, "CameraLoop", _Recorded),
        patch.object(entrypoint, "PreviewPublisher", _FakePublisher),
        patch.object(builders, "_build_pipelines", _record),
        patch.object(builders, "_build_canteen", lambda *_a, **_k: {}),
        patch.object(builders, "_build_classroom", _record),
    ):
        entrypoint._serve(_group(*cameras), cameras, _FakeHeartbeat(), _ImmediateStop())

    assert "hall_right" not in seen, (
        "a disabled camera was handed to the pipeline builder -- the filter is too late"
    )

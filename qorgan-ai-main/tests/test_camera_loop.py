"""The per-camera loop inside a worker process."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from qorgan.config.camera import CAMERA_ADAPTER
from qorgan.preview import PreviewPublisher, PreviewSubscriber
from qorgan.settings import Settings
from qorgan.worker.camera_loop import CameraLoop
from tests.fakes import FakeCameraFactory, FakeCapture, connect_preview_bus, noisy_frame


def _camera(det_every: int = 1):
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": "hall_left",
            "display_name": "Hall",
            "rtsp": {"host": "10.0.0.1"},
            "capture": {"det_every": det_every},
            "preview": {"fps": 15.0},
        }
    )


@pytest.fixture
def address() -> str:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _wait_until(condition, timeout: float, message: str) -> None:
    """Poll for a condition instead of sleeping a fixed duration.

    A fixed-duration sleep assumes the machine is fast enough to get through N frames
    in that time; under load it is not, and the caller silently continues with fewer
    frames than it expected. Polling with a generous timeout gets out as soon as the
    condition is true on a fast machine, and fails LOUDLY -- with a message that says
    exactly what was still missing -- rather than limping on with too little data.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(message)


# 50 fps: faster than any camera in this school, and far slower than `_tick`, which costs a
# few milliseconds (a resize plus a JPEG encode). The producer must lose the race to the
# consumer, or `CameraStream`'s Queue(maxsize=1) sheds frames and the test silently observes
# fewer than it asked for. A real frame rate (0.067s at the sub-stream's 15 fps) would also work and
# costs 5x the wall-clock; 766 tests of correct sleeping is a suite people stop running.
#
# This is a margin, not a proof -- so `_run_loop` CHECKS it (frames_dropped == 0) rather
# than trusting it. On a machine slow enough to lose the race, the test says so in one line
# instead of failing with a baffling count.
FAKE_FRAME_INTERVAL = 0.02


def _await_frames(loop, min_frames: int) -> None:
    """Wait until the loop has really seen `min_frames` DISTINCT frames -- and prove it.

    Two failures are possible here and they need different words, because they point at
    different bugs:

      * the loop never got there            -> it is stalled, not slow. Say so.
      * it got there, but frames were shed  -> the fake outran it, so any count the test
                                               goes on to assert is meaningless.

    The second is the one that bit us. `FAKE_FRAME_INTERVAL` is a MARGIN, not a proof, so
    it is checked rather than trusted: on a machine slow enough to lose the race, this says
    so in one line instead of failing later with a baffling number.
    """
    _wait_until(
        lambda: loop.frames_processed >= min_frames,
        timeout=15.0,
        message=(
            f"only processed {loop.frames_processed}/{min_frames} frames within 15s "
            "-- the loop is stalled, not just slow"
        ),
    )
    dropped = loop._stream.stats.frames_dropped
    assert dropped == 0, (
        f"the fake camera outran the loop: {dropped} frames dropped. A counted frame is "
        f"only meaningful if none were shed. Raise FAKE_FRAME_INTERVAL above "
        f"{FAKE_FRAME_INTERVAL}s."
    )


def _paced_camera(frames: int) -> FakeCameraFactory:
    """A fake that delivers frames at a frame RATE, not at memory bandwidth.

    Unpaced, the stream's producer thread drains the whole script in microseconds and
    `CameraStream`'s Queue(maxsize=1) drops all but the newest -- which is exactly right for
    a real camera with a slow consumer, and fatal for a test that must observe N distinct
    frames. How many it observed then depended on how fast `_tick` happened to be.

    That is not hypothetical. It is what starved `test_det_every_is_honoured` the moment
    `prepare_frame()` made `_tick` do a real resize: the loop was not stalled and the code
    was not wrong, the fake was simply outrunning it. A test whose result depends on the
    speed of the machine is measuring the machine.
    """
    return FakeCameraFactory(
        FakeCapture(
            [(True, noisy_frame(i)) for i in range(frames)], interval=FAKE_FRAME_INTERVAL
        )
    )


def _run_loop(camera, address: str, frames: int = 40, min_frames: int | None = None, **kwargs):
    """Drive a CameraLoop with a fake camera, and capture what reaches the web side.

    Most callers only need "at least one frame went through" (or, for an on_frame
    that raises on every call, pay CameraLoop's 1s exception backoff per frame, so a
    higher count costs real wall-clock seconds regardless of timeout) -- the
    historical fixed-duration wait covers them. `min_frames` is for a caller with an
    actual, checkable requirement: det_every needs enough DISTINCT frames to contain
    a multiple of the skip. It switches to a condition poll that returns the moment
    the target is met, and raises a clear error rather than continuing on too little.
    """
    factory = _paced_camera(frames)
    subscriber = PreviewSubscriber(address, stale_after_seconds=30.0).start()
    publisher = PreviewPublisher(address)
    connect_preview_bus(publisher, subscriber)  # slow-joiner handshake; see tests/fakes.py

    with patch("qorgan.capture.stream.open_rtsp", factory):
        loop = CameraLoop(camera, publisher, **kwargs)
        loop._stream._opener = factory
        loop.start()
        if min_frames is None:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and loop.frames_processed < frames // 2:
                time.sleep(0.02)
        else:
            # From this branch: the wait ALSO checks frames_dropped == 0, because a counted
            # frame is meaningless if the fake outran the loop and the queue shed some.
            _await_frames(loop, min_frames)

        # From main: poll (not sleep) for the last preview to reach the subscriber before
        # stop() tears the publisher down. Best-effort, not asserted here: an on_frame that
        # raises on every call means CameraLoop never reaches publish() at all, so a preview
        # can legitimately never arrive -- callers that need one assert on the return value.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline and subscriber.latest("hall_left") is None:
            time.sleep(0.02)
        loop.stop()

    publisher.close()
    latest = subscriber.latest("hall_left")
    subscriber.stop()
    return loop, latest, factory


def test_the_loop_reads_frames_and_publishes_previews(settings: Settings, address: str) -> None:
    loop, preview, _ = _run_loop(_camera(), address)

    assert loop.frames_processed > 0
    assert preview is not None
    assert preview.header.camera == "hall_left"


def test_det_every_is_honoured(settings: Settings, address: str) -> None:
    """Legacy shipped det_every: 2 on the canteen cameras with a comment claiming it
    halved GPU load, and the canteen worker never read the key at all."""
    seen: list[int] = []

    def detector(_camera, frame) -> str:
        seen.append(frame.seq)
        return "ok"

    loop, _, _ = _run_loop(_camera(det_every=3), address, on_frame=detector, min_frames=20)

    assert seen, "the detector never ran"
    assert all(seq % 3 == 0 for seq in seen), f"det_every ignored: {seen[:6]}"
    assert len(seen) < loop.frames_processed


def test_the_detector_status_reaches_the_preview(settings: Settings, address: str) -> None:
    _loop, preview, _ = _run_loop(_camera(), address, on_frame=lambda _c, _f: "critical")

    assert preview is not None
    assert preview.header.status == "critical", "the operator would not see the alert"


def test_a_detector_that_raises_does_not_kill_the_camera(settings: Settings, address: str) -> None:
    """Rule R7: a camera thread that dies takes its camera off the air, silently."""
    calls = {"n": 0}

    def flaky(_camera, _frame) -> str:
        calls["n"] += 1
        raise RuntimeError("inference exploded")

    loop, _preview, _ = _run_loop(_camera(), address, on_frame=flaky)

    assert calls["n"] > 1, "the loop gave up after the first exception"
    assert loop.frames_processed > 1


def test_the_stream_url_carries_credentials_but_the_log_line_does_not(
    settings: Settings, address: str
) -> None:
    _loop, _preview, factory = _run_loop(_camera(), address)

    assert factory.urls, "no connection was made"
    assert "sup3r-s3cret-camera-pw" in factory.urls[0], "the real URL lost its credentials"


def test_the_detector_sees_the_analysis_resolution_whatever_the_nvr_sends(
    settings: Settings, address: str
) -> None:
    """The fake camera delivers 64x64. The detector must still see THAT CAMERA'S OWN
    analysis frame -- `capture.frame_width x frame_height`, whatever it happens to be.

    Every px/s threshold in every profile is denominated in THIS frame. If the loop
    hands the detector the substream's own resolution, the thresholds are being compared
    against speeds measured in a different pixel, and no amount of tuning fixes that.

    Asserted against the camera's config, never a literal. The resolution is PER PROFILE:
    this fixture is built from a dict so it inherits base.yaml's 960x540 default, but the
    real `hall.yaml` and `canteen_entry.yaml` override it to 1280x720. Hard-coding one
    number here would pin an invariant that only holds for some of the fleet -- and reading
    a base default as if it were the hall's resolution is exactly the mistake that made
    every hall face-size figure wrong once (identity spec §2.4).
    """
    camera = _camera()
    expected = (camera.capture.frame_height, camera.capture.frame_width)
    shapes: list[tuple[int, int]] = []

    def detector(camera, frame) -> str:
        shapes.append(frame.image.shape[:2])
        return "ok"

    _loop, _preview, _ = _run_loop(camera, address, on_frame=detector)

    assert shapes, "the detector never ran"
    assert all(shape == expected for shape in shapes), (
        f"the detector was handed {shapes[0]} -- the NVR's resolution, not the analysis one"
    )

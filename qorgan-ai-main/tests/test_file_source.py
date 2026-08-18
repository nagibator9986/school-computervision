"""A recorded clip driven through the production path, as a camera would be.

RTSP has never been opened from any machine this project runs on. The seam that was meant
to make that survivable -- `CaptureOpener` in `capture/stream.py`, "injectable so the tests
can drive a fake camera without an RTSP server" -- existed in `CameraStream` and dead-ended
in `CameraLoop`, which hardcoded the RTSP URL factory and never forwarded `opener`. These
tests are about the other half: a camera whose frames come off disk, running the same
`prepare_frame`, the same detector callback, the same preview publisher, the same
everything.

Every clip written here is SYNTHETIC NOISE this test generates. Nothing under `eval/`,
`bullying_camera/`, `canteen/` or `media/` is opened, read, or decoded: those are
recordings of children.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from qorgan.capture.clip import PacedClip
from qorgan.capture.source import frame_source
from qorgan.capture.stream import CameraStream, open_rtsp
from qorgan.config.camera import CAMERA_ADAPTER
from qorgan.config.common import RtspSettings, fps_agrees
from qorgan.config.loader import ConfigError
from qorgan.enums import ClipEnd
from qorgan.preview import PreviewPublisher, PreviewSubscriber
from qorgan.settings import Settings
from qorgan.worker.camera_loop import CameraLoop
from qorgan.worker.entrypoint import _every_source_finished
from tests.fakes import FakeCameraFactory, connect_preview_bus, noisy_frame

PASSWORD = "sup3r-s3cret-camera-pw"  # what the `settings` fixture gives every camera
CLIP_FPS = 25.0


def _camera(source: dict | None = None):
    data: dict = {
        "camera_type": "bullying",
        "role": "main_hall",
        "name": "hall_left",
        "display_name": "Hall",
        "rtsp": {"host": "10.0.0.1"},
        "preview": {"fps": 15.0},
    }
    if source is not None:
        data["source"] = source
    return CAMERA_ADAPTER.validate_python(data)


def _write_clip(path: Path, *, frames: int, fps: float = CLIP_FPS) -> Path:
    """A real video file, made of noise this function generates.

    Real because the whole claim is that `cv2.VideoCapture` opening a file behaves like it
    opening a stream, and a fake capture cannot test that claim -- it is the claim.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened(), (
        "OpenCV cannot write mp4v here, so this test cannot make a clip to read back. "
        "That is an environment problem, not a pass."
    )
    for index in range(frames):
        writer.write(noisy_frame(index))
    writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


@pytest.fixture
def address() -> str:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _wait_until(condition, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(message)


# -- which source a camera has, and how it says so ---------------------------


def test_a_camera_with_no_source_block_reads_the_camera(settings: Settings) -> None:
    """The fleet ships without `source:`, and must go on opening RTSP exactly as before."""
    source = frame_source(_camera())

    assert source.opener is open_rtsp
    assert source.reconnects is True, "a camera that is off the network must be retried"
    assert source.url_factory() == f"rtsp://admin:{PASSWORD}@10.0.0.1:554/Streaming/Channels/102"


def test_the_url_still_carries_the_password_and_nothing_printable_does(
    settings: Settings,
) -> None:
    """R4, unweakened by the new seam.

    The URL is built by a FACTORY on each connect and is never stored on anything a repr
    could reach -- `capture/stream.py` says so, and routing the factory through a new
    dataclass is exactly the change that could have quietly broken it. The legacy printed
    this password into every log file and drew it onto debug JPEGs an unauthenticated web
    UI served (audit C-02).
    """
    source = frame_source(_camera())

    assert PASSWORD in source.url_factory(), "the real URL lost its credentials"
    assert PASSWORD not in source.where, "the loggable description carries the password"
    assert PASSWORD not in repr(source), "a repr of the source would leak the password"
    assert source.where == "rtsp://10.0.0.1:554/Streaming/Channels/102"


def test_a_file_backed_camera_names_the_clip_and_not_a_url_it_is_not_reading(
    settings: Settings,
) -> None:
    """The log line is the operator's only clue about where the frames came from. Printing
    `rtsp://10.0.0.1/...` while reading a file off disk would point them at the network."""
    source = frame_source(_camera({"path": "clips/hall.mp4", "at_end": "stop"}))

    assert source.url_factory() == str(Path("clips/hall.mp4"))
    assert "clips/hall.mp4" in source.where
    assert "stop" in source.where, "the end-of-clip behaviour is invisible to whoever reads the log"
    assert "rtsp://" not in source.where
    assert PASSWORD not in source.where


# -- the end of the clip is configuration, not a guess ------------------------


def test_looping_is_the_default_and_stopping_is_available() -> None:
    """Both answers are defensible -- a demonstration wants the hall to keep moving, a run
    over the corpus wants the clip counted once -- so it is a key and not a habit."""
    assert _camera({"path": "a.mp4"}).source.at_end is ClipEnd.LOOP
    assert _camera({"path": "a.mp4", "at_end": "stop"}).source.at_end is ClipEnd.STOP


def test_a_clip_told_to_loop_is_a_source_that_comes_back_and_one_told_to_stop_is_not(
    settings: Settings,
) -> None:
    looping = frame_source(_camera({"path": "a.mp4", "at_end": "loop"}))
    stopping = frame_source(_camera({"path": "a.mp4", "at_end": "stop"}))

    assert looping.reconnects is True
    assert stopping.reconnects is False


def test_a_typo_under_source_is_a_startup_error_and_not_a_silent_default() -> None:
    """R10. `at_the_end:` looks like it works; a config that accepted it would loop a clip
    somebody meant to run once, and the run would never end."""
    with pytest.raises(ValueError, match="at_the_end"):
        _camera({"path": "a.mp4", "at_the_end": "stop"})


def test_an_empty_path_is_refused() -> None:
    with pytest.raises(ValueError, match="path"):
        _camera({"path": ""})


def test_the_shipped_fleet_reads_cameras_and_not_files(settings: Settings) -> None:
    """The demonstration source is opt-in, per camera, and nothing on site has opted in.
    A file left switched on in a config file is a school watching a recording."""
    from qorgan.config.loader import load_cameras
    from tests.conftest import CONFIG_DIR

    cameras = load_cameras(CONFIG_DIR)

    assert cameras, "no cameras loaded; this test would pass vacuously"
    assert [name for name, camera in cameras.items() if camera.source is not None] == []


def test_a_config_error_still_names_the_file(settings: Settings, tmp_path: Path) -> None:
    """Sanity on the new block's failure path: a bad `source:` in a real camera file is
    reported the way every other bad key is, by filename."""
    import shutil

    from qorgan.config.loader import load_cameras
    from tests.conftest import CONFIG_DIR

    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    camera = directory / "cameras" / "hall_left.yaml"
    camera.write_text(
        camera.read_text(encoding="utf-8") + '\nsource:\n  path: "a.mp4"\n  at_end: "rewind"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="hall_left.yaml"):
        load_cameras(directory)


# -- the stream, on a real file ----------------------------------------------


def test_the_reader_does_not_drain_the_file_at_memory_speed(
    settings: Settings, tmp_path: Path
) -> None:
    """The claim, measured on the wall clock rather than on an injected one.

    Asserted as a LOWER bound only. Load on the machine can only make the elapsed time
    larger, so this direction never goes flaky -- and it is the direction that matters:
    the defect being guarded against is a whole clip arriving in a few milliseconds, with
    every px/s the detector computes inflated by however fast this CPU decodes.
    """
    frames = 12
    clip = _write_clip(tmp_path / "hall.mp4", frames=frames)
    camera = _camera({"path": str(clip), "at_end": "stop"})
    source = frame_source(camera)

    started = time.monotonic()
    # `CaptureOpener` takes the camera's RtspSettings alongside the URL -- every opener
    # does, so that `open_rtsp` can put the timeouts in as constructor parameters. A file
    # opener ignores them; it is handed them anyway so there is one opener shape.
    capture = source.opener(source.url_factory(), camera.rtsp)
    try:
        assert isinstance(capture, PacedClip)
        assert capture.fps == pytest.approx(CLIP_FPS, abs=0.5)
        read = sum(1 for _ in iter(lambda: capture.read()[0], False))
    finally:
        capture.release()
    elapsed = time.monotonic() - started

    assert read == frames, f"read {read} of {frames} frames back out of the clip"
    floor = (frames - 1) / CLIP_FPS * 0.95
    assert elapsed >= floor, (
        f"{frames} frames came out in {elapsed:.3f}s, under the {floor:.3f}s the recording "
        "itself lasts -- the file was drained at decode speed, so every timestamp the "
        "detector sees is compressed and every speed it measures is inflated"
    )


def test_a_clip_that_runs_out_finishes_the_stream_instead_of_reconnecting_forever(
    settings: Settings, tmp_path: Path
) -> None:
    """`_run` reconnects for ever, which is right for a camera and wrong for a file.

    Left alone it would reopen the clip every two seconds and replay the same children,
    logging a disconnection each time -- an accidental loop nobody chose, wearing the
    disguise of a network fault.
    """
    clip = _write_clip(tmp_path / "hall.mp4", frames=6)
    camera = _camera({"path": str(clip), "at_end": "stop"})
    source = frame_source(camera)
    opens: list[str] = []

    def counting_opener(target: str, rtsp: RtspSettings):
        opens.append(target)
        return source.opener(target, rtsp)

    # A short reconnect delay so a stream that WRONGLY retries has time to prove it inside
    # this test. It travels inside `RtspSettings` because `CameraStream` now takes the
    # whole block rather than the delay lifted out of it -- same number, same effect, one
    # place. `model_copy` rather than a literal: everything else about the camera's timing
    # budget stays exactly what the config says.
    impatient = camera.rtsp.model_copy(update={"reconnect_delay_seconds": 0.05})
    stream = CameraStream(
        "hall_left",
        url_factory=source.url_factory,
        rtsp=impatient,
        opener=counting_opener,
        reconnects=source.reconnects,
    ).start()
    try:
        _wait_until(
            lambda: stream.stats.finished,
            timeout=10.0,
            message="the clip ended and the stream never reported itself finished",
        )
        # Give a reconnecting stream time to prove itself wrong.
        time.sleep(0.3)
        assert len(opens) == 1, f"the clip was reopened {len(opens)} times"
        assert stream.stats.reconnects == 0
        assert stream.stats.frames_published == 6
    finally:
        stream.stop()


def test_a_looping_camera_never_reports_itself_finished(
    settings: Settings, tmp_path: Path
) -> None:
    clip = _write_clip(tmp_path / "hall.mp4", frames=4)
    camera = _camera({"path": str(clip), "at_end": "loop"})
    source = frame_source(camera)

    stream = CameraStream(
        "hall_left",
        url_factory=source.url_factory,
        rtsp=camera.rtsp,
        opener=source.opener,
        reconnects=source.reconnects,
    ).start()
    try:
        _wait_until(
            lambda: stream.stats.frames_published > 4,
            timeout=10.0,
            message="the clip never started over",
        )
        assert stream.finished is False, "a looping source called itself finished"
        assert stream.stats.reconnects == 0, "it looped by RECONNECTING, which is a 2s gap"
    finally:
        stream.stop()


# -- the whole loop, on a real file ------------------------------------------


def _run_clip_through_the_loop(camera, address: str, *, min_frames: int):
    subscriber = PreviewSubscriber(address, stale_after_seconds=30.0).start()
    publisher = PreviewPublisher(address)
    connect_preview_bus(publisher, subscriber)

    seen: list[tuple[int, int]] = []

    def detector(_camera, frame) -> str:
        seen.append(frame.image.shape[:2])
        return "ok"

    loop = CameraLoop(camera, publisher, on_frame=detector).start()
    try:
        _wait_until(
            lambda: loop.frames_processed >= min_frames,
            timeout=30.0,
            message=f"only {loop.frames_processed}/{min_frames} frames reached the loop",
        )
    finally:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and subscriber.latest("hall_left") is None:
            time.sleep(0.02)
        loop.stop()
        publisher.close()

    latest = subscriber.latest("hall_left")
    subscriber.stop()
    return loop, seen, latest


def test_a_recorded_clip_drives_the_whole_production_path(
    settings: Settings, address: str, tmp_path: Path
) -> None:
    """The point of the exercise: frames off disk, and every layer after the decode is the
    one that runs on site -- the same `prepare_frame` at the camera's own analysis
    resolution, the same detector callback, the same preview on the panel."""
    clip = _write_clip(tmp_path / "hall.mp4", frames=40)
    camera = _camera({"path": str(clip), "at_end": "stop"})
    expected = (camera.capture.frame_height, camera.capture.frame_width)

    loop, seen, preview = _run_clip_through_the_loop(camera, address, min_frames=10)

    assert seen, "the detector never ran on a single frame of the clip"
    assert all(shape == expected for shape in seen), (
        f"the detector was handed {seen[0]} -- the clip's own resolution, not the analysis "
        "one. Every px/s threshold in every profile is denominated in the analysis frame."
    )
    assert preview is not None, "nothing reached the panel"
    assert preview.header.camera == "hall_left"


def test_the_loop_measures_the_clips_rate_and_not_this_machines_decode_speed(
    settings: Settings, address: str, tmp_path: Path
) -> None:
    """`_observe_rate` is the instrument that catches `capture.stream_fps` being wrong. It
    must go on working when the frames come from a file -- and on an UNPACED file it would
    report several hundred fps and denounce a configuration that was perfectly correct.

    The clip is written at 25 fps against a configured 15, so the loop should also DISAGREE
    with the config here, which is the true answer for this camera today.

    Bounded above rather than pinned: load on the machine can only slow delivery down, so
    the upper bound is the load-robust direction -- and it is the one the defect breaks, by
    a factor of tens.
    """
    clip = _write_clip(tmp_path / "hall.mp4", frames=60)
    camera = _camera({"path": str(clip), "at_end": "stop"})

    loop, _seen, _preview = _run_clip_through_the_loop(camera, address, min_frames=45)

    assert loop.measured_fps is not None, "the loop never got enough frames to measure a rate"
    assert loop.measured_fps < CLIP_FPS * 1.5, (
        f"the loop measured {loop.measured_fps:.1f} fps from a {CLIP_FPS:.0f} fps recording "
        "-- the file is being drained faster than it was filmed"
    )
    assert not fps_agrees(float(camera.capture.stream_fps), loop.measured_fps), (
        "this clip runs at 25 fps and the camera is configured for 15; the loop's whole "
        "job here is to notice that"
    )


def test_the_camera_loop_reports_a_finished_clip(
    settings: Settings, address: str, tmp_path: Path
) -> None:
    clip = _write_clip(tmp_path / "hall.mp4", frames=10)
    camera = _camera({"path": str(clip), "at_end": "stop"})

    subscriber = PreviewSubscriber(address, stale_after_seconds=30.0).start()
    publisher = PreviewPublisher(address)
    connect_preview_bus(publisher, subscriber)
    loop = CameraLoop(camera, publisher).start()
    try:
        _wait_until(
            lambda: loop.finished,
            timeout=20.0,
            message="the clip ran out and the loop never said it had finished",
        )
        assert loop.frames_processed > 0
    finally:
        loop.stop()
        publisher.close()
        subscriber.stop()


def test_a_camera_that_is_merely_off_the_air_never_finishes_the_worker(
    settings: Settings, address: str
) -> None:
    """The distinction the worker's stop condition rests on.

    A clip that has run out is finished. A camera that cannot be reached is DISCONNECTED,
    and a worker that shut down over it would take the whole group off the air the first
    time a switch rebooted -- the failure this rewrite exists to prevent (R7).

    **The camera is unreachable by INJECTION, not by pointing at an address that happens
    not to answer.** This test first did the latter, and it was wrong twice over. It made
    the result depend on the machine's network stack -- and, measured, it hung: the reader
    thread sits inside `cv2.VideoCapture(url, CAP_FFMPEG)`, which does not look at
    `self._stop` and does not return until FFmpeg's own 30 s stream timeout fires, so
    `stop()` returned after its 5 s join with the thread still running. See the report on
    that finding; it is a property of `CameraStream`, not of this test, and it is NOT
    fixed here.

    `frame_source` is still what builds the source, so this remains the RTSP branch --
    the real URL factory, `reconnects=True` -- with only the socket replaced. A capture
    that reports `isOpened() is False` is exactly what an unreachable camera produces
    once the connect has failed, and it produces it instantly and identically everywhere.
    """
    subscriber = PreviewSubscriber(address, stale_after_seconds=30.0).start()
    publisher = PreviewPublisher(address)
    # The production source for this camera, with the network taken out of it.
    factory = FakeCameraFactory()
    unreachable = replace(frame_source(_camera()), opener=factory)
    loop = CameraLoop(_camera(), publisher, source=unreachable)
    loop.start()
    try:
        _wait_until(
            # `or finished` so that a stream which WRONGLY gives up is caught by the
            # assertion below, which can say so, rather than by this wait timing out with
            # a message about connecting that would send the reader the wrong way.
            lambda: loop._stream.stats.reconnects >= 1 or loop.finished,
            timeout=10.0,
            message="the camera never even tried to connect, so this proves nothing",
        )
        # Still the CAMERA branch, and it cannot quietly become a file test: the URL the
        # opener was handed is the one `build_url` made.
        assert factory.urls[0].startswith("rtsp://"), factory.urls[0]
        assert loop.finished is False, "an unreachable camera reported itself finished"
        assert _every_source_finished([loop]) is False
    finally:
        loop.stop()
        publisher.close()
        subscriber.stop()


def test_a_group_with_no_cameras_is_not_a_group_that_has_finished() -> None:
    """`all([])` is True, and a worker that read an empty group as "done" would exit at
    once instead of reporting a configuration nobody meant to write."""
    assert _every_source_finished([]) is False

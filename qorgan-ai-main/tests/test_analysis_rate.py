"""The rate the detector actually runs at, and everything that claims to know it.

**A measuring instrument calibrated in units the measured thing does not use reports a
real number and a false conclusion.** That is what this file exists to prevent, and it is
the same disease as every other entry in HANDOFF.md's table -- true in one layer, silently
wrong in the next -- except here the wrong layer is the one that grades the detector.

Before this file existed:

  * `display_fps` had three readers and **none of them was the production loop**. The loop
    (`worker/camera_loop.py`) reads `det_every` and processes one frame in every `det_every`
    *that the stream hands it*. The stream hands it the sub-stream, at its own rate.
  * The eval harness computed `display_fps / det_every` and called that the analysis rate.
    On the hall that was 10/1 = 10 fps for a loop running at 15. Every eval number with a
    per-second denominator was off by 1.5x, in a direction nothing announced.
  * `display_fps` was therefore a knob that looked like it controlled the detector and
    controlled only the instrument measuring it -- the exact shape of the three dead knobs
    retired at `121b539`.

The fix is not a rename. It is that the field now states a **fact about the camera** (what
the NVR delivers on the analysis channel), production **checks that fact against reality**
at runtime, and the harness derives its rate from the same field. A knob nobody reads is
a lie; a knob that is checked is a measurement.

The frame COUNTERS in `detection/constants.py` are a separate, deliberately unfinished
piece of this: they are still denominated in frames chosen for ~10 fps, and at 15 fps they
mean 1.5x less time than their comments claim. Retuning them changes detector behaviour
across four modules, which is an on-site decision with real labels -- not one to make from
a desk. So they are left alone and their assumption is made **loud** instead. See
`test_the_counters_do_not_hide_the_rate_they_assume`.
"""

from __future__ import annotations

import logging

import pytest

from qorgan.config.camera import CAMERA_ADAPTER, BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.detection.constants import ASSUMED_ANALYSIS_FPS, counter_drift
from qorgan.evaluation.noise_floor import analysis_fps
from qorgan.preview import PreviewPublisher, PreviewSubscriber
from qorgan.settings import Settings

# What the sub-stream (channel 102) delivers, per docs/next-session-handoff.md §2: the
# camera's own web UI reports 1280x720 @ 15 fps. Asserted here so that a change to the
# shipped config has to come past a test that says where the number came from.
SUBSTREAM_FPS = 15


def _bullying_cameras() -> list[tuple[str, BullyingCamera]]:
    return [
        (name, camera)
        for name, camera in sorted(load_cameras().items())
        if isinstance(camera, BullyingCamera)
    ]


@pytest.fixture
def address() -> str:
    """A free loopback port. Bound and released, so two concurrent suites do not collide
    on a fixed number -- the recorded reproducer for this project's zmq flake."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


# -- the harness and production must agree on the rate ----------------------


def test_the_harness_rate_is_derived_from_the_stream_every_camera_is_fed() -> None:
    """Pins the DERIVATION, not the number.

    An `== 15.0` assertion would have passed just as happily against `display_fps` on the
    day someone typed 15 into it for the wrong reason. What must hold is that the rate the
    harness computes is the rate the loop runs at: one frame in every `det_every` of what
    the stream delivers. Nothing else may enter this formula.
    """
    for name, camera in _bullying_cameras():
        expected = camera.capture.stream_fps / camera.capture.det_every
        assert camera.capture.analysis_fps == pytest.approx(expected), name
        assert analysis_fps(camera.capture.stream_fps, camera.capture.det_every) == pytest.approx(
            expected
        ), name


def test_the_hall_is_graded_at_the_substream_rate_not_at_a_display_number() -> None:
    """The concrete case that was wrong: hall, `det_every: 1`.

    Production analyses every sub-stream frame -- 15 fps. The harness said 10.
    """
    cameras = dict(_bullying_cameras())
    for name in ("hall_left", "hall_right"):
        camera = cameras[name]
        assert camera.capture.det_every == 1, f"{name}: this test's premise moved"
        assert camera.capture.analysis_fps == pytest.approx(float(SUBSTREAM_FPS)), name


def test_det_every_still_divides_the_rate_it_is_the_one_real_frame_knob() -> None:
    """`canteen_inside` was wrong in kind, not degree: `display_fps: 5`, `det_every: 2`
    made the harness say 2.5 fps where production analyses every other 15 fps frame.

    `det_every` is the knob production genuinely honours, so it must genuinely divide.
    """
    inside = load_cameras()["canteen_inside_left"]
    assert inside.capture.det_every == 2, "this test's premise moved"
    assert inside.capture.analysis_fps == pytest.approx(SUBSTREAM_FPS / 2)


def test_the_dead_knob_is_gone_and_a_config_still_carrying_it_will_not_start() -> None:
    """`extra="forbid"` (R10) turns the stale key into a startup error rather than a
    silently ignored one. A rename that left the old key accepted would have reproduced
    the original defect exactly: a value someone sets, and nothing reads."""
    with pytest.raises(Exception, match="display_fps"):
        CAMERA_ADAPTER.validate_python(
            {
                "camera_type": "bullying",
                "role": "main_hall",
                "name": "hall_left",
                "display_name": "Hall",
                "rtsp": {"host": "10.0.0.1"},
                "capture": {"display_fps": 10},
            }
        )


# -- production reads the field, so it can no longer drift unnoticed --------


def _loop_camera(stream_fps: int) -> BullyingCamera:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": "hall_left",
            "display_name": "Hall",
            "rtsp": {"host": "10.0.0.1"},
            "capture": {"stream_fps": stream_fps},
            "preview": {"fps": 15.0},
        }
    )


def _drive_loop(camera, address: str, frames: int = 90):
    """Run the loop against a fake that delivers at a known, steady rate."""
    import time
    from unittest.mock import patch

    from qorgan.worker.camera_loop import CameraLoop
    from tests.fakes import FakeCameraFactory, FakeCapture, connect_preview_bus, noisy_frame

    interval = 0.02  # 50 fps: far faster than any camera here, and far faster than 15
    factory = FakeCameraFactory(
        FakeCapture([(True, noisy_frame(i)) for i in range(frames)], interval=interval)
    )
    subscriber = PreviewSubscriber(address, stale_after_seconds=30.0).start()
    publisher = PreviewPublisher(address)
    connect_preview_bus(publisher, subscriber)

    with patch("qorgan.capture.stream.open_rtsp", factory):
        loop = CameraLoop(camera, publisher)
        loop._stream._opener = factory
        loop.start()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and loop.frames_processed < frames - 10:
            time.sleep(0.02)
        processed = loop.frames_processed
        loop.stop()

    publisher.close()
    subscriber.stop()
    assert processed >= frames - 10, f"the loop stalled at {processed}/{frames}"
    return loop


def test_the_loop_measures_the_rate_it_is_actually_being_fed(
    settings: Settings, address: str
) -> None:
    """The claim `stream_fps` makes is checkable, so the loop checks it.

    Nothing in this repo has ever met a camera. The one thing that will be true on site
    and is not true here is what the NVR really delivers -- so the number must be measured
    where it can be, not asserted from a screenshot forever.
    """
    loop = _drive_loop(_loop_camera(stream_fps=SUBSTREAM_FPS), address)

    assert loop.measured_fps is not None, "the loop never measured its own frame rate"
    assert loop.measured_fps > float(SUBSTREAM_FPS), (
        f"the fake delivers ~50 fps; the loop measured {loop.measured_fps:.1f}"
    )


def test_a_stream_that_disagrees_with_the_config_says_so_out_loud(
    settings: Settings, address: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Configured 15, delivered ~50. On site this is how a wrong `stream_fps` announces
    itself in the first minute instead of quietly mis-scaling every eval number.

    Both numbers must appear: "they disagree" is not actionable, "configured 15, measured
    50" is.
    """
    with caplog.at_level(logging.WARNING, logger="qorgan.worker.camera_loop"):
        _drive_loop(_loop_camera(stream_fps=SUBSTREAM_FPS), address)

    warnings = [r for r in caplog.records if "stream_fps" in r.getMessage()]
    assert warnings, f"no divergence warning; saw {[r.getMessage() for r in caplog.records]}"

    record = warnings[0]
    assert getattr(record, "configured_fps", None) == pytest.approx(float(SUBSTREAM_FPS))
    assert getattr(record, "measured_fps", 0.0) > float(SUBSTREAM_FPS)


def test_a_stream_that_matches_the_config_stays_quiet(
    settings: Settings, address: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A check that fires on everything is noise, and noise is how a real warning gets
    filtered out of a log by the person reading it."""
    with caplog.at_level(logging.WARNING, logger="qorgan.worker.camera_loop"):
        _drive_loop(_loop_camera(stream_fps=50), address)

    assert not [r for r in caplog.records if "stream_fps" in r.getMessage()], (
        "warned even though the configured rate matched what was delivered"
    )


# -- the frame counters: not retuned, but no longer silent ------------------


def test_the_counters_do_not_hide_the_rate_they_assume() -> None:
    """`SUSTAINED_FRAMES = 2` is two frames of evidence. What that means in SECONDS
    depends entirely on the frame rate, and the value was chosen for ~10 fps.

    This is deliberately NOT fixed by retuning: the counters are read by `gates.py`,
    `pipeline.py` and `scoring.py`, and changing them changes what the detector does. It is
    an on-site decision with real labels. What is fixed is that the assumption is now a
    named constant anyone can find and compare against, rather than a sentence in a
    docstring.
    """
    assert pytest.approx(10.0) == ASSUMED_ANALYSIS_FPS


def test_counter_drift_says_how_far_production_has_moved_from_that_assumption() -> None:
    """1.5 is not a rounding error. It means every frame counter buys 1.5x less time than
    its comment claims: `BRIEF_ENCOUNTER_FRAMES = 4` was "under half a second" and is 0.27 s.
    """
    assert counter_drift(15.0) == pytest.approx(1.5)
    assert counter_drift(ASSUMED_ANALYSIS_FPS) == pytest.approx(1.0)


def test_every_shipped_camera_reports_its_drift_rather_than_inheriting_it_silently() -> None:
    """The check that would have caught the original defect. Every bullying camera now has
    a drift figure derived from config, so `config validate` can print it and nobody has to
    already know to ask."""
    for name, camera in _bullying_cameras():
        drift = counter_drift(camera.capture.analysis_fps)
        assert drift > 0.0, name
        assert drift == pytest.approx(camera.capture.analysis_fps / ASSUMED_ANALYSIS_FPS), name


def test_config_validate_prints_the_drift_so_it_cannot_be_inherited_unread(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """`qorgan config validate` is a documented day-one command (HANDOVER.md §6). The
    warning belongs where the person setting the system up will actually see it."""
    from qorgan.cli import main

    code = main(["config", "validate"])
    out = capsys.readouterr().out

    assert code == 0
    assert "counter" in out.lower(), out
    assert "10" in out and "15" in out, (
        "the drift line must name BOTH rates: the one the counters assume and the one "
        f"production runs at. Got:\n{out}"
    )

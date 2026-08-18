"""Stopping a camera that is not answering.

Measured on this machine, not assumed: a reader thread sits INSIDE
`cv2.VideoCapture(url, CAP_FFMPEG)` for ~30 s when the address goes nowhere, and
`self._stop` is not looked at until that returns. `CameraStream.stop()` joined for 5 s,
the join expired, and `stop()` returned as though it had worked -- dropping the thread
handle on the way out, so nothing anywhere could later tell that the reader was still
alive. The supervisor's grace is 10 s, and the largest worker group is four cameras
stopped one after another, so the process was killed and the log said "worker ignored
terminate": a true sentence about the worker, pointing away from the camera that caused it.

**Not one test here touches the network.** A test that waits out a real TCP timeout is a
test about this machine's routing table. The reader's opener is injectable, so "the camera
never answers" is a fake that blocks on an Event -- deterministic, and the same shape of
block as the real one.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import pytest

from qorgan.capture.stream import CameraStream, open_rtsp
from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.config.common import STOP_MARGIN_SECONDS, RtspSettings
from qorgan.config.loader import load_cameras, load_workers
from qorgan.supervisor.managed import TERMINATE_GRACE_SECONDS
from qorgan.worker.camera_loop import CameraLoop, stop_all
from qorgan.worker.camera_loop import worst_case_stop_seconds as budget
from tests.conftest import CONFIG_DIR
from tests.fakes import FakeCameraFactory, FakeCapture, noisy_frame

# A password that could not be anything else, so a leak into a log line is unmistakable.
PASSWORD = "n0t-in-a-l0g-please"
URL = f"rtsp://admin:{PASSWORD}@10.0.0.1:554/Streaming/Channels/102"

# Quick enough that the whole file is a few seconds, and the two values differ so a test
# can tell which one a derived number came from.
QUICK = RtspSettings(host="10.0.0.1", open_timeout_seconds=0.2, read_timeout_seconds=0.1)

# Deliberately NOT 5.0: 1.4 + 1.0 margin = 2.4, which is on the other side of the 5.0 the
# old `stop()` hardcoded, so a timing assertion can tell the two apart in a direction no
# amount of machine load can fake.
DERIVED = RtspSettings(host="10.0.0.1", open_timeout_seconds=1.4, read_timeout_seconds=0.2)

# An upper bound on how long a wedged fake stays wedged if a test somehow fails to release
# it. The reader threads are daemons, so this only bounds the mess, it does not cause it.
WEDGE_SECONDS = 20.0


class WedgedOpener:
    """A camera that accepts the connection and then says nothing.

    This is `cv2.VideoCapture(url, CAP_FFMPEG)` against a host on a VLAN that is not up:
    the call does not fail, it does not return, and the thread that made it cannot reach
    its stop flag while it is in there.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self._release = threading.Event()

    def __call__(self, _url: str, _rtsp: RtspSettings) -> FakeCapture:
        self.entered.set()
        self._release.wait(WEDGE_SECONDS)
        return FakeCapture([], opened=False)

    def release(self) -> None:
        self._release.set()


class NullPublisher:
    """Enough of PreviewPublisher for a loop that will never get a frame to publish."""

    def publish(self, *_args: object, **_kwargs: object) -> None:
        return None


def _wedged_stream(rtsp: RtspSettings = QUICK) -> tuple[CameraStream, WedgedOpener]:
    opener = WedgedOpener()
    stream = CameraStream("hall_left", lambda: URL, rtsp, opener=opener)
    stream.start()
    assert opener.entered.wait(timeout=5.0), "the reader never reached the opener"
    return stream, opener


def _camera(name: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": name,
            "display_name": name,
            "rtsp": {
                "host": "10.0.0.1",
                "open_timeout_seconds": QUICK.open_timeout_seconds,
                "read_timeout_seconds": QUICK.read_timeout_seconds,
            },
        }
    )


def _wedged_loops(count: int) -> tuple[list[CameraLoop], list[WedgedOpener]]:
    loops: list[CameraLoop] = []
    openers: list[WedgedOpener] = []
    for index in range(count):
        loop = CameraLoop(_camera(f"wedged_{index}"), NullPublisher())
        opener = WedgedOpener()
        loop._stream._opener = opener
        loop.start()
        assert opener.entered.wait(timeout=5.0), "a reader never reached the opener"
        loops.append(loop)
        openers.append(opener)
    return loops, openers


def _release(openers: list[WedgedOpener], loops: list[CameraLoop]) -> None:
    for opener in openers:
        opener.release()
    for loop in loops:
        loop.stop()


# -- the timeout itself ----------------------------------------------------


def test_this_opencv_build_can_bound_the_open() -> None:
    """The premise the whole fix rests on. If a future pin loses these properties, the
    parameter list below becomes a list of numbers OpenCV silently ignores -- the fix
    would stop working and every other test here would still pass, because every other
    test uses a fake opener."""
    assert hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC")
    assert hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC")


def test_the_timeouts_travel_as_constructor_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured, on this build, against an unroutable address:

        VideoCapture(url, CAP_FFMPEG)                            30.05 s
        VideoCapture(url, CAP_FFMPEG, [OPEN_TIMEOUT_MSEC, 2000])  2.20 s
        VideoCapture(); set(OPEN_TIMEOUT_MSEC, 2000); open(url)  30.02 s

    The third line is why this test exists. `.set()` after construction is the obvious
    way to write it, it raises nothing, it returns True -- and it buys exactly nothing,
    because the wait is over by the time it runs.
    """
    constructed: list[tuple[object, ...]] = []
    afterwards: list[tuple[int, object]] = []

    class Recording:
        def __init__(self, *args: object) -> None:
            constructed.append(args)

        def set(self, prop: int, value: object) -> bool:
            afterwards.append((prop, value))
            return True

    monkeypatch.setattr(cv2, "VideoCapture", Recording)
    open_rtsp(URL, RtspSettings(host="h", open_timeout_seconds=3.5, read_timeout_seconds=7.25))

    params = list(constructed[0][2])  # type: ignore[call-overload]
    assert params[params.index(int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC)) + 1] == 3500
    assert params[params.index(int(cv2.CAP_PROP_READ_TIMEOUT_MSEC)) + 1] == 7250
    assert not [p for p, _ in afterwards if p == cv2.CAP_PROP_OPEN_TIMEOUT_MSEC], (
        "the open timeout was applied with .set() -- measured at 30.02 s, i.e. not applied"
    )


def test_the_stop_timeout_is_derived_from_the_two_blocks_it_waits_out() -> None:
    """Not a fourth knob. A separate key could be set SHORTER than the block it exists to
    outlast, which is the exact arrangement that made `stop()` report a success it had
    not achieved."""
    assert DERIVED.stop_timeout_seconds == pytest.approx(1.4 + STOP_MARGIN_SECONDS)
    assert QUICK.stop_timeout_seconds == pytest.approx(0.2 + STOP_MARGIN_SECONDS)


# -- stop() telling the truth ----------------------------------------------


def test_stop_says_so_when_the_reader_is_still_blocked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The defect, stated as a test. The old `stop()` returned None here and logged
    nothing, so an unstoppable camera left no trace anywhere in the system."""
    stream, opener = _wedged_stream()
    try:
        with caplog.at_level(logging.ERROR, logger="qorgan.capture.stream"):
            stopped = stream.stop()

        assert stopped is False, "stop() reported success while the reader was still alive"
        records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert records, "an unstoppable camera left no trace in the log"
        assert getattr(records[0], "camera", None) == "hall_left", (
            "the log does not name the camera, so it cannot be told apart from the "
            "supervisor's later 'worker ignored terminate'"
        )
        assert "camera" in str(getattr(records[0], "consequence", "")), (
            "the log states the fact but not the false conclusion it must head off: that "
            "the worker, not the camera, was at fault"
        )
    finally:
        opener.release()
        stream.stop()


def test_stop_keeps_the_handle_of_a_thread_it_could_not_stop() -> None:
    """The old code cleared `self._thread` whether the join worked or not, so the one
    object that knew the reader was still running threw away its only handle on it."""
    stream, opener = _wedged_stream()
    try:
        assert stream.stop() is False
        assert stream._thread is not None, "the handle was dropped"
        assert stream._thread.is_alive(), "the premise is gone: the thread did stop"
    finally:
        opener.release()
        stream.stop()


def test_stop_returns_true_when_the_reader_really_ended() -> None:
    """The positive control. Without it, `stop()` could return False unconditionally and
    every failure test above would still pass."""
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(i)) for i in range(50)]))
    stream = CameraStream("hall_left", lambda: URL, QUICK, opener=factory).start()
    assert stream.read(timeout=2.0) is not None, "the fake camera never delivered a frame"

    assert stream.stop() is True
    assert stream._thread is not None
    assert not stream._thread.is_alive()


def test_the_stop_timeout_comes_from_the_config_not_a_constant() -> None:
    """`DERIVED` waits 2.4 s; the old code waited a hardcoded 5.0 s. The gap is what this
    measures, and it is wide enough that no amount of load on this machine closes it."""
    stream, opener = _wedged_stream(DERIVED)
    try:
        started = time.monotonic()
        assert stream.stop() is False
        elapsed = time.monotonic() - started
    finally:
        opener.release()
        stream.stop()

    assert elapsed >= DERIVED.stop_timeout_seconds - 0.1, (
        f"gave up after {elapsed:.2f}s, before the {DERIVED.stop_timeout_seconds}s the "
        "config asks for"
    )
    assert elapsed < 5.0, f"waited {elapsed:.2f}s -- that is the old hardcoded 5.0, not config"


def test_a_stopped_stream_refuses_to_start_again() -> None:
    """The loaded gun. `stop()` used to clear `self._thread`, so this second `start()` was
    ACCEPTED -- and then handed back a thread that returned on its first line, because
    `self._stop` is never unset. No exception, no log, `is_alive()` briefly True, and the
    camera unwatched from then on. Nothing on today's paths restarts a stream (the
    supervisor restarts whole processes), which is exactly why it would have gone
    unnoticed until something did."""
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(1))] * 5))
    stream = CameraStream("hall_left", lambda: URL, QUICK, opener=factory).start()
    assert stream.stop() is True

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        stream.start()


# -- the whole group's shutdown budget -------------------------------------


def test_stop_all_names_the_cameras_that_did_not_stop() -> None:
    loops, openers = _wedged_loops(2)
    try:
        assert sorted(stop_all(loops)) == ["wedged_0", "wedged_1"]
    finally:
        _release(openers, loops)


def test_stop_all_is_silent_when_every_camera_stops() -> None:
    """The positive control for the one above."""
    loops: list[CameraLoop] = []
    for index in range(2):
        loop = CameraLoop(_camera(f"quick_{index}"), NullPublisher())
        loop._stream._opener = FakeCameraFactory(FakeCapture([(True, noisy_frame(index))] * 50))
        loops.append(loop.start())

    assert stop_all(loops) == []


def test_four_wedged_cameras_stop_together_inside_the_grace() -> None:
    """The consequence on site, with the real numbers.

    `bullying_stairs_yard` holds four cameras and the supervisor allows 10 s. One after
    another, four cameras cost four times one camera; together they cost one. The second
    assertion is what keeps this test honest: it fails if the parallel stop is quietly
    replaced by a for-loop that happens to be fast enough with these tiny timeouts.

    This measures the WAIT, which is what the grace pays for, and the fake cannot do
    otherwise: real OpenCV serialises the FFmpeg open, so four real readers would come
    back at 4, 8, 12 and 16 s and three of them would still be running when this returns.
    That is why `unstopped` is asserted to be all four here rather than none -- the
    contract is "bounded and honest", not "everything really stopped".
    """
    loops, openers = _wedged_loops(4)
    try:
        started = time.monotonic()
        unstopped = stop_all(loops)
        elapsed = time.monotonic() - started
    finally:
        _release(openers, loops)

    assert len(unstopped) == 4, "the premise is gone: some camera actually stopped"
    assert elapsed < TERMINATE_GRACE_SECONDS, (
        f"four cameras took {elapsed:.2f}s against a {TERMINATE_GRACE_SECONDS}s grace: "
        "the supervisor would kill this worker instead of stopping it"
    )

    # The teeth. The bound above has none with timeouts this small -- a sequential stop
    # of four QUICK cameras costs 4.8 s and would clear a 10 s grace comfortably, so the
    # first assertion alone stayed GREEN when `stop_all` was replaced by a for-loop.
    # (`test_the_shipped_configuration_stops_inside_the_supervisors_grace` is where the
    # 10 s bound is tested against numbers that can actually breach it.)
    #
    # What separates the two arrangements is the shape of the cost, so measure that:
    # together, four cameras cost ONE camera's wait; one after another, four of them.
    # `LOOP_JOIN_SECONDS` is deliberately absent -- the loop threads are already gone by
    # the time the reader join expires, so counting it inflated the sequential estimate
    # to 12.8 s and put the threshold above the 4.8 s a real for-loop actually spends.
    together = QUICK.stop_timeout_seconds
    one_after_another = len(loops) * together
    assert elapsed < (together + one_after_another) / 2, (
        f"{elapsed:.2f}s sits nearer the {one_after_another:.2f}s a sequential stop costs "
        f"than the {together:.2f}s a parallel one does: these cameras were not stopped "
        "together"
    )


def test_no_log_line_from_a_failed_stop_carries_the_password(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rule R4. This is a brand-new set of log lines about a URL that would not open, and
    the URL is the one thing that must never be printed. The legacy printed it into every
    log file and drew it onto debug JPEGs an unauthenticated web UI served (audit C-02).
    """
    loops, openers = _wedged_loops(2)
    try:
        with caplog.at_level(logging.DEBUG):
            stop_all(loops)
        leaked = [r for r in caplog.records if PASSWORD in _rendered(r)]
        assert not leaked, f"the password reached {len(leaked)} log record(s)"
    finally:
        _release(openers, loops)


def _rendered(record: logging.LogRecord) -> str:
    """The message AND every extra hung off the record: a structured field reaches the
    log file exactly like the message does, and is the easier of the two to forget."""
    return record.getMessage() + " " + " ".join(str(v) for v in record.__dict__.values())


def test_the_shipped_configuration_stops_inside_the_supervisors_grace() -> None:
    """The arithmetic that chose a parallel stop, checked against the files that ship.

    Re-derived here rather than quoted, because both halves move: a school raising
    `rtsp.read_timeout_seconds`, or an engineer adding a fifth camera to a group, changes
    this and would otherwise find out months later as an unexplained kill.
    """
    cameras = load_cameras(CONFIG_DIR)
    workers = load_workers(cameras, CONFIG_DIR)

    parallel = budget(cameras.values())
    assert parallel < TERMINATE_GRACE_SECONDS, (
        f"a group needs {parallel}s to stop and the supervisor allows "
        f"{TERMINATE_GRACE_SECONDS}s: lower the rtsp timeouts"
    )

    biggest = max(len(group.cameras) for group in workers.groups)
    assert biggest == 4, f"workers.yaml's largest group now holds {biggest} cameras, not 4"
    assert biggest * parallel >= TERMINATE_GRACE_SECONDS, (
        f"stopping {biggest} cameras one after another would now cost "
        f"{biggest * parallel}s and still fit the {TERMINATE_GRACE_SECONDS}s grace -- so "
        "this suite would stay green if stop_all went back to a for-loop. The reason for "
        "stopping them together has expired; check the timeouts before trusting it."
    )

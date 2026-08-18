"""The RTSP reader thread."""

from __future__ import annotations

import time

import pytest

from qorgan.capture.quality import QualityPolicy
from qorgan.capture.stream import CameraStream
from qorgan.config.common import RtspSettings
from tests.fakes import FakeCameraFactory, FakeCapture, frozen_frame, noisy_frame

URL = "rtsp://user:pw@10.0.0.1:554/Streaming/Channels/102"

# A reconnect delay of 0.01 s so a reconnect test does not wait two real seconds, and
# short timeouts so a stop test does not wait five. Everything else is the shipped default.
FAST = RtspSettings(
    host="10.0.0.1",
    reconnect_delay_seconds=0.01,
    open_timeout_seconds=0.2,
    read_timeout_seconds=0.2,
)


def _stream(factory: FakeCameraFactory, **kwargs: object) -> CameraStream:
    return CameraStream("hall_left", lambda: URL, FAST, opener=factory, **kwargs)


def _drain(stream: CameraStream, count: int, timeout: float = 2.0) -> list:
    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < count and time.monotonic() < deadline:
        frame = stream.read(timeout=0.05)
        if frame is not None:
            frames.append(frame)
    return frames


def test_frames_are_delivered_in_order() -> None:
    # A long script: the reader runs ahead of the test and DROPS old frames by design
    # (the queue holds one slot), so a five-frame script could be exhausted before the
    # test observes three of them.
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(i % 8)) for i in range(500)]))
    with _stream(factory) as stream:
        frames = _drain(stream, 3)

    assert len(frames) >= 3
    assert [f.seq for f in frames] == sorted({f.seq for f in frames})
    assert all(f.camera == "hall_left" for f in frames)


def test_a_frame_is_never_delivered_twice() -> None:
    """The legacy bug: read() returned the SAME frame again when the queue was empty
    and the frame was under 0.35 s old, so one physical moment was analysed up to
    three times and every temporal counter inflated."""
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(i)) for i in range(3)]))
    with _stream(factory) as stream:
        frames = _drain(stream, 3)
        # Ask again, immediately, with the stream now exhausted:
        repeat = stream.read(timeout=0.1)

    seqs = [f.seq for f in frames]
    assert len(seqs) == len(set(seqs)), "the same frame was handed out more than once"
    assert repeat is None, "read() invented a frame when there was no new one"


def test_a_slow_reader_gets_the_newest_frame_not_a_backlog() -> None:
    script_length = 50
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(i)) for i in range(script_length)]))
    with _stream(factory) as stream:
        factory.opened[0].exhausted.wait(timeout=2.0)
        # `exhausted` is set from inside the reader thread's *next* read() call, which
        # happens strictly after the final real frame was already queued -- but a fixed
        # sleep here is still a guess about how long it takes for that to be visible.
        # Poll the actual counter the publish increments instead of hoping 0.05s was
        # enough: under load it can still be in flight when a fixed sleep is not.
        assert _wait_for(lambda: stream.stats.frames_published >= script_length, timeout=2.0), (
            "not every scripted frame was published before the feed was exhausted"
        )
        latest = stream.read(timeout=0.5)

    assert latest is not None
    assert stream.stats.frames_dropped > 0, "the queue should have shed old frames"
    # We got a recent frame, not the one from the start of the run.
    assert latest.seq > 1


def test_the_reader_reconnects_when_the_feed_drops() -> None:
    first = FakeCapture([(True, noisy_frame(1)), (False, None)])
    second = FakeCapture([(True, noisy_frame(2)) for _ in range(10)])
    factory = FakeCameraFactory(first, second)

    with _stream(factory) as stream:
        _drain(stream, 2)

    assert len(factory.urls) >= 2, "it did not reconnect"
    assert first.released, "the dead capture was not released"
    assert stream.stats.reconnects >= 1


def test_the_reader_survives_an_exception_and_reconnects() -> None:
    """Rule R7. Legacy's event loop had a `finally` and no `except`, so one
    `database is locked` killed the thread forever and the system then looked
    healthy while recording nothing."""

    class Exploding(FakeCapture):
        def read(self):  # type: ignore[override]
            raise RuntimeError("decoder blew up")

    factory = FakeCameraFactory(Exploding([]), FakeCapture([(True, noisy_frame(9))] * 5))
    with _stream(factory) as stream:
        frames = _drain(stream, 1)
        assert stream._thread is not None
        assert stream._thread.is_alive(), "the reader thread died"

    assert frames, "it never recovered"
    assert stream.stats.last_error is None or "RuntimeError" not in str(stream.stats.last_error)


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_frozen_feed_forces_a_reconnect() -> None:
    policy = QualityPolicy(corrupt_frames_before_reconnect=3)
    frozen = FakeCapture([(True, frozen_frame()) for _ in range(20)])
    good = FakeCapture([(True, noisy_frame(i)) for i in range(10)])
    factory = FakeCameraFactory(frozen, good)

    with _stream(factory, quality=policy) as stream:
        assert _wait_for(lambda: len(factory.urls) >= 2), "it never abandoned the dead feed"
        assert stream.stats.frames_corrupt >= 3
        assert frozen.released


def test_the_corruption_check_gives_up_rather_than_loop_forever() -> None:
    """A camera that is merely dark and static must not be reconnected forever.
    If the check keeps ending sessions and the picture never improves, the check is
    wrong about this camera -- so it disables itself, loudly (audit H-13)."""
    policy = QualityPolicy(corrupt_frames_before_reconnect=2, reconnects_before_giving_up=2)
    sessions = [FakeCapture([(True, frozen_frame())] * 6) for _ in range(6)]
    factory = FakeCameraFactory(*sessions)

    with _stream(factory, quality=policy) as stream:
        assert _wait_for(lambda: stream.stats.quality_check_disabled), (
            "it would have reconnected forever, exactly when the hallway was quiet"
        )
        # Having given up, it now analyses the picture it actually has.
        before = stream.stats.frames_published
        assert _wait_for(lambda: stream.stats.frames_published > before)


def test_the_url_is_rebuilt_on_every_connection_and_never_stored() -> None:
    """The URL carries the camera password, so it is produced on demand and does not
    sit on the instance where a repr or a log line could pick it up."""
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(1))] * 3))
    with _stream(factory) as stream:
        _drain(stream, 1)
        assert "pw" not in repr(stream)
        assert not any("pw" in str(v) for v in vars(stream).values() if isinstance(v, str))

    assert factory.urls[0] == URL  # it did use the real URL


def test_starting_twice_is_an_error() -> None:
    factory = FakeCameraFactory(FakeCapture([(True, noisy_frame(1))] * 3))
    with _stream(factory) as stream, pytest.raises(RuntimeError, match="already started"):
        stream.start()

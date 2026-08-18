"""The preview bus, over real sockets."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from qorgan.config.common import PreviewSettings
from qorgan.preview import PreviewPublisher, PreviewSubscriber, encode_preview
from tests.fakes import connect_preview_bus, noisy_frame

# The schema caps preview fps at 15 -- nobody can see more than that of a corridor,
# and the encode is not free. Use the ceiling so the transport tests are not throttled.
SETTINGS = PreviewSettings(fps=15.0, width=320, jpeg_quality=70)


@pytest.fixture
def address() -> str:
    # A distinct port per test, so tests cannot collide on a bound address.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


@pytest.fixture
def bus(address: str):
    subscriber = PreviewSubscriber(address, stale_after_seconds=5.0).start()
    publisher = PreviewPublisher(address)
    # NOT "retry on flake": ZeroMQ's PUB socket silently DROPS anything published
    # before the SUB's connect finishes -- there is no handshake and no error. A
    # single fixed sleep before the first real publish is a guess about how long
    # that takes; publishing until the subscriber actually confirms receipt, bounded
    # by a deadline, is the only correct way to synchronise the two. See
    # tests/fakes.py::connect_preview_bus.
    connect_preview_bus(publisher, subscriber)
    yield publisher, subscriber
    publisher.close()
    subscriber.stop()


def _await(subscriber: PreviewSubscriber, camera: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        preview = subscriber.latest(camera)
        if preview is not None:
            return preview
        time.sleep(0.02)
    return None


# -- encoding --------------------------------------------------------------


def test_a_preview_is_downscaled_and_encoded_once() -> None:
    """Legacy pushed full-size frame COPIES around and re-encoded per browser client.
    A 1280x720 BGR frame is 2.7 MB; this should be a few tens of KB."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = noisy_frame(1)[0, 0]

    jpeg = encode_preview(frame, SETTINGS)

    assert jpeg is not None
    assert len(jpeg) < frame.nbytes // 20
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == SETTINGS.width  # downscaled to the configured width


def test_a_small_frame_is_not_upscaled() -> None:
    jpeg = encode_preview(np.zeros((64, 64, 3), dtype=np.uint8), SETTINGS)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 64


# -- transport -------------------------------------------------------------


def test_a_frame_published_by_a_worker_reaches_the_web_process(bus) -> None:
    publisher, subscriber = bus

    assert publisher.publish("hall_left", noisy_frame(1), SETTINGS)

    preview = _await(subscriber, "hall_left")
    assert preview is not None
    assert preview.header.camera == "hall_left"
    assert preview.header.seq == 1
    assert cv2.imdecode(np.frombuffer(preview.jpeg, np.uint8), cv2.IMREAD_COLOR) is not None


def test_each_camera_has_its_own_slot(bus) -> None:
    publisher, subscriber = bus

    publisher.publish("hall_left", noisy_frame(1), SETTINGS)
    publisher.publish("stairs_floor1", noisy_frame(2), SETTINGS, status="alert")

    assert _await(subscriber, "hall_left") is not None
    alert = _await(subscriber, "stairs_floor1")
    assert alert is not None
    assert alert.header.status == "alert"
    assert set(subscriber.cameras()) == {"hall_left", "stairs_floor1"}


def test_only_the_newest_frame_per_camera_is_kept(bus) -> None:
    """One slot per camera. The dashboard wants the current picture, not a backlog."""
    publisher, subscriber = bus

    base = time.time()
    for index in range(1, 6):
        # Step `now` past the rate limit so every frame really is sent, and give the
        # subscriber a moment to drain: the send high-water mark is deliberately low,
        # so a burst is DROPPED rather than queued. Old previews are worthless.
        assert publisher.publish("hall_left", noisy_frame(index), SETTINGS, now=base + index)
        time.sleep(0.05)

    # Wait for the CONDITION, not for a duration. A fixed sleep here asserts that this
    # machine drained the socket within 0.4s -- which is a fact about the machine, not
    # about the code. On a loaded box the slot still holds seq 4 and the test fails for a
    # reason that has nothing to do with the behaviour under test.
    #
    # Timing out is still a real failure, and it says the right thing: the newest frame
    # never arrived. That is exactly the bug this test exists to catch.
    deadline = time.monotonic() + 3.0
    preview = None
    while time.monotonic() < deadline:
        preview = subscriber.latest("hall_left")
        if preview is not None and preview.header.seq == 5:
            break
        time.sleep(0.02)

    assert preview is not None, "no preview ever arrived"
    assert preview.header.seq == 5, (
        f"an older frame was left in the slot: seq {preview.header.seq}, expected 5"
    )


# -- rate limiting and staleness ------------------------------------------


def test_the_preview_rate_is_limited(bus) -> None:
    """A 25 fps camera must not become a 25 fps preview: encoding is not free, and
    nobody can see 25 fps of a corridor anyway."""
    publisher, _ = bus
    slow = PreviewSettings(fps=3.0)

    sent = [
        publisher.publish("hall_left", noisy_frame(i), slow, now=1000.0 + i * 0.1)
        for i in range(10)
    ]

    assert sum(sent) < 5, "the rate limit did nothing"
    assert sent[0] is True


def test_a_disabled_preview_publishes_nothing(bus) -> None:
    publisher, subscriber = bus
    off = PreviewSettings(enabled=False)

    assert not publisher.publish("hall_left", noisy_frame(1), off)
    assert subscriber.latest("hall_left") is None


def test_a_stale_preview_is_reported_as_offline_not_shown(bus) -> None:
    """Showing a five-minute-old frame as if it were live tells the operator the
    hallway is calm when in fact nobody is watching it."""
    publisher, subscriber = bus
    publisher.publish("hall_left", noisy_frame(1), SETTINGS)
    assert _await(subscriber, "hall_left") is not None

    future = time.time() + 3600
    assert subscriber.latest("hall_left", now=future) is None
    assert not subscriber.is_live("hall_left", now=future)


def test_an_unknown_camera_is_not_live(bus) -> None:
    _, subscriber = bus
    assert subscriber.latest("no_such_camera") is None


def test_a_publisher_refuses_port_zero_instead_of_publishing_into_the_void() -> None:
    """Port 0 means "any free port", and only the SUBSCRIBER can resolve that -- it learns
    the port by binding it. The publisher is a different PROCESS and cannot discover it.

    Without this guard the misconfiguration is silent and expensive: the SUB binds some
    random port, the PUB connects to port 0, no preview ever arrives, and nothing says so.
    The operator sees a blank camera and goes looking for a broken camera.
    """
    import pytest

    from qorgan.preview import PreviewPublisher

    with pytest.raises(ValueError, match="port 0"):
        PreviewPublisher("tcp://127.0.0.1:0")

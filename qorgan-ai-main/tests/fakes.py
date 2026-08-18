"""A fake cv2.VideoCapture, so the reader thread is testable without an RTSP server."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from qorgan.config.common import PreviewSettings, RtspSettings
from qorgan.preview import PreviewPublisher, PreviewSubscriber

Script = Iterator[tuple[bool, np.ndarray | None]]


def noisy_frame(seed: int, *, brightness: int = 128, spread: int = 40) -> np.ndarray:
    """A normal, textured picture."""
    rng = np.random.default_rng(seed)
    return rng.integers(
        max(0, brightness - spread), min(255, brightness + spread), (64, 64, 3), dtype=np.uint8
    )


def dark_frame(seed: int) -> np.ndarray:
    """A dark corridor at night: almost no texture, but real sensor noise.

    The legacy called this "broken" and reconnected. It is not broken.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 6, (64, 64, 3), dtype=np.uint8)


def frozen_frame() -> np.ndarray:
    """A grey 'no signal' card: featureless AND identical every time. Genuinely broken."""
    return np.full((64, 64, 3), 16, dtype=np.uint8)


class FakeCapture:
    """Plays back a scripted sequence of (ok, frame) pairs, then reports failure."""

    def __init__(
        self,
        script: list[tuple[bool, np.ndarray | None]],
        *,
        opened: bool = True,
        interval: float = 0.0,
    ) -> None:
        self._script = list(script)
        self._opened = opened
        self._index = 0
        self._interval = interval
        self.released = False
        self.exhausted = threading.Event()

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """One frame. `interval` makes this behave like a camera rather than like RAM.

        A real RTSP source delivers frames at a FRAME RATE. Without an interval this fake
        delivers them at memory bandwidth, so the stream's producer thread drains the whole
        script in microseconds -- and `CameraStream`'s Queue(maxsize=1) correctly drops all
        but the newest, because that is what you want from a real camera when the consumer
        is slow. The consequence is that a test can never observe more than a handful of
        DISTINCT frames, however long it waits, and how many it gets depends on how fast
        `_tick` happens to be.

        That is not hypothetical: it is what starved `test_det_every_is_honoured` the moment
        `prepare_frame()` made `_tick` do a real resize. The loop was not stalled and the
        code was not wrong -- the fake was outrunning it. A test whose result depends on the
        speed of the machine is measuring the machine.
        """
        if self._index >= len(self._script):
            self.exhausted.set()
            return False, None
        if self._interval:
            time.sleep(self._interval)
        item = self._script[self._index]
        self._index += 1
        return item

    def release(self) -> None:
        self.released = True

    def set(self, *_args: object) -> bool:
        return True


class FakeCameraFactory:
    """Hands out a fresh FakeCapture per connection attempt, recording each URL.

    It takes the `RtspSettings` too, and keeps them, because that is the only way a test
    can show that the timeouts the config declares are the timeouts the opener is handed.
    A fake that quietly accepted `**kwargs` would let `open_timeout_seconds` go the way
    of `display_fps`: declared, validated, and read by nobody who matters.
    """

    def __init__(self, *sessions: FakeCapture) -> None:
        self._sessions = list(sessions)
        self.urls: list[str] = []
        self.settings: list[RtspSettings] = []
        self.opened: list[FakeCapture] = []

    def __call__(self, url: str, rtsp: RtspSettings) -> FakeCapture:
        self.urls.append(url)
        self.settings.append(rtsp)
        capture = self._sessions.pop(0) if self._sessions else FakeCapture([], opened=False)
        self.opened.append(capture)
        return capture


PREVIEW_HANDSHAKE_CAMERA = "__slow_joiner_handshake__"


def connect_preview_bus(
    publisher: PreviewPublisher, subscriber: PreviewSubscriber, *, timeout: float = 3.0
) -> None:
    """Block until a real PUB/SUB connection is up, then leave no trace of doing so.

    This is NOT "retry on flake". ZeroMQ's PUB socket fans out only to peers it is
    CURRENTLY connected to: a message published before the SUB's connect finishes is
    not queued for later delivery -- it is simply gone. This is the "slow joiner"
    problem: there is no connect handshake and no error to catch when it happens. A
    fixed sleep before the first real publish is a guess about how long the connect
    takes, and a guess is sometimes wrong under load. Publishing a throwaway frame in
    a loop until the subscriber reports having actually seen it -- bounded by a
    deadline, so a genuine failure still raises loudly -- is the only correct way to
    synchronise a protocol that offers no acknowledgement of its own.

    A dedicated camera name keeps this from being mistaken for a real published
    frame. The publisher's send HWM lets a few duplicates queue up while this
    function is still noticing the first one arrived, so `_drain_handshake` mops
    those up too -- otherwise a straggler could reappear in a test's
    `subscriber.cameras()`, or in the '__slow_joiner_handshake__' bucket of its own
    seq counter, moments after this function returned.
    """
    settings = PreviewSettings(fps=15.0)
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        # A synthetic, strictly increasing `now` so the publisher's own per-camera
        # rate limit never blocks a retry -- only the network connection may.
        publisher.publish(PREVIEW_HANDSHAKE_CAMERA, noisy_frame(0), settings, now=attempt * 0.1)
        if _pop_handshake(subscriber):
            _drain_handshake(subscriber)
            return
        time.sleep(0.02)
    raise AssertionError("PUB/SUB never connected within the deadline")


def _pop_handshake(subscriber: PreviewSubscriber) -> bool:
    with subscriber._lock:
        return subscriber._latest.pop(PREVIEW_HANDSHAKE_CAMERA, None) is not None


def _drain_handshake(subscriber: PreviewSubscriber, quiet_for: float = 0.15) -> None:
    """Keep popping until nothing has arrived for `quiet_for` seconds straight."""
    quiet_since = time.monotonic()
    while time.monotonic() - quiet_since < quiet_for:
        if _pop_handshake(subscriber):
            quiet_since = time.monotonic()
        time.sleep(0.02)

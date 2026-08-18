"""Web side of the preview bus: keep the latest JPEG for each camera, and nothing more.

Bounded by construction. There is exactly one slot per camera, holding one already-
encoded JPEG (tens of KB), and a frame older than its TTL is reported as offline
rather than shown as if it were live -- a stale preview on a safety dashboard is worse
than an honest "no signal", because it tells the operator the hallway is calm when in
fact nobody is watching it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import zmq

from qorgan.logging_setup import get_logger
from qorgan.preview.protocol import PreviewHeader

logger = get_logger(__name__)

RECEIVE_HIGH_WATER_MARK = 8
POLL_TIMEOUT_MS = 250


@dataclass(frozen=True, slots=True)
class Preview:
    header: PreviewHeader
    jpeg: bytes
    received_at: float

    def age_seconds(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.received_at


class PreviewSubscriber:
    """Runs in the web process. One background thread; the rest is a dict lookup."""

    def __init__(
        self,
        address: str,
        *,
        stale_after_seconds: float = 10.0,
        context: zmq.Context | None = None,
    ) -> None:
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.set_hwm(RECEIVE_HIGH_WATER_MARK)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")  # every camera
        self.address = self._bind(address)

        self._stale_after = stale_after_seconds
        self._latest: dict[str, Preview] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        logger.info("preview subscriber bound", extra={"address": self.address})

    def _bind(self, address: str) -> str:
        """Bind and return the address actually bound.

        A port of 0 means "pick one for me": bind_to_random_port asks the OS for a
        free port and binds it atomically, in one call, so there is no window between
        choosing a port and owning it in which something else can steal it. Callers
        that need the real address (tests, mainly) read it back off `self.address`.
        """
        if address.endswith(":0"):
            base = address.rsplit(":", 1)[0]
            port = self._socket.bind_to_random_port(base)
            return f"{base}:{port}"
        self._socket.bind(address)
        return address

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> PreviewSubscriber:
        self._thread = threading.Thread(target=self._run, name="preview-sub", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._socket.close(linger=0)

    def __enter__(self) -> PreviewSubscriber:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- reading -----------------------------------------------------------

    def latest(self, camera: str, *, now: float | None = None) -> Preview | None:
        """The freshest preview for one camera, or None if it is missing or stale."""
        with self._lock:
            preview = self._latest.get(camera)
        if preview is None or preview.age_seconds(now) > self._stale_after:
            return None
        return preview

    def is_live(self, camera: str, *, now: float | None = None) -> bool:
        return self.latest(camera, now=now) is not None

    def cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._latest)

    # -- the thread --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._receive_once()
            except Exception:
                logger.exception("preview receive failed")
                self._stop.wait(0.5)

    def _receive_once(self) -> None:
        if not self._socket.poll(timeout=POLL_TIMEOUT_MS):
            return
        _topic, raw_header, jpeg = self._socket.recv_multipart()
        header = PreviewHeader.decode(raw_header)
        with self._lock:
            self._latest[header.camera] = Preview(header=header, jpeg=jpeg, received_at=time.time())

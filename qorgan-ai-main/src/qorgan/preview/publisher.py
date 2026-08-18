"""Worker side of the preview bus.

Rate-limited, downscaled, encoded once. If nobody is watching, this costs almost
nothing; if a browser is slow, the worker does not care.

That last property is the point. In the legacy, a slow browser could not stall the
detector either -- that part was well built -- but the frames lived in a process-global
registry shared between the web app and the workers, which is why the web layer had to
import the worker module, which is why loading a web route pulled in torch (M-23).
A socket cuts the knot: the worker publishes into the void and never blocks.
"""

from __future__ import annotations

import time

import numpy as np
import zmq

from qorgan.config.common import PreviewSettings
from qorgan.logging_setup import get_logger
from qorgan.preview.protocol import PreviewHeader, topic

# **`cv2` IS IMPORTED LAZILY, INSIDE THE ONE FUNCTION THAT ENCODES.** The WEB process is a
# preview SUBSCRIBER and never publishes, but `qorgan/preview/__init__.py` re-exports both
# halves — so importing the subscriber dragged OpenCV, and OpenCV's system GL libraries,
# into the dashboard container for code it never runs. The worker imports it on the first
# frame exactly as before.

logger = get_logger(__name__)

# If the subscriber falls behind, drop old frames rather than queue them. A preview
# is only worth showing while it is current.
SEND_HIGH_WATER_MARK = 4


class PreviewPublisher:
    """One per worker process. Publishes for every camera that process owns.

    The PUB socket *connects* and the SUB socket *binds*, which is the inverse of the
    usual arrangement. That is deliberate: there are N worker processes and one web
    process, so binding on the subscriber means one fixed address instead of a port
    per worker, and workers can come and go as the supervisor restarts them without
    the web process caring.
    """

    def __init__(self, address: str, *, context: zmq.Context | None = None) -> None:
        # Port 0 means "bind any free port", and only the SUBSCRIBER can act on that --
        # it learns the port by binding it. A publisher is a different PROCESS and has no
        # way to discover a randomly chosen port, so `:0` here is always a misconfiguration.
        #
        # Without this it fails silently and expensively: the SUB binds some random port,
        # the PUB connects to port 0, no preview ever arrives, and nothing anywhere says
        # so. The operator sees a blank camera and starts debugging the camera.
        #
        # Tests bind the subscriber on :0 and pass the REAL address (subscriber.address)
        # here, which is exactly the supported path.
        if address.endswith(":0"):
            raise ValueError(
                f"preview publisher cannot connect to {address!r}: port 0 means 'any free "
                "port', which only the subscriber can resolve. Pass the address the "
                "subscriber actually bound (PreviewSubscriber.address), or configure a "
                "real port."
            )

        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.set_hwm(SEND_HIGH_WATER_MARK)
        self._socket.connect(address)
        self._address = address
        self._last_sent: dict[str, float] = {}
        self._seq: dict[str, int] = {}
        logger.info("preview publisher connected", extra={"address": address})

    def should_publish(self, camera: str, settings: PreviewSettings, now: float) -> bool:
        """Rate limit per camera, so a 25 fps stream does not become a 25 fps preview."""
        if not settings.enabled:
            return False
        last = self._last_sent.get(camera)
        return last is None or (now - last) >= (1.0 / settings.fps)

    def publish(
        self,
        camera: str,
        frame: np.ndarray,
        settings: PreviewSettings,
        *,
        status: str = "ok",
        measured_fps: float | None = None,
        now: float | None = None,
    ) -> bool:
        """Send one preview frame. Returns False if it was rate-limited away.

        `measured_fps` is what the caller has measured this stream to deliver, or None
        while it does not yet know. None is passed through as None rather than filled in
        from the configuration: the whole value of the number is that it is a measurement,
        and a default would make "we have not measured" indistinguishable from "we
        measured exactly what we configured".
        """
        moment = now if now is not None else time.time()
        if not self.should_publish(camera, settings, moment):
            return False

        jpeg = encode_preview(frame, settings)
        if jpeg is None:
            return False

        self._last_sent[camera] = moment
        self._seq[camera] = self._seq.get(camera, 0) + 1
        height, width = frame.shape[:2]
        header = PreviewHeader(
            camera=camera,
            seq=self._seq[camera],
            published_at=moment,
            width=width,
            height=height,
            status=status,
            measured_fps=measured_fps,
        )
        self._socket.send_multipart([topic(camera), header.encode(), jpeg], flags=zmq.NOBLOCK)
        return True

    def close(self) -> None:
        self._socket.close(linger=0)


def encode_preview(frame: np.ndarray, settings: PreviewSettings) -> bytes | None:
    """Downscale, then JPEG-encode. Once, here, not per browser client."""
    # AT THE TOP OF THE FUNCTION, not inside the branch below. A local `import` makes the
    # name local for the WHOLE function, so an import inside the `if` left `cv2.imencode`
    # further down referring to an unbound local whenever the frame did not need resizing —
    # an UnboundLocalError on the common path, which is exactly what the suite caught.
    import cv2  # lazy: see the note above the imports

    if frame.size == 0:
        return None

    height, width = frame.shape[:2]
    if width > settings.width:
        scale = settings.width / width
        frame = cv2.resize(
            frame, (settings.width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    return buffer.tobytes() if ok else None

"""RTSP reader: one background thread per camera, publishing decoded frames.

Two legacy defects are designed out rather than patched:

**Stale-frame reuse.** Legacy's `read()` returned the *same* frame object again if the
queue was empty and the frame was less than 0.35 s old. The detection loop therefore
processed one physical moment up to three times, and every temporal counter — contact
frames, overlap frames, persistence — inflated accordingly. Here a frame is handed out
exactly once. If there is no new frame, `read()` returns None and the caller waits.

**Silent death.** A worker thread must never die (rule R7). Every loop body here is
wrapped; an exception reconnects and logs, it does not end the thread.

The source is injected (`opener`), so the same reader drives a recorded clip -- see
`capture/source.py` and `capture/clip.py`. Only one thing changes for a finite source:
`reconnects=False` lets the thread END when the source runs out, which is the single
exception to the paragraph above and is not silent (`stats.finished`, and a log line).
"""

# **`cv2` IS IMPORTED LAZILY, INSIDE EACH FUNCTION THAT USES IT.** Importing this
# module must not require OpenCV: the dashboard process reaches these files through the
# CLI parser and through `qorgan.notify`, and it neither decodes nor encodes a frame.
# The same separation `INTEGRATION.md` §1 keeps between the web and the model stack, for
# the same reason — and it takes ~120 MB and two system GL libraries out of the
# dashboard's container. Workers import it on first use exactly as before.

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from queue import Empty, Queue

import numpy as np

from qorgan.capture.quality import QualityPolicy, is_corrupt
from qorgan.config.common import RtspSettings
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

BACKOFF_MAX = 30.0


class EndReason(StrEnum):
    """Why a connection ended. Only CORRUPT counts towards disabling the quality check."""

    STOPPED = "stopped"
    NOT_OPENED = "not_opened"
    READ_FAILED = "read_failed"
    CORRUPT = "corrupt"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SessionResult:
    frames: int
    reason: EndReason


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame. Handed to exactly one reader, exactly once."""

    image: np.ndarray
    seq: int
    captured_at: float  # time.monotonic(), for age; not a wall clock
    camera: str

    def age_seconds(self) -> float:
        return time.monotonic() - self.captured_at


@dataclass
class StreamStats:
    frames_published: int = 0
    frames_dropped: int = 0  # reader too slow; we keep the newest, not the oldest
    frames_corrupt: int = 0
    reconnects: int = 0
    connected: bool = False
    last_frame_at: float | None = None
    last_error: str | None = None
    quality_check_disabled: bool = False
    # The source ended and does not come back. Only ever true for a finite source (a
    # recorded clip with `at_end: stop`); a camera is never finished, it is disconnected.
    finished: bool = False


# Injectable so the tests can drive a fake camera without an RTSP server. The settings
# travel WITH the url rather than being captured at construction, so a fake opener sees
# exactly the timeouts production would have used and can be asserted on.
CaptureOpener = Callable[[str, RtspSettings], "cv2.VideoCapture"]


def open_rtsp(url: str, rtsp: RtspSettings) -> cv2.VideoCapture:
    """Open one RTSP stream, with a bounded wait.

    **The timeouts go in as CONSTRUCTOR PARAMETERS and must stay there.** The wait happens
    inside the constructor, so the obvious `capture.set(CAP_PROP_OPEN_TIMEOUT_MSEC, ...)`
    on the line after arrives once the wait is already over. Measured on this build
    (opencv-python 4.13.0) against an unroutable address: constructor with a 2000 ms open
    timeout returned in 2.20 s, the same constructor without one in 30.05 s, and `.set()`
    followed by `.open()` in 30.02 s -- i.e. the `.set()` bought nothing.

    That 30 s is the reason a worker could not be stopped: this call is where the reader
    thread was, and `self._stop` is not checked until it returns.
    """
    import cv2  # lazy — see the note above the imports
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [
            int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC),
            int(rtsp.open_timeout_seconds * 1000),
            int(cv2.CAP_PROP_READ_TIMEOUT_MSEC),
            int(rtsp.read_timeout_seconds * 1000),
        ],
    )
    # Keep the driver's own buffer tiny: we want the newest frame, not a backlog.
    # This one is a `.set()` on purpose -- it is not a timeout, nothing blocks on it, and
    # it has always worked here.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraStream:
    """Reads one camera in its own thread. Start it, read from it, stop it."""

    def __init__(
        self,
        camera: str,
        url_factory: Callable[[], str],
        rtsp: RtspSettings,
        *,
        quality: QualityPolicy | None = None,
        opener: CaptureOpener = open_rtsp,
        reconnects: bool = True,
    ) -> None:
        self.camera = camera
        # A factory, not a string: the URL carries credentials, so it is built on
        # demand and never stored on the instance where a repr could leak it.
        self._url_factory = url_factory
        self._policy = quality or QualityPolicy()
        # The whole RtspSettings, not three numbers lifted out of it. The reconnect delay,
        # the open timeout and the stop timeout are one budget for one camera, and lifting
        # them out one at a time is how a caller ends up passing a stop timeout that does
        # not match the open timeout it is waiting out.
        self._rtsp = rtsp
        self._reconnect_delay = rtsp.reconnect_delay_seconds
        self._opener = opener
        # Does this source come back? A camera does -- it is off the network for a while
        # and then it is not, which is why `_run` reconnects forever and why a thread that
        # gave up would take a camera off the air silently (R7). A recorded clip played to
        # its end does NOT come back, and reconnecting to it would either replay the same
        # children over and over or print a retry line every two seconds for the rest of
        # the process's life. Defaults to the camera's answer.
        self._reconnects = reconnects

        self._queue: Queue[Frame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self.stats = StreamStats()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> CameraStream:
        """Start the reader thread. **A CameraStream is single use.**

        A stopped stream refuses to restart, and that refusal is a fix, not a limitation.
        `stop()` used to drop `self._thread`, so a second `start()` sailed past the guard
        below -- and then handed back a thread that returned on its first line, because
        `self._stop` is never unset. `start()` succeeded, `is_alive()` said True for an
        instant, and the camera was silently unwatched from then on. Nothing on today's
        paths does this (the supervisor restarts whole processes), which is exactly why it
        would have gone unnoticed until the day something did. Build a new CameraStream.
        """
        if self._stop.is_set():
            raise RuntimeError(
                f"{self.camera}: this stream was stopped and cannot be restarted -- its "
                f"stop flag stays set, so the new thread would exit at once and leave the "
                f"camera unwatched with no error anywhere. Build a new CameraStream."
            )
        if self._thread is not None:
            raise RuntimeError(f"{self.camera}: already started")
        self._thread = threading.Thread(target=self._run, name=f"camera:{self.camera}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> bool:
        """Ask the reader thread to stop and wait for it. **True only if it really ended.**

        It RETURNS the failure rather than raising it. `worker/entrypoint.py` stops every
        camera and then closes the preview publisher from one `finally`; an exception out
        of here would skip the cameras after this one and the `publisher.close()` after
        them, so one unreachable camera would leak a ZeroMQ socket and leave the rest of
        the group running. A caller that must know is told, and a caller that must finish
        can finish -- which is why `stop_all` in `worker/camera_loop.py` collects these
        booleans and names the cameras rather than propagating anything.

        `self._thread` is deliberately NOT cleared. The old code cleared it whether or not
        the join worked, so the one object that knew the thread was still alive threw away
        its only handle on it, and every later `is_alive()` answered about nothing.

        A False here is a fact about the CAMERA, and on a group where several cameras are
        down it is the normal answer rather than a rare one: OpenCV serialises the FFmpeg
        open, so four readers with a 4 s open timeout come back at 4, 8, 12 and 16 s
        (measured), and no wait that fits the supervisor's 10 s grace collects them all.
        The thread is a daemon and the process is leaving; what must not happen is
        pretending it left with us.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True

        waited = self._rtsp.stop_timeout_seconds if timeout is None else timeout
        thread.join(timeout=waited)
        if not thread.is_alive():
            return True

        logger.error(
            "camera reader will not stop: it is still blocked inside the camera call",
            extra={
                # The camera NAME. Never the URL and never the factory's output: that
                # string carries the RTSP password (rule R4), which is why this class
                # holds a factory instead of a URL in the first place.
                "camera": self.camera,
                "waited_seconds": round(waited, 1),
                "consequence": (
                    "this worker's shutdown is late because this camera is not "
                    "answering. If the supervisor now reports 'worker ignored "
                    "terminate' for this group, the worker is not the fault -- the "
                    "camera named here is."
                ),
            },
        )
        return False

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        # A statement, never `return self.stop()`: a truthy __exit__ swallows the
        # exception the `with` body raised, and a clean stop would then hide it.
        self.stop()

    # -- reading -----------------------------------------------------------

    @property
    def finished(self) -> bool:
        """The source has ended AND everything it produced has been handed out.

        Both halves matter. `stats.finished` alone would be true while the last frame of
        the clip is still sitting in the queue, so a caller that stops on it would drop
        the frame -- and on a corpus run the last frame of the clip is as much evidence as
        the first. A camera never reaches this state at all.
        """
        return self.stats.finished and self._queue.empty()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """The next frame, or None if none arrived in time.

        Never returns a frame twice. A caller that falls behind sees the newest
        frame and misses the ones in between, which is the correct trade for a
        live detector: analysing a stale moment is worse than skipping it.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def _publish(self, image: np.ndarray) -> None:
        with self._lock:
            self._seq += 1
            frame = Frame(
                image=image, seq=self._seq, captured_at=time.monotonic(), camera=self.camera
            )
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.stats.frames_dropped += 1
            except Empty:  # pragma: no cover - lost the race, the slot is free now
                pass
        self._queue.put(frame)
        self.stats.frames_published += 1
        self.stats.last_frame_at = frame.captured_at

    # -- the thread --------------------------------------------------------

    def _run(self) -> None:
        """Connect, read until the feed breaks, reconnect. Forever, without dying."""
        backoff = self._reconnect_delay
        corrupt_endings = 0

        while not self._stop.is_set():
            try:
                result = self._session()
            except Exception as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("camera session failed", extra={"camera": self.camera})
                result = SessionResult(0, EndReason.ERROR)

            if self._stop.is_set():
                break

            self.stats.connected = False
            if not self._reconnects:
                self._finish(result)
                return

            self.stats.reconnects += 1

            # Only sessions the corruption check itself ended count towards giving up.
            if result.reason is EndReason.CORRUPT:
                corrupt_endings += 1
                self._maybe_disable_quality_check(corrupt_endings)
            else:
                corrupt_endings = 0

            backoff = self._reconnect_delay if result.frames else min(backoff * 2, BACKOFF_MAX)
            logger.warning(
                "camera disconnected, retrying",
                extra={
                    "camera": self.camera,
                    "reason": result.reason.value,
                    "backoff_seconds": round(backoff, 1),
                },
            )
            self._stop.wait(backoff)

    def _finish(self, result: SessionResult) -> None:
        """The source ended and does not come back. Say so once, and let the thread end.

        This is the ONLY way a reader thread here is allowed to stop other than being told
        to (R7: a thread that dies silently takes its camera off the air). It is not a
        failure being swallowed -- `reason` and `last_error` are still recorded and still
        logged, so a clip that ended because the file was missing is distinguishable from
        one that ended because it ran out of frames.
        """
        self.stats.finished = True
        logger.info(
            "frame source finished and will not be reopened",
            extra={
                "camera": self.camera,
                "reason": result.reason.value,
                "frames": result.frames,
            },
        )

    def _maybe_disable_quality_check(self, consecutive: int) -> None:
        """The check keeps ending sessions and the picture never improves, so the
        check is wrong about this camera. Switch it off rather than loop forever."""
        if not self._policy.enabled or self.stats.quality_check_disabled:
            return
        if consecutive < self._policy.reconnects_before_giving_up:
            return

        self._policy = replace(self._policy, enabled=False)
        self.stats.quality_check_disabled = True
        logger.error(
            "corruption check disabled after %d reconnects in a row: the picture is "
            "probably fine and simply dark or static. Analysing it as-is.",
            consecutive,
            extra={"camera": self.camera},
        )

    def _session(self) -> SessionResult:
        """One connection, from open to broken."""
        capture = self._opener(self._url_factory(), self._rtsp)
        try:
            if not capture.isOpened():
                self.stats.last_error = "could not open stream"
                return SessionResult(0, EndReason.NOT_OPENED)

            self.stats.connected = True
            self.stats.last_error = None
            return self._pump(capture)
        finally:
            capture.release()

    def _pump(self, capture: cv2.VideoCapture) -> SessionResult:
        import cv2  # lazy — see the note above the imports
        previous: np.ndarray | None = None
        corrupt_streak = 0
        delivered = 0

        while not self._stop.is_set():
            ok, image = capture.read()
            if not ok or image is None:
                self.stats.last_error = "read failed"
                return SessionResult(delivered, EndReason.READ_FAILED)

            if is_corrupt(image, previous, self._policy):
                self.stats.frames_corrupt += 1
                corrupt_streak += 1
                previous = image
                if corrupt_streak >= self._policy.corrupt_frames_before_reconnect:
                    self.stats.last_error = "stream appears frozen or blank"
                    return SessionResult(delivered, EndReason.CORRUPT)
                continue

            corrupt_streak = 0
            previous = image
            delivered += 1
            self._publish(image)

        return SessionResult(delivered, EndReason.STOPPED)

"""One camera's loop, inside a worker process.

Phase 1: read frames, publish previews, count. The detection pipeline lands in Phase 2
and hangs off `on_frame` -- the loop's shape does not change when it arrives.

Every loop body is wrapped. A camera thread that dies takes its camera off the air
silently, which is the failure mode the whole rewrite exists to prevent (rule R7).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from qorgan.capture import CameraStream, Frame, prepare_frame
from qorgan.capture.source import FrameSource, frame_source
from qorgan.config.camera import CameraConfig
from qorgan.config.common import fps_agrees
from qorgan.logging_setup import get_logger
from qorgan.preview import PreviewPublisher

logger = get_logger(__name__)

FrameHandler = Callable[[CameraConfig, Frame], str]

# How many delivered frames to watch before believing a rate. Too few and a burst of
# buffered frames on connect reads as a fast camera; this is a couple of seconds at any
# real rate, and the check runs once rather than continuously.
FPS_SAMPLE_FRAMES = 30

# How long this loop's OWN thread may take to notice a stop, on top of however long its
# reader takes. The loop thread blocks in exactly one place, `stream.read(timeout=1.0)`,
# and `self._stop` is set before the reader is stopped -- so it has the whole of the
# reader's stop to exit and this join normally returns instantly. 2 s is the allowance for
# an `on_frame` that is mid-inference when the signal lands.
LOOP_JOIN_SECONDS = 2.0

# FPS_TOLERANCE -- how far the delivered rate may sit from `capture.stream_fps` before it
# is worth saying out loud -- now lives in `config.common`, beside the field it judges, and
# is applied through `fps_agrees` below. It moved because it gained a second reader: the
# camera page draws a "config != stream" badge from the same constant. Two copies of that
# number is how a log line and a dashboard end up disagreeing about one camera, which is
# the shape of defect this codebase keeps finding. It is NOT re-exported here: a name kept
# alive for old readers is a second place to look.


def _no_detection(_camera: CameraConfig, _frame: Frame) -> str:
    """Phase 1 placeholder. Phase 2 replaces this with the detector."""
    return "ok"


class CameraLoop:
    """Owns one camera: its frame reader, its preview publishing, its counters.

    "Frame reader", not "RTSP reader": the reader is whatever `capture/source.py` says this
    camera's frames come from, which is the NVR unless the camera's config points it at a
    recording. Everything below this line is the same either way, and that is the whole
    reason the file source is worth having.
    """

    def __init__(
        self,
        camera: CameraConfig,
        publisher: PreviewPublisher,
        *,
        on_frame: FrameHandler = _no_detection,
        source: FrameSource | None = None,
    ) -> None:
        self.camera = camera
        self._publisher = publisher
        self._on_frame = on_frame
        # Where the frames come from is now ONE decision, made in `capture/source.py` from
        # the camera's own config, and passed all the way through to `CameraStream`.
        # `CameraStream` was always built to take an `opener` from outside -- and this loop
        # hardcoded the RTSP URL factory and never forwarded `opener` at all, so the seam
        # existed and dead-ended here. That is why a recorded clip could not be driven
        # through the production path, and why the path has never been run end to end.
        #
        # `source=` overrides the config for a caller that already has one; everything
        # about the frames after this line is identical either way, which is the point.
        self._source = source or frame_source(camera)
        self._stream = CameraStream(
            camera.name,
            url_factory=self._source.url_factory,
            # The WHOLE RtspSettings, not the reconnect delay lifted out of it. This line
            # was `reconnect_delay=camera.rtsp.reconnect_delay_seconds`; `CameraStream`
            # now needs the open and the stop timeout out of the same block, and takes
            # them together so that a caller cannot hand it a stop timeout that does not
            # match the open timeout it is waiting out. The reconnect delay is not lost --
            # `CameraStream` reads it off `rtsp` instead of being handed it here.
            #
            # It is passed for a FILE-backed camera too, and that is deliberate: `rtsp`
            # stays required on every camera (`config/camera.py`), and a clip being read
            # still has to stop inside the supervisor's grace like everything else.
            rtsp=camera.rtsp,
            opener=self._source.opener,
            reconnects=self._source.reconnects,
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_processed = 0

        # The delivered frame rate, measured rather than assumed. `capture.stream_fps` is
        # read off a screenshot of the camera's web UI; this is the only place in the system
        # that can check it against a real NVR, and on site it is the first thing that will
        # disagree with the configuration.
        self._first_frame_at: float | None = None
        self._rate_checked = False
        self.measured_fps: float | None = None

    def start(self) -> CameraLoop:
        logger.info(
            "camera loop starting",
            # `FrameSource.where`, which is `safe_url` for a camera and never `build_url`:
            # the real URL carries the password, and the legacy printed it into every log
            # file and drew it onto debug JPEGs (C-02). It also names the CLIP when this
            # camera is reading one -- a log that said `rtsp://...` while the frames came
            # off disk would be the operator's only clue, pointing the wrong way.
            extra={"camera": self.camera.name, "stream": self._source.where},
        )
        self._stream.start()
        self._thread = threading.Thread(
            target=self._run, name=f"loop:{self.camera.name}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> bool:
        """Stop this camera. **True only if both of its threads really ended.**

        The flag is set FIRST and the reader stopped second, on purpose: that gives this
        loop's own thread the whole of the reader's stop window to notice, so the join
        below almost always returns at once rather than adding to the budget.

        Neither thread handle is cleared. See `CameraStream.stop` -- an object that keeps
        the handle can still answer `is_alive()` honestly, and `start()` can still refuse.
        (`CameraLoop.start()` needs no guard of its own: it calls `CameraStream.start()`
        first, which raises before a second loop thread can be created.)
        """
        self._stop.set()
        stopped = self._stream.stop()
        if self._thread is not None:
            self._thread.join(timeout=LOOP_JOIN_SECONDS)
            if self._thread.is_alive():
                logger.error(
                    "camera loop will not stop",
                    extra={
                        "camera": self.camera.name,
                        # Worth printing here because "which stream is not answering" is
                        # the next question, and this answers it without leaking anything:
                        # `FrameSource.where` is `safe_url` for a camera (the real URL
                        # carries the password -- R4, C-02) and the CLIP PATH for a camera
                        # reading a file. This line used to read `safe_url(camera.rtsp)`,
                        # which was right when every camera was an NVR; it would now print
                        # an `rtsp://` host for a wedged file reader and send whoever is
                        # diagnosing the shutdown to the network instead of the disk.
                        "stream": self._source.where,
                        "waited_seconds": LOOP_JOIN_SECONDS,
                    },
                )
                stopped = False
        return stopped

    @property
    def connected(self) -> bool:
        return self._stream.stats.connected

    @property
    def finished(self) -> bool:
        """This camera's source has ended and every frame it produced has been processed.

        Only a finite source ever says yes -- a recorded clip with `at_end: stop`. A camera
        that is off the network is DISCONNECTED, which is a different thing that must never
        stop the worker: that is the failure the whole rewrite exists to prevent (R7).
        """
        return self._stream.finished

    def _run(self) -> None:
        skip = self.camera.capture.det_every
        while not self._stop.is_set():
            try:
                self._tick(skip)
            except Exception:
                logger.exception("camera loop failed", extra={"camera": self.camera.name})
                self._stop.wait(1.0)

    def _tick(self, skip: int) -> None:
        frame = self._stream.read(timeout=1.0)
        if frame is None:
            return

        self.frames_processed += 1
        self._observe_rate()

        # ONE preprocessing function, shared with the eval harness (qorgan.capture.frames).
        # Production used to hand YOLO whatever the NVR sent while the harness resized to
        # capture.frame_width x frame_height -- so a px/s threshold tuned on the bench was
        # denominated in different pixels from the one production compared against.
        # BullyingDetector was ALREADY constructed with frame_width/frame_height, so the
        # zone maths was in the camera's configured frame (1280x720 on the hall, per
        # hall.yaml -- NOT base.yaml's 960x540 default) while the boxes were in substream
        # pixels. There is no fleet-wide analysis resolution: it is per profile, which is
        # why this reads self.camera.capture and never a constant.
        prepared = replace(frame, image=prepare_frame(frame.image, self.camera.capture))

        # det_every is honoured here, for every camera type. The legacy shipped
        # det_every: 2 on the canteen cameras with a comment about halving GPU load,
        # and then never read the key.
        status = "ok"
        if prepared.seq % skip == 0:
            status = self._on_frame(self.camera, prepared)

        # The measured rate rides along with the frame. It was computed here and read
        # nowhere but a log line, so the one person who needs it -- whoever is looking at
        # the dashboard asking why this camera lags -- could not see it. The web process
        # cannot measure it: it receives previews at `preview.fps` (3/s), which is a rate
        # we chose and says nothing about what the NVR delivers.
        self._publisher.publish(
            self.camera.name,
            prepared.image,
            self.camera.preview,
            status=status,
            measured_fps=self.measured_fps,
        )

    def _observe_rate(self) -> None:
        """Measure what the stream really delivers, and say so once if it is not what the
        config claims.

        This is what stops `stream_fps` being the knob its predecessor was. `display_fps`
        was set in five files and read by nobody in production, so it could be wrong for
        the whole life of the project without anything noticing -- and it was, by 1.5x,
        which silently mis-scaled every per-second number the eval harness printed.

        Measured over the FIRST `FPS_SAMPLE_FRAMES` frames and then never again: this is a
        configuration check, not a health metric. A camera whose rate sags under load is a
        different problem with a different signal.
        """
        now = time.monotonic()
        if self._first_frame_at is None:
            self._first_frame_at = now
            return
        if self._rate_checked or self.frames_processed <= FPS_SAMPLE_FRAMES:
            return

        elapsed = now - self._first_frame_at
        if elapsed <= 0:
            return

        self._rate_checked = True
        # frames_processed counts the first frame, which started the clock rather than
        # taking any time; the interval count is one lower than the frame count.
        self.measured_fps = (self.frames_processed - 1) / elapsed

        configured = float(self.camera.capture.stream_fps)
        if fps_agrees(configured, self.measured_fps):
            return

        logger.warning(
            "stream_fps does not match what this camera delivers",
            extra={
                "camera": self.camera.name,
                "configured_fps": configured,
                "measured_fps": round(self.measured_fps, 2),
                # Said plainly, because the consequence is not obvious from two numbers:
                # every per-second figure the eval harness prints for this camera is scaled
                # by the configured value, and the detector runs at the measured one.
                "consequence": (
                    "eval rates and noise floors for this camera are denominated in "
                    "capture.stream_fps; correct it to the measured rate"
                ),
            },
        )


def stop_all(loops: Sequence[CameraLoop]) -> list[str]:
    """Stop every camera AT ONCE. Returns the cameras that did not stop.

    **Parallel because sequential does not fit.** The supervisor allows a worker
    `TERMINATE_GRACE_SECONDS = 10.0` to shut down; `bullying_stairs_yard` in
    config/workers.yaml holds FOUR cameras. One camera costs
    `rtsp.stop_timeout_seconds` (5.0 s at the shipped defaults) plus `LOOP_JOIN_SECONDS`,
    so stopping four one after another is up to 28 s against a 10 s grace -- the
    supervisor gives up, kills the process, and the group loses its clean heartbeat and
    its `publisher.close()`.

    Shrinking the timeouts does not rescue the sequential version: four cameras inside
    10 s needs under 2.5 s each, so an open timeout under about 1.5 s, which is too tight
    for an NVR under load and would trade a shutdown bug for a reconnect storm. Stopping
    them together makes the budget the cost of the SLOWEST camera instead of the sum, and
    -- the part that matters for the next person -- makes it independent of how many
    cameras a group has, so adding a fifth camera cannot quietly break shutdown again.

    Measured on this machine, four readers wedged on an unroutable address:
    sequential 20.02 s (over the grace), together 5.00 s (inside it). The 28 s above and
    the 20.02 s here are the CAP and the OBSERVED cost, not two estimates of one thing:
    `LOOP_JOIN_SECONDS` is in the cap and is almost never spent, because the loop thread
    is already gone by the time the reader's join expires.

    **What this does NOT do, and it is worth knowing before you change a timeout.**
    OpenCV SERIALISES the FFmpeg open: four threads opening at once, each with a 4000 ms
    open timeout, returned at 4.05 / 8.09 / 12.12 / 16.17 s, and eight of them at 4.05 s
    intervals all the way to 32.34 s. So when a whole group is unreachable, the readers
    cannot all finish inside any wait short enough to fit the grace, and `stop_all` will
    correctly report most of them as unstopped. That is the truth about the cameras, not
    a fault in the shutdown: the threads are daemons, the process is on its way out, and
    what the grace is really protecting -- the final heartbeat, the pipelines and
    `publisher.close()` -- is reached in time precisely because nothing waits for them.

    Failures are returned, not raised: the caller still has pipelines to stop and a socket
    to close. See `CameraStream.stop`.
    """
    if not loops:
        return []
    with ThreadPoolExecutor(max_workers=len(loops), thread_name_prefix="stop") as pool:
        outcomes = list(pool.map(_stop_one, loops))
    return [name for name, stopped in outcomes if not stopped]


def _stop_one(loop: CameraLoop) -> tuple[str, bool]:
    """One camera's stop, with its name, and never an exception.

    An exception here would come back out of `pool.map` and abandon the other cameras'
    results -- the same "one bad camera takes the shutdown down with it" shape that
    `stop()` returning instead of raising exists to prevent. The name is read BEFORE the
    try for the same reason: reading it inside the handler would let the handler raise.
    """
    name = loop.camera.name
    try:
        return name, loop.stop()
    except Exception:
        logger.exception("stopping this camera raised", extra={"camera": name})
        return name, False


def worst_case_stop_seconds(cameras: Iterable[CameraConfig]) -> float:
    """The longest `stop_all` can WAIT for these cameras.

    MAX, not SUM, because `stop_all` joins them at once -- and that is the whole reason
    this is a number worth printing at startup rather than discovering after a kill.

    It bounds the wait, NOT the life of the reader threads. Those are daemons, and on a
    group whose cameras are all unreachable some of them outlive this wait by design --
    OpenCV serialises the FFmpeg open (measured: four concurrent 4 s opens returned at
    4.05 / 8.09 / 12.12 / 16.17 s), so no wait that fits a 10 s grace could collect them
    all. They are reported and left; the interpreter ends them when the process exits.

    It covers the CAMERA LOOPS and nothing else. The rest of `_serve`'s shutdown is the
    detection pipelines and `publisher.close()`; the publisher is a `socket.close(linger=0)`
    and is free, and each pipeline's thread blocks at most 0.5 s (`queue.get(timeout=0.5)`
    in `worker/bullying.py`) plus one validation job, under a 5 s join cap that has never
    been measured against a wedged validator on this machine. Those stops are still
    sequential. The number is named here rather than folded in silently, because folding
    an unmeasured 5 s cap into a measured budget would make the budget look measured.
    """
    return max((c.rtsp.stop_timeout_seconds for c in cameras), default=0.0) + LOOP_JOIN_SECONDS

"""A recorded clip, read the way a camera is read: in real time, and never twice at once.

`cv2.VideoCapture` opens a file and an RTSP URL with the same call, which is why the seam
in `capture/stream.py` can carry a file at all. It does NOT make the two behave alike, and
the two differences are exactly the ones that would corrupt the analysis:

**A file decodes at memory speed.** A camera delivers at a frame rate. `CameraStream`
stamps every frame with `time.monotonic()`, and `worker/bullying.py` hands that stamp to
the detector, whose speeds are px per SECOND. Read a 15 fps clip flat out and consecutive
frames arrive a millisecond apart instead of 67 ms, so every measured speed is inflated by
~60x while every threshold in every profile stays where it was. The value is right where it
is produced and silently wrong one layer on -- R2, in the time dimension. So `PacedClip`
hands frames over at the rate the CONTAINER declares, and refuses to run if the container
will not say what that rate is.

**A file ends.** A camera does not. What should happen then is a real decision with two
defensible answers, so it is config (`ClipEnd`) and not a guess made here.

`open_clip` below is the function `identity/streams.py` used to define and `identity/cli.py`
used to call. It MOVED here rather than being written a second time, and there is now one
way in this tree to open a recorded file.
"""

# **`cv2` IS IMPORTED LAZILY, INSIDE EACH FUNCTION THAT USES IT.** Importing this
# module must not require OpenCV: the dashboard process reaches these files through the
# CLI parser and through `qorgan.notify`, and it neither decodes nor encodes a frame.
# The same separation `INTEGRATION.md` §1 keeps between the web and the model stack, for
# the same reason — and it takes ~120 MB and two system GL libraries out of the
# dashboard's container. Workers import it on first use exactly as before.

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


def open_clip(path: Path) -> cv2.VideoCapture:
    """The clip, or a loud failure. Never a handle that silently reads nothing."""
    import cv2  # lazy — see the note above the imports
    handle = cv2.VideoCapture(str(path))
    if not handle.isOpened():
        raise OSError(f"opencv could not open the clip {path}")
    return handle


def declared_fps(handle: cv2.VideoCapture, path: Path) -> float:
    """The frame rate the container declares, or a refusal.

    Guessing a rate here would put every per-second figure the detector produces into
    units nobody chose -- the same mistake `evaluation/video.py::clip_duration` refuses to
    make about a clip's length, for the same reason. A file whose header will not say how
    fast it was recorded cannot stand in for a camera, and saying so is cheaper than a run
    whose numbers are wrong in a way nothing prints.
    """
    import cv2  # lazy — see the note above the imports
    fps = handle.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not math.isfinite(fps):
        raise OSError(
            f"{path}: the container declares no usable frame rate (fps={fps}). A clip "
            "standing in for a camera must be handed over at the rate it was recorded "
            "at, or every px/s the detector measures is denominated in decode speed. "
            "Re-mux the file with a declared rate rather than picking one here."
        )
    return float(fps)


class PacedClip:
    """One clip, standing in for one camera's stream.

    Duck-types the part of `cv2.VideoCapture` that `CameraStream` uses -- `isOpened`,
    `read`, `release` -- so nothing downstream needs to know which kind of source it has.
    """

    def __init__(
        self,
        handle: cv2.VideoCapture,
        fps: float,
        *,
        camera: str,
        loop: bool,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import cv2  # lazy — see the note above the imports
        self._handle = handle
        self._camera = camera
        self._loop = loop
        # Injected so a test can pin the pacing without spending real seconds on it, and
        # so what the pacing DID can be asserted rather than inferred from a stopwatch.
        self._sleep = sleep
        self._clock = clock

        self.fps = fps
        self._period = 1.0 / fps
        self._due: float | None = None
        self.loops = 0
        self.late_frames = 0

    def isOpened(self) -> bool:
        return bool(self._handle.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        """The next frame, no sooner than the clip's own rate allows."""
        image = self._decode()
        if image is None:
            return False, None
        self._pace()
        return True, image

    def release(self) -> None:
        self._handle.release()

    # -- the two ways a file is not a camera -------------------------------

    def _decode(self) -> np.ndarray | None:
        ok, image = self._handle.read()
        if ok and image is not None:
            return image
        if not self._loop:
            return None

        self._rewind()
        ok, image = self._handle.read()
        # An empty (or unseekable) clip: looping cannot conjure a frame, and retrying
        # would spin the reader thread at full speed forever. End the source instead.
        return image if ok and image is not None else None

    def _rewind(self) -> None:
        import cv2  # lazy — see the note above the imports
        self._handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.loops += 1
        if self.loops == 1:
            logger.info(
                "clip reached its end and started over",
                extra={"camera": self._camera, "clip_fps": round(self.fps, 2)},
            )

    def _pace(self) -> None:
        """Hand frames over `1/fps` apart on the wall clock, and never build up a debt.

        The schedule is absolute, not "sleep 1/fps after each decode": decoding costs real
        time, and adding a fixed sleep on top would deliver every frame slightly late and
        drift further with every one of them. `_due` moves by exactly one period per frame,
        so the handovers stay a period apart whatever the decode cost.

        When the decoder cannot keep up at all, the debt is FORGIVEN rather than repaid:
        `_due` is reset to now. Repaying it would fire off a burst of frames with no sleep
        between them the moment the machine caught up -- a burst whose timestamps say they
        are milliseconds apart, which is the very thing this method exists to prevent.
        """
        now = self._clock()
        if self._due is None:  # the first frame started the clock; it did not take time
            self._due = now + self._period
            return

        delay = self._due - now
        if delay > 0:
            self._sleep(delay)
            self._due += self._period
            return

        self._due = now + self._period
        self._fell_behind()

    def _fell_behind(self) -> None:
        """Said out loud, once: the pacing promise is the reason the numbers mean anything.

        A machine that cannot decode this clip in real time is handing the detector frames
        further apart than they were recorded, so every speed it measures is LOW by however
        far behind it is. Silently limping is what would make that invisible.
        """
        self.late_frames += 1
        if self.late_frames == 1:
            logger.warning(
                "cannot decode this clip in real time; frames are reaching the detector "
                "later than they were recorded, so every px/s it measures is understated",
                extra={"camera": self._camera, "clip_fps": round(self.fps, 2)},
            )


def open_paced_clip(path: Path, *, camera: str, loop: bool) -> PacedClip:
    """A clip, opened and wrapped so that it behaves like the camera it stands in for."""
    handle = open_clip(path)
    try:
        fps = declared_fps(handle, path)
    except OSError:
        handle.release()
        raise
    return PacedClip(handle, fps, camera=camera, loop=loop)

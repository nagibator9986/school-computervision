"""The last few seconds of one camera, small enough to keep for every camera at once.

An event's clip has to cover the moment *before* the alarm. By the time the slow tier has
judged a candidate the shove is already over, so the frames must be kept as they arrive --
before anybody knows whether they will matter. That is exactly the shape of the legacy's
worst memory defect: a single suppressed burst pinned ~460 MB of 1280x720 frames that were
never freed (rule R8). A buffer built for this job is the most likely place to rebuild it.

So it is bounded **twice**, and the two bounds answer different questions:

  * a **frame count**, derived from `bullying.clip_seconds` and the camera's analysis rate.
    It decides how much footage a clip may contain -- a policy about video;
  * a **byte budget**, which decides what that footage is allowed to cost. It is the bound
    that holds when the first one is wrong: a camera reconfigured to a larger analysis
    frame, a clip length raised to the schema's maximum, a stream delivering at twice its
    configured rate. A limit derived from configuration is not a limit, it is a hope about
    configuration, and this module refuses to depend on one.

And the frames are held **JPEG-encoded, at the resolution the worker analyses** -- never
the HD burst stream the legacy kept. Three seconds of 1280x720 at 15 fps is ~124 MB as
ndarrays and ~4 MB as JPEG, and that difference is whether six cameras in one worker cost
three quarters of a gigabyte or a few dozen megabytes. The cost is one encode per frame on
the capture thread, paid so that the worst case is survivable rather than merely unlikely.

Decoding is deliberately lazy (`decode`): the clip is streamed into the video writer one
frame at a time, so writing three seconds of footage never materialises three seconds of
raw frames.
"""

# **`cv2` IS IMPORTED LAZILY, INSIDE EACH FUNCTION THAT USES IT.** Importing this
# module must not require OpenCV: the dashboard process reaches these files through the
# CLI parser and through `qorgan.notify`, and it neither decodes nor encodes a frame.
# The same separation `INTEGRATION.md` §1 keeps between the web and the model stack, for
# the same reason — and it takes ~120 MB and two system GL libraries out of the
# dashboard's container. Workers import it on first use exactly as before.

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Iterator, Sequence

import numpy as np

from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

# Evidence for a human to look at, re-encoded into mp4 afterwards. 70 is what the preview
# publisher uses on the same pixels; higher buys detail this substream does not have.
CLIP_JPEG_QUALITY = 70

# **Per camera.** The largest worker group the planner will build is six cameras, so this
# is a ceiling of ~48 MB of frame buffer for a whole worker process, whatever the cameras
# are configured to do. It is a constant rather than a config key on purpose: it is a
# property of the machine this runs on, not of a camera, and a memory bound a YAML file can
# raise is a memory bound in name only.
CLIP_BUDGET_BYTES = 8 * 1024 * 1024


class ClipBuffer:
    """A camera's recent frames, JPEG-encoded, bounded by count AND by bytes.

    Written by the capture thread and read by the validation thread, so every mutation of
    the deque and its running byte total happens under one lock: the two must never be read
    apart, or the budget is enforced against a total that is briefly a lie.
    """

    def __init__(
        self, seconds: float, fps: float, *, budget_bytes: int = CLIP_BUDGET_BYTES
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.max_frames = max(1, math.ceil(seconds * self.fps))
        self.budget_bytes = budget_bytes
        self.evicted_over_budget = 0

        self._frames: deque[tuple[float, bytes]] = deque()
        self._nbytes = 0
        self._lock = threading.Lock()

    def append(self, timestamp: float, image: np.ndarray) -> None:
        """Keep this frame, and drop whatever no longer fits. Never raises: a frame that
        cannot be encoded is one frame missing from a clip, not a blind camera."""
        payload = _encode(image)
        if payload is None:
            return

        with self._lock:
            self._frames.append((timestamp, payload))
            self._nbytes += len(payload)

            while len(self._frames) > self.max_frames:
                self._drop_oldest()
            # The budget outranks the frame count, and keeps one frame back so that a
            # pathologically large single frame empties the buffer rather than the buffer
            # emptying itself forever.
            while len(self._frames) > 1 and self._nbytes > self.budget_bytes:
                self._drop_oldest()
                self.evicted_over_budget += 1

    def window(self, until: float) -> list[bytes]:
        """The buffered frames up to and including `until`, oldest first.

        Bounded by construction: it can only ever return what the buffer holds, which is
        `max_frames` frames and `budget_bytes` of memory.
        """
        with self._lock:
            return [payload for stamp, payload in self._frames if stamp <= until]

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def _drop_oldest(self) -> None:
        _, payload = self._frames.popleft()
        self._nbytes -= len(payload)


def decode(payloads: Sequence[bytes]) -> Iterator[np.ndarray]:
    """Encoded frames back to pixels, ONE AT A TIME.

    A generator rather than a list because the caller writes them straight into a video
    writer: three seconds of 1280x720 is ~124 MB if it is materialised and one frame if it
    is not. Undecodable frames are skipped -- a clip missing a frame is still evidence.
    """
    import cv2  # lazy — see the note above the imports
    for payload in payloads:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("a buffered clip frame could not be decoded")
            continue
        yield image


def _encode(image: np.ndarray) -> bytes | None:
    """NOT cv2.imwrite, and not a file: this never touches the disk. `imencode` on bytes we
    hold ourselves is also what `events.recorder` does, and for the same reason -- OpenCV's
    path-taking calls fail silently on the Cyrillic paths this school uses."""
    import cv2  # lazy — see the note above the imports
    if image.size == 0:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, CLIP_JPEG_QUALITY])
    if not ok:
        return None
    return bytes(buffer.tobytes())

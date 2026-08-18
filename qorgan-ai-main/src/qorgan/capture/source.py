"""Where one camera's frames come from. One decision, made in one place.

`CameraStream` was already built to take its source from outside -- "Injectable so the
tests can drive a fake camera without an RTSP server" -- but `CameraLoop` hardcoded the
RTSP URL factory and never passed `opener` on, so the seam existed and stopped one layer
short of anything that could use it. This module is the missing half: it answers "what
does this camera read from?" from the camera's own config, and `CameraLoop` asks.

**The credential property is preserved exactly.** The RTSP URL is still built by a factory
called on each connect and is still never stored anywhere a repr could reach it. `where`
is the only string here that is ever logged, and for RTSP it is `safe_url` -- the redacted
one -- because the legacy printed the password into every log file and drew it onto debug
JPEGs an unauthenticated UI served (audit C-02, rule R4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qorgan.capture.clip import open_paced_clip
from qorgan.capture.stream import CaptureOpener, open_rtsp
from qorgan.config.camera import CameraConfig
from qorgan.config.source import FileSource
from qorgan.enums import ClipEnd
from qorgan.rtsp import build_url, safe_url


@dataclass(frozen=True, slots=True)
class FrameSource:
    """Everything `CameraStream` needs in order to open one camera, and nothing else."""

    # A factory, not a string. For RTSP the URL carries credentials, so it is built on
    # demand and never stored -- the property `capture/stream.py` documents and this must
    # not weaken. For a file it is simply the path, built the same way so there is one
    # shape of source and not two.
    url_factory: Callable[[], str]
    opener: CaptureOpener

    # Safe to log, always. For RTSP this is `safe_url`, never `build_url`.
    where: str

    # Does this source come back after a session ends? A camera does; a clip played to its
    # end with `at_end: stop` does not, and retrying it forever would bury the end of the
    # run under a reconnect message every two seconds.
    reconnects: bool


def frame_source(camera: CameraConfig) -> FrameSource:
    """This camera's frames: its NVR, or the recording standing in for it."""
    clip = camera.source
    if clip is None:
        return _from_the_nvr(camera)
    return _from_a_file(camera.name, clip)


def _from_the_nvr(camera: CameraConfig) -> FrameSource:
    """The fleet's real source, byte for byte what `CameraLoop` used to build inline."""
    return FrameSource(
        url_factory=lambda: build_url(camera.name, camera.rtsp),
        opener=open_rtsp,
        where=safe_url(camera.rtsp),
        reconnects=True,
    )


def _from_a_file(name: str, clip: FileSource) -> FrameSource:
    """A recorded clip, paced to its own frame rate.

    `reconnects` follows `at_end`, and the two say the same thing from opposite ends. A
    LOOPING clip never ends a session by running out -- `PacedClip` rewinds inside `read()`
    -- so reconnecting is only ever about a genuine failure, and a source that starts over
    is one that comes back. A STOPPING clip is finished when it is finished.
    """
    path = Path(clip.path)
    loop = clip.at_end is ClipEnd.LOOP
    return FrameSource(
        url_factory=lambda: str(path),
        # `_rtsp` is accepted and ignored, and both halves of that are deliberate.
        #
        # ACCEPTED because `CaptureOpener` now takes the camera's `RtspSettings` alongside
        # the URL: the open and read timeouts have to reach `cv2.VideoCapture` as
        # CONSTRUCTOR parameters, since the wait happens inside the constructor and a
        # `.set()` afterwards arrives too late (see `capture.stream.open_rtsp`). Every
        # opener therefore has this shape, or `_session` raises TypeError on the first
        # connect -- which is precisely how this line was caught.
        #
        # IGNORED because those timeouts bound a NETWORK wait. A file is opened off the
        # local disk and there is no unroutable address to be wedged against, so honouring
        # `open_timeout_seconds` here would invent a deadline for a wait that does not
        # happen. `at_end`, which is a property of the clip, is what governs this source.
        opener=lambda target, _rtsp: open_paced_clip(Path(target), camera=name, loop=loop),
        where=f"{path.as_posix()} (at end: {clip.at_end.value})",
        reconnects=loop,
    )

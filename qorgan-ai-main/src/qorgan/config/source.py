"""Where a camera's frames come from, when they do not come from the camera.

RTSP has never been opened from any machine this project has run on, and the school has
657 recorded clips from its own hall cameras and 14 from the canteen. Without this block
the whole live path -- detection, event, snapshot, preview, panel -- cannot be exercised
end to end at all, and the seam that was designed for it (`CaptureOpener` in
`capture/stream.py`) stopped at `CameraLoop`, which hardcoded the RTSP URL factory.

Setting `source:` on a camera swaps ONLY where the frames come from. Everything after the
decode is the production path, unchanged and unaware: the same `prepare_frame`, the same
detector, the same thresholds, the same preview publisher. That is the whole point -- a
demonstration that took a different code path would demonstrate the wrong code.

Absent (the default, and what the fleet ships with) means RTSP.
"""

from __future__ import annotations

from pydantic import Field

from qorgan.config.common import Base
from qorgan.enums import ClipEnd


class FileSource(Base):
    """A recorded clip standing in for one camera's stream."""

    # Where the recording is. Relative paths resolve against the process's working
    # directory, exactly as every other path in this system does.
    #
    # NOT a directory or a glob: one camera is one stream, and a "play this folder"
    # option would quietly decide an ordering and a join between clips that nobody chose.
    path: str = Field(min_length=1)

    # Loop for a demonstration, stop for an honest run over the corpus. See `ClipEnd`:
    # the choice is real and it is config, not a guess buried in the reader.
    at_end: ClipEnd = ClipEnd.LOOP

    # There is deliberately NO "play as fast as you can" switch, and no fps override.
    #
    # A file decodes as fast as the CPU allows; a camera delivers in real time. The
    # detector times everything off `Frame.captured_at`, a wall clock -- speeds are px per
    # SECOND, and `worker/bullying.py` hands `frame.captured_at` straight to the detector.
    # Hand it an unpaced file and every speed is multiplied by however fast this machine
    # happens to decode, while every threshold stays where it was. The number would be
    # right in the decoder and silently wrong one layer later, which is this codebase's
    # signature defect (R2). So `capture/clip.py` paces the file at the rate the CONTAINER
    # declares, always, and there is no key here to switch that off.
    #
    # Nor is the rate taken from `capture.stream_fps`: pacing a 25 fps recording at a
    # configured 15 would stretch the video's own time by 1.67x and scale every px/s with
    # it -- the same defect, wearing the config's clothes. The container's rate is the
    # only one that makes a wall-clock second equal a second of the scene that was filmed.

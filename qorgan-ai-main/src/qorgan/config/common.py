"""Config building blocks shared by every camera type."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

FORBID = ConfigDict(extra="forbid", frozen=True)

# How far the DELIVERED frame rate may sit from `CaptureSettings.stream_fps` before the
# difference is worth telling somebody about. Generous on purpose: an NVR does not hold an
# exact rate, and a check that fires on everything gets filtered out by the person reading
# it, which is the same as not having one. 15 vs 10 is a factor of 1.5 and must be caught;
# 15 vs 14.2 must not.
#
# It lives beside the field it judges, and NOT in the worker that first needed it, because
# there are now two readers: `worker/camera_loop.py` writes the log line and
# `web/routes/cameras.py` draws the badge. Two copies of this number is how the log and the
# dashboard end up disagreeing about the same camera -- the exact shape of defect this
# codebase keeps finding.
FPS_TOLERANCE = 0.25

# Slack between the longest thing a reader thread can be blocked on and how long we wait
# for it to notice a stop. It covers what happens AFTER the blocking call returns --
# release the capture, unwind the session, re-check the flag -- and nothing more. It is
# not a safety cushion for an unknown wait; every wait it covers is bounded by a key
# below. See `RtspSettings.stop_timeout_seconds`.
STOP_MARGIN_SECONDS = 1.0


def fps_agrees(configured: float, measured: float) -> bool:
    """Is the delivered rate close enough to the configured one to call them the same?"""
    return abs(measured - configured) <= configured * FPS_TOLERANCE


class Base(BaseModel):
    """Every config model forbids unknown keys — a typo is a startup error, not a default (R10)."""

    model_config = FORBID


class ZoneRect(Base):
    """A rectangle in normalised frame coordinates, origin top-left."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> ZoneRect:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"zone must have x1<x2 and y1<y2, got {self!r}")
        return self

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class RtspSettings(Base):
    """Where the stream is. Credentials are NOT here — they come from the environment (R4)."""

    host: str
    port: int = Field(default=554, ge=1, le=65535)
    # Continuous analysis runs on the low-res substream.
    analysis_path: str = "/Streaming/Channels/102"
    # The HD main stream, opened briefly for evidence capture. None = no burst.
    burst_path: str | None = "/Streaming/Channels/101"
    reconnect_delay_seconds: float = Field(default=2.0, gt=0)
    # stream_queue_size DELETED. It claimed to bound a frame queue; no queue was ever
    # sized from it. `capture.stream` reads the newest frame and drops the rest, which
    # is what the key was pretending to arrange.

    # HOW LONG THE FFMPEG BACKEND MAY BLOCK. Both are read by `capture.stream.open_rtsp`,
    # which is the only place in this tree that constructs a VideoCapture on a camera.
    #
    # These are new keys, and the reason they had to exist is measured, not argued. On
    # this build (opencv-python 4.13.0), against an unroutable address:
    #
    #     VideoCapture(url, CAP_FFMPEG)                            30.05 s
    #     VideoCapture(url, CAP_FFMPEG, [OPEN_TIMEOUT_MSEC, 2000])  2.20 s
    #     VideoCapture(); set(OPEN_TIMEOUT_MSEC, 2000); open(url)  30.02 s
    #
    # The third line is the trap: the wait happens INSIDE the constructor, so a `.set()`
    # afterwards arrives thirty seconds late. The timeout has to travel as a CONSTRUCTOR
    # PARAMETER or it does nothing at all.
    #
    # Those 30 s were not merely slow. The reader thread sits inside that constructor and
    # never reaches its stop flag, so `CameraStream.stop()` waited, gave up, and returned
    # as though it had succeeded -- and the supervisor, whose grace is 10 s, killed the
    # worker and logged "worker ignored terminate" while the actual cause was a camera
    # that did not answer.
    #
    # 4 s each: a LAN camera answers a DESCRIBE in well under a second, and a stream that
    # has delivered nothing for four seconds (sixty frames at capture.stream_fps) is not
    # a stream having a hiccup. They are deliberately much smaller than the 10 s grace --
    # see `stop_timeout_seconds` and `worker.camera_loop.worst_case_stop_seconds`.
    open_timeout_seconds: float = Field(default=4.0, gt=0, le=30)
    read_timeout_seconds: float = Field(default=4.0, gt=0, le=30)

    @property
    def stop_timeout_seconds(self) -> float:
        """How long a reader thread may take to notice it has been told to stop.

        DERIVED, and deliberately not a fourth knob. A reader thread that is not stopping
        is blocked in exactly one of two places -- opening the stream, or waiting for the
        next packet on an open one -- and both are bounded above by the keys here. An
        independent key would let somebody set a stop timeout SHORTER than the block it
        exists to wait out, which is precisely how `stop()` came to report a success it
        had not achieved.
        """
        return max(self.open_timeout_seconds, self.read_timeout_seconds) + STOP_MARGIN_SECONDS


class CaptureSettings(Base):
    frame_width: int = Field(default=960, ge=160)
    frame_height: int = Field(default=540, ge=120)

    # What the NVR DELIVERS on the analysis channel (sub-stream 102) -- a fact about the
    # camera, not a policy about us. This replaced `display_fps`, which was a policy nobody
    # applied: the production loop never read it, so it moved only the eval harness's idea
    # of the rate, and the harness graded a 15 fps loop as though it ran at 10.
    #
    # 15 is what the camera's own web UI reports (1280x720 @ 15 fps, 2048 Kbps; handoff §2).
    # That is a screenshot, not a measurement this tree can re-run -- so the loop MEASURES
    # the delivered rate and warns when reality disagrees with this number
    # (`worker/camera_loop.py`). Settle it on site with `ffprobe` on a live RTSP pull, or an
    # ISAPI `Streaming/channels/102` query; then correct this key from what the log says.
    stream_fps: int = Field(default=15, ge=1, le=60)

    # Run detection on every Nth frame. Legacy shipped det_every: 2 on the canteen
    # cameras with a comment claiming it halved GPU load; the canteen worker never
    # read the key (audit: dead config). Here it is honoured by every worker.
    det_every: int = Field(default=1, ge=1, le=10)

    @property
    def analysis_fps(self) -> float:
        """The rate the DETECTOR sees: one frame in every `det_every` of what arrives.

        This is the single derivation of the analysis rate. The eval harness and every
        per-second figure it prints come through here, so the instrument is denominated in
        the same units as the thing it measures (R2, applied to the frame rate).
        """
        from qorgan.detection.constants import analysis_fps

        return analysis_fps(self.stream_fps, self.det_every)


class YoloSettings(Base):
    model: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"
    imgsz: int = Field(default=640, ge=320, le=1920)
    conf: float = Field(default=0.25, gt=0.0, lt=1.0)
    iou: float = Field(default=0.5, gt=0.0, lt=1.0)


class PreviewSettings(Base):
    """What the worker publishes to the web process over ZeroMQ.

    Never raw frames: JPEG-encoded once, downscaled, rate-limited.
    """

    enabled: bool = True
    fps: float = Field(default=3.0, gt=0, le=15)
    width: int = Field(default=640, ge=160, le=1920)
    jpeg_quality: int = Field(default=70, ge=30, le=95)


# DebugSettings DELETED, with `CameraBase.debug`. It offered save_frames /
# save_face_crops / draw_overlays; no code read any of the three, so switching them ON
# would have done nothing -- which makes it worse than absent. The legacy behaviour it
# described (writing ~8 JPEGs per tick, forever, with the RTSP password drawn on the
# image) is simply not ported, and there is nothing to switch off.

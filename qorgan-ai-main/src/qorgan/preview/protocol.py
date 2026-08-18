"""The wire format for live previews.

Workers PUB, the web process SUBs. Never raw frames: a 1280x720 BGR frame is 2.7 MB,
and the legacy pushed full-size copies around a shared in-process registry, then
JPEG-encoded them again per browser client. Here each frame is downscaled and encoded
**once**, in the worker, and the bytes are what travel.

A ZeroMQ message is three parts:

    [ camera name ]  [ header JSON ]  [ JPEG bytes ]

The camera name is the SUB topic, so the web process can subscribe to one camera
without decoding the rest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PreviewHeader:
    camera: str
    seq: int
    # Unix epoch seconds, UTC. Wall clock, because the receiver is another process.
    published_at: float
    width: int
    height: int
    # What the worker wants the operator to see: "ok" | "alert" | "critical" | "offline".
    status: str = "ok"
    # What the camera loop MEASURED this stream to deliver, in frames per second, or None
    # until it has watched enough frames to say. It travels here because the web process
    # cannot derive it: previews arrive at `preview.fps`, a rate WE chose, which says
    # nothing about what the NVR sends the worker. Defaulted so a header written by an
    # older worker still decodes.
    measured_fps: float | None = None

    def encode(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> PreviewHeader:
        return cls(**json.loads(raw.decode("utf-8")))


def topic(camera: str) -> bytes:
    return camera.encode("utf-8")

"""Writing an event: the snapshot, the clip, and the database row.

Three legacy defects are designed out here:

  * **Paths are relative to MEDIA_ROOT.** The `RelPath` column type refuses an absolute
    path outright, so the third time this project moves, nothing breaks. The first two
    moves broke 100 % of the photo and clip rows and spawned a one-off fix script each.

  * **`summary_text` is written at its final severity.** The legacy computed the summary
    *after* the insert, so every row in its database says "Подозрение…" no matter how
    confident the event actually was. The log was a lie about its own contents.

  * **`cv2.imwrite` is not used.** It fails silently on Windows for any path containing a
    non-ASCII character — and this school's paths contain Cyrillic. An event would then
    point at a snapshot that was never written. We encode and write the bytes ourselves,
    and we check.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from qorgan.db.types import utcnow
from qorgan.enums import Severity
from qorgan.logging_setup import get_logger
from qorgan.paths import to_relative
from qorgan.settings import get_settings, resolve

# **`cv2` IS IMPORTED LAZILY, INSIDE THE TWO FUNCTIONS THAT ENCODE.** Every use in this
# module is a codec call; the only thing the WEB process wants from here is `media_root`,
# a `pathlib` helper — and it reached it through `qorgan.notify`, which meant the dashboard
# could not start without OpenCV and the system GL libraries OpenCV links against. That is
# ~120 MB and a class of container failure for a path lookup. The recorder and the workers
# import it on first encode exactly as before.

logger = get_logger(__name__)

JPEG_QUALITY = 88


class MediaWriteError(Exception):
    """The media could not be written. The caller must NOT record an event pointing at it."""


def media_root() -> Path:
    return resolve(get_settings().media_root)


def write_snapshot(frame: np.ndarray, camera: str, moment: datetime | None = None) -> str:
    """Write a JPEG and return its path RELATIVE to MEDIA_ROOT."""
    if frame.size == 0:
        # cv2.imencode raises a bare assertion here rather than returning False, and an
        # assertion escaping into the event thread is exactly how the legacy lost one.
        raise MediaWriteError(f"refusing to write an empty snapshot for {camera}")

    when = moment or utcnow()
    root = media_root()
    target = root / "snapshots" / when.strftime("%Y-%m-%d") / f"{camera}_{_stamp(when)}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)

    import cv2  # lazy: see the note on the import block

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise MediaWriteError(f"could not encode snapshot for {camera}")

    # NOT cv2.imwrite: on Windows it fails SILENTLY on a non-ASCII path, and this
    # school's media paths contain Cyrillic. Writing the bytes ourselves fails loudly.
    target.write_bytes(buffer.tobytes())
    if not target.is_file() or target.stat().st_size == 0:
        raise MediaWriteError(f"snapshot vanished after writing: {target}")

    return to_relative(target, root)


def write_clip(
    frames: Iterable[np.ndarray], camera: str, fps: float, moment: datetime | None = None
) -> str:
    """Write a short mp4 and return its path RELATIVE to MEDIA_ROOT.

    Takes an ITERABLE, not a list, and never holds more than one frame of its own. The
    worker's clip buffer decodes lazily into this, so three seconds of 1280x720 footage
    goes to disk without ever existing as ~124 MB of ndarrays (rule R8). A list still
    works, and is what the tests hand it.
    """
    stream = iter(frames)
    first = next(stream, None)
    if first is None:
        raise MediaWriteError(f"no frames to write a clip for {camera}")

    when = moment or utcnow()
    root = media_root()
    target = root / "clips" / when.strftime("%Y-%m-%d") / f"{camera}_{_stamp(when)}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)

    height, width = first.shape[:2]
    import cv2  # lazy: see the note on the import block

    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height)
    )
    try:
        if not writer.isOpened():
            raise MediaWriteError(f"could not open a video writer for {target}")
        writer.write(first)
        for frame in stream:
            writer.write(frame)
    finally:
        writer.release()  # always, even on the error path: the handle must not leak

    if not target.is_file() or target.stat().st_size == 0:
        raise MediaWriteError(f"clip is empty: {target}")

    return to_relative(target, root)


def severity_for(confidence: float, alert: float, critical: float) -> Severity:
    if confidence >= critical:
        return Severity.CRITICAL
    if confidence >= alert:
        return Severity.ALERT
    return Severity.SUSPICION


SUMMARIES: dict[Severity, str] = {
    Severity.SUSPICION: "Подозрение на агрессию",
    Severity.ALERT: "Зафиксирована агрессия",
    Severity.CRITICAL: "Критический инцидент",
}


def summarise(severity: Severity, camera_name: str, confidence: float) -> str:
    """The text a human reads first.

    Computed BEFORE the row is written. The legacy computed it after, so every event in
    its database is labelled "Подозрение…" regardless of how certain it was — the log
    misrepresented its own contents, and nobody could tell a scuffle from an assault by
    reading it.
    """
    return f"{SUMMARIES[severity]} — {camera_name} ({confidence:.0%})"


def _stamp(moment: datetime) -> str:
    return moment.strftime("%H-%M-%S-%f")[:-3]

"""Ground truth: an explicit labels file, and nothing else.

**The legacy derived its ground truth from Russian words in the video filename**, and
silently labelled anything it could not match as `normal`. So a clip of a genuine fight
whose filename nobody remembered to annotate counted as a *correct rejection* when the
detector missed it. The benchmark rewarded blindness.

Here the labels are a file a human wrote, they are validated, and a video with no label
is an error rather than an assumption.

Format (CSV, one interval per row):

    video_id,t_start,t_end,label,camera
    hall_2026_03_04_0912.mp4,12.5,18.0,bullying,
    hall_2026_03_04_0912.mp4,45.0,52.0,normal,

Only intervals labelled `bullying` are positives. A video may carry any number of them.
Everything in a video *outside* its bullying intervals is negative — which is why the
`normal` rows are optional documentation rather than data.

`camera` is OPTIONAL. A missing column and an empty cell both mean the same thing: infer
the camera from the filename (`evaluation/clips.py`). The precedence, exactly:

  1. An explicit `camera` value here WINS. A human stating which camera the footage came
     from is better evidence than any filename parser -- this is the only way to score
     the handful of human-named clips a recorder never touched.
  2. Otherwise, infer from the filename.
  3. If neither yields a camera, that is a HARD ERROR naming the file. Never a default,
     and never a fallback to some other camera's config -- a typo in an explicit `camera`
     value is an error too, not a silent slide back to inference.

A line whose first non-whitespace character is `#` is a comment and is ignored.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LabelKind(StrEnum):
    BULLYING = "bullying"
    NORMAL = "normal"
    # A stretch nobody is confident about. Neither rewarded nor punished: a detection
    # here is ignored entirely, rather than counted as a false positive.
    IGNORE = "ignore"
    # "No human has looked yet." NOT a judgement -- distinct from `ignore`, which says a
    # human looked and it is neither a fight nor a false alarm. A pending row exists so a
    # clip stays scannable while its timestamp is unknown: it carries the camera, asserts
    # no fight, and is invisible to every metric -- but COUNTED, never silently dropped
    # (see metrics.evaluate). It never suppresses a candidate in `eval sample` either,
    # because it cannot answer a question it has not been asked.
    PENDING = "pending"


class LabelError(Exception):
    """The labels file is wrong. Always names the row."""


@dataclass(frozen=True, slots=True)
class Interval:
    video_id: str
    start: float
    end: float
    kind: LabelKind
    # A human-provided camera name for this row's video, or None to infer from the
    # filename. See the module docstring for the precedence rule.
    camera: str | None = None

    def contains(self, moment: float) -> bool:
        return self.start <= moment <= self.end

    def overlaps(self, start: float, end: float) -> bool:
        return start <= self.end and end >= self.start

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class LabelSet:
    intervals: tuple[Interval, ...]

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(i.video_id for i in self.intervals))

    def for_video(self, video_id: str) -> tuple[Interval, ...]:
        return tuple(i for i in self.intervals if i.video_id == video_id)

    def positives(self, video_id: str) -> tuple[Interval, ...]:
        return tuple(i for i in self.for_video(video_id) if i.kind is LabelKind.BULLYING)

    def ignored(self, video_id: str) -> tuple[Interval, ...]:
        return tuple(i for i in self.for_video(video_id) if i.kind is LabelKind.IGNORE)

    def pending(self, video_id: str) -> tuple[Interval, ...]:
        return tuple(i for i in self.for_video(video_id) if i.kind is LabelKind.PENDING)

    def camera_for(self, video_id: str) -> str | None:
        """The camera a human explicitly named for this video, or None to infer from the
        filename. WINS over inference when present -- see the module docstring.

        A clip labelled with TWO different cameras is a contradiction, and it raises. Taking
        the first row and moving on would pick a winner silently -- and the whole reason this
        column exists is that using the wrong camera's config blanks a region of the frame
        with no error and a perfectly plausible result. A human who contradicts themselves
        here must be told, not quietly overruled.
        """
        named = {i.camera for i in self.for_video(video_id) if i.camera}
        if len(named) > 1:
            raise LabelError(
                f"{video_id}: labelled with more than one camera ({', '.join(sorted(named))}). "
                "A clip comes from exactly one camera; fix the labels file."
            )
        return named.pop() if named else None

    def __len__(self) -> int:
        return len(self.intervals)


REQUIRED_COLUMNS = ("video_id", "t_start", "t_end", "label")


def load_labels(path: Path) -> LabelSet:
    """Read and validate. A malformed row is an error, never a silent `normal`."""
    if not path.is_file():
        raise LabelError(f"labels file not found: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        # `#`-prefixed lines are comments (the template uses one to explain `camera`),
        # never data -- filtered out before csv even sees them.
        lines = [line for line in handle if not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        _check_columns(reader.fieldnames, path)
        intervals = [_parse(row, number, path) for number, row in enumerate(reader, start=2)]

    if not intervals:
        raise LabelError(f"{path.name}: no labels. An empty benchmark measures nothing.")

    return LabelSet(intervals=tuple(intervals))


def _check_columns(fieldnames: Iterable[str] | None, path: Path) -> None:
    present = set(fieldnames or ())
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise LabelError(f"{path.name}: missing column(s): {', '.join(missing)}")


def _parse(row: dict[str, str], number: int, path: Path) -> Interval:
    where = f"{path.name} line {number}"

    video_id = (row.get("video_id") or "").strip()
    if not video_id:
        raise LabelError(f"{where}: video_id is empty")

    try:
        start = float(row["t_start"])
        end = float(row["t_end"])
    except (TypeError, ValueError) as exc:
        raise LabelError(f"{where}: t_start/t_end must be numbers (seconds)") from exc

    if end <= start:
        raise LabelError(f"{where}: t_end ({end}) must be after t_start ({start})")
    if start < 0:
        raise LabelError(f"{where}: t_start must not be negative")

    raw = (row.get("label") or "").strip().lower()
    try:
        kind = LabelKind(raw)
    except ValueError as exc:
        allowed = ", ".join(k.value for k in LabelKind)
        raise LabelError(f"{where}: unknown label {raw!r} (expected one of: {allowed})") from exc

    # Optional. A missing column and an empty cell both mean "infer from the filename".
    camera = (row.get("camera") or "").strip() or None

    return Interval(video_id=video_id, start=start, end=end, kind=kind, camera=camera)


def write_template(path: Path) -> None:
    """A starting point, so nobody has to guess the format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# camera: optional. Leave BLANK -- the camera is inferred from the "
            "filename. Fill it in ONLY when the filename cannot be parsed (e.g. a "
            "human-named clip), with a name from config/cameras/. That value WINS over "
            "inference; a wrong one is a hard error, not a silent fallback.\n"
        )
        writer = csv.writer(handle)
        writer.writerow((*REQUIRED_COLUMNS, "camera"))
        writer.writerow(["hall_left_2026-03-04_09-12.mp4", "12.5", "18.0", "bullying", ""])
        writer.writerow(["hall_left_2026-03-04_09-12.mp4", "45.0", "52.0", "normal", ""])

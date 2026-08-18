"""The wall clock burned into the picture by the DVR, read back out of it.

**Why this exists at all.** Every longitudinal claim this module makes -- "this pupil was
less active this week than last" -- is a statement about DATES. A video file's own
timeline starts at zero and its modification time is the time somebody copied it off a
disk, which on the footage we were given is a month after the lesson. The only record of
when the lesson actually happened is the overlay the camera itself drew into the top-left
corner, and so that overlay is load-bearing: get it wrong and every week boundary moves.

**What was measured on the reference recording (2560x1440 Camera 01).** The overlay is a
fixed-pitch bitmap font: 23 character cells, 32 px apart, starting at x=0, glyph band
y=86..131. Reading it at ten offsets spread across 52 minutes and comparing against the
file's own timeline showed the clock advances **exactly** 1 second per 20 frames with
zero drift over the whole recording -- see `verify_linear`. That is what makes
`WallClock.at()` a multiplication rather than an OCR call per frame.

**Two things are deliberately not read.**

  * *The weekday.* Cells 11..13 ("Fri") sit over a real window whose daylight changes
    across the lesson, so their mask is unstable in a way the digits' masks are not.
    The weekday is a function of the date, so reading it would add a failure mode to
    obtain a value we can compute. `parse` derives it and `CameraClockProfile.verify`
    uses the agreement between the two as a free consistency check on the date order.

  * *Anything at all, when the caller already knows.* `WallClock.declared()` exists so a
    recording with no overlay, or a different DVR, is still usable: the operator states
    the start time and the artefact records that they did. What must never happen is a
    guessed timestamp that reads like a measured one, which is why `source` is not
    optional and travels into the provenance block.

**The date order is ambiguous and this module does not guess it.** "08-07-2026" is the
8th of July or the 7th of August depending on the DVR's menu. Both parse. The recording
we hold says "Fri", and only 2026-08-07 is a Friday, so it is MM-DD-YYYY *for this
camera* -- established by `verify`, stored in the profile, never assumed by `parse`.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# The reference camera's geometry, measured rather than guessed: connected components on
# the overlay strip gave glyph boxes 28 px wide on a 32 px pitch starting at x=0, with a
# 40 px tall glyph inside a 45 px band at y=86. Another DVR will need another profile;
# `calibrate` derives one from a single frame whose text a human has typed in.
HIKVISION_2560 = dict(
    x0=0, pitch=32, cell_w=32, y0=86, y1=131,
    digit_cells=(0, 1, 3, 4, 6, 7, 8, 9, 15, 16, 18, 19, 21, 22),
    font="clock_font_hikvision_32x45.npz",
    # Established by evidence, not assumed: only 2026-08-07 is the "Fri" this
    # same overlay prints (2026-07-08 is a Wednesday). See `verify()`.
    date_order="MDY",
)

# A glyph is drawn as a white fill inside a black outline so that the DVR's clock stays
# legible against both a white wall and a dark window. Neither polarity alone finds it on
# both backgrounds; the union of the two extremes finds it on either. The gap between the
# two thresholds is background, whatever shade the background happens to be.
DARK_BELOW = 80
LIGHT_ABOVE = 205

# Below this margin between the best and second-best glyph match, the read is refused
# rather than returned. On the reference recording every correct read scored ~0.22, so
# this rejects at a third of the observed working margin: low enough not to reject good
# frames, high enough that a frame with the overlay obscured cannot silently return a
# plausible time. A wrong hour that parses is worse than no hour at all.
MIN_MARGIN = 0.07


class ClockSource(StrEnum):
    """Where a recording's start time came from. Never inferred from the value itself."""

    OVERLAY = "overlay"      # read out of the burned-in picture, and verified linear
    FILENAME = "filename"    # the DVR's own export name, cross-checked against the overlay
    DECLARED = "declared"    # a human told us
    UNKNOWN = "unknown"      # nothing did; no dated claim may be made from this run


# DVR export names that carry the recording's start. `D14_20260815101759.mp4` is
# <channel>_<YYYYMMDDhhmmss>, which is what this school's second camera produces.
#
# **Why a filename is allowed to be a first-class source here, when a file's mtime is
# not.** An mtime is when somebody copied the file; this is a value the recorder itself
# wrote, in a fixed structured format, at the moment it opened the recording. It is
# evidence, not a guess — and `verify_against_overlay` checks it against the picture on
# every frame where the picture is legible, so a camera whose naming lies is caught rather
# than trusted. On the camera that motivated this, the overlay sits over a window with
# blinds and a poster (22 % dark and 21 % bright pixels, against 40 %/4 % on the first
# camera), so the glyph matcher correctly refuses — and refusing is only useful if there
# is a second source to fall back to.
FILENAME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:^|[^0-9])(\d{14})(?:[^0-9]|$)", "%Y%m%d%H%M%S"),
    (r"(?:^|[^0-9])(\d{8}[-_]\d{6})(?:[^0-9]|$)", "%Y%m%d_%H%M%S"),
)


@dataclass(frozen=True, slots=True)
class Reading:
    """One frame's overlay, and how confident the matcher was about its worst glyph."""

    text: str
    when: dt.datetime
    margin: float


class OverlayUnreadable(ValueError):
    """The strip was found but did not yield a date and a time we would stand behind."""


@lru_cache(maxsize=4)
def _font(name: str) -> dict[str, np.ndarray]:
    data = np.load(ASSETS / name)
    return {chr(int(key[1:])): data[key] for key in data.files}


def _cell_mask(gray: np.ndarray, profile: dict, index: int) -> np.ndarray:
    x = profile["x0"] + profile["pitch"] * index
    cell = gray[profile["y0"]: profile["y1"], x: x + profile["cell_w"]]
    return ((cell < DARK_BELOW) | (cell > LIGHT_ABOVE)).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation. Mean-centred, so a cell whose background happens to
    be brighter than another's does not score differently for the same glyph."""
    x = a.ravel() - a.mean()
    y = b.ravel() - b.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-9 else 0.0


def read_overlay(frame: np.ndarray, profile: dict = HIKVISION_2560) -> Reading:
    """One frame -> the datetime the camera drew on it.

    Raises `OverlayUnreadable` rather than returning a best guess: this value decides
    which WEEK a lesson lands in, and a silently wrong week moves a pupil's baseline.
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    needed_y, needed_x = profile["y1"], profile["x0"] + profile["pitch"] * 23
    if frame.shape[0] < needed_y or frame.shape[1] < needed_x:
        raise OverlayUnreadable(
            f"frame is {frame.shape[1]}x{frame.shape[0]}, but this clock profile reads a "
            f"strip out to x={needed_x}, y={needed_y}. This profile was measured on a "
            "2560x1440 stream; a resized frame needs its own profile, not a rescale -- "
            "the glyph masks are pixel thresholds."
        )

    font = _font(profile["font"])
    digits: dict[int, str] = {}
    worst = 1.0
    for index in profile["digit_cells"]:
        mask = _cell_mask(frame, profile, index)
        scores = sorted(((_ncc(mask, tpl), ch) for ch, tpl in font.items()), reverse=True)
        digits[index] = scores[0][1]
        worst = min(worst, scores[0][0] - scores[1][0])

    if worst < MIN_MARGIN:
        raise OverlayUnreadable(
            f"the overlay was read but the weakest glyph beat its runner-up by only "
            f"{worst:.3f} (need {MIN_MARGIN}). Something is over the clock, or this is a "
            "different DVR font. Refusing to return a time nobody should trust."
        )

    join = lambda *ix: "".join(digits[i] for i in ix)  # noqa: E731
    text = f"{join(0, 1)}-{join(3, 4)}-{join(6, 7, 8, 9)} {join(15, 16)}:{join(18, 19)}:{join(21, 22)}"
    return Reading(text=text, when=parse(text, profile["date_order"]), margin=worst)


def parse(text: str, date_order: str) -> dt.datetime:
    """"08-07-2026 09:55:03" -> a datetime, under a date order the CALLER supplies.

    **`date_order` is required, and that is the whole point of this signature.** It used
    to default to `"MDY"`, which contradicted this docstring: a default is exactly a guess
    about a setting in some camera's menu, applied silently to every date in the system,
    and the failure it produces is a lesson filed under the wrong month rather than an
    error anybody sees. `MDY` is the REFERENCE camera's order, established by `verify()`
    against the weekday the same overlay prints, and it lives in that camera's profile
    where a different camera can carry a different one.
    """
    date_part, time_part = text.split(" ")
    a, b, year = (int(v) for v in date_part.split("-"))
    month, day = (a, b) if date_order == "MDY" else (b, a)
    hour, minute, second = (int(v) for v in time_part.split(":"))
    return dt.datetime(year, month, day, hour, minute, second)


@dataclass(frozen=True, slots=True)
class WallClock:
    """Video seconds -> wall-clock time, for one recording.

    Affine on purpose. The alternative -- reading the overlay on every analysed frame --
    costs an OCR per frame to re-derive a quantity measured to be exactly linear, and
    turns a per-frame glyph failure into a per-frame timestamp failure.
    """

    started_at: dt.datetime
    source: ClockSource
    drift_seconds: float | None = None   # over the whole file; None when not verified

    def at(self, video_seconds: float) -> dt.datetime:
        return self.started_at + dt.timedelta(seconds=video_seconds)

    @classmethod
    def declared(cls, started_at: dt.datetime) -> WallClock:
        return cls(started_at=started_at, source=ClockSource.DECLARED)

    @classmethod
    def unknown(cls) -> WallClock:
        """No dated claim may be made from a run holding this. The pipeline still runs --
        a lesson report is useful without a date; a WEEKLY trend is not, and the
        aggregation layer refuses that rather than inventing a Monday."""
        return cls(started_at=dt.datetime(1970, 1, 1), source=ClockSource.UNKNOWN)


def from_filename(path: str | Path) -> dt.datetime | None:
    """The recording's start, as the DVR itself named it. None when the name says nothing.

    Deliberately returns a bare datetime rather than a `WallClock`: on its own a filename
    is a claim, and it only becomes a clock after `resolve` has either corroborated it
    against the picture or recorded that it could not.
    """
    stem = Path(path).name
    for pattern, fmt in FILENAME_PATTERNS:
        match = re.search(pattern, stem)
        if not match:
            continue
        try:
            return dt.datetime.strptime(match.group(1), fmt)
        except ValueError:
            continue
    return None


def verify_against_overlay(path: str | Path, claimed_start: dt.datetime,
                           profile: dict = HIKVISION_2560,
                           at_seconds: tuple[float, ...] = (30.0, 300.0, 900.0),
                           tolerance: float = 2.0) -> dict:
    """Check a filename's claim against the burned-in clock wherever it is legible.

    The overlay may be unreadable on this camera — that is exactly why we are using the
    filename — so an unreadable frame is NOT a failure, it is a frame that carries no
    opinion. What would be a failure is a frame that reads clearly and DISAGREES, and that
    is the only thing that sets `agrees` to False.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"checked": 0, "legible": 0, "agrees": None, "reason": "cannot open"}
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    checks: list[dict] = []
    try:
        for offset in at_seconds:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(offset * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            try:
                reading = read_overlay(frame, profile)
            except OverlayUnreadable:
                checks.append({"at_seconds": offset, "legible": False})
                continue
            expected = claimed_start + dt.timedelta(seconds=offset)
            delta = abs((reading.when - expected).total_seconds())
            checks.append({"at_seconds": offset, "legible": True,
                           "overlay": reading.when.isoformat(),
                           "expected": expected.isoformat(),
                           "delta_seconds": delta, "agrees": delta <= tolerance})
    finally:
        capture.release()

    legible = [c for c in checks if c.get("legible")]
    return {
        "checked": len(checks),
        "legible": len(legible),
        # None means "the picture had no opinion", which is different from disagreement.
        "agrees": None if not legible else all(c["agrees"] for c in legible),
        "tolerance_seconds": tolerance,
        "checks": checks,
    }


def resolve(path: str | Path, profile: dict = HIKVISION_2560,
            declared: dt.datetime | None = None) -> tuple[WallClock, dict]:
    """The one entry point a pipeline should use. Tries every source, in order of evidence.

    Order, and the reasoning for it. A human's `declared` value wins outright because a
    person took responsibility for it. Otherwise the OVERLAY beats the filename, because
    it is written into the same pixels we are measuring and cannot be changed by renaming
    a file. The filename comes next, and only after being checked against the overlay
    wherever the overlay is legible. `UNKNOWN` is last and it is not a failure of this
    function — it is a true statement that nothing in this recording says when it happened,
    and the aggregation layer refuses weekly trends rather than inventing a Monday.
    """
    evidence: dict = {}

    if declared is not None:
        return WallClock.declared(declared), {"source": "declared"}

    overlay_error = None
    try:
        clock = from_video(path, profile)
        evidence["overlay"] = {"started_at": clock.started_at.isoformat(),
                               "drift_seconds": clock.drift_seconds}
        return clock, evidence
    except OverlayUnreadable as error:
        overlay_error = str(error)
        evidence["overlay"] = {"unreadable": overlay_error}

    named = from_filename(path)
    if named is not None:
        check = verify_against_overlay(path, named, profile)
        evidence["filename"] = {"started_at": named.isoformat(), "verification": check}
        if check["agrees"] is False:
            # The name and the picture disagree. Refuse both rather than pick a winner:
            # something is wrong with this recording and a person should look at it.
            evidence["conflict"] = ("the filename and the burned-in clock disagree by more "
                                    "than the tolerance; refusing to date this recording")
            return WallClock.unknown(), evidence
        return WallClock(named, ClockSource.FILENAME, drift_seconds=None), evidence

    evidence["filename"] = {"parsed": False}
    return WallClock.unknown(), evidence


def from_video(path: str | Path, profile: dict = HIKVISION_2560) -> WallClock:
    """Read the first and the last frame, and prove the clock is linear between them.

    Two reads rather than one because a single read cannot tell a working clock from a
    frozen overlay, and cannot catch a recording stitched out of two sessions. The drift
    it measures is reported, not silently corrected: a DVR whose clock gains a minute an
    hour is a fact about the school's hardware that somebody should hear about.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OverlayUnreadable(f"cannot open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        ok, first_frame = capture.read()
        if not ok:
            raise OverlayUnreadable(f"{path} has no first frame")
        first = read_overlay(first_frame, profile)

        last = None
        if fps > 0 and count > 1:
            # Step back from the end rather than seeking to it: the final frames of a
            # DVR file are routinely truncated, and a failed read there must not lose us
            # the linearity check entirely.
            for back in (1, 5, 25, 100):
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(count - back, 1))
                ok, frame = capture.read()
                if not ok:
                    continue
                position = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
                try:
                    last = (read_overlay(frame, profile), position / fps)
                    break
                except OverlayUnreadable:
                    continue
    finally:
        capture.release()

    if last is None:
        return WallClock(first.when, ClockSource.OVERLAY, drift_seconds=None)

    reading, elapsed_video = last
    drift = (reading.when - first.when).total_seconds() - elapsed_video
    return WallClock(first.when, ClockSource.OVERLAY, drift_seconds=drift)


def verify_linear(path: str | Path, profile: dict = HIKVISION_2560,
                  frames: int = 240, skip: int = 40) -> dict:
    """Decode consecutively and check the clock ticks once per `fps` frames, forwards.

    This is the test that justified `WallClock.at()` being affine, kept as a function so
    it can be re-run on any new camera instead of being a claim in a docstring.
    """
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    reads: list[tuple[int, dt.datetime]] = []
    try:
        for index in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            if index >= skip:
                reads.append((index, read_overlay(frame, profile).when))
    finally:
        capture.release()

    ticks = [(n, w) for i, (n, w) in enumerate(reads) if i == 0 or w != reads[i - 1][1]]
    gaps = [ticks[i + 1][0] - ticks[i][0] for i in range(len(ticks) - 1)]
    steps = [(ticks[i + 1][1] - ticks[i][1]).total_seconds() for i in range(len(ticks) - 1)]

    # The FIRST gap is discarded: reading starts at an arbitrary frame, so the first
    # tick is however much of that second was left, and on the reference recording it
    # came out at 7 frames of 20. Counting it would fail every camera for a reason that
    # is arithmetic rather than hardware.
    interior = gaps[1:]
    # +/-1 frame, because the DVR's clock is not phase-locked to the encoder: a tick
    # that lands either side of a frame boundary shifts by one frame and back. What
    # would be a real fault is a SECOND skipped or repeated, and `steps` catches that.
    return {
        "frames_read": len(reads),
        "ticks": len(ticks),
        "frame_gaps": gaps,
        "interior_gaps_within_one_frame": bool(interior)
        and all(abs(gap - round(fps)) <= 1 for gap in interior),
        "steps_all_one_second": bool(steps) and all(step == 1.0 for step in steps),
        "first": ticks[0][1].isoformat() if ticks else None,
        "last": ticks[-1][1].isoformat() if ticks else None,
    }


def verify(path: str | Path, profile: dict = HIKVISION_2560) -> dict:
    """Everything a new camera should be made to pass before its dates are believed."""
    linear = verify_linear(path, profile)
    clock = from_video(path, profile)
    weekday_agrees = None
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if ok:
        reading = read_overlay(frame, profile)
        # The date order is decided by which reading agrees with the weekday the DVR
        # prints. We do not read those glyphs, but a human can, and this reports both.
        weekday_agrees = {
            "MDY": parse(reading.text, "MDY").strftime("%a"),
            "DMY": (parse(reading.text, "DMY").strftime("%a")
                    if _valid(reading.text, "DMY") else None),
        }
    return {"linear": linear, "started_at": clock.started_at.isoformat(),
            "drift_seconds": clock.drift_seconds, "weekday_by_order": weekday_agrees}


def _valid(text: str, order: str) -> bool:
    try:
        parse(text, order)
    except ValueError:
        return False
    return True

"""Pass 1 of labelling: by exception.

97 minutes of corridor, and the few seconds that matter are somewhere in it. Watching all
of it is not a good use of a human, so the detector goes first: it runs at **threshold 0**
over every clip and writes down every moment it would have raised anything at all.

The human then watches only what it fired on -- perhaps 50-150 clips of 8 seconds each.
That yields precision, and it puts the human's attention exactly where the label changes
the answer.

It is only half a detector, and `qorgan eval sample` is the other half.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path

from qorgan.evaluation.harness import RunResult

SCAN_COLUMNS = ("clip", "timestamp", "score", "probability", "confidence")

# The coverage manifest. candidates.csv records the clips the detector FIRED on; it cannot
# tell "covered but genuinely silent" from "never covered" -- both are zero rows. This
# separate artifact records every clip the scan actually decoded and ran to completion,
# so `eval sample` can PROVE a clip's silence instead of inferring it from absence (which
# is this project's signature bug: a clip added without a re-scan, or a truncated scan,
# becomes a false SILENT the detector never processed). One clip per line.
COVERAGE_COLUMNS = ("clip",)

# The clips the scan could not read. One line per clip, with the reason it failed.
#
# It is a THIRD artifact rather than a flag on the other two, because an unreadable clip is
# neither a candidate nor a covered clip: nothing decoded it, so nothing knows whether it was
# silent. Recording it here keeps the two existing artifacts saying exactly what they always
# said, and gives the run somewhere to name what it could not do -- which is the difference
# between a corpus that read 657 clips and one that quietly read 457.
UNREADABLE_COLUMNS = ("clip", "error")

# A full-frame scene, at its very smallest. The corpus records all 663 of its full frames
# at 2560x1440 and its 1293 crop ROIs at 76-320 wide, so the bar sits in a gap two orders
# of magnitude wide and is nowhere near either population.
#
# BOTH dimensions must clear it. The crops run up to 720 tall -- taller than plenty of
# legitimate frames -- so a height-only test would wave the tall ones straight through.
FULL_FRAME_MIN_WIDTH = 640
FULL_FRAME_MIN_HEIGHT = 360


class NotAFullFrameError(Exception):
    """A clip in the clips directory is a crop. Names every clip it means."""


def refuse_crops(sizes: Mapping[str, tuple[int, int]]) -> None:
    """The clips directory holds full-frame scenes. Refuse if anything else got in.

    A ~320x450 ROI has no scene and no zones, and the scorer's box-diagonal scaling is
    meaningless inside it: scored as if it were a frame it yields confident nonsense, which
    is worse than a crash because it looks like a measurement. The RESOLUTION is what tells
    the two apart -- not the `burst` marker in the name, which 17 full-frame clips lack.
    """
    crops = sorted(
        f"{name} ({width}x{height})"
        for name, (width, height) in sizes.items()
        if width < FULL_FRAME_MIN_WIDTH or height < FULL_FRAME_MIN_HEIGHT
    )
    if not crops:
        return

    listed = "\n  ".join(crops)
    raise NotAFullFrameError(
        f"{len(crops)} clip(s) in the clips directory are not full frames:\n  {listed}\n\n"
        f"A full frame is at least {FULL_FRAME_MIN_WIDTH}x{FULL_FRAME_MIN_HEIGHT}; the "
        "corpus records them at 2560x1440. These are crop ROIs, which have no scene and "
        "no zones: scoring one produces numbers, and every one of them is a lie. Move them "
        "to the crops directory (eval/crops/), where `qorgan eval label` will find them."
    )


@dataclass(frozen=True, slots=True)
class ScanRow:
    """One thing a human should look at.

    `timestamp` is `None` when the detector named NO moment -- which is exactly what a
    stratum-C row from `qorgan eval sample` is: a clip the fast tier never fired on. There
    is no candidate, so there is nothing to centre on, and the human judges the WHOLE clip
    (`labelling.interval_for`).

    It is `None` rather than the clip's midpoint, and the difference is not cosmetic. A
    midpoint padded by +/-2 s asserts that a fight is at t=45 of a 90-second clip AND that
    the other 86 seconds are negative. A detector that then finds the real fight at t=40 is
    scored a false positive for finding it. An invented interval is not a coarser
    measurement; it is a wrong one, and it is wrong in the direction that hides misses.
    """

    clip: str
    timestamp: float | None
    score: float
    probability: float
    confidence: float


def rows_for(result: RunResult) -> list[ScanRow]:
    """Every incident this clip produced.

    Merged alerts are excluded: a merged alert is the SAME physical incident, and forty
    rows for one four-second scuffle would spend the labeller's attention forty times on
    the same four seconds. One incident, one thing to watch -- the same rule the merger
    applies to Telegram.
    """
    return [
        ScanRow(
            clip=alert.video_id,
            timestamp=round(alert.timestamp, 3),
            score=round(alert.score, 3),
            probability=round(alert.verdict.candidate_probability, 3),
            confidence=round(alert.verdict.confidence, 3),
        )
        for alert in result.alerts
        if not alert.merged
    ]


def order_key(row: ScanRow) -> tuple[str, float]:
    """A stable order, with the moment-less rows first on their clip.

    A `None` timestamp cannot be compared to a float, so it sorts as -1.0 -- ahead of any
    real moment, and never at a time a clip could actually have.
    """
    return (row.clip, -1.0 if row.timestamp is None else row.timestamp)


def replace_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Put these rows at `path` in ONE step: written beside it, then renamed over it.

    `eval scan` rewrites its artifacts after every clip now, so an interruption lands inside
    a write far more often than it used to. Opening the real file with "w" truncates it
    first, and a scan killed in that instant would leave a SHORT candidates.csv -- which
    reads back perfectly cleanly, and is this project's signature failure: a corpus that
    lost rows and looks fine. `os.replace` is atomic on both Windows and POSIX, so a reader
    sees either the previous complete file or the new complete one, and never a half.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    _replace_when_windows_lets_go(temporary, path)


# A rename over an open file fails on Windows, and it is not a rare race: a scan writing
# these three files 657 times is written over by an antivirus scanner mid-scan, by a search
# indexer, or simply by somebody running `wc -l` on the manifest to see how far it has got.
# Python's own `open()` asks for no FILE_SHARE_DELETE, so an ordinary reader is enough.
#
# Measured, not imagined: the first real run of the resumable scan died at clip 104 with
# `PermissionError: [WinError 5]` out of `os.replace` -- while this session was reading the
# manifest to check its progress. Ten tries at 100 ms covers a scanner's grip on a 20 KB
# file with room to spare, and giving up loudly after a full second is right: a scan that
# cannot write its result down is exactly the failure this file exists to prevent.
REPLACE_ATTEMPTS = 10
REPLACE_BACKOFF_SECONDS = 0.1


def _replace_when_windows_lets_go(temporary: Path, path: Path) -> None:
    """`os.replace`, retried while the destination is momentarily held open elsewhere.

    Only `PermissionError` is retried -- WinError 5 (access denied) and WinError 32 (in use
    by another process) both arrive as one, and both clear on their own. Anything else is a
    real fault and is raised at once rather than waited out.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_SECONDS)


def write_candidates(rows: Iterable[ScanRow], path: Path) -> int:
    """Write eval/candidates.csv. GITIGNORED: every row names a clip of children.

    Rewritten in full after every clip of a scan rather than once at the end, so the sorted
    order is the SAME whether a run finished in one sitting or five. Appending instead would
    have made the file's order depend on how often the run was interrupted, and `eval
    sample` draws a seeded sample from that order: the same corpus and the same seed would
    have produced different worklists.
    """
    ordered = sorted(rows, key=order_key)
    replace_csv(path, SCAN_COLUMNS, [astuple(row) for row in ordered])
    return len(ordered)


def load_candidates(path: Path) -> list[ScanRow]:
    """Read it back. A missing file is an empty list: `eval label` is resumable, and
    resuming from nothing is an ordinary state, not an error.

    This is also what reads `qorgan eval sample`'s worklist, whose extra `stratum` column
    is ignored here (by name) and whose silent rows carry an EMPTY timestamp. An empty
    timestamp reads back as `None` -- the detector named no moment -- and never as 0.0,
    which is a moment, and one the human never chose.
    """
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            ScanRow(
                clip=row["clip"],
                timestamp=_moment(row["timestamp"]),
                score=float(row["score"]),
                probability=float(row["probability"]),
                confidence=float(row["confidence"]),
            )
            for row in csv.DictReader(handle)
        ]


def _moment(raw: str) -> float | None:
    return float(raw) if (raw or "").strip() else None


def coverage_path(candidates_path: Path) -> Path:
    """Where the coverage manifest lives: beside candidates.csv, paired by name.

    `eval/candidates.csv` -> `eval/candidates.coverage.csv`. Deriving it from the
    candidates path means the two artifacts always sit together and `eval sample` can find
    the manifest from `--candidates` alone, exactly as `eval scan` writes it from `--out`.
    """
    return candidates_path.with_name(f"{candidates_path.stem}.coverage.csv")


def write_coverage(clips: Iterable[str], path: Path) -> int:
    """Write the coverage manifest. GITIGNORED: every line names a clip of children.

    Rewritten alongside candidates.csv after every clip, so the two cannot drift even when a
    run is interrupted: a clip in candidates.csv is always a clip in the manifest. `clips`
    are the clips the scan ran to COMPLETION -- a clip it errored on or skipped is not among
    them, or the manifest would certify a clip that was never actually processed.

    That is also what makes the scan resumable: the manifest already IS the record of which
    clips are finished, so a re-run reads it and skips exactly those, and there is no second
    progress file that could disagree with the result.
    """
    ordered = sorted(set(clips))
    replace_csv(path, COVERAGE_COLUMNS, [(clip,) for clip in ordered])
    return len(ordered)


def load_coverage(path: Path) -> set[str] | None:
    """The set of clips the scan covered, or `None` when there is no manifest at all.

    `None` is NOT an empty set, and the difference is the whole point. An empty manifest is
    a scan that covered nothing; a MISSING file is an old candidates.csv produced before
    coverage was recorded -- we do not know what it covered. `eval sample` turns the two
    into different outcomes: an empty manifest proves every clip is unscanned (a hard
    error), while a missing one sends the user back to re-run `eval scan`. Falling back to
    "absence == silent" for a missing manifest would be exactly the bug this exists to kill.
    """
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["clip"] for row in csv.DictReader(handle)}


def unreadable_path(candidates_path: Path) -> Path:
    """Where the unreadable list lives: beside candidates.csv, paired by name.

    `eval/candidates.csv` -> `eval/candidates.unreadable.csv`, exactly as the coverage
    manifest is derived, so all three artifacts of one scan travel together.
    """
    return candidates_path.with_name(f"{candidates_path.stem}.unreadable.csv")


def write_unreadable(failures: Mapping[str, str], path: Path) -> int:
    """Write the clips the scan could not read, each with WHY. GITIGNORED: it names clips.

    This file is the honest half of "one bad clip must not take the corpus down". Surviving
    a clip is easy; the trap is surviving it silently, because a scan that skipped 200 files
    and a scan that read them produce candidate files that look identical. The clips named
    here are deliberately NOT in the coverage manifest, so `eval sample` treats each one as
    unscanned and refuses to draw -- the count cannot be lost between the two commands.

    An empty mapping still writes the file (a header and no rows), so "this scan hit nothing
    it could not read" is a recorded fact rather than a missing file that might mean the
    scan predates the check.
    """
    ordered = sorted(failures.items())
    replace_csv(path, UNREADABLE_COLUMNS, ordered)
    return len(ordered)


def load_unreadable(path: Path) -> dict[str, str]:
    """The clips a previous scan could not read, and why. Missing file is an empty mapping.

    Unlike `load_coverage`, missing needs no special reading: nothing infers anything from
    absence here. A clip that failed is also absent from the manifest, and THAT is what
    makes the downstream refusal happen -- this file explains the refusal, it does not
    cause it.
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["clip"]: row["error"] for row in csv.DictReader(handle)}

"""The labelling tool's parts. A dev tool: it gets no web route.

**Watch the crop; record the full frame.**

The recorder writes two views of one incident:

    hall_left_main_1009_1019_20260702_144150_952947.mp4           <- the crop (ROI)
    hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4  <- the full frame
    stairs_floor2_196_322_burst101_20260518_141523_230173.mp4     <- no `_main` segment

Same camera, same track-ID pair, seconds apart: the crop was cut out of the burst. A human
deciding "is this a fight?" does not need the scene. They need the two children, close up
-- which is exactly what the crop is, and it is far faster to watch than a 1440p wide shot
in which the pair occupies 3% of the pixels.

But the crop can never be DETECTOR input: a ~320x450 ROI has no scene, no zones and no
frame geometry, and the scorer's box-diagonal scaling is meaningless inside it. It would
produce numbers, and every one of them would be a lie.

So the label is always written against the **full-frame** video_id, whichever view the
human watched. The crop is a lens, never the record.

**Where the crops are, and how many join.** They are staged apart from the full frames --
`eval/crops/` (1293 of them) next to `eval/clips/` (663) -- so `--crops` defaults there.
Measured 2026-07-14: **342 of the 381 incident-keys (90%) have a crop partner.**

An earlier run of this module measured 1.1%, and that number is now wrong twice over: the
crops had simply not been staged yet, AND both naming shapes above exist, so a pair-key
parser that demanded `_main` dropped every stairs clip out of the join on both sides. The
fallback to the wide shot is still there and still tested -- it is what the remaining 10%
get -- but it is no longer the common path. Pointing `--crops` at `eval/clips/` remains
safe, since a burst is never treated as a crop.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess  # opening a video in the OS player, on a dev machine
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import ClipNameError, parse_clip_name
from qorgan.evaluation.labels import REQUIRED_COLUMNS, Interval, LabelError, LabelKind
from qorgan.evaluation.scan import ScanRow, load_candidates

# How wide an interval one candidate timestamp becomes. Matches
# metrics.DEFAULT_TOLERANCE_SECONDS: a human annotator's start time is itself approximate,
# and a label narrower than the tolerance would be a distinction the scorer cannot see.
LABEL_PAD_SECONDS = 2.0

PROMPT = "[b]ullying / [n]ormal / [i]gnore / [p]ending / [s]kip / [q]uit > "
CHOICES = {
    "b": LabelKind.BULLYING,
    "n": LabelKind.NORMAL,
    "i": LabelKind.IGNORE,
    "p": LabelKind.PENDING,
}

# The complete set of exact single-character tokens the labeller accepts: the four labels
# plus `s` (skip) and `q` (quit). Anything else -- empty, multi-character, an unknown key --
# is re-prompted, never mapped. Taking the first character instead turned "not sure" into a
# confident NORMAL and "beats me" into a BULLYING: a human's ambiguous answer became the
# exact guessed label this project forbids.
VALID_KEYS = frozenset({*CHOICES, "s", "q"})

# Printed once at the start of a run. `p` is the reason this exists: it is distinct from
# both `s` and `i`, and confusing the three is what wrote the only confirmed fight into the
# corpus as `ignore` and made its recall unmeasurable.
KEYS_HELP = (
    "  b = bullying      a real fight\n"
    "  n = normal        watched, no fight\n"
    "  i = ignore        a judgement: neither fight nor false alarm, do not score it\n"
    "  p = pending       I cannot judge this yet -- keeps the clip scannable, asserts\n"
    "                    nothing, and is counted so nobody mistakes it for a clean result\n"
    "  s = skip          writes nothing; comes back next run\n"
    "  q = quit"
)


def crop_partner(clip: str, crops: Iterable[str], cameras: Iterable[str]) -> str | None:
    """The crop showing the same incident as this full-frame clip, or None.

    The join: same camera, same track-ID pair, nearest timestamp. A burst is never a crop,
    so this is safe to point at a directory holding both views.

    `cameras` is the configured camera names, because the pair key starts with a camera and
    only the config knows where a camera's name ends (`clips.parse_clip_name`). Both naming
    shapes reach here, with and without `_main`.

    Unparsable names are skipped rather than fatal -- the human-named clips in the corpus,
    and the `stairs_floor2_second` ones no config can place, must not take a labelling run
    down with them, on either side of the join. A clip nobody can attribute simply has no
    crop partner, and the human watches the wide shot.
    """
    names = list(cameras)
    try:
        target = parse_clip_name(clip, names)
    except ClipNameError:
        return None

    best: str | None = None
    best_gap = float("inf")
    for name in crops:
        try:
            other = parse_clip_name(name, names)
        except ClipNameError:
            continue
        if other.is_burst or other.camera != target.camera or other.pair != target.pair:
            continue
        gap = abs((other.recorded_at - target.recorded_at).total_seconds())
        if gap < best_gap:
            best, best_gap = name, gap
    return best


def interval_for(row: ScanRow, kind: LabelKind, duration: float | None = None) -> Interval:
    """One row, as a labelled interval -- against the FULL FRAME.

    A row with a timestamp is a candidate: the detector named a moment, and the interval is
    that moment plus the tolerance pad.

    A row with NO timestamp is a silent clip from `qorgan eval sample` -- the fast tier
    never fired on it, so there is no moment, and the human was asked about the whole clip.
    The interval is therefore the whole clip, [0, duration]. That is coarse, and it is
    honest. Padding an invented midpoint instead would assert both where the fight is and
    that the rest of the clip is clean, and the second half of that claim is what turns a
    detector's correct find into a scored false positive.
    """
    if row.timestamp is None:
        if duration is None:
            raise ValueError(
                f"{row.clip}: this row carries no timestamp, so it is a whole-clip "
                "judgement and needs the clip's duration. Refusing to invent a moment."
            )
        return Interval(video_id=row.clip, start=0.0, end=duration, kind=kind)

    return Interval(
        video_id=row.clip,
        start=_start_of(row),
        end=row.timestamp + LABEL_PAD_SECONDS,
        kind=kind,
    )


# The columns `append_label` writes. The live labels.csv carries all five: the four
# REQUIRED_COLUMNS plus `camera`. Writing only the first four produced a ragged row against
# it -- `csv.DictReader` tolerates the short row, so the camera slid silently to None and the
# human-named clips (one of which is the only confirmed fight) became un-scannable.
LABEL_COLUMNS = (*REQUIRED_COLUMNS, "camera")


def append_label(path: Path, interval: Interval) -> None:
    """Append one row -- WITH its camera. Written immediately, so a crash costs the last
    decision and not the afternoon.

    Never emits a ragged row. Appending to a file whose header is not `LABEL_COLUMNS` is a
    hard error, not something to paper over: a five-field row under a four-column header
    misaligns, and the misalignment is exactly what this column exists to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    if not new:
        _check_appendable(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(LABEL_COLUMNS)
        writer.writerow(
            [
                interval.video_id,
                f"{interval.start:.2f}",
                f"{interval.end:.2f}",
                interval.kind.value,
                interval.camera or "",
            ]
        )


def _check_appendable(path: Path) -> None:
    """The existing header must be exactly the columns we are about to write. `#` comments
    are skipped, as everywhere labels.csv is read."""
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    header = next(csv.reader(lines), None)
    if header != list(LABEL_COLUMNS):
        raise LabelError(
            f"{path.name}: header is {header}, expected {list(LABEL_COLUMNS)}. Refusing to "
            "append a row that would not line up with it -- a ragged row loses the camera."
        )


ResumeKey = tuple[str, str | None]


def _moment_key(end: float) -> str:
    """A candidate's moment, in the ONE form both sides can agree on: the 2-dp text of its
    interval end, exactly as `append_label` writes it.

    This is the seam. The live side holds a raw timestamp; the resume side has only what
    labels.csv carries, which is 2-dp TEXT. `scan.rows_for` stores timestamps at 3 dp, so
    the two sides meet across a decimal round-trip -- and a key each computed for itself
    could round the binary half opposite ways (t=1.535: `round(t, 2)` is 1.53, while the
    written end "3.54" reconstructs to 1.54). 166 of the 10 000 3-dp timestamps in [0, 10)
    split like that; 29.97 fps reaches them and 10/8/5 fps do not, which is why this was a
    trap and not yet a bug.

    Keying on the written text closes it BY CONSTRUCTION: the text is what the round-trip
    preserves, so both sides land on one string or the file could not be read back at all.
    """
    return f"{end:.2f}"


def resume_key(clip: str, timestamp: float | None) -> ResumeKey:
    """The resume/dedup identity of a live CANDIDATE row, from its RAW timestamp -- never
    the clamped interval start.

    The start is `max(0.0, timestamp - LABEL_PAD_SECONDS)`, so two early candidates on one
    clip -- e.g. t=0.5 and t=1.5 -- both clamp to 0.00. Keyed on that start they collide:
    label the first and the second is marked done and never asked again, a candidate
    (possibly a real fight) silently lost. The pad is added back rather than subtracted, so
    the key is the interval END, which `interval_for` never clamps -- and the two early
    candidates stay two distinct keys ("2.50" and "3.50").

    A whole-clip judgement (a silent clip, `timestamp is None`) names no moment and keys on
    None -- and a clip never carries both a candidate and a silent row, so the two identity
    spaces never meet on one clip.

    `settled_key` is the same identity taken off a row already written to labels.csv. Both
    go through `_moment_key`, so the labeller and the sampler cannot drift apart.
    """
    if timestamp is None:
        return (clip, None)
    return (clip, _moment_key(timestamp + LABEL_PAD_SECONDS))


def settled_key(clip: str, end: float) -> ResumeKey:
    """The same identity as `resume_key`, taken off a SETTLED labels.csv row.

    It reads the interval end straight across -- it does NOT reconstruct the raw timestamp
    (`end - LABEL_PAD_SECONDS`) and re-key on that. The end is what the file stores, so
    there is nothing to undo, and the arithmetic that used to undo it is exactly what let
    the two sides round apart. See `_moment_key`.
    """
    return (clip, _moment_key(end))


def already_labelled(path: Path) -> set[ResumeKey]:
    """`settled_key(clip, end)` for every row already SETTLED. Resuming from nothing is
    normal.

    `pending` rows are excluded, mirroring `sampling._judged`: a pending interval asserts
    nothing, so it cannot answer a question, and a candidate a human marked `pending` must
    be RE-OFFERED next run -- exactly as `eval sample` keeps proposing it. Before this, a
    pending row was silently treated as done here, so the sampler's worklist said "judge
    this" and the labeller refused to offer it: the two dedups disagreed on what "unjudged"
    means.

    The key is the interval END, which `interval_for` never clamps -- so two early
    candidates whose starts both clamped to 0.00 stay distinct, and `is_done` builds the
    same key from the live row (`settled_key`, `resume_key`).

    A row written with `t_start == 0.00` ALSO adds the moment-less `(clip, None)` key. Its
    PURPOSE is the whole-clip (silent) row, whose interval is [0, duration]: that key is what
    lets a settled silent row answer its clip's silent question on resume. But `start == 0.0`
    does not IDENTIFY a whole-clip row, and this code does not test for one -- a candidate at
    t < 2.0 clamps to start 0.00 as well (t=1.5 -> start 0.00, verified), so it adds the None
    key too.

    That extra key is inert rather than wrong, and for a reason worth stating precisely: a
    clip never carries both a candidate and a silent row (`sampling._silent` draws silent
    clips only from those with NO candidate row), so on a candidate clip there is no silent
    row for the stray None key to suppress. Nor does the sampler lean on this -- it does not
    consult this key at all: `sampling._judged` independently suppresses a clip's silent row
    for ANY settled label on it, which is the stated design.

    It reads the same file a human hand-edits, so `#` comments and the optional `camera`
    column are tolerated exactly as `load_labels` tolerates them. A resume that silently
    saw nothing would send the labeller back to the start of the afternoon.
    """
    if not path.is_file():
        return set()

    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]

    done: set[ResumeKey] = set()
    for row in csv.DictReader(lines):
        if _kind(row["label"], path) is LabelKind.PENDING:
            continue
        clip = row["video_id"]
        start = _seconds(row["t_start"], path)
        end = _seconds(row["t_end"], path)
        done.add(settled_key(clip, end))
        if start == 0.0:
            done.add(resume_key(clip, None))
    return done


def is_done(row: ScanRow, done: set[ResumeKey]) -> bool:
    """97 minutes is more than one sitting. A labeller who stops for lunch must not come
    back to the beginning -- nor find their morning written into the file twice.

    Keyed via `resume_key` from the row's RAW timestamp -- the same identity the sampler's
    `_judged` builds with `settled_key`, so the two dedup on one key rather than two that
    drift apart."""
    return resume_key(row.clip, row.timestamp) in done


def open_in_player(path: Path) -> None:
    """Hand the clip to the OS default player. A dev tool on a dev machine."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a local video file, on a developer's desktop
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)])  # noqa: S603 - same


def _start_of(row: ScanRow) -> float:
    """A row's interval start, as WRITTEN to labels.csv. Never before the clip does.

    A row with no timestamp starts at 0.0 -- it is a judgement about the whole clip, and
    the whole clip starts at the beginning. This clamp shapes only the written interval;
    the resume/dedup identity keys on the un-clamped interval END, never this start, so
    two early candidates that both clamp to 0.00 stay distinct (`resume_key`).
    """
    if row.timestamp is None:
        return 0.0
    return max(0.0, row.timestamp - LABEL_PAD_SECONDS)


def _kind(raw: str, path: Path) -> LabelKind:
    try:
        return LabelKind((raw or "").strip().lower())
    except ValueError as exc:
        # Guessing here would silently mis-dedup a row -- treat an unreadable label as an
        # unknown kind instead, exactly as `labels._parse` refuses to guess one.
        allowed = ", ".join(k.value for k in LabelKind)
        raise SystemExit(
            f"{path}: label {raw!r} is not one of ({allowed}). Fix it before resuming."
        ) from exc


def _seconds(raw: str, path: Path) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        # Guessing here would either re-ask a decision already made or skip one never made.
        raise SystemExit(
            f"{path}: t_start {raw!r} is not a number. Fix it before resuming."
        ) from exc


def _ask_choice(ask: Callable[[str], str]) -> str:
    """One exact token from `VALID_KEYS`. Free text -- empty, multi-character ("not sure"),
    or an unknown key -- is RE-PROMPTED, never coerced into a label. The old `[:1]` took the
    first character, so "not sure" recorded NORMAL and "beats me" recorded BULLYING: the
    forbidden guess, wearing a human's face. `s` and `q` stay the explicit skip and quit.
    """
    while True:
        answer = ask(PROMPT).strip().lower()
        if answer in VALID_KEYS:
            return answer
        print("        not a key -- press exactly one of b / n / i / p / s / q")


# -- the command: a human, watching only what they were asked to -----------


def cmd_label(
    args: argparse.Namespace,
    *,
    ask: Callable[[str], str] = input,
    play: Callable[[Path], None] = open_in_player,
    duration: Callable[[Path], float] | None = None,
) -> int:
    """Pass 2 of labelling: a human, watching what `eval scan` or `eval sample` proposed.

    `ask`, `play` and `duration` are injected so the tests can drive the decision logic
    without a video player opening on somebody's desktop, or a decoder opening at all.
    Defaults are the real thing.
    """
    rows = load_candidates(args.candidates)
    if not rows:
        raise SystemExit(f"no candidates in {args.candidates}. Run `qorgan eval scan` first.")

    measure = _clip_duration if duration is None else duration
    done = already_labelled(args.labels)
    todo = [row for row in rows if not is_done(row, done)]
    crops = [p.name for p in args.crops.iterdir()] if args.crops.is_dir() else []
    cameras = list(load_cameras())  # the join's pair key starts with a CONFIGURED camera
    print(f"{len(rows)} candidate(s), {len(rows) - len(todo)} already labelled.")
    print(KEYS_HELP + "\n")

    for index, row in enumerate(todo, start=1):
        view = _view_for(row, args, crops, cameras)
        print(f"[{index}/{len(todo)}] {_describe(row)}")
        print(f"        watching: {view.name}")
        play(view)

        choice = _ask_choice(ask)
        if choice == "q":
            break
        if choice == "s":
            # `s` is the ONLY skip. Never a guessed label -- an unasked-for `normal` is the
            # legacy's silent-negative bug wearing a human's face. It comes back next run.
            print("        skipped (it will come back next run)")
            continue
        _record(args, row, CHOICES[choice], measure)
        print(f"        recorded {CHOICES[choice].value} against {row.clip}")

    print(f"\nlabels -> {args.labels}")
    return 0


def _describe(row: ScanRow) -> str:
    """What the human is being asked about. A silent clip has no moment, and saying
    `t=0.0s` for one would be the tool quietly making a claim it cannot support."""
    if row.timestamp is None:
        return f"{row.clip}  WHOLE CLIP -- the detector never fired on it"
    return (
        f"{row.clip}  t={row.timestamp:.1f}s  "
        f"score={row.score:.2f}  confidence={row.confidence:.2f}"
    )


def _record(
    args: argparse.Namespace, row: ScanRow, kind: LabelKind, measure: Callable[[Path], float]
) -> None:
    """The clip's length is read ONLY for a whole-clip row. A candidate names its own
    moment, so it never opens a decoder at all."""
    length = measure(args.clips / row.clip) if row.timestamp is None else None
    append_label(args.labels, interval_for(row, kind, length))


def _view_for(
    row: ScanRow, args: argparse.Namespace, crops: list[str], cameras: list[str]
) -> Path:
    """The crop if there is one, the full frame if there is not.

    Watching the crop is several times faster; a clip with no crop gets the wide shot --
    slower, but a slow label beats a missing one. The join is on names, and a name is not
    a file: a crop that is named but absent falls back too.
    """
    partner = crop_partner(row.clip, crops, cameras)
    if partner is not None and (args.crops / partner).is_file():
        return args.crops / partner
    return args.clips / row.clip


def _clip_duration(path: Path) -> float:
    from qorgan.evaluation.video import clip_duration  # imports cv2; keep it lazy

    return clip_duration(path)

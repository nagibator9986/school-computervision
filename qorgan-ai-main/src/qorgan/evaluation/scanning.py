"""`qorgan eval scan`: the whole corpus through the detector, written down as it goes.

Split out of `evaluation/cli.py` for the same reason `labelling.py` is: the command has
grown a resume mechanism and a failure policy, and those belong beside each other rather
than inside an argument parser.

**A 657-clip job that can only finish whole loses everything to any interruption.** The
first real run proved it. It crashed on clip 170 -- the host could not allocate the 10.5 MB
for one decoded frame -- and the 169 clips already scanned, tens of minutes of GPU, were
never written down at all, because both artifacts were written once, at the end. A closed
laptop lid or a Ctrl+C would have cost exactly the same. The crash is a bug to chase; the
all-or-nothing shape is a defect in its own right, and it is the one fixed here.

So the artifacts are rewritten after EVERY clip and a re-run resumes from them. The
mechanism is `qorgan eval label`'s, not a second one: there, the output IS the progress
record -- `labelling.append_label` writes each decision the moment it is made and
`already_labelled` reads them back to skip what is done. Here the coverage manifest already
says precisely which clips ran to completion, so it is what a resume skips. Nothing else is
consulted, because a separate checkpoint file could disagree with the result, and then one
of the two would be lying.

**A clip that cannot be read no longer takes the corpus down.** It is recorded by name in
`candidates.unreadable.csv`, with the reason, and the scan moves on. It is deliberately NOT
added to the coverage manifest, so it is never mistaken for a clip the detector watched and
found nothing in: `eval sample` refuses to draw while any clip is unscanned, which is how
the count survives to the end of the pipeline instead of evaporating. The summary prints it
and the exit status is non-zero. A corpus where 200 files were silently skipped and a corpus
where 200 files were read give different numbers and look identical, and that is the exact
defect this project exists to remove.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from qorgan.config.camera import BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import ClipNameError, camera_for
from qorgan.evaluation.harness import run
from qorgan.evaluation.labels import load_labels
from qorgan.evaluation.scan import (
    NotAFullFrameError,
    ScanRow,
    coverage_path,
    load_candidates,
    load_coverage,
    load_unreadable,
    refuse_crops,
    rows_for,
    unreadable_path,
    write_candidates,
    write_coverage,
    write_unreadable,
)
from qorgan.models.validate import SkeletonView

CLIP_SUFFIXES = {".mp4", ".avi", ".mkv"}

# Ctrl+C. The partial result on disk is complete as far as it goes, so this is not a
# failure -- but it is not a finished scan either, and a caller has to be able to tell the
# two apart. 130 is the shell's own convention for "killed by SIGINT".
INTERRUPTED_EXIT = 130

# Every clip, paired with the camera whose zones it must be scored against.
Plan = list[tuple[Path, BullyingCamera]]


@dataclass
class ScanProgress:
    """What the scan has established so far, and where it is written.

    Held in memory and on disk at once: `flush` rewrites all three artifacts together, so
    they cannot disagree about which clips a run has finished -- not even if the run is
    killed between two of the writes, because each one lands atomically (`scan.replace_csv`).
    """

    out: Path
    rows: list[ScanRow] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)
    unreadable: dict[str, str] = field(default_factory=dict)

    def flush(self) -> None:
        write_candidates(self.rows, self.out)
        write_coverage(self.covered, coverage_path(self.out))
        write_unreadable(self.unreadable, unreadable_path(self.out))


def cmd_scan(args: argparse.Namespace) -> int:
    """Run the detector over the whole corpus at threshold 0. What did it fire on?

    Resumable and interruptible: run it again with the same `--out` and it continues from
    the clip after the last one it finished. Exit status is 0 for a complete scan, 1 when
    some clip could not be read, and `INTERRUPTED_EXIT` when Ctrl+C stopped it -- three
    outcomes that a human and a script both need to tell apart.
    """
    plan = _scan_plan(args.clips, args.labels)
    progress = _resume(args.out, plan)
    todo = [pair for pair in plan if pair[0].name not in progress.covered]
    print(f"{len(plan)} clip(s), {len(plan) - len(todo)} already scanned.")

    interrupted = _scan_corpus(todo, progress, args.device)
    # Even a run with nothing left to do leaves all three artifacts on disk and agreeing.
    progress.flush()
    return _report(progress, plan, interrupted=interrupted)


def _resume(out: Path, plan: Plan) -> ScanProgress:
    """Pick up where the last run stopped -- from the artifacts themselves.

    The coverage manifest lists exactly the clips a previous run decoded and ran to
    completion, so those are the clips this run skips, and the candidate rows belonging to
    them are carried forward unchanged.

    A MISSING manifest beside an existing candidates.csv discards that file rather than
    resuming from it. `load_coverage` returns `None` there precisely because nobody can know
    what it covered (`scan.load_coverage`), and keeping its rows while re-scanning every clip
    would double every candidate in it.

    Both are narrowed to the clips this run was pointed at, so the artifacts always describe
    THIS corpus. A candidate row naming a clip no longer in the directory is drift, and
    `eval sample` rejects the pair outright when it finds one -- so carrying it forward would
    only postpone a hard error into a command that cannot fix it. The cost is that scanning a
    subset writes artifacts about that subset; the flags said the subset.
    """
    planned = {clip.name for clip, _ in plan}
    covered = (load_coverage(coverage_path(out)) or set()) & planned
    rows = [row for row in load_candidates(out) if row.clip in covered]
    # Carried forward only so an interrupted re-run does not forget what an earlier run
    # could not read. Every one of these is retried below -- none of them is covered -- and
    # the entry is dropped the moment its clip is attempted again.
    failed = {n: why for n, why in load_unreadable(unreadable_path(out)).items() if n in planned}
    return ScanProgress(out=out, rows=rows, covered=covered, unreadable=failed)


def _scan_corpus(todo: Plan, progress: ScanProgress, device: str) -> bool:
    """Scan what is left, writing after every clip. True when Ctrl+C stopped it.

    KeyboardInterrupt is the operator, not a fault: it ends the loop with the artifacts
    already written and complete as far as they go, and the flush here covers the narrow
    window where the interrupt landed inside the previous one.
    """
    poses: dict[str, SkeletonView] = {}
    try:
        for index, (clip, camera) in enumerate(todo, start=1):
            if camera.name not in poses:
                poses[camera.name] = _pose(camera, device)
            _scan_into(progress, clip, camera, poses[camera.name], device, f"[{index}/{len(todo)}]")
            progress.flush()
    except KeyboardInterrupt:
        progress.flush()
        return True
    return False


def _scan_into(
    progress: ScanProgress,
    clip: Path,
    camera: BullyingCamera,
    pose: SkeletonView,
    device: str,
    marker: str,
) -> None:
    """One clip into the progress record. A clip that fails is NAMED and survived.

    Certified covered only after `_scan_one` returned: a clip whose scan raised is not in
    the manifest, so nothing downstream can read its lack of candidates as silence. The
    catch is deliberately broad -- a decoder can fail as an `OSError`, a `cv2.error`, a
    `MemoryError` or (as it did on clip 170) a bare `SystemError` out of a C extension, and
    a list of the ones seen so far is a list that is wrong the next time.

    Flushed, because a 40-minute scan is normally run with its output redirected to a file,
    and Python block-buffers a redirected stdout: without this the log stays empty for
    minutes at a time and an operator cannot tell a slow clip from a hung run.
    """
    try:
        found = _scan_one(clip, camera, pose, device)
    except Exception as exc:
        reason = _reason(exc)
        progress.unreadable[clip.name] = reason
        print(f"  {marker} {clip.name} ({camera.name}): UNREADABLE -- {reason}", flush=True)
        return

    progress.unreadable.pop(clip.name, None)  # it read this time; the old entry is stale
    progress.rows.extend(found)
    progress.covered.add(clip.name)
    print(f"  {marker} {clip.name} ({camera.name}): {len(found)} candidate(s)", flush=True)


def _reason(exc: BaseException) -> str:
    """The failure, in one CSV cell. The type is kept: `MemoryError` and `OSError` on the
    same clip mean opposite things -- the first is the machine, the second is the file."""
    return f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ").strip()


def _report(progress: ScanProgress, plan: Plan, *, interrupted: bool) -> int:
    """The summary, and the exit status that matches it.

    The unreadable COUNT is printed every time there is one, and it is why this can exit
    non-zero. Candidates.csv from a corpus that skipped 200 clips is indistinguishable from
    one that read them, so the difference has to appear somewhere a human actually looks.
    """
    print(
        f"\n{len(progress.rows)} candidate(s) from {len(progress.covered)} clip(s) "
        f"-> {progress.out}"
    )
    print(
        f"coverage: {len(progress.covered)}/{len(plan)} clip(s) proven scanned "
        f"-> {coverage_path(progress.out)}"
    )
    if progress.unreadable:
        print(f"\n!! {len(progress.unreadable)} clip(s) COULD NOT BE READ:")
        for name, reason in sorted(progress.unreadable.items()):
            print(f"     {name}: {reason}")
        print(f"   listed in {unreadable_path(progress.out)}")
        print("   They are NOT covered and they are NOT silent -- nothing decoded them.")
        print("   `qorgan eval sample` refuses to draw until they are scanned or removed.")

    if interrupted:
        print(f"\nSTOPPED at the operator's request. {len(progress.covered)} clip(s) are done and")
        print("written. Run the same command again to continue from the next one.")
        return INTERRUPTED_EXIT
    if progress.unreadable:
        return 1
    print("Now label them:  qorgan eval label")
    return 0


# -- the pre-flight: everything that must be true before a frame is decoded ----


def _scan_plan(clips_dir: Path, labels_path: Path) -> Plan:
    """Every clip, paired with the camera whose zones it must be scored against.

    Resolved for the WHOLE corpus before a single frame is decoded. A clip whose camera
    cannot be proved is a hard error, and 657 clips is a long time on a GPU: discovering an
    un-placeable clip forty minutes in used to throw the run away, and even now that the run
    survives it would leave a half-scanned corpus over a fault the first second can see.

    The precedence is `eval run`'s, unchanged (`clips.camera_for`): an explicit `camera` in
    labels.csv wins over inference from the filename, and neither is a hard error. That is
    the ONLY way the human-named clips -- one of which is the only confirmed fight in the
    corpus -- get scanned at all. labels.csv need not exist yet: scan runs BEFORE the
    labelling it exists to make possible.

    Every clip must also BE a full frame. Same reason, same second.
    """
    clips = _clips_in(clips_dir)
    _refuse_crops(clips)
    labels = load_labels(labels_path) if labels_path.is_file() else None
    cameras = load_cameras()

    plan: Plan = []
    for clip in clips:
        explicit = labels.camera_for(clip.name) if labels else None
        try:
            plan.append((clip, camera_for(clip.name, cameras, explicit=explicit)))
        except ClipNameError as exc:
            raise SystemExit(
                f"{exc}\n\nName this clip's camera explicitly: add a row to "
                f"{labels_path} with video_id={clip.name!r} and a `camera` column value."
            ) from exc
    return plan


def _scan_one(clip: Path, camera: BullyingCamera, pose: SkeletonView, device: str) -> list[ScanRow]:
    """One clip, start to finish, through the same `run` the benchmark uses.

    One clip at a time, and every candidate cropped as `run` reaches it: batching candidates
    across clips (or deferring their crops) would ask the rolling frame buffer for frames it
    has already evicted -- `build_crops` raises `CropProvenanceError` rather than validate a
    candidate against whatever frames it happens to be holding.
    """
    from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

    return rows_for(run(VideoSource(clip, camera, device=device), camera.bullying, pose=pose))


def _refuse_crops(clips: list[Path]) -> None:
    """The clips directory is full-frame scenes. A crop in there gets scored as one.

    Checked in the pre-flight, beside the camera check: a 320x450 ROI scored as a scene does
    not crash, it returns confident nonsense, and nonsense that survives a whole scan is what
    a calibration then gets built on.
    """
    try:
        refuse_crops({clip.name: _frame_size(clip) for clip in clips})
    except NotAFullFrameError as exc:
        raise SystemExit(str(exc)) from exc


def _frame_size(path: Path) -> tuple[int, int]:
    from qorgan.evaluation.video import frame_size  # imports cv2; keep it lazy

    return frame_size(path)


def _clips_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"no clips directory: {directory}")
    clips = sorted(p for p in directory.iterdir() if p.suffix.lower() in CLIP_SUFFIXES)
    if not clips:
        raise SystemExit(f"no clips in {directory}. Nothing to scan.")
    return clips


def _pose(camera: BullyingCamera, device: str) -> SkeletonView:
    """The REAL pose model.

    Without it, `validation_score` is 0.0, every verdict is capped at 0.72, and the PR curve
    is empty above it -- which is where the notify threshold (0.85) lives. A benchmark that
    cannot produce a single alert production would send is not a benchmark.
    """
    from qorgan.models.pose import PoseEstimator  # imports ultralytics; keep it lazy

    return PoseEstimator(camera.bullying.skeleton, device=device)

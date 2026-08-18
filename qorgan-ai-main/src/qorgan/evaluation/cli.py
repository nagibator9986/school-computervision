"""`qorgan eval` — run the benchmark, sweep the threshold, guard against regressions."""

from __future__ import annotations

import argparse
from pathlib import Path

from qorgan.config.camera import BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import ClipNameError, camera_for
from qorgan.evaluation.harness import RunResult, run
from qorgan.evaluation.labelling import cmd_label
from qorgan.evaluation.labels import LabelSet, load_labels, write_template
from qorgan.evaluation.metrics import Metrics, evaluate
from qorgan.evaluation.noise_floor import measure, synthetic_floor
from qorgan.evaluation.regression import Baseline, check
from qorgan.evaluation.report import print_curve, print_suppressions, warn_when_no_positives
from qorgan.evaluation.sampling import (
    DEFAULT_SEED,
    DEFAULT_SILENT_SAMPLE,
    SampleError,
    Stratum,
    counts,
    draw,
    write_sample,
)
from qorgan.evaluation.scan import coverage_path, load_candidates, load_coverage

# `eval scan` moved to its own module when it grew a resume mechanism (`scanning`), the way
# `eval label` lives in `labelling`. `_scan_plan` and `_pose` went with it because scanning
# is where they are defined, not because scanning is their only caller: `cmd_sample` needs
# the same clip-to-camera plan and `_evaluate` needs the same pose model, and a second copy
# of either here is how the two commands would start disagreeing about what a clip IS.
from qorgan.evaluation.scanning import _pose, _scan_plan, cmd_scan
from qorgan.models.validate import SkeletonView

BASELINE_PATH = Path("eval/baseline.json")
LABELS_PATH = Path("eval/labels.csv")
CLIPS_DIR = Path("eval/clips")
CROPS_DIR = Path("eval/crops")
CANDIDATES_PATH = Path("eval/candidates.csv")
SAMPLE_PATH = Path("eval/sample.csv")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="evaluate the detector against labelled clips")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_cmd = sub.add_parser("run", help="score the detector and print a PR curve")
    _common(run_cmd)
    run_cmd.set_defaults(func=cmd_run)

    gate_cmd = sub.add_parser("gate", help="fail if this change made the detector worse")
    _common(gate_cmd)
    gate_cmd.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    gate_cmd.set_defaults(func=cmd_gate)

    save_cmd = sub.add_parser("save-baseline", help="record the current score as the baseline")
    _common(save_cmd)
    save_cmd.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    save_cmd.add_argument("--note", default="", help="why this baseline changed")
    save_cmd.set_defaults(func=cmd_save)

    template = sub.add_parser("template", help="write a labels.csv template")
    template.add_argument("--out", type=Path, default=LABELS_PATH)
    template.set_defaults(func=cmd_template)

    _scan_parser(sub)
    _sample_parser(sub)
    _label_parser(sub)

    floor = sub.add_parser(
        "noise-floor",
        help="measure what a camera reports when NOTHING is happening",
        description=(
            "Feed this ten minutes of a QUIET corridor -- people walking, nobody "
            "fighting. It reports what the detector claims happened, which is all noise "
            "by construction, and tells you where acceleration_threshold has to sit to "
            "be above it. Without a clip it falls back to a synthetic walk, which is a "
            "LOWER bound on the real floor: real footage has motion blur, compression "
            "and occlusion, and every one of those shakes the box harder."
        ),
    )
    floor.add_argument("clip", type=Path, nargs="?", help="a quiet recording. Omit for synthetic.")
    floor.add_argument("--camera", default="hall_left")
    floor.add_argument("--device", default="cuda:0")
    floor.set_defaults(func=cmd_noise_floor)


def _scan_parser(sub: argparse._SubParsersAction) -> None:
    scan_cmd = sub.add_parser(
        "scan",
        help="run the detector over every clip at threshold 0 and list what it fired on",
        description=(
            "Pass 1 of labelling, by exception. 97 minutes of corridor is too much to "
            "watch; the detector goes first and writes down every moment it would have "
            "raised anything at all. A human then watches only those.\n\n"
            "RESUMABLE. The result is written after every clip, so a crash, a closed lid "
            "or a Ctrl+C costs the clip in flight and nothing else: run it again with the "
            "same --out and it continues where it stopped. A clip that cannot be read is "
            "named in candidates.unreadable.csv, left OUT of the coverage manifest, and "
            "counted in the summary -- it is never quietly folded in with the silent ones."
        ),
    )
    scan_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    scan_cmd.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    scan_cmd.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        help="read ONLY for its `camera` column, so the clips no filename can place "
        "(the human-named ones) can be scanned at all. Need not exist yet.",
    )
    scan_cmd.add_argument("--device", default="cuda:0")
    scan_cmd.set_defaults(func=cmd_scan)


def _sample_parser(sub: argparse._SubParsersAction) -> None:
    sample_cmd = sub.add_parser(
        "sample",
        help="draw the stratified worklist a human must label to measure MISSES",
        description=(
            "Precision is half a detector. The scan found three populations, not one: the "
            "alerts (precision), the clips the SKELETON vetoed, and the clips the fast "
            "tier never fired on. The skeleton is holding back half of everything the fast "
            "tier proposes -- if any of those is a real fight, that is a child nobody "
            "helped, and no precision measurement can see it. Draws every candidate "
            "stratum whole and a seeded sample of the silent clips."
        ),
    )
    sample_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    sample_cmd.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    sample_cmd.add_argument("--out", type=Path, default=SAMPLE_PATH)
    sample_cmd.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        help="a JUDGED candidate here is not proposed again (candidate by candidate, never "
        "the whole clip; a `pending` row suppresses nothing). Also supplies `camera`.",
    )
    sample_cmd.add_argument("--count", type=int, default=DEFAULT_SILENT_SAMPLE)
    sample_cmd.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sample_cmd.set_defaults(func=cmd_sample)


def _label_parser(sub: argparse._SubParsersAction) -> None:
    label_cmd = sub.add_parser(
        "label",
        help="watch each candidate and label it",
        description=(
            "Opens the CROP -- the two children, close up -- in your default video "
            "player, falling back to the full frame when no crop partner exists. The "
            "label is always recorded against the FULL-FRAME clip, whichever view you "
            "watched: the crop is a lens, never the record. Resumable; a dev tool."
        ),
    )
    label_cmd.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    label_cmd.add_argument("--clips", type=Path, default=CLIPS_DIR)
    label_cmd.add_argument(
        "--crops",
        type=Path,
        default=CROPS_DIR,
        help="where the crop ROIs are. Need not exist: every candidate then falls back to "
        "its full frame. Safe to point at --clips if both views live together.",
    )
    label_cmd.add_argument("--labels", type=Path, default=LABELS_PATH)
    label_cmd.set_defaults(func=cmd_label)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--clips", type=Path, default=CLIPS_DIR)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="where YOLO and the pose model run. 663 clips is a long time on a CPU.",
    )


# -- commands --------------------------------------------------------------


def cmd_template(args: argparse.Namespace) -> int:
    write_template(args.out)
    print(f"wrote {args.out}")
    print("Label every clip. A clip with no label is an error, not a `normal`.")
    return 0


def cmd_noise_floor(args: argparse.Namespace) -> int:
    camera = _camera(args.camera)
    metrics = camera.bullying.metrics
    fps = camera.capture.analysis_fps

    if args.clip is None:
        print(f"No clip given. Using a SYNTHETIC calm corridor at {fps:g} fps.")
        print("This is a lower bound on the real floor. Record the corridor.\n")
        floor = synthetic_floor(metrics, fps)
    else:
        from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

        if not args.clip.is_file():
            raise SystemExit(f"no such clip: {args.clip}")
        print(f"Measuring {args.clip} as {args.camera} ...\n")
        floor = measure(VideoSource(args.clip, camera, device=args.device), metrics)

    if floor.samples == 0:
        raise SystemExit("no tracks found. Was anybody in shot?")

    print(floor.summary(current_threshold=metrics.acceleration_threshold))
    return 0 if metrics.acceleration_threshold > floor.acceleration.peak else 1


def cmd_run(args: argparse.Namespace) -> int:
    labels, results = _evaluate(args)
    predictions = [p for r in results for p in r.predictions]

    metrics = evaluate(labels, predictions, threshold=args.threshold)
    print(f"\n{len(results)} clip(s), {len(predictions)} alert(s)\n")
    print(metrics.summary())
    # recall over a corpus with pending intervals is a guess; say so on the same screen.
    if metrics.pending_intervals:
        print(
            f"\n!! {metrics.pending_intervals} PENDING interval(s) unjudged: the recall above "
            "is NOT measured over the whole corpus. Label them (`qorgan eval label`) first."
        )
    # ...and a corpus with no positives at all cannot measure recall in the first place.
    warn_when_no_positives(metrics)

    print_curve(labels, predictions)
    print_suppressions(results)
    return 0


def _refuse_when_pending(metrics: Metrics, action: str) -> None:
    """`gate` and `save-baseline` refuse while any pending interval exists: recall is fiction."""
    if metrics.pending_intervals:
        raise SystemExit(
            f"{action} refuses: {metrics.pending_intervals} pending interval(s) in "
            f"{LABELS_PATH} -- recall is fiction until judged. Label or remove them, then retry."
        )


def cmd_gate(args: argparse.Namespace) -> int:
    if not args.baseline.is_file():
        print(f"no baseline at {args.baseline}. Run `qorgan eval save-baseline` first.")
        return 1

    labels, results = _evaluate(args)
    predictions = [p for r in results for p in r.predictions]
    metrics = evaluate(labels, predictions, threshold=args.threshold)
    _refuse_when_pending(metrics, "gate")

    result = check(Baseline.load(args.baseline), metrics)
    print(result.report())
    return 0 if result.passed else 1


def cmd_sample(args: argparse.Namespace) -> int:
    """Draw the labelling worklist: all of A, all of B, and a seeded sample of C.

    The camera comes from the clip, exactly as it does for `scan` and `run` -- the strata
    boundaries are per-camera config, and a clip nobody can attribute is a hard error here
    too (`_scan_plan`). Nothing is decoded: this reads the scan's output, not the video.

    Silence is PROVEN, never guessed: the coverage manifest beside --candidates says which
    clips the scan actually ran on. A drift between the two artifacts, an unscanned clip, or
    a missing manifest is a hard, actionable error (`draw` -> `_prove_coverage`), surfaced
    here as a clean exit rather than a traceback.
    """
    plan = _scan_plan(args.clips, args.labels)
    labels = load_labels(args.labels) if args.labels.is_file() else None

    try:
        worklist = draw(
            clips=[clip.name for clip, _ in plan],
            candidates=load_candidates(args.candidates),
            cameras={clip.name: camera for clip, camera in plan},
            coverage=load_coverage(coverage_path(args.candidates)),
            labelled=labels,
            count=args.count,
            seed=args.seed,
        )
    except SampleError as exc:
        raise SystemExit(str(exc)) from exc
    if not worklist:
        raise SystemExit("nothing left to label. Every clip is already in labels.csv.")

    written = write_sample(worklist, args.out)
    tally = counts(worklist)
    print(f"\n{written} row(s) -> {args.out}  (seed {args.seed})")
    for stratum, total in tally.items():
        print(f"  {stratum.value:<22} {total:>4}   {_MEASURES[stratum]}")
    print("\nWhat was left out is in the log -- read it. A stratum nobody drew is a")
    print("question nobody answered, not a question with the answer 'no'.")
    print(f"\nNow label them:  qorgan eval label --candidates {args.out}")
    return 0


_MEASURES = {  # one entry per stratum: cmd_sample indexes this, a latent KeyError otherwise
    Stratum.ALERT: "precision: how often we cry wolf",
    Stratum.NEAR_MISS: "skeleton confirmed, below the alert -- where is the boundary?",
    Stratum.SKELETON_SUPPRESSED: "did the SKELETON veto a real fight?",
    Stratum.BELOW_CAP: "the fast tier fired under the cap -- a weak miss or noise?",
    Stratum.SILENT: "did the FAST TIER miss a fight?",
}


def cmd_save(args: argparse.Namespace) -> int:
    labels, results = _evaluate(args)
    predictions = [p for r in results for p in r.predictions]
    metrics = evaluate(labels, predictions, threshold=args.threshold)
    _refuse_when_pending(metrics, "save-baseline")

    baseline = Baseline.from_metrics(metrics, videos=len(results), note=args.note)
    baseline.save(args.baseline)
    print(f"baseline saved to {args.baseline}: {metrics.summary()}")
    return 0


# -- plumbing --------------------------------------------------------------


def _evaluate(args: argparse.Namespace) -> tuple[LabelSet, list[RunResult]]:
    from qorgan.evaluation.video import VideoSource  # imports ultralytics; keep it lazy

    labels = load_labels(args.labels)
    cameras = load_cameras()
    poses: dict[str, SkeletonView] = {}

    results = []
    for video_id in labels.videos:
        clip = args.clips / video_id
        if not clip.is_file():
            print(f"  ! missing clip, skipping: {clip}")
            continue

        # The camera comes from the CLIP, not from a flag. hall_left masks a reflective
        # column that hall_right cannot see; one flag for both blanks part of the frame.
        # labels.csv's explicit `camera` column, when present, wins over inference -- it
        # is the only way to score the human-named clips a recorder never touched.
        try:
            camera = camera_for(video_id, cameras, explicit=labels.camera_for(video_id))
        except ClipNameError as exc:
            raise SystemExit(str(exc)) from exc

        if camera.name not in poses:
            poses[camera.name] = _pose(camera, args.device)

        print(f"  running {video_id} as {camera.name} ...")
        results.append(
            run(
                VideoSource(clip, camera, device=args.device),
                camera.bullying,
                pose=poses[camera.name],
            )
        )

    if not results:
        raise SystemExit(f"no clips found in {args.clips}. Nothing to measure.")
    return labels, results


def _camera(name: str) -> BullyingCamera:
    cameras = load_cameras()
    camera = cameras.get(name)
    if camera is None:
        raise SystemExit(f"unknown camera {name!r}. Known: {', '.join(sorted(cameras))}")
    if not isinstance(camera, BullyingCamera):
        raise SystemExit(f"{name} is not a bullying camera")
    return camera

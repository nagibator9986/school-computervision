"""`qorgan identity` -- the questions that are about optics, not thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qorgan.identity.streams import DEFAULT_FRAMES, DEFAULT_STRIDE, StreamSpec

# Exit codes. A script must be able to tell "this camera recognises nobody" (1) from
# "I could not answer" (2) from "yes" (0) -- the legacy discovered the first one in
# month four, from an event log full of Unknown.
USABLE = 0
REFUSED = 1
UNANSWERED = 2

DESCRIPTION = (
    "Can this camera recognise anybody at the resolution the worker actually feeds it?\n\n"
    "Samples frames PER STREAM -- the analysis substream AND the HD burst -- and reports "
    "each one's face-size distribution and the fraction clearing the recognition gate, "
    "measured at the resolution THAT stream is really analysed at. The analysis figure is "
    "computed on frames scaled to capture.frame_width x frame_height, read from THIS "
    "CAMERA'S merged config and never assumed. There is no fleet-wide analysis resolution, "
    "and this text will not name one: it used to list which profiles override the default, "
    "the list said two, and there were three -- in the help of the very command that exists "
    "to stop you trusting a quoted number.\n\n"
    "Measured on this school's hall (1280x720): 0 of 14 970 faces clear the strict 60px "
    "gate -- the median face is 11.5px and the largest in the corpus is 50px. Lower the bar "
    "to the 38px small-face gate and 77 faces get through, of which NONE is recognised "
    "(best score 0.350, against a min_score of 0.45). Zero recognitions in 14 970 faces. "
    "The same faces on the 2560x1440 burst give 2.2% -- true, and about a stream the worker "
    "never analyses.\n\n"
    "If this says ~0%, MOVE THE CAMERA. Do not lower the gate: there is nothing under it "
    "to recover."
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("identity", help="face recognition diagnostics")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    camera_cmd = sub.add_parser(
        "camera-report",
        help="can this camera recognise anybody, at the resolution the worker feeds it?",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    camera_cmd.add_argument("source", help="a camera name from config/cameras, or a path to a clip")
    camera_cmd.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    camera_cmd.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="take every Nth frame, to span the clip"
    )
    camera_cmd.add_argument(
        "--as-camera",
        default=None,
        help="for a clip: analyse it the way this camera would (its capture settings)",
    )
    camera_cmd.set_defaults(func=cmd_camera_report)


def cmd_camera_report(args: argparse.Namespace) -> int:
    from qorgan.config.identity import FaceGate, FaceModelSettings
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.camera import CameraCannotRecognise, refuse_if_hopeless

    try:
        plan = _plan(args)
    except (LookupError, OSError) as exc:
        print(f"cannot read {args.source!r}: {exc}", file=sys.stderr)
        return UNANSWERED

    # camera-report needs boxes, never vectors, so it takes the cheap half of the
    # recognizer: detect_faces() finds the faces without paying for the 512-d embedding
    # that this question has no use for.
    recognizer = FaceRecognizer.shared(FaceModelSettings())

    print(f"{args.source}: can this camera recognise anybody at the resolution the worker")
    print("actually feeds it? Per STREAM, because the answer is different on each.\n")

    reports = [_run(plan, spec, recognizer, FaceGate(), args) for spec in plan.streams]
    for report, spec in zip(reports, plan.streams, strict=True):
        print(report.summary())
        print(f"  {spec.note()}\n")

    deciding = [
        report for report, spec in zip(reports, plan.streams, strict=True) if spec.gates_identity
    ]
    try:
        for report in deciding:
            refuse_if_hopeless(report)
    except CameraCannotRecognise as refused:
        print(str(refused), file=sys.stderr)
        return REFUSED

    if all(report.usable for report in deciding):
        return USABLE

    print(
        "No faces were seen on the stream that decides. That is not a pass -- it is a "
        "measurement that did not happen. Run it again while somebody walks through.",
        file=sys.stderr,
    )
    return UNANSWERED


def _run(plan: _Plan, spec: StreamSpec, detector: Any, gate: Any, args: argparse.Namespace) -> Any:
    from qorgan.identity.camera import measure_faces
    from qorgan.identity.streams import sample

    handle = plan.open(spec)
    try:
        return measure_faces(
            sample(handle, spec, frames=args.frames, stride=args.stride),
            detector,
            gate,
            source=f"{args.source}/{spec.name}",
        )
    finally:
        handle.release()


class _Plan:
    """What to open, and which streams to measure. A clip, or a camera in config/cameras."""

    def __init__(self, streams: tuple[StreamSpec, ...], clip: Path | None, camera: Any) -> None:
        self.streams = streams
        self._clip = clip
        self._camera = camera

    def open(self, spec: StreamSpec) -> Any:
        from qorgan.capture.clip import open_clip
        from qorgan.identity.streams import open_camera_stream

        if self._clip is not None:
            return open_clip(self._clip)
        return open_camera_stream(self._camera, spec)


def _plan(args: argparse.Namespace) -> _Plan:
    from qorgan.config.common import CaptureSettings
    from qorgan.config.loader import load_cameras
    from qorgan.identity.streams import clip_streams, streams_for

    path = Path(args.source)
    cameras = load_cameras()

    if path.is_file():
        capture = cameras[args.as_camera].capture if args.as_camera else CaptureSettings()
        return _Plan(clip_streams(capture), clip=path, camera=None)

    camera = cameras.get(args.source)
    if camera is None:
        raise LookupError(
            f"no clip at {args.source!r} and no camera called that. Known cameras: "
            f"{', '.join(sorted(cameras))}"
        )
    return _Plan(streams_for(camera), clip=None, camera=camera)

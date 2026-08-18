"""`qorgan classvision` — the seam between the offline analyser and this database.

`classvision` runs on a laptop or a GPU box for tens of minutes with torch, ultralytics,
cv2 and insightface loaded, and writes ONE document: `<lesson>.analysis.json`, schema
`classvision/1.1`. This package reads that document and nothing else. It never imports a
model stack, and `classvision` never imports `qorgan` — the two are checked in both
directions by tests on either side of the seam.

That separation is not tidiness. Three things ride on it (`classvision/INTEGRATION.md` §1,
§9): the dashboard does not inherit a 2 GB dependency tree; a slow model run cannot block a
page load; and YOLO11 is AGPL-3.0, so keeping the network-served process out of the
derivative work is doing legal work. It must not be "simplified" later.

**What lives where in this package.**

* `importer.py` — the refusals, and one artefact turned into rows. Refuses by NAME.
* `places.py`   — which known place a discovered seat is, and which lesson a run belongs to
  (a re-analysis, a DVR continuation, an overlapping hour).
* `frames.py`   — the video-classification view: pre-rendered frames plus the boxes drawn
  over them, either extracted here with ffmpeg or supplied by a classvision-side render.
* `readings.py` — a language model's orientation note, and the guard that refuses any digit
  in it that is not a place number.
* `demo.py`     — a synthetic term for ONE class, every row flagged `is_demo`.

**The two flags this CLI asks for and the artefact cannot supply.** `--room-key` and
`--class`: the analyser measured a FILE and has no field for which room the camera hangs in
or which class was in it. `clip_15min.mp4` and `test_camera.mp4` are the same camera and
would otherwise become two rooms that never accumulate together. Where the key came from is
STORED (`classvision_lessons.camera_key_source`), because an operator's assertion and a
filename are not the same quality of evidence.

Exit codes follow `qorgan identity`: `0` stored, `1` it ran and refused, `2` it could not
run. A back-fill that treats a refused artefact as success is how a term's data quietly
acquires a doubled lesson.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STORED = 0
REFUSED = 1
UNANSWERED = 2

IMPORT_HELP = (
    "Load a classvision artefact (schema classvision/1.x) into this school's database.\n\n"
    "Idempotent on the artefact's own `run_id`, which is a hash of the video content plus "
    "every setting that could change a number -- so re-importing the same file writes "
    "nothing, and re-analysing the same lesson under a different threshold creates a NEW "
    "run rather than overwriting last month's measurement.\n\n"
    "Refuses, by name, on: an unknown schema major; a wall clock with no zone (pass "
    "--timezone, or SCHOOL_TIMEZONE); no wall clock at all (--allow-unclocked stores it "
    "with a NULL date and excludes it from every weekly figure); an hour that overlaps one "
    "already stored (--allow-overlap records the override on the row for ever); the same "
    "run already filed under a different room."
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """`qorgan classvision <verb>`. One group, six verbs."""
    parser = subparsers.add_parser("classvision", help="offline classroom analyses")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    _add_import(sub)
    _add_demo(sub)
    _add_frames(sub)

    # The writer `places.attested_person` documented as deliberately absent. It is
    # present now and still refuses without a named signatory and a named document.
    from qorgan.classvision.attest import add_attest_parser
    add_attest_parser(sub)

    reading = sub.add_parser("reading", help="store a language model's orientation note")
    reading.add_argument("path", help="a reading JSON (see classvision.readings.load)")
    reading.set_defaults(func=_cmd_reading)

    status = sub.add_parser("status", help="what is in these tables, real and demo apart")
    status.set_defaults(func=_cmd_status)


def _add_import(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser(
        "import",
        help="load an artefact",
        description=IMPORT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmd.add_argument("paths", nargs="+", help="one or more <lesson>.analysis.json")
    cmd.add_argument("--room-key", help="which room this camera watches (operator assertion)")
    cmd.add_argument("--class", dest="class_key", help="which class was in the room")
    cmd.add_argument("--timezone", help="zone for a wall clock read off the picture")
    cmd.add_argument("--school", metavar="SLUG", help="required once a second school exists")
    # The flag that makes the fabricated data visible. It is on the ROW, not on a report
    # setting, so nothing downstream can render a demo figure as a measurement by forgetting
    # a template variable.
    cmd.add_argument(
        "--demo", action="store_true", help="flag every row as a demonstration, not a measurement"
    )
    cmd.add_argument("--allow-unclocked", action="store_true")
    cmd.add_argument("--allow-overlap", action="store_true")
    cmd.add_argument(
        "--no-teacher",
        dest="include_teacher",
        action="store_false",
        help="drop the adult's block (§12.5 is the school's decision, recorded either way)",
    )
    cmd.set_defaults(func=_cmd_import)


def _add_demo(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser(
        "demo",
        help="synthesise a term for one class, every row flagged as a demonstration",
        description=(
            "Builds a plausible term so the weekly view has something to show. Shapes are "
            "DERIVED from the real artefacts passed with --from (coverage, index and event "
            "distributions), never from invented constants, and every row carries is_demo=1. "
            "What exactly is synthetic is listed by `qorgan.classvision.demo`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmd.add_argument("--weeks", type=int, default=6)
    cmd.add_argument("--class", dest="class_key", default="8-А")
    cmd.add_argument("--room-key", default="demo_room_8a")
    cmd.add_argument(
        "--from",
        dest="sources",
        action="append",
        help="a real artefact to take the distributions from (repeatable, required)",
    )
    cmd.add_argument("--school", metavar="SLUG")
    cmd.add_argument("--seed", type=int, default=20260817, help="the term is reproducible")
    cmd.add_argument(
        "--replace", action="store_true",
        help="delete the previous demo term for this room first (with its notes; real "
             "lessons in the same room are never touched)",
    )
    cmd.set_defaults(func=_cmd_demo)


def _add_frames(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser(
        "frames",
        help="the video-classification view: frames on disk plus the boxes over them",
        description=(
            "Either --manifest (a classvision-side render, boxes from the detector) or "
            "--video with --at (this side extracts stills with ffmpeg and derives the "
            "rectangles from each PLACE's own geometry). The second is honest only because "
            "`box_source` and the caveat on every row say which it was."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmd.add_argument("--run", dest="run_id", help="the artefact run_id these frames belong to")
    cmd.add_argument("--video", help="the recording to take stills from")
    cmd.add_argument("--at", help="video seconds, comma separated (e.g. 300,900,1800)")
    cmd.add_argument("--manifest", help="a frames manifest written by a classvision render")
    cmd.add_argument("--school", metavar="SLUG")
    cmd.set_defaults(func=_cmd_frames)


def _school_id(args: argparse.Namespace) -> int | None:
    from qorgan.schools import school_id_for_slug

    slug = getattr(args, "school", None)
    return school_id_for_slug(slug) if slug else None


def _cmd_import(args: argparse.Namespace) -> int:
    """Import every path given, and report each one separately.

    A directory back-fill imports many files and some of them are refused; printing one
    verdict per file, and returning REFUSED if any was, is what stops a cron treating a
    doubled lesson as success.
    """
    from qorgan.classvision.importer import Refusal, Unreadable, import_artefact

    worst = STORED
    for path in args.paths:
        try:
            result = import_artefact(
                Path(path),
                school_id=_school_id(args),
                camera_key=args.room_key,
                class_key=args.class_key,
                timezone=args.timezone,
                is_demo=args.demo,
                allow_unclocked=args.allow_unclocked,
                allow_overlap=args.allow_overlap,
                include_teacher=args.include_teacher,
            )
        except Refusal as exc:
            print(f"{path}: ОТКАЗ [{exc.code}] {exc}", file=sys.stderr)
            worst = max(worst, REFUSED)
            continue
        except Unreadable as exc:
            print(f"{path}: не прочитан — {exc}", file=sys.stderr)
            worst = max(worst, UNANSWERED)
            continue
        for line in result.report_ru():
            print(line)
    return worst


def _cmd_demo(args: argparse.Namespace) -> int:
    # `DemoRefused` is imported from the module that DEFINES it. It was briefly defined in
    # both modules, and the duplicate silently shadowed the real one: the refusal reached an
    # operator as a traceback while this `except` sat there looking correct.
    from qorgan.classvision.demo import generate
    from qorgan.classvision.distributions import DemoRefused

    try:
        result = generate(
            sources=[Path(p) for p in (args.sources or [])],
            weeks=args.weeks,
            class_key=args.class_key,
            camera_key=args.room_key,
            school_id=_school_id(args),
            seed=args.seed,
            replace=args.replace,
        )
    except DemoRefused as exc:
        print(f"ДЕМО не построено: {exc}", file=sys.stderr)
        return REFUSED
    for line in result.report_ru():
        print(line)
    return STORED


def _cmd_frames(args: argparse.Namespace) -> int:
    from qorgan.classvision.frames import FramesRefused, from_manifest, from_video

    seconds = [float(s) for s in args.at.split(",")] if args.at else []
    try:
        if args.manifest:
            result = from_manifest(Path(args.manifest), school_id=_school_id(args))
        else:
            result = from_video(
                run_id=args.run_id,
                video=Path(args.video) if args.video else None,
                seconds=seconds,
                school_id=_school_id(args),
            )
    except FramesRefused as exc:
        print(f"кадры не записаны: {exc}", file=sys.stderr)
        return REFUSED
    for line in result.report_ru():
        print(line)
    return STORED


def _cmd_reading(args: argparse.Namespace) -> int:
    from qorgan.classvision.readings import ReadingRefused, load

    try:
        stored = load(Path(args.path))
    except ReadingRefused as exc:
        print(f"записка не сохранена: {exc}", file=sys.stderr)
        return REFUSED
    for line in stored.report_ru():
        print(line)
    return STORED


def _cmd_status(_args: argparse.Namespace) -> int:
    """Row counts, with the demonstration rows counted SEPARATELY and never folded in."""
    from qorgan.classvision.importer import status

    for line in status(_school_id(_args)):
        print(line)
    return STORED

"""`qorgan weapons` -- the two questions to ask before trusting this module with anything.

    weapons weights <camera>        what would actually load, and does it refuse?
    weapons camera-report <camera>  can this camera see an object that size at all?

Both are written to be run on the school's machine by somebody who has not read the code,
and both are written so that a NO is loud. The exit codes are `identity`'s, and for the
same reason it gives: a script has to tell "this camera cannot do this" (1) from "I could
not answer" (2) from "yes" (0). The legacy discovered its equivalent of a 1 in month four,
from an event log full of Unknown.
"""

from __future__ import annotations

import argparse
import sys

from qorgan.weapons.feasibility import DEFAULT_HFOV_DEGREES, DEFAULT_OBJECT_CM

USABLE = 0
REFUSED = 1
UNANSWERED = 2

WEIGHTS_DESCRIPTION = (
    "What weapons weights would this camera actually load?\n\n"
    "It loads them. It does not check that a path exists -- that is the check the "
    "previous system made, and their `best.pt` is a 0-BYTE FILE that passed it for "
    "months while the module fell back to motion analysis and reported healthy.\n\n"
    "Four separate refusals, because they fail in four different ways and mean four "
    "different things: the file is missing; the file is empty or far too small to be "
    "weights; the file does not load; or it loads and is the wrong KIND of model -- a "
    "classifier has no boxes, so it can never answer §12.1's «рядом с человеком» and "
    "would run at full GPU cost in silence forever. There is a fifth: weights that load "
    "and detect but cannot produce any class this camera is configured to alarm on.\n\n"
    "Prints the file, its size, a fingerprint of it, its classes, and what a human wrote "
    "in `weapons.model.evaluated_on`. Nothing here evaluates anything -- a number "
    "invented by this command would be the most dangerous kind of provenance."
)

REPORT_DESCRIPTION = (
    "Can this camera see an object of a given size at a given distance?\n\n"
    "Pinhole arithmetic at the resolution the WORKER analyses, read from this camera's "
    "own merged config and never assumed -- the same question `qorgan identity "
    "camera-report` asks about faces, and it has the same kind of answer.\n\n"
    "What we told the school (docs/questions-for-school.md §7): a knife in a hand at the "
    "entrance is a 100+ px object and will work; the same knife down a corridor at 15 m "
    "is ~15 px and WILL NEVER WORK. That is not a threshold problem and no confidence "
    "setting moves it.\n\n"
    "A PASS is a necessary condition, not a promise: this models optics and nothing else "
    "-- not motion blur, not substream compression, not whether a blade is edge-on, and "
    "certainly not how good any particular weights are. A FAIL is decisive.\n\n"
    "If it says NO, MOVE THE CAMERA. Do not lower weapons.min_object_pixels: there is "
    "nothing under it to recover."
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("weapons", help="weapons detection diagnostics (§12.1)")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    weights = sub.add_parser(
        "weights",
        help="load this camera's weapons weights and say exactly what they are",
        description=WEIGHTS_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    weights.add_argument("camera", help="a camera name from config/cameras")
    weights.set_defaults(func=cmd_weights)

    report = sub.add_parser(
        "camera-report",
        help="can this camera see an object that size at that distance?",
        description=REPORT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report.add_argument("camera", help="a camera name from config/cameras")
    report.add_argument(
        "--object-cm",
        type=float,
        default=DEFAULT_OBJECT_CM,
        help=f"visible size of the object (default {DEFAULT_OBJECT_CM:g}: a knife in a hand)",
    )
    report.add_argument(
        "--hfov-deg",
        type=float,
        default=None,
        help=(
            "the lens's horizontal field of view. Defaults to this camera's own "
            "`weapons.lens_hfov_degrees`, so the terminal and the /weapons panel cannot "
            "give different answers about the same camera. Read the real figure off the "
            "camera's datasheet and put it in the YAML -- the answer moves a long way "
            f"with it (the schema's own default is {DEFAULT_HFOV_DEGREES:g}: a 4 mm lens "
            'on 1/2.8", CHOSEN and not measured).'
        ),
    )
    report.set_defaults(func=cmd_camera_report)


def _weapons_camera(name: str):
    """The named camera, if it is a weapons camera. Raises LookupError otherwise."""
    from qorgan.config.camera import WeaponsCamera
    from qorgan.config.loader import load_cameras

    cameras = load_cameras()
    camera = cameras.get(name)
    if camera is None:
        raise LookupError(f"no camera called {name!r}. Known: {', '.join(sorted(cameras))}")
    if not isinstance(camera, WeaponsCamera):
        raise LookupError(
            f"{name!r} is a {camera.camera_type.value} camera, not a weapons camera. "
            "Only a weapons camera carries a `weapons:` block to report on."
        )
    return camera


def cmd_weights(args: argparse.Namespace) -> int:
    """Load the weights and print what they are, or print why they were refused."""
    from qorgan.config.loader import ConfigError
    from qorgan.weapons.model import YoloWeaponModel
    from qorgan.weapons.weights import WeaponWeightsUnusable

    try:
        camera = _weapons_camera(args.camera)
    except (ConfigError, LookupError) as exc:
        print(str(exc), file=sys.stderr)
        return UNANSWERED

    settings = camera.weapons.model
    try:
        model = YoloWeaponModel(
            settings,
            camera.name,
            tuple(camera.weapons.target_classes),
            device="cpu",
            confusable_classes=tuple(camera.weapons.confusable_classes),
        )
    except WeaponWeightsUnusable as refused:
        print(str(refused), file=sys.stderr)
        return REFUSED
    except Exception as exc:  # a torch/ultralytics failure: case 3 of weights.py
        print(
            f"the weapons weights at {settings.model} did not load: "
            f"{type(exc).__name__}: {exc}\n"
            "The file is present and is large enough to be a model, so this is a "
            "truncated download, the wrong format, or a git-lfs pointer committed as "
            "though it were the artefact. The pipeline does not start.",
            file=sys.stderr,
        )
        return REFUSED

    _print_weights(model.weights, camera)
    return USABLE


def _print_weights(weights, camera) -> None:
    print(f"{camera.name}: the weapons weights that would run\n")
    print(f"  file        {weights.file.path}")
    print(f"  size        {weights.file.size_mb:.1f} MB")
    print(f"  fingerprint {weights.file.fingerprint}  (sha256 of the first 1 MiB)")
    print(f"  task        {weights.task}")
    print(f"  classes     {', '.join(sorted(weights.class_names))}")
    print(f"  evaluated   {weights.evaluated_on or 'НЕ УКАЗАНО — nobody has written this down'}")
    usable = weights.describes(camera.weapons.target_classes)
    print(f"  targets     {', '.join(usable)}  (of {', '.join(camera.weapons.target_classes)})")
    _print_screen_three(weights, camera)


def _print_screen_three(weights, camera) -> None:
    """Whether «нож или ручка» can fire on these weights. **THE ONLY PLACE THAT CAN SAY.**

    `/weapons` cannot answer it -- the web process never opens the model (R3) -- so this
    command is the only thing in the system able to, and it did not try. `target_classes`
    were printed AND checked; `confusable_classes`, which are screen 3's whole input, were
    printed nowhere and checked against nothing. Measured: weights with classes
    `(knife, person, cell phone)` against the shipped confusables intersect to nothing, and
    everything reported healthy.
    """
    from qorgan.weapons.weights import inert_confusables

    declared = tuple(camera.weapons.confusable_classes)
    inert = inert_confusables(weights, declared)
    producible = [name for name in declared if name not in inert]
    print(f"  confusable  {', '.join(producible) or 'НИ ОДНОГО'}  (of {', '.join(declared)})")
    if not declared or not inert:
        return

    print()
    if len(inert) == len(declared):
        print(
            "  SCREEN 3 IS DEAD on this camera. These weights emit none of "
            "`weapons.confusable_classes`, so an ambiguous «нож или ручка» can never be "
            "withheld: the check is in the configuration and is not in the running system.",
            file=sys.stderr,
        )
    else:
        print(
            f"  SCREEN 3 IS PARTIAL: these weights cannot emit {', '.join(sorted(inert))}.",
            file=sys.stderr,
        )
    print(
        "  This is not a refusal, and deliberately so: a detector trained on knives has no "
        "reason to emit `pen`. But the class NAMES are a convention this project invented "
        "-- COCO says `cell phone` where the schema says `phone` -- and only somebody "
        "holding the weights can reconcile them. Bring `weapons.confusable_classes` into "
        "line with the `classes` line above.",
        file=sys.stderr,
    )


def cmd_camera_report(args: argparse.Namespace) -> int:
    """The optics question. Needs no weights, and is the one to ask first."""
    from qorgan.config.loader import ConfigError
    from qorgan.weapons.feasibility import assess

    try:
        camera = _weapons_camera(args.camera)
    except (ConfigError, LookupError) as exc:
        print(str(exc), file=sys.stderr)
        return UNANSWERED

    report = assess(
        camera=camera.name,
        # The resolution the WORKER analyses, from this camera's merged config. Never a
        # fleet-wide number: `identity/cli.py` learned that the hard way -- its own help
        # text listed which profiles override the default, said two, and there were three.
        frame_width=camera.capture.frame_width,
        min_object_pixels=camera.weapons.min_object_pixels,
        object_cm=args.object_cm,
        # This camera's own lens, unless the operator overrode it on the command line.
        # Never the module constant: two cameras with different lenses have different
        # answers, and §12.1's whole honesty problem is a fleet where they differ.
        hfov_deg=camera.weapons.lens_hfov_degrees if args.hfov_deg is None else args.hfov_deg,
    )
    print(report.summary())

    if report.usable:
        return USABLE
    print(
        "\nThis camera cannot see an object that size at any of these distances. Move it "
        "closer to where a weapon would be carried past it -- an entrance, a doorway, a "
        "turnstile. Lowering weapons.min_object_pixels does not help: there is nothing "
        "under it to recover.",
        file=sys.stderr,
    )
    return REFUSED

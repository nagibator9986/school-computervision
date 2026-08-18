"""Command line entry point: `qorgan <command>`."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qorgan.enums import UserRole
from qorgan.logging_setup import setup_logging
from qorgan.settings import get_settings

if TYPE_CHECKING:  # config pulls pydantic and every camera model; the CLI stays light.
    from qorgan.config.camera import CameraConfig
    from qorgan.config.canteen import SessionRules
    from qorgan.config.classroom import LessonRules

# MIN_PASSWORD_LENGTH used to live here. It now lives in `qorgan.passwords` beside
# BCRYPT_MAX_BYTES and is applied by `hash_password`, which this command and the /users
# page both go through -- one minimum, not one per front door.


def _cmd_config_validate(_args: argparse.Namespace) -> int:
    """Resolve every camera config and print it. Credentials are never shown."""
    from qorgan.capture.source import frame_source
    from qorgan.config.loader import ConfigError, load_cameras, load_workers

    try:
        cameras = load_cameras()
        workers = load_workers(cameras)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(cameras)} camera(s):\n")
    for name, camera in sorted(cameras.items(), key=lambda item: item[1].priority):
        group = workers.group_for(name)
        state = "" if camera.enabled else "  [disabled]"
        # `frame_source(...).where`, not `safe_url(camera.rtsp)`: this is the day-one
        # command an engineer runs to see what the fleet is about to do, and a camera that
        # has been pointed at a recording must SAY so here. Printing its RTSP URL would be
        # true about the config and false about what the worker is going to open -- the
        # same one-layer-off falsehood this whole tree keeps finding. Never `build_url`.
        print(f"  {name:<24} {camera.role.value:<16} {frame_source(camera).where}{state}")
        print(f"  {'':<24} worker: {group.name if group else '-'}  {camera.display_name}")

    print(f"\n{len(workers.groups)} worker group(s):")
    for group in workers.groups:
        print(f"  {group.name:<24} {group.device:<10} {', '.join(group.cameras)}")

    _print_counter_drift(cameras)
    print("\nOK")
    return 0


def _print_counter_drift(cameras: dict[str, CameraConfig]) -> None:
    """Say, on the day-one command, that the detector's frame counters were chosen for a
    different frame rate than the fleet runs at.

    This is not a config error -- nothing here is misconfigured -- so it does not fail the
    command. It is printed because the alternative is what already happened: the assumption
    sat in a docstring, nobody knew to ask, and the eval harness spent the project grading
    production at 10 fps for a loop running at 15. An assumption you have to already know
    about is not documented.
    """
    from qorgan.detection.constants import ASSUMED_ANALYSIS_FPS, counter_drift

    # Drift within a percent of 1.0 is a rounding artefact, not a divergence worth a
    # paragraph on the setup screen.
    negligible = 0.01

    rates = {camera.capture.analysis_fps for camera in cameras.values()}
    drifted = {rate for rate in rates if abs(counter_drift(rate) - 1.0) > negligible}
    if not drifted:
        return

    print(
        f"\nNOTE — the detector's frame counters assume {ASSUMED_ANALYSIS_FPS:g} fps "
        f"(src/qorgan/detection/constants.py):"
    )
    for name, camera in sorted(cameras.items()):
        rate = camera.capture.analysis_fps
        drift = counter_drift(rate)
        if abs(drift - 1.0) <= negligible:
            continue
        # A counter is a number of FRAMES; what it is worth is a number of SECONDS, and
        # that is the assumed rate over the real one. Above 1.0 the counter spans longer
        # than intended, below it shorter. Printed as a duration rather than as the raw
        # drift because "1.5x the rate" and "1.5x the time" point in opposite directions,
        # and stating the ratio without its unit is how this whole divergence started.
        spans = ASSUMED_ANALYSIS_FPS / rate
        print(
            f"  {name:<24} analyses at {rate:g} fps  ->  every frame counter spans "
            f"{spans:.2f}x the time it was chosen for"
        )
    print(
        "  Not an error and not rescaled here: the counters are read by three detection\n"
        "  modules, so changing them changes what the detector does. That is an on-site\n"
        "  decision with real labels. Do not re-derive a threshold from an eval run\n"
        "  without reading this line first."
    )


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Is this machine actually able to run the AI stack on the GPU?"""
    from qorgan.gpu import inspect_gpu

    report = inspect_gpu()
    print(f"torch:            {report.torch_version}")
    print(f"CUDA device:      {report.device_name or 'NONE'}")
    print(f"onnx advertises:  {', '.join(report.onnx_providers) or 'none'}")
    # The line that matters. The one above is what the wheel was COMPILED with, and it
    # says CUDA even when every session silently runs on the CPU (spec §3).
    print(f"onnx SESSION ran: {report.onnx_session_provider or 'FAILED TO BUILD'}")

    if report.ok:
        print("\nOK — both torch and onnxruntime can use the GPU.")
        return 0

    print("\nNOT OK:")
    for problem in report.problems():
        print(f"  - {problem}")
    return 1


def _canteen_session_rules(cameras: dict[str, CameraConfig]) -> SessionRules | None:
    """The canteen's meal-session rules, or None where the install has no canteen.

    Off the first canteen camera, exactly as `worker/entrypoint.py::_sessions_for` reads
    them: `max_session_minutes` is one fact about one canteen that happens to be written
    down in a per-camera block.
    """
    from qorgan.config.camera import CanteenCamera

    return next(
        (c.canteen.session for c in cameras.values() if isinstance(c, CanteenCamera)), None
    )


def _classroom_lesson_rules(cameras: dict[str, CameraConfig]) -> LessonRules | None:
    """The lesson rules, or None where the install has no classroom camera.

    Off the first classroom camera, exactly as `_canteen_session_rules` reads the meal
    rules above: `max_lesson_minutes` is one fact about how long a lesson may run, which
    happens to be written down in a per-camera block.
    """
    from qorgan.config.camera import ClassroomCamera

    return next(
        (c.classroom.lesson for c in cameras.values() if isinstance(c, ClassroomCamera)), None
    )


def _cmd_supervisor(_args: argparse.Namespace) -> int:
    from qorgan.config.loader import load_cameras, load_workers
    from qorgan.supervisor import Supervisor

    cameras = load_cameras()
    workers = load_workers(cameras)
    # The supervisor force-closes the meal sessions no exit camera ever closed, and the
    # lessons no empty room ever ended. It is the only process that can: see
    # `Supervisor._sweep_stale_sessions`. Passing the rules is what makes the sweeps run
    # at all -- `close_sessions_nobody_exited` spent five releases reachable only from a
    # test for want of exactly this line.
    Supervisor(
        workers,
        session_rules=_canteen_session_rules(cameras),
        lesson_rules=_classroom_lesson_rules(cameras),
    ).run()
    return 0


def _cmd_web(_args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    # 127.0.0.1 by default. Exposing a dashboard of children's faces to the school
    # network is an explicit act, not a default (the legacy bound 0.0.0.0 with no auth).
    uvicorn.run(
        "qorgan.web:create_app",
        factory=True,
        host=settings.web_host,
        port=settings.web_port,
        log_config=None,
    )
    return 0


def _cmd_user_add(args: argparse.Namespace) -> int:
    """The other front door onto the same table as the /users page.

    Every rule it applies -- the password minimum, bcrypt's 72 bytes, the username, the
    uniqueness check -- now comes from `qorgan.accounts`, which is also what the web route
    calls. Restated here, each of them would be a second number that nobody ever sees
    beside the first, and the laxer of the two doors becomes how weak accounts get made.

    What stays here is what only a terminal has: the typed-twice confirmation. There is no
    "show password" and no going back to fix a typo, so a mistyped password would create an
    account nobody can log into and give no hint why. What only a SHELL has -- the
    `superadmin` role and `--school` -- is in `_write_the_account` below.
    """
    from qorgan.accounts import AccountError, SchoolUndecided
    from qorgan.passwords import PasswordRejected
    from qorgan.schools import SchoolError

    refusal = _impossible_combination(args)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    password = getpass.getpass("password: ")
    if password != getpass.getpass("repeat: "):
        print("passwords do not match", file=sys.stderr)
        return 1

    try:
        _write_the_account(args, password)
    except SchoolUndecided as exc:
        # The one refusal a flag can answer, so it names the flag. The RULE lives in
        # `qorgan.accounts`, where both doors meet it; what a shell types to satisfy it is
        # this door's own business. An error that states a rule and not its remedy is the
        # shape of unhelpfulness this project keeps paying for.
        print(
            f"{exc}\n\nName one, for example:\n"
            f"  qorgan user add {args.username} --role {args.role} --school <slug>",
            file=sys.stderr,
        )
        return 1
    except (AccountError, PasswordRejected, SchoolError) as exc:
        # `exc` names the rule, never the password: this goes to a terminal that scrolls
        # back and, on a scheduled run, into a log file.
        print(str(exc), file=sys.stderr)
        return 1

    print(f"created {args.username} ({args.role})")
    return 0


def _impossible_combination(args: argparse.Namespace) -> str | None:
    """Argument pairs that cannot succeed, refused BEFORE anything is typed twice.

    Nothing should be typed twice for a command that cannot work, and a refusal arriving
    after two password prompts reads as the password having been wrong -- which sends
    somebody to reset a password that was correct.
    """
    if args.role == UserRole.SUPERADMIN.value and args.school is not None:
        return (
            "--school does not apply to the superadmin: that account belongs to no school, "
            "which is what the role is. Drop --school."
        )
    return None


def _write_the_account(args: argparse.Namespace, password: str) -> None:
    """Which door this is: the installation's, or one school's.

    **`--role superadmin` is what this door has and the browser form does not.**
    `parse_role` refuses that value and must go on refusing it -- the accounts page is
    gated on MANAGE_USERS, which a school's own ADMIN holds, so a form that accepted it
    would let one headteacher mint an account reaching every school on the installation. A
    shell on the server is the installation and a form served to a tenant is not, so this
    routes it to `create_superadmin` rather than widening the rule for both doors. Until
    that existed, the command offered the value in `--help`, took it from argparse and then
    refused it as "unknown role".

    That branch takes no school. Every other role belongs to exactly one, named by
    `--school` or -- on the single-school installation every deployment is today --
    inferred by `resolve_school_id`. `school_id_for_slug` turns the name a person can type
    into the id the tables carry.
    """
    from qorgan.accounts import create_account, create_superadmin, parse_role
    from qorgan.schools import school_id_for_slug

    if args.role == UserRole.SUPERADMIN.value:
        create_superadmin(args.username, password)
        return

    school_id = school_id_for_slug(args.school) if args.school is not None else None
    create_account(args.username, password, parse_role(args.role), school_id=school_id)


def _cmd_janitor(args: argparse.Namespace) -> int:
    """Retention. Run this on a schedule: nothing in this system grows forever."""
    from qorgan.maintenance import sweep

    report = sweep(media_days=args.media_days, attempt_days=args.attempt_days, dry_run=args.dry_run)
    print(("DRY RUN — " if args.dry_run else "") + report.summary())
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    """A copy of the database. Put this on a schedule too — see docs/windows-autostart.md.

    Errors are printed, not raised: this runs unattended from Task Scheduler, which
    understands an exit code and cannot read a traceback.
    """
    from qorgan.maintenance import BackupError, backup_database

    try:
        report = backup_database(Path(args.to) if args.to else None)
    except BackupError as exc:
        print(f"backup failed: {exc}", file=sys.stderr)
        return 1

    print(f"database backed up to {report.summary()}")
    return 0


def _cmd_db_upgrade(_args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    settings = get_settings()
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, "head")
    print(f"database at head: {settings.database_url}")
    return 0


def _cmd_db_current(_args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.current(config, verbose=True)
    return 0


def _add_maintenance_parsers(subparsers: argparse._SubParsersAction) -> None:
    """The two that belong on a schedule. Neither is a default: see
    `docs/windows-autostart.md` §4."""
    backup = subparsers.add_parser("backup", help="copy the database to a dated file")
    backup.add_argument(
        "--to",
        metavar="PATH",
        help="where to write it (default: <database dir>/backups/qorgan-<date>.sqlite3)",
    )
    backup.set_defaults(func=_cmd_backup)

    janitor = subparsers.add_parser("janitor", help="delete expired media and stale debug rows")
    janitor.add_argument("--media-days", type=int, default=90)
    janitor.add_argument("--attempt-days", type=int, default=30)
    janitor.add_argument("--dry-run", action="store_true")
    janitor.set_defaults(func=_cmd_janitor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qorgan", description="Qorgan AI v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="configuration")
    config_sub = config_parser.add_subparsers(dest="subcommand", required=True)
    config_sub.add_parser("validate", help="load and validate every camera config").set_defaults(
        func=_cmd_config_validate
    )

    db_parser = subparsers.add_parser("db", help="database")
    db_sub = db_parser.add_subparsers(dest="subcommand", required=True)
    db_sub.add_parser("upgrade", help="migrate to head").set_defaults(func=_cmd_db_upgrade)
    db_sub.add_parser("current", help="show the revision").set_defaults(func=_cmd_db_current)

    from qorgan.classvision import add_parser as add_classvision_parser
    from qorgan.evaluation.cli import add_parser as add_eval_parser
    from qorgan.faces.cli import add_parser as add_pupils_parser
    from qorgan.identity.cli import add_parser as add_identity_parser
    from qorgan.planning.cli import add_parser as add_planning_parser
    from qorgan.weapons.cli import add_parser as add_weapons_parser

    add_eval_parser(subparsers)
    # The offline classroom analyses (`classvision/INTEGRATION.md`). Imported inside this
    # function like every other group, so `qorgan --help` does not pull SQLAlchemy models.
    add_classvision_parser(subparsers)
    add_pupils_parser(subparsers)
    add_identity_parser(subparsers)
    add_planning_parser(subparsers)
    add_weapons_parser(subparsers)

    subparsers.add_parser("doctor", help="check the GPU stack").set_defaults(func=_cmd_doctor)
    subparsers.add_parser("supervisor", help="run the worker fleet").set_defaults(
        func=_cmd_supervisor
    )
    subparsers.add_parser("web", help="run the dashboard").set_defaults(func=_cmd_web)

    _add_maintenance_parsers(subparsers)

    _add_user_parsers(subparsers)

    return parser


def _add_user_parsers(subparsers: argparse._SubParsersAction) -> None:
    """The second front door onto the `users` table; the accounts page is the first."""
    user_parser = subparsers.add_parser("user", help="dashboard users")
    user_sub = user_parser.add_subparsers(dest="subcommand", required=True)
    add_user = user_sub.add_parser("add", help="create a user (prompts for the password)")
    add_user.add_argument("username")
    # **The whole enum, INCLUDING `superadmin`, and `_cmd_user_add` is what makes that
    # honest rather than a lie.** The browser form's list is `accounts.ASSIGNABLE_ROLES`,
    # which excludes this one on purpose; these choices come off `UserRole` because this
    # door IS the installation. A command that offers a choice it will always reject --
    # and then calls that choice "unknown" -- is a lie in two directions at once, and that
    # is what this line was while nothing could create a superadmin.
    #
    # Read off the enum rather than typed out, so a role added later is offered here
    # automatically; `test_every_role_the_command_offers_can_actually_be_created` is what
    # then requires it to work.
    #
    # The help text says SHELL ONLY, because the five values are NOT peers and nothing
    # else on that screen says so: four of them the accounts page can also assign, and
    # this one it refuses. That asymmetry is the whole security property of the role, and
    # this string is what the person choosing a role actually reads.
    add_user.add_argument(
        "--role",
        choices=[r.value for r in UserRole],
        default=UserRole.OPERATOR.value,
        help=(
            "superadmin is SHELL ONLY -- the accounts page cannot assign it. It belongs "
            "to no school and manages the schools register (§14)"
        ),
    )
    # **The flag two docstrings already promised.** `db/models/school.py` named it on
    # `School.slug` before it was built, and `schools.rename_school` refuses to edit a
    # slug because this flag carries it. Until it existed there was no way to say which
    # school an account was for, so `qorgan user add <name> --role admin` on a two-school
    # installation raised `UndecidedSchool` -- a RuntimeError nothing caught -- and the
    # installer got a traceback on a day-one action, on the one configuration this branch
    # exists to enable.
    #
    # Optional, not required: every deployment today serves one school, and `resolve_
    # school_id` falls back to the only one there is. It becomes obligatory exactly when
    # there are two, which is when a default would have to guess.
    add_user.add_argument(
        "--school",
        metavar="SLUG",
        help="which school the account belongs to; required once a second school exists",
    )
    add_user.set_defaults(func=_cmd_user_add)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("cli")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

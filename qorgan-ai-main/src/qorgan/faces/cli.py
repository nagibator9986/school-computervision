"""`qorgan pupils` — import photographs, and answer the canteen's questions."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from qorgan.config.identity import FaceModelSettings

if TYPE_CHECKING:
    # Type-only: `cmd_report` imports `day_report` lazily, and this must not make importing
    # the CLI pull the report layer (and its engine) in at parse time.
    from qorgan.canteen.reports import DayReport


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("pupils", help="pupils, staff and the canteen record")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    _add_roster_parser(sub)

    import_cmd = sub.add_parser(
        "import", help="import the same directory tree, zipped (this is the web upload path)"
    )
    import_cmd.add_argument("archives", type=Path, nargs="+")
    import_cmd.set_defaults(func=cmd_import)

    gallery_cmd = sub.add_parser(
        "gallery-report",
        help="can this gallery work at all, and who is enrolled twice?",
        description=(
            "The cross-person similarity matrix, the duplicate enrolments (two ids, one "
            "human -- six of them in this school), the impostor ceiling and the floor it "
            "puts under min_score, the photos that could not be enrolled, and what the "
            "impostor rate extrapolates to at 500, 800 and 1200 pupils."
        ),
    )
    gallery_cmd.set_defaults(func=cmd_gallery_report)

    _add_merge_parser(sub)

    report_cmd = sub.add_parser(
        "report", help="who ate, and who has no meal record, on a given day"
    )
    report_cmd.add_argument(
        "--day", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)"
    )
    report_cmd.add_argument("--csv", type=Path, help="write the full report to a CSV file")
    report_cmd.set_defaults(func=cmd_report)


def _add_roster_parser(sub: argparse._SubParsersAction) -> None:
    """The photo directory, the names table, and the explicitly decided carry-over list."""
    roster_cmd = sub.add_parser(
        "import-roster",
        help="import the school's photo directory. The FOLDER decides who is who.",
        description=(
            "Walks the tree. A folder named 1-А .. 11-Б is a class of pupils; `staff` is "
            "staff; `учитель` is staff with the position учитель. The FILENAME only "
            "carries the school's id -- and it lies about the type: the учитель folder "
            "contains files named student_469_….jpg. Two filename forms are read: "
            "student|staff_<id>_<timestamp>.jpg (2025) and <id>.jpg (2026). Anything else "
            "is a HARD ERROR naming the file. It is never a guessed identity."
        ),
    )
    roster_cmd.add_argument("directory", type=Path, help="e.g. student_photos/student_photos")
    roster_cmd.add_argument(
        "--names",
        type=Path,
        help=(
            "CSV of external_id,full_name,class_name -- the school's ID -> name table. "
            "Joined on the ID, never on the photo filename: the school's own sheet gets "
            "the extension wrong for three of its 141 rows, and a filename join loses "
            "those three children in silence. Without this the panel shows `Ученик 10, "
            "1А`. A row with no photograph, or a photograph with no row, REFUSES the "
            "import before anything is written."
        ),
    )
    roster_cmd.add_argument(
        "--carry-over",
        type=Path,
        help=(
            "file of external_ids (one per line, # comments) to take from an OLD delivery "
            "-- the staff the 2026 delivery does not contain. Taking the whole staff/ and "
            "учитель/ folders instead would recreate two people who are each enrolled "
            "twice (staff_464 = student_477 at 0.999; staff_334 = student_470 at 0.984) "
            "and four whose photographs contain no detectable face. An id on the list that "
            "is not in the tree refuses the import: a typo must not read as success."
        ),
    )
    roster_cmd.set_defaults(func=cmd_import_roster)


def _add_merge_parser(sub: argparse._SubParsersAction) -> None:
    """Both ids, named in full, by a human. There is no --all and no winner-picking."""
    merge_cmd = sub.add_parser(
        "merge",
        help="two ids are one human: re-point everything onto one of them",
        description=(
            "Executes a decision a HUMAN made. `gallery-report` finds the duplicates; it "
            "does not resolve them, because which id is canonical is a decision only the "
            "school can make -- and 7-А 438/439 may be identical twins, which arithmetic "
            "cannot settle. This never runs automatically. Both ids must be named in "
            "full; there is no --all, and nothing here picks a winner. Merging across the "
            "pupil/staff line decides whether that person is FED -- staff never open a "
            "meal session -- and the summary says so out loud."
        ),
    )
    merge_cmd.add_argument("keep_id", help="the external_id to keep, e.g. staff_334")
    merge_cmd.add_argument("drop_id", help="the external_id to retire, e.g. student_470")
    merge_cmd.add_argument(
        "--reactivate",
        action="store_true",
        help=(
            "revive a keep_id that an earlier merge retired. This is how a merge made in "
            "the WRONG DIRECTION is undone -- without it, merging back is refused because "
            "the id you are merging into is inactive. Not the default: an inactive id can "
            "also mean the person left the school, and reviving one of those silently "
            "would be its own quiet mistake."
        ),
    )
    merge_cmd.set_defaults(func=cmd_merge)


def cmd_import_roster(args: argparse.Namespace) -> int:
    from qorgan.faces.importer import ImportReport, import_directory
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.carryover import read_carry_over
    from qorgan.identity.names import read_names
    from qorgan.identity.roster import BadFilename

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 1

    settings = FaceModelSettings()
    recognizer = FaceRecognizer.shared(settings)

    try:
        names = read_names(args.names) if args.names else None
        only = read_carry_over(args.carry_over) if args.carry_over else None
        report = import_directory(
            args.directory, recognizer, settings, report=ImportReport(), names=names, only=only
        )
    except (BadFilename, ValueError) as exc:
        # Loud, and it names the file. A refusal is recoverable; a quiet guess is a child
        # eating someone else's lunch.
        print(f"\nIMPORT REFUSED: {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())

    # Non-zero when somebody was imported that the system can NEVER recognise -- the four
    # staff photographs with no detectable face are the known case. They are created on
    # purpose (they are on the roster whether or not we can see them, see `_store`), and
    # `summary()` names every one of them; this is what stops a script sailing past that
    # paragraph. Same reasoning, and the same exit code, as `gallery-report` on duplicates:
    # a person who can never be recognised sits on every "no meal record" report for ever,
    # and the fix is a new photograph from the school, which nobody asks for if the import
    # says nothing but a cheerful total.
    return 1 if report.has_unenrollable else 0


def cmd_import(args: argparse.Namespace) -> int:
    from qorgan.faces.importer import ImportReport, import_archive
    from qorgan.faces.recognizer import FaceRecognizer
    from qorgan.identity.roster import BadFilename

    settings = FaceModelSettings()
    recognizer = FaceRecognizer.shared(settings)
    report = ImportReport()

    for archive in args.archives:
        if not archive.is_file():
            print(f"  ! not a file, skipping: {archive}")
            continue
        print(f"  importing {archive.name} ...")
        try:
            import_archive(archive, recognizer, settings, report=report)
        except (BadFilename, ValueError) as exc:
            print(f"\nIMPORT REFUSED: {exc}", file=sys.stderr)
            return 1

    print()
    print(report.summary())
    return 0


def cmd_gallery_report(_args: argparse.Namespace) -> int:
    from qorgan.identity.report import gallery_report

    report = gallery_report(FaceModelSettings())
    print(report.summary())

    # A duplicate enrolment is not a failure to import; it is a question for the school.
    # It exits non-zero so a script does not sail past it.
    return 1 if report.duplicates else 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Never automatic, and it never picks a winner: the human names both ids."""
    from qorgan.identity.merge import merge_persons, resolve_external

    try:
        keep = resolve_external(args.keep_id)
        drop = resolve_external(args.drop_id)
        result = merge_persons(keep, drop, reactivate=args.reactivate)
    except (LookupError, ValueError) as exc:
        # Refusing is recoverable. Merging the wrong two children is not: one of them
        # stops existing.
        print(f"merge refused: {exc}", file=sys.stderr)
        return 1

    print(result.summary())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from qorgan.canteen.reports import day_report

    day = args.day or _today()
    report = day_report(day)

    print(report.summary())
    print()

    if report.never_came:
        # "No meal record", NOT "did not eat" -- a child who ate unrecognised lands in
        # this list under their real name. See `DayReport.did_not_eat`.
        print(f"No meal record today ({len(report.never_came)}):")
        for person in sorted(report.never_came, key=lambda p: (p.class_name or "", p.display)):
            print(f"  {person.class_name or '—':<6} {person.display}")

    # Unconditional: the claim is in `summary()` on every day, so the caveat must be too.
    # One wording for every surface -- see `DayReport.caveat()`.
    for line in report.caveat():
        print(f"\n{line}")

    if report.came_but_did_not_eat:
        print(f"\nCame but did not eat ({len(report.came_but_did_not_eat)}):")
        for meal in report.came_but_did_not_eat:
            dwell = f"{meal.dwell_seconds:.0f}s" if meal.dwell_seconds else "—"
            print(f"  {meal.class_name or '—':<6} {meal.display:<30} {dwell}")

    _print_recognition_counts(report)

    if args.csv:
        _write_csv(args.csv, report)
        print(f"\nwrote {args.csv}")

    return 0


def _print_recognition_counts(report: DayReport) -> None:
    """Both counts, ALWAYS, including zero -- unlike the lists, which print only when full.

    That asymmetry is deliberate, and the rule is list-vs-scalar, not consistency.

    A list is not a measurement. "Nobody came but did not eat" is fully expressed by
    printing nothing, and a "(0):" heading over an empty list is noise.

    These two are scalar INSTRUMENTS, and their entire job is to tell a measured zero
    apart from an unmeasured one. Hidden at zero, the blank says "zero today", "never
    computed" and "silently dropped between `day_report` and here" all at once, and the
    reader cannot tell which -- so if either count is ever dropped, this command simply
    goes quiet, and a broken instrument reads exactly like a good day. That is not
    hypothetical: `forced_unknown` was dropped that way on the web page and went unnoticed
    for precisely this reason. A printed `0` is a finding; a blank is not.

    Two counts, never a total: the predicates overlap (see `DayReport.forced_unknown`).
    """
    print(
        f"\n{report.unknown_sessions} session(s) could not be attributed to a pupil. "
        "If this number is large, face recognition is not working — the legacy system "
        "had 1816 of 1820."
    )

    print(
        f"\nСессий закрыто по таймауту (выход не распознан): {report.forced_unknown}. "
        "This is the measured price of the exit camera's strict min_score (spec A §2.2) "
        "— a session we could not close is a hole we can count, not a false meal record "
        "we cannot detect. If this number is large, re-check the exit threshold."
    )


def _today() -> date:
    from datetime import datetime

    from qorgan.settings import get_settings

    return datetime.now(tz=get_settings().tz).date()


def _write_csv(path: Path, report) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "name", "outcome", "dwell_seconds"])

        for meal in sorted(report.ate, key=lambda m: (m.class_name or "", m.display)):
            writer.writerow(
                [meal.class_name or "", meal.display, "ate", f"{meal.dwell_seconds or 0:.0f}"]
            )
        for meal in report.came_but_did_not_eat:
            outcome = meal.outcome.value if meal.outcome else "unknown"
            writer.writerow(
                [meal.class_name or "", meal.display, outcome, f"{meal.dwell_seconds or 0:.0f}"]
            )
        for person in report.never_came:
            # `never_came` is a stable data contract with the school's spreadsheet, not an
            # assertion about the child -- see the web export, which must write the exact
            # same token. The caveat below, not the token, is what carries the truth.
            writer.writerow([person.class_name or "", person.display, "never_came", ""])

        # Same spreadsheet, same claim, same caveat as the web export: a row reading
        # `never_came` beside a real child's name is what reaches the parent, so the
        # sentence that makes it honest has to travel in the file with it. A blank
        # separator row, then one caveat per row -- csv.writer quotes the commas for us.
        writer.writerow([])
        for line in report.caveat():
            writer.writerow([line, "", "", ""])

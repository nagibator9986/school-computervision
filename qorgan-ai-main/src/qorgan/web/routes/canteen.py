"""The canteen journal, and the report the school actually asks for."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from datetime import date, datetime

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from qorgan.canteen.reports import day_report
from qorgan.roles import Capability
from qorgan.settings import get_settings
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

router = APIRouter()

# The canteen, and only the canteen. §14: a canteen worker holds this and nothing else.
viewer = Depends(require_capability(Capability.VIEW_CANTEEN))


def _today() -> date:
    return datetime.now(tz=get_settings().tz).date()


@router.get("/canteen")
def canteen_page(request: Request, day: str | None = None, user=viewer) -> Response:
    when = _parse(day)
    report = day_report(when, school_of(user))

    return render(
        request,
        "canteen.html",
        day=when.isoformat(),
        report=report,
        ate=sorted(report.ate, key=lambda m: (m.class_name or "", m.display)),
        did_not_eat=sorted(
            report.came_but_did_not_eat, key=lambda m: (m.class_name or "", m.display)
        ),
        never_came=sorted(report.never_came, key=lambda p: (p.class_name or "", p.display)),
    )


def _meal_row(meal, outcome: str) -> list[str]:
    """One meal, one row. `ate` and `came_but_did_not_eat` differ only in the token."""
    return [meal.class_name or "", meal.display, outcome, f"{meal.dwell_seconds or 0:.0f}"]


def _csv_rows(report) -> Iterator[str]:
    """The export, one row at a time, quoted by the `csv` module rather than by hand.

    Never an f-string: `display` holds a comma for every pupil in the school TODAY -- with
    no ID -> name roster `display_name` returns `Ученик 333, 5-А`, and says so. Interpolated,
    that comma opened a fifth column and pushed `outcome` and `dwell_seconds` one right, so
    `never_came` -- kept precisely BECAUSE the school may filter on it -- was not in the
    outcome column at all, and the filter matched nothing. `_write_csv` in the CLI writes
    these same rows through the same module; two exports of one day disagreeing is how a
    value goes true in one layer and wrong in the next.

    Still streamed: a row is written into the buffer, yielded, and the buffer emptied.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    def flush() -> str:
        row = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        return row

    writer.writerow(["class", "name", "outcome", "dwell_seconds"])
    # utf-8-sig: without the BOM, Excel opens Cyrillic as mojibake and a headteacher
    # concludes, reasonably, that the system is broken.
    yield "﻿" + flush()

    for meal in report.ate:
        writer.writerow(_meal_row(meal, "ate"))
        yield flush()

    for meal in report.came_but_did_not_eat:
        writer.writerow(_meal_row(meal, meal.outcome.value if meal.outcome else "unknown"))
        yield flush()

    for person in report.never_came:
        # `never_came` is a CODE, not a sentence: a data contract with the school, whose
        # spreadsheet may filter or macro on this exact string, so renaming it would
        # silently match nothing -- the same falsehood pushed across the boundary, to
        # break where we cannot see it. False on its own (an unrecognised child who ate
        # lands here), it never travels on its own: the caveat below is IN the file and
        # carries the truth. Keep the caveat, keep the token. Same token in `_write_csv`.
        writer.writerow([person.class_name or "", person.display, "never_came", ""])
        yield flush()

    yield _caveat_rows(report, writer, flush)


def _caveat_rows(report, writer, flush) -> str:
    """The caveat has to be IN the file. This spreadsheet is what the school actually
    reads and forwards; a row reading `never_came` beside a real child's name is a claim
    we cannot support, and it is the one that reaches the parent.

    Shape: a blank separator row, then one caveat per row in the first column. A trailing
    notes block is what a spreadsheet reader expects at the foot of a sheet, and it
    survives Excel. Quoting matters -- the sentences contain commas -- and it is the same
    `csv.writer` that quotes the rows above, which is the point: the care this comment
    described was applied only here, one line below the data rows, where the hazard was
    just as live and went unguarded for far longer.
    """
    writer.writerow([])
    for line in report.caveat():
        writer.writerow([line, "", "", ""])
    return flush()


@router.get("/canteen/export.csv")
def export(day: str | None = None, user=viewer) -> StreamingResponse:
    """The school asks for this in a spreadsheet, and the school is right to.

    Scoped like the page it exports, and worth stating because this is the artefact that
    LEAVES the building: it is mailed, forwarded and opened on other people's computers.
    A CSV carrying another school's children would not be a page somebody closed.
    """
    when = _parse(day)
    report = day_report(when, school_of(user))

    return StreamingResponse(
        _csv_rows(report),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="canteen-{when.isoformat()}.csv"'},
    )


def _parse(day: str | None) -> date:
    if not day:
        return _today()
    try:
        return date.fromisoformat(day)
    except ValueError:
        return _today()

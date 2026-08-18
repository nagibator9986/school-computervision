"""The lesson reports: counts per anonymous track, never without their caveat.

Two routes, one capability, and the capability was added in the same change as the routes
-- `qorgan.roles` is explicit that a permission guarding nothing is a guess.

**The caveat is not decoration and it is not optional.** `LessonReport.caveat()` is
rendered by both templates, from the one wording in `classroom/reports.py`, for the same
reason `DayReport.caveat()` is: the canteen learned that four hand-written copies of a
caveat is four chances for one of them to quietly stop being true. The two sentences it
always carries are the two that matter -- a track is not a child, and no threshold behind
these numbers has been checked against a real lesson.

There is no CSV export here, deliberately, and the contrast with `/canteen` is the point.
The canteen exports because the school asked for a spreadsheet it forwards. A file of
per-child-looking rows, detached from the page that explains they are anonymous tracks
counted by unvalidated thresholds, is precisely the artefact that would get forwarded to a
parent as though it meant something. When the school asks for the export, it gets built
with the caveat inside the file, the way `_caveat_rows` does it for the canteen.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import Response

from qorgan.classroom.reports import lesson_report, recent_lessons
from qorgan.roles import Capability
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

router = APIRouter()

viewer = Depends(require_capability(Capability.VIEW_LESSON_METRICS))

# How many lessons the index shows. Bounded because a page that renders every lesson the
# school has ever recorded gets slower every week until somebody calls it broken.
RECENT_LIMIT = 50


@router.get("/lessons")
def lessons_page(request: Request, user=viewer) -> Response:
    """Recent observation periods, newest first. Each carries its own caveat."""
    return render(
        request, "lessons.html", lessons=recent_lessons(RECENT_LIMIT, school_of(user))
    )


@router.get("/lessons/{lesson_id}")
def lesson_page(request: Request, lesson_id: int, user=viewer) -> Response:
    """One lesson: every track observed, including the ones that did nothing.

    Every track, not only the ones with a non-zero count -- "nobody put their hand up"
    and "we did not look" must not render identically. See `classroom/reports.py`.
    """
    report = lesson_report(lesson_id, school_of(user))
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such lesson")
    return render(request, "lesson.html", report=report)

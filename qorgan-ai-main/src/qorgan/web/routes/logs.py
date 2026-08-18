"""The diagnostics page: what is wrong right now, and what went wrong recently.

Client §9 asks for five things under one heading -- "**Logs**: ошибки камер; ошибки
моделей; ошибки базы; причины пропуска тревог; перезапуски" -- and they do not share a
source. `qorgan.diagnostics` is where each one is fetched from and where the reasoning for
each choice is written down; this file only assembles them and hands them to a template.

Three things this route does NOT do, each paid for by the system it replaces:

  * **It changes nothing.** No rotation, no cleanup, no heartbeat, no worker poke. The
    legacy had `POST /page-activate/{page}` restart the AI workers -- with a five-second
    `thread.join()` inside the HTTP handler -- every time somebody opened a tab, so which
    cameras were covered depended on which tab was open. There is no unsafe method here at
    all; the filter is a GET, which is also why no form on this page carries a CSRF token.
    `render()` still puts one on the page, so the logout form in the header works and so
    the next author to add a form here starts from a page that has one.

  * **It renders one page.** Both lists are paginated -- the alerts in SQL, the journal
    over a bounded tail of each log file. The legacy loaded the whole events table and the
    whole canteen log on every render, every 2.5 seconds, per client (audit M-19).

  * **It is not for everyone.** See `Capability.VIEW_DIAGNOSTICS`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from qorgan.diagnostics import alerts, logfiles
from qorgan.diagnostics.workers import worker_rows
from qorgan.roles import Capability
from qorgan.web.security import require_capability
from qorgan.web.templating import render

router = APIRouter()

# A log record quotes whatever the emitting module put in it: camera names, RTSP hosts,
# file paths, exception strings. §14 gives the оператор безопасности alerts, verdicts and
# clips -- not the server -- so this is the school's администратор/суперадминистратор and
# the developer, and it is a capability of its own rather than a corner of VIEW_CAMERAS.
viewer = Depends(require_capability(Capability.VIEW_DIAGNOSTICS))


@router.get("/logs")
def logs_page(
    request: Request,
    page: int = 1,
    alerts_page: int = 1,
    category: str | None = None,
    user=viewer,
) -> Response:
    """Two independent paginations on one page, because they are two different tables.

    A single shared `page` would move both lists at once, so paging to the third page of
    the journal would silently empty the alerts panel and read as "no undelivered alerts".

    Neither page number is clamped here. Each list clamps its own against its own count
    and hands back the page it actually served, because the number in the query string,
    the rows selected and the "3 / 3" on screen have to be one value -- computing it twice
    is how it becomes two.
    """
    return render(
        request,
        "logs.html",
        journal=logfiles.recent(category, page),
        categories=logfiles.CATEGORY_LABELS,
        category=category or "",
        workers=worker_rows(),
        # `user.school_id`, NOT `school_of(user)`. This page is the one place a schoolless
        # account is meant to be -- §14's "управление серверами" -- so refusing it a school
        # would 403 the суперадминистратор off the log viewer entirely. The alerts panel is
        # the only part of it that is a school's data, and `undelivered` answers None with
        # an empty page rather than with everybody's.
        undelivered=alerts.undelivered(alerts_page, user.school_id),
        telegram_configured=alerts.telegram_configured(),
    )

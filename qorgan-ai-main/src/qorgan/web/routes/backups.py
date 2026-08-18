"""Backups: has a copy of the database been taken, and take one now.

`qorgan backup` exists and runs at 03:00 from Task Scheduler. Nobody at the school has a
terminal, so nobody there can answer "did it run?" — and the backup that has silently not
happened since March is exactly the one they will reach for. This page answers it, and
lets a person take one without an engineer.

Three things it does NOT do, each on purpose:

  * **It does not reimplement the backup.** `VACUUM INTO`, the WAL, the read-back
    integrity check and the refusal to overwrite all live in `qorgan.maintenance.backup`
    and are called from here unchanged. A second implementation is a second set of
    guarantees to keep in step, and the one on the page would be the one nobody tests.
  * **It does not take a backup on a page load.** Only the POST does. See `backups_page`.
  * **It does not serve a backup file.** There is no download route, and the comment
    beside the capabilities below says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fastapi import APIRouter, Depends, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.logging_setup import get_logger
from qorgan.maintenance import BackupError, BackupFile, backup_directory, existing_backups
from qorgan.roles import Capability
from qorgan.settings import get_settings
from qorgan.web.security import current_user, require_capability
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()

# Reading the list and pressing the button are two different rights -- see `qorgan.roles`
# for why, at length. In one line: the first answers a question, the second writes a file
# containing every child in the school onto the disk that holds the recordings.
viewer = Depends(require_capability(Capability.VIEW_BACKUPS))
maker = Depends(require_capability(Capability.CREATE_BACKUP))

# Why there is no download route, stated where somebody would add one.
#
# A backup is not "a file about the school", it is the school: every face embedding, every
# meal, every incident summary, in one file small enough to attach to an email. Serving it
# over HTTP turns one authenticated session -- or one browser left open in a staffroom --
# into an offline, permanent copy of every child's data, and nothing afterwards can
# un-copy it. `/media` at least serves one photograph per request, behind a capability
# split precisely so that the incident evidence and the pupil register are separate
# grants; a backup contains BOTH and everything else besides.
#
# So the page lists names, times and sizes -- enough to answer "is there a backup and how
# old is it?" -- and moving a copy off the machine stays a deliberate act at the
# filesystem, performed by a named person. If that is ever revisited, the path must be
# confined with `paths.ensure_within` the way `web.routes.media` does it, and it must
# still be a different capability from this one.
#
# The same divisor `BackupReport.summary` uses, so the CLI and this page cannot report
# different sizes for one file.
BYTES_PER_MB = 1e6


@dataclass(frozen=True, slots=True)
class BackupRow:
    """One row, already formatted. Templates do no arithmetic and no timezone work."""

    name: str
    taken_at: str
    size: str


@router.get("/backups")
def backups_page(request: Request, _user=viewer) -> Response:
    """Read the folder and report. **A GET must never write a backup.**

    The legacy restarted the AI workers from `POST /page-activate/{page}` every time a tab
    was opened, so what the system did depended on which page somebody was looking at
    (audit H-18). Here the same mistake would be worse: a browser prefetch, an uptime
    probe or a held-down F5 would each write a full copy of the database, and the disk
    that fills is the one the recordings are written to.
    """
    rows, folder, listing_error = _listing()
    state = request.app.state.backups.state()

    return render(
        request,
        "backups.html",
        backups=rows,
        folder=folder,
        listing_error=listing_error,
        running=state.running,
        started_at=_moment(state.started_at),
        last=state.last,
    )


@router.post("/backups")
def take_a_backup(request: Request, _user=maker) -> Response:
    """Ask for a backup and answer immediately; the copy runs on its own thread.

    The redirect is not decoration: without it the browser's reload button re-posts, and
    every reload would be another full copy of the database on the same disk.
    """
    user = current_user(request)
    started = request.app.state.backups.request(by=user.username)

    logger.info(
        "backup requested from the dashboard",
        extra={"by": user.username, "started": started},
    )
    return RedirectResponse("/backups", status_code=status.HTTP_303_SEE_OTHER)


def _listing() -> tuple[list[BackupRow], str, str]:
    """The rows, the folder they are in, and why there are none if we cannot tell.

    A `BackupError` here means DATABASE_URL is PostgreSQL or in-memory: `VACUUM INTO`
    cannot copy either. That is a sentence on the page, not a 500 -- an operator seeing a
    server error concludes the backups are broken, when what is true is that this
    installation backs up by other means.
    """
    try:
        folder, files = backup_directory(), existing_backups()
    except BackupError as exc:
        return [], "", str(exc)
    return [_row(file) for file in files], str(folder), ""


def _row(file: BackupFile) -> BackupRow:
    """Converted for a human here, once, and nowhere else.

    `modified_at` is aware UTC (see `BackupFile`). The school's zone -- not the server's,
    and not UTC -- because `default_destination` already writes the file NAME in the
    school's zone, and a page whose "when" column disagreed with the name printed beside
    it would be this project's signature fault in its purest form.
    """
    when = file.modified_at.astimezone(get_settings().tz)
    return BackupRow(
        name=file.path.name,
        taken_at=when.strftime("%Y-%m-%d %H:%M"),
        size=f"{file.size_bytes / BYTES_PER_MB:.1f} МБ",
    )


def _moment(when: datetime | None) -> str:
    """When the running backup started, in the school's zone -- the SAME conversion `_row`
    does, because two times on one page in two different zones is worse than one.

    Shown at all because "копирование выполняется" with no time cannot distinguish a copy
    that started ten seconds ago from one that has been stuck since Tuesday, and the
    second is the state somebody needs to act on.
    """
    return when.astimezone(get_settings().tz).strftime("%H:%M") if when else ""

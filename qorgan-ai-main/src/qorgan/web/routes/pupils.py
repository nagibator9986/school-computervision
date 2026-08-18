"""The pupil register, and one child's meal history.

Two rules, each paid for by the legacy:

  * **Pagination is not optional.** The legacy read the entire persons table on every
    render, every 2.5 seconds, per client (audit M-19). The read model in
    `identity.registry` cannot return more than a page, so this route cannot ask it to.

  * **A GET renders and does nothing else.** Opening a tab in the legacy POSTed
    `/page-activate/{page}`, which restarted the AI workers with a five-second
    `thread.join()` inside the HTTP handler. Nothing here writes.

The register carries no photograph, which is why it asks only for `VIEW_PUPILS`. The
photographs are on the duplicate page, and that page asks for both (see `duplicates.py`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import Response

from qorgan.identity.registry import person_history, pupil_page
from qorgan.roles import Capability
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

router = APIRouter()

# The register, and only the register. §14: a canteen worker holds the canteen journal and
# does not hold this -- the roll of every child in the school is not canteen work.
viewer = Depends(require_capability(Capability.VIEW_PUPILS))

# One child's meal record is BOTH the register and the canteen, so it names both. Neither
# grant on its own opens it: `require_capability` means all of them, never any of them.
history_viewer = Depends(require_capability(Capability.VIEW_PUPILS, Capability.VIEW_CANTEEN))


@router.get("/pupils")
def pupils_page(request: Request, page: int = 1, user=viewer) -> Response:
    return render(request, "pupils.html", registry=pupil_page(school_of(user), page))


@router.get("/pupils/{person_id}/canteen")
def pupil_canteen_page(
    person_id: int, request: Request, page: int = 1, user=history_viewer
) -> Response:
    """This child's meal sessions. A 404 if nobody holds that id -- never an empty page,
    which would read as "this child never ate" for an id that does not exist.

    The same 404 covers a child who belongs to another school, deliberately: an id that
    resolved differently depending on whether it existed elsewhere would let anyone here
    enumerate the roll of every other school on the installation, one number at a time.
    """
    history = person_history(person_id, school_of(user), page)
    if history is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such person")

    return render(request, "pupil_canteen.html", history=history)

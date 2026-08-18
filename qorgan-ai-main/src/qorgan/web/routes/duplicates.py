"""Two ids, one human: the page where the school decides, and can change its mind.

**Merging is gated on `MERGE_PERSONS`, which is not a view capability.** Everything else
in this application asks "may this person LOOK at X". This asks "may this person DECIDE
that two school ids are one child" -- a mutation that re-points meal sessions, retires an
id, and on `student_470 / staff_334` decides whether a child is FED. Reading the pairs is
how you prepare that decision, and whoever prepares it is not always entitled to make it.
See `qorgan.roles.Capability.MERGE_PERSONS`.

**Every refusal is rendered, never raised.** `merge_persons` refuses in whole sentences --
"it is inactive, so everything moved onto it would leave the gallery entirely",
"was merged into X, not into Y" -- and those sentences are the entire safety mechanism for
an operator who has the direction wrong. A 500 throws them away, and an operator who
cannot read why they were refused will try again with different ids until something works.

**No side effects on GET.** The legacy restarted the AI workers when somebody opened a
tab. Nothing is merged, reactivated or written by a GET here; the two mutations are POSTs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.identity.duplicates import duplicates_view
from qorgan.identity.merge import merge_persons
from qorgan.logging_setup import get_logger
from qorgan.roles import Capability
from qorgan.web.security import current_user, require_capability, school_of
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()

PAGE = "/pupils/duplicates"

# Reading the six pairs. The photographs on this page are NOT covered by it: they are
# served by `/media`, which asks for VIEW_PUPIL_PHOTOS on the resolved path, and the
# template draws them only for a grant that holds it.
viewer = Depends(require_capability(Capability.VIEW_PUPILS))

# Deciding. Its own capability, deliberately not held by an operator or a developer.
merger = Depends(require_capability(Capability.VIEW_PUPILS, Capability.MERGE_PERSONS))


@router.get(PAGE)
def duplicates_page(request: Request, user=viewer) -> Response:
    return render(
        request, "duplicates.html", view=duplicates_view(school_id=school_of(user)), error=None
    )


@router.post(f"{PAGE}/merge")
def merge_pair(
    request: Request, keep_id: int = Form(...), drop_id: int = Form(...), user=merger
) -> Response:
    """Execute a decision a human just made, naming both ids. Nothing picks a winner."""
    return _apply(request, keep_id, drop_id, reactivate=False)


@router.post(f"{PAGE}/undo")
def undo_merge(
    request: Request, keep_id: int = Form(...), drop_id: int = Form(...), user=merger
) -> Response:
    """Reverse a merge by merging back the other way, reviving the id it retired.

    `reactivate=True` is passed HERE, in the handler, and it is the whole of the undo: the
    domain refuses to revive an inactive keeper without it, because `is_active=False` also
    means "left the school". The refusals it can raise -- not retired by a merge, retired
    by a merge into somebody else -- are surfaced by `_apply` as readable sentences.
    """
    return _apply(request, keep_id, drop_id, reactivate=True)


def _apply(request: Request, keep_id: int, drop_id: int, *, reactivate: bool) -> Response:
    """Both mutations, because they differ only in that flag and must not differ in how
    they refuse. A refusal renders the page with the reason on it (400); a success
    redirects, so a reload does not re-post -- and the consequence of the merge stays
    visible on the page afterwards rather than flashing past once."""
    user = current_user(request)
    try:
        # `school_id=` is not optional here, and omitting it did not leak -- it made the
        # page DEAD. `merge_persons` falls back to "the only school there is", which
        # RAISES once there are two; that is a `RuntimeError`, the `except` below catches
        # only `(LookupError, ValueError)`, and the result was a 500 in place of the
        # readable refusal this entire module exists to produce -- on a school's OWN
        # legitimate merge, not merely on a cross-school one.
        #
        # Both sibling calls in this file already passed the school. This one did not, and
        # nothing noticed, because every test in the suite runs on ONE school, where the
        # fallback happens to be right.
        result = merge_persons(
            keep_id, drop_id, reactivate=reactivate, school_id=school_of(user)
        )
    except (LookupError, ValueError) as refused:
        logger.warning(
            "merge refused",
            extra={
                "keep": keep_id,
                "drop": drop_id,
                "reactivate": reactivate,
                "by": user.username,
                "reason": str(refused),
            },
        )
        return render(
            request,
            "duplicates.html",
            view=duplicates_view(school_id=school_of(user)),
            error=str(refused),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    logger.warning(
        "persons merged from the web — a human decided these two ids are one person",
        extra={
            "keep": result.keep_external,
            "drop": result.drop_external,
            "reactivate": reactivate,
            "crosses_person_type": result.crosses_person_type,
            "by": user.username,
        },
    )
    return RedirectResponse(PAGE, status_code=status.HTTP_303_SEE_OTHER)

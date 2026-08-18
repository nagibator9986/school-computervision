"""The schools register in the browser. §14's "управление школами".

Gated on `Capability.MANAGE_SCHOOLS`, held by `UserRole.SUPERADMIN` alone. The capability
and this page arrived in the same change, which is the rule `qorgan.roles` closes on: a
permission guarding nothing is a guess.

**What is NOT on this page is the design.** It shows counts -- pupils, cameras, accounts
-- and never a child's name, photograph, meal or incident. The superadmin holds no
child-facing capability at all, so there is no route here to widen later by accident; the
boundary is in `ROLE_CAPABILITIES` and this file could not cross it if it tried.

HTTP only, as in `routes/users.py`: the rules are in `qorgan.schools`, because a CLI is a
second front door onto the same table. Plain forms, no JavaScript, CSRF from `render()`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.roles import Capability
from qorgan.schools import SchoolError, create_school, list_schools, rename_school
from qorgan.web.security import require_capability
from qorgan.web.templating import render

router = APIRouter()

superadmin = Depends(require_capability(Capability.MANAGE_SCHOOLS))

REFUSED = status.HTTP_400_BAD_REQUEST


@router.get("/schools")
def schools_page(request: Request, _user=superadmin) -> Response:
    """The register. Reads, and does nothing else."""
    return _page(request)


@router.post("/schools")
def create(
    request: Request,
    slug: str = Form(...),
    name: str = Form(...),
    _user=superadmin,
) -> Response:
    """Add a school. The act that makes this installation multi-tenant in earnest."""
    try:
        create_school(slug, name)
    except SchoolError as exc:
        return _page(request, error=str(exc), form_slug=slug, form_name=name)
    return RedirectResponse("/schools", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/schools/{school_id}/name")
def rename(
    request: Request, school_id: int, name: str = Form(...), _user=superadmin
) -> Response:
    """Rename one. The row migration 0009 wrote is named to be corrected here."""
    try:
        rename_school(school_id, name)
    except SchoolError as exc:
        return _page(request, error=str(exc))
    return RedirectResponse("/schools", status_code=status.HTTP_303_SEE_OTHER)


def _page(
    request: Request,
    *,
    error: str | None = None,
    form_slug: str = "",
    form_name: str = "",
) -> Response:
    """One function for the read path and the refusal path, so they cannot drift."""
    return render(
        request,
        "schools.html",
        schools=list_schools(),
        error=error,
        form_slug=form_slug,
        form_name=form_name,
        status_code=REFUSED if error else status.HTTP_200_OK,
    )

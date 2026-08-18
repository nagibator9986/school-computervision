"""Account management in the browser.

Until this page existed the only way to give somebody a login was `qorgan user add` at a
shell on the server, which means the school cannot take on a canteen worker without the
vendor. That dependency is the shape of the legacy relationship this rewrite exists to end.

**This is the one page where "who can press it" matters most, and none of the rules are
in this file.** They are in `qorgan.accounts`, because `qorgan user add` is a second front
door onto the same table and a rule written in a route is a rule the CLI does not have.
What this file does is HTTP: gate on the capability, read the form, turn a refusal into a
page the school can read.

The gate is `Capability.MANAGE_USERS`, held by `UserRole.ADMIN` alone -- see `qorgan.roles`
for why not DEVELOPER. Creating an account is a privilege-granting act: whoever can create
one with `role=admin` can create it for themselves, and an admin reaches the live corridor
cameras, the bullying log, and the photograph of every pupil in the school.

**CSRF matters here more than anywhere.** `SameSite=lax` still sends the session cookie on
a top-level navigation, and an auto-submitting form IS one -- so without the token an admin
who clicks a link in an email would silently mint the attacker an admin account. The check
is deny-by-default in middleware (`qorgan.web.csrf`); the forms carry `csrf_token`, which
`render()` puts on every page.

**No JavaScript.** Plain forms, server-rendered. The legacy built this kind of table with
`innerHTML` from server JSON, so a pupil named `<img src=x onerror=...>` was stored XSS in
the operator's browser (audit H-05) -- and a username is user-supplied text of exactly that
shape. Jinja's autoescape covers the server-rendered case and covers nothing that JS builds
from strings, so the page builds nothing from strings.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.accounts import (
    ASSIGNABLE_ROLES,
    PAGE_SIZE,
    AccountError,
    UnknownAccount,
    create_account,
    list_accounts,
    parse_role,
    set_active,
    set_role,
)
from qorgan.passwords import PasswordRejected
from qorgan.roles import Capability
from qorgan.web.security import current_user, require_capability
from qorgan.web.templating import render

router = APIRouter()

# The whole gate, in one line, and the shortest grant list in the system.
manager = Depends(require_capability(Capability.MANAGE_USERS))

# A refusal the school caused (a taken username, a short password) is not a server fault
# and must not be reported as one. 400 with the form still on screen is what lets somebody
# correct it; a 500 is what makes them phone the vendor.
REFUSED = status.HTTP_400_BAD_REQUEST


@router.get("/users")
def users_page(request: Request, page: int = 1, _user=manager) -> Response:
    """The list. **Reads, and does nothing else.**

    The legacy's `POST /page-activate/{page}` restarted the AI workers -- with a
    five-second `thread.join()` inside the HTTP handler -- every time somebody opened a
    tab, so which cameras were being analysed depended on which tab was open.
    """
    return _page(request, page=page)


@router.post("/users")
def create(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    _user=manager,
) -> Response:
    """Create a login. The privilege-granting act this whole page is gated for."""
    try:
        create_account(
            username, password, parse_role(role), school_id=_school_of(request)
        )
    except (AccountError, PasswordRejected) as exc:
        # `username` goes back into the form so the admin can fix a typo; `password` does
        # NOT. Re-rendering a form usually means putting the values back, and putting THIS
        # value back writes it into the page source, into the browser's back-forward cache,
        # and into anything that saves the page.
        return _page(request, page=1, error=str(exc), form_username=username)

    # 303 to a GET, so a refresh does not re-post. The password was in the body of a POST
    # and stays there: never a query string, which is written to browser history, to the
    # referrer of every asset the next page loads, and to every proxy log in between.
    return RedirectResponse("/users", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/users/{user_id}/role")
def change_role(
    request: Request, user_id: int, role: str = Form(...), _user=manager
) -> Response:
    return _apply(
        request,
        lambda actor: set_role(
            user_id,
            parse_role(role),
            acting_user_id=actor,
            school_id=_school_of(request),
        ),
    )


@router.post("/users/{user_id}/active")
def change_active(
    request: Request, user_id: int, active: bool = Form(...), _user=manager
) -> Response:
    """Retire an account, or bring one back. There is no delete route, and that is a
    decision rather than an omission -- see `qorgan.accounts`: `Event.reviewed_by_id` is
    ON DELETE SET NULL, so deleting a row would blank the reviewer on every bullying event
    that person ruled on."""
    return _apply(
        request,
        lambda actor: set_active(
            user_id, active, acting_user_id=actor, school_id=_school_of(request)
        ),
    )


def _school_of(request: Request) -> int | None:
    """**The school comes from the session, exactly like the acting user does.**

    Every account rule below -- who is listed, who may be demoted, who counts as the last
    remaining administrator -- is scoped by this, so if it came off the form or the URL the
    caller would choose which school's staff they are administering. `None` is not a hole:
    it means the account belongs to no school (the суперадминистратор), and `qorgan.
    accounts` then falls back to the only school there is and RAISES if there are several,
    rather than picking one. A superadmin has no business on this page in any case -- they
    do not hold MANAGE_USERS -- so the fallback is unreachable rather than lenient.
    """
    return current_user(request).school_id


def _apply(request: Request, change) -> Response:
    """Run a change as the logged-in user, and turn its refusal into a readable page.

    **The acting user comes from the SESSION, never from the form.** It decides the "not
    your own account" rule, so a form field would let the caller choose which rule applies
    to them -- and a check the attacker parameterises is not a check.
    """
    try:
        change(current_user(request).id)
    except UnknownAccount:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such account") from None
    except (AccountError, PasswordRejected) as exc:
        return _page(request, page=1, error=str(exc))

    return RedirectResponse("/users", status_code=status.HTTP_303_SEE_OTHER)


def _page(
    request: Request,
    *,
    page: int,
    error: str | None = None,
    form_username: str = "",
) -> Response:
    """The list, rendered. One function so the refusal path cannot drift from the read
    path -- an error page missing half its context is how a refusal becomes a dead end."""
    page = max(1, page)
    accounts, total = list_accounts(page, school_id=_school_of(request))

    return render(
        request,
        "users.html",
        accounts=accounts,
        page=page,
        pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
        total=total,
        # NOT `UserRole`. The superadmin role exists and is not assignable from
        # here: a school's headteacher offered it in this dropdown could mint an
        # account that reaches every school on the installation. `parse_role`
        # refuses it as well, because a dropdown is not a check.
        roles=[role.value for role in ASSIGNABLE_ROLES],
        error=error,
        form_username=form_username,
        status_code=REFUSED if error else status.HTTP_200_OK,
    )

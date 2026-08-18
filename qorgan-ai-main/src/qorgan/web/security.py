"""Authentication and roles.

The legacy dashboard had ~50 endpoints, zero authentication, and bound to 0.0.0.0.
Anyone on the school network could open the pupil registry and look at photographs of
children, watch live video of them, read the recognition log, and delete pupils. That
is the most serious finding in the audit (C-01), and it is the reason this module
exists before any page does.

Deny by default: the middleware rejects every request that is not explicitly public.
Adding a route cannot accidentally expose it -- you have to opt out, in writing.

Authorisation is by capability, never by rank -- see `qorgan.roles` for why a rank was
the wrong shape and what it cost the school's §14.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, Response

from qorgan.db.engine import session_scope
from qorgan.db.models import User
from qorgan.db.types import utcnow
from qorgan.passwords import verify_password
from qorgan.roles import Capability, capabilities_for

SESSION_USER_KEY = "user_id"

# Routes reachable without a session. Everything else requires one.
PUBLIC_PATHS: frozenset[str] = frozenset({"/login", "/healthz"})
PUBLIC_PREFIXES: tuple[str, ...] = ("/static/",)

# The bcrypt adapter and the password policy used to live here. They are now in
# `qorgan.passwords`: the CLI and `qorgan.accounts` need them too, and a module outside the
# web package importing this one for `hash_password` meant importing `qorgan.web`, which
# builds the whole FastAPI app -- a real import cycle, not merely untidy layering.


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def authenticate(username: str, password: str) -> User | None:
    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        # Recorded at the moment the password checked out, because the accounts page shows
        # it and an admin decides who to retire from it. The column has existed since the
        # first migration and nothing ever wrote it, so every row read "never logged in" --
        # a claim, displayed as a fact, that was only ever an unwritten column.
        #
        # Flushed BEFORE the expunge: the expunge only detaches the instance so the caller
        # gets a usable object outside the session, and the UPDATE has to be in the
        # transaction `session_scope` is about to commit.
        user.last_login_at = utcnow()
        session.flush()
        session.expunge(user)
        return user


def load_user(user_id: int) -> User | None:
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None or not user.is_active:
            return None
        session.expunge(user)
        return user


class AuthMiddleware(BaseHTTPMiddleware):
    """Deny by default. A new route is protected unless it is on the public list."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if is_public(request.url.path):
            return await call_next(request)

        user_id = request.session.get(SESSION_USER_KEY)
        user = load_user(user_id) if user_id else None

        if user is None:
            # The USER key, not the whole session. The session carries two unrelated
            # things -- who is logged in, and this session's CSRF token -- and clearing
            # both here made logging in from a browser IMPOSSIBLE:
            #
            #   1. GET /login mints the token and renders the form carrying it;
            #   2. the browser then fetches /favicon.ico, which is not public, so this
            #      branch runs and wipes the token it never had any business touching;
            #   3. POST /login presents a token the session no longer knows -> 403.
            #
            # Every browser requests a favicon, so this was not an edge case: it was
            # every login, every time. It survived because `TestClient` fetches exactly
            # the paths a test names, and the defect lives in the GAP between the GET and
            # the POST -- which is why `test_a_browser_can_log_in` below walks that gap.
            #
            # Session fixation is still defended, in the place that owns it: `login()`
            # clears the whole session on SUCCESSFUL authentication, so a token planted
            # before a victim logs in cannot survive into their authenticated session. A
            # CSRF token is a per-session nonce, not a credential; dropping an anonymous
            # visitor's is protection against nothing.
            request.session.pop(SESSION_USER_KEY, None)
            if request.url.path.startswith("/api/"):
                return Response(status_code=status.HTTP_401_UNAUTHORIZED)
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

        request.state.user = user
        return await call_next(request)


def current_user(request: Request) -> User:
    user = getattr(request.state, "user", None)
    if user is None:  # pragma: no cover - the middleware guarantees this
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user


def school_of(user: User) -> int:
    """The one school this request may read, taken from the account making it.

    Never from the URL, a form field or a query string. A `?school_id=` that the server
    honoured would make every page on the installation readable by anybody who could
    guess a number, which is the whole leak this module's tenancy work exists to prevent
    -- and it would look exactly like a working feature.

    `school_id IS NULL` means the суперадминистратор, who belongs to no school. They hold
    `MANAGE_SCHOOLS` and `VIEW_DIAGNOSTICS` and no child-facing capability at all, so no
    route that calls this should ever be reachable by them -- and if one day one is, this
    refuses rather than picking a school for them. Choosing would mean showing one
    school's children to the one account on the installation that reaches all of them.
    """
    if user.school_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this account belongs to no school, so it has no school's data to read",
        )
    return user.school_id


def require_capability(*needed: Capability) -> Callable[[Request], User]:
    """Route dependency: `user = Depends(require_capability(Capability.VIEW_CANTEEN))`.

    Gate on what the route IS, never on how senior the caller is. This replaced a rank
    (`require_role`) under which every gated route asked for OPERATOR, so granting a
    canteen worker the canteen also granted them the bullying log -- the school's §14
    says БЕЗ доступа к буллингу, and an ordering cannot express it.

    All of `needed`, not any: a route that names two capabilities means both.
    """
    required = frozenset(needed)

    def dependency(request: Request) -> User:
        user = current_user(request)
        if not required <= capabilities_for(user.role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires: {', '.join(sorted(c.value for c in required))}",
            )
        return user

    return dependency


def landing_for(user: User) -> str:
    """Where a login lands. `/` is the camera wall, and it is no longer everyone's page.

    A canteen worker has no camera capability, so sending them to `/` after a correct
    password would 403 -- a login that dead-ends is a login the school reports as broken.

    **The superadmin arm was missing, and that dead end was real.** That role holds neither
    VIEW_CAMERAS nor VIEW_CANTEEN, so it fell through to `/login` -- and `/login` is a
    PUBLIC path, so `AuthMiddleware` never sets `request.state.user`, `render` is handed
    `user=None`, and `base.html` draws no nav. A correct password therefore returned the
    installation's own account to a blank login form with no error, no username and no link
    to anything: **indistinguishable from a wrong password.**

    **What hid it was a status-only assertion, not a missing account.** This paragraph used
    to say the latter, and that was measurably false: `tests/test_schools_page.py::_login`
    has signed a superadmin in on every suite run since that file was written -- long before
    any path could create one, because the fixture minted the row itself -- and it asserted
    `status_code == 303`. **303 is exactly what the dead end returns.** The suite walked
    into the defect and straight out the other side. A comment naming the wrong cause is
    worse than no comment, because the next person reasons from it and concludes this was
    unreachable.

    **MANAGE_SCHOOLS is deliberately LAST.** Only SUPERADMIN holds it, and that role holds
    neither capability above, so appending the arm cannot move any other role's landing.
    Putting it first could, silently -- every role's landing is one line, and nobody logs
    in as four roles to notice. The order that actually bites is VIEW_CANTEEN before
    VIEW_CAMERAS: OPERATOR, ADMIN and DEVELOPER hold both, so that swap moves three roles
    off the camera wall at once. `test_each_role_lands_on_a_page_it_can_actually_open`
    pins all five landings and opens each one, which is what makes such a swap red.

    `/login` stays as the fallback and is unreachable for all five roles. It is kept
    because a role granted nothing at all must not bounce `/login` -> `/login`;
    `routes/login.py::login_form` refuses that loop for the same reason.
    """
    capabilities = capabilities_for(user.role)
    if Capability.VIEW_CAMERAS in capabilities:
        return "/"
    # BEFORE the canteen, and the order is the point. A psychologist holds VIEW_CANTEEN
    # (§13's «посещаемость» is the canteen record -- see roles.py), so without this line a
    # correct password would land them on the school's lunch journal rather than on their
    # own cabinet. Landing pages follow the job, not the alphabet of the capability set.
    if Capability.VIEW_PSYCHOLOGIST_CABINET in capabilities:
        return "/psychologist"
    if Capability.VIEW_CANTEEN in capabilities:
        return "/canteen"
    if Capability.MANAGE_SCHOOLS in capabilities:
        return "/schools"
    return "/login"

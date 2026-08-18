"""Proof that a state-changing request came from our own page.

Authentication answers *who is asking*. It does not answer *did they mean to ask*, and
CSRF is the attack that lives in that gap: the victim is genuinely logged in, and the
browser attaches their cookie to a request another site composed.

**`SameSite=lax` does not close it.** `lax` withholds the cookie from background
sub-requests but still sends it on a **top-level navigation** — and a form the attacker's
page auto-submits *is* a top-level navigation. So an operator clicking a link in an email
can silently mark a bullying event "false positive", or (once the pupils section lands)
merge two children's identities. `lax` is a useful second layer and it is not this one.

Two design choices, both deliberate:

**Deny by default, in middleware, keyed on the METHOD.** Every unsafe method is refused
until it proves itself; there is no list of protected paths to keep up to date. An
allow-list would have to be edited by whoever adds the next route, and the author who
forgets is precisely the author who opens the hole. This is the same shape as
`AuthMiddleware`, which refuses everything not explicitly public, and for the same reason.

**The token lives in the session, not in a second cookie.** It is issued once per session
and dies with it — `login()` clears the session against fixation, so the token cannot be
planted before a victim logs in and reused after.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

SESSION_CSRF_KEY = "csrf_token"
FORM_FIELD = "csrf_token"
HEADER_NAME = "X-CSRF-Token"

# GET/HEAD/OPTIONS/TRACE change nothing, so requiring a token on them would only break
# reading. Everything else is unsafe -- naming the SAFE ones is what makes this
# deny-by-default: a method nobody thought of is guarded, not exempt.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

TOKEN_BYTES = 32


def issue_token(request: Request) -> str:
    """This session's token, minted on first sight.

    Read by `web.templating.render`, so every page the server renders carries it and no
    route has to remember. A page that cannot get the token easily is a page whose author
    will find a way around the check.
    """
    token = request.session.get(SESSION_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        request.session[SESSION_CSRF_KEY] = token
    return token


class CsrfMiddleware(BaseHTTPMiddleware):
    """Refuse any unsafe request that cannot show this session's token."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method in SAFE_METHODS:
            return await call_next(request)

        expected = request.session.get(SESSION_CSRF_KEY)
        presented = await _presented_token(request)

        # `compare_digest` on two strs, and only after both are known non-empty: a missing
        # token and a wrong one are the same refusal, but `compare_digest(None, ...)` is a
        # TypeError, and an exception here would fail OPEN through the error handler.
        if not expected or not presented or not secrets.compare_digest(expected, presented):
            return _refused(request)

        return await call_next(request)


async def _presented_token(request: Request) -> str:
    """The token this request offers: header first, then form field.

    The header is what HTMX and `fetch` can set; the field is what a plain `<form>` can
    carry. One mechanism, two spellings -- because a page with no safe way to send it
    sends it unsafely, or its author disables the check.

    **Reading the body here would otherwise eat it.** `BaseHTTPMiddleware` hands the route
    a *different* `Request` object built over the same ASGI `receive` channel, so the
    parsed-form cache does not travel with it and the stream arrives already drained: the
    route then sees no fields at all and FastAPI answers 422. Caught by
    `test_our_own_page_can_still_review_an_event`, which is why that test asserts the
    product still works rather than only that the attack is refused -- a CSRF layer that
    breaks every form is a CSRF layer somebody deletes on Monday.

    So the body is read once and then replayed for whoever comes next.
    """
    header = request.headers.get(HEADER_NAME)
    if header:
        return header

    content_type = request.headers.get("content-type", "")
    if not content_type.startswith(("application/x-www-form-urlencoded", "multipart/form-data")):
        return ""

    body = await request.body()
    _replay(request, body)

    form = await request.form()
    value = form.get(FORM_FIELD)
    return value if isinstance(value, str) else ""


def _replay(request: Request, body: bytes) -> None:
    """Put the bytes back, so the route reads the same request we just inspected."""

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive


def _refused(request: Request) -> Response:
    """403, and it says what to do. A bare 'Forbidden' on a form the user filled in
    honestly reads as "the system is broken", which is how a real protection gets turned
    off by whoever is asked to fix it."""
    detail = (
        "CSRF token missing or invalid. Reload the page and try again — this request "
        "could not be shown to have come from this dashboard."
    )
    if request.url.path.startswith("/api/") or "application/json" in request.headers.get(
        "accept", ""
    ):
        return JSONResponse({"detail": detail}, status_code=status.HTTP_403_FORBIDDEN)
    return PlainTextResponse(detail, status_code=status.HTTP_403_FORBIDDEN)

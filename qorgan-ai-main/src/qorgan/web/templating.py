"""Jinja setup. Autoescaping is on, and it stays on.

The legacy escaped nothing that mattered: its templates built DOM with `innerHTML`
from server JSON, so a pupil named `<img src=x onerror=...>` gave stored XSS in the
operator's browser (audit H-05). Jinja's autoescape does not help you there, because
the injection happens in JavaScript after the page renders.

So the rule in v2 is: **the server renders the DOM; JavaScript does not build it from
strings.** Where JS must inject text, it uses textContent, never innerHTML.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

from qorgan.roles import capabilities_for
from qorgan.web.csrf import issue_token
from qorgan.web.security import landing_for

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.autoescape = True


def render(request: Request, name: str, *, status_code: int = 200, **context: object):
    user = getattr(request.state, "user", None)
    # The nav is drawn from the same table the routes are gated on, so a link a role
    # cannot follow is never drawn. Two sources of truth would mean a canteen worker
    # clicking "События" into a 403 and reporting the system as broken.
    capabilities = capabilities_for(user.role) if user is not None else frozenset()
    # The brand link in `base.html` used to compute its own destination -- `/` if the role
    # held view_cameras, else `/canteen` -- which is a SECOND source of truth about where a
    # role belongs, and it was wrong for the superadmin: that role holds neither, so the
    # masthead of the register they had just landed on linked to a 403. One answer now, the
    # same one `login()` redirects to, so the link and the landing cannot disagree.
    landing = landing_for(user) if user is not None else "/login"
    return templates.TemplateResponse(
        request,
        name,
        # Issued here so EVERY page has it without any route remembering. A page that
        # cannot reach the token easily is a page whose author routes around the check.
        {
            "user": user,
            "capabilities": capabilities,
            "landing": landing,
            "csrf_token": issue_token(request),
            **context,
        },
        status_code=status_code,
    )

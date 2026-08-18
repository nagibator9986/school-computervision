"""Logging in the way a BROWSER does, not the way a test client does.

The difference is the whole point of this file. `TestClient` fetches exactly the paths a
test names; a browser fetches a favicon, follows a redirect, and may have another tab
open. Every login test in this suite went

    GET /login  ->  POST /login

with nothing in between, and passed -- while logging in from Chrome was IMPOSSIBLE, every
time, because `AuthMiddleware` cleared the WHOLE session (this session's CSRF token
included) on any request from somebody not logged in, and `/favicon.ico` is such a request.

`docs/next-session-handoff.md` listed "the web panel is used by anyone" as **never**, with
"every page is proven by TestClient, not by a browser a human drove". This is what that
was worth: the first human to drive it could not get past the login form.

So these tests assert the GAP, not the endpoints. Anything that walks GET -> POST directly
cannot see this class of defect at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import csrf_token

PASSWORD = "correct-horse-battery"
USERNAME = "operator1"


@pytest.fixture(name="app")
def _app(settings: Settings, session: Session):
    del settings  # applied via the fixture
    return create_app()


@pytest.fixture(name="account")
def _account(session: Session) -> User:
    user = User(
        username=USERNAME, password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR
    )
    session.add(user)
    session.commit()
    return user


@pytest.fixture(name="browser")
def _browser(app) -> Iterator[TestClient]:
    """A client that arrives the way a browser does: no session, nothing logged in."""
    with TestClient(app, follow_redirects=False) as client:
        yield client


def _log_in(browser: TestClient, token: str):
    return browser.post(
        "/login",
        data={"username": USERNAME, "password": PASSWORD, "csrf_token": token},
        follow_redirects=False,
    )


@pytest.mark.parametrize(
    "interruption",
    [
        "/favicon.ico",  # every browser, unprompted, on every page load
        "/",  # the tab that was open before, refreshing itself
        "/events",  # a bookmark that redirected them to the login form
    ],
)
def test_a_browser_can_log_in(account, browser: TestClient, interruption: str) -> None:
    """A page fetched between rendering the form and submitting it must not void it.

    Parametrised over three ordinary browser behaviours rather than the favicon alone:
    the defect was never about favicons, it was about ANY anonymous request touching a
    protected path, and a test naming one of them would let the next one back in.
    """
    token = csrf_token(browser)

    browser.get(interruption)

    response = _log_in(browser, token)
    assert response.status_code != 403, (
        f"a browser that fetched {interruption} between opening the login form and "
        f"submitting it was refused: {response.text[:160]}"
    )
    assert response.status_code == 303, response.status_code


def test_an_anonymous_visit_leaves_the_token_alone(account, browser: TestClient) -> None:
    """The token the browser holds is still the one the server expects afterwards.

    Asserted on the VALUE, not merely on the login succeeding, so a future change that
    rotates the token per request fails here saying so -- rather than downstream, as a
    mysterious 403 on a form the user filled in honestly.
    """
    before = csrf_token(browser)

    browser.get("/favicon.ico")
    browser.get("/events")

    assert csrf_token(browser) == before, (
        "an anonymous request replaced this session's CSRF token"
    )


def test_logging_out_still_ends_the_session(account, browser: TestClient) -> None:
    """The fix must not let a session survive a logout.

    Three places clear session state and only one of them was wrong. `logout()` clears
    everything, `login()` clears everything on SUCCESS (session fixation), and the
    middleware's anonymous branch must clear only the user. This pins the difference so
    that a later simplification cannot quietly collapse the three into one.
    """
    token = csrf_token(browser)
    assert _log_in(browser, token).status_code == 303

    assert browser.get("/events").status_code == 200

    assert browser.post(
        "/logout", data={"csrf_token": csrf_token(browser)}
    ).status_code == 303
    assert browser.get("/events").status_code == 303, "the session outlived a logout"

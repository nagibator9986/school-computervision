"""Every state-changing request must prove it came from our own page.

**`SameSite=lax` is not this protection, and believing it is was the gap.** `lax` still
sends the cookie on a TOP-LEVEL NAVIGATION, and a form POST that a page auto-submits is a
top-level navigation. So an operator who clicks a link in an email lands on a page that
posts to `/events/{id}/review` — or, once the pupils section lands, to a merge — with their
session cookie attached, and the browser does it willingly. The operator sees a blank tab;
the school gets a bullying event marked "false positive" by nobody, or two children's
identities fused.

This is the class of hole the whole rewrite exists to close: the legacy dashboard had ~50
endpoints, no authentication at all, bound to `0.0.0.0` (audit C-01). Authentication was
added; *authenticity of the request* was not, and an authenticated victim is exactly what
CSRF uses.

**Deny by default, like `AuthMiddleware` above it.** The check is in middleware and covers
every unsafe method, so a route added next month is protected without its author knowing
this file exists. `test_a_brand_new_mutating_route_is_protected_without_opting_in` is the
one that matters: an allow-list would have to be edited by whoever adds a route, and the
person who forgets is the person who introduces the hole.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, Severity, UserRole
from qorgan.events.store import record_event
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app

PASSWORD = "correct-horse-battery"
TOKEN_FIELD = "csrf_token"
TOKEN_HEADER = "X-CSRF-Token"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()
            client = stack.enter_context(TestClient(app, follow_redirects=False))
            posted = client.post(
                "/login",
                data={"username": username, "password": PASSWORD, TOKEN_FIELD: _token(client)},
            )
            assert posted.status_code == 303, f"login failed: {posted.status_code}"
            return client

        yield make


def _token(client: TestClient) -> str:
    """The token our own login page would have carried."""
    page = client.get("/login")
    found = re.search(rf'name="{TOKEN_FIELD}"\s+value="([^"]+)"', page.text)
    assert found, "the login form carried no CSRF token"
    return found.group(1)


def _session_token(client: TestClient) -> str:
    """The token for an already-authenticated session, read off a rendered page."""
    page = client.get("/events")
    found = re.search(rf'name="{TOKEN_FIELD}"\s+value="([^"]+)"', page.text)
    if found:
        return found.group(1)
    found = re.search(r'content="([^"]+)"\s+name="csrf-token"', page.text) or re.search(
        r'name="csrf-token"\s+content="([^"]+)"', page.text
    )
    assert found, "no CSRF token reached the page; nothing on it could post safely"
    return found.group(1)


@pytest.fixture
def camera(session: Session) -> Camera:
    row = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.commit()
    return row


def _event(camera: Camera) -> int:
    verdict = Verdict(
        confidence=0.9,
        candidate_probability=0.9,
        validation_score=1.0,
        skeleton_confirmed=True,
        capped=False,
    )
    return record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=verdict,
        severity=Severity.CRITICAL,
        summary_text="test",
        track_ids="1,2",
    )


# -- the attack, refused ------------------------------------------------------


def test_a_cross_site_post_without_a_token_is_refused(
    client_for: ClientFor, camera: Camera
) -> None:
    """The email-link attack, in one request. The session cookie IS sent -- `lax` allows
    it on a top-level navigation -- so authentication does not stop this. Only the token
    does."""
    event_id = _event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(f"/events/{event_id}/review", data={"verdict": "false_positive"})

    assert response.status_code == 403, (
        "a bullying event was reviewed by a request that proved nothing about its origin"
    )


def test_a_stolen_looking_token_from_elsewhere_is_refused(
    client_for: ClientFor, camera: Camera
) -> None:
    """A token has to be THIS session's. Accepting any well-formed value would be a
    check that cannot fail, which is worse than no check because it reads as one."""
    event_id = _event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(
        f"/events/{event_id}/review",
        data={"verdict": "false_positive", TOKEN_FIELD: "not-the-one-we-issued"},
    )

    assert response.status_code == 403


def test_our_own_page_can_still_review_an_event(client_for: ClientFor, camera: Camera) -> None:
    """The protection must not break the product. An operator reviewing an event from the
    events page is the entire point of the page."""
    event_id = _event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(
        f"/events/{event_id}/review",
        data={"verdict": "false_positive", TOKEN_FIELD: _session_token(client)},
    )

    assert response.status_code in (200, 303), (
        f"the real page was blocked from doing its job: {response.status_code}"
    )


def test_htmx_may_send_the_token_in_a_header(client_for: ClientFor, camera: Camera) -> None:
    """HTMX posts without a form body. Header or field -- one mechanism, two spellings,
    because a page that cannot send the token safely will send it unsafely."""
    event_id = _event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(
        f"/events/{event_id}/review",
        data={"verdict": "false_positive"},
        headers={TOKEN_HEADER: _session_token(client)},
    )

    assert response.status_code in (200, 303)


# -- deny by default ----------------------------------------------------------


def test_a_brand_new_mutating_route_is_protected_without_opting_in(
    app, client_for: ClientFor
) -> None:
    """**The test that decides whether this holds in a year.**

    An allow-list has to be edited by whoever adds a route, and the author who forgets is
    the author who opens the hole. So the guard is on the METHOD, not on a list of paths:
    anything unsafe is refused until it proves itself, exactly as `AuthMiddleware` refuses
    anything not explicitly public.
    """

    @app.post("/a-route-nobody-told-the-csrf-layer-about")
    def _new_route(request: Request) -> dict[str, bool]:
        return {"mutated": True}

    client = client_for(UserRole.ADMIN)

    assert client.post("/a-route-nobody-told-the-csrf-layer-about").status_code == 403, (
        "a route added without touching the CSRF layer was unprotected -- which is how "
        "every route eventually is"
    )


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_every_unsafe_method_is_covered_not_only_post(
    app, client_for: ClientFor, method: str
) -> None:
    """POST is not the only way to change something, and a guard that only knows POST is
    a guard somebody routes around with `PUT`."""

    @app.api_route("/another-mutating-route", methods=["PUT", "PATCH", "DELETE"])
    def _new_route(request: Request) -> dict[str, bool]:
        return {"mutated": True}

    client = client_for(UserRole.ADMIN)

    assert getattr(client, method)("/another-mutating-route").status_code == 403


def test_reading_is_never_blocked(client_for: ClientFor) -> None:
    """GET/HEAD change nothing, and a CSRF check on a read is a broken dashboard."""
    client = client_for(UserRole.OPERATOR)

    assert client.get("/events").status_code == 200
    assert client.get("/").status_code == 200


# -- login is not an exception ------------------------------------------------


def test_logging_in_requires_a_token_from_our_own_login_page(
    client_for: ClientFor, session: Session
) -> None:
    """Login CSRF is a real attack: an attacker who logs a victim into an account THEY
    control watches everything the victim then does in it. The login form is served by us
    and carries a token like any other."""
    session.add(
        User(username="victim", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR)
    )
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as client:
        response = client.post("/login", data={"username": "victim", "password": PASSWORD})

        assert response.status_code == 403, "a login could be forced from another site"


def test_the_token_changes_when_the_session_does(client_for: ClientFor) -> None:
    """`login()` clears the session against fixation. The token must not survive that:
    a value that outlives the session it authenticates is a value an attacker can plant
    before the victim logs in and reuse afterwards."""
    client = client_for(UserRole.OPERATOR)
    after_login = _session_token(client)

    client.post("/logout", data={TOKEN_FIELD: after_login})
    fresh = _token(client)

    assert fresh != after_login, "the CSRF token survived a session change"

"""Authentication. The fix for the worst finding in the audit.

The legacy dashboard had ~50 endpoints, no authentication of any kind, and bound to
0.0.0.0. Anyone on the school network could browse photographs of children, watch them
live, read the recognition log, and delete pupils (C-01).

The rule here is deny-by-default: a route is protected unless it is explicitly listed
as public. `test_every_route_is_protected` walks the real route table and enforces it,
so a future route cannot be exposed by forgetting something.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette.routing import Mount, Route

from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from qorgan.web.security import PUBLIC_PATHS, PUBLIC_PREFIXES, is_public
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings  # applied via the fixture
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client


@pytest.fixture
def operator(session: Session) -> User:
    return _user(session, "operator1", UserRole.OPERATOR)


def _user(session: Session, username: str, role: UserRole) -> User:
    user = User(username=username, password_hash=hash_password(PASSWORD), role=role)
    session.add(user)
    session.commit()
    return user


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/login",
        data=with_token(client, {"username": username, "password": PASSWORD}),
    )
    assert response.status_code == 303, "login failed"


# -- deny by default -------------------------------------------------------


def test_the_dashboard_is_not_readable_without_logging_in(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_a_camera_preview_is_not_readable_without_logging_in(client: TestClient) -> None:
    """This is live video of children. It was wide open in the legacy."""
    response = client.get("/preview/hall_left.jpg")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_an_api_call_without_a_session_gets_401_not_a_redirect(client: TestClient) -> None:
    assert client.get("/api/cameras").status_code == 401


def test_every_route_is_protected_unless_explicitly_public(app) -> None:
    """The guard rail. Add a route, and it is private -- you must opt out in writing."""
    exposed = []
    for route in app.routes:
        if isinstance(route, Mount) or not isinstance(route, Route):
            continue
        if is_public(route.path):
            exposed.append(route.path)

    assert set(exposed) <= PUBLIC_PATHS, (
        f"these routes are public and should not be: {sorted(set(exposed) - PUBLIC_PATHS)}"
    )
    assert PUBLIC_PREFIXES == ("/static/",), "the public prefix list grew; is that intended?"


def test_healthz_is_public_and_says_nothing_about_the_school(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -- logging in ------------------------------------------------------------


def test_a_valid_login_reaches_the_dashboard(client: TestClient, operator: User) -> None:
    _login(client, operator.username)
    response = client.get("/")
    assert response.status_code == 200
    assert "Камеры" in response.text


def test_a_wrong_password_is_rejected(client: TestClient, operator: User) -> None:
    response = client.post(
        "/login",
        data=with_token(client, {"username": operator.username, "password": "wrong"}),
    )
    assert response.status_code == 401
    assert client.get("/").status_code == 303  # still not logged in


def test_an_unknown_user_and_a_wrong_password_look_identical(
    client: TestClient, operator: User
) -> None:
    """Telling them apart hands out a free list of who works at the school."""
    wrong_password = client.post(
        "/login",
        data=with_token(client, {"username": operator.username, "password": "x"}),
    )
    no_such_user = client.post(
        "/login",
        data=with_token(client, {"username": "ghost", "password": "x"}),
    )

    assert wrong_password.status_code == no_such_user.status_code == 401
    assert wrong_password.text == no_such_user.text


def test_a_deactivated_user_cannot_log_in(client: TestClient, session: Session) -> None:
    user = _user(session, "former_staff", UserRole.ADMIN)
    user.is_active = False
    session.commit()

    response = client.post(
        "/login",
        data=with_token(client, {"username": user.username, "password": PASSWORD}),
    )
    assert response.status_code == 401


def test_deactivating_a_user_ends_their_session_immediately(
    client: TestClient, session: Session, operator: User
) -> None:
    """The session cookie must not outlive the account. Somebody leaves the school on
    Friday; their laptop must not still be watching the corridor on Monday."""
    _login(client, operator.username)
    assert client.get("/").status_code == 200

    operator.is_active = False
    session.commit()

    assert client.get("/").status_code == 303


def test_logging_out_ends_the_session(client: TestClient, operator: User) -> None:
    _login(client, operator.username)
    assert client.post("/logout", data=with_token(client)).status_code == 303
    assert client.get("/").status_code == 303


def test_the_password_is_never_stored_in_the_clear(session: Session, operator: User) -> None:
    stored = session.get(User, operator.id)
    assert stored.password_hash != PASSWORD
    assert PASSWORD not in stored.password_hash
    assert stored.password_hash.startswith("$2")  # bcrypt


# -- roles -----------------------------------------------------------------
#
# Which role may do what now lives in test_web_capability_roles.py, next to the §14
# requirement that shaped it. These two stay here because they are about a session
# reaching a page at all. They no longer nest: roles used to be a strict total order
# (developer implied admin implied operator), and the two below were named for it.
# An admin reaches the dashboard because ROLE_CAPABILITIES grants an admin
# VIEW_CAMERAS, not because an admin outranks anybody.


def test_an_admin_can_do_operator_things(client: TestClient, session: Session) -> None:
    admin = _user(session, "headteacher", UserRole.ADMIN)
    _login(client, admin.username)
    assert client.get("/").status_code == 200


def test_a_developer_can_do_operator_things(client: TestClient, session: Session) -> None:
    developer = _user(session, "dev", UserRole.DEVELOPER)
    _login(client, developer.username)
    assert client.get("/api/cameras").status_code == 200

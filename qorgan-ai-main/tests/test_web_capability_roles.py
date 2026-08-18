"""§14: access is by capability, not by rank.

The school's §14 lists roles that OVERLAP rather than nest: canteen staff see the canteen
"БЕЗ доступа к буллингу" -- no access to bullying data. The model this replaced was a
strict total order (operator implied admin implied developer) and every gated route
required exactly OPERATOR. So the only way to let a canteen worker open /canteen was to
grant OPERATOR, which by the ordering also opened the bullying event log, the review
controls, the live corridor previews, and /media -- snapshots and clips of children.

A rank cannot say "the canteen and nothing else": every role able to see the canteen sat
at or above the role that sees everything. That is why this file exists, and why the
canteen worker below is the test that matters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, Severity, UserRole
from qorgan.events.recorder import write_snapshot
from qorgan.events.store import record_event
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about."""
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(
                User(username=username, password_hash=hash_password(PASSWORD), role=role)
            )
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login",
                data=with_token(client, {"username": username, "password": PASSWORD}),
            )
            assert response.status_code == 303, "login failed"
            return client

        yield make


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


def _snapshot() -> str:
    """A real file under media_root, so a refusal is a refusal and not an accident of
    absence -- a 404 would pass an `is not 200` assertion while proving nothing."""
    frame = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    return write_snapshot(frame, "hall_left", utcnow())


def _event(camera: Camera, snapshot: str | None = None) -> int:
    return record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия",
        track_ids="3,7",
        snapshot_path=snapshot,
    )


# -- §14: the canteen worker -------------------------------------------------


def test_a_canteen_worker_reads_the_journal_and_is_refused_the_bullying_log(
    client_for: ClientFor,
) -> None:
    """§14: столовая — БЕЗ доступа к буллингу.

    The requirement the total order could not express, in one test: the worker does their
    job, and the bullying log is shut.
    """
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get("/canteen").status_code == 200, "a canteen worker cannot do their job"
    assert client.get("/canteen/export.csv").status_code == 200

    assert client.get("/events").status_code == 403, (
        "§14 violated: the canteen worker read the bullying event log"
    )


def test_a_canteen_worker_cannot_review_a_bullying_event(
    client_for: ClientFor, camera: Camera
) -> None:
    """Reviewing is a judgement on a child. It is not canteen work."""
    event_id = _event(camera)
    client = client_for(UserRole.CANTEEN_STAFF)

    response = client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "false_positive"}),
    )
    assert response.status_code == 403, "§14 violated: the canteen worker judged a bullying event"


def test_a_canteen_worker_is_refused_bullying_media(client_for: ClientFor) -> None:
    """/media is snapshots and clips of children. §14 keeps canteen staff out of it."""
    path = _snapshot()
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get(f"/media/{path}").status_code == 403, (
        "§14 violated: the canteen worker fetched a bullying snapshot"
    )


def test_a_canteen_worker_is_refused_the_live_previews(client_for: ClientFor) -> None:
    """Live video of a school corridor is not canteen work either."""
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get("/").status_code == 403
    assert client.get("/api/cameras").status_code == 403
    assert client.get("/preview/hall_left.jpg").status_code == 403


def test_a_canteen_worker_lands_somewhere_they_are_allowed_to_be(
    client_for: ClientFor,
) -> None:
    """A login that dead-ends on 403 is a login the school reports as broken. The landing
    page follows the capabilities, because "/" is no longer everyone's page."""
    client = client_for(UserRole.CANTEEN_STAFF)

    response = client.post(
        "/login",
        data=with_token(client, {"username": "user_canteen_staff", "password": PASSWORD}),
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/canteen"


# -- the mechanism -----------------------------------------------------------


def test_the_canteen_capability_does_not_carry_the_bullying_capability() -> None:
    """The contract, below the HTTP layer: granting the canteen grants the canteen."""
    canteen = capabilities_for(UserRole.CANTEEN_STAFF)

    assert Capability.VIEW_CANTEEN in canteen
    assert Capability.VIEW_BULLYING not in canteen
    assert Capability.REVIEW_BULLYING not in canteen
    # `VIEW_MEDIA` was one grant over a mixed tree and is now two. Both are named here,
    # rather than the check being dropped: a split that quietly stopped asserting one half
    # would be how §14 came back for the gallery while the tests still looked green.
    assert Capability.VIEW_BULLYING_MEDIA not in canteen
    assert Capability.VIEW_PUPIL_PHOTOS not in canteen
    assert Capability.VIEW_CAMERAS not in canteen
    # /logs quotes whatever a module put in a log record -- camera names, RTSP hosts,
    # exception strings. Named here for the same reason as the media split above: a
    # capability added later and never asserted against is how §14 comes back quietly.
    assert Capability.VIEW_DIAGNOSTICS not in canteen


def test_the_operator_does_not_hold_the_diagnostics_capability() -> None:
    """§14 gives the оператор безопасности "просмотр тревог; подтверждение/отклонение
    событий; просмотр клипов". The server is not on that list, and this is the first
    capability the ADMIN and DEVELOPER sets hold that the operator's does not -- the
    difference `roles.py` said would arrive when the debug views landed."""
    assert Capability.VIEW_DIAGNOSTICS not in capabilities_for(UserRole.OPERATOR)
    assert Capability.VIEW_DIAGNOSTICS in capabilities_for(UserRole.ADMIN)
    assert Capability.VIEW_DIAGNOSTICS in capabilities_for(UserRole.DEVELOPER)


def test_every_role_states_its_capabilities_in_writing() -> None:
    """Deny by default, for roles. A new role must say what it may do -- it cannot inherit
    a set by sitting above someone in an ordering, because there is no ordering."""
    assert set(ROLE_CAPABILITIES) == set(UserRole)


def test_an_unmapped_role_is_denied_rather_than_crashing() -> None:
    """If a role ever reaches the check without an entry, it gets nothing. The failure of
    a lookup must not be an open door."""
    assert capabilities_for("no_such_role") == frozenset()  # type: ignore[arg-type]


def test_a_role_with_nowhere_to_land_gets_the_login_page_not_a_redirect_loop(
    app, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role granted nothing has nowhere to land, and /login must not send it to /login.

    No role is empty today, so this guards the next one: the browser answer to a loop is
    ERR_TOO_MANY_REDIRECTS, which the school reports as "the system is down" rather than
    as the permissions mistake it actually is.
    """
    monkeypatch.setitem(ROLE_CAPABILITIES, UserRole.CANTEEN_STAFF, frozenset())
    session.add(
        User(
            username="nowhere",
            password_hash=hash_password(PASSWORD),
            role=UserRole.CANTEEN_STAFF,
        )
    )
    session.commit()

    with TestClient(app, follow_redirects=False) as client:
        client.post(
            "/login",
            data=with_token(client, {"username": "nowhere", "password": PASSWORD}),
        )
        assert client.get("/login").status_code == 200, "login redirected to itself"


# -- what already worked must keep working -----------------------------------


def test_an_operator_still_reads_and_reviews_the_bullying_log(
    client_for: ClientFor, camera: Camera
) -> None:
    event_id = _event(camera)
    client = client_for(UserRole.OPERATOR)

    assert client.get("/events").status_code == 200
    response = client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "false_positive"}),
    )
    assert response.status_code == 303


def test_an_operator_still_reaches_the_canteen_the_cameras_and_the_media(
    client_for: ClientFor,
) -> None:
    path = _snapshot()
    client = client_for(UserRole.OPERATOR)

    assert client.get("/canteen").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/cameras").status_code == 200
    assert client.get(f"/media/{path}").status_code == 200


@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.DEVELOPER])
def test_an_admin_and_a_developer_still_reach_what_an_operator_reaches(
    client_for: ClientFor, role: UserRole
) -> None:
    """The total order let these two through every operator route. Removing the ordering
    must not quietly take that away -- the school's admin still runs the school."""
    client = client_for(role)

    assert client.get("/").status_code == 200
    assert client.get("/events").status_code == 200
    assert client.get("/canteen").status_code == 200

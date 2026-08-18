"""Who may press the buttons on the accounts page.

**This is not merely another view.** Every other page shows the school something. This one
hands out the right to see it: whoever can create an account with `role=admin` can create
one for themselves, and an admin holds every capability in the system -- the live corridor
cameras, the bullying log, the enrolment gallery of every pupil's photograph. A hole here
is not one page leaking; it is the capability model becoming decorative.

**Which capability, and why not a rank.** `MANAGE_USERS`, held by `UserRole.ADMIN` alone.
Not DEVELOPER: `qorgan.roles` records that ADMIN and DEVELOPER hold the operator set "and
no more" and "will differ here when those land" -- this is the first one that landed, and a
developer login is the vendor's. A vendor able to mint themselves an admin account at the
school is the arrangement the audit condemned, not a convenience.

**No delete button at all.** `Event.reviewed_by_id` is `ON DELETE SET NULL`, so removing a
row would silently blank the reviewer on every bullying event that person ruled on -- a
judgement about a named child, unattributed. Accounts are retired, never deleted.

The invariant "at least one admin can still log in" is enforced below HTTP and is tested
in test_accounts_rules.py, because HTTP cannot reach every way of breaking it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, User
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, Severity, UserRole
from qorgan.events.store import record_event
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
NEW_PASSWORD = "zebra-lantern-9174-quiet"

LoginAs = Callable[..., TestClient]
ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def login_as(settings: Settings, session: Session) -> Iterator[LoginAs]:
    """A logged-in client for a named account. Named rather than role-keyed because the
    rules below turn on WHICH admin is asking, and two admins cannot share a username.

    **A FRESH `create_app()` per client, never two clients over one app.** Several tests
    here log in as two people, and entering the lifespan twice over one app re-assigns
    `app.state.notifier`: the first notification worker is then orphaned -- nothing holds
    it, shutdown stops the second one twice -- and it polls `session_scope()` for the rest
    of the session, rebuilding the module-global engine whenever it happens to wake. That
    surfaces as `no such table: users` in some unrelated test's setup minutes later, which
    is exactly the failure conftest documents. This fixture hit it.

    The lifespan is entered rather than skipped because `with_token` reads the token off a
    rendered page and, for a logged-in session, `/login` redirects to `/` -- the camera
    wall, which reads `app.state.previews`.
    """
    del settings  # applied via the fixture; create_app() reads it
    with ExitStack() as stack:

        def make(username: str, role: UserRole) -> TestClient:
            _account(session, username, role)
            client = stack.enter_context(TestClient(create_app(), follow_redirects=False))
            posted = client.post(
                "/login", data=with_token(client, {"username": username, "password": PASSWORD})
            )
            assert posted.status_code == 303, "login failed"
            return client

        yield make


@pytest.fixture
def client_for(login_as: LoginAs) -> ClientFor:
    return lambda role: login_as(f"user_{role.value}", role)


def _account(session: Session, username: str, role: UserRole, *, active: bool = True) -> User:
    row = User(
        username=username, password_hash=hash_password(PASSWORD), role=role, is_active=active
    )
    session.add(row)
    session.commit()
    return row


def _by_name(session: Session, username: str) -> User:
    session.expire_all()
    found = session.scalar(select(User).where(User.username == username))
    assert found is not None
    return found


def _count(session: Session) -> int:
    session.expire_all()
    return len(list(session.scalars(select(User.id)).all()))


# -- the capability ----------------------------------------------------------


def test_only_the_admin_role_manages_accounts() -> None:
    """The table, below HTTP. Every other role here can, by definition of this capability,
    hand itself the whole system if it holds it."""
    assert Capability.MANAGE_USERS in capabilities_for(UserRole.ADMIN)

    for role in (UserRole.OPERATOR, UserRole.DEVELOPER, UserRole.CANTEEN_STAFF):
        assert Capability.MANAGE_USERS not in capabilities_for(role), (
            f"{role.value} can grant itself every capability in the system"
        )


def test_every_role_still_states_its_capabilities_in_writing() -> None:
    """Adding a capability must not quietly drop a role out of the table: a role the table
    forgets can do nothing, which fails shut but is still a silent outage on a Monday."""
    assert set(ROLE_CAPABILITIES) == set(UserRole)


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.DEVELOPER, UserRole.CANTEEN_STAFF])
def test_a_role_without_the_capability_cannot_read_the_account_list(
    client_for: ClientFor, role: UserRole
) -> None:
    """The list is who works here and what each of them can reach. That is reconnaissance,
    and giving it away is not free."""
    assert client_for(role).get("/users").status_code == 403


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.DEVELOPER, UserRole.CANTEEN_STAFF])
def test_a_role_without_the_capability_cannot_create_an_account(
    client_for: ClientFor, session: Session, role: UserRole
) -> None:
    """**The one that matters.** Creating an account is a privilege-granting act. The row
    count is asserted as well as the status: a 403 that still wrote is a 403 that proved
    nothing, and this is the page where that distinction is the whole product."""
    client = client_for(role)
    before = _count(session)

    response = client.post(
        "/users",
        data=with_token(client, {"username": "mine", "password": NEW_PASSWORD, "role": "admin"}),
    )

    assert response.status_code == 403
    assert _count(session) == before, f"{role.value} created an account"


def test_an_operator_cannot_promote_themselves_to_admin(
    client_for: ClientFor, session: Session
) -> None:
    """Escalation end to end. The row is what matters; the 403 on the way there is only
    evidence, and an operator who reaches this route owns the school's cameras."""
    client = client_for(UserRole.OPERATOR)
    them = _by_name(session, "user_operator")

    response = client.post(f"/users/{them.id}/role", data=with_token(client, {"role": "admin"}))

    assert response.status_code == 403
    assert _by_name(session, "user_operator").role is UserRole.OPERATOR


def test_an_operator_cannot_retire_the_admin_who_would_stop_them(
    client_for: ClientFor, session: Session
) -> None:
    """Locking the admins out is the same attack from the other end."""
    client = client_for(UserRole.OPERATOR)
    boss = _account(session, "head", UserRole.ADMIN)

    response = client.post(f"/users/{boss.id}/active", data=with_token(client, {"active": "false"}))

    assert response.status_code == 403
    assert _by_name(session, "head").is_active is True


def test_the_nav_offers_the_page_only_to_whoever_can_open_it(client_for: ClientFor) -> None:
    """Drawn from the same table the route is gated on. Two sources of truth would mean an
    operator clicking a link into a 403 and reporting the system as broken.

    Read off whatever page each role can actually open rather than off `/` for both: the
    nav lives in `base.html`, so every rendered page carries it, and asserting on the very
    page under test is a stronger reading than asserting on the camera wall.
    """
    assert 'href="/users"' in client_for(UserRole.ADMIN).get("/users").text
    assert 'href="/users"' not in client_for(UserRole.OPERATOR).get("/events").text


# -- acting on your own account ----------------------------------------------


def test_an_admin_cannot_retire_themselves(login_as: LoginAs, session: Session) -> None:
    """`load_user` refuses an inactive account, so this lands on your very NEXT request:
    the browser bounces to /login and the page that could undo it needs the account you
    just shut. There is no self-service case here -- a colleague can retire you, and one
    exists in this test, so refusing costs the school nothing."""
    me = login_as("head", UserRole.ADMIN)
    _account(session, "deputy_head", UserRole.ADMIN)  # not the last admin: this is the SELF rule
    mine = _by_name(session, "head")

    response = me.post(f"/users/{mine.id}/active", data=with_token(me, {"active": "false"}))

    assert response.status_code == 400
    assert _by_name(session, "head").is_active is True
    assert me.get("/users").status_code == 200, "the admin locked themselves out"


def test_an_admin_cannot_change_their_own_role(login_as: LoginAs, session: Session) -> None:
    """Demoting yourself is retiring yourself by another spelling: the capability that
    could put it back is the one you just gave away."""
    me = login_as("head", UserRole.ADMIN)
    _account(session, "deputy_head", UserRole.ADMIN)
    mine = _by_name(session, "head")

    response = me.post(f"/users/{mine.id}/role", data=with_token(me, {"role": "canteen_staff"}))

    assert response.status_code == 400
    assert _by_name(session, "head").role is UserRole.ADMIN
    assert me.get("/users").status_code == 200


def test_the_only_admin_is_refused_before_the_self_rule_is_even_reached(
    login_as: LoginAs, session: Session
) -> None:
    """A school with ONE admin: the last-admin invariant is what refuses, and it refuses
    first, because "you would be the last one" is the reason a headteacher needs to read.
    "You cannot do this to yourself" invites them to ask a colleague, and there is none."""
    me = login_as("head", UserRole.ADMIN)
    mine = _by_name(session, "head")

    response = me.post(f"/users/{mine.id}/active", data=with_token(me, {"active": "false"}))

    assert response.status_code == 400
    assert _by_name(session, "head").is_active is True
    assert me.get("/users").status_code == 200


def test_an_admin_may_retire_another_admin_while_one_remains(
    login_as: LoginAs, session: Session
) -> None:
    """The rules above must not add up to "nothing can ever change". A school whose
    headteacher leaves has to be able to close that account, and this is that day."""
    me = login_as("head", UserRole.ADMIN)
    leaving = _account(session, "departing_head", UserRole.ADMIN)

    response = me.post(f"/users/{leaving.id}/active", data=with_token(me, {"active": "false"}))

    assert response.status_code == 303, response.text
    assert _by_name(session, "departing_head").is_active is False


def test_an_admin_may_change_another_accounts_role(login_as: LoginAs, session: Session) -> None:
    them = _account(session, "moved", UserRole.CANTEEN_STAFF)
    me = login_as("head", UserRole.ADMIN)

    response = me.post(f"/users/{them.id}/role", data=with_token(me, {"role": "operator"}))

    assert response.status_code == 303, response.text
    assert _by_name(session, "moved").role is UserRole.OPERATOR


def test_a_role_that_does_not_exist_changes_nothing(login_as: LoginAs, session: Session) -> None:
    """There is deliberately no superadmin. The role arrives off a form, which means it is
    a string somebody types.

    **`"psychologist"` left this list on 2026-07-28, deliberately and in the same change
    that added `/psychologist`.** It was correct here while §13's pages did not exist;
    `roles.py` refuses a role whose pages nobody built. `"superadmin"` has not moved: §14
    describes one and `UserRole` still does not have one.
    """
    them = _account(session, "moved", UserRole.CANTEEN_STAFF)
    me = login_as("head", UserRole.ADMIN)

    for invented in ("superadmin", "ADMIN", "psychologist "):
        response = me.post(f"/users/{them.id}/role", data=with_token(me, {"role": invented}))
        assert response.status_code in (400, 422), f"{invented!r} was accepted as a role"

    assert _by_name(session, "moved").role is UserRole.CANTEEN_STAFF


def test_an_account_that_does_not_exist_is_a_404_not_a_500(
    login_as: LoginAs, session: Session
) -> None:
    """The real account is exercised FIRST, on the same URLs. Without it a 404 is what a
    missing route returns too, and this test would be green against no feature at all --
    it was, before the route existed."""
    me = login_as("head", UserRole.ADMIN)
    real = _account(session, "moved", UserRole.CANTEEN_STAFF)

    assert (
        me.post(f"/users/{real.id}/role", data=with_token(me, {"role": "operator"})).status_code
        == 303
    ), "the route is not there; the 404s below would prove nothing"
    assert (
        me.post(f"/users/{real.id}/active", data=with_token(me, {"active": "false"})).status_code
        == 303
    )

    assert me.post("/users/9999/role", data=with_token(me, {"role": "operator"})).status_code == 404
    assert (
        me.post("/users/9999/active", data=with_token(me, {"active": "false"})).status_code == 404
    )


# -- what retiring an account actually does ----------------------------------


def test_a_retired_account_cannot_log_in(login_as: LoginAs, session: Session) -> None:
    me = login_as("head", UserRole.ADMIN)
    leaving = _account(session, "departing_head", UserRole.ADMIN)
    me.post(f"/users/{leaving.id}/active", data=with_token(me, {"active": "false"}))

    # No `with`, so no lifespan and no background threads: a login needs neither, and every
    # avoidable notification worker is another chance at the orphaning described on
    # `login_as` above.
    fresh = TestClient(create_app(), follow_redirects=False)
    attempt = fresh.post(
        "/login", data=with_token(fresh, {"username": "departing_head", "password": PASSWORD})
    )

    assert attempt.status_code != 303, "a retired account still logs in"


def test_a_retired_account_loses_the_session_it_already_had(
    login_as: LoginAs, session: Session
) -> None:
    """Deactivation that only stops the NEXT login is not deactivation: the person walked
    out of the building with a live session in a browser. It is also exactly why acting on
    your own account is refused -- the effect lands on the actor, one request later."""
    admin = login_as("head", UserRole.ADMIN)
    operator = login_as("op", UserRole.OPERATOR)
    them = _by_name(session, "op")
    assert operator.get("/events").status_code == 200

    admin.post(f"/users/{them.id}/active", data=with_token(admin, {"active": "false"}))

    assert operator.get("/events").status_code == 303, "a retired account kept working"


def test_a_retired_account_can_be_brought_back(login_as: LoginAs, session: Session) -> None:
    """Retiring is not a one-way door. A staff member returning from leave should not need
    a second account -- a second account is how one person becomes two rows in the review
    history of the same child's incident."""
    me = login_as("head", UserRole.ADMIN)
    them = _account(session, "on_leave", UserRole.OPERATOR, active=False)

    response = me.post(f"/users/{them.id}/active", data=with_token(me, {"active": "true"}))

    assert response.status_code == 303, response.text
    assert _by_name(session, "on_leave").is_active is True


def test_retiring_keeps_the_judgements_they_signed(login_as: LoginAs, session: Session) -> None:
    """Why there is no delete button. `Event.reviewed_by_id` is ON DELETE SET NULL, so a
    removed row would blank the reviewer on every bullying event that person ruled on --
    a decision about a named child, silently unattributed. Retiring keeps the record."""
    admin = login_as("head", UserRole.ADMIN)
    reviewer = _account(session, "reviewer", UserRole.OPERATOR)
    event_id = _event(session)
    session.get(Event, event_id).reviewed_by_id = reviewer.id
    session.commit()

    retired = admin.post(
        f"/users/{reviewer.id}/active", data=with_token(admin, {"active": "false"})
    )

    # Asserted, not assumed: a request that 404s leaves the event untouched too, so
    # without this line the assertions below are green against no feature at all.
    assert retired.status_code == 303, "nothing was retired; the rest of this proves nothing"
    session.expire_all()
    assert session.get(User, reviewer.id) is not None, "the account was deleted, not retired"
    assert session.get(User, reviewer.id).is_active is False
    assert session.get(Event, event_id).reviewed_by_id == reviewer.id


def _event(session: Session) -> int:
    camera = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()
    return record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=Verdict(0.91, 0.85, 0.7, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия",
        track_ids="3,7",
    )

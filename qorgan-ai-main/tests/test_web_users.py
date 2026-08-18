"""The accounts page: listing accounts and creating one.

Until now the only way to give somebody a login was `qorgan user add` at a shell on the
server. That is not a rule anybody chose -- it is the absence of a page -- and its cost is
that the school cannot onboard its own staff without the vendor. The legacy solved that by
having no accounts at all: ~50 endpoints, zero authentication, bound to 0.0.0.0 (audit
C-01). This page is the other answer.

Who may press the buttons is a separate subject and lives in test_web_users_privileges.py
-- it is the part that decides whether this page is a fix or a second C-01.

What this file asserts:

  * the page shows what exists, a page at a time (the legacy loaded whole tables on
    every render, every 2.5 s, per client -- audit M-19);
  * a created account really can log in, with the role it was given;
  * the password rules are the SAME rules the CLI applies, because two doors with
    different locks is one door;
  * the password does not reach the HTML, the log, or a URL;
  * a username is user-supplied text and is escaped (the legacy built its DOM with
    innerHTML from server JSON, so a name like `<img src=x onerror=...>` was stored XSS
    in the operator's browser -- audit H-05).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import BCRYPT_MAX_BYTES, MIN_PASSWORD_LENGTH, hash_password
from qorgan.settings import Settings, get_settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
# Distinctive on purpose: every "did this leak?" assertion below greps for this exact
# string, and a password that looked like the other test data would hide in the noise.
NEW_PASSWORD = "zebra-lantern-9174-quiet"


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def admin(session: Session) -> User:
    user = User(username="head", password_hash=hash_password(PASSWORD), role=UserRole.ADMIN)
    session.add(user)
    session.commit()
    return user


@pytest.fixture
def client(app, admin: User) -> Iterator[TestClient]:
    """The admin's browser. ONE client per app, entered once.

    The lifespan is entered because `with_token` reads the token off a rendered page and,
    for a logged-in session, `/login` redirects to `/` -- the camera wall, which reads
    `app.state.previews`. Entering it twice over one app orphans a notification worker that
    then polls `session_scope()` forever; see `login_as` in test_web_users_privileges.py.
    """
    with TestClient(app, follow_redirects=False) as test_client:
        posted = test_client.post(
            "/login",
            data=with_token(test_client, {"username": admin.username, "password": PASSWORD}),
        )
        assert posted.status_code == 303, "login failed"
        yield test_client


def _fresh_browser(app) -> TestClient:
    """Somebody else's browser, deliberately WITHOUT a second lifespan on the same app.

    It is logged out, so `with_token` finds the token on `/login` itself and never probes
    `/`. Nothing it then does reads `app.state`, so no background thread is needed -- and
    starting a second pair underneath the `client` fixture is what orphans a worker.
    """
    return TestClient(app, follow_redirects=False)


def _create(client: TestClient, **fields: str):
    return client.post("/users", data=with_token(client, dict(fields)))


def _usernames(session: Session) -> list[str]:
    return list(session.scalars(select(User.username)).all())


# -- the list ----------------------------------------------------------------


def test_the_page_lists_the_accounts_that_exist(client: TestClient, session: Session) -> None:
    session.add(
        User(
            username="canteen_lady",
            password_hash=hash_password(PASSWORD),
            role=UserRole.CANTEEN_STAFF,
        )
    )
    session.commit()

    page = client.get("/users")

    assert page.status_code == 200
    assert "head" in page.text
    assert "canteen_lady" in page.text
    assert "canteen_staff" in page.text


def test_each_rows_dropdown_preselects_the_role_that_row_actually_has(
    client: TestClient, session: Session
) -> None:
    """Otherwise an admin who presses "Сменить" without touching the dropdown silently
    moves somebody to whatever was preselected -- a canteen worker becoming an operator,
    which §14 exists to prevent, with nobody having chosen it.

    The hazard is that `AccountRow.role` is a str and `UserRole` is an enum: compare the
    wrong pair and every option is unselected, and the browser then shows the FIRST role in
    the list as though it were the current one.
    """
    session.add(
        User(
            username="canteen_lady",
            password_hash=hash_password(PASSWORD),
            role=UserRole.CANTEEN_STAFF,
        )
    )
    session.commit()

    page = client.get("/users")

    assert page.text.count("selected") == 2, "a row shows a role the account does not have"
    assert '<option value="canteen_staff" selected>' in page.text
    assert '<option value="admin" selected>' in page.text


def test_times_are_shown_on_the_schools_clock_not_in_utc(
    client: TestClient, session: Session, settings: Settings
) -> None:
    """Stored UTC, read by a headteacher in Almaty. Rendering the stored value straight is
    the signature mistake in its purest form: correct in the column, five hours wrong on
    the screen, and wrong in the direction that makes last night's login look like this
    morning's. `UtcDateTime` hands back a tz-aware value precisely so this is a conversion
    and not a guess.
    """
    moment = datetime(2026, 3, 1, 22, 30, tzinfo=UTC)
    account = session.get(User, 1)
    account.created_at = moment
    session.commit()

    page = client.get("/users")

    assert moment.astimezone(settings.tz).strftime("%Y-%m-%d %H:%M") in page.text
    assert moment.strftime("%Y-%m-%d %H:%M") not in page.text, "the page is showing UTC"


def test_a_login_is_recorded_so_a_dormant_account_can_be_found(
    client: TestClient, session: Session
) -> None:
    """The column this page's most useful sort would need. It existed and nothing ever
    wrote it, so the page would have shown an em dash beside every account forever -- and
    "never logged in" is exactly the claim an admin would act on when deciding whom to
    retire. A column displayed as a fact has to be one.
    """
    session.expire_all()
    account = session.get(User, 1)
    assert account.last_login_at is not None, "logging in did not record a login"

    page = client.get("/users")

    assert account.last_login_at.astimezone(get_settings().tz).strftime("%Y-%m-%d %H:%M") in (
        page.text
    )


def test_the_list_is_paginated(client: TestClient, session: Session) -> None:
    """The legacy loaded the entire table on every render, per client, every 2.5 seconds
    (M-19). Fine at 400 rows, fatal at 40 000, and worse every day the system runs."""
    from qorgan.accounts import PAGE_SIZE

    for index in range(PAGE_SIZE + 3):
        session.add(
            User(
                username=f"staff{index:03d}",
                password_hash=hash_password(PASSWORD),
                role=UserRole.OPERATOR,
            )
        )
    session.commit()

    first = client.get("/users")
    second = client.get("/users?page=2")

    assert first.text.count('class="account"') == PAGE_SIZE, "a page is not a page"
    assert 0 < second.text.count('class="account"') < PAGE_SIZE
    assert "staff000" in first.text
    assert "staff000" not in second.text, "page 2 repeated page 1"


def test_opening_the_page_changes_nothing(client: TestClient, session: Session) -> None:
    """ZERO side effects on load. The legacy's `POST /page-activate/{page}` restarted the
    AI workers -- with a five-second thread.join() inside the HTTP handler -- every time
    somebody opened a tab, so coverage depended on which tab was open."""
    # Expired first, so the fingerprint is read from the database rather than from a copy
    # this session cached before the fixture logged in -- logging in legitimately writes
    # `last_login_at`, and comparing a pre-login copy against a post-login read would blame
    # the two GETs below for a change neither of them made.
    session.expire_all()
    before = session.get(User, 1)
    assert before is not None
    fingerprint = (before.username, before.role, before.is_active, before.updated_at)

    assert client.get("/users").status_code == 200
    assert client.get("/users").status_code == 200

    session.expire_all()
    after = session.get(User, 1)
    assert after is not None
    assert (after.username, after.role, after.is_active, after.updated_at) == fingerprint
    assert session.scalar(select(func.count(User.id))) == 1, "reading the page wrote a row"


# -- creating an account -----------------------------------------------------


def test_a_created_account_can_log_in_with_the_role_it_was_given(
    client: TestClient, app, session: Session
) -> None:
    """End to end, because every intermediate assertion can be true while this is false:
    the row can exist, the hash can be well-formed, and the person still cannot get in."""
    response = _create(client, username="newop", password=NEW_PASSWORD, role="operator")
    assert response.status_code == 303, response.text

    fresh = _fresh_browser(app)
    logged_in = fresh.post(
        "/login", data=with_token(fresh, {"username": "newop", "password": NEW_PASSWORD})
    )
    assert logged_in.status_code == 303, "the account this page created cannot log in"
    assert fresh.get("/events").status_code == 200, "created without the role it was given"
    assert fresh.get("/users").status_code == 403, "an operator was created as an admin"


def test_a_username_that_is_already_taken_is_refused(client: TestClient, session: Session) -> None:
    """`users.username` is UNIQUE, so the alternative to this check is a 500 on a form the
    school filled in honestly -- which reads as "the system is broken"."""
    response = _create(client, username="head", password=NEW_PASSWORD, role="operator")

    assert response.status_code == 400
    session.expire_all()
    assert _usernames(session) == ["head"], "a duplicate username was accepted"


def test_a_role_that_does_not_exist_creates_nothing(client: TestClient, session: Session) -> None:
    """There is deliberately no superadmin. A role arriving off a form is a string an
    attacker types, and the set of roles is not negotiable by them.

    **`"psychologist"` used to be in this list and has been REMOVED deliberately, in the
    same change that built §13's pages.** It belonged here for as long as the role guarded
    nothing (`roles.py`: a permission guarding nothing is a guess), and taking it out is
    therefore not a weakening of this test -- it is the world changing. `"superadmin"`
    stays, because §14 describes one and `UserRole` still has none: the multi-school routes
    it would guard do not exist yet.

    The near-misses matter as much as the invented names: `"admin "` and `"ADMIN"` are the
    shapes a real role is nearly typed in, and `UserRole` is exact.
    """
    for invented in ("superadmin", "admin ", "ADMIN", "psychologist "):
        response = _create(client, username="ghost", password=NEW_PASSWORD, role=invented)
        assert response.status_code in (400, 422), f"{invented!r} was accepted as a role"

    session.expire_all()
    assert _usernames(session) == ["head"]


def test_a_blank_username_creates_nothing(client: TestClient, session: Session) -> None:
    response = _create(client, username="   ", password=NEW_PASSWORD, role="operator")

    assert response.status_code == 400
    session.expire_all()
    assert _usernames(session) == ["head"]


# -- the password ------------------------------------------------------------


def test_the_password_reaches_neither_the_page_nor_the_log_nor_a_url(
    client: TestClient, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    """A password is write-only. It goes in the form and into bcrypt, and nowhere else.

    The URL matters as much as the page: a query string is written to the browser history,
    to the referrer of every asset the next page loads, and to any proxy log between the
    school and this server. A redirect carrying it would put it in all three at once.
    """
    with caplog.at_level(logging.DEBUG):
        response = _create(client, username="newop", password=NEW_PASSWORD, role="operator")

    assert NEW_PASSWORD not in response.text
    assert NEW_PASSWORD not in response.headers.get("location", "")
    assert NEW_PASSWORD not in str(response.url)
    for record in caplog.records:
        # `extra=` fields never reach `caplog.text`, and `extra={"password": ...}` is
        # exactly the mistake this guards, so the record's whole payload is searched.
        assert NEW_PASSWORD not in repr(record.__dict__), f"logged by {record.name}"

    session.expire_all()
    stored = session.scalar(select(User).where(User.username == "newop"))
    assert stored is not None
    assert NEW_PASSWORD not in stored.password_hash
    assert stored.password_hash.startswith("$2"), "not a bcrypt hash"


def test_a_refused_form_does_not_hand_the_password_back_to_the_browser(
    client: TestClient,
) -> None:
    """The refusal path is the one that forgets. Re-rendering a form usually means putting
    the values back, and putting THIS value back writes it into the page source."""
    response = _create(client, username="head", password=NEW_PASSWORD, role="operator")

    assert response.status_code == 400
    assert NEW_PASSWORD not in response.text, "the rejected password was echoed into the HTML"
    assert "head" in response.text, "the username was not preserved, so the form is unusable"


def test_a_short_password_is_refused_by_the_page_exactly_as_by_the_cli(
    client: TestClient, session: Session
) -> None:
    """One minimum, not two. `qorgan user add` has refused short passwords since day one;
    a browser form with a laxer rule would silently become the way to create weak accounts,
    and nobody would ever see the two numbers side by side."""
    short = "a" * (MIN_PASSWORD_LENGTH - 1)

    response = _create(client, username="weak", password=short, role="operator")

    assert response.status_code == 400
    session.expire_all()
    assert _usernames(session) == ["head"], "an account below the minimum was created"


def test_a_password_at_the_minimum_is_accepted(client: TestClient) -> None:
    """The boundary in the other direction: an off-by-one here locks out a valid password
    and the school reports it as "the form does not work"."""
    exact = "b" * MIN_PASSWORD_LENGTH

    response = _create(client, username="exact", password=exact, role="operator")

    assert response.status_code == 303, response.text


def test_a_password_past_bcrypts_72_bytes_is_refused_rather_than_truncated(
    client: TestClient, app, session: Session
) -> None:
    """bcrypt silently ignores everything past 72 bytes. Truncating would mean the school
    types a 90-character passphrase and gets a 72-character one -- and the 72-character
    PREFIX would then be a working password for that account, which nobody was told."""
    too_long = "q" * (BCRYPT_MAX_BYTES + 1)

    response = _create(client, username="verbose", password=too_long, role="operator")

    assert response.status_code == 400, "a password longer than bcrypt can read was accepted"
    session.expire_all()
    assert _usernames(session) == ["head"]

    prefix = too_long[:BCRYPT_MAX_BYTES]
    fresh = _fresh_browser(app)
    attempt = fresh.post(
        "/login", data=with_token(fresh, {"username": "verbose", "password": prefix})
    )

    assert attempt.status_code != 303, "the truncated prefix logged in; it was truncated"


def test_a_multibyte_password_is_measured_in_bytes_not_characters(client: TestClient) -> None:
    """bcrypt's limit is 72 BYTES. Cyrillic is two bytes a character in UTF-8, so a
    36-character Russian passphrase is already at the limit and a 40-character one is over
    it. Counting characters here would let it through and hand it to bcrypt to truncate --
    a value true in the form layer and quietly wrong one layer down."""
    forty_cyrillic_characters = "п" * 40

    response = _create(client, username="rus", password=forty_cyrillic_characters, role="operator")

    assert response.status_code == 400, "80 bytes of password passed a 72-byte limit"


# -- a username is user-supplied text ----------------------------------------


def test_a_username_shaped_like_a_script_is_escaped_not_executed(
    client: TestClient, session: Session
) -> None:
    """Audit H-05. The legacy rendered names into the DOM with innerHTML from server JSON,
    so `<img src=x onerror=...>` in a name ran in the operator's browser."""
    attack = '<img src=x onerror="alert(1)">'
    session.add(
        User(username=attack, password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR)
    )
    session.commit()

    page = client.get("/users")

    assert page.status_code == 200
    assert attack not in page.text, "a username was rendered as markup"
    # The whole payload, entity-encoded. Asserting the ABSENCE of `onerror=` would pass
    # for the wrong reason and fail for the wrong one: `=` is not an escaped character, so
    # the substring survives escaping as inert text. What makes it inert is that `<` and
    # the quotes did not -- so that is what gets asserted.
    assert "&lt;img src=x onerror=&#34;alert(1)&#34;&gt;" in page.text, (
        "the username is not on the page at all; nothing was proven"
    )
    assert "<img" not in page.text, "an img tag was built out of a username"


# -- CSRF: on this page a forged request creates an ACCOUNT -------------------


def test_creating_an_account_without_a_token_is_refused(
    client: TestClient, session: Session
) -> None:
    """`SameSite=lax` still sends the cookie on a top-level navigation, and an auto-
    submitting form IS one. So an admin who clicks a link in an email would otherwise mint
    the attacker an account -- an admin account, on a system full of children's faces."""
    response = client.post(
        "/users", data={"username": "attacker", "password": NEW_PASSWORD, "role": "admin"}
    )

    assert response.status_code == 403
    session.expire_all()
    assert _usernames(session) == ["head"], "a forged request created an account"


def test_changing_a_role_without_a_token_is_refused(client: TestClient, session: Session) -> None:
    victim = User(
        username="op", password_hash=hash_password(PASSWORD), role=UserRole.CANTEEN_STAFF
    )
    session.add(victim)
    session.commit()

    response = client.post(f"/users/{victim.id}/role", data={"role": "admin"})

    assert response.status_code == 403
    session.expire_all()
    assert session.get(User, victim.id).role is UserRole.CANTEEN_STAFF

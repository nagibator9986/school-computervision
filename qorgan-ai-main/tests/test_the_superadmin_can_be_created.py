"""The role that manages schools, created the way an installation creates it.

`/schools` is gated on `Capability.MANAGE_SCHOOLS`, which `UserRole.SUPERADMIN` holds
alone. **Nothing in the tree could create a SUPERADMIN.** The form's `ASSIGNABLE_ROLES`
excludes it -- correctly, because the accounts page is gated on MANAGE_USERS, which a
school's own ADMIN holds -- `parse_role` therefore refused it as an "unknown role", and
`qorgan user add` went through `parse_role` too while `--help` cheerfully offered
`--role superadmin`. The comment in `accounts.py` that said how to make one named a flag
(`--superadmin`) that occurred nowhere else in the tree. So the page was reachable by
nobody, permanently, and MANAGE_SCHOOLS guarded nothing.

**2583 tests stayed green over it, and this file is mostly about why.** Every superadmin
test in the suite mints its user as an ORM row -- `tests/test_schools_page.py`'s `accounts`
fixture writes `User(role=SUPERADMIN, school_id=None)` straight into the database -- and
nothing under `tests/` touched `parse_role` at all. A test that does not travel the path a
human travels proves something other than what it appears to prove.

So every test here goes through the real command, and the fixture asserts the database was
EMPTY first: a row minted by hand cannot be mistaken for a row `qorgan user add` wrote,
which is the one thing the existing superadmin tests cannot tell apart.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan import accounts, cli
from qorgan.accounts import ASSIGNABLE_ROLES, RoleRejected, parse_role
from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.conftest import SRC_DIR
from tests.web_login import with_token

USERNAME = "installer"
PASSWORD = "correct-horse-battery"


@pytest.fixture
def created_by_the_installation(
    settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch
) -> User:
    """`qorgan user add installer --role superadmin`, run the way an installer runs it.

    **The empty-database assertion is what binds this fixture to the command.** Without
    it every test below would pass just as happily against a row written with
    `session.add(User(...))` -- which is precisely how a page nobody could reach kept a
    green suite. The password is answered at the prompt, so the hash the login later
    checks is the one this command wrote.
    """
    del settings  # applied by the fixture; the CLI reads it through get_settings()
    assert session.scalar(select(func.count(User.id))) == 0, (
        "this database already holds an account, so nothing below could tell a row the "
        "CLI wrote from a row a fixture minted -- which is the whole point of this file"
    )
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: PASSWORD)

    assert cli.main(["user", "add", USERNAME, "--role", UserRole.SUPERADMIN.value]) == 0, (
        "`qorgan user add <name> --role superadmin` created no account. argparse offers "
        "that value in --help; if nothing accepts it, the only role that holds "
        "MANAGE_SCHOOLS cannot be created by anybody and /schools is unreachable forever."
    )

    session.expire_all()
    rows = session.scalars(select(User)).all()
    assert len(rows) == 1, f"the command wrote {len(rows)} accounts, not one"
    return rows[0]


@pytest.fixture
def one_account_per_role(
    settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One account per role, each created by the command an installer actually runs.

    Not `session.add(User(...))`: this file's whole subject is that a row minted by hand
    proves something other than what it appears to prove. The username is the role's own
    name, so the login below reads as what it is.
    """
    del settings  # applied by the fixture
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: PASSWORD)
    for role in UserRole:
        assert cli.main(["user", "add", role.value, "--role", role.value]) == 0, (
            f"`qorgan user add {role.value} --role {role.value}` failed, so nothing "
            "below can be asked about that role"
        )
    session.expire_all()


def test_the_installation_can_create_the_role_that_manages_schools(
    created_by_the_installation: User,
) -> None:
    """The finding itself. `qorgan user add alice --role superadmin` was offered by
    `--help`, accepted by argparse, and then refused as "unknown role" -- about a role
    that is in the enum and in the help text."""
    assert created_by_the_installation.role is UserRole.SUPERADMIN
    assert created_by_the_installation.is_active is True


def test_the_account_the_command_wrote_belongs_to_no_school(
    created_by_the_installation: User,
) -> None:
    """`users.school_id IS NULL` for this role and for no other.

    Documented on the model since it was written and enforced by nothing. Had the writer
    resolved a school for it, the one account that reaches every school would sit on one
    school's staff list (`accounts.list_accounts`), where a headteacher could see it and
    try to retire it, and `web.security.school_of` would hand it a school's data instead
    of making the refusal it is written to make.
    """
    assert created_by_the_installation.school_id is None, (
        "the command gave the superadmin a school. That row is now a member of one "
        "school's staff, listed on its /users page and offered its data by school_of."
    )


def test_the_superadmin_the_command_created_reaches_the_schools_register(
    created_by_the_installation: User,
) -> None:
    """The other half, and the one the finding is about: the page -- reached the way a
    person reaches it, by logging in and GOING WHERE THE SERVER SENDS THEM.

    **This test asserted only that some redirect happened, and a re-review measured what
    that was hiding.** `POST /login` answered `303 Location: /login`, so a correct password
    returned the installation's own account to a blank login form -- no nav, no username,
    no link to the register, no error -- indistinguishable from a wrong password. The
    status is 303 either way, so asserting the status alone is the difference between a
    test that documents that defect and one that catches it. This test was itself an
    instance of the class it exists to close: it did not travel the path a human travels,
    it typed the URL.

    Logged in with the password TYPED AT THE PROMPT, so the hash being checked is the one
    `qorgan user add` wrote. A row minted by hand cannot satisfy both this and the
    empty-database assertion in the fixture, which is what keeps this test on the path.
    """
    del created_by_the_installation  # the command IS the fixture; this test uses its login
    with TestClient(create_app(), follow_redirects=False) as client:
        signed_in = client.post(
            "/login", data=with_token(client, {"username": USERNAME, "password": PASSWORD})
        )
        assert signed_in.status_code == 303, "the account the command created cannot log in"

        landed_on = signed_in.headers["location"]
        assert landed_on != "/login", (
            "a correct password sent the installation's own account BACK TO THE LOGIN "
            "FORM. That page carries no nav, no username and no error, so it cannot be "
            "told apart from a failed login -- `landing_for` has no arm for the one role "
            "that holds MANAGE_SCHOOLS, and this account therefore cannot get anywhere "
            "from a browser however correct its password is."
        )
        page = client.get(landed_on)

    assert page.status_code == 200, (
        f"logging in landed on {landed_on!r}, which answered HTTP {page.status_code}. A "
        "landing page the account cannot open is the same dead end, one redirect later."
    )
    assert 'id="second-school-warning"' in page.text, (
        f"logging in landed on {landed_on!r}, and that is not the schools register. "
        "MANAGE_SCHOOLS is held by this role alone, so the register is the page this "
        "account exists to open."
    )


def test_the_form_still_refuses_the_role_the_shell_can_create() -> None:
    """The security property the whole finding sits on top of, asked directly.

    **Nothing under `tests/` touched `parse_role` before this line.** That is how a
    dropdown, a parser and a command line could disagree without anything going red. The
    accounts page is gated on MANAGE_USERS, held by a school's own ADMIN -- so a form that
    accepted `role=superadmin` would let one headteacher mint an account reaching every
    school on the installation. A shell on the server is the installation; a form served
    to a tenant is not, and that asymmetry is the fix rather than an inconsistency in it.
    """
    assert UserRole.SUPERADMIN not in ASSIGNABLE_ROLES, (
        "the superadmin is assignable from the accounts page. One school's headteacher "
        "can now mint an account that reads every school on this installation."
    )
    with pytest.raises(RoleRejected):
        parse_role(UserRole.SUPERADMIN.value)


def _modules_that_use(name: str, package: Path) -> list[str]:
    """Files under `package` that IMPORT or CALL `name` -- not ones that merely mention it.

    **This was a substring grep, and it cried wolf on its own author.** `web/security.py`
    explains the login dead end it fixes by naming `accounts.create_superadmin` in prose,
    and the grep called that a caller. A guard that cannot tell "used" from "written
    about" is a guard somebody deletes the first time it is wrong -- and this one is
    protecting the asymmetry the whole finding rests on, so it has to be right about what
    it accuses. Parsed instead: an `ImportFrom` naming it, or a call to it under either
    spelling.
    """
    found = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = isinstance(node, ast.ImportFrom) and any(
                alias.name == name for alias in node.names
            )
            called = isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == name)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            )
            if imported or called:
                found.append(str(path.relative_to(SRC_DIR)))
                break
    return sorted(found)


def test_the_web_package_cannot_reach_the_installations_own_door() -> None:
    """`create_superadmin` is the way through that the form does not have, so no module
    under `qorgan/web/` may import or call it. Asserted against the source rather than
    described: "somebody would notice" is what the comment naming `--superadmin` assumed.

    The `hasattr` is not ceremony. Without it this check passes vacuously the moment the
    function is renamed or deleted -- a search for a symbol that no longer exists finds
    nothing and reads exactly like safety, which is the failure mode this file is about.
    """
    assert hasattr(accounts, "create_superadmin"), (
        "qorgan.accounts.create_superadmin is gone, so the search below proves nothing. "
        "If the installation's door has been renamed, rename it here too."
    )

    callers = _modules_that_use("create_superadmin", SRC_DIR / "qorgan" / "web")

    assert not callers, (
        f"{callers} can create a superadmin. Every page in that package is reachable "
        "with a capability a school's own ADMIN holds and can grant."
    )


# Where a correct password puts each role. **Written down rather than derived from
# `landing_for`**, because deriving it would restate the code and agree with any version of
# it -- including a reordered one.
#
# The reorder that actually bites is not the superadmin's. OPERATOR, ADMIN and DEVELOPER
# hold VIEW_CAMERAS *and* VIEW_CANTEEN, so checking VIEW_CANTEEN first moves three of the
# five roles off the camera wall in one line. The previous version of this test asserted
# only `!= "/login"` and opened nothing, so all three stayed green through exactly that.
# A test that asserts only a refusal is this project's most repeated defect.
#
# **PSYCHOLOGIST arrived from the other branch and this map is what caught it.** The two
# roles were built in parallel: `feat/psychologist-cabinet` added the role and the arm in
# `landing_for`, `feat/multi-school` added this map, and neither could see the other. The
# merge left the code right and the map five-sixths complete, and
# `test_the_landing_map_names_every_role_the_system_has` went red for exactly that -- which
# is the whole reason it asserts over `UserRole` instead of over its own keys.
#
# `/psychologist` and NOT `/canteen`: the role holds VIEW_CANTEEN too (§13's «посещаемость»
# is the canteen record), so this entry also pins the ORDER of the two arms in
# `landing_for`. Swap them and a psychologist lands on the school's lunch journal.
LANDS_ON = {
    UserRole.OPERATOR: "/",
    UserRole.ADMIN: "/",
    UserRole.DEVELOPER: "/",
    UserRole.CANTEEN_STAFF: "/canteen",
    UserRole.PSYCHOLOGIST: "/psychologist",
    UserRole.SUPERADMIN: "/schools",
}


def test_the_landing_map_names_every_role_the_system_has() -> None:
    """A role added later must not be silently untested by the map above."""
    assert set(LANDS_ON) == set(UserRole), (
        f"the landing map covers {sorted(r.value for r in LANDS_ON)} but the system has "
        f"{sorted(r.value for r in UserRole)}. A role with no landing lands on /login, "
        "which is a blank form with no error -- decide where it goes."
    )


@pytest.mark.parametrize("role", [r.value for r in UserRole])
def test_each_role_lands_on_a_page_it_can_actually_open(
    one_account_per_role: None, role: str
) -> None:
    """Log in as every role, follow the redirect, and OPEN what is on the other side.

    Three assertions, and each one closes a different way this has failed:

      * **the exact landing**, so a reorder of `landing_for` goes red rather than quietly
        moving three roles off the camera wall;
      * **HTTP 200 on that page**, because a landing the role cannot open is the same dead
        end one redirect later -- `!= "/login"` would not have noticed;
      * **the brand link points at the landing**, which is the masthead the superadmin used
        to meet: `base.html` computed its own destination and sent that role to `/canteen`,
        a 403, from the top of the page it had just been given.

    Every account here is made by `qorgan user add`, so this walks the whole path an
    installer walks: create, log in, arrive, and click the first thing on the screen.
    """
    expected = LANDS_ON[UserRole(role)]
    with TestClient(create_app(), follow_redirects=False) as client:
        signed_in = client.post(
            "/login", data=with_token(client, {"username": role, "password": PASSWORD})
        )
        assert signed_in.status_code == 303, f"{role} could not log in"

        landed_on = signed_in.headers["location"]
        assert landed_on == expected, (
            f"{role} lands on {landed_on!r}, not {expected!r}. If `landing_for` was "
            "reordered, note that OPERATOR, ADMIN and DEVELOPER hold both VIEW_CAMERAS "
            "and VIEW_CANTEEN, so one swapped line moves three roles at once."
        )
        opened = client.get(landed_on)

    assert opened.status_code == 200, (
        f"{role} lands on {landed_on!r}, which answers HTTP {opened.status_code}. A "
        "landing page the role cannot open is a login the school reports as broken."
    )
    assert f'class="brand" href="{landed_on}"' in opened.text, (
        f"the masthead link on {landed_on!r} does not point where {role} belongs. That is "
        "a second source of truth about where a role goes, and it 403'd the superadmin "
        "from the top of the page it had just landed on."
    )


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            return action.choices[name]
    raise AssertionError(f"`qorgan {name}` is not a subcommand any more")


def role_choices() -> list[str]:
    """The values `qorgan user add --help` actually offers, read off the built parser.

    **Off the parser, not off `UserRole`.** The claim worth guarding is that the parser's
    offer and what `_cmd_user_add` accepts agree -- offering a value and then calling it an
    "unknown role" is the original defect. Parametrising over the enum instead asserted
    something wider and already untrue: "every role is CLI-creatable" is false the moment a
    second school exists, which is what `--school` is for.
    """
    add_user = _subparser(_subparser(cli.build_parser(), "user"), "add")
    role = next(a for a in add_user._actions if "--role" in a.option_strings)
    assert role.choices, "`--role` offers no choices, so this check would prove nothing"
    return list(role.choices)


@pytest.mark.parametrize("role", role_choices())
def test_every_role_the_command_offers_can_actually_be_created(
    settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    """Every value `--help` advertises must work when used.

    A command that offers a choice it will always reject, and then calls that choice
    *unknown*, is a lie in two directions at once -- which is exactly what `superadmin`
    was. This is the agreement between the parser's `choices` and `_cmd_user_add`, and
    nothing else: it is parametrised over the parser, so narrowing the offer narrows the
    check with it rather than leaving a decision-forcing trap that is merely noise.
    """
    del settings  # applied by the fixture
    name = f"someone_{role}"
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: PASSWORD)

    assert cli.main(["user", "add", name, "--role", role]) == 0, (
        f"`qorgan user add <name> --role {role}` is offered by --help and refused when used"
    )

    session.expire_all()
    created = session.scalar(select(User).where(User.username == name))
    assert created is not None and created.role.value == role


def test_only_a_superadmin_belongs_to_no_school(
    settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The test `db/models/auth.py:27` has named since it was written, which did not
    exist.** `grep -rn only_a_superadmin_belongs_to_no_school .` returned exactly one hit
    -- that comment. The same disease as the `--superadmin` flag this file was opened for,
    one module over, and in the very file cited as the documentation for the NULL.

    `users.school_id IS NULL` means "not any one school's", and three places branch on it:
    `web.security.school_of` refuses a school's data to such a row, `accounts.list_accounts`
    keeps it off every school's staff list, and `diagnostics.alerts.undelivered` returns an
    empty panel for it. Nothing asserted it held in the database.

    Asked of every role, through the real command, because the invariant has two halves and
    a test of one half is a test of neither: the superadmin MUST have no school, and every
    other role MUST have one. `school_id=None` silently becomes the sole school's id
    (`school_key` carries a column default), so the wrong half is the easy one to get.
    """
    del settings  # applied by the fixture
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: PASSWORD)
    for role in UserRole:
        assert cli.main(["user", "add", f"every_{role.value}", "--role", role.value]) == 0

    session.expire_all()
    rows = session.scalars(select(User)).all()
    assert len(rows) == len(UserRole), "one account per role, or this asks less than it says"
    schoolless = sorted(user.role.value for user in rows if user.school_id is None)

    assert schoolless == [UserRole.SUPERADMIN.value], (
        f"the accounts belonging to no school are {schoolless}. NULL here has exactly one "
        "meaning -- the суперадминистратор, who is not inside a school. A second role with "
        "NULL makes school_of refuse somebody who should be let in; a superadmin WITHOUT "
        "NULL puts the one account that reaches every school onto one school's staff list."
    )

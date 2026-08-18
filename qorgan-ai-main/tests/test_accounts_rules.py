"""The account rules themselves, below any HTTP.

They live in `qorgan.accounts` rather than in the route because there are already TWO front
doors -- `qorgan user add` at a shell and the accounts page in a browser -- and a rule
written in one of them is a rule the other does not have. The CLI has refused short
passwords since day one; a browser form that restated the minimum as its own number would
quietly become the way to create weak accounts, and nobody would ever see the two numbers
side by side. So the tests here call the module both front doors call.

**Why the last-admin invariant is tested here and not over HTTP.** Only ADMIN holds
`MANAGE_USERS`, so the only person who can retire the last active admin is that admin -- at
HTTP the target is always the actor, and `test_the_only_admin_is_refused_before_the_self_
rule_is_even_reached` covers that path. The invariant is nonetheless checked against the
row count inside the writing transaction, for the two cases HTTP cannot show: two admins
each retiring the other at the same moment (both pass a check made before either wrote),
and the next front door somebody builds. This file is where a check with no reachable
route is either justified or deleted; it is justified, and here is the proof it fires.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.accounts import (
    LastActiveAdmin,
    NotYourOwnAccount,
    UsernameRejected,
    UsernameTaken,
    create_account,
    set_active,
    set_role,
)
from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import (
    BCRYPT_MAX_BYTES,
    MIN_PASSWORD_LENGTH,
    PasswordTooLong,
    PasswordTooShort,
    hash_password,
    verify_password,
)
from qorgan.settings import Settings

PASSWORD = "correct-horse-battery"


def _account(session: Session, username: str, role: UserRole, *, active: bool = True) -> User:
    row = User(
        username=username, password_hash=hash_password(PASSWORD), role=role, is_active=active
    )
    session.add(row)
    session.commit()
    return row


def _reload(session: Session, user_id: int) -> User:
    session.expire_all()
    found = session.get(User, user_id)
    assert found is not None
    return found


# -- one password rule, not one per front door -------------------------------


def test_the_minimum_length_is_enforced_where_the_password_is_hashed() -> None:
    """Not in the route and not in the CLI: in the one function both of them must call to
    turn a password into a stored value. A rule you can bypass by calling the layer
    underneath it is a rule for whoever remembers it."""
    with pytest.raises(PasswordTooShort):
        hash_password("a" * (MIN_PASSWORD_LENGTH - 1))

    assert hash_password("a" * MIN_PASSWORD_LENGTH).startswith("$2")


def test_bcrypts_72_bytes_still_rejects_rather_than_truncates() -> None:
    """The property that must survive this change. bcrypt ignores everything past 72 bytes,
    so truncating would make the 72-byte PREFIX a working password for an account whose
    owner typed 90 characters and was never told."""
    with pytest.raises(PasswordTooLong):
        hash_password("q" * (BCRYPT_MAX_BYTES + 1))


def test_the_maximum_is_bytes_because_that_is_what_bcrypt_reads() -> None:
    """Cyrillic is two bytes a character in UTF-8, so 40 Russian letters is 80 bytes. A
    maximum counted in characters is true in the form and wrong one layer down, where the
    tail is discarded and the 72-byte prefix quietly becomes a working password."""
    with pytest.raises(PasswordTooLong):
        hash_password("п" * 40)

    assert hash_password("п" * 36).startswith("$2"), "36 Cyrillic characters is exactly 72 bytes"


def test_the_minimum_is_characters_because_that_is_what_a_guesser_guesses() -> None:
    """The same confusion of units, pointed the other way -- and this one lets a weak
    password IN. Nine Russian letters is 18 bytes: measured in bytes it clears a bar of 10
    while being four characters short of it. `qorgan user add` has always counted
    characters, and moving the rule must not have quietly changed what it counts."""
    with pytest.raises(PasswordTooShort):
        hash_password("п" * (MIN_PASSWORD_LENGTH - 1))


def test_a_rejected_password_never_appears_in_the_error(session: Session) -> None:
    """The refusal is rendered to the browser and written to the log. A message that
    quotes the value it refused puts the password in both."""
    secret = "z" * (BCRYPT_MAX_BYTES + 1)

    with pytest.raises(PasswordTooLong) as raised:
        hash_password(secret)
    assert secret not in str(raised.value)

    short = "y" * (MIN_PASSWORD_LENGTH - 1)
    with pytest.raises(PasswordTooShort) as raised_short:
        hash_password(short)
    assert short not in str(raised_short.value)


def test_the_cli_and_the_page_refuse_the_same_passwords(
    settings: Settings, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`qorgan user add` is the other front door. It must not be the lax one."""
    from qorgan import cli

    del settings  # applied via the fixture

    for typed in ("a" * (MIN_PASSWORD_LENGTH - 1), "q" * (BCRYPT_MAX_BYTES + 1)):
        monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt, typed=typed: typed)
        assert cli.main(["user", "add", "cli_user", "--role", "operator"]) == 1

    session.expire_all()
    assert session.scalar(select(User).where(User.username == "cli_user")) is None


def test_a_created_account_verifies_with_the_password_it_was_given(session: Session) -> None:
    create_account("newop", "zebra-lantern-9174", UserRole.OPERATOR)

    session.expire_all()
    stored = session.scalar(select(User).where(User.username == "newop"))
    assert stored is not None
    assert verify_password("zebra-lantern-9174", stored.password_hash)
    assert stored.role is UserRole.OPERATOR
    assert stored.is_active is True


# -- usernames ---------------------------------------------------------------


def test_a_username_is_taken_only_once(session: Session) -> None:
    """`users.username` is UNIQUE. Without this the alternative is an IntegrityError, which
    reaches a headteacher as a 500 on a form they filled in honestly."""
    _account(session, "head", UserRole.ADMIN)

    with pytest.raises(UsernameTaken):
        create_account("head", PASSWORD, UserRole.OPERATOR)


@pytest.mark.parametrize("bad", ["", "   ", "\t", "a" * 65])
def test_a_username_that_the_column_cannot_hold_is_refused(session: Session, bad: str) -> None:
    """64 characters is what the column holds. SQLite does not enforce VARCHAR length, so a
    65-character name would be accepted here and truncated by PostgreSQL later -- the same
    value, two different people, depending on which database the school is running."""
    with pytest.raises(UsernameRejected):
        create_account(bad, PASSWORD, UserRole.OPERATOR)


def test_surrounding_whitespace_is_not_a_second_account(session: Session) -> None:
    """`" head"` and `"head"` are the same person to everyone except a UNIQUE index."""
    _account(session, "head", UserRole.ADMIN)

    with pytest.raises(UsernameTaken):
        create_account("  head  ", PASSWORD, UserRole.OPERATOR)


# -- at least one admin must be able to log in -------------------------------


def test_the_last_active_admin_cannot_be_retired_by_anyone(session: Session) -> None:
    """Zero active admins means `MANAGE_USERS` is held by nobody and no browser anywhere
    can create an admin again. Recovery is a shell on the server -- i.e. the vendor -- and
    needing the vendor to unlock your own school is the relationship this rewrite ends.

    The actor here is a second admin, which HTTP cannot produce for this case: it is the
    shape of the concurrent pair of requests, checked where the write happens.
    """
    only_admin = _account(session, "head", UserRole.ADMIN)
    someone_else = _account(session, "deputy", UserRole.OPERATOR)

    with pytest.raises(LastActiveAdmin):
        set_active(only_admin.id, False, acting_user_id=someone_else.id)

    assert _reload(session, only_admin.id).is_active is True


def test_the_last_active_admin_cannot_be_demoted(session: Session) -> None:
    only_admin = _account(session, "head", UserRole.ADMIN)
    someone_else = _account(session, "deputy", UserRole.OPERATOR)

    with pytest.raises(LastActiveAdmin):
        set_role(only_admin.id, UserRole.OPERATOR, acting_user_id=someone_else.id)

    assert _reload(session, only_admin.id).role is UserRole.ADMIN


def test_an_admin_who_cannot_log_in_is_not_an_admin_who_remains(session: Session) -> None:
    """**The signature mistake, guarded.** Two ADMIN rows exist, so a check that counts
    rows sees cover and allows the write. One of them is retired and cannot log in, so
    there is exactly one door left and this closes it.

    This is the same defect the project has now hit three times: a count that is true in
    the row layer and false in the layer that decides who can actually get in.
    """
    _account(session, "retired_head", UserRole.ADMIN, active=False)
    working = _account(session, "head", UserRole.ADMIN)
    someone_else = _account(session, "deputy", UserRole.OPERATOR)

    with pytest.raises(LastActiveAdmin):
        set_active(working.id, False, acting_user_id=someone_else.id)

    assert _reload(session, working.id).is_active is True


def test_the_only_admin_hears_about_the_last_admin_not_about_themselves(session: Session) -> None:
    """Both rules fire on a one-admin school retiring itself, so the ORDER decides what
    the headteacher reads. "You would be the last one" is actionable -- promote somebody
    first. "You cannot do this to yourself" tells them to ask a colleague who is not
    there, and a refusal that suggests an impossible remedy reads as a broken system."""
    only_admin = _account(session, "head", UserRole.ADMIN)

    with pytest.raises(LastActiveAdmin):
        set_active(only_admin.id, False, acting_user_id=only_admin.id)


def test_a_second_active_admin_makes_the_first_retirable(session: Session) -> None:
    """The invariant must permit the ordinary case, or it is an outage with a rationale."""
    leaving = _account(session, "head", UserRole.ADMIN)
    staying = _account(session, "deputy_head", UserRole.ADMIN)

    set_active(leaving.id, False, acting_user_id=staying.id)

    assert _reload(session, leaving.id).is_active is False


def test_retiring_a_non_admin_is_never_blocked_by_the_admin_invariant(session: Session) -> None:
    admin = _account(session, "head", UserRole.ADMIN)
    worker = _account(session, "canteen_lady", UserRole.CANTEEN_STAFF)

    set_active(worker.id, False, acting_user_id=admin.id)

    assert _reload(session, worker.id).is_active is False


def test_promoting_somebody_to_admin_is_not_blocked(session: Session) -> None:
    admin = _account(session, "head", UserRole.ADMIN)
    worker = _account(session, "op", UserRole.OPERATOR)

    set_role(worker.id, UserRole.ADMIN, acting_user_id=admin.id)

    assert _reload(session, worker.id).role is UserRole.ADMIN


# -- your own account --------------------------------------------------------


def test_nobody_acts_on_their_own_account(session: Session) -> None:
    """Checked in the module, not in the route, so the next caller cannot forget it."""
    me = _account(session, "head", UserRole.ADMIN)
    _account(session, "deputy_head", UserRole.ADMIN)  # so it is the SELF rule that refuses

    with pytest.raises(NotYourOwnAccount):
        set_active(me.id, False, acting_user_id=me.id)
    with pytest.raises(NotYourOwnAccount):
        set_role(me.id, UserRole.OPERATOR, acting_user_id=me.id)

    assert _reload(session, me.id).is_active is True
    assert _reload(session, me.id).role is UserRole.ADMIN


def test_reactivating_your_own_account_is_refused_too(session: Session) -> None:
    """Symmetry is not the reason -- reachability is. An inactive account has no session,
    so an actor can only ever be active; the rule stays whole so that the day a token or a
    script acts on somebody's behalf, "acting on yourself" still means one thing."""
    me = _account(session, "head", UserRole.ADMIN)
    _account(session, "deputy_head", UserRole.ADMIN)

    with pytest.raises(NotYourOwnAccount):
        set_active(me.id, True, acting_user_id=me.id)

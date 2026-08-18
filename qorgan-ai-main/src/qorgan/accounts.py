"""Creating and retiring dashboard accounts.

**Why this is a module and not the body of a route.** There are two front doors into the
`users` table -- `qorgan user add` at a shell on the server, and the accounts page in a
browser -- and a rule written inside one of them is a rule the other does not have. The CLI
has refused short passwords since day one; a browser form restating that minimum as its own
number would be two numbers nobody ever sees side by side, and the day they drift apart the
laxer door silently becomes how weak accounts get made. Both doors call this module, and
the password rules themselves live one layer further down still, in `hash_password`, which
is the only way a password becomes a stored value at all.

**One invariant is enforced here that no caller can opt out of: at least one ADMIN must be
able to log in.** `Capability.MANAGE_USERS` is held by ADMIN alone, so zero active admins
means nobody anywhere can create an admin from a browser again, and recovery is a shell on
the server -- i.e. the vendor. Needing the supplier to unlock your own school is the
relationship this rewrite exists to end, so it is not something a UI should be able to do
by accident.

**And one thing this module deliberately does not offer: delete.** `Event.reviewed_by_id`
is `ON DELETE SET NULL`, so removing a row would blank the reviewer on every bullying event
that person ever ruled on -- a judgement about a named child, silently unattributed.
Accounts are retired (`is_active=False`) and can be brought back.

`is_active` here answers exactly one question -- "may this account log in" -- and it is
checked by `security.load_user` on every request, so retiring somebody ends the session
they already have. It is not also used to mean "left the school"; that conflation on
`persons.is_active` cost this project a migration (0005) and, before it, face recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, null, select
from sqlalchemy.orm import Session

from qorgan.db.engine import session_scope
from qorgan.db.models import School, UndecidedSchool, User
from qorgan.db.tenancy import resolve_school_id
from qorgan.enums import UserRole
from qorgan.logging_setup import get_logger
from qorgan.passwords import hash_password
from qorgan.settings import get_settings

logger = get_logger(__name__)

PAGE_SIZE = 25

# What `users.username` actually holds: String(64). SQLite does not enforce VARCHAR length
# and PostgreSQL does, so a 65-character name accepted here would be stored whole on the
# school's install and truncated on the next one -- the same person, two different rows,
# depending on which database is underneath. Checked where the value arrives instead.
USERNAME_MAX_LENGTH = 64


class AccountError(ValueError):
    """Refused for a reason the person who hit it can act on.

    The message is rendered into the page and written to the log, so it names the rule and
    never the password. A bare "Forbidden" on a form somebody filled in honestly reads as
    "the system is broken", which is how a real protection gets switched off by whoever is
    asked to fix it (the same reasoning as `csrf._refused`).
    """


class UsernameRejected(AccountError):
    pass


class UsernameTaken(AccountError):
    pass


class RoleRejected(AccountError):
    pass


class UnknownAccount(AccountError):
    pass


class NotYourOwnAccount(AccountError):
    pass


class LastActiveAdmin(AccountError):
    pass


class SchoolUndecided(AccountError):
    """Several schools exist and this call did not say which one the account is for.

    **An `AccountError`, and that is the whole point of the class.** The underlying
    `db.models.school.UndecidedSchool` is a `RuntimeError`, so neither front door caught
    it: `qorgan user add <name> --role admin` on a two-school installation handed the
    installer a stack trace -- an ordinary day-one action failing on exactly the
    configuration this branch exists to enable. Measured before the change: `--role admin`
    and `--role operator` both escaped uncaught, while `--role superadmin` returned 0,
    because that row resolves no school at all.

    Raised HERE rather than caught in the CLI so that both doors get it. That a caller must
    say which school is a rule, and rules live in this module; which flag a shell types to
    satisfy it is that door's own business and is added there.
    """


@dataclass(frozen=True, slots=True)
class AccountRow:
    """One row of the list, already formatted.

    Dates arrive as strings because the template decides nothing: business logic in a
    template is invisible to every test in this suite.
    """

    id: int
    username: str
    role: str
    is_active: bool
    last_login: str
    created: str


# The roles a form may set. **`SUPERADMIN` is not one of them, and that is the whole
# reason this names a set instead of using `UserRole`.** The page behind it is gated on
# MANAGE_USERS, which a school ADMIN holds -- so if `parse_role` accepted every member of
# the enum, one school headteacher could POST `role=superadmin` and mint an account that
# reaches EVERY school on the installation. The role that manages schools is created by
# the installation itself, at a shell -- `qorgan user add <name> --role superadmin`, which
# reaches `create_superadmin` below and never this set -- and never through a form served
# to a tenant. `PSYCHOLOGIST` IS in this set: §13's role is a member of one school's staff,
# minted by that school's own headteacher like every other account, and the sentence here
# that used to read "there is still deliberately no psychologist role" stopped being true
# on 2026-07-28 when §13's pages landed. Deriving the tuple from `UserRole` rather than
# listing it by hand is what kept that correct without anyone editing this line.
#
# **THE FLAG THIS COMMENT USED TO NAME (`--superadmin`) EXISTED NOWHERE IN THE TREE.** For
# as long as it did not, nothing could create a SUPERADMIN at all: the CLI went through
# `parse_role` like the form, so `qorgan user add alice --role superadmin` was offered by
# `--help`, accepted by argparse, and then refused as "unknown role" -- about a role that
# is in the enum and in the help text. `/schools` was therefore reachable by nobody,
# permanently, and `Capability.MANAGE_SCHOOLS` guarded nothing. The suite stayed green
# because every superadmin test minted its user as a database row and no test anywhere
# called `parse_role`. `tests/test_the_superadmin_can_be_created.py` walks the command.
ASSIGNABLE_ROLES: tuple[UserRole, ...] = tuple(
    role for role in UserRole if role is not UserRole.SUPERADMIN
)


def parse_role(raw: str) -> UserRole:
    """A role arriving off a form is a string somebody typed.

    **This said "no psychologist role and no superadmin" until 2026-07-28, and it was
    correct while it said so.** `roles.py` refuses a permission that guards nothing, so
    §13's role could not exist before §13's pages did. The pages arrived; the role arrived
    with them, in the same change, which is the order that keeps the rule true. It is
    written down here rather than quietly dropped because the same rule then admitted the
    second role for the same reason: the old sentence made the superadmin conditional on
    "the multi-school routes existing to be guarded", and `/schools` now exists, so
    `SUPERADMIN` is in `UserRole` too.

    **Both roles are now real, and they arrive here by different doors.** `PSYCHOLOGIST`
    is assignable from the form like any other role. `SUPERADMIN` exists in the enum and
    is unreachable from here: see `ASSIGNABLE_ROLES`.
    """
    try:
        parsed = UserRole(raw)
        if parsed not in ASSIGNABLE_ROLES:
            raise ValueError(raw)
        return parsed
    except ValueError:
        # The rejected string is NOT quoted back. It is attacker-controlled text on its way
        # to an error banner, and while Jinja escapes it, the safe habit is not to carry it.
        raise RoleRejected("unknown role") from None


def list_accounts(page: int, *, school_id: int | None = None) -> tuple[list[AccountRow], int]:
    """One page of accounts, and how many there are.

    Paginated because the legacy loaded whole tables on every render, per client, every
    2.5 seconds (audit M-19) -- fine at 400 rows and fatal at 40 000. A school's staff list
    is small today; the query that assumes so is the one nobody revisits.
    """
    offset = (max(1, page) - 1) * PAGE_SIZE
    with session_scope() as session:
        # One school's staff, never the installation's. The суперадминистратор has
        # `school_id IS NULL` and so appears on nobody's list -- correct: they are not a
        # member of any school's staff, and a headteacher able to see the one account that
        # reaches every school would also be able to try to retire it.
        school = resolve_school_id(session, school_id)
        mine = User.school_id == school
        total = int(session.scalar(select(func.count(User.id)).where(mine)) or 0)
        rows = session.scalars(
            select(User).where(mine).order_by(User.username).limit(PAGE_SIZE).offset(offset)
        ).all()
        return [_row(user) for user in rows], total


def create_account(
    username: str, password: str, role: UserRole, *, school_id: int | None = None
) -> int:
    """A new login. The privilege-granting act, and the reason `/users` is ADMIN-only.

    **`school_id` is IGNORED when `role` is SUPERADMIN.** That row belongs to no school by
    definition, so there is nothing for a caller to name and nothing sensible to do with a
    value if one arrives. Written down rather than left to be discovered: a parameter that
    one role silently discards is the sort of quiet disagreement between layers this
    codebase keeps paying for. No caller does it today -- `create_superadmin` passes
    nothing, and the web route reaches here only through `parse_role`, which refuses that
    role outright.
    """
    name = _clean_username(username)
    # Before the uniqueness check on purpose: hashing is where the password rules live, and
    # a short password must be refused whether or not the name happens to be free.
    hashed = hash_password(password)

    with session_scope() as session:
        if session.scalar(select(User.id).where(User.username == name)) is not None:
            # Checked rather than left to the UNIQUE index: the alternative is an
            # IntegrityError reaching a headteacher as a 500 on a form they filled in
            # honestly.
            raise UsernameTaken("that username is already taken")

        user = User(
            username=name,
            password_hash=hashed,
            role=role,
            school_id=_school_for(session, role, school_id),
        )
        session.add(user)
        session.flush()
        # No password, and no hash either. A log line is the one artefact that outlives
        # the request, gets copied into a ticket, and is read by whoever is on call.
        logger.info("account created", extra={"username": name, "role": role.value})
        return user.id


def _school_for(session: Session, role: UserRole, school_id: int | None) -> Any:
    """Which school a new account belongs to -- and SQL NULL for the one that belongs to none.

    **NULL for the SUPERADMIN and for no other role.** `db/models/auth.py` has documented
    that since it was written and nothing enforced it. Resolving a school for this row
    would put the one account that reaches every school onto one school's staff list
    (`list_accounts`), where that school's headteacher can see it and try to retire it, and
    would give `web.security.school_of` a school to hand back instead of the refusal it is
    written to make. Enforced here, at the single writer, rather than at each caller: a
    caller that forgets produces a row nothing downstream can tell from a correct one.

    **`null()` AND NOT `None`, AND THAT IS NOT A STYLE CHOICE.** `school_key` carries a
    column default (`_default_school_id`), and SQLAlchemy omits a column from the INSERT
    when its mapped value IS `None` -- so the Python default fires and `school_id=None` is
    silently replaced by the sole school's id. Measured, not assumed:

        None -> 1        null() -> None        omitted -> 1
        None, two schools -> raises UndecidedSchool

    A SQL NULL is a value, so it is emitted and no default fires. Anything writing
    `school_id=None` and expecting NULL is getting a school, and has no way to notice.
    """
    if role is UserRole.SUPERADMIN:
        return null()
    try:
        return resolve_school_id(session, school_id)
    except UndecidedSchool as exc:
        # Translated, not propagated. `UndecidedSchool` is a RuntimeError and neither door
        # catches those, so this reached the installer as a traceback -- see
        # `SchoolUndecided`. The slugs are IN the message because "name a school" is not
        # actionable without knowing what the names are, and the person reading this is at
        # a shell on a machine whose schools they may not have created.
        known = ", ".join(sorted(session.scalars(select(School.slug)).all()))
        raise SchoolUndecided(
            "this installation has more than one school, so nothing here can choose which "
            f"one this account belongs to. The schools are: {known}"
        ) from exc


def create_superadmin(username: str, password: str) -> int:
    """The installation's own account: created at a shell, never from a form.

    **This is the door `qorgan user add --role superadmin` goes through, and the reason
    `/schools` is reachable at all.** `Capability.MANAGE_SCHOOLS` is held by SUPERADMIN
    alone, so for as long as no path could create one, the widest grant on the
    installation guarded a page nobody could open. See the note on `ASSIGNABLE_ROLES`.

    **It is a separate function rather than a wider `parse_role`, and that is the whole
    security property.** The accounts page is gated on `MANAGE_USERS`, which a school's
    own ADMIN holds -- so a form able to reach this would let one headteacher mint an
    account that reads every school on the installation. `qorgan.web` imports
    `create_account` and `parse_role`; it must not import this, and
    `test_the_web_package_cannot_reach_the_installations_own_door` is what says so rather
    than this paragraph.

    It takes no `school_id` because there is none to take: this row resolves no school at
    all, which is what `test_the_account_the_command_wrote_belongs_to_no_school` measures.
    """
    return create_account(username, password, UserRole.SUPERADMIN)


def set_role(
    user_id: int, role: UserRole, *, acting_user_id: int, school_id: int | None = None
) -> None:
    """Move somebody between roles. Promotion and demotion are the same operation."""
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        user = _load(session, user_id, school)
        _refuse_if_it_closes_the_last_door(
            session, user, school, still_an_admin=(role is UserRole.ADMIN)
        )
        _refuse_if_it_is_your_own(user, acting_user_id)

        was = user.role
        user.role = role
        logger.info(
            "account role changed",
            extra={"username": user.username, "from": was.value, "to": role.value},
        )


def set_active(
    user_id: int, active: bool, *, acting_user_id: int, school_id: int | None = None
) -> None:
    """Retire an account, or bring one back. There is no delete; see the module docstring."""
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        user = _load(session, user_id, school)
        _refuse_if_it_closes_the_last_door(session, user, school, still_an_admin=active)
        _refuse_if_it_is_your_own(user, acting_user_id)

        user.is_active = active
        logger.info("account active changed", extra={"username": user.username, "active": active})


def _refuse_if_it_closes_the_last_door(
    session: Session, user: User, school_id: int, *, still_an_admin: bool
) -> None:
    """At least one ADMIN must still be able to log in after this write.

    **Checked FIRST, before the "not your own account" rule, because both fire on a school
    with a single admin retiring itself and the order decides what that headteacher reads.**
    "You would be the last admin" is actionable -- promote somebody, then try again. "You
    cannot do this to yourself" sends them to a colleague who does not exist.

    **ACTIVE admins, not ADMIN rows.** An admin who cannot log in is not cover for anything,
    and counting rows would see two where there is one door. That is this project's
    signature defect -- a value true in one layer and quietly wrong in the next -- and it
    has now been paid for three times (`newly_bound`, `persons.is_active`, and the merge
    column that fixed it).

    Inside the writing transaction, so that two admins retiring each other at the same
    moment cannot both pass a check made before either of them wrote.
    """
    if user.role is not UserRole.ADMIN or not user.is_active:
        return  # not currently a way in; this change cannot remove the last one
    if still_an_admin:
        return  # they come out of this an active admin, so nothing is lost
    if _active_admins(session, school_id) <= 1:
        raise LastActiveAdmin(
            "this is the only administrator who can still log in — give somebody else the "
            "administrator role first, or nobody will be able to manage accounts again"
        )


def _refuse_if_it_is_your_own(user: User, acting_user_id: int) -> None:
    """Nobody edits their own account here.

    `load_user` refuses an inactive account on every request, so retiring yourself lands on
    your very NEXT one: the browser bounces to /login and the page that could undo it needs
    the account you just shut. Demoting yourself is the same thing spelled differently --
    the capability that could put it back is the one you gave away.

    There is no self-service case to lose. Somebody leaving the school is retired by a
    colleague, which is also the person who should be deciding it.
    """
    if user.id == acting_user_id:
        raise NotYourOwnAccount(
            "you cannot change your own account here — ask another administrator"
        )


def _active_admins(session: Session, school_id: int) -> int:
    """**Admins of THIS school.** Another school's administrator is not a door into this
    one, so counting them would let the last real door here be closed."""
    return int(
        session.scalar(
            select(func.count(User.id)).where(
                User.school_id == school_id,
                User.role == UserRole.ADMIN,
                User.is_active.is_(True),
            )
        )
        or 0
    )


def _load(session: Session, user_id: int, school_id: int) -> User:
    """The account, if it is this school's. A `session.get` here would not have asked.

    The id comes off a URL (`/users/{id}/role`). Fetched by primary key alone, one
    school's administrator could demote or retire another school's -- including their last
    active admin, locking a school nobody here has heard of out of its own system. "No such
    account" is also the right answer: whether that id exists elsewhere on the installation
    is not this school's business, and a distinguishable 404 is a directory of the others.
    """
    user = session.scalar(select(User).where(User.id == user_id, User.school_id == school_id))
    if user is None:
        raise UnknownAccount("no such account")
    return user


def _clean_username(raw: str) -> str:
    """Trimmed, then checked. `" head"` and `"head"` are the same person to everybody
    except a UNIQUE index, and letting both exist is how one member of staff becomes two
    rows in the review history of the same child's incident."""
    name = raw.strip()
    if not name:
        raise UsernameRejected("a username is required")
    if len(name) > USERNAME_MAX_LENGTH:
        raise UsernameRejected(f"a username may be at most {USERNAME_MAX_LENGTH} characters")
    return name


def _row(user: User) -> AccountRow:
    return AccountRow(
        id=user.id,
        username=user.username,
        role=user.role.value,
        is_active=user.is_active,
        last_login=_when(user.last_login_at),
        created=_when(user.created_at),
    )


def _when(value: datetime | None) -> str:
    """A stored instant, on the clock the reader is actually looking at.

    Everything goes into the database as tz-aware UTC (`db.types.UtcDateTime` refuses
    anything else), and the school is in Asia/Almaty -- five hours away. Rendering the
    stored value straight would be the signature mistake in its purest form: right in the
    column, five hours wrong on the screen, and wrong in the direction that makes last
    night's login look like this morning's. The value arrives tz-aware precisely so that
    this is a conversion rather than a guess.

    NOTE: `web.routes.events` still formats `occurred_at` without this conversion, so
    incident times on that page are shown in UTC. Not changed from here -- it is a
    different page with its own tests -- but the two do not currently agree.
    """
    if value is None:
        return "—"
    return value.astimezone(get_settings().tz).strftime("%Y-%m-%d %H:%M")

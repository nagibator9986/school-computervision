"""The register of schools on this installation. §14's "управление школами".

**Numbers about schools, never data about children.** Every row this module produces is a
count -- how many pupils, cameras and accounts a school has -- and nothing here returns a
name, a photograph, an incident or a meal. That is the boundary the SUPERADMIN role is
drawn on (`qorgan.roles`): the person who administers twenty schools' machines is not
thereby a person twenty schools have entrusted with their children.

The rules live here rather than in the route for the same reason `qorgan.accounts` does: a
rule written in a route is a rule that only that route has. **Said straight, because the
sentence here used to say the opposite:** there is no `qorgan school add` command, and
`create_school` below is the only thing in the system that adds one -- reached from the
/schools page and from nowhere else. The single exception is the default row migration 0009
writes, which `create_school`'s own docstring records; it is not a second front door, it is
the installation's first school arriving before this table did. The old wording called that
absent command "a second front door onto the same table" in the present tense, which made
this module's own docstring the third phantom CLI claim found on this branch --
`tests/test_the_prose_and_the_parser_agree.py` is what now says no. Building the command is
a product decision and not a repair, so it was not taken here; the rules stay in this module
so that the day it IS taken, the second door inherits them instead of restating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import func, select

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Person, School, User
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


class SchoolError(Exception):
    """Something the caller can fix, and must be told about in words."""


class SlugRejected(SchoolError):
    pass


class SlugTaken(SchoolError):
    pass


class UnknownSchool(SchoolError):
    pass


@dataclass(frozen=True)
class SchoolRow:
    """One school, as the register shows it. Counts only -- see the module docstring."""

    id: int
    slug: str
    name: str
    is_active: bool
    pupils: int
    cameras: int
    accounts: int


def list_schools() -> tuple[SchoolRow, ...]:
    """Every school, with how much of the installation each one is using."""
    pupils = (
        select(func.count(Person.id))
        .where(Person.school_id == School.id, Person.is_active.is_(True))
        .scalar_subquery()
    )
    cameras = (
        select(func.count(Camera.id)).where(Camera.school_id == School.id).scalar_subquery()
    )
    accounts = (
        select(func.count(User.id))
        .where(User.school_id == School.id, User.is_active.is_(True))
        .scalar_subquery()
    )

    with session_scope() as session:
        rows = session.execute(
            select(School.id, School.slug, School.name, School.is_active, pupils, cameras, accounts)
            .order_by(School.slug)
        ).all()

    return tuple(SchoolRow(*row) for row in rows)


def school_id_for_slug(slug: str) -> int:
    """The school with this short name, for a caller who has typed one.

    **The slug is the part a person can hold in their head; the id is what the tables
    carry.** `rename_school` below refuses to edit a slug for exactly this reason -- so
    this lookup, a log line and a future URL cannot be silently re-pointed by somebody
    correcting the sign on the building.

    `db/models/school.py` named the `--school` flag on `School.slug` before it was built,
    and `rename_school` names it too. This is that flag's half of the bargain, and
    until it existed `qorgan user add --role admin` had no way to say which school it
    meant -- so on a two-school installation it handed the installer an uncaught
    `UndecidedSchool` traceback. Measured on the tree before this change: not rc 1, a
    stack. The refusal that replaces it lives in `qorgan.accounts`, where both front doors
    meet it.
    """
    clean = slug.strip().lower()
    with session_scope() as session:
        found = session.scalar(select(School.id).where(School.slug == clean))
        if found is None:
            known = ", ".join(sorted(session.scalars(select(School.slug)).all()))
            raise UnknownSchool(
                f"no school on this installation has the short name {clean!r}. "
                f"The schools are: {known}"
            )
        return int(found)


def create_school(slug: str, name: str) -> int:
    """Add a school. **Nothing else in the system creates one, and that is deliberate.**

    A school is a claim about the world -- an institution with children in it -- so it is
    made by a person who typed it, never inferred from a camera appearing in a config file
    or a roster arriving in a folder. The default row that migration 0009 writes is the one
    exception, and it exists because the installation already served a school before this
    table did.
    """
    clean_slug = slug.strip().lower()
    clean_name = " ".join(name.split())
    if not SLUG.match(clean_slug):
        raise SlugRejected(
            "the short name must be 3-64 characters of lowercase latin letters, digits "
            "and hyphens — it is what a command line and a URL will carry"
        )
    if not clean_name:
        raise SlugRejected("the school needs a name somebody would recognise")

    with session_scope() as session:
        if session.scalar(select(School.id).where(School.slug == clean_slug)) is not None:
            raise SlugTaken(f"a school with the short name {clean_slug!r} already exists")
        school = School(slug=clean_slug, name=clean_name)
        session.add(school)
        session.flush()
        logger.info("school created", extra={"slug": clean_slug})
        return school.id


def rename_school(school_id: int, name: str) -> None:
    """The display name, and only that.

    The slug is NOT editable here. It is what a `--school` flag, a log line and a future
    URL carry, so changing it would silently re-point every one of them; renaming the sign
    on the building is a different act from renaming the building in every record.
    """
    clean = " ".join(name.split())
    if not clean:
        raise SlugRejected("the school needs a name somebody would recognise")

    with session_scope() as session:
        school = session.get(School, school_id)
        if school is None:
            raise UnknownSchool("no such school")
        school.name = clean
        logger.info("school renamed", extra={"slug": school.slug})

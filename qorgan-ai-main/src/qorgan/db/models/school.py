"""The tenant boundary: one row per school, and the key every root table points at.

`REWRITE_SPEC.md` §1 says one school, "but the architecture must not obstruct the move to
many". This is that move, and it is deliberately small: a register of schools, and a
`school_id` on the four tables nothing else in the schema can answer for. Which four, and
why those four, is argued in `tests/test_tenancy_guard.py` -- next to the guard that
enforces it, rather than here where nothing checks it.

**THE DEFAULT ON `school_id` IS THE SUBTLE PART, SO READ IT BEFORE COPYING IT.**

A row that does not name a school is filed under the only school there is. On a
single-school installation -- which is every installation today -- that is the truth, and
it is what lets `Camera(name="hall_left")` keep meaning what it has always meant in this
codebase and in eighty places in the suite.

The moment a second school exists, `sole_school_id` stops guessing and RAISES. That
direction matters more than the convenience: the failure mode of a default is that it
picks one, and a default that picks one out of two would file a school's camera under
somebody else's, quietly, on the day the installation grew. This one refuses instead, by
name, at the insert that could not decide.

It is also worth being clear about what the default can and cannot cost. It applies to
INSERTs only. A misfiled insert is visible -- the camera appears on the wrong school's
wall and someone says so. An unfiltered SELECT is not visible at all, which is why the
guard is aimed there and this is allowed to be convenient.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, ForeignKey, String, select
from sqlalchemy.orm import Mapped, mapped_column

from qorgan.db.models.base import Base, TimestampMixin


class UndecidedSchool(RuntimeError):
    """A row had to name a school and could not. Never silently resolved."""


class School(Base, TimestampMixin):
    """One school. Rows of this table are the tenants; they are not a school's data."""

    __tablename__ = "schools"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stable, human-typable, and what the `--school` flag takes. This line named that flag
    # before anything implemented it, and then went on calling it forthcoming for a commit
    # after it shipped -- which is why the tense here is now checked rather than trusted
    # (`tests/test_the_prose_and_the_parser_agree.py`). Globally unique because a school
    # register with two `gymnasium-4`s is a register nobody can address.
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # A school that has left. Its rows stay -- they are its children's records, and the
    # foreign keys below are RESTRICT rather than CASCADE for exactly that reason.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


def sole_school_id(connection: Any) -> int:
    """The only school in this database. Raises, loudly, if that is not what it is."""
    ids = connection.execute(select(School.id).order_by(School.id).limit(2)).scalars().all()
    if len(ids) == 1:
        return int(ids[0])
    if not ids:
        raise UndecidedSchool(
            "this row must belong to a school and there are none in the database. A "
            "school is created by migration 0009 (for an existing installation) or on "
            "the superadmin's page; nothing invents one, because a school is a claim "
            "about the world and not a default."
        )
    raise UndecidedSchool(
        f"this row must belong to a school and there are {len(ids)} of them, so nothing "
        "here can choose. Name the school at the call site: pass `school_id=`. This is "
        "the check that stops a second school's arrival from quietly refiling rows."
    )


def _default_school_id(context: Any) -> int:
    return sole_school_id(context.connection)


def school_key(**kwargs: Any) -> Any:
    """The `school_id` column, written once so the four root tables cannot diverge.

    RESTRICT, not CASCADE. Deleting a school must not be a way to delete every
    photograph, meal and incident belonging to its children in one statement that
    reports success -- this repository has already had one migration that did exactly
    that shape of thing (`migrations/env.py::_suspend_foreign_keys`).
    """
    return mapped_column(
        ForeignKey("schools.id", ondelete="RESTRICT"),
        index=True,
        default=_default_school_id,
        **kwargs,
    )

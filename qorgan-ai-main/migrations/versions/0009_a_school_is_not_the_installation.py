"""a school is not the installation

`schools`, and `school_id` on the four tables nothing else in the schema can answer for:
`cameras`, `persons`, `users`, `meal_windows`. Every other table reaches a school through
a foreign key that cannot be NULL, and `tests/test_tenancy_guard.py` argues which is which
next to the guard that enforces it.

**THE UNIQUE CONSTRAINTS ARE THE HALF THAT BREAKS QUIETLY.** `persons.external_id` was
globally unique and `cameras.name` was too. Under one school that is correct and invisible.
Under two it is wrong in a way that produces no error message anybody can act on: the
second school imports its roster and half of it is rejected as duplicates of children it
has never met, because the first school also has a pupil numbered 7. Both become composite
-- unique WITHIN a school -- and that is the point of this migration as much as the column
is.

`users.username` stays globally unique, deliberately. A login form carries no school, so
the lookup behind it cannot be scoped by one; see `db/models/auth.py`.

**A DEFAULT SCHOOL ROW IS CREATED AND EVERY EXISTING ROW IS GIVEN TO IT.** This
installation serves one school today, so that is a statement of fact rather than a guess
-- the guess would be leaving the rows unowned and letting something later decide. It is
named `Школа (переименуйте на странице суперадминистратора)` on purpose: it is the one
thing in this migration a human must correct, and the page that lets them exists in the
same change.

**WHY THIS MIGRATION IS THE DANGEROUS SHAPE, AND WHAT PROTECTS IT.** SQLite cannot ALTER a
column, so `render_as_batch` rebuilds each table: copy, move rows, **DROP the original**,
rename. `events`, `canteen_sessions` and `recognition_attempts` reference `cameras` with
`ondelete="CASCADE"`, so that DROP is a delete of every camera as far as the foreign keys
are concerned -- and it takes the school's entire incident history with it while the
migration reports success. This migration rebuilds `cameras` THREE times.

`migrations/env.py::_suspend_foreign_keys` is what stops it, on a fresh connection before
any transaction is open, because inside a transaction `PRAGMA foreign_keys` is a documented
no-op. That protection is not assumed here: `tests/test_migration_keeps_the_events.py`
runs this migration over two identically populated databases with the guard on and off and
measures that the event survives only where it is on.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-28 21:10:00.000000

RENUMBERED 0008 -> 0009 when this branch was merged. `feat/psychologist-cabinet` and
`feat/multi-school` were written in parallel and BOTH numbered their migration 0008 with
`down_revision = '0007'`. Git merges that silently -- two differently-named files, no
textual conflict, a green suite -- and the damage only appears at `alembic upgrade`, where
the two become sibling heads and ONE OF THEM SIMPLY NEVER RUNS on a real installation.
This project has had that exact defect once before with two 0006s, where `alembic heads`
answered "0006 (head)" twice behind nothing louder than a UserWarning.

The referral migration kept 0008 because it was merged first; this one follows it.
Nothing about the order is meaningful -- they touch different tables -- and the only thing
that matters is that the chain is linear and `alembic heads` names exactly one revision.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_SLUG = "default"
DEFAULT_NAME = "Школа (переименуйте на странице суперадминистратора)"
ROOT_TABLES = ("cameras", "persons", "users", "meal_windows")


def _create_default_school() -> int:
    """The school this installation already serves, written down for the first time."""
    now = datetime.now(UTC).replace(tzinfo=None)
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "INSERT INTO schools (slug, name, is_active, created_at, updated_at) "
            "VALUES (:slug, :name, 1, :now, :now)"
        ),
        {"slug": DEFAULT_SLUG, "name": DEFAULT_NAME, "now": now},
    )
    school_id = connection.execute(
        sa.text("SELECT id FROM schools WHERE slug = :slug"), {"slug": DEFAULT_SLUG}
    ).scalar_one()
    return int(school_id)


def _own_the_existing_rows(table: str, school_id: int) -> None:
    """Add the column nullable, fill it, and only then refuse NULL.

    Three steps rather than one because `NOT NULL` on an ADD COLUMN needs a value for
    every row that already exists, and the honest value is "the school that has been
    here all along" -- not a server-side default left behind for the next writer to
    inherit silently.
    """
    with op.batch_alter_table(table) as batch:
        batch.add_column(sa.Column("school_id", sa.Integer(), nullable=True))

    op.get_bind().execute(
        sa.text(f"UPDATE {table} SET school_id = :school_id"), {"school_id": school_id}
    )

    with op.batch_alter_table(table) as batch:
        # `users.school_id` stays nullable: NULL there means the суперадминистратор, who
        # manages the schools and so belongs to none of them (db/models/auth.py).
        if table != "users":
            batch.alter_column("school_id", existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key(
            op.f(f"fk_{table}_school_id_schools"),
            "schools",
            ["school_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_index(op.f(f"ix_{table}_school_id"), ["school_id"])


def upgrade() -> None:
    op.create_table(
        "schools",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_schools")),
        sa.UniqueConstraint("slug", name=op.f("uq_schools_slug")),
    )

    school_id = _create_default_school()
    for table in ROOT_TABLES:
        _own_the_existing_rows(table, school_id)

    # The half that breaks quietly. See the module docstring.
    with op.batch_alter_table("persons") as batch:
        batch.drop_constraint("uq_persons_external_id", type_="unique")
        batch.create_unique_constraint(
            "uq_persons_school_external_id", ["school_id", "external_id"]
        )
    with op.batch_alter_table("cameras") as batch:
        batch.drop_constraint("uq_cameras_name", type_="unique")
        batch.create_unique_constraint("uq_cameras_school_name", ["school_id", "name"])


def downgrade() -> None:
    """Reversible only while there is one school. It refuses rather than choosing one.

    Going back means restoring a GLOBAL unique on `persons.external_id`. With two schools
    in the database that constraint cannot hold -- both may have a pupil numbered 7 -- so
    the downgrade would either fail on an index build halfway through or, worse, be run
    against a database where somebody had already deleted "the duplicates" to make it
    pass. Refusing by name is the only answer that cannot lose a child's record.
    """
    schools = op.get_bind().execute(sa.text("SELECT COUNT(*) FROM schools")).scalar_one()
    if schools > 1:
        raise RuntimeError(
            f"refusing to downgrade: {schools} schools exist, and this downgrade restores "
            "a GLOBAL unique on persons.external_id, which two schools' rosters can "
            "legitimately violate. Move the other schools out first."
        )

    with op.batch_alter_table("cameras") as batch:
        batch.drop_constraint("uq_cameras_school_name", type_="unique")
        batch.create_unique_constraint("uq_cameras_name", ["name"])
    with op.batch_alter_table("persons") as batch:
        batch.drop_constraint("uq_persons_school_external_id", type_="unique")
        batch.create_unique_constraint("uq_persons_external_id", ["external_id"])

    for table in ROOT_TABLES:
        with op.batch_alter_table(table) as batch:
            batch.drop_index(op.f(f"ix_{table}_school_id"))
            batch.drop_constraint(op.f(f"fk_{table}_school_id_schools"), type_="foreignkey")
            batch.drop_column("school_id")
    op.drop_table("schools")

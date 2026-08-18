"""inactive answered two questions

`persons.merged_into_id` — which id this one was merged into, when a merge is what retired
it.

**This column exists because a boolean was answering "no" to two different questions.**
`is_active=False` meant BOTH "this id was merged into another one" and "this person left
the school", and those need opposite treatment. `qorgan pupils merge --reactivate` exists
to undo a merge that went the wrong way — across the pupil/staff line a merge decides
whether a child appears in the meal record at all — and with nothing recording the
difference, the flag had to take the operator's word for which case it was. On that
operation, being wrong either revives a pupil the school expelled or refuses a child their
meal history back.

**It is the third one of these in this project, and the first cost it face recognition.**
`newly_bound=False` meant both "this track is already bound to a person" and "nobody was
recognised", so the canteen could not tell a pupil it already knew from a stranger. That is
not a curiosity; it is the shape to watch for. A boolean is a single bit and a bit is one
question. When "false" has two causes and the caller must act differently on each, the
answer is a column, not a comment.

Existing rows get NULL, and NULL means **"not retired by a merge, or retired before this
column existed"** — an unknown, never an assertion that somebody left. Nothing is
back-filled: which merges happened before today is not recoverable from the rows, and
inventing it is precisely what the legacy's `apply_runtime_migrations()` did to
`person_type` on every boot, silently reverting the school's own corrections (audit H-02).
So a person retired before this migration cannot be un-merged by the flag, and is refused
with a message that says why rather than guessing.

`ondelete="SET NULL"`: deleting the survivor must not cascade into deleting the people who
were merged into them. The pointer is a note about history, not ownership.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-24 16:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch_alter_table: SQLite cannot ADD a column with a foreign key in place, so
    # Alembic rebuilds the table. Named constraint, or the rebuild has nothing to call it.
    with op.batch_alter_table("persons") as batch:
        batch.add_column(sa.Column("merged_into_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_persons_merged_into_id",
            "persons",
            ["merged_into_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_persons_merged_into_id", "persons", ["merged_into_id"])


def downgrade() -> None:
    op.drop_index("ix_persons_merged_into_id", table_name="persons")
    with op.batch_alter_table("persons") as batch:
        batch.drop_constraint("fk_persons_merged_into_id", type_="foreignkey")
        batch.drop_column("merged_into_id")

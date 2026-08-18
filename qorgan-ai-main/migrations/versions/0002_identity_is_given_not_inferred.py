"""identity is given, not inferred

`persons.full_name` becomes nullable: the school sent ids, not names, and the name is a
display field we do not have yet.

`ExternalIdSource.GENERATED` is gone: nothing derives an identity from a name any more, so
the column narrows. Rows that carry it were created by the deleted name-based import --
their `external_id` is a hash of a filename, which is exactly the invented identity the
spec forbids. They are deleted rather than relabelled, because calling a guess a roster
entry would be a lie in the database. `qorgan pupils import-roster` recreates all 142
correctly, from the school's own ids.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-13 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A person whose identity was invented from a filename is precisely what §1.2 forbids.
    # photos and embeddings cascade; canteen_sessions.person_id is SET NULL.
    op.execute(sa.text("DELETE FROM persons WHERE external_id_source = 'GENERATED'"))

    with op.batch_alter_table('persons', schema=None) as batch_op:
        batch_op.alter_column(
            'full_name',
            existing_type=sa.String(length=255),
            nullable=True,
        )
        batch_op.alter_column(
            'external_id_source',
            existing_type=sa.Enum(
                'GENERATED', 'ROSTER', name='externalidsource', native_enum=False
            ),
            type_=sa.Enum('ROSTER', name='externalidsource', native_enum=False),
            existing_nullable=False,
        )


def downgrade() -> None:
    # Going back needs a name in every row, and there are none. The id is what we have.
    op.execute(
        sa.text(
            "UPDATE persons SET full_name = 'Ученик ' || external_id "
            "WHERE full_name IS NULL"
        )
    )

    with op.batch_alter_table('persons', schema=None) as batch_op:
        batch_op.alter_column(
            'external_id_source',
            existing_type=sa.Enum('ROSTER', name='externalidsource', native_enum=False),
            type_=sa.Enum(
                'GENERATED', 'ROSTER', name='externalidsource', native_enum=False
            ),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'full_name',
            existing_type=sa.String(length=255),
            nullable=False,
        )

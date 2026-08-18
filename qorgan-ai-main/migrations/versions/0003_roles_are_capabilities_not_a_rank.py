"""roles are capabilities, not a rank

`users.role` gains CANTEEN_STAFF. The column is a VARCHAR with a CHECK constraint
(native_enum=False), and the old constraint names three roles, so a canteen worker
could not be inserted into an existing database at all until it is widened.

This is the schema half of the school's §14: canteen staff work "БЕЗ доступа к
буллингу". The old role model was a strict total order -- operator implied admin
implied developer -- and every gated route required OPERATOR, so the only way to let a
canteen worker open the canteen journal was to make them an operator, which also opened
the bullying event log, the review controls, the corridor previews, and /media.
`qorgan.roles` is the other half: what each role may do, as a set.

Existing rows are untouched and stay valid. 'operator', 'admin' and 'developer' keep
exactly the access they had -- this migration only widens what a role is ALLOWED to be.
Nothing is re-derived from anything: the legacy re-computed person_type on every boot
from 24 LIKE patterns and silently reverted the school's own corrections (audit M-18).

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-17 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_ROLES = ('OPERATOR', 'ADMIN', 'DEVELOPER')
_NEW_ROLES = ('OPERATOR', 'ADMIN', 'DEVELOPER', 'CANTEEN_STAFF')


def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column(
            'role',
            existing_type=sa.Enum(*_OLD_ROLES, name='userrole', native_enum=False),
            type_=sa.Enum(*_NEW_ROLES, name='userrole', native_enum=False),
            existing_nullable=False,
        )


def downgrade() -> None:
    # A canteen worker has no role to go back to. The old model has no way to say "the
    # canteen and nothing else" -- that is the whole reason 0003 exists -- so every role
    # left to relabel them into (operator, admin, developer) opens the bullying log.
    #
    # Deleted rather than relabelled, for the reason 0002 deleted invented persons: a
    # canteen worker stored as an admin is a lie in the database, and this lie hands the
    # school's dinner staff the bullying event log the moment anyone reactivates the
    # account. `qorgan user add` recreates the account; nothing recreates that mistake.
    # events.reviewed_by_id is ON DELETE SET NULL, and a canteen worker has no review
    # capability, so no verdict can be lost here.
    op.execute(sa.text("DELETE FROM users WHERE role = 'CANTEEN_STAFF'"))

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column(
            'role',
            existing_type=sa.Enum(*_NEW_ROLES, name='userrole', native_enum=False),
            type_=sa.Enum(*_OLD_ROLES, name='userrole', native_enum=False),
            existing_nullable=False,
        )

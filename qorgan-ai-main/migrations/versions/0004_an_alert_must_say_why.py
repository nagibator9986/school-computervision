"""an alert must say why

`events.reasons` — the skeleton's own reasons for an event, packed by
`qorgan.events.reasons`.

This is the schema half of the school's §7: the Telegram message must carry the reasons,
the time and the event id. The reasons existed on the Verdict object all along and went
to the log line and stopped there, because the table had nowhere to put them. So the
alert a teacher read could say it was 93% certain and never say what it had seen — and
"someone went down" and "someone waved their arms" are the same severity, the same
confidence, and not the same message.

Existing rows get '' and stay valid: an event with nothing to say for itself is the
normal state of every event the skeleton could not look at, and of every event written
before this column existed. Nothing is re-derived from anything — the reasons of a past
event are not recoverable and are not guessed at (the legacy re-computed person_type on
every boot from 24 LIKE patterns and silently reverted the school's corrections, audit
M-18).

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-17 15:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default='': the column is NOT NULL and existing rows need a value. It stays
    # on the column afterwards so that any writer which does not know about `reasons`
    # yet inserts an empty string rather than failing.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('reasons', sa.Text(), nullable=False, server_default='')
        )


def downgrade() -> None:
    # The reasons are lost, and that is what a downgrade of this migration means: the
    # column is the only place they are kept. The events themselves survive.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.drop_column('reasons')

"""a silence must also say why

`events.telegram_skip_reason` — why a recorded event was deliberately not sent to
Telegram, as a closed set of tokens (`qorgan.enums.TelegramSkipReason`).

0004 gave an alert that DID arrive the words to explain itself. This is the other half,
and it is the half the school asks about first: *"почему мне не пришло уведомление о
драке?"* Until now nothing anywhere could answer that. The decision not to send left no
row, no column and no field — `notifications` records deliveries that were attempted, so
an event nobody tried to deliver simply is not in it, and an unsent alert was
indistinguishable from a quiet afternoon. We admitted as much to the school in writing
(`docs/client-note-2026-07-17.md` §6).

**Why this column is on `events` and not on `notifications`.** A notification that was
never sent has no row to carry the reason. Putting it there would mean inventing a row for
a delivery that never happened — and "НЕ ДОСТАВЛЕНО" on /notifications would then cover
both "the router ate it" and "the system correctly withheld it", which are the two
different questions the school is trying to tell apart. The reason also belongs to the
same judgement as `confidence`, `skeleton_confirmed` and `reasons`, which are the inputs
it was decided from.

**A VARCHAR over a closed set, not free text.** The legacy assembled this as a sentence at
each call site, so one cause was worded three ways and none could be filtered or counted.

NULL means "not withheld". For rows written before this column existed that is an
unknown, not a claim that they were sent, and nothing is back-filled: whether a past event
raised a Telegram is not recoverable from the row, and re-deriving it from today's
thresholds is precisely the invention that `apply_runtime_migrations()` performed on
`person_type` every boot, silently reverting the school's own corrections (audit H-02).
The thresholds may have been retuned since; the afternoon has not.

**RENUMBERED 0006 -> 0007 WHEN THIS BRANCH WAS INTEGRATED, AND ONLY HERE.** It was
written as 0006/0005, and on `feat/telegram-skip-reason` alone that is correct.
`feat/classroom-lesson-metrics` independently wrote its own 0006 off the same 0005.
Nothing collided textually -- the two files have different names, so git merged both
without a word -- and alembic answered `0006 (head)` twice with only a UserWarning, which
means one of the two migrations would simply never have run on the school's machine.

The lesson migration keeps 0006 because it is the one that rebuilds `cameras`, and
`migrations/env.py::_suspend_foreign_keys` -- the guard that stops that rebuild
cascade-deleting every event -- arrives with it. Everything ordered after it is covered.

The dependency created here belongs to the integration tree and must NOT be pushed back
onto `feat/telegram-skip-reason`: this column has nothing to do with lessons, and pinning
its `down_revision` to 0006 on the branch would break that branch's own merge.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-28 09:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0007'
down_revision: str | None = '0006'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Spelled out rather than imported from qorgan.enums, so that this migration keeps
# describing the schema of ITS revision after the enum grows a member. A migration that
# imports the live enum is a migration that quietly changes what it did last year.
_REASONS = (
    'SKELETON_NOT_RUN',
    'NO_SKELETON_CONFIRMATION',
    'WEAK_EVIDENCE_ONLY',
    'LOW_CONFIDENCE',
    'ALREADY_NOTIFIED',
)


def upgrade() -> None:
    # Nullable, with no server_default: NULL is "not withheld", and every existing row
    # gets it because no better answer exists for them and a guess would be worse.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'telegram_skip_reason',
                sa.Enum(*_REASONS, name='telegramskipreason', native_enum=False),
                nullable=True,
            )
        )


def downgrade() -> None:
    # The reasons are lost, and that is what going back means: this column is the only
    # place a decision NOT to notify is recorded anywhere. The events survive; the
    # question the school asks first stops having an answer again.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.drop_column('telegram_skip_reason')

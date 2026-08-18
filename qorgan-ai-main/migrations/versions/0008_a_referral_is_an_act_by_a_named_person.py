"""a referral is an act by a named person

Three things arrive together, and they have to: `roles.py` refuses a permission that
guards nothing, so the psychologist's role, the pages it opens and the column §9 asks an
operator to write are one change or they are three guesses.

**1. `events.status` IS NOT TOUCHED — and an earlier draft of this revision touched it.**

That draft added `REFERRED_TO_PSYCHOLOGIST` to `EventStatus` and widened this VARCHAR from
14 to 24 to hold it, reading §9's list of marks — «подтверждено; ложное срабатывание;
требует доп. проверки; **передано психологу**; закрыто» — as a description of that enum.
It is not one: two of those five have never been members either. §9 says what an operator
must be able to RECORD; `status` answers only one of those questions, namely whether the
detector was right.

It came out before this revision had run anywhere, because writing a referral into `status`
was measured to destroy the other fact in BOTH directions: the review controls are drawn
only for an unreviewed event, so a referred incident could never afterwards be marked
confirmed or false from a browser — every referred incident fell out of the loop that
corrects the detector — and in the other order a confirmed incident, once referred, stopped
saying it had been confirmed, unrecoverably. A referral is not a verdict, so it gets its own
columns; they are next.

**2. `events.referred_by_id` and `events.referred_at`.** WHO handed the child on, and WHEN.
Not folded into `status`, and not folded into `reviewed_by_id`:

  * `status` is the event's CURRENT state. An operator who refers an incident and then
    confirms it as bullying overwrites the status, and if the referral lived only there,
    confirming would silently un-refer a child. The cabinet therefore selects on
    `referred_at IS NOT NULL`.
  * `reviewed_by_id` is who ruled on whether this was an assault. That is a different
    judgement, frequently by a different person, and reusing the column would have made
    "who decided this was a fight" and "who sent this child to the psychologist" one fact.

Both NULL, no server default, nothing back-filled. NULL means "never referred", and for
rows written before this column existed that is an unknown rather than a claim — the same
rule 0007 applied to `telegram_skip_reason` and for the same reason: re-deriving a past
decision from today's state is what `apply_runtime_migrations()` did to `person_type` on
every boot, silently reverting the school's own corrections (audit H-02).

**3. `psychologist_notes`.** The only table in this schema written entirely by a human, and
the only one no administrator can read. §13: «обычный оператор не должен видеть
конфиденциальные записи психолога». It starts empty, which is the honest state — nobody
wrote a note before there was a page to write one on.

**`users.role` needs no change, and that is measured rather than assumed.** 0003 widened
that column to VARCHAR(13) for `CANTEEN_STAFF`; `PSYCHOLOGIST` is 12 characters, so the
new role fits the column exactly as it stands and `compare_metadata` sees no drift. A
no-op `alter_column` written for symmetry would claim this migration did something it did
not. If a longer role name ever arrives, that is the day this column is widened.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-28 18:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0008'
down_revision: str | None = '0007'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    # `events` is still REBUILT here, even though no column changes type: SQLite cannot add
    # a foreign key with ALTER TABLE, so `create_foreign_key` below forces batch mode to
    # copy -> move rows -> DROP the original -> rename. `notifications` references `events`
    # with ON DELETE CASCADE, so that DROP deletes every notification in the school's
    # database unless `env.py::_suspend_foreign_keys` has turned enforcement off first.
    # Measured on this revision, not inherited from the one that found it: with the guard
    # neutered the notification was gone and alembic still exited 0 and stamped 0008.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.add_column(sa.Column('referred_by_id', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column('referred_at', qorgan.db.types.UtcDateTime(), nullable=True)
        )
        batch_op.create_foreign_key(
            'fk_events_referred_by_id_users',
            'users',
            ['referred_by_id'],
            ['id'],
            ondelete='SET NULL',
        )

    op.create_table(
        'psychologist_notes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.Integer(), nullable=False),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['person_id'],
            ['persons.id'],
            name=op.f('fk_psychologist_notes_person_id_persons'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['author_id'],
            ['users.id'],
            name=op.f('fk_psychologist_notes_author_id_users'),
            ondelete='SET NULL',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_psychologist_notes')),
    )
    op.create_index(
        op.f('ix_psychologist_notes_person_id'), 'psychologist_notes', ['person_id']
    )
    op.create_index(
        'ix_psychologist_notes_person_created',
        'psychologist_notes',
        ['person_id', 'created_at'],
    )


def downgrade() -> None:
    # The notes are DESTROYED, and there is nowhere else they exist. This is the one step
    # of this downgrade that loses something a human wrote rather than something a camera
    # measured; take a backup first (`qorgan backup`, or the /backups page).
    op.drop_index('ix_psychologist_notes_person_created', table_name='psychologist_notes')
    op.drop_index(op.f('ix_psychologist_notes_person_id'), table_name='psychologist_notes')
    op.drop_table('psychologist_notes')

    # **No status is relabelled, because no status was ever written.** An earlier draft of
    # this revision put the referral into `events.status` and this downgrade had to rewrite
    # those rows to 'REVIEWED' on the way back, since SQLAlchemy's non-native Enum raises on
    # the way OUT and /events would have 500'd for the school. That whole class of problem
    # went away with the token: `status` never held a referral, so going back cannot leave a
    # value the old code chokes on.
    #
    # What IS lost here is the referral itself — who handed a child on, and when — because
    # it lives only in the two columns dropped below. Nothing deletes an event: 0003 could
    # delete its canteen_staff rows because an account is recreatable, and a bullying
    # incident is not.
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.drop_constraint('fk_events_referred_by_id_users', type_='foreignkey')
        batch_op.drop_column('referred_at')
        batch_op.drop_column('referred_by_id')

"""a lesson survives the worker watching it

`lessons` and `lesson_tracks` — §12.4, cut back to the four facts the school was promised
in writing (`docs/questions-for-school.md` §8): how many times a hand went up, how many
times someone stood up, how long they were away from their place, and whether they were
there at all.

**Why these are tables and not a dictionary in the worker.** A lesson runs 45 minutes. The
legacy kept meal sessions in a RAM dict inside a module-global singleton, and a restart
silently lost every open one — for a canteen that is a few minutes of children; for a
lesson it is almost the whole record, because almost the whole record is "open". The
worker writes its ledgers through on `flush_interval_seconds`, so a crash costs half a
minute of counting and never the lesson, and re-attaches to the still-open row when it
comes back (`lessons.resumed_count` records that it happened, because ByteTrack's ids
restart with the process and every child in the room becomes a new track and a new row).

**NEITHER TABLE HAS A `person_id`, AND NEITHER MAY EVER GAIN ONE.** Every other
observation table here points at `persons`. These do not, because §8 promised the school
no identification inside a classroom — and this school's own footage says the promise
costs nothing to keep: 14 970 corridor faces, median 11.5 px, **zero** recognised, and a
face across a classroom is no larger. A nullable `person_id` here could only ever be
filled by a method already measured to produce nothing, and every report joined through it
would read as a claim about a named child.

`lessons.min_presence_seconds` is a copy of the camera's configured threshold, taken when
the lesson opens. It is stored rather than re-read because a report is a statement about a
past hour: re-reading today's YAML would silently restate an old lesson under a threshold
nobody applied to it — a value true in one layer and quietly wrong in the next, which is
the defect this project keeps finding.

The four counters on `lessons` (`ambiguous_observations`, `unclaimed_observations`,
`dropped_tracks`, `resumed_count`) are part of the result, not diagnostics: they say how
much of the lesson the system could not see, and no surface may render the totals without
them. Four columns rather than one quality score because they have four different causes
and four different cures.

Nothing is back-filled and there is nothing to back-fill: both tables are new and start
empty, which is the honest state — no lesson was observed before the code existed to
observe one.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-28 14:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = '0006'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CAMERA_TYPE_AFTER = sa.Enum(
    "BULLYING", "CANTEEN", "CLASSROOM", name="cameratype", native_enum=False
)
CAMERA_TYPE_BEFORE = sa.Enum("BULLYING", "CANTEEN", name="cameratype", native_enum=False)


def upgrade() -> None:
    # `CameraType` gained CLASSROOM, and with `native_enum=False` that is a VARCHAR whose
    # LENGTH is the longest member: 8 ("bullying") becomes 9 ("classroom"). Widening it is
    # not cosmetic -- on a backend that enforces the length, inserting a classroom camera
    # into the old column fails. `test_the_migration_matches_the_models` caught this, which
    # is the whole reason that test compares the built schema against the models rather
    # than trusting that a migration was remembered.
    with op.batch_alter_table("cameras") as batch:
        batch.alter_column(
            "camera_type",
            existing_type=CAMERA_TYPE_BEFORE,
            type_=CAMERA_TYPE_AFTER,
            existing_nullable=False,
        )

    op.create_table(
        "lessons",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column(
            "state",
            sa.Enum("OPEN", "CLOSED", name="lessonstate", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "close_reason",
            sa.Enum(
                "EMPTY_ROOM", "TIMEOUT", name="lessonclosereason", native_enum=False
            ),
            nullable=True,
        ),
        sa.Column("started_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("ended_at", qorgan.db.types.UtcDateTime(), nullable=True),
        sa.Column("min_presence_seconds", sa.Float(), nullable=False),
        sa.Column("ambiguous_observations", sa.Integer(), nullable=False),
        sa.Column("unclaimed_observations", sa.Integer(), nullable=False),
        sa.Column("dropped_tracks", sa.Integer(), nullable=False),
        sa.Column("resumed_count", sa.Integer(), nullable=False),
        sa.Column("created_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name=op.f("fk_lessons_camera_id_cameras"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lessons")),
    )
    op.create_index(op.f("ix_lessons_camera_id"), "lessons", ["camera_id"])
    op.create_index(op.f("ix_lessons_started_at"), "lessons", ["started_at"])
    op.create_index("ix_lessons_camera_state", "lessons", ["camera_id", "state"])

    op.create_table(
        "lesson_tracks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("lesson_id", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("last_seen_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("observed_seconds", sa.Float(), nullable=False),
        sa.Column("observations", sa.Integer(), nullable=False),
        sa.Column("settled", sa.Boolean(), nullable=False),
        sa.Column("hand_raises", sa.Integer(), nullable=False),
        sa.Column("stands", sa.Integer(), nullable=False),
        sa.Column("away_seconds", sa.Float(), nullable=False),
        sa.Column("brief_excursions", sa.Integer(), nullable=False),
        sa.Column("created_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.Column("updated_at", qorgan.db.types.UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["lesson_id"],
            ["lessons.id"],
            name=op.f("fk_lesson_tracks_lesson_id_lessons"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lesson_tracks")),
        # The worker UPSERTS a track's row on every flush, so this constraint is what
        # makes "write the totals so far" idempotent instead of appending a new row every
        # thirty seconds for forty-five minutes.
        sa.UniqueConstraint(
            "lesson_id", "track_id", name="uq_lesson_tracks_lesson_track"
        ),
    )
    op.create_index(op.f("ix_lesson_tracks_lesson_id"), "lesson_tracks", ["lesson_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_lesson_tracks_lesson_id"), table_name="lesson_tracks")
    op.drop_table("lesson_tracks")
    op.drop_index("ix_lessons_camera_state", table_name="lessons")
    op.drop_index(op.f("ix_lessons_started_at"), table_name="lessons")
    op.drop_index(op.f("ix_lessons_camera_id"), table_name="lessons")
    op.drop_table("lessons")

    # Narrow `camera_type` back. **This is the one step of this downgrade that can fail on
    # a populated database, and it should**: a school that has configured a classroom
    # camera has rows reading 'CLASSROOM' in a column about to be told 9 characters are 8.
    # Silently truncating them would leave a camera row of a type no code recognises,
    # which is worse than a refusal a human reads. Delete the classroom cameras first if
    # this is genuinely what you want.
    with op.batch_alter_table("cameras") as batch:
        batch.alter_column(
            "camera_type",
            existing_type=CAMERA_TYPE_AFTER,
            type_=CAMERA_TYPE_BEFORE,
            existing_nullable=False,
        )

"""a place outlives a track

Eight tables under `classvision_`, holding what an OFFLINE analyser computed from a
recording and accumulating it per PLACE — `classvision_lessons`, `_runs`, `_places`,
`_place_lessons`, `_teacher_lessons`, `_frames`, `_readings`, `_attestations`. The contract
they implement is `classvision/INTEGRATION.md`; the argument for each column is in
`src/qorgan/db/models/classvision.py` and is not repeated here.

**THIS MIGRATION TOUCHES NOTHING THAT EXISTS.** No column is added to `lessons` or
`lesson_tracks`, so the live classroom worker and `/lessons` are unaffected and the promise
in `db/models/classroom.py` — that neither table has a `person_id` and neither may gain one
— stays literally true. That promise is not circumvented by a trick here: a `LessonTrack`
row is a ByteTrack id that dies at the first long occlusion, so a `person_id` on it could
never carry the four-week trend it would have been added for, while looking exactly as
though it could. The stable object is the SEAT, which the old schema has no concept of, and
that is why this is eight new tables instead of two new columns.

**WHY THIS ONE IS SAFE IN THE SHAPE THAT MADE 0009 DANGEROUS.** 0009 rebuilt `cameras`
three times, and a batch rebuild is a DROP as far as foreign keys are concerned — which is
why `migrations/env.py::_suspend_foreign_keys` exists and why
`tests/test_migration_keeps_the_events.py` measures that it works. Here every
`batch_alter_table` is against a table this migration has just CREATED and which no other
table references, so there is no history to lose. Nothing is back-filled and there is
nothing to back-fill: no recording was analysed before the code existed to analyse one.

**THE ORDER IS A FOREIGN KEY ORDER, NOT A PREFERENCE.** `classvision_lessons` and
`classvision_places` first, because everything else points at one of them:
`_runs` -> `_lessons`; `_place_lessons` -> `_lessons`, `_runs`, `_places`, `persons`;
`_teacher_lessons` and `_frames` -> `_lessons` + `_runs`; `_readings` -> `_runs`;
`_attestations` -> `_places` + `persons`.

**`is_demo` IS NOT NULL ON `_lessons`, `_runs` AND `_places`.** Those three are the tables
that originate a row; everything else reaches one through a key that cannot be NULL. The
column exists because this branch ships a demonstration generator (`qorgan classvision
demo`), and a fabricated number that looks measured is the one unacceptable outcome of a
demonstration. It is NOT NULL rather than nullable-defaulting-false so that a writer has to
answer the question — «это измерение или показ?» — instead of inheriting an answer.

**`school_id` IS ON `_lessons` AND `_places` AND ON NOTHING ELSE HERE.** Those two are root
tables in the sense `tests/test_tenancy_guard.py` argues: nothing else in the schema can
answer for them, because `camera_key` is an operator's assertion about which room a FILE
came from and not a foreign key into `cameras`. The rest reach a school through a NOT NULL
key, and a second `school_id` on them would be a second answer to a question that already
has one.

`downgrade()` drops all eight. That is safe precisely because nothing outside this set
references them.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-17 21:15:19.833145
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import qorgan.db.types  # noqa: F401 -- custom column types used in the schema

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('classvision_lessons',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('school_id', sa.Integer(), nullable=False),
    sa.Column('is_demo', sa.Boolean(), nullable=False),
    sa.Column('camera_key', sa.String(length=64), nullable=False),
    sa.Column('camera_key_source', sa.String(length=24), nullable=False),
    sa.Column('class_key', sa.String(length=64), nullable=False),
    sa.Column('started_at', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('ended_at', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('date_local', sa.Date(), nullable=True),
    sa.Column('iso_year', sa.Integer(), nullable=True),
    sa.Column('iso_week', sa.Integer(), nullable=True),
    sa.Column('timezone', sa.String(length=64), nullable=True),
    sa.Column('duration_minutes', sa.Float(), nullable=False),
    sa.Column('selected_run_id', sa.String(length=32), nullable=False),
    sa.Column('continues_lesson_id', sa.Integer(), nullable=True),
    sa.Column('part_count', sa.Integer(), nullable=False),
    sa.Column('overlap_allowed', sa.Boolean(), nullable=False),
    sa.Column('overlap_note', sa.Text(), nullable=True),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['continues_lesson_id'], ['classvision_lessons.id'], name=op.f('fk_classvision_lessons_continues_lesson_id_classvision_lessons'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['school_id'], ['schools.id'], name=op.f('fk_classvision_lessons_school_id_schools'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_lessons'))
    )
    with op.batch_alter_table('classvision_lessons', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_lessons_date_local'), ['date_local'], unique=False)
        batch_op.create_index('ix_classvision_lessons_demo', ['school_id', 'is_demo'], unique=False)
        batch_op.create_index('ix_classvision_lessons_room_date', ['school_id', 'camera_key', 'class_key', 'date_local'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_lessons_school_id'), ['school_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_lessons_started_at'), ['started_at'], unique=False)

    op.create_table('classvision_places',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('school_id', sa.Integer(), nullable=False),
    sa.Column('is_demo', sa.Boolean(), nullable=False),
    sa.Column('camera_key', sa.String(length=64), nullable=False),
    sa.Column('class_key', sa.String(length=64), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('label_ru', sa.String(length=64), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('anchor_x', sa.Float(), nullable=False),
    sa.Column('anchor_y', sa.Float(), nullable=False),
    sa.Column('anchor_scale', sa.Float(), nullable=False),
    sa.Column('first_run_id', sa.String(length=32), nullable=False),
    sa.Column('first_seen_at', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['school_id'], ['schools.id'], name=op.f('fk_classvision_places_school_id_schools'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_places')),
    sa.UniqueConstraint('school_id', 'camera_key', 'class_key', 'ordinal', name='uq_classvision_places_room_ordinal')
    )
    with op.batch_alter_table('classvision_places', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_places_school_id'), ['school_id'], unique=False)

    op.create_table('classvision_attestations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('place_id', sa.Integer(), nullable=False),
    sa.Column('person_id', sa.Integer(), nullable=False),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('attested_by', sa.String(length=200), nullable=False),
    sa.Column('attested_at', sa.Date(), nullable=False),
    sa.Column('decision_ref', sa.String(length=200), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_classvision_attestations_person_id_persons'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['place_id'], ['classvision_places.id'], name=op.f('fk_classvision_attestations_place_id_classvision_places'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_attestations'))
    )
    with op.batch_alter_table('classvision_attestations', schema=None) as batch_op:
        batch_op.create_index('ix_classvision_attestations_lookup', ['place_id', 'valid_from'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_attestations_person_id'), ['person_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_attestations_place_id'), ['place_id'], unique=False)

    op.create_table('classvision_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('lesson_id', sa.Integer(), nullable=False),
    sa.Column('is_demo', sa.Boolean(), nullable=False),
    sa.Column('run_id', sa.String(length=32), nullable=False),
    sa.Column('schema_version', sa.String(length=32), nullable=False),
    sa.Column('video_path', sa.Text(), nullable=False),
    sa.Column('video_sha256', sa.String(length=64), nullable=False),
    sa.Column('video_bytes', sa.Integer(), nullable=False),
    sa.Column('started_at', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('clock_source', sa.String(length=16), nullable=False),
    sa.Column('clock_drift_seconds', sa.Float(), nullable=True),
    sa.Column('sample_fps', sa.Float(), nullable=False),
    sa.Column('analysed_frames', sa.Integer(), nullable=False),
    sa.Column('duration_seconds', sa.Float(), nullable=False),
    sa.Column('thresholds_sha', sa.String(length=16), nullable=False),
    sa.Column('model_weights', sa.String(length=64), nullable=False),
    sa.Column('model_imgsz', sa.Integer(), nullable=True),
    sa.Column('model_device', sa.String(length=16), nullable=True),
    sa.Column('room_layout', sa.JSON(), nullable=False),
    sa.Column('session', sa.JSON(), nullable=True),
    sa.Column('pupil_places', sa.Integer(), nullable=False),
    sa.Column('adult_seat_id', sa.Integer(), nullable=True),
    sa.Column('observations_total', sa.Integer(), nullable=False),
    sa.Column('observations_unassigned', sa.Integer(), nullable=False),
    sa.Column('observations_unreadable', sa.Integer(), nullable=False),
    sa.Column('frames_with_no_person', sa.Integer(), nullable=False),
    sa.Column('seats_never_settled', sa.Integer(), nullable=False),
    sa.Column('provenance', sa.JSON(), nullable=False),
    sa.Column('uncertainty', sa.JSON(), nullable=False),
    sa.Column('caveats', sa.JSON(), nullable=False),
    sa.Column('unmeasured', sa.JSON(), nullable=False),
    sa.Column('analysed_at', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('imported_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['lesson_id'], ['classvision_lessons.id'], name=op.f('fk_classvision_runs_lesson_id_classvision_lessons'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_runs')),
    sa.UniqueConstraint('run_id', name='uq_classvision_runs_run_id')
    )
    with op.batch_alter_table('classvision_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_runs_lesson_id'), ['lesson_id'], unique=False)
        batch_op.create_index('ix_classvision_runs_started', ['started_at'], unique=False)

    op.create_table('classvision_frames',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('lesson_id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('video_seconds', sa.Float(), nullable=False),
    sa.Column('wall_clock', qorgan.db.types.UtcDateTime(), nullable=True),
    sa.Column('image_path', qorgan.db.types.RelPath(length=255), nullable=False),
    sa.Column('image_width', sa.Integer(), nullable=False),
    sa.Column('image_height', sa.Integer(), nullable=False),
    sa.Column('source_video', sa.Text(), nullable=True),
    sa.Column('box_source', sa.String(length=24), nullable=False),
    sa.Column('boxes', sa.JSON(), nullable=False),
    sa.Column('caveat_ru', sa.Text(), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['lesson_id'], ['classvision_lessons.id'], name=op.f('fk_classvision_frames_lesson_id_classvision_lessons'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['classvision_runs.id'], name=op.f('fk_classvision_frames_run_id_classvision_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_frames')),
    sa.UniqueConstraint('run_id', 'video_seconds', name='uq_classvision_frames_run_second')
    )
    with op.batch_alter_table('classvision_frames', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_frames_lesson_id'), ['lesson_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_frames_run_id'), ['run_id'], unique=False)

    op.create_table('classvision_place_lessons',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('lesson_id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('place_id', sa.Integer(), nullable=True),
    sa.Column('seat_id', sa.Integer(), nullable=False),
    sa.Column('seat_label', sa.String(length=32), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('place_match', sa.String(length=16), nullable=False),
    sa.Column('place_match_reason', sa.Text(), nullable=True),
    sa.Column('place_match_distance', sa.Float(), nullable=True),
    sa.Column('centre_x', sa.Float(), nullable=False),
    sa.Column('centre_y', sa.Float(), nullable=False),
    sa.Column('scale_px', sa.Float(), nullable=False),
    sa.Column('person_id', sa.Integer(), nullable=True),
    sa.Column('identity_method', sa.String(length=32), nullable=False),
    sa.Column('identity_reason', sa.Text(), nullable=True),
    sa.Column('coverage', sa.Float(), nullable=False),
    sa.Column('observations', sa.Integer(), nullable=False),
    sa.Column('observed_seconds', sa.Float(), nullable=False),
    sa.Column('settled', sa.Boolean(), nullable=False),
    sa.Column('settle_refusal', sa.Text(), nullable=True),
    sa.Column('absent_observations', sa.Integer(), nullable=False),
    sa.Column('unreadable_observations', sa.Integer(), nullable=False),
    sa.Column('hand_unmeasurable_observations', sa.Integer(), nullable=False),
    sa.Column('hand_raises', sa.Integer(), nullable=False),
    sa.Column('stands', sa.Integer(), nullable=False),
    sa.Column('away_episodes', sa.Integer(), nullable=False),
    sa.Column('board_visits', sa.Integer(), nullable=False),
    sa.Column('head_down_episodes', sa.Integer(), nullable=False),
    sa.Column('turned_away_episodes', sa.Integer(), nullable=False),
    sa.Column('activity_index', sa.Float(), nullable=True),
    sa.Column('activity_reason', sa.Text(), nullable=False),
    sa.Column('activity_parts', sa.JSON(), nullable=False),
    sa.Column('within_lesson', sa.JSON(), nullable=False),
    sa.Column('ledger', sa.JSON(), nullable=False),
    sa.Column('timeline', sa.JSON(), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['lesson_id'], ['classvision_lessons.id'], name=op.f('fk_classvision_place_lessons_lesson_id_classvision_lessons'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_classvision_place_lessons_person_id_persons'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['place_id'], ['classvision_places.id'], name=op.f('fk_classvision_place_lessons_place_id_classvision_places'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['run_id'], ['classvision_runs.id'], name=op.f('fk_classvision_place_lessons_run_id_classvision_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_place_lessons')),
    sa.UniqueConstraint('run_id', 'seat_id', name='uq_classvision_place_lessons_run_seat')
    )
    with op.batch_alter_table('classvision_place_lessons', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_place_lessons_lesson_id'), ['lesson_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_place_lessons_person_id'), ['person_id'], unique=False)
        batch_op.create_index('ix_classvision_place_lessons_place', ['place_id', 'lesson_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_place_lessons_place_id'), ['place_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_classvision_place_lessons_run_id'), ['run_id'], unique=False)

    op.create_table('classvision_readings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('section', sa.String(length=32), nullable=False),
    sa.Column('target_key', sa.String(length=64), nullable=False),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('model', sa.String(length=64), nullable=True),
    sa.Column('prompt_version', sa.String(length=32), nullable=False),
    sa.Column('body', sa.Text(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=True),
    sa.Column('guard_passed', sa.Boolean(), nullable=False),
    sa.Column('guard_offending', sa.JSON(), nullable=False),
    sa.Column('guard_reason_ru', sa.Text(), nullable=False),
    sa.Column('generated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['classvision_runs.id'], name=op.f('fk_classvision_readings_run_id_classvision_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_readings')),
    sa.UniqueConstraint('run_id', 'section', 'target_key', name='uq_classvision_readings_run_section')
    )
    with op.batch_alter_table('classvision_readings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_readings_run_id'), ['run_id'], unique=False)

    op.create_table('classvision_teacher_lessons',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('lesson_id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('place_id', sa.Integer(), nullable=True),
    sa.Column('place_missing_reason', sa.Text(), nullable=True),
    sa.Column('seat_id', sa.Integer(), nullable=True),
    sa.Column('attributed_share_of_lesson_percent', sa.Float(), nullable=True),
    sa.Column('pose_coverage', sa.Float(), nullable=True),
    sa.Column('board_zone_configured', sa.Boolean(), nullable=False),
    sa.Column('board_minutes_of_lesson', sa.Float(), nullable=True),
    sa.Column('board_share_of_lesson_percent', sa.Float(), nullable=True),
    sa.Column('board_occupancy_available', sa.Boolean(), nullable=False),
    sa.Column('transitions_excluding_out_of_frame', sa.Integer(), nullable=True),
    sa.Column('pose_transitions', sa.Integer(), nullable=True),
    sa.Column('presence', sa.JSON(), nullable=True),
    sa.Column('board', sa.JSON(), nullable=False),
    sa.Column('board_occupancy', sa.JSON(), nullable=False),
    sa.Column('pose_metrics', sa.JSON(), nullable=False),
    sa.Column('not_an_assessment_ru', sa.Text(), nullable=False),
    sa.Column('created_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.Column('updated_at', qorgan.db.types.UtcDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['lesson_id'], ['classvision_lessons.id'], name=op.f('fk_classvision_teacher_lessons_lesson_id_classvision_lessons'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['place_id'], ['classvision_places.id'], name=op.f('fk_classvision_teacher_lessons_place_id_classvision_places'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['run_id'], ['classvision_runs.id'], name=op.f('fk_classvision_teacher_lessons_run_id_classvision_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_classvision_teacher_lessons')),
    sa.UniqueConstraint('run_id', name='uq_classvision_teacher_lessons_run')
    )
    with op.batch_alter_table('classvision_teacher_lessons', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_classvision_teacher_lessons_lesson_id'), ['lesson_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('classvision_teacher_lessons', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_teacher_lessons_lesson_id'))

    op.drop_table('classvision_teacher_lessons')
    with op.batch_alter_table('classvision_readings', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_readings_run_id'))

    op.drop_table('classvision_readings')
    with op.batch_alter_table('classvision_place_lessons', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_place_lessons_run_id'))
        batch_op.drop_index(batch_op.f('ix_classvision_place_lessons_place_id'))
        batch_op.drop_index('ix_classvision_place_lessons_place')
        batch_op.drop_index(batch_op.f('ix_classvision_place_lessons_person_id'))
        batch_op.drop_index(batch_op.f('ix_classvision_place_lessons_lesson_id'))

    op.drop_table('classvision_place_lessons')
    with op.batch_alter_table('classvision_frames', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_frames_run_id'))
        batch_op.drop_index(batch_op.f('ix_classvision_frames_lesson_id'))

    op.drop_table('classvision_frames')
    with op.batch_alter_table('classvision_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_classvision_runs_started')
        batch_op.drop_index(batch_op.f('ix_classvision_runs_lesson_id'))

    op.drop_table('classvision_runs')
    with op.batch_alter_table('classvision_attestations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_attestations_place_id'))
        batch_op.drop_index(batch_op.f('ix_classvision_attestations_person_id'))
        batch_op.drop_index('ix_classvision_attestations_lookup')

    op.drop_table('classvision_attestations')
    with op.batch_alter_table('classvision_places', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_places_school_id'))

    op.drop_table('classvision_places')
    with op.batch_alter_table('classvision_lessons', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_classvision_lessons_started_at'))
        batch_op.drop_index(batch_op.f('ix_classvision_lessons_school_id'))
        batch_op.drop_index('ix_classvision_lessons_room_date')
        batch_op.drop_index('ix_classvision_lessons_demo')
        batch_op.drop_index(batch_op.f('ix_classvision_lessons_date_local'))

    op.drop_table('classvision_lessons')

"""Lessons, and the anonymous tracks observed during them.

**There is no `person_id` on either table, and its absence is the design.** Every other
observation table in this schema points at `persons`: a meal session does, a recognition
attempt does, a face embedding does. These two do not, and they must not gain one. §8
promised the school there would be no identification inside a classroom, and the
measurement behind that promise is this school's own: 14 970 corridor faces at a median
of 11.5 px yielded **zero** recognitions, and a face across a classroom is no larger. A
nullable `person_id` here would be a column that could only ever be filled by a method
already measured to produce nothing -- and once one existed, every report joined through
it would look like it was about children rather than about tracks.

**A lesson is a ROW from the moment it opens, not a dict entry**, for the reason the
canteen learned the hard way (`db/models/canteen.py`): the legacy kept sessions in RAM
inside a module global and a restart silently lost every open one. A lesson is 45 minutes
long, so the window in which a restart can destroy it is most of its life. Here the
worker writes its ledgers through every `flush_interval_seconds` and re-attaches to the
open row when it comes back up.

**The four doubt counters on `Lesson` are not diagnostics; they are part of the result.**
They say how much of the lesson the system could not see, and the report is not allowed
to render the totals without them. They are separate columns rather than one quality
score because they have four different causes and four different cures -- see
`classroom/lesson.py::Doubt`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.types import UtcDateTime
from qorgan.enums import LessonCloseReason, LessonState


class Lesson(Base, TimestampMixin):
    """One camera's observation period. Not a timetabled lesson -- we have no timetable.

    It opens when the room stops being empty and closes when it has been empty again for
    a while, or on the `max_lesson_minutes` ceiling. Nothing here claims to know what was
    being taught, or to whom, or by whom.
    """

    __tablename__ = "lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )

    state: Mapped[LessonState] = mapped_column(
        Enum(LessonState, native_enum=False), default=LessonState.OPEN, nullable=False
    )
    close_reason: Mapped[LessonCloseReason | None] = mapped_column(
        Enum(LessonCloseReason, native_enum=False)
    )

    started_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    # The presence threshold IN FORCE WHEN THIS LESSON RAN, copied off the camera's config
    # at the moment the lesson opened. Stored rather than re-read, because a report is a
    # statement about a past hour: re-reading today's YAML would silently restate last
    # month's lesson under a threshold nobody applied to it, which is this codebase's
    # signature defect (a value true in one layer and quietly wrong in the next).
    min_presence_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # -- what we could not see. See the module docstring; never render totals without it.
    #
    # Skeletons dropped because the geometry was ambiguous -- an anchor inside two
    # people's boxes, or two anchors inside one. Rows of desks make this ordinary.
    ambiguous_observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Skeletons inside no tracked box: usually the adult at the front, sometimes a body
    # at the frame edge, always every skeleton before tracking starts.
    unclaimed_observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # New tracks refused because `max_tracks` was already full (rule R8). Non-zero means
    # the room outgrew the cap and part of it went unwatched.
    dropped_tracks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # How many times a worker attached to this already-open lesson, i.e. restarted mid-
    # lesson. It matters to the reader because ByteTrack's ids restart with the process:
    # after a resume, every child in the room is a NEW track and a SECOND row, so the
    # track count jumps without a child having moved.
    resumed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    tracks = relationship(
        "LessonTrack", back_populates="lesson", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_lessons_camera_state", "camera_id", "state"),)


class LessonTrack(Base, TimestampMixin):
    """One anonymous track's totals. `track_id` is ByteTrack's, and it is not a child.

    Unique on (lesson, track), because the worker UPSERTS this row on every flush: the
    same track is written many times during a lesson, each time with its totals so far.
    """

    __tablename__ = "lesson_tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # ByteTrack's id, unique only within one run of one worker on one camera. It is
    # deliberately not called anything that sounds like a person.
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)

    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    observed_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Frames actually folded in. Distinct from `observed_seconds`: a track present for
    # ten minutes but visible in a tenth of the frames has been measured much less than
    # its duration suggests, and only this column says so.
    observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Whether a seated baseline was ever established. **`stands` and `away_seconds` are
    # UNMEASURED, not zero, when this is false**, and the report must render the two
    # differently -- a boolean whose false has two causes is the defect migration 0005
    # was written about. `hand_raises` needs no baseline and is exempt.
    settled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    hand_raises: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stands: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    away_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Excursions discarded for being shorter than `min_away_seconds`. The direct measure
    # of whether that threshold is anywhere near right -- hundreds of these means the
    # tracker is jittering, not that the children kept nearly leaving.
    brief_excursions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    lesson = relationship("Lesson", back_populates="tracks")

    __table_args__ = (
        UniqueConstraint("lesson_id", "track_id", name="uq_lesson_tracks_lesson_track"),
    )

"""Meal sessions, meal windows, and the recognition attempts behind them.

The single biggest correctness change in the rewrite: **a session is a row**.

Legacy kept sessions in a RAM dict inside a module-global singleton, so a process
restart silently lost every open session; and one code path popped a session from
that dict without ever writing it to the database, so the pupil vanished from the
record entirely. Here a session is persisted from the moment it opens, and it moves
through an explicit state machine.
"""

from __future__ import annotations

from datetime import datetime, time

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, Integer, String, Time
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.school import school_key
from qorgan.db.types import UtcDateTime
from qorgan.enums import CloseReason, IdentitySource, MealKind, SessionOutcome, SessionState


class MealWindow(Base, TimestampMixin):
    """Breakfast / lunch. A pupil is counted once per window, not once per day."""

    __tablename__ = "meal_windows"

    id: Mapped[int] = mapped_column(primary_key=True)
    # A root table, and one of the four, because no key on it reaches a school: a meal
    # window is a wall-clock fact about ONE school's day. Two schools on one installation
    # do not eat at the same time, and a shared breakfast window would count one school's
    # pupils inside the other's hours.
    school_id: Mapped[int] = school_key(nullable=False)
    kind: Mapped[MealKind] = mapped_column(Enum(MealKind, native_enum=False), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Local school time, not UTC: a meal window is a wall-clock fact.
    starts_at: Mapped[time] = mapped_column(Time, nullable=False)
    ends_at: Mapped[time] = mapped_column(Time, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class CanteenSession(Base, TimestampMixin):
    __tablename__ = "canteen_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    # NULL means the entry camera opened the session for a face it could not identify.
    # An inside camera may still late-bind an identity to it.
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )
    meal_window_id: Mapped[int | None] = mapped_column(
        ForeignKey("meal_windows.id", ondelete="SET NULL")
    )

    entry_camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    exit_camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL")
    )

    state: Mapped[SessionState] = mapped_column(
        Enum(SessionState, native_enum=False), default=SessionState.OPEN, nullable=False
    )
    outcome: Mapped[SessionOutcome | None] = mapped_column(
        Enum(SessionOutcome, native_enum=False)
    )
    close_reason: Mapped[CloseReason | None] = mapped_column(Enum(CloseReason, native_enum=False))

    identity_source: Mapped[IdentitySource | None] = mapped_column(
        Enum(IdentitySource, native_enum=False)
    )
    identity_score: Mapped[float | None] = mapped_column(Float)

    opened_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    dwell_seconds: Mapped[float | None] = mapped_column(Float)

    person = relationship("Person")

    __table_args__ = (
        Index("ix_canteen_sessions_state", "state"),
        Index("ix_canteen_sessions_person_opened", "person_id", "opened_at"),
    )


class RecognitionAttempt(Base):
    """Every recognition decision, kept for calibration. Subject to the retention janitor.

    This is the data the legacy project never had: it tuned 18 overlapping thresholds
    by feel, and 1816 of its 1820 canteen records ended up with student_id = NULL.
    """

    __tablename__ = "recognition_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("canteen_sessions.id", ondelete="SET NULL")
    )
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)

    top1_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL")
    )
    top1_score: Mapped[float | None] = mapped_column(Float)
    top2_score: Mapped[float | None] = mapped_column(Float)
    gap: Mapped[float | None] = mapped_column(Float)
    face_width: Mapped[int | None] = mapped_column(Integer)
    face_height: Mapped[int | None] = mapped_column(Integer)

    accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Why it was accepted or rejected, as a stable machine-readable token.
    reason: Mapped[str] = mapped_column(String(64), default="", nullable=False)

"""Bullying events and the notifications raised from them."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.types import RelPath, UtcDateTime
from qorgan.enums import (
    EventStatus,
    EventType,
    NotificationChannel,
    NotificationStatus,
    Severity,
    TelegramSkipReason,
)


class Event(Base, TimestampMixin):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType, native_enum=False), nullable=False
    )

    # UTC, timezone-aware. The column type rejects a naive datetime.
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)

    # The full confidence trail, so the eval harness can reconstruct any decision.
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_probability: Mapped[float] = mapped_column(Float, nullable=False)
    validation_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    skeleton_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    severity: Mapped[Severity] = mapped_column(Enum(Severity, native_enum=False), nullable=False)

    # Written once, at the final severity. Legacy computed the summary AFTER the
    # insert, so every row in its database says "Подозрение..." no matter how
    # confident the event actually was (audit M-03).
    summary_text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # WHY the system decided this was an assault: the skeleton's own reasons, packed by
    # `qorgan.events.reasons`. The confidence says how sure it was; only this says what it
    # saw, and "someone went down" and "someone waved their arms" are not the same alert.
    # Empty is normal and not an error — the skeleton is often unable to look at all.
    reasons: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)

    # Why nobody was told (client §7's `_telegram_skip_reason`), or NULL because they were.
    #
    # **On the event and not on `notifications`, because a notification that was never sent
    # has no row to put it on.** Storing it there would mean inventing a row for a delivery
    # that never happened, under a fourth status — and every count on /notifications, and
    # the word "НЕ ДОСТАВЛЕНО" itself, would then mean two different things at once: a
    # message the router ate, and a message the system correctly chose not to send. Those
    # are the two separate questions the school asks, and merging them loses both answers.
    #
    # It also belongs to the same judgement as the four fields above it. `confidence`,
    # `skeleton_confirmed` and `reasons` are the inputs; this is what was decided from them.
    #
    # NULL means "not withheld" — which for rows written before this column existed is an
    # unknown, not an assertion that they were sent. Nothing is back-filled: whether a 2026
    # event raised a Telegram is not recoverable from the row, and guessing it from today's
    # thresholds is exactly the second source of truth this column exists to remove.
    telegram_skip_reason: Mapped[TelegramSkipReason | None] = mapped_column(
        Enum(TelegramSkipReason, native_enum=False)
    )

    # Relative to MEDIA_ROOT.
    snapshot_path: Mapped[str | None] = mapped_column(RelPath(512))
    clip_path: Mapped[str | None] = mapped_column(RelPath(512))

    track_ids: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    # Set when this event was deduplicated into an earlier one for the same incident.
    merged_into_id: Mapped[int | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"))

    status: Mapped[EventStatus] = mapped_column(
        Enum(EventStatus, native_enum=False), default=EventStatus.NEW, nullable=False
    )
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    # §9's «передано психологу», as the two facts that make it a referral rather than a
    # label: WHO handed this child on, and WHEN.
    #
    # **Separate from `status`, and separate from `reviewed_by_id`, because a referral is
    # not a verdict and it does not expire.** `status` is the event's current state, so an
    # operator who refers an incident and afterwards confirms it as bullying overwrites the
    # status — and if the referral lived only there, confirming would silently un-refer the
    # child. `reviewed_by_id` is the person who ruled on whether this was an assault, which
    # is a different judgement and often a different person. So the psychologist's cabinet
    # selects on `referred_at IS NOT NULL` and never on the status token.
    #
    # **Written by exactly one code path** — `web.routes.events.refer_event`, guarded by
    # `Capability.REFER_TO_PSYCHOLOGIST`. `/events/{id}/review` REFUSES the referral
    # verdict, so there is no door through which the status can be set with nobody
    # recorded beside it. Nothing computed anywhere may write these: §8 promised the
    # school no referral FROM THE SYSTEM, and a nullable column is exactly where such a
    # thing would arrive quietly.
    #
    # NULL means "never referred". Nothing is back-filled; no past event's referral is
    # recoverable, and inventing one would be inventing a decision about a child.
    referred_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    referred_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    notifications: Mapped[list[Notification]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_events_status", "status"),
        Index("ix_events_camera_occurred", "camera_id", "occurred_at"),
    )


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel: Mapped[NotificationChannel] = mapped_column(
        Enum(NotificationChannel, native_enum=False), nullable=False
    )
    # Every attempt is logged, not just the failures of a placeholder provider
    # that always failed (audit M-14).
    status: Mapped[NotificationStatus] = mapped_column(
        Enum(NotificationStatus, native_enum=False),
        default=NotificationStatus.QUEUED,
        nullable=False,
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    provider_message_id: Mapped[str | None] = mapped_column(String(64))
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    event: Mapped[Event] = relationship(back_populates="notifications")

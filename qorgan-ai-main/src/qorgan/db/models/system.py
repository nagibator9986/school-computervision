"""Operational state: worker health, mode history, and non-secret app settings."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.types import UtcDateTime
from qorgan.enums import ModeSource, SystemMode, WorkerState


class WorkerHeartbeat(Base, TimestampMixin):
    """One row per worker group. The supervisor writes it; the web process reads it.

    This is how the dashboard shows worker health WITHOUT importing a worker module.
    Legacy's web layer imported CAMERA_REGISTRY straight out of the bullying worker,
    so loading a web route pulled in YOLO and torch as a side effect (audit M-23).
    """

    __tablename__ = "worker_heartbeats"

    id: Mapped[int] = mapped_column(primary_key=True)
    group_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    pid: Mapped[int | None] = mapped_column(Integer)
    state: Mapped[WorkerState] = mapped_column(
        Enum(WorkerState, native_enum=False), default=WorkerState.STARTING, nullable=False
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    restart_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class ModeLog(Base):
    __tablename__ = "mode_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    previous_mode: Mapped[SystemMode | None] = mapped_column(
        Enum(SystemMode, native_enum=False)
    )
    new_mode: Mapped[SystemMode] = mapped_column(
        Enum(SystemMode, native_enum=False), nullable=False
    )
    source: Mapped[ModeSource] = mapped_column(Enum(ModeSource, native_enum=False), nullable=False)
    switched_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)


class AppSetting(Base, TimestampMixin):
    """Operator-editable settings.

    NO SECRETS. Legacy kept a second Telegram bot token in a table exactly like this
    one, in plaintext, and handed the whole table to any anonymous caller of
    /settings?format=json (audit H-04). tests/test_no_secrets.py enforces the rule.
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(64))

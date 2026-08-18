"""The camera registry.

The row holds identity and host only. It does NOT hold a stream URL, because the
legacy `cameras.stream_url` column held `rtsp://user:password@...` in plaintext for
all 15 rows -- the same password for the whole fleet -- and the dashboard served that
column to anyone who asked. Credentials come from the environment; the URL is built
at use time (see qorgan.rtsp).
"""

from __future__ import annotations

from sqlalchemy import Boolean, Enum, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.school import school_key
from qorgan.enums import CameraRole, CameraType


class Camera(Base, TimestampMixin):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = school_key(nullable=False)
    # Matches the config filename under config/cameras/. Unique WITHIN a school and
    # not beyond it: `hall_left` is a corridor in every school that has a corridor,
    # and the config directory that names it belongs to one installation of one
    # school. A global unique here would make the second school unable to install.
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    location: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    camera_type: Mapped[CameraType] = mapped_column(
        Enum(CameraType, native_enum=False), nullable=False
    )
    role: Mapped[CameraRole] = mapped_column(Enum(CameraRole, native_enum=False), nullable=False)

    rtsp_host: Mapped[str] = mapped_column(String(128), nullable=False)
    rtsp_port: Mapped[int] = mapped_column(Integer, default=554, nullable=False)

    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("school_id", "name", name="uq_cameras_school_name"),
        Index("ix_cameras_camera_type", "camera_type"),
        Index("ix_cameras_role", "role"),
    )

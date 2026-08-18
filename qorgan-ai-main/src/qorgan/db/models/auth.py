"""Dashboard users.

Legacy had ~50 endpoints, zero authentication, bound to 0.0.0.0. Anyone on the school
network could view children's photos and live video, and delete pupils (audit C-01).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.school import school_key
from qorgan.db.types import UtcDateTime
from qorgan.enums import UserRole


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # **NULL means "not any one school's": the суперадминистратор, who exists to manage
    # the schools and therefore cannot sit inside one.** Every other role must name a
    # school, and `test_only_a_superadmin_belongs_to_no_school` is what says so -- a
    # nullable column whose NULL has one meaning is fine, one whose NULL has two is the
    # defect migration 0005 was written about.
    school_id: Mapped[int | None] = school_key(nullable=True)

    # **Globally unique, deliberately: this is the one place multi-school does NOT
    # partition.** A login form carries no school, because the person typing has not been
    # identified yet and there is nothing to scope the lookup by. Per-school usernames
    # would mean either asking a headteacher which school they are before they prove who
    # they are, or two accounts answering to one name with the database picking -- and
    # `test_an_unknown_user_and_a_wrong_password_look_identical` already records what this
    # project thinks of a login that can be resolved two ways.
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # A hash. Never a password, and never a reversible token.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, native_enum=False), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

"""Column types that refuse to store the two things the legacy system got wrong.

`UtcDateTime` — legacy stored naive local-time ISO strings and compared them by
string prefix. Anything tz-naive is rejected here, at bind time.

`RelPath` — legacy stored absolute paths. The project then moved twice, and each
move broke 100% of the photo and clip rows and spawned another one-off fix script.
An absolute path cannot enter the database through this type (rule R6).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from sqlalchemy import DateTime, String
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC. Naive datetimes are a bug, not an input."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            raise ValueError(
                "naive datetime rejected: store timezone-aware UTC "
                "(datetime.now(UTC)), not local wall-clock time"
            )
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        # SQLite drops the tzinfo on the way in; put it back rather than hand the
        # application a naive datetime it will silently treat as local.
        return value if value.tzinfo else value.replace(tzinfo=UTC)


class RelPath(TypeDecorator[str]):
    """A media path relative to MEDIA_ROOT. Absolute paths and `..` are rejected."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        text = str(value).replace("\\", "/").strip()
        if not text:
            raise ValueError("media path must not be empty")
        if PurePosixPath(text).is_absolute() or PureWindowsPath(str(value)).is_absolute():
            raise ValueError(
                f"absolute path rejected: {value!r}. Store the path relative to MEDIA_ROOT "
                "so that moving the project does not break every row."
            )
        if ".." in PurePosixPath(text).parts:
            raise ValueError(f"path traversal rejected: {value!r}")
        return text

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        return value


def utcnow() -> datetime:
    return datetime.now(UTC)

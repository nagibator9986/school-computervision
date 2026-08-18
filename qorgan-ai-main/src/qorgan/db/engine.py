"""The database engine. SQLite in WAL mode, connections actually closed.

Legacy ran journal_mode=delete with no busy_timeout, opened a fresh connection on
every service call (including once per face, per frame), and never closed any of them:
they died when the garbage collector got round to it. Under load a writer blocked every
reader and the workers saw `database is locked` -- which then killed the event-writing
thread forever, because its loop had a `finally` and no `except` (audit H-01, H-07).

Going through SQLAlchemy means moving to PostgreSQL later is a URL change.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from qorgan.logging_setup import get_logger
from qorgan.settings import get_settings, resolve

logger = get_logger(__name__)

BUSY_TIMEOUT_MS = 10_000
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

T = TypeVar("T")


def _apply_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _prepare_sqlite_path(url: str) -> None:
    """SQLite will happily create an empty database in a directory that exists.
    It will not create the directory."""
    marker = ":///"
    if "sqlite" not in url or marker not in url:
        return
    raw = url.split(marker, 1)[1]
    if raw and raw != ":memory:":
        resolve(Path(raw)).parent.mkdir(parents=True, exist_ok=True)


def build_engine(url: str | None = None) -> Engine:
    target = url or get_settings().database_url
    _prepare_sqlite_path(target)

    engine = create_engine(
        target,
        future=True,
        pool_pre_ping=True,
        # Workers run their DB work on their own threads; SQLAlchemy's pool handles
        # the handing-out, and every connection is genuinely returned.
        connect_args={"timeout": BUSY_TIMEOUT_MS / 1000} if "sqlite" in target else {},
    )
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine


def get_engine() -> Engine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


def reset_engine() -> None:
    """Test hook, and used by workers after a fork/spawn."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success, roll back on failure, close either way. No exceptions."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def with_retry(operation: Callable[[], T], *, attempts: int = 4, delay: float = 0.25) -> T:
    """Retry a locked write instead of letting it kill the caller's thread.

    WAL plus a busy_timeout makes `database is locked` rare, not impossible. When it
    does happen it must be a retry, never a dead worker.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts:
                raise
            wait = delay * (2 ** (attempt - 1))
            logger.warning(
                "database locked, retrying", extra={"attempt": attempt, "wait_seconds": wait}
            )
            time.sleep(wait)
    raise AssertionError("unreachable")

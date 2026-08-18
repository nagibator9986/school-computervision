"""Workers report they are alive. The supervisor and the web UI both read it.

The heartbeat goes into the database rather than a shared memory structure, for two
reasons. It survives a supervisor restart, and it lets the dashboard show worker health
*without importing a worker module* — the legacy web layer imported CAMERA_REGISTRY
straight out of the bullying worker, so loading a web route pulled in YOLO and torch
as a side effect (audit M-23).
"""

from __future__ import annotations

import os
import threading

from sqlalchemy import select

from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.enums import WorkerState
from qorgan.logging_setup import get_logger
from qorgan.redaction import redact

logger = get_logger(__name__)


def write_heartbeat(
    group: str,
    state: WorkerState,
    *,
    frames_processed: int = 0,
    last_error: str | None = None,
    restarted: bool = False,
) -> None:
    """Upsert this worker's row. Never raises: a failed heartbeat must not kill a worker."""

    def _write() -> None:
        with session_scope() as session:
            row = session.scalar(select(WorkerHeartbeat).where(WorkerHeartbeat.group_name == group))
            if row is None:
                row = WorkerHeartbeat(group_name=group)
                session.add(row)
            row.pid = os.getpid()
            row.state = state
            row.last_seen_at = utcnow()
            row.frames_processed = frames_processed
            # R4: `last_error` is `f"{type(exc).__name__}: {exc}"` off a crashing worker
            # (worker/entrypoint.py), and what a worker's exceptions are about is the
            # RTSP URL — `rtsp://user:password@host/...`, carrying audit C-03's one
            # shared fleet camera password. That went into a column the dashboard
            # renders and the sqlite file carries. Scrub it here, at the write, with the
            # same redactor the log formatter uses — not at each read site, which only
            # ever covers the read sites somebody remembered.
            row.last_error = redact(last_error) if last_error else None
            if restarted:
                row.restart_count += 1

    try:
        with_retry(_write)
    except Exception:
        logger.exception("heartbeat write failed", extra={"group": group})


class Heartbeat:
    """Beats on a background thread for as long as the worker lives."""

    def __init__(self, group: str, interval_seconds: float = 5.0) -> None:
        self.group = group
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frames = 0
        self._error: str | None = None

    def record(self, *, frames: int | None = None, error: str | None = None) -> None:
        with self._lock:
            if frames is not None:
                self._frames = frames
            self._error = error

    def start(self) -> Heartbeat:
        write_heartbeat(self.group, WorkerState.STARTING)
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat:{self.group}", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frames, error = self._frames, self._error
            state = WorkerState.DEGRADED if error else WorkerState.RUNNING
            write_heartbeat(self.group, state, frames_processed=frames, last_error=error)
            self._stop.wait(self._interval)

    def stop(self, state: WorkerState = WorkerState.STOPPED) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None
        with self._lock:
            frames = self._frames
        write_heartbeat(self.group, state, frames_processed=frames)

    def __enter__(self) -> Heartbeat:
        return self.start()

    def __exit__(self, exc_type: type | None, *_rest: object) -> None:
        self.stop(WorkerState.CRASHED if exc_type else WorkerState.STOPPED)

"""Worker health and restart counts, from the table the supervisor writes.

Lives here rather than in `web.routes.cameras`, where it started, because two pages now
answer "how many times has this worker restarted": the camera wall and /logs. Two copies
of this query would drift, and the way they would drift is silently -- the school would
read one number on one page and a different number on the other, and neither page would
look wrong. One function, one answer.

Bounded WITHOUT pagination, which is deliberate and is not the legacy's whole-table
render: `worker_heartbeats.group_name` is UNIQUE, so this table has exactly one row per
worker group in `workers.yaml`. It is bounded by the installation's configuration and not
by how long the system has been running, which is the property that made the legacy's
event and canteen renders fatal (audit M-19).
"""

from __future__ import annotations

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models import WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.redaction import redact


def worker_rows() -> list[dict[str, object]]:
    """One row per worker group, oldest heartbeat and all."""
    now = utcnow()
    with session_scope() as session:
        rows = session.scalars(select(WorkerHeartbeat)).all()
        return [
            {
                "group": row.group_name,
                "state": row.state.value,
                "pid": row.pid,
                "restarts": row.restart_count,
                "seconds_since_beat": (
                    round((now - row.last_seen_at).total_seconds(), 1) if row.last_seen_at else None
                ),
                # REDACTED HERE, and this is not belt-and-braces. `last_error` is written
                # by `Heartbeat.record(error=f"{type(exc).__name__}: {exc}")` from an
                # arbitrary exception, and an RTSP failure carries the whole
                # `rtsp://user:password@host` URL. It goes straight into the database: it
                # never passes the log formatter, so `RedactingFormatter` -- the thing that
                # makes the log FILES safe -- does nothing for this column at all. The
                # legacy's C-02 was this exact string reaching a screen.
                "last_error": redact(row.last_error) if row.last_error else None,
            }
            for row in sorted(rows, key=lambda r: r.group_name)
        ]

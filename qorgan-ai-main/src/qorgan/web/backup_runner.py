"""Taking a backup without holding the HTTP request open.

**This module exists because of audit H-18.** The legacy dashboard restarted the AI
workers from inside a request handler and waited on a five-second `thread.join()` before
answering. A `VACUUM INTO` of the school's database is not five seconds — it is however
long it takes to write a full copy of a growing file to a spinning disk, seconds today
and minutes in a year — and a web worker blocked on it is serving nobody, including the
operator watching a corridor at that moment. Whoever presses the button would then press
it again, and the school would conclude the page is broken.

So the handler does exactly two things: it asks, and it answers. The work happens on a
thread, and the page reports what happened afterwards.

**One at a time, refused rather than queued.** A backup is a second full copy of the
database on the same disk that holds the recordings. Two of them in flight is twice the
write bandwidth and twice the space for no benefit — a double-click, an impatient
operator, or a browser retrying a POST must not multiply it. A queue would only defer the
same multiplication; a refusal is the honest answer, and the page says so.

**And that guarantee is exactly one process wide — say so rather than imply more.** This
lock knows about the web process. The 03:00 scheduled `qorgan backup` is a *different*
process and this cannot see it, so a button pressed at 03:00:30 does run alongside it.
That is survivable and was designed for: `default_destination` is dated to the second and
`backup_database` refuses to overwrite, so the two get different files and neither
corrupts the other — the cost is disk, not data. It would stop being survivable if the
dashboard were ever run with more than one uvicorn worker, because then "one at a time"
would silently mean "one per worker" while this docstring still claimed otherwise.
`cli._cmd_web` starts a single worker; if that changes, this guard has to move to
something both processes can see (the filesystem, or a row).

**The thread never dies of an exception** (rule R7). A backup that failed must leave a
message where the person who pressed the button will look for it, because the alternative
is a page that says nothing and a school that believes it has a backup.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from qorgan.db.types import utcnow
from qorgan.logging_setup import get_logger
from qorgan.maintenance import BackupError, backup_database

logger = get_logger(__name__)

# Shutdown only. Long enough that a backup in flight finishes and the file is complete,
# rather than the process exiting mid-VACUUM and leaving a half-written copy that looks
# like a backup in the listing.
SHUTDOWN_JOIN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the last backup did. `error` is a message for a human, and it is not HTML."""

    ok: bool
    finished_at: datetime
    name: str
    error: str


@dataclass(frozen=True, slots=True)
class RunnerState:
    running: bool
    started_at: datetime | None
    last: Outcome | None


class BackupRunner:
    """One backup at a time, on a thread that is not the request's.

    Held on `app.state`, like the notification worker: one per application, so that
    "is a backup running?" is a fact about the machine rather than about a request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._started_at: datetime | None = None
        self._last: Outcome | None = None
        self._idle = threading.Event()
        self._idle.set()

    def request(self, *, by: str) -> bool:
        """Start a backup. Returns False when one is already running.

        `_running` is set HERE, under the lock, before the thread is even started — not by
        the thread once it wakes up. Otherwise two POSTs arriving back to back would both
        see "not running" and both start, which is exactly the case this guard is for.
        """
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._started_at = utcnow()
            self._idle.clear()
            self._thread = threading.Thread(
                target=self._run, args=(by,), name="backup", daemon=True
            )
            self._thread.start()
            return True

    def state(self) -> RunnerState:
        """A snapshot, taken under the lock so a page never renders half a transition."""
        with self._lock:
            return RunnerState(
                running=self._running,
                started_at=self._started_at if self._running else None,
                last=self._last,
            )

    def wait(self, timeout: float) -> bool:
        """Block until no backup is running. **Never call this from a request handler** —
        that is precisely the five-second join this module exists to avoid. It is here for
        tests, which have to observe a finished backup, and for nothing else."""
        return self._idle.wait(timeout)

    def stop(self) -> None:
        """Shutdown, not a request. Called from the app's lifespan when the web process is
        going away, where waiting for a copy to finish is the correct thing to do."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=SHUTDOWN_JOIN_SECONDS)

    def _run(self, by: str) -> None:
        """The backup itself, off the request thread. Reuses `qorgan backup` whole.

        No destination is passed: `backup_database` picks the dated default beside the
        database and refuses to overwrite. A destination chosen here would be a second
        opinion about where backups live, and the page reads them back from the first one.
        """
        try:
            outcome = self._take(by)
            # Recorded BEFORE the waiter is released, deliberately. Anything that waits
            # for the backup to finish and then asks what happened must not be able to
            # observe "finished" with the previous answer still in place.
            with self._lock:
                self._last = outcome
        finally:
            with self._lock:
                self._running = False
            self._idle.set()

    def _take(self, by: str) -> Outcome:
        try:
            report = backup_database()
        except BackupError as exc:
            logger.error("backup from the dashboard failed", extra={"by": by, "error": str(exc)})
            return Outcome(ok=False, finished_at=utcnow(), name="", error=str(exc))
        except Exception as exc:  # R7: this thread does not die, it reports
            logger.exception("backup from the dashboard failed unexpectedly", extra={"by": by})
            return Outcome(
                ok=False, finished_at=utcnow(), name="", error=f"{type(exc).__name__}: {exc}"
            )

        logger.info(
            "backup taken from the dashboard", extra={"by": by, "summary": report.summary()}
        )
        # The NAME, not the path: the page lists this folder anyway, and a full filesystem
        # path on screen is one more place the machine's layout leaks into a shared room.
        return Outcome(ok=True, finished_at=utcnow(), name=report.destination.name, error="")

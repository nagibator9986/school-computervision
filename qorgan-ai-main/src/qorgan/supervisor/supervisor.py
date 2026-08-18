"""The supervisor: owns the process table, restarts what dies, kills what wedges.

This is the single most important reliability feature in the rewrite, and the legacy
system has no equivalent at all. There, workers were daemon threads inside the web
process; when one died — and one `database is locked` was enough — the thread was gone
for good, the dashboard still looked healthy, and the system quietly recorded nothing.

Two failure modes, two responses:

- **Crashed** (the process is gone): restart it, backing off if it keeps dying, so a
  camera with a bad config does not spin the CPU restarting forever.
- **Wedged** (the process is alive but has stopped beating): kill it, then restart it.
  A worker deadlocked on a decoder or an inference call is worse than a dead one,
  because nothing looks wrong.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from datetime import timedelta

from sqlalchemy import select

from qorgan.canteen.sessions import close_sessions_nobody_exited
from qorgan.classroom.store import close_stale_lessons
from qorgan.config.canteen import SessionRules
from qorgan.config.classroom import LessonRules
from qorgan.config.workers import WorkersConfig
from qorgan.db.engine import session_scope
from qorgan.db.models import WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.enums import WorkerState
from qorgan.logging_setup import get_logger
from qorgan.supervisor.heartbeat import write_heartbeat
from qorgan.supervisor.managed import (
    ManagedWorker,
    ProcessFactory,
    RestartPolicy,
    spawn_worker_process,
)

logger = get_logger(__name__)

TICK_SECONDS = 1.0

# The stale-session sweep's own cadence, because `tick()` is 1 s and the deadline the
# sweep enforces is `max_session_minutes` — 90 by default. `canteen_sessions` is never
# pruned and `state != CLOSED` is not a constraint SQLite can take an index on, so each
# sweep walks the whole year's sessions; doing that 86 400 times a day to answer a
# question that only changes every ninety minutes is scans nobody reads. This is a cost
# knob and not a domain one: any value well under `max_session_minutes` force-closes the
# same sessions on the same day, and a session may linger at most this long past the
# deadline before it is noticed.
SWEEP_SECONDS = 60.0


class Supervisor:
    def __init__(
        self,
        config: WorkersConfig,
        *,
        factory: ProcessFactory = spawn_worker_process,
        tick_seconds: float = TICK_SECONDS,
        session_rules: SessionRules | None = None,
        lesson_rules: LessonRules | None = None,
    ) -> None:
        policy = RestartPolicy(
            delay_seconds=config.restart_backoff_seconds,
            max_delay_seconds=config.restart_backoff_max_seconds,
        )
        self.config = config
        self.workers = [
            ManagedWorker(group=group, policy=policy, factory=factory) for group in config.groups
        ]
        self._tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._session_rules = session_rules
        self._lesson_rules = lesson_rules
        # None, not 0.0: a supervisor that has just come up sweeps on its first tick.
        # Sessions go stale precisely while nobody is running, and a monotonic clock
        # starts near zero -- a 0.0 sentinel would skip that first sweep.
        self._last_sweep: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Block until stopped. Ctrl-C and SIGTERM shut the fleet down cleanly."""
        self._install_signal_handlers()
        logger.info(
            "supervisor starting",
            extra={"groups": [w.name for w in self.workers]},
        )
        try:
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(self._tick_seconds)
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        logger.info("supervisor stopping, shutting down workers")
        for worker in self.workers:
            worker.stop()

    def _install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: object) -> None:
            logger.info("received signal, shutting down", extra={"signal": signum})
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            # Not on the main thread (tests): nothing to install, and nothing to fix.
            with contextlib.suppress(ValueError):
                signal.signal(sig, handle)

    # -- the loop ----------------------------------------------------------

    def tick(self) -> None:
        """One pass over the process table. Safe to call directly from a test."""
        stale = self._stale_heartbeats()
        for worker in self.workers:
            try:
                self._supervise(worker, stale)
            except Exception:
                logger.exception("supervision failed", extra={"group": worker.name})
        # ONE timer, both sweeps. `_sweep_due` CONSUMES the interval, so leaving each
        # sweep to ask for itself would let the first one take the tick and starve the
        # second -- which is the shape of the bug that left the meal sweep unreachable in
        # the first place. Asked once, here, and the answer given to both.
        if self._sweep_due():
            self._sweep_stale_sessions()
            self._sweep_stale_lessons()

    def _sweep_stale_sessions(self) -> None:
        """Close the meal sessions no exit camera ever closed.

        A child the exit camera never recognised keeps `state != CLOSED` forever, and
        `SessionManager.open` answers `already_inside` to every meal they come to after
        that: they silently stop being recorded as fed. `max_session_minutes` is what is
        meant to end that, and until this call existed nothing in `src/` ever ran it.

        **Here, and not in the canteen worker, because a worker cannot be alone with it.**
        `workers.yaml` puts the canteen in TWO groups today (`canteen_corridor` and
        `canteen_inside`), and `worker/entrypoint.py` builds a `SessionManager` in each —
        so a sweep there would be two processes closing the same rows, each stamping its
        own `closed_at`, each logging a count that is not the number of children lost.
        `max_session_minutes` is one fact about one canteen, not one per process. The
        supervisor is the only singleton by construction: it owns the process table, so a
        second one is not a config choice but a broken install. It is also the process
        still up while the group that lost the child is crash-looping — which is the one
        moment a sweep living inside that group would not run.

        Never raises. A supervisor that dies of a sweep stops restarting workers (R7).
        """
        if self._session_rules is None:
            return  # an installation with no canteen cameras has no meal sessions
        try:
            close_sessions_nobody_exited(self._session_rules)
        except Exception:
            logger.exception("the stale meal-session sweep failed")

    def _sweep_stale_lessons(self) -> None:
        """Close the lessons nobody ended, for the same reasons and in the same place.

        A lesson left OPEN is not merely untidy: `classroom.store.open_or_resume` re-
        attaches to the newest open lesson on this camera, so tomorrow morning's tracks
        would be counted into yesterday's lesson, under a `started_at` from the day
        before, and the report would describe two rooms at once.

        Here rather than in the classroom worker for the reason the meal sweep is here: a
        worker cannot be alone with it. It is also the process still running while the
        group that abandoned the lesson is crash-looping -- which is exactly when a lesson
        gets abandoned, and exactly when a sweep living in that group would not run.

        Never raises. A supervisor that dies of a sweep stops restarting workers (R7).
        """
        if self._lesson_rules is None:
            return  # an installation with no classroom cameras has no lessons
        try:
            close_stale_lessons(self._lesson_rules)
        except Exception:
            logger.exception("the stale lesson sweep failed")

    def _sweep_due(self) -> bool:
        now = time.monotonic()
        if self._last_sweep is not None and now - self._last_sweep < SWEEP_SECONDS:
            return False
        self._last_sweep = now
        return True

    def _supervise(self, worker: ManagedWorker, stale: set[str]) -> None:
        if worker.is_alive():
            if worker.name in stale and worker.uptime() > worker.group.heartbeat_timeout_seconds:
                self._kill_wedged(worker)
            return

        if worker.process is not None:
            worker.note_exit()  # it was running, and now it is not
        if worker.may_start():
            # `note_exit` already counted this death, so a pending restart shows as
            # restarts > 0 before the new process is up.
            restarting = worker.restarts > 0
            worker.start()
            if restarting:
                # The supervisor is the only process that knows a restart happened; the
                # dashboard reads restart_count from the DB. Without this a crash-looping
                # camera reports restarts: 0 and looks perfectly healthy (the STARTING beat
                # the child writes on boot carries restarted=False and never bumps it).
                write_heartbeat(worker.name, WorkerState.STARTING, restarted=True)

    def _kill_wedged(self, worker: ManagedWorker) -> None:
        """Alive, but not beating. Worse than dead, because nothing looks wrong."""
        logger.error(
            "worker is wedged (no heartbeat), killing it",
            extra={
                "group": worker.name,
                "timeout_seconds": worker.group.heartbeat_timeout_seconds,
                "pid": worker.process.pid if worker.process else None,
            },
        )
        worker.stop()
        worker.note_exit()

    def _stale_heartbeats(self) -> set[str]:
        """Groups whose last heartbeat is older than their timeout.

        A group that has never beaten at all is NOT stale: it may still be loading its
        models, which takes seconds. `uptime()` in _supervise is what covers that.
        """
        now = utcnow()
        stale: set[str] = set()
        with session_scope() as session:
            rows = session.scalars(select(WorkerHeartbeat)).all()
            beats = {row.group_name: (row.last_seen_at, row.state) for row in rows}

        for worker in self.workers:
            last_seen, state = beats.get(worker.name, (None, None))
            if last_seen is None or state is WorkerState.STOPPED:
                continue
            if now - last_seen > timedelta(seconds=worker.group.heartbeat_timeout_seconds):
                stale.add(worker.name)
        return stale

    # -- introspection -----------------------------------------------------

    def status(self) -> dict[str, dict[str, object]]:
        return {
            worker.name: {
                "alive": worker.is_alive(),
                "restarts": worker.restarts,
                "uptime_seconds": round(worker.uptime(), 1),
                "cameras": worker.group.cameras,
            }
            for worker in self.workers
        }

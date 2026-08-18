"""One supervised worker process, with its restart policy."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from qorgan.config.workers import WorkerGroup
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

TERMINATE_GRACE_SECONDS = 10.0


class Process(Protocol):
    """The slice of multiprocessing.Process we depend on, so tests can substitute a fake."""

    pid: int | None
    exitcode: int | None

    def start(self) -> None: ...
    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...


# Builds the process that runs one group. Injectable for tests.
ProcessFactory = Callable[[WorkerGroup], Process]


def spawn_worker_process(group: WorkerGroup) -> Process:
    from qorgan.worker.entrypoint import run_group

    # Windows has no fork, so the target must be a module-level function and the
    # argument must be picklable. The group NAME travels, not the config object:
    # the child re-reads and re-validates config for itself.
    context = mp.get_context("spawn")
    return context.Process(target=run_group, args=(group.name,), name=f"worker:{group.name}")


@dataclass
class RestartPolicy:
    delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    # A worker that survives this long is considered healthy again, and its backoff resets.
    healthy_after_seconds: float = 60.0


@dataclass
class ManagedWorker:
    """A worker group, the process running it, and when it may next be restarted."""

    group: WorkerGroup
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    factory: ProcessFactory = spawn_worker_process

    process: Process | None = None
    restarts: int = 0
    started_at: float | None = None
    not_before: float = 0.0
    _backoff: float = 0.0

    @property
    def name(self) -> str:
        return self.group.name

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def uptime(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def start(self) -> None:
        self.process = self.factory(self.group)
        self.process.start()
        self.started_at = time.monotonic()
        logger.info(
            "worker started",
            extra={"group": self.name, "pid": self.process.pid, "cameras": self.group.cameras},
        )

    def note_exit(self) -> None:
        """The process is gone. Schedule the restart, backing off if it keeps dying."""
        exitcode = self.process.exitcode if self.process else None
        lived = self.uptime()

        if lived >= self.policy.healthy_after_seconds:
            self._backoff = self.policy.delay_seconds  # it was healthy; start over
        else:
            self._backoff = min(
                max(self._backoff * 2, self.policy.delay_seconds), self.policy.max_delay_seconds
            )

        self.restarts += 1
        self.not_before = time.monotonic() + self._backoff
        self.process = None
        self.started_at = None

        logger.error(
            "worker died, will restart",
            extra={
                "group": self.name,
                "exitcode": exitcode,
                "lived_seconds": round(lived, 1),
                "restart_in_seconds": round(self._backoff, 1),
                "restarts": self.restarts,
            },
        )

    def may_start(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) >= self.not_before

    def stop(self, grace: float = TERMINATE_GRACE_SECONDS) -> None:
        """Ask nicely, then insist. A wedged worker must not block shutdown."""
        if self.process is None or not self.process.is_alive():
            self.process = None
            return

        self.process.terminate()
        self.process.join(timeout=grace)

        if self.process.is_alive():
            logger.warning("worker ignored terminate, killing", extra={"group": self.name})
            self.process.kill()
            self.process.join(timeout=grace)

        self.process = None
        self.started_at = None

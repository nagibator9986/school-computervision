"""The supervisor. Legacy has no equivalent: its workers were daemon threads inside
the web process, and one `database is locked` killed the event thread forever while
the dashboard went on looking perfectly healthy.
"""

from __future__ import annotations

import argparse
import ast
import re
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.canteen.reports import day_report
from qorgan.canteen.sessions import SessionManager
from qorgan.cli import _cmd_supervisor
from qorgan.config.canteen import MealOutcomeRules, SessionRules
from qorgan.config.workers import WorkerGroup, WorkersConfig
from qorgan.db.models import Camera, CanteenSession, Person, WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.enums import (
    CameraRole,
    CameraType,
    CloseReason,
    PersonType,
    SessionOutcome,
    SessionState,
    WorkerState,
)
from qorgan.settings import Settings, get_settings
from qorgan.supervisor.heartbeat import write_heartbeat
from qorgan.supervisor.managed import ManagedWorker, RestartPolicy
from qorgan.supervisor.supervisor import Supervisor
from tests.conftest import SRC_DIR


class FakeProcess:
    """A process we can kill on demand, without spawning anything."""

    def __init__(self, group: WorkerGroup) -> None:
        self.group = group
        self.pid = 4242
        self.exitcode: int | None = None
        self._alive = False
        self.terminated = False
        self.killed = False
        self.ignores_terminate = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def die(self, exitcode: int = 1) -> None:
        self._alive = False
        self.exitcode = exitcode

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_terminate:
            self.die(-15)

    def kill(self) -> None:
        self.killed = True
        self.die(-9)

    def join(self, timeout: float | None = None) -> None:
        pass


class Spawner:
    """Records every process it hands out, so a test can reach in and kill one."""

    def __init__(self) -> None:
        self.made: list[FakeProcess] = []

    def __call__(self, group: WorkerGroup) -> FakeProcess:
        process = FakeProcess(group)
        self.made.append(process)
        return process


def _config(**kwargs: object) -> WorkersConfig:
    return WorkersConfig(
        groups=[
            {"name": "bullying_hall", "cameras": ["hall_left"], "heartbeat_timeout_seconds": 30.0}
        ],
        restart_backoff_seconds=0.01,
        restart_backoff_max_seconds=0.08,
        **kwargs,
    )


# -- starting and restarting ----------------------------------------------


def test_the_first_tick_starts_every_group(session: Session) -> None:
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)

    supervisor.tick()

    assert len(spawner.made) == 1
    assert spawner.made[0].is_alive()
    assert supervisor.status()["bullying_hall"]["alive"] is True


def test_a_crashed_worker_is_restarted(session: Session) -> None:
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)

    supervisor.tick()
    spawner.made[0].die(exitcode=1)
    supervisor.tick()  # notices the death, schedules the restart
    _wait_out_backoff(supervisor)
    supervisor.tick()  # restarts it

    assert len(spawner.made) == 2, "it did not come back"
    assert spawner.made[1].is_alive()
    assert supervisor.workers[0].restarts == 1


def test_a_restarted_worker_reports_a_nonzero_restart_count_to_the_dashboard(
    session: Session,
) -> None:
    """A crash-looping camera is the one an operator most needs to see. The supervisor
    knows it restarted (`worker.restarts`), but the dashboard reads `restart_count` from
    the DB -- and nothing ever wrote it there. So a worker dying every few seconds showed
    `restarts: 0` and looked perfectly healthy.
    """
    # The one query behind both pages that answer "how many restarts": the camera wall and
    # /logs. It moved out of `web.routes.cameras` when the second one needed it -- two
    # copies would drift, and the school would read two different numbers and see nothing
    # wrong with either page.
    from qorgan.diagnostics.workers import worker_rows

    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)

    supervisor.tick()  # first start
    write_heartbeat("bullying_hall", WorkerState.RUNNING)  # the worker's own beat: restart_count 0
    spawner.made[0].die(exitcode=1)
    supervisor.tick()  # notices the death, schedules the restart
    _wait_out_backoff(supervisor)
    supervisor.tick()  # restarts it

    assert supervisor.workers[0].restarts == 1  # the supervisor knows

    # ...and the dashboard, which reads the DB, must know it too.
    row = {r["group"]: r for r in worker_rows()}["bullying_hall"]
    assert row["restarts"] == 1, "the dashboard still reads restarts: 0 for a worker that restarted"


def test_a_worker_that_keeps_dying_is_backed_off(session: Session) -> None:
    """Otherwise a camera with a bad config spins the CPU restarting forever.

    Asserts on the backoff DELAY, not on the wall-clock deadline: `not_before` is a
    monotonic timestamp, so differencing it measures how long the test itself took as
    much as it measures the policy.
    """
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)
    worker = supervisor.workers[0]

    backoffs = []
    for _ in range(4):
        supervisor.tick()
        if worker.process is not None:
            worker.process.die()
        supervisor.tick()  # notices the death and schedules the restart
        backoffs.append(worker._backoff)
        _wait_out_backoff(supervisor)

    assert worker.restarts >= 3
    assert backoffs == sorted(backoffs), f"the backoff did not grow: {backoffs}"
    assert backoffs[-1] > backoffs[0], "the backoff never actually increased"


def test_the_backoff_is_capped(session: Session) -> None:
    worker = ManagedWorker(
        group=WorkerGroup(name="g_one", cameras=["hall_left"]),
        policy=RestartPolicy(delay_seconds=1.0, max_delay_seconds=4.0),
        factory=Spawner(),
    )
    for _ in range(10):
        worker.start()
        worker.process.die()  # type: ignore[union-attr]
        worker.note_exit()

    assert worker._backoff <= 4.0


# -- the wedged worker: alive, but not beating ----------------------------


def test_a_wedged_worker_is_killed_and_restarted(session: Session) -> None:
    """A worker deadlocked on a decoder or an inference call is WORSE than a dead one:
    the process is up, so nothing looks wrong, and the camera sees nothing."""
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)
    supervisor.tick()
    wedged = spawner.made[0]

    _beat(session, "bullying_hall", age_seconds=120)  # last heard from 2 minutes ago
    supervisor.workers[0].started_at -= 120  # and it has been up long enough to know better

    supervisor.tick()

    assert wedged.terminated, "the wedged worker was left running"
    assert not wedged.is_alive()
    assert supervisor.workers[0].restarts == 1


def test_a_wedged_worker_that_ignores_terminate_is_killed(session: Session) -> None:
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)
    supervisor.tick()
    stubborn = spawner.made[0]
    stubborn.ignores_terminate = True

    _beat(session, "bullying_hall", age_seconds=120)
    supervisor.workers[0].started_at -= 120
    supervisor.tick()

    assert stubborn.killed, "a worker that ignores SIGTERM must be killed"


def test_a_worker_still_loading_its_models_is_not_killed(session: Session) -> None:
    """Loading YOLO and InsightFace takes seconds. A worker that has not beaten YET
    is not a wedged worker."""
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)

    supervisor.tick()  # started; no heartbeat row exists at all
    supervisor.tick()

    assert spawner.made[0].is_alive(), "it was killed before it ever got going"
    assert supervisor.workers[0].restarts == 0


def test_a_beating_worker_is_left_alone(session: Session) -> None:
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)
    supervisor.tick()

    _beat(session, "bullying_hall", age_seconds=0)
    supervisor.workers[0].started_at -= 120
    supervisor.tick()

    assert spawner.made[0].is_alive()
    assert not spawner.made[0].terminated
    assert len(spawner.made) == 1


# -- shutdown --------------------------------------------------------------


def test_shutdown_stops_every_worker(session: Session) -> None:
    spawner = Spawner()
    supervisor = Supervisor(_config(), factory=spawner)
    supervisor.tick()

    supervisor.shutdown()

    assert all(not p.is_alive() for p in spawner.made)
    assert spawner.made[0].terminated


def test_supervision_of_one_group_does_not_take_down_the_others(session: Session) -> None:
    """The supervisor itself must never die (rule R7)."""

    class Exploding(Spawner):
        def __call__(self, group: WorkerGroup) -> FakeProcess:
            if group.name == "bad":
                raise RuntimeError("cannot spawn")
            return super().__call__(group)

    config = WorkersConfig(
        groups=[
            {"name": "bad", "cameras": ["hall_left"]},
            {"name": "good", "cameras": ["hall_right"]},
        ],
        restart_backoff_seconds=0.01,
    )
    supervisor = Supervisor(config, factory=Exploding())

    supervisor.tick()  # must not raise

    assert supervisor.status()["good"]["alive"] is True


# -- heartbeats ------------------------------------------------------------


def test_a_heartbeat_is_upserted_not_duplicated(session: Session) -> None:
    write_heartbeat("bullying_hall", WorkerState.RUNNING, frames_processed=10)
    write_heartbeat("bullying_hall", WorkerState.RUNNING, frames_processed=25)

    rows = session.scalars(select(WorkerHeartbeat)).all()
    assert len(rows) == 1
    assert rows[0].frames_processed == 25
    assert rows[0].pid is not None


def test_a_failing_heartbeat_does_not_kill_the_worker(session: Session, monkeypatch) -> None:
    """A worker that cannot write its heartbeat is degraded, not dead."""

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr("qorgan.supervisor.heartbeat.with_retry", explode)
    write_heartbeat("bullying_hall", WorkerState.RUNNING)  # must not raise


def test_the_deletion_record_names_the_cadence_the_worker_actually_beats_at() -> None:
    """`heartbeat_interval_seconds` was deleted, and its record must stay true.

    The record in `config/workers.py` says what reads the value instead -- a fixed
    literal in `worker/entrypoint.py`. It named the wrong one: "a fixed 1 s", where the
    code passes 5.0. That is this codebase's disease in miniature: a comment asserting
    what the code does not do, and nothing to say no. The record is a claim about a
    literal, so it is checkable, so it is checked.
    """
    entrypoint = ast.parse((SRC_DIR / "qorgan" / "worker" / "entrypoint.py").read_text("utf-8"))
    cadences = [
        keyword.value.value
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Heartbeat"
        for keyword in node.keywords
        if keyword.arg == "interval_seconds" and isinstance(keyword.value, ast.Constant)
    ]
    assert cadences == [5.0], f"worker/entrypoint.py beats at {cadences}, not a single 5.0"

    record = (SRC_DIR / "qorgan" / "config" / "workers.py").read_text("utf-8")
    stated = re.search(r"heartbeat_interval_seconds DELETED.*?fixed ([\d.]+) s", record, re.S)
    assert stated, "the deletion record no longer states the cadence it points at"
    assert float(stated.group(1)) == cadences[0], (
        f"the record says the worker beats every {stated.group(1)} s; entrypoint.py "
        f"passes {cadences[0]}. Deleting a key is fine. Misdescribing what replaced it "
        f"is how the next person learns to distrust every comment in the file."
    )


# -- the meal session nobody ever closed -----------------------------------


@pytest.fixture
def canteen(session: Session) -> SessionManager:
    """A camera row for a session to point at, and the machine that opens one."""
    entry = Camera(
        name="canteen_entry",
        display_name="Вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(entry)
    session.commit()
    return SessionManager(SessionRules(), MealOutcomeRules(), entry.id)


@pytest.fixture
def pupil(session: Session) -> Person:
    person = Person(
        external_id="gen-alice",
        full_name="Петрова Мария",
        person_type=PersonType.STUDENT,
    )
    session.add(person)
    session.commit()
    return person


def test_a_pupil_whose_exit_was_never_recognised_can_eat_again_tomorrow(
    session: Session, canteen: SessionManager, pupil: Person
) -> None:
    """The child this sweep exists for, driven through the path that runs in production.

    `force_close_stale` always worked when a test called it by hand, and that is exactly
    why the defect survived: nothing in `src/` called it. A pupil the exit camera never
    recognised kept `state != CLOSED` for the rest of the school year, `_active_for` kept
    finding it, and every future `open()` answered `already_inside` — the child silently
    stopped being recorded as fed. So this drives `Supervisor.tick()`, not the method.
    """
    stale = canteen.open(pupil.id, utcnow() - timedelta(minutes=120))
    assert stale.opened

    Supervisor(_config(), factory=Spawner(), session_rules=SessionRules()).tick()

    session.expire_all()
    row = session.get(CanteenSession, stale.session_id)
    assert row.state is SessionState.CLOSED, "nothing in production closes a stale session"
    assert row.outcome is SessionOutcome.UNKNOWN
    assert row.close_reason is CloseReason.TIMEOUT

    tomorrow = canteen.open(pupil.id)
    assert tomorrow.opened, f"the child is still blocked from every future meal: {tomorrow.reason}"


def test_the_forced_unknown_instrument_moves_off_zero(
    session: Session, canteen: SessionManager, pupil: Person
) -> None:
    """`DayReport.forced_unknown` is the number the exit threshold was argued on.

    `config/canteen.py` justifies the strict exit `min_score = 0.50` by promising the
    sessions it costs us are counted as `forced_unknown` — "if that number spikes, the
    threshold is too high, and we will SEE it rather than guess". Nothing in `src/` ever
    produced a TIMEOUT close, so the count was 0 by construction: a threshold chosen on
    the promise of an instrument that could not move. It has to move because the
    supervisor ticked, not because a test reached in and closed the session itself.
    """
    opened_at = utcnow() - timedelta(minutes=120)
    canteen.open(pupil.id, opened_at)
    day = opened_at.astimezone(get_settings().tz).date()
    assert day_report(day).forced_unknown == 0

    Supervisor(_config(), factory=Spawner(), session_rules=SessionRules()).tick()

    assert day_report(day).forced_unknown == 1, "the instrument still reads zero"


def test_the_supervisor_command_hands_the_sweep_the_real_canteen_rules(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two tests above build the supervisor themselves, so they cannot see this.

    `qorgan supervisor` is what actually runs, and a sweep it never passes rules to is
    the same defect one level up: `_sweep_stale_sessions` would return on its first line
    forever, every test above would stay green, and `forced_unknown` would go back to
    being 0 by construction. So the wiring is asserted against the real `config/`, which
    is the only place `max_session_minutes` is a fact rather than a default.
    """
    built: dict[str, object] = {}

    class Recording:
        def __init__(self, _config: WorkersConfig, **kwargs: object) -> None:
            built.update(kwargs)

        def run(self) -> None:
            """The real one blocks until SIGTERM."""

    monkeypatch.setattr("qorgan.supervisor.Supervisor", Recording)

    _cmd_supervisor(argparse.Namespace())

    rules = built.get("session_rules")
    assert isinstance(rules, SessionRules), "qorgan supervisor starts a sweep that cannot sweep"
    assert rules.max_session_minutes == 90.0


# -- helpers ---------------------------------------------------------------


def _beat(session: Session, group: str, *, age_seconds: float) -> None:
    write_heartbeat(group, WorkerState.RUNNING)
    session.commit()
    row = session.scalars(select(WorkerHeartbeat).where(WorkerHeartbeat.group_name == group)).one()
    row.last_seen_at = utcnow() - timedelta(seconds=age_seconds)
    session.commit()


def _wait_out_backoff(supervisor: Supervisor) -> None:
    for worker in supervisor.workers:
        worker.not_before = 0.0


@pytest.fixture(autouse=True)
def _quiet_logs(settings) -> None:
    """The supervisor logs an error on every simulated death; that is expected here."""

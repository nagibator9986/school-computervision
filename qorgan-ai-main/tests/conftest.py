from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session

from qorgan.db.engine import build_engine, reset_engine
from qorgan.db.models import Base, School
from qorgan.settings import Settings, _cached_settings, override_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SRC_DIR = REPO_ROOT / "src"


def _assert_testing_this_checkout() -> None:
    """The code under test must be the code in THIS checkout.

    `pip install -e .` writes an absolute path into a .pth file. In a git worktree that
    path still points at the ORIGINAL clone -- so `pytest` here would import, and pass
    against, a completely different tree from the one being edited. Nothing fails; the
    tests are simply green about the wrong source. Two parallel worktrees would silently
    grade each other's homework.

    PYTHONPATH fixes it. This makes forgetting to set PYTHONPATH a loud error instead of
    a quiet lie -- because an instruction is not a check.
    """
    import qorgan

    loaded = Path(qorgan.__file__).resolve()
    if SRC_DIR not in loaded.parents:
        raise RuntimeError(
            f"pytest is importing qorgan from {loaded}\n"
            f"but this checkout is           {SRC_DIR}\n\n"
            "You are testing a different tree from the one you are editing. Run pytest "
            f'with:  PYTHONPATH="{SRC_DIR}"'
        )


_assert_testing_this_checkout()


def _point_the_default_database_somewhere_harmless() -> None:
    """Make the DEFAULT database harmless for the whole run, not just inside a test.

    Every test's `settings` fixture points at its own throwaway file, so while a test is
    running nothing can reach the developer's database. **The damage happens between
    tests.** `qorgan.db.engine` keeps the engine in a module global built lazily from
    whatever `get_settings()` returns at the moment of the first call, and a test's
    teardown leaves BOTH the override cleared and the engine `None`. Any thread still
    alive in that gap -- the notifier is a daemon and its `stop()` joins with a 5 s
    timeout, so a slow drain outlives it -- calls `session_scope()`, gets the shipped
    defaults, and `_prepare_sqlite_path` *creates the directory*: a real
    `<worktree>/data/qorgan.sqlite3` appears, and the module global points at it for every
    later caller in the process.

    That produced two different false reds on 2026-07-24 -- `no such table: events` and
    `database is locked` -- in tests that had nothing to do with the cause, and both were
    blamed on the branch under test before being traced.

    **Called at conftest IMPORT time, and that is the whole trick.** `get_settings()` is
    `_OVERRIDE or _cached_settings()`, and `_cached_settings` is `@lru_cache(maxsize=1)`:
    the first call anywhere in the process freezes the defaults forever. A session-scoped
    autouse fixture is too late -- that was tried, and the value had already been cached
    during collection, so the stray database still appeared. conftest.py is imported
    before any test module, which is the earliest hook there is; the cache is cleared as
    well, so an import that got in first is corrected rather than trusted.

    **Why this and not a reordering of the fixture teardown.** Measured: while the old
    engine object is still alive, `session_scope()` keeps using it and never consults the
    missing settings. The dangerous state is not *between* `override_settings(None)` and
    `reset_engine()`, it is *after both* -- so swapping those two lines narrows nothing.
    Nor is "join every stray thread" enough: that only covers threads somebody remembered
    to name. Moving the default makes the unsafe outcome unreachable whatever the timing.
    """
    safe = Path(tempfile.mkdtemp(prefix="qorgan-tests-")) / "never-the-real-one.sqlite3"
    os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{safe.as_posix()}"
    _cached_settings.cache_clear()


_point_the_default_database_somewhere_harmless()


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    """Real Settings, pointed at a throwaway directory. Never touches the dev database."""
    value = Settings(
        # _env_file=None: only .env.example exists today, so this and every raw
        # Settings(...) in the suite is hermetic by luck, not by guarantee. The
        # fields below are already overridden for exactly this reason, but the
        # ones that are not (secret_key, web_port, ...) would silently start
        # reading a real .env the day one is created, as HANDOFF tells developers
        # to do. _env_file=None makes every test build Settings from defaults and
        # these explicit kwargs only, regardless of what is on disk.
        _env_file=None,
        database_url=f"sqlite+pysqlite:///{(tmp_path / 'test.sqlite3').as_posix()}",
        media_root=tmp_path / "media",
        log_dir=tmp_path / "logs",
        config_dir=CONFIG_DIR,
        rtsp_user="admin",
        rtsp_password=SecretStr("sup3r-s3cret-camera-pw"),
        telegram_bot_token=SecretStr("123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        telegram_chat_id="-1001234567890",
        log_json=False,
        # Port 0: PreviewSubscriber asks the OS for a free port and binds it in one
        # call (see qorgan.preview.subscriber._bind). The production default is a
        # fixed port, which two test suites -- or even two tests -- cannot both bind
        # at once; nothing here needs that specific port, only *a* free one.
        preview_address="tcp://127.0.0.1:0",
    )
    override_settings(value)
    # RESET AT SETUP TOO, not only at teardown -- and this line is the whole fix for a
    # flake that reddened three unrelated branches' runs.
    #
    # `get_engine()` binds a MODULE GLOBAL lazily from whatever `get_settings()` says at
    # the instant of the first call. Between one test's teardown and the next test's first
    # database call, the override is gone and `_engine` is None; a leaked daemon polling
    # `session_scope()` in that gap binds the global to the DEFAULT database, which is
    # harmless (see `_point_the_default_database_somewhere_harmless`) but has NO SCHEMA.
    # Nothing in the next test's setup cleared it, so that test's production code paths
    # answered from the poisoned global and died with `no such table: events` / `users` --
    # looking exactly like a bug in whatever branch was under review.
    #
    # Resetting here means a test begins from its own settings whatever a previous test or
    # its ghosts left behind. Deliberately NOT "find and stop every leaking thread": that
    # only ever covers the threads somebody remembered, and this covers the ones nobody has
    # found yet. See `tests/test_engine_isolation.py`.
    reset_engine()
    yield value
    override_settings(None)
    reset_engine()


DEFAULT_SCHOOL_SLUG = "default"


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    """An empty database with the full schema, and the one school that owns it.

    The school row is here rather than in each test because it is what migration 0009
    creates for a real installation: every existing deployment serves exactly one school,
    and a database with none is not a state this system is ever in. Without it every
    `Camera(...)` and `Person(...)` in the suite would have to name a school that has only
    one possible value -- eighty edits saying nothing.

    **It is one school, not two, and that matters.** `school_id` falls back to "the only
    school there is" and RAISES when there are several (`db/models/school.py`), so a test
    that wants to prove isolation has to create its second school itself and name the
    school at every call -- which is exactly what `test_tenancy_isolation` does, and why
    it cannot accidentally pass on a fixture's leniency.
    """
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        db_session.add(School(slug=DEFAULT_SCHOOL_SLUG, name="Тестовая школа"))
        db_session.commit()
        yield db_session
    engine.dispose()


# Every thread this project starts, by the prefix it is named with. The list is checked
# against `src/` by `tests/test_thread_guard.py`, so a new kind of worker cannot be added
# without this guard learning about it -- an allow-list nobody maintains is an allow-list
# that quietly stops covering things.
WORKER_THREAD_PREFIXES = (
    "backup",
    "camera:",
    "heartbeat:",
    "loop:",
    "notifier",
    "preview-sub",
    "validate:",
    # `worker/weapons.py` -- the thread that writes an alert's snapshot, clip and row off
    # the frame loop. Same shape as `validate:` and named separately so that a leak names
    # the pipeline it came from rather than a generic worker.
    "weapon-alert:",
)

# How long a stopping thread may take to actually die before we call it a leak. `stop()`
# on these joins with its own timeout, so by the time a test's fixtures have torn down the
# thread is normally already gone; this is slack for a scheduler, not a retry loop.
THREAD_SHUTDOWN_GRACE_SECONDS = 3.0


def _live_worker_threads() -> dict[int, str]:
    return {
        thread.ident: thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name.startswith(WORKER_THREAD_PREFIXES)
    }


@pytest.fixture(autouse=True)
def _no_worker_thread_outlives_its_test() -> Iterator[None]:
    """**No test may leave a background worker running.**

    This is the invariant, and it is worth more than any number of repeated runs: a
    deterministic check on the CAUSE beats sampling for a symptom whose appearance is
    decided by a background poll's timing.

    What it is for, concretely. A leaked `NotificationWorker` polls `session_scope()`
    forever. `get_engine()` binds the module-global engine lazily from whatever settings
    are installed at that instant, so a poll landing at the wrong moment points the global
    at a database that is not the running test's -- and the NEXT test to authenticate dies
    with `no such table: users`, in a file with nothing to do with the cause. That defect
    reached `main` twice, was "fixed" once at the wrong layer, and was finally caught by
    the owner running the suite by hand rather than by anything in here.

    Autouse and dependency-free ON PURPOSE: pytest sets such a fixture up before the
    others and finalises it after them, so this runs once every other teardown has had its
    chance to stop what it started.

    A failure here is not this guard being fussy. It names a test that started something
    and walked away, and the thread it names is still running now.
    """
    before = _live_worker_threads()
    yield

    deadline = time.monotonic() + THREAD_SHUTDOWN_GRACE_SECONDS
    leaked = {k: v for k, v in _live_worker_threads().items() if k not in before}
    while leaked and time.monotonic() < deadline:
        time.sleep(0.05)
        leaked = {k: v for k, v in _live_worker_threads().items() if k not in before}

    assert not leaked, (
        f"this test left {len(leaked)} worker thread(s) running: {sorted(leaked.values())}. "
        "A leaked worker keeps calling session_scope(), which rebinds the module-global "
        "engine and makes a LATER test fail with `no such table: ...` -- so the red lands "
        "on somebody else. Stop what you start: close the TestClient, stop the loop, join "
        "the worker."
    )

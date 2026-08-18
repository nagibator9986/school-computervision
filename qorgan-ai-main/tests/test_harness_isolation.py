"""The test harness must not be able to write to the developer's own database.

**A measuring instrument that intermittently reports a false red is worse than a slow
one: it teaches you to re-run until green, which is how a real failure gets re-run away.**
This file is about the instrument, not about the product.

The defect it pins, observed twice on 2026-07-24 (once as `no such table: events`, once as
`database is locked`, and both times blamed on the branch under test before being traced):

  * `qorgan.db.engine` keeps the engine in a MODULE GLOBAL, built lazily from whatever
    `get_settings()` returns at the moment of the first call;
  * the `settings` fixture used to tear down by calling `override_settings(None)` FIRST and
    `reset_engine()` second;
  * so any thread still running in that window -- the notifier is a daemon, and `stop()`
    joins with a 5 s timeout, so a slow drain outlives it -- calls `session_scope()`, gets
    DEFAULT settings, and builds a global engine pointed at `<cwd>/data/qorgan.sqlite3`.

`_prepare_sqlite_path` then *creates the directory*, so the leak is not merely wrong: it
manufactures a real database file inside whichever worktree the suite ran in, and every
subsequent test in that process shares the poisoned global. The next failure looks like it
belongs to whatever test was unlucky enough to run next.

That is why this is tested by BEHAVIOUR -- "no database appears where the developer keeps
theirs" -- and not by asserting the order of two calls in a fixture. The order is the
current fix; the behaviour is the requirement, and it stays true if the fix changes.
"""

from __future__ import annotations

import threading
from pathlib import Path

from sqlalchemy import text

from qorgan.db.engine import reset_engine, session_scope
from qorgan.settings import Settings, get_settings, override_settings
from tests.conftest import REPO_ROOT


def _developers_database_file() -> Path:
    """`<repo>/data/qorgan.sqlite3` -- the file `qorgan web` and `qorgan db upgrade` use.

    **Written as the literal repo-relative location on purpose, not read back out of
    `Settings`.** The fix for this defect works by moving what the DEFAULT settings
    resolve to; asking `Settings` where the default points would therefore follow the fix
    around and assert that a throwaway file was not created, which is vacuous. The
    requirement is about a place on disk: nothing a test run does may put a database
    *here*. The path is `settings.py`'s shipped default, also documented in `HANDOVER.md`
    §2.
    """
    return (REPO_ROOT / "data" / "qorgan.sqlite3").resolve()


def _build_an_engine_with_no_settings_installed() -> list[str]:
    """Do exactly what a leaked daemon thread does between two tests, and report.

    Extracted so the test below stays inside the 50-line limit (`test_code_limits.py`),
    which is enforced and not to be loosened.
    """
    touched: list[str] = []
    done = threading.Event()

    def leaked_thread() -> None:
        try:
            with session_scope() as db:
                db.execute(text("SELECT 1"))
            touched.append("built an engine with no settings installed")
        except Exception as exc:  # refusing to build IS the safe outcome here
            touched.append(f"refused: {type(exc).__name__}")
        finally:
            done.set()

    thread = threading.Thread(target=leaked_thread, name="notifier", daemon=True)
    thread.start()
    done.wait(timeout=10.0)
    thread.join(timeout=5.0)
    return touched


def test_a_thread_that_outlives_a_test_cannot_create_the_developers_database(
    settings: Settings,
) -> None:
    """The exact shape of the leak, driven deliberately.

    A background thread calls `session_scope()` after the settings that told it where the
    database lives have been taken away. It must never end up at the default location --
    the file `qorgan web` serves.

    THE WINDOW IS AFTER TEARDOWN COMPLETES, not inside it. Measured rather than assumed:
    while the old engine object is still alive `session_scope()` keeps using it and never
    consults the missing settings. The rebuild -- and the damage -- happens on the first
    call once BOTH the override is gone AND the engine is None, which is the state every
    test leaves behind between its teardown and the next test's setup.
    """
    target = _developers_database_file()
    existed = target.exists()
    stamp = target.stat().st_mtime_ns if existed else None

    override_settings(None)
    reset_engine()
    touched = _build_an_engine_with_no_settings_installed()
    override_settings(settings)
    reset_engine()

    if existed:
        assert target.stat().st_mtime_ns == stamp, (
            f"the suite WROTE to {target} -- that is the database `qorgan web` serves"
        )
    else:
        assert not target.exists(), (
            f"the suite CREATED {target}. A test run must never manufacture a database "
            f"where the developer keeps theirs. The thread: {touched}"
        )


def test_tearing_down_settings_leaves_no_usable_global_engine(settings: Settings) -> None:
    """After teardown there must be no engine still bound to the finished test's database.

    A surviving engine is how one test's rows become another test's mystery: the global is
    shared, so whatever was left behind is what the next caller silently gets.
    """
    with session_scope() as db:
        db.execute(text("SELECT 1"))

    reset_engine()
    override_settings(None)

    import qorgan.db.engine as engine_module

    assert engine_module._engine is None, (
        "a live engine survived teardown, still pointed at the previous test's database"
    )
    override_settings(settings)


def test_the_settings_fixture_restores_a_clean_slate(settings: Settings) -> None:
    """The fixture's own contract: inside a test, settings are the test's own."""
    assert get_settings().database_url == settings.database_url
    assert "test.sqlite3" in settings.database_url, (
        "the settings fixture is not pointing at a throwaway database"
    )

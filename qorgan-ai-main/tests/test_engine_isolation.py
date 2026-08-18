"""One test's leaked thread must not decide which database the next test talks to.

**This is the second half of a fix that was shipped as if it were whole**, and the
difference matters more than the pages waiting behind it.

`fix/test-teardown-race` (merged at `c1f77bb`) pointed the DEFAULT `DATABASE_URL` at a
throwaway file, so a thread that outlives its test can no longer create or write
`<repo>/data/qorgan.sqlite3`. That part holds and is separately tested in
`test_harness_isolation.py`. It fixed **where** a leaked thread writes. It did not fix
**that the leaked thread reassigns the module-global engine**, and the second half is what
kept producing red runs in tests that had nothing to do with the branch under review:

    tests/test_worker_bullying.py ... no such table: events
    tests/test_web_preview.py     ... no such table: users

The mechanism, read off `db/engine.py` rather than guessed:

  * `build_engine(url)` returns an engine and does **not** touch the global. The `session`
    fixture calls it that way, so a test's own tables are created on an engine nobody
    shares.
  * `get_engine()` is the lazy global: `if _engine is None: _engine = build_engine()` —
    from whatever `get_settings()` says **at that instant**. Every production code path
    (`session_scope`) goes through it.
  * The `settings` fixture resets the engine **only on teardown**. Between one test's
    teardown and the next test's first database call there is a window in which the
    override is gone and `_engine` is None. A leaked daemon — the notifier polls every two
    seconds — calls `session_scope()` in that window, and the global is now an engine
    pointed at the harmless-but-**schemaless** default file.
  * Nothing in the next test's SETUP clears it. Its `settings` fixture installs an
    override; its `session` fixture builds its own engine; but `session_scope()` still
    returns the poisoned global, which has no tables. The test dies looking like its own
    bug.

So the durable fix is to reset the global **at setup as well as at teardown**: whatever a
previous test or its ghosts left behind, a test begins with a global that will be built
from its own settings. That is robust against leaks nobody has found yet, which matters
because closing the leaks one at a time only ever covers the threads somebody remembered.

The two tests below are **ordered on purpose**, and pytest runs them in file order (no
`pytest-randomly` here). The first plays the ghost; the second is the victim. Splitting
them across two tests is the point: the damage crosses a test boundary, so a single
self-contained test cannot reproduce it.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import text

import qorgan.db.engine as engine_module
from qorgan.db.engine import reset_engine, session_scope
from qorgan.settings import Settings, override_settings


@pytest.fixture
def a_ghost_that_outlives_the_test():
    """Poisons the global engine AFTER the `settings` fixture has torn down.

    That timing is the entire defect and it is why a self-contained test cannot show it:
    the poisoning test's OWN teardown calls `reset_engine()`, so anything done inside the
    test body is tidied up after itself. A real leaked daemon does not poll politely
    inside a test body -- it polls forever, including in the gap after teardown, when the
    override is gone and `_engine` is None.

    Requested FIRST in the signature below, so pytest sets it up before `settings` and
    therefore finalises it after: this teardown runs in exactly that gap.
    """
    yield

    override_settings(None)
    reset_engine()

    done = threading.Event()

    def leaked() -> None:
        try:
            with session_scope() as db:
                db.execute(text("SELECT 1"))
        except Exception:  # noqa: S110 - a refusal is safe too; either way the window closes
            pass
        finally:
            done.set()

    thread = threading.Thread(target=leaked, name="notifier", daemon=True)
    thread.start()
    done.wait(timeout=10.0)
    thread.join(timeout=5.0)

    assert engine_module._engine is not None, (
        "the leaked thread did not bind the global engine -- the mechanism this pair "
        "reproduces has changed, and the test below now proves nothing"
    )


def test_a_leaked_thread_can_still_bind_the_global_engine_to_the_wrong_database(
    a_ghost_that_outlives_the_test, settings: Settings
) -> None:
    """The ghost. Its body does nothing; the damage is done by the fixture's finaliser,
    after this test is over, which is where the real one happens."""
    del a_ghost_that_outlives_the_test, settings


def test_the_next_test_still_reaches_its_own_database(settings: Settings, session) -> None:
    """The victim, and the actual requirement.

    A test must talk to ITS database through the ordinary production path, no matter what
    the previous test left in the global. `users` exists because the `session` fixture
    created the whole schema on this test's own file; if `session_scope()` answers from
    the previous test's leftover engine, that table is missing and this fails exactly the
    way the real flake did.
    """
    del session  # requested for its schema creation, not used directly

    with session_scope() as db:
        db.execute(text("SELECT COUNT(*) FROM users"))

"""Starting the web app twice must not leave the first app's workers running.

**This is the defect that made a green suite untrustworthy**, and it is the second time
this leak has been closed: `fix/test-isolation-leak` stopped a stray thread from binding
the global engine BETWEEN tests, and this stops the strays existing.

The mechanism, measured rather than reasoned about:

    app = create_app()
    with TestClient(app):          # lifespan runs: app.state.notifier = worker A (started)
        with TestClient(app):      # lifespan runs AGAIN: app.state.notifier = worker B
            ...                    # A is now unreachable and nobody holds a reference
        # inner exit: shutdown stops app.state.notifier, which is B
    # outer exit: shutdown stops app.state.notifier, which is STILL B
    # worker A's thread is alive, and stays alive for the rest of the session

Every `client_for` fixture in this suite opens one `TestClient` per role against one `app`
fixture, so a test that asks for two roles orphans one worker, a test that asks for three
orphans two. Measured across a full run: hundreds of leaked `notifier` and `preview-sub`
threads, concentrated in the six files that compare roles.

**Why that turns into a failing test somewhere else.** A leaked `NotificationWorker` polls
`session_scope()` every couple of seconds forever. `get_engine()` builds the module-global
engine lazily from whatever settings are installed at that instant, so a poll landing in
the wrong moment binds the global to a database that is not the running test's -- and the
next test to authenticate reads a database with no `users` table. The failure surfaces in
whichever test is unlucky, looking exactly like that test's own bug. It reads as flaky
because it is: the same collection order passes and fails on different runs, since what
decides it is a background poll's timing.

The fix is in `lifespan`, not in the fixtures. Eleven `client_for` copies could each be
taught to close the previous client, and the twelfth written next month would not be. A
lifespan that starts a worker on top of a live one and drops the reference is wrong on its
own terms, whoever calls it.
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from qorgan.settings import Settings
from qorgan.web.app import create_app

WORKER_NAMES = ("notifier", "preview-sub")


def _worker_threads() -> set[int]:
    return {
        thread.ident
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name.startswith(WORKER_NAMES)
    }


def _running_worker_names() -> set[str]:
    """Which KINDS of worker are alive, by name.

    Counting threads is not enough to prove the app still works: with only the count, a
    live preview subscriber satisfies an assertion meant to be about the notifier. That is
    not hypothetical -- it is what let a deliberate sabotage that stopped the notifier one
    line after starting it pass this file twice.
    """
    return {
        name
        for name in WORKER_NAMES
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name.startswith(name)
    }


def test_entering_the_app_twice_leaves_no_worker_behind(settings: Settings, session) -> None:
    """**The test that makes the suite's greens mean something.**

    Nested deliberately: that is the shape `client_for` produces when a test asks for two
    roles, and it is the shape that leaks. The assertion is on THREADS rather than on
    `app.state`, because a stopped worker still sitting in `app.state` is harmless and a
    running thread nobody can reach is the whole problem.
    """
    del session  # the schema has to exist for the login path the workers touch

    before = _worker_threads()
    app = create_app()

    with TestClient(app), TestClient(app):
        pass

    leaked = _worker_threads() - before
    assert not leaked, (
        f"{len(leaked)} worker thread(s) outlived the app. A leaked NotificationWorker "
        "polls session_scope() forever and rebinds the global engine, which is how a test "
        "in another file dies with `no such table: users`."
    )


def test_three_clients_leave_nothing_either(settings: Settings, session) -> None:
    """Three roles in one test is not hypothetical -- `test_web_capability_roles.py`
    parametrises over four. Two leaks would satisfy a test that only checked for one."""
    del session

    before = _worker_threads()
    app = create_app()

    with TestClient(app), TestClient(app), TestClient(app):
        pass

    assert not _worker_threads() - before


def test_the_ordinary_single_client_still_works(settings: Settings, session) -> None:
    """The other half: stopping the previous worker must not stop the current one before
    it has served anybody. A fix that shut everything down would satisfy the tests above
    perfectly and leave the dashboard dead."""
    del session

    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        # RUNNING, not merely present. `is not None` was the first version of this
        # assertion and it passed with the worker deliberately stopped one line after it
        # was started -- an over-fix that would have shipped a dashboard whose notifier
        # never sends anything. A stopped worker is still an object.
        running = _running_worker_names()
        assert running == set(WORKER_NAMES), (
            f"the app started but these workers are not running: "
            f"{sorted(set(WORKER_NAMES) - running)}"
        )
        assert app.state.notifier is not None
        assert app.state.previews is not None

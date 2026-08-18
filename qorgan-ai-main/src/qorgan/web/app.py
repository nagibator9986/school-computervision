"""The FastAPI application.

Two rules shape this file:

- **Deny by default.** AuthMiddleware runs on every request; a route is protected
  unless it is explicitly public. The legacy had 50 endpoints and no auth at all.
- **No page-load side effects.** The legacy had `POST /page-activate/{page}` restart
  the AI workers -- with a five-second `thread.join()` inside the HTTP handler -- every
  time somebody opened a tab, which is why coverage depended on which tab was open.
  Nothing here starts, stops, or reconfigures a worker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

from qorgan.config.camera import CameraConfig
from qorgan.config.loader import ConfigError, load_cameras, load_workers
from qorgan.config.workers import WorkersConfig
from qorgan.logging_setup import get_logger, setup_logging
from qorgan.notify import NotificationWorker, TelegramClient
from qorgan.preview import PreviewSubscriber
from qorgan.settings import get_settings
from qorgan.web.backup_runner import BackupRunner
from qorgan.web.csrf import CsrfMiddleware
from qorgan.web.routes import (
    backups,
    canteen,
    classvision,
    duplicates,
    events,
    health,
    lessons,
    login,
    logs,
    media,
    notifications,
    psychologist,
    pupils,
    users,
    weapons,
)
from qorgan.web.routes import cameras as camera_routes
from qorgan.web.routes import schools as school_routes
from qorgan.web.routes import settings as settings_routes
from qorgan.web.security import AuthMiddleware

logger = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def _worker_assignments(
    cameras: dict[str, CameraConfig],
) -> tuple[WorkersConfig | None, str | None]:
    """Which worker group runs which camera -- read ONCE, at startup, never per request.

    Returns the config or the reason there isn't one, and does not raise. The supervisor
    is what must refuse to run on a broken `workers.yaml`; the web process is a reader,
    and the single most important thing this file can say -- "this enabled camera is
    assigned to no worker group, so nobody is watching it" -- is a message the school only
    ever sees on a page that opened. Taking the dashboard down to punish a config mistake
    removes the only place that reports it.

    Loading cameras above DOES raise, and deliberately: with no camera list there is no
    page to draw at all.
    """
    try:
        return load_workers(cameras), None
    except ConfigError as exc:
        logger.error("workers.yaml could not be read", extra={"error": str(exc)})
        return None, str(exc)


def _stop_quietly(worker: object | None) -> None:
    """Stop a worker if there is one, and never let that failure take the app down.

    `stop()` on both of these is idempotent -- it sets an event and joins a thread it then
    forgets -- so calling it on something already stopped is free.
    """
    stop = getattr(worker, "stop", None)
    if stop is None:
        return
    try:
        stop()
    except Exception:
        logger.exception("could not stop a previous worker before replacing it")


def _stop_any_previous_workers(app: FastAPI) -> None:
    """Stop whatever this app object is already running, before replacing it.

    In production `lifespan` runs once and there is nothing here to stop. It runs more than
    once whenever one app object is served twice -- which the test suite does every time a
    test wants two roles at once -- and the second run used to overwrite `app.state.notifier`
    with a new worker while the first kept running, unreachable and unstoppable: shutdown
    can only stop what `app.state` still points at.

    A leaked NotificationWorker is not idle. It polls `session_scope()` forever, and
    `get_engine()` binds the module-global engine lazily from whatever settings are
    installed at that instant -- so a poll landing at the wrong moment points the global at
    a database that is not the running test's, and the next test to authenticate dies with
    `no such table: users`, in a file that has nothing to do with the cause. It reads as
    flaky because it is: what decides it is a background poll's timing.

    Fixed here rather than in the eleven fixtures that produce the shape, because the
    twelfth fixture written next month would reproduce it. Starting a worker on top of a
    live one and dropping the reference is wrong on its own terms, whoever calls it.
    """
    _stop_quietly(getattr(app.state, "notifier", None))
    _stop_quietly(getattr(app.state, "previews", None))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Subscribe to the preview bus and drain the notification queue.

    Note what is NOT here: starting workers. The web process does not own the fleet and
    cannot start one; the supervisor does. In the legacy, opening a page restarted the
    AI workers — with a five-second thread.join() inside the HTTP handler.

    The notifier lives here rather than in a detection worker deliberately: a worker must
    never block on a network call, and the legacy called a blocking requests.post from
    inside the frame loop, so an unreachable Telegram stalled the detector.
    """
    settings = get_settings()
    app.state.cameras = load_cameras()
    app.state.workers, app.state.workers_error = _worker_assignments(app.state.cameras)
    # Before starting anything: an app served twice would otherwise leave the first run's
    # workers alive and unreachable. See `_stop_any_previous_workers` for why that ends as
    # `no such table: users` in an unrelated test.
    _stop_any_previous_workers(app)

    app.state.previews = PreviewSubscriber(
        settings.preview_address,
        stale_after_seconds=settings.preview_stale_after_seconds,
    ).start()
    app.state.notifier = NotificationWorker(TelegramClient.from_settings()).start()

    logger.info("web started", extra={"cameras": len(app.state.cameras)})
    try:
        yield
    finally:
        app.state.notifier.stop()
        app.state.previews.stop()
        # A join at SHUTDOWN, which is the opposite of the legacy's join inside a request
        # (H-18): here the process is going away, and exiting mid-VACUUM would leave a
        # half-written file in backups/ that the page would list as a backup.
        app.state.backups.stop()


# Every route module, mounted in this order. A list rather than eighteen `include_router`
# lines in `create_app`, because three branches merged in the same week each added one and
# the function went over its length limit doing nothing but growing -- the limit was
# measuring the list, not the logic.
#
# **A module missing from here is a page that does not exist**, silently and with a green
# suite, which is why `tests/test_the_pages_promise_only_what_exists.py` and the capability
# tests walk the real app rather than the router modules.
ROUTE_MODULES = (
    health,
    login,
    camera_routes,
    events,
    weapons,
    notifications,
    canteen,
    lessons,
    logs,
    psychologist,
    classvision,
    duplicates,
    pupils,
    media,
    settings_routes,
    school_routes,
    backups,
    users,
)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging("web")

    app = FastAPI(title="Qorgan AI", lifespan=lifespan, docs_url=None, redoc_url=None)

    # Built with the app rather than in the lifespan: it opens nothing and starts no
    # thread until somebody presses the button, and a route reading `app.state` must not
    # depend on which entry point constructed the application. It is stopped in the
    # lifespan's `finally`, where there is a shutdown to hook.
    app.state.backups = BackupRunner()

    # Order matters: SessionMiddleware must be added AFTER AuthMiddleware so that it
    # runs BEFORE it -- Starlette applies middleware in reverse. Without the session
    # loaded first, the auth middleware would find no cookie and lock everyone out.
    app.add_middleware(AuthMiddleware)
    # Between the two on purpose. Starlette runs the LAST-added first, so the order is
    # Session -> Csrf -> Auth: the token lives in the session, so the session must be
    # loaded before it can be checked, and an unsafe request is refused before any
    # route runs.
    app.add_middleware(CsrfMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key.get_secret_value(),
        # Follows the transport, not the name of the environment. Keyed off qorgan_env
        # this marked cookies Secure on plain-HTTP "prod" installs, where the browser
        # then never sent them back and the login redirected to itself forever. Settings
        # refuses to build a prod that cannot carry a Secure cookie, so this cannot lie.
        https_only=settings.web_https,
        same_site="lax",
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    for module in ROUTE_MODULES:
        app.include_router(module.router)
    return app

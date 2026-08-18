"""What must never come off the /logs page: a secret, a child, or live markup.

The two error strings this page reads out of the DATABASE -- `worker_heartbeats.last_error`
and `notifications.last_error` -- never pass through the log formatter, so
`RedactingFormatter`, which is what makes the log FILES safe, does nothing for them at
all. `notifications.last_error` is built from an httpx exception raised against
`https://api.telegram.org/bot<TOKEN>/sendPhoto`, and httpx puts the URL it was calling
into the exception text. That is audit C-02 -- `rtsp://user:password@host` printed into
every log line and drawn onto debug JPEGs -- arriving by a different road.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Notification, WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.enums import NotificationChannel, NotificationStatus, UserRole, WorkerState
from tests.web_logs_support import (  # noqa: F401 - fixtures are used by name
    ClientFor,
    an_event,
    app,
    camera,
    client_for,
    json_logs,
    record,
    write,
)

# Neither of these is in `Settings.secret_values()` for this suite, and that is the point:
# the redactor must catch them by SHAPE. A page that only masks the secrets it was told
# about is a page that leaks the per-camera password nobody registered.
UNKNOWN_CAMERA_PASSWORD = "hunter2-not-in-settings"
UNKNOWN_BOT_TOKEN = "987654321:AAFakeTokenThatIsLongEnough1234567890abc"


def test_a_bot_token_stored_in_a_notification_error_never_reaches_the_page(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    session.add(
        Notification(
            event_id=an_event(camera),
            channel=NotificationChannel.TELEGRAM,
            status=NotificationStatus.FAILED,
            attempts=6,
            last_error=(
                "ConnectError: [Errno 111] Connection refused for url "
                f"'https://api.telegram.org/bot{UNKNOWN_BOT_TOKEN}/sendPhoto'"
            ),
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert UNKNOWN_BOT_TOKEN not in page, "the Telegram bot token was rendered to a browser"
    assert "AAFakeToken" not in page
    assert "api.telegram.org" in page, "the useful half of the error was thrown away too"


def test_a_camera_password_stored_in_a_worker_error_never_reaches_the_page(
    client_for: ClientFor, session: Session
) -> None:
    """`worker_heartbeats.last_error` is `f"{type(exc).__name__}: {exc}"` from an arbitrary
    exception, and an RTSP failure carries the whole URL. Same database-not-formatter hole."""
    session.add(
        WorkerHeartbeat(
            group_name="hall",
            state=WorkerState.CRASHED,
            pid=4242,
            restart_count=7,
            last_seen_at=utcnow(),
            last_error=(
                "OSError: could not open "
                f"rtsp://admin:{UNKNOWN_CAMERA_PASSWORD}@10.0.0.9:554/Streaming/Channels/102"
            ),
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert UNKNOWN_CAMERA_PASSWORD not in page, "the RTSP password was rendered to a browser"
    assert "10.0.0.9" in page, "the host is what an engineer actually needs"


def test_the_same_password_never_reaches_the_camera_pages_either(
    client_for: ClientFor, session: Session
) -> None:
    """**The same secret, the other two pages that render the same column.**

    `/logs` was guarded from the day it was written. `/` and `/cameras` were not, and the
    gap was invisible: the redaction lives in `diagnostics.workers.worker_rows`, and while
    `routes/cameras.py` kept its OWN copy of that query the two surfaces disagreed with
    nothing to say so. Merging the cameras branch would have restored that copy -- reading
    `row.last_error` raw -- and every test in this suite stayed green while the dashboard
    printed the camera password.

    Proven, not assumed: with the merge resolved to a single redacted source, reverting
    `_heartbeats()` to the raw query left all 1664 tests passing. This test is what that
    sabotage should have hit, so it asserts the RENDERED PAGE rather than the helper -- the
    helper already has tests, and they are the ones that stayed green.
    """
    # A REAL worker group from config/workers.yaml. `/cameras` maps heartbeats onto
    # cameras by group name, so an invented name renders nothing at all and the test would
    # pass by showing no error rather than by redacting one -- which is how a leak test
    # becomes decoration. `/logs` lists every heartbeat row and does not care.
    session.add(
        WorkerHeartbeat(
            group_name="bullying_hall",
            state=WorkerState.CRASHED,
            pid=4242,
            restart_count=7,
            last_seen_at=utcnow(),
            last_error=(
                "OSError: could not open "
                f"rtsp://admin:{UNKNOWN_CAMERA_PASSWORD}@10.0.0.9:554/Streaming/Channels/102"
            ),
        )
    )
    session.commit()
    client = client_for(UserRole.ADMIN)

    for path in ("/", "/cameras"):
        page = client.get(path).text
        assert UNKNOWN_CAMERA_PASSWORD not in page, (
            f"{path} rendered the RTSP password to a browser"
        )
        assert "10.0.0.9" in page, (
            f"{path} lost the host, which is the part an engineer actually needs"
        )


def test_a_password_in_a_log_file_never_reaches_the_page(
    client_for: ClientFor, json_logs: Path
) -> None:
    """A line already on disk was written by a process whose redactor knew a different set
    of secrets -- or was written before the redactor existed. The file is not trusted."""
    write(
        json_logs,
        "worker-hall",
        record(f"stream failed: rtsp://admin:{UNKNOWN_CAMERA_PASSWORD}@10.0.0.9:554/x"),
    )

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert UNKNOWN_CAMERA_PASSWORD not in page
    assert "10.0.0.9" in page


def test_an_identifier_for_a_child_is_not_rendered_by_this_page(
    client_for: ClientFor, json_logs: Path
) -> None:
    """Legacy printed children's full names to stdout (audit M-16). v2 logs `person_id`
    instead -- better, and still not something a page about broken cameras needs. Extras
    are rendered from an allow-list, so a field nobody classified is not shown."""
    write(
        json_logs,
        "worker-canteen",
        record(
            "recognition failed",
            logger="qorgan.worker.canteen",
            level="WARNING",
            person_id=4711,
            pupil_name="Айгерим Смагулова",
            camera="canteen_entry",
        ),
    )

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "recognition failed" in page, "the diagnostic itself must survive"
    assert "Айгерим" not in page, "a child's name was rendered onto the diagnostics page"
    assert "4711" not in page, "person_id is not needed to diagnose a camera"
    assert "canteen_entry" in page, "the camera is the whole point of the record"


def test_a_log_message_containing_markup_is_escaped(
    client_for: ClientFor, json_logs: Path
) -> None:
    """A log line is the most attacker-influenced text in the system: it quotes camera
    names, file paths and exception strings. The legacy built this DOM with innerHTML from
    server JSON, so a pupil named `<img src=x onerror=...>` was stored XSS (audit H-05)."""
    write(json_logs, "worker-hall", record("<img src=x onerror=alert(1)>"))

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "<img src=x onerror" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page

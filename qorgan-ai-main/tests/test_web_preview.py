"""The live preview page, and the frames behind it."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape
from sqlalchemy.orm import Session

import qorgan.web
from qorgan.config.common import PreviewSettings
from qorgan.db.models import User
from qorgan.enums import UserRole, WorkerState
from qorgan.passwords import hash_password
from qorgan.preview import PreviewPublisher
from qorgan.settings import Settings
from qorgan.supervisor.heartbeat import write_heartbeat
from qorgan.web.app import create_app
from tests.fakes import noisy_frame
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
PREVIEW = PreviewSettings(fps=15.0, width=320)

# The one consumer of /api/cameras. Read as a file, not paraphrased: a test that restates
# what the JavaScript "obviously" reads is a test of the restatement.
PREVIEW_JS = Path(qorgan.web.__file__).resolve().parent / "static" / "preview.js"


@pytest.fixture
def bus_settings(settings: Settings) -> Settings:
    # The base `settings` fixture already points preview_address at port 0 -- an OS-
    # assigned ephemeral port, bound for real inside PreviewSubscriber -- rather than
    # a port chosen here and released before the real bind happens. Kept as its own
    # fixture so the dependency (client -> bus_settings -> settings) stays explicit.
    return settings


@pytest.fixture
def client(bus_settings: Settings, session: Session) -> Iterator[TestClient]:
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as test_client:
        test_client.post(
            "/login",
            data=with_token(test_client, {"username": "op", "password": PASSWORD}),
        )
        yield test_client


@pytest.fixture
def publisher(client: TestClient) -> Iterator[PreviewPublisher]:
    # Connect to the address the SUB socket actually bound, read back off the running
    # app, instead of guessing a port ahead of time. No sleep here: ZeroMQ's PUB/SUB
    # has no connect handshake, so no fixed pause could ever prove the connection is
    # up. `_publish_and_wait` below is what actually establishes and confirms it, by
    # publishing until the subscriber is seen to have received it.
    pub = PreviewPublisher(client.app.state.previews.address)
    yield pub
    pub.close()


def _publish_and_wait(client: TestClient, publisher: PreviewPublisher, camera: str) -> None:
    """Publish until received, bounded by a deadline -- not sleep-and-hope.

    A frame published before the SUB socket's connect completes is dropped, not
    queued (the "slow joiner" problem); there is no event to wait on, so republishing
    until the effect is observed is the only correct synchronisation, not a retry
    used to paper over flakiness.
    """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        publisher.publish(camera, noisy_frame(1), PREVIEW, now=time.time() + 100)
        if client.get(f"/preview/{camera}.jpg").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError(f"no preview ever arrived for {camera}")


def test_the_dashboard_lists_every_configured_camera(client: TestClient) -> None:
    """Coverage comes from config, not from what the worker happened to publish, and
    certainly not from which tab is open (rule R3)."""
    response = client.get("/")
    assert response.status_code == 200
    for camera in ("hall_left", "canteen_entry", "stairs_floor2_aux"):
        assert camera in response.text


def test_a_frame_published_by_a_worker_is_served_to_the_browser(
    client: TestClient, publisher: PreviewPublisher
) -> None:
    _publish_and_wait(client, publisher, "hall_left")

    response = client.get("/preview/hall_left.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic
    assert response.headers["cache-control"] == "no-store, private"


def test_a_camera_with_no_worker_reports_offline_rather_than_a_stale_frame(
    client: TestClient,
) -> None:
    """503, not a five-minute-old picture. A stale preview tells the operator the
    corridor is calm when in truth nobody is watching it at all."""
    response = client.get("/preview/yard_entry.jpg")
    assert response.status_code == 503


def test_an_unknown_camera_is_a_404_not_a_503(client: TestClient) -> None:
    assert client.get("/preview/not_a_camera.jpg").status_code == 404


def test_the_api_reports_which_cameras_are_live(
    client: TestClient, publisher: PreviewPublisher
) -> None:
    _publish_and_wait(client, publisher, "hall_left")

    payload = client.get("/api/cameras").json()
    by_name = {camera["name"]: camera for camera in payload["cameras"]}

    assert by_name["hall_left"]["live"] is True
    assert by_name["hall_left"]["status"] == "ok"
    assert by_name["yard_entry"]["live"] is False
    assert by_name["yard_entry"]["status"] == "offline"


def test_worker_health_is_read_from_the_database_not_from_a_worker_module(
    client: TestClient, session: Session
) -> None:
    """The legacy web layer imported CAMERA_REGISTRY out of the bullying worker, so
    loading a web route pulled in YOLO and torch as a side effect (M-23)."""
    write_heartbeat("bullying_hall", WorkerState.RUNNING, frames_processed=99)

    payload = client.get("/api/cameras").json()
    groups = {worker["group"]: worker for worker in payload["workers"]}

    assert groups["bullying_hall"]["state"] == "running"
    assert groups["bullying_hall"]["seconds_since_beat"] is not None


def test_the_json_api_carries_everything_the_dashboard_page_shows(client: TestClient) -> None:
    """One source of truth for two surfaces, checked rather than asserted in a comment.

    `/api/cameras` and `/` are both built from `_camera_rows` / `_worker_rows` today, so
    the JSON is exactly as wide as the page — but nothing stopped a future edit from
    giving the page a field the JSON does not carry, and a value that is true on one
    surface and missing from the next is this project's signature defect. This is the
    check that says no.
    """
    payload = client.get("/api/cameras").json()
    page = client.get("/").text

    assert payload["cameras"], "no cameras at all; the check below would prove nothing"
    for camera in payload["cameras"]:
        for field in ("name", "display_name", "location", "role"):
            assert str(escape(str(camera[field]))) in page, f"{field} is on one surface only"


def test_the_api_sends_every_key_its_only_consumer_reads(client: TestClient) -> None:
    """The compatibility constraint on this endpoint, read off the consumer itself.

    `preview.js` dereferences `camera.<key>` on the payload. A key renamed in
    `_camera_rows` is not a Python error and not a JavaScript error either — `undefined`
    is a value — so the badge would simply start printing nothing and the page would look
    merely quiet. Nothing else in the suite ties the two files together.
    """
    used = set(re.findall(r"\bcamera\.([A-Za-z_][A-Za-z0-9_]*)", PREVIEW_JS.read_text("utf-8")))
    assert used, "no camera field reads found; the regex, not the API, is what broke"

    sent = set(client.get("/api/cameras").json()["cameras"][0])

    assert used <= sent, f"preview.js reads keys /api/cameras does not send: {sorted(used - sent)}"


def test_the_web_process_never_imports_a_worker_module() -> None:
    """Rule R3, checked mechanically rather than by good intentions."""
    import sys

    for module in list(sys.modules):
        if module.startswith("qorgan.web"):
            del sys.modules[module]

    before = set(sys.modules)
    import qorgan.web.app  # noqa: F401

    newly_imported = set(sys.modules) - before
    leaked = [m for m in newly_imported if m.startswith(("qorgan.worker", "ultralytics", "torch"))]
    assert not leaked, f"importing the web app dragged in: {leaked}"

"""The camera health page: what every camera is doing, and how much of that we KNOW.

Four defects shape this file -- three from the legacy panel, one from this repo's own
history:

  * **Live corridor video is not everyone's page.** §14 says canteen staff work "БЕЗ
    доступа к буллингу", and a frame of the main hall is a picture of children. The page
    is gated on the capability that already guards `/preview/{camera}.jpg`, because it
    shows the same frames.

  * **Zero side effects on load.** The legacy restarted the AI workers whenever a tab was
    opened -- with a five-second `thread.join()` inside the HTTP handler -- so coverage
    depended on which tab somebody had open. Coverage comes from config (rule R3), and
    this page is a reader.

  * **Pagination.** The legacy re-rendered the entire world every 2.5 s, per client
    (audit M-19). Every row here carries a live JPEG the browser re-fetches, so an
    unbounded list is a bandwidth fault that grows with the school.

  * **A number true in one layer and silently wrong in the next.** `capture.stream_fps` is
    what one camera's own web UI claimed in a screenshot; what the stream actually delivers
    is measured by the camera loop. Printing the configured number in a column an operator
    reads as fact would be the `display_fps` defect (see tests/test_analysis_rate.py) moved
    to the presentation layer -- so the page always says which of the two it is showing,
    and says "не измерено" when nobody has measured anything.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.common import PreviewSettings
from qorgan.db.models import User, WorkerHeartbeat
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
PAGE = "/cameras"

ClientFor = Callable[[UserRole], TestClient]

# base.yaml ships this fleet-wide, and every camera inherits it. The page must never
# present it as a measurement.
CONFIGURED_FPS = "15"


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about."""
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login",
                data=with_token(client, {"username": username, "password": PASSWORD}),
            )
            assert response.status_code == 303, "login failed"
            return client

        yield make


@pytest.fixture
def client(client_for: ClientFor) -> TestClient:
    return client_for(UserRole.OPERATOR)


@pytest.fixture
def publisher(client: TestClient) -> Iterator[PreviewPublisher]:
    # The address the SUB socket actually bound, read back off the running app -- not a
    # port guessed ahead of time. See tests/test_web_preview.py for why there is no sleep.
    pub = PreviewPublisher(client.app.state.previews.address)
    yield pub
    pub.close()


CARD = re.compile(r'data-camera="([a-z0-9_]+)"(.*?)</article>', re.S)


def _cards(html: str) -> dict[str, str]:
    """Each camera's own slice of the page.

    Scoped deliberately: an assertion that only searches the whole document is satisfied
    by ANY camera's text, so "hall_left reports no measurement" would pass while hall_left
    showed a number and some other row carried the words.
    """
    return {name: body for name, body in CARD.findall(html)}


def _publish_and_wait(client: TestClient, publisher: PreviewPublisher, camera: str, **header):
    """Publish until the web process is seen to have received it, bounded by a deadline.

    A frame published before the SUB socket's connect completes is dropped rather than
    queued (ZeroMQ's slow joiner), and there is no event to wait on -- so republishing
    until the effect is observed is the synchronisation, not a retry papering over flake.
    """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        publisher.publish(camera, noisy_frame(1), PREVIEW, now=time.time() + 100, **header)
        if client.get(f"/preview/{camera}.jpg").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError(f"no preview ever arrived for {camera}")


# -- who may look at a school corridor ---------------------------------------


def test_the_page_needs_a_session(app) -> None:
    """The legacy panel had ~50 endpoints, no auth, and bound to 0.0.0.0 (audit C-01)."""
    with TestClient(app, follow_redirects=False) as anonymous:
        assert anonymous.get(PAGE).status_code == 303


def test_a_canteen_worker_is_refused_the_camera_health_page(client_for: ClientFor) -> None:
    """§14: столовая — БЕЗ доступа к буллингу.

    This page renders the same corridor frames as `/preview/{camera}.jpg`, so it must be
    shut to exactly the same people. A page that shows children's faces to whoever can
    open a URL is the finding the whole capability model exists to prevent.
    """
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get(PAGE).status_code == 403


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.ADMIN, UserRole.DEVELOPER])
def test_the_roles_that_already_watch_the_corridors_still_can(
    client_for: ClientFor, role: UserRole
) -> None:
    """A refusal that also refuses the operator is a broken page, not a safe one."""
    assert client_for(role).get(PAGE).status_code == 200


# -- what the page says about a camera ---------------------------------------


def test_each_camera_is_shown_with_its_role_worker_group_and_device(client: TestClient) -> None:
    """Name, role, and who is actually running it. `hall_left` sits in `bullying_hall` on
    cuda:0 in workers.yaml, and that mapping is the only answer to "who watches this?"."""
    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "Холл слева" in card
    assert "main_hall" in card
    assert "bullying_hall" in card, "the page does not say which worker owns this camera"
    assert "cuda:0" in card


def test_the_page_shows_the_latest_frame_for_each_camera(client: TestClient) -> None:
    """Through the authenticated handler, never StaticFiles: these are children."""
    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "/preview/hall_left.jpg" in card


def test_a_display_name_that_looks_like_markup_cannot_inject_script(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy built this DOM with `innerHTML` from server JSON, so a pupil named
    `<img src=x onerror=...>` was stored XSS in the operator's browser (audit H-05).
    A camera's display_name comes from a YAML file a human edits, by the same route."""
    cameras = client.app.state.cameras
    hostile = cameras["hall_left"].model_copy(
        update={"display_name": '<img src=x onerror="alert(1)">'}
    )
    monkeypatch.setitem(cameras, "hall_left", hostile)

    response = client.get(PAGE)

    assert "<img src=x onerror=" not in response.text
    assert "&lt;img src=x onerror=" in response.text, "the display name was not escaped"


def test_a_disabled_camera_is_not_reported_as_a_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`enabled: false` means nobody is meant to be watching it, and no worker runs it --
    so the absence of frames is the configuration working, not a fault. Reporting it as a
    fault is how a real outage gets lost among rows that are red every single day."""
    cameras = client.app.state.cameras
    monkeypatch.setitem(cameras, "yard_entry", cameras["yard_entry"].model_copy(
        update={"enabled": False}
    ))

    card = _cards(client.get(f"{PAGE}?page=2").text)["yard_entry"]

    assert "отключена" in card
    assert "нет кадров от камеры" not in card, "a camera nobody watches was reported broken"
    assert "воркер не работает" not in card


def test_a_silent_camera_and_a_stopped_worker_are_different_answers(client: TestClient) -> None:
    """No preview arrived. That is ONE observation with two very different causes, and the
    page may only report the one the evidence supports.

    If the worker group is alive and still sending nothing, the camera is the suspect. If
    the group never reported at all, nothing whatsoever is known about the camera -- and
    "нет сигнала" beside its name would send an electrician to a working camera while the
    supervisor sat crashed. True at the layer that produced it ("no frames here"), false one
    layer up ("this camera is broken").
    """
    write_heartbeat("bullying_hall", WorkerState.RUNNING, frames_processed=1)

    cards = _cards(client.get(PAGE).text)

    assert "нет кадров от камеры" in cards["hall_left"], "a live worker's silence was excused"
    assert "воркер не работает" in cards["stairs_floor1"], (
        "a camera was blamed for a worker group that has never reported"
    )


def test_the_page_says_when_it_was_rendered(client: TestClient) -> None:
    """Every number on this page is a snapshot taken at one instant, and it stays that
    instant: nothing here refreshes itself.

    That is the point rather than a limitation. `/` is the live wall. A page where the
    status badge updated itself every second while the frame rate beside it was five
    minutes old would be two truths in one row -- so this one is coherent and says which
    instant it belongs to. An undated snapshot is read as "now".
    """
    body = client.get(PAGE).text

    assert "Снимок на" in body, "the page does not say which moment it is describing"
    assert "preview.js" not in body, (
        "a self-refreshing widget on a snapshot page makes half the row current "
        "and half of it stale"
    )


# -- the frame rate, and which frame rate it is ------------------------------


def test_a_camera_nobody_has_measured_says_so_instead_of_repeating_the_config(
    client: TestClient,
) -> None:
    """`capture.stream_fps` is a fact READ OFF A SCREENSHOT of one camera's web UI. It is
    not a measurement, and the page must not let an operator read it as one.

    This is the `display_fps` defect at the presentation layer: a number that was true as
    "what we configured" and false as "what the camera delivers", with nothing in between
    saying which one you were looking at.
    """
    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "не измерено" in card, "the page implied a measurement it does not have"
    assert CONFIGURED_FPS in card, "the configured rate is not shown at all"


def test_the_rate_the_worker_measured_reaches_the_page(
    client: TestClient, publisher: PreviewPublisher
) -> None:
    """The loop measures what the stream really delivers. That number stayed inside the
    worker process, where the one person who needs it -- whoever is looking at the
    dashboard wondering why the camera lags -- could not see it."""
    _publish_and_wait(client, publisher, "hall_left", measured_fps=49.5)

    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "49.5" in card, "the measured rate never reached the page"
    assert "не измерено" not in card
    # And it is dated. The loop counts over its first 30 frames and then stops, by design.
    # Presented as plain "измерено", a number from the worker's first two seconds reads as
    # the current rate -- the same trick the configured value plays one row above.
    assert "при запуске" in card, "a startup measurement was presented as the live rate"


def test_a_measured_rate_that_contradicts_the_config_is_flagged_on_the_page(
    client: TestClient, publisher: PreviewPublisher
) -> None:
    """Configured 15, delivered ~50. The loop already logs this; a log line is read by
    whoever is reading logs, and this page is read by whoever is asking the question.

    Both numbers, and the verdict. "They disagree" is not actionable; "configured 15,
    measured 49.5" is.
    """
    _publish_and_wait(client, publisher, "hall_left", measured_fps=49.5)

    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "конфиг ≠ поток" in card, "a stream at 3.3x the configured rate raised nothing"


def test_a_measured_rate_that_matches_the_config_is_not_flagged(
    client: TestClient, publisher: PreviewPublisher
) -> None:
    """A warning that fires on everything is filtered out by the person reading it, which
    is the same as not having one. An NVR does not hold an exact rate: 15 vs 14.2 is fine,
    15 vs 10 is not, and the tolerance that decides is the SAME one the worker logs by."""
    _publish_and_wait(client, publisher, "hall_left", measured_fps=14.2)

    card = _cards(client.get(PAGE).text)["hall_left"]

    assert "14.2" in card
    assert "конфиг ≠ поток" not in card


# -- errors belong to whoever actually owns them -----------------------------


def test_a_worker_error_is_attributed_to_the_group_and_not_to_one_camera(
    client: TestClient,
) -> None:
    """`bullying_hall` runs hall_left AND hall_right, and its heartbeat carries ONE error
    for the process. Printing it beside hall_left as "hall_left's error" invents a fact:
    the failure may be hall_right's, or neither camera's. The row names the group.

    This is the disease this repo keeps finding -- a value true at the layer that produced
    it (the process failed) and false one layer up (this camera failed).
    """
    write_heartbeat("bullying_hall", WorkerState.CRASHED, last_error="RTSP read failed")

    cards = _cards(client.get(PAGE).text)

    for camera in ("hall_left", "hall_right"):
        assert "RTSP read failed" in cards[camera]
        assert "ошибка группы bullying_hall" in cards[camera], (
            "a worker-group error was pinned on a single camera"
        )
    assert "RTSP read failed" not in cards["stairs_floor1"], (
        "another group's error leaked onto this camera"
    )


# -- pagination --------------------------------------------------------------


def test_every_camera_appears_on_exactly_one_page(client: TestClient) -> None:
    """The legacy loaded everything, every render, per client (M-19). Here each row also
    carries a live JPEG, so the list is bounded -- and a bounded list must still be able
    to show you every camera, or a camera silently stops existing."""
    first = set(_cards(client.get(f"{PAGE}?page=1").text))
    second = set(_cards(client.get(f"{PAGE}?page=2").text))
    configured = set(client.app.state.cameras)

    assert first, "page 1 is empty"
    assert first < configured, "the page is not paginated at all"
    assert not (first & second), "a camera appears on two pages"
    assert first | second == configured, "a camera exists in config and on no page"


def test_a_page_number_past_the_end_shows_the_last_page_not_an_empty_one(
    client: TestClient,
) -> None:
    """`?page=99` is a bookmark someone kept after the fleet shrank. An empty camera list
    on a safety dashboard reads as "nothing is being watched", which is a false alarm the
    school acts on."""
    assert _cards(client.get(f"{PAGE}?page=99").text), "an out-of-range page showed nothing"


# -- rule R3: the page changes nothing ---------------------------------------


def test_opening_the_page_starts_nothing(client: TestClient, session: Session) -> None:
    """The legacy restarted the AI workers when a tab was opened, with a five-second
    `thread.join()` in the HTTP handler, and analysed the stairs only while somebody was
    looking at the stairs page.

    Catches the LASTING half: a worker, a poller, a connection left running per request.
    The transient half -- a probe that opens a socket and is gone before anyone counts --
    is invisible from here, and is caught by the test below instead. (Written the other way
    round first: a sabotage that spawned a thread and returned immediately kept this test
    green, which is how the gap was found rather than assumed.)
    """
    write_heartbeat("bullying_hall", WorkerState.RUNNING, frames_processed=7)
    client.get(PAGE)  # warm up: first-request setup is not a per-request side effect
    threads_before = len(threading.enumerate())

    client.get(PAGE)

    assert len(threading.enumerate()) == threads_before, "the page spawned a thread"
    session.expire_all()
    rows = session.scalars(select(WorkerHeartbeat)).all()
    assert [(r.group_name, r.restart_count, r.frames_processed) for r in rows] == [
        ("bullying_hall", 0, 7)
    ], "rendering the page wrote to the worker table"


# Everything a handler would need to start a process, spawn a thread, or reach out to a
# camera. None of it belongs in a module whose entire job is to read state somebody else
# produced -- and a GET that opens an RTSP socket is the specific temptation of a page with
# a "reachable" column on it.
FORBIDDEN_IMPORTS = frozenset(
    {"threading", "subprocess", "multiprocessing", "socket", "cv2", "requests", "httpx", "urllib"}
)


def test_the_camera_routes_cannot_reach_out_or_start_anything() -> None:
    """Rule R3, checked mechanically rather than by good intentions.

    The same shape as `test_the_web_process_never_imports_a_worker_module`, and for the
    same reason: watching for the EFFECT of a side effect misses the ones that finish
    quickly, so the capability to have one is what gets denied. Reachability on this page
    means "a worker sent us a frame recently" -- already known, nothing to go and ask.
    """
    import ast
    from pathlib import Path

    from qorgan.web.routes import cameras

    tree = ast.parse(Path(cameras.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        # ast.walk, not tree.body: an import hidden inside the handler is exactly how a
        # side effect gets added without touching the top of the file.
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert not (imported & FORBIDDEN_IMPORTS), (
        f"the camera routes can now start or contact something: "
        f"{sorted(imported & FORBIDDEN_IMPORTS)}"
    )


def test_the_page_offers_no_control_that_changes_what_is_watched(client: TestClient) -> None:
    """Coverage is decided by config, never by the UI (R3). The legacy's
    `POST /page-activate/{page}` is the control that must not exist, so the page has no
    unsafe method at all -- a token-bearing POST from our own page still finds nothing.
    """
    response = client.post(PAGE, data=with_token(client))

    assert response.status_code == 405, "the camera page accepts a state-changing request"
    assert 'action="/cameras' not in client.get(PAGE).text

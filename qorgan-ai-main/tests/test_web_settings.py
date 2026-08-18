"""The settings page: shows everything, changes nothing.

Three legacy defects meet on this one page, which is why it is tested this hard.

  * **It leaked the bot token.** `/settings?format=json` handed the whole settings table
    -- including a plaintext Telegram bot token -- to any anonymous caller (audit H-04).
  * **It had side effects on load.** `POST /page-activate/{page}` restarted the AI workers,
    with a five-second `thread.join()` inside the HTTP handler, every time somebody opened
    a tab (audit H-18). Coverage depended on which browser tab was open.
  * **It was a second source of truth.** A browser form that edits YAML makes the file and
    the form disagree, and "true in one layer, silently wrong in the next" is this
    project's signature disease -- `min_score: 0.50` in `config/identity.py`, overridden by
    every profile, in force on zero cameras.

So: read-only, capability-gated, and no write path of any kind. The tests below are the
things that must break if any of the three comes back.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models import AppSetting, ModeLog, User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.roles import Capability, capabilities_for
from qorgan.settings import Settings, override_settings
from qorgan.web.csrf import SAFE_METHODS
from qorgan.web.routes import settings as settings_routes
from tests.conftest import CONFIG_DIR, SRC_DIR
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]

TEMPLATE = SRC_DIR / "qorgan" / "web" / "templates" / "settings.html"


@pytest.fixture
def app(settings: Settings, session: Session):
    from qorgan.web.app import create_app

    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login", data=with_token(client, {"username": username, "password": PASSWORD})
            )
            assert response.status_code == 303, "login failed"
            return client

        yield make


@pytest.fixture
def admin(client_for: ClientFor) -> TestClient:
    return client_for(UserRole.ADMIN)


# -- who may look ------------------------------------------------------------


def test_the_settings_page_needs_a_session(app) -> None:
    with TestClient(app, follow_redirects=False) as anonymous:
        assert anonymous.get("/settings").status_code == 303


@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.DEVELOPER])
def test_the_roles_that_run_the_installation_read_the_configuration(
    client_for: ClientFor, role: UserRole
) -> None:
    """`roles.py` said these two "differ from an operator only in routes that do not exist
    yet -- settings, debug views -- and they will differ here when those land". This is
    that landing."""
    assert client_for(role).get("/settings").status_code == 200


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.CANTEEN_STAFF])
def test_watching_children_is_not_the_same_right_as_reading_the_installation(
    client_for: ClientFor, role: UserRole
) -> None:
    """Capability per page, not per rank. An operator reviews events; the detector's
    thresholds, the ROI rectangles and the RTSP hosts of every camera in the school are a
    different question, and §14's roles overlap rather than nest."""
    assert client_for(role).get("/settings").status_code == 403


def test_the_capability_is_granted_and_not_inherited() -> None:
    assert Capability.VIEW_SETTINGS in capabilities_for(UserRole.ADMIN)
    assert Capability.VIEW_SETTINGS in capabilities_for(UserRole.DEVELOPER)
    assert Capability.VIEW_SETTINGS not in capabilities_for(UserRole.OPERATOR)
    assert Capability.VIEW_SETTINGS not in capabilities_for(UserRole.CANTEEN_STAFF)


def test_the_nav_never_draws_a_link_the_role_cannot_follow(client_for: ClientFor) -> None:
    """One source of truth for the menu and the gate. A link into a 403 is reported by the
    school as "the system is broken"."""
    assert 'href="/settings"' in client_for(UserRole.ADMIN).get("/").text
    assert 'href="/settings"' not in client_for(UserRole.OPERATOR).get("/").text


# -- what it shows -----------------------------------------------------------


def test_a_value_is_shown_beside_the_file_that_actually_supplies_it(admin: TestClient) -> None:
    """§9 asks to see confidence, thresholds, ROI, cooldowns. Showing the number alone is
    what let `min_score: 0.50` be documented in one file and overridden in six others."""
    page = admin.get("/settings").text

    assert "config/profiles/hall.yaml" in page
    assert "config/base.yaml" in page
    assert "config/cameras/hall_left.yaml" in page
    assert "Холл слева" in page


def test_the_detector_thresholds_and_the_roi_are_actually_on_the_page(admin: TestClient) -> None:
    page = admin.get("/settings").text

    assert "bullying.metrics.acceleration_threshold" in page, "a threshold §9 asks for"
    assert "bullying.zones.normal_flow" in page, "the ROI §9 asks for"
    assert "canteen.entry.recognition.min_score" in page, "the recognition floor"


def test_the_page_says_the_files_are_not_necessarily_what_is_running(admin: TestClient) -> None:
    """The page reads the files. The supervisor read them WHEN IT STARTED. Those are two
    different facts, and presenting the first as the second would be a fresh instance of
    the exact disease this page is built to expose."""
    page = admin.get("/settings").text

    assert "перезапуск" in page.lower(), "nothing warns that an edited file is not yet in force"


# -- the token that must never be rendered -----------------------------------


def test_the_telegram_token_is_never_rendered(admin: TestClient, settings: Settings) -> None:
    """Audit H-04: the legacy settings page served the live bot token to anonymous callers.
    R4 keeps it in the environment; this keeps it out of the HTML."""
    token = settings.telegram_bot_token.get_secret_value()
    assert token, "the fixture must supply a token or this test proves nothing"

    page = admin.get("/settings").text

    assert token not in page
    assert "AAxxxx" not in page, "a prefix of the token is still the token"
    assert settings.telegram_chat_id not in page, "the chat id names where alerts go"


def test_telegram_is_shown_as_on_or_off_and_nothing_more(admin: TestClient) -> None:
    page = admin.get("/settings").text

    assert "Telegram" in page
    assert "включён" in page or "отключён" in page


# -- no write path of any kind -----------------------------------------------


def test_the_settings_router_exposes_no_unsafe_method() -> None:
    """The page is a reader. Not "has no write today" -- has no way to write.

    A POST here would be a second source of truth for a value that lives in YAML, and the
    supervisor would go on running what it loaded at startup: the browser and the file
    would disagree, with nothing to say which was in force.
    """
    offenders = [
        (route.path, sorted(set(route.methods) - SAFE_METHODS))
        for route in settings_routes.router.routes
        if set(getattr(route, "methods", set())) - SAFE_METHODS
    ]
    assert not offenders, f"the settings page can be POSTed to: {offenders}"


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                names.add(function.attr)
            elif isinstance(function, ast.Name):
                names.add(function.id)
    return names


WRITERS = frozenset(
    {"dump", "dump_all", "safe_dump", "write_text", "write_bytes", "open", "mkdir", "unlink"}
)


@pytest.mark.parametrize(
    "module",
    [
        SRC_DIR / "qorgan" / "web" / "routes" / "settings.py",
        SRC_DIR / "qorgan" / "config" / "provenance.py",
    ],
    ids=lambda p: p.name,
)
def test_nothing_behind_the_settings_page_can_write_a_file(module: Path) -> None:
    """Belt to the router's braces. A write bolted onto some other router but implemented
    here would still be a browser editing YAML, and the route test would not see it."""
    assert not (_calls(module) & WRITERS), (
        f"{module.name} calls a filesystem writer; the settings page must only read"
    )


def test_the_template_offers_no_form(admin: TestClient) -> None:
    """An inert control that looks live is worse than an honest absence, and a live one
    would be a browser editing YAML. Neither belongs in this template.

    If a control ever becomes real, this test is meant to be edited deliberately, with the
    reason written down -- not to quietly stop being true.
    """
    markup = TEMPLATE.read_text(encoding="utf-8")

    assert "<form" not in markup
    assert "<button" not in markup
    assert 'method="post"' not in markup.lower()
    assert admin.get("/settings").status_code == 200


# -- zero side effects on load -----------------------------------------------


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.yaml"))
    }


def test_opening_the_page_changes_no_configuration_file(admin: TestClient) -> None:
    """Audit H-18: opening a legacy page restarted the AI workers. This page is the closest
    descendant of that mistake, so a GET must change nothing at all -- starting with the
    files it is reading."""
    before = _fingerprint(CONFIG_DIR)
    assert before, "no config files fingerprinted; this test would pass vacuously"

    admin.get("/settings")
    admin.get("/settings")

    assert _fingerprint(CONFIG_DIR) == before


def test_opening_the_page_writes_no_row(admin: TestClient, session: Session) -> None:
    """`app_settings` and `mode_logs` are where a write would land if somebody made this
    page "remember" something. Nothing may appear in either from a GET."""
    admin.get("/settings")

    session.expire_all()
    assert session.scalar(select(func.count(AppSetting.key))) == 0
    assert session.scalar(select(func.count(ModeLog.id))) == 0


def test_the_page_is_the_same_twice(admin: TestClient) -> None:
    """A page that differs between two identical GETs did something the second time."""
    assert admin.get("/settings").text == admin.get("/settings").text


# -- the two controls that are NOT available ---------------------------------


def test_the_operating_mode_is_shown_as_something_this_process_cannot_switch(
    admin: TestClient,
) -> None:
    """There is no control channel from the web process to the supervisor. The supervisor
    reads `config/workers.yaml` and the cameras' `enabled:` at startup and owns the process
    table; nothing in `src/` writes `mode_logs` or reads `app_settings`.

    R3 is the reason this is a feature and not a gap: coverage stays decided by config and
    the supervisor, never by which browser tab is open.
    """
    page = admin.get("/settings").text

    assert "config/workers.yaml" in page, "the page must name what actually decides coverage"
    assert "недоступно" in page, "the control is not marked unavailable"


def test_disabling_a_camera_points_at_the_file_and_not_at_a_switch(admin: TestClient) -> None:
    page = admin.get("/settings").text

    assert "config/cameras/hall_left.yaml" in page
    assert "/settings/cameras" not in page, "a control that does not exist must not be linked"


# -- escaping ----------------------------------------------------------------


def test_a_camera_name_out_of_yaml_cannot_inject_script(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Config is not trusted input either: it is edited by hand, on site, by whoever is
    holding the laptop. The legacy built its DOM with innerHTML from server JSON and a
    pupil's name gave stored XSS in the operator's browser (audit H-05)."""
    import shutil

    from qorgan.web.app import create_app

    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    camera = directory / "cameras" / "hall_left.yaml"
    camera.write_text(
        camera.read_text(encoding="utf-8").replace(
            'display_name: "Холл слева"', "display_name: \"<img src=x onerror='alert(1)'>\""
        ),
        encoding="utf-8",
    )
    override_settings(settings.model_copy(update={"config_dir": directory}))

    session.add(User(username="a", password_hash=hash_password(PASSWORD), role=UserRole.ADMIN))
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as client:
        client.post("/login", data=with_token(client, {"username": "a", "password": PASSWORD}))
        page = client.get("/settings").text

    assert "<img src=x onerror=" not in page
    assert "&lt;img src=x onerror=" in page, "the camera name was not escaped"


def test_a_broken_config_file_is_reported_rather_than_crashing_the_page(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The one page that can explain a config error must not be the page a config error
    takes down. `qorgan config validate` is a terminal away; the person holding the laptop
    is looking at a browser."""
    import shutil

    from qorgan.web.app import create_app

    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    session.add(User(username="a", password_hash=hash_password(PASSWORD), role=UserRole.ADMIN))
    session.commit()
    override_settings(settings.model_copy(update={"config_dir": directory}))

    with TestClient(create_app(), follow_redirects=False) as client:
        client.post("/login", data=with_token(client, {"username": "a", "password": PASSWORD}))
        # Broken only AFTER the app has started: create_app() loads the fleet at lifespan,
        # and a startup failure is a different (already covered) story from a file edited
        # while the dashboard is up -- which is precisely when somebody opens this page.
        (directory / "cameras" / "hall_left.yaml").write_text("nope: [", encoding="utf-8")
        response = client.get("/settings")

    assert response.status_code == 200
    assert "hall_left.yaml" in response.text

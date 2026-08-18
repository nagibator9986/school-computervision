"""The three ways a school LAN deployment silently loses its front door.

The client asked for the dashboard to be reachable from other computers on the school
network (their §3). That is a non-loopback bind in front of children's photographs and
live corridor video -- the exact surface the legacy system exposed to everyone (C-01).

v2 fixed authentication. These tests cover the three ways the fix could be handed back:

1. The session cookie is signed with a key that is printed in the source, so anyone
   who can read the repo can forge a session and walk straight past the login.
2. The .env that holds the real key is looked for in the current directory, so a
   process started from anywhere else (Task Scheduler has no working directory)
   silently falls back to that same published key.
3. QORGAN_ENV=prod marks the cookie Secure, so over plain http:// the browser never
   sends it back and the login redirects to itself forever, with no diagnostic.

Each is individually survivable. Together they are: "I set WEB_HOST=0.0.0.0 like the
manual said" -> children's video on the LAN behind a key from GitHub.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.orm import Session

from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.settings import Settings, override_settings, project_root
from qorgan.web.app import create_app
from tests.conftest import CONFIG_DIR, SRC_DIR
from tests.web_login import with_token

# Not a secret: a stand-in for what `secrets.token_urlsafe(48)` produces, long enough
# that the redactor treats it as one and distinct from the published default.
A_GENERATED_KEY = SecretStr("Vn7-test-only-not-a-real-key-" + "q" * 32)

PASSWORD = "correct-horse-battery"


def _settings(**overrides: object) -> Settings:
    """Settings as a deployment would build them, minus the disk it would touch."""
    base: dict[str, object] = {
        "_env_file": None,
        "config_dir": CONFIG_DIR,
        "preview_address": "tcp://127.0.0.1:0",
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _web_settings(base: Settings, **overrides: object) -> Settings:
    """A variant of the suite's throwaway settings, keeping its database.

    The database_url must stay the fixture's: the `session` fixture created the schema
    in that file, and the app's auth middleware opens its own connection through the
    global settings. Pointing them at two different files makes every request 500 with
    "no such table: users" -- a fixture bug wearing the costume of a real failure.
    """
    return _settings(
        database_url=base.database_url,
        media_root=base.media_root,
        log_dir=base.log_dir,
        secret_key=A_GENERATED_KEY,
        **overrides,
    )


def _plain_http_client() -> TestClient:
    """The dashboard as the school reaches it: http://, no certificate anywhere."""
    return TestClient(create_app(), follow_redirects=False, base_url="http://school-lan")


# -- 1. a key from the public source, on the school network ------------------------


# S104 throughout: binding to every interface is the deployment under test, not a slip.
@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "school-server"])  # noqa: S104
def test_a_dashboard_on_the_school_network_signed_with_a_published_key_refuses_to_start(
    host: str,
) -> None:
    """Anyone who can read settings.py can forge an operator session. Do not start."""
    with pytest.raises(ValidationError) as caught:
        _settings(web_host=host)
    assert "SECRET_KEY" in str(caught.value)


def test_the_refusal_tells_the_operator_which_variable_to_set_and_how_to_generate_it() -> None:
    """An error a school administrator cannot act on is an error that gets worked around."""
    with pytest.raises(ValidationError) as caught:
        _settings(web_host="0.0.0.0")  # noqa: S104
    message = str(caught.value)
    assert "SECRET_KEY" in message
    assert "secrets.token_urlsafe" in message, "the message must carry the command to run"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.5", "localhost", "::1"])
def test_a_developer_on_loopback_may_still_run_with_the_dev_key(host: str) -> None:
    """The dev key on loopback reaches nobody. Breaking this would be gratuitous."""
    assert _settings(web_host=host).web_host == host


def test_a_generated_key_is_what_the_school_network_bind_was_waiting_for() -> None:
    settings = _settings(web_host="0.0.0.0", secret_key=A_GENERATED_KEY)  # noqa: S104
    assert settings.web_host == "0.0.0.0"  # noqa: S104


def test_production_with_the_published_key_refuses_even_when_bound_to_loopback() -> None:
    """A reverse proxy terminating TLS forwards to 127.0.0.1 -- the bind looks private
    while the dashboard is on the LAN. Host alone cannot see that; QORGAN_ENV can."""
    with pytest.raises(ValidationError) as caught:
        _settings(qorgan_env="prod", web_host="127.0.0.1", web_https=True)
    assert "SECRET_KEY" in str(caught.value)


# -- 2. the .env that was never read -----------------------------------------------


def test_the_env_file_is_found_next_to_the_install_not_wherever_the_process_was_started() -> None:
    configured = Settings.model_config.get("env_file")
    assert isinstance(configured, Path), "env_file must be a resolved path, not a bare name"
    assert configured.is_absolute(), "a relative env_file is read against the current directory"
    assert configured == project_root() / ".env"


def test_an_env_file_in_the_current_directory_cannot_shadow_the_one_beside_the_install(
    tmp_path: Path,
) -> None:
    """Task Scheduler starts processes with no working directory at all. Reading .env
    from wherever we happened to land is how SECRET_KEY silently became the dev key
    while the real .env sat unread next to the code.
    """
    (tmp_path / ".env").write_text("SECRET_KEY=decoy-from-the-wrong-directory\n", encoding="utf-8")

    environment = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
    environment["PYTHONPATH"] = str(SRC_DIR)
    result = subprocess.run(  # noqa: S603 - our own interpreter, literal argv
        [
            sys.executable,
            "-c",
            "from qorgan.settings import Settings;"
            "print(Settings().secret_key.get_secret_value() == 'decoy-from-the-wrong-directory')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "a .env in the working directory was read"


# -- 3. the login that redirects to itself forever ---------------------------------


def test_production_without_tls_refuses_to_start_rather_than_looping_the_login_forever() -> None:
    """Secure cookies over plain http:// are never returned by the browser: the operator
    types the right password and lands back on the login screen, with nothing logged."""
    with pytest.raises(ValidationError) as caught:
        _settings(qorgan_env="prod", secret_key=A_GENERATED_KEY, web_https=False)
    message = str(caught.value)
    assert "WEB_HTTPS" in message
    assert "QORGAN_ENV" in message


def test_production_behind_a_tls_proxy_starts() -> None:
    settings = _settings(qorgan_env="prod", secret_key=A_GENERATED_KEY, web_https=True)
    assert settings.web_https is True


@pytest.mark.parametrize(
    ("web_https", "expect_secure"),
    [(False, False), (True, True)],
)
def test_the_session_cookie_is_marked_secure_only_when_the_dashboard_really_speaks_https(
    settings: Settings, session: Session, web_https: bool, expect_secure: bool
) -> None:
    """Secure must follow the transport, not the name of the environment. Deriving it
    from QORGAN_ENV is what made 'prod' mean 'unusable over http' with no diagnostic."""
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    override_settings(_web_settings(settings, web_https=web_https))
    try:
        with _plain_http_client() as client:
            # The flag is read off the GET, not off the login POST, because the session
            # cookie is now first written THERE -- issuing the CSRF token starts the
            # session. It also has to be read there: when `Secure` is set and the
            # transport is plain http (the [True-True] case), the browser will not send
            # the cookie back at all, so the token cannot round-trip and the POST is a
            # 403 rather than a 303. That is not new breakage, it is the documented
            # "prod without TLS logs in forever" failure arriving sooner and louder --
            # and this test is about the flag, which the GET shows directly.
            page = client.get("/login")
            cookie = page.headers["set-cookie"]
            assert ("secure" in cookie.lower()) is expect_secure
    finally:
        override_settings(settings)


def test_a_correct_password_over_plain_http_reaches_the_dashboard_not_the_login_screen(
    settings: Settings, session: Session
) -> None:
    """The bug in the domain's own words: right password, and the dashboard bounces you
    back to the login screen forever."""
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    override_settings(_web_settings(settings, web_https=False))
    try:
        with _plain_http_client() as client:
            client.post("/login", data=with_token(client, {"username": "op", "password": PASSWORD}))
            landing = client.get("/")
            assert landing.status_code == 200, "login looped back to the login screen"
    finally:
        override_settings(settings)

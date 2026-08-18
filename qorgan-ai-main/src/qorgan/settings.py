"""Process-wide settings. Secrets live here and nowhere else (rule R4).

Everything is read from the environment (optionally seeded from a .env file).
No secret may appear in a YAML file, in the database, in a log line, or burnt
into a debug image — the legacy system did all four.
"""

from __future__ import annotations

import ipaddress
import os
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qorgan.redaction import MIN_SECRET_LENGTH

ROOT_MARKER: Path = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Repo root. Relative paths in settings resolve against this."""
    return ROOT_MARKER


# Absolute, and deliberately not ".env". pydantic-settings resolves a relative env_file
# against the *current working directory at instantiation*, while config_dir/media_root
# resolve against the install root -- so running qorgan from anywhere else read no .env
# at all and silently fell back to the dev key, a blank RTSP password and no Telegram,
# with nothing logged. Windows Task Scheduler starts processes with no working
# directory, which is exactly how this system is going to be launched.
ENV_FILE: Path = ROOT_MARKER / ".env"

# The key that ships in the source. Anyone who can read this file can sign a session
# cookie with it, so it is only ever safe where nobody else can reach the socket.
DEV_SECRET_KEY = "dev-only-insecure-key"  # noqa: S105 - named so we can recognise and refuse it

GENERATE_KEY_COMMAND = 'python -c "import secrets; print(secrets.token_urlsafe(48))"'


def binds_beyond_this_machine(host: str) -> bool:
    """Is the dashboard reachable from another computer on the school network?

    Anything not demonstrably loopback counts as exposed. A hostname we cannot parse
    resolves somewhere we cannot see, and guessing "probably local" is how live video
    of children ends up served to the LAN.
    """
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


class RtspCredentials:
    """A username/password pair for one camera. Never stringifies its secret."""

    __slots__ = ("password", "user")

    def __init__(self, user: str, password: SecretStr) -> None:
        self.user = user
        self.password = password

    def __repr__(self) -> str:
        return f"RtspCredentials(user={self.user!r}, password=SecretStr('**********'))"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    qorgan_env: str = "dev"

    # Web. Binds to loopback by default; exposing it is an explicit act (R5).
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    secret_key: SecretStr = SecretStr(DEV_SECRET_KEY)

    # Is the dashboard actually served over TLS -- directly, or behind a proxy that
    # terminates it? This is the only honest source for the cookie's Secure flag. It
    # used to be derived from qorgan_env, which knows nothing about the transport.
    web_https: bool = False

    # Camera credentials. Per-camera overrides go through credentials_for().
    rtsp_user: str = "admin"
    rtsp_password: SecretStr = SecretStr("")

    # Notifications. Blank token disables Telegram entirely.
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""

    # Storage.
    database_url: str = "sqlite+pysqlite:///./data/qorgan.sqlite3"
    media_root: Path = Path("./media")
    log_dir: Path = Path("./logs")
    config_dir: Path = Path("./config")

    # The preview bus. Workers PUB here, the web process SUBs. Loopback only: these
    # are frames of children in a school corridor and they do not leave the machine.
    preview_address: str = "tcp://127.0.0.1:5556"
    preview_stale_after_seconds: float = 10.0

    school_timezone: str = "Asia/Almaty"

    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("school_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        ZoneInfo(value)  # raises for an unknown zone; fail at startup, not at report time
        return value

    @model_validator(mode="after")
    def _refuse_the_published_key_where_it_can_be_reached(self) -> Settings:
        """A dev key is fine on loopback and nowhere else.

        Model-level rather than per-field: the danger is not web_host and not secret_key,
        it is the pair. Neither field can see the other.

        prod counts as exposed regardless of host, because the standard deployment --
        a proxy terminating TLS and forwarding to 127.0.0.1 -- has a loopback bind and
        a dashboard on the LAN. Judging exposure by host alone is blind exactly there.
        """
        if self.secret_key.get_secret_value() != DEV_SECRET_KEY:
            return self

        if binds_beyond_this_machine(self.web_host):
            reach = f"WEB_HOST={self.web_host} accepts connections from other computers"
        elif self.qorgan_env == "prod":
            reach = "QORGAN_ENV=prod, which is not a place for a key that ships in the source"
        else:
            return self  # loopback, dev: the developer's normal path

        raise ValueError(
            f"SECRET_KEY is still the built-in default and {reach}. Session cookies signed "
            "with it can be forged by anyone who has read the source, which would hand over "
            "live video and photographs of children.\n"
            f"Generate a key:  {GENERATE_KEY_COMMAND}\n"
            f"Then set SECRET_KEY in {ENV_FILE}"
        )

    @model_validator(mode="after")
    def _refuse_production_that_cannot_carry_a_secure_cookie(self) -> Settings:
        """prod without TLS used to be an invisible infinite redirect.

        Secure cookies are never sent over plain http://, so the login accepted the right
        password, set a cookie the browser then refused to return, and bounced back to the
        login screen -- forever, logging nothing. Fail here, where we can still say why.
        """
        if self.qorgan_env == "prod" and not self.web_https:
            raise ValueError(
                "QORGAN_ENV=prod marks the session cookie Secure, and browsers never send a "
                "Secure cookie over plain http:// -- the login would accept the correct "
                "password and then redirect to itself forever, with nothing in the log.\n"
                "Either serve the dashboard over HTTPS (directly or behind a proxy that "
                "terminates TLS) and set WEB_HTTPS=true, or run with QORGAN_ENV=dev."
            )
        return self

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.school_timezone)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token.get_secret_value() and self.telegram_chat_id)

    def credentials_for(self, camera_name: str) -> RtspCredentials:
        """Credentials for one camera, falling back to the fleet-wide default.

        Per-camera override: RTSP_USER__HALL_LEFT / RTSP_PASSWORD__HALL_LEFT.
        The audit's C-03 finding was one shared near-dictionary password across
        the whole fleet; this makes per-NVR passwords possible without touching code.
        """
        suffix = camera_name.upper()
        user = os.getenv(f"RTSP_USER__{suffix}") or self.rtsp_user
        raw_password = os.getenv(f"RTSP_PASSWORD__{suffix}")
        password = SecretStr(raw_password) if raw_password else self.rtsp_password
        return RtspCredentials(user=user, password=password)

    def secret_values(self) -> list[str]:
        """Every secret string known to this process, for the log redactor."""
        candidates = [
            self.rtsp_password.get_secret_value(),
            self.telegram_bot_token.get_secret_value(),
            self.secret_key.get_secret_value(),
            # Not a SecretStr, and registered anyway. The chat id names the staff group
            # the school's bullying alerts go to; it is env-only for the same reason the
            # token is (R4). The token is caught by shape as well as by value -- there is
            # a regex for it in `redaction` -- but `-1001234567890` looks like any other
            # number, so being listed HERE is the only thing that keeps it out of a log
            # line, or off the delivery-history page, which renders `last_error` text
            # that a third party wrote.
            self.telegram_chat_id,
            *(v for k, v in os.environ.items() if k.startswith("RTSP_PASSWORD__")),
        ]
        return [c for c in candidates if c and len(c) >= MIN_SECRET_LENGTH]


_OVERRIDE: Settings | None = None


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    return _OVERRIDE or _cached_settings()


def override_settings(settings: Settings | None) -> None:
    """Test hook. Production code never calls this."""
    global _OVERRIDE  # noqa: PLW0603
    _OVERRIDE = settings


def resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root() / p).resolve()


settings_field_names: tuple[str, ...] = tuple(Settings.model_fields)
_ = Field  # re-exported for callers that build partial settings in tests

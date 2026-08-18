"""Secrets come from the environment, and media paths stay inside MEDIA_ROOT."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.config.common import RtspSettings
from qorgan.paths import PathOutsideRoot, ensure_within, to_absolute, to_relative
from qorgan.rtsp import build_url, safe_url
from qorgan.settings import Settings
from tests.conftest import REPO_ROOT


def test_a_secret_does_not_leak_through_repr(settings: Settings) -> None:
    assert settings.rtsp_password.get_secret_value() not in repr(settings)
    assert settings.telegram_bot_token.get_secret_value() not in repr(settings)


def test_credentials_come_from_the_environment(settings: Settings) -> None:
    credentials = settings.credentials_for("hall_left")
    assert credentials.user == "admin"
    assert credentials.password.get_secret_value() == "sup3r-s3cret-camera-pw"
    assert "sup3r" not in repr(credentials)


def test_a_camera_can_have_its_own_password(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy used one near-dictionary password across the entire fleet (audit C-03)."""
    monkeypatch.setenv("RTSP_PASSWORD__HALL_LEFT", "only-for-this-nvr")
    assert settings.credentials_for("hall_left").password.get_secret_value() == "only-for-this-nvr"
    assert settings.credentials_for("hall_right").password.get_secret_value() != "only-for-this-nvr"


def test_the_url_you_may_print_has_no_credentials_in_it(settings: Settings) -> None:
    rtsp = RtspSettings(host="192.168.1.4")

    real = build_url("hall_left", rtsp)
    assert "sup3r-s3cret-camera-pw" in real  # it is a working URL...

    printable = safe_url(rtsp)
    assert "@" not in printable  # ...and this one is the one that reaches a log
    assert printable == "rtsp://192.168.1.4:554/Streaming/Channels/102"


def test_a_burst_url_is_refused_when_the_camera_has_no_burst_stream(settings: Settings) -> None:
    rtsp = RtspSettings(host="192.168.1.5", burst_path=None)
    with pytest.raises(ValueError, match="no rtsp.burst_path"):
        build_url("canteen_inside_left", rtsp, burst=True)


def test_telegram_is_disabled_without_a_token() -> None:
    # `_env_file=None` so this stays hermetic: Settings is a pydantic BaseSettings
    # with env_file=".env", and only .env.example exists today. This test passing
    # is currently luck, not a guarantee -- the moment a real .env exists (the
    # project's own HANDOFF tells developers to create one), a raw Settings() call
    # with no other overrides would silently inherit its database_url, media_root,
    # and preview_address instead of the defaults this test actually means to check.
    settings = Settings(telegram_bot_token="", telegram_chat_id="-100", _env_file=None)
    assert not settings.telegram_enabled


def test_a_media_path_round_trips_relative(settings: Settings, tmp_path: Path) -> None:
    root = tmp_path / "media"
    (root / "snapshots").mkdir(parents=True)
    absolute = root / "snapshots" / "event_1.jpg"
    absolute.touch()

    relative = to_relative(absolute, root)
    assert relative == "snapshots/event_1.jpg"
    assert to_absolute(relative, root) == absolute.resolve()


def test_a_path_outside_the_media_root_is_refused(settings: Settings, tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    with pytest.raises(PathOutsideRoot):
        to_relative(tmp_path / "elsewhere.jpg", root)
    with pytest.raises(PathOutsideRoot):
        to_absolute("../../windows/system32/config", root)


def test_ensure_within_blocks_traversal(tmp_path: Path) -> None:
    """Used by the ZIP import, where the member names are untrusted (audit H-03:
    the legacy called extractall() with no member validation at all)."""
    root = tmp_path / "safe"
    root.mkdir()
    assert ensure_within(root, root / "a" / "b.jpg")
    with pytest.raises(PathOutsideRoot):
        ensure_within(root, root / ".." / "escaped.jpg")


def test_env_example_documents_every_secret_setting() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("SECRET_KEY", "RTSP_USER", "RTSP_PASSWORD", "TELEGRAM_BOT_TOKEN", "DATABASE_URL"):
        assert name in example, f"{name} is undocumented in .env.example"

"""A secret must not survive the trip to a log file."""

from __future__ import annotations

import logging
from pathlib import Path

from qorgan.logging_setup import setup_logging
from qorgan.redaction import MASK, Redactor, redact
from qorgan.settings import Settings


def test_url_credentials_are_masked_even_when_the_password_is_unknown() -> None:
    text = "connecting to rtsp://admin:hunter2@192.168.1.4:554/Streaming/Channels/102"
    result = redact(text)
    assert "hunter2" not in result
    assert "admin:hunter2" not in result
    # The host and channel survive -- that is what you actually need in a log line.
    assert "192.168.1.4:554/Streaming/Channels/102" in result


def test_telegram_token_shape_is_masked_inside_an_api_url() -> None:
    """The token appears as `/bot<token>/`, with no word boundary before the digits.
    A regex anchored on \\b would sail straight past it."""
    token = "123456789:AAH1qQabcdefghijklmnopqrstuvwxyz012345"
    result = redact(f"calling https://api.telegram.org/bot{token}/sendPhoto")

    assert "AAH1qQ" not in result
    assert MASK in result
    assert "api.telegram.org" in result  # the useful part survives


def test_known_secret_literal_is_masked_anywhere_it_appears() -> None:
    redactor = Redactor(["sup3r-s3cret-camera-pw"])
    assert "sup3r" not in redactor.redact("password was sup3r-s3cret-camera-pw all along")


def test_a_short_string_is_not_treated_as_a_secret() -> None:
    """Otherwise a 1-character secret would mask every letter in every log line."""
    redactor = Redactor(["ab"])
    assert redactor.redact("a plain sentence about a cab") == "a plain sentence about a cab"


def test_secrets_do_not_reach_the_log_file(settings: Settings, tmp_path: Path) -> None:
    setup_logging("redaction_test")
    logging.getLogger("test").error(
        "stream failed: rtsp://admin:%s@192.168.1.4:554/x",
        settings.rtsp_password.get_secret_value(),
    )
    logging.shutdown()

    written = (tmp_path / "logs" / "redaction_test.log").read_text(encoding="utf-8")
    assert settings.rtsp_password.get_secret_value() not in written
    assert "192.168.1.4" in written  # the useful part is still there

"""Scrub secrets out of anything on its way to a log file, a JSON dump, or an image overlay.

The legacy system printed full `rtsp://user:password@host/...` URLs to stdout and
rendered them as text onto debug JPEGs that an unauthenticated web UI served
(audit C-02). Redaction is therefore not a nicety here; it is a hard requirement,
and it is applied at the formatter so nothing can route around it.
"""

from __future__ import annotations

import logging
import re

MASK = "***REDACTED***"

# Below this length a "secret" is more likely to be a substring of ordinary words,
# and masking it would shred every log line it touches.
MIN_SECRET_LENGTH = 4

# rtsp://user:password@host  ->  rtsp://***:***@host
_URL_CREDENTIALS = re.compile(r"(?P<scheme>\w+://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@")

# Telegram bot tokens look like 1234567890:AA... (>=30 chars after the colon).
# No leading \b: in the wild the token appears as `api.telegram.org/bot<token>/`,
# and there is no word boundary between the "t" of "bot" and the first digit.
_TELEGRAM_TOKEN = re.compile(r"\d{6,}:[A-Za-z0-9_-]{30,}")


class Redactor:
    """Masks known secret values plus anything that structurally looks like a secret."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        self._literals: list[str] = []
        for secret in secrets or []:
            self.add_secret(secret)

    def add_secret(self, value: str) -> None:
        if value and len(value) >= MIN_SECRET_LENGTH and value not in self._literals:
            self._literals.append(value)
            # Longest first, so a secret that contains another is masked whole.
            self._literals.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        for literal in self._literals:
            text = text.replace(literal, MASK)
        text = _URL_CREDENTIALS.sub(r"\g<scheme>***:***@", text)
        return _TELEGRAM_TOKEN.sub(MASK, text)


_REDACTOR = Redactor()


def get_redactor() -> Redactor:
    return _REDACTOR


def redact(text: str) -> str:
    return _REDACTOR.redact(text)


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and scrubs its output.

    Applied at format time rather than as a logging.Filter because a Filter cannot
    see the interpolated message, only the args.
    """

    def __init__(self, inner: logging.Formatter, redactor: Redactor | None = None) -> None:
        super().__init__()
        self._inner = inner
        self._redactor = redactor or _REDACTOR

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.redact(self._inner.format(record))

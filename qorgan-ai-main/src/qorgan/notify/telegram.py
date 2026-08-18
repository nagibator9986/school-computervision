"""Telegram. It actually uploads the evidence.

**The legacy's media delivery was broken by design, not by accident.** It sent a text
message containing links like `http://127.0.0.1:8000/media/snapshots/...` — the loopback
address *of the server*. A teacher receiving that alert on their phone, at home, could
never open it. Not once, not ever. The system that existed to show staff what happened
sent them a URL to their own phone's localhost.

So: `sendPhoto` and `sendVideo`, with the bytes.

Everything else here exists because an alert about a child being hurt must not be lost:

  * A **429** from Telegram is not a failure. It is Telegram telling us exactly how long
    to wait, in the `retry_after` field. The legacy ignored it and dropped the message.
  * A network blip is not a failure either — it is a retry, with backoff.
  * Every attempt is written to the database, successes included. The legacy logged only
    the failures of a WhatsApp *placeholder* provider that always failed (audit M-14).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from qorgan.logging_setup import get_logger
from qorgan.settings import get_settings

logger = get_logger(__name__)

API = "https://api.telegram.org"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Telegram's own limits. Exceeding them is a permanent failure, not a retryable one.
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024

TOO_MANY_REQUESTS = 429


@dataclass(frozen=True, slots=True)
class SendResult:
    ok: bool
    message_id: str | None = None
    error: str | None = None
    # Telegram told us how long to wait. Honour it rather than guessing.
    retry_after: float | None = None
    # A permanent failure: retrying will never help. A bad token, a wrong chat id, a file
    # too large. Retrying these forever is how a queue fills with poison.
    permanent: bool = False


class TelegramClient:
    """Thin, synchronous, and honest about what failed."""

    def __init__(self, token: str, chat_id: str, *, base_url: str = API) -> None:
        self._token = token
        self._chat_id = chat_id
        self._base = f"{base_url}/bot{token}"

    @classmethod
    def from_settings(cls) -> TelegramClient | None:
        """None when Telegram is not configured. That is a valid state, not an error:
        the system must run without it."""
        settings = get_settings()
        if not settings.telegram_enabled:
            return None
        return cls(settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id)

    # -- sending -------------------------------------------------------------

    def send_message(self, text: str) -> SendResult:
        return self._post("sendMessage", data={"chat_id": self._chat_id, "text": text})

    def send_photo(self, photo: Path, caption: str) -> SendResult:
        """The snapshot itself, not a link to it."""
        too_big = _too_big(photo, MAX_PHOTO_BYTES)
        if too_big:
            return too_big

        with photo.open("rb") as handle:
            return self._post(
                "sendPhoto",
                data={"chat_id": self._chat_id, "caption": caption},
                files={"photo": (photo.name, handle, "image/jpeg")},
            )

    def send_video(self, video: Path, caption: str) -> SendResult:
        too_big = _too_big(video, MAX_VIDEO_BYTES)
        if too_big:
            return too_big

        with video.open("rb") as handle:
            return self._post(
                "sendVideo",
                data={"chat_id": self._chat_id, "caption": caption, "supports_streaming": "true"},
                files={"video": (video.name, handle, "video/mp4")},
            )

    # -- the wire ------------------------------------------------------------

    def _post(self, method: str, **kwargs) -> SendResult:
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                response = client.post(f"{self._base}/{method}", **kwargs)
        except httpx.HTTPError as exc:
            # The network is down. That is temporary, and this alert still matters.
            return SendResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        return _interpret(response)


def _interpret(response: httpx.Response) -> SendResult:
    if response.status_code == TOO_MANY_REQUESTS:
        # Telegram is telling us precisely how long to wait. The legacy ignored this and
        # simply lost the message.
        retry_after = _retry_after(response)
        return SendResult(
            ok=False,
            error=f"rate limited; retry after {retry_after:.0f}s",
            retry_after=retry_after,
        )

    try:
        payload = response.json()
    except ValueError:
        return SendResult(ok=False, error=f"HTTP {response.status_code}: unreadable response")

    if payload.get("ok"):
        result = payload.get("result") or {}
        return SendResult(ok=True, message_id=str(result.get("message_id", "")))

    description = str(payload.get("description", "unknown error"))
    return SendResult(
        ok=False,
        error=f"HTTP {response.status_code}: {description}",
        # A bad token or a wrong chat id will still be bad in five minutes. Retrying it
        # forever fills the queue with poison and delays every real alert behind it.
        permanent=response.status_code in (400, 401, 403, 404),
    )


def _retry_after(response: httpx.Response) -> float:
    try:
        payload = response.json()
        return float(payload["parameters"]["retry_after"])
    except (ValueError, KeyError, TypeError):
        return float(response.headers.get("retry-after", 30))


def _too_big(path: Path, limit: int) -> SendResult | None:
    if not path.is_file():
        return SendResult(ok=False, error=f"missing media: {path.name}", permanent=True)

    size = path.stat().st_size
    if size > limit:
        return SendResult(
            ok=False,
            error=f"{path.name} is {size / 1e6:.1f} MB; Telegram's limit is {limit / 1e6:.0f} MB",
            permanent=True,
        )
    return None

"""One logging setup for every process (web, supervisor, workers).

Legacy logged with bare print(), including children's full names, unrotated,
from ~18k lines across three god-files. Here: structured records, rotation,
and every line passed through the redactor on the way out.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

from qorgan.redaction import RedactingFormatter, get_redactor
from qorgan.settings import get_settings, resolve

MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "process": record.processName,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _text_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s %(levelname)-7s %(processName)-16s %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_handlers(log_path: Path, as_json: bool) -> list[logging.Handler]:
    inner = JsonFormatter() if as_json else _text_formatter()
    formatter = RedactingFormatter(inner)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(RedactingFormatter(_text_formatter()))

    return [file_handler, stream_handler]


def setup_logging(process_name: str = "main") -> None:
    """Idempotent. Safe to call once per process, at the top of its entry point."""
    settings = get_settings()

    redactor = get_redactor()
    for secret in settings.secret_values():
        redactor.add_secret(secret)

    log_dir = resolve(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    for handler in _build_handlers(log_dir / f"{process_name}.log", settings.log_json):
        root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    logging.getLogger("uvicorn.access").propagate = False
    logging.captureWarnings(capture=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

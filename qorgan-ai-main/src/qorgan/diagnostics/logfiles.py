"""Recent failures, read back out of the rotating files `logging_setup` writes.

**Why files and not a table.** Camera connection errors, model failures and database
errors have no table in this system, and inventing one that only the web process filled
in would record the failures of the process that is least likely to have them.
`worker_heartbeats.last_error` is one string per group, overwritten by the next heartbeat
five seconds later -- current state, not history. The files are the only history there is.

**Why JSON only.** `log_json=false` writes `%(asctime)s %(levelname)-7s %(processName)s
...`, and a regular expression over that free text is how a level or a category becomes
plausible and wrong -- a message that happens to contain the word ERROR is not an error.
So text logs are reported UNAVAILABLE, with the setting to change. Blank is better than
confident and wrong.

**Why only part of each file.** Rotation keeps up to six 10 MB files per process, and
there can be a file per worker group. Reading all of that to render fifty rows is the
legacy's whole-table render wearing a different hat (audit M-19). This reads the tail of
each stream, and the page states the horizon it actually looked at -- an unstated horizon
is the one everybody misreads.

**Nothing here writes.** No `mkdir`, no rotation, no truncation: a missing directory is
reported, not created. The legacy restarted the AI workers when somebody opened a tab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qorgan.redaction import redact
from qorgan.settings import get_settings, resolve

LOG_PAGE_SIZE = 50

# Per log stream, newest end first. Six streams at this size is a ~1.5 MB read to render
# one page -- bounded by the deployment's worker count, never by uptime.
TAIL_BYTES = 256 * 1024

# A diagnostics page is about what went wrong. INFO is the running commentary, and it is
# also where the recognition detail lives -- so this floor keeps the page useful AND keeps
# the routine per-child records off it. The one decision that must not be lost, the
# bullying worker's `notify` verdict, is deliberately a WARNING at the source.
SHOWN_LEVELS = frozenset({"WARNING", "ERROR", "CRITICAL"})

# The five things the school asked for (client §9), keyed to the module that emits them.
# Longest prefix wins, which is what lets `qorgan.worker.bullying` sit under alerts while
# its siblings do not. A logger nobody classified lands in `other` rather than being
# dropped: a record we cannot categorise is still a record that something went wrong.
CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qorgan.capture", "camera"),
    ("qorgan.rtsp", "camera"),
    ("qorgan.preview", "camera"),
    ("qorgan.worker.camera_loop", "camera"),
    ("qorgan.models", "model"),
    ("qorgan.detection", "model"),
    ("qorgan.faces", "model"),
    ("qorgan.db", "database"),
    ("qorgan.notify", "alerts"),
    ("qorgan.worker.bullying", "alerts"),
    ("qorgan.supervisor", "workers"),
    ("qorgan.worker.entrypoint", "workers"),
)

CATEGORY_LABELS: tuple[tuple[str, str], ...] = (
    ("camera", "Ошибки камер"),
    ("model", "Ошибки моделей"),
    ("database", "Ошибки базы"),
    ("alerts", "Пропуск тревог"),
    ("workers", "Воркеры"),
    ("other", "Прочее"),
)

# The structured extras this page will render, as an ALLOW-list. The line it draws is
# *what broke*, never *who was in front of the camera*: `camera`, `group` and `severity`
# are in; `person_id`, `track_id`, `pair` and `session_id` are out, and so is every key
# nobody has classified. Legacy printed children's full names to stdout (audit M-16); v2
# logs a `person_id` instead, which is better and is still not something a page about a
# dead RTSP connection needs to show. Deny by default, the same rule `AuthMiddleware`
# applies to routes and `capabilities_for` applies to roles.
SHOWN_FIELDS: frozenset[str] = frozenset(
    {
        "camera",
        "group",
        "groups",
        "cameras",
        "stream",
        "pid",
        "signal",
        "timeout_seconds",
        "model",
        "count",
        "event_id",
        "notify",
        "severity",
        "confidence",
        "skeleton_confirmed",
        "reasons",
        "attempt",
        "attempts",
        "retry_in_seconds",
        "permanent",
        "error",
        "reason",
    }
)

_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class LogEntry:
    ts: str
    level: str
    logger: str
    category: str
    message: str
    traceback: str | None
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Journal:
    """One page of the journal, plus what it took to build it.

    `unavailable` is a sentence, not a flag: a page that shows nothing without saying why
    is read as "nothing went wrong". `unreadable` and `files` are the same honesty at a
    smaller scale -- they are how a reader knows whether the emptiness is real.
    """

    entries: tuple[LogEntry, ...]
    page: int
    pages: int
    total: int
    unavailable: str | None
    files: tuple[str, ...]
    unreadable: int
    horizon_kb: int = TAIL_BYTES // 1024


def categorise(logger_name: str) -> str:
    best, category = 0, "other"
    for prefix, name in CATEGORY_PREFIXES:
        if logger_name.startswith(prefix) and len(prefix) > best:
            best, category = len(prefix), name
    return category


def recent(category: str | None, page: int, page_size: int = LOG_PAGE_SIZE) -> Journal:
    """One page of recent failures, newest first, across every process's log stream."""
    unavailable = _why_unavailable()
    if unavailable is not None:
        return Journal((), 1, 1, 0, unavailable, (), 0)

    entries, unreadable, files = _collect(resolve(get_settings().log_dir))

    # A category nobody defined matches no logger, and an empty journal is READ as "nothing
    # went wrong". So an unrecognised filter is ignored rather than applied -- the same
    # fallback `events._page` makes for a status it cannot parse.
    if category in {code for code, _ in CATEGORY_LABELS}:
        entries = [entry for entry in entries if entry.category == category]

    # Sorted across streams: a camera worker and the supervisor write separate files, and
    # the story of a crash is the two of them interleaved.
    entries.sort(key=_when, reverse=True)

    total = len(entries)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), pages)
    start = (page - 1) * page_size

    return Journal(
        tuple(entries[start : start + page_size]), page, pages, total, None, files, unreadable
    )


def _why_unavailable() -> str | None:
    settings = get_settings()
    if not settings.log_json:
        return (
            "Журнал недоступен: LOG_JSON=false — записи пишутся свободным текстом. "
            "Разбирать его регулярным выражением значит угадывать уровень и источник, "
            "а угаданная диагностика хуже её отсутствия. Включите LOG_JSON=true."
        )
    directory = resolve(settings.log_dir)
    if not directory.is_dir():
        # Reported, never created. Creating it here would be a side effect of a GET, and
        # it would also hide the actual fault: nothing is logging to this machine.
        return f"Журнал недоступен: каталог {directory} не существует."
    return None


def _collect(directory: Path) -> tuple[list[LogEntry], int, tuple[str, ...]]:
    entries: list[LogEntry] = []
    unreadable = 0
    names: list[str] = []

    for path in sorted(directory.glob("*.log")):
        names.append(path.name)
        for line in _stream_lines(path, TAIL_BYTES):
            record = _decode(line)
            if record is None:
                unreadable += 1
            elif str(record.get("level", "")) in SHOWN_LEVELS:
                entries.append(_entry(record))

    return entries, unreadable, tuple(names)


def _decode(line: str) -> dict[str, object] | None:
    if not line.strip():
        return None
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _entry(record: dict[str, object]) -> LogEntry:
    """One record, scrubbed and narrowed on the way out.

    `redact` runs here even though `RedactingFormatter` already ran at write time, and
    that is not belt-and-braces: the writing process masks the secrets ITS settings knew,
    so a per-camera `RTSP_PASSWORD__HALL_LEFT` set for a worker is not a literal the web
    process ever registered -- and a file on disk may predate any of it. The structural
    patterns (`user:password@host`, a Telegram token) do not need to have been told.
    """
    name = str(record.get("logger", ""))
    trace = record.get("exception")
    return LogEntry(
        ts=str(record.get("ts", "")),
        level=str(record.get("level", "")),
        logger=name,
        category=categorise(name),
        message=redact(str(record.get("message", ""))),
        traceback=redact(str(trace)) if trace else None,
        fields=tuple(
            (key, redact(str(value)))
            for key, value in record.items()
            if key in SHOWN_FIELDS and value is not None
        ),
    )


def _when(entry: LogEntry) -> datetime:
    """Sorted on the parsed instant, never on the string.

    `JsonFormatter` writes `%Y-%m-%dT%H:%M:%S%z`, and a lexicographic sort of that is
    correct only while every process shares one UTC offset. It does today, and a value
    that is true only because of a coincidence in the current deployment is exactly the
    kind that goes wrong somewhere else.
    """
    try:
        parsed = datetime.fromisoformat(entry.ts)
    except ValueError:
        return _EPOCH  # an unparseable timestamp sorts oldest; it is not dropped
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _stream_lines(path: Path, budget: int) -> list[str]:
    """The tail of one stream, spilling into its first rotation backup if it has to.

    Without the spill, opening this page a second after a 10 MB file rolled over would
    show an almost empty journal and no hint that anything was missing -- the emptiest
    possible answer at the moment something is most likely to be wrong.
    """
    lines = _tail(path, budget)
    spent = min(path.stat().st_size, budget)

    backup = path.parent / f"{path.name}.1"  # RotatingFileHandler's naming
    if spent < budget and backup.is_file():
        return _tail(backup, budget - spent) + lines
    return lines


def _tail(path: Path, budget: int) -> list[str]:
    size = path.stat().st_size
    start = max(0, size - budget)
    with path.open("rb") as handle:
        handle.seek(start)
        blob = handle.read()

    # errors="replace": a tail that starts mid-character must not raise. A log file is
    # read for the crash, and refusing to open it during one is the worst possible time.
    lines = blob.decode("utf-8", errors="replace").splitlines()
    # The first line of a mid-file seek is a fragment. Dropped rather than counted as
    # unreadable, because it is neither broken nor ours to report.
    return lines[1:] if start > 0 and lines else lines

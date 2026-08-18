"""The diagnostics page: what is wrong right now, and what went wrong recently.

The school asked for it in one line (client §9): "**Logs**: ошибки камер; ошибки моделей;
ошибки базы; причины пропуска тревог; перезапуски." Five categories that do not share a
source, and the tests here are about telling the truth about each of them -- who may read
it, how much of it is rendered at once, and what it says when it has nothing to show.

What must never come OFF this page -- a secret, a child, live markup -- is
`test_web_logs_leaks.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Notification, WorkerHeartbeat
from qorgan.db.types import utcnow
from qorgan.diagnostics.logfiles import LOG_PAGE_SIZE
from qorgan.enums import NotificationChannel, NotificationStatus, UserRole, WorkerState
from qorgan.logging_setup import setup_logging
from qorgan.settings import Settings
from tests.web_logs_support import (  # noqa: F401 - fixtures are used by name
    ClientFor,
    an_event,
    app,
    camera,
    client_for,
    json_logs,
    record,
    write,
)

# -- §14: who this page is for -----------------------------------------------


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.CANTEEN_STAFF])
def test_the_diagnostics_page_is_not_for_everyone(client_for: ClientFor, role: UserRole) -> None:
    """§14 divides on the job, not on seniority.

    "Оператор безопасности: просмотр тревог; подтверждение/отклонение событий; просмотр
    клипов" -- the server is not on that list, and this page is the server. The canteen
    worker is the harder case only in the legacy's model; here it is the same refusal.
    """
    assert client_for(role).get("/logs").status_code == 403


@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.DEVELOPER])
def test_an_admin_and_a_developer_read_the_diagnostics_page(
    client_for: ClientFor, role: UserRole
) -> None:
    """§14's суперадминистратор ("серверами") and `UserRole.DEVELOPER` ("debug views")."""
    assert client_for(role).get("/logs").status_code == 200


def test_the_nav_never_offers_a_link_the_role_cannot_follow(client_for: ClientFor) -> None:
    """One source of truth for the gate and for the link. An operator clicking "Логи" into
    a 403 reports the system as broken rather than the permission as intended."""
    assert '"/logs"' not in client_for(UserRole.OPERATOR).get("/").text
    assert '"/logs"' in client_for(UserRole.ADMIN).get("/").text


# -- pagination is mandatory (audit M-19) -------------------------------------


def test_the_log_view_is_paginated_rather_than_rendering_everything(
    client_for: ClientFor, json_logs: Path
) -> None:
    total = LOG_PAGE_SIZE * 2 + 10
    write(
        json_logs,
        "worker-hall",
        *(
            record(f"запись-{i:03d}", ts=f"2026-07-25T10:{i // 60:02d}:{i % 60:02d}+0500")
            for i in range(total)
        ),
    )
    client = client_for(UserRole.ADMIN)

    first = client.get("/logs").text
    second = client.get("/logs?page=2").text
    third = client.get("/logs?page=3").text

    assert first.count("запись-") == LOG_PAGE_SIZE, "the page rendered more than one page"
    assert second.count("запись-") == LOG_PAGE_SIZE
    assert third.count("запись-") == 10

    # Newest first: the most recent record is on page 1 and the oldest is on the last.
    assert f"запись-{total - 1:03d}" in first
    assert "запись-000" not in first
    assert "запись-000" in third


# -- the five categories the school asked for ---------------------------------


def test_worker_restarts_come_from_the_table_the_supervisor_writes(
    client_for: ClientFor, session: Session
) -> None:
    """`restart_count` is bumped by the supervisor and by nothing else. Counting restarts
    by grepping log text would disagree with it the moment a line rotated away."""
    session.add(
        WorkerHeartbeat(
            group_name="hall_bullying",
            state=WorkerState.CRASHED,
            pid=99,
            restart_count=12,
            last_seen_at=utcnow(),
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "hall_bullying" in page
    assert "12" in page


def test_an_undelivered_alert_says_how_many_attempts_and_what_failed(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """"Причины пропуска тревог", from the notification rows themselves."""
    event_id = an_event(camera)
    session.add(
        Notification(
            event_id=event_id,
            channel=NotificationChannel.TELEGRAM,
            status=NotificationStatus.FAILED,
            attempts=6,
            last_error="HTTP 401: Unauthorized",
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "HTTP 401: Unauthorized" in page
    assert f"#{event_id}" in page
    assert "Холл слева" in page


def test_a_delivered_alert_is_not_listed_as_a_problem(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """The panel answers "what did NOT arrive". A sent alert in it is noise that hides the
    one that matters."""
    session.add(
        Notification(
            event_id=an_event(camera),
            channel=NotificationChannel.TELEGRAM,
            status=NotificationStatus.SENT,
            attempts=1,
            sent_at=utcnow(),
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "Все поставленные в очередь тревоги доставлены" in page


def test_a_page_number_past_the_end_does_not_claim_every_alert_arrived(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """An OFFSET past the last row returns nothing, and "nothing" on this panel READS as
    "all delivered" -- the empty state says so in words. So a mistyped page number would
    answer the question "did my alerts go out?" with a confident yes."""
    session.add(
        Notification(
            event_id=an_event(camera),
            channel=NotificationChannel.TELEGRAM,
            status=NotificationStatus.FAILED,
            attempts=6,
            last_error="HTTP 401: Unauthorized",
        )
    )
    session.commit()

    page = client_for(UserRole.ADMIN).get("/logs?alerts_page=99").text

    assert "Все поставленные в очередь тревоги доставлены" not in page
    assert "HTTP 401: Unauthorized" in page, "the last page should have been shown instead"


def test_telegram_being_unconfigured_is_reported_as_the_reason_nothing_is_sent(
    client_for: ClientFor, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single commonest answer to "why was no alert sent", and it is a live fact rather
    than a guess: `NotificationWorker.start` says so in the log and then does nothing."""
    monkeypatch.setattr(settings, "telegram_chat_id", "")

    assert "Telegram не настроен" in client_for(UserRole.ADMIN).get("/logs").text


def test_the_category_filter_narrows_to_one_source(client_for: ClientFor, json_logs: Path) -> None:
    write(
        json_logs,
        "worker-hall",
        record("camera session failed", logger="qorgan.capture.stream"),
        record("database is locked", logger="qorgan.db.engine"),
    )

    page = client_for(UserRole.ADMIN).get("/logs?category=database").text

    assert "database is locked" in page
    assert "camera session failed" not in page


def test_a_category_nobody_defined_shows_everything_rather_than_nothing(
    client_for: ClientFor, json_logs: Path
) -> None:
    """Same lie one panel down: an unknown filter matches no logger, and an empty journal
    reads as "nothing went wrong". `events._page` already falls back this way for an
    unparseable status; a filter is not a place to be creative."""
    write(json_logs, "worker-hall", record("camera session failed"))

    assert "camera session failed" in client_for(UserRole.ADMIN).get("/logs?category=nope").text


# -- honest about what it cannot show -----------------------------------------


def test_text_logs_are_reported_unavailable_rather_than_parsed_by_guesswork(
    client_for: ClientFor, settings: Settings
) -> None:
    """`log_json=false` writes `%(asctime)s %(levelname)-7s ...`, and a regex over that is
    how a level or a category becomes plausible and wrong. The `settings` fixture ships
    this state, so this is the default the test suite runs under."""
    assert settings.log_json is False

    page = client_for(UserRole.ADMIN).get("/logs").text

    assert "LOG_JSON" in page, "the page did not say WHY the journal is unavailable"
    assert "недоступен" in page


def test_the_page_reads_the_format_logging_setup_actually_writes(
    client_for: ClientFor, json_logs: Path
) -> None:
    """The reader is bound to the writer, not to a fixture's idea of the writer.

    Hand-rolled JSON in the other tests would keep passing after `JsonFormatter` changed a
    key -- true in the test layer and wrong in the running system, which is this project's
    signature disease.
    """
    root = logging.getLogger()
    keep = list(root.handlers)
    try:
        setup_logging("worker-real")
        logging.getLogger("qorgan.capture.stream").error("the real writer wrote this")
        for handler in root.handlers:
            handler.flush()

        page = client_for(UserRole.ADMIN).get("/logs").text
    finally:
        # Restored BEFORE the new handlers are closed, not after. The notifier thread is
        # still running and still logging; between a `removeHandler` and a `close` it
        # would find a shut file and print `I/O operation on closed file` to stderr.
        mine = list(root.handlers)
        root.handlers[:] = keep
        for handler in mine:
            handler.close()

    assert (json_logs / "worker-real.log").exists()
    assert "the real writer wrote this" in page


# -- zero side effects on page load (the legacy restarted workers) ------------


def test_opening_the_page_changes_nothing_on_disk_or_in_the_database(
    client_for: ClientFor, json_logs: Path, session: Session
) -> None:
    """The legacy restarted the AI workers on tab open, with a 5 s join inside the HTTP
    handler. A GET here reads: no rotation, no cleanup, no heartbeat, no queue poke."""
    written = write(json_logs, "worker-hall", record("something broke"))
    before_names = sorted(p.name for p in json_logs.iterdir())
    before_bytes = written.read_bytes()
    before_rows = session.scalar(select(func.count(WorkerHeartbeat.id)))

    assert client_for(UserRole.ADMIN).get("/logs").status_code == 200

    assert sorted(p.name for p in json_logs.iterdir()) == before_names, "a file appeared or rotated"
    assert written.read_bytes() == before_bytes, "the page rewrote the log it was reading"
    assert session.scalar(select(func.count(WorkerHeartbeat.id))) == before_rows


def test_the_page_still_carries_this_session_s_csrf_token(client_for: ClientFor) -> None:
    """Every rendered page carries it, so the logout form on this one works and so the next
    form added here cannot be the one whose author routes around the check."""
    page = client_for(UserRole.ADMIN).get("/logs").text

    assert 'name="csrf-token"' in page
    assert 'name="csrf_token"' in page

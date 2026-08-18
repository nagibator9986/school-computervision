"""The backup page: `qorgan backup` for the person who has no terminal.

§11 item 14 was built as a CLI command and a scheduled task. Neither is visible to the
people who run the school, so nobody there can answer "did last night's backup run?" —
and a backup nobody checks is exactly the backup that has silently not happened for three
months. This page answers that question and lets a human take one now.

What is tested here, and why each of it matters more than the button working:

  * **A GET never takes a backup.** The legacy restarted the AI workers from a page load
    (audit H-18), so what the system did depended on which tab somebody had open. Writing
    a full copy of the database from a page render is the same defect with a disk-fill
    attached: a browser prefetch, a monitoring probe or a refresh would do it.
  * **The handler does not wait for the job.** H-18's other half was a five-second
    `thread.join()` *inside* the request. `VACUUM INTO` on the school's database is
    seconds today and minutes in a year.
  * **Reading the list and pressing the button are two grants.** The button writes a file
    containing every child's data; reading the list does not.
  * **The file is never served.** A backup is the entire database of every child in the
    school. See `test_no_route_hands_out_a_backup_file`.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, User
from qorgan.enums import CameraRole, CameraType, UserRole
from qorgan.maintenance.backup import BackupError, BackupReport, backup_directory
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    application = create_app()
    yield application
    # A backup thread that outlived its test would call `get_engine()` after this test's
    # settings override is gone, poisoning the module-global engine for whatever runs
    # next -- the stray-thread hazard conftest.py documents at length, which surfaces as
    # `no such table: events` in a test that has nothing to do with the cause. The app's
    # lifespan joins the thread on shutdown; this asserts that it actually did, so a leak
    # fails HERE, loudly, instead of somewhere else, quietly.
    assert application.state.backups.wait(timeout=30), "a backup thread outlived its test"


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about. **One open at a time.**

    Every client here is handed the SAME application, because `app.state.backups` is the
    object half these tests assert on. `TestClient.__enter__` runs that application's
    LIFESPAN -- so two clients open at once would run it twice, and the second run
    replaces `app.state.previews` and `app.state.notifier` with a fresh pair while the
    first pair's threads keep running with nobody holding a reference to stop them.

    That is not a tidiness point. A leaked notifier calls `session_scope()` every two
    seconds forever, including after this test's settings override is torn down, and
    `qorgan.db.engine` then caches a module-global engine pointed at the default database
    -- which has no schema. The next test to touch the database fails with `no such table:
    events` and looks like a bug in whatever it was testing. conftest.py documents that
    hunt at length; this fixture leaked exactly that way for two tests before the threads
    were counted. So opening a client closes the previous one.
    """
    opened: list[ExitStack] = []

    def make(role: UserRole) -> TestClient:
        while opened:
            opened.pop().close()

        username = f"user_{role.value}"
        session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
        session.commit()

        stack = ExitStack()
        opened.append(stack)
        client = stack.enter_context(TestClient(app, follow_redirects=False))
        posted = client.post(
            "/login",
            data=with_token(client, {"username": username, "password": PASSWORD}),
        )
        assert posted.status_code == 303, "login failed"
        return client

    yield make
    while opened:
        opened.pop().close()


@pytest.fixture
def camera(session: Session) -> Camera:
    row = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.commit()
    return row


def _folder() -> Path:
    folder = backup_directory()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _fake_backup(name: str = "qorgan-2026-01-02-030405.sqlite3", size: int = 2048) -> Path:
    """A file in the backups folder. Never a real backup: the listing reads the directory
    and the file's own size, and must not need to open anything to draw a row."""
    path = _folder() / name
    path.write_bytes(b"\x00" * size)
    return path


def _rows_in(backup: Path, table: str) -> int:
    """Open the BACKUP itself — not the original — and read it."""
    with sqlite3.connect(backup) as connection:
        return connection.execute(f"select count(*) from {table}").fetchone()[0]  # noqa: S608


def _taken(folder: Path | None = None) -> list[Path]:
    folder = folder or backup_directory()
    return sorted(folder.glob("*.sqlite3")) if folder.exists() else []


# -- reading the list --------------------------------------------------------


def test_the_page_lists_the_backups_that_exist(client_for: ClientFor) -> None:
    """The question the school actually has: is there a copy, and how old is it?"""
    _fake_backup("qorgan-2026-01-02-030405.sqlite3", size=3_000_000)
    _fake_backup("qorgan-2026-01-03-030405.sqlite3", size=4_000_000)

    page = client_for(UserRole.ADMIN).get("/backups")

    assert page.status_code == 200
    assert "qorgan-2026-01-02-030405.sqlite3" in page.text
    assert "qorgan-2026-01-03-030405.sqlite3" in page.text
    # Same divisor as `BackupReport.summary`, which is what the CLI prints for the same
    # file: two surfaces reporting different sizes for one backup is how trust in both goes.
    assert "3.0" in page.text and "4.0" in page.text


def test_a_fresh_install_with_no_backups_yet_still_renders(client_for: ClientFor) -> None:
    """`backups/` does not exist until the first backup is taken.

    "No backup has ever been taken" is the single most important thing this page can say,
    and it must SAY it: a blank table reads as "loading", or as a broken query, or as a
    page that has not finished being built.
    """
    page = client_for(UserRole.ADMIN).get("/backups")

    assert page.status_code == 200
    assert "ни разу не запускалось" in page.text


def test_the_page_says_the_backup_sits_on_the_same_disk(client_for: ClientFor) -> None:
    """The page must not imply that "backed up" means "safe". `backups/` is the same disk
    as the database, so a dead disk takes both. Getting the copy off the machine is a
    decision for the school, and a page that does not say so has made it for them."""
    page = client_for(UserRole.ADMIN).get("/backups")

    assert "тот же диск" in page.text.lower()


def test_the_page_says_a_backup_is_childrens_data(client_for: ClientFor) -> None:
    """Every face embedding, every meal, every incident summary, in one file that is easy
    to email. It belongs where the roster photographs are allowed to be and nowhere else."""
    page = client_for(UserRole.ADMIN).get("/backups")

    assert "данные детей" in page.text.lower()


def test_only_the_backups_folder_is_listed(client_for: ClientFor, settings: Settings) -> None:
    """The live database sits one directory up. Listing it here would offer the operator a
    file that is not a backup and cannot be treated as one."""
    _fake_backup()

    page = client_for(UserRole.ADMIN).get("/backups")

    assert "test.sqlite3" not in page.text, "the live database was listed as a backup"


def test_a_backup_file_name_is_escaped(client_for: ClientFor) -> None:
    """A file name is a string from the filesystem, not from us — anyone with the folder
    open can create one. The legacy built its DOM from server strings and a pupil named
    `<img src=x onerror=...>` gave stored XSS in the operator's browser (audit H-05)."""
    _fake_backup("qorgan-a&b.sqlite3")

    page = client_for(UserRole.ADMIN).get("/backups")

    assert "qorgan-a&amp;b.sqlite3" in page.text
    assert "qorgan-a&b.sqlite3" not in page.text


def test_the_time_shown_is_the_schools_time_not_the_servers(
    client_for: ClientFor, settings: Settings
) -> None:
    """A file's mtime is an INSTANT with no timezone in it. `datetime.fromtimestamp(ts)`
    quietly stamps the SERVER's zone onto it, and `default_destination` already dates the
    file name in the SCHOOL's — so the same backup would carry two different times, one in
    its name and one in this column. That is this project's signature disease: a value
    true in one layer and silently wrong in the next.
    """
    settings.school_timezone = "Pacific/Kiritimati"  # UTC+14, nowhere near any server
    made = _fake_backup()
    expected = datetime.fromtimestamp(made.stat().st_mtime, tz=UTC).astimezone(
        ZoneInfo("Pacific/Kiritimati")
    )

    page = client_for(UserRole.ADMIN).get("/backups")

    assert expected.strftime("%Y-%m-%d %H:%M") in page.text


def test_an_install_this_cannot_back_up_gets_a_sentence_not_a_server_error(
    client_for: ClientFor, settings: Settings
) -> None:
    """`VACUUM INTO` is SQLite's, and the spec keeps the door open to PostgreSQL (§4.3).

    Two things must both hold. The page renders — a 500 tells the school the backups are
    broken when the truth is that this installation backs up by other means. And the empty
    table must NOT then claim "no backup has ever been taken": we did not look, and an
    unmeasured zero reported as a measured one is the fault this project keeps finding.
    """
    client = client_for(UserRole.ADMIN)
    settings.database_url = "postgresql+psycopg://qorgan@db/qorgan"

    page = client.get("/backups")

    assert page.status_code == 200
    assert "SQLite" in page.text, "the page did not say why it cannot list anything"
    assert "ни разу не запускалось" not in page.text


# -- taking one --------------------------------------------------------------


def test_opening_the_page_never_takes_a_backup(app, client_for: ClientFor) -> None:
    """The legacy restarted the AI workers from a page load (audit H-18), so what the
    system did depended on which tab was open. A page render that writes a full copy of
    the database is that defect with a disk-fill attached: a prefetch, a monitoring probe
    or an F5 held down would do it."""
    client = client_for(UserRole.ADMIN)

    client.get("/backups")
    client.get("/backups")

    state = app.state.backups.state()
    assert not state.running and state.last is None, "a GET started a backup"
    assert _taken() == [], "a GET wrote a backup to disk"


def test_pressing_the_button_takes_a_real_backup(
    app, client_for: ClientFor, camera: Camera
) -> None:
    """The whole point. And it is checked by opening the copy, not by trusting the page:
    `qorgan backup` reads its own output back before calling it a backup, and a second
    surface that reports success without that check would undo the guarantee."""
    client = client_for(UserRole.ADMIN)

    response = client.post("/backups", data=with_token(client))

    assert response.status_code == 303
    assert app.state.backups.wait(timeout=30), "the backup never finished"
    taken = _taken()
    assert len(taken) == 1, f"expected one backup, found {taken}"
    assert _rows_in(taken[0], "cameras") == 1
    assert taken[0].name in client.get("/backups").text


def test_the_handler_does_not_wait_for_the_backup(
    app, client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The H-18 test.** The legacy restarted the workers inside a request and waited on
    a five-second `thread.join()`. `VACUUM INTO` on this database is seconds today and
    minutes in a year, and a web worker blocked on it serves nobody — including the
    operator watching a corridor.

    The job is held open by a gate: the request must have finished and answered while the
    backup is provably still running.
    """
    gate, started = threading.Event(), threading.Event()

    def slow(destination: Path | None = None) -> BackupReport:
        started.set()
        gate.wait(timeout=10)
        return BackupReport(destination=Path("held-open.sqlite3"), bytes_written=1)

    monkeypatch.setattr("qorgan.web.backup_runner.backup_database", slow)
    client = client_for(UserRole.ADMIN)

    response = client.post("/backups", data=with_token(client))

    assert response.status_code == 303
    assert started.wait(timeout=10), "the backup never started"
    assert app.state.backups.state().running, "the handler waited for the backup to finish"
    page = client.get("/backups").text
    assert "выполняется" in page.lower()
    # With no time, "копирование выполняется" cannot tell a copy that started ten seconds
    # ago from one that has been stuck since Tuesday.
    assert re.search(r"начато в \d\d:\d\d", page), "the running banner does not say since when"

    gate.set()
    assert app.state.backups.wait(timeout=10)


def test_two_presses_do_not_write_two_copies(
    app, client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One backup is a full second copy of the database on the same disk. A double-click,
    an impatient operator or a browser retry must not multiply it — filling the disk stops
    the recordings, which is a worse outcome than a missing backup."""
    gate, started, runs = threading.Event(), threading.Event(), []

    def slow(destination: Path | None = None) -> BackupReport:
        runs.append(1)
        started.set()
        gate.wait(timeout=10)
        return BackupReport(destination=Path("held-open.sqlite3"), bytes_written=1)

    monkeypatch.setattr("qorgan.web.backup_runner.backup_database", slow)
    client = client_for(UserRole.ADMIN)

    client.post("/backups", data=with_token(client))
    assert started.wait(timeout=10)
    client.post("/backups", data=with_token(client))

    gate.set()
    assert app.state.backups.wait(timeout=10)
    assert len(runs) == 1, "a second backup started while the first was still running"


def test_a_failure_is_shown_on_the_page_and_escaped(
    app, client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backup that failed must say so where the person who pressed the button will look.
    The message quotes a PATH, and this school's paths are Cyrillic and hand-typed — so it
    is untrusted text and it is escaped like any other."""

    def broken(destination: Path | None = None) -> BackupReport:
        raise BackupError("<script>alert(1)</script> диск заполнен")

    monkeypatch.setattr("qorgan.web.backup_runner.backup_database", broken)
    client = client_for(UserRole.ADMIN)

    client.post("/backups", data=with_token(client))
    assert app.state.backups.wait(timeout=10)
    page = client.get("/backups")

    assert "диск заполнен" in page.text, "the failure was not reported"
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text


# -- who may do which ---------------------------------------------------------


def test_the_page_needs_a_session(settings: Settings, session: Session) -> None:
    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get("/backups").status_code == 303


def test_an_operator_and_a_canteen_worker_are_refused_the_page(client_for: ClientFor) -> None:
    """Watching a corridor and serving lunch are not maintenance. The list names every
    copy of the school's whole database that exists on this machine."""
    assert client_for(UserRole.OPERATOR).get("/backups").status_code == 403
    assert client_for(UserRole.CANTEEN_STAFF).get("/backups").status_code == 403


def test_an_operator_cannot_take_a_backup(client_for: ClientFor) -> None:
    client = client_for(UserRole.OPERATOR)

    assert client.post("/backups", data=with_token(client)).status_code == 403
    assert _taken() == []


def test_taking_a_backup_is_a_separate_grant_from_reading_the_list(
    app, client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Why two capabilities and not one.** Reading the list answers "has a backup
    happened?" — a question a headteacher is entitled to ask and which changes nothing.
    Pressing the button writes a new file containing every child in the school to a disk
    that also holds the recordings. Those are different rights, and a model that cannot
    say "may check, may not take" cannot grant the first without the second.

    No role holds exactly this pair today; the grant is what is under test, not the row.
    """
    monkeypatch.setitem(
        ROLE_CAPABILITIES,
        UserRole.OPERATOR,
        frozenset({Capability.VIEW_CAMERAS, Capability.VIEW_BACKUPS}),
    )
    client = client_for(UserRole.OPERATOR)

    assert client.get("/backups").status_code == 200, "the list was shut to a grant that reads it"
    assert client.post("/backups", data=with_token(client)).status_code == 403
    assert not app.state.backups.state().running
    assert _taken() == []


def test_a_post_without_a_csrf_token_is_refused(app, client_for: ClientFor) -> None:
    """Without this, any page on the internet could make the school's server write
    backups — one per visit — until the disk holding the recordings is full."""
    client = client_for(UserRole.ADMIN)

    response = client.post("/backups", data={})

    assert response.status_code == 403
    assert not app.state.backups.state().running
    assert _taken() == []


def test_no_route_hands_out_a_backup_file(client_for: ClientFor) -> None:
    """**Download is deliberately not offered.** A backup is the whole database of every
    child: faces, meals, incidents. Serving it turns one authenticated session — or one
    cookie left open on a staffroom machine — into an offline copy of the school. Getting
    it off the machine is a decision with a person attached, made at the filesystem."""
    made = _fake_backup()
    made.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)
    client = client_for(UserRole.ADMIN)

    for url in (f"/backups/{made.name}", f"/backups/download?name={made.name}"):
        response = client.get(url)
        assert response.status_code != 200, f"{url} served something"
        assert "SQLite format 3" not in response.text

    assert 'href="/backups/' not in client.get("/backups").text, "the page linked the file"


def test_the_nav_link_is_only_drawn_for_someone_who_can_open_it(client_for: ClientFor) -> None:
    """One source of truth for the gate and for the menu. A link into a 403 is reported by
    the school as a broken system, not as the permission it is."""
    assert 'href="/backups"' in client_for(UserRole.ADMIN).get("/").text
    assert 'href="/backups"' not in client_for(UserRole.OPERATOR).get("/").text


def test_the_capability_table_states_both_grants() -> None:
    """Below HTTP: the contract itself. Deny by default — a role that is not written down
    as maintenance gets neither half."""
    for role in (UserRole.ADMIN, UserRole.DEVELOPER):
        assert Capability.VIEW_BACKUPS in capabilities_for(role)
        assert Capability.CREATE_BACKUP in capabilities_for(role)

    for role in (UserRole.OPERATOR, UserRole.CANTEEN_STAFF):
        assert Capability.VIEW_BACKUPS not in capabilities_for(role)
        assert Capability.CREATE_BACKUP not in capabilities_for(role)

"""The CSV the school downloads: what is actually IN the file, parsed as CSV.

Split out of `test_web_canteen.py` when that file reached the 500-line cap (R1). The seam
is by subject, not by size, and the subject is the export: the page renders HTML a person
reads, the export writes a file a SPREADSHEET reads, and a spreadsheet reads columns. So
these tests parse the bytes with the `csv` module rather than matching substrings -- a
substring cannot tell a value in its column from the same value spilled into the next one,
which is exactly how the unquoted-comma defect survived every test in this suite.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, CanteenSession, Person, User
from qorgan.enums import (
    CameraRole,
    CameraType,
    PersonType,
    SessionOutcome,
    SessionState,
    UserRole,
)
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"


@pytest.fixture
def client(settings: Settings, session: Session) -> Iterator[TestClient]:
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as test_client:
        test_client.post(
            "/login",
            data=with_token(test_client, {"username": "op", "password": PASSWORD}),
        )
        yield test_client


@pytest.fixture
def camera(session: Session) -> Camera:
    row = Camera(
        name="canteen_entry",
        display_name="Вход в столовую",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.commit()
    return row


def _pupil(session: Session, name: str) -> Person:
    person = Person(
        external_id=f"gen-{name}",
        full_name=name,
        person_type=PersonType.STUDENT,
        class_name="5А",
    )
    session.add(person)
    session.commit()
    return person


def _meal(session: Session, camera: Camera, person: Person, outcome: SessionOutcome) -> None:
    from qorgan.settings import get_settings

    local_noon = datetime.now(tz=get_settings().tz).replace(hour=12, minute=0, second=0)
    session.add(
        CanteenSession(
            person_id=person.id,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=outcome,
            opened_at=local_noon.astimezone(UTC),
            dwell_seconds=300.0,
        )
    )
    session.commit()


def _nameless_pupil(session: Session, external_id: str, class_name: str) -> Person:
    """The pupil the school actually has TODAY: an id and a class, and no name.

    `display_name` falls back to `Ученик 333, 5-А` -- WITH A COMMA -- and its own docstring
    says that is the present state, not an edge case: there is no ID -> name roster yet, so
    this is EVERY pupil. Every other seed in this suite passes a `full_name`, which is
    exactly why an export that never quoted its commas passed every test for so long.
    """
    person = Person(
        external_id=external_id,
        full_name=None,
        person_type=PersonType.STUDENT,
        class_name=class_name,
    )
    session.add(person)
    session.commit()
    return person


def _columns(body: str) -> list[list[str]]:
    """The export as a spreadsheet reads it: parsed rows, not a haystack of characters.

    The BOM is stripped here and asserted separately -- a reader that silently ate it would
    hide the mojibake defect the BOM exists to prevent.
    """
    return list(csv.reader(io.StringIO(body.lstrip("﻿"))))


def _unknown_entry(session: Session, camera: Camera) -> None:
    """An entry the recogniser could not attribute to anybody: person_id IS NULL."""
    from qorgan.settings import get_settings

    local_noon = datetime.now(tz=get_settings().tz).replace(hour=12, minute=0, second=0)
    session.add(
        CanteenSession(
            person_id=None,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=SessionOutcome.ATE,
            opened_at=local_noon.astimezone(UTC),
            dwell_seconds=300.0,
        )
    )
    session.commit()


def test_the_canteen_exports_a_csv(client: TestClient, session: Session, camera: Camera) -> None:
    """The school asks for a spreadsheet, and the school is right to."""
    ate = _pupil(session, "Петрова Мария")
    _meal(session, camera, ate, SessionOutcome.ATE)

    response = client.get("/canteen/export.csv")

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    body = response.content.decode("utf-8")
    assert body.startswith("﻿"), "no BOM: Excel will render Cyrillic as mojibake"
    assert "Петрова Мария" in body


def test_the_csv_carries_the_caveat_the_school_reads(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The CSV is the surface the school ACTUALLY reads (see the route's own docstring).

    A row saying `never_came` under a real child's name, with no caveat in the file, is the
    exact claim we cannot support -- and it is the one that reaches the parent.
    """
    _pupil(session, "Иванов Иван")
    _unknown_entry(session, camera)

    body = client.get("/canteen/export.csv").content.decode("utf-8")

    assert "нет записи о питании" in body
    assert "не означает, что ученик не ел" in body
    assert "до 1 из перечисленных" in body, "the CSV caveat must say UP TO N"


def test_the_csv_caveat_omits_the_up_to_n_line_when_nothing_is_unattributed(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Part 1 always; part 2 only when there is an N to talk about."""
    _pupil(session, "Иванов Иван")

    body = client.get("/canteen/export.csv").content.decode("utf-8")

    assert "нет записи о питании" in body
    assert "могли поесть" not in body


def test_the_csv_outcome_column_keeps_the_never_came_token(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """`never_came` is a DATA CONTRACT with the school, not a sentence we are asserting.

    The token would be a false claim on its own -- an unrecognised child who ate lands in
    this list -- but it does not travel on its own: the caveat is in the file, and with the
    caveat present `never_came` reads as an opaque code. Renaming it would push the same
    disease across the system boundary instead: a filter or macro in the school's own
    spreadsheet keyed on `never_came` would silently match nothing, and break where we
    cannot see it. Truth is the caveat's job; the token's job is to stay still.
    """
    _nameless_pupil(session, "student_333", "5-А")

    rows = _columns(client.get("/canteen/export.csv").content.decode("utf-8"))

    pupil = [row for row in rows if row and row[1].startswith("Ученик 333")]
    assert pupil, "the pupil left the export entirely"
    assert pupil[0][2] == "never_came", "the token is not in the outcome column"
    assert "no_meal_record" not in "".join("".join(row) for row in rows)


def test_every_column_lands_in_its_column_for_a_pupil_whose_name_holds_a_comma(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Today's normal pupil has no name, so `display` is `Ученик 333, 5-А` -- with a comma.

    An export that interpolates that into an f-string emits a five-column row, and every
    column after `name` shifts by one: the outcome column reads ` 5-А`, not `never_came`.
    The token is then not merely wrong, it is not in the outcome column at all -- so the
    school's filter, keyed on that column, matches nothing, and the data contract this
    branch chose to KEEP is void. The CLI's `_write_csv` uses `csv.writer` and gets this
    right, so the two exports of one day disagree, which is this codebase's signature
    disease.

    Asserted on PARSED columns for every row kind, because a substring cannot tell a value
    in its column from the same value spilled into the next one.
    """
    ate = _nameless_pupil(session, "student_111", "5-А")
    _meal(session, camera, ate, SessionOutcome.ATE)
    did_not_eat = _nameless_pupil(session, "student_222", "5-А")
    _meal(session, camera, did_not_eat, SessionOutcome.NOT_ATE)
    _nameless_pupil(session, "student_333", "5-А")  # no meal record at all

    body = client.get("/canteen/export.csv").content.decode("utf-8")

    assert body.startswith("﻿"), "no BOM: Excel will render Cyrillic as mojibake"
    rows = _columns(body)

    assert rows[0] == ["class", "name", "outcome", "dwell_seconds"]
    assert ["5-А", "Ученик 111, 5-А", "ate", "300"] in rows
    assert ["5-А", "Ученик 222, 5-А", "not_ate", "300"] in rows
    assert ["5-А", "Ученик 333, 5-А", "never_came", ""] in rows


def test_the_caveat_rows_keep_their_sentence_in_one_column(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The caveat is what makes `never_came` honest, so it has to survive as a sentence.

    The route always quoted these -- its comment reasons that unquoted commas would spill
    into columns. That care was applied one line BELOW the data rows, where the same hazard
    was live and unguarded. This pins the half that was already right.
    """
    _nameless_pupil(session, "student_333", "5-А")
    _unknown_entry(session, camera)

    rows = _columns(client.get("/canteen/export.csv").content.decode("utf-8"))

    caveat = [row for row in rows if row and "не означает, что ученик не ел" in row[0]]
    assert caveat, "the caveat left the file, and never_came now travels alone"
    assert all(len(row) == 4 for row in caveat), "a caveat sentence spilled across columns"
    assert [] in rows, "the blank separator row before the caveat is gone"


def test_both_exports_write_the_same_outcome_token(
    client: TestClient, session: Session, camera: Camera, tmp_path: Path
) -> None:
    """The two spreadsheets of one day must never disagree on the token OR its column.

    The web export and the CLI's `_write_csv` are two renderings of the same report, and
    the school may receive either. One saying `never_came` and the other `no_meal_record`
    for the same pupil on the same day is precisely how a value goes true in one layer and
    wrong in the next -- and the token is a contract, so a drift here breaks a filter in
    their spreadsheet, not ours. This binds the CLI, which the web test above cannot see.

    Seeded with a nameless pupil -- today's real pupil, whose `display` holds a comma --
    and asserted by column. This test used to say `",never_came," in body` over the whole
    file, which stayed true while the token sat in the WRONG COLUMN in one of the two
    exports: a test that cannot see the defect it names is what let the two disagree.
    """
    from qorgan.canteen.reports import day_report
    from qorgan.faces.cli import _today, _write_csv

    _nameless_pupil(session, "student_333", "5-А")

    web = _columns(client.get("/canteen/export.csv").content.decode("utf-8"))
    path = tmp_path / "day.csv"
    _write_csv(path, day_report(_today()))
    written = _columns(path.read_text(encoding="utf-8-sig"))

    expected = ["5-А", "Ученик 333, 5-А", "never_came", ""]
    assert web[0] == written[0] == ["class", "name", "outcome", "dwell_seconds"]
    assert expected in web, "the web export"
    assert expected in written, "the CLI export"

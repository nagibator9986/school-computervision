"""The canteen journal: what the /canteen page and its CSV export tell the school.

Split out of `test_web_pages.py` when that file reached the 500-line cap (R1). The seam is
by subject, not by size: the events log and /media are a different surface from the canteen
journal, and the canteen journal is where the numbers a headteacher acts on are rendered.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, CanteenSession, Person, User
from qorgan.db.types import utcnow
from qorgan.enums import (
    CameraRole,
    CameraType,
    CloseReason,
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


def test_the_canteen_page_names_the_pupils_with_no_meal_record(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The question the legacy could not answer at all -- named honestly.

    Not "who did not eat": the page cannot know that, and neither can this test. It knows
    who has no meal record, which is what it says.
    """
    ate = _pupil(session, "Петрова Мария")
    _pupil(session, "Иванов Иван")  # never came
    _meal(session, camera, ate, SessionOutcome.ATE)

    response = client.get("/canteen")

    assert response.status_code == 200
    assert "Иванов Иван" in response.text
    assert "Нет записи о питании (1)" in response.text


def test_the_canteen_page_needs_a_session(settings: Settings, session: Session) -> None:
    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get("/canteen").status_code == 303


def test_an_empty_canteen_day_renders(client: TestClient) -> None:
    response = client.get("/canteen?day=2020-01-01")

    assert response.status_code == 200
    assert "У каждого ученика есть запись о питании" in response.text


def test_a_nonsense_date_falls_back_to_today(client: TestClient) -> None:
    assert client.get("/canteen?day=not-a-date").status_code == 200


def test_unattributed_sessions_are_surfaced_not_hidden(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """1816 of the legacy's 1820 records were unattributed. If recognition is failing, the
    page must SAY so rather than quietly reporting that nobody ate."""
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

    response = client.get("/canteen")

    assert "не удалось привязать к ученику" in response.text


def _timed_out(session: Session, camera: Camera, person: Person) -> None:
    """A session the janitor force-closed: the pupil entered, but no exit was recognised.

    This is what `DayReport.forced_unknown` counts — the measured price of a strict exit
    `min_score`. Note `close_reason`, not `person_id`: the pupil IS known here.
    """
    from qorgan.settings import get_settings

    local_noon = datetime.now(tz=get_settings().tz).replace(hour=12, minute=0, second=0)
    session.add(
        CanteenSession(
            person_id=person.id,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=SessionOutcome.UNKNOWN,
            close_reason=CloseReason.TIMEOUT,
            opened_at=local_noon.astimezone(UTC),
            dwell_seconds=None,
        )
    )
    session.commit()


def _tiles(html: str) -> dict[str, str]:
    """The rendered tiles, as {label: value}.

    Parsed out of the real HTML rather than matched as a bare substring, so that a tile
    which is ABSENT and a tile which reads `0` cannot look the same to the test — which is
    the whole point of the thing being tested.
    """
    pattern = (
        r'<div class="tile[^"]*">\s*<span class="value">([^<]*)</span>\s*'
        r'<span class="label">([^<]*)</span>'
    )
    return {label.strip(): value.strip() for value, label in re.findall(pattern, html)}


def test_the_recognition_tiles_render_a_measured_zero_not_a_blank(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """A tile that disappears at zero cannot say "measured zero today".

    It renders identically to "no data", to "the query is broken", and — as this very
    template proved — to "the value was silently dropped between the report and the page".
    `forced_unknown` reached the template and was thrown away there, and nobody saw it
    precisely because a dropped value and a zero both show as nothing. A count whose job
    is to distinguish measured-zero from unmeasured must SHOW its zero.
    """
    _pupil(session, "Присутствующий Ученик")

    response = client.get("/canteen?day=2020-01-01")

    tiles = _tiles(response.text)
    assert tiles.get("вход не распознан") == "0", "the entry-recognition tile vanished at zero"
    assert tiles.get("выход не распознан") == "0", "the exit-recognition tile vanished at zero"


def test_forced_unknown_sessions_are_surfaced_on_the_canteen_page(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """`forced_unknown` is the instrument that measures the cost of a strict exit
    threshold: a child entered and was never recognised on the way out. It is printed by
    the CLI, but an operator lives on this page — a cost nobody looks at is not measured.

    Seeded at three, not zero: a page that renders nothing would satisfy "0 appears".
    """
    for name in ("Первый Пупил", "Второй Пупил", "Третий Пупил"):
        _timed_out(session, camera, _pupil(session, name))

    response = client.get("/canteen")

    assert response.status_code == 200
    assert "3 сессий закрыто по таймауту" in response.text, "the count never reached the page"
    # The warning above and the tile are two renderings of one number. Pin the TILE too:
    # asserting only the warning leaves the tile free to display anything at all.
    assert _tiles(response.text)["выход не распознан"] == "3", "the tile contradicts the warning"


def test_forced_unknown_is_not_conflated_with_unattributed_entries(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Two different failures on two different cameras: `unknown_sessions` is an entry
    never attributed to anybody, `forced_unknown` is a known pupil never recognised at the
    exit. They are never summed, and the page must not let a human sum them.
    """
    from qorgan.settings import get_settings

    local_noon = datetime.now(tz=get_settings().tz).replace(hour=12, minute=0, second=0)
    session.add(
        CanteenSession(
            person_id=None,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=SessionOutcome.ATE,
            close_reason=CloseReason.EXIT_CAMERA,
            opened_at=local_noon.astimezone(UTC),
            dwell_seconds=300.0,
        )
    )
    session.commit()
    _timed_out(session, camera, _pupil(session, "Четвёртый Пупил"))
    _timed_out(session, camera, _pupil(session, "Пятый Пупил"))

    response = client.get("/canteen")

    assert "1 сессий не удалось привязать к ученику" in response.text
    assert "2 сессий закрыто по таймауту" in response.text
    # Seeded 1 and 2, deliberately unequal: each tile must show its OWN count. A tile wired
    # to the other number — one copy-paste away, the two lines are adjacent and identical
    # in shape — passes any assertion that only checks "a number is present".
    tiles = _tiles(response.text)
    assert tiles["вход не распознан"] == "1", "the entry tile shows the wrong count"
    assert tiles["выход не распознан"] == "2", "the exit tile shows the wrong count"


def test_yesterday_can_be_asked_for(client: TestClient) -> None:
    yesterday = (utcnow() - timedelta(days=1)).date().isoformat()
    assert client.get(f"/canteen?day={yesterday}").status_code == 200


# -- "never came" does not mean "did not eat" --------------------------------


def _unknown_entry(session: Session, camera: Camera) -> None:
    """An entry the recogniser could not attribute to anybody: person_id IS NULL.

    Such a session yields no `Meal` for any named pupil (`_meals_between` joins Person),
    so whoever this was is absent from `seen` -- and if they are on the roster they land in
    `never_came`. That is why every surface showing `never_came` needs the caveat.
    """
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


def test_the_page_says_never_came_means_no_meal_record_even_with_nothing_unknown(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Part 1 of the caveat, which is true on every day regardless of any count.

    `never_came` is "we have no meal record for this pupil today". It is NOT "this pupil
    did not eat" -- and the page must not let a headteacher read the second from the first.
    Asserted on a day with ZERO unknown sessions on purpose: this half of the caveat does
    not depend on N, because a child the detector never saw at all produces no session of
    any kind and still lands here.
    """
    _pupil(session, "Иванов Иван")

    response = client.get("/canteen")

    assert "нет записи о питании" in response.text
    assert "не означает, что ученик не ел" in response.text


def test_the_page_caveats_never_came_with_up_to_n_when_sessions_are_unattributed(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """Part 2: only when unknown_sessions > 0, and strictly UP TO N.

    Never exactly N: an unattributed session may be staff, a visitor, or a child who also
    has a recognised session elsewhere in the day. Two unknown sessions do not mean two of
    these pupils ate.
    """
    _pupil(session, "Иванов Иван")
    _pupil(session, "Петров Пётр")
    _unknown_entry(session, camera)
    _unknown_entry(session, camera)

    response = client.get("/canteen")

    assert "до 2 из перечисленных" in response.text, "the caveat must say UP TO N, not N"


def test_the_page_does_not_claim_never_came_is_certain_when_nothing_is_unknown(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The trap in part 2: N == 0 must not read as "then never_came is certain".

    A child the detector never saw produces NO session, named or unknown, so
    `unknown_sessions` does not count them -- and they still land in never_came. That third
    failure mode is unmeasured, so the page must simply not make the claim.

    The second assertion below used to read `"точно" not in response.text`, which could not
    fail: no surface contains the lowercase word (the caveat says «Точнее», capital Т, and
    only when N > 0 -- so on this zero-N page it is not rendered at all). A guard that
    cannot fire is not a guard. What the test actually means is that the page must still
    CARRY the unconditional half of the caveat: silence here is precisely what would let
    a zero N read as certainty.
    """
    _pupil(session, "Иванов Иван")

    response = client.get("/canteen")

    assert "могли поесть" not in response.text, "the up-to-N caveat fired with nothing unknown"
    assert "не означает, что ученик не ел" in response.text, (
        "with nothing unknown the page dropped the caveat, and the list reads as certain"
    )


def test_the_tile_does_not_say_never_came_where_no_caveat_can_reach_it(
    client: TestClient, session: Session, camera: Camera
) -> None:
    """The tile is the number an operator actually glances at, and it sits far above the
    caveat under the table. A label reading "не приходили" there is the claim we cannot
    support, rendered in the one place the explanation cannot follow it. Label it after
    what it counts -- pupils with no meal record -- so it is honest at a glance.
    """
    _pupil(session, "Иванов Иван")

    tiles = _tiles(client.get("/canteen").text)

    assert tiles.get("нет записи о питании") == "1"
    assert "не приходили" not in tiles, "the tile asserts an absence we did not measure"

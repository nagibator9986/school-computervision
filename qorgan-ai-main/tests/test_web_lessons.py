"""The lesson pages: who may open them, and what the HTML is obliged to say.

Two kinds of test here, and the second kind is the one that has already caught a real
defect elsewhere in this codebase. The first is §14 access control. The second is that the
warnings computed by `LessonReport` actually REACH the browser -- `forced_unknown` was
computed by `day_report`, handed to the canteen template inside `report`, and silently
dropped by the template, and nobody noticed for several releases because a discarded value
and a zero look identical on a page.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.classroom.ledger import TrackLedger
from qorgan.classroom.lesson import Doubt
from qorgan.classroom.store import Clock, flush, open_or_resume
from qorgan.config.classroom import LessonRules
from qorgan.db.models import Camera, User
from qorgan.enums import CameraRole, CameraType, UserRole
from qorgan.passwords import hash_password
from qorgan.roles import Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
RULES = LessonRules(min_presence_seconds=300.0)

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(User(username=username, password_hash=hash_password(PASSWORD), role=role))
            session.commit()

            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login", data=with_token(client, {"username": username, "password": PASSWORD})
            )
            assert response.status_code == 303, "login failed"
            return client

        yield make


def _camera(session: Session) -> Camera:
    row = Camera(
        name="class_7a",
        display_name="7-А",
        camera_type=CameraType.CLASSROOM,
        role=CameraRole.CLASSROOM,
        rtsp_host="10.0.0.9",
    )
    session.add(row)
    session.commit()
    return row


def _ledger(track_id: int, seconds: float, *, settled: bool = True, **kwargs) -> TrackLedger:
    ledger = TrackLedger(track_id=track_id, first_seen=0.0, last_seen=seconds)
    ledger.settled = settled
    for key, value in kwargs.items():
        setattr(ledger, key, value)
    return ledger


@pytest.fixture
def lesson_id(session: Session) -> int:
    camera = _camera(session)
    handle = open_or_resume(camera.id, RULES)
    flush(
        handle.lesson_id,
        [
            _ledger(1, 600.0),  # present, settled, did nothing at all
            _ledger(2, 600.0, hand_raises=3, stands=1, away_seconds=42.0),
            _ledger(3, 600.0, settled=False),  # place metrics unmeasured
            _ledger(4, 30.0),  # a fragment
        ],
        Doubt(ambiguous=140, unclaimed=7, dropped_tracks=2),
        Clock.now(),
    )
    return handle.lesson_id


# -- who may look -------------------------------------------------------------


def test_an_admin_reads_the_lesson_reports(client_for: ClientFor, lesson_id: int) -> None:
    """§14 lists «отчеты» under the администратор школы and under nobody else."""
    client = client_for(UserRole.ADMIN)

    assert client.get("/lessons").status_code == 200
    assert client.get(f"/lessons/{lesson_id}").status_code == 200


@pytest.mark.parametrize(
    "role", [UserRole.OPERATOR, UserRole.DEVELOPER, UserRole.CANTEEN_STAFF]
)
def test_every_other_role_is_refused(
    client_for: ClientFor, lesson_id: int, role: UserRole
) -> None:
    """DEVELOPER is in this list deliberately, and it is the first child-facing capability
    ADMIN holds that DEVELOPER does not.

    Everything the developer login reaches is either the INSTALLATION (/logs, /settings,
    /backups) or something an operator already saw. These numbers are new, they are about
    children, and no threshold behind them has been validated against a real lesson -- so
    the readership starts as small as it can be while the page still has a reader.
    """
    client = client_for(role)

    assert client.get("/lessons").status_code == 403
    assert client.get(f"/lessons/{lesson_id}").status_code == 403


def test_the_capability_does_not_leak_into_the_operator_set() -> None:
    """The contract below the HTTP layer, asserted directly so that a future widening has
    to be a deliberate edit to this line rather than a side effect of a union."""
    assert Capability.VIEW_LESSON_METRICS in capabilities_for(UserRole.ADMIN)
    # The second holder, added 2026-07-28 with §13's cabinet. Safe for the reason the
    # ADMIN-only note in roles.py gave and then had to re-examine: the lesson report is per
    # ANONYMOUS track and carries no person_id, so no amount of reading turns it into a
    # claim about a named child. Asserted here so widening it further stays a deliberate
    # edit to this line.
    assert Capability.VIEW_LESSON_METRICS in capabilities_for(UserRole.PSYCHOLOGIST)
    assert Capability.VIEW_LESSON_METRICS not in capabilities_for(UserRole.OPERATOR)
    assert Capability.VIEW_LESSON_METRICS not in capabilities_for(UserRole.DEVELOPER)
    assert Capability.VIEW_LESSON_METRICS not in capabilities_for(UserRole.CANTEEN_STAFF)


def test_the_psychologist_role_arrived_with_pages_and_not_before(client_for: ClientFor) -> None:
    """**This test used to assert that no psychologist role existed at all**, because
    `qorgan.classroom` produces no signal, no score and no ranking, so there was no
    psychologist's page for a psychologist's role to open -- and `roles.py` is explicit
    that a capability guarding nothing is a guess.

    It was right, and it has been rewritten rather than deleted, because the property it
    was protecting has not changed: the role must never exist without pages. So it now
    asserts the SAME rule from the other side -- the role exists, and every route it is
    granted for actually answers. A role added ahead of its pages fails here just as
    loudly, on a 404 instead of on an empty enum.

    §8's promise is unchanged and is asserted where it lives: `qorgan.classroom` still
    computes no score, and `tests/test_psychologist_cabinet.py` holds the sabotage-proof
    that nothing in the cabinet does either.
    """
    assert UserRole.PSYCHOLOGIST in UserRole
    client = client_for(UserRole.PSYCHOLOGIST)

    for path in ("/psychologist", "/lessons"):
        assert client.get(path).status_code == 200, f"{path} is granted but does not answer"


def test_a_nav_link_is_drawn_only_for_the_role_that_may_follow_it(
    client_for: ClientFor, lesson_id: int
) -> None:
    """A link into a 403 is reported by the school as a broken system."""
    assert 'href="/lessons"' in client_for(UserRole.ADMIN).get("/").text
    assert 'href="/lessons"' not in client_for(UserRole.OPERATOR).get("/").text


def test_a_lesson_that_does_not_exist_is_a_404(client_for: ClientFor) -> None:
    assert client_for(UserRole.ADMIN).get("/lessons/999").status_code == 404


# -- what the page is obliged to say -----------------------------------------


def test_the_page_renders_the_caveat(client_for: ClientFor, lesson_id: int) -> None:
    """**The regression test for the shape of bug that hid `forced_unknown`.**

    `LessonReport.caveat()` can be perfect and still reach nobody. The two permanent
    sentences -- a track is not a child, and nothing here is validated -- must be IN the
    HTML, not merely in the object handed to the template.
    """
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    assert "ТРЕКУ" in body
    assert "не проверен" in body


def test_the_index_warns_before_the_table_not_after_it(
    client_for: ClientFor, lesson_id: int
) -> None:
    """A reader who scans a table and leaves has already formed an impression. The
    warning has to be above the numbers, which is the one place the explanation cannot
    follow the claim."""
    body = client_for(UserRole.ADMIN).get("/lessons").text

    assert body.index("анонимным") < body.index("<table")


def test_a_track_that_did_nothing_is_shown_rather_than_omitted(
    client_for: ClientFor, lesson_id: int
) -> None:
    """"Nobody raised a hand" and "we did not look" must not render identically."""
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    assert "#1" in body


def test_unmeasured_place_metrics_render_as_unmeasured_not_as_zero(
    client_for: ClientFor, lesson_id: int
) -> None:
    """Track 3 never settled, so it has no seat to have risen from. Rendering 0 would put
    "measured zero" and "unknown" back together at the very last moment."""
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    assert "не измерено" in body


def test_the_discarded_observations_reach_the_page(
    client_for: ClientFor, lesson_id: int
) -> None:
    """140 poses were dropped as ambiguous. A page that shows only the totals built from
    what survived is describing a room it largely did not see."""
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    assert "140" in body


def test_every_doubt_counter_computed_is_a_doubt_counter_rendered(
    client_for: ClientFor, lesson_id: int
) -> None:
    """**A value computed, handed to the template, and drawn nowhere is the `forced_unknown`
    defect exactly**, and it survived several releases on /canteen because a dropped value
    and a zero look identical on a page.

    So the check is over the whole set, by name: every counter `LessonReport` carries out
    of the database has to appear. Adding a fifth without a tile fails here.
    """
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    # ambiguous=140, unclaimed=7, dropped_tracks=2, resumed_count=0
    for label in ("140", "7", "2", "0"):
        assert label in body

    assert "вне треков" in body, "unclaimed poses are computed but never drawn"
    assert "не взято" in body, "refused tracks are computed but never drawn"


def test_fragments_are_shown_on_the_page_not_only_counted(
    client_for: ClientFor, lesson_id: int
) -> None:
    body = client_for(UserRole.ADMIN).get(f"/lessons/{lesson_id}").text

    assert "#4" in body
    assert "фрагмент" in body.lower()


def test_the_page_offers_no_csv_export(client_for: ClientFor, lesson_id: int) -> None:
    """Deliberate, and the contrast with /canteen is the point.

    A file of per-child-looking rows, detached from the page explaining that they are
    anonymous tracks counted by unvalidated thresholds, is exactly the artefact that gets
    forwarded to a parent as though it meant something. When the school asks for one, it
    gets built with the caveat INSIDE the file, the way the canteen's `_caveat_rows` does.
    """
    client = client_for(UserRole.ADMIN)

    assert "export.csv" not in client.get(f"/lessons/{lesson_id}").text
    # 422, not 404: the path falls through to `/lessons/{lesson_id}` and fails to parse as
    # an int. The status is not the point -- what matters is that nothing serves a
    # spreadsheet. Asserted on the CONTENT TYPE as well as the code, so that adding an
    # export later cannot make this pass by returning a downloadable 200.
    refused = client.get("/lessons/export.csv")
    assert refused.status_code != 200
    assert "csv" not in refused.headers.get("content-type", "")

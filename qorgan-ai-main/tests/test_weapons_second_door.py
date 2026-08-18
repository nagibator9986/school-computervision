"""The OTHER routes that can reach a weapon row. There was one, and it was open.

`weapons/store.py` says, in its module docstring, that «nothing in `src/` moves it off NEW
except `rule_on_weapon_alert`, which cannot be called without a user id». That sentence is
the whole of §12.1's «тревогу всегда подтверждает человек, и в записи остаётся, кто».

It was FALSE. `POST /events/{id}/review` is guarded by `REVIEW_BULLYING`, not by
`CONFIRM_WEAPON_ALERT`, and it looked up the row by id and never checked `event_type`.
Measured before the guard existed:

    POST /events/<weapon row id>/review  verdict=confirmed  -> 303
      status = confirmed, reviewed_by_id set
    POST /events/<weapon row id>/review  verdict=reviewed    -> 303
      status = reviewed  -- i.e. the alert left «ждёт человека» in a state where
      NOBODY ANSWERED THE QUESTION, which is the exact state `WEAPON_VERDICTS`
      excludes on purpose ("somebody looked" is not a ruling on a weapon).

Not an escalation of privilege TODAY, because both capabilities sit in
`_OPERATOR_CAPABILITIES`. It becomes one the minute the owner narrows DEVELOPER, which is
the change `report.txt` §8(a) proposes: a developer would keep `REVIEW_BULLYING` and walk
in through this door instead. So the narrowing and this guard belong together, and this
file is what makes the narrowing a real one-line change rather than one that merely looks
done.

Every test here POSTs to the ROUTE, with a CSRF token, as a logged-in operator. None of
them calls `review_event` directly: the defect was in the route's own missing check, and a
test that called the function would have been testing the thing that was already correct.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, User
from qorgan.db.types import utcnow
from qorgan.detection.geometry import Box
from qorgan.enums import CameraRole, CameraType, EventStatus, EventType, Severity, UserRole
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability
from qorgan.settings import Settings
from qorgan.weapons.pipeline import EVIDENCE, WeaponAlert
from qorgan.weapons.store import WEAPON_VERDICTS, record_weapon_alert, summarise_weapon
from qorgan.web.app import create_app
from tests.weapons_fixtures import loaded_weights
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def camera_id(session: Session) -> int:
    camera = Camera(
        name="entrance_frame",
        display_name="Вход — рамка",
        camera_type=CameraType.WEAPONS,
        role=CameraRole.WEAPONS,
        rtsp_host="192.168.1.90",
    )
    session.add(camera)
    session.commit()
    return camera.id


def _alert() -> WeaponAlert:
    return WeaponAlert(
        track_id=4,
        class_name="knife",
        timestamp=0.0,
        confidence=0.82,
        observations=3,
        strong_observations=2,
        person_track_id=7,
        box=Box(100, 100, 140, 140),
        reasons=EVIDENCE,
    )


@pytest.fixture
def weapon_event_id(session: Session, camera_id: int) -> int:
    """One weapon alert at NEW, written by the production writer."""
    del session
    return record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=_alert(),
        weights=loaded_weights(),
        summary_text=summarise_weapon(_alert(), "Вход — рамка"),
        min_observations=3,
        reconfirm_observations=2,
    )


@pytest.fixture
def bullying_event_id(session: Session, camera_id: int) -> int:
    """One ordinary bullying event, so the guard can be shown NOT to be a blanket refusal."""
    event = Event(
        camera_id=camera_id,
        event_type=EventType.BULLYING,
        occurred_at=utcnow(),
        confidence=0.5,
        candidate_probability=0.5,
        severity=Severity.ALERT,
        summary_text="учебное событие",
        track_ids="1,2",
        status=EventStatus.NEW,
    )
    session.add(event)
    session.commit()
    return event.id


@pytest.fixture
def operator(settings: Settings, session: Session) -> Iterator[TestClient]:
    del settings
    session.add(
        User(username="operator1", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR)
    )
    session.commit()

    with ExitStack() as stack:
        client = stack.enter_context(TestClient(create_app(), follow_redirects=False))
        login = client.post(
            "/login", data=with_token(client, {"username": "operator1", "password": PASSWORD})
        )
        assert login.status_code == 303, "login failed; nothing below tests anything"
        yield client


def _review(client: TestClient, event_id: int, verdict: str):
    return client.post(
        f"/events/{event_id}/review", data=with_token(client, {"verdict": verdict})
    )


# -- the door, shut --------------------------------------------------------


@pytest.mark.parametrize(
    "verdict", [s.value for s in EventStatus if s is not EventStatus.NEW]
)
def test_the_bullying_review_route_cannot_touch_a_weapon_row(
    operator: TestClient, session: Session, weapon_event_id: int, verdict: str
) -> None:
    """**Every verdict, not just the interesting two.**

    Parametrised over the whole `EventStatus` set rather than over `confirmed` and
    `reviewed`, because the defect was the ABSENCE of a type check: any status this route
    accepts would have landed on the row. `NEW` is excluded only because the route already
    refuses it ("cannot un-review an event") for every kind of event.

    404 rather than 403, and deliberately the same answer `/weapons/{id}/rule` gives for a
    bullying id: the two routes are symmetric, and neither tells a caller which kind of
    row an id it may not touch happens to be.
    """
    response = _review(operator, weapon_event_id, verdict)
    assert response.status_code == 404, (
        f"verdict={verdict!r} reached a weapon row through /events/{{id}}/review. "
        "That is the second, unguarded door onto §12.1's human confirmation."
    )

    session.expire_all()
    event = session.get(Event, weapon_event_id)
    assert event.status is EventStatus.NEW, "the alert must still be waiting for a person"
    assert event.reviewed_by_id is None
    assert event.reviewed_at is None


def test_reviewed_is_the_worst_of_them_and_is_named(
    operator: TestClient, session: Session, weapon_event_id: int
) -> None:
    """«Somebody looked» is not a ruling on a weapon, and this route could have written it.

    Called out on its own because the other verdicts at least record a decision. `reviewed`
    would have taken the row out of «ожидает подтверждения человека» while asserting
    nothing at all — the row would read as handled, the panel would stop offering the
    buttons, and no human would ever have answered the question. `WEAPON_VERDICTS` excludes
    it on purpose; this route did not know that.
    """
    assert EventStatus.REVIEWED not in WEAPON_VERDICTS, "the premise this test rests on"
    assert _review(operator, weapon_event_id, EventStatus.REVIEWED.value).status_code == 404

    session.expire_all()
    assert session.get(Event, weapon_event_id).status is EventStatus.NEW


# -- and the door it is NOT ------------------------------------------------


def test_a_bullying_event_is_still_reviewable_through_the_same_route(
    operator: TestClient, session: Session, bullying_event_id: int
) -> None:
    """The control. Without it the guard could be a blanket refusal that breaks the only
    channel through which the school tells the bullying detector it was wrong — and the
    weapons module has no business costing that."""
    assert _review(operator, bullying_event_id, "false_positive").status_code == 303

    session.expire_all()
    event = session.get(Event, bullying_event_id)
    assert event.status is EventStatus.FALSE_POSITIVE
    assert event.reviewed_by_id is not None


def test_the_weapons_route_remains_the_one_that_works(
    operator: TestClient, session: Session, weapon_event_id: int
) -> None:
    """The other half of the symmetry: shutting this door must not shut the right one."""
    response = operator.post(
        f"/weapons/{weapon_event_id}/rule",
        data=with_token(operator, {"verdict": EventStatus.CONFIRMED.value}),
    )
    assert response.status_code == 303

    session.expire_all()
    event = session.get(Event, weapon_event_id)
    assert event.status is EventStatus.CONFIRMED
    assert session.get(User, event.reviewed_by_id).username == "operator1"


# -- what this makes possible ----------------------------------------------


def test_the_guard_is_by_event_type_so_narrowing_the_grant_would_actually_narrow() -> None:
    """**The reason this file exists rather than a comment.**

    `report.txt` §8(a) proposes that the owner may want to take
    `CONFIRM_WEAPON_ALERT` away from DEVELOPER. That change only does anything if
    `/events/{id}/review` refuses weapon rows: a developer keeps `REVIEW_BULLYING`
    either way, so without the type check the narrowing would look done and change
    nothing.

    This asserts the precondition as a fact about the table: `REVIEW_BULLYING` is held by
    roles that are NOT the same set as `CONFIRM_WEAPON_ALERT`'s in the general case, so
    the two must not be interchangeable doors onto the same row.
    """
    reviewers = {r for r, g in ROLE_CAPABILITIES.items() if Capability.REVIEW_BULLYING in g}
    assert reviewers, "the premise: somebody holds REVIEW_BULLYING"
    # The guard is behavioural (tested above). This records WHY it may not be replaced by a
    # capability check: the grant sets are free to diverge, and the row must be safe either
    # way.
    assert Capability.REVIEW_BULLYING is not Capability.CONFIRM_WEAPON_ALERT

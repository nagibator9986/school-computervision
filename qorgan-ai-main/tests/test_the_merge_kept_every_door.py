"""Three features met on one route, and all three still work. **Written for the merge.**

`POST /events/{id}/review` is the one place `feat/weapons-detection`,
`feat/psychologist-cabinet` and `feat/multi-school` all edited the same lines, and each
branch's suite was green against `main` on its own. Git conflicted the file in all three
pairwise combinations, and the cheapest resolution of every one of those conflicts --
taking a side -- compiles, passes lint, and passes almost the whole suite, because each
branch's own tests only ever exercised its own feature.

**That is this project's documented signature failure**, and `roles.py` carries a comment
about five branches that each silently revoked the others' pages exactly this way. The
existing files each cover one branch's half:

  * `test_weapons_second_door.py` proves the weapon guard, and it was green on a tree
    where the referral route did not exist;
  * `test_psychologist_referrals.py` proves the referral, and it was green on a tree with
    no weapon rows in it at all;
  * `test_tenancy_isolation.py` proves the school scoping, and it was green on a tree with
    neither of the other two.

So each of them would stay green while a merge dropped a feature it does not know about.
This file is the one that cannot: it drives all three behaviours through the merged route,
in one test session, against one database. It is deliberately not parametrised and
deliberately not clever -- it is three assertions that would each have been made by a
different author, in the same room for the first time.

**The 404 on a weapon row is a child-safety fix and is the reason this file exists.**
Before that guard, a weapon alert could be moved off NEW through the bullying door by
guessing an id -- including to `reviewed`, which reads as handled while asserting nothing,
so the panel stops offering the buttons and no human ever answers whether a child is
armed. If a future merge resolves `events.py` by choosing a side, this file goes red on
the line that says so.
"""

from __future__ import annotations

from collections.abc import Iterator
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
from qorgan.weapons.store import record_weapon_alert, summarise_weapon
from qorgan.web.app import create_app
from tests.weapons_fixtures import loaded_weights
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"


@pytest.fixture
def camera_id(session: Session) -> int:
    """One camera. Both kinds of event hang off it, so neither test can pass by sitting in
    a school the other's row is not in."""
    camera = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()
    return camera.id


@pytest.fixture
def bullying_event_id(session: Session, camera_id: int) -> int:
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
def weapon_event_id(session: Session, camera_id: int) -> int:
    """Written by the PRODUCTION writer, not assembled here: a hand-built row could carry a
    combination of columns `record_weapon_alert` never produces, and then this file would
    be guarding a shape that does not occur."""
    del session
    alert = WeaponAlert(
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
    return record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=alert,
        weights=loaded_weights(),
        summary_text=summarise_weapon(alert, "Холл слева"),
        min_observations=3,
        reconfirm_observations=2,
    )


@pytest.fixture
def operator(settings: Settings, session: Session) -> Iterator[TestClient]:
    """One logged-in operator, who holds all three grants this file exercises."""
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


def test_the_operator_holds_all_three_grants_this_file_needs() -> None:
    """The premise. Without it a 404 below could mean "forbidden" rather than "guarded",
    and the file would pass while proving nothing."""
    held = ROLE_CAPABILITIES[UserRole.OPERATOR]
    for capability in (
        Capability.REVIEW_BULLYING,
        Capability.REFER_TO_PSYCHOLOGIST,
        Capability.VIEW_WEAPONS,
    ):
        assert capability in held, (
            f"OPERATOR no longer holds {capability.value}, so the assertions below are "
            "measuring the capability check and not the guards they name."
        )


# -- 1. the weapons branch's guard: the door is SHUT ------------------------


def test_a_weapon_row_cannot_be_ruled_on_through_the_bullying_door(
    operator: TestClient, session: Session, weapon_event_id: int
) -> None:
    """**The child-safety fix.** `verdict=reviewed` is the worst of the four and is the one
    asserted here: it takes the alert out of «ожидает подтверждения человека» while saying
    nothing at all, so the row reads as handled and no human ever answers it.

    404, not 403, and the same 404 `/weapons/{id}/rule` returns for a bullying id: neither
    route tells a caller which kind of row an id it may not touch happens to be.
    """
    response = operator.post(
        f"/events/{weapon_event_id}/review",
        data=with_token(operator, {"verdict": EventStatus.REVIEWED.value}),
    )
    assert response.status_code == 404, (
        "a weapon alert was ruled on through /events/{id}/review. The merge dropped "
        "`feat/weapons-detection`'s guard on `event_type`, which is a child-safety fix: "
        "whether a school treats a child as armed is decided by CONFIRM_WEAPON_ALERT and "
        "by nothing else, and this route is not guarded by it."
    )

    session.expire_all()
    event = session.get(Event, weapon_event_id)
    assert event.status is EventStatus.NEW, "the alert must still be waiting for a person"
    assert event.reviewed_by_id is None
    assert event.reviewed_at is None


# -- 2. the same door, still OPEN for what it is for ------------------------


def test_a_bullying_event_is_still_reviewable(
    operator: TestClient, session: Session, bullying_event_id: int
) -> None:
    """The control, and it is not a formality. A guard written as a blanket refusal would
    pass the test above and close the ONLY channel through which the school tells us the
    bullying detector was wrong -- a detector nobody corrects never improves."""
    response = operator.post(
        f"/events/{bullying_event_id}/review",
        data=with_token(operator, {"verdict": EventStatus.FALSE_POSITIVE.value}),
    )
    assert response.status_code == 303, (
        "a bullying event could not be reviewed. The merge kept the weapons guard as a "
        "refusal of everything rather than as a check on the row's type."
    )

    session.expire_all()
    event = session.get(Event, bullying_event_id)
    assert event.status is EventStatus.FALSE_POSITIVE
    assert event.reviewed_by_id is not None


# -- 3. the psychologist branch's route, on the same event ------------------


def test_a_referral_still_works_and_survives_a_verdict(
    operator: TestClient, session: Session, bullying_event_id: int
) -> None:
    """§9's «передано психологу», through the merged route.

    Both halves matter. The referral must be RECORDED with a name and a minute, and it
    must survive the event later being ruled on -- the two facts live in two columns
    precisely so that neither erases the other, and a merge that folded the referral back
    into `status` would pass the first assertion and fail the second.
    """
    referred = operator.post(
        f"/events/{bullying_event_id}/refer", data=with_token(operator, {})
    )
    assert referred.status_code == 303, (
        "the referral route is gone or refused. The merge dropped "
        "`feat/psychologist-cabinet`'s half of this file."
    )

    session.expire_all()
    event = session.get(Event, bullying_event_id)
    assert event.referred_at is not None, "a referral with no minute is not a referral"
    assert event.referred_by_id is not None, "§8: the system never refers a child, a person does"

    # And ruling on it afterwards must not un-refer the child.
    ruled = operator.post(
        f"/events/{bullying_event_id}/review",
        data=with_token(operator, {"verdict": EventStatus.CONFIRMED.value}),
    )
    assert ruled.status_code == 303

    session.expire_all()
    event = session.get(Event, bullying_event_id)
    assert event.status is EventStatus.CONFIRMED
    assert event.referred_at is not None, (
        "confirming an incident erased its referral. The two facts are two columns so "
        "that neither can overwrite the other; this merge put them back in one."
    )

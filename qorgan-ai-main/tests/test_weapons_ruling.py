"""§12.1: «тревога никогда не действует автоматически — подтверждает человек».

A false gun alert in a school has consequences of its own, and they land on a child. So
what this module produces is a QUESTION, and the answer is somebody's name on the row.

**These tests submit the form the page actually renders.** They read the action and the
verdict off the HTML and post to that, rather than naming `/weapons/1/rule` in the test.
The difference is not pedantry: this suite has already shipped a green test asserting an
event could be reassigned to a psychologist while the page had stopped drawing the button,
and a green login suite while logging in from a browser was impossible. A test that names
the URL itself is testing the router. `_confirm_form` makes these test the page.

The second thing here is the capability. `CONFIRM_WEAPON_ALERT` is separate from
`VIEW_WEAPONS` because whoever holds it decides whether a school treats a child as armed
-- and R5's route walk cannot catch a route guarded by the WRONG capability, only one
guarded by none. So the roles are checked by hand, in both directions.
"""

from __future__ import annotations

import re
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
from qorgan.weapons.store import (
    WEAPON_VERDICTS,
    record_weapon_alert,
    rule_on_weapon_alert,
    summarise_weapon,
)
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


def _alert(track_id: int = 4) -> WeaponAlert:
    return WeaponAlert(
        track_id=track_id,
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
def event_id(session: Session, camera_id: int) -> int:
    """One alert, written by the production writer. Never a hand-built row."""
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
def client_for(settings: Settings, session: Session) -> Iterator[ClientFor]:
    del settings
    app = create_app()
    with ExitStack() as stack:

        def make(role: UserRole) -> TestClient:
            username = f"user_{role.value}"
            session.add(
                User(username=username, password_hash=hash_password(PASSWORD), role=role)
            )
            session.commit()
            client = stack.enter_context(TestClient(app, follow_redirects=False))
            response = client.post(
                "/login", data=with_token(client, {"username": username, "password": PASSWORD})
            )
            assert response.status_code == 303, f"{role} could not log in"
            return client

        yield make


@pytest.fixture
def operator(client_for: ClientFor) -> TestClient:
    return client_for(UserRole.OPERATOR)


def _confirm_form(page: str) -> tuple[str, str]:
    """The action and verdict of the CONFIRM button, as the page renders them.

    Raises rather than returning None when there is no such button: "the page does not
    offer this" must fail the test that is about pressing it, not silently skip it.
    """
    for action, body in re.findall(
        r'<form method="post" action="([^"]+)">(.*?)</form>', page, re.DOTALL
    ):
        verdict = re.search(r'name="verdict" value="([^"]+)"', body)
        if verdict and verdict.group(1) == EventStatus.CONFIRMED.value:
            return action, verdict.group(1)
    raise AssertionError("the page renders no confirm button; nothing can be ruled on")


# -- the row is born unanswered -------------------------------------------


def test_a_weapon_alert_is_written_at_status_new(session: Session, event_id: int) -> None:
    """Nothing in `src/` moves it off NEW except a person's ruling."""
    event = session.get(Event, event_id)
    assert event.status is EventStatus.NEW
    assert event.reviewed_by_id is None
    assert event.reviewed_at is None


def test_the_summary_a_phone_will_show_asks_rather_than_asserts(
    session: Session, event_id: int
) -> None:
    """The notifier reads this string back off the row, so it is what appears on a
    teacher's phone -- and «Обнаружено оружие» on a phone IS an automatic action,
    whatever the status column says sixty seconds later."""
    summary = session.get(Event, event_id).summary_text
    assert summary.startswith("Возможное оружие")
    assert "Требуется подтверждение человека" in summary
    assert "Обнаружено оружие" not in summary


def test_the_row_records_which_weights_said_it(session: Session, event_id: int) -> None:
    """"Which model said this?" asked six months later cannot be answered from a config
    file that has been edited since."""
    assert f"weights:{loaded_weights().file.fingerprint}" in session.get(Event, event_id).reasons


def test_the_row_is_a_weapon_and_not_a_bullying_event(session: Session, event_id: int) -> None:
    """`skeleton_confirmed` is False and means NOT APPLICABLE -- there is no pose tier
    here -- which is why /events filters to BULLYING and this page is separate."""
    event = session.get(Event, event_id)
    assert event.event_type is EventType.WEAPON
    assert event.skeleton_confirmed is False


# -- a person answers it, through the page ---------------------------------


def test_the_page_offers_a_confirm_button_for_a_new_alert(
    operator: TestClient, event_id: int
) -> None:
    page = operator.get("/weapons").text
    assert "ожидает подтверждения человека" in page
    action, _ = _confirm_form(page)
    assert action == f"/weapons/{event_id}/rule"


def test_pressing_the_button_the_page_renders_records_who_pressed_it(
    operator: TestClient, session: Session, event_id: int
) -> None:
    """**The requirement, end to end.** The form comes off the page; the name comes off
    the session."""
    action, verdict = _confirm_form(operator.get("/weapons").text)
    response = operator.post(action, data=with_token(operator, {"verdict": verdict}))
    assert response.status_code == 303

    session.expire_all()
    event = session.get(Event, event_id)
    assert event.status is EventStatus.CONFIRMED
    assert event.reviewed_at is not None
    assert session.get(User, event.reviewed_by_id).username == "user_operator"


def test_the_page_then_shows_who_decided(
    operator: TestClient, session: Session, event_id: int
) -> None:
    """«В записи остаётся, кто подтвердил» is a thing on a screen, not a line in an audit
    log nobody opens."""
    del session, event_id
    action, verdict = _confirm_form(operator.get("/weapons").text)
    operator.post(action, data=with_token(operator, {"verdict": verdict}))

    page = operator.get("/weapons").text
    assert "решение принял: user_operator" in page
    assert "ожидает подтверждения человека" not in page


def test_the_name_recorded_is_the_session_and_not_a_form_field(
    operator: TestClient, session: Session, event_id: int
) -> None:
    """A username in the POST body must not be able to sign somebody else's name."""
    session.add(
        User(username="headteacher", password_hash=hash_password(PASSWORD), role=UserRole.ADMIN)
    )
    session.commit()

    action, verdict = _confirm_form(operator.get("/weapons").text)
    operator.post(
        action,
        data=with_token(operator, {"verdict": verdict, "username": "headteacher", "user_id": "2"}),
    )

    session.expire_all()
    reviewer = session.get(User, session.get(Event, event_id).reviewed_by_id)
    assert reviewer.username == "user_operator"


def test_a_false_positive_is_the_other_answer(
    operator: TestClient, session: Session, event_id: int
) -> None:
    page = operator.get("/weapons").text
    assert 'value="false_positive"' in page, "the page must offer both answers"
    operator.post(
        f"/weapons/{event_id}/rule",
        data=with_token(operator, {"verdict": EventStatus.FALSE_POSITIVE.value}),
    )
    session.expire_all()
    assert session.get(Event, event_id).status is EventStatus.FALSE_POSITIVE


# -- what is not an answer -------------------------------------------------


def test_somebody_looked_is_not_a_ruling(operator: TestClient, event_id: int) -> None:
    """`REVIEWED` would leave the row reading as handled while asserting nothing."""
    assert EventStatus.REVIEWED not in WEAPON_VERDICTS
    response = operator.post(
        f"/weapons/{event_id}/rule",
        data=with_token(operator, {"verdict": EventStatus.REVIEWED.value}),
    )
    assert response.status_code == 400


def test_an_invented_verdict_is_refused(operator: TestClient, event_id: int) -> None:
    response = operator.post(
        f"/weapons/{event_id}/rule", data=with_token(operator, {"verdict": "probably"})
    )
    assert response.status_code == 400


def test_a_bullying_event_cannot_be_confirmed_as_a_weapon(
    operator: TestClient, session: Session, camera_id: int
) -> None:
    """A caller must not be able to turn one kind of record into another by guessing an
    id."""
    event = Event(
        camera_id=camera_id,
        event_type=EventType.BULLYING,
        occurred_at=utcnow(),
        confidence=0.5,
        candidate_probability=0.5,
        severity=Severity.ALERT,
        summary_text="drill",
        track_ids="1,2",
        status=EventStatus.NEW,
    )
    session.add(event)
    session.commit()

    response = operator.post(
        f"/weapons/{event.id}/rule", data=with_token(operator, {"verdict": "confirmed"})
    )
    assert response.status_code == 404
    session.expire_all()
    assert session.get(Event, event.id).status is EventStatus.NEW


def test_the_store_refuses_a_verdict_that_is_not_a_ruling(event_id: int) -> None:
    """The refusal lives in the store as well as the route: a second caller one day must
    not be able to reach past it."""
    with pytest.raises(ValueError):
        rule_on_weapon_alert(event_id, EventStatus.REVIEWED, 1, "somebody")


def test_ruling_on_an_event_that_does_not_exist_says_so(session: Session) -> None:
    del session
    assert rule_on_weapon_alert(999_999, EventStatus.CONFIRMED, 1, "somebody") is False


# -- who may answer it -----------------------------------------------------


def test_the_two_grants_are_separate_capabilities() -> None:
    """Reading a log and deciding a child is armed are not one right."""
    assert Capability.VIEW_WEAPONS is not Capability.CONFIRM_WEAPON_ALERT


def test_a_canteen_worker_reaches_neither_the_page_nor_the_ruling(
    client_for: ClientFor, event_id: int
) -> None:
    """§14: кантина «БЕЗ доступа к буллингу», and a weapon alert is further from the
    canteen journal than a bullying event is. R5 walks the route table and would pass
    with either capability on this route; only this test says which."""
    canteen = client_for(UserRole.CANTEEN_STAFF)
    assert canteen.get("/weapons").status_code == 403
    response = canteen.post(
        f"/weapons/{event_id}/rule", data=with_token(canteen, {"verdict": "confirmed"})
    )
    assert response.status_code == 403


def test_the_operator_holds_both_because_section_14_says_so(operator: TestClient) -> None:
    """§14 gives the оператор безопасности «просмотр тревог; подтверждение/отклонение
    событий». An operator who could not rule would leave §12.1's human confirmation with
    nobody to make it."""
    del operator
    granted = ROLE_CAPABILITIES[UserRole.OPERATOR]
    assert Capability.VIEW_WEAPONS in granted
    assert Capability.CONFIRM_WEAPON_ALERT in granted


def test_exactly_these_roles_may_declare_a_child_armed() -> None:
    """**Written down because it is a policy decision, not an implementation detail.**

    `CONFIRM_WEAPON_ALERT` was granted by putting it in `_OPERATOR_CAPABILITIES`, which
    ADMIN and DEVELOPER both inherit -- so a DEVELOPER account, which exists to let the
    SUPPLIER debug the system, can currently rule that a school's pupil was carrying a
    weapon, with their name on the record.

    That follows the table's existing shape exactly (DEVELOPER already holds
    REVIEW_BULLYING for the same reason), so it is not a mistake and it is not being
    changed here on a guess: five branches each rewrote these two lines, and any single
    version taken alone would have silently revoked the others' pages. Narrowing it is
    the owner's call.

    What this test does is make the breadth VISIBLE. If somebody widens or narrows it,
    they edit this list and have to mean it.
    """
    may_rule = {
        role for role, granted in ROLE_CAPABILITIES.items()
        if Capability.CONFIRM_WEAPON_ALERT in granted
    }
    assert may_rule == {UserRole.OPERATOR, UserRole.ADMIN, UserRole.DEVELOPER}
    assert UserRole.CANTEEN_STAFF not in may_rule


def test_no_role_can_view_weapon_alerts_without_being_able_to_answer_them() -> None:
    """A page whose single control the reader may not press leaves the question unasked.

    Stated as a property over the whole table rather than over the roles that exist
    today, so a new role cannot be added into the gap.
    """
    for role, granted in ROLE_CAPABILITIES.items():
        if Capability.VIEW_WEAPONS in granted:
            assert Capability.CONFIRM_WEAPON_ALERT in granted, role


def test_nobody_reads_a_weapon_alert_without_being_able_to_reach_the_page() -> None:
    """`roles.py` claims these two grants cannot drift apart. Nothing asserted it.

    A weapon alert raises a Telegram, so the row's `summary_text` -- «Возможное оружие:
    нож ...» -- already appears on `/notifications`, which is gated on `VIEW_BULLYING`. A
    role holding `VIEW_BULLYING` and not `VIEW_WEAPONS` would therefore read the alert on
    one page and be refused the page that explains it and the button that answers it. No
    such role exists today; the comment in `roles.py` says so as a fact about "any
    arrangement of these two grants", and this is what makes that sentence true rather
    than true-for-now.
    """
    for role, granted in ROLE_CAPABILITIES.items():
        if Capability.VIEW_BULLYING in granted:
            assert Capability.VIEW_WEAPONS in granted, role


def test_ruling_requires_the_csrf_token_like_every_other_mutation(
    operator: TestClient, event_id: int
) -> None:
    response = operator.post(f"/weapons/{event_id}/rule", data={"verdict": "confirmed"})
    assert response.status_code == 403


# -- and how a human gets to the page at all -------------------------------


def test_the_nav_link_is_drawn_for_somebody_who_may_open_it(client_for: ClientFor) -> None:
    """**The path a person actually takes.**

    Every other test in this file and in `test_weapons_panel.py` reaches `/weapons` by
    naming the URL, and `test_web_auth.py` walks the route TABLE. None of them would have
    noticed a page that nothing links to. That is not a hypothetical gap in this suite: a
    green test asserted an event could be handed to a psychologist while the page had
    stopped drawing the button, and every login test passed while logging in from a
    browser was impossible. So the link is asserted where a browser finds it -- on the
    landing page, in the words on the menu.
    """
    page = client_for(UserRole.OPERATOR).get("/").text
    assert 'href="/weapons"' in page
    assert "Оружие" in page, "the link must also be findable by what it says"


def test_the_nav_link_is_not_drawn_for_somebody_who_would_get_a_403(
    client_for: ClientFor,
) -> None:
    """A link into a 403 is reported by the school as a broken system, not as the
    permission it is. Drawn from the same capability the route is gated on."""
    assert 'href="/weapons"' not in client_for(UserRole.CANTEEN_STAFF).get("/").text

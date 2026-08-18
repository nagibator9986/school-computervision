"""§9's «передано психологу»: an act by a named person, and never by the system.

The thing under test is a distinction, not a feature. `docs/questions-for-school.md` §8
promised the school «Никаких диагнозов и никаких направлений к психологу ОТ СИСТЕМЫ», and
client §9 requires that an operator be able to mark exactly that. Both hold, because a
HUMAN referring is the product and the SYSTEM referring is forbidden. So the tests here ask
two questions over and over:

  * does the row say WHO and WHEN, on every path that can set the mark?
  * is there any path that sets the mark WITHOUT a name -- i.e. any way for the system to
    have referred a child?

The second is why `/events/{id}/review` is tested for a refusal rather than left alone: it
already accepted "any EventStatus except NEW", so the moment the enum grew a member that
handler became a second door through which the referral could arrive unsigned.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, User
from qorgan.enums import EventStatus, UserRole
from qorgan.psychologist.referrals import UnknownEvent, refer, referred_incidents
from qorgan.roles import Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from qorgan.web.routes.events import REFERRED_FILTER
from tests.psychologist_fakes import ClientFor, a_camera, an_event, client_factory
from tests.web_login import with_token

# The BUILDERS are shared (`psychologist_fakes`); the FIXTURES are declared here. Importing
# a fixture from another module re-binds the name, ruff reads the import as a definition
# and calls every use of it a redefinition (F811) -- fairly, because it is not one. Three
# short declarations per module is the price of one shared world; the world itself is not
# copied, which is the part that could drift.


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    yield from client_factory(app, session)


@pytest.fixture
def camera(session: Session) -> Camera:
    return a_camera(session)


def _event_row(session: Session, event_id: int) -> Event:
    session.expire_all()
    found = session.get(Event, event_id)
    assert found is not None
    return found


# -- a referral is not a status ----------------------------------------------


def test_a_referral_is_not_an_event_status() -> None:
    """**The correction of 2026-07-29.** §9 lists five marks — «подтверждено; ложное
    срабатывание; требует доп. проверки; передано психологу; закрыто» — and it is tempting
    to read that as the contents of `EventStatus`. It is not: two of the five have never
    been members either. `EventStatus` answers one question, whether the detector was
    right, and «передано психологу» is not an answer to it.

    Asserted as an absence because the presence cost something real; see
    `test_referring_an_event_leaves_it_reviewable_from_the_page`.
    """
    assert not any("psycholog" in member.value for member in EventStatus)
    assert {member.value for member in EventStatus} == {
        "new",
        "reviewed",
        "confirmed",
        "false_positive",
    }


def test_the_events_page_offers_the_referral_as_a_filter(
    client_for: ClientFor, camera: Camera
) -> None:
    """The person who has to find these incidents still needs to filter for them — the
    filter just selects on `referred_at IS NOT NULL` instead of on a status token."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)
    body = client.get("/events").text

    assert f'value="{REFERRED_FILTER}"' in body, "no way to filter for referred incidents"
    assert "передано психологу" in body

    # And it FILTERS, rather than merely appearing in the box. The summary text is the
    # marker: an event id is a bare digit and would match half the page.
    assert "Зафиксирована агрессия" not in _filtered(client, REFERRED_FILTER)
    client.post(f"/events/{event_id}/refer", data=with_token(client))
    assert "Зафиксирована агрессия" in _filtered(client, REFERRED_FILTER)


def _filtered(client, filter_value: str) -> str:
    return client.get(f"/events?status_filter={filter_value}").text


# -- who, and when -----------------------------------------------------------


def test_a_referral_records_the_person_who_made_it(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """**The whole point.** A mark with nobody's name on it is the system referring a
    child, which is the thing §8 promised the school would never happen."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(f"/events/{event_id}/refer", data=with_token(client))
    assert response.status_code == 303

    row = _event_row(session, event_id)
    operator = session.scalar(select(User).where(User.username == "user_operator"))
    assert row.referred_by_id == operator.id, "the referral has no author"
    assert row.referred_at is not None, "the referral has no time"
    # And the verdict column is untouched: referring is not reviewing.
    assert row.status is EventStatus.NEW


def test_the_page_says_who_referred_it_and_when(
    client_for: ClientFor, camera: Camera
) -> None:
    """A name held only in a column is a name nobody reads. `forced_unknown` was computed,
    handed to a template and drawn nowhere for several releases."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)
    client.post(f"/events/{event_id}/refer", data=with_token(client))

    body = client.get("/events").text
    assert "Передано психологу" in body
    assert "user_operator" in body


def test_referring_twice_does_not_rewrite_the_first_handover(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """The handover happened when it happened. A second press is not a new fact about the
    child, and overwriting the time would quietly move a decision to today's date."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)

    client.post(f"/events/{event_id}/refer", data=with_token(client))
    first = _event_row(session, event_id).referred_at

    client.post(f"/events/{event_id}/refer", data=with_token(client))
    assert _event_row(session, event_id).referred_at == first


def test_referring_an_event_that_does_not_exist_is_a_404_not_a_silent_success(
    client_for: ClientFor,
) -> None:
    client = client_for(UserRole.OPERATOR)
    assert client.post("/events/999/refer", data=with_token(client)).status_code == 404


def test_the_writer_refuses_an_unknown_event_below_the_http_layer(session: Session) -> None:
    """Checked in the module both front doors would call, not in the route -- the rule
    `qorgan.accounts` is built around."""
    del session
    with pytest.raises(UnknownEvent):
        refer(999, user_id=1, username="ghost")


# -- the second door, which had to be closed ---------------------------------


def test_the_review_handler_cannot_record_a_referral(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """A referral arriving through the review handler would carry no name and no minute --
    a child recorded as handed over by nobody, at no time, which is the SYSTEM referring a
    child and is what §8 promised the school would never happen.

    This used to need an explicit branch in `review_event`. It no longer does: the enum has
    no such member, so the string fails the parse and the handler answers 400 without
    knowing what it refused. The test is kept pointing at the STRING rather than at the
    enum for that reason -- it asserts the door is shut, not how it is shut, and it would
    catch a future member added back without the branch.
    """
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(
        f"/events/{event_id}/review",
        data=with_token(client, {"verdict": "referred_to_psychologist"}),
    )

    assert response.status_code == 400, "the referral was set through the review handler"
    row = _event_row(session, event_id)
    assert row.referred_at is None, "a referral was recorded with nobody's name on it"
    assert row.status is EventStatus.NEW


def test_an_ordinary_verdict_still_works(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """The refusal above must not have closed the door it was standing next to."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)

    response = client.post(
        f"/events/{event_id}/review", data=with_token(client, {"verdict": "false_positive"})
    )

    assert response.status_code == 303
    assert _event_row(session, event_id).status is EventStatus.FALSE_POSITIVE


# -- neither fact may erase the other, in either order -----------------------


def _review_controls(client, event_id: int) -> set[str]:
    """The verdicts the EVENTS PAGE actually offers for this event, read off the HTML.

    **This is the whole lesson of 2026-07-29.** The test below used to POST straight to
    `/events/{id}/review` and passed for months of nothing while the page had stopped
    drawing the form that posts there: a referred event was `status == referred_to_
    psychologist`, and the review controls are inside `{% if event.status == "new" %}`. The
    assertion was true and the behaviour was unreachable from a browser.

    So the reachability is asserted from the markup a browser would parse, and only then is
    the POST made. The suite's own history says why: every login test passed while logging
    in from a browser was impossible, because `TestClient` fetches exactly the paths a test
    names.
    """
    body = client.get("/events").text
    forms = re.findall(
        rf'<form[^>]*action="/events/{event_id}/review"[^>]*>(.*?)</form>', body, re.DOTALL
    )
    verdicts = set()
    for form in forms:
        found = re.search(r'name="verdict"\s+value="([^"]+)"', form)
        if found:
            verdicts.add(found.group(1))
    return verdicts


def test_referring_an_event_leaves_it_reviewable_from_the_page(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """**IMPORTANT-1, regression.** Referring used to write the status, which hid the
    review controls for good — so every referred incident fell silently out of the loop
    that corrects the detector. `review_event`'s own docstring: «This is the only channel
    through which the school tells us we were wrong, and a detector nobody corrects never
    improves».

    Driven through the page: the controls must still be OFFERED after the referral, and
    then pressing one must still work.
    """
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)
    assert _review_controls(client, event_id) == {"confirmed", "false_positive"}

    client.post(f"/events/{event_id}/refer", data=with_token(client))

    assert _review_controls(client, event_id) == {"confirmed", "false_positive"}, (
        "referring a child removed the school's only way to tell us the detector was wrong"
    )
    response = client.post(
        f"/events/{event_id}/review", data=with_token(client, {"verdict": "confirmed"})
    )
    assert response.status_code == 303

    row = _event_row(session, event_id)
    assert row.status is EventStatus.CONFIRMED
    assert row.referred_at is not None, "confirming an assault erased the referral"

    still_listed = [item.event_id for item in referred_incidents()]
    assert event_id in still_listed, "the cabinet lost a referred child to a later verdict"


def test_referring_an_event_does_not_erase_a_verdict_already_reached(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """**The other order, which lost data outright.** Confirm, then refer: the status used
    to be overwritten with the referral token and the verdict was UNRECOVERABLE —
    `reviewed_at`/`reviewed_by_id` record who ruled and when, but never what they ruled.

    §14 gives the psychologist «подтверждённые случаи», and under the old shape the act of
    handing one over is exactly what erased the evidence that it was confirmed.
    """
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)

    client.post(f"/events/{event_id}/review", data=with_token(client, {"verdict": "confirmed"}))
    client.post(f"/events/{event_id}/refer", data=with_token(client))

    row = _event_row(session, event_id)
    assert row.status is EventStatus.CONFIRMED, "referring the child erased the verdict"
    assert row.referred_at is not None

    referral = next(item for item in referred_incidents() if item.event_id == event_id)
    assert referral.status == "confirmed", "the cabinet cannot see it was confirmed"


def test_the_list_shows_the_current_status_rather_than_pretending(
    client_for: ClientFor, camera: Camera
) -> None:
    """"Referred, and since confirmed" and "referred, and untouched since" are different
    situations, and the psychologist is entitled to tell them apart."""
    event_id = an_event(camera)
    client = client_for(UserRole.OPERATOR)
    client.post(f"/events/{event_id}/refer", data=with_token(client))
    client.post(f"/events/{event_id}/review", data=with_token(client, {"verdict": "confirmed"}))

    referral = next(item for item in referred_incidents() if item.event_id == event_id)
    assert referral.status == "confirmed"
    assert referral.referred_by == "user_operator"


# -- who may make the mark ---------------------------------------------------


def test_the_operator_and_the_admin_may_refer(client_for: ClientFor, camera: Camera) -> None:
    """§9 gives the mark to «сотрудник», and §14 gives the headteacher the school."""
    for role in (UserRole.OPERATOR, UserRole.ADMIN):
        assert Capability.REFER_TO_PSYCHOLOGIST in capabilities_for(role)

    event_id = an_event(camera)
    client = client_for(UserRole.ADMIN)
    assert client.post(f"/events/{event_id}/refer", data=with_token(client)).status_code == 303


def test_the_developer_login_may_not_refer_a_child(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """**The deliberate divergence.** Every other operator capability flows into DEVELOPER.
    This one does not: referring a named child to the school psychologist is a claim the
    SCHOOL makes about a person, which is the line `MERGE_PERSONS` and `MANAGE_USERS`
    already sit on, and the supplier's debug login does not stand on it.
    """
    assert Capability.REFER_TO_PSYCHOLOGIST not in capabilities_for(UserRole.DEVELOPER)

    event_id = an_event(camera)
    client = client_for(UserRole.DEVELOPER)
    response = client.post(f"/events/{event_id}/refer", data=with_token(client))

    assert response.status_code == 403
    assert _event_row(session, event_id).referred_at is None


def test_a_canteen_worker_may_not_refer_a_child(
    client_for: ClientFor, camera: Camera
) -> None:
    """§14: столовая — БЕЗ доступа к буллингу, and a referral is about a bullying event."""
    event_id = an_event(camera)
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.post(f"/events/{event_id}/refer", data=with_token(client)).status_code == 403


def test_the_psychologist_may_not_refer_a_child_to_themselves(
    client_for: ClientFor, camera: Camera
) -> None:
    """They are the recipient of a handover, not its author -- and they do not hold
    VIEW_BULLYING either, so the log a referral is made from is shut to them."""
    assert Capability.REFER_TO_PSYCHOLOGIST not in capabilities_for(UserRole.PSYCHOLOGIST)

    event_id = an_event(camera)
    client = client_for(UserRole.PSYCHOLOGIST)

    assert client.post(f"/events/{event_id}/refer", data=with_token(client)).status_code == 403
    assert client.get("/events").status_code == 403


def test_the_button_is_drawn_only_for_the_role_that_may_press_it(
    client_for: ClientFor, camera: Camera
) -> None:
    """A control that 403s is reported by the school as a broken system."""
    an_event(camera)

    assert "/refer" in client_for(UserRole.OPERATOR).get("/events").text
    assert "/refer" not in client_for(UserRole.DEVELOPER).get("/events").text

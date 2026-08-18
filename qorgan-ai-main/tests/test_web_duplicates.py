"""Six pairs of ids that are one human, and the school's decision about them.

`qorgan pupils gallery-report` finds them and refuses to resolve them: which id is
canonical is a decision only the school can make, and `7-А 438/439` may be identical twins
(`identity/report.py`, `identity/merge.py`). This page is where that decision gets made,
so everything here is about making it hard to make it wrongly and quietly:

  * **The photographs are behind `VIEW_PUPIL_PHOTOS`,** which is not the capability that
    opens the register. Reading who is enrolled twice and looking at two children's faces
    are different questions, and `/media` decides on the RESOLVED path, not the URL.

  * **Merging has its own capability, `MERGE_PERSONS`.** It is not a view. A merge
    re-points meal sessions and retires an id; on `student_470 / staff_334` it decides
    whether a child is FED, because staff never open a meal session and nothing reports
    the loss -- the number that would have said so is the number that stops being produced.

  * **The merge is reversible from the page that made it.** `--reactivate` exists because
    a merge across the pupil/staff line was a one-way door: the summary said "merge back
    the other way" and merging back was refused. Its two refusals -- "not retired by a
    merge" and "merged into somebody else" -- must arrive as readable sentences, not as a
    500, because a refusal the operator cannot read is a refusal they will work around.

Every merge below goes through the HTTP ROUTE. A test that called `merge_persons` directly
would prove the domain reversible and say nothing about the page, which is the layer that
can quietly pass `reactivate=False`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import Person, User
from qorgan.enums import PersonType, UserRole
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.identity_merge_fakes import face, same_face
from tests.web_login import with_token
from tests.web_pupil_fakes import enrol, meal

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]

PHOTO_SRC = re.compile(r'src="(/media/people/[^"]+)"')


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
            assert (
                client.post(
                    "/login",
                    data=with_token(client, {"username": username, "password": PASSWORD}),
                ).status_code
                == 303
            ), "login failed"
            return client

        yield make


@pytest.fixture
def admin(client_for: ClientFor) -> TestClient:
    """The role that may actually decide. See `roles.py`: ADMIN is the first role to hold
    something an operator does not, and this is it."""
    return client_for(UserRole.ADMIN)


@pytest.fixture
def pair(session: Session) -> tuple[Person, Person, float]:
    """Two ids, one human -- the shape of all six of this school's pairs.

    The cosine is computed here rather than assumed, so a test that asserts the score is
    on the page is comparing against the vectors, not against the page's own arithmetic.
    """
    first = face(11)
    second = same_face(first, 12)
    a = enrol(session, "student_470", class_name="7-А", vector=first)
    b = enrol(session, "student_471", class_name="7-А", vector=second)
    return a, b, float(np.dot(first, second))


@pytest.fixture
def crossing_pair(session: Session) -> tuple[Person, Person]:
    """`student_470 / staff_334`: the one pair of the six that crosses the pupil/staff
    line, which is the pair where the decision decides whether a child eats."""
    first = face(21)
    second = same_face(first, 22)
    pupil = enrol(session, "student_470", class_name="7-А", vector=first)
    staff = enrol(
        session,
        "staff_334",
        class_name=None,
        person_type=PersonType.STAFF,
        position="охранник",
        vector=second,
    )
    return pupil, staff


def _merge(client: TestClient, keep: Person, drop: Person):
    return client.post(
        "/pupils/duplicates/merge",
        data=with_token(client, {"keep_id": keep.id, "drop_id": drop.id}),
    )


def _undo(client: TestClient, keep: Person, drop: Person):
    return client.post(
        "/pupils/duplicates/undo",
        data=with_token(client, {"keep_id": keep.id, "drop_id": drop.id}),
    )


def _reload(session: Session, person: Person) -> Person:
    session.expire_all()
    return session.get(Person, person.id)


# -- the capability is new, and it is not a view ------------------------------


def test_merging_is_its_own_capability() -> None:
    """A view capability may not be reused for a mutation of a child's identity."""
    assert Capability.MERGE_PERSONS is not Capability.VIEW_PUPILS
    assert Capability.MERGE_PERSONS is not Capability.VIEW_PUPIL_PHOTOS


def test_an_operator_reads_the_pairs_but_cannot_resolve_them(
    client_for: ClientFor, pair: tuple[Person, Person, float]
) -> None:
    """Preparing the decision and making it are different rights. An operator sees the six
    pairs; the school's admin is the account that speaks for the school."""
    a, b, _ = pair
    operator = client_for(UserRole.OPERATOR)

    assert operator.get("/pupils/duplicates").status_code == 200
    assert _merge(operator, a, b).status_code == 403


def test_a_developer_cannot_resolve_them_either(
    client_for: ClientFor, pair: tuple[Person, Person, float]
) -> None:
    """A developer account exists to debug the system, not to make a claim about who a
    child is. Nothing they need is lost -- the pairs are still readable."""
    a, b, _ = pair

    assert Capability.MERGE_PERSONS not in capabilities_for(UserRole.DEVELOPER)
    assert _merge(client_for(UserRole.DEVELOPER), a, b).status_code == 403


def test_a_canteen_worker_reaches_none_of_it(client_for: ClientFor) -> None:
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get("/pupils/duplicates").status_code == 403


def test_the_button_is_not_drawn_for_a_role_that_cannot_press_it(
    client_for: ClientFor, pair: tuple[Person, Person, float]
) -> None:
    """Drawn from the same table the route is gated on. A control that 403s is a control
    the school reports as broken."""
    body = client_for(UserRole.OPERATOR).get("/pupils/duplicates").text

    assert "/pupils/duplicates/merge" not in body


# -- what the page has to show -----------------------------------------------


def test_the_pair_shows_both_sides_the_score_the_class_and_the_type(
    admin: TestClient, pair: tuple[Person, Person, float]
) -> None:
    a, b, similarity = pair

    body = admin.get("/pupils/duplicates").text

    assert a.external_id in body
    assert b.external_id in body
    assert f"{similarity:.3f}" in body, "the similarity score is not on the page"
    assert "7-А" in body
    assert "student" in body


def test_both_photographs_are_on_the_page(
    admin: TestClient, pair: tuple[Person, Person, float]
) -> None:
    """Side by side, because this decision is made by LOOKING. Arithmetic said 0.99 and
    arithmetic cannot tell identical twins from a duplicate enrolment."""
    body = admin.get("/pupils/duplicates").text

    assert len(PHOTO_SRC.findall(body)) == 2, "both photographs must be shown"


def test_the_photographs_are_behind_the_photograph_capability(
    admin: TestClient,
    client_for: ClientFor,
    pair: tuple[Person, Person, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Sabotage target.** The photograph URLs this page renders must be served only to a
    grant that names `VIEW_PUPIL_PHOTOS` -- both directions asserted, because a check that
    only asserts the refusal stays green when the whole area is deleted from the map and
    everybody is refused.
    """
    url = PHOTO_SRC.findall(admin.get("/pupils/duplicates").text)[0]

    assert admin.get(url).status_code == 200, "the admin cannot see the photograph they must judge"

    # The register without the photographs: a real role granted exactly that, because
    # `roles.py` refuses to invent a role for a job nobody does yet.
    monkeypatch.setitem(
        ROLE_CAPABILITIES,
        UserRole.DEVELOPER,
        frozenset({Capability.VIEW_PUPILS, Capability.VIEW_CANTEEN}),
    )

    assert client_for(UserRole.DEVELOPER).get(url).status_code == 403, (
        "a grant that never named VIEW_PUPIL_PHOTOS fetched a child's photograph"
    )


def test_a_viewer_without_the_photographs_gets_the_page_and_is_told_why(
    client_for: ClientFor, pair: tuple[Person, Person, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken-image icon is not an explanation. If the photographs are not drawn, the
    page has to say that they were withheld rather than that there are none."""
    monkeypatch.setitem(
        ROLE_CAPABILITIES,
        UserRole.DEVELOPER,
        frozenset({Capability.VIEW_PUPILS, Capability.VIEW_CANTEEN}),
    )

    body = client_for(UserRole.DEVELOPER).get("/pupils/duplicates").text

    assert PHOTO_SRC.findall(body) == []
    assert "Фотографии скрыты" in body


def test_a_name_on_this_page_cannot_inject_script(admin: TestClient, session: Session) -> None:
    """The legacy built its DOM with innerHTML from server JSON (audit H-05)."""
    first = face(31)
    enrol(session, "student_480", full_name='<img src=x onerror="alert(1)">', vector=first)
    enrol(session, "student_481", vector=same_face(first, 32))

    body = admin.get("/pupils/duplicates").text

    assert "<img src=x onerror=" not in body
    assert "&lt;img src=x onerror=" in body


def test_a_pair_crossing_the_pupil_staff_line_says_so(
    admin: TestClient, crossing_pair: tuple[Person, Person]
) -> None:
    """The stakes on this one pair are not bookkeeping: staff never open a meal session,
    so keeping the staff id for a child removes them from the meal record, and nothing
    reports it."""
    body = admin.get("/pupils/duplicates").text

    assert "ученик/сотрудник" in body, "the pupil/staff crossing was not called out"


# -- a GET renders, and merges nobody -----------------------------------------


def test_opening_the_page_merges_nobody(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    """No side effects on page load. The legacy restarted the AI workers on tab open."""
    a, b, _ = pair

    admin.get(f"/pupils/duplicates?keep_id={a.id}&drop_id={b.id}")
    admin.get("/pupils/duplicates")

    assert _reload(session, a).is_active is True
    assert _reload(session, b).is_active is True


# -- merging through the route ------------------------------------------------


def test_the_school_can_merge_from_the_page(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    a, b, _ = pair

    response = _merge(admin, a, b)

    assert response.status_code == 303
    assert _reload(session, b).is_active is False, "the dropped id was not retired"
    assert _reload(session, b).merged_into_id == a.id, "nothing records WHY it is inactive"
    assert _reload(session, a).is_active is True


def test_a_completed_merge_is_listed_with_its_consequence(
    admin: TestClient, crossing_pair: tuple[Person, Person], session: Session
) -> None:
    """The warning does not flash past once. It stays on the page for as long as the merge
    stands, because the person who has to notice it may not be the person who clicked."""
    pupil, staff = crossing_pair
    _merge(admin, staff, pupil)

    body = admin.get("/pupils/duplicates").text

    assert "staff_334" in body
    assert "student_470" in body
    assert "ученик/сотрудник" in body


def test_the_dropped_ids_meals_follow_it_onto_the_kept_id(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    """**Why the six pairs matter at all.** Their meals are SPLIT across two ids, so the
    school's canteen record is already wrong for these six people. A merge that retired an
    id and left its meal sessions behind would make the record worse, not better -- and it
    is the registry's own history page that has to show them arriving.
    """
    a, b, _ = pair
    meal(session, b.id)

    _merge(admin, a, b)

    history = admin.get(f"/pupils/{a.id}/canteen")
    assert history.status_code == 200
    assert "Сессии в столовой" in history.text
    assert "(1)" in history.text, "the dropped id's meal did not follow it onto the kept id"


def test_merging_somebody_into_themselves_is_a_readable_refusal(
    admin: TestClient, pair: tuple[Person, Person, float]
) -> None:
    a, _b, _ = pair

    response = _merge(admin, a, a)

    assert response.status_code == 400
    assert "itself" in response.text


def test_merging_an_id_nobody_holds_is_a_readable_refusal(
    admin: TestClient, pair: tuple[Person, Person, float]
) -> None:
    a, _b, _ = pair

    response = admin.post(
        "/pupils/duplicates/merge",
        data=with_token(admin, {"keep_id": a.id, "drop_id": 9999}),
    )

    assert response.status_code == 400
    assert "9999" in response.text, "the refusal did not name the id it could not find"


# -- the undo, through the route ----------------------------------------------


def test_a_merge_made_from_the_page_can_be_undone_from_the_page(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    """**Sabotage target, and the reason `--reactivate` exists.**

    A merge across the pupil/staff line drops a child out of the meal record entirely, and
    `MergeResult.summary()` says "merge back the other way" -- which was itself refused,
    because the id you would merge back into is the one the merge retired. That made a
    decision the school makes by looking at a photograph a ONE-WAY DOOR.

    Through the ROUTE, both ways. The domain being reversible proves nothing about a
    handler that forgets to pass `reactivate=True`.
    """
    a, b, _ = pair
    assert _merge(admin, a, b).status_code == 303

    response = _undo(admin, b, a)

    assert response.status_code == 303, "a web-initiated merge could not be undone from the web"
    assert _reload(session, b).is_active is True, "the retired id was not revived"
    assert _reload(session, b).merged_into_id is None, "it still claims to be a merged-away id"
    assert _reload(session, a).is_active is False
    assert _reload(session, a).merged_into_id == b.id


def test_the_undo_needs_the_merge_capability_too(
    admin: TestClient,
    client_for: ClientFor,
    pair: tuple[Person, Person, float],
    session: Session,
) -> None:
    """Undoing is as much a claim about who a child is as merging was."""
    a, b, _ = pair
    _merge(admin, a, b)

    assert _undo(client_for(UserRole.OPERATOR), b, a).status_code == 403
    assert _reload(session, b).is_active is False


def test_reviving_somebody_no_merge_retired_is_refused_in_words(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    """`is_active=False` also means "left the school". Reactivating one of those is a new
    claim about who is enrolled, not a correction -- and the operator has to be able to
    READ that, or they will try again with different ids until something works."""
    a, b, _ = pair
    left_the_school = _reload(session, b)
    left_the_school.is_active = False
    left_the_school.merged_into_id = None
    session.commit()

    response = _undo(admin, b, a)

    assert response.status_code == 400
    assert "not retired by a merge" in response.text
    assert _reload(session, b).is_active is False, "the refusal did not actually refuse"


def test_undoing_against_the_wrong_partner_names_the_real_one(
    admin: TestClient, pair: tuple[Person, Person, float], session: Session
) -> None:
    """The operator is holding external ids, so the refusal names the person -- "person id
    135" sends them to the database to find out who they were just told about."""
    a, b, _ = pair
    stranger = enrol(session, "student_999", class_name="7-А")
    _merge(admin, a, b)

    response = _undo(admin, b, stranger)

    assert response.status_code == 400
    assert "student_470" in response.text, "the refusal did not name who b was merged into"
    assert _reload(session, b).is_active is False


def test_a_refusal_still_renders_the_page(
    admin: TestClient, pair: tuple[Person, Person, float]
) -> None:
    """A refusal is a page with an explanation on it, not a stack trace. The operator must
    be able to look at the same six pairs while reading why this one was refused."""
    a, _b, _ = pair

    response = _merge(admin, a, a)

    assert "student_470" in response.text
    assert response.headers["content-type"].startswith("text/html")

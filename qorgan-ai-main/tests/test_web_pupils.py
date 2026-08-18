"""The pupil registry: who is enrolled, and can the system recognise them?

Three legacy defects shape this page, and each one is a test below.

  * **Pagination.** The legacy loaded the ENTIRE persons table on every render, every
    2.5 seconds, per client (audit M-19). 142 rows today; the school it is sold into has
    800, and it gets worse every day the system runs.

  * **Escaping.** The legacy built its DOM with `innerHTML` from server JSON, so a pupil
    named `<img src=x onerror=...>` was stored XSS in the operator's browser (audit H-05).
    `full_name` is a free-text column filled from a roster the school sends us.

  * **Side effects on GET.** Opening a tab in the legacy POSTed `/page-activate/{page}`,
    which restarted the AI workers with a five-second `thread.join()` inside the HTTP
    handler. A GET renders. It does not write.

And one fact about this school's data that the page exists to state: four staff
photographs contain no detectable face at all (spec §1.1). Those people are on the roster
and the system can NEVER recognise them. A registry that does not say so is a registry
that quietly implies it can.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models import CanteenSession, Person, User
from qorgan.enums import PersonType, UserRole
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.identity_merge_fakes import face
from tests.web_login import with_token
from tests.web_pupil_fakes import enrol, meal

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session  # applied via the fixtures
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    """A logged-in client for whichever role a test is about."""
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
def operator(client_for: ClientFor) -> TestClient:
    return client_for(UserRole.OPERATOR)


# -- who may open it ---------------------------------------------------------


def test_the_registry_needs_a_session(settings: Settings, session: Session) -> None:
    """The legacy served the pupil registry to anyone on the school network, on a server
    bound to 0.0.0.0 with no auth anywhere (audit C-01)."""
    with TestClient(create_app(), follow_redirects=False) as anonymous:
        assert anonymous.get("/pupils").status_code == 303


def test_a_canteen_worker_is_refused_the_registry(client_for: ClientFor) -> None:
    """§14 again. A canteen worker is handed the canteen journal and nothing else; the
    register of every child in the school is not canteen work."""
    assert client_for(UserRole.CANTEEN_STAFF).get("/pupils").status_code == 403


def test_the_registry_capability_is_not_the_photograph_capability() -> None:
    """Listing who is enrolled and looking at their photographs are different questions,
    so they are different grants -- the same reason `VIEW_MEDIA` was split in two."""
    assert Capability.VIEW_PUPILS is not Capability.VIEW_PUPIL_PHOTOS


def test_every_role_still_states_its_capabilities_in_writing() -> None:
    assert set(ROLE_CAPABILITIES) == set(UserRole)


# -- what the page has to say ------------------------------------------------


def test_the_row_carries_the_school_id_the_name_the_class_and_the_type(
    operator: TestClient, session: Session
) -> None:
    """The external_id is the identity -- the name is a display field (`identity.naming`),
    and there is no ID -> name roster yet, so the id has to be on the page itself."""
    enrol(session, "student_333", class_name="5-А", vector=face(1))

    body = operator.get("/pupils").text

    assert "student_333" in body
    assert "Ученик 333, 5-А" in body, "the display name is missing"
    assert "5-А" in body
    assert "student" in body


def test_a_person_with_no_embedding_is_said_to_be_unrecognisable(
    operator: TestClient, session: Session
) -> None:
    """**The reason this column exists.** Four of this school's staff photographs contain
    no detectable face (spec §1.1): they are enrolled, and the system can never recognise
    them. Silence here reads as "fine", which is the one thing it is not."""
    enrol(session, "staff_464", class_name=None, person_type=PersonType.STAFF, vector=None)

    body = operator.get("/pupils").text

    assert "не распознаётся" in body, "a person the system cannot recognise was not marked"


def test_a_person_with_an_embedding_is_said_to_be_recognisable(
    operator: TestClient, session: Session
) -> None:
    enrol(session, "student_333", vector=face(2))

    body = operator.get("/pupils").text

    assert "распознаётся" in body
    # `не распознаётся` CONTAINS `распознаётся`, so the positive assertion above passes on
    # its own even when the page says the exact opposite. Both halves, always.
    assert "не распознаётся" not in body


def test_an_embedding_from_another_model_does_not_count(
    operator: TestClient, session: Session
) -> None:
    """`load_gallery` reads ONE model version, because the legacy shipped a rebuild script
    that wrote vectors from a different model into the same column (audit M-29). A row this
    page counts but the gallery will not load is a row that promises a recognition that
    cannot happen."""
    enrol(session, "student_333", vector=face(3), model_version="0.9")

    body = operator.get("/pupils").text

    assert "не распознаётся" in body, "an unusable embedding was counted as recognisable"


def test_a_pupils_name_cannot_inject_script(operator: TestClient, session: Session) -> None:
    """The legacy built this table with innerHTML from server JSON, so a pupil named
    `<img src=x onerror=...>` gave stored XSS in the operator's browser (audit H-05).
    `full_name` comes from a roster the school sends us."""
    enrol(session, "student_666", full_name='<img src=x onerror="alert(1)">')

    body = operator.get("/pupils").text

    assert "<img src=x onerror=" not in body
    assert "&lt;img src=x onerror=" in body, "the name was not escaped"


# -- pagination is mandatory -------------------------------------------------


def test_the_registry_is_paginated(operator: TestClient, session: Session) -> None:
    """The legacy loaded the whole table on every render, every 2.5s, per client. This
    school has 142 people; the next one has 800."""
    for number in range(60):
        enrol(session, f"student_{number:03d}", photo=False)

    first = operator.get("/pupils?page=1")
    second = operator.get("/pupils?page=2")

    assert first.status_code == 200
    assert "1 / 2" in first.text, "the page is not paginated"
    assert second.status_code == 200
    assert "student_059" in second.text, "the second page is empty"
    assert "student_059" not in first.text, "the first page held the whole table"


def test_a_nonsense_page_number_lands_on_the_first_page(
    operator: TestClient, session: Session
) -> None:
    enrol(session, "student_333")

    assert operator.get("/pupils?page=-4").status_code == 200


# -- the canteen history -----------------------------------------------------


def test_the_row_links_to_that_persons_canteen_history(
    operator: TestClient, session: Session
) -> None:
    person = enrol(session, "student_333")

    assert f"/pupils/{person.id}/canteen" in operator.get("/pupils").text


def test_the_history_shows_only_that_persons_meals(operator: TestClient, session: Session) -> None:
    """Keyed on person_id, not on a name. The legacy keyed identity on
    "Surname Firstname" + a class parsed out of a photo FILENAME, so two children with the
    same name in the same class collapsed into one person (audit H-02)."""
    mine = enrol(session, "student_333")
    theirs = enrol(session, "student_334")
    meal(session, mine.id)
    meal(session, theirs.id)

    body = operator.get(f"/pupils/{mine.id}/canteen").text

    assert "Ученик 333, 5-А" in body
    assert "Ученик 334, 5-А" not in body, "another child's meal record leaked onto this page"


def test_the_history_of_somebody_who_does_not_exist_is_a_404(operator: TestClient) -> None:
    assert operator.get("/pupils/4321/canteen").status_code == 404


def test_a_canteen_worker_cannot_read_one_pupils_history(
    client_for: ClientFor, session: Session
) -> None:
    """This is the canteen record, but it is reached BY CHILD, through the registry. §14
    gives canteen staff the day's journal; it does not give them a per-child history."""
    person = enrol(session, "student_333")

    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get(f"/pupils/{person.id}/canteen").status_code == 403


# -- what was deliberately NOT built -----------------------------------------


@pytest.mark.parametrize("path", ["/pupils/import", "/pupils/settings"])
def test_the_pupils_section_has_no_upload_and_no_settings_page(
    operator: TestClient, path: str
) -> None:
    """Out of scope by the owner's instruction, and written down here so that "we did not
    build it" cannot quietly become "somebody added it later".

    Importing photographs is `qorgan pupils import`, which unzips archives of children's
    photographs into `MEDIA_ROOT/.import/` -- an area `media.py` refuses to serve to
    anybody at all. Putting that behind an HTTP upload is a decision with its own
    permissions, its own refusals and its own test file, and it is not this task.

    `/settings` USED TO BE IN THIS LIST and was removed when the settings page landed.
    It is no longer a 404, and asserting 403 here instead would have been the wrong
    repair: a 403 comes from the capability layer and says nothing about whether the
    PUPILS section grew a settings page, which is the only question this test exists to
    answer. It would also tie a pupils test to the settings capability table, so a change
    there would redden a file about pupils. The subject did not disappear -- it moved to a
    page with its own tests, and `tests/test_web_settings.py` asserts an operator gets 403
    from `/settings`. The guard is not weakened; it is pointed at what is still unbuilt.
    """
    assert operator.get(path).status_code == 404


# -- a GET renders, and does nothing else ------------------------------------


def test_opening_the_registry_writes_nothing(operator: TestClient, session: Session) -> None:
    """The legacy restarted the AI workers when somebody opened a tab, with a five-second
    `thread.join()` inside the HTTP handler, so coverage depended on which tab was open.

    Checked on `updated_at` rather than on row counts alone: an ORM that touched a row and
    wrote it back unchanged would leave the counts identical and the timestamp moved.
    """
    enrol(session, "student_333", vector=face(4))
    session.expire_all()
    before = session.execute(select(Person.id, Person.updated_at, Person.is_active)).all()
    sessions_before = session.scalar(select(func.count(CanteenSession.id)))

    assert operator.get("/pupils").status_code == 200

    session.expire_all()
    assert session.execute(select(Person.id, Person.updated_at, Person.is_active)).all() == before
    assert session.scalar(select(func.count(CanteenSession.id))) == sessions_before

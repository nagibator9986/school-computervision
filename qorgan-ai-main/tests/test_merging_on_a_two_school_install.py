"""Merging two ids into one, on the installation the tenancy work was written for.

`test_tenancy_isolation.py` asks whether one school can READ another's data. This asks a
different question about the same page, and it is the one that was missed: **does the
feature still WORK once a second school exists?**

It was not working. `web/routes/duplicates.py` passed `school_id=` on both of its read
paths and not on the write, so `merge_persons` fell back to "the only school there is" --
which RAISES on two. That is a `RuntimeError`, the handler catches only `(LookupError,
ValueError)`, and a headteacher merging two ids **belonging to their own school** got a
500 in place of the readable refusal the whole module is built to produce.

Nothing caught it, and the reason is worth stating because it applies to the entire branch:
**every other test in this suite runs on ONE school, where the fallback happens to be
correct.** A fallback that is right in the only configuration anybody tests is invisible
until the configuration it is wrong in arrives -- which is exactly what a multi-school
branch is for. So the cross-school refusal below is only half of this file; the half that
matters more is the legitimate merge that must still succeed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Person, School, User
from qorgan.enums import PersonType, UserRole
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.conftest import DEFAULT_SCHOOL_SLUG
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
MERGE = "/pupils/duplicates/merge"


def _person(session: Session, school_id: int, external_id: str) -> Person:
    row = Person(
        school_id=school_id,
        external_id=external_id,
        full_name=f"Ученик {external_id}",
        person_type=PersonType.STUDENT,
        class_name="5-А",
    )
    session.add(row)
    session.flush()
    return row


@pytest.fixture
def two_schools(session: Session) -> tuple[int, int, int, int]:
    """Two schools; two pupils in ours to merge, one in theirs to be refused.

    Returns (keep, drop, theirs, ours_school_id).
    """
    ours = session.scalar(select(School.id).where(School.slug == DEFAULT_SCHOOL_SLUG))
    assert ours is not None
    other = School(slug="gymnasium-4", name="Гимназия №4")
    session.add(other)
    session.flush()

    keep = _person(session, int(ours), "a1")
    drop = _person(session, int(ours), "a2")
    theirs = _person(session, int(other.id), "b1")

    session.add(
        User(
            school_id=int(ours),
            username="head",
            password_hash=hash_password(PASSWORD),
            # ADMIN is the only role holding MERGE_PERSONS. Not the operator, not the
            # developer -- see `roles.py`.
            role=UserRole.ADMIN,
        )
    )
    session.commit()
    return keep.id, drop.id, theirs.id, int(ours)


@pytest.fixture
def client(settings: Settings, two_schools: tuple[int, int, int, int]) -> Iterator[TestClient]:
    with TestClient(create_app(), follow_redirects=False) as test_client:
        response = test_client.post(
            "/login", data=with_token(test_client, {"username": "head", "password": PASSWORD})
        )
        assert response.status_code == 303, "could not log in; nothing below would mean anything"
        yield test_client


def test_a_school_can_still_merge_its_own_two_ids_when_another_school_exists(
    client: TestClient, two_schools: tuple[int, int, int, int], session: Session
) -> None:
    """**The regression this file was written for.** A 500 here, not a leak.

    Both ids belong to the school making the request. There is nothing ambiguous about
    this merge and nothing to refuse -- it is the ordinary operation the page exists for,
    and the arrival of an unrelated second school had killed it.
    """
    keep_id, drop_id, _, _ = two_schools
    response = client.post(
        MERGE, data=with_token(client, {"keep_id": str(keep_id), "drop_id": str(drop_id)})
    )

    assert response.status_code == 303, (
        f"a headteacher could not merge two of their OWN school's ids: HTTP "
        f"{response.status_code}. 500 means the school fallback raised and the handler "
        "did not catch it -- the page is dead on a two-school installation."
    )
    dropped = session.get(Person, drop_id)
    session.refresh(dropped)
    assert dropped.is_active is False
    assert dropped.merged_into_id == keep_id


def test_merging_another_schools_pupil_is_refused_readably_and_not_with_a_500(
    client: TestClient, two_schools: tuple[int, int, int, int], session: Session
) -> None:
    """A cross-school merge must be refused the way every other refusal is: on the page.

    The status matters as much as the refusal. A 500 is not a safe failure here -- it is
    indistinguishable from the system being broken, and the operator's next move after an
    error page is to try again, or to ask somebody to "fix" the check.
    """
    keep_id, _, theirs_id, _ = two_schools
    response = client.post(
        MERGE, data=with_token(client, {"keep_id": str(keep_id), "drop_id": str(theirs_id)})
    )

    assert response.status_code == 400, (
        f"expected a readable refusal, got HTTP {response.status_code}"
    )
    theirs = session.get(Person, theirs_id)
    session.refresh(theirs)
    assert theirs.is_active is True, "another school's pupil was retired by this merge"
    assert theirs.merged_into_id is None


def test_the_other_schools_pupil_is_untouched_by_a_legitimate_merge(
    client: TestClient, two_schools: tuple[int, int, int, int], session: Session
) -> None:
    """The control: a merge that succeeds must not reach across while doing it."""
    keep_id, drop_id, theirs_id, _ = two_schools
    client.post(MERGE, data=with_token(client, {"keep_id": str(keep_id), "drop_id": str(drop_id)}))

    theirs = session.get(Person, theirs_id)
    session.refresh(theirs)
    assert theirs.is_active is True
    assert theirs.merged_into_id is None

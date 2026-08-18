"""Merging two ids that are one human. Never automatic; always a decision a human made."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import RecognitionPolicy
from qorgan.db.models import CanteenSession, FaceEmbedding, Person, PersonPhoto
from qorgan.enums import PersonType
from qorgan.faces.gallery import load_gallery
from qorgan.faces.matching import Reason, identify
from qorgan.identity.merge import merge_persons, resolve_external
from qorgan.settings import Settings
from tests.identity_merge_fakes import MODEL_NAME, MODEL_VERSION
from tests.identity_merge_fakes import face as _face
from tests.identity_merge_fakes import open_session as _open_session
from tests.identity_merge_fakes import person as _person
from tests.identity_merge_fakes import same_face as _same_face

# -- the merge ---------------------------------------------------------------


def test_photos_embeddings_and_sessions_all_re_point(settings: Settings, session: Session) -> None:
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    _open_session(session, drop.id)

    result = merge_persons(keep.id, drop.id)

    assert result.photos_moved == 1
    assert result.embeddings_moved == 1
    assert result.sessions_moved == 1

    session.expire_all()
    assert session.scalars(select(PersonPhoto)).all()[0].person_id == keep.id
    assert {e.person_id for e in session.scalars(select(FaceEmbedding))} == {keep.id}
    assert session.scalars(select(CanteenSession)).one().person_id == keep.id


def test_the_dropped_id_is_deactivated_not_deleted(settings: Settings, session: Session) -> None:
    """The id existed. The school issued it, and a record that says so is worth keeping."""
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    merge_persons(keep.id, drop.id)

    session.expire_all()
    assert session.get(Person, drop.id) is not None
    assert session.get(Person, drop.id).is_active is False
    assert session.get(Person, keep.id).is_active is True


def test_merging_a_person_into_themselves_is_refused(settings: Settings, session: Session) -> None:
    keep = _person(session, "staff_334", _face(1))

    with pytest.raises(ValueError, match="itself"):
        merge_persons(keep.id, keep.id)


def test_merging_someone_who_does_not_exist_is_refused(
    settings: Settings, session: Session
) -> None:
    keep = _person(session, "staff_334", _face(1))

    with pytest.raises(LookupError, match="9999"):
        merge_persons(keep.id, 9999)


def test_an_external_id_resolves_to_a_person(settings: Settings, session: Session) -> None:
    keep = _person(session, "staff_334", _face(1))

    assert resolve_external("staff_334") == keep.id

    with pytest.raises(LookupError, match="student_999"):
        resolve_external("student_999")


def test_merging_twice_moves_nothing_the_second_time(settings: Settings, session: Session) -> None:
    """Idempotent. The second run finds nothing left to move and doubles nothing."""
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))
    _open_session(session, drop.id)

    first = merge_persons(keep.id, drop.id)
    second = merge_persons(keep.id, drop.id)

    assert (first.photos_moved, first.embeddings_moved, first.sessions_moved) == (1, 1, 1)
    assert (second.photos_moved, second.embeddings_moved, second.sessions_moved) == (0, 0, 0)

    session.expire_all()
    assert len(session.scalars(select(PersonPhoto)).all()) == 2  # keep's own, plus drop's
    assert len(session.scalars(select(FaceEmbedding)).all()) == 2
    assert len(session.scalars(select(CanteenSession)).all()) == 1


def test_keeping_an_id_that_was_already_retired_is_refused(
    settings: Settings, session: Session
) -> None:
    """Merging a live pupil INTO a retired id would take the live one out of the gallery.

    `load_gallery` reads active people only, so an inactive `keep_id` swallows every photo
    and embedding and then shows none of them. The person does not become hard to
    recognise; they stop existing. That is the erasure this whole module is built to
    prevent, so it is refused rather than performed.
    """
    him = _face(1)
    retired = _person(session, "student_470", him)
    live = _person(session, "staff_334", _same_face(him, 2))

    merge_persons(live.id, retired.id)  # retired is now is_active=False

    with pytest.raises(ValueError, match="inactive"):
        merge_persons(retired.id, live.id)

    session.expire_all()
    assert session.get(Person, live.id).is_active is True


def test_the_merge_is_recorded_with_who_was_merged_into_whom(
    settings: Settings, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    with caplog.at_level("WARNING"):
        merge_persons(keep.id, drop.id)

    merged = [r for r in caplog.records if "merged" in r.getMessage()]
    assert merged, "the merge left no record of itself"
    assert merged[0].keep == "staff_334"
    assert merged[0].drop == "student_470"


# -- crossing the pupil/staff line decides whether a person is fed ------------


def test_crossing_the_pupil_staff_line_is_reported_loudly(
    settings: Settings, session: Session
) -> None:
    """**Staff never open a meal session** (`gallery.PersonInfo.is_staff`).

    Two of the six duplicate pairs cross this line -- `staff_464 / учитель_477` and
    `student_470 / staff_334`. So which id is kept is not bookkeeping: it decides whether
    that person is fed. Keep the staff row for someone who is really a pupil in 11-А and
    the child silently drops out of the canteen record -- nobody is told, and the number
    that would have said so is the one that stops being produced.

    The merge cannot know which way round is right. It can refuse to let it pass unseen.
    """
    him = _face(1)
    keep = _person(session, "staff_334", him, person_type=PersonType.STAFF)
    drop = _person(session, "student_470", _same_face(him, 2), person_type=PersonType.STUDENT)

    result = merge_persons(keep.id, drop.id)

    assert result.crosses_person_type is True
    summary = result.summary()
    assert "staff" in summary.lower()
    assert "meal session" in summary.lower()


def test_a_merge_within_one_person_type_says_nothing_about_meals(
    settings: Settings, session: Session
) -> None:
    """The warning must MEAN something -- so it must be absent when it does not apply."""
    him = _face(1)
    keep = _person(session, "student_371", him)
    drop = _person(session, "student_472", _same_face(him, 2))

    result = merge_persons(keep.id, drop.id)

    assert result.crosses_person_type is False
    assert "meal session" not in result.summary().lower()


# -- THE regression test: the gap collapse, pinned (spec §2.5, §6) ------------


def test_a_human_under_two_ids_is_ambiguous_and_a_merge_makes_him_visible(
    settings: Settings, session: Session
) -> None:
    """**This is §2.5 turned into a regression test, so the failure cannot return
    silently.**

    The top hall matches were id=334 and id=470 — the duplicate pair. Both are the same
    human, so top-1 and top-2 are BOTH him, and the gap collapses:

        40px  id=334  score 0.604   gap +0.001   ->  after merge  +0.413

    min_gap is 0.05. **He is rejected as AMBIGUOUS every time.** This is the 1816-NULL
    mechanism alive in this data — and note that ranking by PERSON (last session's fix)
    cannot help, because the two rows genuinely ARE different persons in the database.

    For those six children the system is not inaccurate. It is BLIND, and no threshold
    anywhere would have shown why.
    """
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2, strength=0.999))
    _person(session, "student_333", _face(9))  # somebody else entirely

    policy = RecognitionPolicy()  # min_score 0.50, min_gap 0.05

    before = load_gallery(MODEL_NAME, MODEL_VERSION)
    rejected = identify(him, before.matrix, before.person_ids, policy)

    assert not rejected.accepted, "the duplicate did not collapse the gap; the test is broken"
    assert rejected.reason is Reason.AMBIGUOUS
    assert rejected.gap < policy.min_gap
    assert {r.person_id for r in rejected.ranked[:2]} == {keep.id, drop.id}

    merge_persons(keep.id, drop.id)

    after = load_gallery(MODEL_NAME, MODEL_VERSION)
    accepted = identify(him, after.matrix, after.person_ids, policy)

    assert accepted.accepted, "merging did not make him visible again"
    assert accepted.person_id == keep.id
    assert accepted.reason is Reason.ACCEPTED
    assert accepted.gap > policy.min_gap


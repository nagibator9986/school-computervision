"""Undoing a merge that went the wrong way.

Split out of `test_identity_merge.py` at the 500-line cap. These are a different subject
from "does a merge move the rows": they are about the fact that the merge which decides
whether a child appears in the meal record was, until this session, a one-way door.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Person
from qorgan.enums import PersonType
from qorgan.identity.merge import merge_persons
from qorgan.settings import Settings
from tests.identity_merge_fakes import face as _face
from tests.identity_merge_fakes import open_session as _open_session
from tests.identity_merge_fakes import person as _person
from tests.identity_merge_fakes import same_face as _same_face

# -- undoing a merge that went the wrong way ---------------------------------
#
# Rehearsed end-to-end on the real roster (2026-07-24, throwaway database): merging the
# 11-А child onto `staff_334` moves their three meal sessions and then removes them from
# the canteen roster entirely -- `day_report` selects `person_type == STUDENT AND
# is_active`, and after the merge neither id satisfies it. `MergeResult.summary()` warns
# about exactly this and tells the operator to "merge back the other way". That instruction
# could not be followed: the dropped id is deactivated, and merging INTO an inactive id is
# refused -- by a message that asks "Did you mean to merge the other way round?", which is
# the thing the operator just did. There was no reactivate command anywhere in the CLI.
#
# So the highest-stakes operation in the system -- the one that decides whether a child is
# fed -- was a one-way door, on a decision the school makes from a photograph.


def test_a_merge_that_went_the_wrong_way_can_be_undone(
    settings: Settings, session: Session
) -> None:
    """The remedy the warning names must exist, and must put everything back.

    Not a convenience. `crosses_person_type` fires on a judgement about a child, made by
    someone reading a similarity score, and a wrong answer silently drops that child out of
    the meal record. A decision that reversible on paper must be reversible in the tool.
    """
    him = _face(1)
    child = _person(session, "student_470", him)
    staff = _person(session, "staff_334", _same_face(him, 2), person_type=PersonType.STAFF)
    _open_session(session, child.id)

    merge_persons(staff.id, child.id)  # the wrong way round

    back = merge_persons(child.id, staff.id, reactivate=True)

    session.expire_all()
    assert session.get(Person, child.id).is_active is True, "the child is still retired"
    assert session.get(Person, staff.id).is_active is False
    assert back.photos_moved == 2, "both photos should have come back"
    assert back.embeddings_moved == 2
    assert back.sessions_moved == 1, "the child's meal history did not come back"


def test_the_child_returns_to_the_canteen_roster_after_the_undo(
    settings: Settings, session: Session
) -> None:
    """The measurable consequence, not the flags. `day_report`'s roster is
    `person_type == STUDENT AND is_active`; the whole point of undoing is that the child
    is countable again."""
    him = _face(1)
    child = _person(session, "student_470", him)
    staff = _person(session, "staff_334", _same_face(him, 2), person_type=PersonType.STAFF)

    merge_persons(staff.id, child.id)
    session.expire_all()
    assert _on_canteen_roster(session, child.id) is False, "premise moved: still on the roster"

    merge_persons(child.id, staff.id, reactivate=True)

    session.expire_all()
    assert _on_canteen_roster(session, child.id) is True, (
        "the child is still absent from the roster the 'did not eat' report is built from"
    )


def _on_canteen_roster(session: Session, person_id: int) -> bool:
    """Exactly `canteen.reports.day_report`'s roster predicate, not a paraphrase."""
    return session.scalar(
        select(Person.id).where(
            Person.id == person_id,
            Person.person_type == PersonType.STUDENT,
            Person.is_active.is_(True),
        )
    ) is not None


def test_reviving_a_retired_id_stays_refused_unless_it_is_asked_for(
    settings: Settings, session: Session
) -> None:
    """The guard is not weakened. `is_active=False` also means "left the school", and a
    merge must never silently bring such a person back into the gallery. The flag is the
    human saying which of the two meanings applies."""
    him = _face(1)
    retired = _person(session, "student_470", him)
    live = _person(session, "staff_334", _same_face(him, 2))

    merge_persons(live.id, retired.id)

    with pytest.raises(ValueError, match="inactive"):
        merge_persons(retired.id, live.id)  # no flag: still refused


def test_the_refusal_names_the_way_out(settings: Settings, session: Session) -> None:
    """The old message asked "Did you mean to merge the other way round?" -- which is what
    the operator had just done. A refusal that points back at the cause is how somebody
    concludes the child cannot be recovered."""
    him = _face(1)
    retired = _person(session, "student_470", him)
    live = _person(session, "staff_334", _same_face(him, 2))
    merge_persons(live.id, retired.id)

    with pytest.raises(ValueError, match="--reactivate"):
        merge_persons(retired.id, live.id)


def test_the_summary_says_when_an_id_was_brought_back(
    settings: Settings, session: Session
) -> None:
    """Reviving an id is a real change to who exists. It is never silent."""
    him = _face(1)
    child = _person(session, "student_470", him)
    staff = _person(session, "staff_334", _same_face(him, 2), person_type=PersonType.STAFF)
    merge_persons(staff.id, child.id)

    summary = merge_persons(child.id, staff.id, reactivate=True).summary()

    assert "student_470" in summary
    assert "reactivat" in summary.lower(), summary


# -- "inactive" answered two different questions -----------------------------
#
# `is_active=False` meant BOTH "this id was merged away" AND "this person left the
# school". `--reactivate` had to trust the operator about which, on the one operation
# where being wrong either revives an expelled pupil or refuses to give a child their
# meal record back. `persons.merged_into_id` is the discriminator: set when a merge
# retires an id, cleared when that merge is undone, NULL for anybody retired some other
# way.


def test_a_merge_records_where_the_dropped_id_went(settings: Settings, session: Session) -> None:
    him = _face(1)
    keep = _person(session, "staff_334", him)
    drop = _person(session, "student_470", _same_face(him, 2))

    merge_persons(keep.id, drop.id)

    session.expire_all()
    assert session.get(Person, drop.id).merged_into_id == keep.id
    assert session.get(Person, keep.id).merged_into_id is None, "the survivor went nowhere"


def test_undoing_a_merge_clears_the_pointer(settings: Settings, session: Session) -> None:
    """After the undo the child is a normal active person again, not one carrying a
    stale note that they live somewhere else."""
    him = _face(1)
    child = _person(session, "student_470", him)
    staff = _person(session, "staff_334", _same_face(him, 2), person_type=PersonType.STAFF)

    merge_persons(staff.id, child.id)
    merge_persons(child.id, staff.id, reactivate=True)

    session.expire_all()
    assert session.get(Person, child.id).merged_into_id is None
    assert session.get(Person, staff.id).merged_into_id == child.id


def test_reactivate_refuses_somebody_who_was_never_merged(
    settings: Settings, session: Session
) -> None:
    """**The reason the column exists.** An id inactive for any other reason — the pupil
    left the school — is not a merge to undo, and `--reactivate` used to revive them
    anyway, because nothing recorded the difference.
    """
    him = _face(1)
    gone = _person(session, "student_470", him)
    live = _person(session, "staff_334", _same_face(him, 2))

    gone.is_active = False  # left the school. No merge involved.
    session.commit()

    with pytest.raises(ValueError, match="not retired by a merge"):
        merge_persons(gone.id, live.id, reactivate=True)

    session.expire_all()
    assert session.get(Person, gone.id).is_active is False, "an expelled pupil was revived"


def test_reactivate_refuses_to_undo_a_merge_that_did_not_happen(
    settings: Settings, session: Session
) -> None:
    """`--reactivate` undoes THIS merge, not merges in general.

    If the kept id was retired into somebody else, reversing it against an unrelated
    person is not an undo — it is a new claim about who these people are, made by a flag
    whose whole justification is that it only reverses.
    """
    him = _face(1)
    child = _person(session, "student_470", him)
    staff = _person(session, "staff_334", _same_face(him, 2), person_type=PersonType.STAFF)
    other = _person(session, "student_402", _face(9))

    merge_persons(staff.id, child.id)  # child -> staff

    with pytest.raises(ValueError, match="staff_334"):
        merge_persons(child.id, other.id, reactivate=True)  # ...undone against a stranger

    session.expire_all()
    assert session.get(Person, child.id).is_active is False


def test_the_refusal_says_which_of_the_two_reasons_it_could_not_tell_apart(
    settings: Settings, session: Session
) -> None:
    """A row retired before this column existed also has NULL, and that is not the same
    as "left the school" — it is "unknown". The message must not assert the stronger
    thing, because whoever reads it is about to decide something about a child."""
    him = _face(1)
    gone = _person(session, "student_470", him)
    live = _person(session, "staff_334", _same_face(him, 2))
    gone.is_active = False
    session.commit()

    with pytest.raises(ValueError) as caught:
        merge_persons(gone.id, live.id, reactivate=True)

    message = str(caught.value)
    assert "student_470" in message
    assert "left the school" in message
    assert "before this column existed" in message, message

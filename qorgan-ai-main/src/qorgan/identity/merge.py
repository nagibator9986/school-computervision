"""Two ids, one human. **Never automatic.**

`gallery-report` detects six duplicate enrolments. It does not resolve them, and it must
not: which id is canonical is a decision only the school can make, and `7-А 438/439` --
adjacent ids, same class, both in the first enrolment batch, the lowest score of the six --
may be identical twins. Arithmetic cannot settle that, and a system that guessed would be
making up an identity, which is the one thing this module exists to stop.

So this command executes a decision a human already made. It re-points the photos, the
embeddings, the canteen sessions and the psychologist's notes from `drop_id` onto
`keep_id`, and deactivates `drop_id` -- it does not delete it, because the school issued
that id and a record saying so is worth keeping.

Measured effect (spec §2.7): the gap collapse is real -- 0.001 -> **0.413** after merge --
and it is a property of the GALLERY (one human enrolled twice sits in his own top-2), so it
holds on any camera at any resolution. The accompanying A/B (accepts 3 -> 9, gap-kills
6 -> 0) was measured on gallery faces >=38 px **at HD**, which is >=19 px at the hall's real
1280x720 -- below the production gate entirely. So that accept-count does NOT describe the
hall, where nothing is recognised regardless. Merging bites where faces are big enough to be
recognised at all: the CANTEEN.

It does not improve the system. It makes six specific people VISIBLE to it.

**ONE of the six pairs crosses the pupil/staff line: `student_470 / staff_334`.** (This
said TWO, counting `staff_464 / учитель_477`. Measured against `persons.person_type` on the
imported roster, 2026-07-24: BOTH of those are STAFF, because the folder decides the type
and `учитель` maps to STAFF -- `identity/roster.py::folder_role`. Neither of them opens a
meal session either way, so no answer there changes whether anybody is fed.) Staff never
open a meal session -- see `faces.gallery.PersonInfo.is_staff`. So on that one pair, which
id is kept is not bookkeeping: it decides whether that person is FED. Keep the staff
row for someone who is really a pupil and the child drops out of the canteen record
silently, because the number that would have reported it is the very number that stops
being produced. `MergeResult.summary()` says so
out loud; it cannot know which way round is right, but it can refuse to let it pass unseen.

**`RecognitionAttempt` is deliberately NOT re-pointed, and that is not an oversight.**

It is a LOG of what the matcher actually decided, not a record of who somebody is. Moving
`top1_person_id` from the dropped id to the kept one would rewrite history: "the matcher
chose 470" would quietly become "the matcher chose 334", and the rows would then describe a
decision that was never made.

Worse, those rows are the EVIDENCE FOR THE MERGE. They are where you can watch the gap
collapse to ~0.001 because the person's own duplicate was sitting in top-2 -- which is the
whole argument that these six people were invisible rather than misidentified. Rewrite the
log to agree with the conclusion and you have destroyed the only thing that justified it.

A log you edit to match a later decision is not a log. Anyone reading these rows must expect
`top1_person_id` to point at people who are now inactive; that is the correct reading of a
historical record, not a bug to be tidied away.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import (
    CanteenSession,
    FaceEmbedding,
    Person,
    PersonPhoto,
    PsychologistNote,
)
from qorgan.db.tenancy import owned_by, resolve_school_id
from qorgan.enums import PersonType
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MergeResult:
    keep_id: int
    drop_id: int
    keep_external: str
    drop_external: str
    photos_moved: int
    embeddings_moved: int
    sessions_moved: int
    keep_type: PersonType
    drop_type: PersonType
    # Defaulted so that records built before this field existed still construct; `_merge`
    # always writes it. Notes follow the person for the same reason meal sessions do.
    notes_moved: int = 0
    keep_reactivated: bool = False

    @property
    def crosses_person_type(self) -> bool:
        """A pupil merged onto a staff id, or the reverse. Not a detail: see `summary`."""
        return self.keep_type is not self.drop_type

    def summary(self) -> str:
        report = (
            f"Merged {self.drop_external} (id {self.drop_id}) into "
            f"{self.keep_external} (id {self.keep_id}).\n"
            f"  photos:     {self.photos_moved}\n"
            f"  embeddings: {self.embeddings_moved}\n"
            f"  sessions:   {self.sessions_moved}\n"
            f"  notes:      {self.notes_moved}\n"
            f"\n{self.drop_external} is now inactive. It leaves the gallery, so it can no "
            "longer sit in top-2 and kill the gap on its own twin — which is what made "
            "this person invisible to the system rather than merely hard to recognise."
        )
        if self.keep_reactivated:
            report += (
                f"\n\n{self.keep_external} was itself retired and has been REACTIVATED by "
                "this merge, because you asked for it with --reactivate. It is back in the "
                "gallery and, if it is a pupil, back on the canteen roster."
            )
        if self.crosses_person_type:
            report += "\n\n" + self._person_type_warning()
        return report

    def _person_type_warning(self) -> str:
        """The kept id decides whether this person eats. Say so, every time."""
        kept_staff = self.keep_type is PersonType.STAFF
        consequence = (
            "STAFF NEVER OPEN A MEAL SESSION. If this person is really a pupil, they have "
            "just left the canteen record: they will not be counted as fed, and nothing "
            "will report it — the number that would have said so is the one that stops "
            "being produced."
            if kept_staff
            else "This person will now open meal sessions and be counted as fed, which "
            "staff are not. If they are really staff, they are now in the canteen record."
        )
        return (
            "!! THIS MERGE CROSSED THE PUPIL/STAFF LINE.\n"
            f"     kept:    {self.keep_external} is {self.keep_type.value}\n"
            f"     dropped: {self.drop_external} was {self.drop_type.value}\n"
            f"   {consequence}\n"
            "   If that is the wrong way round, merge back the other way. Nothing else "
            "will tell you."
        )


def resolve_external(external_id: str, school_id: int | None = None) -> int:
    """`student_470` -> the database id, WITHIN one school.

    **`external_id` stopped being globally unique at migration 0009**, because two schools
    may each enrol a pupil numbered 7. Unscoped, this returns whichever row the database
    happened to reach first -- so `qorgan pupils merge student_470 staff_334` could resolve
    to another school's child and then re-point their photographs, embeddings and meal
    sessions onto somebody they have never met. The composite unique makes the scoped
    lookup exact again.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        person_id = session.scalar(
            select(Person.id).where(
                Person.external_id == external_id, owned_by(Person, school)
            )
        )
    if person_id is None:
        raise LookupError(f"no person holds external_id {external_id!r}")
    return int(person_id)


def merge_persons(
    keep_id: int, drop_id: int, *, reactivate: bool = False, school_id: int | None = None
) -> MergeResult:
    """Re-point everything from `drop_id` onto `keep_id` and deactivate `drop_id`.

    `reactivate` revives a `keep_id` that is itself retired, which is how a merge made in
    the wrong direction is undone. It is off by default and never inferred: `is_active=False`
    also means "left the school", and quietly pulling such a person back into the gallery
    would be a second silent wrong in place of the first.
    """
    if keep_id == drop_id:
        raise ValueError(f"cannot merge person {keep_id} into itself")

    result = with_retry(lambda: _merge(keep_id, drop_id, reactivate, school_id))
    logger.warning(
        "persons merged — a human decided these two ids are one person",
        extra={
            "keep": result.keep_external,
            "drop": result.drop_external,
            "sessions_moved": result.sessions_moved,
            "crosses_person_type": result.crosses_person_type,
        },
    )
    return result


def _merge(
    keep_id: int, drop_id: int, reactivate: bool = False, school_id: int | None = None
) -> MergeResult:
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        keep = _require(session, keep_id, school)
        drop = _require(session, drop_id, school)
        revived = _resolve_an_inactive_keeper(session, keep, drop, reactivate)

        photos = _repoint(session, PersonPhoto, keep_id, drop_id)
        embeddings = _repoint(session, FaceEmbedding, keep_id, drop_id)
        sessions = _repoint(session, CanteenSession, keep_id, drop_id)
        # **Notes move with the person, and `RecognitionAttempt` still does not.** The
        # distinction is the one this module already draws: a recognition attempt is a LOG
        # of what the matcher decided, and rewriting it would make the rows describe a
        # decision that was never made. A psychologist's note is not a record of anything
        # the system did — it is a record ABOUT a person, written by a human who met them.
        # Leaving it behind on a retired id would not preserve history, it would hide
        # everything ever written about that child behind an id no page lists.
        notes = _repoint(session, PsychologistNote, keep_id, drop_id)

        # Not deleted. The school issued that id, and a record saying it existed --
        # and that a human decided it was a duplicate -- is worth keeping.
        drop.is_active = False
        # WHY it is inactive, so `--reactivate` never has to guess. See the column.
        drop.merged_into_id = keep_id

        return MergeResult(
            keep_id=keep_id,
            drop_id=drop_id,
            keep_external=keep.external_id,
            drop_external=drop.external_id,
            photos_moved=photos,
            embeddings_moved=embeddings,
            sessions_moved=sessions,
            notes_moved=notes,
            keep_type=keep.person_type,
            drop_type=drop.person_type,
            keep_reactivated=revived,
        )


def _require(session: Session, person_id: int, school_id: int) -> Person:
    """NOT `session.get`. Both ids arrive off a form on `/pupils/duplicates`, and a merge
    is the most destructive thing this system does to an identity: it re-points meals,
    photographs and embeddings and retires an id. Fetched by primary key alone, one
    school could execute that against another school's children.

    "No person with id N" is the same answer an id nobody holds already gets -- whether it
    exists in another school is not this school's business to be told.
    """
    person = session.scalar(
        select(Person).where(Person.id == person_id, owned_by(Person, school_id))
    )
    if person is None:
        raise LookupError(f"no person with id {person_id}")
    return person


def _resolve_an_inactive_keeper(
    session: Session, keep: Person, drop: Person, reactivate: bool
) -> bool:
    """Merging INTO a retired id erases the person instead of resolving them -- unless
    reviving that id is precisely what the human is asking for. Returns whether it was.

    `load_gallery` reads active people only. An inactive `keep_id` would swallow every
    photo and embedding and then show none of them: the person would not become harder to
    recognise, they would stop existing. That is the exact failure this module exists to
    prevent, so it is refused rather than performed. (The reverse -- re-merging an id that
    is already dropped -- is fine, and is how this stays idempotent.)

    **But the refusal used to be the end of the road, and that was a defect in the highest
    stakes path here.** A merge across the pupil/staff line drops a child out of the meal
    record entirely (`day_report`'s roster is `person_type == STUDENT AND is_active`, and
    after a wrong-way merge neither id satisfies it). `MergeResult.summary()` warns and
    says "merge back the other way" -- and merging back is exactly this case, so it was
    refused, by a message that asked "Did you mean to merge the other way round?": the
    thing the operator had just done. No CLI command reactivated anybody. A decision the
    school makes by looking at a photograph was a one-way door.

    So reviving is allowed, and it is never inferred. It is also no longer taken on the
    operator's word: `is_active=False` also means "left the school", and `merged_into_id`
    is what tells the two apart, so the flag can REFUSE rather than trust.
    """
    if keep.is_active:
        return False
    if not reactivate:
        raise ValueError(
            f"cannot merge into {keep.external_id} (id {keep.id}): it is inactive, so "
            "everything moved onto it would leave the gallery entirely. If this is an "
            "id you retired with an earlier merge and you are putting it back, pass "
            "--reactivate (qorgan pupils merge --reactivate <keep> <drop>). If it is "
            "inactive because that person left the school, you want the other id."
        )

    _refuse_to_revive_what_no_merge_retired(session, keep, drop)
    keep.is_active = True
    keep.merged_into_id = None  # the merge that retired it has just been undone
    return True


def _refuse_to_revive_what_no_merge_retired(session: Session, keep: Person, drop: Person) -> None:
    """`--reactivate` reverses a merge. It is not a general "make this person exist again".

    Without `merged_into_id` this could not be checked at all, so the flag took the
    operator's word on the one operation where being wrong either revives a pupil the
    school expelled or refuses a child their meal history. Now the row says which case it
    is, and the two wrong cases are named separately because they are different mistakes.
    """
    if keep.merged_into_id is None:
        raise ValueError(
            f"{keep.external_id} (id {keep.id}) is inactive but was not retired by a "
            "merge, so there is no merge here to undo. Either that person left the "
            "school, or they were retired before this column existed — the row does not "
            "say which, and this will not guess. Reactivating them would be a new claim "
            "about who is enrolled, not a correction."
        )
    if keep.merged_into_id != drop.id:
        # Named, not numbered: the operator is holding external ids, and "person id 135"
        # sends them to the database to find out who they were told about.
        #
        # Scoped off `keep`, which `_require` has already confirmed is this school's. A
        # merge cannot cross a school, so the id it points at is this school's too -- and
        # this lookup exists only to put a NAME in an error message, which is precisely
        # the incidental read that ends up quoting another school's child back at somebody.
        actual = session.scalar(
            select(Person).where(
                Person.id == keep.merged_into_id, owned_by(Person, keep.school_id)
            )
        )
        into = f"{actual.external_id} (id {keep.merged_into_id})" if actual else "a deleted row"
        raise ValueError(
            f"{keep.external_id} (id {keep.id}) was merged into {into}, "
            f"not into {drop.external_id} (id {drop.id}). "
            "--reactivate undoes THAT merge; reversing it against somebody else is a new "
            "claim about who these people are, which is the one thing this command does "
            "not make on its own."
        )


def _repoint(session: Session, model: type, keep_id: int, drop_id: int) -> int:
    """Hand every row this person owns to the person we are keeping."""
    result = session.execute(
        update(model).where(model.person_id == drop_id).values(person_id=keep_id)
    )
    return int(result.rowcount or 0)

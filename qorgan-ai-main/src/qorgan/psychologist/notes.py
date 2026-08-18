"""The psychologist's own notes about one child. Confidential (client §13).

**The confidentiality is a capability on the ROUTE, and this module is written so that it
can be.** Nothing here is called from a page that is reachable without
`Capability.VIEW_PSYCHOLOGIST_NOTES`; there is no "load everything and hide some of it in
the template" path, because a body that has been fetched is a body that a later refactor
renders. §13 states the boundary in one line — «обычный оператор не должен видеть
конфиденциальные записи психолога» — and a template `{% if %}` is not a boundary, it is a
default.

**A note is never edited and never deleted here.** The two are missing for the reason
`qorgan.accounts` has no delete: an editable record of what somebody concluded about a
child, with no history, is a record that can be made to have always said something else.
If the school needs a correction, it is a new note, and the earlier one stays. That is a
decision worth revisiting WITH the school rather than by adding a route.

**No category, no severity, no status token.** `PsychologistNote.body` is free text on
purpose: a closed set of categories is a taxonomy, a taxonomy gets counted, and a count of
«тревожных» notes per child is precisely the ranking §8 promised the school this system
would not produce.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models import Person, PsychologistNote, User
from qorgan.db.tenancy import owned_by, resolve_school_id
from qorgan.logging_setup import get_logger
from qorgan.notify.message import local_time

logger = get_logger(__name__)

# An upper bound on one note, in characters. **CHOSEN, NOT MEASURED** -- nobody has
# written a note in this system yet. It exists so that the column is not an unbounded
# write path from a browser form (R8's spirit: nothing here grows without a limit), and it
# is generous enough that a page of prose about a child fits comfortably.
MAX_NOTE_LENGTH = 4000


class NoteRejected(ValueError):
    """Refused for a reason the person who hit it can act on.

    The message names the rule and NEVER quotes the body back: this text is confidential,
    it is on its way to an error banner and to a log line, and a refusal that repeats what
    it refused puts a note about a child in both.
    """


@dataclass(frozen=True, slots=True)
class Note:
    id: int
    # The school's wall clock, converted once here rather than in Jinja -- the events page
    # learned that a second formatter is a second thing to forget.
    created_at: str
    # The account that wrote it, or "—" if that account has since been removed. Never
    # blank, and never attributed to somebody else.
    author: str
    body: str


@dataclass(frozen=True, slots=True)
class NoteHistory:
    """Everything the notes page renders: who it is about, and what has been written."""

    person_id: int
    external_id: str
    display: str
    class_name: str | None
    notes: tuple[Note, ...]


def notes_for(person_id: int, *, school_id: int | None = None) -> NoteHistory | None:
    """Every note about this child, newest first. `None` if nobody holds that id.

    Not paginated, unlike the meal history: a meal session arrives twice a day for eleven
    years, and a note is written by a person who has met the child. If a school ever fills
    a page with these, that is the day this grows a limit -- and it will be a real number
    by then instead of a guess.

    **Scoped, and `None` is also the answer for another school's child.** `person_id`
    arrives from a URL. These are the most confidential rows in the schema -- what one
    school's psychologist wrote about one school's child -- so an id belonging elsewhere
    is answered exactly as a missing one is, and the caller cannot tell which it was.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        person = session.scalar(
            select(Person).where(Person.id == person_id, owned_by(Person, school))
        )
        if person is None:
            return None

        rows = session.execute(
            select(PsychologistNote, User.username)
            # OUTER: `author_id` is ON DELETE SET NULL, and an inner join would make the
            # note itself disappear the day the account did -- deleting the record by
            # deleting the person who wrote it.
            .outerjoin(User, User.id == PsychologistNote.author_id)
            .join(Person, Person.id == PsychologistNote.person_id)
            .where(PsychologistNote.person_id == person_id, owned_by(Person, school))
            .order_by(PsychologistNote.created_at.desc(), PsychologistNote.id.desc())
        ).all()

        return NoteHistory(
            person_id=person.id,
            external_id=person.external_id,
            display=person.display,
            class_name=person.class_name,
            notes=tuple(_note(row, username) for row, username in rows),
        )


def add_note(person_id: int, *, author_id: int, body: str, school_id: int | None = None) -> int:
    """Write one note. Returns its id.

    Refuses an unknown person rather than storing a confidential paragraph about nobody --
    which would be a row no page can ever show and no one can ever find to remove.

    **A child in another school is "нет такого ученика" here**, and refusing at the same
    door as a missing id is deliberate: the alternative is writing one school's
    confidential paragraph onto another school's pupil, where the psychologist who could
    read it never met the child and the one who wrote it can never find it again.
    """
    text = _clean(body)
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        known = session.scalar(
            select(Person.id).where(Person.id == person_id, owned_by(Person, school))
        )
        if known is None:
            raise NoteRejected("нет такого ученика")

        note = PsychologistNote(person_id=person_id, author_id=author_id, body=text)
        session.add(note)
        session.flush()
        # The id and the length. **NOT the body**: a log line outlives the request, is
        # copied into tickets and is read on call, and this text is the one thing in the
        # system §13 says an operator must not see.
        logger.info(
            "psychologist note written",
            extra={"note_id": note.id, "person_id": person_id, "length": len(text)},
        )
        return note.id


def _clean(body: str) -> str:
    text = body.strip()
    if not text:
        raise NoteRejected("пустая заметка не сохраняется")
    if len(text) > MAX_NOTE_LENGTH:
        raise NoteRejected(f"заметка длиннее {MAX_NOTE_LENGTH} символов не сохраняется")
    return text


def _note(row: PsychologistNote, username: str | None) -> Note:
    return Note(
        id=row.id,
        created_at=local_time(row.created_at),
        author=username or "—",
        body=row.body,
    )


# There is deliberately no `note_count()` here, and it was written and then removed.
# The pupil page would have shown "N заметок" beside its link -- but that page is reachable
# by an ADMIN, who does not hold `VIEW_PSYCHOLOGIST_NOTES`, and "the psychologist has
# written three times about this child" is itself a disclosure about the child. The link is
# drawn from the capability instead, exactly as the nav links are, and it carries no count.
# A function guarding nothing is the same guess as a permission guarding nothing.

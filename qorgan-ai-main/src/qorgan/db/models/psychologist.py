"""The psychologist's own notes (client §13).

**This is the only table in the schema written entirely by a human, and the only one an
administrator cannot read.** Every other table here records what a camera saw. This one
records what a person concluded, and §13 says so directly: «обычный оператор не должен
видеть конфиденциальные записи психолога».

That boundary is a capability on the ROUTE, not a condition in a template. Nothing that
loads a note body is reachable without `Capability.VIEW_PSYCHOLOGIST_NOTES`
(`web/routes/psychologist.py`), because a body fetched and then hidden by an `{% if %}`
has already been rendered into the response object that a later refactor forgets to guard.

**`person_id` is NOT NULL and the note is about a named child, deliberately.** Everything
else the cabinet shows is either anonymous (`lesson_tracks`, which may never gain a
`person_id` — see `qorgan.classroom`) or an identity the cameras established. A note is
neither: it is a human writing about a child they know by name, which is the one place in
this system where naming a child needs no measurement to justify it.

**A merge moves these notes.** `identity/merge.py` re-points photos, embeddings and
canteen sessions when a human decides two school ids are one person; notes go with them,
or a merge would silently hide everything the psychologist ever wrote about that child
behind a retired id. `RecognitionAttempt` is deliberately NOT moved because it is a LOG of
what the matcher decided; a note is not a log of a decision the system made, it is a record
about a person, so it follows the person.

`author_id` is `ON DELETE SET NULL` for the reason `Event.reviewed_by_id` is: accounts are
retired, never deleted (`qorgan.accounts`), and a note whose author was blanked is still a
note. `person_id` cascades because a note about nobody is a confidential paragraph with no
subject — but nothing in this system deletes a person either: a merge deactivates
(`identity/merge.py`).
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from qorgan.db.models.base import Base, TimestampMixin


class PsychologistNote(Base, TimestampMixin):
    __tablename__ = "psychologist_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # NULL once the account is gone. See the module docstring.
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    # Free text, and it stays free text. There is no severity, no category and no status
    # token here on purpose: a closed set of categories is a taxonomy, a taxonomy gets
    # counted, and a count of «тревожных» notes per child is the ranking §8 promised the
    # school this system would not produce.
    body: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("ix_psychologist_notes_person_created", "person_id", "created_at"),)

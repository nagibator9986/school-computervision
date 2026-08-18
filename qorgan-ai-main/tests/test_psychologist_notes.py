"""§13's one explicit boundary: «обычный оператор не должен видеть конфиденциальные
записи психолога».

**Every assertion here greps for `SECRET_BODY` in a response the wrong person got back.**
That shape is deliberate and it is the only shape that catches the failure this file
exists for. A test asserting `403` alone would still pass if the body were rendered under
a 200 somewhere else, and a test asserting an `{% if %}` in a template would be asserting
about the template rather than about what left the server. So: a note is written, and then
every page reachable by an operator and by an administrator is fetched and searched for its
text.

**This file is the sabotage target for this branch.** Widen `notes_viewer` in
`web/routes/psychologist.py` to a capability an operator holds and
`test_an_operator_cannot_read_a_note_on_any_page_they_can_open` goes red on the grep, not
merely on the status code -- which is the point: if breaking the guard leaves the operator
still unable to read the notes, the test was checking something else.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Person, PsychologistNote
from qorgan.enums import UserRole
from qorgan.identity.merge import merge_persons
from qorgan.psychologist.notes import MAX_NOTE_LENGTH, NoteRejected, add_note, notes_for
from qorgan.roles import Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.psychologist_fakes import ClientFor, a_camera, a_meal, a_pupil, client_factory
from tests.web_login import with_token

# Distinctive on purpose: every "did this leak?" assertion below greps for this exact
# string, and a body that looked like the other test data would hide in the noise.
SECRET_BODY = "zebra-lantern-9174-конфиденциально-о-ребёнке"

# Every page an OPERATOR or an ADMIN can reach that names a pupil or an incident. If a note
# body can be found on any of them, the boundary §13 states in one line is not there.
# `/psychologist` is on the list because an ADMIN holds the cabinet grant and neither notes
# grant -- the cabinet is exactly where a "just show a count" edit would land.
PAGES_THAT_MUST_NOT_CARRY_A_NOTE = (
    "/psychologist",
    "/psychologist/pupils/{person_id}",
    "/pupils",
    "/pupils/{person_id}/canteen",
    "/events",
    "/canteen",
)


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


@pytest.fixture
def pupil(session: Session) -> Person:
    return a_pupil(session)


def _write_a_note(client_for: ClientFor, person_id: int, body: str = SECRET_BODY) -> None:
    """Through the real form, by the real role. Not `add_note` directly: this is the path a
    school actually uses, and it is the path a capability change would break."""
    client = client_for(UserRole.PSYCHOLOGIST)
    response = client.post(
        f"/psychologist/notes/{person_id}", data=with_token(client, {"body": body})
    )
    assert response.status_code == 303, "the psychologist could not write their own note"


# -- the boundary ------------------------------------------------------------


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.ADMIN, UserRole.CANTEEN_STAFF])
def test_only_the_psychologist_holds_the_notes_capabilities(role: UserRole) -> None:
    """The contract below the HTTP layer. ADMIN is in this list on purpose: these are the
    only two capabilities in the enum an administrator does not hold, and that is a decision
    (`roles.py`) rather than an oversight -- so widening it has to be an edit to this line.
    """
    held = capabilities_for(role)
    assert Capability.VIEW_PSYCHOLOGIST_NOTES not in held
    assert Capability.WRITE_PSYCHOLOGIST_NOTES not in held

    psychologist = capabilities_for(UserRole.PSYCHOLOGIST)
    assert Capability.VIEW_PSYCHOLOGIST_NOTES in psychologist
    assert Capability.WRITE_PSYCHOLOGIST_NOTES in psychologist


def test_an_operator_cannot_read_a_note_on_any_page_they_can_open(
    client_for: ClientFor, camera: Camera, pupil: Person, session: Session
) -> None:
    """**The test this branch is sabotaged against.**

    Not `assert 403`: a status code says the notes ROUTE is shut, and the disclosure §13
    forbids is the TEXT arriving anywhere. So the note is written, and then every page an
    operator can open is fetched and searched for it.
    """
    a_meal(session, pupil, camera)
    _write_a_note(client_for, pupil.id)
    operator = client_for(UserRole.OPERATOR)

    # **The text first, the status code second, and the order is the whole point.** What
    # §13 forbids is the operator READING the note; the 403 is only the mechanism that
    # stops them. Asserting the status first would mean a widened capability failed here
    # on "200 != 403" -- true, but it would not have shown that the body came back with
    # it, and "the guard changed" is a weaker claim than "the child's file leaked".
    denied = operator.get(f"/psychologist/notes/{pupil.id}")
    assert SECRET_BODY not in denied.text, (
        "an operator read the psychologist's note about a child -- §13 in one line: "
        "«обычный оператор не должен видеть конфиденциальные записи психолога»"
    )
    assert denied.status_code == 403

    for path in PAGES_THAT_MUST_NOT_CARRY_A_NOTE:
        page = operator.get(path.format(person_id=pupil.id))
        assert SECRET_BODY not in page.text, f"a psychologist's note leaked onto {path}"


def test_an_administrator_cannot_read_a_note_either(
    client_for: ClientFor, camera: Camera, pupil: Person, session: Session
) -> None:
    """The administrator holds the cabinet and every other capability in the enum. These
    two are the exception, and it is a default rather than a wall: MANAGE_USERS can mint a
    psychologist account. What it cannot do is quietly open a colleague's file."""
    a_meal(session, pupil, camera)
    _write_a_note(client_for, pupil.id)

    admin = client_for(UserRole.ADMIN)
    assert admin.get("/psychologist").status_code == 200, "the admin lost the cabinet"

    denied = admin.get(f"/psychologist/notes/{pupil.id}")
    assert SECRET_BODY not in denied.text, "an administrator read a colleague's note"
    assert denied.status_code == 403

    for path in PAGES_THAT_MUST_NOT_CARRY_A_NOTE:
        assert SECRET_BODY not in admin.get(path.format(person_id=pupil.id)).text


def test_an_operator_cannot_write_a_note_in_the_psychologists_name(
    client_for: ClientFor, pupil: Person, session: Session
) -> None:
    """Reading and writing are separate grants and an operator holds neither. A note
    attributed to a psychologist who did not write it is worse than a leaked one."""
    operator = client_for(UserRole.OPERATOR)
    response = operator.post(
        f"/psychologist/notes/{pupil.id}", data=with_token(operator, {"body": SECRET_BODY})
    )

    assert response.status_code == 403
    session.expire_all()
    written = session.scalar(select(func.count(PsychologistNote.id)))
    assert written == 0, "an operator wrote a note"


def test_the_psychologist_reads_the_body_they_wrote(
    client_for: ClientFor, pupil: Person
) -> None:
    """The other direction, and it is not decoration: a boundary that also shuts out the
    person it was built for is a bug that every 403-only test above would call a pass."""
    _write_a_note(client_for, pupil.id)

    page = client_for(UserRole.PSYCHOLOGIST).get(f"/psychologist/notes/{pupil.id}")
    assert page.status_code == 200
    assert SECRET_BODY in page.text
    assert "user_psychologist" in page.text, "the note does not say who wrote it"


def test_the_body_never_reaches_the_log(
    client_for: ClientFor, pupil: Person, caplog: pytest.LogCaptureFixture
) -> None:
    """A log line outlives the request, is pasted into tickets and is read on call. §13's
    one confidential text is the last thing that belongs in it."""
    with caplog.at_level(logging.INFO):
        _write_a_note(client_for, pupil.id)

    assert caplog.records, "nothing was logged at all; this test would pass vacuously"
    assert SECRET_BODY not in caplog.text


# -- what a note may be ------------------------------------------------------


def test_an_empty_note_is_refused(pupil: Person) -> None:
    with pytest.raises(NoteRejected):
        add_note(pupil.id, author_id=1, body="   \n  ")


def test_a_note_longer_than_the_limit_is_refused(pupil: Person) -> None:
    """R8's spirit: a browser form is not an unbounded write path into the database."""
    with pytest.raises(NoteRejected):
        add_note(pupil.id, author_id=1, body="я" * (MAX_NOTE_LENGTH + 1))


def test_a_refusal_does_not_quote_the_text_back(pupil: Person) -> None:
    """The message goes to an error banner and to a log line. Repeating what was refused
    puts a note about a child in both."""
    with pytest.raises(NoteRejected) as refused:
        add_note(pupil.id, author_id=1, body=SECRET_BODY + "x" * MAX_NOTE_LENGTH)

    assert SECRET_BODY not in str(refused.value)


def test_a_note_about_nobody_is_refused(session: Session) -> None:
    """A confidential paragraph under an id nobody holds is a row no page can show and
    nobody can find to remove."""
    del session  # the schema, so that this fails on the rule rather than on a missing table
    with pytest.raises(NoteRejected):
        add_note(999, author_id=1, body=SECRET_BODY)


def test_a_rejected_note_re_renders_the_page_without_echoing_it(
    client_for: ClientFor, pupil: Person
) -> None:
    """400 rather than a redirect, so the author reads the rule beside their own text --
    and the rule, not the text, is what the page says back."""
    client = client_for(UserRole.PSYCHOLOGIST)
    response = client.post(
        f"/psychologist/notes/{pupil.id}", data=with_token(client, {"body": "   "})
    )

    assert response.status_code == 400
    assert SECRET_BODY not in response.text


def test_there_is_no_route_that_edits_or_deletes_a_note(app) -> None:
    """**Asserted against the real route table**, not against a docstring. An editable
    record of what somebody concluded about a child, with no history, is a record that can
    be made to have always said something else. A correction is a NEW note."""
    methods = {
        (route.path, method)
        for route in app.routes
        if getattr(route, "path", "").startswith("/psychologist/notes")
        for method in getattr(route, "methods", set())
    }

    assert not {m for _, m in methods} & {"PUT", "PATCH", "DELETE"}, methods
    assert {m for _, m in methods} >= {"GET", "POST"}, "the notes page lost a verb"


# -- a note follows the child ------------------------------------------------


def test_a_merge_moves_the_notes_onto_the_surviving_id(
    client_for: ClientFor, session: Session
) -> None:
    """A human decides two school ids are one child. If the notes stayed behind, everything
    ever written about that child would sit under an id no page lists -- which is hiding
    the history rather than keeping it."""
    keep = a_pupil(session, "student_470")
    drop = a_pupil(session, "staff_334")
    _write_a_note(client_for, drop.id)

    result = merge_persons(keep.id, drop.id)

    assert result.notes_moved == 1
    session.expire_all()
    surviving = notes_for(keep.id)
    assert surviving is not None
    assert [note.body for note in surviving.notes] == [SECRET_BODY]

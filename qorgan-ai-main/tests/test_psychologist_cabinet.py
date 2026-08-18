"""The cabinet index: what it says is working, and whether that is true.

The old system was not killed by missing features. It was killed by pages that looked like
they were working, so the assertions here are mostly of one shape: **build a world, then
check that the page's own description of that world is the one the rows support.** A block
that says «сигнал живой» over an empty table is the exact defect this module was written
against, and it is a defect no status code catches.

Two decisions are pinned here because they are the ones a later edit would reverse without
noticing:

  * **The empty canteen block is SHOWN, not hidden.** The school has to be able to see that
    the mechanism exists and what it is waiting for. Hiding an empty block is the friendlier
    page and the dishonest one -- it makes "we have not built it" and "it has nothing to say
    yet" look identical.
  * **Nothing computes a recommendation.** `test_no_module_in_the_cabinet_computes_a_
    recommendation` reads the package's own source, because this is a promise made to the
    school in writing (`docs/questions-for-school.md` §8) and prose in a docstring does not
    keep it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Person
from qorgan.enums import UserRole
from qorgan.psychologist.cabinet import CANTEEN_IS_WAITING_FOR_THE_CAMERA, cabinet_view
from qorgan.psychologist.signals import SignalState
from qorgan.roles import Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.conftest import REPO_ROOT, SRC_DIR
from tests.psychologist_fakes import (
    ClientFor,
    a_camera,
    a_lesson,
    a_meal,
    a_pupil,
    an_event,
    an_unattributed_meal,
    client_factory,
)
from tests.web_login import with_token

# Words that would only appear in this package if something in it had started deciding for
# a person. Deliberately not a list of function names: the failure §8 forbids is a
# CONCLUSION reaching the reader, and it arrives as vocabulary long before it arrives as an
# algorithm. A legitimate future use would have to be argued for by editing this line.
FORBIDDEN_VOCABULARY = (
    "рекоменд",  # «рекомендуется проверка психологом» -- the line §8 forbids by name
    "риск",
    "тревожн",
    "подозритель",
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


# The three capabilities §13 actually asks for: the cabinet, and the two halves of the
# psychologist's own notes. Everything else the role holds is a WIDENING that exists so
# those three pages can work, and the school is entitled to be told about each one.
SECTION_13_CAPABILITIES = frozenset(
    {
        Capability.VIEW_PSYCHOLOGIST_CABINET,
        Capability.VIEW_PSYCHOLOGIST_NOTES,
        Capability.WRITE_PSYCHOLOGIST_NOTES,
    }
)

# Each widening, and the heading in `docs/questions-for-school.md` §10.2 that discloses it.
# **A capability granted to this role without a line here fails the test below**, which is
# the only mechanism that stops a widening reaching a real school unmentioned. It caught
# nothing when written -- it was written because three of these four had already reached
# the document's first draft unmentioned, and only the canteen was named.
DISCLOSED_IN_SECTION_10 = {
    Capability.VIEW_BULLYING_MEDIA: "Снимки и клипы инцидентов",
    Capability.VIEW_CANTEEN: "Журнал столовой по всей школе",
    Capability.VIEW_PUPILS: "Список учеников",
    Capability.VIEW_LESSON_METRICS: "Показатели уроков",
    # The fifth. Its heading names the unit of account, because that is the whole difference
    # between it and the fourth: `VIEW_LESSON_METRICS` is anonymous permanently, this one
    # stops being anonymous the day the school signs a seating plan. A disclosure that said
    # only «разбор уроков» would read as a duplicate of item 4 and hide exactly that.
    Capability.VIEW_CLASSROOM_ANALYSIS: "Разбор записей уроков по местам",
}


def _block(key: str):
    return next(block for block in cabinet_view().blocks if block.key == key)


# -- the canteen block, which is empty and stays visible ---------------------


def test_the_empty_canteen_block_is_shown_rather_than_hidden(
    client_for: ClientFor, session: Session
) -> None:
    """**The owner's decision, pinned.** The canteen camera has not been moved, so this
    signal has nothing in it -- and the school must still be able to see that the mechanism
    exists and what it is waiting for. A page that hides its empty blocks makes "we never
    built it" and "it has nothing to say yet" look the same.
    """
    del session
    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist")

    assert page.status_code == 200
    assert "Посещаемость столовой" in page.text, "the empty block was hidden"
    # Against the constant rather than against a quoted sentence: the owner marked this
    # wording for review before the pilot, so rewording it must not cost a red test. What
    # is asserted separately is that the reason survives the rewording -- a caption that
    # says the block is empty without saying what it is waiting for explains nothing.
    assert CANTEEN_IS_WAITING_FOR_THE_CAMERA in page.text
    assert "камер" in CANTEEN_IS_WAITING_FOR_THE_CAMERA, (
        "the caption no longer says the block is waiting for the canteen camera, which is "
        "the only reason it is empty and the only thing the school needs from it"
    )


def test_the_canteen_block_is_empty_until_a_session_names_a_child(
    camera: Camera, session: Session
) -> None:
    """`person_id IS NULL` is an entry nobody was recognised at, which is almost every
    session in the school's database today. Counting those would make this signal look
    alive while naming nobody."""
    an_unattributed_meal(session, camera)
    an_unattributed_meal(session, camera)

    block = _block("canteen")
    assert block.state is SignalState.EMPTY
    assert block.count == 0


def test_the_canteen_block_goes_live_when_a_child_is_actually_recognised(
    camera: Camera, pupil: Person, session: Session
) -> None:
    """The other direction. A block that can only ever say EMPTY is a block whose state is
    a constant, and a constant dressed as a measurement is what this file exists to catch.
    """
    a_meal(session, pupil, camera)
    an_unattributed_meal(session, camera)

    block = _block("canteen")
    assert block.state is SignalState.LIVE
    assert block.count == 1, "the unattributed entry was counted as a named child"


# -- the classroom block, which is not empty and is not about anybody --------


def test_the_classroom_block_stays_anonymous_however_many_lessons_arrive(
    camera: Camera, session: Session
) -> None:
    """**ANONYMOUS is not a third point on the EMPTY-to-LIVE line.** It says the rows ARE
    arriving and still cannot become a statement about a named child, because
    `lesson_tracks` carries no `person_id` and may never gain one. Showing this as EMPTY
    would promise the school that waiting fixes it."""
    for _ in range(3):
        a_lesson(session, camera)

    block = _block("classroom")
    assert block.state is SignalState.ANONYMOUS
    assert block.count == 3


def test_the_classroom_block_is_anonymous_even_with_nothing_in_it(
    camera: Camera, session: Session
) -> None:
    del camera, session
    assert _block("classroom").state is SignalState.ANONYMOUS


def test_the_page_says_the_classroom_half_cannot_name_a_child(
    client_for: ClientFor, camera: Camera, session: Session
) -> None:
    """The §8 contradiction, left with the school in plain words rather than resolved by a
    nullable foreign key onto a pupil."""
    a_lesson(session, camera)
    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist")

    assert "АНОНИМНОМУ ТРЕКУ" in page.text
    assert "14 970" in page.text, "the claim is stated without the measurement behind it"
    assert "questions-for-school.md §10" in page.text


# -- the referral block ------------------------------------------------------


def test_the_referral_block_is_empty_until_a_person_presses_the_button(
    camera: Camera, session: Session
) -> None:
    """An event existing is not a referral. Only a person is."""
    del session
    an_event(camera)

    block = _block("referrals")
    assert block.state is SignalState.EMPTY
    assert "Пока никто ничего не передавал" in " ".join(block.lines)


def test_the_referral_block_counts_what_a_person_handed_over(
    client_for: ClientFor, camera: Camera
) -> None:
    event_id = an_event(camera)
    operator = client_for(UserRole.OPERATOR)
    operator.post(f"/events/{event_id}/refer", data=with_token(operator))

    block = _block("referrals")
    assert block.state is SignalState.LIVE
    assert block.count == 1


def test_the_cabinet_shows_who_referred_the_child(
    client_for: ClientFor, camera: Camera
) -> None:
    """A name held only in a column is a name nobody reads."""
    event_id = an_event(camera)
    operator = client_for(UserRole.OPERATOR)
    operator.post(f"/events/{event_id}/refer", data=with_token(operator))

    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist")
    assert "user_operator" in page.text


# -- who may open it ---------------------------------------------------------


def test_the_cabinet_is_shut_to_everyone_who_was_not_granted_it(
    client_for: ClientFor,
) -> None:
    """§14 gives this page to the psychologist and -- so that somebody at the school other
    than the psychologist can see that referrals are arriving -- to the administrator. The
    failure this whole module is written against is a page that LOOKS like it is working,
    and nobody notices that from inside the role that owns it."""
    for role in (UserRole.OPERATOR, UserRole.DEVELOPER, UserRole.CANTEEN_STAFF):
        assert Capability.VIEW_PSYCHOLOGIST_CABINET not in capabilities_for(role)
        assert client_for(role).get("/psychologist").status_code == 403


def test_the_psychologist_does_not_reach_the_raw_bullying_log(
    client_for: ClientFor, camera: Camera
) -> None:
    """§14 gives them «подтвержденные случаи», not the candidate log with its false
    positives. The cabinet is a smaller disclosure through a DIFFERENT door, not the same
    door with a filter on it -- a filter is a page decision and a door is a capability."""
    an_event(camera)
    psychologist = client_for(UserRole.PSYCHOLOGIST)

    assert Capability.VIEW_BULLYING not in capabilities_for(UserRole.PSYCHOLOGIST)
    assert psychologist.get("/events").status_code == 403
    assert psychologist.get("/notifications").status_code == 403


def test_a_correct_password_lands_the_psychologist_in_their_own_cabinet(
    client_for: ClientFor,
) -> None:
    """They also hold VIEW_CANTEEN, so without the ordering in `web/security.landing_for`
    a correct login would drop them on the school's lunch journal — a page about the whole
    school's lunch rather than the one they were given an account for. Landing follows the
    job, not the order the capability set happens to be tested in.

    Read off `/login`, which redirects an already-authenticated client to its landing page:
    that is the same `landing_for` the POST uses, and it is reachable without replaying the
    login by hand. `/` is NOT the landing here and must not be — a psychologist holds no
    VIEW_CAMERAS and gets a 403 from it, which is what makes the ordering matter at all.
    """
    client = client_for(UserRole.PSYCHOLOGIST)
    assert client.get("/").status_code == 403, "the psychologist reached the camera wall"

    landing = client.get("/login", follow_redirects=False)
    assert landing.status_code == 303
    assert landing.headers["location"] == "/psychologist"


def test_every_widening_of_the_psychologists_rights_is_disclosed_to_the_school() -> None:
    """**The school is told about ALL of them, not the convenient one.**

    §10.2's first draft admitted the canteen journal and stayed silent about three other
    grants -- including `VIEW_BULLYING_MEDIA`, which reaches snapshots and clips of
    incidents nobody referred. That is video of children, it is the heaviest of the four,
    and `roles.py` had ALREADY called it "a real widening" in a code comment. A confession
    in a comment is not a disclosure: the school does not read this repository.

    So the document is checked against the permission table rather than against anybody's
    memory of it. This is the test that makes «мы это не прячем» a fact.
    """
    doc = (REPO_ROOT / "docs" / "questions-for-school.md").read_text(encoding="utf-8")
    section = doc.partition("## 10.")[2]
    assert section, "§10 is gone from the document four source files cite"

    held = capabilities_for(UserRole.PSYCHOLOGIST)
    widenings = held - SECTION_13_CAPABILITIES

    undocumented = sorted(c.value for c in widenings if c not in DISCLOSED_IN_SECTION_10)
    assert not undocumented, (
        f"the psychologist holds {undocumented} beyond what §13 asks for, and nothing in "
        "this test says how the school is told. Add the grant to DISCLOSED_IN_SECTION_10 "
        "and write the disclosure into §10.2 -- or do not grant it."
    )

    missing = sorted(
        DISCLOSED_IN_SECTION_10[c] for c in widenings if DISCLOSED_IN_SECTION_10[c] not in section
    )
    assert not missing, f"§10.2 no longer discloses: {missing}"


def test_the_heaviest_widening_is_named_as_video_of_children() -> None:
    """`VIEW_BULLYING_MEDIA` is the one grant on this role that opens recordings of
    children who were never referred to this psychologist. §10.2 has to say that in those
    words, because "материалы инцидента" is true and does not tell a headteacher what they
    are agreeing to."""
    doc = (REPO_ROOT / "docs" / "questions-for-school.md").read_text(encoding="utf-8")

    assert Capability.VIEW_BULLYING_MEDIA in capabilities_for(UserRole.PSYCHOLOGIST)
    assert "это видео детей" in doc
    assert "**любого**" in doc, "§10.2 no longer says the clips are not only the referred ones"


# -- the line nothing in here may cross --------------------------------------


def test_no_module_in_the_cabinet_computes_a_recommendation() -> None:
    """**§8, in writing: «Никаких диагнозов и никаких направлений к психологу ОТ СИСТЕМЫ».**

    Read off the source rather than asserted in a docstring, because a docstring is not a
    guard. What is scanned is the CODE: the package's docstrings quote the forbidden
    sentence in order to forbid it, and a scan that could not tell those apart would have
    to be either wrong or ignored.
    """
    offenders = [
        f"{path.name}: {word!r}"
        for path in sorted((SRC_DIR / "qorgan" / "psychologist").rglob("*.py"))
        for word in FORBIDDEN_VOCABULARY
        if word in _executable_text(path).lower()
    ]

    assert not offenders, (
        "the cabinet started deciding something for a person: "
        + ", ".join(offenders)
        + ". §8 promised the school the system never does this; a referral is an act by a "
        "named human and nothing else."
    )


def _executable_text(path: Path) -> str:
    """The module with its docstrings and comments blanked out.

    Prose is where this package ARGUES for the rule -- `psychologist/__init__.py` quotes
    «рекомендуется проверка школьным психологом» in order to forbid it -- so a scan that
    could not tell prose from code would have to be either wrong or exempted, and an
    exemption list is how a guard stops guarding.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            for index in range(first.lineno - 1, first.end_lineno):
                lines[index] = ""

    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_the_page_itself_carries_no_recommendation(
    client_for: ClientFor, camera: Camera, pupil: Person, session: Session
) -> None:
    """The rendered HTML, because a template is source the scan above does not read."""
    a_meal(session, pupil, camera)
    a_lesson(session, camera)
    psychologist = client_for(UserRole.PSYCHOLOGIST)

    for path in ("/psychologist", f"/psychologist/pupils/{pupil.id}"):
        body = psychologist.get(path).text.lower()
        for word in FORBIDDEN_VOCABULARY:
            assert word not in body, f"{path} suggests something to the reader: {word!r}"

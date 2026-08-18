"""Five config-fed surfaces refuse on a multi-school installation, and only then.

`/`, `/cameras`, `/api/cameras`, `/preview/{camera}.jpg` and `/settings` are built from the
camera YAML, which has no school dimension whatsoever -- so on a two-school installation
they can show every school's corridors or none. The tenancy guard is silent about all five,
by construction: it scans database queries, and there is no query here.

**The argument that settled it was this branch's own inconsistency.** The system is willing
to crash a detection worker into a restart loop rather than guess which school a database
row belongs to, and was simultaneously willing to serve live video of another school's
children rather than refuse. Both cannot be right. `web/config_scope.py` makes the second
behave like the first.

**BOTH DIRECTIONS ARE TESTED HERE, AND THE SECOND MATTERS MORE.** A refusal that fires on
one school would break the only configuration that exists today -- every real installation
-- in exchange for protecting a configuration that does not exist yet. So every route is
asked twice: it must refuse when there are two schools, and it must serve exactly as before
when there is one. The single-school half is not a formality; it is the regression this
change could most easily have been.

The refusals undo themselves. When the camera configuration learns to name a school, these
five call sites come out together with `test_schools_page.py::test_the_camera_config_still_
has_no_school_and_so_the_warning_still_applies`, which is what will go red to say so.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import null, select
from sqlalchemy.orm import Session

from qorgan.config.common import PreviewSettings
from qorgan.db.models import School, User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.preview import PreviewPublisher
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.fakes import noisy_frame
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
REFUSED = 409
OTHER_SLUG = "gymnasium-4"

# Matches `tests/test_web_preview.py`, which is where this machinery comes from.
PREVIEW = PreviewSettings(fps=15.0, width=320)


def _publish_and_wait(client: TestClient, publisher: PreviewPublisher, camera: str) -> None:
    """Publish until received, bounded by a deadline -- not sleep-and-hope.

    A frame published before the SUB socket's connect completes is dropped rather than
    queued (ZeroMQ's "slow joiner"), and there is no event to wait on, so republishing until
    the effect is OBSERVED is the only correct synchronisation. Lifted from
    `test_web_preview.py` deliberately: two different waits for the same handshake is two
    chances for one of them to be subtly wrong.
    """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        publisher.publish(camera, noisy_frame(1), PREVIEW, now=time.time() + 100)
        if client.get(f"/preview/{camera}.jpg").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError(f"no preview ever arrived for {camera}")

# The five surfaces built from camera configuration. `hall_left` is a camera the shipped
# config really defines, so the preview route gets past its own name check and reaches the
# gate rather than 404-ing before it.
CONFIG_FED = (
    "/",
    "/cameras",
    "/api/cameras",
    "/preview/hall_left.jpg",
    "/settings",
)

# Surfaces that read the DATABASE, which IS scoped per school. They must keep working with
# two schools -- otherwise this change has quietly turned a camera problem into an outage.
#
# `/schools` is deliberately NOT in this tuple. It needs MANAGE_SCHOOLS, which the ADMIN
# these fixtures log in as does not hold, so its parameter would have passed on a 403 --
# structurally, whatever the gate did. It gets its own test below, asked with an account
# that can actually open it.
DATABASE_FED = ("/events", "/notifications", "/pupils", "/canteen")


def _accounts(session: Session) -> None:
    """Two accounts, because no single role can ask every question this file asks.

    ADMIN holds VIEW_CAMERAS and VIEW_SETTINGS, so it can ask all five config-fed
    surfaces. It does NOT hold MANAGE_SCHOOLS -- that is the superadmin alone -- so the
    schools page needs the second account. Asking it as ADMIN returns a 403 that has
    nothing to do with the camera gate, which is exactly the vacuous check this file
    used to contain.
    """
    session.add(
        User(
            username="head",
            password_hash=hash_password(PASSWORD),
            role=UserRole.ADMIN,
            school_id=session.scalar(select(School.id)),
        )
    )
    session.add(
        User(
            username="root",
            password_hash=hash_password(PASSWORD),
            role=UserRole.SUPERADMIN,
            # Belongs to no school. That is what the role IS, and `school_of` refuses to
            # guess one for it.
            #
            # **`null()` and NOT `None`, and this said `None` until it was measured.**
            # `school_key` carries a column default, and SQLAlchemy omits a column whose
            # mapped value IS `None` from the INSERT -- so the default fired and this line
            # gave the superadmin the default school, which `accounts.create_account` can
            # no longer produce. In THIS file it was worse than a misfiling: `_accounts`
            # runs before the second school is added, and on a two-school database `None`
            # RAISES `UndecidedSchool`. Swapping those two statements -- the obvious tidy-up
            # in a file whose whole subject is two schools -- errored on the one account
            # documented as belonging to none. `null()` is a value, so no default fires and
            # the order stops mattering.
            school_id=null(),
        )
    )
    session.commit()


def _client(settings: Settings, username: str = "head") -> TestClient:
    client = TestClient(create_app(), follow_redirects=False)
    client.__enter__()
    response = client.post(
        "/login", data=with_token(client, {"username": username, "password": PASSWORD})
    )
    assert response.status_code == 303, (
        f"{username} could not log in; nothing below would mean anything"
    )
    return client


@pytest.fixture
def one_school(settings: Settings, session: Session) -> Iterator[TestClient]:
    """The installation every real deployment is: exactly one school."""
    _accounts(session)
    client = _client(settings)
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def two_schools(settings: Settings, session: Session) -> Iterator[TestClient]:
    """The same installation the moment a superadmin adds a second school."""
    _accounts(session)
    session.add(School(slug=OTHER_SLUG, name="Гимназия №4"))
    session.commit()
    client = _client(settings)
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def two_schools_as_superadmin(settings: Settings, session: Session) -> Iterator[TestClient]:
    """Two schools, seen by the one account that may look at the register."""
    _accounts(session)
    session.add(School(slug=OTHER_SLUG, name="Гимназия №4"))
    session.commit()
    client = _client(settings, "root")
    yield client
    client.__exit__(None, None, None)


@pytest.mark.parametrize("path", CONFIG_FED)
def test_a_config_fed_page_refuses_when_a_second_school_exists(
    two_schools: TestClient, path: str
) -> None:
    """Refuse, rather than serve another school's cameras -- or another school's live video."""
    response = two_schools.get(path)

    assert response.status_code == REFUSED, (
        f"{path} answered HTTP {response.status_code} on a two-school installation. It is "
        "built from camera configuration that cannot name a school, so anything it "
        "returned was every school's cameras -- and for /preview, live video of another "
        "school's corridor."
    )


@pytest.mark.parametrize("path", CONFIG_FED)
def test_the_refusal_explains_itself_rather_than_being_a_500_or_a_blank(
    two_schools: TestClient, path: str
) -> None:
    """A refusal nobody can read gets treated as a fault, and faults get "fixed"."""
    response = two_schools.get(path)
    body = response.text

    assert "школ" in body, (
        f"{path} refused without saying that schools are the reason: {body[:200]!r}"
    )
    assert len(body.strip()) > 80, f"{path} returned an all-but-empty body: {body!r}"


@pytest.mark.parametrize("path", CONFIG_FED)
def test_the_same_page_serves_normally_on_a_single_school_installation(
    one_school: TestClient, path: str
) -> None:
    """**The reverse check, and the more important one.**

    Every installation in existence today has exactly one school. A gate that fired here
    would break all of them to protect a configuration that does not yet exist -- which
    would be a far worse trade than the exposure it closes.

    `/preview/hall_left.jpg` is asserted as "did not refuse" rather than 200: with no
    worker publishing frames in a test process it answers 503, which is its own correct
    behaviour and not this gate's business.
    """
    response = one_school.get(path)

    assert response.status_code != REFUSED, (
        f"{path} refused on an installation with ONE school. That is every real "
        "deployment, and this gate has just taken the camera wall away from all of them."
    )
    if path != "/preview/hall_left.jpg":
        assert response.status_code == 200, (
            f"{path} answered HTTP {response.status_code} on a single-school installation"
        )


@pytest.mark.parametrize("path", DATABASE_FED)
def test_the_database_fed_pages_keep_working_with_two_schools(
    two_schools: TestClient, path: str
) -> None:
    """The blast radius, bounded by measurement rather than by intention.

    Only the surfaces fed by unscoped CONFIG are closed. Everything fed by the database is
    scoped per school and must keep working -- otherwise adding a school is an outage
    rather than a limitation, and the difference matters to whoever is on the phone about
    it.

    **200, not `in (200, 403)`.** The looser form is how this test was vacuous: a 403 for a
    capability the logged-in role never held would satisfy it no matter what the gate did.
    Every path here is one an ADMIN genuinely holds, so 200 is the only acceptable answer.
    """
    response = two_schools.get(path)

    assert response.status_code == 200, (
        f"{path} answered HTTP {response.status_code} with two schools. It reads the "
        "database, which is scoped, so it should be unaffected by the camera-config gate."
    )


def test_the_schools_page_stays_open_to_the_superadmin(
    two_schools_as_superadmin: TestClient,
) -> None:
    """The loudest claim this file makes, finally asked of an account that can answer it.

    `/schools` has to keep working with two schools: it is the only page that shows the
    tenancy, and closing it in the very state it exists to explain would leave nobody able
    to see what happened. It is asked as the SUPERADMIN because MANAGE_SCHOOLS is the
    superadmin alone -- asked as anybody else it 403s for a reason that has nothing to do
    with this change, which is what let the old version of this check pass whatever the
    code did.

    Note what is NOT claimed here, or anywhere else any more: that the second school can be
    REMOVED from this page. `qorgan.schools` has `list_schools`, `create_school` and
    `rename_school`, and no delete. The page shows the second school; it cannot undo it.
    """
    response = two_schools_as_superadmin.get("/schools")

    assert response.status_code == 200, (
        f"the superadmin cannot open the schools register (HTTP {response.status_code}) on "
        "a two-school installation -- the one state that page exists to explain."
    )
    assert OTHER_SLUG in response.text, (
        f"the register opened but does not list {OTHER_SLUG!r}, so it is not actually "
        "showing the second school the rest of this file is about."
    )


@pytest.mark.parametrize("path", CONFIG_FED)
def test_the_refusal_never_sends_the_reader_where_their_role_cannot_go(
    two_schools: TestClient, path: str
) -> None:
    """The refusal is read by VIEW_CAMERAS / VIEW_SETTINGS holders. `/schools` is not theirs.

    The first version of this refusal told the reader to go and look at the schools page.
    Measured: operator, admin and developer all get 403 from `/schools`, because
    MANAGE_SCHOOLS is the superadmin's alone. So the one instruction on the page was "do
    the thing you cannot do" -- the same defect this branch had already caught once, in the
    warning that promised a school could be deleted.

    A refusal that tells you to do something impossible reads as a broken system, and a
    broken system is what gets "fixed" by switching the protection off.
    """
    body = two_schools.get(path).text

    assert 'href="/schools"' not in body, (
        f"{path} links to /schools, which the roles that see this refusal cannot open "
        "(403). Name the superadmin instead of linking the page."
    )
    assert "суперадминистратор" in body, (
        f"{path} refuses without telling the reader who can actually act on it. The only "
        "account that can is the superadmin, and the reader is not it."
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "1 школа"),
        (2, "2 школы"),
        (4, "4 школы"),
        (5, "5 школ"),
        (11, "11 школ"),
        (21, "21 школа"),
    ],
)
def test_the_count_reads_as_russian(count: int, expected: str) -> None:
    """"5 школы" is wrong, and the sentence carrying it is asking to be believed.

    Agreement is decided in Python, not in Jinja, so it is testable at all -- which is the
    same reason every other formatting decision in this codebase sits outside the template.
    """
    from qorgan.web.config_scope import schools_phrase

    assert schools_phrase(count) == expected


def _refuse_a_frame_that_is_really_there(settings: Settings, session: Session):
    """Publish a real frame, prove it is served, add a school, and return the second answer.

    The order is the whole point. The frame is confirmed present -- fetched, 200, JPEG
    magic -- BEFORE the second school exists, so whatever comes back afterwards is a
    refusal of something that was demonstrably there.
    """
    client = _client(settings)
    try:
        publisher = PreviewPublisher(client.app.state.previews.address)
        try:
            _publish_and_wait(client, publisher, "hall_left")

            live = client.get("/preview/hall_left.jpg")
            assert live.status_code == 200
            assert live.content[:2] == b"\xff\xd8", (
                "the frame served on one school is not a JPEG, so the caller cannot tell "
                "afterwards whether the refusal withheld anything real"
            )

            # The gate reads the school count per request, so the installation becomes
            # multi-school right here, with the frame already published and still held by
            # the subscriber. Nothing about the preview pipeline changes.
            session.add(School(slug=OTHER_SLUG, name="Гимназия №4"))
            session.commit()

            return client.get("/preview/hall_left.jpg")
        finally:
            publisher.close()
    finally:
        client.__exit__(None, None, None)


def test_a_frame_that_really_exists_is_still_refused_when_a_second_school_appears(
    settings: Settings, session: Session
) -> None:
    """The refusal, verified against a LIVE preview pipeline rather than a status code.

    A real frame is published, proved present by fetching it and checking the JPEG magic,
    and only then is the second school added and the frame asked for again. The picture is
    known to be there, so whatever comes back is known to be withholding something real.

    **WHAT THIS CATCHES THAT THE STATUS-ONLY TEST DOES NOT, MEASURED RATHER THAN ASSUMED.**
    Both directions were sabotaged, and the answer was not the one this docstring first
    claimed:

      * **409 returned WITH the JPEG body** -- the status says refused and the picture goes
        out anyway. Status-only test: PASSED. This test: FAILED. That is the only regression
        of the two that actually leaks video of a corridor, and nothing else in the suite
        sees it.
      * **The gate moved BELOW the frame lookup.** Status-only test: FAILED. This test:
        PASSED -- because with the frame present the relocated gate still refuses it
        correctly. An earlier version of this comment claimed the opposite; it was written
        from reasoning rather than measurement, which is the habit this whole branch has
        been correcting.

    So the two are complementary rather than one being stronger, and the pair is what covers
    the surface. The positive direction was proved with a real frame from the start; the
    refusal was proved only by a status code, which was the weakest evidence behind the
    loudest claim this branch makes.
    """
    _accounts(session)
    refused = _refuse_a_frame_that_is_really_there(settings, session)

    assert refused.status_code == REFUSED, (
        f"a live frame was still served on a two-school installation: HTTP "
        f"{refused.status_code}. This is live video of a corridor, and the configuration "
        "cannot say whose corridor it is."
    )
    assert refused.content[:2] != b"\xff\xd8", (
        "the refusal carried a JPEG body. The status said 409 and the picture went out "
        "anyway -- which is the one failure the status code alone could never show."
    )
    assert refused.headers["content-type"].startswith("text/plain"), (
        f"expected a readable text refusal, got {refused.headers['content-type']!r}"
    )
    assert "школ" in refused.text

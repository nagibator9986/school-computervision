"""`/media` is three different kinds of thing behind one URL, and one permission.

`MEDIA_ROOT` holds:

  * `snapshots/` and `clips/` — the bullying evidence. A frame, and a video, of an
    incident involving named children.
  * `people/` — the enrolment gallery. Every pupil in the school, one clear photograph
    each, filed under their class.

A single `VIEW_MEDIA` covered all three, and `media.py` said so in a comment rather than
in a check: *"One capability for the whole tree, because the tree is mixed … Anyone
holding this holds all three."*

That is the §14 problem again, one layer down. §14 was solved for ROUTES — a canteen
worker reaches `/canteen` and not `/events` — by replacing a rank with capabilities. But
inside `/media` the rank came back as a shape: the tree is one grant, so any role that
needs any part of it takes all of it. The psychologist §13 describes needs the clip of the
incident they are asked about. Granting that under one capability hands them a
photographic register of every child in the school, which no part of their work touches.

This was written while the psychologist role did not exist, and said so: "nobody is harmed
by this today … that is exactly when to fix it, because the capability model is the thing a
future role is granted through, and a model that cannot express 'clips but not the gallery'
will be discovered at the moment somebody is being granted access."

**That role arrived on 2026-07-28, and it holds `VIEW_BULLYING_MEDIA` and not
`VIEW_PUPIL_PHOTOS`.** The split is no longer hypothetical, and the test below that used to
model it by monkeypatching an operator now uses the real role.

**The capability is decided by the RESOLVED path, never by the URL string.** `..` is a
valid string, and `people/../clips/x.mp4` starts with `people/` while being a clip. See
`test_the_permission_follows_the_file_not_the_url`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import User
from qorgan.db.types import utcnow
from qorgan.enums import UserRole
from qorgan.events.recorder import media_root, write_snapshot
from qorgan.passwords import hash_password
from qorgan.paths import to_relative
from qorgan.roles import ROLE_CAPABILITIES, Capability, capabilities_for
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

ClientFor = Callable[[UserRole], TestClient]


@pytest.fixture
def app(settings: Settings, session: Session):
    del settings, session
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
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


def _snapshot() -> str:
    """A real bullying snapshot, written by the production writer."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    return str(write_snapshot(frame, "hall_left", moment=utcnow()))


def _pupil_photo() -> str:
    """A pupil's enrolment photograph, where `faces.importer` puts them: people/<class>/."""
    target = media_root() / "people" / "5-А" / "student_333_1778595343147.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return str(to_relative(target))


def _staged_import() -> str:
    """`faces.importer` unzips web-uploaded archives under MEDIA_ROOT/.import/.

    It is neither of the two kinds this module names, and it is a directory full of
    children's photographs. Nothing should serve it.
    """
    target = media_root() / ".import" / "batch" / "1-А" / "student_401_1778595343147.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return str(to_relative(target))


# -- the two kinds are separately grantable ----------------------------------


def test_the_gallery_and_the_incident_evidence_are_different_permissions() -> None:
    """The contract, below HTTP. One grant cannot be the answer to both questions."""
    assert Capability.VIEW_BULLYING_MEDIA is not Capability.VIEW_PUPIL_PHOTOS


def test_the_single_media_capability_is_gone() -> None:
    """Not renamed alongside the new ones — REMOVED.

    Leaving `VIEW_MEDIA` in the enum would let a role keep the whole tree while the new
    capabilities looked like the policy, which is the same defect with better vocabulary.
    """
    assert not hasattr(Capability, "VIEW_MEDIA")


# -- today's access is preserved exactly -------------------------------------


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.ADMIN, UserRole.DEVELOPER])
def test_the_roles_that_saw_everything_still_see_everything(
    client_for: ClientFor, role: UserRole
) -> None:
    """A permission split must not quietly become a permission cut. These three hold both
    halves today and must still hold both: the school's headteacher discovering on Monday
    that snapshots 403 is not an acceptable way to learn about a refactor."""
    snapshot, photo = _snapshot(), _pupil_photo()
    client = client_for(role)

    assert client.get(f"/media/{snapshot}").status_code == 200, "lost the incident evidence"
    assert client.get(f"/media/{photo}").status_code == 200, "lost the pupil gallery"


def test_a_canteen_worker_still_reaches_neither(client_for: ClientFor) -> None:
    """§14 as it already stood. Splitting the capability must not open a door."""
    snapshot, photo = _snapshot(), _pupil_photo()
    client = client_for(UserRole.CANTEEN_STAFF)

    assert client.get(f"/media/{snapshot}").status_code == 403
    assert client.get(f"/media/{photo}").status_code == 403


def test_the_canteen_role_holds_neither_half_in_the_table() -> None:
    canteen = capabilities_for(UserRole.CANTEEN_STAFF)

    assert Capability.VIEW_BULLYING_MEDIA not in canteen
    assert Capability.VIEW_PUPIL_PHOTOS not in canteen


def test_every_role_still_states_its_capabilities_in_writing() -> None:
    assert set(ROLE_CAPABILITIES) == set(UserRole)


# -- the split actually holds ------------------------------------------------


def test_holding_only_the_clips_does_not_hand_over_the_gallery(
    client_for: ClientFor,
) -> None:
    """**The reason this exists, and it is no longer hypothetical.** The psychologist §13
    describes is asked about one incident and needs its clip. Under one capability that
    grant also handed them a photograph of every child in the school.

    This used to monkeypatch an OPERATOR down to the incident half, because the
    psychologist role did not exist and `roles.py` refuses to invent roles for features
    nobody has built. The role exists now, so the test asks the real one — which is
    strictly stronger: a future edit that quietly adds `VIEW_PUPIL_PHOTOS` to the
    psychologist fails here, and it could not have failed against a monkeypatched set.
    """
    snapshot, photo = _snapshot(), _pupil_photo()
    client = client_for(UserRole.PSYCHOLOGIST)

    assert client.get(f"/media/{snapshot}").status_code == 200, "cannot see the incident"
    assert client.get(f"/media/{photo}").status_code == 403, (
        "the incident clip grant also handed over the school's photo register"
    )


def test_holding_only_the_gallery_does_not_hand_over_the_incidents(
    client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction: whoever maintains the roster has no business in the evidence."""
    monkeypatch.setitem(
        ROLE_CAPABILITIES, UserRole.OPERATOR, frozenset({Capability.VIEW_PUPIL_PHOTOS})
    )
    snapshot, photo = _snapshot(), _pupil_photo()
    client = client_for(UserRole.OPERATOR)

    assert client.get(f"/media/{photo}").status_code == 200
    assert client.get(f"/media/{snapshot}").status_code == 403


def test_the_permission_follows_the_file_not_the_url(
    client_for: ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`people/../snapshots/x.jpg` is a snapshot whose URL begins `people/`.

    A prefix test on the raw string reads the first segment and grants the gallery half,
    then serves a bullying snapshot. `ensure_within` does not catch it — the path never
    leaves MEDIA_ROOT. The capability has to be decided from the RESOLVED path.

    **The separator is percent-encoded, and that is the whole test.** Written as plain
    `people/../snapshots/x.jpg`, httpx collapses the `..` before the request is sent: the
    server receives `snapshots/x.jpg` and the case never arrives. Measured, not assumed —
    a probe route logged `path='snapshots/x.jpg'` for the plain spelling and
    `path='people/../snapshots/x.jpg'` for this one. The first version of this test used
    the plain spelling, passed, and went on passing with the defence sabotaged, which is
    how it was caught: it was asserting that the CLIENT normalises.

    An attacker does not use httpx. Anything that speaks HTTP down a socket sends the
    bytes it likes, and `%2F` is the reproducible way to make this client do the same.
    """
    monkeypatch.setitem(
        ROLE_CAPABILITIES, UserRole.OPERATOR, frozenset({Capability.VIEW_PUPIL_PHOTOS})
    )
    snapshot = _snapshot().replace("\\", "/")
    client = client_for(UserRole.OPERATOR)

    disguised = "people/..%2F" + snapshot.replace("/", "%2F")

    assert client.get(f"/media/{disguised}").status_code == 403, (
        "a gallery-only grant fetched a bullying snapshot by putting `people/..` in front"
    )


# -- anything the map does not name is refused -------------------------------


def test_a_directory_nobody_classified_is_refused_to_everyone(client_for: ClientFor) -> None:
    """Deny by default, applied to the media tree itself.

    `.import/` is where `faces.importer` unzips uploaded archives — children's photographs,
    mid-import. Under one whole-tree capability it was served to anyone holding it. A new
    directory under MEDIA_ROOT must not become readable by existing grants merely by
    existing; it has to be classified, in writing, like a role.
    """
    staged = _staged_import()

    for role in (UserRole.OPERATOR, UserRole.ADMIN, UserRole.DEVELOPER):
        assert client_for(role).get(f"/media/{staged}").status_code == 403, (
            f"{role.value} fetched a staged import that no capability names"
        )

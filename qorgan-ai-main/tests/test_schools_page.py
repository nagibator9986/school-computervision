"""The schools register: who may open it, what it does, and what it warns about.

**This page had 290 lines of production code and not one behavioural test.** A review
proved what that costs by swapping `require_capability(Capability.MANAGE_SCHOOLS)` for
`Capability.VIEW_CANTEEN` in `web/routes/schools.py` -- making a canteen worker able to
create and rename schools -- and running 565 tests across every plausible mask
(`web|role|auth|capab|school|tenancy|account`). All 565 passed.

That is not a gap in R5. R5 walks the real route table and proves every route DEMANDS a
session, which this one does either way; it has no opinion about WHICH capability was
named. The capability is a one-token decision in a decorator, it is the whole of the
authorisation, and nothing was checking it. So the first test below is the one that
matters: each role that must NOT reach this page is named, and asked.

**MANAGE_SCHOOLS is the widest grant on the installation.** Whoever holds it decides which
tenants exist. It is deliberately held by `SUPERADMIN` alone -- not by a school's ADMIN,
who could then rename another school, and not by DEVELOPER, which would recreate "the
supplier can always let themselves in" that the audit condemned.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import null, select
from sqlalchemy.orm import Session

from qorgan.db.models import School, User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.roles import ROLE_CAPABILITIES, Capability
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"
PAGE = "/schools"


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/login", data=with_token(client, {"username": username, "password": PASSWORD})
    )
    assert response.status_code == 303, f"{username} could not log in"


@pytest.fixture
def accounts(session: Session) -> Session:
    """One account per role, so every role can be ASKED rather than reasoned about.

    The superadmin has `school_id=null()` -- it belongs to no school, which is the whole
    point of the role and is what `web.security.school_of` refuses to guess at.

    **`null()` and NOT `None`, and this line said `None` until it was measured.**
    `school_key` carries a column default (`_default_school_id`), and SQLAlchemy omits a
    column from the INSERT when its mapped value IS `None`, so the default fired: this
    fixture quietly gave the superadmin the default school -- a row `accounts.create_account`
    can no longer produce. Nothing went red, because every assertion in this file is about
    a CAPABILITY and `/schools` never calls `school_of`. Measured: `None -> 1`,
    `null() -> None`, and on a two-school database `None` RAISES `UndecidedSchool` rather
    than picking one. `tests/test_the_superadmin_can_be_created.py` now asserts the
    invariant itself, through the command that writes it.
    """
    school = session.scalar(select(School.id))
    for role in UserRole:
        session.add(
            User(
                username=role.value,
                password_hash=hash_password(PASSWORD),
                role=role,
                school_id=null() if role is UserRole.SUPERADMIN else school,
            )
        )
    session.commit()
    return session


@pytest.fixture
def client(settings: Settings, accounts: Session) -> Iterator[TestClient]:
    with TestClient(create_app(), follow_redirects=False) as test_client:
        yield test_client


# Every role that is NOT the superadmin. Derived from the enum rather than typed out, so a
# role added later is refused by default here instead of being silently untested.
OTHER_ROLES = tuple(role for role in UserRole if role is not UserRole.SUPERADMIN)


@pytest.mark.parametrize("role", [r.value for r in OTHER_ROLES])
def test_only_the_superadmin_may_open_the_register(client: TestClient, role: str) -> None:
    """The test the review's sabotage proved was missing.

    A canteen worker, an operator, an admin and a developer must all be refused. The
    school's own headteacher is on that list deliberately: MANAGE_SCHOOLS would let one
    school rename another.
    """
    _login(client, role)
    response = client.get(PAGE)

    assert response.status_code == 403, (
        f"{role} opened the schools register (HTTP {response.status_code}). Whoever reaches "
        "this page decides which tenants exist on this installation."
    )


@pytest.mark.parametrize("role", [r.value for r in OTHER_ROLES])
def test_only_the_superadmin_may_create_a_school(client: TestClient, role: str) -> None:
    """Reading is refused above; WRITING is a separate route and gets asked separately."""
    _login(client, role)
    response = client.post(
        PAGE, data=with_token(client, {"slug": "sneaked-in", "name": "Пробная"})
    )

    assert response.status_code == 403, (
        f"{role} created a school (HTTP {response.status_code})"
    )


def test_the_superadmin_may_open_it(client: TestClient) -> None:
    """The control. Without it, a page that 403s at everybody would pass every test above."""
    _login(client, UserRole.SUPERADMIN.value)
    response = client.get(PAGE)

    assert response.status_code == 200, (
        "the superadmin cannot open their own page, so the refusals above are not "
        "authorisation working -- they are the page being broken for everybody."
    )


def test_manage_schools_is_held_by_exactly_one_role() -> None:
    """The capability table itself, asked directly rather than through a page.

    The parametrised tests above would still pass if MANAGE_SCHOOLS were granted to a role
    that has no account in this fixture. This closes that.
    """
    holders = sorted(
        role.value
        for role, granted in ROLE_CAPABILITIES.items()
        if Capability.MANAGE_SCHOOLS in granted
    )
    assert holders == [UserRole.SUPERADMIN.value], (
        f"MANAGE_SCHOOLS is held by {holders}. It decides which tenants exist; §14 gives "
        "'управление школами' to the суперадминистратор and to nobody else."
    )


def test_the_superadmin_holds_no_child_facing_capability() -> None:
    """The boundary the whole role is drawn on, asserted rather than described.

    This is the account that can reach every school, so it must reach the fewest children.
    If a later change grants it VIEW_PUPILS or VIEW_CAMERAS "for convenience", that is the
    moment one login starts seeing twenty schools' corridors.
    """
    child_facing = {
        Capability.VIEW_CAMERAS,
        Capability.VIEW_BULLYING,
        Capability.REVIEW_BULLYING,
        Capability.VIEW_CANTEEN,
        Capability.VIEW_PUPILS,
        Capability.VIEW_PUPIL_PHOTOS,
        Capability.VIEW_BULLYING_MEDIA,
        Capability.VIEW_LESSON_METRICS,
        Capability.MERGE_PERSONS,
    }
    granted = ROLE_CAPABILITIES[UserRole.SUPERADMIN] & child_facing

    assert granted == set(), (
        f"the superadmin now holds {sorted(c.value for c in granted)}. That is the one "
        "account on the installation able to reach every school."
    )


def test_a_school_can_be_created_and_renamed(client: TestClient, session: Session) -> None:
    """The feature itself, through the form a person actually submits."""
    _login(client, UserRole.SUPERADMIN.value)

    created = client.post(
        PAGE, data=with_token(client, {"slug": "gymnasium-4", "name": "Гимназия №4"})
    )
    assert created.status_code == 303
    school = session.scalar(select(School).where(School.slug == "gymnasium-4"))
    assert school is not None, "the form did not create the school"

    renamed = client.post(
        f"{PAGE}/{school.id}/name",
        data=with_token(client, {"name": "Гимназия №4 имени Абая"}),
    )
    assert renamed.status_code == 303
    session.refresh(school)
    assert school.name == "Гимназия №4 имени Абая"
    assert school.slug == "gymnasium-4", "the slug must not be editable; commands carry it"


@pytest.mark.parametrize(
    ("slug", "why"),
    [
        ("Школа", "cyrillic - a slug is carried by a command line and a URL"),
        ("ab", "too short"),
        ("-leading", "must start alphanumeric"),
        ("default", "already taken by the row migration 0009 writes"),
    ],
)
def test_a_refused_school_renders_the_reason_and_does_not_500(
    client: TestClient, slug: str, why: str
) -> None:
    """A refusal is a sentence on the page, never a stack trace.

    The status matters as much as the message: a 500 is indistinguishable from the system
    being broken, and the next move after an error page is to try again or to ask somebody
    to loosen the check.
    """
    _login(client, UserRole.SUPERADMIN.value)
    response = client.post(PAGE, data=with_token(client, {"slug": slug, "name": "Проба"}))

    assert response.status_code == 400, f"{slug!r} ({why}) gave HTTP {response.status_code}"
    assert "warning" in response.text, "the refusal was not rendered where the person can read it"


def test_the_register_shows_counts_and_never_a_childs_name(
    client: TestClient, session: Session
) -> None:
    """The claim `qorgan/schools.py` opens with, asked instead of trusted.

    A pupil is enrolled with a distinctive name; the page must show that the school has
    one pupil and must not show who they are.
    """
    from qorgan.db.models import Person
    from qorgan.enums import PersonType

    school_id = session.scalar(select(School.id))
    session.add(
        Person(
            school_id=school_id,
            external_id="7",
            full_name="Иванов Пётр",
            person_type=PersonType.STUDENT,
        )
    )
    session.commit()

    _login(client, UserRole.SUPERADMIN.value)
    page = client.get(PAGE)

    assert page.status_code == 200
    assert "Иванов" not in page.text, (
        "the schools register rendered a pupil's NAME. This page is counts only: the "
        "person who administers twenty schools' machines is not somebody twenty schools "
        "have entrusted with their children."
    )


# The three constraints that become wrong the moment a second school exists, each keyed on
# a phrase the warning block must carry. Asserted as text because the point of the block is
# that a HUMAN reads it: a constraint recorded only in a test file is one nobody deploying
# will ever meet, which is the standard this branch already set for the Telegram queue.
DEPLOYMENT_LIMITS = {
    "detection stops installation-wide": "ensure_cameras",
    "cameras/previews/settings are installation-wide": "/preview",
    "alerts all reach one chat": "Telegram",
}


@pytest.mark.parametrize("limit", sorted(DEPLOYMENT_LIMITS))
def test_the_page_warns_about_what_a_second_school_breaks(
    client: TestClient, limit: str
) -> None:
    """The warning has to be on the page that creates the thing it warns about.

    `notify/queue.py`'s exemption demands the constraint stand "where the person adding the
    second school will meet it", and for a long while it did not stand anywhere a person
    would look -- it was a comment in a test file. This is that comment, on the form.
    """
    _login(client, UserRole.SUPERADMIN.value)
    page = client.get(PAGE)

    assert page.status_code == 200
    assert 'id="second-school-warning"' in page.text, (
        "the warning block is gone from the schools page. Somebody creating the second "
        "school now meets no notice that detection will stop installation-wide."
    )
    assert DEPLOYMENT_LIMITS[limit] in page.text, (
        f"the warning no longer mentions {limit!r} (looked for "
        f"{DEPLOYMENT_LIMITS[limit]!r}). If that limit has been FIXED, delete this "
        "parameter and the bullet together; if it has not, put it back."
    )


# The detection bullet does not only WARN, it explains why, and the explanation names the
# CLI. It used to name it wrongly: «Флага --school пока нет» -- there is no --school flag
# yet -- written on the superadmin's own landing page, about a flag that had shipped in the
# same commit and that `qorgan user add --help` was advertising. Nothing caught it, because
# the assertion above looks for the marker `ensure_cameras` and never reads the reason.
#
# So the reason is pinned to the parser, not to a string somebody keeps in step by hand. The
# claim the page makes is "naming a school is something only `qorgan user add` can do", and
# that is exactly `--school`'s location in the built parser. Move the flag, or add it to a
# second command, and this fails by name rather than leaving the page quietly false.
#
# Five commands now, not one. What has NOT changed is what the bullet rests on: none of the
# five is a camera. A worker registering one still has nothing to name a school with.
SCHOOL_FLAG_IS_ON = [("classvision", "attest"), ("classvision", "demo"),
                     ("classvision", "frames"), ("classvision", "import"), ("user", "add")]


def test_the_warning_names_the_school_flag_the_cli_actually_has(client: TestClient) -> None:
    """The reason clause on the detection bullet, checked against `build_parser()`.

    Both halves, because either alone is satisfiable while the page lies: the parser half
    would pass with the page saying anything at all, and the page half would pass with the
    flag deleted from the CLI entirely.
    """
    from tests.prose_scan import NODES

    offering = sorted(path for path, (flags, _) in NODES.items() if "--school" in flags)
    assert offering == SCHOOL_FLAG_IS_ON, (
        f"`--school` is offered by {offering}, not by {SCHOOL_FLAG_IS_ON}. The schools page "
        "lists by name the commands that can name a school, and that list is now wrong. Fix "
        "the page with the flag, not afterwards -- last time it was the other way round and "
        "the page spent a commit denying a flag that existed. If the new command registers "
        "a CAMERA, the bullet's whole argument has changed and the prose needs rewriting, "
        "not extending."
    )

    _login(client, UserRole.SUPERADMIN.value)
    page = client.get(PAGE)

    assert page.status_code == 200
    assert "qorgan user add --school" in page.text, (
        "the detection warning no longer names `qorgan user add --school` as the way a "
        "school is named. That bullet explains WHY detection stops installation-wide; "
        "without the reason it is a warning with no remedy attached."
    )


def _ways_the_config_could_name_a_school() -> list[str]:
    """Every route by which the camera configuration might learn about schools.

    A FIELD on a camera model is only one of them, and not the likeliest. Splitting the
    YAML per school -- `config/schools/<slug>/cameras.yaml`, with `load_cameras(school)` --
    scopes the configuration completely while no model ever grows a school field, because
    the school is carried by where the FILE is. On that path a field-only check stays
    green, the school count is still greater than one, and all five pages go on refusing
    for no reason: a camera wall switched off on a system whose job is watching corridors.

    So the loader signatures, the settings and the shape of the config tree are triggers
    too. Returns the reasons found, so the caller can name them.
    """
    return sorted(
        _camera_model_school_fields()
        + _loader_school_parameters()
        + _settings_that_identify_a_school()
        + _config_dirs_named_after_a_school()
    )


def _camera_model_school_fields() -> list[str]:
    """A school field on any member of the `CameraConfig` union.

    `CameraConfig` is `Annotated[A | B | C, Field(discriminator=...)]`, so the first
    `get_args` unwraps the Annotated and the second splits the union. Read off the union
    rather than listed by hand, so a fourth camera kind is covered automatically.
    """
    from typing import get_args

    from qorgan.config.camera import CameraConfig

    members = get_args(get_args(CameraConfig)[0])
    assert members, (
        "could not read the members of the CameraConfig union, so this check is blind and "
        "would pass whatever the config layer grew. Fix the introspection, not the assert."
    )
    return [
        f"{model.__name__}.{field}"
        for model in members
        for field in model.model_fields
        if "school" in field or "tenant" in field
    ]


def _loader_school_parameters() -> list[str]:
    """A school argument reaching either function that reads the camera configuration."""
    from inspect import signature

    from qorgan.config.loader import load_cameras
    from qorgan.config.provenance import camera_views

    return [
        f"{loader.__name__}({param})"
        for loader in (load_cameras, camera_views)
        for param in signature(loader).parameters
        if "school" in param or "tenant" in param
    ]


# What makes a setting a TENANCY setting rather than a fact about the one school served.
# `Settings.school_timezone` already exists and is the second kind: the wall clock of this
# installation's school, which every deployment has and which scopes nothing. Matching
# "school" alone would make this check permanently red, which is worse than not having it --
# so the name must also look like an IDENTIFIER.
#
# **KNOWN HAZARD, LEFT AS IT IS ON PURPOSE.** These are substrings, so `"id"` would also
# match a future `school_holidays` ("hol-ID-ays") and make this check permanently red on a
# setting that scopes nothing -- the same shape of false positive `school_timezone` already
# caused. It is not fixed today because the function returns an empty list today, so there
# is nothing to be wrong about yet, and anchoring these at word parts would need a
# convention for reading snake_case that no other check here needs. Whoever adds a
# `school_*` setting that is not a tenancy key should split on "_" and compare whole
# segments instead of widening the exclusions one name at a time.
SCHOOL_IDENTITY_HINTS = ("slug", "id", "key", "code", "ref")


def _settings_that_identify_a_school() -> list[str]:
    """A school-identifying SETTING, with the loaders untouched.

    The config would then be scoped by whichever school the process was started for, while
    a signature check stays green. This is the blind spot I named as uncovered and judged
    unlikely; a review demonstrated it passes, so it is covered rather than argued about.
    """
    return [
        f"Settings.{field}"
        for field in Settings.model_fields
        if ("school" in field or "tenant" in field)
        and any(hint in field for hint in SCHOOL_IDENTITY_HINTS)
    ]


def _config_dirs_named_after_a_school() -> list[str]:
    """Config subdirectories whose name is a school's slug -- the per-school YAML layout.

    Two rules, and BOTH are needed. The literal one came first, the slug-matching one was
    added later, and adding it while dropping the literal one was a trade rather than an
    extension: `config/schools/` is not itself slug-named -- the slug is one level down --
    so scanning only `config_dir` and `config_dir/cameras` for slug-named children left a
    real `config/schools/default/cameras.yaml` GREEN. That is the layout three docstrings
    in this branch call the likeliest, so the trade lost precisely the documented path.

    Slug matching is what keeps `profiles/`, `cameras/` and the rest from looking like
    tenancy; the literal check is what catches the layout whose slugs are nested.

    **RAISES when there is no database to ask, and that is deliberate.** The first version
    swallowed the error and returned `[]`, so the whole directory trigger was a silent
    no-op: the caller had no `session` fixture, there was no database, and creating a real
    `config/cameras/default/` left the check GREEN. Found by sabotaging it. A trigger that
    cannot tell "nothing found" from "could not look" is not a trigger, so the caller is
    made to supply a database instead.
    """
    from qorgan.db.engine import session_scope
    from qorgan.settings import get_settings

    with session_scope() as db:
        slugs = {slug for (slug,) in db.execute(select(School.slug)).all()}
    assert slugs, (
        "no schools in the database, so this check cannot recognise a per-school config "
        "directory by name. Give the calling test the `session` fixture."
    )

    config_dir = Path(get_settings().config_dir)
    found = ["config/schools/ exists"] if (config_dir / "schools").is_dir() else []
    return found + [
        f"{root.name}/{child.name}/ is named after a school"
        for root in (config_dir, config_dir / "cameras", config_dir / "schools")
        if root.is_dir()
        for child in root.iterdir()
        if child.is_dir() and child.name in slugs
    ]


def test_the_camera_config_still_has_no_school_and_so_the_warning_still_applies(
    session: Session,
) -> None:
    """The expiry condition for the second bullet, as a check rather than a note.

    The config layer has no school dimension, so the five surfaces built on it refuse
    outright once a second school exists (`web/config_scope.py`). The day the config CAN
    name a school, that refusal and this warning both become wrong -- and this is what goes
    red to say so, rather than the notice on the page quietly starting to lie.

    Takes `session` because one of the four triggers needs a database: recognising a
    per-school config DIRECTORY means comparing its name against the real `School.slug`
    values. Without it that trigger was a silent no-op -- creating a real
    `config/cameras/default/` left this test green, which a sabotage found.
    """
    school_ish = _ways_the_config_could_name_a_school()

    assert not school_ish, (
        f"the camera configuration can now name a school: {school_ish}. Scope "
        "web/routes/cameras.py (`/`, `/cameras`, `/api/cameras`, `/preview/{camera}.jpg`) "
        "and web/routes/settings.py by the acting user's school; then delete the refusal "
        "in web/config_scope.py, its five call sites, "
        "tests/test_the_camera_pages_refuse_two_schools.py, the "
        "'cameras/previews/settings are installation-wide' bullet in schools.html and its "
        "parameter above. Until all of that is done, the page is warning about a "
        "limitation that no longer describes the system."
    )

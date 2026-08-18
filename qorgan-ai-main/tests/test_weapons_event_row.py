"""What a weapon alert holds in the `events` table, and why it needed no migration.

**The migration question, answered by measurement rather than by reading.** Three times
this week a migration that rebuilt a table in SQLite cascade-nulled everything referencing
it, at exit code 0; the last measurement put it at eight tables. So the first thing to
establish about `EventType.WEAPON` is whether it needs DDL at all.

It does not, and that is measured here rather than asserted: `event_type` is
`Enum(..., native_enum=False)` with SQLAlchemy 2.0's default `create_constraint=False`, so
the column is a plain `VARCHAR` with **no CHECK constraint naming the permitted values**.
A new member of the Python enum therefore changes no DDL. `test_the_migrated_schema_has_no
_check_constraint_on_event_type` measures exactly that on a database built by the real
migration chain, and `test_a_weapon_row_round_trips_through_the_real_migration_chain`
proves the consequence by writing one and reading it back. If either ever goes red, a
migration IS needed -- and then `migrations/env.py::_suspend_foreign_keys` becomes
load-bearing, because `events` is referenced by `notifications` and by itself.

The rest of this file is about the three bullying-shaped columns whose meaning changes on
a weapon row. A column that means two things is the defect `migrations/0005` exists about.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select

from qorgan.db.engine import build_engine, session_scope
from qorgan.db.models import Camera, Event
from qorgan.db.types import utcnow
from qorgan.detection.geometry import Box
from qorgan.enums import CameraRole, CameraType, EventStatus, EventType, Severity
from qorgan.events.reasons import unpack_reasons
from qorgan.notify.message import REASON_LABELS
from qorgan.settings import Settings
from qorgan.weapons.pipeline import EVIDENCE, WeaponAlert
from qorgan.weapons.store import record_weapon_alert, summarise_weapon
from tests.conftest import REPO_ROOT
from tests.weapons_fixtures import loaded_weights


def _alembic(settings: Settings) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    return config


def _alert(**overrides) -> WeaponAlert:
    fields = {
        "track_id": 4,
        "class_name": "knife",
        "timestamp": 0.0,
        "confidence": 0.82,
        "observations": 3,
        "strong_observations": 2,
        "person_track_id": 7,
        "box": Box(100, 100, 140, 140),
        "reasons": EVIDENCE,
    }
    fields.update(overrides)
    return WeaponAlert(**fields)


def _camera_row() -> int:
    with session_scope() as session:
        camera = Camera(
            name="entrance_frame",
            display_name="Вход — рамка",
            camera_type=CameraType.WEAPONS,
            role=CameraRole.WEAPONS,
            rtsp_host="192.168.1.90",
        )
        session.add(camera)
        session.flush()
        return camera.id


# -- the migration question -------------------------------------------------


def test_the_migrated_schema_has_no_check_constraint_on_event_type(settings: Settings) -> None:
    """**The measurement that says no migration was needed.**

    If this column carried a CHECK naming its permitted values, adding WEAPON would need
    DDL -- and in SQLite that means a table rebuild, which is the operation that
    cascade-nulled eight tables at exit code 0 the last three times it was measured.
    """
    command.upgrade(_alembic(settings), "head")
    engine = build_engine(settings.database_url)
    checks = inspect(engine).get_check_constraints("events")
    columns = {c["name"]: c for c in inspect(engine).get_columns("events")}
    engine.dispose()

    assert not any("event_type" in str(c.get("sqltext", "")) for c in checks), (
        f"event_type is constrained by DDL after all: {checks}. A new EventType now "
        "needs a migration, and a migration on `events` rebuilds a table that "
        "`notifications` and `events` itself reference."
    )
    assert "VARCHAR" in str(columns["event_type"]["type"]).upper()


def test_a_weapon_row_round_trips_through_the_real_migration_chain(settings: Settings) -> None:
    """The consequence, written and read back on a database built by alembic -- not by
    `Base.metadata.create_all`, which is a different artefact and would prove nothing
    about what is on the school's disk."""
    command.upgrade(_alembic(settings), "head")
    camera_id = _camera_row()

    event_id = record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=_alert(),
        weights=loaded_weights(),
        summary_text=summarise_weapon(_alert(), "Вход — рамка"),
        min_observations=3,
        reconfirm_observations=2,
    )

    with session_scope() as session:
        stored = session.get(Event, event_id)
        assert stored.event_type is EventType.WEAPON
        assert stored.status is EventStatus.NEW
        assert stored.severity is Severity.CRITICAL


def test_a_weapon_alert_is_not_a_bullying_row_in_the_database(settings: Settings) -> None:
    """The TYPE on the row, and nothing more than that.

    Renamed. It used to be called `test_weapon_events_do_not_appear_among_the_bullying_
    ones`, which is a claim about `/events` -- and it never fetched `/events`. It writes one
    weapon alert and then asserts that no row has `event_type == BULLYING`, which is true by
    construction and stays true with the page's filter deleted. Measured: with
    `EventType.BULLYING` removed from `web/routes/events.py::_page`, 74 tests across this
    file, `test_events.py`, `test_weapons_ruling.py` and `test_weapons_panel.py` were all
    green. The page-level claim is now made by the test below, which fetches the page.
    """
    command.upgrade(_alembic(settings), "head")
    camera_id = _camera_row()
    record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=_alert(),
        weights=loaded_weights(),
        summary_text="x",
        min_observations=3,
        reconfirm_observations=2,
    )

    with session_scope() as session:
        bullying = session.scalars(
            select(Event).where(Event.event_type == EventType.BULLYING)
        ).all()
        assert bullying == []


def test_a_weapon_alert_is_not_drawn_on_the_bullying_page(settings: Settings) -> None:
    """**`/events` as a browser fetches it, with one row of each kind in the database.**

    Why the filter is not cosmetic: `/events` draws `skeleton_confirmed` as «скелет ✓/✗»,
    and on a weapon row that column is False because there IS no pose tier in that
    pipeline -- so «скелет ✗» beside a possible weapon is a statement nobody made. And
    `VIEW_BULLYING` opens this page while `VIEW_WEAPONS` opens the other, so a weapon row
    here would make the first grant a second door onto the second.

    The pager is checked too. A `total` that counted rows the list does not show would put
    the page count one ahead of the data, which is the shape of defect that survives for
    months because everything looks right on page one.
    """
    command.upgrade(_alembic(settings), "head")
    camera_id = _camera_row()
    record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=_alert(),
        weights=loaded_weights(),
        summary_text="ВОЗМОЖНОЕ ОРУЖИЕ НА СТРАНИЦЕ БУЛЛИНГА",
        min_observations=3,
        reconfirm_observations=2,
    )
    _bullying_row(camera_id, "обычное событие буллинга")

    page = _operator_page(settings, "/events")
    assert "обычное событие буллинга" in page, "the bullying row must still be drawn"
    assert "ВОЗМОЖНОЕ ОРУЖИЕ НА СТРАНИЦЕ БУЛЛИНГА" not in page
    assert "(1)" in page, "the count must be the bullying count, not both rows"


def _bullying_row(camera_id: int, summary: str) -> None:
    """One ordinary bullying event, so the page has something it SHOULD draw.

    Without it this test would pass on an empty page, which is the other way to be green
    about nothing.
    """
    with session_scope() as session:
        session.add(
            Event(
                camera_id=camera_id,
                event_type=EventType.BULLYING,
                occurred_at=utcnow(),
                confidence=0.5,
                candidate_probability=0.5,
                severity=Severity.ALERT,
                summary_text=summary,
                track_ids="1,2",
                status=EventStatus.NEW,
            )
        )


def _operator_page(settings: Settings, path: str) -> str:
    """Fetch `path` as a logged-in operator, through the real login form."""
    from fastapi.testclient import TestClient

    from qorgan.db.models import User
    from qorgan.enums import UserRole
    from qorgan.passwords import hash_password
    from qorgan.web.app import create_app
    from tests.web_login import with_token

    del settings
    password = "correct-horse-battery"
    with session_scope() as session:
        session.add(
            User(
                username="operator1",
                password_hash=hash_password(password),
                role=UserRole.OPERATOR,
            )
        )

    with TestClient(create_app(), follow_redirects=False) as client:
        login = client.post(
            "/login", data=with_token(client, {"username": "operator1", "password": password})
        )
        assert login.status_code == 303, "login failed; nothing below tests anything"
        response = client.get(path)
        assert response.status_code == 200
        return response.text


# -- what the bullying-shaped columns hold on a weapon row -----------------


def _write(settings: Settings, **overrides) -> Event:
    command.upgrade(_alembic(settings), "head")
    camera_id = _camera_row()
    event_id = record_weapon_alert(
        camera_id=camera_id,
        occurred_at=utcnow(),
        alert=_alert(**overrides),
        weights=loaded_weights(),
        summary_text="x",
        min_observations=3,
        reconfirm_observations=2,
    )
    with session_scope() as session:
        session.expunge_all()
        return session.get(Event, event_id)


def test_the_two_gate_columns_hold_how_much_of_each_gate_was_cleared(
    settings: Settings,
) -> None:
    row = _write(settings, observations=3, strong_observations=2)
    assert row.candidate_probability == 1.0
    assert row.validation_score == 1.0


def test_a_gate_fraction_is_capped_at_one(settings: Settings) -> None:
    """These columns read as probabilities on every other row, and a 2.3 in one of them is
    a number somebody will average with the others one day."""
    row = _write(settings, observations=30, strong_observations=20)
    assert row.candidate_probability == 1.0
    assert row.validation_score == 1.0


def test_severity_is_always_critical_and_that_is_about_speed_not_certainty(
    settings: Settings,
) -> None:
    """How SURE the system is lives in `confidence` and in the wording of the summary."""
    row = _write(settings, confidence=0.36)
    assert row.severity is Severity.CRITICAL
    assert row.confidence == 0.36


def test_the_row_names_the_track_and_the_person_beside_it(settings: Settings) -> None:
    """An operator opening the clip is looking for one of these two numbers."""
    row = _write(settings, track_id=11, person_track_id=42)
    assert row.track_ids == "11,42"


def test_the_reasons_carry_the_three_gates_and_the_weights(settings: Settings) -> None:
    reasons = unpack_reasons(_write(settings).reasons)
    assert set(EVIDENCE) <= set(reasons)
    assert f"weights:{loaded_weights().file.fingerprint}" in reasons


def test_the_provenance_slug_survives_the_reason_packing(settings: Settings) -> None:
    """`pack_reasons` refuses a comma, so the fingerprint has to be hex and the token
    fixed. A slug that silently vanished would take the answer to "which model said
    this?" with it."""
    packed = _write(settings).reasons
    assert "," in packed, "the packing is comma separated; the slugs must not contain one"
    assert len([r for r in unpack_reasons(packed) if r.startswith("weights:")]) == 1


# -- the words that reach a phone ------------------------------------------


def test_every_gate_slug_has_a_russian_label_for_the_notifier() -> None:
    """One table and two producers, on purpose: `describe_reasons` reads a row and cannot
    know which pipeline wrote it, so a second lookup would be a second place to forget --
    and the failure mode is a phone message full of `weapon_near_a_person`."""
    for slug in EVIDENCE:
        assert slug in REASON_LABELS, slug
        assert REASON_LABELS[slug].strip()


def test_the_weights_slug_is_deliberately_not_translated() -> None:
    """It is a hex identifier. A made-up Russian phrase for it would be worse than hex."""
    assert not any(key.startswith("weights:") for key in REASON_LABELS)


def test_the_summary_names_the_weapon_in_russian() -> None:
    assert "нож" in summarise_weapon(_alert(), "Вход")
    assert "огнестрельное оружие" in summarise_weapon(_alert(class_name="firearm"), "Вход")


def test_a_class_with_no_russian_word_still_reaches_a_person() -> None:
    """Ugly beats silent."""
    assert "chainsaw" in summarise_weapon(_alert(class_name="chainsaw"), "Вход")

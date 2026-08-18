"""What `/weapons` tells a human, walked the way a browser walks it.

Two §12.1 requirements land on this page and neither is decoration.

**«Панель обязана показывать, какие веса загружены и на чём они проверены.»** A module
that cannot say what it is running is a module nobody can audit. The client's `best.pt`
was 0 bytes for months and no screen anywhere would have said so, so the test that matters
here is `test_a_zero_byte_model_is_the_loudest_thing_on_the_page`.

**The honest limit, per camera.** The client answered on 2026-07-29 that a camera goes at
the entrance specifically so the object is large -- *and the other cameras stay in play*.
So the page carries one feasibility row per camera and no fleet summary. This is the same
failure face recognition had: it "worked", and then somebody measured the corridor and
found 14 970 faces at a median of 11.5 px and zero recognitions. The arithmetic existed
then too. It was in a report nobody ran.

**These tests fetch the page and read what is on it.** Not the row builders: a green test
over `_reach_row` would say nothing about whether the template draws it, and this suite has
already been bitten by exactly that -- a test asserting an event could be reassigned passed
while the page had stopped drawing the button.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from qorgan.db.models import User
from qorgan.enums import UserRole
from qorgan.passwords import hash_password
from qorgan.settings import Settings, override_settings
from qorgan.web.app import create_app
from tests.conftest import CONFIG_DIR
from tests.weapons_fixtures import config_dir_with, plausible_weights, weapons_camera_dict
from tests.web_login import with_token

PASSWORD = "correct-horse-battery"

# Two cameras that disagree, which is the whole point of the block under test.
#
#   entrance: 1280 px wide, a 45° lens -- the camera the client is putting in on purpose
#   corridor:  960 px wide, a 104° wide-angle -- the cameras that "stay in play"
#
# Their answers are computed by hand in `test_the_two_cameras_give_different_answers`, so
# the page is checked against arithmetic done somewhere other than the code under test.
ENTRANCE = "entrance_frame"
CORRIDOR = "corridor_far"


@pytest.fixture
def weights_file(tmp_path: Path) -> Path:
    return plausible_weights(tmp_path, "qorgan-weapons.pt")


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """The client's artefact, reproduced: a file that is there and is 0 bytes."""
    path = tmp_path / "best.pt"
    path.touch()
    return path


@pytest.fixture
def weapons_settings(
    settings: Settings, tmp_path: Path, weights_file: Path, empty_file: Path
) -> Iterator[Settings]:
    entrance = weapons_camera_dict(
        name=ENTRANCE,
        display_name="Вход — рамка",
        capture={"frame_width": 1280, "frame_height": 720},
        weapons={
            "model": {"model": str(weights_file), "evaluated_on": "OD-Weapons v3, 2026-07"},
            "target_classes": ["knife", "firearm"],
            "lens_hfov_degrees": 45.0,
        },
    )
    corridor = weapons_camera_dict(
        name=CORRIDOR,
        display_name="Коридор 2 этаж",
        weapons={"model": {"model": str(empty_file)}, "lens_hfov_degrees": 104.0},
    )
    value = settings.model_copy(
        update={"config_dir": config_dir_with(tmp_path, entrance, corridor)}
    )
    override_settings(value)
    yield value


@pytest.fixture
def client(weapons_settings: Settings, session: Session) -> Iterator[TestClient]:
    """Logged in as an operator, through the real login form, like a browser."""
    del weapons_settings
    session.add(
        User(username="operator1", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR)
    )
    session.commit()

    app = create_app()
    with TestClient(app, follow_redirects=False) as test_client:
        response = test_client.post(
            "/login", data=with_token(test_client, {"username": "operator1", "password": PASSWORD})
        )
        assert response.status_code == 303, "login failed; nothing below tests anything"
        yield test_client


@pytest.fixture
def page(client: TestClient) -> str:
    response = client.get("/weapons")
    assert response.status_code == 200
    return response.text


# -- «какие веса загружены и на чём они проверены» -------------------------


def test_the_page_names_the_weights_file_that_would_run(page: str, weights_file: Path) -> None:
    assert str(weights_file) in page


def test_the_page_shows_a_fingerprint_of_the_bytes_not_just_a_name(
    page: str, weights_file: Path
) -> None:
    """Every Ultralytics training run in the world calls its output `best.pt`, so the file
    name cannot answer "are these the weights we tested?"."""
    from qorgan.weapons.weights import inspect_weights_file

    assert inspect_weights_file(weights_file).fingerprint in page


def test_the_page_says_what_the_weights_were_evaluated_on(page: str) -> None:
    assert "OD-Weapons v3, 2026-07" in page


def test_an_unevaluated_model_says_so_rather_than_leaving_a_blank(page: str) -> None:
    """An unevaluated model that admits it is honest; a blank cell reads as an
    endorsement, and there is no way to tell the two apart afterwards."""
    assert "НЕ УКАЗАНО" in page


def test_a_zero_byte_model_is_the_loudest_thing_on_the_page(page: str, empty_file: Path) -> None:
    """**The failure that was invisible for months, made impossible to miss.**

    `Path.is_file()` is True for this file. The previous system's check passed on it.
    """
    assert str(empty_file) in page, "name the file somebody has to fix"
    assert "0 bytes" in page, "and say what is wrong with it"
    assert "Веса непригодны" in page
    assert "не запустится" in page, "and that the module will not run on that camera"


def test_the_problem_row_is_marked_critical_and_not_merely_present(page: str) -> None:
    """It is rendered in the page's alarm styling. A true statement in grey text at the
    bottom of a table is how the 0-byte model stayed invisible."""
    marker = page[page.index("Веса непригодны") - 200 : page.index("Веса непригодны")]
    assert "critical" in marker


# -- the honest limit, per camera -----------------------------------------


def test_every_weapons_camera_gets_its_own_feasibility_row(page: str) -> None:
    """One row each and no summary line. A module that reports itself by its best camera
    is the module that said face recognition worked."""
    assert page.count(ENTRANCE) >= 1 and page.count(CORRIDOR) >= 1
    assert "Что каждая камера физически способна увидеть" in page


def test_the_two_cameras_give_different_answers(page: str) -> None:
    """Hand-computed, so the page is checked against arithmetic done elsewhere.

    A 20 cm object, pinhole, at the width the WORKER analyses:

      entrance  1280 px, 45°:  span/m = 2*tan(22.5°) = 0.8284
                max distance = 1280*0.2 / (24 * 0.8284) = 12.9 m
      corridor   960 px, 104°: span/m = 2*tan(52°)    = 2.5599
                max distance =  960*0.2 / (24 * 2.5599) =  3.1 m

    Four times the reach for the camera that was put in on purpose. That difference is
    the thing a human has to see, and it is why there is no fleet-wide number.
    """
    assert "12.9" in page
    assert "3.1" in page


def test_the_far_end_of_the_corridor_is_marked_as_never(page: str) -> None:
    """15 m: 21 px at the entrance camera and 5 px at the corridor one, against a 24 px
    gate. **Neither camera can do it**, and the page says so rather than staying quiet."""
    assert "не сработает" in page
    assert page.count("не сработает") == 2, "both cameras fail at 15 m; both must say so"


def test_the_page_refuses_to_suggest_lowering_the_gate(page: str) -> None:
    """The answer to a "no" is to move the camera. There is nothing under the gate to
    recover, and `identity/cli.py` says the same thing about the face gate."""
    assert "под порогом ничего нет" in page


def test_a_pass_is_not_reported_as_a_promise(page: str) -> None:
    """This models optics and nothing else -- not motion blur, not substream compression,
    not whether the blade is edge-on, and certainly not how good any weights are."""
    assert "необходимое условие, а не обещание" in page


def test_a_lens_nobody_checked_is_labelled_an_assumption(
    client: TestClient, settings: Settings, tmp_path: Path, weights_file: Path
) -> None:
    """The default 78° is CHOSEN, not measured, and the answer moves a long way with it.

    Both cameras above state their lens, so neither is labelled. This one does not.
    """
    camera = weapons_camera_dict(
        name="lens_unknown", weapons={"model": {"model": str(weights_file)}}
    )
    override_settings(
        settings.model_copy(update={"config_dir": config_dir_with(tmp_path / "b", camera)})
    )
    page = client.get("/weapons").text
    assert "ПРЕДПОЛОЖЕНИЕ" in page


# -- the states that must not look like a working module ------------------


def test_no_weapons_camera_says_so_instead_of_showing_an_empty_table(
    client: TestClient, settings: Settings
) -> None:
    """**The school's real config today.** There is no weapons camera in
    `config/cameras/`, so this is the state the panel is actually in right now.

    An empty page must read as "the module runs nowhere", never as "nothing has been
    found". And there must be no feasibility table either: a reach table with no rows
    would be a page claiming to have measured ten cameras it has never seen.
    """
    override_settings(settings.model_copy(update={"config_dir": CONFIG_DIR}))
    page = client.get("/weapons").text

    assert "не работает ни на одной камере" in page
    assert "Что каждая камера физически способна увидеть" not in page


def test_an_empty_alert_list_does_not_read_as_an_all_clear(page: str) -> None:
    """«Тревог нет» and «оружия нет» are different statements, and only one is true here.

    The wording was STRENGTHENED after review. The old text said that a camera without
    usable weights does not start — which invited the reader to treat a GREEN weights row as
    «смотрит», and a green row only ever meant "a plausible file is on disk". So the empty
    state now negates the all-clear explicitly and names what a green row does not prove.
    """
    assert "Тревог об оружии нет" in page
    assert "Это НЕ значит «оружия нет»" in page
    assert "не смотрит" in page
    assert "файл на диске правдоподобен" in page, (
        "the empty state must say what a green weights row actually means, because that is "
        "the row the reader looks at next"
    )


# -- a file on disk is not a running module -------------------------------


def test_a_camera_with_no_worker_is_not_reported_as_working(page: str) -> None:
    """**The state that used to render as an ordinary row.**

    Nothing in this test's fixtures runs a worker, so no `worker_heartbeats` row exists for
    the group these cameras belong to. That is the same shape as a dead worker and as a
    case-3 crash loop — weights that clear the size gate and then fail to LOAD — and in all
    three the page used to draw a size, a fingerprint and no warning at all.
    """
    assert "Модуль на этой камере сейчас НЕ РАБОТАЕТ" in page


def test_a_disabled_camera_says_nobody_is_watching_it(
    client: TestClient, settings: Settings, tmp_path: Path, weights_file: Path
) -> None:
    """`enabled: false` is a camera nobody opens, and the panel said nothing about it.

    Perfectly good weights, a perfectly good reach row, and no worker will ever read a frame
    from it. There was no test for `enabled` anywhere in the weapons files.
    """
    camera = weapons_camera_dict(
        name="switched_off", enabled=False, weapons={"model": {"model": str(weights_file)}}
    )
    override_settings(
        settings.model_copy(update={"config_dir": config_dir_with(tmp_path / "off", camera)})
    )
    page = client.get("/weapons").text
    assert "enabled: false" in page
    assert "Модуль на этой камере сейчас НЕ РАБОТАЕТ" in page


@pytest.mark.parametrize(
    "claim", ["модуль работает", "камера работает", "модуль запущен", "всё в порядке"]
)
def test_the_page_never_asserts_that_a_camera_is_working(page: str, claim: str) -> None:
    """There is no positive claim on this page, and that is deliberate.

    The strongest thing it may say about a live worker is that it cannot see inside the
    weights, because it does not open them (R3). A page that said «модуль работает» would be
    the 0-byte dashboard again in a different font.

    Named phrases rather than a scrub of the word «работает»: the page legitimately contains
    «НЕ РАБОТАЕТ», «не работает ни на одной камере», «Работает ли модуль — отсюда не
    подтверждается» and «не срабатывает». A test that stripped those and then searched for a
    substring would be a test about its own string surgery.
    """
    assert claim not in page.lower()


def test_the_declared_confusables_are_shown_as_a_convention_not_a_fact(page: str) -> None:
    """Screen 3 can be dead and silent, and only the CLI can say whether it is.

    The slugs in `confusable_classes` are a naming convention. If the loaded weights emit
    none of them, «нож или ручка» never fires and nothing anywhere says so — measured: with
    weights whose classes are (knife, person, cell phone), the shipped confusables intersect
    to nothing and `refuse_unusable_weights` accepted them silently.
    """
    assert "Спорные классы на этой камере" in page
    assert "ДОГОВОРЁННОСТЬ" in page
    assert "qorgan weapons weights" in page


def test_a_broken_config_does_not_take_down_the_page_that_explains_it(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """On site the laptop shows a browser, not a terminal. The same rule /settings follows."""
    broken = config_dir_with(tmp_path / "c")
    (broken / "cameras" / "hall_left.yaml").write_text("name: [oops\n", encoding="utf-8")
    override_settings(settings.model_copy(update={"config_dir": broken}))

    response = client.get("/weapons")
    assert response.status_code == 200
    assert "Конфигурация не читается" in response.text

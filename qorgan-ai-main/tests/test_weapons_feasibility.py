"""Distance decides, and no threshold changes it. Arithmetic, and what it is allowed to claim.

`qorgan identity camera-report` asks whether a camera can recognise anybody at the
resolution the worker really feeds it. Its answer for this school's hall is no: 14 970
faces, median 11.5 px, zero recognised. That answer existed as arithmetic long before
anybody ran it, and the months in between are the whole reason this module reports the
same thing on a screen (`test_weapons_panel.py`) rather than only in a terminal.

What is asserted here is the arithmetic and its HONESTY -- that a pass is reported as a
necessary condition rather than a promise, that a fail says "move the camera" rather than
"lower the threshold", and that the lens it used is named, because the answer moves a long
way with it and no camera in this repository has ever had its lens measured.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest

from qorgan.cli import build_parser
from qorgan.settings import Settings, override_settings
from qorgan.weapons.cli import REFUSED, UNANSWERED, USABLE
from qorgan.weapons.feasibility import (
    DEFAULT_HFOV_DEGREES,
    DEFAULT_OBJECT_CM,
    apparent_pixels,
    assess,
    max_useful_distance,
)
from tests.weapons_fixtures import config_dir_with, plausible_weights, weapons_camera_dict

GATE = 24.0  # the shipped weapons.min_object_pixels


# -- the pinhole projection ------------------------------------------------


def test_an_object_spanning_the_whole_view_spans_the_whole_frame() -> None:
    """At 1 m a 78° lens sees 2*tan(39°) = 1.6196 m across. An object that wide is the
    frame's whole width."""
    span = 2.0 * math.tan(math.radians(78.0) / 2.0)
    assert apparent_pixels(span, 1.0, 960, 78.0) == pytest.approx(960.0)


def test_twice_as_far_is_half_as_big() -> None:
    near = apparent_pixels(0.2, 3.0, 960, 78.0)
    far = apparent_pixels(0.2, 6.0, 960, 78.0)
    assert far == pytest.approx(near / 2.0)


def test_a_narrower_lens_makes_the_same_object_bigger() -> None:
    """Which is why the lens is asked for and never assumed."""
    wide = apparent_pixels(0.2, 5.0, 960, 104.0)
    narrow = apparent_pixels(0.2, 5.0, 960, 45.0)
    assert narrow > 2 * wide


def test_a_distance_of_zero_is_an_error_and_not_infinity() -> None:
    with pytest.raises(ValueError):
        apparent_pixels(0.2, 0.0, 960, 78.0)


def test_the_max_distance_is_exactly_where_the_gate_is_met() -> None:
    """The round trip: at `max_useful_distance` the object is `min_pixels` across."""
    limit = max_useful_distance(0.2, 960, 78.0, GATE)
    assert apparent_pixels(0.2, limit, 960, 78.0) == pytest.approx(GATE)


def test_a_gate_of_zero_is_an_error(*, gate: float = 0.0) -> None:
    with pytest.raises(ValueError):
        max_useful_distance(0.2, 960, 78.0, gate)


# -- what we told the school -----------------------------------------------


def test_a_knife_at_the_entrance_is_a_hundred_pixel_object() -> None:
    """`docs/questions-for-school.md` §7, checked rather than repeated.

    1280 px (the hall profile's analysis width), the default 78° lens, 1.5 m: a doorway
    or a turnstile.
    """
    pixels = apparent_pixels(DEFAULT_OBJECT_CM / 100.0, 1.5, 1280, DEFAULT_HFOV_DEGREES)
    assert pixels > 100.0
    assert pixels > GATE


def test_the_same_knife_down_a_corridor_can_never_clear_the_gate() -> None:
    """15 m is the far end of the corridor this school's hall cameras look down.

    **Measured here as ~10.5 px, where §7 of the questions to the school says ~15.** The
    conclusion is the same and it is the decisive one -- it is under the 24 px gate by a
    factor of two either way -- but the number in that document is optimistic against
    this arithmetic, and saying so is cheaper than having somebody re-derive it.
    """
    pixels = apparent_pixels(DEFAULT_OBJECT_CM / 100.0, 15.0, 1280, DEFAULT_HFOV_DEGREES)
    assert pixels < GATE
    assert 10.0 < pixels < 11.0


def test_no_confidence_setting_appears_anywhere_in_this_arithmetic() -> None:
    """It is not a threshold problem: the object stops occupying enough pixels to have a
    shape. The function signature is the assertion."""
    import inspect

    parameters = set(inspect.signature(apparent_pixels).parameters)
    assert parameters == {"object_m", "distance_m", "frame_width", "hfov_deg"}


# -- one report per camera -------------------------------------------------


def test_a_report_is_about_one_camera_at_its_own_resolution() -> None:
    report = assess(camera="entrance_frame", frame_width=1280, min_object_pixels=GATE)
    assert report.camera == "entrance_frame"
    assert report.frame_width == 1280


def test_two_cameras_get_two_answers() -> None:
    """The client's answer of 2026-07-29 in one assertion: the entrance camera is placed
    so the object is large, and the other cameras stay in play. They are not the same
    camera and they do not get the same verdict."""
    entrance = assess(camera="a", frame_width=1280, min_object_pixels=GATE, hfov_deg=45.0)
    corridor = assess(camera="b", frame_width=960, min_object_pixels=GATE, hfov_deg=104.0)
    assert entrance.max_useful_distance_m > 4 * corridor.max_useful_distance_m


def test_a_camera_that_can_do_it_nowhere_reports_unusable() -> None:
    """A 320 px substream through a fisheye: nothing at any of the sampled distances."""
    report = assess(camera="fisheye", frame_width=320, min_object_pixels=GATE, hfov_deg=150.0)
    assert report.usable is False
    assert all(not sample.clears_gate for sample in report.samples)


def test_the_summary_names_the_config_key_behind_the_gate() -> None:
    report = assess(camera="a", frame_width=960, min_object_pixels=GATE)
    assert "weapons.min_object_pixels" in report.summary()


def test_the_summary_says_where_the_answer_stops_being_yes() -> None:
    report = assess(camera="a", frame_width=960, min_object_pixels=GATE)
    assert "cannot clear the gate at any confidence" in report.summary()


def test_the_default_distances_are_places_in_a_school() -> None:
    """1.5 m is a doorway, 3 m a lobby, 15 m the far end of a corridor. Not a sweep."""
    report = assess(camera="a", frame_width=960, min_object_pixels=GATE)
    assert [s.distance_m for s in report.samples] == [1.5, 3.0, 5.0, 10.0, 15.0]


# -- the command, and the lens it reads ------------------------------------


@pytest.fixture
def cameras(settings: Settings, tmp_path: Path) -> Iterator[Settings]:
    """Two weapons cameras with DIFFERENT lenses written into their YAML."""
    weights = plausible_weights(tmp_path)
    entrance = weapons_camera_dict(
        name="entrance_frame",
        capture={"frame_width": 1280, "frame_height": 720},
        weapons={"model": {"model": str(weights)}, "lens_hfov_degrees": 45.0},
    )
    corridor = weapons_camera_dict(
        name="corridor_far",
        weapons={"model": {"model": str(weights)}, "lens_hfov_degrees": 104.0},
    )
    value = settings.model_copy(
        update={"config_dir": config_dir_with(tmp_path, entrance, corridor)}
    )
    override_settings(value)
    yield value


def _report(*argv: str) -> int:
    args = build_parser().parse_args(["weapons", "camera-report", *argv])
    return args.func(args)


def test_the_command_reads_this_cameras_own_lens_from_its_config(
    cameras: Settings, capsys
) -> None:
    """R10's subtle half: the key is not merely parsed, it changes the answer.

    Two cameras, two lenses, one command. If `lens_hfov_degrees` were declared and
    ignored, both would print 78°.
    """
    del cameras
    _report("entrance_frame")
    entrance = capsys.readouterr().out
    _report("corridor_far")
    corridor = capsys.readouterr().out

    assert "45°" in entrance
    assert "104°" in corridor


def test_the_entrance_camera_passes_and_says_so_with_an_exit_code(
    cameras: Settings, capsys
) -> None:
    del cameras
    assert _report("entrance_frame") == USABLE
    assert "OK" in capsys.readouterr().out


def test_a_camera_that_cannot_do_it_exits_nonzero(cameras: Settings, capsys) -> None:
    """A script has to be able to tell "this camera cannot do this" from "yes"."""
    del cameras
    assert _report("corridor_far", "--object-cm", "3") == REFUSED
    assert "Move it closer" in capsys.readouterr().err


def test_the_command_refuses_to_answer_by_lowering_the_gate(cameras: Settings, capsys) -> None:
    """There is nothing under the gate to recover, and the message says so instead of
    leaving the obvious wrong fix as the only thing a reader can think of."""
    del cameras
    _report("corridor_far", "--object-cm", "3")
    stderr = capsys.readouterr().err
    assert "Lowering weapons.min_object_pixels does not help" in stderr
    assert "nothing under it to recover" in stderr


def test_an_explicit_lens_overrides_the_configured_one(cameras: Settings, capsys) -> None:
    del cameras
    _report("entrance_frame", "--hfov-deg", "90")
    assert "90°" in capsys.readouterr().out


def test_an_unknown_camera_is_unanswered_rather_than_refused(cameras: Settings) -> None:
    """2, not 1. "I could not answer" and "the answer is no" are different facts and a
    script has to tell them apart -- the legacy discovered its equivalent of a 1 in month
    four, from an event log full of Unknown."""
    del cameras
    assert _report("no_such_camera") == UNANSWERED


def test_a_camera_that_is_not_a_weapons_camera_is_unanswered(cameras: Settings) -> None:
    del cameras
    assert _report("hall_left") == UNANSWERED

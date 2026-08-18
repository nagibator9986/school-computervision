"""Which layer ACTUALLY supplied the value that is in force.

This project's signature disease, in one line: `min_score: 0.50` was declared in
`config/identity.py`, silently overridden to 0.42-0.45 by every canteen profile, and was
therefore in force on ZERO cameras -- while the code went on explaining at length why
0.50 mattered. `config/loader.py::_refuse_a_known_confusion` says it plainly: a default is
not what runs, a YAML file is not what runs, the merged value is.

So the settings page does not show a number. It shows a number AND the file that supplied
it, because a number on its own is exactly what made that mistake invisible for months.

Everything here is a READ. Nothing in this module or the one it tests may write config --
see `test_web_settings.py`, which enforces that as a rule rather than as a habit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from qorgan.config.provenance import (
    LAYER_BASE,
    LAYER_CAMERA,
    LAYER_DEFAULT,
    LAYER_PROFILE,
    CameraView,
    Setting,
    camera_views,
)
from tests.conftest import CONFIG_DIR


@pytest.fixture(scope="module")
def views() -> dict[str, CameraView]:
    """The real fleet config, exactly as the supervisor would resolve it."""
    return {view.name: view for view in camera_views(CONFIG_DIR)}


def _setting(view: CameraView, key: str) -> Setting:
    for section in view.sections:
        for setting in section.settings:
            if setting.key == key:
                return setting
    raise AssertionError(f"{view.name}: no setting {key!r}; the page would not show it at all")


# -- the signature disease ---------------------------------------------------


def test_a_value_a_profile_overrode_names_the_profile_and_not_the_code_default(
    views: dict[str, CameraView],
) -> None:
    """The exact shape of the bug this page exists to make visible.

    `RecognitionPolicy.min_score` is declared in `src/qorgan/config/identity.py`. Every
    canteen profile writes its own, so the declaration is NOT what runs. A page that
    printed the number and stopped would reproduce the months in which nobody could see
    which of the two was in force.
    """
    setting = _setting(views["canteen_entry"], "canteen.entry.recognition.min_score")

    assert setting.layer == LAYER_PROFILE
    assert setting.source == "config/profiles/canteen_entry.yaml", (
        "the page names the wrong file: an operator would edit a value that is overridden"
    )


def test_the_value_shown_is_the_merged_one_not_the_layer_it_was_written_in(
    views: dict[str, CameraView],
) -> None:
    """base.yaml says 960; profiles/hall.yaml says 1280. 1280 is what the detector sees,
    and every px/s threshold in the fleet is denominated in pixels of THAT frame."""
    setting = _setting(views["hall_left"], "capture.frame_width")

    assert setting.value == "1280"
    assert setting.source == "config/profiles/hall.yaml"


# -- each layer names its own file -------------------------------------------


def test_a_value_only_the_camera_file_sets_names_the_camera_file(
    views: dict[str, CameraView],
) -> None:
    setting = _setting(views["hall_left"], "rtsp.host")

    assert setting.layer == LAYER_CAMERA
    assert setting.source == "config/cameras/hall_left.yaml"


def test_all_three_layers_setting_one_key_resolves_to_the_last_one(tmp_path: Path) -> None:
    """The ordering itself, pinned by one key that every layer writes.

    Added after a sabotage run: reversing the search order to base-first broke only ONE of
    the tests around it. The rest looked like tests of the merge and were not -- `rtsp.host`
    is written by the camera file ALONE and `capture.stream_fps` by base alone, so either
    order answers them correctly. Today's fleet happens to contain no key that a camera
    file overrides from its own profile, so the collision is built here rather than hoped
    for: a test that depends on the config staying shaped a particular way stops testing
    the code the day somebody tidies a YAML file.

    base.yaml says 960, profiles/hall.yaml says 1280, and the camera file below says 1024.
    1024 is what the detector sees, and every px/s threshold on this camera is denominated
    in pixels of that frame.
    """
    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    camera = directory / "cameras" / "hall_left.yaml"
    camera.write_text(
        camera.read_text(encoding="utf-8") + "\ncapture:\n  frame_width: 1024\n", encoding="utf-8"
    )

    view = next(v for v in camera_views(directory) if v.name == "hall_left")
    width = _setting(view, "capture.frame_width")

    assert width.value == "1024", "the page shows a width the detector does not use"
    assert width.layer == LAYER_CAMERA
    assert width.source == "config/cameras/hall_left.yaml"
    # ...and the sibling the camera file did NOT override still names the profile, so this
    # is the merge and not a rule that the camera file always wins.
    assert _setting(view, "capture.frame_height").source == "config/profiles/hall.yaml"
    assert _setting(view, "capture.stream_fps").source == "config/base.yaml"


def test_a_fleet_wide_value_names_base_yaml(views: dict[str, CameraView]) -> None:
    setting = _setting(views["hall_left"], "capture.stream_fps")

    assert setting.layer == LAYER_BASE
    assert setting.source == "config/base.yaml"


def test_a_value_nobody_wrote_down_names_the_module_that_declares_it(
    views: dict[str, CameraView],
) -> None:
    """No YAML sets these. "Where is it changed?" is then a .py file, and saying
    `config/base.yaml` would send somebody to a file the key is not in."""
    assert _setting(views["hall_left"], "enabled").layer == LAYER_DEFAULT
    assert _setting(views["hall_left"], "enabled").source == "src/qorgan/config/camera.py"

    gap = _setting(views["canteen_entry"], "canteen.entry.recognition.single_candidate_gap")
    assert gap.layer == LAYER_DEFAULT
    assert gap.source == "src/qorgan/config/identity.py"


def test_every_value_in_force_says_where_it_came_from(views: dict[str, CameraView]) -> None:
    """No blanks. A row whose origin is unknown is a row that teaches nothing, and this
    page's whole claim is that it can answer the question for every value it shows."""
    assert views, "no cameras resolved; this test would pass vacuously"

    unexplained = [
        f"{view.name}: {setting.key}"
        for view in views.values()
        for section in view.sections
        for setting in section.settings
        if not setting.source or setting.layer not in (
            LAYER_BASE,
            LAYER_PROFILE,
            LAYER_CAMERA,
            LAYER_DEFAULT,
        )
    ]
    assert not unexplained, f"values shown with no honest origin: {unexplained[:10]}"


# -- shapes that the merge treats specially ----------------------------------


def test_a_zone_list_is_one_setting_and_not_a_half_inherited_subtree(
    views: dict[str, CameraView],
) -> None:
    """`deep_merge` replaces a list wholesale -- zones are not something you want half
    inherited. Splitting the list into per-element rows on the page would suggest an
    element can be overridden on its own, which is the opposite of what the loader does.
    """
    setting = _setting(views["hall_left"], "bullying.zones.normal_flow")

    assert setting.layer == LAYER_CAMERA
    assert setting.source == "config/cameras/hall_left.yaml"
    assert "0.38" in setting.value, "the zone rectangles are not shown at all"


def test_the_cameras_come_back_in_the_order_the_fleet_is_read_in() -> None:
    """Priority order, the same as `qorgan config validate` and the camera wall. Two
    orderings of one fleet is one more than the school can hold in their head."""
    views = camera_views(CONFIG_DIR)

    assert [v.name for v in views] == [v.name for v in sorted(views, key=lambda v: v.priority)]


# -- secrets ------------------------------------------------------------------


def test_a_credential_that_reached_a_config_value_is_masked(tmp_path: Path) -> None:
    """R4 says no secret lives in YAML, and `test_no_secrets.py` greps for it. This is the
    second layer: if one ever gets in, the page that renders every config value to a
    browser must not be the thing that publishes it.

    The legacy did exactly this -- it drew `rtsp://user:password@host` onto debug JPEGs an
    unauthenticated UI served (audit C-02). Assembled at runtime so this file is not itself
    a credential-shaped string in the repo.
    """
    leaked = "rtsp://admin:" + "hunter2" + "@10.0.0.1"
    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    camera = directory / "cameras" / "hall_left.yaml"
    camera.write_text(
        camera.read_text(encoding="utf-8").replace(
            'location: "Главный холл / центральный вход / левый угол"', f'location: "{leaked}"'
        ),
        encoding="utf-8",
    )

    view = next(v for v in camera_views(directory) if v.name == "hall_left")

    assert "hunter2" not in view.location, "the page would render a camera password"
    assert "hunter2" not in _setting(view, "location").value

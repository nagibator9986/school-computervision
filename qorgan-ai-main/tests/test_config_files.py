"""The shipped configuration must actually load, and say what we think it says."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from qorgan.config.camera import BullyingCamera, CanteenCamera
from qorgan.config.loader import ConfigError, deep_merge, load_cameras, load_workers
from qorgan.enums import CameraRole
from qorgan.rtsp import safe_url
from qorgan.settings import Settings
from tests.conftest import CONFIG_DIR

EXPECTED_ROLES = {
    "hall_left": CameraRole.MAIN_HALL,
    "hall_right": CameraRole.MAIN_HALL,
    "stairs_floor1": CameraRole.STAIRS,
    "stairs_floor2": CameraRole.STAIRS,
    "stairs_floor2_aux": CameraRole.STAIRS,
    "yard_entry": CameraRole.OUTDOOR,
    "canteen_entry": CameraRole.CANTEEN_ENTRY,
    "canteen_exit": CameraRole.CANTEEN_EXIT,
    "canteen_inside_left": CameraRole.CANTEEN_INSIDE,
    "canteen_inside_service": CameraRole.CANTEEN_INSIDE,
}


@pytest.fixture
def cameras(settings: Settings) -> dict[str, BullyingCamera | CanteenCamera]:
    return load_cameras(CONFIG_DIR)


def test_all_ten_cameras_load_with_the_expected_roles(cameras: dict) -> None:
    assert {name: camera.role for name, camera in cameras.items()} == EXPECTED_ROLES


def test_the_spec_headcount_holds(cameras: dict) -> None:
    """6 bullying (hall x2, stairs x3, yard x1) and 4 canteen (entry, exit, 2 inside)."""
    bullying = [c for c in cameras.values() if isinstance(c, BullyingCamera)]
    canteen = [c for c in cameras.values() if isinstance(c, CanteenCamera)]
    assert len(bullying) == 6
    assert len(canteen) == 4


def test_profiles_are_inherited_and_overridden(cameras: dict) -> None:
    hall = cameras["hall_left"]
    assert isinstance(hall, BullyingCamera)
    # From profiles/hall.yaml:
    assert hall.capture.frame_width == 1280
    assert hall.bullying.gates.hall_confirmation.enabled is True
    # From cameras/hall_left.yaml -- the mirror column is unique to this camera:
    assert len(hall.bullying.zones.mirror_ignore) == 1
    assert hall.bullying.zones.normal_flow_alert_threshold == 5.00
    # From base.yaml, untouched by either:
    assert hall.yolo.tracker == "bytetrack.yaml"


def test_a_camera_specific_value_beats_the_profile(cameras: dict) -> None:
    aux = cameras["stairs_floor2_aux"]
    assert isinstance(aux, BullyingCamera)
    assert aux.bullying.metrics.tracker_smoothing == 0.30  # camera file
    assert aux.bullying.gates.staircase_pass.enabled is True  # stairs profile


def test_the_inside_cameras_have_no_burst_stream(cameras: dict) -> None:
    """There is no evidence to capture inside the canteen, only presence."""
    assert cameras["canteen_inside_left"].rtsp.burst_path is None
    assert cameras["canteen_exit"].rtsp.burst_path is not None


def test_the_small_face_path_is_enabled_at_the_entry(cameras: dict) -> None:
    """Younger pupils' faces sit below the size gate. Removing this loses the first-graders."""
    entry = cameras["canteen_entry"]
    assert isinstance(entry, CanteenCamera)
    assert entry.canteen.entry is not None
    assert entry.canteen.entry.small_face.enabled is True
    assert entry.canteen.entry.small_face.min_hits >= 2


def test_no_camera_config_carries_credentials(cameras: dict) -> None:
    for camera in cameras.values():
        url = safe_url(camera.rtsp)
        assert "@" not in url
        assert camera.rtsp.host


def test_every_enabled_camera_is_assigned_to_exactly_one_worker(cameras: dict) -> None:
    """R3: coverage comes from config, never from which browser tab is open."""
    workers = load_workers(cameras, CONFIG_DIR)
    enabled = {name for name, camera in cameras.items() if camera.enabled}
    assert workers.assigned_cameras == enabled
    for name in enabled:
        assert workers.group_for(name) is not None


def test_a_camera_nobody_runs_fails_startup(cameras: dict, tmp_path: Path) -> None:
    """A camera assigned to no worker is a camera nobody is watching. Fail loudly."""
    config_dir = _clone_config(tmp_path)
    (config_dir / "workers.yaml").write_text(
        "groups:\n  - name: only_one\n    cameras: [hall_left]\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="assigned to no worker group"):
        load_workers(cameras, config_dir)


def test_a_worker_group_naming_an_unknown_camera_fails_startup(
    cameras: dict, tmp_path: Path
) -> None:
    config_dir = _clone_config(tmp_path)
    (config_dir / "workers.yaml").write_text(
        "groups:\n"
        "  - name: everything\n"
        f"    cameras: [{', '.join(sorted(cameras))}]\n"
        "  - name: ghost\n"
        "    cameras: [camera_that_does_not_exist]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown camera"):
        load_workers(cameras, config_dir)


def _clone_config(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, destination)
    return destination


def test_deep_merge_replaces_lists_wholesale() -> None:
    """A half-inherited zone list would be a very confusing bug."""
    base = {"zones": {"normal_flow": [1, 2, 3], "keep": 1}}
    merged = deep_merge(base, {"zones": {"normal_flow": [9]}})
    assert merged["zones"]["normal_flow"] == [9]
    assert merged["zones"]["keep"] == 1


# -- the measured floor under min_score, checked where it is actually TRUE ------------
#
# The floor was set in `config/identity.py` (0.50, above the worst measured impostor at
# 0.472) and then quietly overridden back under itself by every canteen profile. It was in
# force on ZERO cameras, and the code went on explaining at length why it mattered.
#
# It was fixed once on `main` and stayed broken on both feature branches, because each
# worktree carries its own copy of `config/profiles/*.yaml`. That is the eighth time this
# codebase has produced the same bug: a value true in one layer and quietly wrong in the
# next, with nothing in between that complains.
#
# So the check does not live on the model (a unit test may legitimately build
# `RecognitionPolicy(min_score=0.1)` to exercise the matching logic -- it is testing a
# decision function, not proposing to run a school on it, and a field validator cannot tell
# the difference). It lives at CONFIG-LOAD time, on the MERGED value, which is the only
# layer that is true.


def test_no_shipped_camera_admits_a_known_confusion(cameras: dict) -> None:
    """THE regression test. It is the one that would have caught eight instances of this.

    Nothing here reads a default or a YAML file: it reads what `load_cameras()` actually
    produced, after base + profile + camera were merged.
    """
    from qorgan.config.identity import MEASURED_IMPOSTOR_CEILING

    offenders = []
    for name, camera in sorted(cameras.items()):
        for where, value in _min_scores(camera):
            if value < MEASURED_IMPOSTOR_CEILING:
                offenders.append(f"{name}.{where} = {value}")

    assert not offenders, (
        f"these thresholds sit BELOW the measured impostor ceiling "
        f"({MEASURED_IMPOSTOR_CEILING}), so they knowingly accept a confusion we have "
        f"already measured in this school's own gallery: {offenders}"
    )


def test_a_profile_below_the_measured_ceiling_refuses_to_LOAD(tmp_path: Path) -> None:
    """A comment is not a check, and a check on the wrong layer is not one either.

    This must fail at LOAD, naming the camera -- not silently produce a camera that
    recognises the wrong child.
    """
    config_dir = _clone_config(tmp_path)
    profile = config_dir / "profiles" / "canteen_exit.yaml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace("min_score: 0.50", "min_score: 0.42", 1),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="0.42"):
        load_cameras(config_dir)


def _min_scores(camera) -> list[tuple[str, float]]:
    """Every effective min_score on one camera, however it got there."""
    canteen = getattr(camera, "canteen", None)
    if canteen is None:
        return []

    found: list[tuple[str, float]] = []
    for role in ("entry", "exit", "inside"):
        block = getattr(canteen, role, None)
        if block is None:
            continue
        for field in ("recognition", "small_face", "soft"):
            policy = getattr(block, field, None)
            if policy is not None and getattr(policy, "enabled", True):
                found.append((f"{role}.{field}", policy.min_score))
    return found


def test_no_source_file_enumerates_which_profiles_override_the_analysis_resolution() -> None:
    """A list of "the profiles that override" is a SECOND source of truth, and it went stale.

    It said `hall` and `canteen_entry`. `canteen_exit` overrides too -- and that is the camera
    which CLOSES meal sessions. The wrong list was sitting in the help text of
    `qorgan identity camera-report`: the command whose entire purpose is that you must not
    trust a quoted resolution, quoting a resolution.

    The fix is not a corrected list. A corrected list goes stale the next time somebody edits
    a YAML file, which is the bug. The fix is NO list: read `capture.frame_width` off the
    camera's own merged config.

    This test fails if anyone writes one again.
    """
    from tests.conftest import SRC_DIR

    overriding = {
        path.stem
        for path in (CONFIG_DIR / "profiles").glob("*.yaml")
        if "frame_width" in path.read_text(encoding="utf-8")
    }
    assert len(overriding) >= 2, "the guard below is meaningless if fewer than two override"

    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Naming TWO OR MORE profiles in one file is an enumeration; naming one (e.g. an
        # example) is not. It is the LIST that rots.
        named = {profile for profile in overriding if f"{profile}.yaml" in text}
        if len(named) >= 2:
            offenders.append(f"{path.relative_to(SRC_DIR)} names {sorted(named)}")

    assert not offenders, (
        "these files enumerate which profiles override the analysis resolution. That list is "
        "a second source of truth and it has already been wrong once, in the help text of the "
        "command built to prevent exactly this. Delete the list; read the camera's config: "
        f"{offenders}"
    )

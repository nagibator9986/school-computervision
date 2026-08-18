"""The config schema must reject what the legacy silently accepted."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qorgan.config.bullying import Confidence, Counters
from qorgan.config.camera import CAMERA_ADAPTER
from qorgan.config.canteen import MealOutcomeRules
from qorgan.config.common import ZoneRect
from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
from qorgan.config.workers import WorkersConfig

MINIMAL_HALL = {
    "camera_type": "bullying",
    "role": "main_hall",
    "name": "hall_left",
    "display_name": "Hall",
    "rtsp": {"host": "10.0.0.1"},
}


def test_unknown_key_is_a_startup_error_not_a_silent_default() -> None:
    """R10. Legacy had 225 unvalidated keys, so a typo just used the default."""
    with pytest.raises(ValidationError, match="proximty_threshold"):
        CAMERA_ADAPTER.validate_python(
            {**MINIMAL_HALL, "bullying": {"metrics": {"proximty_threshold": 160}}}
        )


def test_a_canteen_camera_cannot_carry_a_bullying_block() -> None:
    """Legacy shipped a full bullying scoring block on two canteen cameras."""
    with pytest.raises(ValidationError, match="bullying"):
        CAMERA_ADAPTER.validate_python(
            {
                "camera_type": "canteen",
                "role": "canteen_inside",
                "name": "canteen_inside_left",
                "display_name": "Inside",
                "rtsp": {"host": "10.0.0.1"},
                "canteen": {"inside": {}},
                "bullying": {"metrics": {"proximity_threshold": 160}},
            }
        )


def test_a_canteen_role_must_carry_its_own_block() -> None:
    with pytest.raises(ValidationError, match="requires a canteen.entry block"):
        CAMERA_ADAPTER.validate_python(
            {
                "camera_type": "canteen",
                "role": "canteen_entry",
                "name": "canteen_entry",
                "display_name": "Entry",
                "rtsp": {"host": "10.0.0.1"},
                "canteen": {},
            }
        )


def test_a_canteen_role_must_not_carry_another_roles_block() -> None:
    with pytest.raises(ValidationError, match="must not carry canteen.exit"):
        CAMERA_ADAPTER.validate_python(
            {
                "camera_type": "canteen",
                "role": "canteen_entry",
                "name": "canteen_entry",
                "display_name": "Entry",
                "rtsp": {"host": "10.0.0.1"},
                "canteen": {"entry": {}, "exit": {}},
            }
        )


def test_a_bullying_camera_cannot_take_a_canteen_role() -> None:
    with pytest.raises(ValidationError, match="not a bullying role"):
        CAMERA_ADAPTER.validate_python({**MINIMAL_HALL, "role": "canteen_entry"})


def test_the_skeleton_cap_must_sit_below_the_notify_threshold() -> None:
    """The whole point of the 0.72 cap is that a skeleton-unconfirmed event CANNOT
    reach the Telegram threshold. If someone retunes past that, startup must fail."""
    with pytest.raises(ValidationError, match="strictly below notify_threshold"):
        Confidence(cap_without_skeleton=0.9, notify_threshold=0.85)


def test_the_confidence_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="must equal 1.0"):
        Confidence(candidate_weight=0.7, validation_weight=0.5)


def test_the_default_confidence_block_is_self_consistent() -> None:
    confidence = Confidence()
    assert confidence.cap_without_skeleton < confidence.notify_threshold
    assert confidence.candidate_weight + confidence.validation_weight == 1.0


def test_temporal_window_defaults_to_the_hold_time() -> None:
    """A cross-field derived default, as in the legacy -- but explicit."""
    assert Counters(red_hold_seconds=3.5).temporal_window_seconds == 3.5

    explicit = Counters(red_hold_seconds=3.5, temporal_window_seconds=1.0)
    assert explicit.temporal_window_seconds == 1.0


@pytest.mark.parametrize(
    ("dwell", "expected"),
    [
        (5.0, "not_ate"),
        (19.9, "not_ate"),
        (20.0, "not_ate"),
        (59.9, "not_ate"),
        (60.0, "ate"),
        (600.0, "ate"),
    ],
)
def test_the_meal_ladder_classifies_dwell_time(dwell: float, expected: str) -> None:
    """Two tiers, because the exit guard means classify() never sees a dwell under 30s: a
    session short of ate_at_or_above_seconds did not eat, one at or above it ate. The old
    '<20s came in and left' tier was unreachable and is gone."""
    assert MealOutcomeRules().classify(dwell) == expected


def test_a_soft_accumulator_needs_at_least_two_hits() -> None:
    """One hit is not an accumulator; it is just a lower threshold with extra steps."""
    with pytest.raises(ValidationError):
        SoftAccumulator(min_hits=1)


def test_a_zone_must_be_a_real_rectangle() -> None:
    with pytest.raises(ValidationError, match="x1<x2"):
        ZoneRect(x1=0.8, y1=0.1, x2=0.2, y2=0.5)


def test_a_camera_may_not_be_assigned_to_two_worker_groups() -> None:
    with pytest.raises(ValidationError, match="more than one group"):
        WorkersConfig(
            groups=[
                {"name": "group_one", "cameras": ["hall_left"]},
                {"name": "group_two", "cameras": ["hall_left", "hall_right"]},
            ]
        )


def test_two_worker_groups_may_not_share_a_name() -> None:
    with pytest.raises(ValidationError, match="duplicate group name"):
        WorkersConfig(
            groups=[
                {"name": "group_one", "cameras": ["hall_left"]},
                {"name": "group_one", "cameras": ["hall_right"]},
            ]
        )


# -- the identity config, and the measured floor under min_score ---------------


def test_the_default_min_score_sits_above_the_worst_measured_impostor() -> None:
    """138 real pupils, 9 447 genuine impostor pairs: the WORST scores 0.472.

    The old default was 0.45, which is BELOW that -- margin -0.022. It admitted a known
    confusion. 0.50 sits inside the band that is empty from 0.48 to 0.77 (spec §2.2).
    """
    worst_measured_impostor = 0.472

    assert RecognitionPolicy().min_score > worst_measured_impostor


def test_the_binding_settings_default_to_something_a_queue_can_live_with() -> None:
    """Five children queuing over ten seconds must cost 5 embeddings, not ~200."""
    binding = BindingSettings()

    assert binding.min_face_frames >= 2, "one frame is not corroboration, it is a glance"
    assert binding.max_wait_seconds > 0
    assert binding.max_attempts >= 1


def test_the_identity_config_does_not_drag_in_the_canteen() -> None:
    """THE point of the move. A hall camera needs RecognitionPolicy and must not have to
    import a module about meal sessions to get it (spec §5).

    Both import forms are checked. Reading only `ast.ImportFrom` would let a plain
    `import qorgan.config.canteen` walk straight past the one invariant this whole task
    exists to establish -- a check that looks like coverage and is not.
    """
    import ast

    from tests.conftest import SRC_DIR

    source = (SRC_DIR / "qorgan" / "config" / "identity.py").read_text(encoding="utf-8")

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    offenders = {name for name in imported if name.startswith("qorgan.config.canteen")}
    assert not offenders, (
        f"config/identity.py imports the canteen: {sorted(offenders)}. That coupling is "
        "exactly what made worker/canteen.py the only place in the system that could "
        "recognise a face."
    )

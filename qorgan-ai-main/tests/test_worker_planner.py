"""Sizing the fleet. The measurement needs a GPU; the PLAN does not, so it is tested here."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.config.workers import WorkersConfig
from qorgan.planning.costs import Costs, _cost_of, _lay_out, group_cost, plan_groups

# Measured on the RTX 3050 Laptop. Real numbers, so the test exercises real arithmetic.
COSTS = Costs(context_mb=140.0, yolo_mb=15.0, pose_mb=20.0, insightface_mb=700.0)


def _bullying(name: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": "main_hall",
            "name": name,
            "display_name": name,
            "rtsp": {"host": "10.0.0.1"},
        }
    )


def _canteen(name: str, role: str, block: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "canteen",
            "role": role,
            "name": name,
            "display_name": name,
            "rtsp": {"host": "10.0.0.2"},
            "canteen": {block: {}},
        }
    )


def _fleet() -> dict[str, CameraConfig]:
    return {
        "hall_left": _bullying("hall_left"),
        "hall_right": _bullying("hall_right"),
        "canteen_entry": _canteen("canteen_entry", "canteen_entry", "entry"),
        "canteen_exit": _canteen("canteen_exit", "canteen_exit", "exit"),
        "canteen_inside_left": _canteen("canteen_inside_left", "canteen_inside", "inside"),
    }


# -- the cost model -----------------------------------------------------------


def test_a_canteen_group_pays_for_one_yolo_PER_CAMERA() -> None:
    """**The thing §4.7 says must not be missed.**

    The canteen cameras gained a PersonDetector in §4.4, and Ultralytics keeps its tracker
    state on the model object -- so two cameras sharing one would hand each other's
    children the same track ids. One YOLO per CAMERA. InsightFace is one per PROCESS.
    """
    fleet = _fleet()
    one = group_cost(COSTS, [fleet["canteen_entry"]])
    two = group_cost(COSTS, [fleet["canteen_entry"], fleet["canteen_exit"]])

    assert one == pytest.approx(140.0 + 15.0 + 700.0)
    assert two == pytest.approx(140.0 + 30.0 + 700.0), "the second camera's YOLO was free"


def test_a_bullying_group_pays_for_one_pose_model_and_no_insightface() -> None:
    """The pose model has no per-camera state, so the group shares one. InsightFace is not
    loaded at all -- a hall camera does not recognise faces."""
    fleet = _fleet()
    cost = group_cost(COSTS, [fleet["hall_left"], fleet["hall_right"]])

    assert cost == pytest.approx(140.0 + 30.0 + 20.0)


def test_a_group_with_no_cameras_costs_nothing() -> None:
    assert group_cost(COSTS, []) == 0.0


# -- the plan -----------------------------------------------------------------


def test_a_big_gpu_gets_one_camera_per_process() -> None:
    """The spec asks for one OS process per camera. On a GPU that fits it, that is what it
    gets, and nothing in the code changes."""
    plan = plan_groups(_fleet(), COSTS, total_mb=12288.0)

    assert len(plan.groups) == 5
    assert sorted(c for g in plan.groups for c in g.cameras) == sorted(_fleet())


def test_a_small_gpu_groups_the_canteen_cameras_because_insightface_is_the_wall() -> None:
    """The expensive thing is NOT the CUDA context (~140 MB, models included). It is
    InsightFace buffalo_l, at ~700 MB per instance. Grouping the CANTEEN cameras is what
    makes the fleet fit."""
    plan = plan_groups(_fleet(), COSTS, total_mb=4096.0)

    canteen_groups = [
        g for g in plan.groups if any("canteen" in name for name in g.cameras)
    ]
    assert len(canteen_groups) < 3, "each canteen camera got its own 700 MB InsightFace"
    assert plan.estimated_mb(_fleet()) <= 4096.0 * (1 - plan.headroom)


def test_every_camera_lands_in_exactly_one_group() -> None:
    """A camera nobody runs is a camera nobody is watching."""
    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    assigned = [camera for group in plan.groups for camera in group.cameras]
    assert sorted(assigned) == sorted(fleet)
    assert len(assigned) == len(set(assigned))


def test_a_gpu_too_small_for_even_one_canteen_camera_refuses_to_pretend() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        plan_groups(_fleet(), COSTS, total_mb=512.0)


# -- a fleet of only ONE kind -------------------------------------------------
#
# `range(len(canteen), 0, -1)` is EMPTY when there are no canteen cameras, so a fleet of
# only one kind used to enter neither loop and fall out to the "does not fit" error --
# for a fleet that fits with room to spare. A phased rollout (bullying first, canteen
# later) hits this on day one.


def _bullying_only() -> dict[str, CameraConfig]:
    return {"hall_left": _bullying("hall_left"), "hall_right": _bullying("hall_right")}


def _canteen_only() -> dict[str, CameraConfig]:
    return {
        "canteen_entry": _canteen("canteen_entry", "canteen_entry", "entry"),
        "canteen_exit": _canteen("canteen_exit", "canteen_exit", "exit"),
        "canteen_inside_left": _canteen("canteen_inside_left", "canteen_inside", "inside"),
    }


def test_a_bullying_only_fleet_that_fits_gets_planned() -> None:
    """Zero canteen cameras is a real deployment, not an impossibility. Measured: two
    bullying processes cost 2 x 175 = 350 MB against a 2662 MB budget."""
    fleet = _bullying_only()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    assert [(g.name, g.cameras) for g in plan.groups] == [
        ("bullying_1", ["hall_left"]),
        ("bullying_2", ["hall_right"]),
    ]
    assert plan.estimated_mb(fleet) == pytest.approx(350.0)


def test_a_canteen_only_fleet_that_fits_gets_planned() -> None:
    """Measured: three canteen processes cost 3 x 855 = 2565 MB against a 2662 MB budget.
    It fits one-per-camera, so that is what it gets."""
    fleet = _canteen_only()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    assert [(g.name, g.cameras) for g in plan.groups] == [
        ("canteen_1", ["canteen_entry"]),
        ("canteen_2", ["canteen_exit"]),
        ("canteen_3", ["canteen_inside_left"]),
    ]
    assert plan.estimated_mb(fleet) == pytest.approx(2565.0)


# -- the mixed fleet must not move --------------------------------------------

# Measured on the real RTX 3050 with `scripts/vram_spike.py`, and the real school fleet
# is 6 bullying + 4 canteen. This is the plan the operator gets today; fixing the
# one-kind bug must not change it by a single group.
REAL_COSTS = Costs(context_mb=81.0, yolo_mb=62.0, pose_mb=12.0, insightface_mb=708.0)


def _real_fleet() -> dict[str, CameraConfig]:
    names = [
        "hall_left",
        "hall_right",
        "stairs_floor1",
        "stairs_floor2",
        "stairs_floor2_aux",
        "yard_entry",
    ]
    fleet: dict[str, CameraConfig] = {name: _bullying(name) for name in names}
    fleet["canteen_entry"] = _canteen("canteen_entry", "canteen_entry", "entry")
    fleet["canteen_exit"] = _canteen("canteen_exit", "canteen_exit", "exit")
    fleet["canteen_inside_left"] = _canteen("canteen_inside_left", "canteen_inside", "inside")
    fleet["canteen_inside_service"] = _canteen(
        "canteen_inside_service", "canteen_inside", "inside"
    )
    return fleet


def test_the_real_mixed_fleets_plan_is_unchanged() -> None:
    """Regression pin. The mixed fleet always worked -- which is why nobody noticed the
    one-kind bug -- so it must come out of the fix byte-identical."""
    fleet = _real_fleet()
    plan = plan_groups(fleet, REAL_COSTS, total_mb=8192.0)

    assert [(g.name, g.cameras) for g in plan.groups] == [
        ("bullying_1", ["hall_left"]),
        ("bullying_2", ["hall_right"]),
        ("bullying_3", ["stairs_floor1"]),
        ("bullying_4", ["stairs_floor2"]),
        ("bullying_5", ["stairs_floor2_aux"]),
        ("bullying_6", ["yard_entry"]),
        ("canteen_1", ["canteen_entry"]),
        ("canteen_2", ["canteen_exit"]),
        ("canteen_3", ["canteen_inside_left"]),
        ("canteen_4", ["canteen_inside_service"]),
    ]
    assert plan.estimated_mb(fleet) == pytest.approx(4334.0)


# -- the refusal must survive, and its numbers must be true -------------------


def _reported_mb(message: str) -> float:
    match = re.search(r"needs ([\d.]+) MB", message)
    assert match is not None, f"the error stopped naming a cost: {message}"
    return float(match.group(1))


def test_a_fleet_that_genuinely_does_not_fit_is_still_refused_with_a_TRUE_cost() -> None:
    """The refusal is real domain knowledge and must survive the fix. But the number it
    quotes must be the real cost of the layout it names -- the whole disease here is a
    true-sounding message attached to a condition it does not describe."""
    fleet = _fleet()
    with pytest.raises(ValueError, match="does not fit") as excinfo:
        plan_groups(fleet, COSTS, total_mb=512.0)

    message = str(excinfo.value)
    # The layout it names: "even one process per KIND" -- ALL the bullying cameras in one
    # process, ALL the canteen cameras in one. Measured: 190 + 885 = 1075 MB.
    smallest = _lay_out(
        fleet,
        ["hall_left", "hall_right"],
        1,
        ["canteen_entry", "canteen_exit", "canteen_inside_left"],
        1,
    )
    true_cost = _cost_of(smallest, fleet, COSTS)
    assert true_cost == pytest.approx(1075.0)
    assert _reported_mb(message) == pytest.approx(true_cost, abs=0.5)
    # And that cost really is over the budget it quotes. The old error said 465 MB did
    # not fit in 5325 MB; its own numbers disproved it.
    assert true_cost > 512.0 * (1 - 0.35)
    assert "InsightFace alone is 700 MB" in message


def test_a_one_kind_fleet_that_does_not_fit_is_refused_with_its_own_true_cost() -> None:
    """A bullying-only fleet CAN genuinely not fit -- on a small enough card. It must
    still be refused, and the cost quoted must be the bullying layout's real cost, not a
    number borrowed from a canteen process that this fleet does not have."""
    fleet = _bullying_only()
    # Measured: both hall cameras in one process cost 140 + 2*15 + 20 = 190 MB.
    # 256 MB card -> 166 MB budget. This fleet genuinely does not fit.
    with pytest.raises(ValueError, match="does not fit") as excinfo:
        plan_groups(fleet, COSTS, total_mb=256.0)

    message = str(excinfo.value)
    assert _reported_mb(message) == pytest.approx(190.0, abs=0.5)
    # InsightFace is not in this plan at all. Naming it here would be the same lie in a
    # new hat: a true fact about a component this fleet never loads.
    assert "InsightFace" not in message


# -- the empty fleet ----------------------------------------------------------


def test_an_empty_fleet_is_an_error_not_an_empty_plan() -> None:
    """Zero cameras cannot be PLANNED -- there is nothing to plan. An empty plan is a
    workers.yaml that loads, validates, starts, and watches nothing: a camera nobody runs
    is a camera nobody is watching, and zero cameras nobody runs is the whole school. So
    this is an error, and it names the real condition. Note what it must NOT say: the old
    code reached the "does not fit" refusal here and reported that a 0 MB fleet did not
    fit in a 5325 MB budget."""
    with pytest.raises(ValueError, match="no cameras") as excinfo:
        plan_groups({}, COSTS, total_mb=4096.0)

    assert "does not fit" not in str(excinfo.value)


# -- the file it writes -------------------------------------------------------


def test_the_yaml_it_writes_is_a_valid_workers_config() -> None:
    """It must load through the same schema the supervisor loads. A planner that writes a
    file the system cannot read is worse than no planner."""
    import yaml

    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    parsed = yaml.safe_load(plan.to_yaml(fleet, device_name="NVIDIA GeForce RTX 4070"))
    config = WorkersConfig.model_validate(parsed)

    assert config.assigned_cameras == set(fleet)


def test_the_yaml_records_what_was_measured_and_on_what() -> None:
    """The current file's numbers are sound but they are for the WRONG GPU. A file that
    does not say which GPU it was measured on is how that happens."""
    fleet = _fleet()
    yaml_text = plan_groups(fleet, COSTS, total_mb=4096.0).to_yaml(
        fleet, device_name="NVIDIA GeForce RTX 4070"
    )

    assert "RTX 4070" in yaml_text
    assert "700" in yaml_text  # the InsightFace cost, which is the whole story
    assert "4096" in yaml_text


def test_the_planner_never_writes_a_config_two_groups_could_share_a_camera_in() -> None:
    fleet = _fleet()
    plan = plan_groups(fleet, COSTS, total_mb=4096.0)

    # WorkersConfig raises on a camera in two groups. Prove it would have caught us.
    with pytest.raises(ValidationError, match="more than one group"):
        WorkersConfig(
            # Names must satisfy WorkerGroup.name's own pattern (>= 3 chars) or the
            # field-level check fires first and the message under test never appears.
            groups=[
                {"name": "grp_a", "cameras": ["hall_left"]},
                {"name": "grp_b", "cameras": ["hall_left"]},
            ]
        )
    assert WorkersConfig(groups=[g.model_dump() for g in plan.groups])

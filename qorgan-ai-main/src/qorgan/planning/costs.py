"""What a worker process costs, and how many cameras fit in one. **Pure.**

`config/workers.yaml` was measured on a 4 GB RTX 3050 and under-uses the school's 4070. The
numbers themselves are sound. But they are numbers for the WRONG GPU, and we cannot measure
a GPU we do not have, and we will not guess -- guessing at a number is what this whole
rewrite exists to stop.

So the arithmetic lives here, pure and tested, and `measure.py` supplies the four numbers
it needs by running real processes on the real card.

**The expensive thing is not the CUDA context.** Measured: ~140 MB, models included. It is
**InsightFace buffalo_l, at roughly 700 MB per instance** -- which is the audit's H-12
finding ("up to 5 InsightFace instances in one process") turned into a hard wall: one
instance per PROCESS is just as fatal on a small card. Grouping the canteen cameras is
therefore the whole game.

**And since §4.4 the canteen cameras carry a YOLO too** -- one per CAMERA, not one per
group, because Ultralytics keeps its tracker state on the model object and two cameras
sharing one would hand each other's children the same track ids. A track id is now an
identity, so that is not an optimisation we can take.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qorgan.config.camera import BullyingCamera, CameraConfig, CanteenCamera
from qorgan.config.workers import WorkerGroup

# How much of the card we refuse to plan into. Activations under real load are not in the
# static measurement, and a worker that OOMs at lunchtime is a worker that is not watching.
DEFAULT_HEADROOM = 0.35


@dataclass(frozen=True, slots=True)
class Costs:
    context_mb: float
    yolo_mb: float  # per CAMERA
    pose_mb: float  # per bullying GROUP
    insightface_mb: float  # per canteen GROUP


@dataclass(frozen=True, slots=True)
class Plan:
    groups: tuple[WorkerGroup, ...]
    costs: Costs
    total_mb: float
    headroom: float

    def estimated_mb(self, cameras: Mapping[str, CameraConfig]) -> float:
        return sum(
            group_cost(self.costs, [cameras[name] for name in group.cameras])
            for group in self.groups
        )

    def to_yaml(self, cameras: Mapping[str, CameraConfig], device_name: str) -> str:
        return _render(self, cameras, device_name)


def group_cost(costs: Costs, cameras: Sequence[CameraConfig]) -> float:
    """One process, N cameras. What does it hold?"""
    if not cameras:
        return 0.0

    total = costs.context_mb
    # One YOLO per camera. Not negotiable: the tracker state lives on the model.
    total += costs.yolo_mb * len(cameras)

    if any(isinstance(camera, BullyingCamera) for camera in cameras):
        # No per-camera state, so the group shares one.
        total += costs.pose_mb
    if any(isinstance(camera, CanteenCamera) for camera in cameras):
        # One InsightFace per PROCESS. Ever. This is the wall.
        total += costs.insightface_mb

    return total


def plan_groups(
    cameras: Mapping[str, CameraConfig],
    costs: Costs,
    total_mb: float,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> Plan:
    """One process per camera if the card can take it; otherwise group, canteen first."""
    budget = total_mb * (1.0 - headroom)

    bullying = [n for n, c in cameras.items() if isinstance(c, BullyingCamera)]
    canteen = [n for n, c in cameras.items() if isinstance(c, CanteenCamera)]

    # Zero cameras is not a plan of zero groups, it is a question with no answer. An empty
    # workers.yaml loads, validates, starts, and watches nothing -- and a camera nobody
    # runs is a camera nobody is watching. This used to fall through to the refusal below
    # and report that a 0 MB fleet did not fit a 5325 MB budget, which is a lie about a
    # condition that has nothing to do with the size of the card.
    if not cameras:
        raise ValueError(
            "there are no cameras to plan. Nothing here is too big for the card: the "
            "fleet is empty, and an empty plan is a workers.yaml that watches nothing."
        )

    # `max(..., 1)`: range(0, 0, -1) is EMPTY, so a fleet with zero cameras of one kind
    # used to enter neither loop and fall out to the refusal below -- for a fleet that
    # fits with room to spare. `_chunks` emits no groups for an empty list, so asking for
    # one group of nothing costs nothing and lays out only the kind that is present. With
    # both kinds present, max() changes nothing and the mixed fleet's plan is untouched.
    for canteen_groups in range(max(len(canteen), 1), 0, -1):
        for bullying_groups in range(max(len(bullying), 1), 0, -1):
            groups = _lay_out(cameras, bullying, bullying_groups, canteen, canteen_groups)
            if _cost_of(groups, cameras, costs) <= budget:
                return Plan(
                    groups=tuple(groups), costs=costs, total_mb=total_mb, headroom=headroom
                )

    smallest = _lay_out(cameras, bullying, 1, canteen, 1)
    # InsightFace is the wall, and naming it is the point of this message -- but only for
    # a fleet that actually loads one. Quoting a canteen cost at an operator whose fleet
    # is all bullying cameras is a true number about a component they do not run, which
    # is the same false confidence in a new hat.
    wall = f" InsightFace alone is {costs.insightface_mb:.0f} MB." if canteen else ""
    raise ValueError(
        f"this fleet does not fit: even one process per KIND needs "
        f"{_cost_of(smallest, cameras, costs):.0f} MB, and the budget is {budget:.0f} MB "
        f"({total_mb:.0f} MB less {headroom:.0%} headroom).{wall} Buy a bigger card or "
        "run fewer cameras — do not raise the headroom and hope."
    )


def _lay_out(
    cameras: Mapping[str, CameraConfig],
    bullying: list[str],
    bullying_groups: int,
    canteen: list[str],
    canteen_groups: int,
) -> list[WorkerGroup]:
    groups = [
        WorkerGroup(name=f"bullying_{index + 1}", cameras=chunk)
        for index, chunk in enumerate(_chunks(bullying, bullying_groups))
    ]
    groups += [
        WorkerGroup(name=f"canteen_{index + 1}", cameras=chunk)
        for index, chunk in enumerate(_chunks(canteen, canteen_groups))
    ]
    return groups


def _chunks(names: list[str], count: int) -> list[list[str]]:
    """Split as evenly as possible. Empty groups are never emitted."""
    if not names:
        return []
    count = min(count, len(names))
    size, extra = divmod(len(names), count)
    chunks = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        chunks.append(names[start:end])
        start = end
    return chunks


def _cost_of(
    groups: list[WorkerGroup], cameras: Mapping[str, CameraConfig], costs: Costs
) -> float:
    return sum(
        group_cost(costs, [cameras[name] for name in group.cameras]) for group in groups
    )


# Every key written below must exist on WorkersConfig: extra="forbid" makes a stray one a
# startup error. `heartbeat_interval_seconds` used to be written here and was read by
# nothing; it is gone from both. See tests/test_config_deadkeys.py.
def _render(plan: Plan, cameras: Mapping[str, CameraConfig], device_name: str) -> str:
    estimated = plan.estimated_mb(cameras)
    lines = [
        "# Which cameras run in which worker process.",
        "#",
        f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {device_name} "
        f"({plan.total_mb:.0f} MB).",
        "#",
        "# Do not hand-edit the numbers below without re-measuring. The previous version of",
        "# this file was measured on an RTX 3050 and shipped to a school with a 4070: the",
        "# numbers were correct and they were for the wrong GPU.",
        "#",
        f"#   CUDA context + process       ~{plan.costs.context_mb:.0f} MB",
        f"#   YOLOv8n + ByteTrack          ~{plan.costs.yolo_mb:.0f} MB   PER CAMERA (the "
        "tracker state lives",
        "#                                            on the model; two cameras sharing one",
        "#                                            would swap their children's track ids)",
        f"#   YOLOv8n-pose                 ~{plan.costs.pose_mb:.0f} MB   per bullying group "
        "(no per-camera state)",
        f"#   InsightFace buffalo_l        ~{plan.costs.insightface_mb:.0f} MB   per canteen "
        "group. THIS IS THE WALL.",
        "#",
        f"# This layout: ~{estimated:.0f} MB of {plan.total_mb:.0f} MB, "
        f"{plan.headroom:.0%} held back for activations under load.",
        "#",
        "# What this file does NOT change:",
        "#   - Coverage comes from here, never from which browser tab the operator has open",
        "#     (R3). Legacy analysed the stairs only while somebody was looking at them.",
        "#   - The supervisor restarts a group that dies, and kills one that wedges (R7).",
        "#",
        "# Every enabled camera appears in exactly one group, or startup fails. A camera",
        "# nobody runs is a camera nobody is watching.",
        "",
        "groups:",
    ]
    for group in plan.groups:
        cost = group_cost(plan.costs, [cameras[name] for name in group.cameras])
        lines.append(f"  - name: {group.name}")
        lines.append(f"    cameras: [{', '.join(group.cameras)}]")
        lines.append(f'    device: "{group.device}"')
        lines.append(f"    # ~{cost:.0f} MB")
        lines.append("")

    lines += [
        "restart_backoff_seconds: 2.0",
        "restart_backoff_max_seconds: 60.0",
        "",
    ]
    return "\n".join(lines)

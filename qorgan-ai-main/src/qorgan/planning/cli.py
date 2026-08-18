"""`qorgan plan-workers` — measure this GPU, then write config/workers.yaml for it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qorgan.planning.costs import Costs

WORKERS_PATH = Path("config/workers.yaml")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "plan-workers",
        help="measure this GPU and write config/workers.yaml for it",
        description=(
            "config/workers.yaml was measured on a 4 GB RTX 3050 and under-uses the "
            "school's 4070. The numbers are sound and they are for the WRONG GPU. We "
            "cannot measure a GPU we do not have, and we will not guess -- so this runs "
            "on the machine it will run on, loads exactly what each kind of worker loads "
            "in production, and writes the file. The current config ships as the fallback."
        ),
    )
    parser.add_argument("--out", type=Path, default=WORKERS_PATH)
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.35,
        help="fraction of the card held back for activations under load",
    )
    parser.add_argument("--dry-run", action="store_true", help="print it; do not write it")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config/workers.yaml (default: refuse)",
    )
    parser.set_defaults(func=cmd_plan_workers)


def _measured_on(text: str) -> str | None:
    """Pull the device an existing config/workers.yaml was measured on out of its own
    header comment, if it has one. Best-effort: a hand-edited or foreign file may not."""
    for line in text.splitlines():
        marker = "MEASURED ON: "
        if marker in line:
            after = line.split(marker, 1)[1]
            return after.split(" (", 1)[0].strip()
    return None


def _refuse_to_overwrite(out: Path, new_device: str) -> None:
    """`out` already exists and `--force` was not passed. Refuse loudly rather than
    silently clobbering a real, committed measurement with this machine's numbers --
    the project's signature failure mode, this time self-inflicted."""
    old_device = _measured_on(out.read_text(encoding="utf-8"))
    print(f"{out} already exists and will not be overwritten.", file=sys.stderr)
    if old_device is not None:
        print(
            f"  existing file measured on: {old_device}\n  just measured on:           "
            f"{new_device}",
            file=sys.stderr,
        )
    print(
        "Pass --force to overwrite it with this machine's numbers, or --dry-run to "
        "preview without writing.",
        file=sys.stderr,
    )


def _print_measurement(device: str, total: float, costs: Costs) -> None:
    print(f"device: {device}   total VRAM: {total:.0f} MB")
    print(f"  CUDA context      ~{costs.context_mb:.0f} MB")
    print(f"  YOLOv8n + track   ~{costs.yolo_mb:.0f} MB   per CAMERA")
    print(f"  YOLOv8n-pose      ~{costs.pose_mb:.0f} MB   per bullying group")
    print(f"  InsightFace       ~{costs.insightface_mb:.0f} MB   per canteen group\n")


def cmd_plan_workers(args: argparse.Namespace) -> int:
    from qorgan.config.loader import load_cameras
    from qorgan.planning.costs import plan_groups
    from qorgan.planning.measure import device_name, gpu_total_mb, measure_costs

    cameras = {name: camera for name, camera in load_cameras().items() if camera.enabled}
    if not cameras:
        print("no enabled cameras; nothing to plan", file=sys.stderr)
        return 1

    try:
        costs = measure_costs()
        total = gpu_total_mb()
        device = device_name()
    except (RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"cannot measure this machine: {exc}", file=sys.stderr)
        print(
            f"{args.out} stands unchanged. It is the fallback, and it is honest about "
            "which GPU it was measured on.",
            file=sys.stderr,
        )
        return 1

    _print_measurement(device, total, costs)

    try:
        plan = plan_groups(cameras, costs, total, headroom=args.headroom)
    except ValueError as exc:
        print(f"cannot plan a fleet that fits: {exc}", file=sys.stderr)
        return 1

    text = plan.to_yaml(cameras, device)
    if args.dry_run:
        print(text)
        return 0

    if args.out.exists() and not args.force:
        _refuse_to_overwrite(args.out, device)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.out}: {len(plan.groups)} group(s), "
        f"~{plan.estimated_mb(cameras):.0f} / {total:.0f} MB"
    )
    return 0

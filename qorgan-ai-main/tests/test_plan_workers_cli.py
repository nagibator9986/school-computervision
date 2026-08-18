"""`qorgan plan-workers` writes config/workers.yaml BY DEFAULT -- only ``--dry-run`` used to
suppress that. Run it on a dev box and it silently overwrites the committed fallback with
THAT machine's GPU numbers: a wrong value presented as truth, this project's signature
failure mode.

Write-by-default stays (the school machine is meant to write for real). What this file
locks in is the guard: an EXISTING config/workers.yaml is never overwritten by accident.
Only an explicit ``--force`` may replace it. ``--dry-run`` keeps computing and printing,
never writing, regardless of whether the file already exists.

The GPU measurement itself needs a real CUDA device, so every test here fakes
``load_cameras``, ``measure_costs``, ``gpu_total_mb`` and ``device_name`` -- exactly like
`test_worker_planner.py` fakes the pure planning arithmetic instead of the GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.planning import cli as plan_workers_cli
from qorgan.planning.costs import Costs

COSTS = Costs(context_mb=140.0, yolo_mb=15.0, pose_mb=20.0, insightface_mb=700.0)

OLD_DEVICE = "NVIDIA GeForce RTX 3050 Laptop GPU"
OLD_TOTAL_MB = 4096.0
NEW_DEVICE = "NVIDIA GeForce RTX 4070"
NEW_TOTAL_MB = 12288.0


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


def _canteen(name: str) -> CameraConfig:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "canteen",
            "role": "canteen_entry",
            "name": name,
            "display_name": name,
            "rtsp": {"host": "10.0.0.2"},
            "canteen": {"entry": {}},
        }
    )


def _fleet() -> dict[str, CameraConfig]:
    # One of each kind, the same shape test_worker_planner uses. This comment used to say
    # plan_groups() could not plan an all-bullying (or all-canteen) fleet at all, because
    # it searched canteen-group counts down from len(canteen) and range(0, 0, -1) is
    # empty. That was a bug, not a design; it is fixed, and test_worker_planner pins both
    # one-kind fleets. The fleet stays mixed here because that is what the school runs.
    return {"hall_left": _bullying("hall_left"), "canteen_entry": _canteen("canteen_entry")}


def _fake_measurement(monkeypatch: pytest.MonkeyPatch, *, device: str, total_mb: float) -> None:
    """Stand in for the four things `cmd_plan_workers` measures on a real GPU."""
    import qorgan.config.loader as loader_module
    import qorgan.planning.measure as measure_module

    monkeypatch.setattr(loader_module, "load_cameras", lambda: _fleet())
    monkeypatch.setattr(measure_module, "measure_costs", lambda: COSTS)
    monkeypatch.setattr(measure_module, "gpu_total_mb", lambda: total_mb)
    monkeypatch.setattr(measure_module, "device_name", lambda: device)


def _args(out: Path, *, dry_run: bool = False, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(out=out, headroom=0.35, dry_run=dry_run, force=force)


def test_first_run_writes_when_no_file_exists_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The intended path on the school machine: nothing here yet, so it writes. No guard
    needed, and none should fire."""
    out = tmp_path / "config" / "workers.yaml"
    _fake_measurement(monkeypatch, device=NEW_DEVICE, total_mb=NEW_TOTAL_MB)

    rc = plan_workers_cli.cmd_plan_workers(_args(out))

    assert rc == 0
    assert out.exists()
    assert NEW_DEVICE in out.read_text(encoding="utf-8")


def test_an_existing_file_is_refused_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footgun this guard closes: run on a dev box, and it must NOT silently clobber
    the committed fallback with this machine's numbers."""
    out = tmp_path / "config" / "workers.yaml"
    out.parent.mkdir(parents=True)
    existing_text = f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {OLD_DEVICE} (4096 MB).\n"
    out.write_text(existing_text, encoding="utf-8")

    _fake_measurement(monkeypatch, device=NEW_DEVICE, total_mb=NEW_TOTAL_MB)

    rc = plan_workers_cli.cmd_plan_workers(_args(out))

    assert rc != 0, "an existing config/workers.yaml must not be overwritten silently"
    assert out.read_text(encoding="utf-8") == existing_text, "the existing file must be untouched"


def test_the_refusal_names_the_old_device_the_new_device_and_the_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "config" / "workers.yaml"
    out.parent.mkdir(parents=True)
    out.write_text(
        f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {OLD_DEVICE} (4096 MB).\n",
        encoding="utf-8",
    )

    _fake_measurement(monkeypatch, device=NEW_DEVICE, total_mb=NEW_TOTAL_MB)

    rc = plan_workers_cli.cmd_plan_workers(_args(out))
    message = capsys.readouterr().err

    assert rc != 0
    assert "already exists" in message
    assert OLD_DEVICE in message, "the device the existing file was measured on"
    assert NEW_DEVICE in message, "the device measured just now"
    assert "--force" in message
    assert "--dry-run" in message


def test_force_overwrites_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deliberate re-measure path: --force means "yes, I mean it"."""
    out = tmp_path / "config" / "workers.yaml"
    out.parent.mkdir(parents=True)
    out.write_text(
        f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {OLD_DEVICE} (4096 MB).\n",
        encoding="utf-8",
    )

    _fake_measurement(monkeypatch, device=NEW_DEVICE, total_mb=NEW_TOTAL_MB)

    rc = plan_workers_cli.cmd_plan_workers(_args(out, force=True))

    assert rc == 0
    assert NEW_DEVICE in out.read_text(encoding="utf-8")


def test_dry_run_never_writes_even_when_a_file_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run keeps its current meaning: compute and print, never write -- and that is
    true whether or not config/workers.yaml already exists. It must not trip the guard."""
    out = tmp_path / "config" / "workers.yaml"
    out.parent.mkdir(parents=True)
    existing_text = f"# WRITTEN BY `qorgan plan-workers`. MEASURED ON: {OLD_DEVICE} (4096 MB).\n"
    out.write_text(existing_text, encoding="utf-8")

    _fake_measurement(monkeypatch, device=NEW_DEVICE, total_mb=NEW_TOTAL_MB)

    rc = plan_workers_cli.cmd_plan_workers(_args(out, dry_run=True))

    assert rc == 0
    assert out.read_text(encoding="utf-8") == existing_text, "dry-run must never write"

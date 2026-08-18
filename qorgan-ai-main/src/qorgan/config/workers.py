"""Which cameras run in which worker process.

The spec asks for one OS process per camera. That does not fit this machine: the GPU
is a 4 GB RTX 3050, and ten CUDA contexts (~300-500 MB each on Windows before a single
weight loads) would exhaust it on their own. So the camera-to-process mapping is a
config knob rather than a hardcoded 1:1.

Each worker process loads its models once and runs its cameras as threads. On a larger
GPU you write one camera per group and nothing in the code changes.

What does NOT change: coverage comes from this file, never from which browser tab the
operator has open (rule R3), and the supervisor restarts a dead group (rule R7).
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field, model_validator

from qorgan.config.common import Base


class WorkerGroup(Base):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    cameras: list[str] = Field(min_length=1)
    device: str = "cuda:0"
    # Killed and restarted if it has not published a heartbeat in this long.
    heartbeat_timeout_seconds: float = Field(default=30.0, gt=0)


class WorkersConfig(Base):
    groups: list[WorkerGroup] = Field(min_length=1)
    restart_backoff_seconds: float = Field(default=2.0, gt=0)
    restart_backoff_max_seconds: float = Field(default=60.0, gt=0)
    # heartbeat_interval_seconds DELETED: the worker's cadence is a fixed 5.0 s in
    # worker/entrypoint.py and this key never reached it. `heartbeat_timeout_seconds`
    # above IS live -- the supervisor kills a worker that misses it.

    @model_validator(mode="after")
    def _each_camera_assigned_once(self) -> WorkersConfig:
        names = [group.name for group in self.groups]
        duplicate_groups = sorted({n for n, c in Counter(names).items() if c > 1})
        if duplicate_groups:
            raise ValueError(f"workers: duplicate group name(s): {duplicate_groups}")

        assignments = [camera for group in self.groups for camera in group.cameras]
        duplicate_cameras = sorted({c for c, n in Counter(assignments).items() if n > 1})
        if duplicate_cameras:
            raise ValueError(
                f"workers: camera(s) assigned to more than one group: {duplicate_cameras}"
            )
        return self

    @property
    def assigned_cameras(self) -> set[str]:
        return {camera for group in self.groups for camera in group.cameras}

    def group_for(self, camera: str) -> WorkerGroup | None:
        return next((g for g in self.groups if camera in g.cameras), None)

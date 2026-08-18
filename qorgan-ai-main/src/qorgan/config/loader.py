"""Load and validate the configuration: base.yaml <- profiles/<p>.yaml <- cameras/<c>.yaml.

Defaults live in exactly one place. An unknown key is a startup error, not a silently
applied default -- the legacy had 225 unvalidated keys, so a typo just used the default
and nobody found out for months (rule R10).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from qorgan.config.camera import CAMERA_ADAPTER, CameraConfig
from qorgan.config.workers import WorkersConfig
from qorgan.settings import get_settings, resolve

PROFILE_KEY = "profile"


class ConfigError(Exception):
    """Configuration is wrong. Always names the file that is wrong."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge. A list in the override replaces the base list wholesale --
    zones are not something you want half-inherited."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_camera(config_dir: Path, camera_path: Path, base: dict[str, Any]) -> CameraConfig:
    raw = _read_yaml(camera_path)
    profile_name = raw.pop(PROFILE_KEY, None)
    if not profile_name:
        raise ConfigError(f"{camera_path.name}: must declare a `{PROFILE_KEY}:` key")

    profile_path = config_dir / "profiles" / f"{profile_name}.yaml"
    if not profile_path.is_file():
        raise ConfigError(f"{camera_path.name}: unknown profile {profile_name!r} ({profile_path})")

    merged = deep_merge(deep_merge(base, _read_yaml(profile_path)), raw)
    try:
        camera = CAMERA_ADAPTER.validate_python(merged)
    except ValidationError as exc:
        raise ConfigError(f"{camera_path.name} (profile {profile_name}): {exc}") from exc

    _refuse_a_known_confusion(camera, camera_path, profile_name)
    return camera


def _refuse_a_known_confusion(camera: CameraConfig, path: Path, profile: str) -> None:
    """A recognition threshold below the worst impostor we have MEASURED is not a setting.

    `extra="forbid"` already makes a TYPO a startup error. This makes a KNOWN CONFUSION one.

    It is checked HERE, after base + profile + camera have been merged, because the merged
    value is the only layer that is true. The floor was set in `config/identity.py` and
    silently overridden back under itself by every canteen profile -- so it was in force on
    ZERO cameras while the code explained at length why it mattered. Then it was fixed on one
    branch and stayed broken on two others, because each carries its own copy of the YAML.

    A default is not what runs. A YAML file is not what runs. This is.
    """
    from qorgan.config.identity import MEASURED_IMPOSTOR_CEILING

    canteen = getattr(camera, "canteen", None)
    if canteen is None:
        return

    for role in ("entry", "exit", "inside"):
        block = getattr(canteen, role, None)
        if block is None:
            continue
        for field in ("recognition", "small_face", "soft"):
            policy = getattr(block, field, None)
            if policy is None or not getattr(policy, "enabled", True):
                continue
            if policy.min_score < MEASURED_IMPOSTOR_CEILING:
                raise ConfigError(
                    f"{path.name} (profile {profile}): canteen.{role}.{field}.min_score is "
                    f"{policy.min_score}, BELOW the measured impostor ceiling "
                    f"{MEASURED_IMPOSTOR_CEILING}. In this school's own gallery, 9 447 pairs "
                    f"of DIFFERENT children were scored and the worst pair reaches "
                    f"{MEASURED_IMPOSTOR_CEILING} -- so {policy.min_score} knowingly accepts "
                    f"a confusion we have already measured. Raise it to >= 0.50, or re-derive "
                    f"the ceiling with `qorgan pupils gallery-report` if the roster changed."
                )


def load_cameras(config_dir: Path | str | None = None) -> dict[str, CameraConfig]:
    """Every camera, keyed by name. Includes disabled ones; filter at the call site."""
    directory = resolve(config_dir or get_settings().config_dir)
    base = _read_yaml(directory / "base.yaml")

    cameras: dict[str, CameraConfig] = {}
    for path in sorted((directory / "cameras").glob("*.yaml")):
        camera = _resolve_camera(directory, path, base)
        if camera.name in cameras:
            raise ConfigError(f"{path.name}: duplicate camera name {camera.name!r}")
        if camera.name != path.stem:
            raise ConfigError(f"{path.name}: camera name {camera.name!r} must match the filename")
        cameras[camera.name] = camera

    if not cameras:
        raise ConfigError(f"no camera configs found in {directory / 'cameras'}")
    return cameras


def load_workers(
    cameras: dict[str, CameraConfig], config_dir: Path | str | None = None
) -> WorkersConfig:
    """Worker groups, cross-checked against the cameras that actually exist."""
    directory = resolve(config_dir or get_settings().config_dir)
    try:
        workers = WorkersConfig.model_validate(_read_yaml(directory / "workers.yaml"))
    except ValidationError as exc:
        raise ConfigError(f"workers.yaml: {exc}") from exc

    unknown = sorted(workers.assigned_cameras - set(cameras))
    if unknown:
        raise ConfigError(f"workers.yaml: assigns unknown camera(s): {unknown}")

    enabled = {name for name, camera in cameras.items() if camera.enabled}
    unassigned = sorted(enabled - workers.assigned_cameras)
    if unassigned:
        raise ConfigError(
            f"workers.yaml: enabled camera(s) assigned to no worker group: {unassigned}. "
            "A camera nobody runs is a camera nobody is watching."
        )
    return workers

"""Building blocks for the §12.1 weapons tests. Not a test module.

Centralised for the reason `tests/web_login.py` gives: a helper each test writes for
itself is a helper each test can get subtly wrong, and here the thing most easily got
wrong is the one that matters most -- **what counts as a plausible weights file.**
`plausible_weights()` writes real bytes over the real size gate, so a test that means "the
file is fine" cannot accidentally write four bytes and pass for the wrong reason.

Nothing here constructs a `WeaponView` for production use. `fake_view` exists so the
DECISION can be driven without a GPU and without weights -- which is the only kind of test
available to us, because we have no weapons model at all. It lives in `tests/` and never
in `src/` on purpose: a fake detector reachable from production code is precisely the
fallback this module exists to forbid.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from qorgan.config.camera import CAMERA_ADAPTER, WeaponsCamera
from qorgan.detection.geometry import Box
from qorgan.weapons.model import Sighting
from qorgan.weapons.weights import MIN_PLAUSIBLE_WEIGHTS_BYTES, LoadedWeights, WeightsFile
from tests.conftest import CONFIG_DIR

# A path shaped like the artefact this project would fetch. Never a real file unless a
# test makes one: the point of most of these tests is what happens when it is NOT there.
WEIGHTS_NAME = "qorgan-weapons.pt"


def weapons_camera_dict(**overrides) -> dict:
    """The smallest valid weapons camera, as YAML would give it."""
    camera = {
        "name": "entrance_weapons",
        "display_name": "Вход — рамка",
        "camera_type": "weapons",
        "role": "weapons",
        "rtsp": {"host": "192.168.1.90"},
        "weapons": {"model": {"model": WEIGHTS_NAME}},
    }
    camera.update(overrides)
    return camera


def weapons_camera(**overrides) -> WeaponsCamera:
    """A validated `WeaponsCamera`, through the real discriminated union."""
    camera = CAMERA_ADAPTER.validate_python(weapons_camera_dict(**overrides))
    assert isinstance(camera, WeaponsCamera)
    return camera


# Every camera file must declare a profile, and there is no weapons profile in
# `config/profiles/` because there is no weapons camera in this school yet. Tests write
# their own rather than adding one to the repository: a profile in `config/` that no
# camera names is a file somebody will later assume is in use.
WEAPONS_PROFILE = {"camera_type": "weapons", "role": "weapons"}


def config_dir_with(tmp_path: Path, *cameras: dict) -> Path:
    """A REAL config directory: the repository's own, plus these weapons cameras.

    A copy rather than a hand-built minimum, so `load_cameras()` does the whole job it
    does in production -- base.yaml, the profile lookup, the three-layer merge, the
    discriminated union -- and a page test exercises the same read path the school's
    browser triggers. A test that built a `WeaponsCamera` in Python and handed it to the
    view would skip every one of those layers and still look like coverage.
    """
    directory = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, directory)
    (directory / "profiles" / "weapons.yaml").write_text(
        yaml.safe_dump(WEAPONS_PROFILE), encoding="utf-8"
    )
    for camera in cameras:
        path = directory / "cameras" / f"{camera['name']}.yaml"
        path.write_text(
            yaml.safe_dump({"profile": "weapons", **camera}, allow_unicode=True),
            encoding="utf-8",
        )
    return directory


def plausible_weights(tmp_path: Path, name: str = WEIGHTS_NAME) -> Path:
    """A file that clears the SIZE gate, so a test can be about the next check.

    Deliberately one byte over the limit rather than a comfortable margin: a helper that
    writes a megabyte would hide an off-by-one in the gate itself.
    """
    path = tmp_path / name
    path.write_bytes(b"\x00" * (MIN_PLAUSIBLE_WEIGHTS_BYTES + 1))
    return path


def loaded_weights(
    path: str = "/models/qorgan-weapons.pt",
    *,
    task: str = "detect",
    class_names: tuple[str, ...] = ("knife", "firearm"),
    evaluated_on: str = "",
) -> LoadedWeights:
    """What `YoloWeaponModel` would hold, without needing a card or a checkpoint."""
    return LoadedWeights(
        file=WeightsFile(path=path, size_bytes=6_000_000, fingerprint="abc123def456abcd"),
        task=task,
        class_names=class_names,
        evaluated_on=evaluated_on,
    )


def sighting(
    class_name: str = "knife",
    confidence: float = 0.9,
    *,
    x1: float = 100.0,
    y1: float = 100.0,
    size: float = 40.0,
) -> Sighting:
    """One square sighting of a given size, at a given place."""
    return Sighting(
        class_name=class_name,
        confidence=confidence,
        box=Box(x1, y1, x1 + size, y1 + size),
    )


def person_box(x1: float = 90.0, y1: float = 60.0, w: float = 60.0, h: float = 160.0) -> Box:
    """A person-shaped box that CONTAINS `sighting()`'s default position."""
    return Box(x1, y1, x1 + w, y1 + h)


class FakeWeaponView:
    """A `WeaponView` that returns what a test tells it to, frame by frame.

    This is the synthetic detector §12.1's plan asks the pipeline to be tested against.
    It is also the thing that makes a negative result mean anything: a run that finds
    nothing proves nothing until the same code, on the same frames, has been shown to
    find something when there is something to find.
    """

    def __init__(self, weights: LoadedWeights | None = None) -> None:
        self.weights = weights or loaded_weights()
        self.script: list[list[Sighting]] = []
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Sighting]:
        del image
        index = self.calls
        self.calls += 1
        if index < len(self.script):
            return list(self.script[index])
        return []

"""The camera config: a discriminated union on camera_type.

Legacy shipped a full bullying scoring block on two canteen cameras -- about 25 keys
that no code path ever read for those cameras -- because nothing stopped it. Here the
schema makes that combination unrepresentable.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from qorgan.config.bullying import BullyingConfig
from qorgan.config.canteen import CanteenConfig
from qorgan.config.classroom import ClassroomConfig
from qorgan.config.common import (
    Base,
    CaptureSettings,
    PreviewSettings,
    RtspSettings,
    YoloSettings,
)
from qorgan.config.source import FileSource
from qorgan.config.weapons import WeaponsConfig
from qorgan.enums import (
    BULLYING_ROLES,
    CANTEEN_ROLES,
    CLASSROOM_ROLES,
    WEAPONS_ROLES,
    CameraRole,
    CameraType,
)


class CameraBase(Base):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    display_name: str
    location: str = ""
    priority: int = Field(default=100, ge=0)
    enabled: bool = True

    rtsp: RtspSettings
    capture: CaptureSettings = CaptureSettings()
    yolo: YoloSettings = YoloSettings()
    preview: PreviewSettings = PreviewSettings()

    # Absent = read the camera. Present = read a recorded clip INSTEAD, paced to its own
    # frame rate, with everything after the decode unchanged. `rtsp` stays required either
    # way: a camera that exists on the wall does not stop existing because today's frames
    # come from a file, and a config that dropped it would have to be edited back before
    # the school could go live. See `config/source.py`.
    source: FileSource | None = None
    # debug DELETED, with the DebugSettings model: nothing read any of its three flags.


class BullyingCamera(CameraBase):
    camera_type: Literal[CameraType.BULLYING] = CameraType.BULLYING
    role: CameraRole
    bullying: BullyingConfig = BullyingConfig()

    @model_validator(mode="after")
    def _role_matches_type(self) -> BullyingCamera:
        if self.role not in BULLYING_ROLES:
            raise ValueError(
                f"camera {self.name!r}: role {self.role!r} is not a bullying role "
                f"(expected one of {sorted(r.value for r in BULLYING_ROLES)})"
            )
        return self


class CanteenCamera(CameraBase):
    camera_type: Literal[CameraType.CANTEEN] = CameraType.CANTEEN
    role: CameraRole
    canteen: CanteenConfig = CanteenConfig()

    @model_validator(mode="after")
    def _role_matches_blocks(self) -> CanteenCamera:
        if self.role not in CANTEEN_ROLES:
            raise ValueError(
                f"camera {self.name!r}: role {self.role!r} is not a canteen role "
                f"(expected one of {sorted(r.value for r in CANTEEN_ROLES)})"
            )
        expected = {
            CameraRole.CANTEEN_ENTRY: "entry",
            CameraRole.CANTEEN_EXIT: "exit",
            CameraRole.CANTEEN_INSIDE: "inside",
        }[self.role]
        _require_only_block(self.name, self.role, self.canteen, expected)
        return self


def _require_only_block(name: str, role: CameraRole, canteen: CanteenConfig, expected: str) -> None:
    """A canteen camera carries the block for its role, and no other."""
    blocks = {"entry": canteen.entry, "exit": canteen.exit, "inside": canteen.inside}
    if blocks[expected] is None:
        raise ValueError(
            f"camera {name!r}: role {role.value!r} requires a canteen.{expected} block"
        )
    extras = sorted(key for key, value in blocks.items() if key != expected and value is not None)
    if extras:
        unwanted = ", ".join(f"canteen.{key}" for key in extras)
        raise ValueError(f"camera {name!r}: role {role.value!r} must not carry {unwanted}")


class ClassroomCamera(CameraBase):
    """§12.4: a camera looking at the pupils, counting what a body visibly does.

    It carries `classroom` and nothing else. There is deliberately no `canteen` block and
    no way to give it one: that block is where every recognition threshold in this schema
    lives, and §8 promised the school there would be no identification inside a classroom.
    The discriminated union is what makes that promise unrepresentable rather than merely
    documented -- the same mechanism that stopped the legacy shipping 25 bullying keys
    onto two canteen cameras.
    """

    camera_type: Literal[CameraType.CLASSROOM] = CameraType.CLASSROOM
    role: CameraRole
    classroom: ClassroomConfig = ClassroomConfig()

    @model_validator(mode="after")
    def _role_matches_type(self) -> ClassroomCamera:
        if self.role not in CLASSROOM_ROLES:
            raise ValueError(
                f"camera {self.name!r}: role {self.role!r} is not a classroom role "
                f"(expected one of {sorted(r.value for r in CLASSROOM_ROLES)})"
            )
        return self


class WeaponsCamera(CameraBase):
    """§12.1: a camera whose second detection session looks for a weapon.

    It carries `weapons` and nothing else. There is deliberately no `bullying` block: the
    two tiers have different frame rates, different class maps and different decisions,
    and the legacy's lesson here is precise -- 25 bullying keys shipped onto two canteen
    cameras that never read one of them, because nothing stopped it. The discriminated
    union stops it.

    **`weapons` has no default**, unlike every other camera type's block. That is the
    whole shape of the refusal this module exists for: `WeaponModelSettings.model` is a
    required string, so a weapons camera with no weights named is a STARTUP error in the
    config layer, before a process has been spawned or a frame read. A default here --
    any default -- would put a plausible path in the schema that nothing on disk can
    satisfy, which is precisely how a 0-byte `best.pt` came to look configured.
    """

    camera_type: Literal[CameraType.WEAPONS] = CameraType.WEAPONS
    role: CameraRole
    weapons: WeaponsConfig

    @model_validator(mode="after")
    def _role_matches_type(self) -> WeaponsCamera:
        if self.role not in WEAPONS_ROLES:
            raise ValueError(
                f"camera {self.name!r}: role {self.role!r} is not a weapons role "
                f"(expected one of {sorted(r.value for r in WEAPONS_ROLES)})"
            )
        return self


CameraConfig = Annotated[
    BullyingCamera | CanteenCamera | ClassroomCamera | WeaponsCamera,
    Field(discriminator="camera_type"),
]

CAMERA_ADAPTER: TypeAdapter[CameraConfig] = TypeAdapter(CameraConfig)

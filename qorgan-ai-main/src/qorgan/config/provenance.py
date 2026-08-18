"""For every value in force: what it is, and WHICH FILE put it there.

`loader.py` answers "what runs". This answers "where do I change it", which is a different
question and the one this codebase has been getting wrong.

`min_score: 0.50` was declared in `config/identity.py`, with a long comment on why 0.50 is
a measured floor -- and every canteen profile quietly wrote 0.42-0.45 over it, so the
documented value was in force on ZERO cameras. Nobody was careless; the merge is simply
invisible. `base.yaml <- profiles/<p>.yaml <- cameras/<c>.yaml` is three files deep, and a
number on a screen looks the same whichever of the three supplied it.

So nothing here shows a value on its own. A value travels with its layer and its file, and
`test_config_provenance.py` fails if any value cannot say where it came from.

**This module only reads.** It has no writer, and `test_web_settings.py` enforces that with
an AST scan rather than trusting this sentence: a browser that edits YAML would be a second
source of truth for a value the supervisor loaded at startup, which is the same disease one
layer up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from qorgan.config.camera import CameraConfig
from qorgan.config.loader import PROFILE_KEY, _read_yaml, load_cameras
from qorgan.redaction import redact
from qorgan.settings import get_settings, resolve

LAYER_BASE = "base"
LAYER_PROFILE = "profile"
LAYER_CAMERA = "camera"
# Nothing in any YAML sets it; the value is the field default declared in src/. Naming a
# YAML file for these would send somebody to a file the key does not appear in.
LAYER_DEFAULT = "default"

# Shown in the camera's own heading, so they are not repeated as rows.
HEADER_KEYS = frozenset({"name", "display_name", "camera_type", "role"})

# Values that sit at the top of a camera file rather than inside a named block.
TOP_LEVEL_SECTION = "камера"


@dataclass(frozen=True, slots=True)
class Setting:
    """One value in force, and the file that supplied it."""

    key: str  # dotted, as it is written in YAML: `bullying.metrics.speed_threshold`
    value: str  # rendered for display, JSON-shaped so `true`/`null` read as they do in YAML
    layer: str  # one of the LAYER_* constants
    source: str  # `config/profiles/hall.yaml`, or `src/qorgan/config/identity.py`


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    settings: tuple[Setting, ...]


@dataclass(frozen=True, slots=True)
class CameraView:
    """One camera as the settings page shows it."""

    name: str
    display_name: str
    location: str
    camera_type: str
    role: str
    enabled: bool
    priority: int
    profile: str
    # The merge chain for THIS camera, in the order the loader applies it. Printed in full
    # because "which file wins" is the question, and a reader who cannot see the order
    # cannot check our answer.
    chain: tuple[str, ...]
    sections: tuple[Section, ...]


def config_label(config_dir: Path | str | None = None) -> str:
    """How this installation's config directory is named on screen: `config`, usually.

    Exists so that nothing has to hardcode the string "config/". `config_dir` is a setting
    and an installation may move it, and a page that tells an engineer to edit
    `config/cameras/hall_left.yaml` when the file is somewhere else is precisely the
    failure this module was written to prevent -- just aimed at the reader instead of the
    detector.
    """
    return resolve(config_dir or get_settings().config_dir).name


def camera_views(config_dir: Path | str | None = None) -> list[CameraView]:
    """Every camera, in priority order, with each value attributed to its file.

    Raises `ConfigError` exactly as `load_cameras` does -- the validation is NOT repeated
    here. One resolver, one set of error messages: a second one would eventually disagree
    with the first, and this module exists because of what a disagreement between layers
    costs.
    """
    directory = resolve(config_dir or get_settings().config_dir)
    cameras = load_cameras(directory)
    base = _flatten(_read_yaml(directory / "base.yaml"))

    views = [_view(directory, name, camera, base) for name, camera in cameras.items()]
    # Priority order, the same as `qorgan config validate` and the camera wall.
    return sorted(views, key=lambda view: view.priority)


def _view(
    directory: Path, name: str, camera: CameraConfig, base: dict[str, Any]
) -> CameraView:
    raw = _read_yaml(directory / "cameras" / f"{name}.yaml")
    profile_name = str(raw.pop(PROFILE_KEY, ""))
    profile = _read_yaml(directory / "profiles" / f"{profile_name}.yaml")

    label = config_label(directory)
    # Camera first: the loader merges base <- profile <- camera, so the LAST layer to
    # mention a key is the one in force. Searched in reverse for exactly that reason.
    layers = (
        (LAYER_CAMERA, _flatten(raw), f"{label}/cameras/{name}.yaml"),
        (LAYER_PROFILE, _flatten(profile), f"{label}/profiles/{profile_name}.yaml"),
        (LAYER_BASE, base, f"{label}/base.yaml"),
    )
    # `model_dump`, not attribute access: the effective tree, keyed exactly as YAML writes
    # it, in the order the schema declares. Enum members come back as their string values.
    effective = _flatten(camera.model_dump(mode="json"))

    return CameraView(
        name=str(effective["name"]),
        # `_plain`, not `_text`: these two are the camera's heading, and a heading reading
        # `"Холл слева"` with the quotes still on it is a value pretending to be a literal.
        display_name=_plain(effective["display_name"]),
        location=_plain(effective.get("location", "")),
        camera_type=str(effective["camera_type"]),
        role=str(effective["role"]),
        enabled=bool(effective["enabled"]),
        priority=int(effective["priority"]),
        profile=profile_name,
        chain=tuple(source for _, _, source in reversed(layers)),
        sections=_sections(effective, layers, _declaring_modules(type(camera))),
    )


def _sections(
    effective: dict[str, Any],
    layers: tuple[tuple[str, dict[str, Any], str], ...],
    declared: dict[str, str],
) -> tuple[Section, ...]:
    """Bucketed by the top-level YAML block, in schema order.

    The grouping is done here and not in the template: a template that decides what goes
    where is business logic in a place no test can reach.
    """
    buckets: dict[str, list[Setting]] = {}
    for key, value in effective.items():
        if key in HEADER_KEYS:
            continue
        block = key.split(".")[0] if "." in key else ""
        buckets.setdefault(block, []).append(_attribute(key, value, layers, declared))

    return tuple(
        Section(title=block or TOP_LEVEL_SECTION, settings=tuple(settings))
        for block, settings in buckets.items()
    )


def _attribute(
    key: str,
    value: Any,
    layers: tuple[tuple[str, dict[str, Any], str], ...],
    declared: dict[str, str],
) -> Setting:
    for layer, flat, source in layers:
        if key in flat:
            return Setting(key=key, value=_text(value), layer=layer, source=source)
    return Setting(
        key=key, value=_text(value), layer=LAYER_DEFAULT, source=_declared(key, declared)
    )


def _text(value: Any) -> str:
    """Display form. JSON-shaped, then redacted.

    JSON because YAML and JSON agree on scalars: `true`, `null` and a quoted string read on
    the page the way they must be typed into the file. Python's own `repr` would print
    `True` and `None`, which are not things you may write in YAML.

    Redacted because R4 says no secret lives in config and `test_no_secrets.py` greps for
    one -- and this is the page that would publish it if one ever slipped through. The
    legacy drew `rtsp://user:password@host` onto debug JPEGs an unauthenticated UI served
    (audit C-02); the shape of that mistake is a value reaching a screen unfiltered.
    """
    return redact(json.dumps(value, ensure_ascii=False))


def _plain(value: Any) -> str:
    """Free text out of a config file, for a heading rather than for a value cell.

    Redacted for the same reason as `_text` -- this is free-form text somebody typed into a
    YAML file on site, and the last place a stray credential should surface is the page
    that renders the whole installation to a browser.
    """
    return redact(str(value))


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested mapping -> dotted keys.

    **A list is a LEAF.** `deep_merge` replaces a list wholesale -- "zones are not something
    you want half-inherited" -- so one list is one setting. Splitting it into per-element
    rows would advertise an override the loader does not offer.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _declaring_modules(model: type[BaseModel], prefix: str = "") -> dict[str, str]:
    """Dotted key -> the module whose model declares that field.

    Walked over the CLASSES, from `model_fields` and annotations, never by reading
    attributes off a camera object. Two reasons: a field left at its default has no
    interesting instance to read, and `tests/config_scan.py` proves which config keys are
    dead by resolving who reads what -- a getattr sweep over every field here would look
    exactly like every key being read, and would quietly retire that check.
    """
    modules: dict[str, str] = {}
    for name, field in model.model_fields.items():
        dotted = f"{prefix}{name}"
        modules[dotted] = model.__module__
        nested = _model_of(field.annotation)
        if nested is not None:
            modules.update(_declaring_modules(nested, f"{dotted}."))
    return modules


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The config model an annotation resolves to, or None if it is a leaf.

    `list[ZoneRect]` is deliberately a leaf -- see `_flatten`. `EntryPolicy | None` is not:
    an optional block still declares its fields somewhere, and "somewhere" is the answer
    this module exists to give.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is Annotated:
        return _model_of(get_args(annotation)[0])
    if origin in (Union, UnionType):
        models = [
            arg
            for arg in get_args(annotation)
            if isinstance(arg, type) and issubclass(arg, BaseModel)
        ]
        # A discriminated union (CameraConfig) has no single declaring module, and no field
        # inside a camera is one. Ambiguity is reported as "leaf", never guessed.
        return models[0] if len(models) == 1 else None
    return None


def _declared(key: str, declared: dict[str, str]) -> str:
    """The .py file that declares this field, for a value no YAML sets.

    Walks up the dotted path: a key under a container the schema does not name field by
    field (a plain `dict[str, ...]`) is still owned by the nearest model above it, and
    pointing at that model is a true answer where pointing at nothing is not.

    Caveat, stated because a half-truth here is worse than none: this names where the FIELD
    is declared. A parent model may hold a default instance that overrides it -- e.g.
    `EntryPolicy.recognition = RecognitionPolicy(min_score=0.50)` in `config/canteen.py`
    over the 0.50 in `config/identity.py`. The field's home is where you go to read the
    reasoning; the value in force is the one shown beside it.
    """
    parts = key.split(".")
    while parts:
        module = declared.get(".".join(parts))
        if module:
            return "src/" + module.replace(".", "/") + ".py"
        parts.pop()
    return ""

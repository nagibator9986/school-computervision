"""Are these actually weapons weights, and did they actually load?

**This module is the whole reason `qorgan.weapons` exists.**

The client ran a weapons module for months with no model in it. Their `best.pt` is a
**0-byte file**, and what it was meant to be was a *violence classifier* rather than a
weapons *detector*. The code checked that a path was configured, found one, and carried
on; when loading failed it logged a warning and fell back to motion analysis. So the
module reported healthy, the dashboard reported healthy, and everything the school
watched working was a motion detector. Nobody was lied to on purpose. Nothing ever
checked.

Four different things can be wrong, they fail in four different ways, and a check that
collapses them into "could not load the model" sends whoever reads it looking in the
wrong place. So each is separated here, each names the file, and **not one of them has a
fallback**:

  1. **The file is not there.** `qorgan.pt` was never fetched, or the path is relative to
     a directory the worker was not started from.
  2. **The file is there and is empty**, or is far too small to be weights. This is the
     one that actually happened. `Path.is_file()` is True for a 0-byte file, which is why
     "the path exists" is not a check and this module measures the SIZE.
  3. **The file is there, has content, and does not load.** Truncated download, wrong
     format, a git-lfs pointer committed as though it were the model.
  4. **It loaded, and it is the wrong kind of model.** This is the client's actual
     defect, one layer deeper than they got to: a *classifier* has no boxes, so it can
     never say where anything is, and «проверка нахождения рядом с человеком» -- §12.1's
     fourth step -- is not a thing you can ask it. It would have loaded cleanly and
     produced nothing, forever.

There is a fifth check, and it is the one nobody thinks of: **weights that load, detect,
and cannot produce any class this camera is configured to alarm on.** A person detector
loads perfectly and has a `person` class; if `target_classes` says `knife`, the pipeline
runs at full cost and is silent by construction. That is the 0-byte failure with a
working model in it, and `refuse_unusable_weights` catches it too.

And a sixth, which is that same defect one level in: **weights that produce every target
and none of the CONFUSABLES.** `confusable_classes` is the whole input to screen 3 of
`weapons/rules.py` -- the «нож или ручка» withholding -- and until `inert_confusables`
existed it was checked against nothing at all, while the targets beside it were checked
loudly and refused outright. So that defence could be present in the configuration and
absent from the running system, silently, forever. It does NOT raise, and
`inert_confusables` says at length why refusing would be worse.

Nothing here imports ultralytics or torch. The FILE checks are arithmetic on bytes and
the LOADED checks take a small structural description, so the whole refusal is testable
without a GPU and without weights -- which matters, because we have no weights to test
with and the refusal is the part that must not be taken on trust.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Ultralytics' own task token for a detector. A `classify` model is the client's exact
# defect; `pose` and `segment` load fine and answer a different question.
DETECT_TASK = "detect"

# Below this, a file is not weights whatever its extension. The smallest real YOLO
# checkpoint in this project (`yolov8n.pt`) is ~6 MB; a git-lfs pointer is ~130 bytes and
# an empty file is 0. 4 KiB is far under any model and far over any accident, so it
# separates the two without needing to know which model somebody chose.
MIN_PLAUSIBLE_WEIGHTS_BYTES = 4096

# How much of the file is hashed for the fingerprint shown on the panel. The whole file
# would be tens of megabytes read on every page load; the first megabyte plus the size is
# enough to tell two checkpoints apart, and it is labelled as a prefix hash on screen so
# nobody mistakes it for a full checksum of the artefact.
FINGERPRINT_BYTES = 1024 * 1024


class WeaponWeightsUnusable(RuntimeError):
    """The weapons model cannot be used. **The pipeline must refuse to start.**

    A distinct exception type rather than a bare RuntimeError so that no caller can
    plausibly catch it alongside something ordinary and continue. There is no handler for
    this anywhere in `src/`, and `tests/test_weapons_refusal.py` asserts that: a module
    that recovers from having no model is the module the client already had.
    """


@dataclass(frozen=True, slots=True)
class WeightsFile:
    """The artefact on disk, as the panel names it."""

    path: str
    size_bytes: int
    # A hash of the first megabyte, not of the whole file. Named `fingerprint` rather
    # than `sha256` for exactly that reason -- a field called sha256 that is not the
    # sha256 of the thing is a value true in one layer and wrong in the next.
    fingerprint: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class LoadedWeights:
    """What the module is running, in the words the panel prints.

    `evaluated_on` comes from the config and is whatever a human wrote there. It is NOT
    computed and NOT inferred: nothing in this tree has evaluated these weights, and a
    number invented here would be the most dangerous kind of provenance -- one that looks
    measured.
    """

    file: WeightsFile
    task: str
    class_names: tuple[str, ...]
    evaluated_on: str

    def describes(self, target_classes: Iterable[str]) -> tuple[str, ...]:
        """Which of the camera's targets these weights can actually produce."""
        available = set(self.class_names)
        return tuple(name for name in target_classes if name in available)


def inspect_weights_file(path: Path | str) -> WeightsFile:
    """Read the artefact, or refuse. Cases 1 and 2 of the module docstring.

    Called BEFORE anything tries to load it, so that the commonest two failures produce a
    sentence about a file rather than a torch traceback about a pickle.
    """
    target = Path(path)
    if not target.is_file():
        raise WeaponWeightsUnusable(
            f"weapons model not found: {target}\n"
            "The weapons pipeline does not start without weights and does not fall back "
            "to anything. Fetch the file, or correct `weapons.model.model` in this "
            "camera's YAML. There is no motion-analysis fallback here on purpose: that "
            "fallback is what let the previous system report a working weapons module "
            "for months with no model in it."
        )

    size = target.stat().st_size
    if size < MIN_PLAUSIBLE_WEIGHTS_BYTES:
        raise WeaponWeightsUnusable(
            f"weapons model is {size} bytes: {target}\n"
            f"That is below {MIN_PLAUSIBLE_WEIGHTS_BYTES} bytes and is not a model. "
            "**This is the exact state the client's own `best.pt` is in -- 0 bytes -- "
            "and the reason this check measures the size rather than asking whether the "
            "path exists.** `Path.is_file()` is True for an empty file. Likely causes: "
            "an interrupted download, a git-lfs pointer committed instead of the "
            "artefact, or a training run that never wrote its weights."
        )

    return WeightsFile(path=str(target), size_bytes=size, fingerprint=_fingerprint(target))


def refuse_unusable_weights(
    loaded: LoadedWeights, target_classes: Sequence[str], camera: str
) -> None:
    """Cases 4 and 5: it loaded, and it is still not a model that can do this job.

    Raises or returns None. It never warns, and it never returns a degraded object for a
    caller to decide about -- the decision is here, once, so that no call site can be
    written that carries on.
    """
    if loaded.task != DETECT_TASK:
        raise WeaponWeightsUnusable(
            f"camera {camera!r}: the weapons weights at {loaded.file.path} are a "
            f"{loaded.task!r} model, not a {DETECT_TASK!r} model.\n"
            "**This is the client's own defect, and it is the one that hides.** A "
            "classifier loads cleanly and produces no boxes at all, so §12.1's «проверка "
            "нахождения рядом с человеком» can never be asked of it -- the module would "
            "run at full GPU cost and be silent forever, looking healthy the whole time. "
            "Their `best.pt` was a violence classifier where a weapons detector was "
            "expected. Train or obtain a detection model."
        )

    if not loaded.class_names:
        raise WeaponWeightsUnusable(
            f"camera {camera!r}: the weapons weights at {loaded.file.path} loaded but "
            "declare no classes at all. Nothing they can output has a name, so nothing "
            "they output can be matched against a target."
        )

    usable = loaded.describes(target_classes)
    if not usable:
        raise WeaponWeightsUnusable(
            f"camera {camera!r}: the weapons weights at {loaded.file.path} cannot "
            f"produce ANY of this camera's target classes.\n"
            f"  configured targets : {sorted(target_classes)}\n"
            f"  the model's classes: {sorted(loaded.class_names)}\n"
            "The pipeline would run at full cost and be silent by construction -- the "
            "0-byte failure again, with a model that works in it. Either these are the "
            "wrong weights, or `weapons.target_classes` names something they were never "
            "trained on."
        )


def inert_confusables(
    loaded: LoadedWeights, confusable_classes: Sequence[str]
) -> tuple[str, ...]:
    """Which declared confusables these weights CANNOT produce. Screen 3's dead half.

    **This is the asymmetry `refuse_unusable_weights` had, named.** Target classes are
    checked against the loaded model loudly, in this file, and a camera whose targets the
    weights cannot emit is refused outright. `confusable_classes` was checked against
    NOTHING -- and it is the input to screen 3 of `weapons/rules.py`, the «нож или ручка»
    withholding. Measured: weights whose classes are `(knife, person, cell phone)` against
    the shipped confusables (`phone`, `pen`, `ruler`, `scissors`, `kitchen_utensil`, `toy`,
    `clothing`) intersect to NOTHING, and the pipeline started without a word. Screen 3 then
    never fires, at any confidence, for the life of the installation. A defence that guards
    nothing is the same defect as a permission that guards nothing.

    **It returns rather than raising, and that is a judgement rather than an oversight.** A
    refusal here would block every plausible weapons model: a detector trained on knives and
    firearms has no reason to emit `pen`, and the shipped default names seven slugs a weapons
    training set is unlikely to contain. Refusing would make the module unusable on the day
    real weights arrive -- a worse failure than a defence that is inert, SO LONG AS the
    inertness is said out loud. So `YoloWeaponModel` logs it at every start, `qorgan weapons
    weights` prints it, and `/weapons` says per camera that these slugs are a convention and
    names the command that checks them.

    The naming problem is real and worth stating: `KNOWN_CONFUSABLES` is a convention this
    project invented. COCO says `cell phone` where this schema says `phone`. Whoever brings
    weights will already be editing `KNOWN_TARGETS` to match their class map -- they are the
    person this function exists to interrupt.
    """
    available = set(loaded.class_names)
    return tuple(name for name in confusable_classes if name not in available)


def _fingerprint(path: Path) -> str:
    """A short, stable name for these particular bytes.

    Its job is to answer "are the weights on the school's machine the ones we tested?"
    when two files share a name -- and every Ultralytics training run in the world calls
    its output `best.pt`, which is why the file name alone cannot answer it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(FINGERPRINT_BYTES))
    return digest.hexdigest()[:16]

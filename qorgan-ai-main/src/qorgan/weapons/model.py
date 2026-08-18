"""The weapons model seam: pixels in, named boxes out. Nothing here decides anything.

The same shape as `models/person.py` and `models/pose.py`, and deliberately so -- the
plug-in point `REWRITE_SPEC.md` §5.1 reserved ("violence model (optional, currently
absent -- keep the plug-in point)") is a *swappable model*, not a swappable pipeline. So
`WeaponView` is the seam, `YoloWeaponModel` is production, and everything that judges
lives in `qorgan.weapons.pipeline`, which is pure and is what the tests drive.

**The seam is not an excuse.** A plug-in point that accepts "no plug-in" is what the
client had. `YoloWeaponModel.__init__` refuses without weights -- in the constructor, so
there is no unloaded object for anybody to hold -- and the pipeline takes a `WeaponView`
it cannot construct itself, so the only way to a running weapons pipeline is through a
model that loaded. A fake detector is a test's business and can never be a production
fallback: nothing in `src/` constructs one.

One model per CAMERA, not per worker group -- the opposite of the pose model and for the
same reason `models/person.py` gives. Ultralytics `predict` is stateless, but the weights
are per-camera CONFIGURABLE (`weapons.model.model`), and two cameras in one group may
legitimately name different files while a group can only hold one pose model. Sharing
would mean silently running one camera's weights on another's frames, and the panel would
name the wrong file for one of them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qorgan.config.weapons import WeaponModelSettings
from qorgan.detection.geometry import Box
from qorgan.logging_setup import get_logger
from qorgan.weapons.weights import (
    LoadedWeights,
    WeightsFile,
    inert_confusables,
    inspect_weights_file,
    refuse_unusable_weights,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Sighting:
    """One thing the weights say they saw, in one frame. Not yet an alert of any kind."""

    class_name: str
    confidence: float
    box: Box

    @property
    def size_pixels(self) -> float:
        """The longer side, which is the dimension a knife is measured by.

        The blade of a knife held in a hand is long and narrow, so its box is long and
        narrow: judging it by area or by width would refuse the very shape the module
        exists to find. The longer side is the one that survives the geometry.
        """
        return max(self.box.width, self.box.height)


class WeaponView(Protocol):
    """Anything that can look at a frame and name what it saw.

    `YoloWeaponModel` in production; a fake in a test. That is the point: the DECISION is
    testable without a GPU and without weights, which matters more here than anywhere
    else in this codebase, because there are no weights to test with.
    """

    @property
    def weights(self) -> LoadedWeights: ...

    def detect(self, image: np.ndarray) -> list[Sighting]: ...


class YoloWeaponModel:
    """One camera's weapons weights, verified at construction.

    The constructor does the loading and the refusing. There is no `load()` to forget and
    no half-built state: either you hold one of these and the weights are usable, or
    `WeaponWeightsUnusable` was raised and there is no object.

    `_load_or_refuse` and `_say_what_loaded` are STEPS OF `__init__`, not a load step
    somebody could call later: both are private, both are called from the constructor and
    nowhere else, and `__init__` does not survive either of them failing. They exist
    because R1 caps a function at 50 lines, and splitting a refusal across functions is
    exactly how a guard that names functions goes blind.

    **What is mechanically checked, and where it stops.** Before the split this was a
    tautology -- the code was inline, there was nothing to call. The split turned it into a
    claim about a `@staticmethod` that takes a path and returns a loaded model, which is the
    shape of a reusable loader. `tests/test_weapons_startup_chain.py` reads this file and
    fails on a step name in `STARTUP_CHAIN` that no longer resolves, on a catch site inside a
    named step that does not re-raise on every path out of it, on an `except` in this file
    outside the named steps, and -- the clause that has been rewritten twice and been wrong
    twice, so here it is as the mechanism rather than as the intention:

        a call to either step that is not lexically inside a `def` this file names as a step

    That rule reads THIS FILE alone, which left the widest caller shape of all unread: another
    module reaching in. A second rule reads every module under `src/` at once and reports

        a call to either step from another module under `src/`, except a bare or
        `self`/`cls`-qualified call in a module that defines that name itself

    Both are NARROWER than "nothing else may call these". Where they stop -- measured, not
    remembered, and green as measured:

      * a SECOND CLASS in this file whose `__init__` calls a step is matched BY NAME.
        `__init__` is a step name here and the match is not scoped to a class, so the call
        reads as construction and is not reported. The looseness runs both ways: a
        `super().__init__()` written in a method of this file that is NOT a step IS reported,
        which is a surprise red rather than a silent green, and has an obvious answer.
      * a CLOSURE defined inside `__init__` that calls a step is lexically inside it -- a
        deferred loader manufactured during construction and called later is green.
      * ONE NAMED STEP CALLING ANOTHER is inside the chain by definition.
      * the cross-module rule speaks only for names that are private by convention, so it
        covers these two steps but not `__init__` -- a protocol slot rather than a name
        anybody chose, and one with correct cross-module callers already. It reads only
        `src/`, so a test or a script reaching in is unseen. Its own-call exception is judged
        PER CALL, so a decoy `def _load_or_refuse` in the reaching module buys nothing; what
        it still passes over is a module that both imports the step under its bare name and
        defines its own `def` of it, which is ambiguous rather than sneaky.
      * neither rule sees a step that is not CALLED BY NAME: `cb = model._load_or_refuse`
        then `cb(p)`, or `getattr(model, "_load_or_refuse")(p)`, are both ways in.

    What IS reported is the shape this started from: `reload()` -- an ordinary method whose
    name is not a step -- swapping weights without re-running `refuse_unusable_weights`,
    leaving `self.weights` describing a file that is no longer running and a classifier able
    to arrive after startup. That cannot be added here without a red, and -- since the second
    rule -- a `weapons/pool.py` doing it by naming either step reds too. Not "cannot be added
    in a pool.py": that absolute was the FIFTH false sentence in this paragraph, written the
    same day the rule was, and the escape was a decoy `def` of the same name beside the
    hot-swap. The rule was narrowed to per-call rather than the sentence to fit it, but the
    bullets above are where its reach stops and the instruction below is the only absolute.

    Four drafts of this paragraph were wrong before this one, each in the same direction:
    claiming reach the rules do not have. A fifth was wrong about the PIN itself -- it said
    the pin ran the real rule over all four shapes listed here, and the fixture it ran is one
    `ast.parse` of one string, which cannot hold a second module. So both sentences above are
    pinned, each by the test that enforces it, and each is lifted out of THIS DOCSTRING rather
    than out of the file, so neither can be demoted to a comment and stay green:

      * `test_model_pys_account_of_the_call_rule_is_still_true` -- the in-file rule, over the
        caller shapes one module can hold;
      * `test_no_other_module_calls_a_private_step_of_startup` -- the cross-module rule, over
        a two-module fixture and over every module under `src/`.

    Reword either sentence and it fails; change either rule and it fails.

    **The rules are narrow. The instruction is not: do not add another caller for either step,
    anywhere.** An instruction cannot be false, and some of the ways to break this one are
    still green.
    """

    def __init__(
        self,
        settings: WeaponModelSettings,
        camera: str,
        target_classes: tuple[str, ...],
        device: str = "cuda:0",
        confusable_classes: tuple[str, ...] = (),
    ) -> None:
        self._settings = settings
        self._device = device
        self.camera = camera

        # Measured BEFORE the load: a 0-byte file raises a torch unpickling error that
        # says nothing about the file being empty, and that error is what the client's
        # code caught and turned into a warning.
        artefact = inspect_weights_file(settings.model)
        self._model = self._load_or_refuse(settings.model, artefact, camera)
        self._names: dict[int, str] = dict(getattr(self._model, "names", {}) or {})
        self.weights = LoadedWeights(
            file=artefact,
            # `task` is read off the loaded object rather than guessed from the
            # extension: a `.pt` says nothing about whether it detects or classifies, and
            # that difference is the whole of the client's defect.
            task=str(getattr(self._model, "task", "") or ""),
            class_names=tuple(self._names.values()),
            evaluated_on=settings.evaluated_on,
        )
        refuse_unusable_weights(self.weights, target_classes, camera)

        self._lock = threading.Lock()
        self._say_what_loaded(artefact, device)
        self._warn_about_inert_screen_three(confusable_classes)

    @staticmethod
    def _load_or_refuse(model_path: str, artefact: WeightsFile, camera: str) -> object:
        """Hand the file to ultralytics, and let nothing out of here but the weights.

        Case 3 -- present, big enough, and does not load -- reaches the operator through
        the heartbeat's `last_error`, and torch's own message for it is
        `OSError: [Errno 22] Invalid argument`. That names nothing. `worker/builders.py`
        promises "the missing file named in every line", and for this one case the promise
        was false: the supervisor logged a crash loop about an errno. So the file is named
        here, and the original is chained rather than swallowed -- `raise ... from exc`
        keeps `test_no_step_of_worker_startup_swallows_the_refusal` satisfied, because the
        rule is "must re-raise", not "must not catch".

        A `staticmethod` taking everything it needs, rather than reading `self._settings`
        and `self.camera`: this runs while the object is still being built, and a helper
        that depends on which lines of `__init__` have run yet is a half-built object by
        another name. It returns the ultralytics object as `object` because nothing in
        this module knows or claims to know its shape -- `detect` asks it for `predict`
        and `sightings_from` reads the answer.
        """
        from ultralytics import YOLO

        try:
            return YOLO(model_path)
        except Exception as exc:
            raise type(exc)(
                f"the weapons weights at {artefact.path} did not load "
                f"({type(exc).__name__}: {exc}). The file is present and is "
                f"{artefact.size_mb:.1f} MB, so this is a truncated download, the wrong "
                "format, or a git-lfs pointer committed as though it were the artefact. "
                f"camera={camera!r}. The pipeline does not start and does not fall back."
            ) from exc

    def _say_what_loaded(self, artefact: WeightsFile, device: str) -> None:
        """One line per camera per start, naming the bytes that are actually running.

        The fingerprint is the load-bearing field. Every Ultralytics run in the world
        writes `best.pt`, so the file NAME is not an identity and "the weights are on the
        machine" is a claim nobody could check; this is the line an operator reads to know
        which artefact a camera is watching with, and `evaluated_on` falls back to
        "(not stated)" rather than to nothing, because a silent field reads as a fine one.
        """
        logger.info(
            "weapons model loaded",
            extra={
                "camera": self.camera,
                "model": artefact.path,
                "fingerprint": artefact.fingerprint,
                "size_mb": round(artefact.size_mb, 1),
                "task": self.weights.task,
                "classes": sorted(self.weights.class_names),
                "evaluated_on": self._settings.evaluated_on or "(not stated)",
                "device": device,
            },
        )

    def _warn_about_inert_screen_three(self, confusable_classes: tuple[str, ...]) -> None:
        """Say, at every start, if screen 3 cannot fire on these weights.

        A WARNING rather than a refusal, for the reason `inert_confusables` gives: refusing
        would block every plausible weapons model. But it is said every single time the
        worker starts, at WARNING, naming the slugs, because the alternative is what the
        measurement found -- the pipeline coming up with the «нож или ручка» check dead and
        nothing anywhere mentioning it.
        """
        if not confusable_classes:
            return
        inert = inert_confusables(self.weights, confusable_classes)
        if not inert:
            return
        logger.warning(
            "weapons screen 3 is INERT on this camera: these weights emit none of its "
            "confusable classes, so an ambiguous «нож или ручка» can never be withheld",
            extra={
                "camera": self.camera,
                "declared_confusables": sorted(confusable_classes),
                "unproducible": sorted(inert),
                "the_models_classes": sorted(self.weights.class_names),
                "all_of_them_inert": len(inert) == len(confusable_classes),
                "consequence": (
                    "weapons.confusable_classes names slugs these weights were never "
                    "trained on. Either correct the names to the model's own class map or "
                    "accept that this screen does nothing on this camera."
                ),
            },
        )

    def detect(self, image: np.ndarray) -> list[Sighting]:
        with self._lock:
            results = self._model.predict(
                image,
                imgsz=self._settings.imgsz,
                conf=self._settings.conf,
                iou=self._settings.iou,
                device=self._device,
                verbose=False,
            )
        return sightings_from(results, self._names)


def sightings_from(results, names: dict[int, str]) -> list[Sighting]:
    """Ultralytics results -> named boxes. Pure, so it is tested without a GPU.

    A class id with no name is DROPPED rather than given one. An unnamed class cannot be
    matched against `target_classes` or against `confusable_classes`, so passing it on
    would put an object into the pipeline that no rule can reach -- and a detection no
    rule can reach is one that alarms or is silent for reasons nobody can look up.
    """
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    sightings: list[Sighting] = []
    triples = zip(
        boxes.cls.int().tolist(),
        boxes.conf.tolist(),
        boxes.xyxy.tolist(),
        strict=False,
    )
    for class_id, confidence, xyxy in triples:
        name = names.get(int(class_id))
        if name is None:
            continue
        x1, y1, x2, y2 = xyxy
        sightings.append(
            Sighting(
                class_name=name,
                confidence=float(confidence),
                box=Box(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return sightings

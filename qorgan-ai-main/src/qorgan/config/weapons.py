"""Weapons config (client §12.1): the thresholds, the classes, and the zone rules.

**READ THIS BEFORE TRUSTING ANY DEFAULT IN THIS FILE, AND BEFORE MOUNTING A CAMERA.**

## The wall

We hold no weapons model. Not one. The client's `best.pt` is a **0-byte file**, and what
it was supposed to be was a *violence classifier*, not a weapons *detector* -- so the
module they watched working for months had no model in it at any point. The code fell back
to motion analysis and logged a warning nobody read, and everything the school observed as
"the bullying module" was a motion detector.

That is the failure this schema and `qorgan.weapons` exist to make impossible. Nothing
here has a fallback. `qorgan.weapons.model.YoloWeaponModel` refuses in its CONSTRUCTOR --
so no caller ever holds an unloaded one -- and an empty file is not weights. See
`qorgan/weapons/weights.py`.

## Distance decides, and no threshold in this file changes that

This is the honest limit, and it belongs beside the camera settings rather than in a
report nobody runs (the same reason `qorgan identity camera-report` exists and says the
same kind of thing about faces):

    A knife in a hand at the school entrance is a 100+ px object. It will work.
    The same knife down a corridor at 15 m is ~15 px. It will never work.

`min_object_pixels` below is where that becomes a refusal rather than a disappointment: an
object smaller than the gate is not a low-confidence detection, it is not a detection at
all, and the module says so instead of quietly finding nothing. Run
`qorgan weapons camera-report <camera>` before deciding where a camera goes; it does the
arithmetic for a given object size and distance at the resolution the worker really
analyses, and it will tell you a corridor camera cannot do this job.

**Do not answer a camera-report that says "no" by lowering `min_object_pixels`.** There is
nothing underneath it to recover -- exactly as `identity/cli.py` says about the face gate.

## What is measured here, and what is chosen

Nothing here is measured. Every number below was reasoned and then chosen, because there
is no model to measure against and no footage of a weapon in this school. Each default
carries the reasoning that produced it and says CHOSEN, in those letters, so that whoever
finally has weights can check the specific assumption instead of re-deriving the lot.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from qorgan.config.common import Base, ZoneRect

# §12.1's target classes, as slugs. Held here rather than in the detector so that the
# schema can REFUSE a target the pipeline has no rule for, and so that a model shipping a
# hundred COCO classes cannot alarm on any of them by accident: a class not named in
# `WeaponsConfig.target_classes` is not a weapon, whatever the weights believe.
KNOWN_TARGETS = ("knife", "axe", "bat", "metal_object", "firearm")

# §12.1's false objects, named by the client: телефон, ручка, линейка, ножницы, кухонные
# предметы, игрушки, элементы одежды.
#
# **These are handled by the ARCHITECTURE, not by hoping the model is good.** Three
# mechanisms act on them and none is a threshold on the weapon class itself:
#
#   1. a class not in `target_classes` can never alarm, so a model that emits `scissors`
#      emits it into a log line;
#   2. `confusable_classes` -- if the weights ALSO report one of these at the same place
#      in the same frame, the model is ambiguous between two labels at one location and
#      the weapon claim is withheld with a recorded reason. A pen and a knife blade are
#      the same handful of pixels at 10 m, and a detector that says both is telling us so;
#   3. `min_object_pixels` refuses the size at which a pen and a knife are
#      indistinguishable in principle.
KNOWN_CONFUSABLES = (
    "phone",
    "pen",
    "ruler",
    "scissors",
    "kitchen_utensil",
    "toy",
    "clothing",
)


class WeaponModelSettings(Base):
    """Which weights, and how they are run. There is no default that means 'none'.

    `model` has no default on purpose. Every other model in this schema defaults to a
    filename that `scripts/fetch_models.py` fetches; there is no such file for weapons and
    inventing one would put a plausible path in a YAML file that nothing can satisfy --
    which is how a 0-byte `best.pt` came to look like a configured model.
    """

    model: str
    imgsz: int = Field(default=640, ge=320, le=1920)

    # The bar a single sighting must clear to be counted at all. CHOSEN. Deliberately
    # LOWER than `WeaponsConfig.reconfirm_confidence`: this one decides what enters a
    # track, and a track needs several observations before it can do anything. Making the
    # entry gate strict would leave nothing to accumulate.
    conf: float = Field(default=0.35, gt=0.0, lt=1.0)
    iou: float = Field(default=0.45, gt=0.0, lt=1.0)

    # What these weights were evaluated on, in the words of whoever evaluated them --
    # a dataset name, a date, a report path. Shown on the panel beside the file itself.
    #
    # **A module that cannot say what it is running is a module nobody can audit**, and
    # the file name alone does not say it: `best.pt` is the name of every Ultralytics run
    # ever made. Empty is allowed and renders as "не указано" rather than as an
    # endorsement -- an unevaluated model that admits it is honest; one that implies an
    # evaluation is the 0-byte file again.
    evaluated_on: str = ""


class OrdinaryToolZone(Base):
    """A part of the frame where a named class of object is ORDINARY, not an alarm.

    **This is a different RULE, not a different threshold, and the distinction is the
    whole of §12.1's kitchen clause.** In a kitchen a knife is a working tool. Raising the
    confidence bar there would mean a cook raises the alarm slightly less often than
    twenty times a day, which is the same product. So inside this rectangle the named
    classes do not alarm AT ALL, at any confidence, and the sighting is recorded with the
    reason `ordinary_in_zone` so the log still knows what was seen.

    **What this rule cannot do, said out loud.** It cannot tell a cook from a pupil
    holding the same knife: identity at this distance does not work in this school (14 970
    corridor faces, median 11.5 px, zero recognised), so nothing here may pretend to. The
    zone therefore suppresses the CLASS, not the person, and the school is told that in
    those terms.

    A class NOT named here still alarms inside the rectangle. A firearm in a kitchen is
    still a firearm, and a zone that silenced everything would be an off switch wearing a
    rule's clothes.
    """

    area: ZoneRect
    # The classes that are ordinary here. Never empty: a zone that makes nothing ordinary
    # changes nothing, and a rule that changes nothing is a rule somebody will trust.
    classes: list[str] = Field(min_length=1)
    # What this place is, for the log line and the panel. "кухня столовой", usually.
    label: str = ""

    @model_validator(mode="after")
    def _classes_are_known_targets(self) -> OrdinaryToolZone:
        unknown = sorted(set(self.classes) - set(KNOWN_TARGETS))
        if unknown:
            raise ValueError(
                f"weapons zone {self.label or self.area!r}: makes ordinary the class(es) "
                f"{unknown}, which are not weapon classes ({sorted(KNOWN_TARGETS)}). A "
                "zone can only exempt something that would otherwise alarm; exempting a "
                "name nothing produces is a rule that does nothing."
            )
        return self


class WeaponsZones(Base):
    """Per-camera zones, in FRACTIONS OF THE FRAME (`ZoneRect`), never pixels.

    The same rectangle type the bullying zones use, and deliberately not a second
    mechanism: a zone in pixels stops being right the moment somebody changes
    `capture.frame_width`, and this project has already paid for a value that was true in
    one layer and quietly wrong in the next.
    """

    ordinary_tools: list[OrdinaryToolZone] = Field(default_factory=list)


class WeaponsConfig(Base):
    """Everything a weapons camera needs. See the module docstring first."""

    model: WeaponModelSettings
    zones: WeaponsZones = WeaponsZones()

    # Which classes are weapons on THIS camera. A subset of KNOWN_TARGETS, and the
    # module's whole answer to §12.1's false-object list: a class not named here cannot
    # alarm, however sure the weights are about it.
    target_classes: list[str] = Field(default_factory=lambda: list(KNOWN_TARGETS))

    # Classes the weights may emit that are KNOWN confusions for a weapon at this
    # distance. A weapon claim overlapping one of these in the same frame is withheld:
    # the model is telling us it cannot decide between two labels at one location, and
    # "нож или ручка" is not an alarm a school can act on.
    confusable_classes: list[str] = Field(default_factory=lambda: list(KNOWN_CONFUSABLES))

    # -- the frame rate -----------------------------------------------------

    # Run the weapon model on every Nth frame that reaches the pipeline. §12.1 asks for a
    # reduced rate explicitly, and the GPU here is shared with the person detector, the
    # pose model and (in some groups) InsightFace.
    #
    # A CONFIG KEY, not a constant, because it is the one knob that trades detection
    # latency for GPU time and the right value depends on what else runs in the process.
    # Read by `qorgan.worker.weapons.WeaponsPipeline.on_frame`, which is the only thing
    # whose behaviour it describes.
    #
    # 3 is CHOSEN. At the shipped `capture.det_every: 1` and 15 fps that is 5 weapon
    # inferences a second; `min_track_observations` below then needs 0.6 s of a weapon
    # being visible before anything can happen, which is far less than the time it takes
    # to carry one anywhere. It is NOT `capture.det_every`: person detection has to stay
    # at full rate because a track id is what "near a person" is measured against.
    analyse_every: int = Field(default=3, ge=1, le=30)

    # -- the size gate: where physics stops being negotiable -----------------

    # The smallest object, in pixels of the ANALYSED frame, this module will call a
    # weapon. Below it a detection is refused outright and counted, rather than passed on
    # at low confidence.
    #
    # 24 px is CHOSEN, and here is the reasoning so it can be argued with. YOLO's finest
    # detection head has stride 8, so an object smaller than ~8 px in the LETTERBOXED
    # input covers less than one cell of the only head that could see it. At the shipped
    # `imgsz: 640` over a 960-px-wide analysis frame the scale is 0.667, so 8 px of input
    # is 12 px of frame; 24 px is two cells, which is the least that could carry a shape.
    #
    # That derivation is architectural rather than empirical -- nothing has been measured,
    # because there are no weights to measure. What IS measured is the consequence: this
    # school's corridor faces have a median of 11.5 px, and a knife blade is not larger
    # than a face. A corridor camera will therefore refuse essentially every knife, and
    # `qorgan weapons camera-report` says so in advance rather than after a term of
    # silence.
    min_object_pixels: float = Field(default=24.0, gt=0.0)

    # The lens's horizontal field of view, in degrees. A property of the HARDWARE, read
    # off the camera's datasheet, and the only input to the feasibility arithmetic that
    # is not already in this file.
    #
    # It is here, per camera, rather than a constant or a command-line flag, and the
    # reason is the client's answer of 2026-07-29: **the entrance camera will be placed
    # so that the object is large, AND the other cameras stay in play.** So the module is
    # asked to be honest about ten cameras that differ, and a fleet-wide number would let
    # the best one speak for the worst. `identity/cli.py` was bitten by exactly that: its
    # own help text listed which profiles override the default, said two, and there were
    # three.
    #
    # Read by `qorgan.web.routes.weapons` -- which is the point of it. Until this key
    # existed the feasibility answer lived behind `qorgan weapons camera-report`, one
    # camera at a time, needing a flag the operator had to know; so in practice nobody
    # would ever have seen it, and «оружие детектируется» would have been said on the
    # strength of the entrance camera while eight others saw a 15-px smudge. That is the
    # face-recognition failure again -- «работало», until somebody measured 11.5 px and
    # zero recognitions out of 14 970. `weapons/cli.py` reads it too, as the default for
    # `--hfov-deg`, so the screen and the terminal cannot disagree.
    #
    # 78.0 is CHOSEN, not measured: a 4 mm lens on a 1/2.8" sensor, the commonest fixed
    # IP camera and what this school's Hikvision units are believed to be. **The answer
    # moves a long way with it** -- halving the field of view doubles the useful distance
    # -- so the panel prints the value it used and says, in words, that a default nobody
    # has checked is an assumption and not a measurement.
    lens_hfov_degrees: float = Field(default=78.0, gt=0.0, lt=180.0)

    # -- the pipeline §12.1 asks for, in order -------------------------------

    # «отслеживание несколько кадров». How many observations above `model.conf` a track
    # needs before it can be considered at all. CHOSEN: 3, which at `analyse_every: 3` and
    # 15 fps is 0.6 s. NEVER 1 -- §12.1 says "НЕ отправлять тревогу по одному кадру" in
    # those words, and `_never_a_single_frame` below refuses a config that says otherwise.
    min_track_observations: int = Field(default=3, ge=2)

    # How long a track survives without being seen again. Two missed analyses at
    # `analyse_every: 3` is 0.4 s; 1.5 s allows a hand to pass behind a body and come back
    # without starting a new track. CHOSEN.
    track_idle_seconds: float = Field(default=1.5, gt=0)

    # Hard ceiling on tracks held at once (rule R8). Track ids only ever go up and the
    # legacy leaked several dicts keyed on them. 64 is far more weapon tracks than a
    # school corridor can hold; the point is that the number exists.
    max_tracks: int = Field(default=64, ge=1)

    # «проверка нахождения рядом с человеком». A weapon box must overlap a person box, or
    # sit within this many person-box diagonals of one. CHOSEN: 0.6. A person's box
    # diagonal spans head to foot; an arm's reach from the box centre is roughly 0.4 of
    # that, and the margin covers a box that clips at the frame edge. A weapon nowhere
    # near a person is a poster, a bin or a picture on a wall.
    near_person_ratio: float = Field(default=0.6, gt=0.0, le=5.0)

    # «повторное подтверждение». A SECOND and stricter gate, not a restatement of the
    # first: after a track has its observations, at least `reconfirm_observations` of them
    # must have cleared `reconfirm_confidence`. Both defaults are CHOSEN.
    #
    # This is the gate that separates "the model kept saying maybe" from "the model was
    # sure more than once", and those are the two cases §12.1's «проверка confidence ->
    # ... -> повторное подтверждение» is asking us to tell apart.
    reconfirm_confidence: float = Field(default=0.60, gt=0.0, lt=1.0)
    reconfirm_observations: int = Field(default=2, ge=1)

    # How long after an alert the same track stays quiet. One knife carried down one
    # corridor is one alert and one human decision, not one per second. CHOSEN: 60 s,
    # long enough that a confirmation can be made before the next one arrives.
    realert_seconds: float = Field(default=60.0, gt=0)

    # How much footage the evidence clip holds, ending at the alarm. Same key and same
    # meaning as `bullying.clip_seconds`, and it sizes the same `events.clip_buffer`,
    # which is bounded in BYTES underneath it (rule R8) -- so raising this cannot make a
    # worker leak.
    clip_seconds: float = Field(default=3.0, gt=0, le=15.0)

    @model_validator(mode="after")
    def _never_a_single_frame(self) -> WeaponsConfig:
        """§12.1, in the client's own words: «НЕ отправлять тревогу по одному кадру».

        `min_track_observations: int = Field(ge=2)` already refuses 1. This refuses the
        arrangement that gets to the same place by another road -- asking for more
        observations than the reconfirmation gate needs would be fine, but asking for
        FEWER means the reconfirmation could be satisfied by observations the track never
        had to collect, and the effective requirement collapses to whichever is smaller.
        """
        if self.reconfirm_observations > self.min_track_observations:
            raise ValueError(
                f"weapons: reconfirm_observations ({self.reconfirm_observations}) is "
                f"greater than min_track_observations ({self.min_track_observations}). "
                "The reconfirmation is drawn from the observations the track already "
                "collected, so asking for more of them than exist makes the alert "
                "unreachable rather than stricter."
            )
        if self.reconfirm_confidence < self.model.conf:
            raise ValueError(
                f"weapons: reconfirm_confidence ({self.reconfirm_confidence}) is below "
                f"model.conf ({self.model.conf}), so every observation that entered the "
                "track already satisfies it and the second gate checks nothing. §12.1 "
                "asks for a RE-confirmation; a gate under the entry bar is a rename."
            )
        return self

    @model_validator(mode="after")
    def _targets_and_confusables_are_disjoint(self) -> WeaponsConfig:
        unknown = sorted(set(self.target_classes) - set(KNOWN_TARGETS))
        if unknown:
            raise ValueError(
                f"weapons.target_classes names {unknown}, which are not weapon classes. "
                f"Known: {sorted(KNOWN_TARGETS)}. A target the pipeline has no rule for "
                "would alarm with no rule behind it."
            )
        both = sorted(set(self.target_classes) & set(self.confusable_classes))
        if both:
            raise ValueError(
                f"weapons: {both} appear in BOTH target_classes and confusable_classes. "
                "A class cannot be the alarm and the thing that withholds it: every "
                "sighting would suppress itself and the module would be silent while "
                "looking like it worked."
            )
        return self

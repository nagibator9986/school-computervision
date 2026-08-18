"""Which sightings survive to be tracked, and which are refused and why.

Pure. No model, no config loading, no I/O, no clock -- everything here is arithmetic on
boxes and names, so §12.1's whole rule set is testable without a GPU and without weights.
That matters more here than anywhere else in this codebase: **there are no weights to test
with, so the tests can only ever be about the rules.**

Four screens, in this order, and the order is deliberate:

  1. **Is it a target at all?** A class not in `target_classes` cannot alarm however sure
     the weights are. This is the cheapest screen and the one that disposes of most of
     §12.1's false-object list by construction rather than by tuning.
  2. **Is it big enough to be anything?** Below `min_object_pixels` a knife and a pen are
     the same handful of pixels, and no threshold anywhere changes that. Refused, and
     COUNTED -- a camera whose refusals are all this one is a camera in the wrong place,
     and that is a fact somebody needs on a screen rather than a shrug.
  3. **Does the model contradict itself here?** If it also reports a known confusable at
     the same place in the same frame, it is telling us it cannot decide between two
     labels at one location. "Нож или ручка" is not an alarm a school can act on.
  4. **Is this place one where that object is ordinary?** §12.1's kitchen clause. A
     RULE, not a raised threshold -- see `config/weapons.py::OrdinaryToolZone`.

Being near a person is checked separately (`nearest_person`) because it is not a property
of the sighting alone and because §12.1 puts it after the confidence check in the
sequence. Everything above can be decided from one frame; that one needs the people.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qorgan.config.weapons import OrdinaryToolZone, WeaponsConfig
from qorgan.detection.geometry import Box, distance, iou
from qorgan.weapons.model import Sighting

# Why a sighting went no further. A closed set of slugs rather than a sentence per call
# site, for the reason `TelegramSkipReason` gives: the legacy assembled its skip reasons
# as prose and the same cause came out worded three ways, so none of them could be
# filtered or counted. These are counted -- see `ScreenTally`.
NOT_A_TARGET = "not_a_target"
BELOW_SIZE_GATE = "below_size_gate"
AMBIGUOUS_WITH_CONFUSABLE = "ambiguous_with_confusable"
ORDINARY_IN_ZONE = "ordinary_in_zone"
NOT_NEAR_A_PERSON = "not_near_a_person"

REFUSALS = (
    NOT_A_TARGET,
    BELOW_SIZE_GATE,
    AMBIGUOUS_WITH_CONFUSABLE,
    ORDINARY_IN_ZONE,
    NOT_NEAR_A_PERSON,
)

# How much a weapon box and a confusable box must overlap before the two are treated as
# the model naming one object twice. CHOSEN: 0.30. Two labels on genuinely different
# objects a hand's width apart overlap far less than this; the same object labelled twice
# overlaps far more.
CONFUSABLE_OVERLAP = 0.30


@dataclass(frozen=True, slots=True)
class Screened:
    """A sighting that survived the frame-local screens, with the zone it was seen in."""

    sighting: Sighting
    # The zone rule that examined it, or None where no zone covers this point. Carried
    # rather than re-derived so the log line and the event agree about which rule ran.
    zone_label: str


@dataclass(frozen=True, slots=True)
class Refused:
    sighting: Sighting
    reason: str
    # What the refusal was measured against, for the log: the size gate that was missed,
    # the confusable class that contradicted it, the zone that made it ordinary.
    detail: str


def screen_frame(
    sightings: Sequence[Sighting], config: WeaponsConfig, width: int, height: int
) -> tuple[list[Screened], list[Refused]]:
    """One frame's sightings, split into those worth tracking and those refused.

    Both halves are returned. The refusals are not debris: which screen a camera's
    sightings die on is the single most useful thing for deciding whether a camera is in
    a place where this can work at all.
    """
    targets = set(config.target_classes)
    confusables = [s for s in sightings if s.class_name in set(config.confusable_classes)]

    kept: list[Screened] = []
    refused: list[Refused] = []
    for sighting in sightings:
        if sighting.class_name not in targets:
            refused.append(Refused(sighting, NOT_A_TARGET, sighting.class_name))
            continue
        verdict = _screen_one(sighting, config, confusables, width, height)
        (refused if isinstance(verdict, Refused) else kept).append(verdict)  # type: ignore[arg-type]
    return kept, refused


def _screen_one(
    sighting: Sighting,
    config: WeaponsConfig,
    confusables: Sequence[Sighting],
    width: int,
    height: int,
) -> Screened | Refused:
    """Screens 2 to 4 for one target-class sighting."""
    if sighting.size_pixels < config.min_object_pixels:
        return Refused(
            sighting,
            BELOW_SIZE_GATE,
            f"{sighting.size_pixels:.0f}px < {config.min_object_pixels:.0f}px",
        )

    contradiction = _contradicted_by(sighting, confusables)
    if contradiction is not None:
        return Refused(sighting, AMBIGUOUS_WITH_CONFUSABLE, contradiction)

    zone = ordinary_zone_for(sighting, config.zones.ordinary_tools, width, height)
    if zone is not None:
        return Refused(sighting, ORDINARY_IN_ZONE, zone.label or "зона")
    return Screened(sighting=sighting, zone_label="")


def _contradicted_by(sighting: Sighting, confusables: Sequence[Sighting]) -> str | None:
    """Did the model put a known false object on the same pixels?

    Screen 3. Returns the contradicting class name, or None. Deliberately not a
    confidence comparison: which of the two labels scored higher is exactly the judgement
    the model has already shown it cannot make reliably at this size.
    """
    for other in confusables:
        if iou(sighting.box, other.box) >= CONFUSABLE_OVERLAP:
            return other.class_name
    return None


def ordinary_zone_for(
    sighting: Sighting, zones: Sequence[OrdinaryToolZone], width: int, height: int
) -> OrdinaryToolZone | None:
    """§12.1's kitchen clause: is this object ordinary WHERE IT IS?

    The centre of the box decides, in normalised frame coordinates -- the same convention
    every zone in this project uses, so that a zone drawn once survives a change of
    `capture.frame_width`.

    A class not named by the zone is untouched by it. A firearm in a kitchen is still a
    firearm, and a zone that silenced everything inside it would be an off switch with a
    rule's name on it.
    """
    center = sighting.box.normalized_center(width, height)
    for zone in zones:
        if sighting.class_name in zone.classes and zone.area.contains(*center):
            return zone
    return None


def nearest_person(
    sighting: Sighting, people: Mapping[int, Box], ratio: float
) -> tuple[int, float] | None:
    """§12.1's «проверка нахождения рядом с человеком»: which person, and how far.

    Returns the track id and the centre-to-centre distance, or None.

    Two ways to be near, because one of them is not enough. A knife held in a hand is
    usually INSIDE the person's box, where a centre-to-centre distance is large and
    meaningless -- the person's centre is their chest and the knife is at their hip. So
    an overlap counts on its own. A knife just leaving a hand, or a hand at the edge of a
    clipped box, is outside it, and that is what the distance term is for.

    Scaled by the PERSON's box diagonal rather than by a pixel constant: a person twice as
    close is twice as big, which is the same reasoning `dynamic_threshold` uses and the
    only thing that makes one number work at both ends of a corridor.
    """
    best: tuple[int, float] | None = None
    for track_id, person in people.items():
        if iou(sighting.box, person) > 0.0:
            gap = 0.0
        else:
            gap = distance(sighting.box.center, person.center)
            if gap > ratio * person.diagonal:
                continue
        if best is None or gap < best[1]:
            best = (track_id, gap)
    return best

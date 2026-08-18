"""The four screens, and §12.1's «проверка нахождения рядом с человеком».

Pure arithmetic on boxes and names, which is the only kind of test available here: **we
have no weapons model**, so nothing can be asserted about what the weights see. What CAN
be asserted is what the module does with what they say, and that is all of §12.1's false
object handling.

**On the kitchen.** The client answered on 2026-07-29 that the kitchen is OUT of the
controlled area, so no kitchen-specific rules were built. What was already built is the
general `ordinary_tools` zone rule, and it is kept and tested here because it is read by
production code (`rules.ordinary_zone_for`, called from `screen_frame`) rather than being
a knob nothing consumes. The tests below are therefore about the ZONE RULE -- one place
where a class is ordinary and elsewhere it is not -- and the kitchen is only ever the
example in the label. `test_a_zone_does_not_silence_a_class_it_does_not_name` is the one
that matters: a zone is a rule about a class, never an off switch for a rectangle.
"""

from __future__ import annotations

import pytest

from qorgan.config.weapons import OrdinaryToolZone, WeaponsConfig, WeaponsZones
from qorgan.detection.geometry import Box
from qorgan.weapons.model import Sighting
from qorgan.weapons.rules import (
    AMBIGUOUS_WITH_CONFUSABLE,
    BELOW_SIZE_GATE,
    NOT_A_TARGET,
    ORDINARY_IN_ZONE,
    nearest_person,
    ordinary_zone_for,
    screen_frame,
)
from tests.weapons_fixtures import person_box, sighting

WIDTH, HEIGHT = 960, 540


def _config(**overrides) -> WeaponsConfig:
    base = {"model": {"model": "qorgan-weapons.pt"}, "target_classes": ["knife", "firearm"]}
    base.update(overrides)
    return WeaponsConfig.model_validate(base)


def _kitchen(**overrides) -> WeaponsConfig:
    """A zone over the LEFT HALF of the frame in which a knife is an ordinary tool."""
    zone = OrdinaryToolZone(
        area={"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0},
        classes=["knife"],
        label="кухня столовой",
    )
    return _config(zones=WeaponsZones(ordinary_tools=[zone]), **overrides)


# -- screen 1: a class nobody named cannot alarm ---------------------------


def test_a_class_not_in_target_classes_never_alarms() -> None:
    """§12.1's false-object list, handled by construction rather than by tuning. A model
    shipping a hundred COCO classes emits them into a log line."""
    kept, refused = screen_frame([sighting("scissors", 0.99)], _config(), WIDTH, HEIGHT)
    assert kept == []
    assert [r.reason for r in refused] == [NOT_A_TARGET]


def test_certainty_does_not_promote_a_non_target() -> None:
    kept, _ = screen_frame([sighting("toy", 1.0, size=400)], _config(), WIDTH, HEIGHT)
    assert kept == []


# -- screen 2: the size gate, where physics stops being negotiable ---------


def test_an_object_below_the_size_gate_is_refused_and_counted() -> None:
    """15 px is the corridor case. It is not a low-confidence detection; it is not a
    detection. And it is COUNTED, because a camera whose refusals are all this one is a
    camera in the wrong place -- a fact for a screen, not a shrug."""
    kept, refused = screen_frame([sighting(size=15.0)], _config(), WIDTH, HEIGHT)
    assert kept == []
    assert [r.reason for r in refused] == [BELOW_SIZE_GATE]
    assert "15px" in refused[0].detail and "24px" in refused[0].detail


def test_the_size_gate_measures_the_LONGER_side() -> None:
    """A blade held in a hand is long and narrow. Judging it by width or area would
    refuse the exact shape this module exists to find."""
    tall = Sighting(class_name="knife", confidence=0.9, box=Box(100, 100, 108, 140))
    assert tall.size_pixels == 40.0

    kept, _ = screen_frame([tall], _config(), WIDTH, HEIGHT)
    assert len(kept) == 1, "8 px wide but 40 px long: that is a knife, not a speck"


def test_an_object_over_the_gate_survives() -> None:
    kept, refused = screen_frame([sighting(size=40.0)], _config(), WIDTH, HEIGHT)
    assert len(kept) == 1 and refused == []


# -- screen 3: the model contradicting itself ------------------------------


def test_a_weapon_claim_overlapping_a_known_confusable_is_withheld() -> None:
    """«Нож или ручка» is not an alarm a school can act on. If the weights put two labels
    on one place, they are telling us they cannot decide."""
    knife = sighting("knife", 0.9, x1=100, y1=100, size=40)
    pen = sighting("pen", 0.8, x1=100, y1=100, size=40)

    kept, refused = screen_frame([knife, pen], _config(), WIDTH, HEIGHT)
    assert kept == []
    reasons = [r.reason for r in refused]
    assert AMBIGUOUS_WITH_CONFUSABLE in reasons
    assert refused[reasons.index(AMBIGUOUS_WITH_CONFUSABLE)].detail == "pen"


def test_a_confusable_somewhere_else_in_the_frame_changes_nothing() -> None:
    """A pupil holding a pen across the corridor does not disarm a knife."""
    knife = sighting("knife", 0.9, x1=100, y1=100, size=40)
    pen = sighting("pen", 0.8, x1=600, y1=400, size=40)

    kept, _ = screen_frame([knife, pen], _config(), WIDTH, HEIGHT)
    assert len(kept) == 1


def test_the_contradiction_is_not_settled_by_whichever_scored_higher() -> None:
    """Which of two labels on one object scored higher is exactly the judgement the model
    has already shown it cannot make at this size."""
    knife = sighting("knife", 0.99, x1=100, y1=100, size=40)
    pen = sighting("pen", 0.30, x1=100, y1=100, size=40)

    kept, _ = screen_frame([knife, pen], _config(), WIDTH, HEIGHT)
    assert kept == [], "0.99 against 0.30 is still two labels on one object"


# -- screen 4: the zone rule (a RULE, not a raised threshold) --------------


def test_a_named_class_inside_the_zone_does_not_alarm_at_any_confidence() -> None:
    """A different RULE, not a different number. Raising a threshold in a kitchen means a
    cook raises the alarm slightly less often than twenty times a day: the same product."""
    inside = sighting("knife", 1.0, x1=100, y1=200, size=40)  # left half
    kept, refused = screen_frame([inside], _kitchen(), WIDTH, HEIGHT)

    assert kept == []
    assert [r.reason for r in refused] == [ORDINARY_IN_ZONE]
    assert refused[0].detail == "кухня столовой"


def test_the_same_class_outside_the_zone_still_alarms() -> None:
    """The divergence, in one pair of tests: identical object, identical confidence,
    different place, opposite answer."""
    outside = sighting("knife", 1.0, x1=700, y1=200, size=40)  # right half
    kept, refused = screen_frame([outside], _kitchen(), WIDTH, HEIGHT)
    assert len(kept) == 1 and refused == []


def test_a_zone_does_not_silence_a_class_it_does_not_name() -> None:
    """**A firearm in a kitchen is still a firearm.** A zone that silenced everything
    inside it would be an off switch wearing a rule's clothes."""
    gun = sighting("firearm", 0.9, x1=100, y1=200, size=40)  # inside the zone
    kept, refused = screen_frame([gun], _kitchen(), WIDTH, HEIGHT)
    assert len(kept) == 1 and refused == []


def test_the_zone_is_read_in_fractions_of_the_frame_not_pixels() -> None:
    """A zone drawn once must survive a change of `capture.frame_width`. The same object
    at the same FRACTION of two differently sized frames gets the same answer."""
    zones = _kitchen().zones.ordinary_tools
    small = sighting("knife", 0.9, x1=100, y1=100, size=40)  # centre x = 120/960 = 0.125
    big = sighting("knife", 0.9, x1=200, y1=200, size=80)  # centre x = 240/1920 = 0.125

    assert ordinary_zone_for(small, zones, 960, 540) is not None
    assert ordinary_zone_for(big, zones, 1920, 1080) is not None


def test_no_zone_means_no_exemption() -> None:
    assert ordinary_zone_for(sighting(), [], WIDTH, HEIGHT) is None


# -- «проверка нахождения рядом с человеком» -------------------------------


def test_a_weapon_inside_a_person_box_is_near_that_person() -> None:
    """A knife in a hand is usually INSIDE the person's box, where a centre-to-centre
    distance is large and meaningless: the chest is the centre, the hand is at the hip.
    So overlap counts on its own."""
    found = nearest_person(sighting(), {7: person_box()}, ratio=0.6)
    assert found is not None
    assert found == (7, 0.0)


def test_a_weapon_nowhere_near_anybody_is_refused() -> None:
    """A poster, a bin, a picture on a wall."""
    far = sighting(x1=900, y1=480, size=30)
    assert nearest_person(far, {7: person_box()}, ratio=0.6) is None


def test_an_empty_frame_has_nobody_to_be_near() -> None:
    assert nearest_person(sighting(), {}, ratio=0.6) is None


def test_the_nearest_person_is_the_one_reported() -> None:
    """The operator opens the clip looking for this track id."""
    people = {1: person_box(x1=90, y1=60), 2: Box(600, 60, 660, 220)}
    found = nearest_person(sighting(), people, ratio=0.6)
    assert found is not None and found[0] == 1


def test_nearness_is_scaled_by_the_person_and_not_a_pixel_constant() -> None:
    """A person twice as close is twice as big. One number has to work at both ends of a
    corridor, which a fixed pixel gap cannot do."""
    gap = sighting(x1=300, y1=100, size=20)
    near_camera = {1: Box(60, 40, 240, 520)}  # a big box: long diagonal
    down_corridor = {1: Box(200, 100, 220, 150)}  # a small one

    assert nearest_person(gap, near_camera, ratio=0.6) is not None
    assert nearest_person(gap, down_corridor, ratio=0.6) is None


# -- the order the screens run in ------------------------------------------


def test_every_refusal_reason_is_a_slug_from_the_closed_set() -> None:
    """The legacy wrote these as prose and the same cause came out worded three ways, so
    none of them could be counted."""
    from qorgan.weapons.rules import REFUSALS

    crowd = [
        sighting("scissors", 0.9),
        sighting("knife", 0.9, x1=500, y1=100, size=10),
        sighting("knife", 1.0, x1=100, y1=200, size=40),
    ]
    _, refused = screen_frame(crowd, _kitchen(), WIDTH, HEIGHT)
    assert {r.reason for r in refused} <= set(REFUSALS)
    assert len(refused) == 3


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [("scissors", NOT_A_TARGET), ("knife", BELOW_SIZE_GATE)],
)
def test_the_cheapest_screen_runs_first(class_name: str, expected: str) -> None:
    """A non-target below the size gate is refused as a non-target: the size check never
    runs. Which screen a sighting dies on is the diagnostic, so the order is load-bearing."""
    _, refused = screen_frame([sighting(class_name, 0.9, size=5.0)], _config(), WIDTH, HEIGHT)
    assert [r.reason for r in refused] == [expected]

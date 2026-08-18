"""Event merging: one incident, one alert.

Without this, a four-second fight becomes dozens of database rows and dozens of Telegram
messages — which trains staff to ignore the alerts, which is worse than having none.
"""

from __future__ import annotations

from qorgan.config.bullying import EventMerge
from qorgan.detection.merging import EventMerger

CONFIG = EventMerge()
CENTER = (0.5, 0.5)


def _merger(**overrides) -> EventMerger:
    return EventMerger(EventMerge.model_validate(overrides))


def test_the_first_event_is_always_new() -> None:
    merger = _merger()
    assert merger.decide((1, 2), 0.0, CENTER).is_new


def test_the_same_pair_again_seconds_later_is_the_same_incident() -> None:
    merger = _merger()
    merger.remember((1, 2), 0.0, 0.9, CENTER)

    decision = merger.decide((1, 2), 3.0, CENTER)

    assert not decision.is_new
    assert decision.reason == "same_pair"
    assert decision.merged_into == (1, 2)


def test_the_same_pair_much_later_is_a_new_incident() -> None:
    """Two children who fight, separate, and fight again a minute later have had two
    fights. The system should say so."""
    merger = _merger()
    merger.remember((1, 2), 0.0, 0.9, CENTER)

    assert merger.decide((1, 2), 60.0, CENTER).is_new


def test_a_retracked_child_does_not_produce_a_second_event() -> None:
    """The tracker loses one child mid-fight and re-acquires them with a fresh id. It is
    the same fight, and the legacy would have logged it twice."""
    merger = _merger()
    merger.remember((1, 2), 0.0, 0.9, CENTER)

    decision = merger.decide((1, 7), 2.0, CENTER)

    assert not decision.is_new
    assert decision.reason == "same_scene"
    assert decision.merged_into == (1, 2)


def test_a_scuffle_in_the_same_corner_seconds_later_is_the_same_scuffle() -> None:
    """The tracker lost BOTH children. Nothing links the ids -- only the place does."""
    merger = _merger()
    merger.remember((1, 2), 0.0, 0.9, (0.30, 0.40))

    decision = merger.decide((8, 9), 3.0, (0.34, 0.43))

    assert not decision.is_new
    assert decision.reason == "same_place"


def test_a_fight_at_the_other_end_of_the_hall_is_a_different_fight() -> None:
    merger = _merger()
    merger.remember((1, 2), 0.0, 0.9, (0.1, 0.1))

    assert merger.decide((8, 9), 3.0, (0.9, 0.9)).is_new


def test_spatial_merging_can_be_turned_off() -> None:
    merger = _merger(merge_by_spatial=False)
    merger.remember((1, 2), 0.0, 0.9, (0.30, 0.40))

    assert merger.decide((8, 9), 3.0, (0.31, 0.41)).is_new


def test_the_same_pair_rule_survives_merging_being_disabled() -> None:
    """Even with merging off, one fight must not alert on every frame. This rule is the
    difference between one alert per incident and one alert per frame."""
    merger = _merger(enabled=False)
    merger.remember((1, 2), 0.0, 0.9, CENTER)

    decision = merger.decide((1, 2), 1.0, CENTER)

    assert not decision.is_new
    assert decision.reason == "same_pair"


def test_a_different_pair_is_new_when_merging_is_disabled() -> None:
    merger = _merger(enabled=False)
    merger.remember((1, 2), 0.0, 0.9, CENTER)

    assert merger.decide((3, 4), 1.0, CENTER).is_new


def test_the_registry_is_bounded() -> None:
    """Rule R8. A camera runs for months; this dict must not grow for months."""
    merger = _merger()

    for index in range(500):
        key = (index * 2, index * 2 + 1)
        merger.decide(key, index * 1.0, CENTER)
        merger.remember(key, index * 1.0, 0.9, CENTER)

    assert len(merger) < 30, f"the merge registry grew to {len(merger)} entries"

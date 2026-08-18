"""One test per suppression gate.

These are the anti-false-positive core of the whole system: the accumulated knowledge of
what, in a real school hallway, merely *looks* like a fight. In the legacy this lived as
a thousand lines of `if ...: continue` inside a 940-line function and had zero tests, so
nobody could change a threshold without risking the lot.

Each test states the real-world scene it protects against.
"""

from __future__ import annotations

import pytest

from qorgan.config.bullying import BullyingConfig
from qorgan.detection.gates import (
    GATE_NAMES,
    GATES,
    GateInput,
    benign_conversation,
    crossing_pass,
    first_firing,
    hall_final_confirmation,
    normal_flow_motion_required,
    proximity_only,
    social_group,
    social_reapproach,
    staircase_pass,
    standing_close_long,
    static_close,
)
from qorgan.detection.pairs import PairFrame, PairState
from qorgan.detection.scoring import Signals, ZoneContext

CONFIG = BullyingConfig()
METRICS = CONFIG.metrics


def _frame(**overrides) -> PairFrame:
    """A neutral pair: close, calm, and NOT touching.

    With threshold 200 and contact_ratio 0.6 the contact bar is at 120 px and the
    hard-contact bar at 108, so the gap must sit above both while staying under 200.
    """
    defaults = {
        "key": (1, 2),
        "gap": 150.0,
        "threshold": 200.0,
        "contact_ratio": METRICS.contact_distance_ratio,
        "overlap": 0.0,
        "approach": 0.0,
        "relative_speed": 0.0,
        "max_acceleration": 0.0,
        "max_direction_change": 0.0,
        "max_speed": 0.0,
    }
    return PairFrame(**{**defaults, **overrides})


def _state(**overrides) -> PairState:
    state = PairState(key=(1, 2))
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def _signals(**overrides) -> Signals:
    defaults = {
        "strong_proximity_drop": False,
        "strong_contact": False,
        "strong_motion_spike": False,
        "strong_direction_change": False,
        "same_direction": False,
        "low_motion": True,
        "motion_present": False,
    }
    return Signals(**{**defaults, **overrides})


def _input(
    frame: PairFrame | None = None,
    state: PairState | None = None,
    signals: Signals | None = None,
    zone: ZoneContext | None = None,
    a_speed: float = 0.0,
    b_speed: float = 0.0,
    config: BullyingConfig | None = None,
) -> GateInput:
    return GateInput(
        frame=frame or _frame(),
        state=state or _state(),
        signals=signals or _signals(),
        zone=zone or ZoneContext(),
        config=config or CONFIG,
        a_speed=a_speed,
        b_speed=b_speed,
    )


# -- the rule list itself --------------------------------------------------


def test_all_ten_gates_are_registered() -> None:
    assert len(GATES) == 10
    assert GATE_NAMES == (
        "static_close",
        "standing_close_long",
        "social_group",
        "social_reapproach",
        "proximity_only",
        "normal_flow_motion_required",
        "crossing_pass",
        "staircase_pass",
        "hall_final_confirmation",
        "benign_conversation",
    )


def test_a_neutral_pair_is_reported_by_the_gate_that_caught_it() -> None:
    """first_firing names the gate, so an operator (and the eval harness) can see WHY
    a pair was suppressed rather than just that it was."""
    calm = _input(state=_state(still_frames=10))
    assert first_firing(calm, post_confirmation=False) == "static_close"


# -- 1. static_close -------------------------------------------------------


def test_static_close_suppresses_two_children_chatting_by_a_locker() -> None:
    data = _input(state=_state(still_frames=5))
    assert static_close(data)


def test_static_close_does_not_suppress_a_pair_that_is_actually_grappling() -> None:
    """Standing still is no defence once somebody has grabbed somebody."""
    data = _input(state=_state(still_frames=10, contact_frames=3, overlap_frames=3))
    assert not static_close(data)


def test_static_close_does_not_suppress_hard_contact() -> None:
    data = _input(frame=_frame(overlap=0.5), state=_state(still_frames=10))
    assert not static_close(data)


# -- 2. standing_close_long ------------------------------------------------


def test_standing_close_long_suppresses_a_long_calm_conversation() -> None:
    data = _input(state=_state(still_frames=12))
    assert standing_close_long(data)


def test_a_sudden_attack_after_standing_still_is_NOT_suppressed() -> None:
    """The most important bypass in the system. A long quiet build-up followed by a
    strike is exactly what real bullying looks like; a gate that swallowed it would be
    worse than no gate at all."""
    lunge = _frame(max_acceleration=METRICS.acceleration_threshold * 2)
    data = _input(frame=lunge, state=_state(still_frames=20))
    assert not standing_close_long(data), "the gate swallowed a real attack"


def test_a_sudden_grab_after_standing_still_is_NOT_suppressed() -> None:
    data = _input(state=_state(still_frames=20, contact_frames=2, overlap_frames=1))
    assert not standing_close_long(data)


# -- 3. social_group -------------------------------------------------------


def test_social_group_suppresses_a_class_walking_to_a_lesson() -> None:
    data = _input(
        zone=ZoneContext(in_normal_flow=True),
        signals=_signals(same_direction=True),
    )
    assert social_group(data)


def test_social_group_only_applies_inside_a_corridor_lane() -> None:
    data = _input(zone=ZoneContext(in_normal_flow=False), signals=_signals(same_direction=True))
    assert not social_group(data)


def test_social_group_does_not_suppress_a_fight_inside_the_corridor() -> None:
    data = _input(
        frame=_frame(overlap=0.5, max_acceleration=METRICS.acceleration_threshold * 3),
        zone=ZoneContext(in_normal_flow=True),
        signals=_signals(same_direction=True),
    )
    assert not social_group(data)


# -- 4. social_reapproach --------------------------------------------------


def test_social_reapproach_suppresses_friends_orbiting_each_other() -> None:
    """Close, drifted apart, close again -- and never once touching. A fight does not
    politely separate and come back."""
    data = _input(
        state=_state(peak_close_frames=6, gap_frames=5, gaps=[210.0, 180.0, 150.0]),
    )
    assert social_reapproach(data)


def test_social_reapproach_does_not_fire_if_they_really_drifted_apart() -> None:
    data = _input(state=_state(peak_close_frames=6, gap_frames=5, gaps=[2000.0, 150.0]))
    assert not social_reapproach(data)


def test_social_reapproach_does_not_suppress_a_pair_that_made_contact() -> None:
    data = _input(state=_state(peak_close_frames=6, gap_frames=5, contact_frames=4, gaps=[210.0]))
    assert not social_reapproach(data)


# -- 5. proximity_only -----------------------------------------------------


def test_proximity_alone_never_raises_an_alert() -> None:
    """If it could, a crowded corridor would alert continuously."""
    data = _input(signals=_signals(motion_present=False))
    assert proximity_only(data)


def test_proximity_with_motion_is_not_suppressed() -> None:
    data = _input(signals=_signals(motion_present=True))
    assert not proximity_only(data)


# -- 6. normal_flow_motion_required ----------------------------------------


def test_a_corridor_lane_demands_a_strong_action_signal() -> None:
    """People walking past each other in a corridor are the single biggest source of
    false positives in the entire system."""
    data = _input(zone=ZoneContext(in_normal_flow=True))
    assert normal_flow_motion_required(data)


def test_a_violent_shove_in_the_corridor_still_gets_through() -> None:
    shove = _frame(max_acceleration=METRICS.acceleration_threshold * 1.5)
    data = _input(frame=shove, zone=ZoneContext(in_normal_flow=True))
    assert not normal_flow_motion_required(data)


def test_hard_contact_in_the_corridor_gets_through() -> None:
    data = _input(frame=_frame(overlap=0.4), zone=ZoneContext(in_normal_flow=True))
    assert not normal_flow_motion_required(data)


# -- 7. crossing_pass ------------------------------------------------------


def test_crossing_pass_suppresses_two_people_walking_past_each_other() -> None:
    walking = METRICS.speed_threshold * 0.5
    data = _input(state=_state(close_frames=2), a_speed=walking, b_speed=walking)
    assert crossing_pass(data)


def test_a_shove_delivered_in_passing_is_NOT_suppressed() -> None:
    """It looks exactly like a crossing until you notice the acceleration."""
    walking = METRICS.speed_threshold * 0.5
    shove = _frame(max_acceleration=METRICS.acceleration_threshold)
    data = _input(frame=shove, state=_state(close_frames=2), a_speed=walking, b_speed=walking)
    assert not crossing_pass(data), "the gate swallowed a shove in passing"


def test_a_lingering_encounter_is_not_a_crossing() -> None:
    walking = METRICS.speed_threshold * 0.5
    data = _input(state=_state(close_frames=20), a_speed=walking, b_speed=walking)
    assert not crossing_pass(data)


# -- 8. staircase_pass -----------------------------------------------------


def test_staircase_pass_suppresses_someone_walking_past_someone_standing() -> None:
    """On a staircase there is nowhere else to go. Passing close is not a choice."""
    gate = CONFIG.gates.staircase_pass
    data = _input(
        zone=ZoneContext(in_staircase=True),
        a_speed=gate.static_speed_threshold - 1,
        b_speed=gate.moving_speed_threshold + 1,
    )
    assert staircase_pass(data)


def test_staircase_pass_does_not_apply_off_the_stairs() -> None:
    gate = CONFIG.gates.staircase_pass
    data = _input(
        zone=ZoneContext(in_staircase=False),
        a_speed=gate.static_speed_threshold - 1,
        b_speed=gate.moving_speed_threshold + 1,
    )
    assert not staircase_pass(data)


def test_a_real_attack_on_the_stairs_still_gets_through() -> None:
    gate = CONFIG.gates.staircase_pass
    data = _input(
        frame=_frame(overlap=0.5),
        zone=ZoneContext(in_staircase=True),
        a_speed=gate.static_speed_threshold - 1,
        b_speed=gate.moving_speed_threshold + 1,
    )
    assert not staircase_pass(data)


# -- 9. hall_final_confirmation --------------------------------------------


def test_the_hall_demands_a_sustained_grapple_before_alerting() -> None:
    """The main hall generated more false positives than every other camera combined."""
    config = BullyingConfig.model_validate(
        {"gates": {"hall_confirmation": {"enabled": True, "sustained_contact_min": 3}}}
    )
    data = _input(config=config)
    assert hall_final_confirmation(data)


def test_a_sustained_grapple_in_the_hall_passes() -> None:
    config = BullyingConfig.model_validate(
        {"gates": {"hall_confirmation": {"enabled": True, "sustained_contact_min": 3}}}
    )
    data = _input(state=_state(contact_frames=5), config=config)
    assert not hall_final_confirmation(data)


def test_the_hall_gate_is_off_unless_a_camera_turns_it_on() -> None:
    assert not hall_final_confirmation(_input())


# -- 10. benign_conversation -----------------------------------------------


def test_benign_conversation_suppresses_a_confirmed_pair_that_never_displaced_anyone() -> None:
    """The deepest idea in the system: an assault displaces its victim. If nobody was
    pushed, pulled, knocked down or made to flinch, it was not an assault."""
    data = _input(state=_state(still_frames=5, overlap_frames=1))
    assert benign_conversation(data)


def test_a_victim_who_was_actually_shoved_is_not_benign() -> None:
    shoved = _frame(max_acceleration=METRICS.acceleration_threshold * 2)
    data = _input(frame=shoved, state=_state(still_frames=5))
    assert not benign_conversation(data)


def test_benign_conversation_runs_only_after_confirmation() -> None:
    """It is the last line of defence, not the first: it can only judge a pair the rest
    of the pipeline already believes in."""
    gate = next(g for g in GATES if g.name == "benign_conversation")
    assert gate.post_confirmation

    data = _input(state=_state(still_frames=5, overlap_frames=1))
    assert first_firing(data, post_confirmation=True) == "benign_conversation"


@pytest.mark.parametrize("gate", GATES, ids=lambda g: g.name)
def test_every_gate_is_pure_and_side_effect_free(gate) -> None:
    """A gate must not mutate the pair it is judging: the pipeline decides what to do
    about a suppression, not the gate."""
    data = _input(state=_state(still_frames=9, contact_frames=1, close_frames=3))
    before = (data.state.contact_frames, data.state.close_frames, data.state.aggression_frames)

    gate.fires(data)

    after = (data.state.contact_frames, data.state.close_frames, data.state.aggression_frames)
    assert before == after, f"{gate.name} mutated the pair state"

"""The ten suppression gates. This is the anti-false-positive core of the system.

Everything else in the detector decides that two people *might* be fighting. These
decide that they are not — that they are friends talking, or a group walking to class,
or two children passing on a staircase. That knowledge was earned over months in a real
school hallway, and it is the single most valuable thing the legacy project produced.

In the legacy it was a thousand lines of `if ...: continue` buried inside a 940-line
function, and it could not be tested, reasoned about, or measured. Here each gate is a
named predicate with a docstring saying what it protects against and a unit test proving
it fires. The pipeline runs them as a list.

A gate that fires **drains the escalation counters** and suppresses the pair for this
frame. Draining, not zeroing: the pair can still build back up if it really is a fight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qorgan.config.bullying import BullyingConfig
from qorgan.detection.constants import (
    BRIEF_ENCOUNTER_FRAMES,
    FIRM_FRAMES,
    SUSTAINED_FRAMES,
)
from qorgan.detection.pairs import PairFrame, PairState
from qorgan.detection.scoring import Signals, ZoneContext


@dataclass(frozen=True, slots=True)
class GateInput:
    """Everything a gate is allowed to look at. Gates are pure: no I/O, no clock."""

    frame: PairFrame
    state: PairState
    signals: Signals
    zone: ZoneContext
    config: BullyingConfig
    a_speed: float
    b_speed: float


GatePredicate = Callable[[GateInput], bool]


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    predicate: GatePredicate
    # Post-confirmation gates run only on an already-confirmed pair and block the alert
    # without draining the counters -- the pair is genuinely engaged, we simply do not
    # believe THIS is an assault.
    post_confirmation: bool = False

    def fires(self, data: GateInput) -> bool:
        return self.predicate(data)


# -- the shared precondition ------------------------------------------------


def is_static_close(data: GateInput) -> bool:
    """Two people standing close together, not moving, not touching.

    The most common non-event in a school: children talking. Note it self-disables the
    moment there is real contact or overlap -- standing still is not a defence once
    somebody has grabbed somebody.
    """
    gate = data.config.gates.static_close
    return (
        data.state.still_frames >= gate.min_frames
        and data.state.contact_frames < SUSTAINED_FRAMES
        and data.state.overlap_frames < SUSTAINED_FRAMES
        and not data.frame.hard_contact
    )


# -- 1. static_close --------------------------------------------------------


def static_close(data: GateInput) -> bool:
    """Close, but still. Two children chatting by a locker."""
    return data.config.gates.static_close.enabled and is_static_close(data)


# -- 2. standing_close_long -------------------------------------------------

STANDING_LONG_FRAMES = 8


def standing_close_long(data: GateInput) -> bool:
    """The sustained version, with the bypass that makes it safe.

    A pair can stand together for a long time and *then* one of them attacks. The
    `sudden_action_after_static` bypass is what stops this gate from swallowing exactly
    that -- the most dangerous case in the whole system, because a long quiet build-up
    is what real bullying looks like.
    """
    gate = data.config.gates.static_close
    if not gate.enabled:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    if gate.sudden_action_bypass_enabled and _sudden_action(data):
        return False

    return (
        state.still_frames >= STANDING_LONG_FRAMES
        and not frame.hard_contact
        and not (
            state.contact_frames >= FIRM_FRAMES
            and state.window_drop > metrics.window_drop_threshold
        )
        and frame.max_acceleration < metrics.acceleration_threshold
        and frame.approach < metrics.speed_threshold * 0.6
    )


def _sudden_action(data: GateInput) -> bool:
    """Something just happened after a long quiet stretch. Let it through."""
    metrics = data.config.metrics
    return (
        data.frame.max_acceleration >= metrics.acceleration_threshold
        or data.state.window_drop > metrics.window_drop_threshold * 2
        or (data.state.contact_frames >= SUSTAINED_FRAMES and data.state.overlap_frames >= 1)
    )


# -- 3. social_group --------------------------------------------------------


def social_group(data: GateInput) -> bool:
    """A group in a corridor lane, all moving the same way. That is a class changeover."""
    gate = data.config.gates.social_group
    if not gate.enabled or not data.zone.in_normal_flow or data.frame.hard_contact:
        return False

    metrics = data.config.metrics
    frame = data.frame

    calm = (
        frame.max_acceleration < metrics.acceleration_threshold * 0.6
        and data.state.window_drop < metrics.window_drop_threshold * 0.8
    )
    if not calm:
        return False

    drifting_together = (
        data.a_speed < metrics.speed_threshold * 0.5
        and data.b_speed < metrics.speed_threshold * 0.5
        and frame.max_direction_change < metrics.direction_change_threshold_deg * 0.5
    )
    return data.signals.same_direction or drifting_together


# -- 4. social_reapproach ---------------------------------------------------


def social_reapproach(data: GateInput) -> bool:
    """Close, drifted apart, came back together. Friends, orbiting each other.

    A fight does not do this: people who are fighting do not politely separate and
    re-approach within a few seconds without ever making contact.
    """
    gate = data.config.gates.social_reapproach
    if not gate.enabled:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    # They drifted apart, but never very far.
    drift_limit = frame.threshold * (1.0 + gate.distance_delta_ratio)

    return (
        state.peak_close_frames >= gate.min_prior_close_frames
        and 0 < state.gap_frames <= gate.max_gap_frames
        and state.max_window_gap <= drift_limit
        and not frame.hard_contact
        and frame.max_acceleration < metrics.acceleration_threshold * 1.2
        and frame.max_direction_change < metrics.direction_change_threshold_deg * 1.2
        and state.overlap_frames < FIRM_FRAMES
        and state.contact_frames < FIRM_FRAMES
    )


# -- 5. proximity_only ------------------------------------------------------


def proximity_only(data: GateInput) -> bool:
    """Close, and nothing else. Proximity alone must never raise an alert -- if it
    could, a crowded corridor would alert continuously."""
    if not data.config.gates.proximity_only.enabled:
        return False
    if not data.config.gates.require_motion_for_alert:
        return False
    return data.frame.is_close and not data.signals.motion_present


# -- 6. normal_flow_motion_required -----------------------------------------


def normal_flow_motion_required(data: GateInput) -> bool:
    """Inside a corridor lane, demand a genuinely strong action signal.

    People walking past each other in a corridor are the dominant false-positive source
    in the entire system. Inside a flow zone the ordinary evidence bar is not enough.
    """
    gate = data.config.gates.normal_flow_motion
    if not gate.enabled or not data.zone.in_normal_flow:
        return False
    if not data.config.gates.require_motion_for_alert:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    has_action = (
        frame.hard_contact
        or state.window_drop > metrics.window_drop_threshold * 1.5
        or frame.max_acceleration > metrics.acceleration_threshold * 1.2
        or frame.approach > metrics.speed_threshold * 1.2
        or (
            state.overlap_frames >= FIRM_FRAMES
            and frame.max_direction_change > metrics.direction_change_threshold_deg
        )
        or (
            state.contact_frames >= FIRM_FRAMES
            and frame.relative_speed > metrics.min_relative_speed
        )
    )
    return not has_action


# -- 7. crossing_pass -------------------------------------------------------


def crossing_pass(data: GateInput) -> bool:
    """Two people walking past each other. Brief, close, and utterly ordinary.

    The action-evidence bypass matters: a shove delivered in passing is a real attack,
    and it looks exactly like a crossing until you notice the acceleration.
    """
    gate = data.config.gates.crossing_pass
    if not gate.enabled:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    both_moving = (
        data.a_speed > metrics.speed_threshold * 0.25
        and data.b_speed > metrics.speed_threshold * 0.25
    )
    brief = state.close_frames <= BRIEF_ENCOUNTER_FRAMES
    no_hard_signals = (
        not frame.hard_contact
        and state.contact_frames < gate.max_contact_frames
        and state.overlap_frames < gate.max_overlap_frames
        and frame.max_acceleration < metrics.acceleration_threshold * 0.9
        and state.window_drop < metrics.window_drop_threshold * 1.5
    )

    if gate.action_evidence_bypass_enabled and _action_evidence(data):
        return False

    return both_moving and brief and no_hard_signals


def _action_evidence(data: GateInput) -> bool:
    """A shove in passing. Not a crossing after all."""
    metrics = data.config.metrics
    return (
        data.frame.max_acceleration >= metrics.acceleration_threshold * 0.9
        or data.state.window_drop >= metrics.window_drop_threshold * 2
        or (data.state.contact_frames >= SUSTAINED_FRAMES and data.state.overlap_frames >= 1)
    )


# -- 8. staircase_pass ------------------------------------------------------


def staircase_pass(data: GateInput) -> bool:
    """One person standing on the stairs, another walking past them.

    On a staircase there is nowhere else to go: passing close is not a choice.
    """
    gate = data.config.gates.staircase_pass
    if not gate.enabled or not data.zone.in_staircase:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    a_static = data.a_speed < gate.static_speed_threshold
    b_static = data.b_speed < gate.static_speed_threshold
    a_moving = data.a_speed > gate.moving_speed_threshold
    b_moving = data.b_speed > gate.moving_speed_threshold
    one_static_one_moving = (a_static and b_moving) or (b_static and a_moving)

    no_genuine_attack = (
        not frame.hard_contact
        and state.contact_frames < gate.max_contact_frames
        and state.overlap_frames < gate.max_overlap_frames
        and frame.max_acceleration < metrics.acceleration_threshold * 1.1
        and state.window_drop < metrics.window_drop_threshold * 1.8
        and frame.max_direction_change < metrics.direction_change_threshold_deg * 1.5
    )
    return one_static_one_moving and no_genuine_attack


# -- 9. hall_final_confirmation ---------------------------------------------


def hall_final_confirmation(data: GateInput) -> bool:
    """Hall cameras demand a sustained grapple or a violent approach before alerting.

    The main hall is wide, busy, and full of doorways. It generated more false positives
    than every other camera combined, so it gets one extra bar to clear.
    """
    gate = data.config.gates.hall_confirmation
    if not gate.enabled:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    strong_displacement = state.window_drop > metrics.window_drop_threshold * 2.0
    extreme_approach = frame.approach > metrics.speed_threshold * 2.0
    sustained_grapple = (
        state.contact_frames >= gate.sustained_contact_min
        or state.overlap_frames >= gate.sustained_overlap_min
    )
    return not (strong_displacement or extreme_approach or sustained_grapple)


# -- 10. benign_conversation (post-confirmation) ----------------------------


def benign_conversation(data: GateInput) -> bool:
    """A confirmed pair -- but the victim never moved.

    This is the last line of defence, and it encodes the deepest idea in the system:
    **an assault displaces its victim.** If nobody was pushed, pulled, knocked down or
    made to flinch, then whatever happened, it was not an assault.
    """
    gate = data.config.gates.benign_conversation
    if not gate.enabled:
        return False

    metrics = data.config.metrics
    frame, state = data.frame, data.state

    no_victim_displacement = (
        not frame.hard_contact
        and frame.max_acceleration < metrics.acceleration_threshold
        and frame.max_direction_change < metrics.direction_change_threshold_deg
        and frame.approach < metrics.speed_threshold * 0.70
        and state.window_drop < metrics.window_drop_threshold * 2.0
    )
    was_standing_still = state.still_frames >= data.config.gates.static_close.min_frames
    no_real_collision = state.overlap_frames <= SUSTAINED_FRAMES

    return no_victim_displacement and was_standing_still and no_real_collision


# -- the rule list ----------------------------------------------------------

GATES: tuple[Gate, ...] = (
    Gate("static_close", static_close),
    Gate("standing_close_long", standing_close_long),
    Gate("social_group", social_group),
    Gate("social_reapproach", social_reapproach),
    Gate("proximity_only", proximity_only),
    Gate("normal_flow_motion_required", normal_flow_motion_required),
    Gate("crossing_pass", crossing_pass),
    Gate("staircase_pass", staircase_pass),
    Gate("hall_final_confirmation", hall_final_confirmation),
    Gate("benign_conversation", benign_conversation, post_confirmation=True),
)

GATE_NAMES: tuple[str, ...] = tuple(gate.name for gate in GATES)


def first_firing(data: GateInput, *, post_confirmation: bool) -> str | None:
    """The name of the first gate that suppresses this pair, or None if it survives."""
    for gate in GATES:
        if gate.post_confirmation != post_confirmation:
            continue
        if gate.fires(data):
            return gate.name
    return None

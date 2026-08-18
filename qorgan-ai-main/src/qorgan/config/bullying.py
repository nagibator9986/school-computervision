"""Bullying detection config.

Every number the detector uses lives here. The legacy system had a good first tier
(~60 named thresholds per camera, well commented) and a bad second tier: scoring
weights, inline multipliers and validation constants scattered as magic numbers
across three files, silently interacting with the YAML. Both tiers are config now.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from qorgan.config.common import Base, ZoneRect

# Floats from YAML will not sum to exactly 1.0.
WEIGHT_SUM_TOLERANCE = 1e-6


class PairMetrics(Base):
    """Thresholds for the per-frame kinematics of a pair of tracked people.

    **All speeds are px/SECOND and accelerations px/second^2.** The legacy measured
    speed in px/FRAME and acceleration in the meaningless mixed unit (px/frame)/s, so
    every threshold below was tuned against a quantity that changed meaning whenever
    the frame rate or `det_every` changed.

    Most values here are the legacy's, converted at its ~10 fps analysis rate. **They
    are therefore provisional starting points, not tuned values**, and the eval harness
    exists to re-derive them against labelled footage. Nothing else was possible: there
    is no exact conversion from a threshold tuned against an uncontrolled frame rate.

    **`acceleration_threshold` is the exception, and it is worth dwelling on why.** Its
    faithful conversion is 80 px/s^2 -- and a faithful conversion of a wrong number is a
    wrong number in better units. A YOLO box wobbles a few pixels on a motionless person;
    the tracker reads that wobble as velocity, and velocity as acceleration. So a calm
    corridor has a *noise floor*, and the converted thresholds (60, 80, 90) all sat
    underneath it: an ordinary walk crossed them on 4-22% of its frames, depending on the
    camera. The detector was reacting to its own bounding box.

    Nothing could be tuned there, because the quantity being thresholded was, in that
    range, mostly noise. See `qorgan.evaluation.noise_floor` -- which measures the floor,
    and `tests/test_noise_floor.py`, which fails the build if any camera's threshold sinks
    back below its own.
    """

    proximity_threshold: float = Field(default=160, gt=0)  # px
    close_distance_ratio: float = Field(default=0.90, gt=0)
    contact_distance_ratio: float = Field(default=0.60, gt=0)

    speed_threshold: float = Field(default=220, gt=0)  # px/s  (legacy 22 px/frame)
    min_relative_speed: float = Field(default=120, ge=0)  # px/s  (legacy 12 px/frame)
    # Above the noise floor of a calm corridor, not a conversion of the legacy's 8.
    acceleration_threshold: float = Field(default=220, gt=0)  # px/s^2
    direction_change_threshold_deg: float = Field(default=35, gt=0, le=180)

    # window_drop is a DISPLACEMENT in px -- how far the pair closed across the recent
    # window -- not a speed. The legacy compared it against `closing_speed_threshold`,
    # which is a per-frame rate: dimensionally incoherent, but every multiplier in every
    # gate (x0.8, x1.15, x1.5, x1.8, x2.0) was tuned around that comparison. Kept as its
    # own key so the incoherence is at least visible and separately tunable.
    window_drop_threshold: float = Field(default=6, gt=0)  # px

    min_box_area: float = Field(default=2500, gt=0)
    min_track_hits: int = Field(default=2, ge=1)
    tracker_max_lost: int = Field(default=18, ge=1)
    tracker_smoothing: float = Field(default=0.55, ge=0.0, le=1.0)
    # Detections whose centre is this close to the frame edge are ignored: a half-visible
    # person has a meaningless box, and a meaningless box has a meaningless velocity.
    edge_margin_ratio: float = Field(default=0.06, ge=0.0, lt=0.4)


class ScoreWeights(Base):
    """The eleven additive contributions to a pair's aggression score.

    Every one of these was a magic number in the legacy worker, which is why the eval
    harness could not tune them: it could not reach them. They are the exact values the
    legacy used.

    The terms are independent -- no elif -- so a pair that is close AND contacting AND
    accelerating collects all three.
    """

    close: float = 0.8  # centres within the dynamic threshold
    window_drop: float = 0.9  # they came together across the window
    approach: float = 1.0  # one is moving hard at the other
    relative_speed: float = 0.8  # they are moving fast relative to each other
    acceleration: float = 1.2  # a lunge or a shove
    direction_change: float = 0.7  # people in corridors walk in straight lines
    contact: float = 0.8  # touching-ish
    hard_contact: float = 0.9  # a grab (stacks with `contact`)
    contact_hold: float = 0.7  # contact SUSTAINED over frames
    overlap_hold: float = 0.9  # boxes overlapping, sustained
    chaotic_struggle: float = 1.1  # the combo below

    # The chaotic-struggle term fires at a LOWER acceleration bar than the `acceleration`
    # term does, so a grappling pair can collect 1.1 without collecting 1.2. That slack
    # is deliberate: a struggle is messy and sustained rather than sharp.
    chaotic_acceleration_factor: float = Field(default=0.7, gt=0, le=1.0)
    # How many frames of contact/overlap count as "sustained".
    hold_min_frames: int = Field(default=2, ge=1)

    # peak_score decays by this factor on every frame the pair survives the gates.
    peak_decay: float = Field(default=0.92, gt=0.0, lt=1.0)


class Counters(Base):
    """Hysteresis on the pair state counters.

    They increment by 1 and decay by 1 per frame rather than resetting to zero.
    That is deliberate and hard-won: a single dropped frame must not erase the
    evidence of a struggle. Preserve it.
    """

    increment: int = Field(default=1, ge=1)
    decay: int = Field(default=1, ge=0)
    min_aggression_frames: int = Field(default=2, ge=1)
    min_pair_persistence_frames: int = Field(default=2, ge=1)
    temporal_window_seconds: float | None = Field(default=None, gt=0)
    red_hold_seconds: float = Field(default=2.5, gt=0)
    # cooldown_seconds DELETED -- nothing ever read it. Alert repetition is held off by
    # `EventMerge.same_pair_window_seconds`, which is live and does the job.

    @model_validator(mode="after")
    def _default_window_to_hold(self) -> Counters:
        # Legacy: temporal_window_seconds defaulted to whatever red_hold_seconds
        # resolved to. A cross-field default, not a literal one.
        if self.temporal_window_seconds is None:
            object.__setattr__(self, "temporal_window_seconds", self.red_hold_seconds)
        return self


class Zones(Base):
    """Per-camera zones. The one thing that is genuinely unique to each camera."""

    # Tracks inside are excluded from pair analysis entirely: a reflective column
    # in the main hall produced phantom people for months.
    mirror_ignore: list[ZoneRect] = Field(default_factory=list)

    # Corridor traffic lanes. People walking past each other are the single largest
    # false-positive source, so score is damped AND the alert bar is raised.
    normal_flow: list[ZoneRect] = Field(default_factory=list)
    normal_flow_score_multiplier: float = Field(default=0.30, gt=0.0, le=1.0)
    normal_flow_alert_threshold: float = Field(default=4.0, gt=0)

    # People legitimately pass close on stairs.
    staircase: list[ZoneRect] = Field(default_factory=list)
    staircase_proximity_multiplier: float = Field(default=1.60, ge=1.0)


class StaticCloseGate(Base):
    """Gates 1 and 2: two people standing close but still, and the sustained version.

    **This `acceleration_threshold` is a CEILING, and `PairMetrics.acceleration_threshold`
    is a FLOOR. Same quantity, opposite jobs — do not "make them consistent".**

    That one asks *is this violent enough to be aggression?* and so must sit ABOVE the
    noise of somebody WALKING (~105 px/s^2). This one asks *are these two standing
    still?* and so must sit above the noise of somebody STANDING (~46 px/s^2) — but well
    BELOW the noise of walking, or a person strolling past would be classified as
    motionless and the pair suppressed. It is the gate that kills the commonest false
    positive in the building (two children close together, talking), so a value that
    cannot fire is not a conservative choice; it is a broken one.

    50 clears standing-still noise on ~96% of frames (measured, +/-3 px box wobble) and
    the counters' hysteresis absorbs the rest. It stays.

    Speeds are px/s and accelerations px/s^2, converted from the legacy's px/frame at
    its ~10 fps analysis rate. Provisional until the eval harness re-derives them.
    """

    enabled: bool = True
    speed_threshold: float = Field(default=40.0, gt=0)  # px/s   (legacy 4.0)
    # A ceiling. See the class docstring before touching it.
    acceleration_threshold: float = Field(default=50.0, gt=0)  # px/s^2 (legacy 5.0)
    relative_speed_threshold: float = Field(default=60.0, gt=0)  # px/s   (legacy 6.0)
    min_frames: int = Field(default=3, ge=1)
    # A real attack after standing still must not be swallowed by the gate.
    sudden_action_bypass_enabled: bool = True
    # sudden_action_speed_ratio DELETED -- `_sudden_action` compares against
    # PairMetrics.acceleration_threshold and window_drop_threshold, never against a ratio.


class SocialGroupGate(Base):
    """Gate 3: a group in a flow zone moving the same direction."""

    enabled: bool = True
    # min_group_size / max_direction_spread_deg DELETED -- `social_group` reads
    # signals.same_direction and hardcoded multipliers of PairMetrics, never these.


class SocialReapproachGate(Base):
    """Gate 4: close, drifted apart, close again. Friends talking."""

    enabled: bool = True
    min_prior_close_frames: int = Field(default=4, ge=1)
    max_gap_frames: int = Field(default=12, ge=1)
    distance_delta_ratio: float = Field(default=0.35, gt=0)
    # requires_hard_action DELETED -- never read.


class ProximityOnlyGate(Base):
    """Gate 5: close, but no motion at all."""

    enabled: bool = True
    # max_speed DELETED. `proximity_only` reads signals.motion_present, not a speed.
    # A name-based scan could not see this one: PairData.max_speed shares the name.


class NormalFlowMotionGate(Base):
    """Gate 6: inside a flow zone, demand a strong action signal."""

    enabled: bool = True
    # min_action_score DELETED -- `normal_flow_motion_required` builds `has_action` from
    # hardcoded multipliers of PairMetrics, never from a score.


class CrossingPassGate(Base):
    """Gate 7: two people walking past each other."""

    enabled: bool = True
    max_contact_frames: int = Field(default=2, ge=0)
    max_overlap_frames: int = Field(default=2, ge=0)
    # Legacy plumbed a `requires_skeleton_aggression` flag the function body never
    # read. If we want the bypass, it has to actually do something:
    action_evidence_bypass_enabled: bool = True
    # action_evidence_min_skeleton_score DELETED -- `_action_evidence` never sees a
    # skeleton; the skeleton runs in the SLOW tier, after the gates.


class StaircasePassGate(Base):
    """Gate 8: one person standing, one walking past, on stairs."""

    enabled: bool = True
    max_contact_frames: int = Field(default=3, ge=0)
    max_overlap_frames: int = Field(default=3, ge=0)
    static_speed_threshold: float = Field(default=40.0, gt=0)  # px/s (legacy 4.0)
    moving_speed_threshold: float = Field(default=80.0, gt=0)  # px/s (legacy 8.0)


class HallConfirmationGate(Base):
    """Gate 9: hall cameras require sustained contact or overlap before alerting."""

    enabled: bool = False
    sustained_contact_min: int = Field(default=10, ge=1)
    sustained_overlap_min: int = Field(default=5, ge=1)
    # min_skeleton_score DELETED -- same reason. The skeleton is not available here.
    # Confidence.min_skeleton_score_for_confirmation is the live one, and it is elsewhere.


class BenignConversationGate(Base):
    """Gate 10: a confirmed pair, but the victim never moved."""

    enabled: bool = True
    # min_victim_displacement_ratio DELETED -- `benign_conversation` measures
    # displacement against PairMetrics thresholds, never against a ratio of its own.


class Gates(Base):
    """The ten suppression gates, as a declarative rule list.

    Each is a named, individually testable predicate rather than an `if ...: continue`
    buried in a 940-line function. This is the anti-false-positive core of the system
    and the most valuable thing the legacy project learned.
    """

    static_close: StaticCloseGate = StaticCloseGate()
    social_group: SocialGroupGate = SocialGroupGate()
    social_reapproach: SocialReapproachGate = SocialReapproachGate()
    proximity_only: ProximityOnlyGate = ProximityOnlyGate()
    normal_flow_motion: NormalFlowMotionGate = NormalFlowMotionGate()
    crossing_pass: CrossingPassGate = CrossingPassGate()
    staircase_pass: StaircasePassGate = StaircasePassGate()
    hall_confirmation: HallConfirmationGate = HallConfirmationGate()
    benign_conversation: BenignConversationGate = BenignConversationGate()
    # separation_guard DELETED, with the SeparationGuard model. No gate ever consulted it:
    # `PairState._guard_after_alert` does the job against PairMetrics thresholds instead.
    require_motion_for_alert: bool = True


class SkeletonSettings(Base):
    enabled: bool = True
    model: str = "yolov8n-pose.pt"
    conf: float = Field(default=0.25, gt=0.0, lt=1.0)
    min_frames: int = Field(default=8, ge=1)
    max_frames: int = Field(default=16, ge=1)
    # run_on_crop / required_for_high_alert DELETED -- neither was ever read. The
    # skeleton always runs on a crop, and `Confidence.cap_without_skeleton` is the
    # mechanism that makes confirmation mandatory for an alert.

    @model_validator(mode="after")
    def _frame_bounds(self) -> SkeletonSettings:
        if self.min_frames > self.max_frames:
            raise ValueError("skeleton.min_frames must be <= max_frames")
        return self


# PoseSettings and ViolenceSettings DELETED, with `BullyingConfig.pose` and `.violence`.
# Nothing read either: no second or third validator exists, the skeleton is the only one.
# ViolenceSettings called itself a plug-in point that was "kept" -- but a declared plug-in
# point with no consumer is a dead key wearing a hat, and it cost a validator, a test and
# six YAML-tunable numbers that did nothing. Spec C will design its own config when it has
# something to plug in.


class Confidence(Base):
    """How a candidate becomes an alert. All of it, in one block.

    Legacy spread these three numbers across three files with three different values.
    """

    candidate_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    validation_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # The cap is the mechanism: an event the skeleton did not confirm cannot reach
    # the notify threshold, so skeleton confirmation is *mandatory* for a Telegram
    # alert. The validator below is what keeps that true if someone retunes.
    cap_without_skeleton: float = Field(default=0.72, ge=0.0, le=1.0)
    # The single skeleton-aggression bar. A skeleton confirms only if its score reaches this
    # (and its evidence is not all weak), and the confidence agreement boost keys off the
    # SAME line. It used to be shadowed: `_skeleton_confirms` ALSO required a hardcoded
    # AGGRESSIVE_SCORE of 0.45, so the effective bar was max(0.45, this) and this knob did
    # nothing for any value <= 0.45 -- most of its range. The default is 0.45 because that
    # WAS the bar in force; the 0.35 it declared before never took effect. Now it is the
    # bar, alone, across its whole range. See detection/validation.py.
    min_skeleton_score_for_confirmation: float = Field(default=0.45, ge=0.0, le=1.0)

    alert_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    critical_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    # Only this fires Telegram. One number, one place.
    notify_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _cap_below_notify(self) -> Confidence:
        if abs(self.candidate_weight + self.validation_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError("confidence: candidate_weight + validation_weight must equal 1.0")
        if self.cap_without_skeleton >= self.notify_threshold:
            raise ValueError(
                f"confidence: cap_without_skeleton ({self.cap_without_skeleton}) must be "
                f"strictly below notify_threshold ({self.notify_threshold}); otherwise a "
                "skeleton-unconfirmed event can raise a Telegram alert"
            )
        if self.alert_threshold > self.critical_threshold:
            raise ValueError("confidence: alert_threshold must be <= critical_threshold")
        return self


class EventMerge(Base):
    """Deduplicate one physical incident into one notification."""

    enabled: bool = True
    same_pair_window_seconds: float = Field(default=15.0, ge=0)
    scene_window_seconds: float = Field(default=8.0, ge=0)
    spatial_threshold: float = Field(default=0.25, gt=0, le=1.0)
    merge_by_spatial: bool = True


class BullyingConfig(Base):
    metrics: PairMetrics = PairMetrics()
    weights: ScoreWeights = ScoreWeights()
    counters: Counters = Counters()
    zones: Zones = Zones()
    gates: Gates = Gates()
    skeleton: SkeletonSettings = SkeletonSettings()
    confidence: Confidence = Confidence()
    event_merge: EventMerge = EventMerge()

    # The score a pair must reach to become a candidate, before zone modifiers.
    alert_score_threshold: float = Field(default=1.4, gt=0)

    # How much footage an event's clip holds, ending at the moment of the alarm. Client
    # §5.5 asks for ~3 s. A number, not a constant, because "how long is the evidence" is
    # a question the school answers and the answer differs per camera: a staircase needs
    # the run-up, a doorway does not.
    #
    # It also sizes the worker's frame ring buffer -- but it does NOT bound it. Raising
    # this to the schema maximum cannot make a worker leak: `events.clip_buffer` enforces
    # a byte budget underneath it (rule R8). The ceiling here is what stops somebody
    # asking for a minute of pre-roll and getting a buffer that quietly holds ten frames.
    clip_seconds: float = Field(default=3.0, gt=0, le=15.0)

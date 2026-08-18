"""The detector: detections in, candidates out. One frame at a time.

This is the only place the pieces are wired together, and the worker and the eval harness
both drive *this* object. That is rule R2 made concrete: there is no second copy of the
pipeline for evaluation to drift away from.

The two-tier structure is preserved from the legacy, because it is correct: cheap work on
every frame (geometry, scoring, gates), expensive work only on the handful of pairs that
survive (pose, skeleton, recording). Nothing in this module loads a model or touches disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qorgan.config.bullying import BullyingConfig
from qorgan.detection.constants import FIRM_FRAMES, SUSTAINED_FRAMES
from qorgan.detection.gates import GateInput, first_firing
from qorgan.detection.geometry import Box
from qorgan.detection.pairs import PairFrame, PairState, PairStore, measure, pair_key
from qorgan.detection.scoring import (
    Signals,
    ZoneContext,
    candidate_probability,
    in_mirror_zone,
    score_pair,
    signals,
    zone_context,
)
from qorgan.detection.tracking import Track, TrackStore

# A pair whose centres are further apart than this fraction of the frame diagonal is
# not a candidate, whatever the pixel numbers say: they are simply not near each other.
NEAR_IN_FRAME_RATIO = 0.22


@dataclass(frozen=True, slots=True)
class Candidate:
    """A pair the detector believes is fighting. Handed to the slow validation tier."""

    key: tuple[int, int]
    timestamp: float
    score: float
    threshold: float  # the EFFECTIVE, zone-adjusted bar it had to clear
    probability: float  # normalised against `threshold`, not the base one
    boxes: tuple[Box, Box]
    center: tuple[float, float]  # normalised 0..1, for spatial event merging
    in_normal_flow: bool
    in_staircase: bool


@dataclass(frozen=True, slots=True)
class Suppression:
    """A pair the detector considered and rejected, and the gate that rejected it.

    Kept because it is the single most useful thing for tuning: it turns "why did it not
    alert?" from an afternoon of print statements into a lookup.
    """

    key: tuple[int, int]
    timestamp: float
    gate: str
    score: float


@dataclass
class FrameResult:
    candidates: list[Candidate] = field(default_factory=list)
    suppressions: list[Suppression] = field(default_factory=list)
    tracks: int = 0
    pairs: int = 0

    @property
    def suppressed_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.suppressions:
            counts[item.gate] = counts.get(item.gate, 0) + 1
        return counts


class BullyingDetector:
    """One per camera. Stateful across frames; pure with respect to the outside world."""

    def __init__(self, config: BullyingConfig, frame_width: int, frame_height: int) -> None:
        self.config = config
        self.width = frame_width
        self.height = frame_height
        self._frame_diagonal = (frame_width**2 + frame_height**2) ** 0.5

        self.tracks = TrackStore(
            smoothing=config.metrics.tracker_smoothing,
            max_lost=config.metrics.tracker_max_lost,
        )
        self.pairs = PairStore()

    def process(self, detections: dict[int, Box], timestamp: float) -> FrameResult:
        """One frame. Returns what survived and what was suppressed, and why."""
        self.tracks.update(detections, timestamp)
        self.pairs.evict(timestamp)

        people = self._eligible(timestamp)
        result = FrameResult(tracks=len(people))

        for index, a in enumerate(people):
            for b in people[index + 1 :]:
                self._judge(a, b, timestamp, result)

        result.pairs = len(self.pairs)
        return result

    # -- who is even eligible ------------------------------------------------

    def _eligible(self, timestamp: float) -> list[Track]:
        """Confirmed, big enough, away from the frame edge, and not a reflection."""
        metrics = self.config.metrics
        margin_x = self.width * metrics.edge_margin_ratio
        margin_y = self.height * metrics.edge_margin_ratio

        return [
            track
            for track in self.tracks.confirmed(metrics.min_track_hits)
            if track.box.area >= metrics.min_box_area
            and not self._near_edge(track.box, margin_x, margin_y)
            and not in_mirror_zone(track.box, self.config.zones, self.width, self.height)
        ]

    def _near_edge(self, box: Box, margin_x: float, margin_y: float) -> bool:
        """A half-visible person has a meaningless box, and a meaningless box has a
        meaningless velocity."""
        cx, cy = box.center
        return (
            cx < margin_x
            or cx > self.width - margin_x
            or cy < margin_y
            or cy > self.height - margin_y
        )

    # -- the per-pair decision ----------------------------------------------

    def _judge(self, a: Track, b: Track, timestamp: float, result: FrameResult) -> None:
        zone = zone_context(a.box, b.box, self.config.zones, self.width, self.height)
        frame = measure(a, b, self.config.metrics, zone.proximity_scale(self.config.zones))

        state = self.pairs.get(pair_key(a.track_id, b.track_id), timestamp)
        sig = signals(frame, state, a, b, self.config)
        state.observe(frame, sig.low_motion, self.config.counters, timestamp)

        score = score_pair(frame, state, self.config, zone)
        threshold = zone.effective_threshold(self.config.zones, self.config.alert_score_threshold)
        data = GateInput(frame, state, sig, zone, self.config, a.speed, b.speed)

        if self._suppressed(data, timestamp, score, result):
            return

        state.record_score(score, self.config.weights.peak_decay)
        escalating = self._high_risk(frame, state, sig) and score >= threshold
        state.escalate(escalating, self.config.counters, timestamp)

        if not self._confirmed(frame, state, sig, zone, threshold):
            return

        # The last line of defence: an assault displaces its victim.
        vetoed = first_firing(data, post_confirmation=True)
        if vetoed:
            result.suppressions.append(Suppression(state.key, timestamp, vetoed, score))
            return

        result.candidates.append(self._candidate(a, b, state, timestamp, score, threshold, zone))

    def _suppressed(
        self, data: GateInput, timestamp: float, score: float, result: FrameResult
    ) -> bool:
        """Did anything stop this pair before it could escalate? Drain it if so."""
        state = data.state

        # A pair that already alerted and then separated must earn fresh physical contact
        # before it can alert again -- otherwise one shove fires on every subsequent frame.
        if state.awaiting_new_contact:
            state.drain(self.config.counters)
            result.suppressions.append(
                Suppression(state.key, timestamp, "post_alert_separation", score)
            )
            return True

        blocked = first_firing(data, post_confirmation=False)
        if blocked:
            state.drain(self.config.counters)
            result.suppressions.append(Suppression(state.key, timestamp, blocked, score))
            return True

        return False

    # -- the two decisions that matter --------------------------------------

    def _high_risk(self, frame: PairFrame, state: PairState, sig: Signals) -> bool:
        """Does this frame look like an attack, as opposed to merely a close pair?

        Any ONE of four independent signatures is enough. They are different shapes of
        the same event: a rush, a grab, a sustained grapple, a violent turn.
        """
        if not self._near_in_frame(frame):
            return False
        if sig.same_direction or sig.low_motion:
            return False

        metrics = self.config.metrics
        return (
            (state.window_drop > metrics.window_drop_threshold and frame.contact_like)
            or (
                frame.hard_contact and frame.max_acceleration > metrics.acceleration_threshold * 0.7
            )
            or (
                state.contact_frames >= SUSTAINED_FRAMES
                and frame.relative_speed > metrics.min_relative_speed * 0.75
            )
            or (
                state.overlap_frames >= SUSTAINED_FRAMES
                and frame.max_direction_change > metrics.direction_change_threshold_deg * 0.75
            )
        )

    def _confirmed(
        self,
        frame: PairFrame,
        state: PairState,
        sig: Signals,
        zone: ZoneContext,
        threshold: float,
    ) -> bool:
        """Enough escalating frames, a high enough peak, real contact, and a coherent
        story from more than one signal.

        Note it tests `peak_score`, not this frame's score: a fight is judged on its
        worst moment, not on whichever frame happened to arrive last.
        """
        counters = self.config.counters
        return (
            state.aggression_frames >= counters.min_aggression_frames
            and state.persistence_frames >= max(counters.min_pair_persistence_frames, FIRM_FRAMES)
            and state.peak_score >= threshold
            and state.contact_frames >= SUSTAINED_FRAMES
            and not sig.same_direction
            and not sig.low_motion
            and self._multi_signal(state, sig, zone)
        )

    def _multi_signal(self, state: PairState, sig: Signals, zone: ZoneContext) -> bool:
        """Either four independent signals agree, or there is a sustained grapple.

        One signal is never enough. That is the whole lesson of the legacy's false
        positives: any single metric can be produced by ordinary corridor life.
        """
        contact_min = FIRM_FRAMES + 1 if zone.in_staircase else FIRM_FRAMES
        overlap_min = FIRM_FRAMES if zone.in_staircase else SUSTAINED_FRAMES

        sustained_grapple = (
            state.contact_frames >= contact_min
            and state.overlap_frames >= overlap_min
            and sig.strong_motion_spike
            and not sig.same_direction
        )
        return sig.multi_signal or sustained_grapple

    # -- emitting ------------------------------------------------------------

    def _near_in_frame(self, frame: PairFrame) -> bool:
        return frame.gap / max(self._frame_diagonal, 1.0) < NEAR_IN_FRAME_RATIO

    def _candidate(
        self,
        a: Track,
        b: Track,
        state: PairState,
        timestamp: float,
        score: float,
        threshold: float,
        zone: ZoneContext,
    ) -> Candidate:
        ax, ay = a.box.normalized_center(self.width, self.height)
        bx, by = b.box.normalized_center(self.width, self.height)

        return Candidate(
            key=state.key,
            timestamp=timestamp,
            score=score,
            threshold=threshold,
            # Normalised against the EFFECTIVE threshold. The legacy used the base one,
            # which saturated every flow-zone event at ~0.997. See detection/scoring.py.
            probability=candidate_probability(state.peak_score, threshold),
            boxes=(a.box, b.box),
            center=((ax + bx) / 2, (ay + by) / 2),
            in_normal_flow=zone.in_normal_flow,
            in_staircase=zone.in_staircase,
        )

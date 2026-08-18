"""What a pupil is doing, per observation and then per stretch of time.

--------------------------------------------------------------------------------
**THE LINE THIS MODULE REFUSES TO CROSS.**

Everything named in `PupilState` is something a camera witnesses: where a body is, which
way a head is turned, whether an arm is up. None of it is a claim about a mind. In
particular there is no `ENGAGED` state and there will not be one, because "вовлечён в
урок" is not visible — a pupil looking straight at the board may be somewhere else
entirely, and one staring at the desk may be concentrating hard. What `metrics/` builds
instead is an index over these observable facts, decomposed on screen so the psychologist
sees the parts and not a verdict. The EU AI Act's treatment of emotion inference in
education is the legal edge of the same point; the honest engineering reason is simply
that the measurement does not exist.

--------------------------------------------------------------------------------
**WHAT THE FOOTAGE TAUGHT, WHICH IS WHY THE THRESHOLDS LOOK THE WAY THEY DO.**

Three findings from labelling real observations by eye (`MEASUREMENTS.md` §5) drive
almost every decision below.

1. **This camera looks DOWN.** Head keypoints sit only ~0.29 shoulder widths above the
   shoulder line (median), not the ~0.6 a frontal view would give. Any threshold copied
   from a frontal-camera intuition is roughly twice too generous here.

2. **A head-relative datum inverts when the head goes down.** Measuring "hand above head"
   fails exactly when a pupil lowers their head, because then everything is above their
   head. `HAND_RAISED` is therefore measured against the SHOULDER line with a threshold
   set from the observed distribution (p95 of wrist-above-shoulder is 0.11, p99 is 0.45),
   not against the head.

3. **The adult dominates every naive statistic.** Of the 24 highest-scoring "arm raised"
   observations in the whole lesson, the large majority were the teacher — he sits
   nearest the lens, so his shoulder width is ~220 px against a pupil's ~70 px, and he
   habitually rests a hand against his head. Hand-labelling the 24 left **two or three**
   genuine pupil hand-raises in 52 minutes. Two consequences, both structural rather than
   cosmetic: the teacher is excluded *before* pupil metrics are computed, not filtered
   afterwards (`room/roles.py`), and the honest expected count of hand-raises in a lesson
   of independent seatwork is small. A detector that returns a satisfying number here
   would be wrong.

--------------------------------------------------------------------------------
**EVERY STATE IS EITHER OBSERVED, OR EXPLICITLY UNKNOWN.**

`UNKNOWN` is a state, not a gap, and it is counted alongside the others. A pupil hidden
behind the pupil in front is not a pupil sitting quietly, and the two must never
aggregate the same way. Every threshold below that is a judgement call rather than a
measurement carries the word CHOSEN, in those letters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from classvision.geometry import (
    HeadDirection,
    Keypoints,
    anchor,
    head_direction,
    head_height,
    shoulder_width,
)


class PupilState(StrEnum):
    """What one pupil was doing in one observation.

    Ordered by the priority in which they are tested, because they are not disjoint: a
    pupil can be out of their place AND have a hand up. `classify` resolves that with a
    fixed precedence and records only the winner, while the underlying booleans stay
    available on `Reading` for metrics that want them independently.
    """

    AWAY_FROM_PLACE = "away_from_place"   # вне своего места
    AT_BOARD = "at_board"                 # у доски
    STOOD_UP = "stood_up"                 # встал на своём месте
    HAND_RAISED = "hand_raised"           # поднял руку
    HEAD_DOWN = "head_down"               # лежит на парте / голова опущена
    TURNED_AWAY = "turned_away"           # смотрит назад / отвернулся
    SEATED = "seated"                     # сидит на месте, лицом вперёд
    UNKNOWN = "unknown"                   # виден, но поза не читается

RU_LABELS: dict[PupilState, str] = {
    PupilState.AWAY_FROM_PLACE: "вне места",
    PupilState.AT_BOARD: "у доски",
    PupilState.STOOD_UP: "встал",
    PupilState.HAND_RAISED: "поднял руку",
    PupilState.HEAD_DOWN: "голова опущена / лежит на парте",
    PupilState.TURNED_AWAY: "отвернулся назад",
    PupilState.SEATED: "сидит на месте",
    PupilState.UNKNOWN: "поза не читается",
}


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Every number the classifier uses, in one object so a run can record what it used.

    A metric compared across two weeks under two different threshold sets is a comparison
    of settings wearing the costume of a comparison of children, so this whole object is
    written into the artefact and a change to it makes a NEW artefact rather than an
    overwrite.
    """

    # Wrist this far above the pupil's own shoulder line counts as a hand up. MEASURED
    # BOUND: across 5 067 arm observations, p95 of wrist-above-shoulder was 0.11 and p99
    # was 0.45 shoulder widths, so 0.45 selects roughly the top 1 % of arms. CHOSEN at
    # that p99 rather than lower because hand-labelling showed the band below it is
    # dominated by hand-to-head and reaching for objects.
    hand_above_shoulder: float = 0.45
    # ...and the forearm must point upward, which removes an arm extended sideways to
    # pass a book. CHOSEN.
    hand_forearm_up: float = 0.15

    # A hand must be up for this many consecutive analysed observations to count as one
    # raise, and be down for as many before another can be counted. Without the second
    # rule a wavering arm is counted repeatedly and the pupil with the least steady arm
    # tops the report. CHOSEN; at 2 fps this is 1.5 s.
    hand_min_hold: int = 3
    hand_min_gap: int = 3

    # Head this far BELOW the pupil's own upright baseline is a lowered head. The raw
    # single-frame test ("nose below the shoulder line") fired on 94 % of observations,
    # so the baseline is per-track and this is a drop from it. CHOSEN.
    head_drop: float = 0.55
    # ...held this long. A pupil reading a book has their head down; a pupil lying on the
    # desk keeps it there. 20 s at 2 fps. CHOSEN, and the single number most likely to
    # need retuning once a teacher says which of these episodes they would call lying.
    head_down_min_seconds: float = 20.0

    # Shoulder line risen this far above the pupil's own seated baseline is standing.
    # Standing raises the shoulders by roughly the seated thigh height, comparable to a
    # child's shoulder width. CHOSEN just under that.
    rise: float = 0.8
    # Anchor this far from the pupil's own settled position is out of place. ~1 seat over.
    away: float = 2.0
    # Excursions shorter than this are tracker jitter, not a pupil leaving. CHOSEN.
    away_min_seconds: float = 5.0

    # How many observations establish a track's baselines before anything is counted
    # against them. Everything before this is measured against nothing and is recorded as
    # UNKNOWN rather than as zero. CHOSEN; 20 observations is 10 s at 2 fps.
    settle_observations: int = 20

    # -- the quality gate on those observations, and why there has to be one -----------
    #
    # A settling observation whose shoulder width departs from the settling window's own
    # median by more than this factor is thrown out of the baseline. It is not a stylistic
    # robustness measure; it repairs a defect that reached a psychologist's report.
    #
    # MEASURED, camera D14, seat at (838, 481). The first twenty observations of that place
    # carried these shoulder widths:
    #
    #     66 55 73  16 14 17 14 11 26  45 65 64 58 64 63 64 60 60 59 59
    #
    # — six of them between 11 and 17 px against the place's settled 57 px, a
    # badly-resolved detection of somebody arriving. `head_up` is head height DIVIDED BY
    # shoulder width, so those six read 2.41 – 3.56 where a real upright head on this
    # camera reads 0.56 – 0.83. They are not tall children; they are a division by a wrong
    # number. p75 of the twenty came out at **1.635** instead of 0.67, and since `classify`
    # calls a head lowered when it falls `head_drop` below the baseline, EVERY subsequent
    # observation of that child qualified: 5 343 of 5 549, 96 % of the lesson, rendered as
    # «22 эпизода с опущенной головой (суммарно 44,5 минуты)» and an activity index of 51
    # against 95-98 for every other place in the room. Nothing anywhere said the baseline
    # was built on rubbish; the number simply arrived, and it is precisely the number a
    # psychologist acts on.
    #
    # CHOSEN at 0.6, derived rather than tasted. An admitted observation whose scale is a
    # factor `r` too small inflates `head_up` by `head_up × (1/r − 1)`, and the baseline is
    # only safe while that inflation stays under `head_drop` — otherwise one survivor can
    # by itself make an upright head read as a lowered one. With the measured upright
    # `head_up` on this camera (median 0.63 across the six D14 places) and `head_drop`
    # 0.55, the requirement is `1/r < 1 + 0.55/0.63`, i.e. `r > 0.535`. 0.6 is that bound
    # with a little margin, and the band is applied symmetrically — a doubled or merged box
    # gives a scale that is too LARGE, which deflates `head_up` and corrupts
    # `seat_shoulder_y` just as thoroughly, only in the direction that under-reports.
    #
    # What it costs on real footage: of the six D14 places, the widest genuine departure
    # inside a settling window was 102 px against a median of 76 (ratio 1.33), comfortably
    # inside the band, and the observations it removes are the ones listed above.
    settle_scale_band: float = 0.6
    # How many observations a place may be watched for while trying to find
    # `settle_observations` mutually consistent ones. CHOSEN at 5x, i.e. 100 observations /
    # 50 s at 2 fps. Reaching it is a REFUSAL, not a fallback: the place keeps no baseline,
    # every observation of it stays UNKNOWN, and `uncertainty.seats_never_settled` counts
    # it. A place the detector cannot resolve for fifty seconds is a place whose "normal
    # posture" would be a description of the detector, and the whole point of the gate is
    # that we would rather say nothing than say that.
    settle_window_limit: int = 5

    # A head direction must differ from this pupil's OWN habitual direction for this long
    # to count as turning away. Per-pupil because the habitual direction is a fact about
    # where their desk is, not about them: on this footage 61 % of all observations read
    # as one particular profile simply because of where the camera hangs. CHOSEN.
    turned_min_seconds: float = 4.0


@dataclass(frozen=True, slots=True)
class Reading:
    """One observation, reduced to the facts the state machine needs.

    Every field is `| None`-able where the camera can genuinely fail to see it, and no
    field is defaulted to a neutral value when it could not be measured. That is the
    whole discipline: `hand_up=None` (could not tell) and `hand_up=False` (told, and no)
    must not aggregate the same way.
    """

    video_seconds: float
    scale: float | None                    # shoulder width, px
    position: tuple[float, float] | None   # shoulder midpoint
    head_up: float | None                  # head height above shoulder line, in scales
    direction: HeadDirection
    hand_up: bool | None


def read(person: Keypoints, video_seconds: float, thresholds: Thresholds) -> Reading:
    """One person in one frame -> the measurements. No history, no verdict."""
    scale = shoulder_width(person)
    position = anchor(person)
    if scale is None or position is None:
        return Reading(video_seconds, None, None, None, HeadDirection.UNKNOWN, None)

    raised = hand_raised_shoulder(person, scale, thresholds)
    return Reading(
        video_seconds=video_seconds,
        scale=scale,
        position=position,
        head_up=head_height(person, scale),
        direction=head_direction(person),
        hand_up=raised,
    )


def hand_raised_shoulder(person: Keypoints, scale: float,
                         thresholds: Thresholds) -> bool | None:
    """A hand up, measured against the SHOULDER line rather than the head.

    `geometry.hand_raised` measures against the head, which is the more intuitive datum
    and the wrong one on this camera: a pupil who lowers their head puts everything above
    it. This uses the shoulder line, whose position does not move when a head drops, with
    a threshold taken from the observed p99 of the wrist-above-shoulder distribution
    rather than from an intuition about where a raised hand ought to be.

    Returns None when neither arm can be read at all — the pupil may well have their hand
    up behind the pupil in front.
    """
    from classvision.geometry import (
        L_ELBOW,
        L_WRIST,
        LOOSE_CONF,
        R_ELBOW,
        R_WRIST,
        STRICT_CONF,
    )

    shoulders = anchor(person)
    if shoulders is None or scale <= 0:
        return None

    readable = False
    for wrist, elbow in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
        if not person.has(wrist, threshold=STRICT_CONF):
            continue
        readable = True
        wrist_y = person.point(wrist)[1]
        if (shoulders[1] - wrist_y) < thresholds.hand_above_shoulder * scale:
            continue
        # The forearm must point upward. An arm extended sideways to hand something over
        # clears the height test and is not a raised hand -- hand-labelling found several.
        if person.has(elbow, threshold=LOOSE_CONF):
            if (person.point(elbow)[1] - wrist_y) < thresholds.hand_forearm_up * scale:
                continue
        return True

    # Read at least one arm and neither was up -> a real "no". Read neither -> "cannot
    # tell", which the ledger counts apart from the noes.
    return False if readable else None


@dataclass(slots=True)
class Baselines:
    """What THIS pupil's normal looks like, learned from their own first observations.

    Everything comparative in this module compares a pupil to themselves — the principle
    the school was promised («сравниваем ребёнка только с ним самим»), arrived at here by
    geometry rather than by policy, because there is no other datum. A pupil at the back
    of the room and one at the front share no pixel scale, no head-height and no habitual
    head direction, so a shared threshold would measure the seating plan.

    **THE BASELINE IS THE MOST DANGEROUS NUMBER IN THIS PACKAGE, BECAUSE NOTHING DOWNSTREAM
    CAN SEE IT.** It is computed once, from a handful of observations, and then every state
    of that place for the rest of the lesson is measured against it. A total computed
    against a wrong baseline is not noisy, it is confidently and consistently wrong for
    forty minutes, and it arrives in the report with a coverage figure beside it saying the
    place was seen 79 % of the time — which is true, and which makes it worse. So the
    settling window has a quality gate (`Thresholds.settle_scale_band`, where the measured
    case that forced it is written down) and settling can be REFUSED, in which case this
    place has no normal posture and every observation of it stays `UNKNOWN`. An unsettled
    place is counted in `uncertainty.seats_never_settled`; a place settled on rubbish is
    counted nowhere, which is the whole argument for preferring the first.
    """

    settled: bool = False
    seat_position: tuple[float, float] | None = None
    seat_shoulder_y: float | None = None
    upright_head: float | None = None
    habitual_direction: HeadDirection | None = None
    scale: float | None = None
    # Why the place never settled, when it never did. `None` while still collecting.
    # A sentence rather than a flag, because "not settled yet" and "watched for fifty
    # seconds and never resolved" are different facts about the recording.
    refusal: str | None = None
    # The candidate observations, kept WHOLE rather than as four parallel lists. They were
    # four lists, and that was the bug's hiding place: `head_up` was appended without the
    # `scale` it had been divided by, so by the time the p75 was taken there was no way
    # left to ask which readings it came from. Keeping the reading intact is what makes the
    # gate below expressible at all.
    _window: list[Reading] = field(default_factory=list)

    def observe(self, reading: Reading, thresholds: Thresholds) -> None:
        # Settled or refused: nothing more to learn, and the window is dropped rather than
        # grown for the remaining thousands of observations of this place.
        if self.settled or self.refusal is not None:
            return
        if reading.position is None or reading.scale is None:
            return
        self._window.append(reading)
        if len(self._window) < thresholds.settle_observations:
            return

        # -- the gate ---------------------------------------------------------------
        # A shoulder width wildly unlike the rest of this window's is a broken detection,
        # not a fact about this child, and `head_up` is measured IN SHOULDER WIDTHS, so a
        # broken one does not add noise to the baseline — it multiplies it. The band is
        # relative to the window's own median so it needs no per-camera calibration: a
        # place genuinely detected small all lesson has a small median and keeps its
        # observations.
        scales = sorted(r.scale for r in self._window)
        median_scale = scales[len(scales) // 2]
        low = median_scale * thresholds.settle_scale_band
        high = median_scale / thresholds.settle_scale_band
        kept = [r for r in self._window if low <= r.scale <= high]

        if len(kept) < thresholds.settle_observations:
            # Not enough consistent evidence YET. Keep watching — but not forever: a place
            # that cannot produce `settle_observations` mutually consistent readings within
            # `settle_window_limit` windows is refused, and refused loudly.
            if len(self._window) >= (thresholds.settle_observations
                                     * thresholds.settle_window_limit):
                self.refusal = (
                    f"за {len(self._window)} наблюдений на этом месте не набралось "
                    f"{thresholds.settle_observations} с согласованной шириной плеч "
                    f"(медиана {median_scale:.0f} px, допуск ×{thresholds.settle_scale_band}"
                    f"..×{1 / thresholds.settle_scale_band:.2f}): разметка человека здесь "
                    "неустойчива, и базовая поза была бы описанием детектора, а не ребёнка")
                self._window = []
            return

        # Medians, not means: a pupil who arrives late and is caught mid-walk contributes
        # a few wild positions, and a mean would move the seat to meet them.
        xs = sorted(r.position[0] for r in kept)
        ys = sorted(r.position[1] for r in kept)
        middle = len(xs) // 2
        self.seat_position = (xs[middle], ys[middle])
        self.seat_shoulder_y = ys[middle]
        self.scale = sorted(r.scale for r in kept)[len(kept) // 2]
        heads = sorted(r.head_up for r in kept if r.head_up is not None)
        if heads:
            # The UPRIGHT head is the high end of the observed distribution, not its
            # middle: the baseline must be "how this pupil sits when sitting up", and a
            # median would bake in however much of the settling window they spent looking
            # down at a book. p75. CHOSEN.
            self.upright_head = heads[int(0.75 * (len(heads) - 1))]
        directions = [r.direction for r in kept if r.direction is not HeadDirection.UNKNOWN]
        if directions:
            self.habitual_direction = max(set(directions), key=directions.count)
        self.settled = True
        self._window = []


def classify(reading: Reading, baselines: Baselines, thresholds: Thresholds, *,
             in_board_zone: bool = False) -> PupilState:
    """One observation + this pupil's own baselines -> one state.

    Precedence, highest first, and the order is a claim about what matters: where a body
    IS beats what it is doing, because "not in their seat" is the fact a teacher would
    notice first and the one least likely to be a measurement artefact.

    Until a pupil has settled, everything is `UNKNOWN` — not `SEATED`. A default of
    SEATED would fill the opening minutes of every lesson with a state we did not
    measure, and those minutes are exactly when pupils are arriving and moving.
    """
    if reading.scale is None or reading.position is None:
        return PupilState.UNKNOWN
    if not baselines.settled or baselines.seat_position is None:
        return PupilState.UNKNOWN

    import math

    displacement = math.dist(baselines.seat_position, reading.position) / reading.scale
    if displacement >= thresholds.away:
        return PupilState.AT_BOARD if in_board_zone else PupilState.AWAY_FROM_PLACE

    if (baselines.seat_shoulder_y is not None
            and (baselines.seat_shoulder_y - reading.position[1]) >= thresholds.rise * reading.scale):
        return PupilState.STOOD_UP

    if reading.hand_up:
        return PupilState.HAND_RAISED

    if (baselines.upright_head is not None and reading.head_up is not None
            and (baselines.upright_head - reading.head_up) >= thresholds.head_drop):
        return PupilState.HEAD_DOWN

    if (baselines.habitual_direction is not None
            and reading.direction is not HeadDirection.UNKNOWN
            and _turned_away(reading.direction, baselines.habitual_direction)):
        return PupilState.TURNED_AWAY

    if reading.direction is HeadDirection.UNKNOWN and reading.head_up is None:
        return PupilState.UNKNOWN

    return PupilState.SEATED


def _turned_away(current: HeadDirection, habitual: HeadDirection) -> bool:
    """Has the head turned away from where THIS pupil's head usually points?

    Absolute directions are useless here: 61 % of all observations on this footage read as
    `profile_left`, which is a fact about where the camera hangs relative to the desks and
    not about any child. What carries information is the DEPARTURE from a pupil's own
    habitual direction — and only in the direction that means turning away from the front
    of the room, which on this camera (mounted at the board, pupils facing it) is `AWAY`
    or the profile opposite to their habitual one.
    """
    if current == habitual:
        return False
    if current is HeadDirection.AWAY:
        return True
    opposite = {
        HeadDirection.PROFILE_LEFT: HeadDirection.PROFILE_RIGHT,
        HeadDirection.PROFILE_RIGHT: HeadDirection.PROFILE_LEFT,
    }
    return opposite.get(habitual) == current

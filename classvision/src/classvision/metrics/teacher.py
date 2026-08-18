"""The teacher half: where the adult was, for how long, and how often that changed.

--------------------------------------------------------------------------------
**WHAT THIS MEASURES AND WHAT IT REFUSES TO.**

The client asked for analysis of whether the teacher «сидит много времени или на доске
объясняет урок, то есть анализ того что он так же активен или нет». Half of that is
measurable and half is not, and the split runs straight through the sentence.

Measurable: the share of the lesson spent at the board, at his own desk, among the pupils,
in transit and out of frame; how many times that changed; how long the longest stretch of
each was; and how much of the room's floor he covered. All of it is position over time,
which is exactly what a camera records.

Not measurable, and deliberately absent: whether the lesson was any good. «Сидит» is not
«не ведёт урок». A teacher seated at a desk may be running a discussion, marking work with
individual pupils, or supervising an exam — and on THIS recording he does something a
position metric would have to be told not to punish: at 11:01 he leaves the front of the
room and sits down *among* the pupils at a middle desk, which is neither «у доски» nor «за
столом» and is obviously not disengagement. There is no quality score here, no "activity
rating" for the adult, no threshold above which a teacher is flagged, and no comparison
between teachers or between lessons of different teachers, because the first person to be
shown such a number would reasonably treat it as an assessment of their work, and it would
not be one.

The neighbouring `qorgan` codebase declines to build this at all (its client §12.5),
reasoning that it is not equipped to tell «сидит» from «не ведёт урок». That reasoning is
right and this module does not overturn it — it satisfies the request by reporting the
position facts and stating, on the same surface, that they are not an evaluation.

--------------------------------------------------------------------------------
**WHY THIS FILE IS SUDDENLY BIGGER THAN IT WAS: THE BOARD IS FINALLY IN FRAME.**

Every earlier camera in this project hung above the board looking back at the class, so
the board was behind the lens. «Стоит у доски» was therefore not a hard measurement, it
was an *absent* one — a teacher at the board was a teacher out of frame — and the honest
report was "at desk vs not at desk", which is what this module used to contain.

Camera D14 is mounted at the back of the room looking forward. The three-panel chalkboard
occupies roughly x 1030..1610, y 135..305 of 2560x1440 (measured off frames, see
`room/zones.py`). So for the first time the sentence the client actually wrote can be
answered, and this module grows a state taxonomy to answer it.

Three facts about D14 shaped every decision below, all read off the footage rather than
assumed:

  1. **The adult moves, and he is not the biggest thing in the picture.** At 10:31 he is
     standing at the board writing; at 10:35 and 10:41 he is seated at the front-left desk
     with a laptop; at 11:01 he is seated among the pupils; at 11:16 he is back at the
     board. When he is at the board he is at the FAR end of the room, so his shoulder
     width there is ~50 px against a near pupil's ~110 px. `room/zones.identify_adult`
     compares shoulder widths and nominates the largest persistent place; on this camera
     that reasoning nominates a front-row child. Scale cannot find this adult, so it is
     the fallback here and a configured `teacher_zone` is the primary route.

  2. **A moving adult has no seat, and the whole identity mechanism of this codebase is
     the seat.** `room/seats.py` exists because children sit still for 45 minutes; that
     argument does not extend to a man who works at three different places in one lesson.
     Attributing him to one discovered seat loses two thirds of him, and attributing him
     to several merges him with whichever pupils share those places. So this module
     carries its own single-target follower (`follow_adult`) over the same anchor stream
     the seats were built from — a tracker for exactly one person, seeded from a human's
     zone, gated, and honest about every frame it could not attribute.

  3. **Pupils sit with their backs to this camera, and the adult does not.** That makes
     face identity even weaker here than the measured 0.30 best-cosine / 0.10 margin, and
     nothing in this module uses it. It also means head direction at the board is worth a
     little — a teacher writing faces away, a teacher explaining faces the class — but
     only a little, because at the board his shoulder width is ~50 px and his ears are a
     few pixels each. That split is reported, flagged INFERRED, with its own coverage.

--------------------------------------------------------------------------------
**THE ADULT MUST BE IDENTIFIED BEFORE ANY OF THIS RUNS**, and it was never optional: of
the 45 hand-raise detections in the whole of the earlier lesson, 35 were the adult, because
he sat nearest the lens and rested a hand against his head. The artefact records which
route identified him (`designated_zone` / `inferred_by_scale` / `none`), whether a human
signed the zones (`zones_confirmed_by`), and what fraction of analysed frames the follower
was actually able to attribute to him.

**A stationary camera cannot see a teacher who leaves the frame**, and on D14 he does —
the near half of the room and the doorway are outside or behind the lens.
`out_of_frame_share_of_lesson` is therefore reported as its own quantity, never folded into
the denominator of anything else, and it must never be read as absence: it means "we did
not see him", not "he was not there", and the two are different facts about a person's
working day.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from classvision.geometry import HeadDirection
from classvision.room.zones import Polygon, RoomLayout, point_in_polygon
from classvision.states import PupilState

Point = tuple[float, float]


# ---------------------------------------------------------------------------------
# THE STATE TAXONOMY
# ---------------------------------------------------------------------------------


class TeacherState(StrEnum):
    """Where the adult was in one observation.

    Five states, tested in the order listed, and the order is a claim: **where a body IS
    beats what it is doing**, the same precedence `states.classify` argues for on the pupil
    side. A teacher pacing back and forth in front of the board while explaining is AT the
    board, not IN TRANSIT, because the first description is the one a person in the room
    would give.

    Each is a predicate over position + zones + speed, each carries a minimum hold before
    it becomes an episode, and each carries its confusion below.
    """

    OUT_OF_FRAME = "out_of_frame"
    AT_BOARD = "at_board"
    AT_DESK = "at_desk"
    MOVING = "moving"
    AMONG_PUPILS = "among_pupils"


RU_LABELS: dict[TeacherState, str] = {
    TeacherState.OUT_OF_FRAME: "вне кадра / не опознан",
    TeacherState.AT_BOARD: "у доски",
    TeacherState.AT_DESK: "за своим столом",
    TeacherState.MOVING: "перемещается по классу",
    TeacherState.AMONG_PUPILS: "среди учеников",
}

# What each state is DEFINED as, and — the part that matters — what it cannot tell apart.
# Rendered next to the numbers, not buried here: a share is only readable by someone who
# knows what got counted into it.
DEFINITIONS_RU: dict[TeacherState, dict[str, str]] = {
    TeacherState.AT_BOARD: {
        "basis": "наблюдение",
        "rule": "плечевая точка взрослого внутри размеченной зоны доски",
        "confusion": "не отличает «пишет на доске» от «стоит рядом и слушает ученика "
                     "у доски»; не отличает объяснение от паузы. Учитель, который ходит "
                     "вдоль доски, объясняя, отнесён сюда, а не к «перемещается».",
    },
    TeacherState.AT_DESK: {
        "basis": "наблюдение",
        "rule": "плечевая точка внутри размеченной зоны учительского стола",
        "confusion": "не отличает работу с журналом или ноутбуком от бездействия. "
                     "«Сидит за столом» не означает «не ведёт урок».",
    },
    TeacherState.AMONG_PUPILS: {
        "basis": "вывод от противного",
        "rule": "в кадре, вне зон доски и стола, скорость ниже порога перемещения — "
                "либо скорость измерить не удалось (первый кадр после перерыва в "
                "слежении); число таких кадров указано отдельно",
        "confusion": "определён исключением: сюда попадёт и взрослый, сидящий за партой "
                     "с учениками, и взрослый, неподвижно стоящий у окна или в проходе. "
                     "Это НЕ показатель работы с учениками.",
    },
    TeacherState.MOVING: {
        "basis": "вывод",
        "rule": "смещение плечевой точки быстрее порога между двумя ПОДРЯД идущими "
                "опознанными кадрами, вне зон доски и стола",
        # NO NUMBER IN THIS STRING. It used to end «на этой записи показатель почти пуст
        # (0,1 % урока)» — a figure measured on one run, frozen into a constant, and
        # rendered by `report/html.py` in the same table row as the live value. On a run of
        # the same footage with a different room profile the row read «перемещается по
        # классу — 1 %» with «показатель почти пуст (0,1 % урока)» printed beside it. The
        # MECHANISM below is a property of the method and holds on every camera; the size
        # of the effect is a measurement and belongs in the cell next to it.
        "confusion": "порог выбран, а не измерен по размеченной записи. Важнее другое: "
                     "малая доля здесь — свойство метода, а не учителя. Переходы между "
                     "местами как раз и есть те моменты, когда слежение теряет взрослого, "
                     "поэтому они попадают во «вне кадра», а не сюда. Читать малое число "
                     "как «учитель почти не ходил по классу» нельзя.",
    },
    TeacherState.OUT_OF_FRAME: {
        "basis": "отсутствие наблюдения",
        "rule": "в этом кадре ни одна фигура не отнесена к взрослому",
        # The last clause used to promise «кадры, в которых слежение отказалось выбирать
        # между двумя кандидатами». The follower does not choose between candidates at all
        # — that was the design it replaced. What it actually does is drop whole tracklets
        # that claim the same seconds as a better-evidenced one, and that count is now
        # published (`conflicting_tracklets_dropped`, listed in `unmeasured_ru`). Describing
        # a mechanism the code does not have is how a caveat stops being checkable.
        "confusion": "«не увидели» — это не «отсутствовал» и не «ничего не делал». Сюда "
                     "попадают и выход из класса, и заслонение учеником, и отрезки, "
                     "отброшенные потому, что на те же секунды претендовала другая, "
                     "лучше обоснованная цепочка слежения.",
    },
}


# ---------------------------------------------------------------------------------
# CHOSEN CONSTANTS. Every one of them is a judgement, and every one says so.
# ---------------------------------------------------------------------------------

# How far the adult may move between two consecutive observations, in HIS OWN shoulder
# widths, that a single step of the path may cover. Used only to keep `floor_coverage` from
# counting a hand-off as a walk: the adult reappearing four shoulder widths away is a
# tracker event, not a stride. CHOSEN from physics — an adult's shoulder width is ~0.40 m, a
# brisk indoor walk ~1.5 m/s, so at 2 fps one stride is ~1.9 shoulder widths and 4.0 doubles
# it to absorb pose jitter and a dropped frame.
MAX_STRIDE_SCALES = 4.0

# -- the numbers that decide who the adult is --------------------------------------------
#
# The follower does not score bodies against each other. It anchors on a human's polygon and
# then extends only across events the DETECTOR vouches for, so these constants govern how
# generous "the detector vouches for it" is allowed to be, and nothing else.

# A detector track id that goes quiet for longer than this is cut into two tracklets.
# CHOSEN at 2 s (four samples at 2 fps): ByteTrack re-uses ids after an absence, and a
# re-used id spanning a longer silence may be two different people. Short enough that a
# re-use across a real absence is split; long enough not to shred a track that flickers.
TRACKLET_GAP_SECONDS = 2.0

# A hand-off: one tracklet ends and another begins within this long and this far, in the
# earlier one's own shoulder widths.
#
# **The distance is MEASURED, and it is the one number in this module that was chosen by
# being wrong first.** Swept against ten hand-labelled frames of D14 (`scratchpad/eval.py`),
# holding everything else fixed:
#
#     hand-off distance   coverage   correct   wrong   silent
#     4.0 shoulder widths   56.8 %      6        2       1
#     2.0 shoulder widths   45.0 %      7        0       2
#     1.5 shoulder widths   45.0 %      7        0       2
#     1.0 shoulder widths   45.0 %      7        0       2
#
# At 4.0 the chain jumps from the teacher to a pupil standing beside him AT THE BOARD —
# twice in ten frames, and both times it then reports a child's position as the teacher's.
# Two people at a chalkboard stand about two shoulder widths apart, which is precisely why
# 4.0 fails there and nowhere else. Below 2.0 nothing further is bought. CHOSEN at 2.0, the
# loosest value that made no mistakes, at a cost of 11.8 points of coverage — a trade this
# module takes every time, because a missing frame is visible in `out_of_frame` and a wrong
# one is invisible everywhere.
#
# The time is CHOSEN at 4 s rather than measured: 2 s and 4 s and 6 s gave identical answers
# on this lesson, so the data does not constrain it and it is set at the middle of the range
# that behaved the same.
HANDOFF_SECONDS = 4.0
HANDOFF_SCALES = 2.0

# How many anchors a tracklet must place inside the operator's teacher-desk polygon before
# it is taken as the adult's. CHOSEN at 20 observations = 10 s at 2 fps: long enough that a
# pupil walking past the teacher's desk cannot seed the whole track, short enough to catch a
# brief return to the desk. This is the ONE place a track can start; everything else is
# reached from here.
MIN_ZONE_OBSERVATIONS = 20

# -- the two numbers the FALLBACK route needs, which are not the two above ----------------
#
# They were the two above, reused, and the reuse was invisible: the `inferred_by_scale`
# branch of `follow_adult` seeded on `math.dist(...) <= MAX_STRIDE_SCALES * t.scale` and
# `t.observations >= MIN_ZONE_OBSERVATIONS`. Neither constant's stated reasoning covers what
# it was doing there. `MAX_STRIDE_SCALES` is justified as "one stride, doubled" — a claim
# about how far a body travels between two consecutive samples — and was being used as the
# radius of a circle around a seat centre, which is a claim about how close a tracklet has
# to start to a place before it counts as starting there. `MIN_ZONE_OBSERVATIONS` is
# justified by what a pupil walking past a drawn polygon can and cannot do, and there is no
# polygon on this route. Same values, so no number moves; separate names, so the next person
# to tune one of them does not silently move the other.
#
# CHOSEN at 4.0 shoulder widths: `identify_adult` on this route nominated a seat by raw
# shoulder width, and a seat centre is a median of scattered anchors, so the radius has to
# be loose enough to admit a tracklet that begins at the edge of that scatter. It is loose
# on purpose and the route it belongs to carries `needs_confirmation` for exactly that
# reason.
SEED_RADIUS_SCALES = 4.0
# CHOSEN at 20 observations = 10 s at 2 fps, the same duration argument as
# `MIN_ZONE_OBSERVATIONS` and none of its polygon argument: a tracklet shorter than ten
# seconds is not enough of anybody to start a chain from.
MIN_SEED_OBSERVATIONS = 20

# Size veto. A tracklet may not join the adult unless its median observed-over-predicted
# shoulder width is within this factor of the seeds'. MEASURED: over the whole lesson the
# adult sits at 1.09-1.28 and pupils at 0.31-1.44, so size CANNOT identify him -- the
# largest child is bigger for her depth than he is. It can only rule things out, and 1.35
# is CHOSEN to rule out the bottom half of that pupil range while admitting every one of his
# own measured values. A veto, never a reason to attribute.
RATIO_TOLERANCE = 1.35

# In the final one-body-at-a-time selection, one observation inside the operator's polygon
# is worth this many ordinary observations. CHOSEN at 1000, i.e. effectively lexicographic:
# a chain that wandered onto a child can be arbitrarily long and still never outweigh the
# stretch a human's zone pinned down.
ZONE_EVIDENCE_WEIGHT = 1000

# Speed above which the adult is described as in transit rather than settled, in his own
# shoulder widths per SECOND. Per second and not per observation, which is the correction
# to the `MOVING_THRESHOLD = 0.12` this module used to carry: a per-observation threshold
# silently means something different at a different sampling rate, so one setting would
# describe two different teachers at 2 fps and at 5 fps. The old name is deleted rather
# than aliased — nothing in the repository imported it, and an alias is how a wrong unit
# survives a rename. CHOSEN: a walk is ~3.7 shoulder
# widths/s, a person shifting in a chair is under ~0.2, and 0.8 sits in the empty middle.
# NOT measured against labelled footage, and labelled as chosen.
MOVING_SPEED = 0.8

# Minimum duration before a run of one state is admitted as an episode. Below these it is
# counted as a discarded short run and reported, never silently folded into a neighbour.
# CHOSEN: 4 s for the three "places" (a teacher who steps to the board for two seconds to
# point at a line has not "stood at the board"), 1.5 s for the two transient states, which
# would otherwise never survive at all.
MIN_HOLD_SECONDS: dict[TeacherState, float] = {
    TeacherState.AT_BOARD: 4.0,
    TeacherState.AT_DESK: 4.0,
    TeacherState.AMONG_PUPILS: 4.0,
    TeacherState.MOVING: 1.5,
    TeacherState.OUT_OF_FRAME: 1.5,
}

# How many interrupting observations a run tolerates before it is closed. Same value and
# same reasoning as `ledger._min_gap`: one stray observation inside a settled stretch is
# far more likely to be the pose model blinking than the teacher teleporting.
MIN_GAP_OBSERVATIONS = 2

# The side of the floor-coverage cell, in shoulder widths, measured in the perspective-
# warped space where one unit is one shoulder width everywhere in the frame. CHOSEN at 1.0:
# a cell the size of the person means "he stood somewhere he had not stood before", which
# is the coarsest honest reading of «покрытие класса». Smaller cells would count pose
# jitter as exploration; larger ones would hide a walk to the far corner.
FLOOR_CELL_SHOULDER_WIDTHS = 1.0


# ---------------------------------------------------------------------------------
# FOLLOWING ONE ADULT WHO WILL NOT SIT STILL
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdultObservation:
    """One person in one analysed frame, reduced to what the follower needs.

    Deliberately the same three quantities `room/seats.Anchor` carries, so the pipeline
    hands this module the anchor stream it already built rather than a second one that can
    drift from it. `index` points back into the detection arrays for anything downstream
    that wants the keypoints.
    """

    video_seconds: float
    position: Point
    scale: float
    index: int
    # The DETECTOR's own track id, or None/-1 where it would not commit. Carried because
    # this module's whole method rests on it: within one tracklet, "is this the same body"
    # is ByteTrack's answer from box overlap and motion, which is far more information than
    # the shoulder point available here. `room/seats.py` is right that a track id cannot
    # carry identity across a lesson; it carries it across seventy-five seconds, and that is
    # enough to chain.
    track_id: int | None = None


@dataclass(slots=True)
class AdultTrack:
    """One adult, followed across the lesson, with every failure counted.

    `state` is parallel to `frames`: exactly one entry per analysed frame, so the shares
    below have a denominator that is the lesson rather than the part of it that went well.
    """

    frames: list[float] = field(default_factory=list)
    position: list[Point | None] = field(default_factory=list)
    scale: list[float | None] = field(default_factory=list)
    index: list[int | None] = field(default_factory=list)
    speed: list[float | None] = field(default_factory=list)     # shoulder widths / second
    state: list[TeacherState] = field(default_factory=list)

    source: str = "none"                 # designated_zone | inferred_by_scale | none
    needs_confirmation: bool = True
    human_confirmed: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def attributed(self) -> int:
        return sum(1 for p in self.position if p is not None)

    @property
    def analysed(self) -> int:
        return len(self.frames)

    @property
    def found(self) -> bool:
        return self.attributed > 0
@dataclass(frozen=True, slots=True)
class Tracklet:
    """A maximal run of one detector track id: the unit this module reasons about.

    Not a person and not a seat — a stretch of frames the *detector* is willing to say are
    the same body. `room/seats.py` is right that a track id cannot carry identity across a
    lesson; what it can carry is identity across seventy-five seconds, and that turns out to
    be the difference between a solvable problem and an unsolvable one.
    """

    members: tuple[int, ...]          # indices into the observation list
    track_id: int
    start_seconds: float
    end_seconds: float
    start_position: Point
    end_position: Point
    scale: float                      # median shoulder width, px
    size_ratio: float                 # median observed / perspective-predicted width
    zone_observations: int            # how many of its anchors are in the teacher's zone

    @property
    def observations(self) -> int:
        return len(self.members)

    @property
    def duration(self) -> float:
        return max(self.end_seconds - self.start_seconds, 0.0)


def build_tracklets(observations: Sequence[AdultObservation], *,
                    teacher_zone: Polygon | None, perspective=None,
                    gap_seconds: float = TRACKLET_GAP_SECONDS) -> list[Tracklet]:
    """Cut the observation stream into runs of one detector track id.

    A track id is split wherever it goes quiet for longer than `gap_seconds`, because
    ByteTrack will happily re-use an id after an absence and the two halves may be two
    different people. Untracked observations (`track_id < 0`) are dropped: an observation
    the detector would not commit to is not evidence of continuity.
    """
    import numpy as np

    usable = perspective is not None and getattr(perspective, "usable", False)
    groups: dict[int, list[int]] = {}
    for index, observation in enumerate(observations):
        if observation.track_id is None or observation.track_id < 0:
            continue
        groups.setdefault(observation.track_id, []).append(index)

    tracklets: list[Tracklet] = []
    for track_id, members in groups.items():
        members.sort(key=lambda i: observations[i].video_seconds)
        run = [members[0]]
        for index in members[1:]:
            if (observations[index].video_seconds
                    - observations[run[-1]].video_seconds) <= gap_seconds:
                run.append(index)
                continue
            tracklets.append(_tracklet(observations, run, track_id, teacher_zone,
                                       perspective, usable, np))
            run = [index]
        tracklets.append(_tracklet(observations, run, track_id, teacher_zone,
                                   perspective, usable, np))

    tracklets.sort(key=lambda t: t.start_seconds)
    return tracklets


def _tracklet(observations, run, track_id, teacher_zone, perspective, usable, np) -> Tracklet:
    scales = [observations[i].scale for i in run]
    if usable:
        ratios = [observations[i].scale
                  / max(float(perspective.scale_at(observations[i].position[1])), 1e-6)
                  for i in run]
    else:
        # No usable perspective fit means the size term is switched OFF, not defaulted to
        # something plausible. 1.0 for everybody makes every ratio comparison vacuous, which
        # is the correct behaviour when the model that would make it meaningful did not fit.
        ratios = [1.0 for _ in run]
    return Tracklet(
        members=tuple(run), track_id=track_id,
        start_seconds=observations[run[0]].video_seconds,
        end_seconds=observations[run[-1]].video_seconds,
        start_position=observations[run[0]].position,
        end_position=observations[run[-1]].position,
        scale=float(np.median(scales)),
        size_ratio=float(np.median(ratios)),
        zone_observations=(0 if teacher_zone is None else
                           sum(1 for i in run
                               if point_in_polygon(observations[i].position, teacher_zone))),
    )


def follow_adult(observations: Sequence[AdultObservation], frames: Sequence[float], *,
                 layout: RoomLayout | None, sample_interval: float,
                 perspective=None,
                 fallback_centre: Point | None = None) -> AdultTrack:
    """Which observations belong to the adult. Anchored by a human, extended only by
    hand-offs the detector itself vouches for, and silent everywhere else.

    --------------------------------------------------------------------------------
    **TWO EARLIER VERSIONS OF THIS FUNCTION WERE WRONG ON REAL FOOTAGE, AND THE ONLY THING
    THAT CAUGHT EITHER OF THEM WAS DRAWING THE ANSWER ONTO THE FRAMES.**

    *Version one, greedy nearest-neighbour.* Seed from the operator's `teacher_zone`, then
    each frame take the nearest body inside a gate. It tracked the teacher for ninety
    seconds and then spent forty-five minutes on a succession of seated children, reporting
    70 % attribution and a confident set of shares. Two of sixteen checked frames were
    right. The failure is structural: **a nearest-neighbour tracker prefers whatever is
    stationary**, so the moment the teacher stands up and walks, some seated child is nearer
    to his last position than he is, and after one bad frame a greedy tracker has no memory
    of having been wrong.

    *Version two, Viterbi over individual observations*, with a cost made of "how adult-sized
    is this body for its depth" plus a travel term. Choosing the whole path at once fixes the
    greedy failure and did not fix the answer, because the premise was wrong: **size does not
    identify this adult.** Measured over the lesson, his observed-over-predicted shoulder
    width is 1.09-1.28 while the pupils run 0.31-1.44 — the largest child in the room is
    *bigger for her depth than he is*. He is a slight young man in a room of eleven-year-olds
    and the camera cannot see the difference. Three of sixteen frames right.

    Both versions produced numbers that looked entirely reasonable in JSON. Neither carried
    any internal signal that it was wrong. That is the argument for
    `scratchpad/verify.py`-style rendering as part of building a metric, not as a check
    afterwards.

    --------------------------------------------------------------------------------
    **WHAT ACTUALLY WORKS IS TO STOP GUESSING AND START CHAINING.**

    Two facts about this footage, both measured:

      1. **The operator's zone is decisive whenever he is in it.** The tracklet covering
         t = 74..647 s has *every one of its 1 147 anchors* inside the drawn teacher-desk
         polygon. That is not an inference about size or position, it is a human saying
         "the adult's desk is here" and a body being there.

      2. **The detector's own tracklets are long.** The median is short (3.5 s — mostly
         fragments), but the p90 is 75 s and forty of them exceed two minutes. ByteTrack
         uses box overlap and motion, which is far more information than the shoulder point
         this module works in, so *within* a tracklet the identity is the detector's problem
         and it has already solved it.

    So: **seed on the zone, extend only across hand-offs the detector's own geometry
    vouches for, and refuse everywhere else.** A hand-off is one tracklet ending and another
    beginning within `HANDOFF_SECONDS` and `HANDOFF_SCALES` shoulder widths of the same
    point — a detector dropping an id and immediately re-creating it on the same body, which
    is a mechanical event rather than a judgement about who somebody is. Size enters only as
    a *veto* (`RATIO_TOLERANCE`), never as a reason to attribute.

    **Measured result on D14.** Ten frames were labelled by eye across the lesson — two at
    the board early, two at his desk, one where he is genuinely absent from frame, one at
    the board mid-lesson, two among the pupils, one at the board late, one walking. This
    method attributes **45.0 % of the lesson**, gets **7 of 7** attributed labels right,
    correctly refuses the frame where he is not there, and stays silent on 2 frames where he
    is visibly at the board but the chain could not reach him. Versions one and two, on the
    same ten frames, scored 2 and 3.

    The other **55 % is reported as `out_of_frame`**, which is a large and uncomfortable
    number and is the honest one: it means "we could not say", not "he was not there", and
    `presence_block` never lets it be read the other way. A method that covered 90 % by
    guessing would be worse, and the two previous versions of this function are the
    evidence. **The 55 % is also not uniformly distributed** — he is hardest to hold when he
    is at the board, small and partly hidden behind pupils, which is exactly the state the
    client asked about. So board time is the quantity most likely to be UNDERSTATED here,
    and that sentence travels in the artefact rather than living only in this docstring.

    --------------------------------------------------------------------------------
    **THE LADDER OF EVIDENCE, unchanged from `room/zones.py` and for the same reason:**

      1. `designated_zone` — an operator drew `teacher_zone` around the adult's desk. The
         only route where a human is accountable for the answer.
      2. `inferred_by_scale` — no zone, so the seed is whichever tracklets sit at
         `fallback_centre` (the seat `identify_adult` nominated on raw shoulder width). On a
         camera like D14 that nomination is wrong more often than right — he is *further*
         from the lens than the pupils for a third of the lesson — so the whole track
         carries `needs_confirmation` and the artefact says which route ran.
      3. Neither — no adult. The report says so plainly.

    **The adult is in one place at a time**, and the final step enforces it: among the
    claimed tracklets, a maximum-weight non-overlapping subset is selected by ordinary
    interval scheduling, weighting zone evidence far above duration. Everything dropped is
    counted in `conflicting_tracklets_dropped`, because two claims on the same second is the
    signature of a chain having wandered, and a reader should see how often it happened.
    """
    track = AdultTrack()
    track.frames = [float(t) for t in frames]
    n = len(track.frames)
    track.position = [None] * n
    track.scale = [None] * n
    track.index = [None] * n
    track.speed = [None] * n
    track.human_confirmed = bool(layout is not None and layout.zones_confirmed_by)

    teacher_zone = layout.teacher_zone if layout is not None else None
    usable_perspective = perspective is not None and getattr(perspective, "usable", False)
    tracklets = build_tracklets(observations, teacher_zone=teacher_zone,
                                perspective=perspective)

    # -- seed --------------------------------------------------------------------------
    if teacher_zone is not None:
        seeds = {k for k, t in enumerate(tracklets)
                 if t.zone_observations >= MIN_ZONE_OBSERVATIONS}
        source = "designated_zone" if seeds else "none"
    elif fallback_centre is not None:
        seeds = {k for k, t in enumerate(tracklets)
                 if math.dist(t.start_position, fallback_centre) <= SEED_RADIUS_SCALES * t.scale
                 and t.observations >= MIN_SEED_OBSERVATIONS}
        source = "inferred_by_scale" if seeds else "none"
    else:
        seeds, source = set(), "none"

    if not seeds:
        track.source = "none"
        track.needs_confirmation = True
        track.state = [TeacherState.OUT_OF_FRAME] * n
        track.diagnostics = {
            "route": "none",
            "why": ("зона учительского стола размечена, но ни один трек не провёл в ней "
                    f"{MIN_ZONE_OBSERVATIONS} наблюдений подряд"
                    if teacher_zone is not None else
                    "зона учительского стола не размечена и по размеру место взрослого "
                    "не выделено: взрослого нечем задать"),
            # The SAME counters the successful path emits, not a shorter dict. A consumer
            # that has to branch on which keys exist will eventually branch wrongly, and
            # "seeded by zone: 0" is a more useful refusal than a missing key.
            "tracklets_total": len(tracklets),
            "tracklets_seeded_by_zone": 0,
            "tracklets_claimed_after_growth": 0,
            "tracklets_kept_after_conflict_resolution": 0,
            "analysed_frames": n,
            "attributed_frames": 0,
            "attributed_share_of_lesson_percent": 0.0 if n else None,
            "teacher_zone_configured": teacher_zone is not None,
            "zones_confirmed_by": (layout.zones_confirmed_by if layout is not None else ""),
        }
        return track

    import numpy as np
    seed_ratio = float(np.median([tracklets[k].size_ratio for k in seeds]))

    # -- grow across hand-offs ---------------------------------------------------------
    claimed = set(seeds)
    vetoed_by_size = 0
    changed = True
    while changed:
        changed = False
        for candidate in range(len(tracklets)):
            if candidate in claimed:
                continue
            other = tracklets[candidate]
            if usable_perspective and not (
                    1.0 / RATIO_TOLERANCE <= other.size_ratio / max(seed_ratio, 1e-6)
                    <= RATIO_TOLERANCE):
                vetoed_by_size += 1
                continue
            if any(_hands_off(tracklets[member], other) for member in claimed):
                claimed.add(candidate)
                changed = True

    # -- one body at a time ------------------------------------------------------------
    #
    # Ordinary weighted interval scheduling over the claimed tracklets: sort by end time,
    # and for each take the best chain that ends before it starts. The weight puts zone
    # evidence far above duration, so a long chain that wandered onto a child can never
    # outvote the stretch a human's polygon pinned down.
    order = sorted(claimed, key=lambda k: tracklets[k].end_seconds)
    ends = [tracklets[k].end_seconds for k in order]
    best = [0.0] * (len(order) + 1)
    take = [False] * len(order)
    choice = [0] * len(order)
    for position, k in enumerate(order):
        weight = (ZONE_EVIDENCE_WEIGHT * tracklets[k].zone_observations
                  + tracklets[k].observations)
        previous = _last_ending_before(ends, tracklets[k].start_seconds, position)
        with_this = weight + best[previous + 1]
        if with_this > best[position]:
            best[position + 1] = with_this
            take[position] = True
            choice[position] = previous
        else:
            best[position + 1] = best[position]
    selected: list[int] = []
    position = len(order) - 1
    while position >= 0:
        if take[position] and best[position + 1] != best[position]:
            selected.append(order[position])
            position = choice[position]
        else:
            position -= 1
    selected.reverse()
    dropped = len(claimed) - len(selected)

    # -- lay the selected tracklets onto the frame grid --------------------------------
    slot = {t: i for i, t in enumerate(track.frames)}
    attributed = 0
    for k in selected:
        for member in tracklets[k].members:
            observation = observations[member]
            index = slot.get(float(observation.video_seconds))
            if index is None or track.position[index] is not None:
                continue
            track.position[index] = observation.position
            track.scale[index] = observation.scale
            track.index[index] = observation.index
            attributed += 1

    for index in range(1, n):
        here, there = track.position[index], track.position[index - 1]
        if here is None or there is None:
            continue
        elapsed = track.frames[index] - track.frames[index - 1]
        if elapsed <= 0:
            continue
        # Speed only across CONSECUTIVE attributed frames. A displacement measured across a
        # gap we could not see is a guess about what he did while invisible, and it is the
        # number that would decide MOVING.
        track.speed[index] = (math.dist(here, there)
                              / max(track.scale[index] or 1e-6, 1e-6) / elapsed)

    track.source = source
    # `designated_zone` is the only route a human is accountable for, and even it wants a
    # human to have said so: an operator who drew a zone and never signed it is a guess
    # with a polygon. Both facts travel, separately.
    track.needs_confirmation = not (source == "designated_zone" and track.human_confirmed)
    track.diagnostics = {
        "route": source,
        "method": "zone_anchored_tracklets_extended_by_detector_handoffs",
        "tracklets_total": len(tracklets),
        "tracklets_seeded_by_zone": len(seeds),
        "tracklets_claimed_after_growth": len(claimed),
        "tracklets_kept_after_conflict_resolution": len(selected),
        "conflicting_tracklets_dropped": dropped,
        "tracklets_vetoed_by_size": vetoed_by_size,
        "adult_size_ratio_median": round(seed_ratio, 3),
        "perspective_usable": bool(usable_perspective),
        "size_veto_active": bool(usable_perspective),
        "analysed_frames": n,
        "attributed_frames": attributed,
        "attributed_share_of_lesson_percent": _pct(attributed / n) if n else None,
        "handoff_seconds": HANDOFF_SECONDS,
        "handoff_scales": HANDOFF_SCALES,
        "ratio_tolerance": RATIO_TOLERANCE,
        "min_zone_observations": MIN_ZONE_OBSERVATIONS,
        "teacher_zone_configured": teacher_zone is not None,
        "zones_confirmed_by": (layout.zones_confirmed_by if layout is not None else ""),
    }
    return track


def _hands_off(one: Tracklet, other: Tracklet) -> bool:
    """Did the detector drop one id and immediately re-create it on the same body?

    Symmetric in time: either may be the earlier. The test is deliberately mechanical —
    a short silence and a short distance — and contains no judgement about who anybody is.
    Distances are in the EARLIER tracklet's own shoulder widths, so the same rule means the
    same physical distance at the front and the back of the room.
    """
    for earlier, later in ((one, other), (other, one)):
        gap = later.start_seconds - earlier.end_seconds
        if not 0.0 <= gap <= HANDOFF_SECONDS:
            continue
        if math.dist(later.start_position, earlier.end_position) \
                <= HANDOFF_SCALES * max(earlier.scale, 1e-6):
            return True
    return False


def _last_ending_before(ends: list[float], when: float, before: int) -> int:
    """Index of the latest interval that finishes before `when`, or -1. Binary search."""
    low, high, found = 0, before - 1, -1
    while low <= high:
        middle = (low + high) // 2
        if ends[middle] < when:
            found, low = middle, middle + 1
        else:
            high = middle - 1
    return found


def classify_track(track: AdultTrack, layout: RoomLayout | None) -> None:
    """Fill `track.state` — one state per analysed frame, in place.

    Precedence, highest first, with the reasoning for each place in the order:

      1. **OUT_OF_FRAME** when nothing was attributed. It is first because it is not a
         judgement about the adult at all, it is a statement about the measurement, and
         letting any other state win a frame we did not see would be inventing data.
      2. **AT_DESK** before AT_BOARD, which looks backwards and is not. The desk zone is a
         small polygon a human drew around one piece of furniture; the board zone is a
         large one over a three-metre wall. On D14 they overlap horizontally, and when a
         specific claim and a loose one disagree the specific one should win.
      3. **AT_BOARD**.
      4. **MOVING** — in frame, outside both zones, faster than `MOVING_SPEED`.
      5. **AMONG_PUPILS** — everything else. Defined by exclusion, and its entry in
         `DEFINITIONS_RU` says so in the words a reader will see.

    A teacher pacing along the board while explaining is AT_BOARD, not MOVING: zone wins
    over speed, deliberately, because "he was at the board" is the sentence a person in the
    room would say and "he was in transit" is not.

    **The one place this function guesses, counted rather than argued away.** `speed` is
    `None` on the first attributed frame after any gap — `follow_adult` refuses to divide a
    displacement by a stretch it could not see. Step 5 then takes such a frame as settled
    and calls it AMONG_PUPILS, which is a decision made in the absence of the quantity the
    decision is about. It is the right default (a body in view outside both zones is
    somewhere among the pupils, whether or not it was walking) but it is not what
    `DEFINITIONS_RU[AMONG_PUPILS]` used to claim — «скорость ниже порога перемещения» is
    false for exactly these frames. Measured on `D14_20260815103136.mp4`: **2 observations
    of 560**, one second of a 46-minute lesson, so the effect on the totals is nil and the
    honest thing is to publish the count rather than the reassurance.
    """
    board = layout.board_zone if layout is not None else None
    desk = layout.teacher_zone if layout is not None else None

    states: list[TeacherState] = []
    settled_without_speed = 0
    for position, speed in zip(track.position, track.speed, strict=True):
        if position is None:
            states.append(TeacherState.OUT_OF_FRAME)
            continue
        if desk is not None and point_in_polygon(position, desk):
            states.append(TeacherState.AT_DESK)
            continue
        if board is not None and point_in_polygon(position, board):
            states.append(TeacherState.AT_BOARD)
            continue
        if speed is not None and speed >= MOVING_SPEED:
            states.append(TeacherState.MOVING)
            continue
        if speed is None:
            settled_without_speed += 1
        states.append(TeacherState.AMONG_PUPILS)
    track.state = states
    track.diagnostics["among_pupils_observations_with_unmeasurable_speed"] = \
        settled_without_speed


# ---------------------------------------------------------------------------------
# EPISODES
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeacherEpisode:
    """One continuous stretch in one state, after the hold and gap rules."""

    state: TeacherState
    start_seconds: float
    end_seconds: float
    observations: int

    @property
    def duration(self) -> float:
        return max(self.end_seconds - self.start_seconds, 0.0)


def episodes(track: AdultTrack, *, sample_interval: float
             ) -> tuple[list[TeacherEpisode], dict[str, int]]:
    """Run-length encode the state stream, applying the hold and gap rules.

    The same two-rule shape as `ledger._advance`, restated here rather than shared, because
    the two operate on different enums with different minima and a single generic version
    would have to be parameterised by both — at which point it is two functions wearing one
    name. What IS shared is the discipline: a run shorter than its state's minimum is
    DISCARDED and counted, never recorded as a shorter episode, because half-second
    fragments accumulating into "minutes at the board" is exactly the artefact the minimum
    exists to prevent.

    Returns the episodes and the count of what was thrown away, per state. Hundreds of
    discards in one state means the threshold or the sampling rate is wrong, and that count
    is the only thing that says so.
    """
    found: list[TeacherEpisode] = []
    discarded: dict[str, int] = {}

    run_state: TeacherState | None = None
    run_start = run_end = 0.0
    run_count = 0
    gap = 0

    def close() -> None:
        nonlocal run_state, run_count
        if run_state is None:
            return
        duration = max(run_end - run_start, 0.0) + sample_interval
        if duration < MIN_HOLD_SECONDS[run_state]:
            discarded[run_state.value] = discarded.get(run_state.value, 0) + 1
        else:
            found.append(TeacherEpisode(run_state, run_start, run_end + sample_interval,
                                        run_count))
        run_state = None
        run_count = 0

    for when, state in zip(track.frames, track.state, strict=True):
        if run_state is not None and state == run_state:
            run_end = when
            run_count += 1
            gap = 0
            continue
        if run_state is not None:
            gap += 1
            if gap < MIN_GAP_OBSERVATIONS:
                continue     # tolerate the interruption; the run continues
            close()
        run_state, run_start, run_end, run_count, gap = state, when, when, 1, 0
    close()

    found.sort(key=lambda e: e.start_seconds)
    return found, discarded


# ---------------------------------------------------------------------------------
# FLOOR COVERAGE
# ---------------------------------------------------------------------------------


def floor_coverage(track: AdultTrack, *, perspective, everyone: Sequence[AdultObservation],
                   cell: float = FLOOR_CELL_SHOULDER_WIDTHS) -> dict[str, Any]:
    """How much of the room the adult's position visited, and against what denominator.

    **The problem with the obvious measure.** "Area covered" wants square metres, which
    wants a homography onto the floor plane and a known room size, and this project has
    neither — nobody measured the classroom and nothing in the recording says how big it
    is. A number in pixels² is worse than useless: a step at the front of the room covers
    twelve times the pixels of the same step at the back, so a pixel-area measure would
    report a teacher who works at the near desks as covering the room and one who works at
    the board as barely moving. It would be a measurement of the camera.

    **What is available instead is `room/perspective.py`**, already fitted for seat
    clustering, which warps the frame into a space where one unit is one shoulder width
    ANYWHERE. In that space a square grid is a physically meaningful grid, so the measure
    is: **count the distinct one-shoulder-width cells the adult's shoulder point visited.**
    Person-sized cells, because the question is "did he go somewhere else", and a cell
    smaller than the person would count pose jitter as exploration.

    **And the denominator is the honest part.** A raw cell count means nothing on its own —
    is 40 cells a lot? So it is reported against the cells that ANY person in this lesson
    visited: the part of the room that this class actually used. That denominator needs no
    floor plan, no room dimensions and no assumption about where the walls are, and it
    answers the question a reader has, which is whether the adult moved around the space
    that was in use or stayed in one corner of it.

    Both numbers are reported, never only the ratio, and the cell size is reported with
    them, because a coverage share without its cell size is a number whose units are a
    setting. When the perspective fit is not usable the whole block refuses: an unwarped
    grid would be a grid of the camera again.
    """
    if perspective is None or not getattr(perspective, "usable", False):
        return {"available": False,
                "why": "перспективная модель комнаты не сошлась; сетка в пикселях измеряла "
                       "бы камеру, а не класс"}

    import numpy as np

    def cells(points: list[Point]) -> set[tuple[int, int]]:
        if not points:
            return set()
        warped = perspective.warp(np.asarray(points, dtype=float))
        return {(math.floor(u / cell), math.floor(v / cell))
                for u, v in warped}

    adult_points = [p for p in track.position if p is not None]
    room_points = [o.position for o in everyone]
    adult_cells = cells(adult_points)
    room_cells = cells(room_points)

    # Path length in the same warped space: shoulder widths actually walked. Reported
    # beside coverage because they answer different questions -- a teacher pacing two
    # metres for forty minutes has a long path and a tiny coverage, and a report that
    # showed only one of them would describe him wrongly either way.
    path = 0.0
    previous: Point | None = None
    for position in track.position:
        if position is None:
            previous = None
            continue
        if previous is not None:
            warped = perspective.warp(np.asarray([previous, position], dtype=float))
            step = math.dist(tuple(warped[0]), tuple(warped[1]))
            # A step longer than one plausible stride is a hand-off, not a walk. Not summed,
            # so the path length cannot be inflated by the tracker re-acquiring him
            # somewhere else.
            if step <= MAX_STRIDE_SCALES:
                path += step
        previous = position

    return {
        "available": True,
        "cell_side_shoulder_widths": cell,
        "cells_visited_by_adult": len(adult_cells),
        "cells_visited_by_anyone": len(room_cells),
        "share_of_room_in_use_percent": (
            _pct(len(adult_cells & room_cells) / len(room_cells)) if room_cells else None),
        "path_length_shoulder_widths": round(path, 1),
        "what_this_is_not": "Это не площадь в метрах: ни размеры класса, ни план пола "
                            "неизвестны. Единица — клетка размером с человека в "
                            "перспективно выпрямленных координатах.",
    }


# ---------------------------------------------------------------------------------
# THE METRICS OBJECT
# ---------------------------------------------------------------------------------


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 1)


def _r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True, slots=True)
class TeacherMetrics:
    """Position facts about the adult. No judgement, and nowhere to put one."""

    available: bool
    coverage: float
    seated_share: float | None
    standing_or_away_share: float | None
    out_of_frame_share: float | None
    # **POSE transitions only: how many times his body went from settled-at-his-desk to
    # upright-or-away and back.** `None` when the pose half did not run, and `None` rather
    # than a number is the whole point of the type.
    #
    # It used to be filled, on the track-only path, from
    # `presence["transitions_between_episodes"]` — the count of boundaries between POSITION
    # episodes, which includes every boundary where the follower started or stopped seeing
    # him. Two different quantities under one key, and the collision was not theoretical:
    # `report/summary.py` prints this field inside a sentence that opens «что делало его
    # тело за его собственным столом, из наблюдений на этом месте, где поза прочиталась»,
    # so on a D14 run with no seat the report said «смен положения — 25» about a lesson in
    # which not one pose observation existed and 17 of those 25 boundaries were us losing
    # him. `cabinet/report.py` puts the same field in a column headed «смен положения»
    # across lessons from different cameras, where the two meanings would have been added
    # up by eye. The position count keeps its own two unambiguous names inside `presence`
    # (`transitions_between_episodes` and `..._excluding_out_of_frame`) and is not copied
    # out to a shorter one.
    transitions: int | None
    longest_seated_minutes: float | None
    longest_standing_minutes: float | None
    motion_per_observation: float | None
    # Observed minutes, so that nobody downstream has to multiply a share by a duration to
    # get one -- and, more importantly, so nobody reaches for `episode_seconds.seated`
    # instead and reports 6 minutes where the answer is 46. See `ledger.summary`.
    observed_minutes: float | None = None
    # "at desk" = SEATED + HEAD_DOWN + TURNED_AWAY + HAND_RAISED. Deliberately NOT called
    # `seated_minutes`: see `to_dict`.
    seated_minutes: float | None = None
    standing_or_away_minutes: float | None = None
    reason: str = ""
    # Everything the D14 taxonomy adds, kept in one sub-dict so that the pose-based fields
    # above keep their exact meaning. They are different measurements of the same man --
    # the ones above ask what his body was doing at his place, these ask where in the room
    # he was -- and merging them into one flat namespace is how "seated" came to mean two
    # things and produce a false sentence. See `ledger.summary`.
    presence: dict[str, Any] | None = None
    board_zone_configured: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "available": self.available,
            "coverage": round(self.coverage, 3),
            # The suffix is not decoration: it names the DENOMINATOR. These are shares of
            # the observations in which the adult was visible, not of the lesson -- he was
            # out of frame for part of it, and `out_of_frame_share_of_lesson` says how much.
            # `at_desk`, not `seated`, and the rename is the second half of the same fix.
            # This quantity sums FOUR states -- SEATED, HEAD_DOWN, TURNED_AWAY,
            # HAND_RAISED -- because all four are the adult at his desk. But `seated` is
            # ALSO the name of one of those four states, whose own total is 8.6 minutes
            # against this 46.3. Calling both "seated" is the identical trap that produced
            # the false sentence in the first place, one level up, and it was found by
            # verifying the fix rather than by review.
            "at_desk_share_of_observed": _pct(self.seated_share),
            "standing_or_away_share_of_observed": _pct(self.standing_or_away_share),
            "out_of_frame_share_of_lesson": _pct(self.out_of_frame_share),
            # ...and the same facts already converted, so a consumer never multiplies.
            "observed_minutes": _r(self.observed_minutes),
            "at_desk_minutes": _r(self.seated_minutes),
            "standing_or_away_minutes": _r(self.standing_or_away_minutes),
            "transitions": self.transitions,
            "longest_at_desk_episode_minutes": _r(self.longest_seated_minutes),
            "longest_standing_episode_minutes": _r(self.longest_standing_minutes),
            "motion_per_observation": _r(self.motion_per_observation, 4),
            "reason": self.reason,
            "not_an_assessment_ru": self.not_an_assessment_ru,
        }
        if self.presence is not None:
            data["presence"] = self.presence
        return data

    @property
    def not_an_assessment_ru(self) -> str:
        """The sentence that must appear wherever any of these numbers appear.

        It changes with the camera, and that is the point: on a camera with no board zone
        the largest caveat is that «у доски» is unmeasurable, and on D14 it is measurable
        and the largest caveat becomes something else. A single frozen sentence would be
        wrong on one of the two cameras, and a caveat that is wrong is worse than none
        because it is still believed.

        **Which is exactly the trap the second branch fell into.** It used to say «доска на
        этой камере находится позади объектива» whenever `board_zone_configured` was false
        — that is, whenever nobody had drawn a polygon. On D14 the board is a metre and a
        half of green chalkboard in the middle of the frame, and a run without `--room`
        printed, as a statement of fact about the hardware, that it was behind the lens.
        The absence of a config file is not an observation about where a camera points.
        What is actually known in that branch is only that the zone was never marked, so
        that is all the sentence now claims; which of the two reasons applies is a fact
        about the room that the operator has, and the report asks for it instead of
        guessing it.
        """
        common = (
            "Это описание положения в пространстве, а не оценка работы учителя. "
            "«Сидит» не означает «не ведёт урок»: за столом можно вести обсуждение, "
            "проверять работы или наблюдать за самостоятельной работой; сидя за партой "
            "рядом с учениками — разбирать задачу. Здесь нет ни балла активности, ни "
            "порога, ни сравнения учителей между собой, и добавлять их нельзя. "
        )
        if self.board_zone_configured:
            return common + (
                "Доска на этой камере видна, поэтому «у доски» измеряется; но время вне "
                "кадра — это «мы не увидели», а не «отсутствовал», и складывать его с "
                "остальным нельзя."
            )
        return common + (
            "Кроме того, для этой камеры не размечена зона доски, поэтому «у доски» здесь "
            "не измеряется вовсе: ноль в этой строке означает «не измеряли», а не «не "
            "подходил к доске». Причина может быть двух родов — доска стоит позади "
            "объектива или её просто никто не разметил, — и по самой записи они не "
            "различаются; это знает тот, кто ставил камеру."
        )


def board_occupancy(everyone: Sequence[AdultObservation], layout: RoomLayout | None,
                    interval: float, analysed_frames: int) -> dict[str, Any]:
    """How long ANYBODY stood at the board — a fact about the room, not about a person.

    **Why this exists beside `board`, which already answers the client's question.**
    `board` reports how long the ADULT was at the board, and that number depends on the
    follower having held on to him. Measured on this lesson, it does not: the adult is
    attributed 4.0 minutes at the board, while somebody is standing at the board for
    **30.6 minutes** of the same lesson. Checked by eye against the recording, the adult is
    unmistakably writing on the board at 10:32:36 — and at that moment the follower has
    him as `out_of_frame`, because at that distance the pose model resolved his shoulders
    at 16.8 px and the size veto that protects the chain from wandering onto a child
    refused the hand-off. The same man at the same place ten minutes later resolves at
    52.1 px and is attributed correctly.

    So `board` is a **lower bound with a known leak**, and a lower bound quoted alone is
    the failure this codebase keeps finding. This quantity has no such leak, because it
    asks a question that needs no identity: was there a person, any person, whose shoulder
    line was inside the operator's board polygon.

    **It is deliberately NOT "the teacher at the board".** Part of the 30.6 minutes is
    pupils: at 11:01:36 a girl is standing at the board pointing at it while the adult sits
    among the class. Reporting this as teacher time would be a straightforward lie. What
    the pair of numbers gives a reader is the truth and its bracket — the adult was at the
    board at least this long, the board was in use at most that long, and the difference is
    pupils plus whatever the follower lost.
    """
    if layout is None or layout.board_zone is None:
        return {
            "available": False,
            "reason_ru": "зона доски не задана для этой камеры — занятость доски не измеряется",
        }

    frames_with_somebody: set[float] = set()
    observations = 0
    for observation in everyone:
        position = getattr(observation, "position", None)
        if position is None:
            continue
        if point_in_polygon(position, layout.board_zone):
            observations += 1
            frames_with_somebody.add(round(float(observation.video_seconds), 3))

    frames = len(frames_with_somebody)
    return {
        "available": True,
        "minutes": round(frames * interval / 60.0, 1),
        "share_of_lesson_percent": _pct(frames / analysed_frames) if analysed_frames else None,
        "frames_with_somebody": frames,
        "observations_in_zone": observations,
        "means_ru": "Сколько времени у доски находился ХОТЬ КТО-ТО — учитель или ученик. "
                    "Личность здесь не устанавливается и не нужна.",
        "why_it_differs_from_board_ru":
            "Это число всегда БОЛЬШЕ времени учителя у доски, и разница складывается из "
            "двух разных вещей: у доски бывают ученики, и слежение теряет взрослого именно "
            "там, где он дальше всего от камеры. Считать разницу временем учителя нельзя, "
            "но и считать время учителя полным — тоже: оно нижняя граница.",
    }


def presence_block(track: AdultTrack, *, sample_interval: float,
                   layout: RoomLayout | None,
                   perspective=None,
                   everyone: Sequence[AdultObservation] = (),
                   head_directions: Counter | None = None,
                   wall_clock=None) -> dict[str, Any]:
    """The state taxonomy turned into numbers, every one of them carrying its denominator.

    **Two families of share, and they are not interchangeable.** This is the same
    distinction `ledger.summary` had to learn the hard way, restated here before it can be
    made again:

      * `*_share_of_lesson` — denominator is every analysed frame in the lesson window,
        including the ones where the adult was not found. These sum to 100 and include
        `out_of_frame`. This is the family that answers «какую часть урока».
      * `*_share_of_attributed` — denominator is only the frames the follower attributed to
        the adult. These also sum to 100 and exclude `out_of_frame`. This is the family
        that answers «из того, что мы видели».

    A reader who takes a number from one family and a sentence from the other produces a
    false statement about a real person, which has already happened once in this project.
    So the denominator is in the name of every field, and the two families are never
    presented without both.

    `episode_minutes_by_state` is a THIRD quantity and smaller than either: only time
    inside runs that survived their minimum hold. It answers «сколько длились эпизоды,
    которые мы готовы назвать событиями», and it is the one to quote next to a count of
    episodes. Quoting it next to a share is the error.
    """
    analysed = track.analysed
    attributed = track.attributed
    interval = sample_interval

    observed_counts = Counter(track.state)
    by_state_lesson = {
        state.value: _pct(observed_counts.get(state, 0) / analysed) if analysed else None
        for state in TeacherState
    }
    in_frame = attributed
    by_state_attributed = {
        state.value: (_pct(observed_counts.get(state, 0) / in_frame) if in_frame else None)
        for state in TeacherState if state is not TeacherState.OUT_OF_FRAME
    }
    minutes_by_state = {
        state.value: round(observed_counts.get(state, 0) * interval / 60.0, 1)
        for state in TeacherState
    }

    found, discarded = episodes(track, sample_interval=interval)
    episode_minutes = {state.value: 0.0 for state in TeacherState}
    episode_counts = {state.value: 0 for state in TeacherState}
    longest = {state.value: 0.0 for state in TeacherState}
    for episode in found:
        episode_minutes[episode.state.value] += episode.duration / 60.0
        episode_counts[episode.state.value] += 1
        longest[episode.state.value] = max(longest[episode.state.value], episode.duration)

    # A transition is a change between two ADJACENT admitted episodes of different states.
    # Counted over episodes rather than over raw frames on purpose: at 2 fps a body on a
    # zone boundary flickers, and a frame-level count would report hundreds of "changes of
    # position" for a man standing still at the edge of the board zone.
    transitions = 0
    for earlier, later in itertools.pairwise(found):
        if earlier.state is not later.state:
            transitions += 1
    # ...and the transitions that involve OUT_OF_FRAME separately, because a change from
    # «не видели» to «у доски» is not the teacher moving, it is us starting to see him.
    transitions_excluding_out_of_frame = sum(
        1 for earlier, later in itertools.pairwise(found)
        if earlier.state is not later.state
        and TeacherState.OUT_OF_FRAME not in (earlier.state, later.state))

    board_observations = observed_counts.get(TeacherState.AT_BOARD, 0)
    board_zone_configured = layout is not None and layout.board_zone is not None

    # **Zero at the board and "we never looked at the board" are different values, and the
    # `board` block used to render them identically.** With no `board_zone` in the layout,
    # `classify_track` cannot ever emit `AT_BOARD`, so every figure below is structurally
    # 0.0 — and the block still printed `0.0 мин · 0 % урока` under the heading «то, о чём
    # спрашивал заказчик», followed by `direction_of_error_ru` = «Скорее занижено.
    # Слежение теряет взрослого чаще всего именно у доски…», which explains a measurement
    # that was never taken. Camera 01's own room profile has `board_zone: null`, so this is
    # the state of the ONLY other configured camera in the repository, not a hypothetical.
    #
    # So the numbers are `None` there and the sentence beside them says the zone is
    # missing. `None` and `0.0` survive JSON and reach `qorgan` as different values;
    # 0.0 with a caveat attached does not.
    if board_zone_configured:
        board_block: dict[str, Any] = {
            "minutes_of_lesson": round(board_observations * interval / 60.0, 1),
            "share_of_lesson_percent": by_state_lesson[TeacherState.AT_BOARD.value],
            "share_of_attributed_percent":
                by_state_attributed.get(TeacherState.AT_BOARD.value),
            "episodes": episode_counts[TeacherState.AT_BOARD.value],
            "longest_episode_minutes": round(longest[TeacherState.AT_BOARD.value] / 60.0, 1),
            "zone_configured": True,
            # The one direction of error this method is known to have, stated beside the
            # number it affects rather than in a footnote. The follower loses the adult most
            # easily where he is smallest and most often hidden behind pupils, and on a
            # camera at the back of the room that place is the board. So this figure is a
            # LOWER BOUND on his time at the board, and the gap between it and the truth is
            # somewhere inside `out_of_frame`.
            "direction_of_error_ru":
                "Скорее занижено. Слежение теряет взрослого чаще всего именно у доски: там "
                "он дальше всего от камеры и его заслоняют ученики, вышедшие к доске. "
                "Разница спрятана в доле «вне кадра», и по этой записи её размер неизвестен.",
        }
    else:
        board_block = {
            "minutes_of_lesson": None,
            "share_of_lesson_percent": None,
            "share_of_attributed_percent": None,
            "episodes": None,
            "longest_episode_minutes": None,
            "zone_configured": False,
            "direction_of_error_ru":
                "Не измерялось. Для этой камеры не размечена зона доски, поэтому состояние "
                "«у доски» не может возникнуть ни в одном кадре. Прочерк здесь — это «не "
                "измеряли», а не «не подходил к доске»; время у доски попало в «среди "
                "учеников» или «перемещается».",
        }

    block: dict[str, Any] = {
        "schema": "teacher_presence/1",
        "identification": {
            "route": track.source,
            "needs_confirmation": track.needs_confirmation,
            "confirmed_by_human": track.human_confirmed,
            **track.diagnostics,
        },
        "analysed_frames": analysed,
        "attributed_frames": attributed,
        "attributed_share_of_lesson_percent": _pct(attributed / analysed) if analysed else None,
        "lesson_minutes": round(analysed * interval / 60.0, 1),
        "sample_interval_seconds": interval,

        "state_share_of_lesson_percent": by_state_lesson,
        "state_share_of_attributed_percent": by_state_attributed,
        "state_minutes_of_lesson": minutes_by_state,

        "episode_counts": episode_counts,
        "episode_minutes_by_state": {k: round(v, 1) for k, v in episode_minutes.items()},
        "longest_episode_minutes_by_state": {k: round(v / 60.0, 1) for k, v in longest.items()},
        "discarded_short_runs": discarded,
        "transitions_between_episodes": transitions,
        "transitions_between_episodes_excluding_out_of_frame":
            transitions_excluding_out_of_frame,

        # The client's sentence, answered directly, with both denominators beside it so it
        # cannot be quoted as one number -- or refused outright when no zone was drawn.
        "board": board_block,
        "board_occupancy": board_occupancy(everyone, layout, interval, analysed),
        "floor_coverage": floor_coverage(track, perspective=perspective, everyone=everyone),
        "per_minute": per_minute(track, sample_interval=interval, wall_clock=wall_clock),
        "definitions_ru": {state.value: {"label": RU_LABELS[state],
                                         **DEFINITIONS_RU[state]}
                           for state in TeacherState},
        "thresholds": {
            "moving_speed_shoulder_widths_per_second": MOVING_SPEED,
            "min_hold_seconds": {s.value: v for s, v in MIN_HOLD_SECONDS.items()},
            "min_gap_observations": MIN_GAP_OBSERVATIONS,
        },
    }

    if head_directions is not None:
        # The «пишет или объясняет» split, offered and immediately qualified. At the board
        # the adult's shoulder width on D14 is ~50 px, so his ears are a couple of pixels
        # and `geometry.head_direction` is reading confidences at the edge of what they
        # carry. Reported with its own coverage and flagged INFERRED so nobody quotes it
        # as an observation.
        total = sum(head_directions.values())
        # **The old `share_of_board_observations_percent` could not be anything but 100, and
        # it was printed as if it were a coverage counter.** It divided "board observations
        # whose head direction was not UNKNOWN" by "board observations" — and
        # `geometry.head_direction` returns UNKNOWN only when it can see neither a head
        # keypoint NOR the two shoulders, while `geometry.anchor` refuses to produce a
        # position at all without those same two shoulders at the same threshold. So every
        # observation the follower could place at the board is, by construction, an
        # observation whose direction is not UNKNOWN. On D14 it duly printed `100.0` beside
        # a caveat that says «доля кадров, где направление вообще прочиталось, указана
        # рядом» — a number that cannot fall, sold as the thing that measures whether the
        # reading worked. That is a coverage counter in name only, which under this
        # project's second rule is worse than none.
        #
        # What actually varies is whether a HEAD was visible at all. `AWAY` is not a sighting
        # of the back of a head; it is the residual bucket for "no ear, no nose, no eye, but
        # shoulders are there", which at a 50 px shoulder width is also what a head too small
        # to resolve looks like. So the honest coverage is the share of board observations
        # carrying a real head keypoint, and `away` is named as the leftover it is.
        away = int(head_directions.get(HeadDirection.AWAY.value, 0))
        head_visible = total - away
        block["board"]["facing"] = {
            "basis": "вывод, слабый",
            "observations": total,
            "head_keypoint_visible_observations": head_visible,
            "head_keypoint_visible_share_of_board_percent":
                _pct(head_visible / board_observations) if board_observations else None,
            # The WHOLE distribution, not two of its five members. The first version
            # printed only `toward_camera` and `away` -- 14.4 % and 3.9 % on this lesson --
            # and a reader would reasonably subtract them from 100 and conclude something
            # about the missing 81.7 %, which is neither of those two things: it is
            # profiles, where he is turned side-on and the camera cannot say whether he is
            # writing or talking. Reporting a partial distribution as if it were a whole
            # one is the same class of error as a share without its denominator.
            "share_by_direction_percent": {
                direction.value: _pct(head_directions.get(direction.value, 0) / total)
                for direction in HeadDirection
                if direction is not HeadDirection.UNKNOWN
            } if total else None,
            "caveat_ru": "У доски взрослый находится в дальнем конце класса: ширина плеч "
                         "около 50 пикселей, уши — единицы пикселей. «Лицом к классу» "
                         "как «объясняет» и «спиной» как «пишет» — это истолкование, а не "
                         "измерение. Доля «спиной» здесь особенно ненадёжна: так помечен "
                         "не только затылок, но и любой кадр, где голову вообще не удалось "
                         "разобрать при видимой линии плеч. Рядом указано, в какой доле "
                         "кадров у доски нашлась хоть одна точка головы.",
        }

    block["unmeasured_ru"] = _unmeasured(layout, track)
    return block


def _unmeasured(layout: RoomLayout | None, track: AdultTrack) -> list[dict[str, str]]:
    """What this half of the report could not see, listed rather than omitted."""
    items: list[dict[str, str]] = [
        {"what": "что именно учитель говорил или объяснял",
         "why": "в файле нет аудиодорожки; речь не измеряется вообще"},
        {"what": "к кому из учеников учитель подходил",
         "why": "«среди учеников» определено исключением и не связывает взрослого с "
                "конкретным местом"},
        {"what": "качество урока",
         "why": "не измеряется и не выводится: камера не отличает «сидит» от «не ведёт "
                "урок». Это отказ, а не пробел в данных"},
    ]
    attributed = track.diagnostics.get("attributed_share_of_lesson_percent")
    if attributed is not None and attributed < 80.0:
        items.append({
            # Comma, not a point: this string is rendered verbatim into a Russian report
            # that writes «48,5 %» three lines above it, and «55.0 %» beside «55,0 %» is
            # how a page starts looking machine-generated.
            "what": "положение взрослого в "
                    f"{str(round(100.0 - attributed, 1)).replace('.', ',')} % кадров урока",
            "why": "слежение за взрослым опирается на размеченную зону стола и на "
                   "передачи трека, которым доверяет сам детектор; всё остальное отнесено "
                   "к «вне кадра». Это «не смогли определить», а не «отсутствовал», и "
                   "складывать эту долю с остальными как время нельзя"})
    if layout is None or layout.teacher_zone is None:
        items.append({"what": "«за своим столом»",
                      "why": "зона учительского стола не размечена для этой камеры"})
    if layout is None or layout.board_zone is None:
        items.append({"what": "«у доски»",
                      "why": "зона доски не размечена для этой камеры; события попадают "
                             "в «среди учеников» или «перемещается»"})
    # This used to read `track.diagnostics["ambiguous_frames_refused"]`, a key nothing in
    # the module has ever written. A refusal that is announced in the report only when a
    # counter fires, and whose counter is never incremented, is a refusal that does not
    # exist; the branch was dead and read as coverage of a case that is not covered. The
    # two counters below ARE written, on every successful run, by `follow_adult` and
    # `classify_track` respectively.
    dropped = track.diagnostics.get("conflicting_tracklets_dropped")
    if dropped:
        items.append({
            "what": f"{dropped} отрезков слежения, отброшенных как одновременные",
            "why": "взрослый может быть только в одном месте, и там, где две цепочки "
                   "претендовали на одни и те же секунды, оставлена та, что опирается на "
                   "размеченную зону; отброшенное время ушло в «вне кадра», а не угадано"})
    blind = track.diagnostics.get("among_pupils_observations_with_unmeasurable_speed")
    if blind:
        items.append({
            "what": f"скорость взрослого в {blind} кадрах из числа отнесённых к «среди "
                    "учеников»",
            "why": "это первые кадры после перерыва в слежении: смещение за невидимый "
                   "промежуток не считается, поэтому «перемещается» или «стоит» для них "
                   "не различается и они отнесены к «среди учеников»"})
    return items


def per_minute(track: AdultTrack, *, sample_interval: float, wall_clock=None
               ) -> list[dict[str, Any]]:
    """One row per minute of the lesson, for charting. Counts, not only shares.

    Counts as well as shares because the last minute of a lesson is usually a partial one,
    and a chart drawn from shares alone renders it at full height — a bar that says the
    teacher spent 100 % of a minute at the board when the minute was eleven seconds long.
    `observations` is on every row so a renderer can fade or drop the short ones.
    """
    if not track.frames:
        return []
    origin = track.frames[0]
    rows: dict[int, Counter] = {}
    for when, state in zip(track.frames, track.state, strict=True):
        minute = int((when - origin) // 60)
        rows.setdefault(minute, Counter())[state.value] += 1

    output: list[dict[str, Any]] = []
    for minute in sorted(rows):
        counts = rows[minute]
        total = sum(counts.values())
        row: dict[str, Any] = {
            "minute": minute,
            "start_s": round(origin + minute * 60.0, 2),
            "observations": total,
            "seconds_covered": round(total * sample_interval, 1),
            "observations_by_state": dict(counts),
            "share_of_minute_percent": {state: _pct(count / total)
                                        for state, count in counts.items()},
        }
        if wall_clock is not None:
            with contextlib.suppress(Exception):
                row["wall"] = wall_clock.at(origin + minute * 60.0).strftime("%H:%M")
        output.append(row)
    return output


def timeline(track: AdultTrack, *, sample_interval: float) -> list[dict[str, Any]]:
    """The lesson as spans, in the shape `report/charts.py` already draws for a seat.

    `measured` is False for OUT_OF_FRAME, which is what tells the strip renderer to hatch
    it rather than colour it: a band that renders "we did not see him" exactly like "he was
    at his desk" tells the reader the opposite of the truth about those minutes.
    """
    found, _ = episodes(track, sample_interval=sample_interval)
    return [{"state": e.state.value, "start_s": round(e.start_seconds, 2),
             "end_s": round(e.end_seconds, 2), "observations": e.observations,
             "measured": e.state is not TeacherState.OUT_OF_FRAME}
            for e in found]


def teacher_metrics(ledger=None, *, track: AdultTrack | None = None,
                    sample_interval: float | None = None,
                    layout: RoomLayout | None = None,
                    perspective=None,
                    everyone: Sequence[AdultObservation] = (),
                    head_directions: Counter | None = None,
                    wall_clock=None) -> TeacherMetrics:
    """Position facts for the adult.

    Two independent sources, and both are optional because on different cameras different
    ones exist:

      * `ledger` — a `SeatLedger` for a seat the adult occupied. This is the pose-based
        half (at his desk / upright / turned away) and it only exists when the adult HAS a
        settled place. On D14 he does not, for most of the lesson.
      * `track` — an `AdultTrack` from `follow_adult`. This is the position-based half, and
        it is the one that answers the client's question.

    With neither, this refuses and says why. It does not fall back to reporting the pose
    half as though it were the whole answer, which is what the previous version had to do
    because the position half did not exist yet.
    """
    presence = None
    board_configured = bool(layout is not None and layout.board_zone is not None)
    if track is not None and track.found:
        if sample_interval is None:
            raise ValueError("a track without its sample_interval cannot be turned into "
                             "minutes; every *_seconds total here is a count of "
                             "observations multiplied by it (see `ledger.sample_interval`)")
        presence = presence_block(track, sample_interval=sample_interval, layout=layout,
                                  perspective=perspective, everyone=everyone,
                                  head_directions=head_directions, wall_clock=wall_clock)

    if ledger is None:
        if presence is None:
            return TeacherMetrics(False, 0.0, None, None, None, 0, None, None, None,
                                  reason="взрослый не опознан ни по зоне, ни по размеру",
                                  board_zone_configured=board_configured)
        out_of_frame = presence["state_share_of_lesson_percent"][
            TeacherState.OUT_OF_FRAME.value]
        return TeacherMetrics(
            available=True,
            coverage=(presence["attributed_share_of_lesson_percent"] or 0.0) / 100.0,
            seated_share=None, standing_or_away_share=None,
            out_of_frame_share=None if out_of_frame is None else out_of_frame / 100.0,
            # None, not `presence["transitions_between_episodes"]`. See the field.
            transitions=None,
            longest_seated_minutes=None, longest_standing_minutes=None,
            motion_per_observation=None,
            observed_minutes=round(track.attributed * sample_interval / 60.0, 1)
            if track is not None and sample_interval else None,
            # Deliberately left None rather than 0.0. The pose half did not run, and a zero
            # here would read as «за столом не сидел», which is a claim nobody made.
            seated_minutes=None, standing_or_away_minutes=None,
            reason="взрослый не привязан к постоянному месту: он перемещается по классу, "
                   "поэтому позные показатели за столом не считались",
            presence=presence, board_zone_configured=board_configured)

    analysed = ledger.observations + ledger.absent_observations
    if analysed == 0:
        return TeacherMetrics(False, 0.0, None, None, None, 0, None, None, None,
                              reason="ни одного проанализированного кадра",
                              presence=presence, board_zone_configured=board_configured)

    coverage = ledger.coverage
    # **`out_of_frame_share_of_lesson` must mean "we did not locate the adult", and from
    # the seat ledger alone it does not.** The ledger counts frames in which the adult's
    # SEAT was empty. On every camera before D14 that was the same fact, because the adult
    # sat at that desk all lesson. On D14 it is a different fact and a false one: he spends
    # two thirds of the lesson away from his desk, plainly inside the frame, at the board
    # and among the pupils. Measured on `D14_20260815103136.mp4`: the ledger says 51,5 %
    # «вне кадра» and the follower says 55,0 %, and they were both in the same artefact
    # under names a reader cannot tell apart — one of them printed on the cabinet's adult
    # page under the heading «вне кадра, %».
    #
    # So when the follower ran, its number is the one that answers the question the name
    # asks, and the seat's emptiness stays where it is honestly labelled:
    # `ledger.absent_observations`, rendered by the summary as «на нём никого не было».
    # One name, one meaning, on both cameras.
    if presence is not None:
        share = presence["state_share_of_lesson_percent"][TeacherState.OUT_OF_FRAME.value]
        out_of_frame = None if share is None else share / 100.0
    else:
        out_of_frame = ledger.absent_observations / analysed

    measured = sum(count for state, count in ledger.state_observations.items()
                   if state != PupilState.UNKNOWN.value)
    if measured == 0:
        return TeacherMetrics(False, coverage, None, None, out_of_frame, 0, None, None,
                              None, reason="взрослый в кадре, но поза ни разу не прочиталась",
                              presence=presence, board_zone_configured=board_configured)

    # "Seated" for the adult means at their own settled place with the shoulder line at
    # its settled height -- exactly the states a pupil would be given, reused deliberately
    # so there is one definition of sitting in this codebase rather than two that drift.
    seated = (ledger.state_observations.get(PupilState.SEATED.value, 0)
              + ledger.state_observations.get(PupilState.HEAD_DOWN.value, 0)
              + ledger.state_observations.get(PupilState.TURNED_AWAY.value, 0)
              + ledger.state_observations.get(PupilState.HAND_RAISED.value, 0))
    upright = (ledger.state_observations.get(PupilState.STOOD_UP.value, 0)
               + ledger.state_observations.get(PupilState.AWAY_FROM_PLACE.value, 0)
               + ledger.state_observations.get(PupilState.AT_BOARD.value, 0))

    seated_states = {PupilState.SEATED, PupilState.HEAD_DOWN, PupilState.TURNED_AWAY,
                     PupilState.HAND_RAISED}
    transitions = 0
    previous: bool | None = None
    longest_seated = longest_standing = 0.0
    for episode in ledger.episodes:
        is_seated = episode.state in seated_states
        if previous is not None and is_seated != previous:
            transitions += 1
        previous = is_seated
        if is_seated:
            longest_seated = max(longest_seated, episode.duration)
        elif episode.state is not PupilState.UNKNOWN:
            longest_standing = max(longest_standing, episode.duration)

    observed_minutes = ledger.observed_seconds / 60.0
    return TeacherMetrics(
        available=True,
        coverage=coverage,
        seated_share=seated / measured,
        standing_or_away_share=upright / measured,
        out_of_frame_share=out_of_frame,
        transitions=transitions,
        longest_seated_minutes=longest_seated / 60.0,
        longest_standing_minutes=longest_standing / 60.0,
        motion_per_observation=ledger.motion_per_observation,
        observed_minutes=observed_minutes,
        seated_minutes=observed_minutes * seated / measured,
        standing_or_away_minutes=observed_minutes * upright / measured,
        presence=presence,
        board_zone_configured=board_configured,
    )

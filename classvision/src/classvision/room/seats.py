"""Seats, discovered from the footage itself — the identity this module actually rests on.

--------------------------------------------------------------------------------
**THE PROBLEM THIS SOLVES, AND WHY THE OBVIOUS ANSWERS DO NOT.**

The client wants per-pupil metrics that accumulate over WEEKS: «первая неделя хорошо,
вторая неделя не очень». That needs a stable identity for a child across a lesson and
across a term. Three candidates, two of which fail on measured evidence:

  * **A tracker id fails within one lesson.** ByteTrack loses a person behind another
    person; one child occluded and re-acquired becomes two ids, and two children who
    cross can swap. A track lives minutes. The neighbouring `qorgan` codebase concluded
    from exactly this that per-child classroom metrics are impossible, and within its
    own frame of reference it was right.

  * **A face fails on this footage too, though not for the reason usually given.** The
    faces here are fine — median **64 px**, near-frontal, nothing like the 11.5 px
    corridor measurement that decided the earlier verdict. They still do not identify:
    matched against the 141-pupil gallery, the median best cosine was **0.30** with a
    margin of **0.10** over the runner-up (`MEASUREMENTS.md` §4). That is a preference,
    not a recognition, and a name attached to it would be a guess wearing a number.

  * **A seat works, and it works because a classroom is not a corridor.** Children sit in
    the same place for 45 minutes and, in a school with a seating plan, for a term. A
    seat is a fixed point in a fixed room observed by a fixed camera: it does not die to
    occlusion, it does not swap when two children cross, and it accumulates hundreds of
    observations per lesson instead of a handful.

So the unit of accumulation here is a **place**, not a track and not a face. Tracks give
short-term continuity between samples; faces contribute evidence; the seat is what
persists. This is the one substantive thing this module knows that the codebase next door
does not, and it is available only because a classroom recording finally exists.

--------------------------------------------------------------------------------
**A SEAT IS STILL NOT A CHILD, AND NOTHING HERE PRETENDS OTHERWISE.**

Everything below produces anonymous places: `seat_3`, not a name. Attaching a name is a
separate, explicit act with its own evidence standard (`identity/`), and when that
standard is not met the report says «место 3» for the whole term and remains useful.
Two children who swap chairs mid-term swap their histories with them, and no amount of
clustering detects that — only a human with the seating plan does. That is stated in the
artefact as a caveat on every surface, not buried here.

--------------------------------------------------------------------------------
**HOW THE SEATS ARE FOUND.**

Not from a plan (we have none) and not from furniture detection (an empty chair is not a
seat that was used). From where people actually were: cluster every observation's anchor
over the whole lesson, and the dense, temporally-persistent clusters are the seats.

The distance between two anchors is normalised by their own shoulder widths, because the
room is viewed at an angle: two pupils sharing a front desk are ~150 px apart while two
sharing a back desk are ~60 px apart, and a pixel threshold would either merge the back
row into one seat or split the front row into two. Normalised, both pairs are about the
same number of shoulder widths apart, which is the whole reason `geometry.py` works in
those units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from classvision.room import perspective

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Anchor:
    """One observation reduced to what seat-finding needs."""

    video_seconds: float
    position: Point
    scale: float          # this person's shoulder width, in pixels
    track_id: int | None


@dataclass(slots=True)
class Seat:
    """A place in the room that somebody occupied, with the evidence that it exists."""

    seat_id: int
    centre: Point
    scale: float                    # median shoulder width observed here
    observations: int
    first_seen: float
    last_seen: float
    # How much of the lesson this place was occupied, as a fraction of the analysed
    # window. The single most useful number for telling a seat from a doorway: a seat is
    # occupied most of the lesson, a spot where somebody paused is not.
    occupancy: float
    spread: float                   # median distance of members from the centre, in scales
    members: list[int] = field(default_factory=list)   # indices into the anchor list

    @property
    def label(self) -> str:
        return f"seat_{self.seat_id}"


# A cluster must hold at least this fraction of the analysed frames to be called a seat.
# Below it, the cluster is somebody who stood there for a while -- the adult at the front,
# a pupil at the board, a visitor in the doorway. CHOSEN, and deliberately low: the cost
# of admitting a non-seat is a place that the report shows as mostly-absent, which is
# visible; the cost of rejecting a real seat is a child who vanishes from the term's data,
# which is not.
MIN_OCCUPANCY = 0.25

# A cluster whose members scatter further than this from their own centre is not a place,
# it is a path -- somebody walking through. CHOSEN from the geometry: two pupils sharing a
# desk sit ~1.5-2.5 shoulder widths apart, so a blob wider than 1.2 is spanning furniture.
MAX_SPREAD_SCALES = 1.2


# Discovery thins the anchors to one observation per place per this many seconds.
#
# **This is not an optimisation, it is a correctness fix, and it was found by running the
# pipeline.** Single-link clustering joins two clusters through any chain of intermediate
# points. At a 10 s sampling interval the anchors are sparse and the classroom's nine
# places separated cleanly. At 2 fps — twenty times denser — a pupil leaning toward a
# neighbour lays a continuous trail of anchors between the two desks, single-link walks
# along it, and the whole room collapsed: **9 seats became 3**. Denser data made the
# answer worse, which is the signature of a chaining failure rather than a threshold one.
#
# Thinning restores the sparsity the method needs while assignment still uses every
# observation, so nothing is thrown away — discovery answers "where are the places", which
# needs spatial coverage, not temporal resolution. 5 s is CHOSEN: comfortably sparser than
# the 2 fps that failed, comfortably denser than the 10 s that worked.
DISCOVERY_THIN_SECONDS = 5.0


# DBSCAN's neighbourhood radius, in shoulder widths, after the perspective warp. Swept
# rather than fixed, and selected by agreement with the detector's own people count.
# The range brackets both regimes seen: well under the ~1.5-2.5 shoulder widths that
# separate two pupils sharing a desk, and above the jitter of a seated body.
EPS_SWEEP = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75)

# What fraction of the discovery frames must hold an anchor within `eps` for a point to be
# a DBSCAN core point. This is the number that kills chaining: a seat is occupied for most
# of the lesson, a trail between two desks for a second or two. CHOSEN at 5 %, well below
# the 25 % a real seat must clear to survive `MIN_OCCUPANCY` anyway, so it removes trails
# without being a second occupancy test in disguise.
CORE_FRACTION = 0.05


class _Auto:
    """Sentinel for `same_place=AUTO`. A class, so it cannot be confused with a float."""

    def __repr__(self) -> str:
        return "AUTO"


AUTO = _Auto()


def _choose_threshold(sweep: dict[float, int], expected_people: float | None
                      ) -> tuple[float, str]:
    """Pick the clustering threshold using evidence that does not come from clustering.

    **The evidence is the detector's own people-per-frame count**, which is produced by a
    completely different mechanism (a detection head) and is therefore not circular. A
    room where the model sees nine people should yield nine places; the threshold that
    achieves that is the right one, and picking it automatically is more honest than
    shipping a constant that happened to work on one sampling rate.

    Among the thresholds that agree with the count, the LARGEST is chosen. The failure
    mode on either side is asymmetric: too small splits one pupil into two places (visible
    immediately as two half-occupied seats side by side), too large merges two pupils into
    one (invisible, and silently averages two children together). Preferring the larger
    end of the agreeing run keeps us as far as possible from the splitting edge while
    still agreeing, and the merging edge is guarded by the agreement test itself.
    """
    if expected_people is None:
        # **This used to return `SAME_PLACE_SCALES` (0.9) and it was a latent disaster.**
        # 0.9 is a threshold for the single-link clusterer that no longer exists; handed
        # to DBSCAN as `eps` it is outside `EPS_SWEEP` (0.20-0.75) entirely, and on this
        # footage it produces **zero seats** — an empty room, reported confidently, with
        # every downstream count correctly summing to nothing. `pipeline.analyse` always
        # supplies the count today, so it never fired; it was one refactor away from
        # silently emptying a school's term.
        #
        # There is no safe default here. The whole method rests on an independent estimate
        # of how many people are in the room, and without one the honest answer is a
        # refusal, not a guess.
        raise ValueError(
            "seat discovery needs `expected_people` — an independent estimate of how many "
            "people are in the room (the detector's median count per frame). The clustering "
            "radius is selected by agreeing with it; there is no defensible default, and "
            "the previous one silently returned an empty room."
        )

    target = round(expected_people)
    agreeing = sorted(value for value, count in sweep.items() if count == target)
    if agreeing:
        return agreeing[-1], f"agrees with {target} people per frame"

    # Nothing matched exactly. Take the closest, and say so -- a near miss is worth
    # reporting as a near miss rather than as a clean answer.
    best = min(sweep, key=lambda v: (abs(sweep[v] - target), -v))
    return best, (f"no threshold gave {target} places; closest was {sweep[best]} "
                  f"at {best}")


def discover(anchors: list[Anchor], *, analysed_frames: int,
             same_place: float | _Auto = AUTO,
             min_occupancy: float = MIN_OCCUPANCY,
             max_spread: float = MAX_SPREAD_SCALES,
             thin_seconds: float = DISCOVERY_THIN_SECONDS,
             expected_people: float | None = None) -> tuple[list[Seat], dict]:
    """Find the seats, and report enough about the finding to argue with it.

    Returns the seats and a diagnostics block. The diagnostics are not decoration: the
    threshold above is a chosen number, and the only defence against a chosen number is
    showing what the answer would have been at other values. `seat_count_by_eps`
    is what a human checks against the pupils they can count in one frame, and
    `expected_people` makes that check automatic — a room where the model routinely sees
    nine people but only three places were found has failed, and must say so rather than
    return three confident seats.
    """
    if not anchors:
        return [], {"anchors": 0, "reason": "no anchors"}

    # Discovery works on the thinned set; assignment (a separate call) uses every anchor.
    sample = _thin(anchors, thin_seconds)
    sample_frames = len({round(a.video_seconds / max(thin_seconds, 1e-6)) for a in sample})

    model = perspective.fit(np.array([a.position for a in sample], dtype=float),
                            np.array([a.scale for a in sample], dtype=float))

    # A point is CORE only if this many neighbours sit within `eps`. Derived from the
    # data rather than chosen: a seat occupied for the whole lesson contributes about
    # `sample_frames` anchors, so a fraction of that is "persistently occupied" while a
    # walk-through is not. The floor of 4 keeps very short recordings clustering at all.
    min_samples = max(4, int(CORE_FRACTION * sample_frames))

    sweep = {value: _count_density(sample, value, min_samples, model, sample_frames,
                                   min_occupancy, max_spread, thin_seconds)
             for value in EPS_SWEEP}
    chosen_by = "given"
    if same_place is AUTO:
        same_place, chosen_by = _choose_threshold(sweep, expected_people)

    labels = _cluster_density(sample, same_place, min_samples, model)

    seats: list[Seat] = []
    rejected: list[dict] = []

    for label in sorted(set(labels)):
        # -1 is DBSCAN's noise label: anchors that were never part of a dense place. They
        # are people walking, leaning between desks, and the frame edge. They are not a
        # seat, and `assign()` still places them by proximity, so nothing is discarded --
        # they simply do not get to define where a place is.
        if label < 0:
            continue
        members = [i for i, value in enumerate(labels) if value == label]
        points = np.array([sample[i].position for i in members], dtype=float)
        scales = np.array([sample[i].scale for i in members], dtype=float)
        centre = (float(np.median(points[:, 0])), float(np.median(points[:, 1])))
        scale = float(np.median(scales))
        spread = float(np.median([math.dist(centre, tuple(p)) for p in points]) / max(scale, 1e-6))
        # Occupancy counts DISTINCT frames, not observations: two detections of the same
        # person in one frame (which happens when a box splits) must not inflate it. It is
        # computed against the THINNED frame count, so it stays a fraction of the lesson
        # rather than a fraction of the sampling rate.
        frames_here = len({round(sample[i].video_seconds / max(thin_seconds, 1e-6))
                           for i in members})
        occupancy = frames_here / max(sample_frames, 1)

        record = dict(centre=centre, observations=len(members), occupancy=round(occupancy, 3),
                      spread=round(spread, 2))
        if occupancy < min_occupancy or spread > max_spread:
            record["why"] = ("too transient" if occupancy < min_occupancy else "too scattered")
            rejected.append(record)
            continue

        times = [sample[i].video_seconds for i in members]
        seats.append(Seat(
            seat_id=0, centre=centre, scale=scale, observations=len(members),
            first_seen=min(times), last_seen=max(times), occupancy=occupancy,
            spread=spread, members=members,
        ))

    # Number the seats by reading order in the image -- top of frame first, then left to
    # right within a row -- so `seat_3` means the same place to a human looking at the
    # picture as it does to the report. Rows are banded by half a scale because a "row"
    # of desks is not a straight line in a wide-angle view.
    seats.sort(key=lambda s: (round(s.centre[1] / max(s.scale, 1e-6) / 1.5), s.centre[0]))
    for number, seat in enumerate(seats, start=1):
        seat.seat_id = number

    diagnostics = {
        "anchors": len(anchors),
        "anchors_used_for_discovery": len(sample),
        "thin_seconds": thin_seconds,
        "analysed_frames": analysed_frames,
        "discovery_frames": sample_frames,
        "seats_found": len(seats),
        "clusters_rejected": rejected,
        "perspective": perspective.summarise(
            model, np.array([a.position for a in sample], dtype=float),
            np.array([a.scale for a in sample], dtype=float)),
        "min_samples": min_samples,
        "method": "dbscan_in_perspective_corrected_space",
        "thresholds": {"eps_shoulder_widths": same_place, "chosen_by": chosen_by,
                       "min_occupancy": min_occupancy, "max_spread_scales": max_spread},
        # What the answer would have been at every threshold tried. If this is flat around
        # the chosen value the choice is safe; if it is a cliff, the room needs a human.
        "seat_count_by_eps": {str(k): v for k, v in sorted(sweep.items())},
    }

    # The automatic sanity check. A room where the model routinely sees nine people but
    # only three places were found has failed, and the failure is silent unless something
    # compares the two. This is the check that would have caught the chaining collapse
    # without a human noticing the seat map looked wrong.
    if expected_people is not None:
        diagnostics["expected_people"] = round(expected_people, 1)
        shortfall = expected_people - len(seats)
        diagnostics["plausible"] = bool(shortfall <= max(1.0, 0.2 * expected_people))
        if not diagnostics["plausible"]:
            diagnostics["warning"] = (
                f"found {len(seats)} places but the detector sees a median of "
                f"{expected_people:.0f} people per frame. Places are probably being merged "
                f"(single-link chaining): try a smaller same_place threshold or a larger "
                f"thin_seconds. These seats should not be trusted."
            )
    return seats, diagnostics


def _thin(anchors: list[Anchor], seconds: float) -> list[Anchor]:
    """One anchor per place per `seconds`, keeping spatial coverage and dropping density.

    Bucketed by time AND by coarse position, so thinning cannot silently drop a whole
    seat: two pupils in the same time bucket occupy different position buckets and both
    survive. The position bucket is deliberately coarse — a third of a shoulder width —
    so it thins a stationary pupil without merging neighbours.
    """
    if seconds <= 0:
        return list(anchors)
    kept: dict[tuple[int, int, int], Anchor] = {}
    for a in anchors:
        cell = max(a.scale / 3.0, 1.0)
        key = (int(a.video_seconds // seconds), int(a.position[0] // cell),
               int(a.position[1] // cell))
        kept.setdefault(key, a)
    return sorted(kept.values(), key=lambda a: a.video_seconds)


def _count_density(sample: list[Anchor], eps: float, min_samples: int, model,
                   sample_frames: int, min_occupancy: float, max_spread: float,
                   thin_seconds: float) -> int:
    """How many places survive at this `eps`. The sweep's inner loop.

    Applies the SAME occupancy and spread filters the real pass applies, so the sweep
    reports what would actually be returned rather than a raw cluster count — a sweep that
    counts something different from what the algorithm returns is a lookup table for the
    wrong question.
    """
    labels = _cluster_density(sample, eps, min_samples, model)
    count = 0
    for label in set(labels):
        if label < 0:
            continue
        members = [i for i, value in enumerate(labels) if value == label]
        points = np.array([sample[i].position for i in members], dtype=float)
        scale = float(np.median([sample[i].scale for i in members]))
        centre = (float(np.median(points[:, 0])), float(np.median(points[:, 1])))
        spread = float(np.median([math.dist(centre, tuple(p)) for p in points])
                       / max(scale, 1e-6))
        frames_here = len({round(sample[i].video_seconds / max(thin_seconds, 1e-6))
                           for i in members})
        if frames_here / max(sample_frames, 1) >= min_occupancy and spread <= max_spread:
            count += 1
    return count


def _cluster_density(anchors: list[Anchor], eps: float, min_samples: int,
                     model) -> list[int]:
    """DBSCAN in perspective-corrected space. The replacement for single-link, and why.

    **Single-link chained, and the chaining got worse with more data — which is the tell.**
    It joins two clusters through ANY path of intermediate points, so a pupil leaning
    toward a neighbour lays a trail of anchors that welds two desks into one place. On a
    sparse 10-second sample the trail had too few points to connect and the room resolved
    into 9 correct seats; at 2 fps it collapsed to 3, and over the full 52 minutes to 5.
    An algorithm that gets *less* correct as you give it more evidence is the wrong
    algorithm, not a badly tuned one.

    DBSCAN breaks the chain because a point only propagates a cluster if it is a CORE
    point — if at least `min_samples` neighbours sit within `eps`. A seat holds a pupil
    for most of the lesson and is dense; a trail is traversed in a second or two and is
    not. So the trail becomes noise (label −1) instead of a bridge, and `assign()` later
    places those observations by proximity anyway, so nothing is thrown away.

    Points are warped by `room/perspective.py` first, so `eps` is one number meaning one
    shoulder width everywhere in a frame whose scale varies twelvefold. That is what makes
    a single `eps` legitimate rather than a compromise between the front and back rows.
    """
    from sklearn.cluster import DBSCAN

    points = np.array([a.position for a in anchors], dtype=float)
    warped = model.warp(points) if model is not None and model.usable else points / max(
        float(np.median([a.scale for a in anchors])), 1e-6)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(warped)
    return [int(v) for v in labels]


def assign(anchors: list[Anchor], seats: list[Seat], *,
           gate_scales: float = 1.5) -> list[int | None]:
    """Which seat each anchor belongs to, or None for "not at any seat".

    `None` is a first-class answer and the reason this is not a nearest-neighbour lookup:
    the adult at the front, a pupil standing at the board, and somebody in the doorway are
    all genuinely at no seat, and forcing them into the closest one would post their
    movements to a child's record. The gate is generous (1.5 shoulder widths) because a
    seated pupil leaning across their desk must not fall out of their own seat.
    """
    if not seats:
        return [None] * len(anchors)
    centres = np.array([s.centre for s in seats], dtype=float)
    out: list[int | None] = []
    for anchor in anchors:
        distances = np.hypot(centres[:, 0] - anchor.position[0],
                             centres[:, 1] - anchor.position[1])
        best = int(np.argmin(distances))
        limit = gate_scales * max(anchor.scale, seats[best].scale, 1e-6)
        out.append(seats[best].seat_id if distances[best] <= limit else None)
    return out

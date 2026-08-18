"""The room: where the board is, where the teacher's desk is, and who the adult is.

**Why the adult has to be separated first, not filtered later.** On this footage the
teacher sits nearest the lens, so his shoulder width is ~220 px against a pupil's ~70 px,
and he moves more than anyone. Of the 45 observations the hand-raise rule fired on across
the whole lesson, **35 were him** — he rests a hand against his head while working. Any
pupil statistic computed before separating him is a statistic about one adult. Filtering
afterwards does not help either, because by then his observations have already set the
scale distribution, the seat clustering and the activity baseline.

**Three ways to identify him, in descending order of trustworthiness, and the artefact
records which one was used.** A room where a human has drawn the teacher's zone once is
the only case where this is certain; everything else is the module guessing, and a guess
that is not labelled as one is the failure this project keeps finding.

  1. `DESIGNATED` — a polygon in the room config. The operator drew it once for this
     camera. Costs one minute per classroom, per install.
  2. `INFERRED_BY_SCALE` — the persistent place with much the largest shoulder width, on
     the reasoning that the adult's desk is nearest the camera and adults are bigger.
     This is what runs when nobody has configured anything, and the artefact carries
     `needs_confirmation: true` so the psychologist's page can say so.
  3. `NONE` — no adult identified. Every person is treated as a pupil and the report says
     that plainly. Correct for a camera pointed at a room with no teacher desk in frame.

**The board zone is a claim about this camera's mounting, and it is not guessable.** On
the first recordings the camera hung above the board looking back at the class, so a pupil
"at the board" was at the BOTTOM edge of the frame, large and foreshortened — the opposite
of where naive intuition puts them. Camera D14 is mounted the other way round: at the back
of the room, looking forward, with the three-panel chalkboard in frame. Where exactly is
NOT restated here: it is `board_surface` in `configs/camera_d14.yaml`, measured once, and
this docstring carried a second, different set of numbers for the same rectangle
(x 1030..1610, y 135..305 against the profile's 1024..1603, 137..313) until a review noticed
that a room had two coordinates in two shipped files. One measured object, one place it is
written down. Get the MOUNTING backwards and «стоит у доски» silently
becomes «ушёл в конец класса». So there is no default board polygon: without one,
`AT_BOARD` is never produced and out-of-place pupils are reported as `AWAY_FROM_PLACE`,
which is true but less informative. A wrong zone would be worse than a missing one.

**Zones are tested against the SHOULDER ANCHOR, not against a foot point, and that is a
correction driven by a measurement.** The obvious convention is the bottom-centre of the
person's box — where they stand on the floor — and `pipeline.analyse` used it. On D14 it
does not work, because the front row of desks stands between the camera and the board and
crops the legs off exactly the people the zone exists to find: in the frame at t = 5 s the
adult is standing AT the board and his box bottom lands at y ≈ 430 where the desk occludes
him, and in the frame at t = 565 s he is SEATED at his own desk and his box bottom lands at
y ≈ 420. Two states a metre and a half apart, four pixels apart in the quantity being
measured. The shoulder midpoint separates them cleanly — y ≈ 190 standing at the board
against y ≈ 336 seated at the desk — for the same reason `geometry.anchor` gives for
preferring it everywhere else: furniture crops boxes, and a box edge measures the furniture.

So a zone here is **the region of the image in which a person's shoulder line appears when
they are in that place**, which for a wall-mounted board means a rectangle over the board
itself rather than a strip of floor in front of it. That is one convention, used by the
pupil path and the teacher path alike, because one polygon interpreted two ways is the
drift this codebase keeps finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

Point = tuple[float, float]
Polygon = tuple[Point, ...]


class AdultSource(StrEnum):
    DESIGNATED = "designated"
    INFERRED_BY_SCALE = "inferred_by_scale"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RoomLayout:
    """Everything about this camera's view of this room that a human decided.

    Stored per camera and versioned with the artefact, because a room whose zones were
    redrawn between two lessons has produced two incomparable reports, and the only way
    to notice is to have both zone sets on file.
    """

    camera: str
    frame_width: int
    frame_height: int
    # All three are tested against the SHOULDER ANCHOR of a person, never against a foot
    # point -- see the module docstring for the measurement that forced that choice.
    #
    # `board_zone`   — where a person's shoulders appear when they are AT the board.
    # `teacher_zone` — the adult's own desk: where his shoulders appear when he is seated
    #                  at it. This is the primary route to identifying the adult, and on a
    #                  camera like D14, where the adult walks the room and is often FURTHER
    #                  from the lens than the pupils, it is the only route that works;
    #                  `identify_adult`'s scale comparison assumes an adult parked nearest
    #                  the camera and quietly nominates a front-row pupil when that is false.
    # `door_zone`    — reserved; nothing consumes it yet, and it is not invented for it.
    board_zone: Polygon | None = None
    teacher_zone: Polygon | None = None
    door_zone: Polygon | None = None
    # Seats the operator has excluded — an empty desk used for storage, a chair that is
    # really the doorway. Named by the seat ids `room/seats.py` discovers.
    excluded_seats: tuple[int, ...] = ()
    # Who drew or checked these polygons, free text, empty when nobody has. A zone is a
    # human claim about a room, and the artefact must be able to say whether a human ever
    # made it: an operator's name here is the difference between «взрослый определён по
    # размеченной зоне» and «программа предположила». It is deliberately not a boolean —
    # "confirmed" with nobody attached to it is how an unchecked default becomes a fact.
    zones_confirmed_by: str = ""
    notes: str = ""

    def contains(self, zone: Polygon | None, point: Point) -> bool:
        return zone is not None and point_in_polygon(point, zone)


@dataclass(slots=True)
class AdultDecision:
    """Who the adult is, how we decided, and whether a human should check."""

    seat_id: int | None
    source: AdultSource
    needs_confirmation: bool
    evidence: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.seat_id is not None


# An adult's shoulder width must exceed the median pupil's by at least this factor before
# scale alone is allowed to nominate them. CHOSEN, and deliberately large: on this footage
# the ratio is ~3.1 (≈220 px against ≈70 px) because the teacher's desk is nearest the
# lens, so a bar at 1.8 is met comfortably here while refusing to fire in a room where
# everyone is a similar distance away -- which is exactly the room where guessing would
# be wrong. When it does not fire, the answer is NONE and the report says so.
ADULT_SCALE_RATIO = 1.8


def identify_adult(seats, layout: RoomLayout | None = None) -> AdultDecision:
    """Which discovered seat, if any, is the adult's.

    `seats` is the list from `room/seats.discover`. The scale comparison uses the median
    of the OTHER seats rather than of all of them, so that one very large person cannot
    raise the bar they are being measured against.
    """
    # `evidence["why"]` is RUSSIAN, and that is not a style preference. `pipeline.assemble`
    # splices this exact string into `uncertainty.notes`, which the report prints verbatim
    # under «ЧТО ИЗМЕРИТЬ НЕ УДАЛОСЬ» — so the psychologist's page read «Взрослому не
    # сопоставлено постоянное место (a teacher_zone is configured but no seat fell inside
    # it -- the zone or the camera moved)». Developer English in the middle of a Russian
    # sentence is the thing that makes a reader stop trusting the numbers around it. The
    # KEYS stay English, because keys are code; the values are read by a human.
    if not seats:
        return AdultDecision(None, AdultSource.NONE, needs_confirmation=False,
                             evidence={"why": "ни одного места не найдено"})

    if layout is not None and layout.teacher_zone is not None:
        for seat in seats:
            if point_in_polygon(seat.centre, layout.teacher_zone):
                return AdultDecision(seat.seat_id, AdultSource.DESIGNATED,
                                     needs_confirmation=False,
                                     evidence={"zone": "teacher_zone", "centre": seat.centre})
        return AdultDecision(None, AdultSource.NONE, needs_confirmation=True,
                             evidence={"why": "зона учительского стола размечена, но центр "
                                              "ни одного найденного места в неё не попал: "
                                              "либо зона, либо камера сдвинулась, либо "
                                              "взрослый нигде не сидел достаточно долго, "
                                              "чтобы место сложилось"})

    ranked = sorted(seats, key=lambda s: s.scale, reverse=True)
    candidate = ranked[0]
    others = [s.scale for s in ranked[1:]]
    if not others:
        return AdultDecision(None, AdultSource.NONE, needs_confirmation=True,
                             evidence={"why": "найдено единственное место; сравнивать "
                                              "размеры не с чем"})

    median_other = sorted(others)[len(others) // 2]
    ratio = candidate.scale / max(median_other, 1e-6)
    if ratio < ADULT_SCALE_RATIO:
        return AdultDecision(
            None, AdultSource.NONE, needs_confirmation=True,
            evidence={"why": "ни одно место не выделяется размером настолько, чтобы "
                             "уверенно назвать его местом взрослого",
                      "largest_scale": round(candidate.scale, 1),
                      "median_other_scale": round(median_other, 1),
                      "ratio": round(ratio, 2), "required": ADULT_SCALE_RATIO},
        )
    return AdultDecision(
        candidate.seat_id, AdultSource.INFERRED_BY_SCALE, needs_confirmation=True,
        evidence={"largest_scale": round(candidate.scale, 1),
                  "median_other_scale": round(median_other, 1),
                  "ratio": round(ratio, 2), "required": ADULT_SCALE_RATIO,
                  "centre": candidate.centre},
    )


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting. Vertices in image pixels, in the frame the layout declares.

    Written out rather than pulled from shapely because it is fifteen lines, it is the
    only geometry this module needs, and a school install with one fewer C extension to
    build on a Windows box is a school install that starts.
    """
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


# `bottom_band(width, height, fraction=0.22)` used to live here: a helper that manufactured
# "the bottom 22 % of the frame" as a STARTING POINT for a board zone on a camera mounted
# above the board. It is deleted rather than kept, and the reason is worth the seven lines
# it replaces.
#
# It had no callers and no test. What it had was a shape: a function that turns two integers
# into a plausible board polygon, sitting in the module whose own docstring says that a board
# zone which is merely plausible produces «стоит у доски» events that are merely plausible,
# «and those go into a child's record». The next person in a hurry would have used it, and
# the artefact would have carried a `board_zone` that nobody measured and nothing marks as
# guessed — `zones_confirmed_by` would be empty, which is the same thing an honestly drawn
# but unsigned polygon says.
#
# It is also now unusable by construction: since `layout_io` requires `board_surface`
# alongside any non-null `board_zone`, a band across the bottom of the frame would have to
# be declared to overlap a chalk rectangle it does not touch. A dead function that the
# loader would refuse is a trap with a friendly name.
#
# A camera with the board behind the lens gets `board_zone: null` and a written reason —
# `configs/camera_01.yaml` is the worked example.

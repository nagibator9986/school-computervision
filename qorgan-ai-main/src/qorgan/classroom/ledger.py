"""One track's running totals for one lesson. No I/O, no config lookup, no database.

A ledger is fixed size: eight counters and four floats, whatever the length of the
lesson. Nothing here grows with the number of frames -- no history list, no per-frame
buffer, no dict keyed on anything (rule R8). A lesson is 45 minutes of frames, and the
legacy's habit of keeping a list per track is what made a long run and a leak the same
thing.

**Every count here is debounced, and the debounce is not tidiness -- it is the metric.**
A raw "was the wrist above the line this frame" summed over a lesson is a count of pose
model frames, not a count of raised hands: at 15 fps a hand held up for four seconds
scores sixty. Worse, a wrist hovering either side of the threshold scores once per
crossing, so the child with the least steady arm tops the report. So a raise is counted
when the pose has been HELD for `min_hold_observations`, and cannot be counted again
until the wrist has been DOWN for `min_gap_observations`. The two thresholds are what
turn a per-frame predicate into an event.

**Zero and unknown are different, and this is the third time this project has paid for
confusing them** (see `migrations/0005`, on a boolean that meant both "merged away" and
"left the school"). A track whose baseline never settled has no seat to be away from and
no seated shoulder line to have risen above, so its `stands` and `away_seconds` are not
zero -- they are unmeasured. `settled` carries that distinction out of here and onto the
row, and the report renders the two differently. Hand raises are exempt: they need no
baseline, so their zero is a real zero even on an unsettled track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qorgan.classroom.posture import (
    anchor,
    hand_raised,
    left_the_place,
    rose_from_seat,
    shoulder_width,
)
from qorgan.config.classroom import HandRaiseRules, PlaceRules
from qorgan.detection.skeleton import Keypoints

Point = tuple[float, float]


@dataclass(slots=True)
class TrackLedger:
    """What one anonymous track did. There is no name here and no column for one."""

    track_id: int
    first_seen: float
    last_seen: float

    observations: int = 0

    # The largest shoulder width this track has shown, used as its unit of length.
    # The LARGEST rather than this frame's: shoulder width foreshortens when a child
    # turns, and a shrinking scale shrinks every threshold with it, which would make a
    # turned child easier to credit with a raised hand than a facing one. The maximum is
    # the closest thing to their true width that arrives without a calibration target.
    scale: float = 0.0

    # The seated baseline, averaged over the settling window and then frozen.
    settled: bool = False
    seat_y: float = 0.0
    seat_xy: Point = (0.0, 0.0)
    _sum_y: float = 0.0
    _sum_x: float = 0.0
    _settle_samples: int = 0

    hand_raises: int = 0
    _hand_up_run: int = 0
    _hand_down_run: int = 0
    _hand_counted: bool = False

    stands: int = 0
    _stand_up_run: int = 0
    _stand_down_run: int = 0
    _standing: bool = False

    away_seconds: float = 0.0
    _away_since: float | None = None

    # Excursions discarded for being shorter than `min_away_seconds`. Kept because it is
    # the direct measure of whether that threshold is set anywhere near right: a lesson
    # where this is in the hundreds is a lesson where the tracker is jittering, not one
    # where the children kept nearly leaving.
    brief_excursions: int = 0

    _place: PlaceRules | None = field(default=None, repr=False)

    def observe(
        self, person: Keypoints, at: float, hand: HandRaiseRules, place: PlaceRules
    ) -> None:
        """Fold one frame's skeleton into the totals.

        **A frame whose shoulders are not visible concludes NOTHING**, and the guard is
        per FRAME rather than per track. It was written per track first -- "has this
        track ever had a usable width" -- and that is a different question with a wrong
        answer: once a child had been measured once, every later frame in which they
        turned away, or were occluded by the child in front, was scored against the
        remembered scale and read as **hand down**. Ten such frames satisfy
        `min_gap_observations`, so the raise counter re-armed, and one continuously
        raised hand was counted twice. Caught by
        `test_an_unmeasurable_frame_is_not_counted_as_hand_down`.

        `last_seen` still advances (the tracker did see this person), but `observations`
        does not: it counts frames actually MEASURED, which is what makes it possible to
        tell a track present for ten minutes and readable throughout from one present for
        ten minutes and readable in a tenth of them.
        """
        self.last_seen = at
        self._place = place

        width = shoulder_width(person)
        shoulders = anchor(person)
        if width is None or shoulders is None:
            return

        self.scale = max(self.scale, width)
        self.observations += 1

        self._observe_hand(person, hand)
        self._settle(shoulders, place)
        self._observe_place(shoulders, at, place)

    # -- hands ---------------------------------------------------------------

    def _observe_hand(self, person: Keypoints, hand: HandRaiseRules) -> None:
        """Needs no baseline: a wrist is measured against that person's own shoulders."""
        if hand_raised(person, self.scale, hand.above_shoulder_ratio):
            self._hand_down_run = 0
            self._hand_up_run += 1
            if self._hand_up_run >= hand.min_hold_observations and not self._hand_counted:
                self.hand_raises += 1
                self._hand_counted = True
            return

        self._hand_up_run = 0
        self._hand_down_run += 1
        if self._hand_down_run >= hand.min_gap_observations:
            self._hand_counted = False

    # -- the seated baseline -------------------------------------------------

    def _settle(self, shoulders: Point, place: PlaceRules) -> None:
        """Average the first `settle_observations` MEASURED frames, then freeze.

        A mean rather than the first frame, because the first frame of a track is the one
        where the box has just appeared and the keypoints are at their worst.

        **Its honest failure mode, which no code here can detect:** a child who is
        standing throughout the settling window is frozen with a standing baseline, and
        for the rest of the lesson their sitting down reads as normal and their standing
        reads as nothing. The report cannot tell this from a child who never stood, and
        neither can this function. It is the strongest argument for the whole package
        being read as counts rather than as conclusions.
        """
        if self.settled:
            return

        self._sum_x += shoulders[0]
        self._sum_y += shoulders[1]
        self._settle_samples += 1
        if self._settle_samples < place.settle_observations:
            return

        self.seat_y = self._sum_y / self._settle_samples
        self.seat_xy = (self._sum_x / self._settle_samples, self.seat_y)
        self.settled = True

    # -- standing, and being away from the place -----------------------------

    def _observe_place(self, shoulders: Point, at: float, place: PlaceRules) -> None:
        if not self.settled:
            return
        # shoulders[1] IS the shoulder line -- see `posture.anchor` on why there is not a
        # second function returning that same number.
        self._observe_stand(shoulders[1], place)
        self._observe_away(shoulders, at, place)

    def _observe_stand(self, shoulder_y: float, place: PlaceRules) -> None:
        """Symmetrically debounced: a stand both begins and ends only after it holds.

        The end matters as much as the start. Ending the moment one frame reads as seated
        would let a single dropped keypoint split one stand into two, and the metric the
        school asked for is «сколько раз встал» -- a count of occasions, which is exactly
        the quantity a flicker inflates.
        """
        if rose_from_seat(self.seat_y, shoulder_y, self.scale, place.rise_ratio):
            self._stand_down_run = 0
            self._stand_up_run += 1
            if self._stand_up_run >= place.min_hold_observations and not self._standing:
                self.stands += 1
                self._standing = True
            return

        self._stand_up_run = 0
        self._stand_down_run += 1
        if self._stand_down_run >= place.min_hold_observations:
            self._standing = False

    def _observe_away(self, point: Point, at: float, place: PlaceRules) -> None:
        """Time away is accrued in whole excursions, never per frame.

        Debounced by DURATION rather than by a frame count, because it is a duration that
        gets reported: an excursion is credited only once it ends, and only if it lasted
        `min_away_seconds`. Accruing per frame would mean tracker jitter around the
        threshold added real seconds to a child's total, a few frames at a time, with
        nothing to distinguish the result from a child who genuinely wandered.
        """
        if left_the_place(self.seat_xy, point, self.scale, place.away_ratio):
            if self._away_since is None:
                self._away_since = at
            return

        self._close_excursion(at, place)

    def _close_excursion(self, at: float, place: PlaceRules) -> None:
        if self._away_since is None:
            return
        spent = at - self._away_since
        self._away_since = None
        if spent >= place.min_away_seconds:
            self.away_seconds += spent
        else:
            self.brief_excursions += 1

    # -- finishing -----------------------------------------------------------

    def finalise(self, at: float) -> None:
        """Close an excursion still open when the track (or the lesson) ends.

        Without this, a child who walks out and does not come back before the bell has
        their entire absence dropped -- the one excursion most worth reporting is the one
        that never closes, and it would be the only kind that never counted.

        Idempotent, because it is called both when a track is retired and again when the
        lesson closes, and a flush may land between them.
        """
        if self._place is not None:
            self._close_excursion(at, self._place)
        self.last_seen = max(self.last_seen, at)

    @property
    def observed_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

"""Bounding the Unknown meal sessions. **Pure: no GPU, no wall clock, no database.**

`SessionManager` dedups by `person_id` — a cooldown, and a check that the pupil is not
already inside. **An Unknown session has no person_id**, so it skips all of that, and
nothing downstream can tell two Unknown sessions apart. Whatever bounds them has to do it
before `open()` is ever called, out of the only evidence there is: where the tracks were,
and when.

**Why it has to be bounded at all.** ByteTrack associates by IOU and a motion model. A long
occlusion at a busy door breaks the association and the same child is issued a NEW track id.
That id resolves independently, so it opens its own Unknown session: one child, two meal
records, the meal split across both, neither of them true. Six people in the school's own
roster hold two IDs each in exactly that shape.

**And the trade runs both ways.** Suppressing a real child's session is exactly as bad as
duplicating one: a duplicate is a record we can find and count, but a child suppressed ate
and is reported as having not, and there is nothing in the data to find them by. So every
rule here is written to fail OPEN, and the module's whole job is to be sure enough.

The clock is an argument. That is what makes "the same child, three seconds later, in the
same doorway" a unit test rather than a thing you find out about in a canteen.
"""

from __future__ import annotations

from dataclasses import dataclass

from qorgan.detection.geometry import Box, iou

# Two person boxes overlapping this much are, for our purposes, in the same place. IOU and
# not raw pixels, because a doorway is a place whatever the camera's zoom; and HIGH, because
# the cost of a false "same place" is a real child's meal session.
SAME_PLACE_IOU = 0.6

# How long a dead track's last sighting stays worth remembering. It only has to outlive the
# person cooldown (5 s) and the binding TTL (3 s); this is generous, and it bounds a dict
# that would otherwise grow all day.
FORGET_TRACK_AFTER_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Sighting:
    """Where and when a camera actually SAW a track. Frame clock, never wall clock."""

    first_seen: float
    last_seen: float
    box: Box

    # The BIGGEST this track ever looked, not the last. A child crossing away from the door
    # shrinks; a fast walker who dies while we are still RETRYING is judged on their final,
    # smallest box -- and refused `box_too_small`, which erases exactly the child the
    # farewell path exists to save. The gate's own words are "a figure at the far end of the
    # room, not a child at the door": someone who was ever child-at-the-door sized WAS at the
    # door.
    largest_area: float = 0.0


@dataclass(frozen=True, slots=True)
class UnknownOpen:
    """The last Unknown session a camera opened, and the track that caused it."""

    at: float
    track_id: int
    sighting: Sighting | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether this nameless track gets a meal record, and why. The reason is always logged.

    A refusal that cannot say why is indistinguishable from a bug, and this one refuses to
    record a child who may well have eaten.
    """

    allowed: bool
    reason: str
    previous_track_id: int | None = None
    seconds_since: float | None = None
    overlap: float | None = None
    box_area: float | None = None


def were_in_shot_together(a: Sighting, b: Sighting) -> bool:
    """Were these two tracks ever in the same frame? **Then they are two children.**

    ByteTrack never gives one person two track ids at the same moment. This is not a
    heuristic — it is the one thing about a pair of tracks we know for certain, and it is
    what keeps the child queuing behind the door from being mistaken for the child in it.
    """
    return a.first_seen <= b.last_seen and b.first_seen <= a.last_seen


class UnknownGuard:
    """One per entry camera. Remembers where the tracks were, and the last Unknown it opened.

    Not thread-safe, because one camera loop owns it.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Sighting] = {}
        self._last: UnknownOpen | None = None

    def note(self, people: dict[int, Box], now: float) -> None:
        """This is what the camera can see. Remember it.

        `first_seen` is what tells two children apart from one child under two track ids;
        `box` is what tells "the same doorway" from "further back in the queue". Neither can
        be recovered later, so both are recorded on every frame, whether we act or not.
        """
        for track_id, box in people.items():
            seen = self._tracks.get(track_id)
            first = seen.first_seen if seen is not None else now
            biggest = max(box.area, seen.largest_area if seen is not None else 0.0)
            self._tracks[track_id] = Sighting(
                first_seen=first, last_seen=now, box=box, largest_area=biggest
            )

        horizon = now - FORGET_TRACK_AFTER_SECONDS
        self._tracks = {
            track_id: seen for track_id, seen in self._tracks.items() if seen.last_seen >= horizon
        }

    def sighting(self, track_id: int) -> Sighting | None:
        return self._tracks.get(track_id)

    def opened(self, track_id: int, now: float) -> None:
        """An Unknown session was just opened for this track. It is the one to beat."""
        self._last = UnknownOpen(at=now, track_id=track_id, sighting=self._tracks.get(track_id))

    def allows(
        self, track_id: int, now: float, *, cooldown: float, min_box_area: float
    ) -> Verdict:
        """Should this nameless track get a meal session of its own? Two gates.

          * A person box below `min_box_area` is not a child at the door. It is a figure at
            the far end of the room, and a meal record made out of one is a hole in the
            register manufactured from nothing.

          * A SPLIT track does not get a second session (`_is_a_split`).

        **Any doubt opens the session.** No box in hand, no overlap, ever seen beside the
        last child — every one of those opens it.
        """
        seen = self._tracks.get(track_id)
        if seen is None:
            # We never saw a box for this track: a farewell for a child already gone. We have
            # nothing to judge them by, so we record them. That is the whole point of Unknown.
            return Verdict(True, "no_box_to_judge_by")

        if seen.largest_area < min_box_area:
            return Verdict(False, "box_too_small", box_area=seen.largest_area)

        return self._is_a_split(seen, cooldown, now)

    def _still_watching(self, last: UnknownOpen) -> Sighting:
        """The previous track as it is NOW, not as it was when its session opened.

        This is the bug that ate a child's meal record. `Sighting` is frozen and `note()`
        rebinds it every frame, so the object `opened()` stored never advances -- its
        `last_seen` is pinned to the instant the session opened. Asking THAT about
        co-visibility only ever asked "was this track already in shot when the last session
        opened?", and it was blind to a child who joins the queue one frame later, stands in
        plain sight beside the first, and steps into the doorway when they go in.

        That child is suppressed, eats, and is reported as not having eaten. The guard
        written to protect them was reading a photograph of the past.

        The live sighting outlives the cooldown by design: `FORGET_TRACK_AFTER_SECONDS` (30 s)
        against a 5 s window. The frozen snapshot is kept only as a fallback for a track the
        camera has already forgotten.
        """
        return self._tracks.get(last.track_id) or last.sighting  # type: ignore[return-value]

    def _is_a_split(self, seen: Sighting, cooldown: float, now: float) -> Verdict:
        """Is this the SAME child the last Unknown session was opened for, under a new id?

        The rule: **no second Unknown session within `cooldown` of the last, from a track
        that was never in shot BESIDE the track that opened it, standing in substantially the
        same place.** All three, or the session opens.

        **The coexistence test is what protects the real second child, and it is why this is
        not a global cooldown.** Two children in a queue are visible TOGETHER — the one behind
        is in frame while the one at the door is still being recognised — and ByteTrack never
        gives one person two ids at once. So two tracks ever seen in the same frame are two
        different children, always, and the one behind keeps their session however soon they
        step forward and wherever they choose to stand.

        **The failure mode, plainly.** A child who reaches the door only AFTER the previous
        track has died, within the cooldown, into substantially the same spot, and whom we
        also fail to recognise, is suppressed — and a child who ate lands on the "did not eat"
        report. All four must hold at once, which is narrow; every other case opens the
        session. It is a real cost, and it is why the window is 5 s and not 60.
        """
        last = self._last
        if last is None or last.sighting is None:
            return Verdict(True, "nothing_to_split_from")

        since = now - last.at
        if since >= cooldown:
            return Verdict(True, "cooldown_expired", seconds_since=since)

        previous = self._still_watching(last)

        if were_in_shot_together(seen, previous):
            # Seen side by side. Two children, and no cooldown may touch the second.
            return Verdict(True, "in_shot_together", previous_track_id=last.track_id)

        overlap = iou(seen.box, previous.box)
        if overlap < SAME_PLACE_IOU:
            return Verdict(True, "a_different_place", overlap=overlap)

        return Verdict(
            False,
            "split_track",
            previous_track_id=last.track_id,
            seconds_since=since,
            overlap=overlap,
        )

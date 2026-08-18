"""The live ledgers for one lesson: bounded, anonymous, and never the whole record.

This is RAM, and it is deliberately not where a lesson lives. The legacy kept canteen
sessions in a dict inside a module-global singleton, so restarting the process silently
lost every child who had walked in and not yet walked out. A lesson is 45 minutes; a
worker restart at minute 40 must not erase it. So `store.py` owns the record, this owns
only the last few seconds of arithmetic, and `flush_interval_seconds` bounds the gap
between them.

**Bounded three ways, because track ids only ever go up (rule R8).** Ledgers are fixed
size; a track unseen for `track_idle_seconds` is retired and removed; and the whole map
is capped at `max_tracks`, past which a new track is REFUSED and counted rather than
admitted. Refusing is the honest half: silently admitting the 81st track would grow the
dict, and silently dropping it without a count would let a report describe a room it had
stopped watching.

**Retirement is not departure.** A track goes idle when ByteTrack loses it, and it loses
people behind other people constantly in a room full of them. The child re-enters as a
NEW track id with a NEW ledger and a fresh seated baseline -- so one child can become
three rows in the report, each with part of their lesson. Nothing here can undo that
(undoing it means recognising the child, which §8 forbids in a classroom and the corridor
measurement says would not work anyway), so the numbers that reveal it are carried to the
report instead: `tracks` against a class size a human already knows, and the count of
fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qorgan.classroom.association import Assignment
from qorgan.classroom.ledger import TrackLedger
from qorgan.config.classroom import HandRaiseRules, LessonRules, PlaceRules


@dataclass(slots=True)
class Doubt:
    """Everything the lesson could not see, counted rather than smoothed over.

    Four counters, and none of them is derivable from another, which is why there are
    four. They fail differently and they are cured differently: ambiguity is a seating
    and camera-angle problem, unclaimed skeletons are a tracking problem, dropped tracks
    are a capacity problem, and a resume is a restart. A single "quality" number would
    average them into something that points at nothing.
    """

    ambiguous: int = 0
    unclaimed: int = 0
    dropped_tracks: int = 0


@dataclass(slots=True)
class LessonAccumulator:
    """The live ledgers, keyed by track id. Anonymous throughout."""

    rules: LessonRules
    doubt: Doubt = field(default_factory=Doubt)
    _ledgers: dict[int, TrackLedger] = field(default_factory=dict)

    def observe(
        self, assignment: Assignment, at: float, hand: HandRaiseRules, place: PlaceRules
    ) -> list[TrackLedger]:
        """Fold one frame in. Returns the ledgers that finished on this frame.

        The caller must persist what comes back: it has been finalised and removed, and
        this object will not mention it again.

        **Retirement happens BEFORE anybody is admitted, and the order is load-bearing.**
        It ran the other way round first, and the cap was then measured against ledgers
        that were about to be thrown away: in a room that turns over, every slot was held
        by a track last seen twenty minutes ago, so a genuinely new child was refused,
        counted as `dropped_tracks`, and the ledgers freed a microsecond later. The room
        emptied itself out of the report while the counter said the cap was doing its job.
        Caught by `test_a_retired_slot_lets_a_new_track_in`.
        """
        self.doubt.ambiguous += assignment.ambiguous
        self.doubt.unclaimed += assignment.unclaimed

        retired = self._retire(at, present=set(assignment.people))

        for track_id, person in assignment.people.items():
            ledger = self._ledger_for(track_id, at)
            if ledger is not None:
                ledger.observe(person, at, hand, place)

        return retired

    def _ledger_for(self, track_id: int, at: float) -> TrackLedger | None:
        """This track's ledger, opening one if the cap allows. `None` means refused."""
        existing = self._ledgers.get(track_id)
        if existing is not None:
            return existing

        if len(self._ledgers) >= self.rules.max_tracks:
            self.doubt.dropped_tracks += 1
            return None

        opened = TrackLedger(track_id=track_id, first_seen=at, last_seen=at)
        self._ledgers[track_id] = opened
        return opened

    def _retire(self, at: float, present: set[int]) -> list[TrackLedger]:
        """Finalise and remove tracks nobody has seen for `track_idle_seconds`.

        Finalised BEFORE removal, so that a child who walked out and never came back has
        their open excursion closed rather than silently discarded -- the absence most
        worth reporting is the one that does not end.

        **A track in THIS frame is never retired, however long the gap before it was.**
        A long occlusion that ByteTrack nevertheless re-associates comes back under the
        SAME id, and retiring it would open a second ledger with a fresh seated baseline
        and zeroed counters -- which then UPSERTS over the first row (the row is unique on
        lesson and track id, and `flush` assigns rather than adds), silently erasing
        everything the track had done before the gap.
        """
        idle = [
            ledger
            for ledger in self._ledgers.values()
            if ledger.track_id not in present
            and at - ledger.last_seen > self.rules.track_idle_seconds
        ]
        for ledger in idle:
            ledger.finalise(ledger.last_seen)
            del self._ledgers[ledger.track_id]
        return idle

    def live(self) -> list[TrackLedger]:
        """The ledgers still running. Written out on every flush, not only at the end:
        that is what makes a mid-lesson restart cost `flush_interval_seconds` of counting
        instead of the whole lesson."""
        return list(self._ledgers.values())

    def finish(self, at: float) -> list[TrackLedger]:
        """The bell. Finalise every remaining ledger and empty the map."""
        remaining = list(self._ledgers.values())
        for ledger in remaining:
            ledger.finalise(at)
        self._ledgers.clear()
        return remaining

"""Deduplicating one physical incident into one event.

A fight lasting four seconds produces a confirmed pair on dozens of consecutive frames.
Without merging, that is dozens of database rows and dozens of Telegram messages for one
shove -- which trains staff to ignore the alerts, which is worse than having none.

Three rules, in priority order. Each answers "is this the same incident?" differently:

  1. **Same pair** — the same two track ids, within 15 s. Strict, and always on.
  2. **Same scene** — any overlapping track id, within 8 s. Catches a fight where the
     tracker loses one child and re-acquires them with a new id: same incident, and the
     legacy would otherwise have logged it twice.
  3. **Same place** — the pair centres are close in the frame, within 8 s. Catches the
     case where the tracker loses BOTH ids -- a scuffle in the same corner of the hall,
     seconds apart, is one scuffle.

A merged event updates the original rather than creating a new one, and **raises no
second notification** — but "no second" is not "none". An incident is notified the first
time its confidence crosses the bar, and that is often not the first look at it: a shove
is ambiguous until the skeleton confirms, by which time every further judgement of it is
a merge. So the notification is a property of the INCIDENT, not of one judgement, and
this module is where it is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorgan.config.bullying import EventMerge

PairKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class MergeDecision:
    """Either this is a new incident, or it is one we already know about."""

    merged_into: PairKey | None
    reason: str

    @property
    def is_new(self) -> bool:
        return self.merged_into is None


NEW_INCIDENT = MergeDecision(None, "new")


@dataclass(slots=True)
class RecentEvent:
    key: PairKey
    timestamp: float
    confidence: float
    center: tuple[float, float]  # normalised 0..1
    event_id: int | None = None
    # Whether this incident already has a video clip on disk. Read by the retro-attach
    # path (spec §5.1's second ordering): a later look at an incident whose row was
    # written without a clip back-fills one, and must not then write a second.
    has_clip: bool = False
    # Whether a human has already been told about this incident. Lives here because this
    # dict is the only thing that knows which judgements are the same fight.
    notified: bool = False


class EventMerger:
    """Remembers recent events for one camera, so it can recognise a repeat."""

    def __init__(self, config: EventMerge) -> None:
        self._config = config
        self._recent: dict[PairKey, RecentEvent] = {}

    def decide(self, key: PairKey, timestamp: float, center: tuple[float, float]) -> MergeDecision:
        """Is this a new incident, or the same one we already recorded?"""
        self._evict(timestamp)

        # 1. The same two people. This rule holds even with merging disabled: it is the
        #    difference between "one alert per fight" and "one alert per frame".
        existing = self._recent.get(key)
        if existing is not None and timestamp - existing.timestamp <= self._same_pair_window:
            return MergeDecision(key, "same_pair")

        if not self._config.enabled:
            return NEW_INCIDENT

        for other_key, event in self._recent.items():
            if timestamp - event.timestamp > self._config.scene_window_seconds:
                continue

            # 2. The tracker lost one of them and gave them a new id. Same fight.
            if set(key) & set(other_key):
                return MergeDecision(other_key, "same_scene")

            # 3. It lost both. Same corner of the hall, seconds apart: same scuffle.
            if self._config.merge_by_spatial and self._close(center, event.center):
                return MergeDecision(other_key, "same_place")

        return NEW_INCIDENT

    def remember(
        self,
        key: PairKey,
        timestamp: float,
        confidence: float,
        center: tuple[float, float],
        event_id: int | None = None,
        has_clip: bool = False,
    ) -> None:
        previous = self._recent.get(key)
        self._recent[key] = RecentEvent(
            key=key,
            timestamp=timestamp,
            confidence=confidence,
            center=center,
            event_id=event_id,
            # Same rule as `notified`, and for the same reason: an incident that already
            # has a clip must not lose that fact and be given a second one.
            has_clip=has_clip or (previous is not None and previous.has_clip),
            # A fresh look at an incident we have already reported must not forget that we
            # reported it, or the next judgement would alert on the same fight again.
            # Anything old enough to be a genuinely NEW incident under these same two ids
            # has already been evicted by `decide`, so there is nothing to inherit from.
            notified=previous is not None and previous.notified,
        )

    def claim_notification(self, key: PairKey, wants_notify: bool) -> bool:
        """Is THIS the judgement that tells a human? True at most once per incident.

        The two ways to get this wrong are not symmetrical. Alerting on every judgement
        is the legacy's spam, which trains staff to ignore alerts. But deciding it on
        `is_new` alone means a fight whose opening frames were ambiguous is recorded at
        its true severity and reported to nobody — the row was written by the
        unconvincing first look, and every look that could have raised the alarm was
        merged into it. A fight in the database that nobody was told about is worse.
        """
        if not wants_notify:
            return False

        incident = self._recent.get(key)
        if incident is None:
            # No memory of this incident, so we cannot know it has been reported. Send:
            # one alert too many is a nuisance, a fight nobody hears about is the bug.
            return True
        if incident.notified:
            return False

        incident.notified = True
        return True

    def note_clip(self, key: PairKey) -> None:
        """This incident now has a clip on disk.

        Recorded against the merger's memory rather than the caller's, because the merger
        is the only thing that knows which judgements are one fight: a fight is judged
        dozens of times, and the second judgement must not write a second clip over the
        first. Deliberately called only AFTER `attach_media` has succeeded -- flagging an
        incident that has no file is how a row ends up pointing at nothing.
        """
        incident = self._recent.get(key)
        if incident is not None:
            incident.has_clip = True

    def get(self, key: PairKey) -> RecentEvent | None:
        return self._recent.get(key)

    def _close(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) < self._config.spatial_threshold

    @property
    def _same_pair_window(self) -> float:
        return self._config.same_pair_window_seconds

    def _evict(self, now: float) -> None:
        """Bounded (rule R8). A camera runs for months; this dict must not."""
        horizon = max(self._same_pair_window, self._config.scene_window_seconds)
        stale = [k for k, e in self._recent.items() if now - e.timestamp > horizon]
        for stale_key in stale:
            del self._recent[stale_key]

    def __len__(self) -> int:
        return len(self._recent)

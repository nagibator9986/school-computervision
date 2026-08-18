"""«Отслеживание несколько кадров» -- §12.1's second step, and the memory bound under it.

The weapons model runs `predict`, not `track`, so nothing upstream hands these sightings
an identity: Ultralytics' tracker keeps its state on the model object and is already spoken
for by the person detector, whose track ids are what «рядом с человеком» is measured
against. Borrowing it here would mean two trackers on one model object, which is the defect
`models/person.py` documents from the other side.

So association is done here, and it is deliberately dull: same class, and close to where
that class was last seen, scaled by the object's own size.

**Bounded twice, and the second bound is the one that holds** -- the rule
`events/clip_buffer.py` states and this module obeys:

  * **A TTL.** A track not seen for `track_idle_seconds` is finished and dropped. This is
    the ordinary bound and it is the one derived from configuration.
  * **A HARD CEILING on the number of tracks**, `max_tracks`, enforced whatever the TTL
    says. It is what holds when the first bound is wrong: a camera pointed at a display
    of cutlery, a model that fires on texture, a `track_idle_seconds` raised in YAML. A
    limit derived from configuration is not a limit, it is a hope about configuration.

Evictions over the ceiling are COUNTED and surfaced, not silently absorbed. A camera that
is evicting is a camera whose output means less than it appears to, and that is a fact for
a screen rather than a debug log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qorgan.detection.geometry import Box, distance
from qorgan.logging_setup import get_logger
from qorgan.weapons.model import Sighting

logger = get_logger(__name__)

# How far an object may move between two analyses and still be the same object, in
# multiples of its OWN box diagonal. CHOSEN: 3.0. Scaled by the object rather than fixed
# in pixels for the reason every distance in this project is -- a knife in the foreground
# is a bigger box AND moves through more pixels for the same real movement. At the shipped
# `analyse_every: 3` on a 15 fps stream that is 0.2 s between looks, in which a hand
# carrying something covers a fraction of a metre.
ASSOCIATION_RADIUS_DIAGONALS = 3.0


@dataclass(slots=True)
class WeaponTrack:
    """One object, seen more than once. **Not an alert.** See `weapons/pipeline.py`."""

    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    box: Box
    # Every observation that entered the track, i.e. everything above `model.conf`.
    observations: int = 0
    # Of those, the ones that also cleared `reconfirm_confidence`. §12.1's «повторное
    # подтверждение» is counted here rather than recomputed later, because a track is the
    # only thing that spans the frames the second gate is about.
    strong_observations: int = 0
    best_confidence: float = 0.0
    # The person this object was last seen beside, or None. Kept per track so that the
    # event can name the track ids the school's operator will look for in the clip.
    person_track_id: int | None = None
    # When a human was last asked about this track. §12.1 gives no answer for the second
    # look at one knife carried down one corridor, and the honest one is: the same
    # question, not a new alarm.
    alerted_at: float | None = None
    # Which zone rules have already examined it, for the log. A set so that a track that
    # crosses out of a kitchen carries both.
    zone_labels: set[str] = field(default_factory=set)

    def observe(self, sighting: Sighting, at: float, strong: bool) -> None:
        self.last_seen = at
        self.box = sighting.box
        self.observations += 1
        self.strong_observations += int(strong)
        self.best_confidence = max(self.best_confidence, sighting.confidence)

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


class WeaponTrackStore:
    """The live tracks for ONE camera, bounded by a TTL and by a hard ceiling."""

    def __init__(self, *, idle_seconds: float, max_tracks: int) -> None:
        self.idle_seconds = idle_seconds
        self.max_tracks = max_tracks
        # Counted, never merely dropped: this number is the difference between "the camera
        # saw nothing" and "the camera saw more than it can hold", and those look
        # identical from the outside.
        self.evicted_over_ceiling = 0

        self._tracks: dict[int, WeaponTrack] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._tracks)

    def observe(
        self, sighting: Sighting, at: float, *, strong: bool, person_track_id: int | None
    ) -> WeaponTrack:
        """Fold one screened sighting into a track, starting one if none matches."""
        track = self._match(sighting)
        if track is None:
            track = self._start(sighting, at)
        track.observe(sighting, at, strong)
        if person_track_id is not None:
            track.person_track_id = person_track_id
        return track

    def expire(self, now: float) -> list[WeaponTrack]:
        """Drop the tracks nobody has seen lately. Returns them, oldest first."""
        stale = [t for t in self._tracks.values() if now - t.last_seen > self.idle_seconds]
        for track in stale:
            del self._tracks[track.track_id]
        return sorted(stale, key=lambda t: t.last_seen)

    def _match(self, sighting: Sighting) -> WeaponTrack | None:
        """The nearest live track of the same class, within the association radius.

        Class first, and no cross-class association at all. A knife that becomes an axe
        between two frames is the model changing its mind, and merging the two would let
        a track accumulate the observations of one object under the name of another --
        which is how a bat and a ruler add up to an alert neither of them earned.
        """
        radius = ASSOCIATION_RADIUS_DIAGONALS * sighting.box.diagonal
        best: tuple[WeaponTrack, float] | None = None
        for track in self._tracks.values():
            if track.class_name != sighting.class_name:
                continue
            gap = distance(track.box.center, sighting.box.center)
            if gap <= radius and (best is None or gap < best[1]):
                best = (track, gap)
        return best[0] if best is not None else None

    def _start(self, sighting: Sighting, at: float) -> WeaponTrack:
        """A new track, making room for it first if the ceiling has been reached."""
        self._make_room()
        track = WeaponTrack(
            track_id=self._next_id,
            class_name=sighting.class_name,
            first_seen=at,
            last_seen=at,
            box=sighting.box,
        )
        self._next_id += 1
        self._tracks[track.track_id] = track
        return track

    def _make_room(self) -> None:
        """Enforce the hard ceiling. The least recently seen track goes.

        Least recently SEEN, not first created: the oldest track may be the one still
        being watched, and evicting it would restart the count on the only object that
        matters while keeping a stack of dead ones.
        """
        while len(self._tracks) >= self.max_tracks:
            oldest = min(self._tracks.values(), key=lambda t: t.last_seen)
            del self._tracks[oldest.track_id]
            self.evicted_over_ceiling += 1
            logger.warning(
                "weapon track evicted at the ceiling",
                extra={
                    "max_tracks": self.max_tracks,
                    "evicted_class": oldest.class_name,
                    "evicted_observations": oldest.observations,
                    "evicted_total": self.evicted_over_ceiling,
                    "consequence": (
                        "this camera is producing more simultaneous weapon tracks than "
                        "the ceiling allows, so some objects are no longer being counted "
                        "across frames and cannot reach the alert gate at all"
                    ),
                },
            )

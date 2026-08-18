"""§12.1's sequence, as one pure object. The worker drives it; a test drives the same one.

    объект обнаружен -> отслеживается несколько кадров -> проверка confidence
    -> проверка нахождения рядом с человеком -> повторное подтверждение
    -> snapshot и clip -> критическое уведомление

The client's shape is correct and is kept intact. The first five steps are here and are
decided without a model, a GPU, a database or a clock of their own; the last two are the
worker's, because they touch disk and the network (`qorgan/worker/weapons.py`).

**Never on one frame.** §12.1 says so in those words and it is enforced in three
independent places, which is deliberate rather than belt-and-braces: the schema refuses
`min_track_observations: 1` (`ge=2`), the schema refuses a reconfirmation gate that cannot
bite (`_never_a_single_frame`), and `_ready` below needs both counts. Any one of them
alone could be edited away by somebody who did not know why it was there.

Rule R2, applied here as it is to the bullying tier: there is exactly ONE of this object.
The eval-style smoke run over the school's corpus drives this class, not a copy of it, so
a negative result from that run is a statement about the code that would run in the
school.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from qorgan.config.weapons import WeaponsConfig
from qorgan.detection.geometry import Box
from qorgan.weapons.model import Sighting
from qorgan.weapons.rules import (
    NOT_NEAR_A_PERSON,
    Refused,
    nearest_person,
    screen_frame,
)
from qorgan.weapons.tracking import WeaponTrack, WeaponTrackStore

# What the module is claiming to have seen, as slugs on the event row. The same closed-set
# discipline `detection.validation.REASON_EVIDENCE` follows, and for the same reason: the
# legacy wrote this kind of thing as a sentence per call site and the same cause came out
# worded three ways, so none of them could be counted.
#
# There are three, they are the three gates §12.1 asks for, and each one is written only
# when that gate actually passed -- so the row says which of them the alert rests on
# rather than restating the configuration.
TRACKED_ACROSS_FRAMES = "weapon_tracked_across_frames"
NEAR_A_PERSON = "weapon_near_a_person"
RECONFIRMED = "weapon_reconfirmed"

EVIDENCE = (TRACKED_ACROSS_FRAMES, NEAR_A_PERSON, RECONFIRMED)


@dataclass(frozen=True, slots=True)
class WeaponAlert:
    """A track that cleared every gate. **Still not a finding.**

    It is a question put to a person: §12.1's alert is confirmed by a human and the record
    says who (`weapons/store.py`). Nothing downstream may treat one of these as a
    statement that a weapon is present, and the wording of every surface that renders it
    is written to that rule.
    """

    track_id: int
    class_name: str
    timestamp: float
    # The best single observation in the track, which is what the panel shows as "how
    # sure". Never an average: the question is whether the model was ever sure, and a mean
    # over a track that begins as the object comes into view answers a different one.
    confidence: float
    # The evidence behind the two counting gates, so the row can be audited against the
    # configuration that produced it rather than against today's YAML.
    observations: int
    strong_observations: int
    # Which person it was beside. The operator opens the clip looking for this track.
    person_track_id: int | None
    box: Box
    reasons: tuple[str, ...]


@dataclass
class FrameOutcome:
    """What one analysed frame produced. Refusals are part of the result, not debris."""

    alerts: list[WeaponAlert] = field(default_factory=list)
    refusals: list[Refused] = field(default_factory=list)
    tracked: int = 0

    @property
    def refused_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.refusals:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


class WeaponsDetector:
    """The decision, for ONE camera. Pure: no model, no disk, no database, no wall clock.

    `width` and `height` are the ANALYSED frame's, because every zone in this project is
    stored as a fraction of the frame and has to be turned back into pixels against the
    resolution the frames actually arrive at -- not the camera's advertised one.
    """

    def __init__(self, config: WeaponsConfig, width: int, height: int) -> None:
        self.config = config
        self.width = width
        self.height = height
        self.tracks = WeaponTrackStore(
            idle_seconds=config.track_idle_seconds, max_tracks=config.max_tracks
        )

    def process(
        self, sightings: Sequence[Sighting], people: Mapping[int, Box], now: float
    ) -> FrameOutcome:
        """One analysed frame: expire, screen, associate, gate. Never raises on bad input.

        **Expiry happens FIRST, before anything can be associated.** `WeaponTrackStore.
        _match` is spatial and has no clock of its own, so a track that has aged out has to
        be gone before a new sighting can join it. Expiring at the END instead — which is
        what this did — enforced `track_idle_seconds` only for as long as frames kept
        ARRIVING: every call dropped what was stale, so an object that went away during a
        running stream was correctly forgotten.

        A stream outage removes exactly that. Measured with the shipped gates: two
        observations, then no `process` call for 9.8 s, then one sighting in the same place
        — and the first frame back completed a three-observation track and alerted, with
        two of its three observations from before the outage and `track_idle_seconds` at
        1.5. RTSP reconnects are routine in this school, and «отслеживание несколько
        кадров» that can be satisfied by one fresh frame plus two from before a
        disconnection is not the guarantee §12.1 asks for. `tracking.py` already said, in
        words, that a track not seen for `track_idle_seconds` is dropped; now it is.
        """
        self.tracks.expire(now)
        kept, refusals = screen_frame(sightings, self.config, self.width, self.height)

        outcome = FrameOutcome(refusals=refusals)
        for screened in kept:
            track = self._observe(screened.sighting, people, now, outcome)
            if track is None:
                continue
            alert = self._maybe_alert(track, now)
            if alert is not None:
                outcome.alerts.append(alert)

        outcome.tracked = len(self.tracks)
        return outcome

    def _observe(
        self,
        sighting: Sighting,
        people: Mapping[int, Box],
        now: float,
        outcome: FrameOutcome,
    ) -> WeaponTrack | None:
        """§12.1's «проверка нахождения рядом с человеком», then fold it into a track.

        **The person check happens at the OBSERVATION and not at the alert gate**, so a
        sighting nowhere near anybody never enters a track at all. That is one place
        rather than two, and it means a track's observation count cannot include frames
        the check would have refused -- which is what would have to be true for the count
        to mean what the panel says it means. A knife on a poster accumulates nothing.
        """
        near = nearest_person(sighting, people, self.config.near_person_ratio)
        if near is None:
            outcome.refusals.append(
                Refused(sighting, NOT_NEAR_A_PERSON, f"{len(people)} people in frame")
            )
            return None

        person_track_id, _gap = near
        return self.tracks.observe(
            sighting,
            now,
            strong=sighting.confidence >= self.config.reconfirm_confidence,
            person_track_id=person_track_id,
        )

    def _maybe_alert(self, track: WeaponTrack, now: float) -> WeaponAlert | None:
        """The last two gates, and the quiet period after one has fired."""
        if not self._ready(track):
            return None
        if track.alerted_at is not None and now - track.alerted_at < self.config.realert_seconds:
            # One knife carried down one corridor is one question for a person, not one a
            # second. The track keeps accumulating; nobody is asked again yet.
            return None

        track.alerted_at = now
        return WeaponAlert(
            track_id=track.track_id,
            class_name=track.class_name,
            timestamp=now,
            confidence=track.best_confidence,
            observations=track.observations,
            strong_observations=track.strong_observations,
            person_track_id=track.person_track_id,
            box=track.box,
            # All three, and only because all three passed: `_ready` demands the two
            # counts and `_observe` is the only door into a track, so being here means the
            # person check passed on every observation this track holds.
            reasons=EVIDENCE,
        )

    def _ready(self, track: WeaponTrack) -> bool:
        """«Отслеживание несколько кадров» AND «повторное подтверждение», both.

        Two counts rather than one, because they are two different questions. The first is
        "did this object persist?", which rejects a single-frame flicker. The second is
        "was the model ever actually sure?", which rejects a shape it kept hedging about
        for a second and a half. A pen at the edge of the size gate can satisfy the first
        indefinitely and will never satisfy the second.
        """
        return (
            track.observations >= self.config.min_track_observations
            and track.strong_observations >= self.config.reconfirm_observations
        )

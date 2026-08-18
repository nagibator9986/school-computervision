"""Bind, retry, evict. **A pure state machine: no GPU, no clock, no database.**

The clock is an argument. That is what makes "a child who turns away for the whole queue
is still recognised after max_wait_seconds" a unit test rather than a thing you find out
about in a canteen.

  * Watch a track. Keep only the **best face seen so far** — one object, not a list (R8).
  * After `min_face_frames` observations OR `max_wait_seconds`, whichever comes first,
    the caller may spend one embedding.
  * Accepted => BOUND. Never recognised again.
  * Rejected => RETRYING, after a backoff, up to `max_attempts`. (This is where the
    small-face path lives: several weak looks at one child are worth more than one.)
  * Out of attempts => EXHAUSTED. **That is a verdict, not a non-event**: an unrecognised
    child still walked in, and somebody still has to open an Unknown session for them.
  * Track lost for `track_ttl_seconds` => evicted. **The next child to get that track id
    is a different child**, and a binding that outlives its track hands one pupil another
    pupil's identity.

**RETRYING is not a verdict.** A track can die in it — a child who walks through in two
seconds cannot fit three attempts at a one-second backoff inside a three-second TTL — and
a caller that simply forgets such a track has lost a child who really did eat. So
`expired()` names the doomed BEFORE `evict()` kills them, and the caller gets its last
look. See `IdentityService._farewells`.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import StrEnum

from qorgan.config.identity import BindingSettings
from qorgan.faces.matching import Recognition
from qorgan.faces.recognizer import FaceBox


class BindState(StrEnum):
    OBSERVING = "observing"  # watching, not yet worth an embedding
    BOUND = "bound"  # we know who this is. Done.
    RETRYING = "retrying"  # a look that was not good enough. Try again shortly.
    EXHAUSTED = "exhausted"  # out of attempts. Stop burning the GPU on this one.


# The two states in which a track's identity is DECIDED, and therefore the two in which a
# caller should act on it. RETRYING and OBSERVING are "not yet", not "no".
RESOLVED = (BindState.BOUND, BindState.EXHAUSTED)


@dataclass(frozen=True, slots=True)
class Binding:
    track_id: int
    state: BindState
    person_id: int | None
    score: float
    attempts: int
    observations: int
    best: FaceBox | None
    first_seen: float
    last_seen: float
    next_attempt_at: float

    @property
    def resolved(self) -> bool:
        return self.state in RESOLVED

    @property
    def ever_had_a_face(self) -> bool:
        """Did we ever have anything to recognise?

        `bind()` clears `best` on a rejection, so a RETRYING track has no face in hand —
        but it certainly had one, or we would not have spent an embedding on it. A track
        that never held a face at all is a back of a head, and it must not manufacture a
        meal session.
        """
        return self.best is not None or self.attempts > 0


class BindingTable:
    """One per camera. Not thread-safe, because one camera loop owns it."""

    def __init__(self, config: BindingSettings) -> None:
        self.config = config
        self._bindings: dict[int, Binding] = {}

    def __len__(self) -> int:
        return len(self._bindings)

    def get(self, track_id: int) -> Binding | None:
        return self._bindings.get(track_id)

    def person_for(self, track_id: int) -> int | None:
        binding = self._bindings.get(track_id)
        return binding.person_id if binding and binding.state is BindState.BOUND else None

    def observe(self, track_id: int, face: FaceBox | None, now: float) -> Binding:
        """This track is still here. Keep its best face; count the look."""
        current = self._bindings.get(track_id)
        if current is None:
            current = Binding(
                track_id=track_id,
                state=BindState.OBSERVING,
                person_id=None,
                score=0.0,
                attempts=0,
                observations=0,
                best=None,
                first_seen=now,
                last_seen=now,
                next_attempt_at=now,
            )

        best = current.best
        if face is not None and (best is None or face.quality > best.quality):
            best = face

        updated = replace(
            current,
            observations=current.observations + (1 if face is not None else 0),
            best=best,
            last_seen=now,
        )
        self._bindings[track_id] = updated
        return updated

    def should_embed(self, track_id: int, now: float) -> bool:
        """Is it worth spending one 512-d ArcFace embedding on this track right now?"""
        binding = self._bindings.get(track_id)
        if binding is None or binding.best is None:
            return False
        if binding.resolved:
            return False
        if now < binding.next_attempt_at:
            return False

        if binding.attempts > 0:
            # A RETRY, and the backoff above is its only gate. We have already decided
            # this track is worth an embedding -- we spent one on it. We are not deciding
            # that again, we are waiting for a BETTER LOOK, and the fresh face we are
            # holding is it. Re-applying `min_face_frames` here would mean a child who
            # failed once had to be seen `min_face_frames` more times before we would try
            # again, which for a queue that moves is never.
            return True

        enough_looks = binding.observations >= self.config.min_face_frames
        # A child who looks at the floor for the whole queue must still be recognised.
        waited_long_enough = (now - binding.first_seen) >= self.config.max_wait_seconds
        return enough_looks or waited_long_enough

    def bind(self, track_id: int, recognition: Recognition, now: float) -> Binding:
        """Apply the one recognition we spent an embedding on."""
        binding = self._bindings[track_id]
        attempts = binding.attempts + 1

        if recognition.accepted:
            updated = replace(
                binding,
                state=BindState.BOUND,
                person_id=recognition.person_id,
                score=recognition.score,
                attempts=attempts,
                last_seen=now,
            )
        else:
            exhausted = attempts >= self.config.max_attempts
            updated = replace(
                binding,
                state=BindState.EXHAUSTED if exhausted else BindState.RETRYING,
                person_id=None,
                score=recognition.score,
                attempts=attempts,
                # Start the next look from scratch: the face that failed is not the face
                # that will succeed.
                observations=0,
                best=None,
                next_attempt_at=now + self.config.retry_backoff_seconds,
                last_seen=now,
            )

        self._bindings[track_id] = updated
        return updated

    def expired(self, live: Collection[int], now: float) -> list[int]:
        """Which tracks are gone? **Names them; does not kill them.**

        The caller gets to look at a dying track before it is forgotten, because a track
        that dies while still RETRYING is a child who walked in and was never recognised.
        Evicting silently is how that child ends up with no record at all.
        """
        cutoff = now - self.config.track_ttl_seconds
        return [
            track_id
            for track_id, binding in self._bindings.items()
            if track_id not in live and binding.last_seen < cutoff
        ]

    def evict(self, live: Collection[int], now: float) -> list[int]:
        """Forget tracks that have been gone longer than the TTL.

        A short absence is a flicker -- a head turns, YOLO loses a frame -- and evicting on
        it would re-embed the same child every second. A long one is a different child
        walking into a recycled track id, and NOT evicting on that is how one pupil is
        recorded eating another pupil's lunch.
        """
        gone = self.expired(live, now)
        for track_id in gone:
            del self._bindings[track_id]
        return gone

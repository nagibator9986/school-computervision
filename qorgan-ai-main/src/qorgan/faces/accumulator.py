"""The small-face path: accept a weaker match if the SAME child keeps coming top.

**Keep this. It is domain knowledge, not a hack.**

Younger pupils are shorter, so their faces land further from the camera and are
systematically smaller than the size gate. A strict single-shot threshold therefore does
not merely recognise them less well — it never recognises the first-graders *at all*.
The school notices this before the engineers do.

The answer the legacy arrived at is sound: if the same person comes out top-1 repeatedly
inside a short window, accept them at a lower score than a single glance would need.
Several weak looks at one child are worth more than one weak look.

What is different here: the legacy wrote this same logic out **four separate times** with
four sets of key names (`soft_recognition_*`, `exit_soft_*`, `exit_candidate_*_soft`,
`entry_small_face_*`), which then drifted apart. One implementation, one config model.

**And the bug this design kills on the way past.** `hits` used to be keyed by `person_id`,
with one accumulator shared across the whole camera. So weak top-1 hits from DIFFERENT
children corroborated each other: a crowd of unknowns at the door could vote a stranger
into being pupil X — and that closes a meal session. The class was called
`TrackAccumulator` and had never known what a track was. Now it does, and the key is
`(track_id, person_id)` (spec §4.5).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from qorgan.config.identity import SoftAccumulator as SoftConfig
from qorgan.faces.matching import Reason, Recognition

Key = tuple[int, int]  # (track_id, person_id)


@dataclass(frozen=True, slots=True)
class Hit:
    track_id: int
    person_id: int
    score: float
    gap: float
    at: float


@dataclass
class TrackAccumulator:
    """Repeated looks at ONE track. Two tracks are two children, and they do not vote
    together."""

    config: SoftConfig
    hits: dict[Key, deque[Hit]] = field(default_factory=dict)

    def observe(self, recognition: Recognition, now: float, track_id: int) -> int | None:
        """Record a look at ONE track. Returns a person id once we have seen enough.

        Only the top-1 counts, and only if it is at least *plausible* — a face that
        matched nobody tells us nothing, and letting it accumulate would eventually
        "recognise" a child out of pure noise.
        """
        if not self.config.enabled:
            return None

        top = recognition.top1
        if top is None:
            return None
        if top.score < self.config.min_score or recognition.gap < self.config.min_gap:
            return None

        window = self.hits.setdefault((track_id, top.person_id), deque())
        window.append(Hit(track_id, top.person_id, top.score, recognition.gap, now))
        self._expire(window, now)

        if len(window) >= self.config.min_hits:
            return top.person_id
        return None

    def _expire(self, window: deque[Hit], now: float) -> None:
        """Hits older than the window are gone. Two glances a minute apart are not
        corroboration; they are two separate children walking past."""
        cutoff = now - self.config.window_seconds
        while window and window[0].at < cutoff:
            window.popleft()

    def evidence(self, track_id: int, person_id: int) -> int:
        return len(self.hits.get((track_id, person_id), ()))

    def clear(self, track_id: int | None = None) -> None:
        """Forget one track, or everything. A track that has ended has no more to say."""
        if track_id is None:
            self.hits.clear()
            return
        for key in [key for key in self.hits if key[0] == track_id]:
            del self.hits[key]

    def prune(self, now: float) -> None:
        """Bounded (rule R8). Track ids only ever increase and a canteen runs all year."""
        for key in list(self.hits):
            window = self.hits[key]
            self._expire(window, now)
            if not window:
                del self.hits[key]


def accept_small_face(
    recognition: Recognition,
    accumulator: TrackAccumulator,
    now: float,
    track_id: int,
) -> Recognition:
    """Give a rejected recognition a second chance via accumulated evidence — **from this
    track, and only this track.**

    Only ever *upgrades* a rejection, and never touches an acceptance — a face the strict
    policy already accepted does not need help, and a face it rejected as AMBIGUOUS is
    rejected for a reason the accumulator cannot cure: seeing two children who look alike
    five times running does not tell you which one it was.
    """
    if recognition.accepted:
        return recognition
    if recognition.reason is Reason.AMBIGUOUS:
        return recognition

    person_id = accumulator.observe(recognition, now, track_id)
    if person_id is None:
        return recognition

    top = recognition.top1
    return Recognition(
        person_id=person_id,
        score=top.score if top else 0.0,
        gap=recognition.gap,
        reason=Reason.ACCEPTED,
        ranked=recognition.ranked,
    )

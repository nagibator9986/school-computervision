"""The impure shell: recognizer + gallery + bindings. Everything it DECIDES is pure.

    detect_faces (27 ms -- the EXPENSIVE half; skipped once every track is resolved)
      -> assign_faces_to_tracks   (pure geometry)
      -> BindingTable             (pure state machine)
      -> embed + identify         (10 ms, ONCE per track)

For five children queuing over ten seconds: **5 embeddings instead of ~200** (spec §4.4).

**Two invariants, and they are not the same one.** Conflating them is how the first draft
of this module rebuilt the 1 816-NULL bug inside the module written to prevent it:

  * the **GPU** invariant -- one track costs ONE embedding, not one per frame. That is
    `should_embed`, and it is what the task is nominally about.

  * the **RECORD** invariant -- every track that ever held a face is resolved EXACTLY
    ONCE, and a track we never managed to recognise still resolves, as UNKNOWN. A child
    who walks through in two seconds and whose track dies while we are still RETRYING is
    still a child who walked in. If they resolve to nothing they ate, left no record, and
    are then missing from the "did not eat" report -- the one report the school asked for.

So `should_act` is True on exactly one frame per track, and it means **decided**, not
**recognised**. `BOUND` decides with a name; `EXHAUSTED` decides without one; a track that
dies unresolved decides without one on the way out (`_farewells`). `RETRYING` decides
nothing -- it is "not yet", not "no". A hole we can count beats a child who silently never
ate.

`embedded` is separate again, and deliberately so: it is True on every frame we spend an
embedding, which is 1..max_attempts times per track. That is what feeds
`RecognitionAttempt` -- **the instrument that measures the unmeasured `min_score`
ceiling.** A calibration table that records only its successes measures nothing.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
from qorgan.detection.geometry import Box
from qorgan.faces.accumulator import TrackAccumulator, accept_small_face
from qorgan.faces.gallery import GalleryCache, PersonInfo
from qorgan.faces.matching import Reason, Recognition, identify
from qorgan.faces.recognizer import FaceBox, FaceRecognizer
from qorgan.identity.binding import Binding, BindingTable, BindState
from qorgan.identity.tracks import assign_faces_to_tracks
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Identified:
    """One track, this frame."""

    track_id: int
    state: BindState
    person_id: int | None
    person: PersonInfo | None
    recognition: Recognition
    face: FaceBox | None

    # True on every frame we spent an embedding on this track (1..max_attempts times).
    # Record a RecognitionAttempt on each -- including the failures, which are the only
    # rows that can tell us WHY recognition failed.
    embedded: bool = False

    # True on the ONE frame this track's identity was DECIDED, and never again. Decided
    # is not the same as recognised: an EXHAUSTED track and a track lost while still
    # retrying are both decided, both anonymous, and both still walked in.
    should_act: bool = False

    @property
    def is_staff(self) -> bool:
        return self.person is not None and self.person.is_staff


class IdentityService:
    """One per camera. Owns that camera's bindings; shares the recognizer and gallery."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        gallery: GalleryCache,
        policy: RecognitionPolicy,
        binding: BindingSettings,
        *,
        soft: SoftAccumulator | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._gallery = gallery
        self._policy = policy
        self._table = BindingTable(binding)
        self._soft = TrackAccumulator(soft) if soft else None

    def state_of(self, track_id: int) -> BindState | None:
        """This track's state, or None if we have never heard of it (or have forgotten)."""
        binding = self._table.get(track_id)
        return binding.state if binding else None

    def on_frame(
        self, image: np.ndarray, person_boxes: dict[int, Box], now: float
    ) -> list[Identified]:
        """Detect a face only for a track that still needs one; embed once per track, ever.

        This module was built on "the expensive half is the 512-d ArcFace embedding". That
        was wrong, and the real model says so:

            detect_faces  27.4 ms
            embed          9.7 ms

        Detection costs nearly THREE TIMES the embedding. So "200 embeddings become 5" was
        true and misleading -- the dominant cost was running unconditionally on every frame
        and never went away.

        Detection exists to find a face for a track that still needs one. Once every track in
        shot is resolved, there is no face left to look for, and the frame costs nothing.
        """
        wanted = self._needs_a_face(person_boxes, now)
        faces = (
            assign_faces_to_tracks(self._recognizer.detect_faces(image), person_boxes)
            if wanted
            else {}
        )

        for track_id in person_boxes:
            self._table.observe(track_id, faces.get(track_id), now)

        # Every track in shot, not only the ones a face was found for. A BOUND track must
        # still report who it is on every frame -- the pipeline needs to know who is in
        # view -- and reporting an identity needs no face. Only ACQUIRING one does, which is
        # why detection is skipped above while this is not.
        found = [
            seen
            for track_id in person_boxes
            if (seen := self._resolve(track_id, faces.get(track_id), image, now)) is not None
        ]
        # Tracks that have walked out of shot get their last word here, and it is the only
        # chance an unresolved one will ever have to be recorded at all.
        found.extend(self._farewells(person_boxes.keys(), now))
        return found

    def _needs_a_face(self, person_boxes: dict[int, Box], now: float) -> bool:
        """Is there any track in shot that a face would still tell us something about?

        A track we have never seen before needs one. A track still OBSERVING or RETRYING
        needs one. A track that is BOUND or EXHAUSTED does not -- its answer is final, and
        looking again cannot change it.

        `now` is unused today and is taken deliberately: a future retry policy that reopens
        a resolved track on a timer has to decide that HERE, where the detector is paid for,
        and not somewhere the cost is invisible.
        """
        for track_id in person_boxes:
            binding = self._table.get(track_id)
            if binding is None or not binding.resolved:
                return True
        return False

    def _resolve(
        self, track_id: int, face: FaceBox | None, image: np.ndarray, now: float
    ) -> Identified | None:
        """Report a bound track; spend an embedding on one that is ready for it."""
        if not self._table.should_embed(track_id, now):
            return self._already_bound(track_id, face)

        # THE expensive call. Once per track, or once per attempt if we keep failing.
        best = self._table.get(track_id).best or face
        gallery = self._gallery.get()
        recognition = identify(
            self._recognizer.embed(image, best), gallery.matrix, gallery.person_ids, self._policy
        )
        recognition = self._soften(recognition, best, now, track_id)

        binding = self._table.bind(track_id, recognition, now)
        self._log(binding, recognition)

        return Identified(
            track_id=track_id,
            state=binding.state,
            person_id=binding.person_id,
            person=gallery.info(binding.person_id) if binding.person_id else None,
            recognition=recognition,
            face=best,
            embedded=True,
            # BOUND: decided, with a name. EXHAUSTED: decided, without one -- and an
            # anonymous child still ate. RETRYING: not yet.
            should_act=binding.resolved,
        )

    def _log(self, binding: Binding, recognition: Recognition) -> None:
        if binding.state is BindState.BOUND:
            logger.info(
                "track bound",
                extra={
                    "track_id": binding.track_id,
                    "person_id": binding.person_id,
                    "score": round(recognition.score, 3),
                    "attempts": binding.attempts,
                },
            )
        elif binding.state is BindState.EXHAUSTED:
            logger.info(
                "track gave up on recognition; it will be recorded as Unknown",
                extra={
                    "track_id": binding.track_id,
                    "score": round(recognition.score, 3),
                    "attempts": binding.attempts,
                    "reason": recognition.reason.value,
                },
            )

    def _already_bound(self, track_id: int, face: FaceBox) -> Identified | None:
        """A track whose identity we already know. Never acts again, never embeds again."""
        person_id = self._table.person_for(track_id)
        if person_id is None:
            return None

        binding = self._table.get(track_id)
        gallery = self._gallery.get()
        return Identified(
            track_id=track_id,
            state=binding.state,
            person_id=person_id,
            person=gallery.info(person_id),
            recognition=Recognition(person_id, binding.score, 0.0, Reason.ACCEPTED),
            face=face,
        )

    # -- the last word --------------------------------------------------------

    def _farewells(self, live: Collection[int], now: float) -> list[Identified]:
        """The tracks that have gone. **An unresolved one still gets to be counted.**

        A track that dies OBSERVING or RETRYING never reached a verdict, and it is the
        fast walker: three attempts at a one-second backoff do not fit inside a
        three-second TTL. Forgetting it silently means a child walked in, ate, and left no
        record -- the exact hole the meal report exists to close.
        """
        doomed = self._table.expired(live, now)
        leaving = [
            binding for binding in (self._table.get(track_id) for track_id in doomed) if binding
        ]
        self.evict(live, now)

        return [
            self._unresolved(binding)
            for binding in leaving
            if not binding.resolved and binding.ever_had_a_face
        ]

    def _unresolved(self, binding: Binding) -> Identified:
        """A child we never managed to name. They still walked in."""
        logger.info(
            "track lost before it could be recognised; recording it as Unknown",
            extra={
                "track_id": binding.track_id,
                "state": binding.state.value,
                "attempts": binding.attempts,
            },
        )
        return Identified(
            track_id=binding.track_id,
            state=binding.state,
            person_id=None,
            person=None,
            recognition=Recognition(None, binding.score, 0.0, Reason.UNRESOLVED),
            face=binding.best,
            embedded=False,
            should_act=True,
        )

    # -- the small-face path ---------------------------------------------------

    def _soften(
        self, recognition: Recognition, face: FaceBox, now: float, track_id: int
    ) -> Recognition:
        """The small-face path. Younger pupils' faces are systematically below the size
        gate, so a strict single-shot threshold never recognises the first-graders at all.

        Keyed by TRACK: several weak looks at ONE child are worth more than one weak look.
        Several weak looks at a CROWD are worth nothing at all, and used to be worth a
        meal session (spec §4.5).
        """
        if self._soft is None:
            return recognition
        if not self._soft.config.face_gate.accepts(face.width, face.height):
            return recognition
        return accept_small_face(recognition, self._soft, now, track_id)

    def evict(self, live: Collection[int], now: float) -> list[int]:
        gone = self._table.evict(live, now)
        if self._soft is not None:
            # A dead track's evidence dies with it. Leaving it behind would let a child
            # who has walked away go on corroborating the next child to be given their
            # track id.
            for track_id in gone:
                self._soft.clear(track_id)
            self._soft.prune(now)
        return gone

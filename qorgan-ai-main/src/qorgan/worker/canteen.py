"""The canteen pipeline for one camera. What it does depends entirely on its role.

    entry   — the ONLY camera that opens a meal session
    exit    — the ONLY camera that closes one
    inside  — confirms presence, may late-bind an identity. Never opens, never closes.

That division is the whole design, and it is what stops a child being recorded as having
left the moment they arrive. The legacy's canteen worker was a single 6 330-line function
with 1 074 local variables and six nested closures, in which all of this was interleaved.

Everything decided here is decided by pure functions in `qorgan.faces` and
`qorgan.canteen.sessions`. This module is plumbing.

**The role decides the CADENCE too, and that is not a tuning knob — it is the record.**

`PersonDetector.detect` is YOLO *and* ByteTrack, and ByteTrack associates by IOU and a
motion model: it needs frame-to-frame continuity or it does not work at all. Behind the
recognition gate it was being asked to associate across 0.25 s at the door and 1.5 s
inside, and at 1.5 s a child crosses the room between looks. Association fails and they are
issued a NEW track id. Not occasionally — structurally.

A new track id resolves independently, so it opens its own meal session: **one child, two
Unknown sessions, the meal split across both, neither record true.** Six people in the
school's own roster hold two IDs each in exactly that shape. We found it in their data and
we were one release from manufacturing it in ours.

So entry and exit track on EVERY frame, because a session RECORD depends on the track. An
inside camera never opens and never closes anything, so a duplicate track there is a
duplicate *confirmation* and no record at all — it stays on its interval, and paying 8 Hz
of YOLO for it would buy nothing. Measured, four canteen cameras, ms of GPU per second of
wall clock: 650.6 before, 553.6 tracking everything, **~300 tracking the doors only.**

The FACE work is throttled by `IdentityService._needs_a_face`, not by a gate here: it stops
looking for a face once every track in shot is BOUND or EXHAUSTED, which is why tracking
every frame is affordable. A blanket throttle over it would undo that.
"""

from __future__ import annotations

from qorgan.canteen.sessions import SESSION_TOO_YOUNG, SessionManager
from qorgan.canteen.unknowns import UnknownGuard
from qorgan.capture import Frame
from qorgan.config.camera import CanteenCamera
from qorgan.config.identity import BindingSettings, RecognitionPolicy, SoftAccumulator
from qorgan.db.engine import session_scope, with_retry
from qorgan.db.models import RecognitionAttempt
from qorgan.db.types import utcnow
from qorgan.enums import CameraRole
from qorgan.identity.service import Identified, IdentityService
from qorgan.logging_setup import get_logger
from qorgan.models.person import PersonDetector

logger = get_logger(__name__)

# The two roles whose act is a RECORD, and which therefore cannot tolerate a broken track.
TRACKS_EVERY_FRAME = (CameraRole.CANTEEN_ENTRY, CameraRole.CANTEEN_EXIT)


class CanteenPipeline:
    """One camera's worth of canteen logic."""

    def __init__(
        self,
        camera: CanteenCamera,
        camera_id: int,
        person: PersonDetector,
        identity: IdentityService,
        sessions: SessionManager,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self._person = person
        self._identity = identity
        self._sessions = sessions
        # None, not 0.0. A monotonic clock starts near zero, so a 0.0 sentinel would make
        # the camera skip its own first frames -- exactly when a queue is forming at the
        # door.
        self._last_scan: float | None = None
        # An Unknown session has no person_id, so `SessionManager` has nothing to dedup it
        # by. This is what bounds them; the tracks' boxes are all the evidence there is.
        self._unknowns = UnknownGuard()
        # Exit only: tracks whose close was DECIDED and then transiently REFUSED, and the
        # frame time each may next be re-attempted at.
        self._retry_close_at: dict[int, float] = {}

    # -- the frame loop ------------------------------------------------------

    def on_frame(self, _camera: CanteenCamera, frame: Frame) -> str:
        """Track the people, recognise each ONCE, and act according to our role.

        **Three different lifetimes, and conflating any two of them corrupts a record.**

          * `embedded` is true on every frame we spent an embedding -- 1..max_attempts
            times per track. Each one is a RecognitionAttempt row, **including the
            failures**. That table is the instrument that measures the unmeasured
            `min_score` ceiling, and a calibration table of successes only measures
            nothing.

          * `should_act` is true on exactly ONE frame per track: the frame its identity
            was DECIDED. Decided is not the same as recognised -- a track we gave up on,
            and a track that walked out of shot before we could name it, are both decided
            and both anonymous, and both of those children still walked in.

          * **DECIDED is not DONE.** Entry's act is terminal -- `open()` cannot fail
            transiently -- but the exit's is an ATTEMPT: `close()` refuses a session that
            is not yet old enough. A refused act is re-queued and retried while the track
            lives (`_retrying`), and at most one of those attempts ever succeeds.

        The person tracker runs on every frame at the doors and on the interval inside. See
        the module docstring: it is the difference between one meal record and two.
        """
        now = frame.captured_at
        if self.camera.role not in TRACKS_EVERY_FRAME and not self._due(now):
            return "ok"

        people = self._person.detect(frame.image)
        self._unknowns.note(people, now)

        acted = False
        for found in self._identity.on_frame(frame.image, people, now):
            if found.embedded:
                self._record_attempt(found)
            if found.should_act or self._retrying(found, now):
                acted |= self._act(found, now)

        self._forget_dead_tracks()
        return "alert" if acted else "ok"

    def _due(self, now: float) -> bool:
        """The INSIDE cameras' gate, and theirs alone.

        It used to gate the person tracker too, on every role -- which asked ByteTrack to
        associate across gaps it cannot associate across, and split one child into two meal
        records. Entry and exit no longer come through here at all.
        """
        if self._last_scan is not None and now - self._last_scan < self._interval():
            return False
        self._last_scan = now
        return True

    def _retrying(self, found: Identified, now: float) -> bool:
        """Is this track owed another attempt at a close that was refused, not done?"""
        at = self._retry_close_at.get(found.track_id)
        return at is not None and now >= at

    def _forget_dead_tracks(self) -> None:
        """Stop retrying a close for a child who has left.

        "Retry while the track LIVES" -- and the identity service is what knows when a track
        stops living, because it owns the TTL that decides it. A pending close whose track
        has been evicted is a child who walked away, and we do not get to close them.
        """
        self._retry_close_at = {
            track_id: at
            for track_id, at in self._retry_close_at.items()
            if self._identity.state_of(track_id) is not None
        }

    # -- acting ---------------------------------------------------------------

    def _act(self, found: Identified, now: float) -> bool:
        role = self.camera.role
        if role is CameraRole.CANTEEN_ENTRY:
            return self._on_entry(found, now)
        if role is CameraRole.CANTEEN_EXIT:
            return self._on_exit(found, now)
        return self._on_inside(found)

    def _on_entry(self, found: Identified, now: float) -> bool:
        """The only camera that opens a session.

        **It opens one for a child we could not name, too**, and that is not a detail.
        `person_id` is None for a track we gave up on and for a track that walked out of
        shot before we could look at it properly -- and both of those children ate. An
        Unknown session is a hole we can count; a child with no session at all is a child
        who silently never appears on the "did not eat" report.
        """
        if found.is_staff:
            # Staff eat too, but they do not get a meal record. Counting a cook as a
            # pupil is how they end up on the "did not eat" report.
            return False

        person_id = found.person_id
        if person_id is None and not self._unknown_is_worth_a_record(found, now):
            return False

        result = self._sessions.open(person_id, utcnow())
        if result.opened:
            if person_id is None:
                self._unknowns.opened(found.track_id, now)
            logger.info(
                "meal session opened",
                extra={
                    "camera": self.camera.name,
                    "person_id": person_id,
                    "track_id": found.track_id,
                    "reason": found.recognition.reason.value,
                    "score": round(found.recognition.score, 3),
                    "session_id": result.session_id,
                },
            )
        return result.opened

    def _unknown_is_worth_a_record(self, found: Identified, now: float) -> bool:
        """Should this NAMELESS track get a meal session of its own?

        The decision is pure and it lives in `canteen.unknowns`. This is the plumbing: it
        hands over the two config keys that bound it, and it LOGS the refusal. A child we
        decline to record is the one thing here that must never happen quietly.
        """
        entry = self.camera.canteen.entry
        if entry is None:
            return True

        verdict = self._unknowns.allows(
            found.track_id,
            now,
            cooldown=entry.person_cooldown_seconds,
            min_box_area=entry.min_person_box_area,
        )
        if verdict.allowed:
            return True

        logger.warning(
            "no Unknown meal session was opened for a track we could not name",
            extra={
                "camera": self.camera.name,
                "track_id": found.track_id,
                "reason": verdict.reason,
                "previous_track_id": verdict.previous_track_id,
                "seconds_since": verdict.seconds_since,
                "overlap": verdict.overlap,
                "box_area": verdict.box_area,
            },
        )
        return False

    # -- the exit --------------------------------------------------------------

    def _on_exit(self, found: Identified, now: float) -> bool:
        """The only camera that closes one. **An ATTEMPT, not an act.**

        Entry's `open()` is terminal: decided once, and done. `close()` is not -- it REFUSES
        a session younger than `exit_min_session_age_seconds`, because the exit camera is
        pointed at the backs of heads and the back it most often sees is that of a child who
        has just walked IN.

        `should_act` fires exactly once per track and it means DECIDED, not DONE. A child
        recognised at the exit while their session is still young would otherwise spend their
        single attempt on a "not yet" -- and if they then linger in shot past 30 s, they never
        get another go. Their session silently never closes and force-closes as UNKNOWN 90
        minutes later. So a TRANSIENT refusal is re-queued and retried for as long as the
        track lives; a success, or any other refusal, ends it. **At most one close per track.**
        """
        if not found.recognition.accepted or found.is_staff:
            self._retry_close_at.pop(found.track_id, None)
            return False

        person_id = found.person_id
        result = self._sessions.close(person_id, utcnow())

        if result.closed:
            self._retry_close_at.pop(found.track_id, None)
            logger.info(
                "meal session closed",
                extra={
                    "camera": self.camera.name,
                    "person_id": person_id,
                    "outcome": result.outcome.value if result.outcome else None,
                    "dwell_seconds": round(result.dwell_seconds or 0.0, 1),
                },
            )
            return True

        if result.reason == SESSION_TOO_YOUNG:
            # Not "no". "Not yet." Come back while this child is still in front of us.
            self._retry_close_at[found.track_id] = now + self._close_retry_interval()
        else:
            self._retry_close_at.pop(found.track_id, None)
        return False

    def _on_inside(self, found: Identified) -> bool:
        """Confirms presence. Never opens, never closes.

        **On late-binding, and why it is deliberately not done here.**

        The obvious move is: we have recognised a pupil inside the canteen who has no
        open session, and there are Unknown sessions lying about — so attach them to one.
        That is exactly what the legacy's `resolve_exit_session` did, and it attached the
        recognised pupil to somebody *else's* oldest Unknown session, handing them another
        child's dwell time and meal status (spec §5.2).

        The reason it cannot work is simple: an inside camera sees a face, not a journey.
        Nothing links this child to any particular Unknown session, so picking one is a
        guess dressed up as data. A binding that is wrong is worse than no binding — the
        wrong child is recorded as having eaten, and the right one appears on the "did not
        eat" report.

        So a pupil seen inside with no session is LOGGED, loudly, as what it actually is:
        evidence that the entry camera missed someone. That is a fixable problem. A
        corrupted meal record is not.
        """
        if not found.recognition.accepted or found.is_staff:
            return False

        person_id = found.person_id
        session_id = self._sessions.active_session_id(person_id)

        if session_id is None:
            logger.warning(
                "a pupil is inside the canteen with no open session — the entry camera missed them",
                extra={
                    "camera": self.camera.name,
                    "person_id": person_id,
                    "score": round(found.recognition.score, 3),
                },
            )
            return False

        return self._sessions.confirm_inside(session_id)

    # -- observability -------------------------------------------------------

    def _record_attempt(self, found: Identified) -> None:
        """Every recognition decision, kept for calibration.

        This is the data the legacy never had. It tuned eighteen overlapping thresholds
        by feel, and 1816 of its 1820 canteen records ended up with student_id = NULL —
        with no record of WHY any of them failed.

        **Written on every EMBEDDING, not on every binding.** A track that fails three
        times writes three rows, all with `accepted = False` and a real reason. Gating
        this on success would leave a table in which `accepted` is always True — which is
        not calibration data, it is a trophy cabinet, and `min_score`'s ceiling is still
        unmeasured (config/identity.py). It is also ~3 rows per child rather than the ~200
        the old worker wrote, which is the actual point of the task.
        """
        recognition = found.recognition
        top1 = recognition.top1
        second = recognition.ranked[1] if len(recognition.ranked) > 1 else None
        face = found.face

        def _write() -> None:
            with session_scope() as session:
                session.add(
                    RecognitionAttempt(
                        camera_id=self.camera_id,
                        occurred_at=utcnow(),
                        top1_person_id=top1.person_id if top1 else None,
                        top1_score=top1.score if top1 else None,
                        top2_score=second.score if second else None,
                        gap=recognition.gap,
                        face_width=face.width if face else None,
                        face_height=face.height if face else None,
                        accepted=recognition.accepted,
                        reason=recognition.reason.value,
                    )
                )

        try:
            with_retry(_write)
        except Exception:
            logger.exception("could not record a recognition attempt")

    # -- config ---------------------------------------------------------------

    def _interval(self) -> float:
        """The INSIDE cameras' recognition interval. Nothing else has one any more.

        Entry and exit used to be gated on 0.25 s here, and that gate sat over the person
        TRACKER as well as the recognition -- which is what split one child into two records.
        They now track on every frame; their face work is throttled by `_needs_a_face`, where
        the cost actually is.
        """
        inside = self.camera.canteen.inside
        return inside.recognition_interval_seconds if inside is not None else 0.25

    def _close_retry_interval(self) -> float:
        """How often to re-attempt a close the exit camera refused as too young.

        **This is what `watch_interval_seconds` is for.** The old per-frame loop re-tried the
        exit close on exactly this cadence, and it is the one thing the key ever meant. It is
        not the tracking cadence: tracking is every frame now, because ByteTrack needs it.
        """
        exit_settings = self.camera.canteen.exit
        return exit_settings.watch_interval_seconds if exit_settings is not None else 0.25

    def stop(self) -> None:
        """Nothing to unwind: the session machine owns the state, and it is in the
        database."""


def build_identity(camera: CanteenCamera, recognizer, gallery) -> IdentityService:
    """One policy object per camera role.

    The legacy had EIGHTEEN overlapping thresholds, with "strong" gates bypassed by
    "soft" gates 0.02 apart — decorative rather than functional, and the fossil record of
    trying to fix a broken recognition pipeline by tuning it.
    """
    policy, soft, binding = _blocks_for(camera)
    return IdentityService(
        recognizer=recognizer,
        gallery=gallery,
        policy=policy,
        binding=binding,
        soft=soft,
    )


def _blocks_for(
    camera: CanteenCamera,
) -> tuple[RecognitionPolicy, SoftAccumulator | None, BindingSettings]:
    canteen = camera.canteen
    if canteen.entry is not None:
        return canteen.entry.recognition, canteen.entry.small_face, canteen.entry.binding
    if canteen.exit is not None:
        return canteen.exit.recognition, canteen.exit.soft, canteen.exit.binding
    if canteen.inside is not None:
        return canteen.inside.recognition, None, canteen.inside.binding
    raise ValueError(f"camera {camera.name!r} has no canteen role block")

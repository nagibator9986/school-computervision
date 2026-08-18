"""The bullying pipeline for one camera: fast tier on every frame, slow tier on candidates.

The two-tier split is the legacy's, and it is correct: cheap geometry on every frame,
expensive pose inference and disk I/O only on the handful of pairs that survive the
gates. Keeping the slow work off the frame loop is what lets one GPU watch six cameras.

What is NOT the legacy's:

  * **Every loop body is wrapped.** The legacy's event-writing loop had a `try/finally`
    with no `except`, so a single `database is locked` killed the thread permanently. The
    dashboard went on looking healthy while the system recorded precisely nothing (H-10).

  * **The queues are bounded, and a full queue is LOUD.** The legacy dropped a fully
    confirmed alert on a `put_nowait` with a bare `pass` — no media, no database row, no
    Telegram, no log line. An assault simply evaporated (M-10).

  * **Frame buffers hold crops, not full HD frames.** The legacy pinned ~900 MB of
    1280x720 frames per queued validation job (H-07).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue

import numpy as np

from qorgan.capture import Frame
from qorgan.config.camera import BullyingCamera
from qorgan.db.types import utcnow
from qorgan.detection.merging import EventMerger, RecentEvent
from qorgan.detection.pipeline import BullyingDetector, Candidate
from qorgan.detection.validation import Verdict
from qorgan.enums import Severity, TelegramSkipReason
from qorgan.events.clip_buffer import ClipBuffer, decode
from qorgan.events.recorder import (
    MediaWriteError,
    severity_for,
    summarise,
    write_clip,
    write_snapshot,
)
from qorgan.events.store import (
    attach_media,
    raise_confidence,
    record_event,
    record_telegram_decision,
)
from qorgan.logging_setup import get_logger
from qorgan.models.person import PersonDetector
from qorgan.models.pose import CROP_BUFFER, PoseEstimator, build_crops
from qorgan.models.validate import validate_candidate
from qorgan.notify import enqueue

logger = get_logger(__name__)

VALIDATION_QUEUE_SIZE = 8


@dataclass(slots=True)
class ValidationJob:
    candidate: Candidate
    crops: list[np.ndarray]
    snapshot: np.ndarray


class BullyingPipeline:
    """Owns the detector, the slow tier and the event writing for ONE camera."""

    def __init__(
        self,
        camera: BullyingCamera,
        camera_id: int,
        person: PersonDetector,
        pose: PoseEstimator,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self._person = person
        self._pose = pose

        self._detector = BullyingDetector(
            camera.bullying, camera.capture.frame_width, camera.capture.frame_height
        )
        self._merger = EventMerger(camera.bullying.event_merge)

        # Bounded (rule R8). Holds recent (timestamp, frame) pairs only long enough to cut
        # a crop from them. Stamped so a candidate's own frame can be told apart from
        # whatever the buffer has been overwritten with since (see CropProvenanceError).
        self._recent: deque[tuple[float, np.ndarray]] = deque(maxlen=CROP_BUFFER)

        # The clip's ring buffer, and a SEPARATE one from the crops above on purpose. The
        # crop buffer is detector input -- its length is what the pose model sees, so
        # resizing it to suit a video would retune the skeleton tier. This one is evidence
        # for a human, sized by `clip_seconds`, JPEG-encoded and bounded in bytes (R8).
        #
        # It exists because a clip that begins when the alarm fires shows the aftermath.
        # The shove is in the seconds BEFORE, which are only available to something that
        # was already keeping them.
        self._clips = ClipBuffer(camera.bullying.clip_seconds, camera.capture.analysis_fps)

        self._queue: Queue[ValidationJob] = Queue(maxsize=VALIDATION_QUEUE_SIZE)

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._validate_loop, name=f"validate:{camera.name}", daemon=True
        )
        self._thread.start()

    # -- the fast tier: called on every frame --------------------------------

    def on_frame(self, _camera: BullyingCamera, frame: Frame) -> str:
        """Detect, track, score, gate. Returns the status shown on the live preview."""
        detections = self._person.detect(frame.image)
        result = self._detector.process(detections, frame.captured_at)

        self._recent.append((frame.captured_at, frame.image))
        # Every frame, not just the interesting ones: whether this second mattered is
        # decided a second from now, by the slow tier, and the frame is gone by then.
        self._clips.append(frame.captured_at, frame.image)

        for candidate in result.candidates:
            self._enqueue(candidate, frame)

        if not result.candidates:
            return "ok"
        return "critical" if len(result.candidates) > 1 else "alert"

    def _enqueue(self, candidate: Candidate, frame: Frame) -> None:
        """Hand a candidate to the slow tier. A full queue is an incident, not a shrug."""
        job = ValidationJob(
            candidate=candidate,
            # The SAME crop builder the harness uses. Crops, not frames: the legacy
            # pinned ~900 MB of HD frames per queued job (H-07). Stamped with the
            # candidate's own timestamp: if this pipeline is ever made to enqueue before
            # judging (it already is) or batch candidates before cropping (it doesn't,
            # yet), a stale crop request raises instead of silently cropping the wrong
            # frames.
            crops=build_crops(list(self._recent), candidate.boxes, candidate.timestamp),
            snapshot=frame.image.copy(),
        )
        try:
            self._queue.put_nowait(job)
        except Full:
            # The legacy did `except Full: pass` here and a confirmed assault vanished
            # without a trace. If we ever drop one, it will be in the log in capitals.
            logger.error(
                "VALIDATION QUEUE FULL — a confirmed candidate was DROPPED",
                extra={
                    "camera": self.camera.name,
                    "pair": candidate.key,
                    "score": round(candidate.score, 2),
                },
            )

    # -- the slow tier -------------------------------------------------------

    def _validate_loop(self) -> None:
        """Never dies. A failure here means one lost event, not a blind camera forever."""
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._handle(job)
            except Exception:
                logger.exception(
                    "validation failed for a candidate",
                    extra={"camera": self.camera.name, "pair": job.candidate.key},
                )
            finally:
                self._queue.task_done()

    def _handle(self, job: ValidationJob) -> None:
        candidate = job.candidate
        confidence_config = self.camera.bullying.confidence

        # crop -> pose -> judge, through the ONE function the eval harness also calls
        # (qorgan.models.validate). Not a copy of it. See that module's docstring.
        verdict = validate_candidate(candidate, job.crops, self._pose, confidence_config)

        severity = severity_for(
            verdict.confidence,
            confidence_config.alert_threshold,
            confidence_config.critical_threshold,
        )
        summary = summarise(severity, self.camera.display_name, verdict.confidence)

        decision = self._merger.decide(candidate.key, candidate.timestamp, candidate.center)
        if not decision.is_new:
            self._merge(decision.merged_into, candidate.timestamp, verdict, severity, summary)
            return

        self._create(job, verdict, severity, summary)

    def _merge(
        self,
        key: tuple[int, int],
        timestamp: float,
        verdict: Verdict,
        severity: Severity,
        summary: str,
    ) -> None:
        """The same incident, again: update the row, and tell a human if this is the look
        that finally crossed the bar.

        A fight's first candidate is often judged before the skeleton can confirm, so the
        row is written below the notify threshold. Seconds later the same pair is judged
        again, the skeleton confirms, and `raise_confidence` lifts that row to CRITICAL --
        inside the merge window, so it lands here. Returning early was the whole bug: the
        database said CRITICAL and Telegram was never told.
        """
        existing = self._merger.get(key)
        if existing is None or existing.event_id is None:
            # No row for this incident: nothing to notify about, nothing to write a reason
            # onto, and nothing to back-fill a clip onto either.
            logger.info(
                "event merged into an existing incident",
                extra={"camera": self.camera.name, "pair": key, "notify": False},
            )
            return

        # Raise the row FIRST: the notifier reads the summary and confidence back out
        # of it, so notifying first would send the ambiguous first look's wording for
        # an incident we now know is critical.
        raise_confidence(existing.event_id, verdict.confidence, severity, summary, verdict.reasons)
        # Back-fill the clip SECOND, and still before the notify, for the same reason the
        # confidence goes first: whoever opens the message should find the video already
        # attached to the row it names.
        self._retro_attach(key, existing.event_id, existing.has_clip, timestamp)
        withheld = self._notify(key, existing.event_id, verdict)

        logger.info(
            "event merged into an existing incident",
            extra={
                "camera": self.camera.name,
                "pair": key,
                "notify": withheld is None,
                "skip_reason": withheld.value if withheld is not None else None,
            },
        )

    def _create(
        self, job: ValidationJob, verdict: Verdict, severity: Severity, summary: str
    ) -> None:
        candidate = job.candidate
        occurred_at = utcnow()
        snapshot_path = self._write_snapshot(job)
        # Written BEFORE the row, so the common case is spec §5.1's first ordering: the
        # media exists by the time the event names it. Either path may hand back None, and
        # None is what goes in the column -- never a path to a file that is not there.
        clip_path = self._write_clip(candidate.timestamp)

        event_id = record_event(
            camera_id=self.camera_id,
            occurred_at=occurred_at,
            verdict=verdict,
            severity=severity,
            summary_text=summary,
            track_ids=f"{candidate.key[0]},{candidate.key[1]}",
            snapshot_path=snapshot_path,
            clip_path=clip_path,
        )

        self._merger.remember(
            candidate.key,
            candidate.timestamp,
            verdict.confidence,
            candidate.center,
            event_id=event_id,
            has_clip=clip_path is not None,
        )

        withheld = self._notify(candidate.key, event_id, verdict)

        self._detector.pairs.get(candidate.key, candidate.timestamp).mark_alerted()

        logger.warning(
            "BULLYING EVENT",
            extra={
                "camera": self.camera.name,
                "event_id": event_id,
                "confidence": round(verdict.confidence, 3),
                "severity": severity.value,
                "skeleton_confirmed": verdict.skeleton_confirmed,
                "reasons": list(verdict.reasons),
                "notify": withheld is None,
                "skip_reason": withheld.value if withheld is not None else None,
            },
        )

    def _notify(
        self, key: tuple[int, int], event_id: int, verdict: Verdict
    ) -> TelegramSkipReason | None:
        """Tell a human, at most once per incident. Returns why nobody was told, or None.

        The at-most-once guarantee has to be made here, against the merger's memory of
        what one incident is: `enqueue` has no uniqueness constraint on event_id, so two
        calls for one event are two rows and two messages on a teacher's phone.

        **The threshold is not consulted here.** `verdict.telegram_skip_reason` already
        carries `should_notify`'s answer from the moment of judgement, together with the
        reason behind it. Asking again would be a second comparison against a number the
        legacy managed to disagree with itself about three times over (§5.4), and it would
        let the recorded reason and the actual decision drift apart by one edit.
        """
        withheld = verdict.telegram_skip_reason
        if self._merger.claim_notification(key, withheld is None):
            record_telegram_decision(event_id, None)
            # Queued as a ROW, in the same breath as the event, and drained by the web
            # process. The legacy spawned a raw thread per alert with a bare requests.post
            # in it, called FROM THE FRAME LOOP -- so an unreachable Telegram stalled the
            # detector for ten seconds, and a network blip lost the alert entirely.
            enqueue(event_id)
            return None

        reason = _why_nobody_was_told(withheld, self._merger.get(key))
        record_telegram_decision(event_id, reason)
        return reason

    def _write_snapshot(self, job: ValidationJob) -> str | None:
        """A picture is better than no picture, but an event with no picture is far
        better than a fight nobody wrote down."""
        try:
            return write_snapshot(job.snapshot, self.camera.name, utcnow())
        except MediaWriteError:
            logger.exception("could not write the snapshot", extra={"camera": self.camera.name})
            return None

    def _write_clip(self, until: float) -> str | None:
        """The seconds leading up to the alarm, as an mp4 — or nothing at all.

        `MediaWriteError` means the file is NOT on disk, so this may only ever return None
        for it. The legacy recorded the path first and wrote the file separately, and all
        447 of its event rows point at clips that do not exist; the dashboard shows a
        broken player and the school concludes the system is lying about the incident too.
        An event with no clip is a loss. An event pointing at a clip that was never written
        is worse than a loss.
        """
        payloads = self._clips.window(until)
        if not payloads:
            # Nothing buffered yet — the first seconds after a camera connects. Not an
            # error, and not worth an exception traceback in the log every time.
            return None
        try:
            # `decode` is a generator: the frames reach the writer one at a time and the
            # whole clip is never in memory at once.
            return write_clip(decode(payloads), self.camera.name, self._clips.fps, utcnow())
        except MediaWriteError:
            logger.exception("could not write the clip", extra={"camera": self.camera.name})
            return None

    def _retro_attach(
        self, key: tuple[int, int], event_id: int, has_clip: bool, until: float
    ) -> None:
        """Spec §5.1's second ordering: the row exists and the media does not, yet.

        The first look at a fight can easily land with nothing to write — the camera has
        just connected, or the disk refused — and that row would then carry a broken video
        player forever. But the same incident is judged again seconds later, and by then
        the buffer holds the footage. So the row is back-filled instead of a second row
        being created for one shove.

        `attach_media` is COALESCE: it sets the clip and cannot wipe the snapshot the
        first look did manage to write (audit M-20).
        """
        if has_clip:
            return
        clip_path = self._write_clip(until)
        if clip_path is None:
            return
        attach_media(event_id, clip_path=clip_path)
        self._merger.note_clip(key)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _why_nobody_was_told(
    withheld: TelegramSkipReason | None, incident: RecentEvent | None
) -> TelegramSkipReason:
    """The reason to record when the claim was refused. Total: there is always an answer.

    **"Already told" outranks the judgement's own reason, and the order is the whole
    correctness of this function.** A four-second fight is judged many times. Once one of
    those judgements has sent the message, every later one is a fresh verdict on the same
    row — and a later look is often weaker than the one that convinced us, so its own
    reason would be something like LOW_CONFIDENCE. Writing that would stamp "не
    отправлено — недостаточная уверенность" onto an incident a teacher has already been
    messaged about: a row that contradicts the phone in their hand.

    `withheld is None` here means this judgement wanted to send and the merger refused,
    and the only branch of `claim_notification` that refuses a willing judgement is the
    one that has already been claimed. So both arms are the same fact, reached two ways.
    """
    if withheld is None or (incident is not None and incident.notified):
        return TelegramSkipReason.ALREADY_NOTIFIED
    return withheld

"""The weapons pipeline for one camera: plumbing only. Everything decided is decided by
`qorgan.weapons`, which is pure.

Two tiers again, and the split is not the bullying one. There the fast tier is geometry
and the slow tier is a pose model; here the FAST tier is the person detector (every frame,
because a track id is what «рядом с человеком» is measured against and a track that skips
frames is a track that loses its id), and the reduced-rate tier is the WEAPON model,
which runs on every `weapons.analyse_every`-th frame. §12.1 asks for the reduced rate in
those words, and the GPU in this process is already shared.

What is deliberately borrowed rather than rebuilt:

  * **`events.clip_buffer.ClipBuffer`.** The evidence clip has to cover the seconds BEFORE
    the alarm, so frames are kept as they arrive. That is the exact shape of the legacy's
    worst memory defect (~460 MB pinned by one suppressed burst), and there is already a
    buffer here that is bounded twice -- by a frame count derived from configuration and
    by a BYTE BUDGET that holds when the first bound is wrong. Writing a second one would
    be writing the untested half of that lesson again.
  * **`events.recorder`.** `write_snapshot` / `write_clip` encode the bytes themselves
    because `cv2.imwrite` fails SILENTLY on this school's Cyrillic paths, and they raise
    rather than returning a path to a file that is not there.

R7: the alert thread's body is wrapped. A failure writing one alert is one lost alert,
never a camera that stops watching. The legacy's event loop had a `try/finally` with no
`except`, so one `database is locked` killed it permanently while the dashboard stayed
green (H-10).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue

import numpy as np

from qorgan.capture import Frame
from qorgan.config.camera import WeaponsCamera
from qorgan.db.types import utcnow
from qorgan.events.clip_buffer import ClipBuffer, decode
from qorgan.events.recorder import MediaWriteError, write_clip, write_snapshot
from qorgan.events.store import record_telegram_decision
from qorgan.logging_setup import get_logger
from qorgan.models.person import PersonDetector
from qorgan.notify import enqueue
from qorgan.weapons.model import WeaponView
from qorgan.weapons.pipeline import WeaponAlert, WeaponsDetector
from qorgan.weapons.store import record_weapon_alert, summarise_weapon

logger = get_logger(__name__)

# Small on purpose. A weapon alert is rare and a backlog of them is not a queue to drain,
# it is a camera producing nonsense -- and the RIGHT answer to that is a loud log line
# rather than memory. Same reasoning as `worker.bullying.VALIDATION_QUEUE_SIZE`.
ALERT_QUEUE_SIZE = 4


@dataclass(slots=True)
class AlertJob:
    alert: WeaponAlert
    snapshot: np.ndarray


class WeaponsPipeline:
    """Owns the two models, the decision and the evidence writing for ONE camera."""

    def __init__(
        self,
        camera: WeaponsCamera,
        camera_id: int,
        person: PersonDetector,
        weapons: WeaponView,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self._person = person
        self._weapons = weapons

        self._detector = WeaponsDetector(
            camera.weapons, camera.capture.frame_width, camera.capture.frame_height
        )
        # Bounded by frames AND by bytes; see the module docstring on why this is borrowed.
        self._clips = ClipBuffer(camera.weapons.clip_seconds, camera.capture.analysis_fps)

        # Counts the frames that reached this pipeline, so `analyse_every` is honoured
        # against what ARRIVES rather than against wall-clock time -- a stream delivering
        # at half its configured rate would otherwise silently double the analysis rate.
        self._frames = 0

        self._queue: Queue[AlertJob] = Queue(maxsize=ALERT_QUEUE_SIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._alert_loop, name=f"weapon-alert:{camera.name}", daemon=True
        )
        self._thread.start()

    # -- the frame loop ------------------------------------------------------

    def on_frame(self, _camera: WeaponsCamera, frame: Frame) -> str:
        """People every frame, weapons every Nth. Returns the live preview's status."""
        self._frames += 1
        # Every frame, not just the analysed ones: whether this second mattered is decided
        # a second from now, and the frame is gone by then.
        self._clips.append(frame.captured_at, frame.image)

        people = self._person.detect(frame.image)
        if self._frames % self.camera.weapons.analyse_every:
            # A frame the weapon model does not look at. The people are still tracked --
            # skipping detection here would break the track ids the next analysed frame
            # measures «рядом с человеком» against.
            return "ok"

        sightings = self._weapons.detect(frame.image)
        outcome = self._detector.process(sightings, people, frame.captured_at)
        self._log_refusals(outcome)

        for alert in outcome.alerts:
            self._enqueue(alert, frame)
        return "critical" if outcome.alerts else "ok"

    def _log_refusals(self, outcome) -> None:
        """Say what was seen and refused, and why.

        Not debris. Which screen a camera's sightings die on is the difference between
        "there are no weapons here" (the hoped-for answer) and "everything this camera
        sees is below the size gate" (a camera in the wrong place). The second is
        invisible unless it is counted, and it was invisible for months in the system
        this replaces.
        """
        if not outcome.refusals:
            return
        logger.info(
            "weapon sightings refused",
            extra={
                "camera": self.camera.name,
                "refused_by": outcome.refused_by,
                "tracked": outcome.tracked,
                "evicted_over_ceiling": self._detector.tracks.evicted_over_ceiling,
            },
        )

    def _enqueue(self, alert: WeaponAlert, frame: Frame) -> None:
        """Hand an alert to the writing thread. A full queue is an incident, not a shrug."""
        try:
            self._queue.put_nowait(AlertJob(alert=alert, snapshot=frame.image.copy()))
        except Full:
            # The legacy did `except Full: pass` on its equivalent and a confirmed assault
            # vanished without a trace (M-10). If we ever drop one, it is in the log in
            # capitals -- and on this pipeline the thing dropped is a possible weapon.
            logger.error(
                "WEAPON ALERT QUEUE FULL — a possible weapon was DROPPED and no human "
                "will be asked about it",
                extra={
                    "camera": self.camera.name,
                    "class": alert.class_name,
                    "confidence": round(alert.confidence, 2),
                },
            )

    # -- writing the evidence, off the frame loop ----------------------------

    def _alert_loop(self) -> None:
        """Never dies. A failure here is one lost alert, not a blind camera forever."""
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._write(job)
            except Exception:
                logger.exception(
                    "could not record a weapon alert",
                    extra={"camera": self.camera.name, "class": job.alert.class_name},
                )
            finally:
                self._queue.task_done()

    def _write(self, job: AlertJob) -> None:
        """Snapshot, clip, row, and one message asking a person to look. In that order.

        The media goes first so that the common case is spec §5.1's first ordering: the
        file exists by the time the row names it. Either write may hand back None, and
        None is what goes in the column -- never a path to a file that is not there. The
        legacy recorded the path first and wrote the file separately, and all 447 of its
        event rows point at clips that do not exist.
        """
        alert = job.alert
        occurred_at = utcnow()
        rules = self.camera.weapons

        event_id = record_weapon_alert(
            camera_id=self.camera_id,
            occurred_at=occurred_at,
            alert=alert,
            weights=self._weapons.weights,
            summary_text=summarise_weapon(alert, self.camera.display_name),
            min_observations=rules.min_track_observations,
            reconfirm_observations=rules.reconfirm_observations,
            snapshot_path=self._snapshot(job),
            clip_path=self._clip(alert.timestamp),
        )

        # `None` = nothing was withheld. Written for the same reason the bullying tier
        # writes it: the column is how the school's first question -- "why was I not
        # told?" -- gets an answer at all, and leaving it unset would make a sent alert
        # indistinguishable from one nobody decided about.
        record_telegram_decision(event_id, None)
        enqueue(event_id)

        logger.warning(
            "POSSIBLE WEAPON — a person must confirm this",
            extra={
                "camera": self.camera.name,
                "event_id": event_id,
                "class": alert.class_name,
                "confidence": round(alert.confidence, 3),
                "observations": alert.observations,
                "strong_observations": alert.strong_observations,
                "person_track": alert.person_track_id,
                "weights": self._weapons.weights.file.fingerprint,
                # Said on every single alert, because the log is where somebody looks
                # after the fact and this is the property §12.1 turns on.
                "auto_actioned": False,
            },
        )

    def _snapshot(self, job: AlertJob) -> str | None:
        try:
            return write_snapshot(job.snapshot, self.camera.name, utcnow())
        except MediaWriteError:
            logger.exception("could not write the snapshot", extra={"camera": self.camera.name})
            return None

    def _clip(self, until: float) -> str | None:
        """The seconds leading up to the alarm, as an mp4 — or nothing at all.

        `MediaWriteError` means the file is NOT on disk, so this may only ever return None
        for it. An alert with no clip is a loss; an alert pointing at a clip that was
        never written is worse, because the operator asked to CONFIRM a weapon opens a
        broken player and has nothing to rule on.
        """
        payloads = self._clips.window(until)
        if not payloads:
            return None
        try:
            return write_clip(decode(payloads), self.camera.name, self._clips.fps, utcnow())
        except MediaWriteError:
            logger.exception("could not write the clip", extra={"camera": self.camera.name})
            return None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

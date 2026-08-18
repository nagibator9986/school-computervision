"""The classroom pipeline for one camera. Plumbing only; everything decided is decided
by pure functions in `qorgan.classroom`.

Two models per frame, and neither can do the other's job. YOLO+ByteTrack supplies boxes
that KEEP their identity between frames; YOLOv8n-pose supplies skeletons that describe
arms but arrive unordered and anonymous. `classroom.association.assign` joins them, and
refuses to guess when the geometry is ambiguous.

**It hangs off the ordinary camera path.** `CameraLoop` opens the stream, honours
`det_every`, resizes through the one shared `prepare_frame`, and calls `on_frame` -- the
same door the bullying and canteen pipelines come through. Nothing here opens a capture
device or knows what kind of thing is behind the stream, which is what will let this run
against a recorded lesson for free the day the file-source camera lands: a file source is
a `CameraStream`, and this never touches one.

**A worker stopping does not end a lesson, and that asymmetry is the point.** `stop()`
flushes and leaves the row OPEN, so the next worker re-attaches to it rather than
starting a second lesson at minute 20 and reporting the first as having ended there. A
lesson ends when the ROOM empties, or when the janitor times it out. The legacy lost
every open canteen session on restart; here a restart costs `flush_interval_seconds`.
"""

from __future__ import annotations

from qorgan.capture import Frame
from qorgan.classroom import store
from qorgan.classroom.association import assign
from qorgan.classroom.lesson import LessonAccumulator
from qorgan.classroom.store import Clock, LessonHandle
from qorgan.config.camera import ClassroomCamera
from qorgan.enums import LessonCloseReason
from qorgan.logging_setup import get_logger
from qorgan.models.person import PersonDetector
from qorgan.models.pose import PoseModel

logger = get_logger(__name__)


class ClassroomPipeline:
    """One classroom camera's worth of lesson counting. Anonymous throughout."""

    def __init__(
        self,
        camera: ClassroomCamera,
        camera_id: int,
        person: PersonDetector,
        pose: PoseModel,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self._person = person
        self._pose = pose

        self._lesson: LessonHandle | None = None
        self._tally: LessonAccumulator | None = None
        # None, not 0.0, for the reason `CanteenPipeline` documents: a monotonic clock
        # starts near zero, so a 0.0 sentinel makes the camera skip its own first frames.
        self._last_flush: float | None = None
        self._last_person_at: float | None = None

    # -- the frame loop ------------------------------------------------------

    def on_frame(self, _camera: ClassroomCamera, frame: Frame) -> str:
        """Track the people, read their skeletons, and fold the frame into the lesson.

        A lesson OPENS on the first frame that holds a tracked person and CLOSES once the
        room has been empty for `end_after_empty_minutes`. An empty room is the only
        evidence of an ending this system has: there is no bell signal and no timetable.
        """
        now = frame.captured_at
        boxes = self._person.detect(frame.image)

        if not boxes:
            self._maybe_end(now)
            return "ok"

        self._last_person_at = now
        if self._lesson is None:
            self._begin(now)

        rules = self.camera.classroom
        skeletons = self._pose.keypoints(
            frame.image, imgsz=rules.pose.imgsz, conf=rules.pose.conf
        )
        assignment = assign(skeletons, boxes)

        assert self._tally is not None  # _begin built it
        retired = self._tally.observe(assignment, now, rules.hand_raise, rules.place)
        if retired or self._flush_due(now):
            self._flush(now, retired)
        return "ok"

    # -- the lesson ----------------------------------------------------------

    def _begin(self, now: float) -> None:
        rules = self.camera.classroom.lesson
        self._lesson = store.open_or_resume(self.camera_id, rules)
        # Seeded with what the lesson already failed to see. `store.flush` ASSIGNS these
        # counters, so a resuming worker starting from zero would overwrite the previous
        # one's measurement -- and the lesson would look better watched after every crash.
        self._tally = LessonAccumulator(rules=rules, doubt=self._lesson.doubt)
        self._last_flush = now
        logger.info(
            "lesson observation started",
            extra={
                "camera": self.camera.name,
                "lesson": self._lesson.lesson_id,
                "resumed": self._lesson.resumed,
            },
        )

    def _maybe_end(self, now: float) -> None:
        """Close the lesson once the room has stayed empty long enough."""
        if self._lesson is None or self._last_person_at is None:
            return

        empty_for = now - self._last_person_at
        if empty_for < self.camera.classroom.lesson.end_after_empty_minutes * 60:
            return

        assert self._tally is not None
        # Finalised at the last moment anybody was SEEN, not at `now`. Finalising at `now`
        # would close every open excursion with the whole empty-room timeout inside it, so
        # a child still away when the lesson ended would be credited with five minutes
        # nobody observed -- the room was empty, which is not evidence about a child.
        self._flush(now, self._tally.finish(self._last_person_at))
        store.close(self._lesson.lesson_id, LessonCloseReason.EMPTY_ROOM)
        logger.info(
            "lesson observation ended",
            extra={"camera": self.camera.name, "lesson": self._lesson.lesson_id},
        )
        self._lesson = None
        self._tally = None
        self._last_person_at = None

    # -- writing through -----------------------------------------------------

    def _flush_due(self, now: float) -> bool:
        if self._last_flush is None:
            return True
        interval = self.camera.classroom.lesson.flush_interval_seconds
        return now - self._last_flush >= interval

    def _flush(self, now: float, retired: list) -> None:
        """Write the retired ledgers AND the live ones through.

        Both, always. Retired ledgers would otherwise be lost between flushes, and live
        ones are what makes a mid-lesson crash cost half a minute instead of the lesson.
        A fresh `Clock` per flush, because it converts the loop's monotonic timestamps to
        instants and a stale one carries the drift between the two clocks into the rows.
        """
        if self._lesson is None or self._tally is None:
            return

        self._last_flush = now
        store.flush(
            self._lesson.lesson_id,
            retired + self._tally.live(),
            self._tally.doubt,
            Clock.now(),
        )

    def stop(self) -> None:
        """Write through and LEAVE THE LESSON OPEN. A worker stopping is not a bell."""
        if self._lesson is None or self._tally is None:
            return
        at = self._last_person_at or 0.0
        self._flush(at, self._tally.finish(at))
        logger.info(
            "worker stopping mid-lesson; lesson left open for the next worker",
            extra={"camera": self.camera.name, "lesson": self._lesson.lesson_id},
        )

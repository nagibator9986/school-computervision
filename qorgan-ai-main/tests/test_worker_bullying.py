"""The bullying pipeline's event writing: one incident, one alert — and never none.

Driven through `_handle`, the slow tier's judgement on one candidate, with a scripted
skeleton. That is the unit the notification contract lives in, and driving it directly is
deliberate: `_validate_loop` catches every exception by design (a failure there must cost
one event, not the camera), so a fault reached through the queue arrives as a log line and
an empty database rather than as a failure that names itself.

No GPU and no model. What is under test is which incidents raise a Telegram message, and
that decision is made from a `Verdict` and a `MergeDecision` — neither of which needs one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.capture import Frame
from qorgan.config.camera import CAMERA_ADAPTER, BullyingCamera
from qorgan.db.models import Camera, Event, Notification
from qorgan.detection.geometry import Box
from qorgan.detection.pipeline import Candidate
from qorgan.detection.validation import SkeletonResult, score_skeleton
from qorgan.enums import CameraRole, CameraType, Severity
from qorgan.events.reasons import unpack_reasons
from qorgan.events.recorder import MediaWriteError, media_root
from qorgan.settings import Settings
from qorgan.worker.bullying import BullyingPipeline, ValidationJob

# The candidate the heuristics hand to the slow tier. probability 0.9 is what makes the
# two judgements below land where they do; see NOT_YET and CONFIRMED.
PROBABILITY = 0.9
CENTER = (0.5, 0.5)
PAIR = (1, 2)

# The skeleton could not look yet: a fight's opening frames give it too few crops. Nothing
# is confirmed and nothing is capped, so the confidence is the heuristics' alone:
# 0.9 * 0.7 = 0.63, well below the 0.85 notify bar.
NOT_YET = SkeletonResult(skipped=True, skip_reason="too_few_crops")

# A second later the skeleton has seen the whole thing: hands, contact, a kick, a sharp
# displacement, and a child on the floor. Score 1.0, confirmed, so
# 0.9 * 0.7 + 1.0 * 0.3 + 0.05 agreement boost = 0.98 — past the 0.95 critical bar.
FULL_FIGHT = (
    "rapid_hand_motion",
    "close_upper_body_contact",
    "kick_like_leg_motion",
    "sudden_body_displacement",
    "body_fall_or_low_posture",
)
# Scored by the SAME function the real pose model's result is scored by, so this fixture
# cannot quietly stop meaning what its name says if the weights are retuned.
CONFIRMED = SkeletonResult(
    reasons=FULL_FIGHT, score=score_skeleton(list(FULL_FIGHT)), skipped=False, skip_reason=""
)


class ScriptedPose:
    """A skeleton whose opinion changes between looks, the way a real one's does.

    This is the whole shape of the problem: the judgement that matters is often not the
    first one. A shove is ambiguous until somebody hits the floor.
    """

    def __init__(self, *looks: SkeletonResult) -> None:
        self._looks = list(looks)
        self.calls = 0

    def validate(self, _crops: list[np.ndarray]) -> SkeletonResult:
        look = self._looks[min(self.calls, len(self._looks) - 1)]
        self.calls += 1
        return look


class NoPeople:
    """`_handle` never runs the person detector; only `on_frame` does."""

    def detect(self, _frame: np.ndarray) -> dict[int, Box]:
        return {}


def _camera() -> BullyingCamera:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": CameraRole.MAIN_HALL.value,
            "name": "main_hall",
            "display_name": "Главный холл",
            "rtsp": {"host": "10.0.0.7", "burst_path": None},
        }
    )


def _job(at: float, key: tuple[int, int] = PAIR) -> ValidationJob:
    candidate = Candidate(
        key=key,
        timestamp=at,
        score=6.0,
        threshold=4.0,
        probability=PROBABILITY,
        boxes=(Box(100.0, 100.0, 200.0, 400.0), Box(180.0, 100.0, 280.0, 400.0)),
        center=CENTER,
        in_normal_flow=False,
        in_staircase=False,
    )
    picture = np.full((64, 64, 3), 100, dtype=np.uint8)
    return ValidationJob(candidate=candidate, crops=[picture, picture], snapshot=picture)


def _feed(pipeline: BullyingPipeline, count: int = 12, start: float = 0.0) -> float:
    """Run frames through the REAL frame-loop entry point and return the last timestamp.

    Through `on_frame`, deliberately — reaching into the buffer directly would test the
    buffer and prove nothing about whether anything in production ever fills it. That is
    the exact defect this file is here about: `write_clip` was complete, tested, and called
    by nobody, so every event carried a JPEG or nothing.
    """
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    stamp = start
    for seq in range(count):
        stamp = start + seq * 0.1
        pipeline.on_frame(
            pipeline.camera,
            Frame(image=image, seq=seq, captured_at=stamp, camera=pipeline.camera.name),
        )
    return stamp


def _clip_files() -> list[Path]:
    return sorted(p for p in (media_root() / "clips").rglob("*.mp4"))


@pytest.fixture
def build(settings: Settings, session: Session) -> Iterator[Callable[..., BullyingPipeline]]:
    """A pipeline against a real camera row, stopped however the test ends."""
    built: list[BullyingPipeline] = []

    def _build(pose: ScriptedPose) -> BullyingPipeline:
        row = Camera(
            name="main_hall",
            display_name="Главный холл",
            camera_type=CameraType.BULLYING,
            role=CameraRole.MAIN_HALL,
            rtsp_host="10.0.0.7",
        )
        session.add(row)
        session.commit()

        pipeline = BullyingPipeline(
            camera=_camera(),
            camera_id=row.id,
            person=NoPeople(),  # type: ignore[arg-type]
            pose=pose,  # type: ignore[arg-type]
        )
        built.append(pipeline)
        return pipeline

    yield _build
    for pipeline in built:
        pipeline.stop()


def _notifications(session: Session, event_id: int) -> list[Notification]:
    return list(
        session.scalars(select(Notification).where(Notification.event_id == event_id)).all()
    )


# -- notify at most once, and never zero -------------------------------------


def test_a_fight_that_only_convinces_us_on_the_second_look_is_still_reported_to_a_human(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """**The fight the database calls CRITICAL that nobody was ever told about.**

    The first candidate is judged before the skeleton can confirm, so it lands below the
    notify bar: an event row is written and — correctly — no Telegram is sent. Three
    seconds later the same two children are judged again, the skeleton confirms, and the
    confidence reaches CRITICAL. But that is the same incident inside the 15 s merge
    window, so it updates the row already written instead of creating a second one.

    The row then says CRITICAL and nobody has been told. "Notify once" (spec §147) means
    *at most* once — not "only if the very first look was already convincing". This breaks
    the client's §7 outright (every event at or above `alert` is sent), and it is the
    product's entire purpose failing silently.
    """
    pipeline = build(ScriptedPose(NOT_YET, CONFIRMED))

    pipeline._handle(_job(at=0.0))
    pipeline._handle(_job(at=3.0))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.severity is Severity.CRITICAL, (
        "this fixture no longer reproduces the bug: the merged judgement must raise the "
        "row to CRITICAL, or there is no silently-unreported fight to catch"
    )
    assert len(_notifications(session, event.id)) == 1, (
        f"the database records a {event.severity.value} fight and nobody was notified"
    )


def test_the_message_explains_the_look_that_actually_raised_the_alarm(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """The other half of the fight nobody was told about — the half that says WHY.

    Same shape: the first look cannot confirm and writes a row with nothing to say for
    itself; three seconds later the skeleton sees a child hit the floor, the row is raised
    to CRITICAL, and *that* is the look that sends the Telegram. The notifier composes the
    message from the ROW, so if the reasons do not move with the confidence, the school
    gets a CRITICAL alert explaining the ambiguous first glance — or explaining nothing at
    all, which is exactly what the message did before client §7 was built.
    """
    pipeline = build(ScriptedPose(NOT_YET, CONFIRMED))

    pipeline._handle(_job(at=0.0))
    pipeline._handle(_job(at=3.0))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.severity is Severity.CRITICAL, "the fixture no longer reproduces the merge"
    assert unpack_reasons(event.reasons) == FULL_FIGHT, (
        "the row that was raised to CRITICAL and sent to Telegram does not carry the "
        "reasons of the look that raised it"
    )
    # The one that matters to a human: somebody went down. It is the only CLEAN evidence
    # in the hierarchy, and the first look had none of it.
    assert "body_fall_or_low_posture" in unpack_reasons(event.reasons)


def test_an_event_records_the_reasons_it_was_judged_on(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """The straight path: one convincing look, one event, and the evidence written down
    with it rather than left in a log line."""
    pipeline = build(ScriptedPose(CONFIRMED))

    pipeline._handle(_job(at=0.0))

    session.expire_all()
    event = session.scalars(select(Event)).one()
    assert unpack_reasons(event.reasons) == FULL_FIGHT


def test_one_fight_judged_twice_does_not_notify_twice(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """The other half of the contract, and the reason the first half is not simply
    "notify on every merge".

    A four-second fight produces a confirmed pair on dozens of frames. The legacy sent one
    Telegram message per frame, which trains staff to ignore the alerts — worse than
    having none. Both looks here are convincing; exactly one message may leave the building.
    """
    pipeline = build(ScriptedPose(CONFIRMED, CONFIRMED))

    pipeline._handle(_job(at=0.0))
    pipeline._handle(_job(at=3.0))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert len(_notifications(session, event.id)) == 1, "one fight, one message"


# -- the clip, in production -------------------------------------------------


def test_a_recorded_event_carries_a_clip_of_the_seconds_before_the_alarm(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """**The defect this file's clip tests exist for.**

    `write_clip` was written, tested, and called from nowhere in `src/` — so every event
    the school ever saw carried a JPEG snapshot or nothing at all, while REWRITE_SPEC §5.1,
    client §6 and §7 all promise a clip and the events page already renders a `<video>` for
    one. A still photograph of two children standing close is not evidence of a shove; the
    two seconds leading up to it are.

    Driven through `on_frame` and then the slow tier, which is the whole point: the buffer
    has to be filled by the production frame loop, not by a test.
    """
    pipeline = build(ScriptedPose(CONFIRMED))
    alarm = _feed(pipeline)

    pipeline._handle(_job(at=alarm))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.clip_path is not None, (
        "the event has no clip: nothing in production called write_clip"
    )
    # R6: the project has moved twice and each move broke 100% of the media rows.
    assert not Path(event.clip_path).is_absolute()
    assert event.clip_path.startswith("clips/")
    assert (media_root() / event.clip_path).stat().st_size > 0
    # The snapshot is still there. A clip is an addition to the evidence, not a swap.
    assert event.snapshot_path is not None


def test_an_event_never_points_at_a_clip_that_was_not_written(
    settings: Settings,
    session: Session,
    build: Callable[..., BullyingPipeline],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MediaWriteError` means the file is NOT on disk — the recorder's docstring says the
    caller must not record an event pointing at it, and the legacy's 447 event rows all
    point at clips that never existed.

    And the event itself must still be written. A disk that refused a video is not a reason
    to forget that a child was assaulted.
    """

    def _refuse(*_args: object, **_kwargs: object) -> str:
        raise MediaWriteError("the disk said no")

    monkeypatch.setattr("qorgan.worker.bullying.write_clip", _refuse)

    pipeline = build(ScriptedPose(CONFIRMED))
    alarm = _feed(pipeline)

    pipeline._handle(_job(at=alarm))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.clip_path is None, "the row names a clip that was never written"
    assert event.severity is Severity.CRITICAL, "the event itself was lost with the clip"


def test_a_row_written_without_a_clip_is_back_filled_by_a_later_look(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """Spec §5.1's second ordering: the event is created before the media exists.

    The first look at this fight lands with an empty buffer — the camera has only just
    connected — so the row is written with no video, and on the events page that is a
    broken player. Three seconds later the same two children are judged again and by then
    there IS footage, so it is attached to the row already written rather than becoming a
    second row for one shove.
    """
    pipeline = build(ScriptedPose(CONFIRMED, CONFIRMED))

    pipeline._handle(_job(at=0.0))  # nothing buffered yet
    session.expire_all()
    assert session.scalars(select(Event)).one().clip_path is None, (
        "this fixture no longer reproduces the ordering: there is no missing clip to attach"
    )

    _feed(pipeline, start=2.0)
    pipeline._handle(_job(at=3.0))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.clip_path is not None, "the later look never back-filled the clip"
    assert (media_root() / event.clip_path).stat().st_size > 0
    assert event.snapshot_path is not None, "attaching the clip erased the snapshot (M-20)"


def test_one_fight_does_not_collect_a_second_clip(
    settings: Settings, session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """The other half of the retro-attach contract, and the reason it is not simply
    "write a clip on every merge".

    A four-second fight is judged on dozens of frames. Re-cutting the video each time
    would fill the disk with near-identical copies of one shove and leave every superseded
    one orphaned — nothing points at them and nothing deletes them.
    """
    pipeline = build(ScriptedPose(CONFIRMED, CONFIRMED))
    alarm = _feed(pipeline)

    pipeline._handle(_job(at=alarm))
    session.expire_all()
    first = session.scalars(select(Event)).one().clip_path

    _feed(pipeline, start=alarm + 0.1)
    pipeline._handle(_job(at=alarm + 1.5))

    session.expire_all()
    event = session.scalars(select(Event)).one()

    assert event.clip_path == first, "the merged look cut a second clip over the first"
    assert len(_clip_files()) == 1, (
        f"one fight left {len(_clip_files())} clips on disk; the extras are orphans"
    )

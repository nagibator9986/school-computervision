"""The weapons pipeline as the worker runs it: two tiers, one thread, and no silent losses.

Everything decided is decided by `qorgan.weapons`, which is pure and tested elsewhere.
What is left here is plumbing, and plumbing is where this project's measured disasters
live:

  * **R7.** The legacy's event loop had a `try/finally` with no `except`, so one
    `database is locked` killed it permanently while the dashboard stayed green (H-10).
    The alert thread here must survive a failure and keep taking work.
  * **`except Full: pass`.** The legacy did that on its equivalent queue and a confirmed
    assault vanished without a trace (M-10). Here the thing dropped is a possible weapon,
    so it goes in the log in capitals.
  * **A path to a file that is not there.** All 447 of the legacy's event rows point at
    clips that do not exist, because it recorded the path first and wrote the file
    separately. Media is written FIRST here and `None` is what goes in the column when
    the write fails.

The two-tier split is the opposite of the bullying one: the fast tier is the PERSON
detector (every frame, because a track id is what «рядом с человеком» is measured against
and a track that skips frames loses its id) and the reduced-rate tier is the weapon model.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.capture.stream import Frame
from qorgan.db.models import Camera, Event
from qorgan.enums import CameraRole, CameraType, EventStatus, EventType
from qorgan.events.clip_buffer import ClipBuffer
from qorgan.settings import Settings, get_settings
from qorgan.worker.weapons import ALERT_QUEUE_SIZE, WeaponsPipeline
from tests.weapons_fixtures import FakeWeaponView, person_box, sighting, weapons_camera

FRAME = np.zeros((540, 960, 3), dtype=np.uint8)


class FakePerson:
    """A person detector that always sees the same person, with the same track id."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> dict[int, object]:
        del image
        self.calls += 1
        return {7: person_box()}


@pytest.fixture
def camera_id(session: Session) -> int:
    camera = Camera(
        name="entrance_frame",
        display_name="Вход — рамка",
        camera_type=CameraType.WEAPONS,
        role=CameraRole.WEAPONS,
        rtsp_host="192.168.1.90",
    )
    session.add(camera)
    session.commit()
    return camera.id


@pytest.fixture
def parts(settings: Settings, camera_id: int) -> Iterator[tuple]:
    """A running pipeline, always stopped again -- conftest fails the test otherwise."""
    del settings
    person, view = FakePerson(), FakeWeaponView()
    pipeline = WeaponsPipeline(
        camera=weapons_camera(), camera_id=camera_id, person=person, weapons=view
    )
    try:
        yield pipeline, person, view
    finally:
        pipeline.stop()


def _frame(seq: int) -> Frame:
    return Frame(image=FRAME, seq=seq, captured_at=seq * 0.2, camera="entrance_frame")


def _feed(pipeline: WeaponsPipeline, frames: int, start: int = 0) -> list[str]:
    return [pipeline.on_frame(pipeline.camera, _frame(seq)) for seq in range(start, start + frames)]


def _wait_for(session: Session, model, expected: int, seconds: float = 5.0) -> int:
    """Rows of `model`, once the alert thread has got round to writing them.

    Per model rather than "wait for the event, then assume the rest": `_write` inserts the
    event, then the telegram decision, then the notification, and a waiter that stops at
    the first of those reads the other two mid-write. That is a flake, and it is the kind
    that appears on somebody else's branch a week later.
    """
    deadline = time.monotonic() + seconds
    count = 0
    while time.monotonic() < deadline:
        session.expire_all()
        count = session.scalar(select(func.count(model.id))) or 0
        if count >= expected:
            return count
    return count


def _wait_for_events(session: Session, expected: int, seconds: float = 5.0) -> int:
    return _wait_for(session, Event, expected, seconds)


# -- the two tiers ---------------------------------------------------------


def test_the_person_detector_runs_on_every_frame(parts) -> None:
    """Skipping it would break the track ids the next analysed frame measures «рядом с
    человеком» against."""
    pipeline, person, _ = parts
    _feed(pipeline, 9)
    assert person.calls == 9


def test_the_weapon_model_runs_at_the_reduced_rate_section_12_1_asks_for(parts) -> None:
    pipeline, _, view = parts
    assert pipeline.camera.weapons.analyse_every == 3
    _feed(pipeline, 9)
    assert view.calls == 3


def test_the_rate_is_counted_against_frames_that_ARRIVE(parts) -> None:
    """Not against a wall clock: a stream delivering at half its configured rate would
    otherwise silently double the analysis rate."""
    pipeline, _, view = parts
    _feed(pipeline, 3)
    assert view.calls == 1
    _feed(pipeline, 3, start=100)
    assert view.calls == 2


def test_a_quiet_camera_reports_ok(parts) -> None:
    pipeline, _, _ = parts
    assert set(_feed(pipeline, 9)) == {"ok"}


# -- an alert becomes a row, a snapshot and a clip -------------------------


def _arm(view: FakeWeaponView, frames: int = 12) -> None:
    """Make the fake detector see a knife on every frame it is asked about."""
    view.script = [[sighting("knife", 0.9)] for _ in range(frames)]


def test_a_confirmed_track_produces_exactly_one_row(parts, session: Session) -> None:
    pipeline, _, view = parts
    _arm(view)
    statuses = _feed(pipeline, 9)

    assert "critical" in statuses, "the live preview must show it too"
    assert _wait_for_events(session, 1) == 1


def test_the_row_is_a_weapon_alert_awaiting_a_person(parts, session: Session) -> None:
    """§12.1's whole point, at the end of the real worker path."""
    pipeline, _, view = parts
    _arm(view)
    _feed(pipeline, 9)
    _wait_for_events(session, 1)

    event = session.scalars(select(Event)).one()
    assert event.event_type is EventType.WEAPON
    assert event.status is EventStatus.NEW
    assert event.reviewed_by_id is None
    assert event.summary_text.startswith("Возможное оружие")


def test_the_evidence_is_written_before_the_row_names_it(parts, session: Session) -> None:
    """A row pointing at a clip that was never written is worse than a row with no clip:
    the operator asked to CONFIRM a weapon opens a broken player and has nothing to rule
    on."""
    pipeline, _, view = parts
    _arm(view)
    _feed(pipeline, 9)
    _wait_for_events(session, 1)

    event = session.scalars(select(Event)).one()
    root = get_settings().media_root
    assert event.snapshot_path and (root / event.snapshot_path).is_file()
    assert event.clip_path and (root / event.clip_path).is_file()


def test_a_notification_is_queued_for_the_alert(parts, session: Session) -> None:
    from qorgan.db.models import Notification

    pipeline, _, view = parts
    _arm(view)
    _feed(pipeline, 9)
    assert _wait_for(session, Notification, 1) == 1


def test_one_knife_carried_past_is_one_question(parts, session: Session) -> None:
    """Not one a second: the realert quiet period is 60 s and this is 12 s of frames."""
    pipeline, _, view = parts
    _arm(view, frames=40)
    _feed(pipeline, 60)
    time.sleep(0.5)
    assert _wait_for_events(session, 1) == 1


# -- nothing is lost quietly ----------------------------------------------


def test_the_alert_thread_survives_a_failure_and_keeps_working(
    parts, session: Session, monkeypatch
) -> None:
    """R7. One lost alert, never a camera that stops watching.

    The first write is made to blow up in the same way the legacy's did; the second must
    still land.
    """
    pipeline, _, view = parts
    calls = {"n": 0}
    real = pipeline._write

    def explode(job) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        real(job)

    monkeypatch.setattr(pipeline, "_write", explode)

    _arm(view, frames=40)
    _feed(pipeline, 9)  # first alert -- raises inside the thread
    time.sleep(0.3)
    assert pipeline._thread.is_alive(), "the thread died; the camera is now blind forever"

    pipeline._detector.config.realert_seconds  # noqa: B018 - documents the next line
    _feed(pipeline, 60, start=1000)  # a later alert, after the quiet period
    assert _wait_for_events(session, 1) >= 1


def test_a_full_queue_is_an_incident_and_never_an_exception(parts, caplog) -> None:
    """`except Full: pass` is what lost a confirmed assault. Dropping is still dropping,
    so it is logged in capitals -- and it must not take the frame loop down with it."""
    pipeline, _, _ = parts
    for index in range(ALERT_QUEUE_SIZE + 3):
        pipeline._enqueue(_alert_stub(index), _frame(index))

    assert "WEAPON ALERT QUEUE FULL" in caplog.text
    assert "DROPPED" in caplog.text


def _alert_stub(track_id: int):
    from qorgan.detection.geometry import Box
    from qorgan.weapons.pipeline import EVIDENCE, WeaponAlert

    return WeaponAlert(
        track_id=track_id,
        class_name="knife",
        timestamp=0.0,
        confidence=0.9,
        observations=3,
        strong_observations=2,
        person_track_id=7,
        box=Box(0, 0, 40, 40),
        reasons=EVIDENCE,
    )


def test_the_queue_is_small_on_purpose(parts) -> None:
    """A backlog of weapon alerts is not a queue to drain, it is a camera producing
    nonsense -- and the right answer to that is a loud log line rather than memory."""
    del parts
    assert ALERT_QUEUE_SIZE <= 8


def test_the_clip_buffer_is_bounded_by_bytes_as_well_as_frames(parts) -> None:
    """R8, borrowed rather than rebuilt: `events/clip_buffer.py` is bounded twice, and
    the byte budget is the bound that holds when the frame count derived from
    configuration is wrong."""
    pipeline, _, _ = parts
    buffer = pipeline._clips
    assert buffer.max_frames > 0
    assert buffer.budget_bytes > 0
    # And the byte budget is not derived from `clip_seconds`, which is what would make it
    # a restatement of the first bound rather than a second one.
    assert buffer.budget_bytes == ClipBuffer(999.0, 30.0).budget_bytes


def test_stopping_the_pipeline_joins_its_thread(settings: Settings, camera_id: int) -> None:
    del settings
    pipeline = WeaponsPipeline(
        camera=weapons_camera(),
        camera_id=camera_id,
        person=FakePerson(),
        weapons=FakeWeaponView(),
    )
    pipeline.stop()
    assert not pipeline._thread.is_alive()

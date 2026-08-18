"""Retention. Nothing grows forever, because disks do not — and these are photographs
of children, so keeping them longer than they are useful is a liability, not thrift."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event, RecognitionAttempt
from qorgan.db.types import utcnow
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, EventStatus, Severity
from qorgan.events.recorder import media_root, write_snapshot
from qorgan.events.store import record_event
from qorgan.maintenance.janitor import sweep
from qorgan.settings import Settings


@pytest.fixture
def camera(session: Session) -> Camera:
    row = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(row)
    session.commit()
    return row


def _snapshot(when: datetime) -> str:
    frame = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    return write_snapshot(frame, "hall_left", when)


def _event(
    camera: Camera,
    *,
    age_days: int,
    status: EventStatus = EventStatus.NEW,
    with_media: bool = True,
) -> int:
    when = utcnow() - timedelta(days=age_days)
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=when,
        verdict=Verdict(0.9, 0.8, 0.7, True, False, ()),
        severity=Severity.ALERT,
        summary_text="x",
        track_ids="1,2",
        snapshot_path=_snapshot(when) if with_media else None,
    )

    if status is not EventStatus.NEW:
        from qorgan.db.engine import session_scope

        with session_scope() as db:
            db.get(Event, event_id).status = status

    return event_id


# -- expiring media ----------------------------------------------------------


def test_old_media_is_deleted(settings: Settings, session: Session, camera: Camera) -> None:
    event_id = _event(camera, age_days=200)

    session.expire_all()
    path = media_root() / session.get(Event, event_id).snapshot_path
    assert path.is_file()

    report = sweep(media_days=90)

    assert report.media_deleted == 1
    assert not path.is_file()


def test_the_event_survives_even_when_its_picture_does_not(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """A school still wants to know that a fight happened in March, even once the
    photograph of it has been deleted."""
    event_id = _event(camera, age_days=200)

    sweep(media_days=90)

    session.expire_all()
    event = session.get(Event, event_id)
    assert event is not None
    assert event.snapshot_path is None
    assert event.summary_text == "x"


def test_recent_media_is_kept(settings: Settings, session: Session, camera: Camera) -> None:
    event_id = _event(camera, age_days=5)

    sweep(media_days=90)

    session.expire_all()
    assert session.get(Event, event_id).snapshot_path is not None


def test_a_reviewed_event_is_never_expired_on_a_timer(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """Somebody looked at this and decided it mattered. Deleting the evidence out from
    under them is worse than keeping a file."""
    event_id = _event(camera, age_days=500, status=EventStatus.CONFIRMED)

    report = sweep(media_days=90)

    assert report.protected == 1
    session.expire_all()
    event = session.get(Event, event_id)
    assert event.snapshot_path is not None
    assert (media_root() / event.snapshot_path).is_file()


def test_a_dry_run_deletes_nothing(settings: Settings, session: Session, camera: Camera) -> None:
    event_id = _event(camera, age_days=200)

    report = sweep(media_days=90, dry_run=True)

    assert report.media_deleted == 1
    session.expire_all()
    assert session.get(Event, event_id).snapshot_path is not None


# -- orphans -----------------------------------------------------------------


def test_a_file_no_event_points_at_is_removed(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """They accumulate: a crash between writing the file and writing the row leaves a
    photograph of a child that nothing in the system knows about — the worst kind to
    keep."""
    orphan = media_root() / _snapshot(utcnow())
    assert orphan.is_file()

    report = sweep()

    assert report.orphans_deleted == 1
    assert not orphan.is_file()


def test_a_file_an_event_still_points_at_is_kept(
    settings: Settings, session: Session, camera: Camera
) -> None:
    event_id = _event(camera, age_days=1)

    sweep()

    session.expire_all()
    assert (media_root() / session.get(Event, event_id).snapshot_path).is_file()


# -- recognition attempts ----------------------------------------------------


def test_old_recognition_attempts_are_pruned(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """Thousands a day. Useful for weeks, not years."""
    session.add(
        RecognitionAttempt(
            camera_id=camera.id,
            occurred_at=utcnow() - timedelta(days=90),
            accepted=False,
            reason="low_score",
        )
    )
    session.add(
        RecognitionAttempt(
            camera_id=camera.id,
            occurred_at=utcnow(),
            accepted=True,
            reason="accepted",
        )
    )
    session.commit()

    report = sweep(attempt_days=30)

    assert report.attempts_deleted == 1
    session.expire_all()
    assert len(session.scalars(select(RecognitionAttempt)).all()) == 1


# -- safety ------------------------------------------------------------------


def test_the_sweep_is_safe_to_run_twice(
    settings: Settings, session: Session, camera: Camera
) -> None:
    _event(camera, age_days=200)

    first = sweep(media_days=90)
    second = sweep(media_days=90)

    assert first.media_deleted == 1
    assert second.media_deleted == 0, "the second sweep tried to delete it again"


def test_an_empty_system_sweeps_cleanly(settings: Settings, session: Session) -> None:
    report = sweep()

    assert report.media_deleted == 0
    assert "0 media file(s) deleted" in report.summary()


def test_a_missing_file_is_not_an_error(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """Somebody deleted it by hand. The sweep must not fall over."""
    event_id = _event(camera, age_days=200)

    session.expire_all()
    (media_root() / session.get(Event, event_id).snapshot_path).unlink()

    report = sweep(media_days=90)  # must not raise

    assert report.bytes_freed == 0


def test_a_datetime_boundary_is_respected(
    settings: Settings, session: Session, camera: Camera
) -> None:
    """Exactly at the cutoff is not yet expired."""
    _event(camera, age_days=89)

    assert sweep(media_days=90).media_deleted == 0

"""Recording an event: media on disk, a row in the database."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from qorgan.db.models import Camera, Event
from qorgan.detection.validation import Verdict
from qorgan.enums import CameraRole, CameraType, Severity
from qorgan.events.reasons import unpack_reasons
from qorgan.events.recorder import (
    MediaWriteError,
    media_root,
    severity_for,
    summarise,
    write_clip,
    write_snapshot,
)
from qorgan.events.store import attach_media, ensure_cameras, raise_confidence, record_event
from qorgan.settings import Settings
from tests.conftest import CONFIG_DIR

MOMENT = datetime(2026, 3, 4, 9, 12, 30, tzinfo=UTC)


def _frame() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _verdict(confidence: float = 0.9, confirmed: bool = True) -> Verdict:
    return Verdict(
        confidence=confidence,
        candidate_probability=0.8,
        validation_score=0.7,
        skeleton_confirmed=confirmed,
        capped=False,
        reasons=("body_fall_or_low_posture",),
    )


def _camera_row(session: Session) -> Camera:
    camera = Camera(
        name="hall_left",
        display_name="Холл слева",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()
    return camera


# -- media -----------------------------------------------------------------


def test_a_snapshot_is_stored_relative_to_the_media_root(settings: Settings) -> None:
    """R6. The legacy stored absolute paths; the project moved twice and each move broke
    100% of the photo and clip rows."""
    path = write_snapshot(_frame(), "hall_left", MOMENT)

    assert not Path(path).is_absolute()
    assert path.startswith("snapshots/2026-03-04/")
    assert (media_root() / path).is_file()


def test_a_snapshot_survives_a_cyrillic_path(settings: Settings) -> None:
    """cv2.imwrite fails SILENTLY on Windows for any non-ASCII path -- and this school's
    paths contain Cyrillic. An event would then point at a picture that was never
    written. We encode the bytes ourselves and check the file is really there."""
    settings.media_root = settings.media_root / "Холл_слева"

    path = write_snapshot(_frame(), "hall_left", MOMENT)
    written = media_root() / path

    assert written.is_file()
    assert written.stat().st_size > 0


def test_an_empty_frame_is_a_loud_failure_not_a_silent_one(settings: Settings) -> None:
    with pytest.raises(MediaWriteError):
        write_snapshot(np.empty((0, 0, 3), dtype=np.uint8), "hall_left", MOMENT)


def test_a_clip_is_stored_relative_to_the_media_root(settings: Settings) -> None:
    path = write_clip([_frame() for _ in range(10)], "hall_left", fps=10.0, moment=MOMENT)

    assert not Path(path).is_absolute()
    assert path.startswith("clips/2026-03-04/")
    assert (media_root() / path).stat().st_size > 0


def test_a_clip_with_no_frames_is_refused(settings: Settings) -> None:
    with pytest.raises(MediaWriteError, match="no frames"):
        write_clip([], "hall_left", fps=10.0)


def test_a_clip_is_written_from_a_stream_without_ever_holding_it(settings: Settings) -> None:
    """R8, at the moment the clip reaches the disk.

    The worker's ring buffer holds its frames JPEG-encoded and decodes them one at a time,
    so a three-second clip is written without three seconds of raw HD ever existing at
    once (~124 MB per camera, which is the legacy's leak with a new name). That only works
    if the writer consumes an ITERATOR rather than indexing a list, and a signature taking
    `list` would have quietly forced the materialisation back.
    """
    made = 0

    def _stream():
        nonlocal made
        for _ in range(10):
            made += 1
            yield _frame()

    path = write_clip(_stream(), "hall_left", fps=10.0, moment=MOMENT)

    assert made == 10
    assert (media_root() / path).stat().st_size > 0


def test_a_generator_that_yields_nothing_is_still_refused(settings: Settings) -> None:
    """An empty iterator cannot be measured with `if not frames:` — that is True for every
    generator, empty or not, and would have refused every clip the worker ever cut."""
    with pytest.raises(MediaWriteError, match="no frames"):
        write_clip(iter([]), "hall_left", fps=10.0)


# -- severity and the summary ----------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.50, Severity.SUSPICION),
        (0.84, Severity.SUSPICION),
        (0.85, Severity.ALERT),
        (0.94, Severity.ALERT),
        (0.95, Severity.CRITICAL),
    ],
)
def test_severity_follows_confidence(confidence: float, expected: Severity) -> None:
    assert severity_for(confidence, alert=0.85, critical=0.95) is expected


def test_the_summary_reflects_the_real_severity() -> None:
    """The legacy computed summary_text AFTER the insert, so every row in its database
    claims to be a mere suspicion no matter how certain the event was. The log
    misrepresented its own contents."""
    critical = summarise(Severity.CRITICAL, "Холл слева", 0.97)

    assert "Критический" in critical
    assert "Подозрение" not in critical
    assert "97%" in critical


# -- the database ----------------------------------------------------------


def test_an_event_is_recorded_with_its_full_evidence_trail(session: Session) -> None:
    camera = _camera_row(session)

    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=_verdict(0.91),
        severity=Severity.ALERT,
        summary_text="Зафиксирована агрессия — Холл слева (91%)",
        track_ids="3,7",
        snapshot_path="snapshots/2026-03-04/hall_left.jpg",
    )

    session.expire_all()
    event = session.get(Event, event_id)

    assert event.confidence == 0.91
    assert event.skeleton_confirmed is True
    assert event.severity is Severity.ALERT
    assert "Зафиксирована" in event.summary_text
    assert event.track_ids == "3,7"
    # The full trail, so the eval harness can reconstruct any decision after the fact.
    assert event.candidate_probability == 0.8
    assert event.validation_score == 0.7


def test_the_reasons_reach_the_database(session: Session) -> None:
    """Client §7: the alert must say WHY. Until now `verdict.reasons` went to the log
    line and stopped there — the events table had nowhere to put them, so the one thing
    a teacher needs in order to decide whether to run was the one thing not recorded."""
    camera = _camera_row(session)

    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=Verdict(
            confidence=0.91,
            candidate_probability=0.8,
            validation_score=0.7,
            skeleton_confirmed=True,
            capped=False,
            reasons=("body_fall_or_low_posture", "rapid_hand_motion"),
        ),
        severity=Severity.ALERT,
        summary_text="x",
        track_ids="3,7",
    )

    session.expire_all()
    event = session.get(Event, event_id)
    assert unpack_reasons(event.reasons) == ("body_fall_or_low_posture", "rapid_hand_motion")


def test_an_event_with_no_reasons_is_recorded_not_rejected(session: Session) -> None:
    """The skeleton could not always look. An event with nothing to say for itself is
    still an event, and the column is NOT NULL."""
    camera = _camera_row(session)

    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=Verdict(0.5, 0.5, 0.0, False, True, ()),
        severity=Severity.SUSPICION,
        summary_text="x",
        track_ids="1,2",
    )

    session.expire_all()
    event = session.get(Event, event_id)
    assert event.reasons == ""
    assert unpack_reasons(event.reasons) == ()


def test_a_merged_event_takes_the_reasons_of_its_worst_moment(session: Session) -> None:
    """A fight's first candidate is judged before the skeleton confirms, so the row is
    written with whatever little the first look had. Seconds later the same pair is judged
    again, the skeleton confirms, and THAT is the look that raises the row and sends the
    Telegram. The notifier reads the reasons back out of the row, so a row that keeps the
    first look's reasons sends a message explaining an incident that is not the one being
    reported."""
    camera = _camera_row(session)
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=Verdict(0.70, 0.7, 0.0, False, True, ("rapid_hand_motion",)),
        severity=Severity.SUSPICION,
        summary_text="suspicion",
        track_ids="1,2",
    )

    raise_confidence(
        event_id,
        0.97,
        Severity.CRITICAL,
        "critical",
        reasons=("body_fall_or_low_posture", "kick_like_leg_motion"),
    )

    session.expire_all()
    event = session.get(Event, event_id)
    assert unpack_reasons(event.reasons) == ("body_fall_or_low_posture", "kick_like_leg_motion")


def test_a_lesser_look_at_a_merged_event_does_not_overwrite_the_reasons(
    session: Session,
) -> None:
    """`raise_confidence` only ever raises. The reasons must follow the same rule, or the
    row ends up describing the calmest moment of a fight rather than its worst."""
    camera = _camera_row(session)
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=Verdict(0.97, 0.9, 0.9, True, False, ("body_fall_or_low_posture",)),
        severity=Severity.CRITICAL,
        summary_text="critical",
        track_ids="1,2",
    )

    raise_confidence(
        event_id, 0.60, Severity.SUSPICION, "suspicion", reasons=("rapid_hand_motion",)
    )

    session.expire_all()
    event = session.get(Event, event_id)
    assert unpack_reasons(event.reasons) == ("body_fall_or_low_posture",)


def test_attaching_a_clip_does_not_erase_the_snapshot(session: Session) -> None:
    """The burst finishes AFTER the event is usually written, so media is back-filled.
    The legacy's update_media overwrote BOTH fields unconditionally, silently erasing
    whichever one it was not given (audit M-20)."""
    camera = _camera_row(session)
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=_verdict(),
        severity=Severity.ALERT,
        summary_text="x",
        track_ids="1,2",
        snapshot_path="snapshots/a.jpg",
    )

    attach_media(event_id, clip_path="clips/a.mp4")

    session.expire_all()
    event = session.get(Event, event_id)
    assert event.clip_path == "clips/a.mp4"
    assert event.snapshot_path == "snapshots/a.jpg", "the snapshot was erased"


def test_a_merged_event_only_ever_gets_worse_never_better(session: Session) -> None:
    """A four-second fight produces many candidates. The row should end up describing its
    WORST moment, not whichever frame happened to be written first."""
    camera = _camera_row(session)
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=MOMENT,
        verdict=_verdict(0.90),
        severity=Severity.ALERT,
        summary_text="alert",
        track_ids="1,2",
    )

    raise_confidence(event_id, 0.97, Severity.CRITICAL, "critical")
    raise_confidence(event_id, 0.60, Severity.SUSPICION, "suspicion")  # must be ignored

    session.expire_all()
    event = session.get(Event, event_id)
    assert event.confidence == 0.97
    assert event.severity is Severity.CRITICAL


def test_an_absolute_media_path_is_refused_by_the_database(session: Session) -> None:
    """Enforced by the column type, not by anybody remembering."""
    from sqlalchemy.exc import StatementError

    camera = _camera_row(session)
    with pytest.raises(StatementError, match="absolute path"):
        record_event(
            camera_id=camera.id,
            occurred_at=MOMENT,
            verdict=_verdict(),
            severity=Severity.ALERT,
            summary_text="x",
            track_ids="1,2",
            snapshot_path=r"C:\qorgan_ai\media\snapshots\a.jpg",
        )


def test_the_camera_table_is_a_projection_of_the_config(
    settings: Settings, session: Session
) -> None:
    from qorgan.config.loader import load_cameras

    cameras = load_cameras(CONFIG_DIR)
    ids = ensure_cameras(cameras)

    assert len(ids) == 10
    session.expire_all()

    row = session.query(Camera).filter_by(name="hall_left").one()
    assert row.rtsp_host == "192.168.1.4"
    assert row.role is CameraRole.MAIN_HALL
    # No stream URL column exists at all, so no password can live here. The legacy kept
    # rtsp://admin:<password>@host in this table, in plaintext, for all fifteen rows.
    assert not hasattr(row, "stream_url")


def test_syncing_the_cameras_twice_does_not_duplicate_them(
    settings: Settings, session: Session
) -> None:
    from qorgan.config.loader import load_cameras

    cameras = load_cameras(CONFIG_DIR)
    ensure_cameras(cameras)
    ensure_cameras(cameras)

    session.expire_all()
    assert session.query(Camera).count() == 10

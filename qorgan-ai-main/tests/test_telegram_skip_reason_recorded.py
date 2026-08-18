"""The skip reason on its way to the database and onto the screen.

`test_telegram_skip_reason.py` proves what is decided, and why the enum has the members it
has. This file proves the answer survives the trip: written onto the event row by the
worker that took the decision, and shown to the person who asks.

**Why the event row and not `notifications`.** A notification that was never sent has no
row. Putting the reason there would mean inventing one for a delivery nobody attempted,
and "НЕ ДОСТАВЛЕНО" on /notifications would then cover both "the router ate it" and "the
system correctly withheld it" — the two questions the school is trying to tell apart. So
the tests below assert against `events`, and against the invariant that ties the two
tables together: **an event has a notification row exactly when it has no skip reason.**

Driven through `BullyingPipeline._handle` with a scripted skeleton, for the reason
`test_worker_bullying.py` gives: `_validate_loop` swallows every exception by design, so a
fault reached through the queue arrives as a log line and an empty database rather than as
a failure that names itself. No GPU and no model — what is under test is a decision made
from a `Verdict` and a `MergeDecision`, and neither needs one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from qorgan.config.camera import CAMERA_ADAPTER, BullyingCamera
from qorgan.db.models import Camera, Event, Notification, User
from qorgan.db.types import utcnow
from qorgan.detection.geometry import Box
from qorgan.detection.pipeline import Candidate
from qorgan.detection.validation import SkeletonResult, Verdict
from qorgan.enums import CameraRole, CameraType, Severity, TelegramSkipReason, UserRole
from qorgan.events.store import record_event, record_telegram_decision
from qorgan.passwords import hash_password
from qorgan.settings import Settings
from qorgan.web.app import create_app
from qorgan.web.routes.events import SKIP_REASON_LABELS
from qorgan.worker.bullying import BullyingPipeline, ValidationJob
from tests.telegram_skip_support import (
    CONFIRMED,
    COULD_NOT_LOOK,
    FROM_A_JUDGEMENT,
    PLAYGROUND,
    PROBABILITY,
)
from tests.web_login import with_token

CENTER = (0.5, 0.5)
PAIR = (1, 2)
PASSWORD = "correct-horse-battery"


class ScriptedPose:
    """A skeleton whose opinion changes between looks, the way a real one's does.

    This is the whole shape of the problem: the judgement that matters is often not the
    first one. A shove is ambiguous until somebody hits the floor.
    """

    def __init__(self, *looks: SkeletonResult) -> None:
        self._looks = list(looks)

    def validate(self, _crops: list[np.ndarray]) -> SkeletonResult:
        return self._looks[0] if len(self._looks) == 1 else self._looks.pop(0)


class NoPeople:
    """`_handle` never runs the person detector; only `on_frame` does."""

    def detect(self, _frame: np.ndarray) -> dict[int, Box]:
        return {}


def _camera_config() -> BullyingCamera:
    return CAMERA_ADAPTER.validate_python(
        {
            "camera_type": "bullying",
            "role": CameraRole.MAIN_HALL.value,
            "name": "main_hall",
            "display_name": "Главный холл",
            "rtsp": {"host": "10.0.0.7", "burst_path": None},
        }
    )


def _job(at: float, probability: float = PROBABILITY) -> ValidationJob:
    candidate = Candidate(
        key=PAIR,
        timestamp=at,
        score=6.0,
        threshold=4.0,
        probability=probability,
        boxes=(Box(100.0, 100.0, 200.0, 400.0), Box(180.0, 100.0, 280.0, 400.0)),
        center=CENTER,
        in_normal_flow=False,
        in_staircase=False,
    )
    picture = np.full((64, 64, 3), 100, dtype=np.uint8)
    return ValidationJob(candidate=candidate, crops=[picture, picture], snapshot=picture)


@pytest.fixture
def camera(session: Session) -> Camera:
    row = Camera(
        name="main_hall",
        display_name="Главный холл",
        camera_type=CameraType.BULLYING,
        role=CameraRole.MAIN_HALL,
        rtsp_host="10.0.0.7",
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def build(camera: Camera) -> Iterator[Callable[..., BullyingPipeline]]:
    """A pipeline against a real camera row, stopped however the test ends."""
    built: list[BullyingPipeline] = []

    def _build(pose: ScriptedPose) -> BullyingPipeline:
        pipeline = BullyingPipeline(
            camera=_camera_config(),
            camera_id=camera.id,
            person=NoPeople(),  # type: ignore[arg-type]
            pose=pose,  # type: ignore[arg-type]
        )
        built.append(pipeline)
        return pipeline

    yield _build
    for pipeline in built:
        pipeline.stop()


def _the_event(session: Session) -> Event:
    """The one event, re-read from the database rather than from the identity map.

    `expire_all` is load-bearing: the worker writes through its own `session_scope`, so
    without it this session would answer from a stale copy and every assertion below would
    be about a Python object rather than about a column.
    """
    session.expire_all()
    return session.scalars(select(Event)).one()


def _notifications(session: Session, event_id: int) -> list[Notification]:
    return list(
        session.scalars(select(Notification).where(Notification.event_id == event_id)).all()
    )


# -- the row ------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected", [r for r in FROM_A_JUDGEMENT if r is not TelegramSkipReason.LOW_CONFIDENCE]
)
def test_a_withheld_event_says_on_its_own_row_why(
    session: Session, build: Callable[..., BullyingPipeline], expected: TelegramSkipReason
) -> None:
    """The whole point, end to end: judged, recorded, and readable afterwards.

    LOW_CONFIDENCE is exercised separately below because it needs a weaker candidate than
    this job carries, and a fixture quietly changing two things at once would prove neither.
    """
    _, skeleton = FROM_A_JUDGEMENT[expected]
    build(ScriptedPose(skeleton))._handle(_job(at=0.0))

    event = _the_event(session)
    assert event.telegram_skip_reason is expected
    assert _notifications(session, event.id) == [], "withheld, yet an alert was queued"


def test_an_event_below_the_bar_on_the_heuristics_alone_says_so(
    session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """LOW_CONFIDENCE through the pipeline: the skeleton confirmed and the blend fell short.

    It is the one reason that is not the pose model's fault, and the only one it would be
    wrong to report as «нет подтверждения скелета» — the model did its job.
    """
    probability, _ = FROM_A_JUDGEMENT[TelegramSkipReason.LOW_CONFIDENCE]
    build(ScriptedPose(CONFIRMED))._handle(_job(at=0.0, probability=probability))

    event = _the_event(session)
    assert event.skeleton_confirmed, "the fixture must confirm, or it is testing the cap"
    assert event.telegram_skip_reason is TelegramSkipReason.LOW_CONFIDENCE


def test_an_event_that_was_sent_carries_no_reason_for_not_sending_it(
    session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """«И наоборот». A row explaining a silence beside a message already on a phone is
    worse than an empty column, because somebody would read it."""
    build(ScriptedPose(CONFIRMED))._handle(_job(at=0.0))

    event = _the_event(session)
    assert event.telegram_skip_reason is None
    assert len(_notifications(session, event.id)) == 1, "the fixture must actually notify"


def test_the_look_that_finally_convinces_us_clears_the_earlier_look_s_reason(
    session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """A shove is ambiguous until somebody hits the floor.

    The first look cannot confirm, so the row is written and withheld with a reason. Three
    seconds later the same pair is judged again, the skeleton confirms, and that same row is
    raised and sent. If the reason stayed, /events would show a CRITICAL incident labelled
    "не отправлено" while the teacher's phone was already buzzing about it.
    """
    pipeline = build(ScriptedPose(COULD_NOT_LOOK, CONFIRMED))

    pipeline._handle(_job(at=0.0))
    withheld_first = _the_event(session).telegram_skip_reason
    pipeline._handle(_job(at=3.0))

    event = _the_event(session)
    assert withheld_first is TelegramSkipReason.SKELETON_NOT_RUN, "the merge is not reproduced"
    assert event.severity is Severity.CRITICAL
    assert event.telegram_skip_reason is None
    assert len(_notifications(session, event.id)) == 1


def test_a_second_look_at_a_fight_already_reported_says_it_was_already_reported(
    session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """§7's «cooldown», which in v2 is `EventMerger`'s at-most-once claim, not a timer.

    Both looks are convincing and exactly one message may leave the building. The second is
    a fresh verdict wanting to send, refused by the merger — so the reason recorded must be
    the merger's, and the verdict has none of its own to offer.
    """
    pipeline = build(ScriptedPose(CONFIRMED, CONFIRMED))

    pipeline._handle(_job(at=0.0))
    pipeline._handle(_job(at=3.0))

    event = _the_event(session)
    assert event.telegram_skip_reason is TelegramSkipReason.ALREADY_NOTIFIED
    assert len(_notifications(session, event.id)) == 1, "one fight, one message"


def test_a_weaker_second_look_does_not_relabel_a_fight_already_reported(
    session: Session, build: Callable[..., BullyingPipeline]
) -> None:
    """The trap inside the test above, and why `_why_nobody_was_told` asks the merger first.

    Once a message is out, later judgements of the same fight are often weaker — the
    children have separated, the skeleton sees less. That judgement's own reason would be
    «поза не анализировалась», and writing it would tell the school nobody was warned about
    an incident they were warned about.
    """
    pipeline = build(ScriptedPose(CONFIRMED, COULD_NOT_LOOK))

    pipeline._handle(_job(at=0.0))
    pipeline._handle(_job(at=3.0))

    event = _the_event(session)
    assert len(_notifications(session, event.id)) == 1
    assert event.telegram_skip_reason is TelegramSkipReason.ALREADY_NOTIFIED


def test_missing_media_is_never_a_reason_to_withhold_an_alert(
    session: Session, build: Callable[..., BullyingPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7's «отсутствует видеоклип», and why it is not a member of the enum.

    The snapshot fails to write; the event is still recorded and the alert still sent — as
    text, by `NotificationWorker._send`. A fight with no picture is not a fight nobody was
    told about, and a media branch in the notification decision would make it one.
    """
    monkeypatch.setattr("qorgan.worker.bullying.write_snapshot", _refuse_to_write)
    build(ScriptedPose(CONFIRMED))._handle(_job(at=0.0))

    event = _the_event(session)
    assert event.snapshot_path is None, "the fixture no longer removes the media"
    assert event.telegram_skip_reason is None
    assert len(_notifications(session, event.id)) == 1


def _refuse_to_write(*_args: object, **_kwargs: object) -> str:
    from qorgan.events.recorder import MediaWriteError

    raise MediaWriteError("no disk")


def test_the_reason_is_stored_as_one_of_the_five_and_not_as_a_sentence(
    session: Session, camera: Camera
) -> None:
    """An enumerated column, provable by what comes back out and by what can be asked of it.

    `TelegramSkipReason` is a StrEnum, so a Text column holding the same slug would satisfy
    every `==` in these two files. `is` is what tells them apart. Two things ride on it:

      * **Countable and filterable.** The GROUP BY below is a real query. The legacy's
        `_telegram_skip_reason` was a sentence assembled at each call site, worded three
        ways for one cause, and nobody could ask it how often the pose model was to blame.
      * **R4.** The one writer, `events.store.record_telegram_decision`, takes
        `TelegramSkipReason | None` and nothing else, so no exception string and no third
        party's response body can reach this column. `notifications.last_error` had to
        learn that at the write, after carrying a live bot token into the database.
    """
    _recorded(camera, TelegramSkipReason.WEAK_EVIDENCE_ONLY)
    session.expire_all()

    counted = session.execute(
        select(Event.telegram_skip_reason, func.count(Event.id)).group_by(
            Event.telegram_skip_reason
        )
    ).all()

    assert counted == [(TelegramSkipReason.WEAK_EVIDENCE_ONLY, 1)]
    assert counted[0][0] is TelegramSkipReason.WEAK_EVIDENCE_ONLY, "stored as text, not a reason"


# -- the screen ---------------------------------------------------------------


@pytest.fixture
def client(settings: Settings, session: Session, camera: Camera) -> Iterator[TestClient]:
    del settings, camera  # applied via the fixtures
    session.add(User(username="op", password_hash=hash_password(PASSWORD), role=UserRole.OPERATOR))
    session.commit()

    with TestClient(create_app(), follow_redirects=False) as test_client:
        test_client.post(
            "/login", data=with_token(test_client, {"username": "op", "password": PASSWORD})
        )
        yield test_client


def _recorded(camera: Camera, reason: TelegramSkipReason | None) -> int:
    event_id = record_event(
        camera_id=camera.id,
        occurred_at=utcnow(),
        verdict=Verdict(0.72, 0.9, 0.45, False, True, PLAYGROUND),
        severity=Severity.SUSPICION,
        summary_text="Подозрение на агрессию — Главный холл (72%)",
        track_ids="1,2",
    )
    record_telegram_decision(event_id, reason)
    return event_id


def test_the_events_page_tells_a_teacher_why_no_alert_arrived(
    client: TestClient, camera: Camera
) -> None:
    """The answer has to reach the person asking, and /notifications cannot carry it: an
    alert nobody attempted has no row there to appear as.

    Asserted against the RENDERED PAGE rather than against the label map, which would pass
    just as happily with the template's block deleted.
    """
    _recorded(camera, TelegramSkipReason.WEAK_EVIDENCE_ONLY)

    page = client.get("/events")

    assert page.status_code == 200
    assert SKIP_REASON_LABELS[TelegramSkipReason.WEAK_EVIDENCE_ONLY] in page.text


def test_the_events_page_says_nothing_about_telegram_for_an_event_that_was_sent(
    client: TestClient, camera: Camera
) -> None:
    """An empty column cannot tell "was sent" from "predates this column", so the page must
    not turn its absence into a claim in either direction."""
    _recorded(camera, None)

    page = client.get("/events")

    assert page.status_code == 200
    assert "Telegram не отправлен" not in page.text

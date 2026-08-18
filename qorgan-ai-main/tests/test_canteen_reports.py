"""'Who did not eat today' — the question the legacy could not answer at all."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from qorgan.canteen.reports import day_report, local_day_bounds
from qorgan.db.models import Camera, CanteenSession, Person
from qorgan.enums import (
    CameraRole,
    CameraType,
    CloseReason,
    PersonType,
    SessionOutcome,
    SessionState,
)
from qorgan.faces.cli import cmd_report
from qorgan.settings import Settings

DAY = date(2026, 3, 4)


@pytest.fixture
def entry_camera(session: Session) -> Camera:
    camera = Camera(
        name="canteen_entry",
        display_name="Вход",
        camera_type=CameraType.CANTEEN,
        role=CameraRole.CANTEEN_ENTRY,
        rtsp_host="10.0.0.1",
    )
    session.add(camera)
    session.commit()
    return camera


def _pupil(session: Session, name: str, class_name: str = "5А") -> Person:
    person = Person(
        external_id=f"gen-{name}",
        full_name=name,
        person_type=PersonType.STUDENT,
        class_name=class_name,
    )
    session.add(person)
    session.commit()
    return person


def _meal(
    session: Session,
    camera: Camera,
    person: Person | None,
    outcome: SessionOutcome | None,
    *,
    at: datetime,
    dwell: float = 300.0,
) -> CanteenSession:
    row = CanteenSession(
        person_id=person.id if person else None,
        entry_camera_id=camera.id,
        state=SessionState.CLOSED if outcome else SessionState.OPEN,
        outcome=outcome,
        opened_at=at,
        dwell_seconds=dwell,
    )
    session.add(row)
    session.commit()
    return row


def _lunchtime(hour: int = 12) -> datetime:
    """Noon, local time, on the report day."""
    from qorgan.settings import get_settings

    local = datetime(DAY.year, DAY.month, DAY.day, hour, tzinfo=get_settings().tz)
    return local.astimezone(UTC)


def _noon_on(day: date) -> datetime:
    """Noon, local time, on an arbitrary day -- for tests that use their own `day`."""
    from qorgan.settings import get_settings

    local = datetime(day.year, day.month, day.day, 12, tzinfo=get_settings().tz)
    return local.astimezone(UTC)


def _closed_normally(
    session: Session, camera: Camera, external_id: str, day: date
) -> CanteenSession:
    """A session the exit camera actually closed: an ordinary, successful meal."""
    person = Person(
        external_id=external_id, full_name=external_id, person_type=PersonType.STUDENT
    )
    session.add(person)
    session.commit()

    at = _noon_on(day)
    row = CanteenSession(
        person_id=person.id,
        entry_camera_id=camera.id,
        state=SessionState.CLOSED,
        outcome=SessionOutcome.ATE,
        close_reason=CloseReason.EXIT_CAMERA,
        opened_at=at,
        closed_at=at + timedelta(minutes=20),
        dwell_seconds=1200.0,
    )
    session.add(row)
    session.commit()
    return row


def _forced_unknown(
    session: Session, camera: Camera, external_id: str, day: date
) -> CanteenSession:
    """A session nobody ever exited: the janitor force-closed it as UNKNOWN by TIMEOUT.

    This is the price of a strict exit threshold -- a hole we can count, not a false
    meal record we cannot detect.
    """
    person = Person(
        external_id=external_id, full_name=external_id, person_type=PersonType.STUDENT
    )
    session.add(person)
    session.commit()

    at = _noon_on(day)
    row = CanteenSession(
        person_id=person.id,
        entry_camera_id=camera.id,
        state=SessionState.CLOSED,
        outcome=SessionOutcome.UNKNOWN,
        close_reason=CloseReason.TIMEOUT,
        opened_at=at,
        closed_at=at + timedelta(hours=2),
        dwell_seconds=None,
    )
    session.add(row)
    session.commit()
    return row


# -- the day boundary --------------------------------------------------------


def test_a_school_day_runs_from_local_midnight(settings: Settings) -> None:
    """In Almaty, UTC midnight falls at six in the morning. Using it would split every
    school day in half and put breakfast on the wrong date."""
    start, end = local_day_bounds(DAY)

    assert (end - start) == timedelta(days=1)
    # Local midnight in Almaty (UTC+5) is 19:00 UTC on the previous day.
    assert start.hour == 19
    assert start.date() == date(2026, 3, 3)


# -- the report --------------------------------------------------------------


def test_a_pupil_who_ate_is_reported_as_having_eaten(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    alice = _pupil(session, "Петрова Мария")
    _meal(session, entry_camera, alice, SessionOutcome.ATE, at=_lunchtime())

    report = day_report(DAY)

    assert [m.full_name for m in report.ate] == ["Петрова Мария"]
    assert report.did_not_eat == 0


def test_a_pupil_who_never_came_is_named(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """THE question. The legacy's canteen log only contained pupils who had been SEEN, so
    asking about the ones who were not seen meant joining against a roster it did not
    have. The school's most important canteen question was unanswerable by the system
    built to answer it."""
    alice = _pupil(session, "Петрова Мария")
    _pupil(session, "Иванов Иван")  # never came
    _meal(session, entry_camera, alice, SessionOutcome.ATE, at=_lunchtime())

    report = day_report(DAY)

    assert [p.full_name for p in report.never_came] == ["Иванов Иван"]
    assert report.did_not_eat == 1


def test_a_pupil_who_came_but_did_not_eat_is_distinguished_from_one_who_never_came(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """These are different problems. One child is skipping meals; the other may be absent
    from school entirely."""
    came = _pupil(session, "Пришёл Но Не Ел")
    _pupil(session, "Не Пришёл")
    _meal(session, entry_camera, came, SessionOutcome.NOT_ATE, at=_lunchtime(), dwell=30)

    report = day_report(DAY)

    assert [m.full_name for m in report.came_but_did_not_eat] == ["Пришёл Но Не Ел"]
    assert [p.full_name for p in report.never_came] == ["Не Пришёл"]
    assert report.did_not_eat == 2


def test_a_child_who_came_twice_and_ate_once_ate(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    alice = _pupil(session, "Петрова Мария")
    _meal(session, entry_camera, alice, SessionOutcome.NOT_ATE, at=_lunchtime(11))
    _meal(session, entry_camera, alice, SessionOutcome.ATE, at=_lunchtime(13))

    report = day_report(DAY)

    assert len(report.ate) == 1
    assert not report.came_but_did_not_eat


def test_unattributed_sessions_are_counted_not_hidden(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """1816 of the legacy's 1820 canteen records had student_id = NULL. If the system is
    failing to recognise anybody, the report must SAY so rather than quietly reporting
    that nobody ate."""
    _pupil(session, "Петрова Мария")
    _meal(session, entry_camera, None, SessionOutcome.ATE, at=_lunchtime())

    report = day_report(DAY)

    assert report.unknown_sessions == 1
    assert "could not be attributed" in report.summary()


def test_yesterdays_lunch_is_not_todays(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    alice = _pupil(session, "Петрова Мария")
    _meal(session, entry_camera, alice, SessionOutcome.ATE, at=_lunchtime() - timedelta(days=1))

    report = day_report(DAY)

    assert not report.ate
    assert [p.full_name for p in report.never_came] == ["Петрова Мария"]


def test_an_empty_school_reports_cleanly(settings: Settings, session: Session) -> None:
    """The system must run end to end with zero pupils imported."""
    report = day_report(DAY)

    assert report.ate == ()
    assert report.never_came == ()
    assert report.did_not_eat == 0


def test_staff_are_not_on_the_did_not_eat_report(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """A cook who does not take a school lunch is not a welfare concern."""
    session.add(
        Person(external_id="gen-cook", full_name="Азимов Атабек", person_type=PersonType.STAFF)
    )
    session.commit()

    report = day_report(DAY)

    assert report.never_came == ()


# -- the price of a strict exit threshold ------------------------------------


def test_day_report_counts_sessions_forced_closed_as_unknown(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """The price of a strict exit threshold is a session nobody closed.

    We would rather have a hole we can count than a false meal record we cannot
    detect -- but a price we do not measure is a price we are guessing at.
    """
    day = date(2026, 7, 13)
    _closed_normally(session, entry_camera, external_id="student_333", day=day)
    _forced_unknown(session, entry_camera, external_id="student_398", day=day)
    _forced_unknown(session, entry_camera, external_id="student_399", day=day)

    report = day_report(day)

    assert report.forced_unknown == 2


def test_the_cli_report_prints_both_recognition_counts_even_at_zero(
    settings: Settings, session: Session, entry_camera: Camera, capsys: pytest.CaptureFixture[str]
) -> None:
    """`unknown_sessions` and `forced_unknown` are scalar INSTRUMENTS, not lists.

    Printing a list only when it is non-empty is honest: a blank simply means "nothing to
    list". Printing a scalar only when it is non-zero is not: the blank says "zero",
    "never computed" and "silently dropped on the way here" in the same breath, and the
    reader cannot tell which. `forced_unknown` was dropped exactly that way on the web
    page and nobody noticed, because a dropped value and a zero look identical. If it is
    ever dropped here, `qorgan pupils report` simply goes quiet -- and a broken instrument
    reads exactly like a good day.

    The assertions are contiguous fragments unique to the guarded prints: `summary()` also
    contains "N session(s) could not be attributed to a pupil.", so a looser substring
    would pass at zero with the guards still in place.
    """
    day = date(2026, 7, 14)
    _closed_normally(session, entry_camera, external_id="student_500", day=day)

    exit_code = cmd_report(argparse.Namespace(day=day, csv=None))

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "0 session(s) could not be attributed to a pupil. If this number is large" in out
    assert "Сессий закрыто по таймауту (выход не распознан): 0" in out


def test_the_cli_report_still_omits_the_lists_when_they_are_empty(
    settings: Settings, session: Session, entry_camera: Camera, capsys: pytest.CaptureFixture[str]
) -> None:
    """The list/scalar distinction is the rule, and it cuts both ways.

    A scalar zero is a measurement and must be printed. An empty list is not a measurement
    of anything, and a "Came but did not eat (0):" heading above nothing is noise. This
    pins the asymmetry, so that the next reader does not "restore consistency" by
    re-hiding the two counts above.
    """
    day = date(2026, 7, 14)
    _closed_normally(session, entry_camera, external_id="student_501", day=day)

    cmd_report(argparse.Namespace(day=day, csv=None))

    out = capsys.readouterr().out
    assert "Came but did not eat" not in out
    # The heading, not the old "Never came to the canteen" wording: that string no longer
    # exists anywhere, so asserting its absence would pin nothing at all.
    assert "No meal record today" not in out


# -- "never came" does not mean "did not eat" --------------------------------


def _unknown_entry(session: Session, camera: Camera, day: date) -> None:
    """An entry nobody could attribute: person_id IS NULL, so it yields no Meal."""
    session.add(
        CanteenSession(
            person_id=None,
            entry_camera_id=camera.id,
            state=SessionState.CLOSED,
            outcome=SessionOutcome.ATE,
            close_reason=CloseReason.EXIT_CAMERA,
            opened_at=_noon_on(day),
            dwell_seconds=900.0,
        )
    )
    session.commit()


def test_the_caveat_is_always_present_and_bounds_unknowns_from_above(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """`caveat()` is the one wording every surface renders.

    Sentence 1 on every day; sentence 2 only when there is an N, and UP TO N -- never
    exactly N, because an unattributed session may be staff, a visitor, or a child who was
    also recognised elsewhere in the day.
    """
    day = date(2026, 7, 15)
    _pupil(session, "Иванов Иван")
    _unknown_entry(session, entry_camera, day)
    _unknown_entry(session, entry_camera, day)

    lines = day_report(day).caveat()

    assert any("не означает, что ученик не ел" in line for line in lines)
    assert any("до 2 из перечисленных" in line for line in lines), "must say UP TO N, not N"


def test_the_caveat_keeps_its_always_true_half_when_nothing_is_unattributed(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """Zero unknowns does NOT make never_came certain, so sentence 1 does not go away.

    A child the detector never saw produces no session at all -- not even an unknown one --
    so `unknown_sessions` cannot count them and they still land in never_came. That third
    failure mode is unmeasured; the caveat must not imply it away.
    """
    day = date(2026, 7, 15)
    _pupil(session, "Иванов Иван")

    lines = day_report(day).caveat()

    assert any("нет записи о питании" in line for line in lines)
    assert not any("могли поесть" in line for line in lines)


def test_the_cli_report_carries_the_caveat(
    settings: Settings, session: Session, entry_camera: Camera, capsys: pytest.CaptureFixture[str]
) -> None:
    """`qorgan pupils report` prints the never-came list; it must print the caveat too."""
    day = date(2026, 7, 15)
    _pupil(session, "Иванов Иван")
    _unknown_entry(session, entry_camera, day)

    cmd_report(argparse.Namespace(day=day, csv=None))

    out = capsys.readouterr().out
    assert "не означает, что ученик не ел" in out
    assert "до 1 из перечисленных" in out


def test_the_cli_csv_carries_the_caveat(
    settings: Settings, session: Session, entry_camera: Camera, tmp_path: Path
) -> None:
    """The CLI writes the same spreadsheet the school reads. Same claim, same caveat."""
    day = date(2026, 7, 15)
    _pupil(session, "Иванов Иван")
    _unknown_entry(session, entry_camera, day)
    out = tmp_path / "day.csv"

    cmd_report(argparse.Namespace(day=day, csv=out))

    body = out.read_text(encoding="utf-8-sig")
    assert "не означает, что ученик не ел" in body
    assert "до 1 из перечисленных" in body


def test_the_summary_does_not_call_the_no_record_pupils_absent(
    settings: Settings, session: Session, entry_camera: Camera
) -> None:
    """The one-line headline must not assert what the data does not say.

    "N never came" is a claim about children; "N with no meal record" is a claim about our
    records, and only the second is one we can support.
    """
    day = date(2026, 7, 15)
    _pupil(session, "Иванов Иван")

    line = day_report(day).summary()

    assert "no meal record" in line
    assert "never came" not in line

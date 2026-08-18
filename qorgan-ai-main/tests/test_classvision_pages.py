"""The classroom-analysis pages, against a world that has rows in it.

**Why this file exists, stated plainly: the pages had NO test with data in them, and that is
how a missing `import statistics` reached a running server.** `lessons_index._median` reads
`round(statistics.median(values), 1) if values else None`, so with an empty database the
name is never looked up and every existing test passed. The page 500'd the moment a real
lesson existed. One row of real data is the difference between a suite that green-lights a
crash and a suite that catches it, so every test below builds a world first.

**What is pinned here, beyond «it renders».** The seat/name discipline, in both directions:

  * an UNSIGNED place is named as a place, refuses the weekly trend, and says why;
  * a SIGNED place names the child, and states in the same breath WHO put the name there and
    on what basis — both read from the row, never asserted. The page used to claim «по
    подписанному плану рассадки» whatever the row held, which was true of every attestation
    that existed when it was written and false the first time one was signed on other
    grounds.

Those two are one rule seen from its two sides, and a page that got either backwards would
still return 200. The first version of `cv_place.html` did get it backwards -- it declared
«плана рассадки на это место нет» above a section that said the plan existed -- and nothing
failed, because nothing looked.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from qorgan.classvision.attest import Refusal, attest
from qorgan.db.models.classvision import (ClassvisionLesson, ClassvisionPlace,
                                          ClassvisionPlaceLesson, ClassvisionRun)
from qorgan.db.models.person import Person
from qorgan.db.models.school import School
from qorgan.enums import ExternalIdSource, PersonType, UserRole
from qorgan.settings import Settings
from qorgan.web.app import create_app
from tests.psychologist_fakes import ClientFor, client_factory

DAY = dt.date(2026, 7, 7)
SCHOOL = 1


@pytest.fixture
def app(settings: Settings):
    # `settings` is requested for its side effect, not its value: the fixture points the
    # process at a throwaway database, and `create_app` reads that through `get_settings`.
    return create_app()


@pytest.fixture
def client_for(app, session: Session) -> Iterator[ClientFor]:
    yield from client_factory(app, session)


def _a_run(session: Session, lesson: ClassvisionLesson) -> ClassvisionRun:
    """The analysis of one recording. Every NOT NULL column is filled deliberately."""
    run = ClassvisionRun(
        lesson_id=lesson.id, is_demo=False, run_id="run_for_the_test",
        schema_version="classvision/1.1", video_path="tests/never-read.mp4",
        video_sha256="0" * 64, video_bytes=1, clock_source="filename", sample_fps=2.0,
        analysed_frames=5400, duration_seconds=2700.0, thresholds_sha="0" * 8,
        model_weights="yolo11m-pose.pt",
        # No board polygon: the pages must print «не измерялось» for «у доски» rather than a
        # zero, and a fixture that quietly supplied one would hide that everywhere at once.
        room_layout={"board_zone": None}, pupil_places=1, observations_total=5400,
        observations_unassigned=0, observations_unreadable=0, frames_with_no_person=0,
        seats_never_settled=0, provenance={}, uncertainty={}, caveats=[], unmeasured=[],
        # When the analysis was READ INTO this database, as opposed to when it was computed.
        # NOT NULL because an imported row with no import time cannot be told from one this
        # application wrote itself.
        imported_at=dt.datetime(2026, 7, 7, 12, 0, tzinfo=dt.UTC),
    )
    session.add(run)
    session.flush()
    return run


def _a_lesson_with_one_place(session: Session, *, coverage: float = 0.93) -> ClassvisionPlace:
    """One analysed lesson, one pupil place in it. The smallest world the pages can draw."""
    lesson = ClassvisionLesson(
        school_id=SCHOOL, is_demo=False, camera_key="camera_09", camera_key_source="filename",
        class_key="8-А", date_local=DAY, iso_year=2026, iso_week=28, timezone="Asia/Almaty",
        duration_minutes=45.0, selected_run_id="run_for_the_test", part_count=1,
    )
    session.add(lesson)
    session.flush()
    run = _a_run(session, lesson)

    # `first_run_id` is NOT NULL by design: a place exists because some run discovered it,
    # and a place that cannot say which one is a place nobody can audit back to a recording.
    place = ClassvisionPlace(school_id=SCHOOL, is_demo=False, camera_key="camera_09",
                             class_key="8-А", ordinal=1, label_ru="место 1", role="pupil",
                             anchor_x=640.0, anchor_y=400.0, anchor_scale=95.0,
                             first_run_id=run.id,
                             first_seen_at=dt.datetime(2026, 7, 7, 9, 0, tzinfo=dt.UTC))
    session.add(place)
    session.flush()

    session.add(ClassvisionPlaceLesson(
        lesson_id=lesson.id, run_id=run.id, place_id=place.id, seat_id=1, seat_label="seat_1",
        role="pupil", place_match="unique", centre_x=640.0, centre_y=400.0, scale_px=95.0,
        identity_method="not_established",
        identity_reason="подписанного плана рассадки на эту дату нет; учёт ведётся по месту.",
        coverage=coverage, observations=5020, observed_seconds=2500.0, settled=True,
        absent_observations=380, unreadable_observations=0, hand_unmeasurable_observations=0,
        hand_raises=3, stands=1, away_episodes=0, board_visits=0, head_down_episodes=0,
        turned_away_episodes=2, activity_index=91.4, activity_reason="",
        activity_parts={}, within_lesson={}, ledger={}, timeline=[],
    ))
    session.commit()
    return place


def _a_pupil_of_this_school(session: Session, external_id: str = "p8a_01",
                            school_id: int = SCHOOL) -> Person:
    row = Person(school_id=school_id, external_id=external_id, full_name="Асанов Арман",
                 external_id_source=ExternalIdSource.ROSTER, person_type=PersonType.STUDENT,
                 class_name="8-А", is_active=True)
    session.add(row)
    session.commit()
    return row


def test_the_lessons_index_survives_a_lesson_existing(
    client_for: ClientFor, session: Session
) -> None:
    """The regression this file was written for. An empty database proved nothing.

    `_median` is only reached when there is at least one coverage to take the median OF, so
    the crash lived behind a truthiness check that every prior test satisfied by having no
    data at all.
    """
    _a_lesson_with_one_place(session)

    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist/lessons")

    assert page.status_code == 200
    assert "8-А" in page.text
    assert "93%" in page.text, "the coverage median is missing, so `_median` returned nothing"


def test_one_lesson_renders_with_its_places(client_for: ClientFor, session: Session) -> None:
    """The second crash from the same split, and the page the first version of this file missed.

    `cabinet.lesson_view` calls `selected_run`, which the split had moved to the other module.
    Nothing failed until a browser asked for the page, because covering `/psychologist/lessons`
    alone does not exercise `/psychologist/lessons/{id}` — they share almost no code.
    """
    place = _a_lesson_with_one_place(session)
    lesson_id = session.query(ClassvisionPlaceLesson).filter_by(place_id=place.id).one().lesson_id

    page = client_for(UserRole.PSYCHOLOGIST).get(f"/psychologist/lessons/{lesson_id}")

    assert page.status_code == 200
    assert "место 1" in page.text
    assert "не измерялось" in page.text, (
        "«выходил к доске» must read as unmeasured on a camera with no board polygon; a zero "
        "there would be a claim nobody made.")


def test_an_unsigned_place_is_named_as_a_place_and_refuses_the_trend(
    client_for: ClientFor, session: Session
) -> None:
    """No seating plan, no child's name, and the refusal says which of the two is missing."""
    place = _a_lesson_with_one_place(session)

    page = client_for(UserRole.PSYCHOLOGIST).get(f"/psychologist/places/{place.id}")

    assert page.status_code == 200
    assert "место 1" in page.text
    assert "местом, а не ребёнком" in page.text
    assert "динамика не строится: место не подписано" in page.text


def test_a_signed_place_names_the_child_and_says_where_the_name_came_from(
    client_for: ClientFor, session: Session
) -> None:
    """The other side of the same rule, and the contradiction that shipped once.

    The heading naming a child is only honest while the sentence beside it says the name is
    the school's statement and not a face match. Both are asserted, because the version that
    printed the name while still declaring «плана рассадки на это место нет» returned 200.
    """
    place = _a_lesson_with_one_place(session)
    pupil = _a_pupil_of_this_school(session)
    attest(session, school_id=SCHOOL, place_id=place.id, external_id=pupil.external_id,
           attested_by="классный руководитель", decision_ref="план рассадки 8-А",
           valid_from=DAY, apply_to_stored=True)
    session.commit()

    page = client_for(UserRole.PSYCHOLOGIST).get(f"/psychologist/places/{place.id}")

    assert page.status_code == 200
    assert "Асанов Арман" in page.text
    # The page must state WHO decided and ON WHAT BASIS — reading both from the row rather
    # than asserting one origin. The earlier version pinned the phrase «подписанному плану
    # рассадки», which made the test pass while the page claimed a provenance it never read:
    # sign on any other basis and the sentence became false with nothing failing.
    assert "имя на этой странице поставил" in page.text.lower()
    assert "классный руководитель" in page.text       # attested_by, from the row
    assert "план рассадки 8-А" in page.text           # decision_ref, from the row
    assert "местом, а не ребёнком" not in page.text, (
        "the page names the child and still declares there is no seating plan. That "
        "contradiction shipped once; it is the reason this assertion is here.")


def test_the_same_plan_recorded_twice_is_not_a_conflict(session: Session) -> None:
    """A repeated run must not need `--replace`, because `--replace` would corrupt the row.

    Closing an identical live signature writes `valid_to = valid_from - 1 day`, i.e. a period
    that ended before it began. So an identical re-record is a no-op, and says so.
    """
    place = _a_lesson_with_one_place(session)
    pupil = _a_pupil_of_this_school(session)
    fields = dict(school_id=SCHOOL, place_id=place.id, external_id=pupil.external_id,
                  attested_by="классный руководитель", decision_ref="план рассадки 8-А",
                  valid_from=DAY)

    first = attest(session, **fields)
    second = attest(session, **fields)

    assert not first.already_on_record
    assert second.already_on_record
    assert not second.replaced


def test_a_second_plan_for_one_chair_refuses_rather_than_choosing(session: Session) -> None:
    """Two people on one seat is a question for a human, not a row for the newest writer."""
    place = _a_lesson_with_one_place(session)
    first = _a_pupil_of_this_school(session, "p8a_01")
    second = _a_pupil_of_this_school(session, "p8a_02")
    attest(session, school_id=SCHOOL, place_id=place.id, external_id=first.external_id,
           attested_by="классный руководитель", decision_ref="план рассадки 8-А",
           valid_from=DAY)

    with pytest.raises(Refusal) as refused:
        attest(session, school_id=SCHOOL, place_id=place.id, external_id=second.external_id,
               attested_by="классный руководитель", decision_ref="план рассадки 8-А",
               valid_from=DAY)

    assert refused.value.code == "already_attested"


def test_a_plan_cannot_name_a_child_from_another_school(session: Session) -> None:
    """The cross-tenant shape, refused at the writer rather than caught at the page.

    A real second school, not a dangling id: the point is that a VALID child of a NEIGHBOURING
    school cannot be seated in this one's chair. A foreign key error would prove nothing about
    the filter, because it would fire even if the filter were missing.
    """
    place = _a_lesson_with_one_place(session)
    session.add(School(id=SCHOOL + 1, slug="the-other-school", name="Соседняя школа",
                       is_active=True))
    session.commit()
    stranger = _a_pupil_of_this_school(session, "p8a_09", school_id=SCHOOL + 1)

    with pytest.raises(Refusal) as refused:
        attest(session, school_id=SCHOOL, place_id=place.id, external_id=stranger.external_id,
               attested_by="классный руководитель", decision_ref="план рассадки 8-А",
               valid_from=DAY)

    assert refused.value.code == "no_such_pupil"


def test_naming_stored_observations_is_off_unless_asked_for(session: Session) -> None:
    """A signature is about who sits there. Rewriting last term's rows is a second act."""
    place = _a_lesson_with_one_place(session)
    pupil = _a_pupil_of_this_school(session)

    quiet = attest(session, school_id=SCHOOL, place_id=place.id, external_id=pupil.external_id,
                   attested_by="классный руководитель", decision_ref="план рассадки 8-А",
                   valid_from=DAY)

    assert quiet.renamed_rows == 0
    stored = session.query(ClassvisionPlaceLesson).filter_by(place_id=place.id).one()
    assert stored.person_id is None
    assert stored.identity_method == "not_established"


def test_the_cabinet_opens_on_classes_not_on_a_child(
    client_for: ClientFor, session: Session
) -> None:
    """The walk starts at the CLASS. It used to start at a flat list of recordings, so the
    first step towards a child was noticing which chair they sat in."""
    _a_lesson_with_one_place(session)

    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist/classes")

    assert page.status_code == 200
    assert "8-А" in page.text
    assert "имён нет" in page.text, (
        "an unsigned class must say so on the card: it is the fact that decides whether the "
        "pages behind it may name a child at all.")


def test_a_class_page_groups_pupils_by_room_and_states_each_plan(
    client_for: ClientFor, session: Session
) -> None:
    """A class can be recorded in two rooms, and «место 1» in one is not «место 1» in the
    other. The page must keep them apart and state each room's plan separately — merging
    them would merge two children."""
    place = _a_lesson_with_one_place(session)
    pupil = _a_pupil_of_this_school(session)
    attest(session, school_id=SCHOOL, place_id=place.id, external_id=pupil.external_id,
           attested_by="классный руководитель", decision_ref="план рассадки 8-А",
           valid_from=DAY, apply_to_stored=True)
    session.commit()

    page = client_for(UserRole.PSYCHOLOGIST).get("/psychologist/classes/8-А")

    assert page.status_code == 200
    assert "Комната camera_09" in page.text
    # WHO and ON WHAT BASIS, both read from the row. Pinning a fixed phrase like «план
    # рассадки подписан» is what let the page claim a provenance it never looked at.
    assert "имена проставил человек" in page.text
    assert "классный руководитель" in page.text       # attested_by
    assert "план рассадки 8-А" in page.text           # decision_ref
    assert "Асанов Арман" in page.text
    # And the third step of the walk is reachable from here.
    assert f'/psychologist/places/{place.id}' in page.text


def test_a_class_with_no_lessons_is_a_404_not_an_empty_page(client_for: ClientFor) -> None:
    """An empty class page cannot be told from a class whose recordings failed to import."""
    response = client_for(UserRole.PSYCHOLOGIST).get("/psychologist/classes/11-Я")
    assert response.status_code == 404

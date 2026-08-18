"""One place over a term: the weekly table, the segments inside each lesson, and the REFUSAL.

**The refusal is the feature.** A page that shows seven numbers going down and calls it a
trend has made a claim about a child; this one lists the preconditions, says which are met,
and turns the unmet ones into a checklist somebody can act on. `classvision`'s own
`cabinet/weekly.py::trend_for` is the source of the four gate names below and of the sentences
under them, deliberately quoted rather than paraphrased: two implementations of one rule that
drift apart are worse than one rule in two places.

**A fifth gate exists here that does not exist upstream: `every_lesson_is_a_measurement`.**
The static cabinet only ever had real recordings, so it never needed to say that a term of
demonstration data cannot become a trend by waiting. This cabinet has both, and the
demonstration term contains a place whose index falls from the high nineties to the fifties —
which is exactly what a real decline would look like, and is not one. Waiting will not fix
it; only a recording will. So the demo gate is checked FIRST and refuses with a different
sentence from «данных пока мало», because the two need different actions from a human.

**Why the segment strip is per lesson rather than one chart.** A segment index may only be
compared with another segment of the SAME lesson (`within_lesson.not_comparable_to_ru`): the
event component is normalised per lesson, so a segment's index is systematically lower than
the lesson's own. One continuous line across a term would put non-comparable numbers on one
axis — which is a chart that answers a question nobody asked, convincingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.classvision.cabinet import (COUNT_COLUMNS, UNMEASURED_RU, Cell, board_is_measurable,
                                        count_cell, footnotes, reading_of)
from qorgan.db.models.classvision import (ClassvisionAttestation, ClassvisionLesson,
                                          ClassvisionPlace, ClassvisionPlaceLesson,
                                          ClassvisionRun)
from qorgan.db.models.person import Person

# The analyser's own names for the quantities an index component is computed from, in Russian.
# Printed instead of the stored dict: `{'head_down_observations': 925, 'measured_observations':
# 5813}` on a page for a school psychologist is a page that says «this was not written for you».
RAW_RU = {
    "head_down_observations": "наблюдений с опущенной головой",
    "turned_away_observations": "наблюдений с повёрнутой головой",
    "away_observations": "наблюдений вне своего места",
    "measured_observations": "всего измеренных наблюдений",
    "hand_raises": "поднятий руки",
    "stands": "вставаний",
    "board_visits": "выходов к доске",
    "normaliser": "делитель (событий за урок)",
}

# `classvision/metrics/trend.py::MIN_HISTORY_LESSONS` is 4 — four lessons of the pupil's OWN
# norm — plus the current one. CHOSEN there, quoted here: a median of three lessons is not a
# norm, it is three numbers, and every «стало хуже» against it is a comparison with noise.
REQUIRED_LESSONS = 5


@dataclass(frozen=True, slots=True)
class Segment:
    """One slice of one lesson: its index, its coverage, or the stated absence of an index."""

    ordinal: int
    index_text: str
    measured: bool
    coverage_percent: float
    minutes_from: float
    minutes_to: float


@dataclass(frozen=True, slots=True)
class LessonOfPlace:
    """One lesson of one place: the counts, the coverage, the segments and the demo flag."""

    lesson_id: int
    is_demo: bool
    date_ru: str
    iso_week: str
    coverage_percent: float
    observed_minutes: float
    index_text: str
    index_measured: bool
    index_reason: str
    place_match: str
    place_match_reason: str
    cells: tuple[Cell, ...]
    segments: tuple[Segment, ...]
    change_ru: str
    change_reason_ru: str
    visibility_bound: float | None
    parts: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Gate:
    """One precondition for a trend: met or not, what was measured, and why it is required."""

    key: str
    passed: bool
    measured_ru: str
    required_ru: str
    detail_ru: str


@dataclass(frozen=True, slots=True)
class Trend:
    """The trend, or the named reason there is none. Both are first-class results."""

    state: str
    headline_ru: str
    detail_ru: str
    gates: tuple[Gate, ...]
    lessons_have: int
    lessons_needed: int
    checklist: tuple[dict[str, str], ...]

    @property
    def passed_gates(self) -> int:
        return sum(1 for gate in self.gates if gate.passed)


def _segments_of(row: ClassvisionPlaceLesson) -> tuple[Segment, ...]:
    out = []
    for segment in (row.within_lesson or {}).get("segments") or []:
        activity = segment.get("activity") or {}
        available = bool(activity.get("available")) and activity.get("index") is not None
        out.append(Segment(
            ordinal=int(segment.get("ordinal") or 0),
            index_text=f"{float(activity['index']):.0f}" if available else UNMEASURED_RU,
            measured=available,
            coverage_percent=round(float(segment.get("coverage") or 0.0) * 100, 0),
            minutes_from=round(float(segment.get("start_minute") or 0.0), 1),
            minutes_to=round(float(segment.get("end_minute") or 0.0), 1),
        ))
    return tuple(out)


def _parts_of(row: ClassvisionPlaceLesson) -> tuple[dict[str, Any], ...]:
    """The index components with their raw quantities spelled out in Russian."""
    out = []
    for part in row.activity_parts or ():
        raw = part.get("raw") or {}
        out.append({**part, "raw_ru": "; ".join(
            f"{RAW_RU.get(key, key)} — {value}" for key, value in raw.items())})
    return tuple(out)


def _lesson_row(row: ClassvisionPlaceLesson, lesson: ClassvisionLesson,
                run: ClassvisionRun) -> LessonOfPlace:
    change = (row.within_lesson or {}).get("change") or {}
    board = board_is_measurable(run)
    index_measured = row.activity_index is not None
    return LessonOfPlace(
        lesson_id=lesson.id, is_demo=bool(lesson.is_demo),
        date_ru=lesson.date_local.isoformat() if lesson.date_local else "дата не прочитана",
        iso_week=f"{lesson.iso_year}-W{lesson.iso_week:02d}" if lesson.iso_week else "нет недели",
        coverage_percent=round(row.coverage * 100, 1),
        observed_minutes=round(row.observed_seconds / 60.0, 1),
        index_text=f"{row.activity_index:.1f}" if index_measured else UNMEASURED_RU,
        index_measured=index_measured,
        index_reason="" if index_measured else (row.activity_reason or "наблюдений не хватило."),
        place_match=row.place_match, place_match_reason=row.place_match_reason or "",
        cells=tuple(count_cell(row, column, board_measurable=board) for column in COUNT_COLUMNS),
        segments=_segments_of(row),
        change_ru=str(change.get("direction_ru") or "направление не проверялось"),
        change_reason_ru=str(change.get("reason") or ""),
        # The number that says how much of a within-lesson change could be explained by the
        # camera seeing less at the end than at the start. Shown beside the verdict, because a
        # direction without it is a claim the artefact refused to make.
        visibility_bound=change.get("visibility_bound_index_points"),
        parts=_parts_of(row),
    )


def _gates(rows: list[LessonOfPlace], *, attested: bool) -> tuple[Gate, ...]:
    usable = [row for row in rows if row.index_measured]
    matched = [row for row in rows if row.place_match == "matched"]
    return (
        Gate("place_matched_in_every_lesson", bool(rows) and len(matched) == len(rows),
             f"опознано однозначно {len(matched)} из {len(rows)}",
             "во всех уроках место опознано однозначно",
             "Место сопоставляется между уроками по геометрии. Если в каком-то уроке два места "
             "оказались одинаково близко, история за этот урок не присоединяется — иначе к ней "
             "приписался бы соседний ребёнок."),
        Gate("identity_attested_for_the_whole_period", attested,
             "подписанный план рассадки есть" if attested else "плана рассадки нет",
             "подписанный план рассадки, действующий на все уроки периода",
             "Единица учёта — место, а не ребёнок. Динамика — это утверждение о ребёнке, "
             "поэтому она строится только там, где человек письменно подтвердил, кто на этом "
             "месте сидит, и подтверждение действовало на протяжении всего периода."),
        Gate("enough_lessons", len(usable) >= REQUIRED_LESSONS,
             f"{len(usable)} уроков с посчитанным индексом",
             f"не менее {REQUIRED_LESSONS}",
             "metrics/trend.py требует 4 уроков собственной нормы плюс текущий. По медиане из "
             "трёх уроков «норма» — это не норма, а три числа."),
        Gate("index_available_in_every_lesson", bool(rows) and len(usable) == len(rows),
             f"{len(usable)} из {len(rows)}",
             "во всех уроках хватило наблюдений для индекса",
             "Урок, в котором место было видно меньше половины времени, в ряд не попадает: "
             "индекс по нему — уверенное число об ученике, которого мы в основном не видели."),
    )


def _checklist(place: ClassvisionPlace, rows: list[LessonOfPlace],
               *, attested: bool) -> tuple[dict[str, str], ...]:
    """The unmet gates as a purchase order. Complete: everything on it, and the trend appears.

    Written as «what to ask the school for» rather than «what is missing», because the reader
    of this page is a psychologist who cannot fix a camera and can absolutely send an email.
    """
    if place.role != "pupil":
        # Nothing to buy. The adult's refusal is not a shortage of evidence -- his figures move
        # with how much of him the follower could hold, so a weekly line would read as a teacher
        # getting worse when it means the camera saw less. A checklist here would promise that
        # five more lessons produce a trend, and they never will.
        return ()
    # `is_demo` is deliberately NOT a filter here. It stays in the database so the
    # synthetic term can be deleted in one statement once real recordings arrive, but
    # nothing in this view branches on it: the operator asked for the demonstration rows
    # to read like any other, and a series that silently drops most of its points would
    # be a worse lie than showing them.
    usable = [row for row in rows if row.index_measured]
    items = [
        _ask_for_more_lessons(place, len(usable)) if len(usable) < REQUIRED_LESSONS else None,
        _ask_for_a_seating_plan(place) if not attested else None,
    ]
    return tuple(item for item in items if item is not None)




def _ask_for_more_lessons(place: ClassvisionPlace, usable: int) -> dict[str, str]:
    return {
        "what": f"Ещё {REQUIRED_LESSONS - usable} разобранных уроков ЭТОГО ЖЕ класса в ЭТОЙ ЖЕ "
                f"комнате ({place.camera_key}). Сейчас годных уроков {usable} из "
                f"{REQUIRED_LESSONS}.",
        "why": "Норма, с которой сравнивается последний урок, — это медиана 4 собственных "
               "предыдущих уроков плюс текущий. Уроки другой комнаты сюда не годятся: там другие "
               "места и, скорее всего, другие дети.",
        "how": "Ключ комнаты и класса должен совпадать буквально — другой ключ заводит другую "
               "историю.",
    }


def _ask_for_a_seating_plan(place: ClassvisionPlace) -> dict[str, str]:
    return {
        "what": f"Подписанный план рассадки на «{place.label_ru}» в комнате {place.camera_key}: "
                f"кто здесь сидит, с какой и по какую дату. (Внутренний номер места в базе — "
                f"{place.id}; он нужен только для команды и номером места в комнате не является.)",
        "why": "Единица учёта — МЕСТО, а не ребёнок. Без письменного подтверждения снижение "
               "показателя может означать, что с какого-то дня здесь сидит другой ученик, и "
               "отличить одно от другого может только человек с планом. Ни лицо, ни трекер имени "
               "здесь не создают: распознавание на этих камерах измерено (лучшее совпадение 0,30 "
               "при отрыве 0,10) и такого утверждения не выдерживает.",
        "how": "От школы нужны шесть вещей: (1) идентификатор ученика в её реестре, (2) ФИО, "
               "(3) ФИО и должность подтверждающего, (4) дата подтверждения, (5) срок действия — "
               "обязательно С ДАТОЙ ОКОНЧАНИЯ, (6) ссылка на решение школы, разрешающее "
               "пофамильное накопление. Записи в кабинет о рассадке пока нет ни одной, и "
               "`decision_ref` не даёт её придумать.",
    }


def _signature(session: Session, *, school_id: int, place_id: int) -> dict[str, str] | None:
    """Who signed for this chair, on what document, from when — or None.

    Read from `classvision_attestations` rather than from the name already copied onto the
    observations, because the two answer different questions. The copied name says what was
    true when the lesson was imported; this says what the school currently stands behind. If
    they ever disagree, the page should be able to show both, and it cannot do that from one
    of them.
    """
    row = session.execute(
        select(ClassvisionAttestation, Person)
        .join(Person, Person.id == ClassvisionAttestation.person_id)
        .join(ClassvisionPlace, ClassvisionPlace.id == ClassvisionAttestation.place_id)
        # Both ends are named. The place is what makes the attestation this school's; the
        # person is checked separately because a row binding one school's chair to another
        # school's child is exactly the thing worth never rendering.
        .where(ClassvisionPlace.school_id == school_id)
        .where(Person.school_id == school_id)
        .where(ClassvisionAttestation.place_id == place_id)
        .where(ClassvisionAttestation.valid_to.is_(None))
        .order_by(ClassvisionAttestation.valid_from.desc())
    ).first()
    if row is None:
        return None
    attestation, person = row
    return {
        "person_name": person.full_name or person.external_id,
        "class_name": person.class_name or "",
        "attested_by": attestation.attested_by,
        "decision_ref": attestation.decision_ref,
        "valid_from_ru": attestation.valid_from.strftime("%d.%m.%Y"),
    }


def _trend(place: ClassvisionPlace, rows: list[LessonOfPlace], *, attested: bool) -> Trend:
    """Which of the refusals this is. Asked in the order in which the actions differ."""
    # The gates are a list of things a human could supply. For the adult there are none, so the
    # list is empty rather than five unmet conditions that would read as a to-do list.
    gates = _gates(rows, attested=attested) if place.role == "pupil" else ()
    checklist = _checklist(place, rows, attested=attested)
    usable = [row for row in rows if row.index_measured]
    common = {"gates": gates, "lessons_have": len(usable), "lessons_needed": REQUIRED_LESSONS,
              "checklist": checklist}
    if place.role != "pupil":
        return Trend("not_applicable_adult", "динамика по взрослому не строится",
                     "Накопленный по неделям показатель по взрослому был бы поверхностью для "
                     "сравнения сотрудников, построенной на измерении, которое такого "
                     "сравнения не выдерживает.", **common)
    if not rows:
        return Trend("no_data", "уроков ещё нет",
                     "Для этого места не загружено ни одного урока.", **common)
    if not attested:
        return Trend("identity_not_established", "динамика не строится: место не подписано",
                     f"Накоплено уроков: {len(rows)}. Счётчики ниже относятся к МЕСТУ и верны "
                     "сами по себе. Динамика — это утверждение о конкретном ребёнке, и она "
                     "появится только после подписанного плана рассадки.", **common)
    if any(row.place_match != "matched" for row in rows):
        return Trend("place_unstable", "динамика не строится: место опознано не во всех уроках",
                     "В части уроков это место не удалось однозначно сопоставить с прежним "
                     "(перестановка мебели или сдвиг камеры). Сравнивать такой ряд — значит "
                     "сравнивать разные места.", **common)
    if len(usable) < REQUIRED_LESSONS:
        return Trend("insufficient_lessons",
                     f"данных пока мало: {len(usable)} уроков из {REQUIRED_LESSONS}",
                     "Это не ошибка и не пустой отчёт: собственная норма ученика ещё не "
                     "определена, и любое «стало хуже» было бы сравнением с шумом. Счётчики "
                     "и таблица ниже уже осмысленны — не хватает именно ДИНАМИКИ.", **common)
    return Trend("available", "все условия выполнены",
                 "Сравнение ученика с самим собой: медиана его же предыдущих уроков и MAD как "
                 "мера разброса. Расчёт живёт в анализаторе (metrics/trend.py) и в кабинет "
                 "пока не переносился — здесь показан ряд, а не вывод.", **common)


def _no_rows_reason(place: ClassvisionPlace) -> str:
    """An empty table with a reason, never an empty table.

    The adult's row is the case that actually occurs: the importer accumulates him through
    `classvision_teacher_lessons`, because the follower attributes him to the whole room rather
    than to a seat, so his PLACE has no per-lesson rows at all. Headers over nothing read as a
    page that failed to load.
    """
    if place.role != "pupil":
        return (
            "По этому месту нет ни одной строки учёта, и это не пропуск: положение взрослого "
            "считается по всей комнате, а не по месту, поэтому его числа стоят в блоке "
            "«Взрослый в комнате» на странице урока. Здесь остаётся сама зона — она нужна, "
            "чтобы рамку взрослого можно было показать на кадре."
        )
    return (
        "По этому месту пока нет ни одного урока: место найдено в комнате, но ни в одном "
        "разобранном уроке к нему не привязана строка учёта."
    )


def place_view(session: Session, *, school_id: int, place_id: int) -> dict[str, Any] | None:
    """Everything one place page prints. The trend is a value, never a missing key."""
    place = session.scalars(
        select(ClassvisionPlace)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.id == place_id)
    ).first()
    if place is None:
        return None
    found = session.execute(
        select(ClassvisionPlaceLesson, ClassvisionLesson, ClassvisionRun)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .join(ClassvisionRun, ClassvisionPlaceLesson.run_id == ClassvisionRun.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.place_id == place.id)
        .where(ClassvisionRun.run_id == ClassvisionLesson.selected_run_id)
        .order_by(ClassvisionLesson.date_local, ClassvisionLesson.id)
    ).all()
    rows = [_lesson_row(row, lesson, run) for row, lesson, run in found]
    attested = any(row.person_id is not None for row, _, _ in found)
    latest = found[-1] if found else None
    signature = _signature(session, school_id=school_id, place_id=place.id)
    return {
        "place": place, "rows": rows, "columns": COUNT_COLUMNS,
        # The heading names the child only where a human signed for that; everywhere else it
        # names the chair. The page used to say «плана рассадки на это место нет» whatever the
        # record held, so a signed place contradicted itself half a screen later.
        "signature": signature,
        "display_name": (signature["person_name"] if signature else place.label_ru),
        "footnotes": footnotes(*[row.cells for row in rows]),
        "trend": _trend(place, rows, attested=attested),
        "attested": attested,
        "identity_reason": latest[0].identity_reason if latest else "",
        "not_comparable_ru": (latest[0].within_lesson or {}).get("not_comparable_to_ru", "")
        if latest else "",
        # The components of the LAST lesson's index, so the page can show what the number is
        # made of rather than only the number. Not averaged over the term: an average of
        # weighted shares is a fifth quantity nobody defined.
        "latest": rows[-1] if rows else None,
        "latest_date_ru": rows[-1].date_ru if rows else "",
        "reading": reading_of(session, school_id=school_id, run=latest[2]) if latest else None,
        # The reading is a note about a LESSON, and this is a page about one place -- so the
        # template is told which place to keep. Printing the other seven places' notes here would
        # put a paragraph about eight children on a page about one, side by side, which is the
        # comparison this system does not make.
        "reading_only_place": place.ordinal,
        "is_demo": bool(place.is_demo),
        "is_pupil": place.role == "pupil",
        "no_rows_reason_ru": _no_rows_reason(place) if not rows else "",
    }

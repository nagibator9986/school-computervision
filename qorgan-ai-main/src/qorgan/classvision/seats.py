"""The rows one artefact writes per PLACE and for the ADULT, and what each refuses.

Split out of `importer.py` because the two jobs answer different questions and the file
limit is not the only reason: this module is where the artefact's vocabulary is turned into
columns, and every choice in it is about NOT losing a distinction on the way — a refused
index is NULL and not zero, an unmatched place is stored in full rather than joined to the
nearest history, an adult the analyser could not locate is listed with the reason rather
than omitted.
"""

from __future__ import annotations

from typing import Any

from qorgan.classvision import places as place_rules
from qorgan.db.models.classvision import (ClassvisionLesson, ClassvisionPlaceLesson,
                                          ClassvisionRun, ClassvisionTeacherLesson)


def float_or_none(value: Any) -> float | None:
    """`0.0` and «не измерялось» are different values and must not become one."""
    return None if value is None else float(value)


def store_places(session: Any, document: dict[str, Any], *, lesson: ClassvisionLesson,
                  run: ClassvisionRun, school: int, result: Any) -> None:
    """Every discovered seat, matched to a place by geometry — or stored unattached."""
    known = place_rules.known_places(session, school_id=school, camera_key=lesson.camera_key,
                                     class_key=lesson.class_key)
    claimed: set[int] = set()
    for seat in sorted(document["seats"], key=lambda s: int(s["seat_id"])):
        centre = tuple(float(v) for v in (seat.get("centre") or (0.0, 0.0)))
        scale = float(seat.get("scale_px") or 0.0)
        role = str(seat.get("role") or "pupil")
        found = place_rules.match_place(centre, scale, known, role)
        if found.place is not None and found.place.id in claimed:
            found = place_rules.PlaceMatch(
                None, "ambiguous",
                f"{found.place.label_ru} уже занято другим местом этого же прогона: "
                "два места не могут быть одним. Привязка не сделана.", found.distance)
        elif found.place is None and found.match == "new":
            new = place_rules.create_place(
                session, school_id=school, camera_key=lesson.camera_key,
                class_key=lesson.class_key, centre=centre, scale=scale, role=role,
                run_id=run.run_id, first_seen_at=lesson.started_at, is_demo=run.is_demo)
            known.append(new)
            result.new_places += 1
            found = place_rules.PlaceMatch(new, "matched", found.reason_ru, found.distance)
        if found.place is not None:
            claimed.add(found.place.id)
        else:
            result.unmatched += 1
        session.add(place_row(session, seat, lesson=lesson, run=run, found=found,
                              school=school, result=result))
        result.places += 1


def place_row(session: Any, seat: dict[str, Any], *, lesson: ClassvisionLesson,
              run: ClassvisionRun, found: place_rules.PlaceMatch, school: int,
              result: Any) -> ClassvisionPlaceLesson:
    ledger = seat.get("ledger") or {}
    counts = ledger.get("counts") or {}
    activity = (seat.get("metrics") or {}).get("activity") or {}
    person_id, method, reason = (None, "not_established", "место не привязано к истории комнаты")
    if found.place is not None:
        person_id, method, reason = place_rules.attested_person(
            session, school_id=school, place_id=found.place.id, on=lesson.date_local)
    if person_id is not None:
        result.named += 1
    return ClassvisionPlaceLesson(
        lesson_id=lesson.id, run_id=run.id,
        place_id=None if found.place is None else found.place.id,
        seat_id=int(seat["seat_id"]), seat_label=str(seat["label"]),
        role=str(seat.get("role") or "pupil"),
        place_match=found.match, place_match_reason=found.reason_ru,
        place_match_distance=found.distance,
        # This run's own geometry, which is what a frame is labelled from. The place's anchor
        # is a different number on purpose (`db/models/classvision.py`).
        centre_x=float((seat.get("centre") or (0.0, 0.0))[0]),
        centre_y=float((seat.get("centre") or (0.0, 0.0))[1]),
        scale_px=float(seat.get("scale_px") or 0.0),
        person_id=person_id, identity_method=method, identity_reason=reason,
        coverage=float(ledger.get("coverage") or 0.0),
        observations=int(ledger.get("observations") or 0),
        observed_seconds=float(ledger.get("observed_seconds") or 0.0),
        settled=bool(ledger.get("settled")),
        settle_refusal=ledger.get("settle_refusal"),
        absent_observations=int(ledger.get("absent_observations") or 0),
        unreadable_observations=int(ledger.get("unreadable_observations") or 0),
        hand_unmeasurable_observations=int(ledger.get("hand_unmeasurable_observations") or 0),
        hand_raises=int(counts.get("hand_raises") or 0),
        stands=int(counts.get("stands") or 0),
        away_episodes=int(counts.get("away_episodes") or 0),
        board_visits=int(counts.get("board_visits") or 0),
        head_down_episodes=int(counts.get("head_down_episodes") or 0),
        turned_away_episodes=int(counts.get("turned_away_episodes") or 0),
        activity_index=float_or_none(activity.get("index")) if activity.get("available") else None,
        activity_reason=str(activity.get("reason") or ""),
        activity_parts=activity.get("parts") or [],
        within_lesson=(seat.get("metrics") or {}).get("within_lesson") or {},
        ledger=ledger,
        timeline=seat.get("timeline") or [],
    )


def store_teacher(session: Any, document: dict[str, Any], *, lesson: ClassvisionLesson,
                   run: ClassvisionRun, school: int, result: Any) -> None:
    """The adult, or a named reason there is no row. Never a silent omission.

    A `presence` block without `attributed_share_of_lesson_percent` is DROPPED: §7 forbids
    rendering any teacher number without it, and the nearest available column means the
    follower's share on one path and the seat's occupancy on the other, so a page reading it
    could not know which guarantee it had.
    """
    teacher = document.get("teacher")
    if not isinstance(teacher, dict):
        result.dropped.append("взрослого в артефакте нет")
        return
    metrics = dict(teacher.get("metrics") or {})
    presence = metrics.pop("presence", None)
    if isinstance(presence, dict) and presence.get("attributed_share_of_lesson_percent") is None:
        result.dropped.append(
            "блок взрослого: в `presence` нет `attributed_share_of_lesson_percent`. Без него "
            "доли по состояниям нельзя показать — «у доски 3 %» без «опознан в 45 % кадров» "
            "сообщает читателю ложь верным числом."
        )
        return
    board = (presence or {}).get("board") or {}
    occupancy = (presence or {}).get("board_occupancy") or {}
    found = adult_place(session, teacher, lesson=lesson, run=run, school=school)
    session.add(ClassvisionTeacherLesson(
        lesson_id=lesson.id, run_id=run.id,
        place_id=None if found.place is None else found.place.id,
        place_missing_reason=None if found.place is not None else found.reason_ru,
        seat_id=teacher.get("seat_id"),
        attributed_share_of_lesson_percent=float_or_none(
            (presence or {}).get("attributed_share_of_lesson_percent")),
        pose_coverage=float_or_none(metrics.get("coverage")),
        board_zone_configured=bool(board.get("zone_configured")),
        board_minutes_of_lesson=float_or_none(board.get("minutes_of_lesson")),
        board_share_of_lesson_percent=float_or_none(board.get("share_of_lesson_percent")),
        board_occupancy_available=bool(occupancy.get("available")),
        transitions_excluding_out_of_frame=(presence or {}).get(
            "transitions_between_episodes_excluding_out_of_frame"),
        pose_transitions=metrics.get("transitions"),
        presence=presence, board=board, board_occupancy=occupancy, pose_metrics=metrics,
        not_an_assessment_ru=str(metrics.get("not_an_assessment_ru") or ""),
    ))
    result.teacher = True


def adult_place(session: Any, teacher: dict[str, Any], *, lesson: ClassvisionLesson,
                 run: ClassvisionRun, school: int) -> place_rules.PlaceMatch:
    """An adult with no `centre` is listed with the artefact's own reason, never dropped."""
    centre = teacher.get("centre")
    if not centre:
        evidence = (teacher.get("identification") or {}).get("evidence") or {}
        return place_rules.PlaceMatch(
            None, "no_geometry",
            str(evidence.get("reason_ru") or evidence.get("reason")
                or "артефакт не смог определить, где взрослый находился. Это «мы не знаем, "
                   "где он был», а не «его не было»."), None)
    known = place_rules.known_places(session, school_id=school, camera_key=lesson.camera_key,
                                     class_key=lesson.class_key)
    point = (float(centre[0]), float(centre[1]))
    scale = float(teacher.get("scale_px") or 0.0)
    found = place_rules.match_place(point, scale, known, "adult")
    if found.place is None and found.match == "new":
        made = place_rules.create_place(
            session, school_id=school, camera_key=lesson.camera_key,
            class_key=lesson.class_key, centre=point, scale=scale, role="adult",
            run_id=run.run_id, first_seen_at=lesson.started_at, is_demo=run.is_demo)
        return place_rules.PlaceMatch(made, "matched", found.reason_ru, found.distance)
    return found

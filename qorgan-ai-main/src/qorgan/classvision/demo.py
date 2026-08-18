"""A synthetic TERM for one class, so the weekly view has something to show. Every row DEMO.

`qorgan classvision demo --weeks 6 --class 8-А --from <real artefact.json>`

**WHAT IS SYNTHETIC, PRECISELY.** Everything below is invented, and everything not listed is
copied or derived from the real artefacts passed with `--from`:

* the LESSONS: one per ISO week (plus a second lesson in one week, because a real timetable
  does that and a weekly aggregate must survive it), Tuesday 09:00 school-local;
* the per-place, per-lesson OBSERVATION HISTOGRAM: how many sampled frames this place spent
  head-down, turned away, away from its place, hand up. Drawn from the pooled distributions of
  the real recordings, per component, never from a round number chosen by hand;
* the COVERAGE of each place in each lesson, drawn from the real coverages (which run
  0.615-0.996 across the two recordings) with per-place bias, so the demo has both a
  well-seen place and a badly-seen one;
* TWO deliberate shapes, which are the point of the exercise: ONE place declines steadily
  across the term (head-down share up, visible actions down, monotone but not artificially
  smooth) and ONE is merely NOISY (large swings, no direction). Everything else is stable.

**WHAT IS REAL.** The room's geometry — every place's centre and shoulder width — is taken
from the real recording that discovered the most places, so the seating plan a page draws is a
real room seen from a real camera. The thresholds block, the caveats and the `unmeasured` list
are copied verbatim from that artefact, and `sample_fps`, lesson duration and the per-state
proportions come from its numbers. The INDEX is not invented at all: it is computed from the
synthetic histogram by the same weighted sum the analyser uses, so its four components sum to
it exactly as they do on a real lesson.

**WHY THE WEIGHTS ARE COPIED HERE RATHER THAN IMPORTED.** `qorgan` may not import
`classvision` — that separation carries the AGPL boundary and keeps torch out of the web
process (`INTEGRATION.md` §1, §9). So these four numbers exist twice, which is exactly the
"second quietly different implementation" this project keeps being bitten by. The mitigation is
a measurement, not a promise: `verify_against(artefact)` recomputes every real place's index
from the artefact's OWN parts and compares it with the artefact's own figure, and `generate`
runs it on every source before writing a single row. If the two implementations ever diverge,
the demo refuses to build.

**WHAT THE DEMO DELIBERATELY DOES NOT INVENT.** No names: no attestation is written, so every
place stays «место N» — a fabricated child is not a demonstration, it is a fabricated child.
No frames: there is no recording, so the video-classification view shows real runs only. No
Mann-Kendall verdict inside a lesson: segment indices are computed honestly, and the direction
block says it was not tested rather than carrying an invented p-value. No `presence` block for
the adult: there is no follower here, so the board figures are NULL and read «не измерялось».
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select

from qorgan.classvision import places as place_rules
from qorgan.classvision.distributions import (DemoRefused, Profile, activity, read_profile,
                                              verify_against)
from qorgan.db.engine import session_scope
from qorgan.db.models.classvision import (ClassvisionLesson, ClassvisionPlace,
                                          ClassvisionPlaceLesson, ClassvisionRun,
                                          ClassvisionTeacherLesson)
from qorgan.db.models.school import sole_school_id
from qorgan.settings import get_settings


# Where the demonstration lesson sits in the school day, and how many segments the
# within-lesson view is cut into. Both CHOSEN, and both are stated on the page as demo facts.
LESSON_START_HOUR = 9
SEGMENTS = 6
DEMO_CAVEAT_RU = (
    "ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ. Этот урок не записывался и не разбирался: счётчики "
    "синтезированы по распределениям реальных записей, чтобы показать, как выглядит семестр. "
    "Ни одно число здесь не является измерением, и складывать эти уроки с реальными нельзя."
)
DEMO_UNMEASURED_RU = {
    "what": "всё, что показано на этом уроке",
    "why": "это демонстрация: записи не существует, измерений не было",
}


@dataclass(slots=True)
class DemoResult:
    class_key: str
    camera_key: str
    lessons: int = 0
    places: int = 0
    rows: int = 0
    declining: str = ""
    noisy: str = ""
    removed: int = 0
    notes: list[str] = field(default_factory=list)

    def report_ru(self) -> list[str]:
        lines = [
            f"ДЕМО построено: класс {self.class_key}, комната {self.camera_key}, "
            f"уроков {self.lessons}, мест {self.places}, строк по местам {self.rows}.",
            f"  устойчивое снижение заложено на {self.declining}; "
            f"шумное место без направления — {self.noisy}.",
            "  все строки помечены is_demo=1; имён нет — план рассадки не подписан; "
            "кадров нет — записи не существует.",
        ]
        if self.removed:
            lines.append(f"  прежняя демонстрация удалена: уроков {self.removed}")
        return lines + [f"  {note}" for note in self.notes]


def generate(*, sources: list[Path], weeks: int = 6, class_key: str = "8-А",
             camera_key: str = "demo_room_8a", school_id: int | None = None,
             seed: int = 20260817, replace: bool = False) -> DemoResult:
    """Build the term. Refuses before writing anything if the copied formula disagrees."""
    profile = read_profile(sources)
    for path in sources:
        problems = verify_against(path)
        if problems:
            raise DemoRefused(
                "пересчёт индекса расходится с артефактом, поэтому демонстрация не строится "
                "(иначе в базе появилось бы второе, слегка другое определение индекса): "
                + "; ".join(problems))
    rng = random.Random(seed)
    zone = ZoneInfo(get_settings().school_timezone)
    result = DemoResult(class_key=class_key, camera_key=camera_key)
    result.notes.append("формы взяты из: " + ", ".join(profile.sources))

    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        if replace:
            result.removed = _remove_previous(session, school, camera_key, class_key)
        places = _make_places(session, profile, school=school, camera_key=camera_key,
                              class_key=class_key, rng=rng)
        result.places = len(places)
        shapes = _shapes(profile, places, rng)
        result.declining = shapes["declining"].label_ru
        result.noisy = shapes["noisy"].label_ru
        for ordinal, moment in enumerate(_dates(weeks, zone), start=1):
            _one_lesson(session, profile, places=places, shapes=shapes, school=school,
                        camera_key=camera_key, class_key=class_key, moment=moment,
                        ordinal=ordinal, rng=rng, zone=zone, result=result)
        return result


def _remove_previous(session: Any, school: int, camera_key: str, class_key: str) -> int:
    """Delete the previous demonstration for this room — and only the demonstration.

    Scoped by `is_demo` on both tables, so a real lesson in the same room can never be caught
    by a rebuild. It takes the demo lessons' RUNS, place-rows, teacher rows and READINGS with
    them (that is what the cascades are for), so a note generated for a demo lesson has to be
    re-loaded after `--replace`.
    """
    lessons = list(session.scalars(
        select(ClassvisionLesson)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionLesson.camera_key == camera_key)
        .where(ClassvisionLesson.class_key == class_key)
        .where(ClassvisionLesson.is_demo.is_(True))))
    for lesson in lessons:
        session.delete(lesson)
    session.execute(
        delete(ClassvisionPlace)
        .where(ClassvisionPlace.school_id == school)
        .where(ClassvisionPlace.camera_key == camera_key)
        .where(ClassvisionPlace.class_key == class_key)
        .where(ClassvisionPlace.is_demo.is_(True)))
    session.flush()
    return len(lessons)


def _dates(weeks: int, zone: ZoneInfo) -> list[datetime]:
    """Tuesdays, one per week, ending this week — plus a second lesson in the middle week.

    Two lessons in one ISO week is what a timetable actually looks like, and a weekly view
    that quietly averages them is a bug this data will find.
    """
    today = datetime.now(zone).date()
    tuesday = today - timedelta(days=(today.weekday() - 1) % 7)
    days = [tuesday - timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]
    middle = days[len(days) // 2] + timedelta(days=2)
    stamps = [datetime(d.year, d.month, d.day, LESSON_START_HOUR, 0, tzinfo=zone) for d in days]
    stamps.append(datetime(middle.year, middle.month, middle.day, LESSON_START_HOUR + 2, 0,
                           tzinfo=zone))
    return sorted(stamps)


def _make_places(session: Any, profile: Profile, *, school: int, camera_key: str,
                 class_key: str, rng: random.Random) -> list[ClassvisionPlace]:
    """Real geometry, demo flag. Re-used when the demo is rebuilt without --replace."""
    existing = place_rules.known_places(session, school_id=school, camera_key=camera_key,
                                       class_key=class_key)
    if existing:
        return [p for p in existing if p.role == "pupil"]
    made: list[ClassvisionPlace] = []
    stamp = f"demo:{rng.getrandbits(32):08x}"
    for x, y, scale in profile.geometry:
        made.append(place_rules.create_place(
            session, school_id=school, camera_key=camera_key, class_key=class_key,
            centre=(x, y), scale=scale, role="pupil", run_id=stamp, first_seen_at=None,
            is_demo=True))
    if profile.adult.get("scale_px"):
        place_rules.create_place(
            session, school_id=school, camera_key=camera_key, class_key=class_key,
            centre=profile.adult["centre"], scale=profile.adult["scale_px"], role="adult",
            run_id=stamp, first_seen_at=None, is_demo=True)
    return made


@dataclass(slots=True)
class Shape:
    """How one place behaves across the term. `drift` is the only invented direction here."""

    place: ClassvisionPlace
    label_ru: str
    coverage: float
    head_down: float
    turned_away: float
    away: float
    events: float
    drift: str = "stable"  # stable | declining | noisy


def _shapes(profile: Profile, places: list[ClassvisionPlace],
            rng: random.Random) -> dict[str, Any]:
    """One shape per place, drawn from the pooled real distributions.

    The declining place is given the WORST-BUT-ONE coverage rather than the best, on purpose:
    a decline on a well-seen place is the easy case, and the page has to be able to say
    «снижение видно, но обзор этого места был неполным» about the hard one.
    """
    shapes: list[Shape] = []
    for place in places:
        shapes.append(Shape(
            place=place,
            label_ru=place.label_ru,
            coverage=rng.choice(profile.coverages),
            head_down=rng.choice(profile.head_down),
            turned_away=rng.choice(profile.turned_away),
            away=rng.choice(profile.away),
            events=float(rng.choice(profile.hand_raises) + rng.choice(profile.stands)),
        ))
    ordered = sorted(shapes, key=lambda s: s.coverage)
    declining = ordered[1] if len(ordered) > 1 else ordered[0]
    noisy = ordered[-1]
    declining.drift = "declining"
    noisy.drift = "noisy"
    return {"all": shapes, "declining": declining, "noisy": noisy}


def _one_lesson(session: Any, profile: Profile, *, places: list[ClassvisionPlace],
                shapes: dict[str, Any], school: int, camera_key: str, class_key: str,
                moment: datetime, ordinal: int, rng: random.Random, zone: ZoneInfo,
                result: DemoResult) -> None:
    minutes = round(rng.choice(profile.durations) + rng.uniform(-3.0, 3.0), 1)
    started = moment.astimezone(UTC)
    day, iso_year, iso_week = place_rules.iso_week_of(started, zone)
    run_id = _demo_run_id(camera_key, class_key, moment)
    lesson = ClassvisionLesson(
        school_id=school, is_demo=True, camera_key=camera_key,
        camera_key_source="demo", class_key=class_key, started_at=started,
        ended_at=started + timedelta(minutes=minutes), date_local=day, iso_year=iso_year,
        iso_week=iso_week, timezone=str(zone), duration_minutes=minutes,
        selected_run_id=run_id, part_count=1)
    session.add(lesson)
    session.flush()
    run = _demo_run(session, profile, lesson=lesson, run_id=run_id, minutes=minutes)
    for shape in shapes["all"]:
        session.add(_row(shape, lesson=lesson, run=run, ordinal=ordinal, rng=rng,
                         minutes=minutes, fps=run.sample_fps))
        result.rows += 1
    _demo_teacher(session, profile, lesson=lesson, run=run, school=school)
    result.lessons += 1


def _demo_run_id(camera_key: str, class_key: str, moment: datetime) -> str:
    """Deterministic, and visibly NOT a content hash: nothing was hashed, nothing was watched."""
    stamp = f"{camera_key}|{class_key}|{moment.isoformat()}".encode()
    return f"demo{hashlib.sha256(stamp).hexdigest()[:12]}"


def _demo_run(session: Any, profile: Profile, *, lesson: ClassvisionLesson, run_id: str,
              minutes: float) -> ClassvisionRun:
    """The provenance row for a lesson nobody recorded, and it says exactly that.

    `video_sha256` carries `демо:` and not a plausible digest: a 64-character hex string in
    that column would be an invitation to go and look for a recording that does not exist.
    """
    fps = profile.sample_fps[0] if profile.sample_fps else 2.0
    run = ClassvisionRun(
        lesson_id=lesson.id, is_demo=True, run_id=run_id, schema_version="classvision/1.1",
        video_path="ДЕМО: записи не существует",
        video_sha256=f"демо:{run_id}", video_bytes=0, started_at=lesson.started_at,
        clock_source="demo", sample_fps=fps,
        analysed_frames=int(minutes * 60 * fps), duration_seconds=minutes * 60.0,
        thresholds_sha="demo", model_weights="ДЕМО: модель не запускалась",
        room_layout=profile.room_layout, pupil_places=len(profile.geometry),
        provenance={"schema": "classvision/1.1", "demo": True,
                    "thresholds": profile.thresholds, "room": {"layout": profile.room_layout},
                    "note_ru": DEMO_CAVEAT_RU},
        uncertainty={"demo": True, "note_ru": DEMO_CAVEAT_RU},
        caveats=[DEMO_CAVEAT_RU, *profile.caveats],
        unmeasured=[DEMO_UNMEASURED_RU, *profile.unmeasured],
        imported_at=datetime.now(UTC))
    session.add(run)
    session.flush()
    return run


def _row(shape: Shape, *, lesson: ClassvisionLesson, run: ClassvisionRun, ordinal: int,
         rng: random.Random, minutes: float, fps: float) -> ClassvisionPlaceLesson:
    coverage, histogram, counts = _draw(shape, ordinal=ordinal, rng=rng, minutes=minutes, fps=fps)
    index = activity(histogram, counts, coverage)
    observations = sum(histogram.values())
    return ClassvisionPlaceLesson(
        lesson_id=lesson.id, run_id=run.id, place_id=shape.place.id,
        seat_id=shape.place.ordinal, seat_label=f"seat_{shape.place.ordinal}", role="pupil",
        place_match="matched",
        place_match_reason="демонстрация: место задано, геометрия взята из реальной записи",
        place_match_distance=0.0,
        person_id=None, identity_method="not_established",
        identity_reason="демонстрация без плана рассадки: имён нет и быть не должно",
        centre_x=shape.place.anchor_x, centre_y=shape.place.anchor_y,
        scale_px=shape.place.anchor_scale,
        coverage=round(coverage, 3), observations=observations,
        observed_seconds=round(observations / fps, 1), settled=True,
        absent_observations=int(run.analysed_frames - observations),
        unreadable_observations=0, hand_unmeasurable_observations=0,
        hand_raises=counts["hand_raises"], stands=counts["stands"], away_episodes=counts["away"],
        board_visits=0, head_down_episodes=counts["head_down_episodes"],
        turned_away_episodes=counts["turned_away_episodes"],
        activity_index=index["index"], activity_reason=index["reason"],
        activity_parts=index["parts"],
        within_lesson=_within(shape, coverage, histogram, counts, minutes, rng),
        ledger={"demo": True, "coverage": round(coverage, 3), "observations": observations,
                "counts": {k: counts[k] for k in ("hand_raises", "stands")},
                "state_observations": histogram, "note_ru": DEMO_CAVEAT_RU},
        timeline=[])


def _draw(shape: Shape, *, ordinal: int, rng: random.Random, minutes: float,
          fps: float) -> tuple[float, dict[str, int], dict[str, int]]:
    """One place, one lesson: coverage, an observation histogram and episode counts.

    The drift is applied to the COMPONENTS, never to the index: a decline invented at the level
    of the total would have components that do not add up to it, and every page here shows the
    components.
    """
    step = ordinal - 1
    swing = rng.uniform(-0.05, 0.05)
    if shape.drift == "declining":
        head_down = min(0.75, shape.head_down + 0.055 * step + swing * 0.4)
        events = max(0.0, shape.events - 0.45 * step)
        coverage = shape.coverage + rng.uniform(-0.03, 0.03)
    elif shape.drift == "noisy":
        head_down = max(0.0, min(0.8, shape.head_down + rng.uniform(-0.22, 0.22)))
        events = max(0.0, shape.events + rng.uniform(-1.5, 1.5))
        coverage = shape.coverage + rng.uniform(-0.08, 0.02)
    else:
        head_down = max(0.0, shape.head_down + swing)
        events = max(0.0, shape.events + rng.uniform(-0.7, 0.7))
        coverage = shape.coverage + rng.uniform(-0.04, 0.04)
    coverage = min(0.998, max(0.42, coverage))
    total = int(minutes * 60 * fps * coverage)
    histogram = {
        "head_down": int(total * head_down),
        "turned_away": int(total * max(0.0, shape.turned_away + swing * 0.5)),
        "away_from_place": int(total * max(0.0, shape.away + swing * 0.3)),
        "hand_raised": int(round(events)) * 6,
        "unknown": int(total * 0.004),
    }
    histogram["seated"] = max(0, total - sum(histogram.values()))
    hands = int(round(events * 0.6))
    counts = {"hand_raises": hands, "stands": int(round(events)) - hands, "board_visits": 0,
              "away": max(0, int(histogram["away_from_place"] / (fps * 25)) ),
              "head_down_episodes": max(0, int(histogram["head_down"] / (fps * 60))),
              "turned_away_episodes": max(0, int(histogram["turned_away"] / (fps * 20)))}
    counts["stands"] = max(0, counts["stands"])
    return coverage, histogram, counts


def _within(shape: Shape, coverage: float, histogram: dict[str, int], counts: dict[str, int],
            minutes: float, rng: random.Random) -> dict[str, Any]:
    """Segment indices, computed the same way — and NO direction verdict.

    The analyser tests a within-lesson direction with Mann-Kendall and a floor built out of
    that lesson's own visibility. Re-implementing that here would be a second statistic with
    the same name, so the block says the test was not run. «Не проверялось» is a usable answer;
    an invented p-value is not.
    """
    segments = []
    for index in range(1, SEGMENTS + 1):
        share = 1.0 / SEGMENTS
        jitter = rng.uniform(0.85, 1.15)
        piece = {k: int(v * share * jitter) for k, v in histogram.items()}
        piece_counts = {"hand_raises": counts["hand_raises"] if index % 3 == 0 else 0,
                        "stands": 0, "board_visits": 0}
        segments.append({
            "ordinal": index,
            "start_minute": round((index - 1) * minutes / SEGMENTS, 2),
            "end_minute": round(index * minutes / SEGMENTS, 2),
            "coverage": round(min(0.999, coverage * jitter), 3),
            "observations": sum(piece.values()),
            "events": piece_counts["hand_raises"],
            "events_are_a_lower_bound": True,
            "activity": activity(piece, piece_counts, min(0.999, coverage * jitter)),
        })
    return {
        "segments_requested": SEGMENTS,
        "segment_minutes": round(minutes / SEGMENTS, 2),
        "segments": segments,
        "change": {"available": False, "direction": "unknown",
                   "direction_ru": "направление внутри урока не проверялось",
                   "refused_because": "demo_no_direction_test",
                   "reason": "это демонстрация: проверка направления (Манна—Кендалла с "
                             "порогом по видимости) здесь не проводится, чтобы не выдавать "
                             "придуманное p-значение за результат."},
        "not_comparable_to_ru": "Индексы сегментов сравнимы только друг с другом внутри "
                                "этого урока: событийная составляющая нормирована на урок.",
    }


def _demo_teacher(session: Any, profile: Profile, *, lesson: ClassvisionLesson,
                  run: ClassvisionRun, school: int) -> None:
    """The adult's POSE row only. No `presence`, so every board figure is NULL, as it must be.

    There is no follower in a demonstration, so «у доски» cannot arise in any frame — exactly
    the state camera_01 is in for real. The row exists so the page shows the adult listed with
    «не измерялось» rather than absent, which is a different fact.
    """
    if not profile.adult.get("metrics"):
        return
    adult = next((p for p in place_rules.known_places(
        session, school_id=school, camera_key=lesson.camera_key, class_key=lesson.class_key)
        if p.role == "adult"), None)
    metrics = dict(profile.adult["metrics"])
    session.add(ClassvisionTeacherLesson(
        lesson_id=lesson.id, run_id=run.id, place_id=None if adult is None else adult.id,
        seat_id=1, attributed_share_of_lesson_percent=None,
        pose_coverage=profile.adult.get("coverage"),
        board_zone_configured=False, board_occupancy_available=False,
        presence=None,
        board={"zone_configured": False, "minutes_of_lesson": None,
               "direction_of_error_ru": "Не измерялось: зона доски не задана, и записи не "
                                        "существует — это демонстрация."},
        board_occupancy={"available": False,
                         "reason_ru": "демонстрация: занятость доски не измеряется"},
        pose_metrics={**metrics, "demo": True},
        not_an_assessment_ru=str(metrics.get("not_an_assessment_ru") or "")))

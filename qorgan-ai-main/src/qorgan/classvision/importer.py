"""One artefact -> rows, and a NAMED refusal for everything this code will not guess.

`qorgan classvision import <artefact.json>`.

**Everything it refuses, and why refusing beats importing.** An unknown schema MAJOR (a
major bump changes what an existing field MEANS, so importing it under this code would put a
differently-defined number into the same column as last term's — `classvision/1.1` renamed
the adult's shares for exactly that reason and deliberately did not alias the old names). A
wall clock with no zone (an hour's error puts a lesson on the wrong day at the boundary,
silently). No wall clock at all, unless `--allow-unclocked`, in which case the run is stored
with NULL dates and excluded from everything longitudinal. An hour that overlaps one already
stored, unless `--allow-overlap`, because two files covering one hour double that hour in
every weekly counter. The same `run_id` already filed under a different room — which is NOT
the same thing as an idempotent no-op: idempotency is on `run_id` alone, so a corrected room
key would otherwise write nothing while printing «уже импортирован», which an operator
fixing a typo reads as «принято».

**Idempotent on `run_id`.** That id is the artefact's hash of the video content plus every
setting that could change a number, so re-importing the same file writes nothing, and a
re-analysis under a different threshold is a NEW run on the SAME lesson which does not move
`selected_run_id`. Moving it is a separate, named act by a person.

**Coverage and the doubt counters are stored with the totals they qualify**, never derived
later, and `activity_index` is NULL — not 0 — when the artefact refused it. The refusal
sentence travels in `activity_reason` so a page has something true to print in place of a
number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select

from qorgan.classvision import places as place_rules
from qorgan.classvision.seats import float_or_none, store_places, store_teacher
from qorgan.db.engine import session_scope
from qorgan.db.models.classvision import (ClassvisionFrame, ClassvisionLesson,
                                          ClassvisionPlaceLesson, ClassvisionReading,
                                          ClassvisionRun)
from qorgan.db.models.school import sole_school_id
from qorgan.settings import get_settings

# A MINOR bump adds fields and is accepted — the unread ones land whole in the JSON columns.
# A MAJOR bump is refused. The asymmetry is the point: an unknown field is a gap in a report,
# a redefined field is a wrong number that looks right.
ACCEPTED_SCHEMA_MAJOR = "classvision/1"


class Refusal(Exception):
    """This artefact must not enter a school's database, and which rule says so.

    Carries a machine-readable `code` beside the sentence, because a term's back-fill is run
    by a script and "which refusal fired" must be answerable without matching on prose.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Unreadable(Exception):
    """The file could not be read or parsed — a statement about us, not about the artefact.

    Kept apart from `Refusal` so that a truncated download cannot report as a schema
    violation and send somebody off to change the schema.
    """


@dataclass(slots=True)
class ImportResult:
    """What was stored, and everything the import decided NOT to store.

    `dropped` and `notes` are part of the result on the same rule as the artefact's own
    `uncertainty` block: an import that silently discarded the adult and an import that had
    no adult to discard must not look identical afterwards.
    """

    run_id: str
    stored: bool
    is_demo: bool
    lesson_id: int | None = None
    places: int = 0
    new_places: int = 0
    unmatched: int = 0
    named: int = 0
    teacher: bool = False
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def report_ru(self) -> list[str]:
        mark = "ДЕМО — не измерение" if self.is_demo else "измерение"
        if not self.stored:
            return [f"{self.run_id}: уже импортирован, ничего не записано ({mark})."]
        lines = [
            f"{self.run_id}: записано ({mark}). Урок #{self.lesson_id}, мест {self.places} "
            f"(новых {self.new_places}, без привязки {self.unmatched}), "
            f"с именем {self.named} — только по подписанному плану рассадки.",
            f"  взрослый: {'записан' if self.teacher else 'не записан'}",
        ]
        lines += [f"  не записано: {reason}" for reason in self.dropped]
        lines += [f"  примечание: {note}" for note in self.notes]
        return lines


def load(path: Path) -> dict[str, Any]:
    """Read one artefact. Raises `Unreadable`; never returns a partial document."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise Unreadable(f"{path}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Unreadable(f"{path} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise Unreadable(f"{path} is JSON but not an object")
    return document


def check(document: dict[str, Any], *, allow_unclocked: bool) -> None:
    """Everything that stops this artefact entering a database, cheapest test first.

    Ordered so the error a human reads is the earliest true thing rather than a downstream
    symptom of it.
    """
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise Refusal("no_provenance", "документ без блока `provenance`")
    schema = str(provenance.get("schema", ""))
    if not schema.startswith(ACCEPTED_SCHEMA_MAJOR + "."):
        raise Refusal(
            "schema_version",
            f"схема {schema!r} не {ACCEPTED_SCHEMA_MAJOR}.x. Мажорная версия меняет СМЫСЛ "
            "существующих полей, и такой импорт положил бы иначе определённое число в ту же "
            "колонку, где лежит прошлый семестр.",
        )
    if not document.get("run_id"):
        raise Refusal("no_run_id", "нет `run_id` — идемпотентности не на чем держаться")
    if not provenance.get("video_sha256"):
        raise Refusal(
            "no_video_hash",
            "`provenance.video_sha256` пуст: это единственное, что связывает строку с "
            "записью, которую можно пересмотреть.",
        )
    if not isinstance(document.get("caveats"), list) or not document["caveats"]:
        raise Refusal(
            "no_caveats",
            "нет блока `caveats`. Они показываются рядом с каждым итогом; документ без них "
            "нельзя показать человеку.",
        )
    discovery = (provenance.get("room") or {}).get("seat_discovery") or {}
    if discovery.get("plausible") is False:
        raise Refusal(
            "implausible_seats",
            "разбор мест сам сообщает `plausible: false` — "
            f"{discovery.get('warning', 'места, вероятно, слиты')}. Двое детей, слитые в одно "
            "место, потом неотличимы.",
        )
    if not document.get("seats"):
        raise Refusal("no_seats", "мест не найдено — накапливать нечего")
    if provenance.get("started_at") in (None, "") and not allow_unclocked:
        raise Refusal(
            "no_wall_clock",
            "`provenance.started_at` пуст: запись нельзя поставить на календарь и сравнить с "
            "другим уроком. С флагом --allow-unclocked она сохранится с пустой датой и будет "
            "исключена из всего, что считается по неделям.",
        )
    _check_every_index_has_its_parts(document)


def _check_every_index_has_its_parts(document: dict[str, Any]) -> None:
    """An index is only ever shown decomposed, so one without components cannot be stored."""
    for seat in document["seats"]:
        activity = (seat.get("metrics") or {}).get("activity") or {}
        if activity.get("available") and not activity.get("parts"):
            raise Refusal(
                "index_without_parts",
                f"место {seat.get('label')} несёт индекс без составляющих. Индекс "
                "показывается только разложенным — иначе это «голое число», которое схема "
                "запрещает.",
            )


def import_artefact(path: Path, *, school_id: int | None = None, camera_key: str | None = None,
                    class_key: str | None = None, timezone: str | None = None,
                    is_demo: bool = False, allow_unclocked: bool = False,
                    allow_overlap: bool = False,
                    include_teacher: bool = True) -> ImportResult:
    """Store one artefact. Refuses by name; writes nothing at all when it refuses."""
    document = load(path)
    check(document, allow_unclocked=allow_unclocked)
    provenance = document["provenance"]
    zone_name = timezone or get_settings().school_timezone
    started_at = _wall_clock(provenance.get("started_at"), zone_name)
    room, source = _room_key(document, camera_key, path)
    klass = class_key or "не указан"

    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        existing = _already_imported(session, school, str(document["run_id"]))
        if existing is not None:
            _refuse_a_second_room(existing, room, klass)
            return ImportResult(run_id=existing.run_id, stored=False, is_demo=existing.is_demo)
        result = ImportResult(run_id=str(document["run_id"]), stored=True, is_demo=is_demo)
        lesson = _lesson_for(session, document, started_at=started_at, room=room, source=source,
                             klass=klass, school=school, is_demo=is_demo,
                             allow_overlap=allow_overlap, result=result, zone=ZoneInfo(zone_name))
        run = _store_run(session, document, lesson=lesson, started_at=started_at,
                         is_demo=is_demo)
        store_places(session, document, lesson=lesson, run=run, school=school,
                     result=result)
        if include_teacher:
            store_teacher(session, document, lesson=lesson, run=run, school=school,
                          result=result)
        else:
            result.dropped.append(
                "блок взрослого — по требованию оператора (--no-teacher). §12.5 остаётся "
                "решением школы, и отказ тоже записан."
            )
        result.lesson_id = lesson.id
        return result


def _wall_clock(raw: Any, zone_name: str) -> datetime | None:
    """The overlay's LOCAL time as an aware UTC datetime, or nothing. Never a guessed zone.

    The clock reader returns what is burned into the picture, which is a school's wall clock
    with no offset on it. `db/types.UtcDateTime` rejects a naive datetime at bind time, and
    rightly — a naive value stored as if it were UTC is the defect that column type exists to
    prevent — so the zone comes from the caller (`SCHOOL_TIMEZONE`) and never from here.
    """
    if raw in (None, ""):
        return None
    try:
        local = datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise Refusal("bad_wall_clock", f"`started_at` не ISO-8601: {raw!r} ({exc})") from exc
    if local.tzinfo is not None:
        return local.astimezone(UTC)
    try:
        return local.replace(tzinfo=ZoneInfo(zone_name)).astimezone(UTC)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise Refusal("bad_timezone", f"неизвестная зона {zone_name!r}: {exc}") from exc


def _room_key(document: dict[str, Any], asserted: str | None, path: Path) -> tuple[str, str]:
    """Which room this recording is of, and HOW WELL WE KNOW — the second half is a column.

    Three sources, best first: the operator's flag; the camera named in the room profile the
    analysis used (a signed file, so an earlier assertion by a person); and last the video
    filename, which matters because two files from one camera would otherwise become two
    rooms that never accumulate together.
    """
    if asserted:
        return asserted, "operator"
    layout = ((document["provenance"].get("room") or {}).get("layout") or {})
    if layout.get("camera"):
        return str(layout["camera"]), "from_room_profile"
    return Path(str(document["provenance"].get("video_path") or path)).stem, "derived_from_filename"


def _already_imported(session: Any, school: int, run_id: str) -> ClassvisionRun | None:
    return session.scalars(
        select(ClassvisionRun)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionRun.run_id == run_id)
    ).first()


def _refuse_a_second_room(existing: ClassvisionRun, room: str, klass: str) -> None:
    """A corrected room key must not read as «принято» while writing nothing."""
    lesson = existing.lesson
    if lesson.camera_key == room or room is None:
        return
    raise Refusal(
        "already_imported_elsewhere",
        f"прогон {existing.run_id} уже импортирован как «{lesson.camera_key} / "
        f"{lesson.class_key}», а сейчас указано «{room} / {klass}». Идемпотентность держится "
        "на run_id, поэтому исправленный ключ комнаты не записал бы НИЧЕГО, и урок навсегда "
        "остался бы в комнате с опечаткой. Удалите прежний импорт или укажите тот же ключ.",
    )


def _lesson_for(session: Any, document: dict[str, Any], *, started_at: datetime | None,
                room: str, source: str, klass: str, school: int, is_demo: bool,
                allow_overlap: bool, result: ImportResult, zone: Any) -> ClassvisionLesson:
    """Attach to the lesson this run belongs to, or open one. Refuses a doubled hour."""
    lesson_block = document.get("lesson") or {}
    duration = float(lesson_block.get("duration_minutes") or 0.0) * 60.0
    link = place_rules.lesson_for_run(
        session, school_id=school, camera_key=room, class_key=klass,
        video_sha256=document["provenance"]["video_sha256"], started_at=started_at,
        duration_seconds=duration)
    result.notes.extend(link.notes or [])
    if link.lesson is not None:
        return link.lesson
    if link.overlaps is not None and not allow_overlap:
        # The time is printed in the SCHOOL's zone, not UTC. An operator comparing this
        # sentence with a timetable would otherwise read a five-hour discrepancy as proof that
        # the refusal is about some other lesson.
        local = link.overlaps.started_at.astimezone(zone)
        raise Refusal(
            "overlapping_lesson",
            f"эта запись перекрывается по времени с уроком #{link.overlaps.id} "
            f"({local:%d.%m %H:%M} по местному времени). Два пересекающихся файла — это один "
            "и тот же час, посчитанный дважды во всех недельных итогах. Флаг --allow-overlap "
            "оставляет пометку в строке и печатает её в отчёте.",
        )
    return _open_lesson(session, document, started_at=started_at, room=room, source=source,
                        klass=klass, school=school, is_demo=is_demo, link=link, zone=zone,
                        forced=allow_overlap and link.overlaps is not None)


def _open_lesson(session: Any, document: dict[str, Any], *, started_at: datetime | None,
                 room: str, source: str, klass: str, school: int, is_demo: bool,
                 link: place_rules.LessonLink, zone: Any, forced: bool) -> ClassvisionLesson:
    lesson_block = document.get("lesson") or {}
    session_block = (document["provenance"].get("session") or {})
    minutes = float(lesson_block.get("duration_minutes") or 0.0)
    day, iso_year, iso_week = place_rules.iso_week_of(started_at, zone)
    lesson = ClassvisionLesson(
        school_id=school,
        is_demo=is_demo,
        camera_key=room,
        camera_key_source=source,
        class_key=klass,
        started_at=started_at,
        ended_at=None if started_at is None else started_at + _seconds(minutes * 60.0),
        date_local=day,
        iso_year=iso_year,
        iso_week=iso_week,
        timezone=None if started_at is None else str(zone),
        duration_minutes=minutes,
        selected_run_id=str(document["run_id"]),
        continues_lesson_id=None if link.continues is None else link.continues.id,
        part_count=len(session_block.get("parts") or []) or 1,
        overlap_allowed=forced,
        overlap_note=(
            f"импортировано поверх пересечения с уроком #{link.overlaps.id} по требованию "
            "оператора (--allow-overlap): этот час может быть посчитан дважды."
            if forced and link.overlaps is not None else None
        ),
    )
    session.add(lesson)
    session.flush()
    if link.precedes is not None:
        # The seam was met from the other side: the stored lesson is the CONTINUATION of this
        # one, so it is the stored row that gets the pointer. Without this the chain would run
        # backwards whenever the longer file of a split recording was imported first.
        link.precedes.continues_lesson_id = lesson.id
        session.flush()
    return lesson


def _seconds(value: float) -> Any:
    from datetime import timedelta

    return timedelta(seconds=value)


def _store_run(session: Any, document: dict[str, Any], *, lesson: ClassvisionLesson,
               started_at: datetime | None, is_demo: bool) -> ClassvisionRun:
    provenance = document["provenance"]
    uncertainty = document.get("uncertainty") or {}
    lesson_block = document.get("lesson") or {}
    model = provenance.get("model") or {}
    run = ClassvisionRun(
        lesson_id=lesson.id,
        is_demo=is_demo,
        run_id=str(document["run_id"]),
        schema_version=provenance["schema"],
        video_path=str(provenance.get("video_path") or ""),
        video_sha256=provenance["video_sha256"],
        video_bytes=int(provenance.get("video_bytes") or 0),
        started_at=started_at,
        clock_source=str(provenance.get("clock_source") or "unknown"),
        clock_drift_seconds=float_or_none(provenance.get("clock_drift_seconds")),
        sample_fps=float(provenance.get("sample_fps") or 0.0),
        analysed_frames=int(provenance.get("analysed_frames") or 0),
        duration_seconds=float(provenance.get("duration_seconds") or 0.0),
        thresholds_sha=thresholds_sha(provenance.get("thresholds") or {}),
        model_weights=str(model.get("weights") or ""),
        model_imgsz=model.get("imgsz"),
        model_device=str(model.get("device") or "") or None,
        room_layout=(provenance.get("room") or {}).get("layout") or {},
        session=provenance.get("session"),
        pupil_places=int(lesson_block.get("pupil_seats") or 0),
        adult_seat_id=lesson_block.get("adult_seat"),
        observations_total=int(uncertainty.get("observations_total") or 0),
        observations_unassigned=int(uncertainty.get("observations_unassigned") or 0),
        observations_unreadable=int(uncertainty.get("observations_unreadable") or 0),
        frames_with_no_person=int(uncertainty.get("frames_with_no_person") or 0),
        seats_never_settled=int(uncertainty.get("seats_never_settled") or 0),
        provenance=provenance,
        uncertainty=uncertainty,
        caveats=document["caveats"],
        unmeasured=lesson_block.get("unmeasured") or [],
        analysed_at=_iso_to_utc(provenance.get("analysed_at")),
        imported_at=datetime.now(UTC),
    )
    session.add(run)
    session.flush()
    return run


def thresholds_sha(thresholds: dict[str, Any]) -> str:
    """A short, order-independent digest of what the numbers were computed under."""
    payload = json.dumps(thresholds, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def status(school_id: int | None = None) -> list[str]:
    """Row counts for ONE school, real and demonstration counted APART.

    Two lines rather than one total, because a figure that adds a demonstration to a
    measurement is the specific output this whole branch exists to make impossible.
    """
    lines: list[str] = []
    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        for label, demo in (("измерения", False), ("ДЕМО", True)):
            mine = (ClassvisionLesson.school_id == school, ClassvisionLesson.is_demo == demo)
            lessons = session.scalar(
                select(func.count()).select_from(ClassvisionLesson).where(*mine)) or 0
            runs = session.scalar(
                select(func.count()).select_from(ClassvisionRun)
                .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
                .where(*mine)) or 0
            rows = session.scalar(
                select(func.count()).select_from(ClassvisionPlaceLesson)
                .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
                .where(*mine)) or 0
            frames = session.scalar(
                select(func.count()).select_from(ClassvisionFrame)
                .join(ClassvisionLesson, ClassvisionFrame.lesson_id == ClassvisionLesson.id)
                .where(*mine)) or 0
            readings = session.scalar(
                select(func.count()).select_from(ClassvisionReading)
                .join(ClassvisionRun, ClassvisionReading.run_id == ClassvisionRun.id)
                .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
                .where(*mine)) or 0
            named = session.scalar(
                select(func.count()).select_from(ClassvisionPlaceLesson)
                .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
                .where(*mine).where(ClassvisionPlaceLesson.person_id.is_not(None))) or 0
            lines.append(
                f"{label}: уроков {lessons}, прогонов {runs}, строк по местам {rows} "
                f"(с именем {named}), кадров {frames}, записок {readings}")
    return lines


def _iso_to_utc(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    moment = datetime.fromisoformat(str(raw))
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)

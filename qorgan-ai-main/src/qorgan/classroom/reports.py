"""The lesson report: four counts per anonymous track, and what they are not.

**Every track observed appears, including the ones whose counts are all zero**, and that
is the shape the canteen taught this codebase. Its «кто не ел» report could not be
answered by the legacy at all, because the canteen log only ever contained children who
had been SEEN -- so the question "who is missing from this log" had nothing to be asked
against. Here the equivalent mistake would be listing only the tracks that raised a hand:
"nobody put their hand up" and "we did not look" would render identically, and the second
is the one that needs saying.

The canteen joins against `persons`, the roster. **This report cannot, and must not.**
There is no roster to join a classroom track against, because a track has no identity --
§8 promised the school no identification in a classroom, and the corridor measurement
(14 970 faces, median 11.5 px, zero recognised) says the promise costs nothing. So the
completeness this report can honestly offer is over TRACKS, not over children, and the
gap between those two is not papered over: it is `caveat()`, which every surface must
render beside the numbers exactly as the canteen renders `DayReport.caveat()`.

**A LESSON ID IN A URL IS THE LEAK THIS FILE HAD TO CLOSE.** `/lessons/{id}` puts a small
integer in front of anyone with `VIEW_LESSON_METRICS`, and a primary-key fetch would hand
whoever types a number the report of another school's classroom -- when it ran, on which
camera, and every movement counter in it. `lesson_report` therefore scopes the lesson, the
camera name and the track rows as three separate statements: the last two are reached from
a lesson already proved to be this school's, but the proof and the reads are different
statements and only one of them is in front of you when you edit this. `None` is the same
answer an id nobody holds already gets, which is right -- whether that lesson exists
elsewhere on the installation is not this school's business to be told.

**There is no score, no ranking and no threshold that fires.** Not an oversight and not a
phase-two feature. The school was told the system reports facts with numbers and a person
decides -- «Никаких диагнозов и никаких направлений к психологу от системы» -- and a
number that flags a child is a decision already taken. Sorting is by track id, so the
report does not even imply an order of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Lesson, LessonTrack
from qorgan.db.tenancy import owned_by, resolve_school_id, scope
from qorgan.enums import LessonCloseReason, LessonState


@dataclass(frozen=True, slots=True)
class TrackRow:
    """One anonymous track. Deliberately carries no name and nothing name-shaped."""

    track_id: int
    observed_seconds: float
    observations: int
    settled: bool
    hand_raises: int
    stands: int
    away_seconds: float
    brief_excursions: int

    @property
    def measured_place(self) -> bool:
        """Whether `stands` and `away_seconds` mean anything for this track.

        False means UNMEASURED, never zero: no seated baseline was ever established, so
        there was nothing to have risen from or moved away from. `hand_raises` needs no
        baseline and is meaningful either way. Surfaces must render the two differently --
        a zero that also means "unknown" is the defect `migrations/0005` was written
        about, one layer down.
        """
        return self.settled


@dataclass(frozen=True, slots=True)
class LessonReport:
    lesson_id: int
    camera_name: str
    started_at: datetime
    ended_at: datetime | None
    state: LessonState
    close_reason: LessonCloseReason | None
    min_presence_seconds: float

    # Tracks observed for at least `min_presence_seconds`.
    present: tuple[TrackRow, ...]
    # Tracks observed for less. Kept and counted, never dropped: they are the direct
    # measure of how badly the tracker is fragmenting, and dropping them would make a
    # room the system kept losing look like a calm one.
    fragments: tuple[TrackRow, ...]

    ambiguous_observations: int
    unclaimed_observations: int
    dropped_tracks: int
    resumed_count: int

    @property
    def tracks(self) -> int:
        """Tracks observed. **NOT the number of children**, in either direction -- see
        `caveat`. One child occluded and re-acquired is two tracks; two children who
        cross can become one."""
        return len(self.present) + len(self.fragments)

    @property
    def unsettled(self) -> int:
        """Present tracks with no seated baseline, whose place metrics are unmeasured."""
        return sum(1 for row in self.present if not row.measured_place)

    def summary(self) -> str:
        return (
            f"lesson {self.lesson_id} on {self.camera_name}: {len(self.present)} track(s) "
            f"observed for {self.min_presence_seconds:.0f}s or more, "
            f"{len(self.fragments)} shorter fragment(s). "
            f"A track is not a child."
        )

    def caveat(self) -> tuple[str, ...]:
        """The sentences that must travel with these numbers, on every surface.

        One wording, rendered by the page and by anything else that grows later, for the
        reason the canteen's caveat is written once: four hand-copied versions is four
        chances for one to quietly stop being true, which is this codebase's signature
        disease.

        The first two hold on every lesson, whatever the counts, and they are the two that
        matter most: a track is not a child, and not one threshold behind these numbers
        has been checked against a real lesson. The rest appear only when there is a
        number to talk about.
        """
        lines = [
            "Все показатели — по анонимному ТРЕКУ, а не по ученику. Система не узнаёт "
            "лица в классе и не пытается: на таком расстоянии это не работает, и это "
            "измерено, а не предположено — из 14 970 лиц с камер школы не узнано ни одно. "
            f"«{self.tracks} трек(ов)» — это НЕ «{self.tracks} учеников»: одного ребёнка, "
            "которого заслонили и снова нашли, система считает двумя треками.",
            "Ни один порог в этом отчёте не проверен на записи настоящего урока — "
            "записей у нас нет. Числа получены расчётом от пропорций тела, а не "
            "измерением. Это счётчики наблюдаемых движений, а не оценка ученика и не "
            "повод для вывода о нём.",
        ]
        lines.extend(self._conditional_lines())
        return tuple(lines)

    def _conditional_lines(self) -> list[str]:
        """The parts that exist only when there is an N. An absent line says nothing."""
        lines: list[str] = []
        if self.unsettled:
            lines.append(
                f"Для {self.unsettled} трек(ов) не удалось определить исходное место. "
                "Для них «вставал» и «вне места» НЕ ИЗМЕРЕНЫ — это не ноль."
            )
        if self.ambiguous_observations:
            lines.append(
                f"{self.ambiguous_observations} наблюдений отброшено: было не определить, "
                "кому из сидящих рядом принадлежит поза. Система предпочитает не "
                "засчитать движение, чем засчитать его не тому."
            )
        if self.resumed_count:
            lines.append(
                f"Во время урока рабочий процесс перезапускался {self.resumed_count} раз. "
                "После перезапуска нумерация треков начинается заново, поэтому один "
                "ученик почти наверняка попал в отчёт несколькими треками."
            )
        if self.dropped_tracks:
            lines.append(
                f"{self.dropped_tracks} новых треков не взяты в работу: достигнут предел "
                "одновременно отслеживаемых. Часть класса в это время не наблюдалась."
            )
        return lines


def lesson_report(lesson_id: int, school_id: int | None = None) -> LessonReport | None:
    """Everything recorded for one lesson of ONE school. `None` if there is no such lesson.

    NOT `session.get(Lesson, lesson_id)`, and all three statements below are scoped
    separately -- the module docstring argues both.

    The presence threshold comes off the LESSON ROW, not from today's configuration: a
    report is a statement about a past hour, and re-reading the YAML would restate it
    under a threshold nobody applied to it.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        lesson = session.scalar(
            scope(select(Lesson), Lesson, school).where(Lesson.id == lesson_id)
        )
        if lesson is None:
            return None

        camera_name = session.scalar(
            select(Camera.name).where(
                Camera.id == lesson.camera_id, owned_by(Camera, school)
            )
        )
        rows = session.scalars(
            scope(select(LessonTrack), LessonTrack, school)
            .where(LessonTrack.lesson_id == lesson_id)
            # By track id, never by any count. A report sorted by "most hand raises"
            # ranks children, and ranking is the judgement the school asked us not to make.
            .order_by(LessonTrack.track_id)
        ).all()

        tracks = [_row(track) for track in rows]
        floor = lesson.min_presence_seconds
        return LessonReport(
            lesson_id=lesson.id,
            camera_name=camera_name or f"camera {lesson.camera_id}",
            started_at=lesson.started_at,
            ended_at=lesson.ended_at,
            state=lesson.state,
            close_reason=lesson.close_reason,
            min_presence_seconds=floor,
            present=tuple(t for t in tracks if t.observed_seconds >= floor),
            fragments=tuple(t for t in tracks if t.observed_seconds < floor),
            ambiguous_observations=lesson.ambiguous_observations,
            unclaimed_observations=lesson.unclaimed_observations,
            dropped_tracks=lesson.dropped_tracks,
            resumed_count=lesson.resumed_count,
        )


def recent_lessons(limit: int = 50, school_id: int | None = None) -> tuple[LessonReport, ...]:
    """This school's most recent lessons, newest first, each carrying its own caveat.

    Unscoped this is the whole installation's teaching day on one page, newest first --
    so a school with more cameras than another would simply push the other's lessons off
    the front page, and neither would be able to tell.
    """
    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        ids = list(
            session.scalars(
                scope(select(Lesson.id), Lesson, school)
                .order_by(Lesson.started_at.desc())
                .limit(limit)
            ).all()
        )
    reports = (lesson_report(lesson_id, school) for lesson_id in ids)
    return tuple(report for report in reports if report is not None)


def _row(track: LessonTrack) -> TrackRow:
    return TrackRow(
        track_id=track.track_id,
        observed_seconds=track.observed_seconds,
        observations=track.observations,
        settled=track.settled,
        hand_raises=track.hand_raises,
        stands=track.stands,
        away_seconds=track.away_seconds,
        brief_excursions=track.brief_excursions,
    )

"""Several DVR files that are one lesson, assembled into one timeline before anything is
measured.

--------------------------------------------------------------------------------
**THE PROBLEM. A lesson is not a file, and on this school's second camera it never is.**

The D14 recorder closes a file and opens the next one whenever it feels like it. The pair
we hold is:

    D14_20260815101759.mp4   817.998 s   10:17:59 -> 10:31:36.998
    D14_20260815103136.mp4  2903.865 s   10:31:36 -> 11:19:59.865

(Those durations are `frame_count / fps` at the fps `decode.probe` actually reports —
**20.00127**, not the round 20 the container header suggests. The difference is 0.05 s over
the short file and 0.19 s over the long one, which is small until it is subtracted to make
a seam: at a round 20 fps the seam computes as -1.05 s and at the true rate as -0.998 s.
Both are inside the tolerance below, so nothing about the merge turns on it — but the
tolerance is JUSTIFIED by that measurement, so the measurement has to be the one the code
makes and not the one arithmetic on a round number suggests.)

which is **one lesson of 62 minutes**, cut in two at a moment that means nothing to anybody
in the room. Analysing the two files separately does not produce a slightly worse report,
it produces two reports that cannot be added together:

  * **Two seat discoveries.** `room/seats.discover` clusters the anchors of whatever it is
    given and numbers the resulting places by reading order. Run twice, it numbers twice,
    and `seat_3` in the first file is `seat_3` in the second only by luck. Nothing in the
    JSON marks the coincidence, so a reader adding the two hand-raise counters is adding
    two different children's counters and cannot tell.
  * **Two sets of baselines.** Every state predicate compares a pupil to their own settled
    posture (`states.Baselines`). A baseline needs observations to settle; splitting the
    lesson makes the pipeline settle twice, and the second settling happens on a room that
    has already been working for thirteen minutes — a different posture from the one it
    would have learnt from the start.
  * **Two lesson windows, two denominators.** `pipeline.lesson_window` measures when the
    room was occupied. Two windows means every share in the report is a share of a
    different thing, and the two are not commensurable even when they look it.

So the merge cannot happen after analysis. There is nothing to merge afterwards: no key
ties one artefact's seat 3 to the other's. It has to happen **before seat discovery**, and
that is the entire content of this module.

--------------------------------------------------------------------------------
**WHAT THIS MODULE REFUSES TO DO.**

It will not concatenate files that are not actually continuous. The failure it exists to
prevent is a coffee break, a class change, or a whole different lesson being glued onto the
end of this one and reported as 90 uninterrupted minutes in which «место 4 отсутствовало
30 минут». Contiguity is therefore **checked against the wall clock, not assumed from the
argument order**, and a real gap raises `NotOneLesson` instead of being averaged into the
result. What the check can and cannot see is stated in `MAX_SEAM_GAP_SECONDS` below.

It also will not merge two files that are not the same view of the same room. Seat centres
are pixel coordinates; a part recorded at a different resolution would place its people in
a coordinate system that means something else, and the clustering would happily average the
two. Frame geometry must match exactly, and the DVR's channel token (`D14_`) must agree
wherever the file names carry one. What no code here can check is whether somebody moved
the camera between the two files — that is stated in the artefact as an assumption rather
than silently relied on.

--------------------------------------------------------------------------------
**TRACK IDS ACROSS A SEAM: THEY DO NOT SURVIVE, AND THAT IS FINE — BUT SAY IT.**

The tracker is restarted for every file (`pipeline.detect_pass` calls
`model.reset_tracker()`), so the second file's track 3 has no relationship whatsoever to
the first file's track 3. This costs almost nothing, because identity in this package rides
on **places, not tracks** (`room/seats.py`): a seat accumulates hundreds of observations
per lesson and does not die at a file boundary. Tracks are used only as a per-observation
label carried on `room.seats.Anchor`, and nothing groups by them.

"Almost nothing" is not "nothing", so the merge does not leave the collision lying around
for a future reader to trip over: every part's ids are renumbered into a disjoint block
(`TRACK_ID_PART_STRIDE`). After the merge, two observations sharing a track id are
guaranteed to come from the same file. Code that assumes track continuity across a seam is
now impossible to write by accident; it would have to invent an id that does not exist.

--------------------------------------------------------------------------------
**THE SEAM IS RECORDED, NOT SMOOTHED.** `provenance.session` carries every constituent
file with its own sha256, start and duration, and every seam with its measured gap. A
reader who wants to know whether an episode at 13 min 37 s is a real event or an artefact
of the join can see that the join is exactly there. An assembled artefact that looked like
a single recording would be the most convincing wrong number this package could produce.

**A GAP IS UNSEEN TIME, NOT AN EMPTY SEAT**, and the arithmetic already keeps them apart.
Seconds nobody recorded contribute no analysed frames, so no seat is marked absent for
them — `absent_observations` counts frames in which a place we WERE looking at held nobody,
and there is no frame here to look at.

**And that means `coverage` does NOT fall across a gap**, which is worth stating because
the obvious guess is that it does. `SeatLedger.coverage` is
`observations / (observations + absent_observations)`, and `pipeline.assemble` calls
`note_absent` only for frames it actually analysed — so unrecorded seconds move neither the
numerator nor the denominator. That is correct: coverage answers «какую долю ТОГО, НА ЧТО
МЫ СМОТРЕЛИ, мы видели», and diluting it with seconds nobody filmed would turn a fact about
the camera's view into a fact about the recorder. The signal that footage is missing is
carried by three other numbers instead, and a reader needs to know it is them and not
coverage: `provenance.analysed_frames` against the window length, the pair
`duration_seconds` (the lesson's span) and `recorded_seconds` (how much footage exists
inside it), and the seam's own sentence in `uncertainty.notes`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from classvision.video import clock as clockmod
from classvision.video import decode

# How far apart two parts' wall clocks may be before this stops being one recording.
#
# CHOSEN at 2.0 s, from the resolution of the evidence rather than from taste. Three
# quantities go into a seam and each has a known error:
#
#   * a part's START comes from the DVR — either the burned-in overlay or the export name
#     (`video/clock.py`) — and both are printed to a whole second, so two starts can be
#     up to 1 s apart from rounding alone;
#   * a part's DURATION is `frame_count / fps`, exact to one frame (0.05 s at 20 fps);
#   * `clock.verify_against_overlay` already treats 2 s as agreement between a file name
#     and the picture, for exactly this reason, and a second tolerance in the same package
#     that means the same thing must not be a different number.
#
# Measured on the pair this was written for: the seam is **-0.998 s** — the second file's
# name claims a start one second before the first file's last frame. That is a clean DVR
# handover seen through one-second stamps, and it is inside 2.0 s with room to spare.
# (Re-derived from `decode.probe`, whose fps is 20.00127 rather than a round 20; the round
# number gives -1.05 s. Same conclusion, but this constant is argued FROM the number, so
# the number here is the one `resolve()` computes.)
#
# The bound is symmetric, and the two directions fail differently. A POSITIVE gap is
# missing footage: seconds of the lesson nobody recorded, which the timeline then carries
# as a hole rather than as silence. A NEGATIVE gap is duplicated footage: the same wall
# seconds present in both files, which would be counted twice if it were kept — so it is
# trimmed, and the trim is reported (`Seam.trimmed_*`).
#
# What 2.0 s deliberately does NOT admit: anything a person would call a break. A DVR
# restart, a swapped disk, a class change or a second lesson are tens of seconds at the very
# least, and every one of them ends up as `NotOneLesson` rather than as a silent 30-minute
# absence recorded against a child's place.
#
# **The same quantity exists once more in this package and the two must move together.**
# `cabinet/store.CONTINUATION_TOLERANCE_SECONDS` is 2.0 for exactly this reason, because the
# store faces the same seam one layer up: it decides whether two ALREADY-ANALYSED artefacts
# are a continuation. They are separate constants because they guard separate decisions —
# this one refuses to merge FILES, that one refuses to double-count LESSONS — but a camera
# whose stamps get coarser invalidates both, and a reader who changes one and not the other
# has left the package believing two different things about one recorder.
MAX_SEAM_GAP_SECONDS = 2.0

# Track ids are renumbered `part_index * TRACK_ID_PART_STRIDE + id`. CHOSEN far above any
# id a tracker will issue in one lesson (ByteTrack allocates a few hundred over 50 minutes
# on this footage), so the blocks cannot touch, and small enough that the result stays an
# ordinary int32 for eight parts. A part index is recoverable as `id // STRIDE`, which is
# what makes "did these two observations come from the same file?" answerable rather than
# an assumption.
TRACK_ID_PART_STRIDE = 1_000_000

# How authoritative each clock source is. Used only to describe the SESSION with the
# weakest evidence any of its parts rests on -- see `Session.clock_source`.
_SOURCE_RANK: dict[str, int] = {
    clockmod.ClockSource.DECLARED.value: 3,
    clockmod.ClockSource.OVERLAY.value: 2,
    clockmod.ClockSource.FILENAME.value: 1,
    clockmod.ClockSource.UNKNOWN.value: 0,
}


class NotOneLesson(ValueError):
    """These files are not one continuous recording of one room, and were not merged.

    Raised rather than worked around. Every alternative — concatenating anyway, picking the
    longest file, dropping the odd one out — produces a report that reads like a measurement
    of a lesson that did not happen.
    """


@dataclass(frozen=True, slots=True)
class Part:
    """One constituent recording, placed on the session timeline."""

    index: int
    path: Path
    sha256: str
    size_bytes: int
    started_at: dt.datetime | None
    clock_source: str
    clock_drift_seconds: float | None
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    # Where this file's t=0 lands on the session timeline. Derived from the WALL CLOCK
    # (`started_at - session start`), not by butt-joining durations: the wall clock is what
    # the report prints and what the weekly trend keys on, so the timeline is anchored to
    # it and a seam's error stays local instead of accumulating into every later part.
    offset_seconds: float

    @property
    def ends_at(self) -> dt.datetime | None:
        if self.started_at is None:
            return None
        return self.started_at + dt.timedelta(seconds=self.duration_seconds)

    @property
    def end_offset_seconds(self) -> float:
        return self.offset_seconds + self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.size_bytes,
            "started_at": None if self.started_at is None else self.started_at.isoformat(),
            "clock_source": self.clock_source,
            "clock_drift_seconds": self.clock_drift_seconds,
            "duration_seconds": round(self.duration_seconds, 3),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "offset_seconds": round(self.offset_seconds, 3),
        }


@dataclass(slots=True)
class Seam:
    """The join between two parts, with the measurement that licensed it.

    Mutable in one respect only: `trimmed_*` are filled in by `detect_session`, because how
    much duplicated footage an overlap actually cost is a property of the sampled
    observations and not of the file headers.
    """

    after_part: int              # this seam follows part `after_part` (0-based)
    at_session_seconds: float    # where the earlier part ends on the session timeline
    at_wall: str | None
    gap_seconds: float           # positive = missing footage, negative = duplicated
    trimmed_observations: int = 0
    trimmed_frames: int = 0

    @property
    def kind(self) -> str:
        if self.gap_seconds > 0.05:
            return "gap"          # seconds of the lesson that nobody recorded
        if self.gap_seconds < -0.05:
            return "overlap"      # the same seconds recorded twice
        return "flush"

    def to_dict(self) -> dict[str, Any]:
        return {
            "after_part": self.after_part,
            "at_session_seconds": round(self.at_session_seconds, 3),
            "at_wall": self.at_wall,
            "gap_seconds": round(self.gap_seconds, 3),
            "kind": self.kind,
            # Only ever non-zero on an overlap. These are the observations that WOULD have
            # been counted twice: the same wall seconds present in both files. Reported
            # rather than quietly dropped, because "we removed 3 observations" and "there
            # were 3 fewer observations" are different facts.
            "trimmed_observations": self.trimmed_observations,
            "trimmed_frames": self.trimmed_frames,
        }


@dataclass(frozen=True, slots=True)
class Session:
    """An ordered set of recordings established to be one continuous lesson."""

    parts: tuple[Part, ...]
    seams: tuple[Seam, ...]
    tolerance_seconds: float
    notes: tuple[str, ...] = ()

    # -- what the session is, as a whole -------------------------------------------
    @property
    def started_at(self) -> dt.datetime | None:
        return self.parts[0].started_at

    @property
    def duration_seconds(self) -> float:
        """The span of the timeline, seams included. NOT the sum of the durations: an
        overlap is counted once and a gap is counted, because both are real seconds of the
        lesson's clock and the report divides by this."""
        return self.parts[-1].end_offset_seconds - self.parts[0].offset_seconds

    @property
    def recorded_seconds(self) -> float:
        """How much footage actually exists, which differs from `duration_seconds` by the
        seams. Two numbers, because a hole in the recording is not the same as a hole in
        the lesson and the artefact must not let them look alike."""
        return sum(p.duration_seconds for p in self.parts) - sum(
            -s.gap_seconds for s in self.seams if s.gap_seconds < 0)

    @property
    def frame_count(self) -> int:
        return sum(p.frame_count for p in self.parts)

    @property
    def size_bytes(self) -> int:
        return sum(p.size_bytes for p in self.parts)

    @property
    def clock_source(self) -> str:
        """The WEAKEST source any part rests on, not the first part's.

        The session start comes from part one, but the claim that these files are one
        lesson rests on every part's clock: a part dated only by its file name is the
        evidence that seam was checked against. Reporting the best of them would describe a
        merge that was never made.
        """
        return min((p.clock_source for p in self.parts),
                   key=lambda s: _SOURCE_RANK.get(s, 0))

    @property
    def digest(self) -> str:
        """Content identity of the ASSEMBLY, not of any file.

        Covers every part's own hash and where it was placed, so re-assembling the same
        files under a different tolerance (a different placement) is a different session and
        therefore a different `run_id`. A digest over the file hashes alone would let two
        different assemblies overwrite each other in the school's database.

        A session of ONE file is that file: nothing was assembled, every measured number is
        the one `pipeline.analyse` would produce, and giving it a second identity would put
        the same measurement in the school's database twice under two `run_id`s.
        """
        if len(self.parts) == 1:
            return self.parts[0].sha256
        payload = "\n".join(
            f"{p.index}:{p.sha256}:{p.offset_seconds:.3f}:{p.duration_seconds:.3f}"
            for p in self.parts)
        return hashlib.sha256(payload.encode()).hexdigest()

    def wall_clock(self) -> clockmod.WallClock:
        """Session seconds -> wall time. Affine, for the same reason one file's is.

        Legitimate across a seam only because the seam was measured: each part is placed at
        its own clock's offset, so this is exact for part one and correct for every later
        part to within the seam's own measured error (-0.998 s on the pair this was written
        for, i.e. inside the one-second stamps the DVR prints).
        """
        if self.started_at is None:
            return clockmod.WallClock.unknown()
        return clockmod.WallClock(
            started_at=self.started_at,
            source=clockmod.ClockSource(self.clock_source),
            drift_seconds=self.parts[0].clock_drift_seconds,
        )

    def part_at(self, session_seconds: float) -> Part | None:
        """Which file a moment on the session timeline came from."""
        for part in self.parts:
            if part.offset_seconds <= session_seconds <= part.end_offset_seconds:
                return part
        return None

    def to_dict(self) -> dict[str, Any]:
        """The `provenance.session` block: how this artefact was assembled, in full.

        Present and non-null ONLY on assembled runs. A single-recording artefact carries
        `null` here, so a consumer can tell the two apart by a field rather than by counting
        something.
        """
        return {
            "assembled": True,
            "digest": self.digest,
            "parts": [p.to_dict() for p in self.parts],
            "seams": [s.to_dict() for s in self.seams],
            "tolerance_seconds": self.tolerance_seconds,
            "duration_seconds": round(self.duration_seconds, 3),
            "recorded_seconds": round(self.recorded_seconds, 3),
            "clock_source": self.clock_source,
            # Two statements a reader should not have to reconstruct from the numbers.
            "track_ids": (
                "переномерованы по файлам (шаг "
                f"{TRACK_ID_PART_STRIDE}): трекер перезапускается на каждом файле, поэтому "
                "непрерывность треков через стык невозможна и ни один счётчик на неё не "
                "опирается — единицей накопления является МЕСТО, а не трек."),
            "assumption": (
                "Проверено: время начала каждого файла, непрерывность на стыках (допуск "
                f"{self.tolerance_seconds} с), одинаковый размер кадра, а также канал DVR — "
                "если имена файлов его содержат (иначе это сказано в notes). НЕ проверено и "
                "принято как допущение: камеру между файлами не двигали. Сдвиг камеры "
                "сместил бы координаты мест, и заметить это может только человек, "
                "посмотревший оба файла."),
            "notes": list(self.notes),
        }


def resolve(paths: Sequence[str | Path], *,
            tolerance_seconds: float = MAX_SEAM_GAP_SECONDS,
            declared: Mapping[str | Path, dt.datetime] | None = None) -> Session:
    """Establish that these files are one lesson, or refuse to say so.

    The order of the arguments is not trusted: each file is dated from its own evidence
    (`video/clock.resolve` — a human's declaration, then the burned-in overlay, then the
    DVR's export name checked against the overlay) and the session is ordered by the result.
    A caller who lists the parts in the wrong order gets the right session; a caller who
    lists parts of two different lessons gets `NotOneLesson`.
    """
    if not paths:
        raise NotOneLesson("no files given; a session is at least one recording")

    declared = {str(Path(k)): v for k, v in (declared or {}).items()}
    entries: list[tuple[dt.datetime | None, str, float | None, Path, decode.VideoInfo]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"нет файла: {path}")
        info = decode.probe(path)
        if info.fps <= 0 or info.frame_count <= 0:
            raise NotOneLesson(
                f"{path.name} reports fps={info.fps} and {info.frame_count} frames, so its "
                "duration is unknown. A seam cannot be checked against a duration that does "
                "not exist.")
        wall, _evidence = clockmod.resolve(path, declared=declared.get(str(path)))
        started = None if wall.source is clockmod.ClockSource.UNKNOWN else wall.started_at
        entries.append((started, wall.source.value, wall.drift_seconds, path, info))

    # -- can these be ordered at all? ----------------------------------------------
    undated = [e for e in entries if e[0] is None]
    if undated and len(entries) > 1:
        names = ", ".join(e[3].name for e in undated)
        raise NotOneLesson(
            f"cannot date {names}: neither the burned-in overlay nor the file name says "
            "when the recording started. Several files can only be established to be one "
            "lesson by their clocks, so an undated part cannot join a session. Supply the "
            "start with `declared=` if a human knows it.")

    entries.sort(key=lambda e: (e[0] or dt.datetime(1970, 1, 1), e[3].name))

    by_hash: dict[str, Path] = {}
    from classvision.report.artefact import sha256_of

    digests: list[str] = []
    for _started, _source, _drift, path, _info in entries:
        digest = sha256_of(path)
        if digest in by_hash:
            raise NotOneLesson(
                f"{path.name} and {by_hash[digest].name} are byte-identical. Merging a file "
                "with itself would double every counter in the report.")
        by_hash[digest] = path
        digests.append(digest)

    # -- same view of the same room? -----------------------------------------------
    notes: list[str] = []
    first = entries[0][4]
    for _started, _source, _drift, path, info in entries[1:]:
        if (info.width, info.height) != (first.width, first.height):
            raise NotOneLesson(
                f"{path.name} is {info.width}x{info.height} but {entries[0][3].name} is "
                f"{first.width}x{first.height}. Seat centres are pixel coordinates in one "
                "frame; two resolutions are two coordinate systems and clustering them "
                "together would put places where nobody sat.")
        if abs(info.fps - first.fps) > 0.01:
            notes.append(
                f"{path.name} снят с частотой {info.fps:.2f} к/с против "
                f"{first.fps:.2f} к/с у первого файла: разбор идёт по времени, поэтому "
                "это не ошибка, но разные потоки — повод проверить, та ли это камера.")

    channels = {_channel(e[3]) for e in entries}
    if len(channels) > 1 and None not in channels:
        raise NotOneLesson(
            "the file names carry different DVR channels ("
            + ", ".join(sorted(str(c) for c in channels))
            + "). Two channels are two cameras, and a merged seat map would be a map of "
              "neither room.")
    if None in channels:
        notes.append("имя файла не содержит канала DVR: принадлежность всех частей одной "
                     "камере не подтверждена именами и принята как допущение.")

    # -- contiguity ------------------------------------------------------------------
    session_start = entries[0][0]
    parts: list[Part] = []
    seams: list[Seam] = []
    for index, ((started, source, drift, path, info), digest) in enumerate(
            zip(entries, digests, strict=True)):
        offset = 0.0 if (started is None or session_start is None) else (
            started - session_start).total_seconds()
        part = Part(
            index=index, path=path, sha256=digest, size_bytes=path.stat().st_size,
            started_at=started, clock_source=source, clock_drift_seconds=drift,
            fps=info.fps, frame_count=info.frame_count,
            width=info.width, height=info.height,
            duration_seconds=info.duration_seconds, offset_seconds=offset,
        )
        if index:
            previous = parts[-1]
            gap = part.offset_seconds - previous.end_offset_seconds
            if abs(gap) > tolerance_seconds:
                raise NotOneLesson(
                    f"{previous.path.name} ends at "
                    f"{previous.ends_at.isoformat() if previous.ends_at else '?'} and "
                    f"{part.path.name} starts at "
                    f"{part.started_at.isoformat() if part.started_at else '?'} — a "
                    f"{'gap' if gap > 0 else 'overlap'} of {abs(gap):.2f} s, beyond the "
                    f"{tolerance_seconds} s that a DVR handover can explain. These are not "
                    "one continuous recording. Analyse them separately: two honest "
                    "artefacts are worth more than one that hides a break in the middle of "
                    "a child's timeline.")
            wall_at = previous.ends_at
            seams.append(Seam(
                after_part=previous.index,
                at_session_seconds=previous.end_offset_seconds,
                at_wall=None if wall_at is None else wall_at.isoformat(),
                gap_seconds=gap,
            ))
        parts.append(part)

    return Session(parts=tuple(parts), seams=tuple(seams),
                   tolerance_seconds=tolerance_seconds, notes=tuple(notes))


def _channel(path: Path) -> str | None:
    """The DVR channel a file name claims (`D14_20260815101759.mp4` -> `D14`).

    Weak evidence deliberately treated as weak: it is used to REFUSE an obviously wrong
    merge, never to authorise one, and a name that carries no channel produces `None` and a
    note rather than a failure.
    """
    import re

    match = re.match(r"^([A-Za-z]+\d*)[-_]\d{8}", path.stem)
    return match.group(1) if match else None


def detect_session(session: Session, settings, *, use_cache: bool = True):
    """Pass one per file, concatenated onto a single timeline in session seconds.

    Deliberately not a new detector. Each part goes through the ordinary
    `pipeline.detect_pass`, keyed on that file's own content hash, so a session run costs
    nothing extra on a file that was already analysed alone and a single-file re-run costs
    nothing extra after a session run. The merge is arithmetic on the arrays afterwards.

    Three things happen in the concatenation and each is a decision:

      * **Times are shifted by the part's offset**, so everything downstream keeps working
        in "seconds since the lesson's first frame" and no other module needs to know that
        a session exists. That is the whole reason this returns a plain `Detections`.
      * **Track ids are moved into disjoint blocks** (`TRACK_ID_PART_STRIDE`), because the
        tracker restarts per file and equal ids across a seam would be a coincidence
        wearing the appearance of continuity.
      * **An overlap is trimmed, not summed.** Where two parts cover the same wall seconds,
        the later part's duplicate frames are dropped and counted in `Seam.trimmed_*`. The
        alternative is one second of the lesson in which every pupil is observed twice,
        which inflates precisely the counters this package exists to state carefully.
    """
    from classvision.pipeline import Detections, detect_pass

    if len(session.parts) > 1 and (settings.start_seconds or settings.end_seconds):
        raise NotOneLesson(
            "`start_seconds`/`end_seconds` cut a single file and would be applied to every "
            "part of a session, silently taking the same minute out of each. Trim the "
            "session by analysing the part you want on its own.")

    times, xys, confs, boxes, tracks, frame_times = [], [], [], [], [], []
    spent = 0.0
    per_part: list[dict[str, Any]] = []
    # The session second up to which footage has already been placed. Everything before it
    # is duplicated content, whichever file it arrives from.
    covered = float("-inf")
    # ...and the last frame actually kept, which is what the sampling cadence is measured
    # from across a seam.
    last_frame = float("-inf")

    for part in session.parts:
        detections = detect_pass(part.path, settings, use_cache=use_cache)
        spent += detections.seconds_spent

        shifted_frames = detections.frame_times.astype(np.float64) + part.offset_seconds
        shifted_times = detections.times.astype(np.float64) + part.offset_seconds
        # A part covers the HALF-OPEN interval [offset, offset + duration): a later part's
        # frame landing exactly on the previous part's end duplicates nothing, so the
        # comparison is `>=` and not `>`. At 2 fps the difference is one sampled frame per
        # seam, which is one frame of a child's lesson thrown away for a rounding
        # convention.
        #
        # The second condition is about the SAMPLING PHASE, and it was added after
        # measuring the seam on the D14 pair. Each file is sampled from its own first
        # frame, so the two phases are unrelated: part one's last sample landed at
        # 817.948 s and part two's first surviving sample at 817.99994 s — **52 ms apart**,
        # against a nominal 0.5 s interval. Both frames are real, but the ledger charges
        # every observation `sample_interval` seconds of a place's time, so that pair
        # bought 1.0 s of "observed" for 0.05 s of footage, at every seat, at every seam.
        # Requiring a full interval since the previous part's last kept frame puts the
        # sampling back on one cadence across the join. The cost is at most one frame per
        # seam; the alternative is a seam that inflates every counter that touches it.
        floor = max(covered, last_frame + settings.sample_interval) if part.index else covered
        keep_frames = shifted_frames >= floor - 1e-9
        keep_times = shifted_times >= floor - 1e-9

        dropped_frames = int((~keep_frames).sum())
        dropped_obs = int((~keep_times).sum())
        if part.index:
            seam = session.seams[part.index - 1]
            seam.trimmed_frames = dropped_frames
            seam.trimmed_observations = dropped_obs

        frame_times.append(shifted_frames[keep_frames])
        times.append(shifted_times[keep_times])
        xys.append(detections.xy[keep_times])
        confs.append(detections.conf[keep_times])
        boxes.append(detections.box[keep_times])
        # -1 means "the tracker had not committed to an id"; it stays -1, because moving it
        # into a part's block would turn "unknown" into a per-file identity that is not one.
        raw = detections.track[keep_times].astype(np.int64)
        tracks.append(np.where(raw < 0, -1, raw + part.index * TRACK_ID_PART_STRIDE))

        per_part.append({
            "index": part.index,
            "path": str(part.path),
            "analysed_frames": int(keep_frames.sum()),
            "observations": int(keep_times.sum()),
            "frames_trimmed_as_duplicate": dropped_frames,
            "observations_trimmed_as_duplicate": dropped_obs,
            "detect_seconds": round(detections.seconds_spent, 1),
            "offset_seconds": round(part.offset_seconds, 3),
        })
        # The FILE's nominal end, not its last sampled frame: what a later part duplicates
        # is footage, and the sampling rate must not decide how much of it is dropped.
        covered = max(covered, part.end_offset_seconds)
        if keep_frames.any():
            last_frame = float(shifted_frames[keep_frames][-1])

    merged = Detections(
        times=np.concatenate(times).astype(np.float32),
        xy=np.concatenate(xys).astype(np.float32),
        conf=np.concatenate(confs).astype(np.float32),
        box=np.concatenate(boxes).astype(np.float32),
        track=np.concatenate(tracks).astype(np.int64),
        frame_times=np.concatenate(frame_times).astype(np.float32),
        seconds_spent=spent,
    )
    diagnostics = {"parts": per_part,
                   "track_id_part_stride": TRACK_ID_PART_STRIDE}
    return merged, diagnostics


def analyse(paths: Sequence[str | Path], settings=None, *, use_cache: bool = True,
            tolerance_seconds: float = MAX_SEAM_GAP_SECONDS,
            declared: Mapping[str | Path, dt.datetime] | None = None):
    """Several files, one lesson, one artefact.

    Everything after the merge is the ordinary single-recording pipeline
    (`pipeline.assemble`) working on a longer timeline: seat discovery runs **once** over
    the whole session, so `seat_3` means one place from the first frame to the last;
    baselines settle once; the lesson window is measured once; and every share in the
    report has one denominator. That is the point of the module, and it is achieved by
    changing nothing downstream.
    """
    from classvision import pipeline

    settings = settings or pipeline.Settings()
    session = resolve(paths, tolerance_seconds=tolerance_seconds, declared=declared)

    if settings.face_corroboration and len(session.parts) > 1:
        # Refused rather than silently skipped. The face pass re-decodes ONE file at the
        # frame times the pose pass used (`identity/faces.collect_from_video`), and those
        # times are now session seconds spanning several files. Feeding it one part would
        # collect faces for part one and quietly report the result as the whole lesson's
        # corroboration -- evidence about 13 minutes presented as evidence about 62.
        raise NotOneLesson(
            "подтверждение по лицам (--faces) не реализовано для склеенной сессии: проход "
            "по лицам декодирует ОДИН файл по временам кадров, а времена сессии охватывают "
            "несколько файлов. Запустите без --faces (имена от этого не зависят: их даёт "
            "только план рассадки) или разберите файлы по отдельности.")

    detections, merge = detect_session(session, settings, use_cache=use_cache)
    first = session.parts[0]

    block = session.to_dict()
    block["merge"] = merge
    source = pipeline.Source(
        path=str(first.path),
        sha256=session.digest,
        size_bytes=session.size_bytes,
        width=first.width, height=first.height, fps=first.fps,
        frame_count=session.frame_count,
        duration_seconds=session.duration_seconds,
        wall=session.wall_clock(),
        session=block,
        # A session of one file HAS a single file to decode faces from and behaves exactly
        # like `pipeline.analyse`. A session of several does not, and `analyse` refused the
        # combination above rather than picking a part; None makes that structural instead
        # of conventional.
        face_video=first.path if len(session.parts) == 1 else None,
    )
    return pipeline.assemble(source, detections, settings)

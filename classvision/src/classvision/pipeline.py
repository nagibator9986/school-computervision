"""One recording in, one artefact out. The order of the stages is the argument.

**Two passes, and the split is what makes the thresholds arguable.** Pass one is the only
part that needs a model or a GPU: decode, detect, pose, and write every observation to a
cache keyed by the video hash and the model settings. Everything after it — seat
discovery, state classification, episodes, metrics — is pure Python over that cache and
runs in seconds. So changing a threshold and re-running costs seconds rather than the
eleven minutes the model costs, which is the difference between thresholds that were
argued about against real footage and thresholds that were guessed once and shipped.

**Seats are discovered before states are classified, and that ordering is load-bearing.**
Every state predicate compares a pupil to their own baseline, and the baseline belongs to
a place. Classifying first and clustering afterwards would mean the baselines were built
from whatever the tracker happened to call the same person, which is the fragile thing we
are trying not to depend on.

**The adult is separated before any pupil number is computed.** Not filtered afterwards:
on this footage he is nearest the lens, so his shoulder width is three times a pupil's and
he alone produced 35 of the lesson's 45 hand-raise detections. Any statistic computed
before removing him is a statistic about one adult.

**The lesson window is detected, never assumed to be the file.** This recording is 52
minutes of which the first 110 seconds are an empty room — measured, not guessed: people
first appear at t = 110 s (09:56:48) and the class is present by t = 150 s. Dividing by
the file duration instead of the occupied window would understate every share by ~4 %,
silently and consistently, which is exactly the kind of error that survives review.

**What this pipeline can never measure, declared rather than omitted.** `ffprobe` reports
`nb_streams=1`: there is **no audio**. The client's «отвечал» — answering aloud — is
therefore not merely hard here, it is absent from the data, and it is listed in
`unmeasured` on every artefact rather than quietly missing from the report. «У доски» is
`unmeasured` unless somebody configures a `board_zone` — and the reason is the *missing
polygon*, not the camera: this paragraph used to say the board was outside the camera's
view unless a zone was configured, which was true of the first recordings and became false
the moment D14 arrived with a chalkboard in the middle of frame. The same sentence had
propagated into `metrics/teacher.not_an_assessment_ru`, where it told a reader, as a fact
about the hardware, that a board they can see in the verification frames was behind the
lens. A report that lists what it could not see is worth more than one that lists only what
it could — and worth less than nothing if it explains the gap with a guess.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from classvision.geometry import (
    HeadDirection,
    Keypoints,
    anchor,
    head_direction,
    shoulder_width,
)
from classvision.identity import assign as identity
from classvision.ledger import SeatLedger
from classvision.metrics import teacher as teachermod
from classvision.metrics import within_lesson as withinmod
from classvision.metrics.activity import activity
from classvision.metrics.teacher import teacher_metrics
from classvision.report import artefact as art
from classvision.room import perspective
from classvision.room.seats import Anchor, Seat, assign, discover
from classvision.room.zones import RoomLayout, identify_adult
from classvision.states import Thresholds, read
from classvision.video import clock as clockmod
from classvision.video import decode


@dataclass(frozen=True, slots=True)
class Settings:
    sample_fps: float = 2.0
    imgsz: int = 1280
    weights: str = "yolo11m-pose.pt"
    device: str = "mps"
    conf: float = 0.30
    start_seconds: float = 0.0
    end_seconds: float | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    layout: RoomLayout | None = None
    cache_dir: Path = Path("classvision/.cache")

    # -- identity, all optional and all off by default ---------------------------------
    #
    # The pipeline's normal answer is anonymous places (`место 3`), and that report is
    # complete: every metric in it is computed per seat and needs no name. Naming is an
    # extra, and it is opt-in at every level, because each of these settings is a claim
    # somebody has to be accountable for.
    #
    # `seat_map` is the ONLY thing that can put a name in the artefact
    # (`identity/seatmap.py`). Without it `SeatRecord.pupil` stays None and the analysis
    # is unchanged.
    seat_map: Path | None = None
    roster_csv: Path | None = None
    photos_root: Path | None = None
    # Restricting the register to one class shrinks the face gallery from 141 candidates
    # to ~12. That is the single biggest lever on the corroboration step's usefulness --
    # see `identity/roster.restrict_to`.
    class_name: str | None = None
    # Faces stay OFF unless asked for. They cost a second decode pass and an ONNX model,
    # they cannot create a name, and on this camera their measured verdict is usually
    # "inconclusive". A school that wants the extra column switches it on knowingly.
    face_corroboration: bool = False
    face_every_seconds: float = 10.0
    identity_standard: identity.Standard = field(default_factory=identity.Standard)

    @property
    def sample_interval(self) -> float:
        return 1.0 / self.sample_fps if self.sample_fps else 0.5


@dataclass(frozen=True, slots=True)
class Source:
    """What the detections are OF: one recording, or several assembled into one lesson.

    **This type exists so that `assemble` cannot tell the difference.** Everything after
    pass one — seat discovery, baselines, the lesson window, every share — is identical
    work whether the observations came from one file or from a DVR's four; the only things
    that differ are the provenance fields and the fact that a session has no single file to
    decode a face out of. Gathering exactly those differences here keeps `session.py` from
    reimplementing the analysis, which is the failure mode that would let two code paths
    drift into two subtly different definitions of the same lesson.

    `sha256` is the identity of the thing measured: a file's content hash for one recording,
    the assembly digest for a session (`session.Session.digest`). Either way `run_id`
    depends on it, so no two differently-assembled analyses can overwrite one another.
    """

    path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    wall: clockmod.WallClock
    # None for a single recording; the `provenance.session` block for an assembled one.
    session: dict[str, Any] | None = None
    # The one file the optional face pass may decode. None means there is no such file, and
    # `_identity` refuses rather than choosing one -- see `session.analyse`.
    face_video: Path | None = None


@dataclass(slots=True)
class Detections:
    """Pass-one output: every person seen in every analysed frame."""

    times: np.ndarray      # (N,) video seconds
    xy: np.ndarray         # (N, 17, 2)
    conf: np.ndarray       # (N, 17)
    box: np.ndarray        # (N, 4)
    track: np.ndarray      # (N,) track id, -1 when the tracker had not committed
    frame_times: np.ndarray   # (F,) the video seconds of every ANALYSED frame,
                              # including the ones in which nobody was found
    seconds_spent: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=self.times, xy=self.xy, conf=self.conf,
                            box=self.box, track=self.track,
                            frame_times=self.frame_times,
                            seconds_spent=np.float32(self.seconds_spent))

    @classmethod
    def load(cls, path: Path) -> Detections:
        data = np.load(path)
        return cls(times=data["times"], xy=data["xy"], conf=data["conf"], box=data["box"],
                   track=data["track"], frame_times=data["frame_times"],
                   seconds_spent=float(data["seconds_spent"]))


def _identity(video: Path | None, settings: Settings, *, seats: list[Seat],
              adult_seat_id: int | None, excluded_seats: set[int],
              anchors: list[Anchor], index_of: list[int],
              seat_of_anchor: list[int | None], detections: Detections,
              recorded_on) -> tuple[dict[int, Any], dict[str, Any]]:
    """Turn discovered places into named pupils — or, far more often, decline to.

    **Everything here is optional and the pipeline is complete without it.** With no
    seat map configured this returns an empty binding set and a block saying so, and the
    artefact is still a full lesson report about anonymous places. That is the normal
    mode, and it is not a degraded one: every metric above this point is computed per
    seat and needs no name.

    **The order of the three inputs is the evidence standard in miniature.** The roster
    bounds who may be named at all; the seat map is the only thing that can propose a
    name; faces may then agree or disagree with that proposal and nothing else. Face
    evidence is gathered LAST and separately, so it is structurally incapable of
    creating a name — there is no code path from an embedding to a `full_name` that does
    not pass through a human's signed seating plan first.

    The face pass costs a second decode of the video, which is why it is off unless
    asked for. It re-uses the person boxes the pose pass already found, so it never
    detects a person of its own: a face this pipeline cannot attribute to a known seat is
    a face it does not look at.
    """
    from classvision.identity import assign as assign_identity
    from classvision.identity import roster as roster_mod
    from classvision.identity import seatmap as seatmap_mod

    roster = None
    if settings.roster_csv is not None:
        roster = roster_mod.load(settings.roster_csv, settings.photos_root,
                                 class_name=settings.class_name)

    seat_map = None
    if settings.seat_map is not None:
        seat_map = seatmap_mod.load(settings.seat_map, roster,
                                    discovered_seat_ids=[s.seat_id for s in seats])

    faces: dict[int, Any] | None = None
    gallery = None
    face_summary: dict[str, Any] | None = None
    # Faces are pointless without a roster to match against AND a seat map to corroborate:
    # with no proposed name there is nothing for the evidence to agree or disagree with,
    # and running the model anyway would produce a "closest pupil" that nothing may use.
    if settings.face_corroboration and roster is not None and seat_map is not None:
        if video is None:
            # An assembled session has no single file to decode at these frame times, and
            # `session.analyse` refuses the combination before it gets here. This is the
            # second lock on the same door: reached only by a caller who built a `Source`
            # by hand, and it raises rather than quietly running the pass on part one and
            # labelling the result as the whole lesson's evidence.
            raise ValueError(
                "face corroboration needs one file to decode; this analysis has no single "
                "source video (an assembled session). See `session.analyse`.")
        from classvision.identity import faces as faces_mod

        gallery = faces_mod.build_gallery(roster, cache_dir=settings.cache_dir)
        boxes_at: dict[float, list[tuple[int, Any]]] = {}
        for position, seat_id in enumerate(seat_of_anchor):
            if seat_id is None:
                continue
            boxes_at.setdefault(anchors[position].video_seconds, []).append(
                (seat_id, detections.box[index_of[position]])
            )
        evidence = faces_mod.collect_from_video(
            video, frame_times=sorted(boxes_at), boxes_at=boxes_at,
            every_seconds=settings.face_every_seconds,
            time_tolerance=max(settings.sample_interval / 2.0, 0.05),
        )
        faces = evidence.aggregate(gallery)
        face_summary = evidence.summary()

    camera = settings.layout.camera if settings.layout else None
    bindings = assign_identity.bind_all(
        seats, seat_map=seat_map, roster=roster, recorded_on=recorded_on,
        camera=camera, adult_seat_id=adult_seat_id, excluded_seats=excluded_seats,
        faces=faces, gallery=gallery, standard=settings.identity_standard,
    )
    block = assign_identity.report(bindings, standard=settings.identity_standard,
                                   seat_map=seat_map, roster=roster, gallery=gallery,
                                   face_summary=face_summary)
    return bindings, block


def detect_pass(video: Path, settings: Settings, *, use_cache: bool = True) -> Detections:
    """Pass one: the only stage that needs the model.

    The cache key includes everything that could change a detection — the video's content
    hash, the weights, the input size, the confidence and the sampling rate. A cache keyed
    on the filename alone would serve last week's detections for this week's settings and
    nothing downstream could tell.
    """
    digest = art.sha256_of(video)[:16]
    key = (f"{digest}_{Path(settings.weights).stem}_{settings.imgsz}_"
           f"{settings.conf}_{settings.sample_fps}_"
           f"{settings.start_seconds:.0f}_{settings.end_seconds or 'end'}")
    path = settings.cache_dir / f"det_{key}.npz"
    if use_cache and path.exists():
        return Detections.load(path)

    from classvision.vision.pose import PoseModel

    model = PoseModel(weights=settings.weights, device=settings.device,
                      imgsz=settings.imgsz, conf=settings.conf)
    model.reset_tracker()

    times, xys, confs, boxes, tracks, frame_times = [], [], [], [], [], []
    started = time.time()
    for sample in decode.samples(video, settings.sample_fps,
                                 start_seconds=settings.start_seconds,
                                 end_seconds=settings.end_seconds):
        frame_times.append(sample.video_seconds)
        frame = model.look(sample.image, sample.index, sample.video_seconds)
        for person in frame.people:
            times.append(sample.video_seconds)
            xys.append(person.keypoints.xy)
            confs.append(person.keypoints.conf)
            boxes.append(person.box)
            tracks.append(-1 if person.track_id is None else person.track_id)

    detections = Detections(
        times=np.array(times, dtype=np.float32),
        xy=np.array(xys, dtype=np.float32) if xys else np.zeros((0, 17, 2), np.float32),
        conf=np.array(confs, dtype=np.float32) if confs else np.zeros((0, 17), np.float32),
        box=np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32),
        track=np.array(tracks, dtype=np.int32),
        frame_times=np.array(frame_times, dtype=np.float32),
        seconds_spent=time.time() - started,
    )
    detections.save(path)
    return detections


def lesson_window(detections: Detections, *, min_people: int = 2,
                  settle_seconds: float = 30.0) -> tuple[float, float, dict]:
    """When the room was actually occupied. Measured, never assumed to be the file.

    `min_people` rather than one, because a caretaker crossing an empty room should not
    open a lesson. `settle_seconds` requires the occupancy to persist, so a single frame
    in which the pose model hallucinated a second person in a chair does not either.
    """
    per_frame = {float(t): 0 for t in detections.frame_times}
    for t in detections.times:
        per_frame[float(t)] = per_frame.get(float(t), 0) + 1

    ordered = sorted(per_frame)
    occupied = [t for t in ordered if per_frame[t] >= min_people]
    if not occupied:
        return (float(ordered[0]) if ordered else 0.0,
                float(ordered[-1]) if ordered else 0.0,
                {"why": "the room was never occupied by more than one person"})

    start = occupied[0]
    for t in occupied:
        later = [u for u in occupied if t <= u <= t + settle_seconds]
        if len(later) >= max(2, int(settle_seconds / max(ordered[1] - ordered[0], 1e-6) * 0.5)):
            start = t
            break

    return float(start), float(occupied[-1]), {
        "file_seconds": float(ordered[-1] - ordered[0]) if ordered else 0.0,
        "occupied_seconds": float(occupied[-1] - start),
        "empty_frames_before_start": sum(1 for t in ordered if t < start),
        "min_people": min_people,
    }


def _seam_notes(source: Source, start: float, end: float) -> tuple[str, ...]:
    """One sentence per join, in the reader's units (minutes into the lesson window).

    Only the seams that fall INSIDE the analysed window are mentioned. A join in the empty
    room before the lesson opened is a fact about the recorder, not about the lesson, and
    an uncertainty note nobody needs is a note everybody stops reading.

    **This note is read by a psychologist, so it is written in Russian and not in machine
    output.** It reached the rendered report as «стык на 11.9-й минуте урока
    (2026-08-15T10:31:36.998000): перекрытие файлов, 1.00 с» — an English decimal point in
    a document where every other number is written «58,1 минуты», and an ISO timestamp with
    microseconds where the rest of the page says «10:19». Both are formatting, and both are
    the kind of formatting that tells a reader this sentence was not meant for them, in the
    one paragraph of the report whose whole job is to be believed.
    """
    block = source.session or {}
    notes: list[str] = []
    for seam in block.get("seams", []):
        at = float(seam["at_session_seconds"])
        if not (start <= at <= end):
            continue
        kind = {"gap": "пропуск записи", "overlap": "перекрытие файлов",
                "flush": "встык"}.get(seam["kind"], seam["kind"])
        trimmed = seam.get("trimmed_observations", 0)
        # `HH:MM:SS` from the seam's own wall time. The sub-second part is dropped on
        # purpose: the DVR prints whole seconds, so the microseconds are arithmetic, not
        # evidence -- and `MAX_SEAM_GAP_SECONDS` exists precisely because they are not
        # known to that precision.
        wall = str(seam.get("at_wall") or "")
        clock = wall[11:19] if len(wall) >= 19 else "—"
        notes.append(
            f"Запись склеена из нескольких файлов; стык на "
            f"{_ru_number((at - start) / 60.0, 1)}-й минуте урока ({clock}): {kind}, "
            f"{_ru_number(abs(float(seam['gap_seconds'])), 2)} с"
            + (f", отброшено {trimmed} повторных наблюдений." if trimmed else ".")
            + " Событие ровно на этой секунде стоит проверить по видео."
        )
    return tuple(notes)


def _ru_number(value: float, decimals: int) -> str:
    """A decimal comma, because the rest of the report has one.

    Deliberately local and three lines rather than an import from `report/summary.py`:
    `pipeline` writes the artefact and `report` renders it, and a dependency in that
    direction would let a rendering change alter what is measured.
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def analyse(video: str | Path, settings: Settings | None = None, *,
            use_cache: bool = True) -> art.Artefact:
    """One recording, one artefact. The whole thing.

    A lesson split across several DVR files is `session.analyse`, which does the same work
    on an assembled timeline; both meet at `assemble` below so that there is one definition
    of "a lesson analysed", not two.
    """
    video = Path(video)
    settings = settings or Settings()

    info = decode.probe(video)
    # `resolve`, not `from_video`, and the difference is a whole camera. `from_video` reads
    # the burnt-in overlay and raises when it cannot; on D14 the overlay sits over a window
    # with blinds and a poster, so the glyph matcher CORRECTLY refuses (margins 0.055/0.038
    # against a 0.07 floor) and every wall-clock label in the report came out `null`.
    # `resolve` is the ladder the clock module was written to expose: a human's declared
    # time, then the overlay, then the file name cross-checked against the overlay wherever
    # it IS legible, then `UNKNOWN`. It was built and verified and the pipeline was still
    # calling the layer underneath it.
    wall, _clock_evidence = clockmod.resolve(video)

    detections = detect_pass(video, settings, use_cache=use_cache)
    source = Source(
        path=str(video), sha256=art.sha256_of(video), size_bytes=video.stat().st_size,
        width=info.width, height=info.height, fps=info.fps,
        frame_count=info.frame_count, duration_seconds=info.duration_seconds,
        wall=wall, session=None, face_video=video,
    )
    return assemble(source, detections, settings)


def assemble(source: Source, detections: Detections,
             settings: Settings) -> art.Artefact:
    """Observations on a timeline -> the artefact. The part that does not care where the
    timeline came from.

    Split out of `analyse` when one lesson turned out to arrive as two files. The split is
    exactly at pass one: above it, files; below it, seconds since the lesson's first frame.
    Nothing below this line knows whether there was a seam, which is why seat ids, baselines
    and the lesson window are computed once for an assembled session rather than once per
    file — and it is why the merge could be added without re-deriving a single threshold.
    """
    video = source.face_video
    wall = source.wall
    start, end, window_notes = lesson_window(detections)

    # -- seats ---------------------------------------------------------------------
    anchors: list[Anchor] = []
    index_of: list[int] = []
    for i in range(len(detections.times)):
        t = float(detections.times[i])
        if not (start <= t <= end):
            continue
        person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
        scale = shoulder_width(person)
        position = anchor(person)
        if scale is None or position is None:
            continue
        anchors.append(Anchor(t, position, scale, int(detections.track[i])))
        index_of.append(i)

    analysed = [float(t) for t in detections.frame_times if start <= t <= end]

    # How many people the detector typically sees, so `discover` can check its own answer
    # against it rather than returning a confidently wrong seat count.
    per_frame: dict[float, int] = {t: 0 for t in analysed}
    for a in anchors:
        per_frame[a.video_seconds] = per_frame.get(a.video_seconds, 0) + 1
    counts = sorted(per_frame.values())
    expected_people = float(counts[len(counts) // 2]) if counts else None

    seats, seat_diagnostics = discover(anchors, analysed_frames=len(analysed),
                                       expected_people=expected_people)
    adult = identify_adult(seats, settings.layout)
    seat_of_anchor = assign(anchors, seats)

    # -- the adult, followed rather than seated -------------------------------------
    #
    # `identify_adult` answers "which discovered PLACE is the adult's", and on a camera
    # where the adult sits at one desk all lesson that is the whole answer. On D14 it is
    # not: he writes at the board, works at his own desk and sits down among the pupils
    # inside one lesson, so he has three places and none of them is his. Worse, at the
    # board he is at the far end of the room and his shoulder width there is smaller than
    # a front-row child's, so the scale route does not merely fail — it nominates a pupil.
    #
    # So the adult gets a single-target follower (`metrics/teacher.follow_adult`) over the
    # same anchor stream the seats were built from, seeded from the operator's
    # `teacher_zone` and falling back to whatever `identify_adult` nominated. Both facts
    # travel into the artefact: which route seeded it, and how much of the lesson it could
    # actually attribute. The seat route is kept, not replaced, because on the earlier
    # cameras it is the better answer and its ledger carries pose facts the follower has
    # no access to.
    # Fitted here rather than taken from `seat_diagnostics`, which carries only a summary
    # of the fit and not the model. Refitting costs a polyfit over eight row-band medians;
    # threading a model through the diagnostics dict would make a report block into an API.
    room_model = perspective.fit(
        np.array([a.position for a in anchors], dtype=float) if anchors
        else np.zeros((0, 2), float),
        np.array([a.scale for a in anchors], dtype=float) if anchors
        else np.zeros((0,), float),
    )

    adult_seat = next((s for s in seats if s.seat_id == adult.seat_id), None)
    track = teachermod.follow_adult(
        [teachermod.AdultObservation(a.video_seconds, a.position, a.scale, index_of[i],
                                     a.track_id)
         for i, a in enumerate(anchors)],
        analysed,
        layout=settings.layout,
        sample_interval=settings.sample_interval,
        perspective=room_model,
        fallback_centre=adult_seat.centre if adult_seat is not None else None,
    )
    teachermod.classify_track(track, settings.layout)

    # -- ledgers -------------------------------------------------------------------
    ledgers: dict[int, SeatLedger] = {
        seat.seat_id: SeatLedger(seat_id=seat.seat_id, thresholds=settings.thresholds,
                                 sample_interval=settings.sample_interval)
        for seat in seats
    }

    by_frame: dict[float, list[int]] = {}
    for position, seat_id in enumerate(seat_of_anchor):
        by_frame.setdefault(anchors[position].video_seconds, []).append(position)

    unassigned = 0
    board_zone = settings.layout.board_zone if settings.layout else None

    # The same stream the ledgers are fed, kept so that `metrics/within_lesson.py` can cut
    # the lesson into segments AFTER the baselines have settled on the whole of it. It is
    # deliberately the identical `Reading` objects and the identical board-zone verdicts:
    # anything re-derived here would make the segments a second analysis of the lesson
    # rather than a partition of the one this artefact reports, and the two could then
    # disagree about a second of a child's day with nothing to say which was right.
    # Memory: one `Reading` (slots) per seat-observation, ~35 000 on the D14 session.
    within_observations: dict[int, list[tuple[float, Any, bool]]] = {
        seat.seat_id: [] for seat in seats}
    within_absences: dict[int, list[float]] = {seat.seat_id: [] for seat in seats}

    for frame_time in analysed:
        present: set[int] = set()
        for position in by_frame.get(frame_time, []):
            seat_id = seat_of_anchor[position]
            if seat_id is None:
                unassigned += 1
                continue
            i = index_of[position]
            person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
            reading = read(person, frame_time, settings.thresholds)
            in_board = False
            if board_zone is not None:
                # The SHOULDER ANCHOR, not the bottom-centre of the box, and the change is
                # a measurement rather than a preference. On camera D14 the front row of
                # desks stands between the lens and the board and crops the legs off
                # exactly the people this test exists to find: the adult standing AT the
                # board has a box bottom at y ~ 430, and the same adult SEATED at his own
                # desk has a box bottom at y ~ 420. Four pixels apart in the quantity being
                # measured, a metre and a half apart in the room. The shoulder line
                # separates them by 145 px, for the reason `geometry.anchor` gives for
                # preferring it everywhere else: furniture crops boxes, so a box edge
                # measures the furniture. `room/zones.py` states this convention once, and
                # the teacher module uses the same one, so a single polygon has a single
                # meaning.
                from classvision.room.zones import point_in_polygon
                in_board = point_in_polygon(anchors[position].position, board_zone)
            ledgers[seat_id].observe(reading, in_board_zone=in_board)
            within_observations[seat_id].append((frame_time, reading, in_board))
            present.add(seat_id)
        for seat in seats:
            if seat.seat_id not in present:
                ledgers[seat.seat_id].note_absent(frame_time)
                within_absences[seat.seat_id].append(frame_time)

    for ledger in ledgers.values():
        ledger.finish()

    # -- identity ------------------------------------------------------------------
    # Deliberately AFTER the ledgers and BEFORE the records. After, because the standard
    # tests a seat's occupancy, which is a measured quantity and not a setting. Before,
    # because a name may only ever be attached to numbers that were computed without it:
    # nothing above this line knows who anybody is, so no metric can have been influenced
    # by a name. If identity is wrong, the names are wrong and the measurements are still
    # right -- which is the property that lets a school correct a seat map and re-run
    # without re-measuring anything.
    excluded = set(settings.layout.excluded_seats) if settings.layout else set()
    recorded_on = (wall.at(start).date()
                   if wall.source != clockmod.ClockSource.UNKNOWN else None)
    bindings, identity_block = _identity(
        video, settings, seats=seats, adult_seat_id=adult.seat_id,
        excluded_seats=excluded, anchors=anchors, index_of=index_of,
        seat_of_anchor=seat_of_anchor, detections=detections, recorded_on=recorded_on,
    )

    # -- records -------------------------------------------------------------------
    seat_records: list[art.SeatRecord] = []
    teacher_record: art.TeacherRecord | None = None

    # Which way the adult's head was turned in the observations the follower placed AT the
    # board -- the «пишет или объясняет» split. Gathered here because only the pipeline has
    # the keypoints; interpreted in `metrics/teacher.py`, which attaches the caveat that at
    # that distance his ears are a couple of pixels and this is an inference, not a reading.
    board_facing: Counter = Counter()
    for position_index, state in enumerate(track.state):
        if state is not teachermod.TeacherState.AT_BOARD:
            continue
        detection_index = track.index[position_index]
        if detection_index is None:
            continue
        direction = head_direction(Keypoints(xy=detections.xy[detection_index],
                                             conf=detections.conf[detection_index]))
        if direction is not HeadDirection.UNKNOWN:
            board_facing[direction.value] += 1

    teacher_kwargs = {
        "track": track,
        "sample_interval": settings.sample_interval,
        "layout": settings.layout,
        "perspective": room_model,
        "everyone": [teachermod.AdultObservation(a.video_seconds, a.position, a.scale, i,
                                             a.track_id)
                     for i, a in enumerate(anchors)],
        "head_directions": board_facing,
        "wall_clock": wall if wall.source != clockmod.ClockSource.UNKNOWN else None,
    }

    for seat in seats:
        ledger = ledgers[seat.seat_id]
        if seat.seat_id == adult.seat_id:
            adult_binding = bindings.get(seat.seat_id)
            teacher_record = art.TeacherRecord(
                seat_id=seat.seat_id,
                centre=seat.centre,
                # NOT rounded -- see the note on `SeatRecord.scale_px` below.
                scale_px=seat.scale,
                identification={"source": adult.source.value,
                                "needs_confirmation": adult.needs_confirmation,
                                "evidence": adult.evidence,
                                # Open-set rejection, stated rather than assumed: the
                                # adult appears in no roster row, so identity refuses this
                                # seat before consulting any evidence. Any face numbers
                                # here have their candidate ids redacted -- the "closest"
                                # pupil to a person who is not a pupil is a child's name
                                # attached to nothing.
                                "roster_membership": "взрослых нет в реестре учеников",
                                "identity": (adult_binding.to_pupil_field()
                                             if adult_binding else None)},
                ledger=ledger.summary(),
                metrics=teacher_metrics(ledger, **teacher_kwargs).to_dict(),
                # The POSITION timeline, not the pose one, because the question this half
                # of the report answers is «где он был». The pose episodes are still in
                # `ledger` above for anyone who wants them; putting them here would draw a
                # strip whose bands say «сидит» while the metrics beside it say «у доски».
                timeline=teachermod.timeline(track,
                                             sample_interval=settings.sample_interval),
                pose_timeline=ledger.timeline(),
            )
            continue
        role = "excluded" if seat.seat_id in excluded else "pupil"
        # ADDITIVE, and the schema stays `classvision/1.1` for exactly the reason
        # `provenance.session` did: nothing is renamed, nothing is removed, and a consumer
        # that ignores the key gets the document it got before, field for field. A version
        # bump exists to make an old reader FAIL where it would otherwise read a changed
        # meaning out of an unchanged name; there is no such name here.
        metrics: dict[str, Any] = {}
        if role == "pupil":
            metrics["activity"] = activity(ledger).to_dict()
            # Computed AFTER `ledger.finish()`, and that ordering is the whole design: the
            # segments share this place's settled baseline instead of each learning its
            # own, so a child who spent the last ten minutes slumped is measured against how
            # they sat at the start rather than against how they sat while slumping.
            metrics["within_lesson"] = withinmod.analyse_seat(
                seat.seat_id, within_observations[seat.seat_id],
                within_absences[seat.seat_id], window=(start, end),
                thresholds=settings.thresholds,
                sample_interval=settings.sample_interval,
                baselines=ledger.baselines,
            ).to_dict()
        seat_records.append(art.SeatRecord(
            seat_id=seat.seat_id, label=seat.label, centre=seat.centre,
            # NOT rounded, for the same reason `lesson.window_seconds` is not: this is a
            # REPRODUCIBILITY PARAMETER, not a display value. `room/seats.assign` gates an
            # observation into a place at `1.5 x max(anchor.scale, seat.scale)`, and the
            # verification overlay re-runs that gate from this very field. Rounded to
            # 0.1 px it moved the gate by up to 0.075 px, which on the D14 session flipped
            # exactly one borderline observation at t = 2 184,91 s and made
            # `overlay.agrees_with_artefact` report 1 mismatch of 325 (seat 4,
            # `away_from_place`, 325 against 324) -- the single check in this package that
            # can falsify the whole pipeline, failing because of a display rounding.
            # `centre` beside it was already full precision; the two are used together and
            # had no business being stored to different precisions. The report rounds it
            # when it prints it, and `identity/seatmap.write_template` already formats it
            # `:.0f` for the operator.
            scale_px=seat.scale, occupancy=round(seat.occupancy, 3),
            role=role,
            # Identity is a separate, explicit act -- see `identity/assign.py`. None here
            # means the layer did not run at all (no seat map configured), which is the
            # default and produces the anonymous «место N» report. When it DID run, this
            # is always a dict carrying `established: true|false` and the numeric reason
            # every gate passed or failed, because "examined and refused" and "never
            # examined" are different facts and `null` cannot tell them apart.
            pupil=(bindings[seat.seat_id].to_pupil_field()
                   if seat.seat_id in bindings else None),
            ledger=ledger.summary(),
            metrics=metrics,
            # The episode stream, so the report can draw WHEN rather than only HOW MUCH.
            # It is the same run-length encoding the totals above were summed from, so a
            # strip and a table on one page cannot disagree.
            timeline=ledger.timeline(),
        ))

    if teacher_record is None:
        # No seat belongs to the adult -- either `identify_adult` refused, or (the D14
        # case) he never stayed anywhere long enough to become one. The follower may still
        # have him, so the record is built from the track alone and the ledger stays empty
        # rather than being faked from someone else's seat. `metrics=` reports
        # `available: False` with a reason when the follower has nothing either, which is
        # a different sentence from "he was never there" and is written as one.
        teacher_record = art.TeacherRecord(
            seat_id=None,
            identification={"source": adult.source.value,
                            "needs_confirmation": adult.needs_confirmation,
                            "evidence": adult.evidence,
                            "follower": track.diagnostics},
            ledger={},
            metrics=teacher_metrics(None, **teacher_kwargs).to_dict(),
            timeline=teachermod.timeline(track,
                                         sample_interval=settings.sample_interval),
        )

    provenance = art.Provenance(
        schema=art.SCHEMA_VERSION,
        video_path=source.path, video_sha256=source.sha256,
        video_bytes=source.size_bytes,
        width=source.width, height=source.height, fps=source.fps,
        frame_count=source.frame_count,
        duration_seconds=round(source.duration_seconds, 2),
        started_at=None if wall.source == clockmod.ClockSource.UNKNOWN
        else wall.started_at.isoformat(),
        clock_source=wall.source.value, clock_drift_seconds=wall.drift_seconds,
        analysed_at=art.utc_now(),
        sample_fps=settings.sample_fps, sample_interval_seconds=settings.sample_interval,
        analysed_frames=len(analysed),
        window_start_seconds=start, window_end_seconds=end,
        model={"weights": settings.weights, "imgsz": settings.imgsz,
               "conf": settings.conf, "device": settings.device,
               "detect_seconds": round(detections.seconds_spent, 1)},
        thresholds=vars(settings.thresholds) if not hasattr(settings.thresholds, "__slots__")
        else {slot: getattr(settings.thresholds, slot)
              for slot in settings.thresholds.__slots__},
        room={"layout": None if settings.layout is None else {
            "camera": settings.layout.camera,
            "board_zone": settings.layout.board_zone,
            "teacher_zone": settings.layout.teacher_zone,
            # Recorded although nothing reads it YET, and that is the reason rather than an
            # objection to it: `run_id` hashes this dict, so a polygon that is asserted by
            # a human but absent here is a setting that can be changed WITHOUT producing a
            # new artefact. The day somebody wires the door zone into a metric, every
            # artefact made before that day would silently share a run_id with one made
            # after a redrawn door. Versioning it now costs one line; noticing later costs
            # a term of incomparable reports.
            "door_zone": settings.layout.door_zone,
            # Who drew or checked the polygons above, empty when nobody has. It is in
            # PROVENANCE and not only in the teacher block because a zone changes numbers
            # for pupils too (`AT_BOARD`), and because a re-run after a zone was redrawn
            # must be a different `run_id` -- which it now is, since `run_id` hashes this
            # whole `room` dict.
            "zones_confirmed_by": settings.layout.zones_confirmed_by,
            "excluded_seats": list(settings.layout.excluded_seats)},
            "perspective": room_model.to_dict(),
            "seat_discovery": seat_diagnostics,
            # The identity stage's own account of itself: the standard applied, and for
            # every seat that was NOT named, which gate stopped it and with what number.
            # It was computed and discarded until ruff noticed the variable was unused --
            # which would have left the artefact silently unable to answer the first
            # question a school asks, «почему у места 3 нет имени?».
            "identity": identity_block},
        software=art.software_versions(),
        # None on an ordinary run; on an assembled one, every constituent file and every
        # seam. A reader must be able to see that this artefact was put together, and
        # where -- see `session.py`.
        session=source.session,
    )

    uncertainty = art.Uncertainty(
        analysed_frames=len(analysed),
        frames_with_no_person=sum(1 for t in analysed if not by_frame.get(t)),
        observations_total=len(anchors),
        observations_unassigned=unassigned,
        observations_unreadable=sum(led.unreadable_observations
                                   for led in ledgers.values()),
        seats_never_settled=sum(1 for led in ledgers.values()
                                if not led.baselines.settled),
        rejected_clusters=seat_diagnostics.get("clusters_rejected", []),
        # Deliberately NOT repeating what `lesson["unmeasured"]` already says. Two lists
        # carrying the same sentence is two places for one of them to stop being true,
        # and the report rendered both, so the reader saw every limitation twice and
        # started skipping the block. `notes` is for run-specific remarks only.
        notes=[note for note in (
            # Two different failures with two different consequences, and they were one
            # sentence until D14 separated them. `adult.seat_id is None` means no PLACE was
            # attributed to the adult, so the pupil counters may include him -- that is the
            # dangerous one. `not track.found` means the FOLLOWER lost him too, so the
            # teacher half is empty. On D14 the first is true and the second is false: he
            # has no seat because he never sits in one place, and the pupil counters are
            # nonetheless clean because the follower attributes his observations. Reporting
            # only the first would have said the pupil numbers were contaminated when they
            # are not.
            (f"Взрослому не сопоставлено постоянное место ({adult.evidence.get('why', '')}); "
             # Comma decimal separator, like every other number in the Russian report. This
             # one arrived straight from `round()` and printed «40.7 %» three lines below
             # «48,5 %».
             + ("но слежение за взрослым отнесло ему "
                f"{str(track.diagnostics.get('attributed_share_of_lesson_percent')).replace('.', ',')}"
                " % кадров, "
                "поэтому его наблюдения не попадают в счётчики учеников."
                if track.found else
                "и слежение за взрослым тоже не нашло его: счётчики учеников могут "
                "включать взрослого.")
             if adult.seat_id is None else ""),
            # Where the record was joined. Not a caveat about the method -- the seams were
            # measured and are inside the tolerance -- but a run-specific fact a reader
            # needs before deciding whether an event at exactly that second is real.
            *_seam_notes(source, start, end),
        ) if note],
    )

    wall_start = wall.at(start) if wall.source != clockmod.ClockSource.UNKNOWN else None
    lesson = {
        # NOT rounded. These two numbers are the window the run actually used, and
        # anything that replays the analysis from the artefact -- the verification
        # overlay, a re-import, a future re-scoring -- must select exactly the same
        # frames. Rounding to 0.1 s dropped the final frame, and every seat's state
        # histogram came out one observation short of the report it was checking.
        # A window is a reproducibility parameter, not a display value; the report
        # rounds it when it prints it.
        "window_seconds": [start, end],
        "window_wall": [wall_start.isoformat() if wall_start else None,
                        wall.at(end).isoformat()
                        if wall.source != clockmod.ClockSource.UNKNOWN else None],
        "duration_minutes": round((end - start) / 60.0, 1),
        "detection": window_notes,
        "pupil_seats": len(seat_records),
        # How many places in a room this size would show a within-lesson direction from
        # noise alone. It travels with the artefact rather than being computed by whoever
        # renders it, for the same reason `caveats` does: a list of places whose index moved
        # is read as a list of children unless this number is printed next to it, and a
        # surface that has to derive it is a surface that can forget to.
        "within_lesson_direction_by_chance": withinmod.expected_by_chance(
            len(seat_records)),
        "adult_seat": adult.seat_id,
        "unmeasured": [
            {"what": "ответил вслух", "why": "в файле нет аудиодорожки"},
            {"what": "стоял у доски",
             "why": "зона доски не задана для этой камеры; события попадают в «вне места»"}
            if board_zone is None else None,
            {"what": "«за своим столом» для взрослого",
             "why": "зона учительского стола не размечена; взрослого приходится искать по "
                    "размеру, а на этой камере он бывает дальше учеников от объектива"}
            if (settings.layout is None or settings.layout.teacher_zone is None) else None,
            # A zone nobody signed is a guess with a polygon, and the artefact says so
            # rather than letting a drawn boundary read as a checked one.
            {"what": "подтверждение разметки зон человеком",
             "why": "зоны заданы, но никто не указан в `zones_confirmed_by`: границы доски "
                    "и стола — предположение оператора, а не проверенный факт"}
            if (settings.layout is not None and not settings.layout.zones_confirmed_by
                and (settings.layout.board_zone or settings.layout.teacher_zone)) else None,
        ],
    }
    lesson["unmeasured"] = [u for u in lesson["unmeasured"] if u]

    return art.Artefact(provenance=provenance, seats=seat_records,
                        teacher=teacher_record, uncertainty=uncertainty, lesson=lesson)

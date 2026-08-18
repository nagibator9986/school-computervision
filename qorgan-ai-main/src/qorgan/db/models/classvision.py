"""Imported offline classroom analyses: accumulated per PLACE, named only by attestation.

**Why these are new tables and not columns on `lessons` / `lesson_tracks`.** Those two record
what the LIVE worker saw through a socket and key on a ByteTrack id; these record what an
OFFLINE analyser computed from a file and key on a seat. They differ in the one property
everything longitudinal depends on: a track dies at the first long occlusion and is reborn
under a new number, while a place is still the same place next Tuesday. Folding the second
into the first would put two definitions of "the thing a count belongs to" in one table.

**Should `LessonTrack` gain a `person_id`? No — and not as a compromise.** The refusal in
`db/models/classroom.py` reads as squeamishness about identification; the stronger reason is
that the column could not carry what it would be added for. A `LessonTrack` row does not
survive one lesson, so a `person_id` on it could never express "this child, four weeks running"
— while looking exactly as though it could. The stable object is the place, which the old
schema has no concept of. `classroom.py`'s promise stays literally true: 0010 touches neither.

**A place is not a child, and there is one route from a place to a name.**
`ClassvisionAttestation` — a dated statement by a named human, `decision_ref` NOT NULL. No
face, no embedding, no inference may fill `ClassvisionPlaceLesson.person_id`, and
`identity_method` records the route so a later reader can check. The measurement behind that
ban is this school's own: 141 roster photos against this room's faces give a median best cosine
of 0.30 with a 0.10 margin (`classvision/MEASUREMENTS.md` §4) — a preference, not a recognition.

**`is_demo` is NOT NULL on the three tables that originate a row** — lessons, runs, places —
and everything else reaches one of them through a key that cannot be NULL. A fabricated
number that looks measured is the one unacceptable outcome of a demonstration, so no
aggregate may add a demo lesson to a real one without grouping on this column and saying so.

**Coverage travels with every count, in the same row, NOT NULL**, and `activity_index` is
nullable while `activity_parts` is not: an index may be refused (coverage below
`classvision.metrics.activity.MIN_COVERAGE`), NULL then means «не измерялось» and never zero,
and a schema in which an index can exist without its components is one in which a page can
render a bare number — which `classvision/DESIGN.md` forbids.

**Tenancy.** `classvision_lessons` and `classvision_places` are ROOT tables as
`tests/test_tenancy_guard.py` defines them — nothing else can answer for them, because
`camera_key` is an operator's assertion about which room a FILE came from, not a foreign key
into `cameras` (a demo room, and a camera analysed before it was enrolled, both have to be
storable). Everything else is derived and reaches its school through a NOT NULL key. No second
answer to one question.

**The prefix is `classvision_` on all eight tables**, unlike the draft in
`classvision/INTEGRATION.md` §2 (`class_analyses`, `seat_attestations`): one prefix means
`sqlite_master` shows this module's whole footprint, and the draft's names read as though
they belonged to two unrelated features.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (JSON, Boolean, Date, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.models.school import school_key
from qorgan.db.types import RelPath, UtcDateTime


class ClassvisionLesson(Base, TimestampMixin):
    """One analysed hour in one room, as the unit a week is built from.

    Separate from the RUN because a lesson may be analysed more than once (a new threshold is a
    new run) and recorded in more than one file (a DVR cut this school's 62-minute lesson into
    13.6 and 48.4 minutes). `selected_run_id` is the run the pages show; moving it is a named
    act, never a side effect of importing. `started_at` is nullable and `iso_week` with it: a
    recording whose clock could not be read is still stored, and is then excluded from every
    weekly figure rather than filed under a guessed Monday.
    """

    __tablename__ = "classvision_lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = school_key(nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # An operator's assertion, and the column beside it says so. The artefact analyses a FILE and
    # has no idea which room the camera hangs in; `clip_15min.mp4` and `test_camera.mp4` are one
    # camera and would otherwise become two rooms that never accumulate together.
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_key_source: Mapped[str] = mapped_column(String(24), nullable=False)
    class_key: Mapped[str] = mapped_column(String(64), nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    # The school's local day and ISO week, computed once at import from `started_at` and the zone
    # it was read in. Re-deriving them under a later SCHOOL_TIMEZONE would silently move last
    # term's lesson across a day boundary, so they are stored.
    date_local: Mapped[date | None] = mapped_column(Date, index=True)
    iso_year: Mapped[int | None] = mapped_column(Integer)
    iso_week: Mapped[int | None] = mapped_column(Integer)
    timezone: Mapped[str | None] = mapped_column(String(64))

    duration_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    selected_run_id: Mapped[str] = mapped_column(String(32), nullable=False)

    # A DVR seam: two recordings, one lesson. Two lessons is the correct STORAGE and one session
    # the correct unit of ACCUMULATION; only a column can tell a weekly aggregate which it has.
    continues_lesson_id: Mapped[int | None] = mapped_column(
        ForeignKey("classvision_lessons.id", ondelete="SET NULL")
    )
    part_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # An operator forced an import that overlaps an hour already stored. Kept for ever and printed
    # on the report: «один и тот же час, посчитанный дважды во всех недельных итогах».
    overlap_allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    overlap_note: Mapped[str | None] = mapped_column(Text)

    runs = relationship("ClassvisionRun", back_populates="lesson",
                        cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_classvision_lessons_room_date", "school_id", "camera_key", "class_key",
              "date_local"),
        Index("ix_classvision_lessons_demo", "school_id", "is_demo"),
    )


class ClassvisionRun(Base, TimestampMixin):
    """One imported artefact: its `run_id`, and everything a number was computed under.

    `run_id` is the artefact's own hash of the video content plus every setting that could change
    a figure, so re-import is idempotent for free and a re-analysis under different thresholds is
    a NEW row rather than an overwrite. A term's trend with a step change nobody can explain is
    the failure that prevents.

    `provenance`, `uncertainty`, `caveats` and `unmeasured` are stored verbatim: they are shown
    beside the numbers and audited, not queried, and a field added upstream must not be silently
    dropped on the way in.
    """

    __tablename__ = "classvision_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_lessons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalised from the lesson ON PURPOSE, and the only such column here: this row is the
    # provenance object somebody will read on its own, and a run that does not say it is a
    # demonstration is the one row that could be quoted as a measurement.
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    # On an assembled session this is the digest of the ASSEMBLY, not of a file (see `session`).
    video_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    video_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    # NOT NULL on the source, nullable on the value: «overlay» / «filename» / «unknown» are
    # three qualities of evidence about when this hour happened.
    clock_source: Mapped[str] = mapped_column(String(16), nullable=False)
    clock_drift_seconds: Mapped[float | None] = mapped_column(Float)

    sample_fps: Mapped[float] = mapped_column(Float, nullable=False)
    analysed_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # A short hash over `provenance.thresholds`: "same rules as last term?" without two blobs.
    thresholds_sha: Mapped[str] = mapped_column(String(16), nullable=False)
    model_weights: Mapped[str] = mapped_column(String(64), nullable=False)
    model_imgsz: Mapped[int | None] = mapped_column(Integer)
    model_device: Mapped[str | None] = mapped_column(String(16))
    room_layout: Mapped[dict] = mapped_column(JSON, nullable=False)
    session: Mapped[dict | None] = mapped_column(JSON)

    pupil_places: Mapped[int] = mapped_column(Integer, nullable=False)
    adult_seat_id: Mapped[int | None] = mapped_column(Integer)

    # -- what the run could not see. Never render a total without these.
    observations_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    observations_unassigned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    observations_unreadable: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    frames_with_no_person: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    seats_never_settled: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    provenance: Mapped[dict] = mapped_column(JSON, nullable=False)
    uncertainty: Mapped[dict] = mapped_column(JSON, nullable=False)
    caveats: Mapped[list] = mapped_column(JSON, nullable=False)
    unmeasured: Mapped[list] = mapped_column(JSON, nullable=False)

    analysed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    imported_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    lesson = relationship("ClassvisionLesson", back_populates="runs")
    places = relationship("ClassvisionPlaceLesson", back_populates="run",
                          cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("run_id", name="uq_classvision_runs_run_id"),
        Index("ix_classvision_runs_started", "started_at"),
    )


class ClassvisionPlace(Base, TimestampMixin):
    """A fixed position in a fixed room used by a fixed class. The unit of accumulation.

    A `seat_id` belongs to one run — `room/seats.py` numbers seats in reading order, so one absent
    pupil shifts every id behind the gap and a term of «место 3» becomes two or three children
    welded together. Everything longitudinal joins THIS table instead.

    The geometry is ANCHORED on the run that first revealed the place and never drifts. A running
    mean would let a slowly-moving camera carry a place across a desk over a term with nothing
    failing; a fixed anchor makes a moved camera show up as one dated run in which every seat is
    unmatched — a fact a human can act on.

    `class_key` is in the key beside `camera_key` because two classes sharing a room have different
    occupants, and merging their histories is the seat-swap defect at the scale of a timetable.
    """

    __tablename__ = "classvision_places"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = school_key(nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)
    class_key: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    label_ru: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # pupil | adult

    anchor_x: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_y: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_scale: Mapped[float] = mapped_column(Float, nullable=False)  # shoulder width, px
    first_run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    first_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        UniqueConstraint("school_id", "camera_key", "class_key", "ordinal",
                         name="uq_classvision_places_room_ordinal"),
    )


class ClassvisionPlaceLesson(Base, TimestampMixin):
    """One place, in one lesson, as one run measured it. The row every page reads.

    `place_id` is nullable and the NULL is a complete record: the seat could not be matched to
    a known place in this room, so the row is stored in full and excluded from every
    accumulation rather than joined to the nearest history. `place_match` says which unmatched
    case it was — new, moved, ambiguous — because each needs a different action from a human.

    `person_id` is filled ONLY from an attestation in force on the recording date. NULL is the
    ordinary case and the report stays useful: it says «место 3».
    """

    __tablename__ = "classvision_place_lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_lessons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    place_id: Mapped[int | None] = mapped_column(
        ForeignKey("classvision_places.id", ondelete="SET NULL"), index=True
    )

    # The artefact's own per-run seat number, kept for tracing back to the JSON and
    # deliberately not the key anything longitudinal joins on.
    seat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    seat_label: Mapped[str] = mapped_column(String(32), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)

    place_match: Mapped[str] = mapped_column(String(16), nullable=False)
    place_match_reason: Mapped[str | None] = mapped_column(Text)
    place_match_distance: Mapped[float | None] = mapped_column(Float)  # shoulder widths

    # THIS RUN's geometry, which is not the place's anchor: the anchor is frozen on the run that
    # discovered the place, and the difference between the two is how a moved camera shows up.
    # It is also what `classvision.frames` draws a rectangle from.
    centre_x: Mapped[float] = mapped_column(Float, nullable=False)
    centre_y: Mapped[float] = mapped_column(Float, nullable=False)
    scale_px: Mapped[float] = mapped_column(Float, nullable=False)  # shoulder width, px

    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )
    identity_method: Mapped[str] = mapped_column(String(32), nullable=False)
    identity_reason: Mapped[str | None] = mapped_column(Text)

    # Coverage FIRST, because no count below it may be read without it.
    coverage: Mapped[float] = mapped_column(Float, nullable=False)
    observations: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    # False means `stands` and the away counts are UNMEASURED, not zero: with no settled
    # baseline there is nothing to compare a posture against (migration 0005).
    settled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    settle_refusal: Mapped[str | None] = mapped_column(Text)
    absent_observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unreadable_observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hand_unmeasurable_observations: Mapped[int] = mapped_column(Integer, default=0,
                                                                nullable=False)

    hand_raises: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stands: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    away_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    board_visits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    head_down_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    turned_away_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # NULL when the index was refused. NULL is not zero, and `activity_reason` is what the
    # page prints instead of a number.
    activity_index: Mapped[float | None] = mapped_column(Float)
    activity_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    activity_parts: Mapped[list] = mapped_column(JSON, nullable=False)
    # Per-segment index, coverage and the ordering verdict (Kendall S/p, the floors,
    # `visibility_bound_index_points`) — whole, because a direction shown without the floor it
    # had to clear is a claim the artefact refused to make.
    within_lesson: Mapped[dict] = mapped_column(JSON, nullable=False)
    ledger: Mapped[dict] = mapped_column(JSON, nullable=False)
    # State runs, each with its own `measured` flag, so a frame at second N can be labelled
    # with what this place was doing — and «не наблюдалось» told apart from «сидит».
    timeline: Mapped[list] = mapped_column(JSON, nullable=False)

    run = relationship("ClassvisionRun", back_populates="places")

    __table_args__ = (
        UniqueConstraint("run_id", "seat_id", name="uq_classvision_place_lessons_run_seat"),
        Index("ix_classvision_place_lessons_place", "place_id", "lesson_id"),
    )


class ClassvisionTeacherLesson(Base, TimestampMixin):
    """The adult in one lesson: position over time, and nothing that ranks a teacher.

    `attributed_share_of_lesson_percent` is filled whenever a `presence` block exists, and the
    importer refuses the block without it: on camera D14 the follower attributes 45 % of the
    lesson to the adult, and «у доски — 3 %» without «опознан в 45 % кадров» has told the reader
    something false with a true number. `board_zone_configured` is what tells a page whether a
    board figure is a measurement at all: with no polygon the state cannot arise in any frame, so
    every board number is NULL and renders «не измерялось» — never «0,0 мин».

    No trend and no week-over-week comparison for this row, deliberately: the adult's numbers move
    with how much of him the follower could hold, so a falling line would read as a teacher
    getting worse when it means the camera saw less of him.
    """

    __tablename__ = "classvision_teacher_lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_lessons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_runs.id", ondelete="CASCADE"), nullable=False
    )
    # NULL with a reason, never an omitted row: «мы не знаем, где он был» and «его не было»
    # are different facts, so an adult the artefact could not place is still listed.
    place_id: Mapped[int | None] = mapped_column(
        ForeignKey("classvision_places.id", ondelete="SET NULL")
    )
    place_missing_reason: Mapped[str | None] = mapped_column(Text)
    seat_id: Mapped[int | None] = mapped_column(Integer)

    attributed_share_of_lesson_percent: Mapped[float | None] = mapped_column(Float)
    # The SEAT's occupancy, named so it cannot be confused with the follower's share above.
    pose_coverage: Mapped[float | None] = mapped_column(Float)
    board_zone_configured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    board_minutes_of_lesson: Mapped[float | None] = mapped_column(Float)
    board_share_of_lesson_percent: Mapped[float | None] = mapped_column(Float)
    board_occupancy_available: Mapped[bool] = mapped_column(Boolean, default=False,
                                                            nullable=False)
    # Position-episode boundaries EXCLUDING those where the follower merely started or stopped
    # seeing him. The other count lives in `presence` and must never be shown as «смен места».
    transitions_excluding_out_of_frame: Mapped[int | None] = mapped_column(Integer)
    # A POSE count from the seat ledger, NULL without one — not the line above (§6b.1).
    pose_transitions: Mapped[int | None] = mapped_column(Integer)

    presence: Mapped[dict | None] = mapped_column(JSON)
    board: Mapped[dict] = mapped_column(JSON, nullable=False)
    board_occupancy: Mapped[dict] = mapped_column(JSON, nullable=False)
    pose_metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    not_an_assessment_ru: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (UniqueConstraint("run_id", name="uq_classvision_teacher_lessons_run"),)


class ClassvisionFrame(Base, TimestampMixin):
    """One rendered frame of the recording, with the boxes and labels that go over it.

    Rows rather than a video because of the seam in `INTEGRATION.md` §1: the web process must
    never gain torch, ultralytics or cv2. The images are pre-rendered on disk under MEDIA_ROOT;
    this table says which second each is, what was labelled on it, and where the labels came
    from.

    `box_source` is the honest half. `place_geometry` means the rectangles are the PLACE's own
    anchor and shoulder width — the region a count was accumulated in — not the detector's
    output for that frame, which no artefact carries. `detector` means a classvision-side
    render supplied them. `caveat_ru` says so in a sentence, because a rectangle drawn on a
    child reads as a detection whatever a column says.
    """

    __tablename__ = "classvision_frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_lessons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    video_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    wall_clock: Mapped[datetime | None] = mapped_column(UtcDateTime)
    # Relative to MEDIA_ROOT. `RelPath` refuses an absolute path at bind time: the legacy
    # stored absolute ones and every project move broke 100 % of the rows (rule R6).
    image_path: Mapped[str] = mapped_column(RelPath(255), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    source_video: Mapped[str | None] = mapped_column(Text)

    box_source: Mapped[str] = mapped_column(String(24), nullable=False)
    boxes: Mapped[list] = mapped_column(JSON, nullable=False)
    caveat_ru: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "video_seconds", name="uq_classvision_frames_run_second"),)


class ClassvisionReading(Base, TimestampMixin):
    """A language model's ORIENTATION note about one run, and the guard's verdict on it.

    Keyed to the run, so a re-analysis cannot inherit the previous reading: the numbers moved,
    and a paragraph about the old ones would be a confident sentence about a measurement that
    no longer exists.

    `guard_passed` is NOT NULL and is the point of the row. The note may contain no digit that
    is not a place number (`classvision/report/orientation.py::check`): the one failure measured
    in this system was a model restating a quantity beside the wrong claim. A failed guard is
    stored rather than discarded — the page shows the deterministic figures, and the audit shows
    why the prose was withheld.
    """

    __tablename__ = "classvision_readings"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # lesson | place | class_term, and which one. Two granularities are two rows, never one
    # overwritten.
    section: Mapped[str] = mapped_column(String(32), nullable=False)
    target_key: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    source: Mapped[str] = mapped_column(String(16), nullable=False)  # gemini | none
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)

    body: Mapped[str] = mapped_column(Text, nullable=False)
    # The structured half when the model was asked for one. Shown as structure, never re-parsed.
    payload: Mapped[dict | None] = mapped_column(JSON)

    guard_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    guard_offending: Mapped[list] = mapped_column(JSON, nullable=False)
    guard_reason_ru: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "section", "target_key",
                         name="uq_classvision_readings_run_section"),
    )


class ClassvisionAttestation(Base, TimestampMixin):
    """A human's dated statement that a PLACE holds a named child. The only route to a name.

    Its own table, with `attested_by` and a validity window, because it is EVIDENCE about a
    period, not a property of a lesson: when a class is re-seated the old rows stay true about
    the old term and a new row starts.

    `place_id`, never `(camera, seat_id)`: a seat number is re-assigned by reading order on
    every run, so an attestation keyed on it points at a different desk the first time a pupil
    is absent.

    `decision_ref` is NOT NULL and is not decoration. Per-pupil accumulation is what
    `docs/questions-for-school.md` §8 told the school would not happen and §10.1 then asked the
    school to decide in writing; it is also personal data under ЗРК 94-V. An attestation nobody
    can trace to that answer is the quiet re-opening of a written promise.
    """

    __tablename__ = "classvision_attestations"

    id: Mapped[int] = mapped_column(primary_key=True)
    place_id: Mapped[int] = mapped_column(
        ForeignKey("classvision_places.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)
    attested_by: Mapped[str] = mapped_column(String(200), nullable=False)
    attested_at: Mapped[date] = mapped_column(Date, nullable=False)
    decision_ref: Mapped[str] = mapped_column(String(200), nullable=False)

    __table_args__ = (Index("ix_classvision_attestations_lookup", "place_id", "valid_from"),)

# Integrating `classvision` into `qorgan-ai-main`

This document is the contract. It is written to be read by whoever does the merge, and by
whoever has to explain the merge to the school.

---

## 0. The thing to settle before any code is written

`qorgan-ai-main` already contains a classroom package, and that package contains a written
promise. Four of them, in fact, stated in `src/qorgan/classroom/__init__.py` and repeated
in `HANDOVER.md` §7 and `src/qorgan/psychologist/__init__.py`:

1. **No identification inside a classroom.** `LessonTrack` has no `person_id` column and
   its docstring says one must never be added.
2. **No conclusion about emotion, attention or engagement.**
3. **No judgement about a teacher's work** — client §12.5 is deliberately not built.
4. **No «лежит на парте» metric** — §12.4, not promised, not computed.

`classvision` does all four. So the honest position is not that the old verdict was wrong;
it is that **one of its two supports has moved and the other has not.**

### What the new measurements change

The stated evidence for refusing classroom identity is a corridor measurement: 14 970
faces, median **11.5 px**, **zero** recognitions. That measurement is real and it is about
a corridor. On the classroom recording we now hold:

| | corridor (old) | this classroom (measured) |
|---|---|---|
| face height, median | 11.5 px | **64 px** (p25 53.5, p75 73.7) |
| how long the same body is available | seconds | **~46 minutes** |
| position of the same person | anywhere | **a fixed desk** |

A classroom genuinely is not a corridor, and the sentence "a face across a classroom is
the same size" — written before anyone had a classroom recording — is not true of this
camera.

### What the new measurements do NOT change

**Face recognition still cannot create a name here, and that is measured, not assumed.**
All 141 roster photos embed successfully. Matching the room's faces against that gallery
gives a median best cosine of **0.30** with a margin of **0.10** over the runner-up
(`MEASUREMENTS.md` §4). ArcFace wants ~0.4–0.5 to call two faces the same person. That is
a preference, not a recognition, and a name attached to it would be a guess wearing a
number.

### Therefore

`classvision` does not identify children by face. It accumulates per **seat**, and a seat
acquires a name **only** from a dated, human-attested seating plan (`identity/seatmap.py`).
Face evidence may corroborate or contradict that plan; it can never originate a name.
There is no code path from an embedding to a `full_name`.

This means the promise is **not circumvented by a technical trick** — it is
*re-opened for the school to decide*, on new evidence, with a mechanism that does not
depend on the thing the school objected to. That decision is the school's, not the
engineer's. The wording to put to them:

> Раньше мы отвечали, что помесячная динамика по конкретному ребёнку в классе невозможна,
> потому что опознать ребёнка в классе нельзя. Появилась запись урока, и она уточняет
> ответ. Опознавать по лицу мы по-прежнему не будем — мы это измерили, и точности не
> хватает (совпадение 0,30 при разнице 0,10 со следующим кандидатом). Но в классе, в
> отличие от коридора, ребёнок сидит на закреплённом месте. Мы предлагаем вести учёт **по
> месту**, а имя брать **только из плана рассадки, который подписывает классный
> руководитель** и который действует на конкретный период. Система имён не угадывает: если
> плана нет, в отчёте будет «место 3», и отчёт остаётся полезным.
>
> Вопрос к школе: согласны ли вы, чтобы к местам привязывались имена из подписанного плана
> рассадки? Это решение о персональных данных детей (ЗРК «О персональных данных и их
> защите» 94-V), и принимать его должна школа, а не мы. Если ответ «нет» — модуль работает
> без имён.

Until that answer exists **in writing**, run with no seat map. Everything works; the
cabinet shows places.

### Should `LessonTrack` gain a `person_id`?

**No — and not as a compromise, but because it would be the wrong column.** The argument
is not squeamishness about the docstring. A `LessonTrack` row is a *ByteTrack id*, which
dies at the first long occlusion and is reborn with a new number. Attaching a person to it
would attach a person to an object that does not survive one lesson, let alone four weeks —
so the column could not carry the trend it was added for, while looking exactly as though
it could. The stable object is the **seat**, and the seat is a new concept that the old
schema does not have. That is why §2 adds tables rather than a column.

---

## 1. The seam

```
  classvision  ──writes──▶  <lesson>.analysis.json  ──reads──▶  qorgan
   (torch, ultralytics,        schema: classvision/1.1        (fastapi, sqlalchemy)
    opencv, insightface)                                       NO torch, ever
```

* `classvision` **never imports** `qorgan`. Verified by grep in CI.
* `qorgan` **never imports** `torch`, `ultralytics`, `cv2`, `insightface`. Verified by an
  import-guard test (§8).
* The only shared thing is the JSON document, whose shape lives in
  `classvision/src/classvision/report/artefact.py` and whose version string is
  `SCHEMA_VERSION = "classvision/1.1"`. **1.1 is not backward compatible with 1.0, deliberately.** The per-state `seconds` key was
split into `observed_seconds_by_state` (sums to `observed_seconds`) and `episode_seconds` (a
subset), and the adult's shares gained their denominator (`at_desk_share_of_observed`,
`out_of_frame_share_of_lesson`). The old names were **removed rather than aliased**, so an
importer written against 1.0 raises `KeyError` instead of reading a quantity that means
something else — which is what happened when both were called `seconds` (`MEASUREMENTS.md` §8).
The importer must therefore reject any `schema_version` it does not know, not coerce it.

A copy of one real artefact is committed to
  `qorgan` as a **test fixture**, so the importer's tests do not need the analyser.

Why a file rather than a shared library or an HTTP call: the analysis runs on a laptop or a
GPU box for tens of minutes, and the web process serves a school over a LAN. Coupling them
means the dashboard inherits a 2 GB dependency tree, an AGPL obligation (see §9) and a
failure mode where a slow model run blocks a page load.

**One artefact may describe several FILES: `provenance.session`.** A DVR splits a lesson
wherever it likes — the D14 camera cut this school's 62-minute lesson into 13.6 and 48.4
minutes — and such a lesson is analysed as one timeline (`classvision analyse-session A B`,
`session.py`) because seat ids and posture baselines must be established once for the whole
hour. The document is otherwise unchanged and remains `classvision/1.1`: nothing was renamed
or removed, and `provenance.session` is `null` on every single-file run. When it is **not**
null, three existing fields mean something a reader must not assume:

| field | on an assembled artefact |
|---|---|
| `provenance.video_sha256` | the digest of the ASSEMBLY (every part's hash and where it was placed), not of a file. Still unique, still the thing `run_id` is built on, so the importer stays idempotent and re-joining the same files under a different tolerance is a different run rather than an overwrite. |
| `provenance.video_path` | the FIRST part only. Every part's path, hash, start and duration is in `provenance.session.parts[]`. |
| `provenance.duration_seconds`, `frame_count` | the span and the frame total of the joined timeline, not of one file. |

`provenance.session.seams[]` records where the joins are — `at_session_seconds`, the wall
time, the measured `gap_seconds` (negative means the files overlapped and the duplicate
observations were trimmed, and how many) — and `uncertainty.notes` carries the same seam in
a sentence for any surface that shows the lesson to a person. **A merged artefact that
looked like a single recording would be the most convincing wrong number this package could
produce**, so no consumer has to infer the assembly: it is a field.

The importer needs no change to accept these, and should not gain one that treats them
specially: an assembled lesson is a lesson.

**What happens if a school imports BOTH the session artefact and the per-file ones.** It is
the obvious operational accident — a cron that runs `analyse` over a directory and an
operator who then runs `analyse-session` on the same pair — and it is already refused,
because the session's wall-clock span contains each part's. Verified in both import orders
on the D14 pair:

```
parts first:      REVIEW_D14_20260815101759   -> OK lesson=1
                  REVIEW_D14_20260815103136   -> OK lesson=2
                  REVIEW_session_room         -> REFUSED [overlapping_lesson]
session first:    REVIEW_session_room         -> OK lesson=1
                  REVIEW_D14_20260815101759   -> REFUSED [overlapping_lesson]
                  REVIEW_D14_20260815103136   -> REFUSED [overlapping_lesson]
```

The refusal says «Два пересекающихся файла — это один и тот же час, посчитанный дважды во
всех недельных итогах», and `--allow-overlap` is the deliberate override, which records the
note in the row and prints it in the report. So the choice between "two lessons" and "one
session" is made once, by a person, and cannot be made twice by accident.

**Where `run_id` comes from and why it matters.** It is a hash of the video's content hash
plus every setting that could change a number — model weights, `imgsz`, sampling rate, all
thresholds, the room layout. Re-running the same lesson with a different threshold produces
a **different `run_id`**, so the importer is idempotent for free *and* a school cannot
silently overwrite last month's measurement with one taken under different rules. A term's
trend that contains a step change nobody can explain is the specific failure this prevents.

---

## 2. New tables

> **Amended after the accumulation layer was actually built.** The four tables below were
> drafted before anything accumulated more than one lesson. Building
> `classvision/src/classvision/cabinet/store.py` — a working store over the real artefacts —
> found one defect in this draft that would have been invisible in review and undetectable
> in production. It is written up in **§2a**, and the `ClassPlace` table and the corrected
> `SeatAttestation` below are the fix. Read §2a before implementing this section.

Four, in `qorgan`'s existing style (`Base`, `TimestampMixin`, `UtcDateTime`, school
scoping, explicit index names). A new file `src/qorgan/db/models/classvision.py`.

```python
"""Imported classroom analyses: per SEAT, and named only by human attestation.

These tables are separate from `lessons`/`lesson_tracks` on purpose. Those record what a
LIVE worker saw and key on a ByteTrack id; these record what an OFFLINE analyser computed
and key on a seat. A seat outlives an occlusion and a track does not, which is the whole
reason a per-child trend is possible here and was not there.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean, Date, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qorgan.db.models.base import Base, TimestampMixin
from qorgan.db.types import UtcDateTime


class ClassAnalysis(Base, TimestampMixin):
    """One imported artefact. `run_id` is the artefact's own; re-import is a no-op."""

    __tablename__ = "class_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    camera: Mapped[str | None] = mapped_column(String(64))
    video_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    # Wall clock, and where it came from. NOT NULL on the source, nullable on the value:
    # a run whose clock could not be read still imports, and every weekly aggregate must
    # then refuse it rather than assume a Monday.
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)
    clock_source: Mapped[str] = mapped_column(String(16), nullable=False)
    lesson_date: Mapped[date | None] = mapped_column(Date, index=True)
    duration_minutes: Mapped[float] = mapped_column(Float, nullable=False)

    analysed_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    pupil_seats: Mapped[int] = mapped_column(Integer, nullable=False)

    # The whole provenance + uncertainty blocks, verbatim. Stored rather than exploded
    # into columns because their job is to be shown next to the numbers and audited, not
    # queried -- and because a schema change upstream must not silently drop a field.
    provenance: Mapped[dict] = mapped_column(JSON, nullable=False)
    uncertainty: Mapped[dict] = mapped_column(JSON, nullable=False)
    caveats: Mapped[dict] = mapped_column(JSON, nullable=False)

    seats = relationship("ClassSeatRecord", back_populates="analysis",
                         cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("school_id", "run_id", name="uq_class_analyses_school_run"),
        Index("ix_class_analyses_school_date", "school_id", "lesson_date"),
    )


class ClassSeatRecord(Base, TimestampMixin):
    """One seat in one analysed lesson.

    `person_id` is nullable and that nullability is the design: NULL means "this place was
    not attested to anybody", which is the ordinary case and a complete record. It is
    filled ONLY from an attested seat map -- never from a face, never from an inference --
    and `identity_method` records which, so a later reader can tell.
    """

    __tablename__ = "class_seat_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("class_analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The artefact's OWN per-run seat number. Kept for tracing back to the JSON, and
    # deliberately NOT the key anything longitudinal joins on -- see §2a.
    seat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    seat_label: Mapped[str] = mapped_column(String(32), nullable=False)

    # The cross-lesson identity. NULL means "this seat could not be matched to a known
    # place in this room", which is a complete and honest record: the row is stored in
    # full and excluded from every accumulation, rather than joined to the nearest history.
    place_id: Mapped[int | None] = mapped_column(
        ForeignKey("class_places.id", ondelete="SET NULL"), index=True
    )
    place_match: Mapped[str] = mapped_column(String(16), nullable=False)
    place_match_reason: Mapped[str | None] = mapped_column(Text)
    place_match_distance: Mapped[float | None] = mapped_column(Float)  # shoulder widths

    role: Mapped[str] = mapped_column(String(16), nullable=False)  # pupil | adult | excluded

    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )
    identity_method: Mapped[str] = mapped_column(String(32), nullable=False)
    identity_reason: Mapped[str | None] = mapped_column(Text)

    # Coverage FIRST, because no count below it may be read without it.
    coverage: Mapped[float] = mapped_column(Float, nullable=False)
    observations: Mapped[int] = mapped_column(Integer, nullable=False)
    settled: Mapped[bool] = mapped_column(Boolean, nullable=False)

    hand_raises: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stands: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    away_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    head_down_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    turned_away_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # NULL when the index was refused (coverage below the floor). NULL is not zero.
    activity_index: Mapped[float | None] = mapped_column(Float)
    activity_parts: Mapped[dict] = mapped_column(JSON, nullable=False)
    ledger: Mapped[dict] = mapped_column(JSON, nullable=False)

    analysis = relationship("ClassAnalysis", back_populates="seats")

    __table_args__ = (
        UniqueConstraint("analysis_id", "seat_id", name="uq_class_seat_records_run_seat"),
        Index("ix_class_seat_records_person", "person_id", "seat_id"),
    )


class ClassPlace(Base, TimestampMixin):
    """A fixed position in a fixed room used by a fixed class. The unit of accumulation.

    This is the table §2a is about. A `seat_id` belongs to one analysis run; a place
    outlives every run, and everything longitudinal joins on it.

    The geometry is ANCHORED on the first run that revealed the place and never drifts.
    A running mean would let a slowly-moving camera carry a place across a desk over a term
    with nothing ever failing; a fixed anchor makes a moved camera show up as a run in
    which every seat is unmatched, on a particular date, which is a fact a human can act on.

    `class_key` is in the key alongside `camera` because two classes sharing one room have
    different occupants, and merging their place histories is the seat-swap defect at the
    scale of a timetable.
    """

    __tablename__ = "class_places"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera: Mapped[str] = mapped_column(String(64), nullable=False)
    class_key: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)   # stable display number
    role: Mapped[str] = mapped_column(String(16), nullable=False)

    anchor_x: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_y: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_scale: Mapped[float] = mapped_column(Float, nullable=False)  # shoulder width px
    first_run_id: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("school_id", "camera", "class_key", "ordinal",
                         name="uq_class_places_room_ordinal"),
    )


class SeatAttestation(Base, TimestampMixin):
    """A human's dated statement that a PLACE holds a named child.

    The only route by which `ClassSeatRecord.person_id` may ever be filled. Kept as its
    own table, with `attested_by` and a validity window, because it is EVIDENCE about a
    period, not a property of a lesson: when the class is re-seated, the old rows stay
    true about the old period and a new row starts.

    **`place_id`, not `(camera, seat_id)`.** That was the draft, and §2a is why it changed:
    a seat number is re-assigned by reading order on every run, so an attestation keyed on
    it points at a different desk the first time a pupil is absent.

    `decision_ref` is NOT NULL and is not decoration. Per-pupil accumulation is what
    `docs/questions-for-school.md` §8 told the school would not happen and §10.1 then asked
    the school to decide about in writing. An attestation nobody can trace to that answer
    is the quiet re-opening of a written promise, and a NOT NULL column is the cheapest
    possible way to stop it happening by accident.
    """

    __tablename__ = "seat_attestations"

    id: Mapped[int] = mapped_column(primary_key=True)
    school_id: Mapped[int] = mapped_column(
        ForeignKey("schools.id", ondelete="CASCADE"), nullable=False, index=True
    )
    place_id: Mapped[int] = mapped_column(
        ForeignKey("class_places.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)
    attested_by: Mapped[str] = mapped_column(String(200), nullable=False)
    attested_at: Mapped[date] = mapped_column(Date, nullable=False)
    decision_ref: Mapped[str] = mapped_column(String(200), nullable=False)

    __table_args__ = (
        Index("ix_seat_attestations_lookup", "school_id", "place_id", "valid_from"),
    )
```

---

## 2a. The defect building the store found: `seat_id` is not a key across lessons

The reference accumulator is
`classvision/src/classvision/cabinet/store.py`, and it exists partly to make this
argument runnable rather than rhetorical.

`room/seats.py` numbers seats by sorting the discovered clusters into reading order:

```python
seats.sort(key=lambda s: (round(s.centre[1] / max(s.scale, 1e-6) / 1.5), s.centre[0]))
for number, seat in enumerate(seats, start=1):
    seat.seat_id = number
```

**One pupil absent on Tuesday is one fewer cluster, and every id behind the gap shifts by
one.** Appending `seat_3` to `seat_3` across a term therefore builds a history out of two
or three different children — and produces a textbook sustained decline that is pure
bookkeeping. `metrics/trend.py` already refuses on exactly this hazard
(«личность на этом месте не подтверждена»), but it cannot fire here, because both rows
genuinely say `seat_3` and there is nothing to notice. The draft above joined
`seat_attestations` on `(camera, seat_id)` and would have inherited the whole defect
silently.

**The fix is a place matched by geometry, with a gate AND a margin.** Distance normalised
by the mean of the two shoulder widths (the room is viewed at an angle, so a pixel gate
spans two back-row desks while missing one front-row pupil leaning forward):

| constant | value | why |
|---|---|---|
| `PLACE_MATCH_GATE_SCALES` | **1.0** shoulder widths | `assign()` uses 1.5 *within* a lesson, where a miss costs one observation; here a miss costs a term of one child's history welded onto another's. Two pupils sharing a desk are 1.5–2.5 apart, so 1.0 cannot span a neighbour. |
| `PLACE_MATCH_MARGIN` | **2.0×** | `MEASUREMENTS.md` §4 applied to geometry rather than faces: a best score with no margin over the runner-up is a preference, not an identification. A re-arranged room produces exactly that case, and the right answer there is `place_id = NULL` and a question for the teacher. |

Measured on the two real artefacts (`full_lesson` 50 min and `clip_15min`, same camera,
different videos, independently discovered seats): **9 places created from the first
recording, 9 of 9 recognised in the second, 0 unmatched, 0 ambiguous.** Shifting every
seat by 600 px produces **0 matched and 9 new places** rather than a silently inherited
history — the moved-camera case, asserted in
`tests/test_cabinet.py::test_a_moved_camera_creates_new_places_rather_than_inheriting_a_history`.

### Two more accumulation-only checks the store added

Neither is expressible in a per-lesson artefact, because both are questions about a
*second* recording:

* **Wall-clock overlap.** `clip_15min.mp4` is a slice of `test_camera.mp4`. Their content
  hashes differ, so nothing upstream can tell they are the same hour, and importing both
  doubles that hour in every weekly counter. The store refuses on an overlap in the same
  room and requires an explicit `--allow-overlap`, which is then recorded on the lesson for
  ever and printed on the report. **This fired on the real `out/` directory**, which is how
  it earned its place.
* **DVR continuation.** The D14 pair is one lesson cut in two. Two recordings is the
  correct storage; **one session** is the correct unit of accumulation, so the second row
  carries `continues_lesson_id` and `weekly.py` chains it. Two constants, and the second
  one is the interesting half:
  * `CONTINUATION_GAP_SECONDS = 180` — generous for a recorder rolling a file, far too
    tight to join two real lessons, since the shortest break in the timetable is 5 minutes
    and a class changeover takes longer than the break.
  * `CONTINUATION_TOLERANCE_SECONDS = 2` — how far a continuation may appear to start
    *before* its predecessor ended. File one starts 10:17:59 and runs 818.00 s, ending at
    10:31:37; file two starts **10:31:36**. A genuine seam therefore reads as a
    one-second OVERLAP, because both clocks are known only to the second. A window of
    `[0, gap]` misses the exact pair the check exists for, so the continuation question is
    asked **before** the overlap refusal and the matched predecessor is excluded from it.
    Both facts are stored: a lesson can be a continuation *and* a forced overlap at once
    (`tests/test_cabinet.py::test_a_lesson_that_is_both_a_continuation_and_a_forced_overlap_keeps_both_marks`).
  * **The seam is looked for in both directions**, and this was a defect before it was a
    design. The first build asked only «какая запись кончается прямо перед этой», so
    importing the real D14 pair long-file-first stored **two** independent lessons —
    «занятий 2» for an hour that had one, every per-lesson figure of it halved — and
    nothing refused and nothing warned, because a lesson followed by another lesson is what
    a normal school day looks like. A directory back-fill takes whatever order the glob
    produced. `_immediately_after` is the mirror check: an already-stored lesson that
    starts within the same window after this one ends is pointed *back* at it, and both
    matches are excluded from the overlap rule, so the refusal cannot depend on typing
    order either (`test_a_split_recording_is_one_session_whichever_file_arrives_first`,
    `test_a_seam_wide_enough_to_read_as_an_overlap_is_not_refused_in_either_order`).
  * **A re-import under a different `--room-key` is refused** (`already_imported_elsewhere`)
    rather than silently doing nothing. Idempotency is on `run_id` alone, so a corrected key
    wrote nothing at all while printing «уже импортирован — ничего не записано» — which an
    operator fixing a typo reads as «принято», leaving the lesson filed under the mistyped
    room for ever.

### What folding a session does and does not preserve

The D14 lesson exists twice over — as one artefact of the concatenated recording, and as
two artefacts the cabinet chains. Comparing the two routes per place
(`MEASUREMENTS.md` §9.2): **observations and coverage agree to within 0.1–4 %, episode
counts do not** (3 vs 0, 17 vs 14, 8 vs 3 for «вставал»). That is not a bug. An episode is
a duration with a hold and a gap, the file seam cuts one in half, and each part
re-establishes its own posture baseline. So `_SummedLedger` adds the artefacts' own episode
COUNTS and never invents an episode object, the index is recomputed from the summed
observation histogram by `metrics/activity.py` itself, and where a concatenated analysis
exists it is the better measurement. `qorgan` must carry the same rule, or a term will mix
two definitions of «сколько раз встал».

---

## 3. Capability

One new member in `src/qorgan/roles.py`, added **in the same change** as the page it
guards — the rule that file states about itself.

```python
    # Imported offline classroom analyses (`/classroom`). Separate from
    # VIEW_LESSON_METRICS, which guards the LIVE worker's anonymous track counts: these
    # rows can carry a pupil's name when a seating plan was attested, so they are a
    # different disclosure and get a different grant.
    VIEW_CLASSROOM_ANALYSIS = "view_classroom_analysis"  # /classroom, /classroom/{id}
```

Held by `UserRole.ADMIN`, `UserRole.DEVELOPER` and `UserRole.PSYCHOLOGIST`. **Not** by
`OPERATOR` — an operator's job is live incidents, and these rows are longitudinal
observations about named children, which §13 keeps to the psychologist. **Not** by
`CANTEEN_STAFF` or `SUPERADMIN`, which hold no child-facing capability at all.

---

## 4. The psychologist cabinet block

**This section is now a port, not a design.** The whole surface exists and runs offline:

| file | what it is |
|---|---|
| `cabinet/store.py` | the SQLite store (§2/§2a in stdlib form) — `connect`, `import_artefact`, `select_run`, `attest`, plus the read helpers `lessons`, `places`, `seat_rows`, `caveats`, `unmeasured`, `summary` |
| `cabinet/weekly.py` | sessions, ISO-week rows, and `TrendView` — the aggregation `qorgan`'s service layer mirrors |
| `cabinet/report.py` | the finished HTML the psychologist opens, including all three empty states |
| `classvision cabinet import / weekly / report / show / select-run / attest` | the CLI |

`qorgan` should port the **shapes and the refusals**, not re-derive them. Every named state
below is a value of `weekly.TrendView.state` and carries its own Russian sentence in
`headline_ru` / `detail_ru`; copying those strings by hand is the four-copies-of-a-caveat
defect this document already warns about twice.

### The `Block` state

`src/qorgan/psychologist/cabinet.py` gains one `Block`. Its `SignalState` is computed, not
fixed, and the three states mean genuinely different things:

| condition | state | what the page says |
|---|---|---|
| no analyses imported | `EMPTY` | «пока пусто» |
| analyses imported, no place attested | `ANONYMOUS` | «накапливается, но без личности» |
| at least one place attested to a pupil | `LIVE` | «сигнал живой» |

`ANONYMOUS` is the important one and it already exists in `signals.py` for exactly this
shape of problem. The lines the block owes, matching what the built page prints:

```python
Block(
    key="classroom_analysis",
    title="Наблюдения в классе",
    state=state,
    count=seat_records_count,
    lines=(
        f"Разобрано уроков: {lessons} (занятий {sessions}). Мест под наблюдением: {places}.",
        f"Имена проставлены для {named} мест из {places} — только по подписанному "
        f"плану рассадки.",
        "Единица учёта — место, а не ребёнок. Система не ставит диагнозов.",
        ("Динамика по ученику появится после "
         f"{MIN_HISTORY_LESSONS + 1} разобранных уроков с подтверждённым местом."),
    ),
)
```

### The trend states, which are results and not errors

`metrics/trend.py` refuses below **5** lessons of a pupil's own history and refuses
outright when identity is not stable. `weekly.trend_for()` surfaces each refusal as a named
state rather than an empty chart, because with one lesson «динамики нет» *is* the answer
and a blank chart gets debugged instead of read:

| `TrendView.state` | when | what the page shows | fixed by |
|---|---|---|---|
| `no_data` | the place has no dated lesson | «уроков ещё нет» | recording one |
| `no_dated_lessons` | lessons exist, clock unreadable | «есть N уроков без даты» + how to fix | a date, not waiting |
| `identity_not_established` | no signed seating plan covering the period | counters in full, the word «динамика» replaced by the reason | a human, never more data |
| `place_unstable` | the place was ambiguous in some lesson | «место опознано не во всех уроках» | re-attesting after a re-arrangement |
| `insufficient_lessons` | fewer than 5 | «данных пока мало: 1 урок из 5» + how many remain | time |
| `available` | all four gates met | direction, magnitude in the pupil's own MADs, and `expected_by_chance()` beside it | — |
| `not_applicable_adult` | the place is the adult's | position per lesson, no index, no weeks, no trend | never |

Every state also carries `gates` — four `{gate, passed, measured, required, detail_ru}`
records, the same shape `identity/assign.py` uses for names — so a reader learns *which* of
four things is missing rather than that something is. That distinction is the whole
usefulness of the page in its first term: «подождите ещё три урока» and «подпишите план
рассадки» are different actions, and a single shrug for both is what makes a dashboard
get ignored.

### Two rules the port must not lose

* **`not_applicable_adult` is not an oversight.** The adult's ledger has the same *shape*
  as a pupil's, so `metrics/activity.py` will happily produce an index from it — the first
  build of this layer did, and printed «индекс 91,2 · поднимал руку 60» for a teacher who
  was pointing at a board. That number is a teaching-quality score assembled out of a
  pupil's vocabulary, which `metrics/teacher.py` and §7 of this document both refuse to
  produce. The index is refused **by role**, with the reason carried in the object exactly
  as `activity()` carries its own refusals.
* **A sparkline below the threshold draws points but no line.** A line through two dots is
  a direction, and a direction is precisely the claim the sentence above it has just said
  cannot be made; a reader who takes the picture and leaves the words behind would leave
  with the opposite of what the page said.
* **A place with no accumulated session shows no counters at all — not zeros.** An
  undated recording is excluded from everything longitudinal, correctly; summing its empty
  session list gave `0` and the class page printed «поднимал руку — 0 · видимость от 0 %»
  for a pupil the same run had watched raise his hand twice, one line under a badge saying
  the lesson had not been accumulated. `PlaceHistory.totals` is `None` per key when there
  is nothing to sum, and `coverage_min` is `None` rather than `0`
  (`test_an_undated_place_shows_no_counters_rather_than_a_row_of_zeros`).
* **An adult the artefact could not place is listed, never omitted.** A `teacher` block
  with no `centre` used to be dropped at import without a word: `clip_15min.named.json`
  stored eight places beside two other runs of the same lesson that stored nine, and the
  D14 class page showed six pupils and no adult at all. «Мы не знаем, где он был» and «его
  не было» are different facts. The import now reports it in `dropped_seats`, and
  `store.adults_without_place()` puts it on the page, quoting the artefact's own reason.
  No geometry is inferred from `identification.evidence.centre` to paper over it.
* **The adult's shares are read only under the names that state their denominator.**
  `classvision/1.0` artefacts call them `seated_share` / `out_of_frame_share`; `1.1` calls
  them `at_desk_share_of_observed` / `out_of_frame_share_of_lesson`, and §6b explains that
  the rename *was* the fix. A `metrics.get(new, metrics.get(old))` bridge put the 1.0
  number under the 1.1 heading — the identical substitution the rename ended. A 1.0 row now
  renders as its own sentence naming its schema, with the raw old-named values shown intact
  (`test_a_1_0_artefacts_adult_shares_are_not_filed_under_a_1_1_column`).

---

## 5. Migration

`migrations/versions/0010_a_seat_outlives_a_track.py`, `down_revision = "0009"`.

Creates the four tables above, in the order `class_places` → `class_analyses` →
`class_seat_records` → `seat_attestations` (both `class_seat_records.place_id` and
`seat_attestations.place_id` point at the first). **Touches nothing existing** — no column
is added to `lessons` or `lesson_tracks`, so the live classroom worker and the `/lessons`
page are unaffected and the promise in `db/models/classroom.py` stays literally true.

`downgrade()` drops the four tables. That is safe precisely because nothing else
references them.

---

## 6. Importer CLI

```bash
qorgan classvision import <artefact.json> [--school <slug>] [--dry-run]
```

Refuses, loudly, when:

* `schema_version` is not one it knows — a future analyser must not be half-imported.
* `clock_source == "unknown"` **and** the caller asked for it to count toward a trend:
  a run with no date can be stored but must not silently land in a week.
* the artefact names a pupil `external_id` that is not in `persons` for that school.
* a seat is named but carries no attestation record — the JSON is not itself an
  attestation; the signature is.

Idempotent on `(school_id, run_id)`: re-importing the same file is a no-op that reports
"already imported", and re-analysing with different thresholds creates a **new** row rather
than mutating the old one.

A reference implementation that emits the rows it *would* insert, so it can be tested
without a database, is in `integration/qorgan_importer.py`.

### 6a. The offline cabinet, which is the same import against SQLite

`integration/qorgan_importer.py` plans rows for `qorgan`'s Postgres and opens no database.
`cabinet/store.py` is the same seam against a single SQLite file a psychologist can copy to
a USB stick, and it is what actually runs today:

```bash
classvision cabinet import out/*.analysis.json --room-key camera01 --class 3-Б \
        [--allow-unclocked] [--allow-overlap] [--select]
classvision cabinet show      --room-key camera01 --class 3-Б
classvision cabinet weekly    --room-key camera01 --class 3-Б [--json]
classvision cabinet report    --out out/cabinet
classvision cabinet select-run <run_id>
classvision cabinet attest --place-id 3 --external-id student_53 \
        --by "Иванова И.И., классный руководитель" --at 2026-09-01 \
        --valid-from 2026-09-01 --valid-to 2026-12-31 --decision "Протокол №14 от 28.08"
```

Exit codes follow the rest of the CLI: `0` usable, `1` it ran and something was refused or
must not be trusted, `2` it could not run. A back-fill that treats a refused artefact as
success is how a term's data quietly acquires a doubled lesson.

**`--room-key` and `--class` are operator assertions, and the store records that they are.**
The artefact analyses a FILE; it does not know which room the camera is in or which class
was in it, and there is no field for either. Three sources, stored in `room_key_source`:
the flag (`operator`), the camera named in the room profile the analysis used
(`from_room_profile` — that profile is a signed file, so it is an earlier assertion by a
person), and last the video filename (`derived_from_filename`). The last one matters,
because `clip_15min.mp4` and `test_camera.mp4` are the *same camera* and would otherwise
become two rooms that never accumulate together — which is exactly why the source is a
stored column rather than a comment.

The flag is `--room-key`, not `--room`, because `analyse --room` already takes a profile
FILE in this CLI. Two flags with one name and two meanings in one program is a defect
waiting for a tired operator.

The two refusals `qorgan`'s importer does not have, because they are questions only an
accumulator can ask, are `overlapping_lesson` and the DVR-continuation detection in §2a.
`qorgan` should carry both: nothing else in the pipeline can see them.

---

## 6b. What schema 1.1 gained for the adult (camera D14)

`TeacherRecord` now carries two timelines and one new metrics sub-block. They are separate
fields on purpose and the importer must keep them separate.

| field | what it is | denominator |
|---|---|---|
| `teacher.timeline` | POSITION episodes — `TeacherState` (`at_board` / `at_desk` / `among_pupils` / `moving` / `out_of_frame`) | — |
| `teacher.pose_timeline` | POSE episodes — `PupilState`, from his seat ledger. Empty when he never settled anywhere | — |
| `metrics.presence.state_share_of_lesson_percent` | every state incl. `out_of_frame`, sums to 100 | every analysed frame |
| `metrics.presence.state_share_of_attributed_percent` | every state EXCEPT `out_of_frame`, sums to 100 | frames attributed to the adult |
| `metrics.presence.state_minutes_of_lesson` | the same counts as minutes | — |
| `metrics.presence.episode_minutes_by_state` | time inside runs that survived a minimum hold — **always smaller** | — |
| `metrics.presence.board` | the client's question, with both denominators and `direction_of_error_ru`. **All five numeric fields are `null` when `zone_configured` is false** — see below | both |
| `metrics.presence.floor_coverage` | cells of one shoulder width visited, against cells anyone visited | cells in use |
| `metrics.presence.per_minute` | one row per minute, counts AND shares, for charting | the minute |
| `metrics.presence.identification` | which route found the adult, and how much it could hold | — |
| `provenance.room.layout.zones_confirmed_by` | who signed the polygons; `""` means nobody | — |

The pose fields (`at_desk_share_of_observed` and friends) are unchanged and still present,
because cameras without a board in frame still produce only those.

### 6b.1 Three things in this block that an importer must not treat as ordinary numbers

Found by review, after the block had been built and its numbers reproduced.

* **`metrics.transitions` is a POSE count and is `null` whenever there is no pose ledger.**
  It counts settled-at-his-desk ↔ upright-or-away, from `SeatLedger.episodes`. It used to
  be filled, on the no-seat path only, from `presence.transitions_between_episodes` — a
  count of POSITION episode boundaries that includes every boundary where the follower
  started or stopped seeing him. Two quantities, one key, and the report renders that key
  as «смен положения» inside a sentence about what his body was doing at his desk. The
  position counts stay inside `presence`, under
  `transitions_between_episodes` and `transitions_between_episodes_excluding_out_of_frame`,
  and are **not** copied to a shorter name. Render the second of those, never the first,
  next to the word «смен места».

* **`metrics.out_of_frame_share_of_lesson` means "the adult was not located".** When a
  `presence` block is present that number comes from the follower. It used to come from the
  seat ledger — the share of frames in which the adult's DESK was empty — which on a camera
  where he walks is a different fact and a false one: on `D14_20260815103136.mp4` the desk
  was empty for 51,5 % of the lesson and the adult was unlocated for 55,0 %, and both were
  in the same artefact. The desk's emptiness is still available, honestly named, as
  `teacher.ledger.absent_observations`.

* **`presence.board.*` is `null`, not `0`, when `zone_configured` is false.**
  `classify_track` cannot produce `at_board` without a polygon, so every figure there was
  structurally zero — and was rendered as «0,0 мин · 0 % урока» under the heading that
  answers the client's question, with «Скорее занижено» beside it. `configs/camera_01.yaml`
  has `board_zone: null`, so this is the state of the other configured camera in the
  repository. An importer must render `null` here as «не измерялось» and must never sum it.
  `presence.state_share_of_lesson_percent.at_board` stays `0.0` on that camera, because the
  five state shares have to keep summing to 100; `board.zone_configured` is what tells you
  whether that zero is a measurement.

* **`classvision_teacher_records` gains one column:
  `attributed_share_of_lesson_percent` (`Float`, nullable).** §7 forbids rendering any
  teacher number without it, and the importer previously had no column to put it in — the
  nearest thing, `coverage`, is the follower's share on one code path and the SEAT's
  occupancy on the other, so a page reading it could not know which guarantee it had.
  `integration/qorgan_importer.py` now **drops** a teacher block that carries a `presence`
  sub-block without this value, with the reason in `Plan.dropped`. `NULL` means the
  artefact has no position taxonomy at all (every camera before D14) and therefore no
  state shares needing a denominator.

---

## 7. What must NOT be built

* No automatic referral. `REFER_TO_PSYCHOLOGIST` stays an act by a named human
  (`0008_a_referral_is_an_act_by_a_named_person.py` is that principle in a migration).
* No sort-by-index on any list of children, and no colour scale that implies a ranking.
  A sort order is a recommendation with the argument left out.
* No CSV export of per-child rows, for the reason `web/routes/lessons.py` already gives:
  a file detached from the page that explains the caveats is the artefact that gets
  forwarded to a parent as though it meant something. If the school asks, build it with
  the caveats **inside** the file, as `canteen._caveat_rows` does.
* No teacher surface for administration. The teacher block is position over time and
  carries `not_an_assessment_ru`; it must not become a staff-comparison page.
* **No teacher number rendered without `attributed_share_of_lesson_percent` beside it.**
  On camera D14 the follower can attribute only 45 % of the lesson to the adult, and the
  state shares are computed against the WHOLE lesson so that `out_of_frame` (55 %) is one
  of them rather than a silent divisor. A page that shows «у доски — 3 %» without «опознан
  в 45 % кадров» has told the reader something false with a true number. The importer must
  refuse a teacher block whose `presence.attributed_share_of_lesson_percent` is missing.
* **No teacher trend across lessons, and no week-over-week teacher comparison.** The pupil
  side accumulates because a seat is stable; the adult's numbers move with how much of him
  the follower could hold, which moves with where he happened to stand. A line going down
  would read as a teacher getting worse when it means the camera saw less of him.

---

## 8. Tests

1. **Import guard.** Fails if `qorgan` gains a heavy dependency:
   ```python
   def test_web_process_never_imports_a_model_stack():
       import subprocess, sys, json
       code = ("import qorgan.web.app, sys, json;"
               "print(json.dumps([m for m in sys.modules"
               " if m.split('.')[0] in {'torch','ultralytics','cv2','insightface','onnxruntime'}]))")
       out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
       assert json.loads(out.stdout) == []
   ```
2. **Reverse guard.** `grep -r "import qorgan" classvision/src` returns nothing.
3. **Fixture round trip.** The committed artefact imports, and importing it twice inserts
   the same number of rows as importing it once.
4. **Refusal tests.** Unknown schema version, unknown `external_id`, named seat with no
   attestation, `clock_source == "unknown"` into a weekly aggregate — each must raise, and
   the test asserts on the *message*, because the message is what an operator acts on.
5. **`person_id` provenance.** No code path fills `ClassSeatRecord.person_id` from anything
   whose `identity_method` is not `seat_map_attested`. Assert it directly.
6. **The accumulation tests already exist** and are the port's specification:
   `classvision/tests/test_cabinet.py`, 28 cases against the real artefacts. The four that
   matter most, because each covers a defect that is invisible once it is in a database:
   * `test_one_part_session_reproduces_the_artefacts_own_index` — the aggregation
     recomputes the index from a **summed ledger** using `metrics/activity.py` itself
     rather than averaging the parts' indices. A one-recording session must reproduce the
     artefact's own number **exactly**, for all 8 pupil places; if it does not, the weekly
     figures are a second, quietly different implementation of the index — the
     `MEASUREMENTS.md` §8 defect one layer up.
   * `test_a_reanalysis_is_a_new_run_and_does_not_replace_the_old_one` — same lesson, new
     run, and `selected_run_id` **does not move**. Moving it is `select_run()`, a separate
     named act.
   * `test_a_moved_camera_creates_new_places_rather_than_inheriting_a_history` — §2a.
   * `test_a_week_with_no_lesson_carries_None_and_not_zero` — a week with no lesson and a
     week with zero hand-raises are different rows, for ever.

   Plus the import guard in this file's own terms:
   `test_the_cabinet_never_imports_a_model_stack` asserts the accumulation layer loads no
   torch, ultralytics, cv2, insightface, onnxruntime **or matplotlib**. It caught a real
   one: `cabinet/report.py` imports `CSS` from `report/html.py` — one look for both
   surfaces rather than a second copy — and `html.py` imported `charts` at module level,
   so obtaining a string loaded ~90 matplotlib modules. `charts` is now imported inside
   `render_report()`.

---

## 9. Licensing — settle this before the school pays for anything

Two obligations that a school deployment inherits and that are easy to miss:

* **Ultralytics (YOLO11) is AGPL-3.0.** A network-served derivative work triggers source
  disclosure. Keeping `classvision` in a separate process that only writes a file is what
  keeps the web application out of that surface, and it is a real reason for the seam in
  §1 — not just tidiness. If a commercial licence is not bought, this separation is doing
  legal work and must not be "simplified" later.
* **InsightFace `buffalo_l` weights are non-commercial.** Face corroboration is off by
  default, which is convenient: a school that never switches it on never touches them.

Both are avoidable — RTMPose/RTMDet (Apache-2.0) is the escape route for the detector.
Decide before deployment, not after.

---

## 10. Kazakhstan personal-data note

Attaching a child's name to behavioural observations is processing personal data under
**ЗРК «О персональных данных и их защите» (94-V)**. Before any seat map is attested:

* the school's written decision, with its date, recorded in the analysis provenance;
* the retention period for artefacts and for the video itself, and something that enforces
  it (`qorgan janitor` is the existing mechanism);
* who may read the cabinet — already enforced by capability, §3;
* the answer to «что мы удаляем и когда» before the first import, not after the first
  request.

Without the seat map none of this is triggered: places are not personal data. That is a
genuine reason to run anonymous by default, not merely a cautious one.

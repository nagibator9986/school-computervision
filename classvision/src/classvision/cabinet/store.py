"""The cabinet store: many lesson artefacts, accumulated without losing what they said.

--------------------------------------------------------------------------------
**THE PROBLEM THIS FILE EXISTS FOR, AND IT IS NOT "WHERE DO WE PUT THE JSON".**

A single artefact is already a complete, honest statement about one hour. Accumulating
lessons looks like appending rows and is not, because the artefact's own key for a place
— `seat_id` — **is not stable across lessons, and nothing in the artefact says so**.

`room/seats.py` assigns ids by sorting discovered clusters into reading order:

```python
seats.sort(key=lambda s: (round(s.centre[1] / max(s.scale, 1e-6) / 1.5), s.centre[0]))
for number, seat in enumerate(seats, start=1):
    seat.seat_id = number
```

One pupil absent on Tuesday means one fewer cluster, which means **every id behind the
gap shifts by one**. Appending `seat_3` to `seat_3` across a term therefore builds a
history out of two or three different children and produces a textbook decline that is
pure bookkeeping. This is the same class of defect `metrics/trend.py` refuses on
(«личность на этом месте не подтверждена») — except here it would be invisible, because
both rows genuinely say `seat_3`.

So the store introduces a unit the artefact does not have: a **place**
(`places`), matched from lesson to lesson **by geometry, with a gate and a margin**, in
shoulder widths rather than pixels. A seat that cannot be matched cleanly gets
`place_id = NULL` and a sentence saying why, and it is then absent from every longitudinal
view rather than silently joined to the wrong history.

--------------------------------------------------------------------------------
**FOUR THINGS THIS SCHEMA IS ORGANISED TO MAKE IMPOSSIBLE.**

1. **A re-import changing anything.** `runs.run_id` is the artefact's own hash of the
   video plus every setting that could change a number. `INSERT OR IGNORE` on it, and the
   second import of the same file writes nothing and says so. Idempotency is a property
   of the key, not of a code path that has to be remembered.

2. **A re-analysis silently replacing a measurement.** A second run of the *same video*
   under different thresholds is the same `lessons` row and a **new** `runs` row. It is
   stored, it is visible, and `lessons.selected_run_id` **does not move** — a term's
   aggregate keeps using the run it has always used until a human runs
   `select_run()` deliberately. Rule 6 of this project in a single column: «a re-run with
   different settings is a NEW artefact, never an overwrite».

3. **Two overlapping recordings both counting as lessons.** `clip_15min.mp4` is a slice of
   `test_camera.mp4`. Their content hashes differ, so nothing upstream can tell they are
   the same hour — but their wall clocks overlap, and importing both doubles one lesson in
   every weekly counter. `import_artefact` refuses on a wall-clock overlap in the same
   room and requires `allow_overlap=True`, which is then recorded on the lesson **for
   ever** and printed on the report, because "an operator once said this was fine" is
   information the reader of the number needs.

4. **A DVR split counting as two lessons.** The D14 pair is one lesson cut in two by the
   recorder. Two `lessons` rows is the correct storage — they are two recordings — but one
   *session* is the correct unit of accumulation, so the second row carries
   `continues_lesson_id` and `weekly.py` chains them. Detected from the clock, at import.

   The seam is measured, not assumed, and it is why the continuation question is asked
   **before** the overlap refusal in (3): file one starts 10:17:59 and runs 818.00 s,
   ending at 10:31:37, while file two starts **10:31:36**. A back-to-back pair therefore
   *looks like* a one-second overlap, because both clocks are known only to the second.
   Ask overlap first and the commonest real case in a school needs an operator override
   every time — and arrives carrying a note saying it was a duplicate.

--------------------------------------------------------------------------------
**WHAT THE ARTEFACT DOES NOT KNOW AND THE OPERATOR MUST SUPPLY.**

`classvision` analyses a FILE. It has no idea which room the camera is in and no idea
which class was in it — there is no field for either, and inventing one from the filename
would be a guess wearing a column name. So `room_key` and `class_key` are **operator
assertions**, recorded with their source (`operator` / `from_room_profile` /
`derived_from_filename` / `from_attested_seat_map` / `not_stated`), and the source is
printed rather than the derived value being accepted quietly. Places are keyed on the
pair: two classes sharing one room have different occupants, and merging their place
histories would be the seat-swap defect at the scale of a timetable.

--------------------------------------------------------------------------------
**REFUSALS, AND WHY THEY ARE REFUSALS.**

The same argument `integration/qorgan_importer.py` makes: a warning on an import is a line
in a log nobody reads, after which the row is in the database for ever and every later
report is computed over it. The refusal set here is deliberately close to that file's —
that file must stay runnable with **zero** `classvision` imports (it is the executable
proof of the seam and it describes `qorgan`'s tables, not ours), so the two are relatives
rather than one calling the other. Where they differ, they differ on purpose: this store
adds the overlap and the DVR-continuation checks, which are questions only an accumulator
can ask, and drops the timezone question, which only a `UtcDateTime` column can ask.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_NAME = "classvision.cabinet"
# Bumped when a column changes MEANING. A store built by an older version is opened
# read-only rather than migrated in place, for the reason the artefact bumps its own major:
# a number whose definition changed sitting in the same column as last term's is the step
# change in a trend that nobody can explain afterwards.
#
# NOT bumped by a new TABLE. `run_notes` was added after this number was last set, and every
# statement in `SCHEMA_SQL` is `IF NOT EXISTS`, so an existing cabinet acquires the table on
# the next `connect()` with no measurement touched and nothing to migrate. Bumping here for
# an additive change would refuse to open every cabinet in the school — `connect()` raises
# rather than migrating — and buy exactly nothing: the danger this version guards against is
# an old NUMBER read under a new definition, and a table that holds no numbers cannot
# produce one.
SCHEMA_VERSION = 1

# Artefact majors this store can read. `classvision/1.x` adds fields; `2.x` would change
# what an existing field means.
ACCEPTED_SCHEMA_MAJOR = "classvision/1"

# How close a seat must be to a known place to BE that place, in shoulder widths.
# CHOSEN at 1.0. `room/seats.py::assign` uses 1.5 within one lesson, so that a seated pupil
# leaning across their desk does not fall out of their own seat; here the gate is tighter
# because the consequence is different. There, a miss costs one observation out of several
# thousand. Here it costs a term of one child's history welded onto another's, and the
# right answer in the ambiguous case is an unmatched seat and a question for the teacher.
# The measured separation between two pupils sharing a desk is ~1.5–2.5 shoulder widths
# (`room/seats.py`), so 1.0 cannot span a neighbour.
PLACE_MATCH_GATE_SCALES = 1.0

# ...and the nearest place must be this many times closer than the second nearest. This is
# `MEASUREMENTS.md` §4 applied to geometry instead of to faces: a best score with no margin
# over the runner-up is a preference, not an identification, and a re-arranged room
# produces exactly that ambiguous case. CHOSEN at 2.0, matching the importer's gate.
PLACE_MATCH_MARGIN = 2.0

# Gap between the end of one recording and the start of the next, in the same room, below
# which they are treated as one interrupted SESSION rather than two lessons. CHOSEN at
# 180 s from the D14 evidence: that DVR's two files are separated by 1 second, and no two
# genuinely distinct lessons in a school are separated by three minutes — the shortest
# break in a Kazakh school timetable is 5 minutes and the changeover of a class takes
# longer than the break. The value is generous enough for a recorder that drops a few
# seconds while rolling a file, and far too tight to join two real lessons.
CONTINUATION_GAP_SECONDS = 180.0

# Wall-clock overlap between two recordings in the same room below which they are treated
# as touching rather than overlapping. CHOSEN at 1 s, the quantisation of the burned-in
# overlay clock itself (`video/clock.py`): a smaller "overlap" than the clock can resolve
# is not evidence of anything.
OVERLAP_TOLERANCE_SECONDS = 1.0

# ...and how far a continuation may appear to start BEFORE its predecessor ended. CHOSEN at
# 2 s, and it is not slack: the D14 pair is measured evidence that a back-to-back pair
# looks like an overlap. File one starts 10:17:59 and runs 818.00 s, which ends it at
# 10:31:37; file two starts 10:31:36. Both start times are read to the second (from the
# filename here, from the burned-in overlay elsewhere), so a genuine seam can land on
# either side of zero and a window of [0, gap] misses half of them. 2 s is twice the
# clock's own quantisation and still two orders of magnitude below any real lesson gap.
CONTINUATION_TOLERANCE_SECONDS = 2.0

# Which artefact `counts` key holds each state's episode count. Written out rather than
# derived so that a state gaining a counter fails here, loudly, instead of aggregating as
# zero somewhere downstream. `seated` and `unknown` have no episode counter in the
# artefact, and are None rather than 0 — see rule 3 of this project: zero and "we do not
# have this" are different values.
COUNT_KEYS: dict[str, str | None] = {
    "hand_raised": "hand_raises",
    "stood_up": "stands",
    "at_board": "board_visits",
    "away_from_place": "away_episodes",
    "head_down": "head_down_episodes",
    "turned_away": "turned_away_episodes",
    "seated": None,
    "unknown": None,
}


class Refusal(Exception):
    """This artefact must not enter the cabinet, and here is the rule that says so.

    Carries a machine-readable `code` as well as a Russian sentence, because a term's
    back-fill is a script and "which refusal fired" must be answerable without matching on
    prose — the same reason `integration/qorgan_importer.py::Refusal` carries one.
    """

    def __init__(self, code: str, message_ru: str) -> None:
        super().__init__(message_ru)
        self.code = code
        self.message_ru = message_ru


class Unreadable(Exception):
    """The file could not be read or parsed.

    Distinct from `Refusal` on purpose. A refusal is a statement about the artefact; this
    is a statement about our ability to look at it. Collapsing the two lets a truncated
    download report as a schema violation, and then somebody goes and changes the schema.
    """


# ---------------------------------------------------------------------------
# The schema.
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One LESSON = one recording of one room by one class. Keyed on the video's content hash
-- because that is the only thing that identifies a recording independently of where the
-- file happens to sit or what it was called.
--
-- What actually keeps one recording out of two classes is `runs.run_id` being a global
-- PRIMARY KEY: the second import of the same artefact under a different room key finds the
-- run and refuses (`already_imported_elsewhere`). This UNIQUE is the weaker statement it
-- looks like — one row per (room, class, video) — and it is here so that the same video
-- deliberately re-imported into a fresh key set cannot land twice in one of them.
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id            INTEGER PRIMARY KEY,
    room_key             TEXT    NOT NULL,
    class_key            TEXT    NOT NULL,
    video_sha256         TEXT    NOT NULL,
    video_path           TEXT    NOT NULL,   -- last known path; NOT an identity
    -- Local wall clock exactly as `video/clock.py` produced it: no timezone, because the
    -- overlay carries none and a guessed offset moves a lesson across a day boundary.
    -- Everything longitudinal here is per ISO week of the SCHOOL's own calendar, which is
    -- the calendar this string is already in, so no conversion is needed or performed.
    started_at           TEXT,
    ended_at             TEXT,
    date_local           TEXT,               -- YYYY-MM-DD, for grouping and display
    iso_year             INTEGER,
    iso_week             INTEGER,
    iso_weekday          INTEGER,
    duration_seconds     REAL,
    -- "overlay" | "filename" | "operator" | "unknown". Never inferred from the value.
    -- A lesson whose clock_source is "unknown" has no date and is excluded from every
    -- weekly view; it is still stored, because refusing to store it would lose the fact
    -- that a recording exists.
    clock_source         TEXT    NOT NULL,
    -- Where room/class came from. "operator" is an assertion by a human; anything else is
    -- a convenience that must be visible on the page, because two lessons filed under a
    -- guessed room key would accumulate as one room.
    room_key_source      TEXT    NOT NULL,
    class_key_source     TEXT    NOT NULL,
    -- The previous recording this one continues, when a DVR split one lesson in two.
    continues_lesson_id  INTEGER REFERENCES lessons(lesson_id),
    -- Set only when an operator forced an import over a wall-clock overlap. Kept for ever
    -- and rendered on the report: "somebody once said this was fine" qualifies every
    -- number computed over it.
    overlap_note         TEXT,
    -- Which run of this lesson the aggregates use. Deliberately NOT a foreign key: `runs`
    -- references `lessons`, and a circular FK in SQLite would have to be deferred, which
    -- buys nothing here. Moved only by `select_run()`, never by an import.
    selected_run_id      TEXT,
    imported_at          TEXT    NOT NULL,
    UNIQUE (room_key, class_key, video_sha256)
);

-- One RUN = one artefact = one measurement of one lesson under one set of settings.
-- Several runs per lesson is the normal, intended state after a threshold is argued about.
CREATE TABLE IF NOT EXISTS runs (
    run_id                    TEXT PRIMARY KEY,
    lesson_id                 INTEGER NOT NULL REFERENCES lessons(lesson_id),
    schema                    TEXT NOT NULL,
    artefact_path             TEXT NOT NULL,
    artefact_sha256           TEXT NOT NULL,  -- the FILE, not the run: catches a truncated copy
    imported_at               TEXT NOT NULL,
    analysed_at               TEXT,
    sample_fps                REAL,
    window_start_seconds      REAL,
    window_end_seconds        REAL,
    window_wall_start         TEXT,
    window_wall_end           TEXT,
    analysed_frames           INTEGER,
    -- Provenance kept whole rather than shredded into columns: the question a psychologist
    -- actually asks of it is «обе недели мерили одинаково?», which is a comparison of the
    -- whole object, and `thresholds_sha` answers it in one equality test.
    model_json                TEXT NOT NULL,
    thresholds_json           TEXT NOT NULL,
    thresholds_sha            TEXT NOT NULL,
    seats_found               INTEGER,
    discovery_plausible       INTEGER,        -- 1 / 0 / NULL when the artefact did not say
    observations_total        INTEGER,
    observations_unassigned   INTEGER,
    observations_unreadable   INTEGER,
    seats_never_settled       INTEGER,
    frames_with_no_person     INTEGER,
    -- Copied, never retyped (`report/artefact.py::CAVEATS_RU`). The report renders these,
    -- so a caveat that changes in the analyser changes on old pages only if they are
    -- re-rendered — which is correct: the caveat that applied is the one that was written
    -- when the measurement was made.
    caveats_json              TEXT NOT NULL,
    unmeasured_json           TEXT NOT NULL,
    -- The adult's record, stored whole and NEVER aggregated across lessons. See
    -- `teacher_record()`: there is no teacher index in this store and nowhere to put one.
    teacher_json              TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_lesson ON runs(lesson_id);

-- A PLACE is the unit of accumulation: a fixed position in a fixed room used by a fixed
-- class. It is what `seat_id` cannot be (see this module's docstring). Its geometry is
-- ANCHORED on the first lesson that revealed it and never drifts: a running mean would let
-- a slowly-moving camera carry a place across a desk over a term without anything
-- failing, whereas a fixed anchor makes the camera move show up as unmatched seats on a
-- particular date, which is a fact a human can act on.
CREATE TABLE IF NOT EXISTS places (
    place_id       INTEGER PRIMARY KEY,
    room_key       TEXT NOT NULL,
    class_key      TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,          -- stable display number within the room
    label_ru       TEXT NOT NULL,             -- «место 3» / «место взрослого»
    role           TEXT NOT NULL,             -- "pupil" | "adult"
    anchor_x       REAL NOT NULL,
    anchor_y       REAL NOT NULL,
    anchor_scale   REAL NOT NULL,
    first_run_id   TEXT NOT NULL,
    first_seen_at  TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE (room_key, class_key, ordinal)
);

-- One place, for one run. The grain the client's sentence needs: «сколько раз он был
-- активен, поднимал руку отвечал».
CREATE TABLE IF NOT EXISTS seat_lessons (
    run_id                TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seat_id               INTEGER NOT NULL,
    lesson_id             INTEGER NOT NULL REFERENCES lessons(lesson_id),
    -- NULL when this seat could not be matched to a known place. Such a row is stored in
    -- full and excluded from everything longitudinal: the alternative is joining it to the
    -- nearest history, which is the defect this whole table exists to prevent.
    place_id              INTEGER REFERENCES places(place_id),
    place_match           TEXT    NOT NULL,   -- "new" | "matched" | "ambiguous" | "moved"
    place_match_reason_ru TEXT    NOT NULL,
    place_match_distance  REAL,               -- in shoulder widths
    seat_label            TEXT    NOT NULL,   -- the artefact's own per-run label
    role                  TEXT    NOT NULL,
    centre_x              REAL,
    centre_y              REAL,
    scale_px              REAL,
    occupancy             REAL,
    settled               INTEGER,
    -- Coverage and its raw parts travel with every count in this row. Rule 2 of this
    -- project is enforced by the schema and not by the discipline of whoever writes the
    -- next query: there is no counter column here without its denominator alongside.
    observations          INTEGER NOT NULL,
    absent_observations   INTEGER NOT NULL,
    unreadable_observations INTEGER NOT NULL,
    hand_unmeasurable_observations INTEGER NOT NULL,
    observed_seconds      REAL,
    coverage              REAL,
    hand_raises           INTEGER,
    stands                INTEGER,
    board_visits          INTEGER,
    away_episodes         INTEGER,
    head_down_episodes    INTEGER,
    turned_away_episodes  INTEGER,
    -- The index as the ARTEFACT computed it, kept for cross-checking. `weekly.py` does not
    -- read it for a session: it recomputes from the summed ledger with the very same
    -- `metrics/activity.py`, and a test asserts the two agree for a one-recording session.
    activity_available    INTEGER NOT NULL,
    activity_index        REAL,
    activity_reason_ru    TEXT,
    activity_parts_json   TEXT,
    state_observations_json TEXT NOT NULL,
    observed_seconds_by_state_json TEXT NOT NULL,
    -- What the artefact's identity stage said about this seat. EVIDENCE, deliberately not
    -- a key: nothing in this store joins through it. The only thing that produces a name
    -- is `attestations`. (`MEASUREMENTS.md` §4: best cosine 0.30, margin 0.10.)
    artefact_pupil_json   TEXT,
    PRIMARY KEY (run_id, seat_id)
);
CREATE INDEX IF NOT EXISTS idx_seat_lessons_place ON seat_lessons(place_id);
CREATE INDEX IF NOT EXISTS idx_seat_lessons_lesson ON seat_lessons(lesson_id);

-- A human being's signed statement that a place is a named child's place. The ONLY route
-- by which a name enters this store, for the reason `identity/seatmap.py` gives at length.
-- Time-bounded, because classes are re-seated: a statement with no end date silently keeps
-- naming last term's children in this term's lessons.
CREATE TABLE IF NOT EXISTS attestations (
    attestation_id INTEGER PRIMARY KEY,
    place_id       INTEGER NOT NULL REFERENCES places(place_id),
    external_id    TEXT NOT NULL,
    full_name      TEXT,
    -- Who said so and when they said it. NOT NULL because an attestation nobody can trace
    -- to a person is not an attestation, it is a row.
    attested_by    TEXT NOT NULL,
    attested_at    TEXT NOT NULL,
    valid_from     TEXT NOT NULL,
    valid_to       TEXT,
    -- The school's written decision that per-pupil accumulation may happen at all. See
    -- INTEGRATION.md §4 and the promise it re-opens; a NOT NULL column is the cheapest
    -- possible way to keep that promise from being re-opened by accident.
    decision_ref   TEXT NOT NULL,
    note           TEXT,
    recorded_at    TEXT NOT NULL,
    UNIQUE (place_id, external_id, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_attestations_place ON attestations(place_id);

-- The machine-written orientation note for ONE run (`report/orientation.py`). Not a
-- measurement and never treated as one: it holds no quantities by construction, and the
-- guard verdict that established that is stored beside it rather than re-derived later.
--
-- THREE THINGS THIS TABLE'S KEY AND CONSTRAINTS MAKE IMPOSSIBLE, in the order they would
-- otherwise happen:
--
-- 1. **A second note for the same run.** `run_id` is the PRIMARY KEY, so re-generating is
--    an UPSERT and not an INSERT. A cron that runs `cabinet note --all` nightly leaves one
--    note per run for ever instead of a pile of near-identical paragraphs, and nothing
--    downstream ever has to choose between two notes about one hour.
--
-- 2. **A note outliving the run it was written about.** This is the requirement that
--    decided the key. `runs.run_id` is the artefact's own hash of the video plus every
--    setting that could change a number, so a lesson re-analysed under different thresholds
--    is a NEW run row — which simply has no note until one is generated for it. The old
--    note stays attached to the old run, is shown only beside that run, and cannot migrate
--    to the new measurement even though both describe the same hour. Keying on `lesson_id`
--    would have done the opposite: the sentence «на месте 3 ребёнок несколько раз
--    отходил» written under one head-down threshold would appear beside the numbers of
--    another. `ON DELETE CASCADE` closes the same door from the other side: a run that is
--    ever removed takes its note with it, rather than leaving an orphan keyed to a string
--    nobody can resolve.
--
-- 3. **A note surviving a change in the numbers it describes.** `bundle_sha256` is the hash
--    of the exact compacted bundle that was sent to the model. `cabinet/notes.py` recomputes
--    that bundle from these same tables when the page is rendered and compares; a mismatch
--    marks the note stale and the page says so instead of showing it. Belt and braces on top
--    of (2): the run id covers everything the ANALYSER could change, and this covers
--    anything the STORE could — a re-import into a rebuilt database, a future column, a
--    corrected import.
--
-- `guard_passed` is 1 / 0 / NULL, and the NULL is load-bearing: no provider means the guard
-- never ran, which is a different fact from the guard having rejected the text, and a
-- BOOLEAN whose false has two causes is the defect this file keeps being written against.
CREATE TABLE IF NOT EXISTS run_notes (
    run_id               TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    -- Empty string when there is no note. `available = 0` is the field to branch on.
    text                 TEXT NOT NULL,
    available            INTEGER NOT NULL,
    -- What produced (or failed to produce) it: "gemini" | "openai" | "none".
    source               TEXT NOT NULL,
    -- What the operator asked for: "auto" | "gemini" | "openai" | "deterministic".
    backend_requested    TEXT NOT NULL,
    model                TEXT,
    prompt_version       TEXT NOT NULL,
    guard_passed         INTEGER,
    guard_reason_ru      TEXT NOT NULL,
    guard_offending_json TEXT NOT NULL,
    -- Why there is no note, when there is none. Empty when there is one.
    reason_ru            TEXT NOT NULL,
    bundle_sha256        TEXT NOT NULL,
    bundle_bytes         INTEGER NOT NULL,
    generated_at         TEXT NOT NULL
);

-- WITHIN-LESSON DYNAMICS: how a place's index moved across its own lesson.
--
-- **A separate table, and additively, for the reason stated at `SCHEMA_VERSION`.** Every
-- statement here is `IF NOT EXISTS`, so an existing cabinet acquires this on the next
-- `connect()` with no measurement touched. Adding COLUMNS to `seat_lessons` would not
-- work that way -- `CREATE TABLE IF NOT EXISTS` never alters an existing table -- and
-- bumping the schema version would refuse to open every cabinet in the school to deliver
-- a quantity none of them previously had.
--
-- **Why this is the only trend in the cabinet that exists today.** A per-child WEEK-over-
-- week trend needs to know that this seat holds the same child as three weeks ago, which
-- needs an attested seating plan (`metrics/trend.py` refuses without one). Within ONE
-- lesson that problem does not arise: the place is the same place for the whole hour, so
-- «стал ли этот ребёнок менее активен к концу урока» is answerable with nothing extra.
-- It is a statement about a PLACE during one lesson and about nothing else -- never a
-- statement about a child across lessons, and never comparable between two lessons.
--
-- `segments_json` carries the per-segment indices AND their coverage, because the whole
-- reading of this table turns on the pair: an index that fell while coverage fell is not
-- an observation about a pupil, it is an observation about the camera. `change_json`
-- carries the verdict with `visibility_bound_index_points` -- how much of the fall the
-- coverage change alone could explain.
CREATE TABLE IF NOT EXISTS seat_within_lesson (
    run_id             TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seat_id            INTEGER NOT NULL,
    place_id           INTEGER REFERENCES places(place_id) ON DELETE SET NULL,
    -- "lower" | "higher" | "steady" | "unknown". Never a flag, never a ranking.
    direction          TEXT NOT NULL,
    direction_ru       TEXT NOT NULL,
    -- Present only when the verdict was computed; NULL when it was refused. NULL and 0.0
    -- are different answers and must not share a column value.
    delta_index_points REAL,
    slope_per_10min    REAL,
    kendall_p          REAL,
    -- The magnitude below which the segmentation itself, or one event appearing or
    -- disappearing, could account for the change. A delta under this is not a finding.
    floor_index_points REAL,
    -- How much of the change the coverage drop alone could explain, in index points.
    visibility_bound   REAL,
    usable_segments    INTEGER NOT NULL,
    refused_segments   INTEGER NOT NULL,
    reason_ru          TEXT NOT NULL,
    refused_because    TEXT,
    segments_json      TEXT NOT NULL,
    change_json        TEXT NOT NULL,
    PRIMARY KEY (run_id, seat_id)
);
CREATE INDEX IF NOT EXISTS ix_within_place ON seat_within_lesson(place_id);
"""


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImportResult:
    """What one import did, including everything it decided NOT to do.

    `notes` and `dropped_seats` are part of the result rather than diagnostics, on the same
    rule as the artefact's own `uncertainty` block: an import that quietly discarded a seat
    and an import that had none to discard must not look identical afterwards.
    """

    run_id: str
    lesson_id: int
    already_imported: bool
    is_new_lesson: bool
    selected: bool                 # did this run become the lesson's selected run?
    seats_stored: int = 0
    places_created: int = 0
    places_matched: int = 0
    seats_unmatched: int = 0
    notes_ru: list[str] = field(default_factory=list)
    dropped_seats: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "lesson_id": self.lesson_id,
            "already_imported": self.already_imported,
            "is_new_lesson": self.is_new_lesson, "selected": self.selected,
            "seats_stored": self.seats_stored, "places_created": self.places_created,
            "places_matched": self.places_matched,
            "seats_unmatched": self.seats_unmatched,
            "notes_ru": self.notes_ru, "dropped_seats": self.dropped_seats,
        }


# ---------------------------------------------------------------------------
# Opening.
# ---------------------------------------------------------------------------


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a cabinet database.

    `sqlite3` from the standard library, no ORM, no migration framework and no new
    dependency — a school's psychologist gets a single file they can copy to a USB stick,
    and this package keeps its promise of adding nothing heavy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_SQL)

    stored = connection.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if stored is None:
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                           (str(SCHEMA_VERSION),))
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_name', ?)",
                           (SCHEMA_NAME,))
        connection.commit()
    elif int(stored["value"]) != SCHEMA_VERSION:
        raise Refusal(
            "store_schema_version",
            f"база кабинета имеет версию схемы {stored['value']}, а этот код — "
            f"{SCHEMA_VERSION}. Обновление на месте не выполняется: столбец, у которого "
            f"изменился смысл, рядом с прошлогодними значениями — это как раз тот скачок "
            f"в динамике, который потом никто не может объяснить.")
    return connection


# ---------------------------------------------------------------------------
# Reading and refusing.
# ---------------------------------------------------------------------------


def load_artefact(path: str | Path) -> dict[str, Any]:
    """Read one artefact, or raise `Unreadable`. Never returns a partial document."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise Unreadable(f"не удалось прочитать {path}: {error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise Unreadable(f"{path} — не читается как JSON: {error}") from error
    if not isinstance(document, dict):
        raise Unreadable(f"{path} — на верхнем уровне не объект")
    return document


def check(document: dict[str, Any], *, allow_unclocked: bool = False) -> None:
    """Everything that stops this artefact entering the cabinet. Raises `Refusal`.

    Ordered cheapest and most fundamental outward, so the sentence a human reads is the
    earliest true thing rather than a downstream symptom of it.
    """
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise Refusal("no_provenance", "в документе нет блока provenance — это не артефакт "
                                       "classvision.")

    schema = str(provenance.get("schema", ""))
    if not schema.startswith(ACCEPTED_SCHEMA_MAJOR + "."):
        raise Refusal(
            "schema_version",
            f"схема {schema!r} — не {ACCEPTED_SCHEMA_MAJOR}.x. Старшая версия меняет не "
            f"состав полей, а СМЫСЛ существующих; такое значение нельзя класть в тот же "
            f"столбец, где лежит прошлый месяц.")

    if not document.get("run_id"):
        raise Refusal("no_run_id", "в документе нет run_id — идемпотентности не на чем "
                                   "держаться.")

    if not provenance.get("video_sha256"):
        raise Refusal("no_video_hash",
                      "provenance.video_sha256 пуст. Это единственное, что связывает "
                      "строку с записью, которую можно пересмотреть.")

    caveats = document.get("caveats")
    if not isinstance(caveats, list) or not caveats:
        raise Refusal("no_caveats",
                      "в документе нет caveats. Они показываются рядом с каждым итогом; "
                      "документ без них нечем сопроводить.")

    discovery = (provenance.get("room") or {}).get("seat_discovery") or {}
    if discovery.get("plausible") is False:
        raise Refusal(
            "implausible_seats",
            "разбор мест в самом артефакте помечен как недостоверный "
            f"({discovery.get('warning', 'места, вероятно, слиты')}). Такие места нельзя "
            "накапливать: двое детей, слитых в одно место, потом неразличимы.")

    if not document.get("seats"):
        raise Refusal("no_seats", "в артефакте нет ни одного места — накапливать нечего.")

    for seat in document["seats"]:
        activity = (seat.get("metrics") or {}).get("activity") or {}
        if activity.get("available") and not activity.get("parts"):
            raise Refusal(
                "index_without_parts",
                f"место {seat.get('label')} несёт индекс активности без составляющих. "
                f"Индекс показывается только разобранным на части; голое число — "
                f"единственное, что эта схема запрещает.")

    if provenance.get("started_at") in (None, "") and not allow_unclocked:
        raise Refusal(
            "no_wall_clock",
            "provenance.started_at пуст — запись нельзя поместить на календарь и нельзя "
            "сравнить ни с одним другим уроком. Импортировать всё равно: "
            "--allow-unclocked; тогда урок хранится без даты и не попадает ни в одну "
            "недельную сводку.")


# ---------------------------------------------------------------------------
# Importing.
# ---------------------------------------------------------------------------


def import_artefact(
    connection: sqlite3.Connection,
    path: str | Path,
    *,
    room_key: str | None = None,
    class_key: str | None = None,
    allow_unclocked: bool = False,
    allow_overlap: bool = False,
    select: str = "if_first",
) -> ImportResult:
    """Ingest one artefact. Idempotent on `run_id`; never overwrites a measurement.

    `select` decides what happens to `lessons.selected_run_id` when this lesson already has
    a run:

      * `"if_first"` (default) — the first run imported for a lesson is selected and a
        later one is stored but NOT selected. This is rule 6 of the project made
        operational: a re-analysis under different thresholds is a new measurement, and
        which of two measurements a term's history uses is a decision by a person.
      * `"always"` — select this run. The caller is asserting the decision explicitly, and
        it is recorded in the result notes.
      * `"never"` — store only.

    **An import is all or nothing.** Anything that raises after the first INSERT rolls the
    whole thing back, because `import_many` continues past a failure and an uncommitted
    half-lesson would otherwise be committed by the NEXT artefact that happened to succeed
    — a term's data acquiring a lesson that no import ever reported writing.
    """
    try:
        return _import(connection, path, room_key=room_key, class_key=class_key,
                       allow_unclocked=allow_unclocked, allow_overlap=allow_overlap,
                       select=select)
    except Exception:
        connection.rollback()
        raise


def _import(
    connection: sqlite3.Connection,
    path: str | Path,
    *,
    room_key: str | None,
    class_key: str | None,
    allow_unclocked: bool,
    allow_overlap: bool,
    select: str,
) -> ImportResult:
    document = load_artefact(path)
    check(document, allow_unclocked=allow_unclocked)

    provenance = document["provenance"]
    run_id = str(document["run_id"])
    now = datetime.now(UTC).isoformat(timespec="seconds")

    room, room_source = _room_key(provenance, room_key)
    klass, class_source = _class_key(document, class_key)

    existing = connection.execute(
        "SELECT r.lesson_id, l.room_key, l.class_key FROM runs r "
        "JOIN lessons l ON l.lesson_id = r.lesson_id WHERE r.run_id = ?",
        (run_id,)).fetchone()
    if existing is not None:
        if (existing["room_key"], existing["class_key"]) != (room, klass):
            # Idempotency on `run_id` is global, so a re-import under a corrected key
            # writes NOTHING and used to report «уже импортирован» — a true sentence that
            # an operator fixing a typo reads as «принято». The lesson then stays filed
            # under the wrong room for ever, and two rooms that should accumulate together
            # never do. Silence is the whole problem, so this is a refusal with a code.
            raise Refusal(
                "already_imported_elsewhere",
                f"прогон {run_id} уже лежит в кабинете под комнатой "
                f"{existing['room_key']!r} и классом {existing['class_key']!r}, а сейчас "
                f"запрошены {room!r} / {klass!r}. Повторный импорт НИЧЕГО не меняет: "
                f"ключи комнаты и класса задаются при первой загрузке. Если прежние ключи "
                f"неверны, урок нужно завести заново в чистой базе — переклеивать их на "
                f"месте эта команда не умеет, потому что вместе с ними пришлось бы "
                f"переносить и историю мест.")
        return ImportResult(
            run_id=run_id, lesson_id=int(existing["lesson_id"]), already_imported=True,
            is_new_lesson=False, selected=False,
            notes_ru=[f"этот прогон ({run_id}) уже импортирован в комнату "
                      f"{existing['room_key']} / класс {existing['class_key']} — ничего "
                      f"не записано."])

    lesson_id, is_new_lesson, lesson_notes = _upsert_lesson(
        connection, document, room=room, room_source=room_source,
        klass=klass, class_source=class_source, allow_overlap=allow_overlap, now=now)

    result = ImportResult(run_id=run_id, lesson_id=lesson_id, already_imported=False,
                          is_new_lesson=is_new_lesson, selected=False)
    result.notes_ru.extend(lesson_notes)

    window_wall = (document.get("lesson") or {}).get("window_wall") or [None, None]
    uncertainty = document.get("uncertainty") or {}
    thresholds = provenance.get("thresholds") or {}
    connection.execute(
        """INSERT INTO runs (run_id, lesson_id, schema, artefact_path, artefact_sha256,
               imported_at, analysed_at, sample_fps, window_start_seconds,
               window_end_seconds, window_wall_start, window_wall_end, analysed_frames,
               model_json, thresholds_json, thresholds_sha, seats_found,
               discovery_plausible, observations_total, observations_unassigned,
               observations_unreadable, seats_never_settled, frames_with_no_person,
               caveats_json, unmeasured_json, teacher_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id, lesson_id, provenance.get("schema"), str(path),
            _file_sha256(path), now, provenance.get("analysed_at"),
            provenance.get("sample_fps"), provenance.get("window_start_seconds"),
            provenance.get("window_end_seconds"),
            window_wall[0] if len(window_wall) > 0 else None,
            window_wall[1] if len(window_wall) > 1 else None,
            provenance.get("analysed_frames"),
            _dump(provenance.get("model")), _dump(thresholds), _settings_sha(thresholds),
            ((provenance.get("room") or {}).get("seat_discovery") or {}).get("seats_found"),
            _tri(((provenance.get("room") or {}).get("seat_discovery") or {}).get("plausible")),
            uncertainty.get("observations_total"),
            uncertainty.get("observations_unassigned"),
            uncertainty.get("observations_unreadable"),
            uncertainty.get("seats_never_settled"),
            uncertainty.get("frames_with_no_person"),
            _dump(document.get("caveats")),
            _dump((document.get("lesson") or {}).get("unmeasured")),
            _dump(document.get("teacher")),
        ))

    _store_seats(connection, document, result, room=room, klass=klass, run_id=run_id,
                 lesson_id=lesson_id, now=now)

    current = connection.execute("SELECT selected_run_id FROM lessons WHERE lesson_id = ?",
                                 (lesson_id,)).fetchone()["selected_run_id"]
    if select == "always" or (select == "if_first" and current is None):
        connection.execute("UPDATE lessons SET selected_run_id = ? WHERE lesson_id = ?",
                           (run_id, lesson_id))
        result.selected = True
        if select == "always" and current not in (None, run_id):
            result.notes_ru.append(
                f"прогон {current} был выбранным для этого урока и заменён на {run_id} по "
                f"явному указанию оператора. Прежний прогон НЕ удалён; сравнить их можно "
                f"командой `cabinet weekly`.")
    elif current is not None and current != run_id:
        result.notes_ru.append(
            f"урок уже измерен прогоном {current}; этот прогон ({run_id}) сохранён как "
            f"ОТДЕЛЬНОЕ измерение и в сводки не попадает. Так и задумано: другие пороги — "
            f"другое измерение, а не уточнение прежнего. Переключить: "
            f"`cabinet select-run {run_id}`.")

    connection.commit()
    return result


def select_run(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    """Make `run_id` the run its lesson's aggregates use. The only way that pointer moves.

    A separate, named act, because moving it changes every weekly number computed over that
    lesson. An import cannot do it by accident and a script cannot do it as a side effect.
    """
    row = connection.execute(
        "SELECT lesson_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise Refusal("unknown_run", f"прогона {run_id} нет в базе.")
    lesson_id = int(row["lesson_id"])
    previous = connection.execute("SELECT selected_run_id FROM lessons WHERE lesson_id = ?",
                                  (lesson_id,)).fetchone()["selected_run_id"]
    connection.execute("UPDATE lessons SET selected_run_id = ? WHERE lesson_id = ?",
                       (run_id, lesson_id))
    connection.commit()
    return {"lesson_id": lesson_id, "previous_run_id": previous, "run_id": run_id}


def attest(connection: sqlite3.Connection, *, place_id: int, external_id: str,
           attested_by: str, attested_at: str, valid_from: str, decision_ref: str,
           full_name: str | None = None, valid_to: str | None = None,
           note: str | None = None) -> int:
    """Record one human statement that a place is a named child's place.

    Every argument without a default is required for a reason stated in the schema: who
    said it, when they said it, from when it holds, and which school decision permits
    per-pupil accumulation at all. There is no code path in this package that writes this
    table from a face, a tracker or a heuristic.
    """
    if connection.execute("SELECT 1 FROM places WHERE place_id = ?",
                          (place_id,)).fetchone() is None:
        raise Refusal("unknown_place", f"места {place_id} нет в базе.")
    for value, name in ((external_id, "external_id"), (attested_by, "attested_by"),
                        (attested_at, "attested_at"), (valid_from, "valid_from"),
                        (decision_ref, "decision_ref")):
        if not str(value).strip():
            raise Refusal("incomplete_attestation",
                          f"поле {name} пусто. Утверждение без него — не утверждение, "
                          f"а строка в таблице.")
    cursor = connection.execute(
        """INSERT OR IGNORE INTO attestations (place_id, external_id, full_name,
               attested_by, attested_at, valid_from, valid_to, decision_ref, note,
               recorded_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (place_id, external_id, full_name, attested_by, attested_at, valid_from,
         valid_to, decision_ref, note, datetime.now(UTC).isoformat(timespec="seconds")))
    connection.commit()
    return int(cursor.lastrowid or 0)


# ---------------------------------------------------------------------------
# Lessons.
# ---------------------------------------------------------------------------


def _upsert_lesson(connection: sqlite3.Connection, document: dict[str, Any], *,
                   room: str, room_source: str, klass: str, class_source: str,
                   allow_overlap: bool, now: str) -> tuple[int, bool, list[str]]:
    provenance = document["provenance"]
    video_sha = str(provenance["video_sha256"])

    row = connection.execute(
        "SELECT * FROM lessons WHERE room_key = ? AND class_key = ? AND video_sha256 = ?",
        (room, klass, video_sha)).fetchone()
    if row is not None:
        return int(row["lesson_id"]), False, []

    started = _parse_local(provenance.get("started_at"))
    duration = _float(provenance.get("duration_seconds"))
    ended = started + timedelta(seconds=duration) if started and duration else None
    notes: list[str] = []
    overlap_note: str | None = None

    # -- DVR continuation, asked FIRST. A recorder that rolls a file produces a pair whose
    # -- ends touch, and whose second-quantised clocks can make them look like a one-second
    # -- overlap (see `CONTINUATION_TOLERANCE_SECONDS`). That is a benign, specific kind of
    # -- touching and it must be recognised before the general overlap rule refuses it --
    # -- otherwise the single most likely real-world case, the D14 pair, needs an operator
    # -- override every time and arrives carrying a note saying it was a duplicate.
    continues = None
    previous = _immediately_before(connection, room, klass, started) if started else None
    if previous is not None:
        continues = int(previous["lesson_id"])
        gap = float(previous["gap"])
        seam = (f"начинается через {gap:.0f} с после конца" if gap >= 0 else
                f"начинается на {abs(gap):.0f} с раньше расчётного конца (часы известны "
                f"с точностью до секунды)")
        notes.append(
            f"запись {seam} урока №{continues} в той же комнате — считается "
            f"ПРОДОЛЖЕНИЕМ того же занятия, а не отдельным уроком. В недельных итогах "
            f"эти два файла — одно занятие.")

    # -- ...and the same question asked FORWARDS, because a directory is imported in
    # -- whatever order a shell glob produced and an operator loads files one at a time.
    # -- Looking only backwards made the whole continuation rule depend on import order:
    # -- loading the D14 pair long-file-first produced TWO sessions, silently, with no
    # -- refusal and no note — the exact failure this check exists to prevent, arrived at
    # -- by typing two filenames the other way round. Measured, not imagined: see
    # -- `test_a_split_recording_is_one_session_whichever_file_arrives_first`.
    continued_by = (_immediately_after(connection, room, klass, ended)
                    if ended is not None else None)
    if continued_by is not None:
        notes.append(
            f"уже загруженная запись №{continued_by['lesson_id']} начинается сразу после "
            f"конца этой ({abs(float(continued_by['gap'])):.0f} с) — она помечена как "
            f"ПРОДОЛЖЕНИЕ этой записи. В недельных итогах эти два файла — одно занятие.")

    # -- Overlap. Two recordings of one room whose wall clocks intersect cannot both be
    # -- whole lessons, and counting both doubles one hour in every weekly total. The
    # -- content hashes differ (a slice of a file is a different file), so this is the only
    # -- place the duplication is visible at all.
    if started and ended:
        ignore = {continues} if continues is not None else set()
        if continued_by is not None:
            ignore.add(int(continued_by["lesson_id"]))
        clash = _overlapping(connection, room, klass, started, ended, ignore=ignore)
        if clash is not None and not allow_overlap:
            raise Refusal(
                "overlapping_lesson",
                f"эта запись ({started:%Y-%m-%d %H:%M}–{ended:%H:%M}) пересекается по "
                f"времени с уже загруженным уроком №{clash['lesson_id']} "
                f"({clash['started_at']}–{clash['ended_at']}) в той же комнате. Два "
                f"пересекающихся файла — это один и тот же час, посчитанный дважды во "
                f"всех недельных итогах. Если это всё-таки разные уроки, повторите с "
                f"--allow-overlap: пометка останется в базе и будет напечатана в отчёте.")
        if clash is not None:
            # Held in its own variable rather than fished back out of `notes` by prefix.
            # `notes` already carries the continuation sentence by this point, and a
            # `notes[0].startswith(...)` test would have silently stored NULL for exactly
            # the lesson that is both a continuation and a forced overlap — losing the
            # override on the one row where it matters most.
            overlap_note = (f"импортирован поверх пересечения по времени с уроком "
                            f"№{clash['lesson_id']} — подтверждено оператором.")
            notes.append(overlap_note)

    iso = started.date().isocalendar() if started else None
    cursor = connection.execute(
        """INSERT INTO lessons (room_key, class_key, video_sha256, video_path, started_at,
               ended_at, date_local, iso_year, iso_week, iso_weekday, duration_seconds,
               clock_source, room_key_source, class_key_source, continues_lesson_id,
               overlap_note, selected_run_id, imported_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
        (room, klass, video_sha, str(provenance.get("video_path") or ""),
         started.isoformat() if started else None,
         ended.isoformat() if ended else None,
         started.date().isoformat() if started else None,
         iso.year if iso else None, iso.week if iso else None, iso.weekday if iso else None,
         duration, str(provenance.get("clock_source") or "unknown"),
         room_source, class_source, continues, overlap_note, now))

    lesson_id = int(cursor.lastrowid or 0)
    if continued_by is not None:
        # The already-stored later half now points BACK at this one, so the chain is the
        # same object whichever file arrived first. Only ever written onto a lesson that
        # heads its own chain (`_immediately_after` filters on that), so no existing link
        # is overwritten and no cycle can be built: the target starts strictly after this
        # lesson's start, and a lesson cannot precede itself.
        connection.execute(
            "UPDATE lessons SET continues_lesson_id = ? WHERE lesson_id = ?",
            (lesson_id, int(continued_by["lesson_id"])))

    if started is None:
        notes.append("у записи нет читаемого времени — урок сохранён, но исключён из всех "
                     "недельных сводок: два урока без дат нельзя расположить по порядку.")
    return lesson_id, True, notes


def _overlapping(connection: sqlite3.Connection, room: str, klass: str,
                 started: datetime, ended: datetime,
                 ignore: set[int] | None = None) -> dict[str, Any] | None:
    """The already-stored lesson this recording collides with, or nothing.

    `ignore` holds the lessons this one was just recognised as a DVR seam with — the half
    it continues and the half that continues it. A seam reads as a one-second overlap on
    second-quantised clocks, and refusing it would put an operator override on the most
    ordinary case there is. Both directions are excluded, not just the backward one:
    otherwise the same pair of files refuses or does not depending on which was typed
    first, which is a property of the operator rather than of the recordings.
    """
    ignore = ignore or set()
    for row in connection.execute(
            "SELECT lesson_id, started_at, ended_at FROM lessons "
            "WHERE room_key = ? AND class_key = ? AND started_at IS NOT NULL",
            (room, klass)):
        if int(row["lesson_id"]) in ignore:
            continue
        other_start = _parse_local(row["started_at"])
        other_end = _parse_local(row["ended_at"])
        if other_start is None or other_end is None:
            continue
        overlap = (min(ended, other_end) - max(started, other_start)).total_seconds()
        if overlap > OVERLAP_TOLERANCE_SECONDS:
            return {"lesson_id": row["lesson_id"], "started_at": row["started_at"],
                    "ended_at": row["ended_at"], "overlap_seconds": overlap}
    return None


def _immediately_before(connection: sqlite3.Connection, room: str, klass: str,
                        started: datetime) -> dict[str, Any] | None:
    """The stored recording this one continues, if any — the seam of a DVR file roll.

    The window is `[-CONTINUATION_TOLERANCE_SECONDS, CONTINUATION_GAP_SECONDS]` and the
    negative half is measured, not defensive: on the D14 pair the second file's start
    (10:31:36) falls one second BEFORE the first file's computed end (10:17:59 + 818.00 s
    = 10:31:37), because both clocks are known only to the second. A `[0, gap]` window
    misses that pair, which is the pair this check exists for.
    """
    best: dict[str, Any] | None = None
    for row in connection.execute(
            "SELECT lesson_id, ended_at FROM lessons "
            "WHERE room_key = ? AND class_key = ? AND ended_at IS NOT NULL",
            (room, klass)):
        other_end = _parse_local(row["ended_at"])
        if other_end is None:
            continue
        gap = (started - other_end).total_seconds()
        if (-CONTINUATION_TOLERANCE_SECONDS <= gap <= CONTINUATION_GAP_SECONDS
                and (best is None or abs(gap) < abs(best["gap"]))):
            best = {"lesson_id": row["lesson_id"], "gap": gap}
    return best


def _immediately_after(connection: sqlite3.Connection, room: str, klass: str,
                       ended: datetime) -> dict[str, Any] | None:
    """The stored recording that continues THIS one, if the halves arrived out of order.

    The mirror image of `_immediately_before`, and it is not symmetry for its own sake. A
    back-fill imports whatever `ls out/*.json` produced, and an operator drags two files
    onto a command line in whatever order they read them. With only the backward check, the
    D14 pair loaded long-file-first stored two independent lessons: «занятий 2» for an hour
    that had one, every per-lesson figure of it halved, and — unlike the overlap case —
    nothing refused and nothing warned, because a lesson followed by another lesson is
    exactly what a normal school day looks like.

    Only lessons that currently head their own chain are considered. A lesson that already
    continues something else has its predecessor, and quietly re-pointing it would rewrite a
    chain that a previous import already established.
    """
    best: dict[str, Any] | None = None
    for row in connection.execute(
            "SELECT lesson_id, started_at FROM lessons "
            "WHERE room_key = ? AND class_key = ? AND started_at IS NOT NULL "
            "AND continues_lesson_id IS NULL", (room, klass)):
        other_start = _parse_local(row["started_at"])
        if other_start is None:
            continue
        gap = (other_start - ended).total_seconds()
        if (-CONTINUATION_TOLERANCE_SECONDS <= gap <= CONTINUATION_GAP_SECONDS
                and (best is None or abs(gap) < abs(best["gap"]))):
            best = {"lesson_id": row["lesson_id"], "gap": gap}
    return best


# ---------------------------------------------------------------------------
# Seats and places.
# ---------------------------------------------------------------------------


def _store_seats(connection: sqlite3.Connection, document: dict[str, Any],
                 result: ImportResult, *, room: str, klass: str, run_id: str,
                 lesson_id: int, now: str) -> None:
    known = _places(connection, room, klass)
    # This lesson's seats are matched one at a time, and a place already claimed by an
    # earlier seat of THIS run is removed from the pool. Without that, two adjacent seats
    # in a re-arranged room can both match the same place and one child's lesson is written
    # twice under another's history.
    claimed: set[int] = set()

    first_seen = (document.get("lesson") or {}).get("window_wall") or [None]
    records = list(document.get("seats") or [])
    teacher = document.get("teacher")
    if isinstance(teacher, dict) and teacher.get("centre"):
        records.append(_teacher_as_seat(teacher))
    elif isinstance(teacher, dict):
        # THE ADULT EXISTS AND WE DO NOT KNOW WHERE HE WAS — and that is a different fact
        # from "there was no adult", which is what dropping the record silently made the
        # cabinet say. Measured on the real directory: `clip_15min.named.json` carries a
        # fully measured adult (`metrics.available` true, 96 % at his desk, 11 changes of
        # position) whose top-level `centre`/`scale_px` are null, and the import printed
        # «мест 8 … без привязки 0» beside two other runs of the SAME lesson that printed
        # 9. A room with a hole in it, reported as a clean import. On `D14_session` the
        # same branch hid «взрослый не опознан ни по зоне, ни по размеру», so the class
        # page showed six pupils and no statement that anyone had looked for an adult.
        #
        # No geometry is invented from `identification.evidence.centre`: that is where the
        # size-ratio test happened to see him once, not the position his record is about,
        # and a place anchored on it would be a guess wearing a column name — the defect
        # this whole module is organised against. So the omission is REPORTED instead.
        metrics = teacher.get("metrics") or {}
        why = str(metrics.get("reason") or "").strip()
        result.dropped_seats.append({
            "seat_label": "взрослый",
            "why_ru": (
                "в артефакте есть запись о взрослом, но нет его положения в кадре "
                "(centre/scale_px пусты)"
                + (f": {why}" if why else "")
                + ". Место взрослого по этому прогону не заводится и не сопоставляется — "
                  "это ОТСУТСТВИЕ ИЗМЕРЕНИЯ, а не отсутствие взрослого в классе. "
                  "Сам разбор взрослого сохранён вместе с прогоном и показан в карточке "
                  "урока."),
        })

    for seat in records:
        centre = seat.get("centre")
        scale = _float(seat.get("scale_px"))
        role = str(seat.get("role") or "pupil")
        if not centre or scale in (None, 0.0):
            result.dropped_seats.append({
                "seat_label": seat.get("label"),
                "why_ru": "у места нет геометрии (centre/scale_px) — сопоставить его с "
                          "местом в комнате нечем.",
            })
            continue

        point = (float(centre[0]), float(centre[1]))
        place_id, match, reason, distance = _match_place(
            point, float(scale), [p for p in known if p["place_id"] not in claimed], role)

        if place_id is None and match == "new":
            place_id = _create_place(connection, room=room, klass=klass, point=point,
                                     scale=float(scale), role=role, run_id=run_id,
                                     first_seen=first_seen[0], now=now)
            known = _places(connection, room, klass)
            result.places_created += 1
        elif place_id is not None:
            result.places_matched += 1
        else:
            result.seats_unmatched += 1
            result.dropped_seats.append({"seat_label": seat.get("label"), "why_ru": reason})

        if place_id is not None:
            claimed.add(place_id)

        ledger = seat.get("ledger") or {}
        counts = ledger.get("counts") or {}
        activity = (seat.get("metrics") or {}).get("activity") or {}
        connection.execute(
            """INSERT INTO seat_lessons (run_id, seat_id, lesson_id, place_id, place_match,
                   place_match_reason_ru, place_match_distance, seat_label, role, centre_x,
                   centre_y, scale_px, occupancy, settled, observations,
                   absent_observations, unreadable_observations,
                   hand_unmeasurable_observations, observed_seconds, coverage, hand_raises,
                   stands, board_visits, away_episodes, head_down_episodes,
                   turned_away_episodes, activity_available, activity_index,
                   activity_reason_ru, activity_parts_json, state_observations_json,
                   observed_seconds_by_state_json, artefact_pupil_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, int(seat["seat_id"]), lesson_id, place_id, match, reason, distance,
             str(seat.get("label") or ""), role, point[0], point[1], float(scale),
             _float(seat.get("occupancy")), _tri(ledger.get("settled")),
             int(ledger.get("observations") or 0),
             int(ledger.get("absent_observations") or 0),
             int(ledger.get("unreadable_observations") or 0),
             int(ledger.get("hand_unmeasurable_observations") or 0),
             _float(ledger.get("observed_seconds")), _float(ledger.get("coverage")),
             counts.get("hand_raises"), counts.get("stands"), counts.get("board_visits"),
             counts.get("away_episodes"), counts.get("head_down_episodes"),
             counts.get("turned_away_episodes"),
             1 if activity.get("available") else 0, _float(activity.get("index")),
             activity.get("reason"), _dump(activity.get("parts")),
             _dump(ledger.get("state_observations") or {}),
             _dump(ledger.get("observed_seconds_by_state") or {}),
             _dump(seat.get("pupil"))))
        _store_within_lesson(connection, run_id, seat, place_id)
        result.seats_stored += 1


def _store_within_lesson(connection: sqlite3.Connection, run_id: str,
                         seat: dict[str, Any], place_id: int | None) -> None:
    """Persist the per-segment dynamics, when the artefact carries them.

    **Absent for a reason, not by accident, on artefacts written before this existed.** The
    key is optional and its absence stores nothing rather than a row of nulls: a place whose
    lesson was analysed by an older build has no within-lesson verdict, and «нет записи»
    must not render as «steady». The cabinet asks this table for a row and treats a missing
    one as "this measurement was not taken", which is what it is.
    """
    within = (seat.get("metrics") or {}).get("within_lesson")
    if not isinstance(within, dict) or not within.get("segments"):
        return

    change = within.get("change") or {}
    segments = within.get("segments") or []
    usable = sum(1 for s in segments if ((s.get("activity") or {}).get("available")))
    connection.execute(
        """INSERT OR REPLACE INTO seat_within_lesson (run_id, seat_id, place_id, direction,
               direction_ru, delta_index_points, slope_per_10min, kendall_p,
               floor_index_points, visibility_bound, usable_segments, refused_segments,
               reason_ru, refused_because, segments_json, change_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, int(seat["seat_id"]), place_id,
         str(change.get("direction") or "unknown"),
         str(change.get("direction_ru") or ""),
         _float(change.get("delta_index_points")),
         _float(change.get("slope_index_points_per_10_minutes")),
         _float(change.get("kendall_p")),
         _float(change.get("floor_index_points")),
         _float(change.get("visibility_bound_index_points")),
         usable, len(segments) - usable,
         str(change.get("reason") or ""),
         change.get("refused_because"),
         _dump(segments), _dump(change)))


def within_lesson_for_place(connection: sqlite3.Connection,
                            place_id: int) -> list[dict[str, Any]]:
    """This place's within-lesson dynamics, one row per lesson, oldest first.

    Joined to the lesson so the caller gets the DATE without a second query, and restricted
    to the run each lesson is currently measured BY (`lessons.selected_run_id`) — otherwise
    a lesson re-analysed under different thresholds would contribute two contradictory
    verdicts to the same page, which is precisely the ambiguity `select-run` exists to
    resolve.

    Undated lessons sort last rather than being dropped: a recording whose clock could not
    be read still has a within-lesson verdict, and that verdict is valid — it is a statement
    about one hour, and it needs no calendar to be true.
    """
    rows = connection.execute(
        """SELECT w.*, l.lesson_id, l.date_local, l.iso_year, l.iso_week
             FROM seat_within_lesson AS w
             JOIN lessons AS l ON l.selected_run_id = w.run_id
            WHERE w.place_id = ?
            ORDER BY (l.date_local IS NULL), l.date_local, l.lesson_id""",
        (int(place_id),)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        data["segments"] = json.loads(data.pop("segments_json") or "[]")
        data["change"] = json.loads(data.pop("change_json") or "{}")
        out.append(data)
    return out


def within_lesson_for_run(connection: sqlite3.Connection,
                          run_id: str) -> dict[int, dict[str, Any]]:
    """Every place's within-lesson dynamics for ONE measurement, keyed by seat_id."""
    out: dict[int, dict[str, Any]] = {}
    for row in connection.execute(
            "SELECT * FROM seat_within_lesson WHERE run_id = ? ORDER BY seat_id",
            (run_id,)):
        data = dict(row)
        data["segments"] = json.loads(data.pop("segments_json") or "[]")
        data["change"] = json.loads(data.pop("change_json") or "{}")
        out[int(data["seat_id"])] = data
    return out


def within_lesson_for(connection: sqlite3.Connection, run_id: str,
                      seat_id: int) -> dict[str, Any] | None:
    """One place's within-lesson dynamics for one measurement, or None if not taken."""
    row = connection.execute(
        "SELECT * FROM seat_within_lesson WHERE run_id = ? AND seat_id = ?",
        (run_id, int(seat_id))).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["segments"] = json.loads(data.pop("segments_json") or "[]")
    data["change"] = json.loads(data.pop("change_json") or "{}")
    return data


def _teacher_as_seat(teacher: dict[str, Any]) -> dict[str, Any]:
    """The adult's record, shaped like a seat so that «все места» really means all of them.

    `report/artefact.py::TeacherRecord` grew `centre`/`scale_px` precisely because a
    consumer that iterated `seats` silently lost the adult, and the verification overlay
    drew their circle at the frame origin. A store that repeated that omission would show a
    room with a hole in it. The adult is stored with `role = "adult"`, gets NO activity
    index (the artefact does not compute one for them) and is never aggregated across
    lessons — see `teacher_record()`.
    """
    centre = teacher.get("centre") or (0.0, 0.0)
    return {
        # `seat_id` 0 when the artefact gave the adult none. It is a KEY, not a
        # measurement: `seat_lessons`'s primary key is (run_id, seat_id) and
        # `room/seats.py` numbers pupil seats from 1, so 0 is the one value that cannot
        # collide with a pupil. Nothing reads it as a position.
        "seat_id": int(teacher.get("seat_id") or 0),
        "label": "adult",
        "centre": centre,
        "scale_px": teacher.get("scale_px"),
        "occupancy": None,
        "role": "adult",
        "pupil": (teacher.get("identification") or {}).get("identity"),
        "ledger": teacher.get("ledger") or {},
        "metrics": {},
    }


def _places(connection: sqlite3.Connection, room: str, klass: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        "SELECT * FROM places WHERE room_key = ? AND class_key = ? ORDER BY ordinal",
        (room, klass))]


def _match_place(point: tuple[float, float], scale: float, known: list[dict[str, Any]],
                 role: str) -> tuple[int | None, str, str, float | None]:
    """Which known place is this seat — by geometry, with a gate AND a margin.

    Distance is normalised by the MEAN of the two shoulder widths, the way
    `room/seats.py` normalises everywhere: the room is viewed at an angle, so a fixed pixel
    gate spans two back-row desks while failing to cover one front-row pupil leaning
    forward. The three unmatched outcomes are named separately because they need different
    actions from a human — nobody has seen this place before, the camera or the furniture
    moved, or the room was re-arranged so that two places now overlap.

    The adult is matched only against adult places and pupils only against pupil places.
    The adult's «место» is where a person stands and sits, not a chair; letting it compete
    with a desk would attach a lesson of teaching to a child's history.
    """
    candidates = [p for p in known if p["role"] == role]
    if not candidates:
        return None, "new", "место в этой комнате встречено впервые.", None

    ranked = sorted(
        ((math.dist(point, (p["anchor_x"], p["anchor_y"]))
          / max((scale + p["anchor_scale"]) / 2.0, 1e-6), p) for p in candidates),
        key=lambda pair: pair[0])
    best_distance, best = ranked[0]

    if best_distance > PLACE_MATCH_GATE_SCALES:
        # Beyond the gate, this is either a genuinely new place (a desk nobody used before)
        # or a moved camera. They are indistinguishable from one lesson, so a new place is
        # created and the distance is recorded: a run that creates a new place for EVERY
        # seat is a moved camera, and it is visible as exactly that on the report.
        return None, "new", (
            f"ближайшее известное место — {best['label_ru']} на расстоянии "
            f"{best_distance:.2f} ширины плеч, за порогом {PLACE_MATCH_GATE_SCALES}: "
            f"считается новым местом."), best_distance

    if len(ranked) > 1:
        runner_up = ranked[1][0]
        if runner_up < best_distance * PLACE_MATCH_MARGIN:
            return None, "ambiguous", (
                f"два известных места одинаково близки ({best['label_ru']} — "
                f"{best_distance:.2f}, {ranked[1][1]['label_ru']} — {runner_up:.2f} "
                f"ширины плеч). Место оставлено без привязки: угадывать, чья это история, "
                f"нельзя."), best_distance

    return int(best["place_id"]), "matched", (
        f"{best['label_ru']}, расстояние {best_distance:.2f} ширины плеч."), best_distance


def _create_place(connection: sqlite3.Connection, *, room: str, klass: str,
                  point: tuple[float, float], scale: float, role: str, run_id: str,
                  first_seen: Any, now: str) -> int:
    ordinal = int(connection.execute(
        "SELECT COALESCE(MAX(ordinal), 0) + 1 AS n FROM places "
        "WHERE room_key = ? AND class_key = ?", (room, klass)).fetchone()["n"])
    label = "место взрослого" if role == "adult" else f"место {ordinal}"
    cursor = connection.execute(
        """INSERT INTO places (room_key, class_key, ordinal, label_ru, role, anchor_x,
               anchor_y, anchor_scale, first_run_id, first_seen_at, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (room, klass, ordinal, label, role, point[0], point[1], scale, run_id,
         str(first_seen) if first_seen else None, now))
    return int(cursor.lastrowid or 0)


# ---------------------------------------------------------------------------
# Reading back.
# ---------------------------------------------------------------------------


def rooms(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every (room, class) pair in the store, with what it holds."""
    return [dict(row) for row in connection.execute(
        """SELECT room_key, class_key,
                  COUNT(*) AS lessons,
                  SUM(CASE WHEN started_at IS NULL THEN 1 ELSE 0 END) AS undated,
                  MIN(date_local) AS first_date, MAX(date_local) AS last_date
           FROM lessons GROUP BY room_key, class_key ORDER BY room_key, class_key""")]


def lessons(connection: sqlite3.Connection, *, room_key: str | None = None,
            class_key: str | None = None) -> list[dict[str, Any]]:
    sql = ("SELECT l.*, r.thresholds_sha, r.sample_fps, r.seats_found, r.analysed_frames, "
           "       r.schema AS run_schema "
           "FROM lessons l LEFT JOIN runs r ON r.run_id = l.selected_run_id")
    where, args = _scope(room_key, class_key, prefix="l.")
    return [dict(row) for row in connection.execute(
        f"{sql}{where} ORDER BY l.started_at IS NULL, l.started_at", args)]


def runs_for_lesson(connection: sqlite3.Connection, lesson_id: int) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        "SELECT * FROM runs WHERE lesson_id = ? ORDER BY imported_at", (lesson_id,))]


def places(connection: sqlite3.Connection, *, room_key: str | None = None,
           class_key: str | None = None) -> list[dict[str, Any]]:
    where, args = _scope(room_key, class_key)
    return [dict(row) for row in connection.execute(
        f"SELECT * FROM places{where} ORDER BY room_key, class_key, ordinal", args)]


def attestations_for(connection: sqlite3.Connection, place_id: int) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        "SELECT * FROM attestations WHERE place_id = ? ORDER BY valid_from", (place_id,))]


def attestation_in_force(connection: sqlite3.Connection, place_id: int,
                         on_date: str | None) -> dict[str, Any] | None:
    """The statement that held on the day the lesson was RECORDED, or nothing.

    `None` for an undated lesson is not an oversight: a lesson we cannot place on a
    calendar cannot be placed on either side of a re-seating, and the honest answer to
    «какой план рассадки действовал» is then «неизвестно», not «сегодняшний».
    """
    if not on_date:
        return None
    for row in attestations_for(connection, place_id):
        if row["valid_from"] <= on_date and (row["valid_to"] is None
                                             or on_date <= row["valid_to"]):
            return row
    return None


def seat_rows(connection: sqlite3.Connection, *, room_key: str | None = None,
              class_key: str | None = None) -> list[dict[str, Any]]:
    """Every stored seat-lesson of the SELECTED runs only, newest last.

    Non-selected runs are deliberately invisible here. They exist, they are listed by
    `runs_for_lesson`, and they are not part of any aggregate — because two measurements of
    one hour summed together is one hour counted twice.
    """
    where, args = _scope(room_key, class_key, prefix="l.")
    return [dict(row) for row in connection.execute(
        f"""SELECT s.*, l.lesson_id AS lid, l.room_key, l.class_key, l.date_local,
                   l.started_at, l.ended_at, l.iso_year, l.iso_week, l.iso_weekday,
                   l.continues_lesson_id, l.clock_source, l.overlap_note,
                   l.duration_seconds
            FROM seat_lessons s
            JOIN lessons l ON l.lesson_id = s.lesson_id
            WHERE s.run_id = l.selected_run_id{where.replace(' WHERE ', ' AND ', 1)}
            ORDER BY l.started_at IS NULL, l.started_at, s.place_id""", args)]


def adults_without_place(connection: sqlite3.Connection, *, room_key: str | None = None,
                         class_key: str | None = None) -> list[dict[str, Any]]:
    """Lessons whose artefact HAS an adult record but never gave it a position.

    Derived rather than stored, from two facts the store already holds: `runs.teacher_json`
    is present, and the run produced no `seat_lessons` row with `role = 'adult'`. That is
    exactly the state `_store_seats` now reports at import time, and the report needs it
    too, because an import message scrolls past once while the page is read all term.

    Rule 4 of this project: anything unmeasurable is listed explicitly. A class page that
    simply had no adult on it could mean either «взрослого в кадре не было» or «мы не
    смогли его найти», and those send a reader to two different places.
    """
    where, args = _scope(room_key, class_key, prefix="l.")
    rows = connection.execute(
        f"""SELECT l.lesson_id, l.date_local, l.started_at, r.run_id, r.teacher_json
            FROM lessons l JOIN runs r ON r.run_id = l.selected_run_id
            WHERE r.teacher_json IS NOT NULL AND r.teacher_json NOT IN ('null', '')
              AND NOT EXISTS (SELECT 1 FROM seat_lessons s
                              WHERE s.run_id = r.run_id AND s.role = 'adult')
            {where.replace(' WHERE ', ' AND ', 1)}
            ORDER BY l.started_at IS NULL, l.started_at""", args)
    out: list[dict[str, Any]] = []
    for row in rows:
        record = json.loads(row["teacher_json"]) or {}
        metrics = record.get("metrics") or {}
        out.append({
            "lesson_id": row["lesson_id"], "date_local": row["date_local"],
            "run_id": row["run_id"],
            # The artefact's OWN sentence about why, never a sentence invented here. When
            # it measured the adult happily and merely failed to publish his coordinates,
            # `reason` is empty and the page says that instead of inventing a cause.
            "reason_ru": str(metrics.get("reason") or "").strip(),
            "measured": bool(metrics.get("available")),
        })
    return out


def save_note(connection: sqlite3.Connection, run_id: str, *, text: str, available: bool,
              source: str, backend_requested: str, model: str | None,
              prompt_version: str, guard_passed: bool | None, guard_reason_ru: str,
              guard_offending: Iterable[str], reason_ru: str, bundle_sha256: str,
              bundle_bytes: int) -> dict[str, Any]:
    """Write (or replace) the orientation note of one run. Returns what is now stored.

    `INSERT ... ON CONFLICT(run_id) DO UPDATE` rather than a delete-then-insert, so that
    re-generating is one statement and cannot leave a run noteless if it fails halfway.
    Replacement is the whole behaviour: one run has one note, and a second generation of the
    same run is a newer statement about the same numbers, not a second opinion to be kept
    beside the first.

    **Deciding whether to replace is the CALLER's job, and `cabinet/notes.py` does not
    replace a good note with a failure.** A nightly job run on a machine whose key expired
    would otherwise erase every note in the cabinet and replace it with «ключа нет» — a
    regression caused by nothing changing.
    """
    if connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        raise Refusal("unknown_run",
                      f"прогона {run_id} нет в базе, поэтому записку не к чему привязать. "
                      f"Сначала `cabinet import`, список прогонов — `cabinet show`.")
    connection.execute(
        """INSERT INTO run_notes (run_id, text, available, source, backend_requested, model,
               prompt_version, guard_passed, guard_reason_ru, guard_offending_json,
               reason_ru, bundle_sha256, bundle_bytes, generated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(run_id) DO UPDATE SET
               text = excluded.text, available = excluded.available,
               source = excluded.source, backend_requested = excluded.backend_requested,
               model = excluded.model, prompt_version = excluded.prompt_version,
               guard_passed = excluded.guard_passed,
               guard_reason_ru = excluded.guard_reason_ru,
               guard_offending_json = excluded.guard_offending_json,
               reason_ru = excluded.reason_ru, bundle_sha256 = excluded.bundle_sha256,
               bundle_bytes = excluded.bundle_bytes,
               generated_at = excluded.generated_at""",
        (run_id, text, 1 if available else 0, source, backend_requested, model,
         prompt_version, _tri(guard_passed), guard_reason_ru,
         _dump(list(guard_offending)), reason_ru, bundle_sha256, int(bundle_bytes),
         datetime.now(UTC).isoformat(timespec="seconds")))
    connection.commit()
    return note_row(connection, run_id) or {}


def note_row(connection: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """The stored note of one run, exactly as stored. `None` when that run has none.

    Deliberately raw: freshness is decided by `cabinet/notes.py`, which can recompute the
    bundle this note was written from and compare hashes. A reader of the store gets the row;
    a renderer of the page gets the verdict.
    """
    row = connection.execute("SELECT * FROM run_notes WHERE run_id = ?",
                             (run_id,)).fetchone()
    return None if row is None else dict(row)


def teacher_record(connection: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """The adult's block for ONE run. There is deliberately no aggregate of these.

    `metrics/teacher.py` is a description of position over time carrying
    `not_an_assessment_ru`, and this store adds nothing to it. A weekly teacher index would
    be a staff-comparison surface built out of a camera that cannot tell «сидит» from «не
    ведёт урок», and INTEGRATION.md §7 forbids exactly that. So the record is stored per
    run, shown per lesson, and never summed.
    """
    row = connection.execute("SELECT teacher_json FROM runs WHERE run_id = ?",
                             (run_id,)).fetchone()
    if row is None or not row["teacher_json"]:
        return None
    return json.loads(row["teacher_json"])


def _runs_in_scope(connection: sqlite3.Connection, column: str, room_key: str | None,
                   class_key: str | None) -> list[sqlite3.Row]:
    where, args = _scope(room_key, class_key, prefix="l.")
    return list(connection.execute(
        f"SELECT r.{column} AS value FROM runs r JOIN lessons l "
        f"ON l.lesson_id = r.lesson_id{where} ORDER BY r.imported_at", args))


def caveats(connection: sqlite3.Connection, *, room_key: str | None = None,
            class_key: str | None = None) -> list[str]:
    """The union of every imported run's caveats, in first-seen order.

    Copied out of the artefacts, never retyped here. `report/artefact.py::CAVEATS_RU`
    exists in one place for the reason the neighbouring codebase paid for: four
    hand-written copies of a caveat is four chances for one of them to quietly stop being
    true. The cabinet is the fifth surface and it takes the same route as the other four.
    """
    seen: list[str] = []
    for row in _runs_in_scope(connection, "caveats_json", room_key, class_key):
        for sentence in json.loads(row["value"] or "[]"):
            if sentence not in seen:
                seen.append(sentence)
    return seen


def unmeasured(connection: sqlite3.Connection, *, room_key: str | None = None,
               class_key: str | None = None) -> list[dict[str, Any]]:
    """Everything the imported runs said they could NOT measure, de-duplicated.

    Rule 4 of this project: anything unmeasurable is listed explicitly, never silently
    omitted — and that has to survive accumulation, otherwise a term's page shows twelve
    lessons' worth of counters with no note that «ответил вслух» was never measurable on
    any of them because the recordings have no audio.

    Counted per RUN and scoped to the class being displayed. A page for 3-Б that said «во
    всех 5 прогонах» while three of them belong to another room would be a true sentence
    about the wrong set, which is the quietest kind of wrong a report can be.
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in _runs_in_scope(connection, "unmeasured_json", room_key, class_key):
        for item in json.loads(row["value"] or "[]"):
            key = f"{item.get('what')}|{item.get('why')}"
            entry = seen.setdefault(key, {**item, "runs": 0})
            entry["runs"] += 1
    return list(seen.values())


def summary(connection: sqlite3.Connection) -> dict[str, Any]:
    """What is in this cabinet, in the terms the CLI prints."""
    def scalar(sql: str) -> int:
        return int(connection.execute(sql).fetchone()[0] or 0)

    return {
        "rooms": rooms(connection),
        "lessons": scalar("SELECT COUNT(*) FROM lessons"),
        "lessons_dated": scalar("SELECT COUNT(*) FROM lessons WHERE started_at IS NOT NULL"),
        "sessions": scalar("SELECT COUNT(*) FROM lessons WHERE continues_lesson_id IS NULL"),
        "runs": scalar("SELECT COUNT(*) FROM runs"),
        "runs_not_selected": scalar(
            "SELECT COUNT(*) FROM runs r JOIN lessons l ON l.lesson_id = r.lesson_id "
            "WHERE l.selected_run_id IS NOT r.run_id"),
        "places": scalar("SELECT COUNT(*) FROM places"),
        "places_pupil": scalar("SELECT COUNT(*) FROM places WHERE role = 'pupil'"),
        "attested_places": scalar("SELECT COUNT(DISTINCT place_id) FROM attestations"),
        "seat_lessons": scalar("SELECT COUNT(*) FROM seat_lessons"),
        "seat_lessons_unplaced": scalar(
            "SELECT COUNT(*) FROM seat_lessons WHERE place_id IS NULL"),
        "threshold_sets": scalar("SELECT COUNT(DISTINCT thresholds_sha) FROM runs"),
        # Notes counted in two numbers, never one. «записок 2» over a cabinet where both
        # were refused by the guard would report coverage that does not exist; the pair
        # answers «сколько прогонов вообще пробовали» and «сколько из них показывается».
        "notes": scalar("SELECT COUNT(*) FROM run_notes"),
        "notes_available": scalar("SELECT COUNT(*) FROM run_notes WHERE available = 1"),
        "unmeasured": unmeasured(connection),
    }


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _scope(room_key: str | None, class_key: str | None,
           prefix: str = "") -> tuple[str, list[Any]]:
    clauses, args = [], []
    if room_key:
        clauses.append(f"{prefix}room_key = ?")
        args.append(room_key)
    if class_key:
        clauses.append(f"{prefix}class_key = ?")
        args.append(class_key)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


def _room_key(provenance: dict[str, Any], given: str | None) -> tuple[str, str]:
    """The room is an operator assertion; every fallback is a convenience that says so.

    Three sources, in descending order of how much a human stands behind them:

      1. `--room-key`, typed by the operator for this import.
      2. `provenance.room.layout.camera` — the camera named in the room profile that was
         used for the analysis. That profile is a file a person wrote and signed
         (`zones_confirmed_by`), so its camera name is an assertion too, just an earlier
         one. Preferring it over the filename is what makes a configured camera Just Work.
      3. The video filename's stem. A DVR name like `D14_20260815101759.mp4` carries the
         camera in its head and using it saves typing, but it is a guess — two cameras can
         share a naming scheme and one camera can be renamed. The source is stored and the
         CLI prints it, because the failure it guards against is silent: `clip_15min` and
         `test_camera` are the SAME camera and would otherwise become two rooms that never
         accumulate together.
    """
    if given:
        return given, "operator"
    layout = (provenance.get("room") or {}).get("layout") or {}
    if layout.get("camera"):
        return str(layout["camera"]), "from_room_profile"
    stem = Path(str(provenance.get("video_path") or "unknown")).stem
    head = stem.split("_")[0]
    if len(stem.split("_")) == 2 and stem.split("_")[1].isdigit() and head:
        return head, "derived_from_filename"
    return stem or "unknown", "derived_from_filename"


def _class_key(document: dict[str, Any], given: str | None) -> tuple[str, str]:
    """The class is an operator assertion too, with one legitimate second source.

    If a signed seating plan was applied during analysis, the artefact carries the class it
    was written for, and that IS a human statement — it came off the plan. Anything else
    falls back to the room, which keeps a single-class room working without a flag while
    remaining visibly a fallback.
    """
    if given:
        return given, "operator"
    for seat in document.get("seats") or []:
        name = ((seat.get("pupil") or {}).get("class_name"))
        if name:
            return str(name), "from_attested_seat_map"
    return "-", "not_stated"


def _parse_local(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _tri(value: Any) -> int | None:
    """True/False/unknown kept as 1/0/NULL. A boolean whose false has two causes is a bug."""
    return None if value is None else (1 if value else 0)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _settings_sha(thresholds: dict[str, Any]) -> str:
    """One string answering «обе недели мерили одинаково?» with an equality test."""
    return hashlib.sha256(_dump(thresholds).encode()).hexdigest()[:12]


def _file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def import_many(connection: sqlite3.Connection, paths: Iterable[str | Path],
                **kwargs: Any) -> list[tuple[str, ImportResult | Refusal | Unreadable]]:
    """Import a directory's worth, continuing past refusals and reporting every one.

    A back-fill that stops on the first bad artefact leaves a term half-loaded and gives no
    list of what to fix. Each outcome is returned paired with its path, refusals included,
    so the caller prints a complete report and decides.
    """
    out: list[tuple[str, ImportResult | Refusal | Unreadable]] = []
    for path in paths:
        try:
            out.append((str(path), import_artefact(connection, path, **kwargs)))
        except (Refusal, Unreadable) as error:
            out.append((str(path), error))
    return out

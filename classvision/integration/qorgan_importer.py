"""Turn one classvision artefact into the rows `qorgan` would insert — and nothing else.

**This module is the executable half of `INTEGRATION.md`.** Read that document for the
argument; this file is the part a test can run. It parses an artefact, refuses the ones
that must not enter a school's database, and returns the rows it WOULD write, as plain
dicts keyed by table name. It opens no database and imports no ORM, so the whole refusal
surface can be exercised in milliseconds without a Postgres, and so this file can live in
the `classvision` repository — which must never import `qorgan` — while describing
`qorgan`'s tables exactly.

--------------------------------------------------------------------------------
**WHY A PLANNER AND NOT AN INSERTER.**

The real command is `qorgan classvision import`, and it is ~40 lines: call `plan()`, open
a session, insert `plan.rows` in table order, commit. Every judgement worth testing —
which artefacts are refused, which seat is linked to which child, what happens on a second
import of the same file — is in `plan()`, where it needs no database at all. The mistake
this shape avoids is the one the neighbouring codebase names repeatedly: a rule that is
true in one layer and quietly wrong in the next. There is one place that decides, and both
the CLI and the tests go through it.

The rows are dicts rather than model instances for the same reason the artefact is JSON
rather than a pickle: the seam has to survive `qorgan` renaming a column without
`classvision` growing a dependency on the rename.

--------------------------------------------------------------------------------
**THE FOUR THINGS THIS FILE REFUSES, AND WHY EACH ONE IS A REFUSAL RATHER THAN A WARNING.**

A warning on an import is a line in a log nobody reads, after which the row is in the
database forever and every later report is computed over it. So:

1. **A schema major this code has not read.** `classvision/1.x` is accepted, `2.x` is not.
   A minor version adds fields; a major one changes what an existing field means, and a
   number whose meaning changed is the step change in a term's trend that nobody can
   explain afterwards.

2. **An artefact whose own seat discovery says it should not be trusted.**
   `room.seat_discovery.plausible is False` is `seats.discover()` reporting that it found
   materially fewer places than the detector saw people — the single-link chaining
   collapse that turned 9 seats into 3 (`room/seats.py`). The artefact already says the
   seats must not be trusted; importing them anyway would put two merged children into one
   term's history, and nothing downstream could ever detect it.

3. **A wall clock with no zone.** `provenance.started_at` is read off a burned-in overlay
   (`video/clock.py`) and is therefore LOCAL school time with no offset, while every
   datetime column in `qorgan` is `UtcDateTime`, which rejects a naive value at bind time.
   Guessing the zone would put a lesson an hour into the wrong day at the boundary, which
   is precisely the sort of quiet wrongness `UtcDateTime` exists to make impossible. So
   the zone is passed in or the import refuses; `--allow-unclocked` stores NULL and the
   run is then excluded from anything longitudinal, which is a smaller loss than a wrong
   date and a visible one.

4. **A face used to CREATE a name.** `SeatRecord.pupil` may carry a suggestion from the
   identity stage. It is recorded as EVIDENCE — a text column, deliberately not a foreign
   key, so no report can ever join through it — and it never becomes `person_id`. The
   measurement behind that refusal is in `MEASUREMENTS.md` §4: median best cosine 0.30
   against the 141-pupil gallery, with 0.10 to the runner-up. The only thing that creates
   a name here is a human attestation with a school decision reference on it.

--------------------------------------------------------------------------------
**HOW A SEAT FINDS ITS CHILD, AND WHY IT IS GEOMETRY AND NOT A SEAT NUMBER.**

`seat_id` is assigned by reading order at the end of `seats.discover()`, so it is a
property of ONE run: a lesson where a pupil is absent discovers eight places instead of
nine and renumbers everything below the gap. Keying an attestation on `seat_id` would
therefore silently shift a whole class's history by one the first time somebody was off
sick. So an attestation carries the seat CENTRE and SCALE it was made against, and a new
run's seats are matched to it by position, in shoulder widths, exactly the unit
`room/seats.py` works in.

The match uses the same discipline the face measurement failed: a gate AND a margin to
the runner-up. A seat that is inside the gate of two attestations is left unlinked with a
reason recorded, because at that point the room has been re-arranged and the person who
knows what happened is the teacher, not this function.

**Attestations are time-bounded, and that is what makes the seat-swap caveat survivable.**
`valid_from`/`valid_to` mean a class can be reseated in March without rewriting February:
a run is matched against the attestations that were in force on the day it was RECORDED,
not the ones in force today.

--------------------------------------------------------------------------------
**WHAT THIS DELIBERATELY DROPS.**

`Artefact.teacher` is not imported, and the drop is recorded rather than silent. The
school was told in writing that nothing would judge a teacher's work (§8, and
`qorgan.classroom` §12.5, deliberately not built). `TeacherRecord` carries no quality
field and cannot acquire one — but a per-adult time series in the school's own database
is a thing a later query can rank, and the protection the promise actually has is the
absence of the rows, not the absence of a column. What survives is one integer on the run:
which seat was the adult's, so a reader can see why a room of nine people reports eight
pupils.

Every field is either mapped, dropped-with-a-reason, or kept whole in a `*_json` column.
`test_every_artefact_field_is_accounted_for` is what stops a fifth category appearing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ---------------------------------------------------------------------------
# What this importer understands.
# ---------------------------------------------------------------------------

# The schema family this code has read. A MINOR bump adds fields and is accepted (the
# unread ones land in the `*_json` columns intact); a MAJOR bump changes what an existing
# field means and is refused. CHOSEN, and the asymmetry is the point: an unknown field is
# a gap in a report, a redefined field is a wrong number that looks right.
ACCEPTED_SCHEMA_MAJOR = "classvision/1"

# Exit codes, following `qorgan identity camera-report`: a script must be able to tell
# "this artefact must not be imported" from "I could not answer".
IMPORTED = 0
REFUSED = 1
UNANSWERED = 2

# How close a seat must be to an attested seat to be considered the same place, in
# shoulder widths. `room/seats.py::assign` uses 1.5 for "a seated pupil leaning across
# their desk must not fall out of their own seat"; this is TIGHTER, at 1.0, because the
# consequence is different. There, a mis-assignment costs one observation; here it costs a
# child's name on another child's term. CHOSEN below the ~1.5-2.5 shoulder widths that
# separate two pupils sharing a desk (`room/seats.py`), so the gate cannot span a
# neighbour.
SEAT_MATCH_GATE_SCALES = 1.0

# ...and the nearest attestation must be this many times closer than the second nearest.
# The lesson of `MEASUREMENTS.md` §4 applied to geometry rather than to faces: a best
# score without a margin to the runner-up is a preference, not an identification. A room
# that has been re-arranged produces exactly the ambiguous case this catches, and the
# right answer there is to leave the seat anonymous and ask the teacher. CHOSEN.
SEAT_MATCH_MARGIN = 2.0

# The tables, in insert order. Written down because the caller inserts them in this order
# and a foreign key does not tolerate being guessed at.
TABLES = (
    "classvision_runs",
    "classvision_seat_records",
)


class Refusal(Exception):
    """This artefact must not enter a school's database, and here is which rule says so.

    Carries a machine-readable `code` as well as the sentence, because the CLI is called
    by a script during a term's back-fill and "which of the refusals fired" must be
    answerable without matching on prose.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Unreadable(Exception):
    """The file could not be read or parsed. Distinct from a Refusal on purpose.

    A refusal is a statement about the artefact; this is a statement about our ability to
    look at it. Collapsing the two would let a truncated download report as a schema
    violation, and somebody would go and change the schema.
    """


# ---------------------------------------------------------------------------
# The human attestation. Reproduced here as a dataclass so this file can be run and
# tested standalone; in `qorgan` it is a row of `classvision_seat_attestations`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attestation:
    """One human's statement that a place in a room is a named child's place.

    `decision_ref` is not optional and is not decoration. Per-pupil accumulation in a
    classroom is the thing `docs/questions-for-school.md` §8 told the school would not
    happen; §10.1 then asked the school whether it wants it, naming the only acceptable
    mechanism («ученик отмечается сам, или это делает учитель»). Until the school answers
    in writing there is no attestation to make, so the reference to that answer is a NOT
    NULL column: an attestation nobody can trace to a school decision is exactly the
    quiet re-opening of a written promise that this whole design exists to prevent.
    """

    seat_label: str
    centre: tuple[float, float]
    scale_px: float
    person_id: int
    person_external_id: str
    valid_from: datetime
    valid_to: datetime | None
    decision_ref: str
    attested_by_id: int | None = None
    note: str | None = None

    def in_force_at(self, moment: datetime | None) -> bool:
        """Was this attestation in force when the lesson was RECORDED?

        `None` (an unclocked run) is never in force: a run we cannot place on a calendar
        cannot be placed on either side of a reseating, and the honest answer to "which
        seating plan applied" is then "we do not know", not "today's".
        """
        if moment is None:
            return False
        if moment < self.valid_from:
            return False
        return self.valid_to is None or moment < self.valid_to


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Plan:
    """The rows an import WOULD write, plus everything it decided not to write.

    `dropped` and `notes` are part of the result and not diagnostics, on the same rule as
    the artefact's own `uncertainty` block: an import that silently discarded a seat and
    an import that had no seat to discard must not look identical afterwards.
    """

    run_id: str
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def seats(self) -> list[dict[str, Any]]:
        return self.rows.get("classvision_seat_records", [])

    @property
    def attested_seats(self) -> int:
        return sum(1 for row in self.seats if row["attestation_person_id"] is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows": self.rows,
            "dropped": self.dropped,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------


def load(path: str | Path) -> dict[str, Any]:
    """Read one artefact. Raises `Unreadable`, never returns a partial document."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise Unreadable(f"cannot read {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Unreadable(f"{path} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise Unreadable(f"{path} is JSON but not an object")
    return document


# ---------------------------------------------------------------------------
# The refusals. One function, so there is one list and one order.
# ---------------------------------------------------------------------------


def check(document: dict[str, Any], *, allow_unclocked: bool = False) -> None:
    """Everything that stops this artefact entering a database. Raises `Refusal`.

    Ordered from the cheapest and most fundamental outward, so that the error a human
    sees is the earliest true thing rather than a downstream symptom of it.
    """
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise Refusal("no_provenance", "the document carries no `provenance` block")

    schema = str(provenance.get("schema", ""))
    if not schema.startswith(ACCEPTED_SCHEMA_MAJOR + "."):
        raise Refusal(
            "schema_version",
            f"schema {schema!r} is not {ACCEPTED_SCHEMA_MAJOR}.x. A major version changes "
            "what an existing field MEANS; importing it under this code would put a "
            "differently-defined number into the same column as last term's.",
        )

    if not document.get("run_id"):
        raise Refusal("no_run_id", "the document carries no `run_id`; idempotency has no key")

    if not provenance.get("video_sha256"):
        raise Refusal(
            "no_video_hash",
            "`provenance.video_sha256` is empty. It is the only thing that ties this row "
            "to a recording somebody can go back and watch.",
        )

    caveats = document.get("caveats")
    if not isinstance(caveats, list) or not caveats:
        raise Refusal(
            "no_caveats",
            "the document carries no `caveats`. They are copied into the database and "
            "rendered beside every total; a document without them cannot be shown.",
        )

    discovery = (provenance.get("room") or {}).get("seat_discovery") or {}
    if discovery.get("plausible") is False:
        raise Refusal(
            "implausible_seats",
            "the artefact's own seat discovery reports `plausible: false` — "
            f"{discovery.get('warning', 'places are probably being merged')}. These seats "
            "must not enter a term's history: two children merged into one place is "
            "undetectable afterwards.",
        )

    if not document.get("seats"):
        raise Refusal("no_seats", "no seats were discovered; there is nothing to accumulate")

    if provenance.get("started_at") in (None, "") and not allow_unclocked:
        raise Refusal(
            "no_wall_clock",
            "`provenance.started_at` is empty, so this run cannot be placed on a "
            "calendar and cannot be compared with any other lesson. Pass "
            "--allow-unclocked to import it anyway; it will be stored with a NULL "
            "timestamp and excluded from anything longitudinal.",
        )

    for seat in document["seats"]:
        activity = (seat.get("metrics") or {}).get("activity") or {}
        if activity.get("available") and not activity.get("parts"):
            raise Refusal(
                "index_without_parts",
                f"seat {seat.get('label')} carries an activity index with no components. "
                "The index is only ever shown decomposed; a bare number is the one thing "
                "this schema forbids.",
            )


# ---------------------------------------------------------------------------
# Planning.
# ---------------------------------------------------------------------------


def plan(
    document: dict[str, Any],
    *,
    camera_id: int,
    school_timezone: str | None = None,
    attestations: tuple[Attestation, ...] = (),
    allow_unclocked: bool = False,
    imported_at: datetime | None = None,
    # OFF by default. See `_handle_the_teacher`: the client asked for teacher analysis in
    # writing and `qorgan.classroom` §12.5 promised the school there would be none, so the
    # choice belongs to the school and is recorded either way. There is also no
    # teacher-facing camera yet, which makes off the honest default today.
    include_teacher: bool = False,
) -> Plan:
    """One artefact -> the rows `qorgan` would insert. Pure; touches no database.

    `camera_id` comes from the caller and never from the artefact: `classvision` analyses
    a FILE and has no idea which of a school's cameras produced it. Making the operator
    name the camera is also what routes the row to a school — `db/tenancy.py` reaches a
    school through the camera, and a row that named its own would be the second answer to
    one question that this schema keeps being bitten by.
    """
    check(document, allow_unclocked=allow_unclocked)

    provenance = document["provenance"]
    started_at = _wall_clock(provenance.get("started_at"), school_timezone)
    result = Plan(run_id=str(document["run_id"]))

    if started_at is None:
        result.notes.append(
            "imported without a wall clock: this run is excluded from every longitudinal "
            "view, because two lessons cannot be ordered without dates."
        )

    result.rows["classvision_runs"] = [
        _run_row(document, camera_id=camera_id, started_at=started_at,
                 imported_at=imported_at or datetime.now(UTC))
    ]

    in_force = tuple(a for a in attestations if a.in_force_at(started_at))
    if attestations and not in_force:
        result.notes.append(
            f"{len(attestations)} attestation(s) exist for this camera and none was in "
            "force on the day this lesson was recorded; every seat stays anonymous."
        )

    rows: list[dict[str, Any]] = []
    for seat in document["seats"]:
        rows.append(_seat_row(seat, run_id=result.run_id, attestations=in_force,
                              started_at=started_at))
    result.rows["classvision_seat_records"] = rows

    _handle_the_teacher(document, result, include=include_teacher)
    _carry_the_unmeasured(document, result)
    return result


def _wall_clock(raw: Any, school_timezone: str | None) -> datetime | None:
    """The overlay's LOCAL time, made into an aware UTC datetime, or nothing.

    The clock reader returns what is burned into the picture, which is a school's wall
    clock with no offset on it. `qorgan.db.types.UtcDateTime` rejects a naive datetime at
    bind time, and rightly: a naive value stored as if it were UTC is the defect that
    whole column type exists to prevent. So the zone is supplied by the caller, from
    `SCHOOL_TIMEZONE`, and never guessed here.
    """
    if raw in (None, ""):
        return None
    try:
        local = datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise Refusal("bad_wall_clock", f"`started_at` is not ISO-8601: {raw!r} ({exc})") from exc
    if local.tzinfo is not None:
        return local.astimezone(UTC)
    if not school_timezone:
        raise Refusal(
            "no_timezone",
            f"`started_at` is {raw!r} — a wall clock read off the picture, with no zone. "
            "Pass --timezone (in qorgan: SCHOOL_TIMEZONE). Nothing here guesses one: an "
            "hour's error puts a lesson on the wrong day at the boundary, silently.",
        )
    try:
        zone = ZoneInfo(school_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise Refusal("bad_timezone", f"unknown timezone {school_timezone!r}: {exc}") from exc
    return local.replace(tzinfo=zone).astimezone(UTC)


def _run_row(document: dict[str, Any], *, camera_id: int, started_at: datetime | None,
             imported_at: datetime) -> dict[str, Any]:
    """The imported analysis run. Promoted columns are the ones a page filters on.

    Everything else stays whole in `provenance_json`, deliberately: a report is a
    statement about a past hour, and re-deriving the thresholds it was computed under
    from today's configuration is this codebase's signature defect (a value true in one
    layer and quietly wrong in the next).
    """
    provenance = document["provenance"]
    uncertainty = document.get("uncertainty") or {}
    lesson = document.get("lesson") or {}

    return {
        "camera_id": camera_id,
        "run_id": str(document["run_id"]),
        "schema_version": provenance["schema"],
        "video_sha256": provenance["video_sha256"],
        "video_bytes": int(provenance.get("video_bytes") or 0),
        "started_at": started_at,
        "clock_source": str(provenance.get("clock_source") or "none"),
        "clock_drift_seconds": _float_or_none(provenance.get("clock_drift_seconds")),
        "duration_seconds": float(lesson.get("duration_minutes") or 0.0) * 60.0,
        "window_start_seconds": float(provenance.get("window_start_seconds") or 0.0),
        "window_end_seconds": _float_or_none(provenance.get("window_end_seconds")),
        "sample_fps": float(provenance.get("sample_fps") or 0.0),
        "analysed_frames": int(provenance.get("analysed_frames") or 0),
        "analysed_at": _iso_to_utc(provenance.get("analysed_at")),
        "imported_at": imported_at,
        # -- what the run could not see. Never render a total without these.
        "observations_total": int(uncertainty.get("observations_total") or 0),
        "observations_unassigned": int(uncertainty.get("observations_unassigned") or 0),
        "observations_unreadable": int(uncertainty.get("observations_unreadable") or 0),
        "frames_with_no_person": int(uncertainty.get("frames_with_no_person") or 0),
        "seats_never_settled": int(uncertainty.get("seats_never_settled") or 0),
        "pupil_seats": int(lesson.get("pupil_seats") or 0),
        # One integer about the adult, and no time series. See the module docstring.
        "adult_seat_id": lesson.get("adult_seat"),
        # Copied, never retyped. `report/artefact.py::CAVEATS_RU` is the one wording.
        "caveats_json": _compact(document["caveats"]),
        "unmeasured_json": _compact(lesson.get("unmeasured") or []),
        "provenance_json": _compact(provenance),
        "uncertainty_json": _compact(uncertainty),
    }


def _seat_row(seat: dict[str, Any], *, run_id: str, attestations: tuple[Attestation, ...],
              started_at: datetime | None) -> dict[str, Any]:
    """One place, for one lesson — the unit of accumulation, and never a child.

    `attestation_person_id` is the only path by which a name reaches these numbers, and it
    arrives from a human. `face_suggested_external_id` beside it is TEXT and not a foreign
    key: it can be displayed as corroboration and it cannot be joined.
    """
    ledger = seat.get("ledger") or {}
    counts = ledger.get("counts") or {}
    seconds = ledger.get("observed_seconds_by_state") or {}
    activity = (seat.get("metrics") or {}).get("activity") or {}
    centre = tuple(seat.get("centre") or (0.0, 0.0))
    scale = float(seat.get("scale_px") or 0.0)

    match, reason, distance = _match(centre, scale, attestations)
    suggestion = seat.get("pupil") or {}

    return {
        "run_id": run_id,
        "seat_id": int(seat["seat_id"]),
        "label": str(seat["label"]),
        "role": str(seat.get("role") or "pupil"),
        "centre_x": float(centre[0]),
        "centre_y": float(centre[1]),
        "scale_px": scale,
        "occupancy": float(seat.get("occupancy") or 0.0),
        "settled": bool(ledger.get("settled")),
        "first_seen_seconds": float(ledger.get("first_seen_s") or 0.0),
        "last_seen_seconds": float(ledger.get("last_seen_s") or 0.0),
        "observations": int(ledger.get("observations") or 0),
        "observed_seconds": float(ledger.get("observed_seconds") or 0.0),
        # -- the four doubt counters, beside the totals they qualify.
        "coverage": float(ledger.get("coverage") or 0.0),
        "absent_observations": int(ledger.get("absent_observations") or 0),
        "unreadable_observations": int(ledger.get("unreadable_observations") or 0),
        "hand_unmeasurable_observations": int(ledger.get("hand_unmeasurable_observations") or 0),
        # -- counts of episodes.
        "hand_raises": int(counts.get("hand_raises") or 0),
        "stands": int(counts.get("stands") or 0),
        "away_episodes": int(counts.get("away_episodes") or 0),
        "board_visits": int(counts.get("board_visits") or 0),
        "head_down_episodes": int(counts.get("head_down_episodes") or 0),
        "turned_away_episodes": int(counts.get("turned_away_episodes") or 0),
        # -- seconds in each state.
        "away_seconds": float(seconds.get("away_from_place") or 0.0),
        "at_board_seconds": float(seconds.get("at_board") or 0.0),
        "stood_up_seconds": float(seconds.get("stood_up") or 0.0),
        "hand_raised_seconds": float(seconds.get("hand_raised") or 0.0),
        "head_down_seconds": float(seconds.get("head_down") or 0.0),
        "turned_away_seconds": float(seconds.get("turned_away") or 0.0),
        "seated_seconds": float(seconds.get("seated") or 0.0),
        "unknown_seconds": float(seconds.get("unknown") or 0.0),
        # -- the index, which is NULL unless it is available, and its parts, which are
        # NOT NULL. A schema in which an index can exist without its components is a
        # schema in which a page can render one.
        "activity_index": _float_or_none(activity.get("index")) if activity.get("available") else None,
        "activity_unavailable_reason": (None if activity.get("available")
                                        else str(activity.get("reason") or "not computed")),
        "activity_parts_json": _compact(activity.get("parts") or []),
        "ledger_json": _compact(ledger),
        # -- identity. One column carries a name and a human put it there.
        "attestation_person_id": match.person_id if match else None,
        "attestation_decision_ref": match.decision_ref if match else None,
        "attestation_match_scales": (round(distance, 3) if distance is not None else None),
        "attestation_note": reason,
        # -- and one column carries what the faces thought, as evidence only.
        "face_suggested_external_id": suggestion.get("external_id"),
        "face_confidence": _float_or_none(suggestion.get("confidence")),
        "face_method": suggestion.get("method"),
        "face_agrees": _agrees(match, suggestion),
    }


def _match(centre: tuple[float, float], scale: float,
           attestations: tuple[Attestation, ...]) -> tuple[Attestation | None, str, float | None]:
    """Which attested place is this one — by geometry, with a gate AND a margin.

    Returns the attestation (or None), the sentence saying why, and the normalised
    distance to the winner. The unmatched cases are named individually because they need
    different actions from a human: nobody has attested this room, the room has moved, or
    the room has been re-arranged so that two attested places now overlap.
    """
    if not attestations:
        return None, "no attestation in force for this camera on that date", None
    # Normalised by the MEAN of the two scales, the way `room/seats.py::_cluster` does it:
    # the room is viewed at an angle, so a fixed pixel gate would span two back-row desks
    # while failing to cover one front-row pupil leaning forward.
    ranked = sorted(
        (
            (math.dist(centre, a.centre) / max((scale + a.scale_px) / 2.0, 1e-6), a)
            for a in attestations
        ),
        key=lambda pair: pair[0],
    )
    best_distance, best = ranked[0]
    if best_distance > SEAT_MATCH_GATE_SCALES:
        return None, (
            f"nearest attested place ({best.seat_label}) is {best_distance:.2f} shoulder "
            f"widths away, past the {SEAT_MATCH_GATE_SCALES} gate: the seating has moved "
            "and needs re-attesting"
        ), best_distance
    if len(ranked) > 1:
        runner_up = ranked[1][0]
        if runner_up < best_distance * SEAT_MATCH_MARGIN:
            return None, (
                f"two attested places are within reach ({best.seat_label} at "
                f"{best_distance:.2f}, {ranked[1][1].seat_label} at {runner_up:.2f} "
                "shoulder widths); refusing to guess which child this is"
            ), best_distance
    return best, f"attested as {best.seat_label} ({best.decision_ref})", best_distance


def _agrees(match: Attestation | None, suggestion: dict[str, Any]) -> bool | None:
    """Did the face evidence agree with the human? `None` when there was nothing to ask.

    Three-valued on purpose. "The faces agreed", "the faces disagreed" and "there was no
    face evidence" are three different things, and a boolean whose false has two causes is
    the defect migration 0005 was written about.
    """
    if match is None or not suggestion.get("external_id"):
        return None
    return str(suggestion["external_id"]) == match.person_external_id


def _handle_the_teacher(document: dict[str, Any], result: Plan, *,
                        include: bool) -> None:
    """Import the adult's position record, or drop it — but never decide that silently.

    **Why this is a switch and not a policy.** Two true things are in tension and neither
    of them is an engineer's to resolve:

      * The client asked for this in writing: «камера смотрит на учителя и анализирует
        что он сидит много времени или на доске объясняет урок». Dropping it by default
        would be quietly delivering less than was ordered, on an engineer's judgement.
      * `qorgan.classroom` records a promise to the SCHOOL that nothing would judge a
        teacher's work (§12.5, deliberately not built). Importing it by default would be
        quietly breaking a promise, also on an engineer's judgement.

    So the flag exists, it has no default hidden inside this function, and whichever way
    it is set the choice is written into `notes` or `dropped` where an auditor sees it.
    What the module never does either way is produce a quality score: the record is share
    of time seated, standing and out of frame, and it travels with
    `not_an_assessment_ru`, which the cabinet must render beside it.
    """
    teacher = document.get("teacher")
    if not teacher:
        return

    if not include:
        result.dropped.append({
            "what": "teacher record",
            "seat_id": teacher.get("seat_id"),
            "why": "not imported: --no-teacher was set. The school was promised no "
                   "judgement about a teacher's work (qorgan.classroom §12.5). Only "
                   "which seat was the adult's is kept, so a reader can see why nine "
                   "people are eight pupils.",
        })
        return

    metrics = teacher.get("metrics") or {}
    presence = metrics.get("presence")
    # **§7 of `INTEGRATION.md` is a rule this function did not enforce.** It says no teacher
    # number may be rendered without `attributed_share_of_lesson_percent` beside it, and
    # that the importer must refuse a teacher block that lacks it — because on camera D14
    # the follower holds only 45 % of the lesson, and «у доски — 3 %» without «опознан в
    # 45 % кадров» is a false statement made of true numbers. The row below carried
    # `coverage`, which is that share on one code path and the SEAT's occupancy on the
    # other, so the guarantee could not be met by reading it. The share now travels as its
    # own column, and a `presence` block without it is dropped rather than imported
    # half-qualified. Artefacts with no `presence` at all (every camera before D14) are
    # unaffected: they have no state shares to qualify.
    if presence is not None and presence.get("attributed_share_of_lesson_percent") is None:
        result.dropped.append({
            "what": "teacher record",
            "seat_id": teacher.get("seat_id"),
            "why": "not imported: the presence block carries state shares but no "
                   "`attributed_share_of_lesson_percent`. INTEGRATION.md §7 forbids "
                   "rendering any of those shares without it, and an importer that "
                   "stores them anyway makes that impossible to honour downstream.",
        })
        return

    result.rows.setdefault("classvision_teacher_records", []).append({
        "run_id": result.run_id,
        "seat_id": teacher.get("seat_id"),
        # The denominator every state share below must be shown against. None on a camera
        # with no position taxonomy, where there are no such shares to qualify.
        "attributed_share_of_lesson_percent": (
            presence.get("attributed_share_of_lesson_percent") if presence else None),
        "identification_source": (teacher.get("identification") or {}).get("source"),
        "needs_confirmation": (teacher.get("identification") or {}).get(
            "needs_confirmation"),
        "coverage": metrics.get("coverage"),
        "at_desk_share_of_observed": metrics.get("at_desk_share_of_observed"),
        "at_desk_minutes": metrics.get("at_desk_minutes"),
        "standing_or_away_share": metrics.get("standing_or_away_share_of_observed"),
        "out_of_frame_share": metrics.get("out_of_frame_share_of_lesson"),
        "transitions": metrics.get("transitions"),
        "longest_at_desk_episode_minutes": metrics.get("longest_at_desk_episode_minutes"),
        # Carried as a column, not as template text, so it cannot be rendered without it.
        "not_an_assessment_ru": metrics.get("not_an_assessment_ru"),
    })
    result.notes.append(
        "запись об учителе импортирована по прямому требованию заказчика (§12.5 "
        "qorgan.classroom обещал обратное — решение о том, что показывать, принимает "
        "школа). Это описание положения в пространстве, а не оценка работы."
    )


def _carry_the_unmeasured(document: dict[str, Any], result: Plan) -> None:
    """Anything the run could not measure is listed, not silently omitted."""
    for item in (document.get("lesson") or {}).get("unmeasured") or []:
        result.notes.append(f"не измерено — {item.get('what')}: {item.get('why')}")
    for note in (document.get("uncertainty") or {}).get("notes") or []:
        result.notes.append(str(note))


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"),
                      default=str)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _iso_to_utc(raw: Any) -> datetime | None:
    """`analysed_at` is written by `report/artefact.py::utc_now()` and is already aware."""
    if raw in (None, ""):
        return None
    moment = datetime.fromisoformat(str(raw))
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(
        tzinfo=UTC)


# ---------------------------------------------------------------------------
# The CLI. In `qorgan` this is `qorgan classvision import`; here it prints the plan.
# ---------------------------------------------------------------------------


def load_attestations(path: str | Path | None) -> tuple[Attestation, ...]:
    """Read a seat map from JSON, for running this file without a database.

    The real command reads `classvision_seat_attestations` for the camera. The shape is
    the same either way, which is the point of having it in one dataclass.
    """
    if path is None:
        return ()
    raw = load(path)
    items = raw.get("attestations", raw if isinstance(raw, list) else [])
    out = []
    for item in items:
        out.append(Attestation(
            seat_label=item["seat_label"],
            centre=(float(item["centre"][0]), float(item["centre"][1])),
            scale_px=float(item["scale_px"]),
            person_id=int(item["person_id"]),
            person_external_id=str(item["person_external_id"]),
            valid_from=_iso_to_utc(item["valid_from"]) or datetime.min.replace(
                tzinfo=UTC),
            valid_to=_iso_to_utc(item.get("valid_to")),
            decision_ref=str(item["decision_ref"]),
            attested_by_id=item.get("attested_by_id"),
            note=item.get("note"),
        ))
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qorgan classvision import",
        description=(
            "Import one classvision analysis artefact. Idempotent on run_id: importing "
            "the same file twice writes nothing the second time, and a re-analysis under "
            "different thresholds is a NEW run rather than an overwrite."
        ),
    )
    parser.add_argument("artefact", help="path to <name>.analysis.json")
    parser.add_argument("--camera-id", type=int, required=True,
                        help="the qorgan camera this recording came from (the artefact "
                             "does not know, and this is also what routes the row to a school)")
    parser.add_argument("--timezone", default=None,
                        help="the school's timezone, for the wall clock read off the "
                             "picture. In qorgan this is SCHOOL_TIMEZONE.")
    parser.add_argument("--attestations", default=None,
                        help="a seat map as JSON, for running without a database")
    parser.add_argument("--allow-unclocked", action="store_true",
                        help="import a run with no readable wall clock; it is then "
                             "excluded from everything longitudinal")
    parser.add_argument("--teacher", action="store_true",
                        help="also import the adult's position record. OFF by default: "
                             "there is no teacher-facing camera yet, and qorgan.classroom "
                             "§12.5 promised the school no teacher judgement. Turning it on "
                             "is the school's decision, and it is recorded in the notes.")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = load(args.artefact)
        attestations = load_attestations(args.attestations)
        result = plan(document, camera_id=args.camera_id, school_timezone=args.timezone, include_teacher=args.teacher,
                      attestations=attestations, allow_unclocked=args.allow_unclocked)
    except Unreadable as exc:
        print(f"could not read the artefact: {exc}", file=sys.stderr)
        return UNANSWERED
    except Refusal as exc:
        print(f"REFUSED [{exc.code}]: {exc}", file=sys.stderr)
        return REFUSED

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return IMPORTED

    _print_summary(result)
    return IMPORTED


def _print_summary(result: Plan) -> None:
    run = result.rows["classvision_runs"][0]
    seats = result.seats
    print(f"run {result.run_id}  camera {run['camera_id']}  schema {run['schema_version']}")
    print(f"  recorded   {run['started_at']}  ({run['clock_source']}, "
          f"drift {run['clock_drift_seconds']}s)")
    print(f"  window     {run['window_start_seconds']:.0f}-"
          f"{run['window_end_seconds']:.0f}s at {run['sample_fps']} fps, "
          f"{run['analysed_frames']} frames")
    print(f"  seats      {len(seats)} ({run['pupil_seats']} pupil, adult at seat "
          f"{run['adult_seat_id']})")
    print(f"  unseen     {run['observations_unassigned']} of {run['observations_total']} "
          f"observations at no seat, {run['observations_unreadable']} unreadable, "
          f"{run['seats_never_settled']} seat(s) never settled")
    print(f"  identity   {result.attested_seats} of {len(seats)} seat(s) attested")
    print()
    header = (f"  {'seat':6} {'role':7} {'cov':>5} {'hands':>5} {'stands':>6} {'away s':>6} "
              f"{'head↓ s':>7} {'index':>6}  identity")
    print(header)
    for row in seats:
        index = "—" if row["activity_index"] is None else f"{row['activity_index']:.1f}"
        who = ("место " + str(row["seat_id"]) if row["attestation_person_id"] is None
               else f"person {row['attestation_person_id']}")
        print(f"  {row['label']:6} {row['role']:7} {row['coverage']:5.3f} "
              f"{row['hand_raises']:5d} {row['stands']:6d} {row['away_seconds']:6.1f} "
              f"{row['head_down_seconds']:7.1f} {index:>6}  {who}")
    print()
    for item in result.dropped:
        print(f"  dropped: {item['what']} (seat {item['seat_id']}) — {item['why']}")
    for note in result.notes:
        print(f"  note: {note}")
    print()
    print("  каждое число выше показывается только вместе с caveats из артефакта:")
    for line in json.loads(run["caveats_json"]):
        print(f"    • {line}")


if __name__ == "__main__":
    raise SystemExit(main())

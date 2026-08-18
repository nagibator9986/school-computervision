"""The evidence standard: when a place in a room may be called a child, and when it may not.

--------------------------------------------------------------------------------
**THE STANDARD, IN FULL, BEFORE ANY CODE.**

A seat receives a name only if **every one** of these holds. Each is a gate; each gate
emits the number it tested and the number it required, per seat, into the artefact.

  1. **A human said so.** There is a seat map (`seatmap.py`), it names this seat, it was
     validated against the register (`roster.py`), and somebody's name is in its
     `attested_by` field. *No other route to a name exists in this codebase.*
  2. **The statement covers this recording.** The lesson's date falls inside the map's
     `valid_from … valid_to`. A recording whose date could not be read fails this gate
     rather than being waved through.
  3. **The named child is on the register.** Closed world. A seat map cannot invent a
     pupil, and if the run was restricted to one class it cannot borrow one from another.
  4. **The seat is a pupil's seat.** The adult's seat is refused by construction — see
     open-set rejection below.
  5. **The seat was actually occupied.** A place seen for 4 % of the lesson is not a
     child's lesson; naming it would attribute one anecdote to a person for a term.
  6. **Face evidence does not contradict the statement** — when face evidence was
     collected at all, and when there is enough of it to mean anything.

Fail any gate and the result is `NOT_ESTABLISHED` (or `CONTRADICTED`, or
`NOT_APPLICABLE`), which is a **value**, carrying the failing gate's numbers. It is not
`None`, it is not an exception, and it is not a name with a low confidence attached. The
report then says «место 3» for that seat and remains completely usable — that is the
design's fallback, not its failure mode.

--------------------------------------------------------------------------------
**WHY FACES MAY ONLY CORROBORATE OR CONTRADICT, NEVER CREATE.**

Measured on this footage (`MEASUREMENTS.md` §4): median best cosine **0.30**, median
margin over the runner-up **0.10**, faces **64 px**. ArcFace wants 0.4–0.5. Aggregating a
lesson's worth of observations (`faces.py`) improves this, but improving a coin toss
produces a better-informed coin toss, not testimony. Meanwhile a seat map is a direct
statement by someone who was in the room.

So the asymmetry is deliberate and it runs in one direction:

  * To **agree** with the human, face evidence must clear a modest bar
    (`CORROBORATE_*`). Agreement changes nothing about the name — it only upgrades
    `method` from `seat_map_attested` to `seat_map_attested+face_corroborated`, so the
    psychologist can see which seats have a second, independent line of support.
  * To **overturn** the human, it must clear a distinctly higher bar
    (`CONTRADICT_*`), and both aggregation methods — mean embedding and vote — must point
    the same way. Overturning does not produce a different name. It produces
    `CONTRADICTED`: no name at all, and a note telling the operator which line of their
    seat map to re-check. **A machine that is not allowed to name a child is not allowed
    to rename one either.**
  * `INCONCLUSIVE` is the expected outcome on this camera and is not a fault. A reader
    who sees mostly-inconclusive corroboration is seeing the measurement in §4, honestly
    reported, and not a broken matcher.

--------------------------------------------------------------------------------
**OPEN-SET REJECTION: THE ADULT IS NOT IN THE REGISTER.**

`roster.csv` lists 141 pupils. The teacher in this room is in none of them, and a matcher
that always returns its best candidate returns **a child's name for the adult**. Two
independent defences, because this failure is both the most likely and the most damaging:

  1. The adult's seat is decided upstream by `room/zones.identify_adult` and arrives here
     as `adult_seat_id`. That seat is `NOT_APPLICABLE` before any evidence is consulted.
  2. Nothing in this module can produce a name from face evidence alone, so even a
     mis-identified adult seat with a seat-map line would still require a human to have
     written a pupil into it.

--------------------------------------------------------------------------------
**THE ARTEFACT NEVER CARRIES A NAME THAT WAS NOT ESTABLISHED.**

Face evidence for a seat that did not clear the standard is written into the artefact with
its **candidate ids removed** — the scores, the margins, the vote shares stay, the names
go. This is not squeamishness. A `not_established` block containing `"best": "student_57"`
is a probable name, published, with deniability; the next person to build a dashboard will
render it, and the sentence «система думает, что это Ахметов» will be true. So the numbers
travel and the names do not, and the only names in any artefact are the ones a human
attested and every gate passed.

Correspondingly `pupil.confidence` is always `null`. There is no probability here to put
in it: identity in this system is attested or it is absent, and a number in that field
would be read as one no matter what the documentation said.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from classvision.identity.faces import AggregateMatch, Gallery
from classvision.identity.roster import Pupil, Roster
from classvision.identity.seatmap import SeatMap


class Outcome(StrEnum):
    """What identity concluded for one seat. Every value is a real answer."""

    ESTABLISHED = "established"
    NOT_ESTABLISHED = "not_established"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"


OUTCOME_RU: dict[Outcome, str] = {
    Outcome.ESTABLISHED: "личность установлена по утверждённому плану рассадки",
    Outcome.NOT_ESTABLISHED: "личность не установлена — место остаётся анонимным",
    Outcome.CONTRADICTED: ("план рассадки расходится с наблюдениями по лицу — имя не "
                           "проставлено, строку плана нужно перепроверить"),
    Outcome.NOT_APPLICABLE: "имя неприменимо (взрослый или исключённое место)",
}


class Corroboration(StrEnum):
    """What the optional face pass had to say. `NOT_ATTEMPTED` is the default everywhere."""

    NOT_ATTEMPTED = "not_attempted"        # faces switched off
    UNAVAILABLE = "unavailable"            # on, but this pupil has no usable photo
    INSUFFICIENT = "insufficient"          # on, but too few usable faces at this seat
    CORROBORATES = "corroborates"
    INCONCLUSIVE = "inconclusive"          # the expected answer on this camera
    CONTRADICTS = "contradicts"


@dataclass(frozen=True, slots=True)
class Standard:
    """Every number the standard tests against, in one object recorded into the artefact.

    Same discipline as `states.Thresholds`: a binding made under one standard and a
    binding made under another are not comparable, so the standard travels with the run
    and a change to it produces a new document.
    """

    # A seat occupied for less than this fraction of the analysed lesson is not named.
    # CHOSEN, and deliberately low: the cost of naming a rarely-occupied seat is a term
    # record built from a handful of observations, which the coverage counters make
    # visible; the cost of refusing a name to a child who was absent for half the lesson
    # is losing that child from the term entirely, which nothing makes visible. 0.10 rules
    # out only the near-phantom places that `room/seats.py` already treats as marginal.
    min_seat_occupancy: float = 0.10

    # How many usable faces a seat needs before face evidence is consulted at all. CHOSEN
    # from the arithmetic that motivates aggregation: averaging n unit vectors suppresses
    # independent noise as ~1/sqrt(n), so 30 observations buys a ~5.5x reduction — enough
    # to make a 0.10 single-frame margin mean something — while at the 10 s face sampling
    # interval it is reachable by any seat occupied for five minutes.
    face_min_observations: int = 30

    # To AGREE with the seating plan. `corroborate_score` sits at the top of the measured
    # single-frame distribution (median 0.30, p75 0.40) because the aggregate should beat
    # a single frame comfortably before it is called support. `corroborate_margin` is the
    # measured median single-frame margin: an aggregate that cannot beat one frame's
    # typical gap has added nothing. Both CHOSEN from §4's numbers, and both are
    # provisional until a lesson with a verified seating plan exists to check them on.
    corroborate_score: float = 0.35
    corroborate_margin: float = 0.10
    corroborate_vote_share: float = 0.50

    # To OVERTURN the seating plan. Higher on every axis, because the thing being
    # overturned is a human statement and the evidence doing the overturning measured 0.30.
    # `contradict_margin` is the gap by which some OTHER pupil must beat the named one —
    # 0.15 is half again the measured median margin, chosen so that ordinary matcher noise
    # cannot reach it. CHOSEN.
    contradict_margin: float = 0.15
    contradict_vote_share: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


@dataclass(frozen=True, slots=True)
class Gate:
    """One test, its measurement, its requirement, and its verdict — in the artefact.

    `measured` and `required` are separate fields rather than a formatted sentence so a
    dashboard can show «занято 6 % при пороге 10 %» in its own words, and so a reviewer
    arguing with a threshold can see the value it was tested against without reading prose.
    """

    name: str
    passed: bool
    measured: Any
    required: Any
    detail_ru: str

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.name, "passed": self.passed, "measured": self.measured,
                "required": self.required, "detail_ru": self.detail_ru}


@dataclass(frozen=True, slots=True)
class Binding:
    """One seat's identity decision, with the whole audit trail that produced it."""

    seat_id: int
    outcome: Outcome
    method: str
    gates: tuple[Gate, ...]
    gates_not_evaluated: tuple[str, ...] = ()
    external_id: str | None = None
    full_name: str | None = None
    class_name: str | None = None
    corroboration: Corroboration = Corroboration.NOT_ATTEMPTED
    face: dict[str, Any] | None = None
    disputed_external_id: str | None = None   # what the plan claimed, when contradicted
    attested_by: str | None = None
    attested_at: str | None = None

    @property
    def established(self) -> bool:
        return self.outcome is Outcome.ESTABLISHED

    @property
    def reason_ru(self) -> str:
        """The one sentence a surface shows. The first failing gate, or the success."""
        for gate in self.gates:
            if not gate.passed:
                return gate.detail_ru
        return OUTCOME_RU[self.outcome]

    def to_pupil_field(self) -> dict[str, Any]:
        """What goes into `artefact.SeatRecord.pupil`.

        Always a dict once identity has run, never a bare `None`: a seat that was examined
        and refused is a different fact from a seat nobody examined, and `pupil: null`
        cannot tell those apart. `external_id`/`full_name` are present-and-null when not
        established, so a consumer reading them gets a null rather than a KeyError and
        cannot accidentally render a half-decided name.

        `confidence` is always null. See the module docstring: there is no probability
        here, and a number in that field would be read as one.
        """
        return {
            "established": self.established,
            "outcome": str(self.outcome),
            "external_id": self.external_id,
            "full_name": self.full_name,
            "class_name": self.class_name,
            "method": self.method,
            "confidence": None,
            "reason_ru": self.reason_ru,
            "attested_by": self.attested_by,
            "attested_at": self.attested_at,
            "disputed_external_id": self.disputed_external_id,
            "corroboration": str(self.corroboration),
            "gates": [g.to_dict() for g in self.gates],
            "gates_not_evaluated": list(self.gates_not_evaluated),
            "face_evidence": self.face,
        }


# The gate names, in the order they are tested. Named constants because the web project
# groups seats by which gate stopped them, and a string typed twice is a bug that shows up
# as an empty group.
GATE_ROLE = "role_is_a_pupil_seat"
GATE_MAP_PRESENT = "seat_map_present_and_checked_against_roster"
GATE_MAP_ATTESTED = "seat_map_attested_by_a_person"
GATE_MAP_CAMERA = "seat_map_written_for_this_camera"
GATE_MAP_VALID = "seat_map_valid_on_the_recording_date"
GATE_MAP_NAMES_SEAT = "seat_map_names_this_seat"
GATE_ROSTER = "named_pupil_is_on_the_register"
GATE_OCCUPANCY = "seat_was_actually_occupied"
GATE_FACE = "face_evidence_does_not_contradict"

GATE_ORDER = (GATE_ROLE, GATE_MAP_PRESENT, GATE_MAP_ATTESTED, GATE_MAP_CAMERA,
              GATE_MAP_VALID, GATE_MAP_NAMES_SEAT, GATE_ROSTER, GATE_OCCUPANCY, GATE_FACE)


def bind(seat: Any, *, seat_map: SeatMap | None, roster: Roster | None,
         recorded_on: date | None, camera: str | None = None,
         adult_seat_id: int | None = None,
         excluded_seats: Iterable[int] = (),
         face: AggregateMatch | None = None, gallery: Gallery | None = None,
         standard: Standard | None = None) -> Binding:
    """Decide one seat, emitting the numeric reason every gate passed or failed.

    Gates are evaluated in `GATE_ORDER` and evaluation STOPS at the first failure. The
    gates that were never reached are listed by name in `gates_not_evaluated` rather than
    reported as passing — a gate that did not run has not been satisfied, and a list of
    nine green ticks where six of them were skipped is the kind of audit trail that is
    worse than none.
    """
    standard = standard or Standard()
    seat_id = int(seat.seat_id)
    occupancy = float(getattr(seat, "occupancy", 0.0) or 0.0)
    gates: list[Gate] = []

    def stop(outcome: Outcome, **extra: Any) -> Binding:
        evaluated = {g.name for g in gates}
        return Binding(
            seat_id=seat_id, outcome=outcome,
            method=("not_applicable" if outcome is Outcome.NOT_APPLICABLE
                    else "not_established"),
            gates=tuple(gates),
            gates_not_evaluated=tuple(n for n in GATE_ORDER if n not in evaluated),
            face=_redacted(face), **extra,
        )

    # -- 4. role: the adult is not in the register at all, so no evidence can apply -----
    excluded = set(int(s) for s in excluded_seats)
    is_adult = adult_seat_id is not None and seat_id == int(adult_seat_id)
    gates.append(Gate(
        name=GATE_ROLE, passed=not (is_adult or seat_id in excluded),
        measured=("взрослый" if is_adult else "исключено" if seat_id in excluded
                  else "ученик"),
        required="ученик",
        detail_ru=("Это место определено как место взрослого. Взрослых нет в реестре "
                   "учеников, поэтому имя не подбирается ни при каких данных."
                   if is_adult else
                   "Место исключено из разбора в конфигурации комнаты." if seat_id in excluded
                   else "Место разбирается как ученическое."),
    ))
    if not gates[-1].passed:
        return stop(Outcome.NOT_APPLICABLE)

    # -- 1. a human said so ------------------------------------------------------------
    has_map = seat_map is not None
    checked = has_map and roster is not None
    gates.append(Gate(
        name=GATE_MAP_PRESENT, passed=checked,
        measured=("план есть и сверен с реестром" if checked else
                  "план есть, но реестр не подключён" if has_map else "плана рассадки нет"),
        required="план рассадки, сверенный с реестром",
        detail_ru=("План рассадки не задан. Имена не проставляются: распознавание лица на "
                   "этой камере измерено (0.30 при отрыве 0.10) и не может создать имя."
                   if not has_map else
                   "План рассадки загружен без реестра учеников, поэтому проверить "
                   "указанные в нём external_id не с чем." if not checked else
                   "План рассадки загружен и сверен с реестром."),
    ))
    if not checked:
        return stop(Outcome.NOT_ESTABLISHED)
    assert seat_map is not None and roster is not None

    gates.append(Gate(
        name=GATE_MAP_ATTESTED, passed=seat_map.attested,
        measured=seat_map.attested_by or None, required="непустое поле attested_by",
        detail_ru=("В плане рассадки не указано, кто его утвердил (attested_by). "
                   "Неподписанное утверждение об именах детей не применяется."
                   if not seat_map.attested else
                   f"План рассадки утвердил: {seat_map.attested_by}"
                   + (f", {seat_map.attested_at.isoformat()}" if seat_map.attested_at else "")),
    ))
    if not gates[-1].passed:
        return stop(Outcome.NOT_ESTABLISHED)

    # The camera check fails only on a MISMATCH. When the run declares no camera there is
    # nothing to disagree with, and refusing every name because a room config is absent
    # would punish the common install for the rare mistake. The map's camera is recorded
    # in the artefact either way, so the check can be made later by a human.
    mismatch = bool(camera and seat_map.camera and camera != seat_map.camera)
    gates.append(Gate(
        name=GATE_MAP_CAMERA, passed=not mismatch,
        measured=camera or "камера в разборе не указана", required=seat_map.camera or "—",
        detail_ru=(f"План рассадки написан для камеры «{seat_map.camera}», а разбор — для "
                   f"«{camera}». Номера мест между камерами не переносятся."
                   if mismatch else
                   f"План рассадки написан для камеры «{seat_map.camera or '—'}»."),
    ))
    if mismatch:
        return stop(Outcome.NOT_ESTABLISHED)

    # -- 2. the statement covers this recording ----------------------------------------
    valid, why = seat_map.applies_on(recorded_on)
    gates.append(Gate(
        name=GATE_MAP_VALID, passed=valid,
        measured=recorded_on.isoformat() if recorded_on else None,
        required=(f"{seat_map.valid_from or '—'} … {seat_map.valid_to or '—'}"),
        detail_ru=why[0].upper() + why[1:] if why else "",
    ))
    if not valid:
        return stop(Outcome.NOT_ESTABLISHED)

    external_id = seat_map.pupil_at(seat_id)
    gates.append(Gate(
        name=GATE_MAP_NAMES_SEAT, passed=external_id is not None,
        measured=external_id, required="external_id в строке assignments",
        detail_ru=(f"Место {seat_id}: {seat_map.why_no_pupil(seat_id)}."
                   if external_id is None else
                   f"План рассадки называет для места {seat_id}: {external_id}."),
    ))
    if external_id is None:
        return stop(Outcome.NOT_ESTABLISHED)

    # -- 3. closed world ---------------------------------------------------------------
    pupil: Pupil | None = roster.get(external_id)
    gates.append(Gate(
        name=GATE_ROSTER, passed=pupil is not None, measured=external_id,
        required=("строка в реестре"
                  + (f" класса {roster.class_name}" if roster.class_name else "")),
        detail_ru=(f"Ученика «{external_id}» нет в реестре — имя не создаётся."
                   if pupil is None else
                   f"{external_id} есть в реестре ({pupil.class_name})."),
    ))
    if pupil is None:
        return stop(Outcome.NOT_ESTABLISHED)

    # -- 5. the seat was actually occupied ---------------------------------------------
    occupied = occupancy >= standard.min_seat_occupancy
    gates.append(Gate(
        name=GATE_OCCUPANCY, passed=occupied, measured=round(occupancy, 3),
        required=standard.min_seat_occupancy,
        detail_ru=(f"Место занято {occupancy * 100:.0f} % урока при пороге "
                   f"{standard.min_seat_occupancy * 100:.0f} %"
                   + (" — слишком мало, чтобы связывать его с человеком."
                      if not occupied else ".")),
    ))
    if not occupied:
        return stop(Outcome.NOT_ESTABLISHED)

    # -- 6. faces may corroborate or contradict, never create --------------------------
    verdict, gate, evidence = _weigh_faces(external_id, face, gallery, standard)
    gates.append(gate)

    attested_at = seat_map.attested_at.isoformat() if seat_map.attested_at else None
    if verdict is Corroboration.CONTRADICTS:
        return Binding(
            seat_id=seat_id, outcome=Outcome.CONTRADICTED, method="not_established",
            gates=tuple(gates), gates_not_evaluated=(),
            corroboration=verdict, face=_redacted(face),
            disputed_external_id=external_id,
            attested_by=seat_map.attested_by, attested_at=attested_at,
        )

    method = ("seat_map_attested+face_corroborated"
              if verdict is Corroboration.CORROBORATES else "seat_map_attested")
    return Binding(
        seat_id=seat_id, outcome=Outcome.ESTABLISHED, method=method,
        gates=tuple(gates), gates_not_evaluated=(),
        external_id=pupil.external_id, full_name=pupil.full_name,
        class_name=pupil.class_name, corroboration=verdict, face=evidence,
        attested_by=seat_map.attested_by, attested_at=attested_at,
    )


def _weigh_faces(external_id: str, face: AggregateMatch | None, gallery: Gallery | None,
                 standard: Standard) -> tuple[Corroboration, Gate, dict[str, Any] | None]:
    """The whole of what face evidence is permitted to do, in one place.

    Returns the verdict, the gate to record, and the evidence block to publish. The
    evidence published for an ESTABLISHED seat keeps its candidate ids — the name is
    already attested, so the numbers beside it are informative rather than suggestive.
    """
    if face is None or gallery is None:
        return Corroboration.NOT_ATTEMPTED, Gate(
            name=GATE_FACE, passed=True, measured="сверка по лицу не проводилась",
            required="не противоречит",
            detail_ru=("Сверка по лицу выключена. Имя стоит на утверждённом плане "
                       "рассадки — это основной и достаточный источник."),
        ), None

    if len(gallery) == 0 or external_id not in gallery.external_ids:
        return Corroboration.UNAVAILABLE, Gate(
            name=GATE_FACE, passed=True, measured="нет эталонного фото",
            required="не противоречит",
            detail_ru=(f"У ученика {external_id} нет пригодной фотографии в реестре, "
                       f"поэтому сверка по лицу невозможна. Имя стоит на плане рассадки."),
        ), face.to_dict()

    if face.usable < standard.face_min_observations:
        return Corroboration.INSUFFICIENT, Gate(
            name=GATE_FACE, passed=True, measured=face.usable,
            required=standard.face_min_observations,
            detail_ru=(f"Пригодных наблюдений лица {face.usable} при необходимых "
                       f"{standard.face_min_observations} — сверка не проводилась. "
                       f"Имя стоит на плане рассадки."),
        ), face.to_dict()

    mean = face.mean_match
    claimed = face.score_of(external_id)
    if mean is None or mean.best_id is None or claimed is None:
        return Corroboration.INSUFFICIENT, Gate(
            name=GATE_FACE, passed=True, measured=face.usable, required="сопоставимый вектор",
            detail_ru="Агрегированный вектор лица не сопоставился с галереей.",
        ), face.to_dict()

    # AGREES: the mean names the attested pupil, clears both absolute and relative bars,
    # and the independent vote agrees.
    if (mean.best_id == external_id and mean.best_score >= standard.corroborate_score
            and mean.margin >= standard.corroborate_margin
            and face.top_vote == external_id
            and face.top_vote_share >= standard.corroborate_vote_share):
        return Corroboration.CORROBORATES, Gate(
            name=GATE_FACE, passed=True,
            measured={"score": round(mean.best_score, 3), "margin": round(mean.margin, 3),
                      "vote_share": round(face.top_vote_share, 3), "faces": face.usable},
            required={"score": standard.corroborate_score,
                      "margin": standard.corroborate_margin,
                      "vote_share": standard.corroborate_vote_share},
            detail_ru=(f"Лицо подтверждает план: совпадение {mean.best_score:.2f} при "
                       f"отрыве {mean.margin:.2f} и доле голосов "
                       f"{face.top_vote_share * 100:.0f} % по {face.usable} наблюдениям."),
        ), face.to_dict()

    # CONTRADICTS: some other registered pupil beats the attested one by more than
    # ordinary matcher noise, on BOTH the mean and the vote.
    rival_gap = mean.best_score - claimed
    if (mean.best_id != external_id and rival_gap >= standard.contradict_margin
            and face.top_vote is not None and face.top_vote != external_id
            and face.top_vote_share >= standard.contradict_vote_share):
        return Corroboration.CONTRADICTS, Gate(
            name=GATE_FACE, passed=False,
            measured={"claimed_pupil_score": round(claimed, 3),
                      "best_other_score": round(mean.best_score, 3),
                      "gap": round(rival_gap, 3),
                      "vote_share_for_other": round(face.top_vote_share, 3),
                      "faces": face.usable},
            required={"gap_below": standard.contradict_margin,
                      "or_vote_share_below": standard.contradict_vote_share},
            detail_ru=(f"Лицо на месте устойчиво совпадает не с тем учеником, которого "
                       f"называет план: указанный получает {claimed:.2f}, другой ученик "
                       f"реестра — {mean.best_score:.2f} (разрыв {rival_gap:.2f}) и "
                       f"{face.top_vote_share * 100:.0f} % голосов по {face.usable} "
                       f"наблюдениям. Имя не проставлено; строку плана нужно "
                       f"перепроверить. Система своего имени не предлагает."),
        ), _redact_dict(face.to_dict())

    return Corroboration.INCONCLUSIVE, Gate(
        name=GATE_FACE, passed=True,
        measured={"claimed_pupil_score": round(claimed, 3),
                  "best_score": round(mean.best_score, 3),
                  "margin": round(mean.margin, 3),
                  "vote_share": round(face.top_vote_share, 3), "faces": face.usable},
        required={"to_corroborate": standard.corroborate_score,
                  "to_contradict_gap": standard.contradict_margin},
        detail_ru=(f"Сверка по лицу не дала однозначного ответа (совпадение с указанным "
                   f"учеником {claimed:.2f}, лучшее {mean.best_score:.2f}, отрыв "
                   f"{mean.margin:.2f}). Это ожидаемый результат для этой камеры и не "
                   f"является ошибкой. Имя стоит на утверждённом плане рассадки."),
    ), face.to_dict()


def _redacted(face: AggregateMatch | None) -> dict[str, Any] | None:
    return _redact_dict(face.to_dict()) if face is not None else None


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Keep every number, remove every candidate id. See the module docstring.

    This is what stops «система думает, что это Ахметов» from being technically true of a
    seat the system explicitly refused to name.
    """
    out = dict(data)
    match = out.get("mean_embedding_match")
    if isinstance(match, dict):
        out["mean_embedding_match"] = {**match, "best": None, "runner_up": None}
    out["votes"] = {"candidates": len(out.get("votes") or {}),
                    "top_count": max((out.get("votes") or {}).values(), default=0)}
    out["top_vote"] = None
    out["redacted"] = ("кандидаты скрыты: место не получило имени, а «наиболее похожий» "
                       "ученик — это и есть та вероятная догадка, которую система не "
                       "публикует")
    return out


def bind_all(seats: Sequence[Any], *, seat_map: SeatMap | None, roster: Roster | None,
             recorded_on: date | None, camera: str | None = None,
             adult_seat_id: int | None = None, excluded_seats: Iterable[int] = (),
             faces: Mapping[int, AggregateMatch] | None = None,
             gallery: Gallery | None = None,
             standard: Standard | None = None) -> dict[int, Binding]:
    """Every discovered seat, decided independently. No cross-seat inference, on purpose.

    A tempting refinement is to reason globally — if seats 2,3,4 are established then the
    remaining child must be at seat 5 — and it is refused here. That inference is only
    valid if the class list is exactly the room's occupants, which is precisely what
    nobody can attest to for a given lesson: children are absent, swap chairs, and sit in
    on other classes. A globally-consistent assignment would manufacture names for exactly
    the seats where the evidence was weakest.
    """
    standard = standard or Standard()
    return {
        int(seat.seat_id): bind(
            seat, seat_map=seat_map, roster=roster, recorded_on=recorded_on,
            camera=camera, adult_seat_id=adult_seat_id, excluded_seats=excluded_seats,
            face=(faces or {}).get(int(seat.seat_id)), gallery=gallery,
            standard=standard,
        )
        for seat in seats
    }


def report(bindings: Mapping[int, Binding], *, standard: Standard | None = None,
           seat_map: SeatMap | None = None, roster: Roster | None = None,
           gallery: Gallery | None = None,
           face_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run-level identity block for the artefact's provenance.

    Counts by outcome and, for the seats that failed, **which gate stopped them** — the
    single most actionable thing a school can be told, because every one of those gates
    has an owner: the office fills the register, the class teacher signs the plan, the
    installer names the camera.
    """
    by_outcome: dict[str, int] = {}
    stopped_at: dict[str, int] = {}
    for binding in bindings.values():
        by_outcome[str(binding.outcome)] = by_outcome.get(str(binding.outcome), 0) + 1
        for gate in binding.gates:
            if not gate.passed:
                stopped_at[gate.name] = stopped_at.get(gate.name, 0) + 1
                break

    return {
        "standard": (standard or Standard()).to_dict(),
        "seat_map": seat_map.summary() if seat_map else None,
        "roster": roster.summary() if roster else None,
        "gallery": gallery.summary() if gallery else None,
        "face_pass": face_summary,
        "seats_examined": len(bindings),
        "outcomes": by_outcome,
        "stopped_at_gate": stopped_at,
        "established_seats": sorted(s for s, b in bindings.items() if b.established),
        "note_ru": ("Имя может появиться только из утверждённого плана рассадки. "
                    "Распознавание лица на этой камере измерено (медиана 0.30 при отрыве "
                    "0.10) и допущено лишь как подтверждение или возражение, но не как "
                    "источник имени."),
    }

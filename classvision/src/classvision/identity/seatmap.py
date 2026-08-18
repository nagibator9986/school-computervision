"""The seating plan: a human being's signed statement about who sits where.

--------------------------------------------------------------------------------
**THIS FILE IS THE ONLY ROUTE BY WHICH A CHILD'S NAME MAY ENTER THIS SYSTEM.**

Nothing else in `classvision` can create a name. Not the tracker, not the face matcher,
not a heuristic about who arrives first. The reason is the measurement in
`MEASUREMENTS.md` §4: matched against the roster gallery, the median best cosine for a
face in this room was **0.30 with a 0.10 margin over the runner-up**. ArcFace wants
0.4–0.5 to call two faces the same person. A name attached to a 0.10 margin is a coin toss
with a preference, and the thing it would be attached to is a psychological record.

So identity here is **attested, not inferred**. A person who knows the class writes the
plan down once, signs it (`attested_by`), dates it (`attested_at`), and says how long it
holds (`valid_from` / `valid_to`). The system's job is to check that statement against the
register, apply it, and record whose statement it was — not to have an opinion about it.

Two consequences that look like inconveniences and are actually the point:

  * **A seat map is per camera and per term, and it expires.** `valid_to` is not optional
    decoration. Classes are re-seated after the holidays; a plan with no end date silently
    keeps naming last term's children in this term's lessons, and the resulting report is
    wrong in the one way nobody checks. A lesson recorded outside the validity window gets
    no names at all — the anonymous `место 3` report, which remains fully useful.
  * **A seat map does not make a seat a child.** `artefact.CAVEATS_RU` states that the
    unit of accumulation is a place. If two children swap chairs on Tuesday, this file
    still says what it said on Monday and their histories swap with them. That is not a
    defect this module can fix; it is a fact this module must not hide, which is why
    `attested_by`/`attested_at` travel into the artefact — «кто и когда это утверждал» is
    the only handle a psychologist has on the question.

--------------------------------------------------------------------------------
**WHY SEAT IDS AND NOT COORDINATES.**

The map keys on the seat ids `room/seats.py` discovered, not on pixel positions, because
the operator should not have to think in pixels. But those ids are assigned by reading
order over discovered clusters, so they are **stable only while the camera and the seating
are stable**: move the camera, re-run, and `seat_3` may be a different desk. That is a
real hazard, and it is handled in three ways rather than pretended away:

  1. The template written by `write_template()` carries each seat's `centre`, `scale_px`
     and `occupancy` as comments, so a human can find the seat in the picture.
  2. The map records the `camera` it was written for, and `assign.py` refuses a map whose
     camera does not match the run.
  3. The map records `discovered_from` — the run id of the analysis whose seats it was
     built from — so a later run that discovers a different number of seats can be
     compared against the one the human actually looked at.

--------------------------------------------------------------------------------
**VALIDATION RETURNS PROBLEMS; IT DOES NOT THROW THEM.**

Same discipline as `roster.py`. One bad line — a pupil id that is not in the register, the
same child written into two seats — invalidates *that assignment*, not the lesson. The bad
entries are dropped from `assignments`, kept in `rejected` with the exact reason, and
`assign.py` reports that reason against the affected seat. A file that cannot be parsed at
all is the one hard error, because then there is no statement to apply.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from classvision.identity.roster import Roster

SEATMAP_VERSION = "classvision.seatmap/1.0"


class SeatMapUnreadable(Exception):
    """The file is absent, or is not parseable as YAML/JSON. The only hard failure here."""


@dataclass(frozen=True, slots=True)
class MapIssue:
    """One rejected or questionable line of the plan."""

    kind: str
    subject: str
    detail_ru: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "subject": self.subject, "detail_ru": self.detail_ru}


KIND_UNKNOWN_PUPIL = "pupil_not_in_roster"
KIND_PUPIL_TWICE = "pupil_in_two_seats"
KIND_BAD_SEAT_ID = "seat_id_not_an_integer"
KIND_NOT_ATTESTED = "not_attested"
KIND_NO_VALIDITY = "no_validity_window"
KIND_BAD_DATE = "unreadable_date"
KIND_NO_ASSIGNMENTS = "no_assignments"
KIND_WRONG_CLASS = "pupil_from_another_class"
KIND_SEAT_NOT_FOUND = "seat_not_discovered"


@dataclass(frozen=True, slots=True)
class SeatMap:
    """One room, one camera, one term: which discovered seat holds which registered pupil."""

    camera: str
    valid_from: date | None
    valid_to: date | None
    assignments: Mapping[int, str]        # seat_id -> external_id, VALID entries only
    attested_by: str
    attested_at: date | None
    class_name: str | None = None
    discovered_from: str | None = None    # run_id of the analysis the seats came from
    notes: str = ""
    source_path: Path | None = None
    source_sha256: str | None = None
    issues: tuple[MapIssue, ...] = ()
    rejected: Mapping[int, str] = field(default_factory=dict)   # seat_id -> why, in RU

    # -- the questions `assign.py` asks -------------------------------------------------

    @property
    def attested(self) -> bool:
        """A plan nobody signed is an anonymous assertion about named children."""
        return bool(self.attested_by.strip())

    def applies_on(self, when: date | None) -> tuple[bool, str]:
        """Does this plan cover the day the lesson was recorded, and if not, why not.

        `when is None` means the recording's wall clock could not be read. That is NOT
        treated as "probably fine": a plan is a statement about a period, and a lesson of
        unknown date cannot be shown to fall inside it. The names are withheld and the
        reason says so.
        """
        if self.valid_from is None and self.valid_to is None:
            return False, "в плане рассадки не указан срок действия (valid_from/valid_to)"
        if when is None:
            return False, ("дата записи не установлена (часы на кадре не прочитаны), "
                           "поэтому нельзя проверить срок действия плана рассадки")
        if self.valid_from is not None and when < self.valid_from:
            return False, (f"запись от {when.isoformat()} раньше начала действия плана "
                           f"({self.valid_from.isoformat()})")
        if self.valid_to is not None and when > self.valid_to:
            return False, (f"запись от {when.isoformat()} позже окончания действия плана "
                           f"({self.valid_to.isoformat()})")
        return True, (f"план действует с {self.valid_from or '—'} по {self.valid_to or '—'}, "
                      f"запись от {when.isoformat()}")

    def pupil_at(self, seat_id: int) -> str | None:
        """The external_id attested for this seat, or None. Never a guess."""
        return self.assignments.get(int(seat_id))

    def why_no_pupil(self, seat_id: int) -> str:
        """The exact reason this seat has no attested pupil — dropped line, or never written."""
        return self.rejected.get(int(seat_id),
                                 "место не указано в плане рассадки")

    def summary(self) -> dict[str, Any]:
        """The provenance block. Carries who attested and when — never the pupils' names.

        The names belong to the seats that were actually established, and those are
        written per seat by `assign.py`. Duplicating the whole plan into the artefact
        would export the class list of every lesson, including for seats that produced no
        measurement at all.
        """
        return {
            "version": SEATMAP_VERSION,
            "path": str(self.source_path) if self.source_path else None,
            "sha256": self.source_sha256,
            "camera": self.camera,
            "class_name": self.class_name,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "attested_by": self.attested_by,
            "attested_at": self.attested_at.isoformat() if self.attested_at else None,
            "discovered_from": self.discovered_from,
            "assignments": len(self.assignments),
            "rejected": {str(k): v for k, v in self.rejected.items()},
            "issues": [i.to_dict() for i in self.issues],
        }


def load(path: str | Path, roster: Roster | None = None, *,
         discovered_seat_ids: Iterable[int] | None = None) -> SeatMap:
    """Read a seat map and check it against the register. Raises only if unreadable.

    `roster` is optional so the file can be linted on its own, but a map loaded without
    one has NOT been checked against the register and `assign.py` treats it as unusable:
    the closed-world guarantee — «имя может появиться только из реестра» — is exactly what
    the roster check provides, and a map that skipped it provides nothing.
    """
    path = Path(path)
    if not path.is_file():
        raise SeatMapUnreadable(f"плана рассадки нет: {path}")

    raw_text = path.read_text(encoding="utf-8")
    data = _parse(raw_text, path)
    if not isinstance(data, dict):
        raise SeatMapUnreadable(f"{path}: ожидался объект с ключами camera/assignments")

    issues: list[MapIssue] = []
    rejected: dict[int, str] = {}

    camera = str(data.get("camera") or "").strip()
    attested_by = str(data.get("attested_by") or "").strip()
    class_name = data.get("class_name") or None
    notes = str(data.get("notes") or "")
    discovered_from = data.get("discovered_from") or None

    valid_from = _as_date(data.get("valid_from"), "valid_from", issues)
    valid_to = _as_date(data.get("valid_to"), "valid_to", issues)
    attested_at = _as_date(data.get("attested_at"), "attested_at", issues)

    if not attested_by:
        issues.append(MapIssue(
            kind=KIND_NOT_ATTESTED, subject=str(path),
            detail_ru=("В плане рассадки не заполнено поле attested_by — не указано, кто "
                       "его утверждает. Имена не будут проставлены."),
        ))
    if valid_from is None and valid_to is None:
        issues.append(MapIssue(
            kind=KIND_NO_VALIDITY, subject=str(path),
            detail_ru=("Не указан срок действия плана (valid_from/valid_to). План без "
                       "срока продолжает называть детей после пересадки класса."),
        ))

    raw_assignments = data.get("assignments") or {}
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        issues.append(MapIssue(
            kind=KIND_NO_ASSIGNMENTS, subject=str(path),
            detail_ru="В плане рассадки нет ни одной заполненной строки assignments.",
        ))
        raw_assignments = {}

    discovered = set(int(s) for s in discovered_seat_ids) if discovered_seat_ids else None
    assignments: dict[int, str] = {}
    seat_of_pupil: dict[str, int] = {}

    for raw_seat, raw_pupil in raw_assignments.items():
        try:
            seat_id = int(str(raw_seat).removeprefix("seat_"))
        except ValueError:
            issues.append(MapIssue(
                kind=KIND_BAD_SEAT_ID, subject=str(raw_seat),
                detail_ru=(f"Ключ «{raw_seat}» не является номером места "
                           f"(ожидается 3 или seat_3). Строка пропущена."),
            ))
            continue

        # An empty value is the template's own default and means "this seat is not
        # claimed" -- the operator filled in the ones they were sure about. Silence is a
        # legitimate answer and must not be reported as an error.
        if raw_pupil is None or not str(raw_pupil).strip():
            continue
        external_id = str(raw_pupil).strip()

        if roster is not None and external_id not in roster:
            reason = (f"ученика «{external_id}» нет в реестре"
                      + (f" класса {roster.class_name}" if roster.class_name else ""))
            issues.append(MapIssue(kind=KIND_UNKNOWN_PUPIL, subject=external_id,
                                   detail_ru=f"Место {seat_id}: {reason}. Строка отклонена."))
            rejected[seat_id] = reason
            continue

        if roster is not None and class_name:
            pupil = roster.get(external_id)
            if pupil is not None and pupil.class_name != class_name:
                reason = (f"ученик «{external_id}» числится в классе {pupil.class_name}, "
                          f"а план заявлен для {class_name}")
                issues.append(MapIssue(kind=KIND_WRONG_CLASS, subject=external_id,
                                       detail_ru=f"Место {seat_id}: {reason}. Строка отклонена."))
                rejected[seat_id] = reason
                continue

        if external_id in seat_of_pupil:
            # One child cannot be in two places. Both entries are dropped rather than one
            # kept: there is no evidence for which of the two the operator meant, and
            # keeping the first would attach a real name to a coin flip.
            other = seat_of_pupil[external_id]
            reason = (f"ученик «{external_id}» указан и на месте {other}, и на месте "
                      f"{seat_id}")
            issues.append(MapIssue(kind=KIND_PUPIL_TWICE, subject=external_id,
                                   detail_ru=f"{reason}. Обе строки отклонены."))
            rejected[seat_id] = reason
            rejected[other] = reason
            assignments.pop(other, None)
            continue

        if discovered is not None and seat_id not in discovered:
            issues.append(MapIssue(
                kind=KIND_SEAT_NOT_FOUND, subject=f"seat_{seat_id}",
                detail_ru=(f"В плане есть место {seat_id}, но при разборе этой записи "
                           f"такое место не обнаружено (найдены: "
                           f"{', '.join(str(s) for s in sorted(discovered))}). "
                           f"Камера или рассадка изменились."),
            ))
            # Kept, not dropped: the map is a statement about the room, and a seat that
            # this particular lesson did not discover simply never comes up for binding.

        assignments[seat_id] = external_id
        seat_of_pupil[external_id] = seat_id

    import hashlib

    return SeatMap(
        camera=camera, valid_from=valid_from, valid_to=valid_to,
        assignments=assignments, attested_by=attested_by, attested_at=attested_at,
        class_name=class_name, discovered_from=discovered_from, notes=notes,
        source_path=path,
        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
        issues=tuple(issues), rejected=rejected,
    )


def _parse(text: str, path: Path) -> Any:
    """YAML if available and the extension says so, otherwise JSON.

    PyYAML is not a hard dependency. A school install that has it gets comments in its
    seat maps, which matter for a file a human edits; one that does not gets JSON, which
    every Python has. The web project imports `report/artefact.py` only and never reaches
    this code either way.
    """
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:   # pragma: no cover - depends on the install
            raise SeatMapUnreadable(
                f"{path}: план в YAML, но PyYAML не установлен ({error}). "
                f"Сохраните план в .json или установите pyyaml."
            ) from error
        try:
            return yaml.safe_load(text)
        except Exception as error:
            raise SeatMapUnreadable(f"{path}: не разбирается как YAML: {error}") from error
    try:
        return json.loads(text)
    except Exception as error:
        raise SeatMapUnreadable(f"{path}: не разбирается как JSON: {error}") from error


def _as_date(value: Any, field_name: str, issues: list[MapIssue]) -> date | None:
    """ISO dates only. A date this module had to guess at is a date it will not enforce."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        issues.append(MapIssue(
            kind=KIND_BAD_DATE, subject=field_name,
            detail_ru=(f"Поле {field_name}={value!r} не читается как дата ГГГГ-ММ-ДД. "
                       f"Поле считается незаполненным."),
        ))
        return None


# --------------------------------------------------------------------------------------
# The template
# --------------------------------------------------------------------------------------

TEMPLATE_HEADER = """\
# План рассадки — classvision
#
# Это единственный способ, которым имя ребёнка попадает в отчёт. Система сама имён не
# присваивает: распознавание лиц на этой камере измерено и признано недостаточным
# (медианное совпадение 0.30 при отрыве 0.10 — см. MEASUREMENTS.md §4).
#
# Как заполнять:
#   1. Откройте кадр записи и найдите место по координатам centre в комментарии.
#   2. Впишите external_id ученика из реестра (roster.csv), например student_57.
#   3. Места, в которых вы не уверены, ОСТАВЬТЕ ПУСТЫМИ. Пустое место останется
#      «место N» — отчёт от этого не теряет смысла. Неверное имя — теряет.
#   4. Заполните attested_by (кто утверждает) и срок действия.
#
# Внимание: единица учёта — МЕСТО, а не ребёнок. Если дети поменяются местами, их
# истории поменяются вместе с ними, и заметить это может только человек.
"""


def write_template(seats: Iterable[Any], path: str | Path, *, camera: str,
                   class_name: str | None = None,
                   valid_from: date | None = None, valid_to: date | None = None,
                   discovered_from: str | None = None,
                   adult_seat_id: int | None = None,
                   roster: Roster | None = None) -> Path:
    """Emit a stub seat map from a discovered seat list, for a human to fill in.

    Every seat gets a line with its `centre`, `scale_px` and `occupancy` in a comment,
    because the operator's actual task is "find this place in the picture" and seat ids
    alone do not help with that. The value is left empty: a template that pre-filled
    plausible names would be the exact failure this whole module exists to prevent.

    The seat the adult detector nominated is emitted commented-out with an explanation.
    The teacher is not in the pupil register (open-set), so a line for them can only ever
    be filled in wrongly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seats = sorted(seats, key=lambda s: int(getattr(s, "seat_id", 0)))

    lines = [TEMPLATE_HEADER, ""]
    lines.append(f"camera: {camera}")
    lines.append(f"class_name: {class_name or ''}"
                 + ("" if class_name else "        # класс из roster.csv, например 5-А"))
    lines.append(f"valid_from: {valid_from.isoformat() if valid_from else ''}"
                 + ("" if valid_from else "        # ГГГГ-ММ-ДД, с какого дня действует"))
    lines.append(f"valid_to: {valid_to.isoformat() if valid_to else ''}"
                 + ("" if valid_to else "          # ГГГГ-ММ-ДД, по какой день действует"))
    lines.append("attested_by:        # ФИО и должность того, кто утверждает рассадку")
    lines.append("attested_at:        # ГГГГ-ММ-ДД, дата утверждения")
    if discovered_from:
        lines.append(f"discovered_from: {discovered_from}   # run_id разбора, из которого "
                     f"взяты места")
    lines.append("notes: ''")
    lines.append("")
    lines.append("assignments:")

    for seat in seats:
        seat_id = int(seat.seat_id)
        centre = getattr(seat, "centre", (0.0, 0.0))
        scale = float(getattr(seat, "scale", 0.0) or 0.0)
        occupancy = float(getattr(seat, "occupancy", 0.0) or 0.0)
        comment = (f"# центр кадра x={centre[0]:.0f} y={centre[1]:.0f}, "
                   f"ширина плеч {scale:.0f} px, занято {occupancy * 100:.0f}% урока")
        if adult_seat_id is not None and seat_id == adult_seat_id:
            lines.append(f"  # {seat_id}:   {comment}")
            lines.append("  #   ^ это место определено как взрослый (учитель). Взрослых нет "
                         "в реестре учеников —")
            lines.append("  #     не вписывайте сюда ученика. Если определено неверно, "
                         "исправьте зону учителя в конфигурации комнаты.")
            continue
        lines.append(f"  {seat_id}:   {comment}")

    if roster is not None and roster.pupils:
        lines.append("")
        lines.append("# Ученики этого реестра (external_id — ФИО):")
        for pupil in sorted(roster.pupils, key=lambda p: p.full_name):
            lines.append(f"#   {pupil.external_id}: {pupil.full_name} ({pupil.class_name})")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def seats_from_artefact(path: str | Path) -> tuple[list[Any], dict[str, Any]]:
    """Read a written artefact back into something `write_template` can consume.

    The template is generated from a REAL run rather than from a room drawing, because the
    seat ids it must key on are the ones discovery produced on that footage. Returns the
    seats (including the adult's) and the context the template header needs.
    """
    from types import SimpleNamespace

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seats = [SimpleNamespace(seat_id=s["seat_id"], centre=tuple(s["centre"]),
                             scale=s["scale_px"], occupancy=s["occupancy"])
             for s in data.get("seats", [])]
    teacher = data.get("teacher") or {}
    adult_seat_id = teacher.get("seat_id")
    if adult_seat_id is not None:
        ledger = teacher.get("ledger") or {}
        seats.append(SimpleNamespace(
            seat_id=adult_seat_id,
            centre=tuple((teacher.get("identification", {}).get("evidence", {})
                          or {}).get("centre", (0.0, 0.0))),
            scale=(teacher.get("identification", {}).get("evidence", {})
                   or {}).get("largest_scale", 0.0),
            occupancy=ledger.get("coverage", 0.0),
        ))
    context = {
        "run_id": data.get("run_id"),
        "video_path": (data.get("provenance") or {}).get("video_path"),
        "started_at": (data.get("provenance") or {}).get("started_at"),
        "adult_seat_id": adult_seat_id,
    }
    return seats, context


def main(argv: list[str] | None = None) -> int:
    """`python -m classvision.identity.seatmap --write-template …` / `--check …`."""
    import argparse

    parser = argparse.ArgumentParser(description="План рассадки: шаблон и проверка")
    parser.add_argument("--write-template", metavar="OUT", type=Path, default=None,
                        help="создать заготовку плана рассадки")
    parser.add_argument("--from-artefact", metavar="JSON", type=Path, default=None,
                        help="взять список мест из готового артефакта разбора")
    parser.add_argument("--check", metavar="MAP", type=Path, default=None,
                        help="проверить существующий план рассадки")
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--photos", type=Path, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--valid-from", default=None)
    parser.add_argument("--valid-to", default=None)
    args = parser.parse_args(argv)

    from classvision.identity import roster as roster_module

    registry = (roster_module.load(args.roster, args.photos, class_name=args.class_name)
                if args.roster else None)

    if args.write_template:
        if not args.from_artefact:
            parser.error("--write-template требует --from-artefact")
        seats, context = seats_from_artefact(args.from_artefact)
        out = write_template(
            seats, args.write_template,
            camera=args.camera or str(context.get("video_path") or "camera"),
            class_name=args.class_name,
            valid_from=date.fromisoformat(args.valid_from) if args.valid_from else None,
            valid_to=date.fromisoformat(args.valid_to) if args.valid_to else None,
            discovered_from=context.get("run_id"),
            adult_seat_id=context.get("adult_seat_id"),
            roster=registry,
        )
        print(f"заготовка плана рассадки: {out}")
        print(f"мест в шаблоне: {len(seats)}  (место взрослого закомментировано: "
              f"{context.get('adult_seat_id')})")
        return 0

    if args.check:
        plan = load(args.check, registry)
        print(f"камера: {plan.camera or '—'}   класс: {plan.class_name or '—'}")
        print(f"срок:   {plan.valid_from or '—'} … {plan.valid_to or '—'}")
        print(f"утвердил: {plan.attested_by or '— НЕ УТВЕРЖДЁН'} "
              f"({plan.attested_at or '—'})")
        print(f"назначено мест: {len(plan.assignments)}")
        for seat_id, external_id in sorted(plan.assignments.items()):
            name = registry.get(external_id) if registry else None
            print(f"  место {seat_id}: {external_id}"
                  + (f" — {name.full_name}" if name else ""))
        for issue in plan.issues:
            print(f"  [{issue.kind}] {issue.subject}: {issue.detail_ru}")
        return 0

    parser.error("укажите --write-template или --check")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

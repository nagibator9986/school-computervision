"""Writing down that a named person decided who sits where.

**Why this writer did not exist until now, and what changed.** `places.attested_person`
carries the note: there was deliberately no way to create a `classvision_attestation`,
because the school's written answer to `docs/questions-for-school.md` §10.1 did not exist
and `decision_ref NOT NULL` is what stops one being invented. That reasoning was right and
it is not overturned here — it is *satisfied*: this command cannot run without an operator
naming the decision and naming themselves, and neither value is defaulted, derived or
guessed. The refusal has moved from "there is no code" to "the code refuses without the
document", which is the same guarantee somewhere it can actually be checked.

**A name attached to stored observations is a change to a stored observation.** That is why
`--apply-to-stored` exists and is off by default. `attested_person` is consulted AT IMPORT,
so a plan signed today does not retro-name last term's lessons by itself; making it do so is
a separate, explicit request that prints exactly how many rows it touched. The alternative —
back-filling silently on every page load — would mean last term's numbers acquiring a child's
name because a form was submitted this morning, with nothing in the record saying so.

**What this still refuses.** It will not name a place whose lessons have no date (there is no
way to know which plan applied), it will not accept a pupil from a different school, and it
will not overwrite a live attestation without `--replace` — a second plan for the same place
and period is two people claiming the same chair, and the honest response is to make somebody
say which one is true.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.db.models.classvision import (
    ClassvisionAttestation,
    ClassvisionPlace,
    ClassvisionPlaceLesson,
)
from qorgan.db.models.person import Person


class Refusal(RuntimeError):
    """A named refusal. The name is the first thing printed, so it can be looked up."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class Result:
    place_id: int
    place_label: str
    person_name: str
    valid_from: dt.date
    replaced: bool = False
    already_on_record: bool = False
    renamed_rows: int = 0
    lessons_without_a_date: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        verb = "уже записано" if self.already_on_record else "подписано"
        out = [f"{verb}: {self.place_label} → {self.person_name}, "
               f"действует с {self.valid_from.isoformat()}"]
        if self.already_on_record:
            out.append("  такая подпись уже стояла — новая запись не создавалась")
        if self.replaced:
            out.append("  прежняя подпись на этот период заменена")
        if self.renamed_rows:
            out.append(f"  имя проставлено в уже сохранённых наблюдениях: {self.renamed_rows}")
        else:
            out.append("  сохранённые наблюдения НЕ тронуты (нужен --apply-to-stored)")
        if self.lessons_without_a_date:
            out.append(f"  уроков без прочитанной даты пропущено: {self.lessons_without_a_date} "
                       "— неизвестно, какой план действовал в тот день")
        out.extend(f"  {note}" for note in self.notes)
        return out


def _place_and_person(session: Session, *, school_id: int, place_id: int,
                      external_id: str) -> tuple[ClassvisionPlace, Person]:
    """Both ends of the binding, each looked up INSIDE the school. Refuses rather than returns.

    The school filter is on both queries and not just one. A place of school A bound to a
    child of school B is the exact shape of a cross-tenant leak, and it would render as an
    ordinary page with an ordinary name on it.
    """
    place = session.scalars(
        select(ClassvisionPlace)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionPlace.id == place_id)
    ).first()
    if place is None:
        raise Refusal("no_such_place",
                      f"места №{place_id} в этой школе нет. Список: qorgan classvision status")

    person = session.scalars(
        select(Person)
        .where(Person.school_id == school_id)
        .where(Person.external_id == external_id)
    ).first()
    if person is None:
        raise Refusal("no_such_pupil",
                      f"ученика с идентификатором {external_id!r} в этой школе нет. "
                      "Реестр загружается через qorgan pupils import-roster.")
    if not person.is_active:
        raise Refusal("pupil_retired", f"{person.full_name} отмечен как выбывший.")
    return place, person


def _is_the_same_plan(row: ClassvisionAttestation, *, person: Person, valid_from: dt.date,
                      valid_to: dt.date | None, attested_by: str, decision_ref: str) -> bool:
    """Is this stored signature the one being written again, in every field?

    Re-recording an identical plan is not a conflict. It happens when a run is repeated after
    a failure, or when the operator cannot tell whether the first attempt went through.
    Refusing would push them towards `--replace`, and `--replace` on an identical plan closes
    the old row with a `valid_to` BEFORE its own `valid_from` — a period that never existed.

    Every field is compared, not just the person: a plan signed by somebody else, or on a
    different document, is a different claim about the same chair even when the child matches.
    """
    return (row.person_id == person.id and row.valid_from == valid_from
            and row.valid_to == valid_to and row.attested_by == attested_by
            and row.decision_ref == decision_ref)


def _write_signature(session: Session, *, school_id: int, place_id: int, person: Person,
                     valid_from: dt.date, valid_to: dt.date | None, attested_by: str,
                     decision_ref: str, replace: bool) -> tuple[bool, bool]:
    """Put the row in, or establish that it is already there. Returns (already, replaced)."""
    live = session.scalars(
        select(ClassvisionAttestation)
        # The caller already proved the place is this school's. Naming it again costs one
        # line and means this query is safe to read on its own, which is the only way a
        # query ever gets read.
        .join(ClassvisionPlace, ClassvisionPlace.id == ClassvisionAttestation.place_id)
        .where(ClassvisionPlace.school_id == school_id)
        .where(ClassvisionAttestation.place_id == place_id)
        .where(ClassvisionAttestation.valid_to.is_(None))
    ).all()

    already = any(_is_the_same_plan(row, person=person, valid_from=valid_from,
                                    valid_to=valid_to, attested_by=attested_by,
                                    decision_ref=decision_ref) for row in live)

    if live and not already and not replace:
        who = ", ".join(str(row.person_id) for row in live)
        raise Refusal(
            "already_attested",
            f"на это место уже есть действующая подпись (person_id {who}). Два плана на один "
            "стул — это два человека на одном месте; закройте прежнюю (--valid-to) или "
            "передайте --replace, чтобы заменить его этим.")

    replaced = False
    if not already:
        for row in live:
            closes_on = valid_from - dt.timedelta(days=1)
            if closes_on < row.valid_from:
                raise Refusal(
                    "replace_predates",
                    f"новый план начинается {valid_from.isoformat()}, а действующий — "
                    f"{row.valid_from.isoformat()}. Закрыть его {closes_on.isoformat()} значит "
                    "записать срок, который кончился раньше, чем начался. Если прежняя подпись "
                    "ошибочна, её надо снять отдельно, а не задвинуть новой.")
            row.valid_to = closes_on
            replaced = True

        session.add(ClassvisionAttestation(
            place_id=place_id, person_id=person.id, valid_from=valid_from, valid_to=valid_to,
            attested_by=attested_by,
            attested_at=dt.date.today(),  # a DATE column: the hour was never signed
            decision_ref=decision_ref,
        ))
        session.flush()
    return already, replaced


def attest(session: Session, *, school_id: int, place_id: int, external_id: str,
           attested_by: str, decision_ref: str, valid_from: dt.date,
           valid_to: dt.date | None = None, replace: bool = False,
           apply_to_stored: bool = False) -> Result:
    """Record one signed seat→pupil binding. Every argument is required for a reason."""
    place, person = _place_and_person(session, school_id=school_id, place_id=place_id,
                                      external_id=external_id)
    already, replaced = _write_signature(
        session, school_id=school_id, place_id=place_id, person=person, valid_from=valid_from,
        valid_to=valid_to, attested_by=attested_by, decision_ref=decision_ref, replace=replace)

    result = Result(place_id=place_id, place_label=place.label_ru or f"место {place.ordinal}",
                    person_name=person.full_name or external_id, valid_from=valid_from,
                    replaced=replaced, already_on_record=already)
    if apply_to_stored:
        result.renamed_rows, result.lessons_without_a_date = _apply_to_stored(
            session, school_id=school_id, place_id=place_id, person_id=person.id,
            valid_from=valid_from, valid_to=valid_to, attested_by=attested_by,
            decision_ref=decision_ref)
    return result


def _apply_to_stored(session: Session, *, school_id: int, place_id: int, person_id: int,
                     valid_from: dt.date, valid_to: dt.date | None,
                     attested_by: str, decision_ref: str) -> tuple[int, int]:
    """Name the observations already in the table, and only the ones the plan covers.

    Rows whose lesson has no date are left alone and counted: a recording that cannot be
    placed on a calendar cannot be placed on either side of a re-seating either.
    """
    from qorgan.db.models.classvision import ClassvisionLesson, ClassvisionRun

    rows = session.execute(
        select(ClassvisionPlaceLesson, ClassvisionLesson.date_local)
        .join(ClassvisionRun, ClassvisionRun.id == ClassvisionPlaceLesson.run_id)
        .join(ClassvisionLesson, ClassvisionLesson.selected_run_id == ClassvisionRun.run_id)
        .where(ClassvisionPlaceLesson.place_id == place_id)
        .where(ClassvisionLesson.school_id == school_id)
    ).all()

    named = undated = 0
    reason = (f"имя проставлено по подписанному плану рассадки: подписал {attested_by}, "
              f"основание — {decision_ref}.")
    for row, date_local in rows:
        if date_local is None:
            undated += 1
            continue
        if date_local < valid_from or (valid_to is not None and date_local > valid_to):
            continue
        row.person_id = person_id
        row.identity_method = "seat_map_attested"
        row.identity_reason = reason
        named += 1
    return named, undated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP = """\
Записать подписанный план рассадки: кто сидит на этом месте.

Это ЕДИНСТВЕННЫЙ путь, которым имя ребёнка попадает в разбор урока. Ни лицо, ни трекер имени
не создают: распознавание лица на этих камерах измерено (0,30 при отрыве 0,10) и такого
утверждения не выдерживает.

Обязательны и не подставляются по умолчанию:
  --by            кто подписал (должность или фамилия) — это акт человека, а не системы
  --decision-ref  на основании чего (номер и дата документа школы)

По умолчанию подпись действует только на НОВЫЕ разборы. Чтобы проставить имя в уже
сохранённые наблюдения, нужен явный --apply-to-stored: имя на прошлогодних числах — это
изменение сохранённого наблюдения, и оно должно быть просьбой, а не побочным эффектом.
"""


def add_attest_parser(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("attest", help="record a signed seating plan for one place",
                         description=HELP,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    cmd.add_argument("--place", type=int, required=True, metavar="N", help="place id")
    cmd.add_argument("--pupil", required=True, metavar="EXTERNAL_ID")
    cmd.add_argument("--by", required=True, help="кто подписал")
    cmd.add_argument("--decision-ref", required=True, help="основание: документ школы")
    cmd.add_argument("--valid-from", required=True, metavar="YYYY-MM-DD")
    cmd.add_argument("--valid-to", metavar="YYYY-MM-DD")
    cmd.add_argument("--replace", action="store_true", help="закрыть действующую подпись")
    cmd.add_argument("--apply-to-stored", action="store_true",
                     help="проставить имя и в уже сохранённых наблюдениях")
    cmd.add_argument("--school", metavar="SLUG")
    cmd.set_defaults(func=_cmd_attest)


def _cmd_attest(args: argparse.Namespace) -> int:
    from qorgan.classvision import _school_id
    from qorgan.db.engine import session_scope
    from qorgan.schools import list_schools

    try:
        valid_from = dt.date.fromisoformat(args.valid_from)
        valid_to = dt.date.fromisoformat(args.valid_to) if args.valid_to else None
    except ValueError as error:
        print(f"ОТКАЗ [bad_date]: {error}")
        return 2

    with session_scope() as session:
        school_id = _school_id(args)
        if school_id is None:
            schools = list_schools()
            if len(schools) != 1:
                print("ОТКАЗ [school_required]: школ несколько — укажите --school SLUG")
                return 2
            school_id = schools[0].id
        try:
            result = attest(session, school_id=school_id, place_id=args.place,
                            external_id=args.pupil, attested_by=args.by,
                            decision_ref=args.decision_ref, valid_from=valid_from,
                            valid_to=valid_to, replace=args.replace,
                            apply_to_stored=args.apply_to_stored)
        except Refusal as refusal:
            print(f"ОТКАЗ [{refusal.code}]: {refusal}")
            return 2
    for line in result.lines():
        print(line)
    return 0

"""A language model's ORIENTATION note, and the guard that decides whether it may be shown.

**The division of labour, which is the whole design.** Every quantity on these pages comes
from the deterministic report. The model is given one job it is genuinely better at than a
format string — saying, in plain language, what a reader should look at first and why — and
that job contains no numbers at all. `classvision/report/orientation.py` documents the
measurement behind that split: asked to restate quantities, a model attached a true number to
the wrong claim, which is undetectable by a reader and the one error class this system cannot
tolerate.

**So the guard refuses any digit that is not a place number this run actually has.** «место 4»
is an identifier; «сорок две минуты», «3 %», «2 раза» are measurements and are refused. The
verdict is STORED either way (`guard_passed`, `guard_offending`, `guard_reason_ru`): a
withheld note is a fact about this run, and a page that silently shows nothing teaches nobody
anything. A failed guard means the page shows the figures and the audit shows why the prose
was not shown.

The check is deliberately crude and crude in the safe direction. It is a floor, not a
ceiling: the prompt also forbids number WORDS, and no regular expression can enforce that.
What this floor guarantees is that no DIGIT reaches a reader through this table.

**Why generation is not in this module.** `google-genai` lives in the analyser's environment,
not in the web process's, and this package's whole reason for existing is that the process
serving a school over a LAN carries no heavy dependency. So the note is generated on the
analyser side (or by any operator with the key) and arrives here as a small JSON document:
`qorgan classvision reading <file.json>`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from qorgan.db.engine import session_scope
from qorgan.db.models.classvision import (ClassvisionLesson, ClassvisionPlace,
                                          ClassvisionPlaceLesson, ClassvisionReading,
                                          ClassvisionRun)
from qorgan.db.models.school import sole_school_id

# A digit is allowed only when it is a place number this run holds. The pattern swallows
# ENUMERATIONS, not just single references, because Russian writes them as «места 8 и 3» — one
# noun governing a list. The first version of this rule (upstream) matched only «мест<ending>
# <number>» and rejected a clean note over the trailing «и 3»; rejecting good notes is not a
# safe failure, because it teaches the reader to ignore the guard.
PLACE_TOKEN = re.compile(r"мест[а-яё]*\s+\d+(?:\s*(?:,|и|или)\s*\d+)*", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")
_QUANTITY = re.compile(r"\d+(?:[.,]\d+)?\s*%?")

SECTIONS = ("lesson", "place", "class_term")


class ReadingRefused(Exception):
    """Nothing was stored. The note itself is never the reason — a missing run is."""


@dataclass(frozen=True, slots=True)
class Guard:
    """Whether a note may be shown, and exactly what disqualified it."""

    passed: bool
    offending: tuple[str, ...] = ()
    allowed: tuple[int, ...] = ()

    @property
    def reason_ru(self) -> str:
        if self.passed:
            return "проверка пройдена: в записке нет величин, только номера мест."
        listed = ", ".join(repr(item) for item in self.offending)
        return (
            f"записка не показывается: в ней есть числа, которые не являются номерами мест "
            f"({listed}). Все величины берутся из таблицы, а не из текста модели."
        )


def check(text: str, allowed_places: tuple[int, ...]) -> Guard:
    """Reject every digit that is not one of this run's place numbers.

    All-or-nothing on a reference: «места 4 и 42» is not a place list with a stray quantity,
    it is a sentence whose meaning nobody can vouch for, and the safe reading of an ambiguous
    case is to surface it rather than to salvage the part that parses.
    """

    def blank_if_all_places(match: re.Match[str]) -> str:
        numbers = [int(n) for n in _NUMBER.findall(match.group(0))]
        return " " if numbers and all(n in allowed_places for n in numbers) else match.group(0)

    remaining = PLACE_TOKEN.sub(blank_if_all_places, text)
    offending = tuple(found.strip() for found in _QUANTITY.findall(remaining) if found.strip())
    return Guard(passed=not offending, offending=offending, allowed=allowed_places)


@dataclass(slots=True)
class StoredReadings:
    run_id: str
    stored: int = 0
    replaced: int = 0
    withheld: int = 0
    notes: list[str] = field(default_factory=list)

    def report_ru(self) -> list[str]:
        lines = [
            f"{self.run_id}: записок сохранено {self.stored} (перезаписано {self.replaced}), "
            f"из них не показываются {self.withheld} — не прошли проверку на числа."
        ]
        return lines + [f"  {note}" for note in self.notes]


def allowed_places(session: Any, run: ClassvisionRun, school: int) -> tuple[int, ...]:
    """Which numbers a note about THIS run may legitimately contain.

    Both vocabularies, because the pages use both: the run's own seat numbers and the room's
    stable place ordinals. A note is written from a bundle that may quote either, and a guard
    that knew only one would refuse a correct sentence.
    """
    seats = session.scalars(
        select(ClassvisionPlaceLesson.seat_id)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionPlaceLesson.run_id == run.id)
    ).all()
    ordinals = session.scalars(
        select(ClassvisionPlace.ordinal)
        .join(ClassvisionPlaceLesson, ClassvisionPlaceLesson.place_id == ClassvisionPlace.id)
        .where(ClassvisionPlace.school_id == school)
        .where(ClassvisionPlaceLesson.run_id == run.id)
    ).all()
    return tuple(sorted({int(v) for v in list(seats) + list(ordinals)}))


def store_reading(session: Any, *, run: ClassvisionRun, school: int, section: str,
                  target_key: str, body: str, source: str, model: str | None,
                  prompt_version: str, payload: dict[str, Any] | None = None,
                  generated_at: datetime | None = None) -> tuple[ClassvisionReading, bool]:
    """Store one note against one run, replacing that run's previous note for the same target.

    Keyed to the RUN, so a re-analysis cannot inherit a reading: the numbers moved, and a
    paragraph about the old ones would be a confident sentence about a measurement that no
    longer exists. Within one run a re-generated note replaces the old one — a model's second
    attempt at the same paragraph is not a second fact.
    """
    if section not in SECTIONS:
        raise ReadingRefused(f"неизвестный раздел {section!r}; допустимы {SECTIONS}")
    verdict = check(body, allowed_places(session, run, school))
    existing = session.scalars(
        select(ClassvisionReading)
        .join(ClassvisionRun, ClassvisionReading.run_id == ClassvisionRun.id)
        .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
        .where(ClassvisionLesson.school_id == school)
        .where(ClassvisionReading.run_id == run.id)
        .where(ClassvisionReading.section == section)
        .where(ClassvisionReading.target_key == target_key)
    ).first()
    row = existing or ClassvisionReading(run_id=run.id, section=section, target_key=target_key)
    row.source = source
    row.model = model
    row.prompt_version = prompt_version
    row.body = body
    row.payload = payload
    row.guard_passed = verdict.passed
    row.guard_offending = list(verdict.offending)
    row.guard_reason_ru = verdict.reason_ru
    row.generated_at = generated_at or datetime.now(UTC)
    session.add(row)
    return row, existing is not None


def load(path: Path, *, school_id: int | None = None) -> StoredReadings:
    """Store one reading document: either one note, or `{"readings": [...]}`.

    Refuses when the run is not in the database. A note is prose ABOUT a measurement, and one
    with nothing to be about is the kind of orphan that later gets read as if it described
    whatever is on the screen.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    items = document.get("readings") if isinstance(document.get("readings"), list) else [document]
    if not items:
        raise ReadingRefused("в файле нет ни одной записки")
    run_id = str(items[0].get("run_id") or document.get("run_id") or "")
    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        run = session.scalars(
            select(ClassvisionRun)
            .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
            .where(ClassvisionLesson.school_id == school)
            .where(ClassvisionRun.run_id == run_id)
        ).first()
        if run is None:
            raise ReadingRefused(
                f"прогон {run_id!r} в базе не найден: записка без разбора — абзац без предмета"
            )
        result = StoredReadings(run_id=run_id)
        for item in items:
            row, replaced = store_reading(
                session, run=run, school=school,
                section=str(item.get("section") or "lesson"),
                target_key=str(item.get("target_key") or ""),
                body=str(item.get("text") or item.get("body") or ""),
                source=str(item.get("source") or "gemini"),
                model=item.get("model"),
                prompt_version=str(item.get("prompt_version") or "orientation/1.1"),
                payload=item.get("payload"),
                generated_at=_moment(item.get("generated_at")),
            )
            result.stored += 1
            result.replaced += int(replaced)
            if not row.guard_passed:
                result.withheld += 1
                result.notes.append(f"{row.section}/{row.target_key or '—'}: {row.guard_reason_ru}")
        return result


def _moment(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(raw))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)

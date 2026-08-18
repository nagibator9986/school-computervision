"""The structured reading: a model asked for STRUCTURE, and forbidden from carrying a quantity.

The client asked for «не просто сырые данные, а полностью структурированные расшифрованные с
помощью Gemini». This is that — and the shape it takes is decided by one measurement rather
than by taste.

**The measurement.** `classvision/report/orientation.py` records what happened when a model was
asked to restate this system's quantities: it wrote «Сидячее положение зафиксировано в 96,5 %
случаев (6,0 минуты)». Both numbers existed in the bundle — 96.5 % was a share of observations,
6.0 minutes was `episode_seconds.seated` — so an existence check passed them, and the sentence
was false. No realistic guard catches that, because the error is not an invented number but a
true one attached to the wrong claim.

**So the division of labour is absolute: prose carries no quantities, and every number on the
page comes from the database beside it.** The model gets the job it is genuinely better at than
a format string — structure and salience, «на что посмотреть в первую очередь и почему» — and
the guard from `classvision.readings.check` refuses any digit that is not a place number this
run actually has. The guard is applied FIELD BY FIELD here, so one bad sentence costs one
sentence: the offending field is withheld, `payload.withheld_fields` says which and why, and the
page prints that as a stated absence. A silently shortened reading would be the same failure in
a politer form.

**Why this module imports nothing at module level.** It is the one file in the package that has
to be readable from two environments that cannot both exist: the analyser's, which has
`google-genai` and no SQLAlchemy, and the web process's, which has SQLAlchemy and must never
gain a model client. So the model half imports `google.genai` inside `ask_model`, the database
half imports the ORM inside its own functions, and the flow across the seam is a JSON document:

    (1) qorgan venv:    dump_bundles(...)      -> bundles.json   (numbers, from the database)
    (2) analyser venv:  write_document(...)    -> reading.json    (structure, from the model)
    (3) qorgan venv:    store_document(...)    -> classvision_readings (guarded, field by field)

**No key present is a stated absence, never an error.** With no `GEMINI_API_KEY` step 2 writes a
document with `available: false` and a reason; nothing is stored, no row is touched, and the
pages show the figures with «записки модели для этого урока нет». The report is complete without
it, and that is the whole reason it is allowed to be optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT_VERSION = "structured-reading/1"
SCHEMA = "classvision-reading/1"

# Reused from `classvision/report/orientation.py::VISIBILITY_CAVEAT_BELOW_PERCENT`, with its
# measurement: across both real recordings the counted episodes have a median length of 7.3 s, so
# a place seen 90 % of a 50-minute lesson is unseen for five minutes — room for forty median
# episodes. Below this line the prose must SAY the view was incomplete, and the decision is made
# here, beside the number, rather than left to the model's impression of which places felt bad.
VISIBILITY_CAVEAT_BELOW_PERCENT = 90.0

# Which counter is unmeasurable under which condition. A counter that is not measured must reach
# the model as `null` and NOT as `0`: rule 7 of the prompt below forbids saying either that the
# behaviour happened or that it did not, and a zero would invite exactly the second.
_NULL_IF_UNSETTLED = ("stands", "away_episodes")

SYSTEM_RU = """\
Ты помогаешь школьному психологу быстро сориентироваться в отчёте о наблюдении за уроком.

Тебе дают измерения по МЕСТАМ в классе. Верни СТРУКТУРУ по заданной схеме: короткий обзор,
заметку по каждому месту, где есть на что посмотреть, заметку про взрослого, список того, что
стоит проверить в самой записи, и список того, о чём по этой записи говорить нельзя.

ЖЁСТКИЕ ПРАВИЛА.

1. НИ ОДНОЙ ЦИФРЫ И НИ ОДНОГО ЧИСЛА СЛОВАМИ — ни минут, ни процентов, ни количества раз.
   Вместо «42 минуты» пиши «большую часть урока», вместо «4 раза» — «несколько раз».
   Единственное исключение — номер места («место 4»): это название, а не измерение. Числа даны
   тебе, чтобы ты понял картину; в ответе их быть не должно. Все величины читатель видит в
   таблице рядом с твоим текстом.
2. Не ставь диагнозов и не предполагай их. Никаких слов из медицины и психиатрии: ни
   «тревожность», ни «расстройство», ни «нарушение», ни «дефицит внимания».
3. Не сравнивай детей между собой и не выстраивай их по порядку. Можно сказать, что на каком-то
   месте картина отличается от остальных, но нельзя говорить, кто «лучше» или «хуже».
4. Не давай рекомендаций и не предлагай мер по отношению к детям. Ты не пишешь, что делать с
   ребёнком. В поле what_to_verify пиши только проверки САМОГО НАБЛЮДЕНИЯ: посмотреть эти минуты
   записи глазами, уточнить план рассадки, проверить, что видно камере.
5. Не говори о вовлечённости, интересе, мотивации, настроении или внимании: камера этого не
   видит. Только физически наблюдаемое — поза, положение в классе, поднятая рука, поворот
   головы, выход с места.
6. НЕ ПУТАЙ «ничего не зафиксировано» и «плохо видно». Различить можно по coverage_percent.
   Высокое coverage_percent и нули в счётчиках означают, что ребёнок спокойно сидел — так и
   пиши. Не объясняй нули плохим обзором, если coverage_percent высокий.
7. Счётчик со значением null на этой камере НЕ ИЗМЕРЯЕТСЯ вовсе: про него нельзя писать ни что
   событие было, ни что его не было. Ноль в счётчике — это «не набралось ни одного эпизода», а
   не «такого не было вовсе»; у каждого места есть zero_but_briefly_seen — список счётчиков, у
   которых ноль стоит только потому, что все включения были слишком короткими. Про такой
   счётчик нельзя писать «не было совсем», «ни разу», «всё время спокойно сидел». Допустима
   ровно одна форма — со словом «длительных»: «длительных отходов от парты не отмечено».
8. У каждого места есть needs_visibility_caveat. Если true — оговорка про неполный обзор
   обязательна, даже если у других мест обзор ещё хуже. Но НЕ ОЦЕНИВАЙ, насколько именно место
   было видно: «большую часть урока не видно» — это величина, сказанная словами. Пиши просто
   «обзор этого места был неполным» и отправляй читателя к таблице.
9. НЕ ОБЪЕДИНЯЙ несколько мест в одну фразу, если сказанное верно не для всех них. Каждому
   месту — своя заметка в поле places, с его номером в поле place.
10. Говори только о том, что есть в сводке. В ней нет ни доски, ни окна, ни двери: «повернулся
   к доске» — это утверждение о том, куда человек смотрел, а измерен только поворот головы
   относительно её обычного положения на этом месте. Пиши «повернул голову в сторону».
11. Не отправляй читателя к видеозаписи, если в сводке сказано, что кадров нет
   (frames = 0): смотреть будет нечего.
12. Простой человеческий язык, без канцелярита. overview — три-четыре предложения. Заметка по
   месту — одно-два. teacher — одно-два предложения о положении в комнате, и это не оценка
   работы учителя.

Пиши по-русски.
"""

USER_TEMPLATE_RU = """\
Вот что измерено на уроке. Числа даны тебе, чтобы ты понял картину, но в ответе их быть
не должно.

{bundle}

Заполни структуру по правилам выше.
"""

# Requested as structured output rather than parsed out of prose: a schema is the difference
# between «расшифровка» and a paragraph somebody has to read twice. Every field is a string or a
# list of strings, because the model may not produce a number in any position.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "place": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["place", "note"],
            },
        },
        "teacher": {"type": "string"},
        "what_to_verify": {"type": "array", "items": {"type": "string"}},
        "not_measured": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overview", "places", "teacher", "what_to_verify", "not_measured"],
}

# Verified working with `google-genai` in the analyser's environment. Kept beside the prompt
# version so a change to either is visible in one diff.
GEMINI_MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# The database half. Imports the ORM inside the functions -- see the module docstring.
# ---------------------------------------------------------------------------


def _place_facts(row: Any, *, board_measurable: bool, label: str, ordinal: int) -> dict[str, Any]:
    """One place, as a salience judgement needs it: how well it was seen, and what happened.

    The unmeasurable counters arrive as `null` and the ones that are zero only because every
    episode was too short arrive in `zero_but_briefly_seen`. Both were prompt rules that a model
    cannot obey without being told which case it has.
    """
    from qorgan.classvision.cabinet import COUNT_COLUMNS

    coverage_percent = round(row.coverage * 100, 1)
    counts: dict[str, int | None] = {}
    brief: list[str] = []
    discarded = (row.ledger or {}).get("discarded_short_runs") or {}
    for column in COUNT_COLUMNS:
        if column.key == "board_visits" and not board_measurable:
            counts[column.key] = None
            continue
        if column.key in _NULL_IF_UNSETTLED and not row.settled:
            counts[column.key] = None
            continue
        value = int(getattr(row, column.key) or 0)
        counts[column.key] = value
        if value == 0 and int(discarded.get(column.state, 0) or 0) > 0:
            brief.append(column.key)
    return {
        "place": ordinal, "label_ru": label, "coverage_percent": coverage_percent,
        "needs_visibility_caveat": coverage_percent < VISIBILITY_CAVEAT_BELOW_PERCENT,
        "activity_index": row.activity_index, "settled": bool(row.settled),
        "counts": counts, "zero_but_briefly_seen": brief,
    }


def bundle_for_lesson(session: Any, *, school_id: int, lesson_id: int) -> dict[str, Any] | None:
    """The numbers one reading is about. Assembled from the database, never from the artefact.

    From the database on purpose: the pages read these rows, so a note written from anything else
    could describe a lesson that is not the one on the screen.
    """
    from qorgan.classvision.cabinet import board_is_measurable, lesson_and_run

    found = lesson_and_run(session, school_id=school_id, lesson_id=lesson_id)
    if found is None:
        return None
    lesson, run = found
    board = board_is_measurable(run)
    rows = _rows_for_bundle(session, school_id=school_id, run=run)
    places = [_place_facts(row, board_measurable=board, label=label, ordinal=ordinal)
              for row, label, ordinal in rows if row.role == "pupil"]
    return {
        "schema": SCHEMA, "run_id": run.run_id, "lesson_id": lesson.id,
        # `is_demo` is deliberately NOT put in front of the model. It stays in the
        # database as the switch that deletes the synthetic term in one statement,
        # but a flag in the bundle is a flag the model writes a sentence about, and
        # the operator asked for those rows to read like any other.
        "camera_key": lesson.camera_key,
        "class_key": lesson.class_key, "date_local": str(lesson.date_local or ""),
        "lesson_minutes": round(lesson.duration_minutes, 1), "pupil_places": len(places),
        "board_measurable": board, "places": places,
        "teacher": _teacher_facts(session, school_id=school_id, run=run),
        "unmeasured": [item.get("what") for item in (run.unmeasured or [])],
        "allowed_places": sorted({place["place"] for place in places}),
    }


def _rows_for_bundle(session: Any, *, school_id: int, run: Any) -> list[tuple[Any, str, int]]:
    """Every place row of one run with its stable label and ordinal, in room order."""
    from sqlalchemy import select

    from qorgan.db.models.classvision import (ClassvisionLesson, ClassvisionPlace,
                                              ClassvisionPlaceLesson)

    found = session.execute(
        select(ClassvisionPlaceLesson, ClassvisionPlace)
        .join(ClassvisionLesson, ClassvisionPlaceLesson.lesson_id == ClassvisionLesson.id)
        .outerjoin(ClassvisionPlace, ClassvisionPlaceLesson.place_id == ClassvisionPlace.id)
        .where(ClassvisionLesson.school_id == school_id)
        .where(ClassvisionPlaceLesson.run_id == run.id)
        .order_by(ClassvisionPlaceLesson.seat_id)
    ).all()
    return [(row, place.label_ru if place else row.seat_label,
             place.ordinal if place else row.seat_id) for row, place in found]


def _teacher_facts(session: Any, *, school_id: int, run: Any) -> dict[str, Any]:
    """What may be said about the adult, and the sentence that says it is not an assessment."""
    from qorgan.classvision.cabinet import teacher_of

    teacher = teacher_of(session, school_id=school_id, run=run)
    if teacher is None:
        return {"present": False, "note_ru": "взрослого в этом разборе нет ни одной строкой"}
    return {
        "present": True,
        "board_measurable": bool(teacher.board_zone_configured),
        "board_minutes": teacher.board_minutes_of_lesson,
        "attributed_share_percent": teacher.attributed_share_of_lesson_percent,
        "at_desk_share_of_observed": (teacher.pose_metrics or {}).get("at_desk_share_of_observed"),
        "out_of_frame_share_of_lesson": (teacher.pose_metrics or {}).get(
            "out_of_frame_share_of_lesson"),
        "not_an_assessment_ru": teacher.not_an_assessment_ru,
    }


def dump_bundles(path: Path, *, school_id: int | None = None,
                 lesson_ids: list[int] | None = None) -> int:
    """Write one bundle per lesson to a file the analyser's environment can read. Returns count."""
    from sqlalchemy import select

    from qorgan.db.engine import session_scope
    from qorgan.db.models.classvision import ClassvisionLesson
    from qorgan.db.models.school import sole_school_id

    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        ids = lesson_ids or list(session.scalars(
            select(ClassvisionLesson.id)
            .where(ClassvisionLesson.school_id == school)
            .order_by(ClassvisionLesson.id)
        ))
        bundles = [bundle_for_lesson(session, school_id=school, lesson_id=one) for one in ids]
    kept = [bundle for bundle in bundles if bundle is not None]
    Path(path).write_text(
        json.dumps({"schema": SCHEMA, "bundles": kept}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return len(kept)


# ---------------------------------------------------------------------------
# The model half. Imports `google.genai` inside the function -- see the module docstring.
# ---------------------------------------------------------------------------


def ask_model(bundle: dict[str, Any], *, model: str = GEMINI_MODEL,
              timeout: float = 90.0) -> dict[str, Any]:
    """Ask for the structure. Returns `{"available": False, "reason": ...}` on any failure.

    `temperature=0` for the reason `orientation.py` gives: two runs of the same lesson that
    differ only in sampling would otherwise produce two different paragraphs about the same
    child. No provider, a timeout and a malformed answer are all the same outcome to the caller
    — a stated absence — because the deterministic report is the product and this is an aid.
    """
    import os

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"available": False, "reason": "GEMINI_API_KEY не задан"}
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=USER_TEMPLATE_RU.format(
                bundle=json.dumps(bundle, ensure_ascii=False, indent=1)),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RU, temperature=0.0,
                response_mime_type="application/json", response_schema=RESPONSE_SCHEMA,
                http_options=types.HttpOptions(timeout=int(timeout * 1000))),
        )
        payload = json.loads(response.text or "{}")
    except Exception as error:  # any provider failure is one outcome to the caller
        return {"available": False, "reason": f"обращение к модели не удалось: {error!r}"[:300]}
    if not isinstance(payload, dict) or not payload.get("overview"):
        return {"available": False, "reason": "модель вернула структуру без обзора"}
    return {"available": True, "model": model, "payload": payload}


def write_document(bundles_path: Path, out_path: Path, *,
                   model: str = GEMINI_MODEL) -> dict[str, Any]:
    """Ask the model about every bundle and write one document. Never raises on a failed call."""
    document = json.loads(Path(bundles_path).read_text(encoding="utf-8"))
    readings, refused = [], []
    for bundle in document.get("bundles") or []:
        answer = ask_model(bundle, model=model)
        if not answer.get("available"):
            refused.append({"run_id": bundle.get("run_id"), "reason": answer.get("reason")})
            continue
        readings.append({
            "schema": SCHEMA, "run_id": bundle["run_id"], "lesson_id": bundle.get("lesson_id"),
            "section": "lesson", "target_key": "", "source": "gemini",
            "model": answer.get("model"), "prompt_version": PROMPT_VERSION,
            "payload": answer["payload"],
        })
    out = {"schema": SCHEMA, "readings": readings, "refused": refused}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Storing, which is where the guard runs -- field by field.
# ---------------------------------------------------------------------------


def _guard_place_notes(payload: dict[str, Any], allowed: tuple[int, ...], labels: dict[int, str],
                       withheld: list[dict[str, str]], prose: list[str]) -> list[dict[str, Any]]:
    """The per-place notes, each guarded on its own, each labelled from the DATABASE.

    A note whose `place` is not a place of this run is dropped with a reason rather than shown
    under an invented heading: the model choosing a subject the measurement does not have is
    exactly the failure this whole design is built around.
    """
    from qorgan.classvision.readings import check

    notes: list[dict[str, Any]] = []
    for note in payload.get("places") or []:
        ordinal = note.get("place")
        text = str(note.get("note") or "").strip()
        if ordinal not in labels:
            withheld.append({"field": f"places[{ordinal}]",
                             "reason_ru": "в этом уроке такого места нет"})
            continue
        verdict = check(text, allowed)
        if not verdict.passed:
            withheld.append({"field": f"places[{ordinal}]", "reason_ru": verdict.reason_ru})
            continue
        notes.append({"place": ordinal, "label_ru": labels[ordinal], "note": text})
        prose.append(text)
    return notes


def _guard_fields(payload: dict[str, Any], allowed: tuple[int, ...],
                  labels: dict[int, str]) -> tuple[dict[str, Any], list[str]]:
    """Apply `classvision.readings.check` to every prose field. Offenders are withheld, not fixed.

    A field is dropped whole rather than edited: a sentence with its quantity deleted is a
    sentence nobody wrote, and «часть текста не показана» is a fact the reader can act on.
    Labels come from the DATABASE, so a note the model attached to a place this run does not
    have is dropped too — with a reason, because that is the model inventing a subject.
    """
    from qorgan.classvision.readings import check

    kept: dict[str, Any] = {"schema": SCHEMA}
    withheld: list[dict[str, str]] = []
    prose: list[str] = []

    for field in ("overview", "teacher"):
        text = str(payload.get(field) or "").strip()
        if not text:
            continue
        verdict = check(text, allowed)
        if not verdict.passed:
            withheld.append({"field": field, "reason_ru": verdict.reason_ru})
            continue
        kept[field] = text
        prose.append(text)

    notes = _guard_place_notes(payload, allowed, labels, withheld, prose)
    if notes:
        kept["places"] = notes

    for field in ("what_to_verify", "not_measured"):
        items = [str(item).strip() for item in (payload.get(field) or []) if str(item).strip()]
        clean = [item for item in items if check(item, allowed).passed]
        if len(clean) < len(items):
            withheld.append({"field": field,
                             "reason_ru": "часть пунктов содержала числа и не показывается"})
        if clean:
            kept[field] = clean
            prose.extend(clean)
    if withheld:
        kept["withheld_fields"] = withheld
    return kept, prose


def store_document(path: Path, *, school_id: int | None = None) -> dict[str, Any]:
    """Store a reading document. The body is every surviving sentence, so the row's guard covers all.

    `ClassvisionReading.body` is composed from the fields that passed, deliberately: the stored
    `guard_passed` then means «every sentence the page will print is clean», rather than «the
    summary paragraph was clean and nobody looked at the rest».
    """
    from sqlalchemy import select

    from qorgan.classvision.readings import ReadingRefused, allowed_places, store_reading
    from qorgan.db.engine import session_scope
    from qorgan.db.models.classvision import ClassvisionLesson, ClassvisionRun
    from qorgan.db.models.school import sole_school_id

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    report: dict[str, Any] = {"stored": 0, "skipped": [], "withheld_fields": 0}
    with session_scope() as session:
        school = school_id or sole_school_id(session.connection())
        for item in document.get("readings") or []:
            run = session.scalars(
                select(ClassvisionRun)
                .join(ClassvisionLesson, ClassvisionRun.lesson_id == ClassvisionLesson.id)
                .where(ClassvisionLesson.school_id == school)
                .where(ClassvisionRun.run_id == str(item.get("run_id")))
            ).first()
            if run is None:
                report["skipped"].append(f"{item.get('run_id')}: прогона нет в базе")
                continue
            labels = {ordinal: label for _, label, ordinal
                      in _rows_for_bundle(session, school_id=school, run=run)}
            kept, prose = _guard_fields(
                item.get("payload") or {}, allowed_places(session, run, school), labels)
            if not prose:
                report["skipped"].append(
                    f"{run.run_id}: после проверки не осталось ни одного предложения — "
                    "прежняя записка не тронута")
                continue
            report["withheld_fields"] += len(kept.get("withheld_fields") or ())
            try:
                store_reading(session, run=run, school=school, section="lesson", target_key="",
                              body="\n".join(prose), source=str(item.get("source") or "gemini"),
                              model=item.get("model"),
                              prompt_version=str(item.get("prompt_version") or PROMPT_VERSION),
                              payload=kept)
            except ReadingRefused as refused:
                report["skipped"].append(f"{run.run_id}: {refused}")
                continue
            report["stored"] += 1
    return report

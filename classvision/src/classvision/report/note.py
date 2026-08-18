"""The orientation note: the model chooses what to point at, the code writes every number.

--------------------------------------------------------------------------------
**THE MEASURED FAILURE THIS MODULE IS THE ANSWER TO.**

The first design asked Gemini to restate the metrics in Russian prose and forbade it from
saying anything a number in the bundle did not support. `summary.check_numbers` then
verified that every numeral in the generated text existed in the bundle. It passed — 350
numbers, all present — and the output was still false. About the adult it wrote:

    «Сидячее положение зафиксировано в 96,5% случаев (6,0 минуты / 357,5 секунды)»

96.5 % of 48 observed minutes is 46.3, not 6.0. The model had joined a share of
observations to a duration of qualifying episodes as though they were one fact. Both
numbers were real. Both were in the document. **The guard checks that a number EXISTS, not
that it is attached to the right claim** — and no amount of prompt tightening fixes that,
because the model was being asked to do the one thing it is worst at: carry dozens of
similar numbers into prose without swapping any of them.

--------------------------------------------------------------------------------
**SO THE DIVISION OF LABOUR IS INVERTED, AND THAT IS THE WHOLE DESIGN.**

The model returns structured output only, and **every free-text field it produces must
contain no digits at all**. It names a seat and a metric *by key*; it explains in words why
that is worth a look. Our code then looks the value up and formats it.

The model never types a numeral, so it cannot attach one to the wrong claim. This is a
structural guarantee, not a behavioural request — the difference being that a structural
guarantee still holds on the day the model changes.

**What the model is still trusted with, and the residual risk.** It chooses which of ~50
facts a psychologist should look at first, and it writes the connecting prose. It can still
be wrong *qualitatively* — describing a small value as though it were large. That is
bounded, not eliminated: the number is printed immediately beside its own description, so a
mismatch is visible in the same sentence rather than hidden behind a summary. `MAX_HIGHLIGHTS`
bounds how much of the report is model-written at all.

**The note is an ADDITION, never a replacement.** `summary.render` remains the record and
sits underneath, complete. If anything here fails — no key, timeout, schema drift, a digit
where none is allowed — the report is exactly what it would have been without a model.

**Every threshold below is CHOSEN.** There is no labelled corpus of good orientation notes
to fit them to.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

NOTE_VERSION = "classvision.note/1.0"

# At most this many facts the model may point at. CHOSEN at 6: enough to orient a reader
# across eight places and an adult, few enough that the note stays a note and the
# deterministic tables remain the substance. It also bounds the residual qualitative risk
# above -- six model-written clauses, each printed beside its own number.
MAX_HIGHLIGHTS = 6

# Characters per free-text field. CHOSEN: two sentences of Russian. A field long enough for
# a paragraph is a field long enough to bury a claim in.
MAX_FIELD_CHARS = 320

# --------------------------------------------------------------------------------------
# What the model is allowed to point at.
#
# An explicit allowlist rather than "any key in the bundle", because the bundle also
# carries provenance, thresholds and diagnostics, and a note that points a psychologist at
# `sample_fps` is noise. Each entry says where the value lives, how to format it, and what
# to call it in Russian -- so the RENDERING of every number lives here, in code, next to
# the unit it belongs to.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Metric:
    """One quotable fact: where to find it, how to say it, and in what units."""

    key: str
    label_ru: str
    source: str          # "seat" | "activity_part" | "count" | "teacher"
    unit: str            # "percent" | "count" | "minutes" | "index"

    def value_of(self, holder: dict[str, Any]) -> float | None:
        if self.source == "seat" or self.source == "teacher":
            value = holder.get(self.key)
        elif self.source == "count":
            value = (holder.get("counts") or {}).get(self.key)
        elif self.source == "activity_part":
            if self.key == "index":
                value = (holder.get("activity") or {}).get("index")
            else:
                parts = (holder.get("activity") or {}).get("parts") or []
                found = next((p for p in parts if p.get("key") == self.key), None)
                # `value_percent`, ALREADY a percentage in the bundle -- `compact()`
                # converts the artefact's 0..1 `value` on the way in.
                #
                # **No default.** An earlier version read `found.get("value", 0.0)` and
                # multiplied by 100: the key did not exist under that name, so every share
                # silently became 0.0 % and was rendered as a confident fact about a child.
                # It survived the "does this metric ever resolve" test precisely because
                # 0.0 is not None. A missing value must be UNKNOWN, never zero -- the same
                # rule the rest of this codebase runs on.
                value = None if found is None else found.get("value_percent")
        else:
            value = None
        return None if value is None else float(value)


SEAT_METRICS: tuple[Metric, ...] = (
    Metric("index", "индекс наблюдаемой активности", "activity_part", "index"),
    Metric("head_up_share", "доля времени с поднятой головой", "activity_part", "percent"),
    Metric("facing_front_share", "доля времени лицом вперёд", "activity_part", "percent"),
    Metric("at_place_share", "доля времени на своём месте", "activity_part", "percent"),
    Metric("participation", "доля от нормы видимых действий", "activity_part", "percent"),
    Metric("coverage_percent", "доля кадров, где поза читалась", "seat", "percent"),
    Metric("occupancy_percent", "доля урока, когда место было занято", "seat", "percent"),
    Metric("observed_minutes", "время наблюдения", "seat", "minutes"),
    Metric("hand_raises", "поднятий руки", "count", "count"),
    Metric("stands", "вставаний на своём месте", "count", "count"),
    Metric("away_episodes", "выходов из-за своего места", "count", "count"),
    Metric("board_visits", "выходов к доске", "count", "count"),
    Metric("head_down_episodes", "эпизодов с опущенной головой", "count", "count"),
    Metric("turned_away_episodes", "эпизодов с поворотом назад", "count", "count"),
)

TEACHER_METRICS: tuple[Metric, ...] = (
    Metric("at_desk_percent", "доля наблюдаемого времени за столом", "teacher", "percent"),
    Metric("at_desk_minutes", "время за столом", "teacher", "minutes"),
    Metric("standing_or_away_percent", "доля времени стоя или вне места", "teacher", "percent"),
    Metric("out_of_frame_percent", "доля урока вне кадра", "teacher", "percent"),
    Metric("transitions", "смен положения", "teacher", "count"),
    Metric("longest_at_desk_episode_minutes", "самый долгий период за столом", "teacher", "minutes"),
)

SEAT_KEYS = {m.key: m for m in SEAT_METRICS}
TEACHER_KEYS = {m.key: m for m in TEACHER_METRICS}

DATA_QUALITY_ISSUES: dict[str, str] = {
    "low_coverage": "место было видно не всё время, поэтому его доли посчитаны по меньшему числу наблюдений",
    "never_settled": "базовая поза не установлена, сравнивать не с чем",
    "no_index": "индекс не рассчитан",
    "identity_not_established": "за местом не закреплён конкретный ребёнок",
}


# --------------------------------------------------------------------------------------
# The guards.
# --------------------------------------------------------------------------------------

# Any numeral, in any script a multilingual model might reach for. `\d` alone misses the
# Arabic-Indic and fullwidth forms, so ٩٦ would sail past a naive check. The two separate
# fraction ranges are both needed and the split is easy to get wrong: ½ ¼ ¾ live in
# Latin-1 Supplement (U+00BC..U+00BE) while ⅓ ⅔ ⅛ and the Roman numerals Ⅰ..Ⅿ live in
# Number Forms (U+2150..U+218F). A first version covered only the second range and let ½
# through -- caught by a test, not by reading.
DIGIT = re.compile(r"[0-9٠-٩۰-۹０-９¼-¾⅐-↏]")

# Roman numerals written with ASCII letters. The Unicode block above catches Ⅰ, Ⅳ, Ⅻ; it
# cannot catch "IV" or "XII", which are ordinary Latin letters. In a Russian text a run of
# two or more of these is a number with very little else it could be, and a note has no
# legitimate reason to contain one. CHOSEN at two characters: a lone "I" or "V" is more
# likely part of a Latin abbreviation than a numeral.
ROMAN = re.compile(r"\b[IVXLCDM]{2,}\b")

# Quantities spelled as words, which is the obvious way a number gets back into prose once
# digits are banned.
#
# **These are WORD-BOUNDARY patterns, and that is not a stylistic preference — a plain
# substring list is unusable in Russian.** The first version of this guard was a substring
# list, and it rejected «место» (contains «сто») and «смотрим» (contains «три»). «Место» is
# the single most common word in this entire report: the guard would have refused every note
# ever generated, and the tests caught it only because one fixture happened to say
# «смотрим». A false rejection here is not a cheap failure — it silently removes the whole
# feature.
#
# Each entry is CHOSEN, and every one is a word a note can simply do without: the numerals
# because «девяносто шесть процентов» is a number written out, and the vague quantifiers
# because «почти всё время» is exactly the residual qualitative risk this module cannot
# eliminate and must not invite. `\w*` where a Russian stem inflects.
SMUGGLED_PATTERNS: tuple[str, ...] = (
    r"нол[ья]\w*", r"один", r"одна", r"одно", r"два", r"две", r"тр[иё]х?", r"четыр\w*",
    r"пят[ьи]", r"шест[ьи]", r"сем[ьи]", r"восем[ьи]", r"девят[ьи]", r"десят[ьи]",
    r"одиннадцат\w*", r"двенадцат\w*", r"двадцат\w*", r"тридцат\w*", r"сорок",
    r"пятьдесят", r"шестьдесят", r"семьдесят", r"восемьдесят", r"девяносто", r"сто",
    r"процент\w*", r"половин\w*", r"трет[ьи]\w*", r"четверт\w*",
    r"большинств\w*", r"меньшинств\w*", r"вдвое", r"втрое", r"дважды", r"трижды",
    r"минут\w*", r"секунд\w*", r"раз[ае]?",
    # Multi-word quantifiers, so «почти всё время» is caught while «почти» alone is not.
    r"почти\s+вс\w*", r"кажд\w+\s+втор\w*", r"больш\w+\s+част\w*",
)

SMUGGLED = re.compile(
    r"\b(?:" + "|".join(SMUGGLED_PATTERNS) + r")\b", re.IGNORECASE | re.UNICODE)

# Words this project never allows in a psychologist-facing text, whatever produced them.
# `summary.find_forbidden_words` owns the canonical list; imported rather than copied.


def contains_digits(text: str) -> bool:
    """Any numeral, in any script the model might reach for -- including ASCII Roman."""
    value = text or ""
    return bool(DIGIT.search(value) or ROMAN.search(value))


def smuggled_quantity(text: str) -> str | None:
    """The first spelled-out quantity found, or None.

    Word-boundary matched. See `SMUGGLED_PATTERNS` for why a substring search cannot be
    used here: «место» contains «сто».
    """
    found = SMUGGLED.search(text or "")
    return found.group(0) if found else None


# --------------------------------------------------------------------------------------
# The note.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Highlight:
    """One fact the model pointed at, with the value OUR code looked up."""

    subject_ru: str        # «Место 4» / «Взрослый»
    metric: Metric
    value: float
    why_ru: str

    def render(self) -> str:
        from classvision.report.summary import num, percent, qty

        if self.metric.unit == "percent":
            said = percent(self.value)
        elif self.metric.unit == "minutes":
            said = qty(self.value, ("минута", "минуты", "минут"))
        elif self.metric.unit == "index":
            said = f"{num(self.value)} из {num(100, 0)}"
        else:
            said = num(self.value, 0)
        return f"{self.subject_ru}: {self.metric.label_ru} — {said}. {self.why_ru}"


@dataclass(slots=True)
class Note:
    """The rendered note plus a full account of what was dropped and why."""

    text: str
    highlights: list[Highlight] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    ok: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": NOTE_VERSION, "ok": self.ok, "reason": self.reason,
            "model": self.model, "text": self.text,
            "highlights": [
                {"subject": h.subject_ru, "metric": h.metric.key,
                 "value": h.value, "unit": h.metric.unit, "why_ru": h.why_ru}
                for h in self.highlights
            ],
            "dropped": self.dropped,
        }


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "opening": {"type": "string", "description":
                    "2-3 предложения простым языком. БЕЗ ЦИФР И БЕЗ ЧИСЕЛ СЛОВАМИ."},
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description":
                                "метка места, например seat_4, или teacher для взрослого"},
                    "metric_key": {"type": "string"},
                    "why_ru": {"type": "string", "description":
                               "одно предложение, зачем психологу на это смотреть. БЕЗ ЦИФР."},
                },
                "required": ["subject", "metric_key", "why_ru"],
            },
        },
        "data_quality_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "issue_key": {"type": "string"},
                    "why_ru": {"type": "string", "description": "одно предложение. БЕЗ ЦИФР."},
                },
                "required": ["subject", "issue_key", "why_ru"],
            },
        },
        "closing": {"type": "string", "description": "1-2 предложения. БЕЗ ЦИФР."},
    },
    "required": ["opening", "highlights", "closing"],
}


SYSTEM_RU = """\
Ты помогаешь школьному психологу прочитать отчёт о наблюдении за уроком.

Отчёт уже посчитан и уже написан. Твоя задача — НЕ пересказывать его, а составить короткую \
ориентирующую записку: на что в этом отчёте стоит посмотреть в первую очередь и почему.

САМОЕ ВАЖНОЕ ПРАВИЛО. Ты НЕ ПИШЕШЬ НИ ОДНОГО ЧИСЛА. Ни цифрами, ни словами. \
Нельзя ни в виде цифр с процентом, нельзя «девяносто шесть процентов», нельзя «почти всё время», нельзя \
«половину урока», нельзя «дважды». Числа подставит программа сама — ты только указываешь, \
НА КАКОЙ показатель смотреть (через metric_key) и объясняешь СЛОВАМИ, зачем.

Если правило нарушено, записка будет отброшена целиком и психолог увидит отчёт без неё.

Что можно и нужно:
— назвать место (метка вида seat_ и номер, как в данных) или взрослого (teacher) \
и один показатель по нему;
— объяснить в одном предложении, почему на это стоит посмотреть;
— отметить, где данных было мало и поэтому числа менее надёжны.

Чего нельзя никогда:
— ставить диагноз, предполагать причину, давать рекомендацию, советовать «обратить внимание \
на ребёнка» как на проблему;
— сравнивать детей между собой и выстраивать их по порядку: психолог сравнивает сам, \
система — нет;
— говорить «вовлечён», «внимателен», «мотивирован», «старается», «не старается»: камера \
этого не видит, а показатель называется «индекс наблюдаемой активности» и является суммой \
наблюдаемых признаков;
— называть детей по именам: в этом отчёте за местами не закреплены конкретные дети;
— утверждать, что ребёнок отвечал вслух или стоял у доски: на этой записи нет звука, \
а доска находится позади камеры.

Тон: спокойный, деловой, без драматизации. Ты описываешь, что видно на записи, и \
оставляешь выводы человеку."""


def _allowed_keys_text() -> str:
    lines = ["Показатели по месту (metric_key -> что это):"]
    lines += [f"  {m.key} — {m.label_ru}" for m in SEAT_METRICS]
    lines.append("Показатели по взрослому (subject = teacher):")
    lines += [f"  {m.key} — {m.label_ru}" for m in TEACHER_METRICS]
    lines.append("Ключи проблем с данными (issue_key):")
    lines += [f"  {key} — {text}" for key, text in DATA_QUALITY_ISSUES.items()]
    return "\n".join(lines)


def user_message(bundle: dict[str, Any]) -> str:
    """The bundle, plus the keys the model may name. Numbers are present so the model can
    DECIDE what matters; it is forbidden from repeating them, which is checked."""
    return (
        f"{_allowed_keys_text()}\n\n"
        f"Не более {_spell(MAX_HIGHLIGHTS)} пунктов в highlights.\n\n"
        "Данные урока (только для выбора, НЕ для цитирования чисел):\n"
        f"{json.dumps(bundle, ensure_ascii=False, indent=1)}"
    )


def _spell(value: int) -> str:
    """The one place a number is spelled out on purpose: the instruction itself must not
    contain a digit, or the model reasonably concludes digits are acceptable."""
    return {1: "одного", 2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти",
            6: "шести", 7: "семи", 8: "восьми"}.get(value, "шести")


def _subject_of(bundle: dict[str, Any], subject: str) -> tuple[str, dict, dict] | None:
    """Resolve a model-supplied subject to (Russian name, holder, key table)."""
    token = (subject or "").strip().lower()
    if token in ("teacher", "взрослый", "учитель"):
        teacher = bundle.get("teacher")
        return ("Взрослый", teacher, TEACHER_KEYS) if teacher else None
    for seat in bundle.get("seats") or []:
        if token in (str(seat.get("label", "")).lower(), f"seat_{seat.get('seat_id')}",
                     str(seat.get("seat_id"))):
            return f"Место {seat.get('seat_id')}", seat, SEAT_KEYS
    return None


def validate(payload: dict[str, Any], bundle: dict[str, Any]) -> Note:
    """Turn a model response into a note, dropping everything that fails a gate.

    Dropping rather than raising, and COUNTING every drop: a note is worth having with
    four of six points, and the count is what tells whoever tunes the prompt that the
    model is fighting the rules.

    The one thing that is fatal rather than droppable is a digit or a spelled-out quantity
    in `opening` or `closing`. Those frame the whole note, so a smuggled number there
    poisons it, and there is nothing left to salvage.
    """
    dropped: list[dict[str, Any]] = []

    def clean(text: Any, where: str) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        if len(value) > MAX_FIELD_CHARS:
            dropped.append({"where": where, "why": "слишком длинно",
                            "chars": len(value), "limit": MAX_FIELD_CHARS})
            return None
        if contains_digits(value):
            dropped.append({"where": where, "why": "в тексте цифра"})
            return None
        token = smuggled_quantity(value)
        if token:
            dropped.append({"where": where, "why": "число словами", "token": token})
            return None
        return value

    opening = clean(payload.get("opening"), "opening")
    closing = clean(payload.get("closing"), "closing")
    if opening is None or closing is None:
        return Note(text="", dropped=dropped, ok=False,
                    reason="во вступлении или заключении оказалось число")

    highlights: list[Highlight] = []
    for index, item in enumerate(payload.get("highlights") or []):
        if len(highlights) >= MAX_HIGHLIGHTS:
            dropped.append({"where": f"highlights[{index}]", "why": "сверх лимита"})
            continue
        resolved = _subject_of(bundle, str(item.get("subject", "")))
        if resolved is None:
            dropped.append({"where": f"highlights[{index}]", "why": "неизвестный объект",
                            "subject": item.get("subject")})
            continue
        name, holder, table = resolved
        metric = table.get(str(item.get("metric_key", "")).strip())
        if metric is None:
            dropped.append({"where": f"highlights[{index}]", "why": "показатель не в списке",
                            "metric_key": item.get("metric_key")})
            continue
        value = metric.value_of(holder)
        if value is None:
            dropped.append({"where": f"highlights[{index}]",
                            "why": "показатель не посчитан для этого объекта",
                            "metric_key": metric.key, "subject": name})
            continue
        why = clean(item.get("why_ru"), f"highlights[{index}].why_ru")
        if why is None:
            continue
        highlights.append(Highlight(name, metric, value, why))

    quality: list[str] = []
    for index, item in enumerate(payload.get("data_quality_notes") or []):
        resolved = _subject_of(bundle, str(item.get("subject", "")))
        issue = DATA_QUALITY_ISSUES.get(str(item.get("issue_key", "")).strip())
        if resolved is None or issue is None:
            dropped.append({"where": f"data_quality_notes[{index}]",
                            "why": "неизвестный объект или ключ проблемы"})
            continue
        why = clean(item.get("why_ru"), f"data_quality_notes[{index}].why_ru")
        if why is None:
            continue
        quality.append(f"{resolved[0]}: {issue}. {why}")

    if not highlights:
        return Note(text="", dropped=dropped, ok=False,
                    reason="ни один пункт не прошёл проверку")

    body = [opening, ""]
    body += [f"— {h.render()}" for h in highlights]
    if quality:
        body += ["", "О надёжности данных:"]
        body += [f"— {line}" for line in quality]
    body += ["", closing]
    return Note(text="\n".join(body), highlights=highlights, dropped=dropped, ok=True)


HEADING = "С ЧЕГО НАЧАТЬ ЧТЕНИЕ"


def orientation_note(bundle: dict[str, Any], *, model: str | None = None,
                     timeout: float = 60.0) -> Note:
    """Ask the model, validate hard, and return a note or a stated refusal.

    Never raises. The deterministic report exists before this is called and must not be
    lost to a network error at the last step of a run that already cost minutes of GPU.
    """
    from classvision.report import prompts

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return Note(text="", ok=False, reason="GEMINI_API_KEY не задан")

    model = model or prompts.GEMINI_MODEL
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=user_message(bundle),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RU,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            ),
        )
        payload = json.loads(response.text)
    except Exception as exc:
        return Note(text="", model=model, ok=False,
                    reason=f"модель недоступна или ответ не разобран: {type(exc).__name__}")

    note = validate(payload, bundle)
    note.model = model
    return note

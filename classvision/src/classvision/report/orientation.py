"""The short orientation note: the one job an LLM is actually better at here.

--------------------------------------------------------------------------------------
**WHY THIS EXISTS, AND WHY IT IS NOT `summary.py` WITH A SHORTER PROMPT.**

We put the full artefact through Gemini once, under a prompt that forbade any claim not
backed by a number in the bundle, and measured the result. Two findings, both negative:

1. **It added nothing.** The model restated the deterministic report's numbers, at greater
   length ("18 эпизодов опускания головы (2541,3 секунды / 42,4 минуты)" for what the
   deterministic generator already wrote as "18 эпизодов с опущенной головой, суммарно
   42,4 минуты"). Constrained to state only what the numbers say, a language model can
   only paraphrase — and a paraphrase of a table is a worse table.

2. **It introduced a false sentence that the guard could not catch.** It wrote «Сидячее
   положение зафиксировано в 96,5 % случаев (6,0 минуты / 357,5 секунды)». 96.5 % of the
   48 observed minutes is 46.4 minutes; the 6.0 came from `episode_seconds.seated`, a
   different quantity that was then called by the same name. **Both numbers existed in the
   bundle, so `NumberCheck` passed them.** The guard verifies that a number is present in
   the source; it cannot verify that it is attached to the right claim, and no realistic
   guard can.

The conclusion is not "write a better prompt". It is that **asking a language model to
restate quantities puts it on the one part of the job where it is worse than a format
string, and where its errors are undetectable.** So the quantities stay entirely in the
deterministic report, and this module gives the model the job it is genuinely better at:
saying, in plain language, *what a reader should look at first and why* — a judgement about
salience, which contains no numbers at all.

--------------------------------------------------------------------------------------
**THE GUARD IS THE DESIGN, NOT AN ADD-ON.** `check(text)` rejects any output containing a
digit, with one narrow exception: a seat label the bundle itself contains, because «место
4» is an identifier and not a measurement. Everything else — durations, counts, shares,
minutes, percentages — is refused, and refusal falls back to no note at all rather than to
an unverified one. A note that cannot contain a quantity cannot misattribute one.

**What the note may and may not say.** It points at places and at kinds of observation. It
must not diagnose, must not rank pupils against each other, must not use clinical
vocabulary, and must not recommend anything. It is orientation for a professional who then
reads the table and the video — the same role a colleague plays when they say "look at the
third desk", which is useful precisely because it is not a conclusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PROMPT_VERSION = "orientation/1.1"

# Below this share of the analysed window, a place's counters must not be quoted without
# saying how much of the lesson it was actually seen. CHOSEN, from the two real recordings
# rather than from taste.
#
# The measurement: across both artefacts, 39 counted episodes have a **median length of
# 7.3 s and a mean of 14.2 s**. A place at exactly this threshold on the 50-minute camera-01
# lesson is unseen for 5.0 minutes — 300 seconds, room for about forty median episodes. So
# at 90 % the unseen slice is already large enough to hide more of a behaviour than the
# visible slice recorded, and every counter for that place is a lower bound of unknown
# tightness. The observed coverages run 61.5 – 99.6 %, and this line puts 5 of the 13 places
# on the qualified side, including D14's «место 4» at 78.8 % — the place the first version of
# this prompt described with no caveat at all while duly qualifying 61.5 % and 65.4 %,
# because nothing told it where the line was and it picked the two worst.
#
# It is deliberately far stricter than `metrics/activity.MIN_COVERAGE` (0.50), and the two
# are not competing: that one decides whether a NUMBER may exist at all, this one decides
# whether PROSE may mention it without a qualification. A note carries no numbers and no
# coverage column of its own, so the only place its reader can learn that a fifth of the
# hour is missing is the sentence itself.
VISIBILITY_CAVEAT_BELOW_PERCENT = 90.0

SYSTEM_RU = """\
Ты помогаешь школьному психологу быстро сориентироваться в отчёте о наблюдении за уроком.

Тебе дают сводку измерений по местам в классе. Твоя задача — написать короткую записку:
на что стоит посмотреть в первую очередь и почему. Это не выводы, а подсказка, куда
направить внимание человека, который дальше сам посмотрит таблицу и запись.

ЖЁСТКИЕ ПРАВИЛА.

1. НИ ОДНОЙ ЦИФРЫ. Не пиши числа ни цифрами, ни словами: ни минут, ни процентов, ни
   количества раз. Вместо «42 минуты» пиши «большую часть урока». Вместо «4 раза» —
   «несколько раз». Единственное исключение — номер места («место 4»), потому что это
   название, а не измерение.
2. Не ставь диагнозов и не предполагай их. Не используй слова из области медицины и
   психиатрии. Не пиши «тревожность», «расстройство», «нарушение», «дефицит внимания».
3. Не сравнивай детей между собой и не выстраивай их по порядку. Можно сказать, что на
   каком-то месте картина отличается от остальных, но нельзя говорить, кто «лучше».
4. Не давай рекомендаций и не предлагай мер. Ты не пишешь, что делать.
5. Не говори о вовлечённости, интересе, мотивации, настроении или внимании. Камера этого
   не видит. Говори только о том, что физически наблюдаемо: поза, положение в классе,
   поднятая рука, поворот головы, выход с места.
6. НЕ ПУТАЙ «ничего не зафиксировано» и «плохо видно». Это разные вещи, и различить их
   можно по полю coverage_percent: оно показывает, какую долю урока место вообще было видно.
   Высокое coverage_percent и нули в счётчиках означают, что ребёнок спокойно сидел — так и пиши.
   Низкое coverage_percent означает, что камера мало что видела, и вот тогда данным по этому месту
   верить нельзя. Не объясняй нули плохим обзором, если coverage_percent высокий.
7. НОЛЬ В СЧЁТЧИКЕ — ЭТО «НЕ НАБРАЛОСЬ НИ ОДНОГО ЭПИЗОДА», А НЕ «ТАКОГО НЕ БЫЛО ВОВСЕ».
   У каждого места есть поле zero_but_briefly_seen — список счётчиков, у которых ноль
   стоит только потому, что все включения оказались слишком короткими: само поведение
   там наблюдалось. Про счётчик из этого списка НЕЛЬЗЯ писать «не было совсем», «ни
   разу», «ни на одном месте», «всё время спокойно сидел».
   Отрицание без слова «длительных» запрещено точно так же: «отходов не отмечено»,
   «руку не поднимал», «наклонов не зафиксировано» читаются как «этого не происходило»,
   а происходило. Допустима ровно одна форма — с явным словом «длительных» или
   «продолжительных»: «длительных отходов от парты не отмечено». Если такая фраза в
   предложение не ложится, просто не упоминай этот счётчик.
   Счётчик со значением null не измеряется на этой камере вовсе: про него нельзя писать
   ни что событие было, ни что его не было.
8. Про плохо видимые места скажи отдельно: это влияет на доверие к их числам сильнее, чем
   сами числа. Не решай сам, какое место видно плохо: у каждого места есть поле
   needs_visibility_caveat. Если оно true — оговорка про обзор обязательна, даже если у
   других мест обзор ещё хуже. Оговорка нужна каждому такому месту, а не только худшему.
   Но НЕ ОЦЕНИВАЙ, насколько именно место было видно. «Большую часть урока не видно»,
   «видно лишь изредка», «почти всё время в кадре» — это величины, сказанные словами
   (правило 1), и они почти всегда расходятся с coverage_percent. Пиши просто «обзор
   этого места был неполным» и отправляй читателя к таблице.
9. НЕ ОБЪЕДИНЯЙ несколько мест в одну фразу, если сказанное верно не для всех них. Фраза
   «на местах 5 и 8 поднимали руку и отходили от парты» неверна, если на месте 8 отходов
   не было. Либо пиши про каждое место отдельным предложением, либо объединяй только то,
   что действительно общее. Это самая частая ошибка в таких записках.
10. Говори только о том, что есть в сводке. В ней нет ни доски, ни окна, ни двери, ни
   учителя: «повернулся к доске» или «отвернулся от доски» — это утверждение о том, куда
   человек смотрел относительно предмета в комнате, а измерен только поворот головы
   относительно её же обычного положения на этом месте. Пиши «повернул голову в сторону»,
   а не «в сторону от доски».
11. От пяти до восьми предложений. Простой человеческий язык, без канцелярита.

Пиши по-русски.
"""

USER_TEMPLATE_RU = """\
Вот что измерено на уроке. Числа даны тебе, чтобы ты понял картину, но в ответе их быть
не должно.

{bundle}

Напиши записку по правилам выше.
"""

# A digit is allowed only when it is a seat number this bundle actually holds. Anything
# else containing a digit is a quantity, and quantities are the class of error this module
# exists to make impossible.
#
# The pattern has to swallow ENUMERATIONS, not just single references, because Russian
# writes them naturally as «места 8 и 3» or «местах 4, 7 и 9» — one noun governing a list.
# The first version matched only «мест<ending> <number>` and duly rejected a perfectly
# clean note over the trailing «и 3». Rejecting good notes is not a safe failure here: it
# trains the reader to ignore the guard.
SEAT_TOKEN = re.compile(
    r"мест[а-яё]*\s+\d+(?:\s*(?:,|и|или)\s*\d+)*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class Check:
    """Whether a note is safe to show, and exactly what disqualified it."""

    passed: bool
    offending: tuple[str, ...] = ()
    allowed_seats: tuple[int, ...] = ()

    @property
    def reason_ru(self) -> str:
        if self.passed:
            return "проверка пройдена: в записке нет величин"
        listed = ", ".join(repr(o) for o in self.offending)
        return (f"записка отклонена: в ней есть числа, которые не являются номерами мест "
                f"({listed}). Показан детерминированный отчёт.")


def seat_labels(bundle: dict[str, Any]) -> tuple[int, ...]:
    """Which seat numbers this bundle legitimately contains."""
    seats = bundle.get("seats") or []
    out: list[int] = []
    for seat in seats:
        value = seat.get("seat_id")
        if isinstance(value, int):
            out.append(value)
    return tuple(sorted(set(out)))


def check(text: str, bundle: dict[str, Any]) -> Check:
    """Reject any digit that is not a seat number this bundle actually holds.

    Deliberately crude, and crude in the safe direction. A note that says «место 4» passes;
    a note that says «место 4 — сорок две минуты» passes the digit test and is caught by
    rule 1 only if the model obeyed it — which is why the prompt forbids number WORDS too
    and why this checker is a floor rather than a ceiling. The floor is what matters: the
    specific failure we measured was a digit pair joined into a false claim, and no digit
    can reach the reader through this.
    """
    allowed = seat_labels(bundle)

    def strip_seat_reference(match: re.Match[str]) -> str:
        """Blank the whole reference only if EVERY number in it is a real seat.

        All-or-nothing on purpose: «места 4 и 42» is not a seat list with a stray quantity,
        it is a sentence whose meaning we cannot vouch for, and the safe reading of an
        ambiguous case is to surface it rather than to salvage the part that parses.
        """
        numbers = [int(n) for n in _NUMBER.findall(match.group(0))]
        return " " if numbers and all(n in allowed for n in numbers) else match.group(0)

    remaining = SEAT_TOKEN.sub(strip_seat_reference, text)
    offending = tuple(o.strip() for o in re.findall(r"\d+(?:[.,]\d+)?\s*%?", remaining)
                      if o.strip())
    return Check(passed=not offending, offending=offending, allowed_seats=allowed)


@dataclass(frozen=True, slots=True)
class Note:
    """The orientation note, or the reason there is not one."""

    text: str
    available: bool
    source: str                    # "gemini" | "openai" | "none"
    model: str | None = None
    prompt_version: str = PROMPT_VERSION
    check: Check | None = None
    reason: str = ""
    bundle_keys: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "text": self.text,
            "source": self.source,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "reason": self.reason,
            "check": None if self.check is None else {
                "passed": self.check.passed,
                "offending": list(self.check.offending),
                "reason_ru": self.check.reason_ru,
            },
        }


def compact_for_note(bundle: dict[str, Any]) -> dict[str, Any]:
    """Strip the bundle to what a salience judgement needs.

    The full bundle is ~32 KB of exhaustive per-state detail. None of it helps a model
    decide what to point at, and all of it is extra surface for the model to quote back.
    What is kept is the shape of each seat: how well it was seen, its index, and the
    handful of facts a colleague would actually mention.
    """
    seats = []
    for seat in bundle.get("seats") or []:
        counts = (seat.get("counts") or {})
        seats.append({
            "seat_id": seat.get("seat_id"),
            "identified": bool(seat.get("pupil_name")),
            # `coverage_percent`, not `coverage`. The first version of this function read
            # a key that does not exist in the bundle, so every seat arrived at the model
            # with `coverage: null` -- and the model correctly answered that it could not
            # tell a quiet pupil from an unseen one. It was right, and the silent None was
            # mine. A missing key that reads as "unknown" is the politest possible version
            # of this bug and it still cost the note its most useful sentence.
            "coverage_percent": seat.get("coverage_percent"),
            # Computed here, beside the number it is about, rather than asked of the model:
            # rule 8 used to say «про плохо видимые места скажи отдельно» and left the model
            # to decide which those were, so it qualified the two worst and said nothing
            # about a place seen four fifths of the hour. A threshold a reader can look up
            # in `VISIBILITY_CAVEAT_BELOW_PERCENT` is a decision; a model's impression of
            # which places felt bad is not. `None` coverage counts as needing the caveat:
            # not knowing how much was seen is the strongest reason to say so.
            "needs_visibility_caveat": (
                seat.get("coverage_percent") is None
                or float(seat["coverage_percent"]) < VISIBILITY_CAVEAT_BELOW_PERCENT),
            "activity_index": (seat.get("activity") or {}).get("index"),
            "hand_raises": counts.get("hand_raises"),
            "stands": counts.get("stands"),
            "away_episodes": counts.get("away_episodes"),
            "head_down_episodes": counts.get("head_down_episodes"),
            "turned_away_episodes": counts.get("turned_away_episodes"),
            "board_visits": counts.get("board_visits"),
            # Which of the zeros above mean «не набралось эпизода», not «не было вовсе».
            # A caller that does not compute it sends an empty list, which reads as "every
            # zero here is a real zero" — the only safe default, and the reason this is a
            # list of the qualified counters rather than a flag on the seat.
            # `cabinet/notes.py::_zero_but_briefly_seen` explains what it costs to omit.
            "zero_but_briefly_seen": list(seat.get("zero_but_briefly_seen") or ()),
        })
    return {
        "lesson_minutes": (bundle.get("lesson") or {}).get("duration_minutes"),
        "pupil_seats": len(seats),
        "seats": seats,
        "unmeasured": [u.get("what") for u in (bundle.get("unmeasured") or [])],
    }


def _call_plain(provider: str, system: str, user: str, model: str,
                timeout: float) -> str:
    """Ask for PROSE, not a JSON object.

    `summary._call_gemini` forces structured output against the full-report schema, which
    is right for that job and wrong for this one: asked for a note, it returned a JSON
    document with `overview`/`seats`/`teacher` keys. A note is one paragraph of Russian and
    has no schema, so this asks for text and gets text.

    `temperature=0` for the same reason the summary uses it: two runs of the same lesson
    that differ only in sampling produce two different sentences about the same child.
    """
    import os

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                              or os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.0,
                http_options=types.HttpOptions(timeout=int(timeout * 1000))))
        return response.text or ""

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
    response = client.responses.create(
        model=model, temperature=0.0,
        instructions=system, input=user)
    return getattr(response, "output_text", "") or ""


def write_note(bundle: dict[str, Any], *, backend: str = "auto",
               timeout: float = 60.0) -> Note:
    """Ask a model for the note, and refuse its answer unless it is clean.

    Returns a `Note` with `available=False` and a stated reason when there is no provider,
    when the call fails, or when the guard rejects the text. Every one of those is a normal
    outcome and none of them is an error: the deterministic report is the product, and this
    is an optional aid on top of it.
    """
    import json

    from classvision.report import prompts, summary

    provider = backend
    if backend == "auto":
        # The public helper, not a private copy: one rule for what this environment
        # can reach, shared with `summary.summarise`, so the two cannot disagree.
        provider = summary.available_backend()
    if provider in (None, "", "deterministic", "none"):
        return Note("", False, "none",
                    reason="ни GEMINI_API_KEY, ни OPENAI_API_KEY не заданы")

    small = compact_for_note(bundle)
    user = USER_TEMPLATE_RU.format(bundle=json.dumps(small, ensure_ascii=False, indent=1))
    model = prompts.GEMINI_MODEL if provider == "gemini" else prompts.OPENAI_MODEL

    try:
        text = _call_plain(provider, SYSTEM_RU, user, model, timeout).strip()
    except Exception as error:  # any provider failure is the same outcome to the caller
        return Note("", False, "none", model=model,
                    reason=f"обращение к модели не удалось: {type(error).__name__}")

    verdict = check(text, bundle)
    if not verdict.passed:
        return Note("", False, "none", model=model, check=verdict,
                    reason=verdict.reason_ru)
    return Note(text, True, provider, model=model, check=verdict)

"""The Russian prompt, the output schema, and the pinned model ids — as source, not strings.

--------------------------------------------------------------------------------
**WHY THE PROMPT IS A FILE AND NOT A LITERAL INSIDE A FUNCTION.**

A prompt is a threshold. It is a chosen value that changes the output, and the artefact
already records every other chosen value that changes an output (`Provenance.thresholds`,
`WEIGHTS` in `metrics/activity.py`) precisely so that «ребёнок стал менее активен» can
never turn out to mean «мы поменяли настройку». `PROMPT_VERSION` below is written into
the summary result for the same reason, and a change to the wording is a change to the
version. A prompt buried in an f-string three calls deep gets edited by whoever is
debugging that day and nobody can afterwards say which lesson was described under which
rules.

--------------------------------------------------------------------------------
**THE GUARDRAILS ARE IN THE PROMPT AND ALSO OUTSIDE IT, BECAUSE A PROMPT IS NOT A
CONTROL.** Everything in `SYSTEM_RU` is a request. `summary.check_numbers` and
`summary.find_forbidden_words` are the enforcement, they run on the model's output
afterwards, and they reject it. The prompt exists to make rejection rare, not to make it
unnecessary. This is the same division of labour as everywhere else in the package: the
model proposes, a deterministic check disposes.

**The caveats are never generated.** `artefact.CAVEATS_RU` is appended verbatim by
`summary.py` after the model has finished, and the prompt is told not to write its own.
A model asked to restate a legal caveat will restate it slightly differently every run,
and «слегка иначе» applied to «система не ставит диагнозов» is exactly the sentence that
must not drift.

--------------------------------------------------------------------------------
**WHY STRUCTURED OUTPUT RATHER THAN FREE TEXT.**

Not for tidiness. The hallucination check has to be able to attribute an unbacked number
to a section, the per-seat paragraphs must map one-to-one onto the seats in the bundle
(a model that quietly writes seven paragraphs for eight places has dropped a child), and
the fixed caveats have to be appended in a fixed position. A JSON envelope makes all three
mechanical. The prose inside each field is still prose.

--------------------------------------------------------------------------------
**MODEL IDS ARE PINNED, AND VERIFIED ON THE DAY.**

Checked 2026-08-12 against the providers' own model lists:

* Gemini — `gemini-3.6-flash` is the current stable flash-tier text model; the SDK call is
  `client.models.generate_content(model=..., contents=..., config=types.
  GenerateContentConfig(system_instruction=..., response_mime_type="application/json",
  response_json_schema=...))`, and the JSON text comes back on `response.text`.
* OpenAI — `gpt-5.6-luna` is the small tier of the current GPT-5.6 family (GA 2026-07-09);
  the Responses API takes the schema at `text.format` (NOT `response_format`, which was
  the Chat Completions shape) and returns `response.output_text`.

Both are overridable by environment variable, because a pinned id is a fact about today
and this file will outlive today. Neither is a default that costs money silently: nothing
here runs at all unless a key is present, and on this machine none is
(`MEASUREMENTS.md` §6), so the deterministic generator is what actually runs.

**Model choice is deliberately the small tier of each family.** The task is to restate
about eighty numbers in grammatical Russian under a list of prohibitions. It is not a
reasoning task, there is no judgement to make — the judgement was made upstream by
`metrics/`, and a larger model given this job has more capacity available for inventing
context that is not in the bundle, not less.
"""

from __future__ import annotations

import os
from typing import Any

# Bump on any change to SYSTEM_RU, USER_TEMPLATE_RU or SUMMARY_JSON_SCHEMA. Recorded in
# the summary result so two lessons summarised under two wordings can be told apart.
PROMPT_VERSION = "classvision.prompt/1.0"

# Verified 2026-08-12. See the module docstring for the call shapes these ids go with.
GEMINI_MODEL = os.environ.get("CLASSVISION_GEMINI_MODEL", "gemini-3.6-flash")
OPENAI_MODEL = os.environ.get("CLASSVISION_OPENAI_MODEL", "gpt-5.6-luna")


# Word stems that must not appear in a generated report. Stems rather than words because
# Russian inflects: «тревожный / тревожность / тревожного» is one prohibition, not three.
#
# The list is short on purpose. It is not an attempt to enumerate every way of being
# judgemental — that is unbounded, and the real defences are the number check and the fact
# that the model is given nothing but counters. These are the specific registers that a
# fluent summariser reaches for by reflex when handed a table of children and a low number:
# clinical labels, ranking language, and the recommendation voice. Each one, once printed
# next to a child's seat, converts a counter into a verdict.
FORBIDDEN_STEMS_RU: tuple[str, ...] = (
    # Clinical / diagnostic register.
    "диагноз", "синдром", "сдвг", "расстройств", "патолог", "норма развития",
    "гиперактивн", "тревожн", "депресс", "аутист", "дефицит внимания", "симптом",
    # Ranking and comparison between children.
    "лучший", "лучшая", "худший", "худшая", "отстаёт", "отстает", "лидер",
    "рейтинг", "в среднем по классу", "по сравнению с одноклассник", "самый активный",
    "наименее активный", "наиболее активный",
    # The recommendation voice.
    "рекомендуем", "рекомендуется", "следует обратить внимание",
    "требует внимания", "нуждается в", "необходимо принять меры",
    # The claim the package exists to refuse.
    "вовлечённост", "вовлеченност", "вовлечён в урок", "вовлечен в урок",
    "мотивац", "заинтересован", "скучал", "устал", "не слушал", "не слушает",
)


SYSTEM_RU = """\
Ты — редактор отчётов системы видеонаблюдения за уроком. Ты пишешь по-русски для \
школьного психолога.

Ты НЕ видишь видео. Ты видишь только JSON — сводку уже посчитанных счётчиков. Твоя \
работа: превратить эти счётчики в связный текст, ничего к ним не добавив. Ты редактор, \
а не аналитик.

ЗАПРЕТЫ. Они не обсуждаются и не смягчаются формулировками.

1. Никаких диагнозов, гипотез о состоянии ребёнка, о его внимании, настроении, \
мотивации, усталости или отношении к уроку. Камера видит положение тела, направление \
головы и поднятую руку — и больше ничего. «Голова опущена» — это положение головы, а не \
скука и не усталость.
2. Никаких сравнений детей между собой, никаких рейтингов, никакого порядка «от \
активных к неактивным», никаких формулировок вроде «активнее остальных». Места \
описывай в том порядке, в каком они идут в JSON.
3. Никакой клинической лексики (диагноз, синдром, расстройство, гиперактивность, \
тревожность) — ни как утверждения, ни как предположения, ни в кавычках.
4. Никаких рекомендаций, советов, выводов о том, что делать. Отчёт заканчивается \
фактами. Решение принимает человек.
5. Ни одного числа, которого нет в JSON. Не складывай, не вычитай, не считай проценты, \
не переводи секунды в минуты — все нужные величины уже посчитаны и лежат в JSON рядом \
друг с другом. Если нужного числа нет, значит про это писать не нужно.
6. Ни одного утверждения, за которым не стоит число из JSON. Никаких «в целом класс \
работал спокойно», «атмосфера была рабочей», «урок прошёл динамично».
7. Слово «вовлечённость» и его производные запрещены. Величина называется «индекс \
наблюдаемой активности», это взвешенная сумма наблюдаемых признаков, и её всегда нужно \
показывать вместе со всеми составляющими из поля parts.
8. Единица учёта — МЕСТО в классе, а не ребёнок. Пиши «место 4», а не «ученик за \
местом 4 обычно…». Если в JSON у места есть pupil_name — можно назвать имя, но и тогда \
это место, за которым сидели.
9. Не пиши предостережений от себя: готовые формулировки будут добавлены к отчёту \
автоматически. Не пересказывай их и не сочиняй похожие.

ЧТО ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ В ТЕКСТЕ.

— Рядом с каждой итоговой величиной — её покрытие: сколько наблюдений, какая доля \
кадров, сколько раз место было пустым или поза не читалась. Число без покрытия здесь \
считается ошибкой.
— Индекс наблюдаемой активности — только вместе с разложением на составляющие \
(значение, вес, вклад каждой).
— Если индекс не посчитан (available = false) — назови причину из поля reason и не \
придумывай замену.
— Всё, что измерить не удалось, — отдельным разделом, из полей unmeasured и notes, \
своими словами, но без новых причин.

КАК ПИСАТЬ.

— Спокойный протокольный русский. Прошедшее время. Без метафор и без эпитетов.
— Числа — цифрами. Десятичный разделитель — запятая: 62,4.
— Согласуй числительные с существительными: 1 раз, 2 раза, 5 раз; 1 минута, \
2,5 минуты, 5 минут; 1 эпизод, 3 эпизода, 11 эпизодов.
— Абзац на место. Не списком, а связным текстом: психолог читает его целиком.
— Не повторяй JSON построчно. Соедини величины в предложения, но не добавляй к ним \
ничего.
"""


USER_TEMPLATE_RU = """\
Ниже — сводка одного урока в формате JSON. Напиши по ней отчёт.

Разделы, которые ты возвращаешь:
— overview: 1–2 абзаца об уроке в целом: когда записан, сколько времени \
проанализировано, сколько мест обнаружено, сколько наблюдений и сколько из них не \
удалось отнести ни к одному месту. Без выводов о классе.
— seats: по одному объекту на каждое место из массива seats, в том же порядке, с тем \
же значением label. Ни одного места не пропускай и ни одного не добавляй.
— teacher: абзац о взрослом за передним столом (поле teacher). Это описание положения \
в пространстве, а не оценка работы учителя; так и напиши. Если teacher отсутствует — \
пустая строка.
— not_measured: абзац о том, что измерить не удалось (поля unmeasured, notes, а также \
места с неполным покрытием и неустановленные личности).

JSON:
{bundle}
"""


# The envelope. Deliberately shallow: the model's job is prose, and every structural
# decision that could go wrong (which seats exist, in what order, what the caveats say) is
# made here rather than by the model.
SUMMARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overview", "seats", "teacher", "not_measured"],
    "properties": {
        "overview": {"type": "string"},
        "seats": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "text"],
                "properties": {
                    # Echoed back so the assembler can verify one paragraph per place and
                    # in the right order, instead of trusting the array position.
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
        "teacher": {"type": "string"},
        "not_measured": {"type": "string"},
    },
}


def user_message(bundle_json: str) -> str:
    """The user turn: the instructions above with the compacted bundle pasted in."""
    return USER_TEMPLATE_RU.format(bundle=bundle_json)

"""The Russian prose report — written from the numbers, and checked back against them.

--------------------------------------------------------------------------------
**WHAT THIS IS, AND THE ONE THING IT IS NOT.**

This module turns an artefact into paragraphs a school psychologist can read. Its input is
`artefact.json` — a bundle of counters that `metrics/` already computed — and *never* a
frame, a keypoint or a face. There is no model here that looks at a child. Every judgement
in this package was made upstream, under thresholds that are written into the artefact and
argued for in `states.py`; this file has no authority to add one, and structurally it
cannot: it never sees the evidence, only the counters.

That distinction is the whole reason the LLM path is safe enough to exist at all. A vision
model asked «что происходит на уроке» is an unsupervised judgement about children. A text
model asked to restate «индекс 62,4; голова поднята 80,4 %; рука поднималась 0 раз» in
grammatical Russian is a typesetting job with a fact-checker bolted to its output.

--------------------------------------------------------------------------------
**THE DETERMINISTIC GENERATOR IS THE PRODUCT. THE LLM IS THE OPTIONAL EXTRA.**

`MEASUREMENTS.md` §6: no `GEMINI_API_KEY` and no `OPENAI_API_KEY` exist on the machine
this has to run on, and a school is not a place where one reliably appears. So `render()`
is not a stub that apologises for a missing key — it is a full report: an overview, a
paragraph per place, the adult, an explicit «что измерить не удалось», and the caveats.
It is what the psychologist actually reads, and it is written to that standard.

It also has two properties the LLM path can never have. It cannot invent a number, because
it only ever prints values it looked up. And it is reproducible: the same artefact yields
the same text forever, so «в прошлый раз формулировка была другая» is never a question
about the child.

The price is the Russian. A generator that concatenates strings produces «2 раз» and
«5 минута» unless somebody does the grammar properly, and a report that cannot inflect a
numeral reads as machine output and gets trusted accordingly. Hence `plural_form()` and
`qty()` below, and hence the fact that every duration in this file goes through them.

--------------------------------------------------------------------------------
**THE HALLUCINATION CHECK IS THE LOAD-BEARING PART.**

`check_numbers()` extracts every numeral from a generated text and requires each one to be
a correct rounding of some number that exists in the bundle. If any is not, the text is
REJECTED — not annotated, not flagged for review — and the deterministic text is returned
in its place. `Summary.source` says which one you got and `Summary.fallback_reason` says
why.

The rule is deliberately "is a correct rounding of", not "appears verbatim". «62,4»,
«62» and «около 62» are all faithful renderings of 62.4, and a checker that rejected them
would be a checker nobody keeps switched on. A tolerance of half a unit in the last
printed digit is exactly the set of strings that a human sub-editor would also accept.
CHOSEN, and it is the only tolerance in this file.

Two known limits, stated rather than hidden:

* **Numbers spelled as words.** «два раза» is a number. `WORD_NUMERALS_RU` catches the
  common forms from «два» upward. It deliberately omits «один/одна/одно», because in
  Russian those double as an indefinite article («одна из составляющих») and treating
  them as numerals produces false rejections of correct text. The prompt therefore
  requires digits, and this gap is the reason it does.
* **A false statement containing no number passes.** «Место 4 вело себя беспокойно» has
  nothing to check. That is what `FORBIDDEN_STEMS_RU` and the prompt's prohibitions are
  for, and it is why the LLM path is optional rather than default. The number check
  bounds the damage; it does not eliminate the category.

--------------------------------------------------------------------------------
**WHY THERE IS NO CLASS AGGREGATE ANYWHERE IN THIS FILE.**

`compact()` could trivially emit the mean, the range and the ordering of the activity
index across the class, and every one of those would be a ranking of children — the thing
`metrics/trend.py` refuses at length and for the same reason: a pupil at the back is 70 px
of shoulder and one at the front is 220 px, so an ordering of the index is substantially
an ordering of how well the camera sees each seat. The bundle carries how many seats got
an index and how many were refused one, and nothing that lets a sentence say which child
came top. A number that is not in the bundle cannot be written, so the omission is not a
style guideline — it is enforced by the same checker as everything else.

--------------------------------------------------------------------------------
**WHY THE STATE LABELS ARE RE-DECLARED HERE AND NOT IMPORTED FROM `states.py`.**

`states.RU_LABELS` is the same information. Importing it would pull `classvision.states`
→ `classvision.geometry` → `numpy` into a module whose entire job is to format strings,
and `artefact.py`'s contract is that the web project imports the artefact and nothing
else. The keys below (`"head_down"`, `"turned_away"`, …) are the artefact's own JSON keys
and are therefore part of that contract already; the labels are the reader-facing wording
for them, which belongs on the reader-facing surface.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from classvision.report import prompts
from classvision.report.artefact import CAVEATS_RU

SUMMARY_VERSION = "classvision.summary/1.0"

# The artefact's own state keys, in the order a paragraph should mention them, with the
# wording the psychologist sees. See the module docstring for why these are not imported.
STATE_LABELS_RU: dict[str, str] = {
    "away_from_place": "вне своего места",
    "at_board": "у доски",
    "stood_up": "стоял на своём месте",
    "hand_raised": "с поднятой рукой",
    "head_down": "с опущенной головой",
    "turned_away": "отвернувшись назад",
    "seated": "сидя на месте",
    "unknown": "поза не читалась",
}

# The counter keys of `ledger.counts`, each with its three Russian forms and the matching
# key in `ledger.seconds` (or None where the counter has no duration).
#
# The forms are written out rather than derived by suffixing. A first attempt generated
# them — label, label + "а", label + "ов" — and produced «2 поднятие рукиа» and «6
# вставание на своём местй», because the head noun of a Russian phrase is not its last
# word and its stem is not its prefix. There are six counters in this package; there is no
# version of this where a rule is cheaper than the table.
EVENT_LABELS_RU: tuple[tuple[str, tuple[str, str, str], str | None], ...] = (
    ("hand_raises",
     ("поднятие руки", "поднятия руки", "поднятий руки"), "hand_raised"),
    ("stands",
     ("вставание на своём месте", "вставания на своём месте",
      "вставаний на своём месте"), "stood_up"),
    ("away_episodes",
     ("выход из-за своего места", "выхода из-за своего места",
      "выходов из-за своего места"), "away_from_place"),
    ("board_visits",
     ("выход к доске", "выхода к доске", "выходов к доске"), "at_board"),
    ("head_down_episodes",
     ("эпизод с опущенной головой", "эпизода с опущенной головой",
      "эпизодов с опущенной головой"), "head_down"),
    ("turned_away_episodes",
     ("эпизод с поворотом назад", "эпизода с поворотом назад",
      "эпизодов с поворотом назад"), "turned_away"),
)

MONTHS_RU_GENITIVE: tuple[str, ...] = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
WEEKDAYS_RU: tuple[str, ...] = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)

CLOCK_SOURCE_RU: dict[str, str] = {
    "overlay": "время считано с таймкода, наложенного камерой на кадр",
    "filename": "время взято из имени файла",
    "manual": "время указано человеком вручную",
    "unknown": "время начала записи неизвестно",
}


# --------------------------------------------------------------------------------------
# Russian number agreement.
# --------------------------------------------------------------------------------------

MINUTES = ("минута", "минуты", "минут")
SECONDS = ("секунда", "секунды", "секунд")
TIMES = ("раз", "раза", "раз")
OBSERVATIONS = ("наблюдение", "наблюдения", "наблюдений")
# The prepositional case, for «в 1 наблюдении» / «в 148 наблюдениях». Russian marks the
# case of a counted noun from the preposition governing it, so a single set of forms per
# noun is not enough and «в 1 наблюдение» is simply ungrammatical. Only the two cases this
# report actually uses are declared.
OBSERVATIONS_IN = ("наблюдении", "наблюдениях", "наблюдениях")
FRAMES = ("кадр", "кадра", "кадров")
PUPIL_PLACES = ("ученическое место", "ученических места", "ученических мест")
# The GENITIVE of the same thing, for «ни за одним ИЗ …». `qty` inflects for a bare count
# («найдено 5 ученических мест»), and after a preposition that governs the genitive the
# same helper produces «ни за одним из 1 ученическое место» and «из 2 ученических места» —
# both wrong, and both reachable on any small group: the forms only coincide from 5 up,
# which is why a room of six hid it. `NUMBERS_GEN` below already exists for exactly this
# reason at exactly one call site; this is the second.
PUPIL_PLACES_GEN = ("ученического места", "ученических мест", "ученических мест")
PLACES = ("место", "места", "мест")
PEOPLE = ("человек", "человека", "человек")
# Used by the hallucination checker's own message, which is Russian shown to a human and
# therefore inflects too: a checker that reports «22 чисел» is one nobody reads carefully.
NUMBERS = ("число", "числа", "чисел")
NUMBERS_GEN = ("числа", "чисел", "чисел")
VALUES = ("значение", "значения", "значений")


def plural_form(value: float, forms: tuple[str, str, str]) -> str:
    """Pick the noun form for a Russian numeral: (1 минута, 2 минуты, 5 минут).

    The fractional case is the one machine-written Russian always gets wrong. A decimal
    quantity takes the genitive singular regardless of its value — «2,5 минуты»,
    «0,5 минуты», «15,2 минуты», never «15,2 минут» and never «1,0 минута» — and the
    genitive singular is the same form as the 2–4 slot for every noun this module uses
    (минута/минуты, раз/раза, секунда/секунды, кадр/кадра, наблюдение/наблюдения). So the
    fractional branch returns `forms[1]`, and where that coincidence ever stops holding
    the noun does not belong in this file.
    """
    if value != int(value):
        return forms[1]
    number = abs(int(value))
    if number % 10 == 1 and number % 100 != 11:
        return forms[0]
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return forms[1]
    return forms[2]


def num(value: float, decimals: int = 1) -> str:
    """A number as Russian typography renders it: decimal comma, no trailing zeros.

    `15.0` prints as «15» rather than «15,0» because the artefact stores minutes as floats
    and a report full of «,0» reads as a spreadsheet. The checker accepts either, since 15
    is a correct rounding of 15.0.
    """
    rounded = round(float(value), decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}".replace(".", ",")


def qty(value: float, forms: tuple[str, str, str], decimals: int = 1) -> str:
    """«5 раз», «2,5 минуты» — the number and its agreeing noun.

    Agreement follows the number as PRINTED, not as stored: 0.5 rendered to zero decimals
    is «0 раз», not «0 раза». A reader agrees the noun with the digits in front of them,
    and the rounding happens here rather than in the caller.
    """
    rounded = round(float(value), decimals)
    return f"{num(rounded, decimals)} {plural_form(rounded, forms)}"


def duration(seconds: float | None, minutes: float | None) -> str:
    """A duration in whichever unit a person would say it in.

    Both values come out of the bundle rather than being converted here, so the checker
    can back either rendering. Under a minute, seconds; above, minutes — «375,5 секунды»
    is a number nobody holds in their head and «6,3 минуты» is.
    """
    if seconds is None:
        return "—"
    if seconds < 60 or minutes is None:
        return qty(seconds, SECONDS)
    return qty(minutes, MINUTES)


def percent(value: float | None) -> str:
    """«80,4 %». A non-breaking space before the sign, as Russian typography wants."""
    if value is None:
        return "—"
    return f"{num(value, 1)} %"


# --------------------------------------------------------------------------------------
# compact() — the bundle.
# --------------------------------------------------------------------------------------


def _pct(fraction: float | None) -> float | None:
    """A 0..1 share as a percentage, rounded once, here, and never again downstream.

    Percentages live in the bundle already converted for one reason: the hallucination
    checker compares the text's numbers against the bundle's numbers literally. If the
    bundle held 0.804 and the text said «80,4 %», the checker would have to accept a
    factor of one hundred between them — and an allowance that wide would also let «8»
    stand for 0.08. Converting once, upstream of the text, keeps the comparison exact.
    """
    return None if fraction is None else round(100.0 * float(fraction), 1)


def _minutes(seconds: float | None) -> float | None:
    return None if seconds is None else round(float(seconds) / 60.0, 1)


def _clock(value: str | None) -> str | None:
    """«2026-08-07T09:59:58.123» -> «09:59». Seconds are noise in a lesson report."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value[11:16] or None


def _date_ru(value: str | None) -> tuple[str | None, str | None, str | None]:
    """-> (iso date, «7 августа 2026 года», «пятница»)."""
    if not value:
        return None, None, None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value[:10] or None, None, None
    return (moment.date().isoformat(),
            f"{moment.day} {MONTHS_RU_GENITIVE[moment.month - 1]} {moment.year} года",
            WEEKDAYS_RU[moment.weekday()])


def compact(artefact: Any) -> dict[str, Any]:
    """Reduce an artefact to the minimal bundle a summary is allowed to talk about.

    **The bundle is a whitelist, not a compression.** Everything in it is a number (or a
    fixed sentence) that the report may mention; everything left out is a number the
    report may not mention, because `check_numbers()` will reject it. So the omissions are
    the design:

    * **Pixel geometry** — seat centres, `scale_px`, cluster spreads, head-direction
      histograms. These describe the camera, not the lesson. «Место 4 находится в точке
      (817, 697)» is true, unreadable, and invites a reader to reason about a seating plan
      the module does not have.
    * **Any cross-seat aggregate of the activity index.** See the module docstring: a
      class mean or an ordering is a ranking of children, and the only defence that
      survives contact with a fluent text generator is the number simply not existing.
    * **Timelines.** A per-observation state sequence is what the chart is for. In prose
      it would become a narrative («сначала… затем…»), and a narrative is where causation
      gets invented.

    Accepts an `Artefact`, an already-parsed dict, or a path to the JSON.
    """
    if hasattr(artefact, "to_dict"):
        data: dict[str, Any] = artefact.to_dict()
    elif isinstance(artefact, (str, Path)):
        data = json.loads(Path(artefact).read_text(encoding="utf-8"))
    else:
        data = dict(artefact)

    provenance = data.get("provenance") or {}
    lesson = data.get("lesson") or {}
    uncertainty = data.get("uncertainty") or {}
    discovery = ((provenance.get("room") or {}).get("seat_discovery")) or {}

    window = lesson.get("window_wall") or [provenance.get("started_at"), None]
    iso_date, date_ru, weekday_ru = _date_ru(window[0])

    seats = [_seat_bundle(seat) for seat in (data.get("seats") or [])]
    with_index = sum(1 for s in seats if s["activity"]["available"])

    observations_total = uncertainty.get("observations_total") or 0
    unassigned = uncertainty.get("observations_unassigned") or 0

    bundle: dict[str, Any] = {
        "schema": SUMMARY_VERSION,
        "run_id": data.get("run_id"),
        "lesson": {
            "date": iso_date,
            "date_ru": date_ru,
            "weekday_ru": weekday_ru,
            "start_time": _clock(window[0]),
            "end_time": _clock(window[1]),
            "duration_minutes": lesson.get("duration_minutes"),
            "pupil_seats": lesson.get("pupil_seats", len(seats)),
            "adult_seats": lesson.get("adult_seat", 0),
            "seats_with_index": with_index,
            "seats_without_index": len(seats) - with_index,
            # The scale the index is expressed on. In the bundle because the sentence
            # «62,4 из 100» is the honest way to write it, and 100 has to be backed.
            "index_scale_max": 100,
            "analysed_frames": provenance.get("analysed_frames"),
            "sample_fps": provenance.get("sample_fps"),
            "clock_source": provenance.get("clock_source"),
            "clock_source_ru": CLOCK_SOURCE_RU.get(
                str(provenance.get("clock_source")), "источник времени не указан"),
            "min_people_in_frame": (lesson.get("detection") or {}).get("min_people"),
        },
        "coverage": {
            "observations_total": observations_total,
            "observations_unassigned": unassigned,
            "observations_unassigned_percent": _pct(
                unassigned / observations_total) if observations_total else None,
            "observations_unreadable": uncertainty.get("observations_unreadable"),
            "frames_with_no_person": uncertainty.get("frames_with_no_person"),
            "seats_never_settled": uncertainty.get("seats_never_settled"),
            # The count only. Each rejected cluster's centre is pixel geometry, and its
            # existence is the fact worth reporting: places the room offered and the
            # module declined to call seats.
            "rejected_clusters": len(uncertainty.get("rejected_clusters") or ()),
        },
        "seat_discovery": {
            "places_found": discovery.get("seats_found"),
            "expected_people_per_frame": discovery.get("expected_people"),
            "agrees_with_detector": discovery.get("plausible"),
            "warning": discovery.get("warning"),
        },
        "seats": seats,
        "teacher": _teacher_bundle(data.get("teacher")),
        "unmeasured": list(lesson.get("unmeasured") or ()),
        "notes": list(uncertainty.get("notes") or ()),
        "caveats": list(data.get("caveats") or CAVEATS_RU),
    }
    return bundle


def _seat_bundle(seat: dict[str, Any]) -> dict[str, Any]:
    """One place, reduced to its counters and their coverage."""
    ledger = seat.get("ledger") or {}
    activity = ((seat.get("metrics") or {}).get("activity")) or {}
    pupil = seat.get("pupil")
    seconds = {key: value for key, value in (ledger.get("observed_seconds_by_state") or {}).items()}

    return {
        "label": seat.get("label"),
        "seat_id": seat.get("seat_id"),
        "role": seat.get("role"),
        # None is the answer this package expects to give: the measured face evidence on
        # this camera (median best cosine 0.30, margin 0.10) does not identify anyone, so
        # a seat carries a name only when `identity/` cleared its own bar.
        "pupil_name": (pupil or {}).get("full_name"),
        # `bool(pupil)` is WRONG here and was the bug: `identity/assign.py` always emits a
        # pupil block, including for every refusal, so the dict's presence says only that
        # identity was considered. The established flag inside it is the answer, and
        # reading the wrapper instead printed «Место 2 — None» for every unnamed seat --
        # a seat announcing a child called None is exactly the kind of confident-looking
        # emptiness this report exists to avoid.
        "identity_established": bool((pupil or {}).get("established")),
        "identity_method": (pupil or {}).get("method"),
        "identity_reason": (pupil or {}).get("reason_ru"),
        "occupancy_percent": _pct(seat.get("occupancy")),
        "coverage_percent": _pct(ledger.get("coverage")),
        "settled": ledger.get("settled"),
        # WHY a place never settled, when it never did — see `states.Baselines`. `settled:
        # false` on its own makes every state at that place UNKNOWN without saying whether
        # the place was barely seen or seen badly, and the report has to be able to say
        # which: the first is a camera angle, the second is a detection this footage cannot
        # support, and they call for different actions from whoever reads it.
        "settle_refusal": ledger.get("settle_refusal"),
        "observations": ledger.get("observations"),
        "observed_minutes": _minutes(ledger.get("observed_seconds")),
        "absent_observations": ledger.get("absent_observations"),
        "unreadable_observations": ledger.get("unreadable_observations"),
        "hand_unmeasurable_observations": ledger.get("hand_unmeasurable_observations"),
        "activity": {
            "available": bool(activity.get("available")),
            "index": activity.get("index"),
            "reason": activity.get("reason") or "",
            "parts": [
                {
                    "key": part.get("key"),
                    "label_ru": part.get("label_ru"),
                    "value_percent": _pct(part.get("value")),
                    "weight_percent": _pct(part.get("weight")),
                    "contribution": part.get("contribution"),
                    "raw": part.get("raw"),
                }
                for part in (activity.get("parts") or ())
            ],
        },
        "counts": dict(ledger.get("counts") or {}),
        "observed_seconds_by_state": seconds,
        "minutes": {key: _minutes(value) for key, value in seconds.items()},
        "state_observations": dict(ledger.get("state_observations") or {}),
        # Runs too short to be called an episode. In the bundle because they are the
        # honest measure of how much jitter the counters above absorbed.
        "discarded_short_runs": dict(ledger.get("discarded_short_runs") or {}),
    }



def _presence_bundle(presence: dict[str, Any] | None) -> dict[str, Any] | None:
    """The adult's position taxonomy, flattened for prose, denominators intact.

    Only the shares OF THE LESSON are carried through. The artefact also holds shares of
    the attributed frames, and they are deliberately left behind here: a sentence is read
    linearly, a reader cannot see which denominator a number carried three clauses ago, and
    offering both to a text generator is offering it the chance to make exactly the mistake
    that produced «сидел 96,5 % времени (6,0 минуты)». One family, the one that includes
    `out_of_frame`, so every number in the paragraph shares a divisor.
    """
    if not presence:
        return None
    of_lesson = presence.get("state_share_of_lesson_percent") or {}
    minutes = presence.get("state_minutes_of_lesson") or {}
    longest = presence.get("longest_episode_minutes_by_state") or {}
    board = presence.get("board") or {}
    floor = presence.get("floor_coverage") or {}
    follower = presence.get("identification") or {}
    return {
        "attributed_percent_of_lesson": presence.get("attributed_share_of_lesson_percent"),
        "lesson_minutes": presence.get("lesson_minutes"),
        # `None`, not `0.0`, when nobody drew a board zone: `classify_track` cannot emit
        # `at_board` at all in that case, so the zero is the shape of the config and not a
        # fact about the adult, and the prose below drops the clause instead of printing
        # «у доски — 0 %» about a lesson in which the board was never looked at.
        "board_zone_configured": bool(board.get("zone_configured")),
        "at_board_percent_of_lesson": (of_lesson.get("at_board")
                                       if board.get("zone_configured") else None),
        "at_board_minutes": (minutes.get("at_board")
                             if board.get("zone_configured") else None),
        "at_desk_percent_of_lesson": of_lesson.get("at_desk"),
        "at_desk_minutes": minutes.get("at_desk"),
        "among_pupils_percent_of_lesson": of_lesson.get("among_pupils"),
        "among_pupils_minutes": minutes.get("among_pupils"),
        "out_of_frame_percent_of_lesson": of_lesson.get("out_of_frame"),
        "out_of_frame_minutes": minutes.get("out_of_frame"),
        "longest_at_desk_minutes": longest.get("at_desk"),
        "board_episodes": board.get("episodes"),
        "board_direction_of_error_ru": board.get("direction_of_error_ru") or "",
        "transitions_excluding_out_of_frame":
            presence.get("transitions_between_episodes_excluding_out_of_frame"),
        "floor_share_of_room_in_use_percent": floor.get("share_of_room_in_use_percent"),
        "identification_route": follower.get("route"),
        "zones_confirmed_by": follower.get("zones_confirmed_by") or "",
    }


def _teacher_bundle(teacher: dict[str, Any] | None) -> dict[str, Any] | None:
    """The adult. Position over time, and the sentence saying it is not an assessment."""
    if not teacher:
        return None
    ledger = teacher.get("ledger") or {}
    metrics = teacher.get("metrics") or {}
    identification = teacher.get("identification") or {}
    evidence = identification.get("evidence") or {}
    seconds = dict(ledger.get("observed_seconds_by_state") or {})

    return {
        "seat_id": teacher.get("seat_id"),
        "identification_source": identification.get("source"),
        "needs_confirmation": identification.get("needs_confirmation"),
        "evidence": {
            "largest_scale_px": evidence.get("largest_scale"),
            "median_other_scale_px": evidence.get("median_other_scale"),
            "ratio": evidence.get("ratio"),
            "required_ratio": evidence.get("required"),
        },
        "available": bool(metrics.get("available")),
        "reason": metrics.get("reason") or "",
        "coverage_percent": _pct(metrics.get("coverage")),
        # WHERE IN THE ROOM he was, when the camera can see the board. Present only on a
        # camera with a `board_zone`; `None` everywhere else, and the prose below omits the
        # whole paragraph rather than printing zeroes for states that were never measurable.
        # Kept as its own sub-dict for the same reason the artefact does: the pose numbers
        # beside it answer a different question and share none of their denominators.
        "presence": _presence_bundle(metrics.get("presence")),
        # Already percentages in the artefact -- see metrics/teacher.py. Not converted.
        "at_desk_percent": metrics.get("at_desk_share_of_observed"),
        "at_desk_minutes": metrics.get("at_desk_minutes"),
                "standing_or_away_percent": metrics.get("standing_or_away_share_of_observed"),
        "standing_or_away_minutes": metrics.get("standing_or_away_minutes"),
        "out_of_frame_percent": metrics.get("out_of_frame_share_of_lesson"),
        "transitions": metrics.get("transitions"),
        "longest_at_desk_episode_minutes": metrics.get("longest_at_desk_episode_minutes"),
        "longest_standing_episode_minutes": metrics.get("longest_standing_episode_minutes"),
        "observations": ledger.get("observations"),
        "observed_minutes": _minutes(ledger.get("observed_seconds")),
        "absent_observations": ledger.get("absent_observations"),
        "counts": dict(ledger.get("counts") or {}),
        "observed_seconds_by_state": seconds,
        "minutes": {key: _minutes(value) for key, value in seconds.items()},
        "not_an_assessment_ru": metrics.get("not_an_assessment_ru") or "",
    }


# --------------------------------------------------------------------------------------
# The deterministic generator.
# --------------------------------------------------------------------------------------

HEADING_OVERVIEW = "УРОК В ЦЕЛОМ"
HEADING_SEATS = "ПО МЕСТАМ"
HEADING_TEACHER = "ВЗРОСЛЫЙ ЗА ПЕРЕДНИМ СТОЛОМ"
HEADING_NOT_MEASURED = "ЧТО ИЗМЕРИТЬ НЕ УДАЛОСЬ"
HEADING_CAVEATS = "КАК ЧИТАТЬ ЭТИ ЧИСЛА"


def render(bundle: dict[str, Any]) -> str:
    """The whole report, deterministically, from the bundle alone.

    Every number printed below is read out of `bundle`; none is computed here. That is not
    a stylistic preference — it is what makes the generator's own output pass
    `check_numbers()`, and `summarise()` runs the checker over this text too. A
    deterministic generator that failed its own hallucination check would mean the bundle
    and the prose had drifted apart, and it is worth finding out from a test rather than
    from a psychologist.
    """
    blocks = [_header(bundle), _overview(bundle), _seats_section(bundle)]
    teacher = _teacher_section(bundle)
    if teacher:
        blocks.append(teacher)
    blocks.append(_not_measured(bundle))
    blocks.append(_caveats(bundle))
    return "\n\n".join(block.strip() for block in blocks if block.strip()) + "\n"


def _sentence(text: str) -> str:
    """Capitalise the first letter and end with a full stop. Nothing else is touched."""
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?", ":")) else text + "."


def _header(bundle: dict[str, Any]) -> str:
    lesson = bundle["lesson"]
    line = "ОТЧЁТ О НАБЛЮДЕНИИ ЗА УРОКОМ"
    when = lesson.get("date_ru")
    weekday = lesson.get("weekday_ru")
    start, end = lesson.get("start_time"), lesson.get("end_time")
    parts = []
    if when:
        parts.append(f"Запись от {when}" + (f" ({weekday})" if weekday else ""))
    if start and end:
        parts.append(f"с {start} до {end}")
    if lesson.get("duration_minutes") is not None:
        parts.append(f"продолжительность {qty(lesson['duration_minutes'], MINUTES)}")
    second = ", ".join(parts) + "." if parts else ""
    run_id = bundle.get("run_id")
    third = f"Идентификатор расчёта: {run_id}." if run_id else ""
    return "\n".join(x for x in (line, second, third) if x)


def _overview(bundle: dict[str, Any]) -> str:
    lesson, coverage = bundle["lesson"], bundle["coverage"]
    discovery = bundle["seat_discovery"]
    sentences: list[str] = []

    if lesson.get("analysed_frames") is not None and lesson.get("sample_fps") is not None:
        sentences.append(
            f"Проанализировано {qty(lesson['analysed_frames'], FRAMES)} записи — "
            f"по {qty(lesson['sample_fps'], FRAMES)} в секунду"
        )
    if lesson.get("clock_source_ru"):
        sentences.append(lesson["clock_source_ru"])
    overview = "; ".join(sentences) + "." if sentences else ""

    places: list[str] = []
    if lesson.get("pupil_seats") is not None:
        places.append(qty(lesson["pupil_seats"], PUPIL_PLACES))
    if lesson.get("adult_seats"):
        places.append(f"{qty(lesson['adult_seats'], PLACES)} взрослого")
    second = ""
    if places:
        second = "В кадре найдено " + " и ".join(places) + "."
        if discovery.get("agrees_with_detector") and discovery.get(
                "expected_people_per_frame") is not None:
            second += (
                " Разбиение на места сошлось с независимой оценкой — детектор видит в "
                f"кадре {qty(discovery['expected_people_per_frame'], PEOPLE)}."
            )
        elif discovery.get("warning"):
            second += " " + str(discovery["warning"])

    third_parts: list[str] = []
    if coverage.get("observations_total") is not None:
        third_parts.append(
            f"всего наблюдений за людьми — {num(coverage['observations_total'], 0)}")
    if coverage.get("observations_unassigned") is not None:
        share = coverage.get("observations_unassigned_percent")
        tail = f" ({percent(share)})" if share is not None else ""
        third_parts.append(
            f"из них {qty(coverage['observations_unassigned'], OBSERVATIONS)}{tail} "
            "не отнесены ни к одному месту — это проходы по классу и всё, что оказалось "
            "дальше полутора ширин плеч от любого найденного места"
        )
    if coverage.get("frames_with_no_person") is not None:
        third_parts.append(
            f"кадров, где не было видно ни одного человека, — "
            f"{num(coverage['frames_with_no_person'], 0)}")
    if coverage.get("seats_never_settled") is not None:
        third_parts.append(
            "мест, для которых не удалось установить базовую позу, — "
            f"{num(coverage['seats_never_settled'], 0)}")
    # Upper-cased by hand rather than with `str.capitalize()`, which also lower-cases
    # everything after the first character and would quietly destroy «МЕСТО» in a caveat
    # or a name in a future sentence.
    third = _sentence("; ".join(third_parts)) if third_parts else ""

    fourth = ""
    if lesson.get("seats_without_index"):
        fourth = (f"Индекс наблюдаемой активности рассчитан для "
                  f"{qty(lesson['seats_with_index'], PLACES)} из "
                  f"{num(lesson['pupil_seats'], 0)}; для остальных наблюдений оказалось "
                  "слишком мало, причина указана в абзаце места.")
    elif lesson.get("seats_with_index") is not None:
        fourth = ("Индекс наблюдаемой активности рассчитан для всех "
                  f"{qty(lesson['seats_with_index'], PUPIL_PLACES)}.")

    body = " ".join(x for x in (overview, second, third, fourth) if x)
    return f"{HEADING_OVERVIEW}\n{body}"


def _seats_section(bundle: dict[str, Any]) -> str:
    scale_max = bundle["lesson"].get("index_scale_max", 100)
    paragraphs = [_seat_paragraph(seat, scale_max) for seat in bundle["seats"]]
    return f"{HEADING_SEATS}\n\n" + "\n\n".join(paragraphs)


def _seat_paragraph(seat: dict[str, Any], scale_max: int = 100) -> str:
    """One place, one paragraph: who is it, how well was it seen, what was counted."""
    identity = ("личность не установлена, учёт ведётся по месту"
                if not seat.get("identity_established")
                else f"{seat.get('pupil_name')} "
                     f"(способ: {seat.get('identity_method')})")
    head = f"Место {num(seat['seat_id'], 0)} — {identity}."

    seen: list[str] = []
    if seat.get("occupancy_percent") is not None:
        seen.append(f"место было занято {percent(seat['occupancy_percent'])} времени "
                    "урока")
    if seat.get("coverage_percent") is not None:
        seen.append(f"поза читалась в {percent(seat['coverage_percent'])} "
                    "проанализированных кадров")
    if seat.get("observations") is not None:
        tail = ""
        if seat.get("observed_minutes") is not None:
            tail = f", суммарно {qty(seat['observed_minutes'], MINUTES)}"
        seen.append(f"это {qty(seat['observations'], OBSERVATIONS)}{tail}")
    coverage_sentence = ("Покрытие наблюдения: " + "; ".join(seen) + ".") if seen else ""

    gaps: list[str] = []
    if seat.get("absent_observations"):
        gaps.append("место было пустым в "
                    f"{qty(seat['absent_observations'], OBSERVATIONS_IN)}")
    if seat.get("unreadable_observations"):
        gaps.append("поза не читалась в "
                    f"{qty(seat['unreadable_observations'], OBSERVATIONS_IN)}")
    if seat.get("hand_unmeasurable_observations"):
        gaps.append("положение рук нельзя было определить в "
                    f"{qty(seat['hand_unmeasurable_observations'], OBSERVATIONS_IN)}")
    gaps_sentence = ("Пропуски: " + "; ".join(gaps) + ".") if gaps else (
        "Пропусков в наблюдении за этим местом не было.")

    activity = seat["activity"]
    if activity["available"] and activity.get("index") is not None:
        pieces = [
            f"{part['label_ru']} — {percent(part['value_percent'])} "
            f"при весе {percent(part['weight_percent'])}, "
            f"вклад {num(part['contribution'], 1)}"
            for part in activity["parts"]
        ]
        index_sentence = (
            f"Индекс наблюдаемой активности — {num(activity['index'], 1)} из "
            f"{num(scale_max, 0)}. Он складывается из наблюдаемых "
            "признаков: " + "; ".join(pieces) + "."
        )
    else:
        index_sentence = ("Индекс наблюдаемой активности не рассчитан: "
                          f"{activity.get('reason') or 'причина не указана'}.")

    counts = seat.get("counts") or {}
    happened: list[str] = []
    absent: list[str] = []
    for key, forms, seconds_key in EVENT_LABELS_RU:
        value = counts.get(key)
        if value is None:
            continue
        if value:
            tail = ""
            secs = (seat.get("observed_seconds_by_state") or {}).get(seconds_key or "", 0)
            if seconds_key and secs:
                tail = (f" (суммарно "
                        f"{duration(secs, (seat.get('minutes') or {}).get(seconds_key))})")
            happened.append(f"{qty(value, forms, 0)}{tail}")
        else:
            # The genitive plural, because the sentence it lands in is «не зафиксировано
            # ни разу: поднятий руки, вставаний…». A counter that read zero is stated,
            # never omitted: «не поднимал руку» and «руку измерить не удалось» are
            # different facts and the report keeps them apart.
            absent.append(forms[2])
    events_sentence = ("Зафиксировано: " + "; ".join(happened) + "."
                       if happened else "Ни одного отдельного события не зафиксировано.")
    if absent:
        events_sentence += " Не зафиксировано ни разу: " + ", ".join(absent) + "."

    seconds = seat.get("observed_seconds_by_state") or {}
    minutes = seat.get("minutes") or {}
    spent = [f"{STATE_LABELS_RU.get(key, key)} — {duration(value, minutes.get(key))}"
             for key, value in seconds.items() if value]
    time_sentence = ("Распределение времени: " + "; ".join(spent) + ".") if spent else ""

    discarded = seat.get("discarded_short_runs") or {}
    discarded_items = [f"{STATE_LABELS_RU.get(key, key)} — {num(value, 0)}"
                       for key, value in discarded.items() if value]
    discarded_sentence = (
        "Отброшено как слишком короткие серии (дрожание разметки, а не события): "
        + "; ".join(discarded_items) + "."
    ) if discarded_items else ""

    return " ".join(x for x in (head, coverage_sentence, gaps_sentence, index_sentence,
                                events_sentence, time_sentence, discarded_sentence) if x)


def _teacher_section(bundle: dict[str, Any]) -> str:
    teacher = bundle.get("teacher")
    if not teacher:
        return ""

    evidence = teacher.get("evidence") or {}
    if teacher.get("identification_source") == "inferred_by_scale":
        who = ("Взрослого программа выделила сама, по размеру фигуры: ширина плеч "
               f"{num(evidence.get('largest_scale_px'), 1)} против "
               f"{num(evidence.get('median_other_scale_px'), 1)} у остальных, отношение "
               f"{num(evidence.get('ratio'), 2)} при требуемом "
               f"{num(evidence.get('required_ratio'), 1)}.")
    else:
        # Translated, not printed raw. «Взрослый определён так: designated» reached the page
        # and is the kind of leak that makes a reader distrust everything around it.
        route = {
            "designated": "по зоне учительского стола, которую разметил человек",
            "designated_zone": "по зоне учительского стола, которую разметил человек",
            "inferred_by_scale": "по размеру фигуры — это предположение программы",
        }
        source = str(teacher.get("identification_source") or "none")
        # «Взрослый выделен не определён.» was on the page. `none` is not a ROUTE, it is
        # the absence of one, so it cannot be substituted into a sentence built for routes;
        # it needs its own. An untranslated source still prints raw, deliberately, because
        # a leak that looks like a leak gets fixed and one that reads smoothly does not.
        if source in route:
            who = f"Взрослый выделен {route[source]}."
        elif source in ("none", "", "None"):
            who = "Взрослого на этой записи выделить не удалось."
        else:
            who = f"Взрослый выделен: {source}."
    signed = ((teacher.get("presence") or {}).get("zones_confirmed_by") or "")
    if (teacher.get("presence") or {}).get("identification_route") == "designated_zone":
        who += (f" Разметку зон подтвердил: {signed}." if signed
                else " Саму разметку зон никто не подтверждал.")
    # ...but only when there IS an identification to confirm. «Взрослого выделить не
    # удалось. Человеком это не подтверждено.» leaves the reader hunting for what «это»
    # refers to, and the answer is: nothing.
    if teacher.get("needs_confirmation") and teacher.get("identification_source") != "none":
        who += " Человеком это не подтверждено."

    seen = ""
    # `observations` comes from the SEAT ledger, and on the camera this whole module was
    # written for the adult has no seat: `pipeline.assemble` builds his record with
    # `ledger={}` whenever `identify_adult` finds no settled place for him, which is the
    # normal D14 outcome once his desk cluster drifts a few pixels out of the drawn
    # polygon. `coverage_percent` is populated on that path anyway — it is then the
    # FOLLOWER's coverage, not the seat's — so testing it alone let this sentence be built
    # out of `None` and `qty()` raised `TypeError: float() argument must be ... not
    # 'NoneType'`, taking the whole text summary and the HTML report with it. Verified: an
    # artefact whose `teacher.seat_id` is null crashed `classvision report` outright.
    # The guard is on the quantity actually being printed, and the paragraph is simply
    # absent when there is no seat to describe — there is no seat, so there is nothing
    # here to say, and inventing zero occupancy for a place nobody sat at would be worse.
    if (teacher.get("coverage_percent") is not None
            and teacher.get("observations") is not None
            and teacher.get("observed_minutes") is not None):
        # A THIRD denominator, and the wording had to change to stop it reading as a
        # correction of the second. This is the SEAT ledger: how often the adult's own
        # place was occupied by anybody at all. It is not how often the adult was found —
        # that is the sentence below — and on this recording they are 48,5 % and 45 %,
        # close enough to be mistaken for each other and about different things.
        # `ledger.summary` is about a PLACE; the follower is about a PERSON.
        seen = (f"Само место у учительского стола было занято в "
                f"{percent(teacher['coverage_percent'])} проанализированных кадров — "
                f"{qty(teacher['observations'], OBSERVATIONS)}, "
                f"суммарно {qty(teacher['observed_minutes'], MINUTES)}")
        if teacher.get("absent_observations"):
            seen += (f"; в {qty(teacher['absent_observations'], OBSERVATIONS_IN)} на нём "
                     "никого не было")
        seen += ". Это про место, а не про человека: сколько удалось увидеть самого "
        seen += "взрослого — следующая цифра."

    # -- the refusal, said out loud ---------------------------------------------------
    #
    # `available: false` means the adult was located by NEITHER route: no zone, no scale,
    # no follower. Every position and pose number below is then absent -- except
    # `transitions`, which `metrics/teacher.py` fills with a literal `0` on that path, and
    # the paragraph printed «Отдельно и по другому знаменателю — что делало его тело за его
    # собственным столом, из наблюдений на этом месте, где поза прочиталась: смен положения
    # — 0.» about a lesson with no place, no observations and no adult. A zero standing in
    # for «не увидели» is rule 3 of this project, and this is the third time it has shipped:
    # the first put a guessed head position into a hand-raise counter, the second returned
    # an empty room from a default clustering threshold, and this is the teacher half.
    #
    # `metrics.reason` was computed, written into the artefact and never printed. The
    # cabinet's adult page already says «Это не ноль.» for exactly this case; the lesson
    # report said nothing at all, which is worse, because the reader gets a heading, a
    # confident-looking paragraph and no hint that it describes nobody.
    if not teacher.get("available"):
        why = str(teacher.get("reason") or "").strip()
        refusal = ("Показатели положения взрослого за этот урок не рассчитывались"
                   + (f": {why}." if why else "."))
        refusal += (" Это не ноль и не «взрослого не было»: это значит, что программа не "
                    "смогла указать, кто из людей в кадре взрослый, и поэтому не считала "
                    "по нему ничего.")
        note = teacher.get("not_an_assessment_ru") or ""
        return f"{HEADING_TEACHER}\n" + " ".join(x for x in (who, refusal, note) if x)

    position: list[str] = []
    if teacher.get("at_desk_percent") is not None:
        # BOTH the share and the minutes, in one phrase. This is the sentence an LLM got
        # wrong by pairing 96,5 % with the wrong duration; stating the right pair here
        # means no reader -- human or model -- has to derive it.
        at_desk = f"за столом — {percent(teacher['at_desk_percent'])} этого времени"
        if teacher.get("at_desk_minutes") is not None:
            at_desk += f" ({qty(teacher['at_desk_minutes'], MINUTES)})"
        position.append(at_desk)
    if teacher.get("standing_or_away_percent") is not None:
        standing = ("стоя или вне своего места — "
                    f"{percent(teacher['standing_or_away_percent'])}")
        if teacher.get("standing_or_away_minutes") is not None:
            standing += f" ({qty(teacher['standing_or_away_minutes'], MINUTES)})"
        position.append(standing)
    if teacher.get("transitions") is not None:
        position.append(f"смен положения — {num(teacher['transitions'], 0)}")
    if teacher.get("longest_at_desk_episode_minutes") is not None:
        position.append("самый долгий непрерывный период за столом — "
                        f"{qty(teacher['longest_at_desk_episode_minutes'], MINUTES)}")
    if teacher.get("longest_standing_episode_minutes") is not None:
        position.append("самый долгий период стоя — "
                        f"{qty(teacher['longest_standing_episode_minutes'], MINUTES)}")
    # **A DIFFERENT MEASUREMENT, and it has to say so out loud.** The paragraph above counts
    # WHERE IN THE ROOM he was, against the whole lesson. This one counts WHAT HIS BODY WAS
    # DOING at his own desk, against the observations where his seat was occupied and the
    # pose read. On this recording they are 31,3 % and 84,3 % of two different things, and
    # a reader meeting them in consecutive sentences without a lead-in will take the second
    # for a correction of the first. That is the same joining error, in prose, that this
    # project already shipped once as «сидел 96,5 % времени (6,0 минуты)».
    lead = ("Отдельно и по другому знаменателю — что делало его тело за его собственным "
            "столом, из наблюдений на этом месте, где поза прочиталась: ")
    position_sentence = (lead + "; ".join(position) + "." if position else "")

    # -- where in the room, when the board is visible ---------------------------------
    #
    # This paragraph comes BEFORE the pose paragraph and opens with the coverage, because
    # a reader who meets «у доски — 3 %» before «опознан в 45 % кадров» has already formed
    # an impression and will not revise it. Every figure here shares one denominator, the
    # whole lesson, and `out_of_frame` is quoted as one of the states rather than as a
    # remainder the reader has to compute.
    place_sentence = ""
    presence = teacher.get("presence")
    if presence and presence.get("attributed_percent_of_lesson") is not None:
        parts = []
        for key, label in (("at_board", "у доски"), ("at_desk", "за своим столом"),
                           ("among_pupils", "среди учеников"),
                           ("out_of_frame", "не опознан в кадре")):
            share = presence.get(f"{key}_percent_of_lesson")
            if share is None:
                continue
            piece = f"{label} — {percent(share)}"
            got = presence.get(f"{key}_minutes")
            if got is not None:
                piece += f" ({qty(got, MINUTES)})"
            parts.append(piece)
        place_sentence = (
            f"Взрослого удалось опознать в "
            f"{percent(presence['attributed_percent_of_lesson'])} кадров урока; "
            "остальное — «не смогли определить», а не «отсутствовал». "
            "От всей длительности урока: " + "; ".join(parts) + ". "
            + str(presence.get("board_direction_of_error_ru") or "")
        )
        if presence.get("transitions_excluding_out_of_frame") is not None:
            place_sentence += (" Смен места (не считая переходов через «не опознан») — "
                               f"{num(presence['transitions_excluding_out_of_frame'], 0)}.")

    frame_sentence = ""
    if not presence and teacher.get("out_of_frame_percent"):
        # The clause after the semicolon used to assert «на этой камере доска находится
        # позади объектива». Nothing here knows that. This branch runs when there is no
        # position taxonomy at all, which happens both on a camera pointed away from the
        # board and on a camera whose zones nobody drew, and the recording does not
        # distinguish them. Same defect as `metrics/teacher.not_an_assessment_ru` carried,
        # in the prose copy of it, and fixed the same way: say what was not measured, do
        # not explain it with a guess about the mounting.
        frame_sentence = (
            f"Вне кадра — {percent(teacher['out_of_frame_percent'])} урока. Это «не "
            "смогли увидеть», а не «отсутствовал»: положение в классе на этой записи не "
            "разбиралось — зоны доски и стола для этой камеры не размечены."
        )

    hands = ""
    raises = (teacher.get("counts") or {}).get("hand_raises")
    if raises:
        seconds = (teacher.get("observed_seconds_by_state") or {}).get("hand_raised")
        minutes = (teacher.get("minutes") or {}).get("hand_raised")
        hands = (
            f"Детектор поднятой руки сработал на взрослом {qty(raises, TIMES, 0)} "
            f"(суммарно {duration(seconds, minutes)}). Этот признак настроен на учениках, "
            "а взрослый сидит ближе всех к объективу и часто подпирает голову рукой, так "
            "что как участие в уроке эти срабатывания не считаются и нигде не суммируются."
        )

    note = teacher.get("not_an_assessment_ru") or ""
    body = " ".join(x for x in (who, seen, place_sentence, position_sentence,
                                frame_sentence, hands, note) if x)
    return f"{HEADING_TEACHER}\n{body}"


def _not_measured(bundle: dict[str, Any]) -> str:
    """The section that must never be an empty list, and never be silently skipped."""
    lines: list[str] = []

    named: list[str] = []
    for item in bundle.get("unmeasured") or ():
        named.append(str(item.get("what") or ""))
        lines.append(f"— «{item.get('what')}» — {item.get('why')}.")
    for note in bundle.get("notes") or ():
        note = str(note).strip()
        # `lesson.unmeasured` and `uncertainty.notes` overlap by design — the first is the
        # list of things the report cannot answer, the second is the run's own log of why.
        # Printing both verbatim reads as the report stuttering, so a note is dropped when
        # it is about a quantity already named above, and kept when it is not.
        if note and not any(what and what in note for what in named):
            lines.append(f"— {note}")

    unidentified = [s for s in bundle["seats"] if not s.get("identity_established")]
    if unidentified:
        lines.append(
            f"— Ни за одним из {qty(len(unidentified), PUPIL_PLACES_GEN, 0)} не закреплён "
            "конкретный ребёнок: распознавание лица на этой записи не даёт достаточных "
            "оснований, поэтому весь отчёт — про места, а не про детей."
        )

    partial = [s for s in bundle["seats"]
               if (s.get("coverage_percent") or 100.0) < 100.0]
    if partial:
        listed = "; ".join(f"место {num(s['seat_id'], 0)} — "
                           f"{percent(s['coverage_percent'])}" for s in partial)
        lines.append(
            f"— Не всё время урока каждое место было видно. Доля кадров, в которых поза "
            f"читалась: {listed}. Все проценты в отчёте посчитаны только по видимым "
            "наблюдениям, пропуски в них не входят."
        )

    refused = [s for s in bundle["seats"] if not s["activity"]["available"]]
    for seat in refused:
        lines.append(f"— Для места {num(seat['seat_id'], 0)} индекс не рассчитан: "
                     f"{seat['activity'].get('reason')}.")

    # A place whose baseline was never established. Every state there is «поза не читается»,
    # so the place LOOKS merely quiet in the tables above — and quiet is exactly the wrong
    # impression. `uncertainty.seats_never_settled` already counts these, but a count is
    # not a reason, and the reason is the part a psychologist can act on.
    for seat in bundle["seats"]:
        if seat.get("settle_refusal"):
            lines.append(
                f"— Для места {num(seat['seat_id'], 0)} не удалось установить базовую позу: "
                f"{seat['settle_refusal']}. Поэтому все состояния на этом месте — «поза не "
                "читается»; это не «ребёнок сидел неподвижно», а «мы не смогли измерить».")

    coverage = bundle["coverage"]
    if coverage.get("observations_unassigned"):
        lines.append(
            f"— {qty(coverage['observations_unassigned'], OBSERVATIONS).capitalize()} "
            f"({percent(coverage.get('observations_unassigned_percent'))} от всех) не "
            "отнесены ни к одному месту и не вошли ни в один счётчик."
        )
    if coverage.get("rejected_clusters"):
        lines.append(
            f"— Ещё {qty(coverage['rejected_clusters'], PLACES)} рассматривались как "
            "возможные и были отклонены: люди задерживались там слишком редко или "
            "слишком разбросанно, чтобы считать это местом."
        )

    teacher = bundle.get("teacher")
    if teacher and teacher.get("out_of_frame_percent"):
        lines.append(
            f"— Взрослый был вне кадра {percent(teacher['out_of_frame_percent'])} урока; "
            "что он делал в это время, запись не показывает."
        )

    lines.append("— Ничего из того, что происходило со звуком, речью или содержанием "
                 "урока, здесь не измерялось.")
    return f"{HEADING_NOT_MEASURED}\n" + "\n".join(lines)


def _caveats(bundle: dict[str, Any]) -> str:
    """The fixed sentences, verbatim from the artefact, never paraphrased and never cut."""
    items = "\n".join(f"— {text}" for text in bundle.get("caveats") or CAVEATS_RU)
    return f"{HEADING_CAVEATS}\n{items}"


# --------------------------------------------------------------------------------------
# The hallucination check.
# --------------------------------------------------------------------------------------

# A numeral, but not one welded to a word. The lookarounds keep the checker out of
# identifiers: `run_id` is a hex string, `seat_2` is a label, and a checker that pulled
# "9" out of "a3f19c" would spend its life reporting phantom numbers. A digit run counts
# only when nothing alphanumeric touches it on either side.
NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])(\d+(?:[.,]\d+)?)(?![0-9A-Za-zА-Яа-яЁё])")

# Numerals spelled as words. From «два» upward, on purpose: «один/одна/одно» function as
# an indefinite article in Russian («одна из составляющих»), and counting them as numerals
# rejects correct sentences. See the module docstring.
WORD_NUMERALS_RU: dict[str, float] = {
    "два": 2, "две": 2, "двух": 2, "двое": 2, "оба": 2, "обе": 2,
    "три": 3, "трёх": 3, "трех": 3, "трое": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4, "четверо": 4,
    "пять": 5, "пяти": 5, "пятеро": 5,
    "шесть": 6, "шести": 6, "шестеро": 6,
    "семь": 7, "семи": 7, "семеро": 7,
    "восемь": 8, "восьми": 8, "восьмеро": 8,
    "девять": 9, "девяти": 9, "девятеро": 9,
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100,
}
WORD_RE = re.compile(r"[А-Яа-яЁё]+")

CONTEXT_CHARS = 60


@dataclass(frozen=True, slots=True)
class Unbacked:
    """One number in the text that no number in the bundle supports."""

    token: str
    value: float
    context: str

    def __str__(self) -> str:
        return f"«{self.token}» (…{self.context}…)"


@dataclass(frozen=True, slots=True)
class NumberCheck:
    """The verdict on one generated text, with enough detail to argue with it."""

    ok: bool
    numbers_checked: int
    backing_values: int
    unbacked: tuple[Unbacked, ...] = ()

    def report_ru(self) -> str:
        # This message is itself Russian shown to a human, so it inflects too. A checker
        # that reports «22 чисел» is a checker whose output nobody reads carefully.
        if self.ok:
            return (f"Проверка чисел пройдена: {qty(self.numbers_checked, NUMBERS, 0)} "
                    f"в тексте, все подтверждены данными "
                    f"({qty(self.backing_values, VALUES, 0)} в сводке).")
        listed = "\n".join(f"  - {item}" for item in self.unbacked)
        return (f"Проверка чисел НЕ пройдена: не подтверждено сводкой "
                f"{num(len(self.unbacked), 0)} из "
                f"{qty(self.numbers_checked, NUMBERS_GEN, 0)}.\n{listed}")


def collect_backing(bundle: Any) -> set[float]:
    """Every number the bundle contains, including those inside its strings.

    Strings are mined as well as numeric leaves, because the bundle's own sentences carry
    figures the report is entitled to repeat — «требуется 50 %» inside an `activity.reason`
    and the date inside `date_ru`. Excluding them would make the checker reject a text for
    quoting the bundle back correctly. Booleans are excluded explicitly: `True` is `1` in
    Python and would silently back the number one everywhere.
    """
    found: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            found.add(float(node))
        elif isinstance(node, str):
            for match in NUMBER_RE.finditer(node):
                found.add(float(match.group(1).replace(",", ".")))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(bundle)
    return found


def _is_backed(value: float, decimals: int, backing: set[float]) -> bool:
    """Is `value` a correct rounding, to `decimals` places, of something in the bundle?

    Half a unit in the last printed digit, and nothing else. «62» backs 62.4 because 62.4
    rounds to 62; «63» does not. Written as a tolerance rather than by re-rounding every
    bundle value so that a text which prints more precision than the bundle holds (a model
    inventing «62,43») is caught rather than being rounded into agreement.
    """
    tolerance = 0.5 * (10.0 ** -decimals) + 1e-9
    return any(abs(value - candidate) <= tolerance for candidate in backing)


def check_numbers(text: str, bundle: Any) -> NumberCheck:
    """Every number in `text` must exist in `bundle`. This is the gate, not a warning.

    Called on the model's output before it is ever shown; a single unbacked number rejects
    the whole text. That severity is deliberate. The failure this defends against is not a
    model that is wrong by a little — it is a model that writes a plausible sentence about
    a child containing a figure nobody measured, and there is no version of that which is
    acceptable at a lower rate.
    """
    backing = collect_backing(bundle)
    unbacked: list[Unbacked] = []
    checked = 0

    for match in NUMBER_RE.finditer(text):
        token = match.group(1)
        value = float(token.replace(",", "."))
        decimals = len(token.split(",")[-1]) if "," in token else (
            len(token.split(".")[-1]) if "." in token else 0)
        checked += 1
        if not _is_backed(value, decimals, backing):
            unbacked.append(Unbacked(token, value, _context(text, match.start())))

    for match in WORD_RE.finditer(text):
        value = WORD_NUMERALS_RU.get(match.group(0).lower())
        if value is None:
            continue
        checked += 1
        if not _is_backed(value, 0, backing):
            unbacked.append(Unbacked(match.group(0), value,
                                     _context(text, match.start())))

    return NumberCheck(ok=not unbacked, numbers_checked=checked,
                       backing_values=len(backing), unbacked=tuple(unbacked))


def _context(text: str, position: int) -> str:
    start = max(0, position - CONTEXT_CHARS // 2)
    return " ".join(text[start:position + CONTEXT_CHARS // 2].split())


def find_forbidden_words(text: str) -> tuple[tuple[str, str], ...]:
    """Clinical, ranking and recommendation vocabulary, as (stem, context) pairs.

    The second gate, and much cruder than the first: a stem list cannot enumerate every
    way of being judgemental. It catches the specific registers a fluent summariser
    reaches for by reflex, and it is a backstop for the failure the number check cannot
    see — a sentence that contains no number at all.
    """
    lowered = text.lower()
    hits: list[tuple[str, str]] = []
    for stem in prompts.FORBIDDEN_STEMS_RU:
        position = lowered.find(stem)
        if position >= 0:
            hits.append((stem, _context(text, position)))
    return tuple(hits)


def prose_only(text: str) -> str:
    """The written part of a report, with the fixed caveats block removed.

    Run the word filter over a whole report and it fires on its own caveats: they contain
    «не ставит диагнозов» and «„Вовлечённость" здесь не измеряется», which are the very
    strings the filter hunts for — stated in order to be refused. Stripping the block that
    `artefact.CAVEATS_RU` owns leaves exactly the text somebody wrote, which is the only
    text a vocabulary filter has any business judging.
    """
    return text.split(HEADING_CAVEATS)[0]


# --------------------------------------------------------------------------------------
# The optional LLM path.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    """A finished report, and the full account of how it came to be that text."""

    text: str
    source: str                       # "deterministic" | "gemini" | "openai"
    bundle: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    prompt_version: str | None = None
    number_check: NumberCheck | None = None
    forbidden: tuple[tuple[str, str], ...] = ()
    # Why the deterministic text is what you are reading. Empty when an LLM text passed.
    fallback_reason: str = ""
    # The text that was rejected, kept for inspection rather than thrown away: a rejected
    # generation is the most useful thing there is for arguing with the prompt.
    rejected_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUMMARY_VERSION,
            "text": self.text,
            "source": self.source,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "fallback_reason": self.fallback_reason,
            "numbers_checked": None if self.number_check is None
            else self.number_check.numbers_checked,
            "numbers_unbacked": [] if self.number_check is None else
            [{"token": u.token, "value": u.value, "context": u.context}
             for u in self.number_check.unbacked],
            "forbidden_words": [{"stem": s, "context": c} for s, c in self.forbidden],
        }


def available_backend() -> str | None:
    """Which provider this environment can actually reach, if any.

    Presence of a key, not presence of a package: a machine with `google-genai` installed
    and no key cannot summarise anything, and the check that matters to a caller is
    whether there is any point trying.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _call_gemini(system: str, user: str, model: str, timeout: float) -> str:
    """google-genai, structured output. Call shape verified 2026-08-12 (see prompts.py)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                          or os.environ["GOOGLE_API_KEY"])
    config: dict[str, Any] = {
        "system_instruction": system,
        "response_mime_type": "application/json",
        # Deterministic-as-possible: this is a restatement task, and sampling variety
        # between two runs of the same lesson is variety in a child's record.
        "temperature": 0.0,
        "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
    }
    try:
        response = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                response_json_schema=prompts.SUMMARY_JSON_SCHEMA, **config))
    except TypeError:
        # Older SDKs take the OpenAPI-subset parameter instead of JSON Schema proper.
        response = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                response_schema=prompts.SUMMARY_JSON_SCHEMA, **config))
    return response.text or ""


def _call_openai(system: str, user: str, model: str, timeout: float) -> str:
    """openai Responses API, structured output. Call shape verified 2026-08-12.

    `text.format`, not `response_format` — the latter is the Chat Completions shape and is
    silently ignored here. A raw JSON Schema rather than a Pydantic model via
    `responses.parse`, because pydantic is not a dependency of this package and a report
    formatter is not where one gets added.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
    response = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        text={"format": {"type": "json_schema", "name": "classvision_summary",
                         "strict": True, "schema": prompts.SUMMARY_JSON_SCHEMA}},
    )
    return response.output_text or ""


def _assemble(sections: dict[str, Any], bundle: dict[str, Any]) -> str:
    """Model prose + fixed structure. The headings and the caveats are never generated.

    Raises `ValueError` when the model's seats do not correspond one-to-one, in order, to
    the bundle's. A missing paragraph is a child dropped from the report, and it is the
    one failure that would look like a clean success on every other check.
    """
    labels = [seat["label"] for seat in bundle["seats"]]
    returned = [str(item.get("label")) for item in (sections.get("seats") or ())]
    if returned != labels:
        raise ValueError(f"места в ответе модели не совпадают со сводкой: "
                         f"{returned} вместо {labels}")

    blocks = [_header(bundle), f"{HEADING_OVERVIEW}\n{sections.get('overview', '').strip()}"]
    paragraphs = "\n\n".join(str(item.get("text", "")).strip()
                             for item in sections["seats"])
    blocks.append(f"{HEADING_SEATS}\n\n{paragraphs}")
    if bundle.get("teacher") and str(sections.get("teacher", "")).strip():
        blocks.append(f"{HEADING_TEACHER}\n{str(sections['teacher']).strip()}")
    blocks.append(f"{HEADING_NOT_MEASURED}\n{str(sections.get('not_measured', '')).strip()}")
    blocks.append(_caveats(bundle))
    return "\n\n".join(block.strip() for block in blocks if block.strip()) + "\n"


def summarise(artefact: Any, *, backend: str = "auto", timeout: float = 60.0) -> Summary:
    """Produce the report. Deterministic unless an LLM is available AND passes the checks.

    `backend`: "deterministic" never calls out; "auto" uses whichever provider has a key;
    "gemini"/"openai" force one and fall back if it is unreachable.

    **Failure is always a fallback, never an exception.** A summary is the last step of a
    run that has already cost minutes of pose estimation, and there is a complete, correct
    report in hand before the network is touched. Nothing a provider can do — timeout,
    rate limit, schema drift, an unbacked number — is worth turning that into a traceback.
    Every fallback records its reason, and the reason is part of the returned object rather
    than a log line nobody reads.
    """
    bundle = compact(artefact)
    deterministic = render(bundle)
    baseline_check = check_numbers(deterministic, bundle)

    if backend == "deterministic":
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason="запрошен детерминированный режим")

    provider = available_backend() if backend == "auto" else backend
    if provider is None:
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason="ни GEMINI_API_KEY, ни OPENAI_API_KEY не заданы")

    # -- THE LLM PATH, and it no longer writes the report ------------------------------
    #
    # It used to: the model was asked to restate every metric in Russian prose, and
    # `check_numbers` verified that each numeral it produced existed in the bundle. That
    # passed (350 of 350) and the text was still false, because the guard checks whether a
    # number EXISTS, not whether it is attached to the right claim. See `report/note.py`
    # and `MEASUREMENTS.md` section 8.
    #
    # Now the model only chooses WHICH facts to point at and explains them in words, with
    # every numeral supplied by our own code. So the deterministic text below is always
    # the report; the note is an orientation paragraph placed above it, and losing it
    # costs a paragraph rather than the document.
    from classvision.report import note as note_module

    model = prompts.GEMINI_MODEL if provider == "gemini" else prompts.OPENAI_MODEL
    if provider != "gemini":
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       model=model, number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason="ориентирующая записка реализована только для Gemini")

    written = note_module.orientation_note(bundle, model=model, timeout=timeout)
    if not written.ok:
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       model=model, number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason=f"записка не построена: {written.reason}")

    # Second line of defence over the FINAL text: every numeral in it was written by us,
    # so this should never fire -- and if it ever does, something upstream is wrong and
    # the deterministic report is what ships.
    combined = f"{note_module.HEADING}\n{written.text}\n\n{deterministic}"
    check = check_numbers(combined, bundle)
    forbidden = find_forbidden_words(prose_only(combined))
    if not check.ok:
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       model=model, number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason=("в собранном тексте нашлось число без подтверждения "
                                        f"({check.unbacked[0].token}) — записка отброшена"),
                       rejected_text=written.text)
    if forbidden:
        return Summary(text=deterministic, source="deterministic", bundle=bundle,
                       model=model, number_check=baseline_check,
                       forbidden=find_forbidden_words(prose_only(deterministic)),
                       fallback_reason=("в записке запрещённая лексика: "
                                        f"{', '.join(w for w, _ in forbidden)}"),
                       rejected_text=written.text)

    return Summary(text=combined, source=provider, bundle=bundle, model=model,
                   prompt_version=note_module.NOTE_VERSION, number_check=check)


# --------------------------------------------------------------------------------------
# CLI: `python -m classvision.report.summary out/clip_15min.analysis.json`
# --------------------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Русский отчёт по артефакту анализа.")
    parser.add_argument("artefact", help="путь к *.analysis.json")
    parser.add_argument("--backend", default="deterministic",
                        choices=("deterministic", "auto", "gemini", "openai"))
    parser.add_argument("--json", action="store_true",
                        help="выдать результат целиком в JSON, а не только текст")
    parser.add_argument("--bundle", action="store_true",
                        help="выдать сводку, которая уходит в модель, и выйти")
    arguments = parser.parse_args(argv)

    bundle = compact(Path(arguments.artefact))
    if arguments.bundle:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0

    result = summarise(Path(arguments.artefact), backend=arguments.backend,
                       timeout=60.0)
    if arguments.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.text)
    if result.number_check is not None:
        print(result.number_check.report_ru(), file=sys.stderr)
    if result.fallback_reason:
        print(f"Источник текста: {result.source} ({result.fallback_reason})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))

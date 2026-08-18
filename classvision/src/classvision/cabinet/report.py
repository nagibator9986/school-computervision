"""The psychologist's cabinet as static HTML: five kinds of page, one argument.

--------------------------------------------------------------------------------
**THE DESIGN IS NOT INVENTED HERE. IT IS THE PLATFORM'S, AND `DESIGN.md` IS THE CONTRACT.**

This cabinet is shown next to `qorgan-ai-main`, and to a viewer it has to be the same
product rather than a tool bolted onto one. So it no longer ships a stylesheet of its own:
`cabinet/skin.py` holds `qorgan-ai-main/src/qorgan/web/static/app.css` **verbatim**, plus a
short tail for the handful of things the platform has no component for, and every page wears
the platform masthead. Every class name used below — `.topbar`, `.tiles`/`.tile`, `.tag`,
`.warning`, `.muted`, `.grid`, `.pages`, plain `table` — is a component that already exists
in the platform; the tail is documented rule by rule, and anything that could not name the
platform component it extends was deleted rather than defended.

**Why the CSS is inlined into every page rather than linked as a sibling file** is argued in
`skin.py`'s own docstring, and the short version is the same argument that put every long
caveat behind a `<details>` instead of deleting it: a psychologist mails ONE page to a
colleague, and an unstyled page of caveats is a page nobody reads to the end. Self-contained
means self-contained — no CDN, no font, no script, no external stylesheet, no request that
leaves the building saying who read a child's page and when.

--------------------------------------------------------------------------------
**THE NUMBERS LEAD; THE REASONING IS ONE CLICK AWAY. NOT ONE CAVEAT WAS DELETED.**

Every caveat here prevents a real misreading, and the previous version of this module proved
that by writing all of them out at full length, in prose, above the numbers — which produced
pages that opened with six paragraphs and a reader who scrolled past the lot. An unread
caveat protects nobody.

So each page now: `.tiles` of the numbers first, then the tables, then the argument. Long
blocks live in a `<details>` whose `<summary>` states what it is about to say («Что этой
записью НЕ измерено», «Почему это не динамика по неделям», «Что запросить у школы»). The one
sentence that changes how a number is read stays visible and un-collapsed beside it — the
board-zero contradiction, «единица учёта — место, а не ребёнок», «это не оценка работы
учителя», the index's lower-bound summary, every «не измерялось» cell. `<details>` needs no
JavaScript, which is the reason it and not a widget: a caveat that needed a script would
disappear on the day the script did.

**The cross-room warning is said once, in full, on `lessons.html`** — the one page where
records from two rooms legitimately sit in one table. Everywhere else it is one amber line
naming both halves of the claim («сняты РАЗНЫМИ камерами», «НЕ является динамикой») and a
link to that page. It used to be repeated verbatim on fifteen pages, which is the volume at
which a warning reads as boilerplate and loses exactly the readers it was written for.

--------------------------------------------------------------------------------
**THE PAGE SET, AND WHY IT IS IN THIS ORDER.**

  1. `index.html` — the cabinet. Organised **by class first and by room second**, because
     the class is what a school has and the room is where the camera happens to hang. The
     previous version listed one card per (room, class) pair with the room first, so two
     rooms sharing a class key read as two classes of one school — and, worse, invited the
     reader to read their two weeks as one term.
  2. `lessons.html` — «Что сравнимо между уроками». The one surface where two recordings
     from two rooms sit in one table, and it earns that by being about the RECORDINGS:
     coverage, places found, room-level event totals, where the adult was. Every column
     carries what it is NOT.
  3. `class-<room>-<class>.html` — one class in one room: its lessons, what happened in the
     room lesson by lesson, its places, its caveats.
  4. `lesson-<room>-<class>-<id>-<date>.html` — one lesson. This is the page somebody
     actually opens first: the questions in the five minutes after an analysis are «что было
     на этом уроке» and «чему в нём можно верить», and answering them out of nine per-place
     pages means holding nine coverage figures in your head.
  5. `place-<room>-<class>-<ordinal>.html` — one place across the term. The within-lesson
     strip leads, because it is the only trend this cabinet can honestly draw today; the
     refusal to draw the other one follows, and it carries a purchase order rather than a
     description of itself.

--------------------------------------------------------------------------------
**THE EMPTY AND INSUFFICIENT STATES ARE DESIGNED, BECAUSE THEY ARE THE STATES THIS PAGE
WILL ACTUALLY BE IN FOR MOST OF ITS FIRST TERM.**

  * **Nothing imported.** «Пока пусто» plus the two sentences saying what to do. A page
    that renders a blank table here gets reported as a bug and the module gets blamed for
    the school not having recorded anything yet.
  * **Lessons imported, no seating plan signed.** The counters are real and are shown in
    full; the word «динамика» is replaced by the reason there is none, the four gates, and
    the list of documents and recordings that would produce one. This is the normal state,
    not a degraded one.
  * **Fewer than five lessons.** «Данных пока мало: 1 урок из 5» — a sentence, in the place
    the chart would be, with the number of remaining lessons named. With one lesson that IS
    the honest answer, and the page has to say it rather than look broken.
  * **A week with no lesson.** «Уроков не загружено — это не ноль, это отсутствие
    наблюдения». Never an empty cell, never a `0`.
  * **One lesson in the whole cabinet.** The cross-lesson table says «в кабинете одна
    запись, сравнивать не с чем» instead of rendering one row and inviting a comparison
    against nothing.

Each is a distinct layout and none of them is an empty version of the full page, because an
empty version of a full page reads as a failure of the page.

--------------------------------------------------------------------------------
**A COUNTER THAT CANNOT BE MEASURED IS NEVER PRINTED AS A NUMBER.**

`weekly.COUNTER_KEYS` has a «выходил к доске» column. On camera01 it is `0` for all eight
pupils and it is zero **by construction**: the board hangs behind the lens,
`configs/camera_01.yaml` records `board_zone: null` as a measured decision, and
`ledger.classify` cannot emit `AT_BOARD` without a polygon. The artefact says so in
`lesson.unmeasured`, and the teacher half of the very same artefact already honoured it
(`presence.board.minutes_of_lesson: null` beside `zone_configured: false`). The pupil half
printed a confident zero, in a column headed with an observable event, for an event that
cannot be observed on that camera.

Every counter cell and every counter TILE on every page here goes through `_counter_cell` /
`_counter_tiles`, which print «не измерялось» plus the artefact's own reason. See
`cabinet/lessons.py::COUNTER_VOIDED_BY` for why the map is keyed on the analyser's own
sentence and which test fails if that sentence is reworded.

--------------------------------------------------------------------------------
**WHAT THIS PAGE REFUSES TO BE.**

No ranking, no sort by index, no colour scale over children, no class average, no «ниже
нормы» badge, no arrow that means «плохо». Places appear in room order, which is geometry.
A direction is always a WORD first — «ниже», «в пределах разброса», «не установлено» — and
`--alert` is reserved for caveats alone, so no verdict on this surface arrives as a colour
and every page survives being printed in grey.

The lesson page lists places whose MEASUREMENT does not support a figure («видно меньше
половины кадров»), and there is deliberately no counterpart list of places «на которые
стоит посмотреть»: that list would be a flag, and rule 1 of this project has no exception
for a gently-worded one.

The adult gets a page of position over time carrying the artefact's own
`not_an_assessment_ru`, and gets no index, no trend and no week table. «Сидит» is not «не
ведёт урок» and there is nowhere on this surface to imply otherwise.

--------------------------------------------------------------------------------
**THE ONE BLOCK ON THIS PAGE THAT WAS NOT WRITTEN BY THIS PACKAGE.**

The machine orientation note (`cabinet/notes.py`, guarded by `report/orientation.py`) is
rendered on the class overview and nowhere else, above the tables, dashed rather than
filled, labelled on both sides. Three separate reasons, none of them decoration:

  * **Above the tables** because its own caption says the numbers are below it, and a
    caption has to be true of the layout and not only of the intention.
  * **Dashed and unfilled** because every block on this surface that holds a measurement is
    a solid `--panel` surface, so the one block that holds none must not be able to pass for
    one at a glance. A psychologist scanning a page reads shapes before words.
  * **On the class page only.** A note names several places in one paragraph — it is a
    judgement about where to look FIRST. Repeated on a child's own page, «на месте 5 руку
    поднимали часто» sitting under the heading of place 3 is a sentence about somebody else,
    and the reader has no way to know.

All four of its states render: shown, rejected by the guard, absent with a reason, and
stale. Rule 4 of this project applies to the note as much as to a measurement — an absence
is reported, never rendered as a page that simply has one section fewer.

**Rendering never calls a provider.** The notes are read out of SQLite. This command runs on
an office machine that may have no internet at all, and a page build that quietly spent
money per class would be a surprise nobody asked for.
"""

from __future__ import annotations

import html as html_escape
import sqlite3
from pathlib import Path
from typing import Any

from classvision.cabinet import lessons as lessons_mod
from classvision.cabinet import notes, skin, store, weekly
from classvision.cabinet.lessons import ComparableView, LessonView
from classvision.cabinet.weekly import COUNTER_KEYS, ClassView, PlaceHistory

# Sparkline geometry, in SVG user units. CHOSEN small: the sparkline is a shape beside a
# number, not a chart to read values off. Anyone who needs the values has the table
# directly beneath it, which is why no axis, no gridline and no tooltip is drawn.
SPARK_W, SPARK_H, SPARK_PAD = 320, 56, 6

# The fewest points that may be DRAWN as a series. CHOSEN 2, and it is a design rule rather
# than a geometry limit: one dot in a framed box reads as a chart, and a flat one-point
# "line" reads as «стабильно», which one lesson cannot support. `DESIGN.md` rule 6 — no
# decoration that implies precision the data lacks. Below this the caption is printed alone.
SPARK_MIN_POINTS = 2


def _e(value: Any) -> str:
    return html_escape.escape("" if value is None else str(value))


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Russian numeral agreement, because «во всех 1 прогонах» is where a reader stops.

    Not a nicety on this surface. Every sentence this page puts a number into is a sentence
    ABOUT the limits of a measurement — «в 1 прогоне», «осталось 3 урока» — and a page that
    cannot conjugate its own refusal reads as a broken template, which is precisely the
    impression the named empty states exist to avoid. `weekly.lessons_word` does this for
    «урок»; this is the same rule for everything else.
    """
    if 11 <= n % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def _page(title: str, body: str, *, active: str = "",
          section: tuple[str, str] | None = None, comparable: bool = True) -> str:
    return skin.page(title, body, active=active, section=section, comparable=comparable)


def _class_page_name(room_key: Any, class_key: Any) -> str:
    """The class page's filename, in ONE place.

    `render` writes the file and the masthead of every lesson and place page links to it, so
    the template `class-<slug>.html` is shared knowledge. Written out twice it is two chances
    for one of them to change — and the failure would be a nav item that 404s in a folder
    with no server to report it.
    """
    return f"class-{_slug(f'{room_key}-{class_key}')}.html"


# ---------------------------------------------------------------------------
# The platform's own headline row. Every page opens with one.
# ---------------------------------------------------------------------------


def _tile(value: str, label: str, *, css: str = "", title: str = "") -> str:
    """One `.tile`: the number at 30px, what it is at 12px underneath.

    `css` takes the platform's own tile accents and nothing else. `void` is this surface's
    addition and is documented in `skin.CABINET_CSS`: it makes «не измерялось» small and
    italic, because a refusal set at 30px would be the loudest thing on the page.

    Both halves are escaped here, so a caller passes TEXT and never markup: a tile holds one
    measured value and one label, and a tile that could carry a tag would be a tile that
    could carry a second sentence.
    """
    attr = f' title="{_e(title)}"' if title else ""
    return (f'<div class="tile {css}"{attr}><span class="value">{_e(value)}</span>'
            f'<span class="label">{_e(label)}</span></div>')


def _tiles(*items: str) -> str:
    kept = [item for item in items if item]
    return f'<div class="tiles">{"".join(kept)}</div>' if kept else ""


def _counter_tiles(totals: dict[str, Any], voided: dict[str, dict[str, Any]]) -> str:
    """The six episode counters as a headline row, with «не измерялось» where it belongs.

    **The markup inside each tile is pinned by a test and is not free to change.** It is
    `<strong>N</strong><br><span class="none">label</span>`, and the test that pins it
    (`test_a_counter_the_camera_cannot_measure_is_printed_as_a_refusal_not_as_zero`) is the
    one that proves camera01's «выходил к доске» never prints as a confident 0 in the place
    a reader quotes from. `skin.CABINET_CSS` styles `.tile strong` for exactly this reason.
    """
    cells = []
    for key, label in COUNTER_KEYS:
        void = voided.get(key) or {}
        if void.get("full"):
            value, css, title = "не измерялось", "void", str(void.get("why") or "")
        elif totals.get(key) is None:
            value, css, title = "—", "void", "за период не накоплено — это не ноль"
        else:
            value, css, title = str(totals[key]), "", str(void.get("why") or "")
        attr = f' title="{_e(title)}"' if title else ""
        cells.append(f'<div class="tile {css}"{attr}><strong>{value}</strong><br>'
                     f'<span class="none">{_e(label)}</span></div>')
    return f'<div class="tiles">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Cells. `None`, `0` and «не измерялось» are three different facts, for ever.
# ---------------------------------------------------------------------------


def _coverage_cell(value: float | None) -> str:
    if value is None:
        return '<td class="none">—</td>'
    return f"<td>{value * 100:.0f}%</td>"


def _percent_cell(value: float | None) -> str:
    """A share the artefact already expressed in percent, or «—» when it did not have one.

    Separate from `_coverage_cell`, which multiplies a fraction by 100. Two cells that
    both print a «%» out of numbers on two different scales is how a 0.965 becomes a «1 %».
    """
    if value is None:
        return '<td class="none">—</td>'
    return f"<td>{float(value):.1f}</td>"


def _counter_cell(value: int | None, void: dict[str, Any] | None,
                  *, no_lesson: bool = False) -> str:
    """One episode counter, with the three states it actually has.

    «Не измерялось» (this camera cannot witness the event at all), «—» (there was no lesson
    to count in) and a number are three different facts, and a table that renders any two of
    them the same way is the defect this project keeps finding in its own output. The reason
    travels in the cell's own `title`, so the sentence is one hover away rather than one page
    away — and `_voided_note` prints it in full underneath, because a tooltip is not a place
    to put something a reader has to see.
    """
    if void is not None and void.get("full"):
        return f'<td class="void" title="{_e(void["why"])}">не измерялось</td>'
    if value is None:
        return ('<td class="none" title="уроков не загружено — это не ноль">—</td>'
                if no_lesson else '<td class="none">—</td>')
    if void is not None:
        return f'<td title="{_e(void["why"])}">{value}<sup>*</sup></td>'
    return f"<td>{value}</td>"


def _counter_heads(voided: dict[str, dict[str, Any]]) -> str:
    return "".join(
        f'<th class="{"void" if voided.get(key, {}).get("full") else ""}">{_e(label)}</th>'
        for key, label in COUNTER_KEYS)


# ---------------------------------------------------------------------------
# The caveats. Collapsed, never removed.
# ---------------------------------------------------------------------------


def _details(summary: str, inner: str, *, css: str = "", open_: bool = False) -> str:
    """A block whose SUMMARY says what the block will say. The whole reorganisation.

    A `<details>` whose summary reads «Подробнее» is a caveat that has been hidden; one whose
    summary reads «Почему это не динамика по неделям» is a caveat that has been filed. The
    difference is the entire justification for collapsing anything on this surface, so this
    helper takes the sentence and not a label.
    """
    if not inner:
        return ""
    mark = " open" if open_ else ""
    klass = f' class="{css}"' if css else ""
    return f"<details{klass}{mark}><summary>{_e(summary)}</summary>{inner}</details>"


# The one visible sentence about what a row on these pages IS. It is not collapsed anywhere,
# on any page, because it is the sentence that changes how every number below it is read: two
# children who swap desks swap histories, and only a human with a seating plan can see that.
READING_LINE_RU = ("Единица учёта — <strong>место в классе, а не ребёнок</strong>. Система "
                   "не ставит диагнозов и не даёт рекомендаций: ниже счётчики наблюдаемых "
                   "фактов, и интерпретирует их человек.")


def _reading_line() -> str:
    return f'<p class="muted">{READING_LINE_RU}</p>'


def _runs_phrase(hit: int, total: int) -> str:
    """«в 1 прогоне из 4» — the two numbers, never one of them dressed up as both.

    The old wording was «во всех {hit} прогонах этого класса», which reads as a statement
    about the class («this class has {hit} runs») while {hit} is only the number of runs
    that carried this particular note. An item present in 2 of a class's 4 runs printed «во
    всех 2 прогонах этого класса» — a false sentence in a generated report, assembled the
    same way as the one this project has already shipped once. Both numbers now appear, and
    «во всех» is used only where it is true.
    """
    word = _plural(total, "прогоне", "прогонах", "прогонах")
    if total and hit >= total:
        if total == 1:
            return "в единственном прогоне этого класса"
        return f"во всех {total} {word} этого класса"
    return f"в {hit} {_plural(hit, 'прогоне', 'прогонах', 'прогонах')} из {total}"


def _adults_note(view: ClassView) -> str:
    """Lessons whose adult was recorded but never located. Never silently absent.

    A class page with no adult on it is ambiguous between «взрослого в кадре не было» and
    «взрослого не удалось найти», and the store used to render both identically: the import
    dropped a teacher record with no `centre` without a word, so `D14/8-А` showed six pupils
    and nothing else, and `clip_15min.named.json` stored eight places where two other runs
    of the same lesson stored nine. Rule 4 of this project: unmeasurable is LISTED.
    """
    if not view.adults_without_place:
        return ""
    rows = "".join(
        f"<li>{_e(row['date_local'] or 'без даты')}: "
        + (_e(row["reason_ru"]) if row["reason_ru"] else
           "разбор взрослого есть, но его положение в кадре в артефакт не попало")
        + "</li>"
        for row in view.adults_without_place)
    n = len(view.adults_without_place)
    # «по 1 уроку», «по 21 уроку», «по 2 урокам» — dative, so the singular form returns for
    # every count ending in 1 except 11, not only for literal 1.
    return (f"<p><strong>Место взрослого не заведено "
            f"{_plural(n, f'по {n} уроку', f'по {n} урокам', f'по {n} урокам')}."
            f"</strong></p>"
            f"<ul>{rows}</ul>"
            f'<p class="muted">Это отсутствие ИЗМЕРЕНИЯ, а не вывод о том, что взрослого '
            f"в классе не было. Положение взрослого по таким урокам не накапливается; "
            f"подставлять вместо него ноль или ближайшее известное место нельзя.</p>")


def _caveats_block(view: ClassView, *, include_unmeasured: bool = True) -> str:
    """«Что не измерено» + «как это читать», both filed behind their own sentence.

    **Nothing here is optional and nothing here is shortened.** The two blocks are collapsed
    because they are the LONG form of statements whose short form is already visible higher
    on the page (`_reading_line`, the «не измерялось» cells, the board contradiction), and a
    reader who has met an argument three times at full length starts skipping the block —
    which is precisely how the neighbouring codebase lost its caveats.

    The first half is skipped on ONE page only: the lesson page prints that run's own
    `unmeasured` list at the top, above every number, which is where it belongs on a page
    about one recording, and printing the class-scoped version again at the bottom put the
    same two sentences twice on one page.
    """
    items = "".join(f"<li>{_e(c)}</li>" for c in view.caveats_ru)
    how = _details(
        f"Как это читать: {len(view.caveats_ru)} "
        f"{_plural(len(view.caveats_ru), 'оговорка', 'оговорки', 'оговорок')}, которые "
        f"меняют смысл всех чисел выше",
        f"<ul>{items}</ul>")
    if not include_unmeasured:
        return how

    unmeasured = "".join(
        f"<li><strong>{_e(u.get('what'))}</strong> — {_e(u.get('why'))} "
        f'<span class="none">({_runs_phrase(int(u.get("runs") or 0), view.runs_in_class)})'
        f"</span></li>"
        for u in view.unmeasured)
    heading = ("Что этой записью НЕ измерено" if len(view.lessons) == 1
               else "Что этими записями НЕ измерено")
    block = ""
    if unmeasured:
        count = len(view.unmeasured)
        block = _details(
            f"{heading} — {count} "
            f"{_plural(count, 'пункт', 'пункта', 'пунктов')}, и ни один из них не ноль",
            f"<ul>{unmeasured}</ul>{_adults_note(view)}", css="warn")
    elif view.adults_without_place:
        block = _details("Что этой записью НЕ измерено: место взрослого",
                         _adults_note(view), css="warn")
    return f"{block}{how}"


def _voided_for_class(view: ClassView) -> dict[str, dict[str, Any]]:
    """Which counters this class's runs said they could not measure, and how completely.

    Two states, and collapsing them would recreate the very defect one level up. When EVERY
    run of the class could not measure the event, the summed counter is not a small number,
    it is not a number at all. When only some runs could not, the sum is a real count over
    the remaining lessons, and the honest rendering is the number PLUS the sentence naming
    which lessons it excludes — deleting it would throw away a measurement that was made.
    """
    out: dict[str, dict[str, Any]] = {}
    for item in view.unmeasured:
        keys = lessons_mod.COUNTER_VOIDED_BY.get(str(item.get("what") or ""), ())
        if not keys:
            continue
        hit = int(item.get("runs") or 0)
        total = max(int(view.runs_in_class or 0), hit)
        full = hit >= total
        why = f"{item.get('what')}: {item.get('why')} ({_runs_phrase(hit, total)})."
        if not full:
            why += (" Число рядом относится только к остальным урокам, а не ко всему "
                    "периоду.")
        for key in keys:
            out[key] = {"full": full, "why": why}
    return out


def _voided_note(voided: dict[str, dict[str, Any]]) -> str:
    """The footnote under any table with a voided column. Never a silent «—».

    Collapsed, and it is the one caveat on the page that may be: the cell itself already
    says «не измерялось» in words, so the reader has met the claim before this block. What
    the block adds is WHY, per column, which is what its summary promises.
    """
    if not voided:
        return ""
    labels = dict(COUNTER_KEYS)
    named = ", ".join(_e(labels.get(key, key)) for key in voided)
    items = "".join(
        f"<li><strong>{_e(labels.get(key, key))}</strong> — {_e(value['why'])}</li>"
        for key, value in voided.items())
    return _details(
        f"Почему «{named}» здесь не число, а «не измерялось»",
        f"<ul>{items}</ul>"
        f'<p class="muted">В таких столбцах стоит «не измерялось», а не ноль. Ноль '
        f"означал бы «смотрели и не увидели»; здесь смотреть было нечем.</p>", css="warn")


def _index_lower_bound_note(voided: dict[str, dict[str, Any]]) -> str:
    """The index itself contains a voided counter, and no page may print it without this.

    `metrics/activity.py` weights «видимые действия» at 0.30 and computes it as
    `min(1, (hand_raises + stands + board_visits) / 3)`. On camera01 `board_visits` cannot be
    anything but zero — the board is behind the lens — so a third of that term's numerator is
    structurally absent, and the index it feeds is a LOWER BOUND rather than a measurement of
    the same quantity the D14 index measures.

    This is not repaired by changing the index: the weights are declared, not fitted, and
    re-deriving them per camera would make two lessons' indices incomparable in a way no
    reader could see. It is repaired by saying it wherever the index appears — which is the
    same remedy `_counter_cell` applies to the counter one level down.

    **The `<summary>` is the claim, not a label for it.** That sentence changes the reading of
    every index on the page, so it is what the reader sees closed; the arithmetic behind it is
    what opens.
    """
    if not voided.get("board_visits", {}).get("full"):
        return ""
    return _details(
        "Индекс на этой камере — нижняя граница: «выход к доске» входит в него нулём",
        "<p>Составляющая «видимые действия» складывает поднятые руки, вставания и выходы к "
        "доске и делит сумму на 3. На этой камере «выход к доске» не измеряется вовсе, "
        "поэтому в эту сумму он входит НУЛЁМ — не потому, что ребёнок не выходил, а потому, "
        "что этого нечем увидеть. По этому слагаемому индекс здесь занижен на неизвестную "
        "величину, и сравнивать индекс с индексом камеры, где доска размечена, нельзя даже "
        "при прочих равных.</p>", css="warn")


def _board_conflicts(views: list[LessonView]) -> str:
    """Every «выходил к доске — 0» in this scope that the same artefact contradicts.

    **Visible, never collapsed, on every page where the zero itself is visible.** It is the
    definition of the sentence `DESIGN.md` rule 1 keeps un-collapsed: the number it qualifies
    is a `0` in a column headed with an observable event, and a reader who does not meet the
    contradiction has read the zero as a fact about children. Rendered on the class page and
    on every place page of the class, not only on the lesson card, because the zero appears
    on all three and the number that contradicts it is per lesson.
    """
    items = [(view, view.board_conflict_ru) for view in views if view.board_conflict_ru]
    if not items:
        return ""
    rows = "".join(
        f"<li><strong>{_e(view.date_local or 'без даты')}</strong> — {_e(text)}</li>"
        for view, text in items)
    return (f'<div class="warning"><strong>Ноль, которому противоречит эта же запись'
            f"</strong><ul>{rows}</ul></div>")


def _totals_qualifiers(history: PlaceHistory) -> str:
    """Everything qualifying the summed figures, beside the summed figures.

    Rule 2 of this project is «every total is rendered with its coverage/uncertainty
    counters», and an operator's `--allow-overlap` override is one of those counters: on
    the real directory, place 3's «поднимал руку — 2» is one hand raised in a 50-minute
    recording plus the SAME hand raised in a 15-minute slice of it. The per-lesson rows
    carried the note; the total, which is the number a reader quotes, did not.
    """
    collected: list[str] = []
    for session in history.sessions:
        for note in session.lesson_notes_ru + session.match_notes_ru:
            if note and note not in collected:
                collected.append(note)
    if not collected:
        return ""
    items = "".join(f"<li>{_e(note)}</li>" for note in collected)
    return _details(
        f"Что оговаривает эти итоги — {len(collected)} "
        f"{_plural(len(collected), 'оговорка', 'оговорки', 'оговорок')}",
        f"<ul>{items}</ul>", css="warn")


# ---------------------------------------------------------------------------
# Two rooms under one class key: once in full, everywhere else in one line.
# ---------------------------------------------------------------------------


def _cross_room_line(rooms: list[dict[str, Any]], *, here: str | None = None,
                     link: str = "lessons.html") -> str:
    """The short form, for the fourteen pages that are not `lessons.html`.

    It carries both halves of the claim — that the records come from DIFFERENT cameras, and
    that the difference between their weeks is therefore NOT a trend — because either half
    alone is a sentence a reader can finish wrongly. The full argument, with the room keys,
    the dates and the place counts the store actually holds, is one link away and is said
    exactly once.
    """
    if len(rooms) < 2:
        return ""
    keys = ", ".join(_e(str(room["room_key"])) for room in rooms)
    where = (f" Эта страница — только комната <strong>{_e(here)}</strong>; числа других "
             f"комнат сюда не входят и войти не могут." if here else "")
    # Both halves of the claim are inside ONE `<strong>` each, with no tag splitting either
    # phrase. That is not typography: the test that proves this line still says both things
    # on all fifteen pages matches on the phrases, and a `<strong>` in the middle of one of
    # them silently satisfies the eye while failing the check that keeps the eye honest.
    return (f'<p class="warning"><strong>Уроки под этим ключом класса сняты РАЗНЫМИ '
            f"камерами</strong> ({keys}), поэтому разница между их неделями "
            f"<strong>НЕ является динамикой</strong> и нигде здесь не выводится.{where} "
            f'<a href="{_e(link)}">Полностью — на «Что сравнимо между уроками» →</a></p>')


def _cross_room_block(rooms: list[dict[str, Any]], class_key: str, *,
                      here: str | None = None, link: str | None = None) -> str:
    """The full form. Rendered on `lessons.html` and nowhere else.

    The sentences come from `lessons.ClassAcrossRooms.warning_ru`, which assembles them out
    of the store's own room keys, dates and place counts — so the warning cannot describe a
    configuration the database does not hold. A frozen paragraph would keep warning after
    somebody had fixed the keys, and a warning that is wrong is worse than none, because it
    is still believed.
    """
    if len(rooms) < 2:
        return ""
    across = lessons_mod.ClassAcrossRooms(class_key=class_key, rooms=rooms)
    bullets = "".join(f"<li>{_e(sentence)}</li>" for sentence in across.warning_ru())
    where = ""
    if here:
        where = (f"<p>Эта страница — только комната <strong>{_e(here)}</strong>. Числа "
                 f"других комнат сюда не входят и войти не могут.</p>")
    tail = (f'<p><a href="{_e(link)}">Что сравнимо между этими записями →</a></p>'
            if link else "")
    return (f'<div class="warning"><strong>Уроки под этим ключом класса сняты РАЗНЫМИ '
            f"камерами</strong><ul>{bullets}</ul>{where}{tail}</div>")


# ---------------------------------------------------------------------------
# The machine note.
# ---------------------------------------------------------------------------


# The two sentences that must travel with every shown note, and the reason they are a
# constant rather than prose typed into the template: they are the only thing standing
# between a fluent Russian paragraph and a reader who takes it for a finding. The first says
# who wrote it; the second says what it structurally cannot contain and where the numbers
# are. `report/orientation.py` guarantees the second by rejecting any digit that is not a
# seat label, so this is a description of an enforced property and not a hope.
NOTE_DISCLAIMER_RU = (
    "Это не измерение и не вывод. В записке по построению нет ни одной величины: проверка "
    "отклоняет любое число, кроме номера места, и отклонённая записка не показывается "
    "вовсе. Все числа — в таблицах ниже, и только они являются отчётом."
)


def _note_shell(date: str, css: str, inner: str) -> str:
    return (f'<div class="machine {css}"><span class="tag">машинная записка</span>'
            f'<span class="muted">урок {_e(date)}</span>{inner}</div>')


def _note_block(view: ClassView, stored: dict[str, Any] | None) -> str:
    """The orientation notes of this class's lessons, one block per lesson.

    **`stored = None` and `stored = {}` are different, and the page says which.** `{}` means
    the notes were looked up and this class has none — «записка не запрашивалась», with the
    command that would write one. `None` means the PAGE BUILD never looked, which happens
    only when a caller of `_overview` omits the argument, and it must not be rendered as the
    first: that would be a false sentence about a note sitting in the database two lines
    away, and false in the direction that hides work rather than inventing it.

    **Only on the class page, never on a place page.** A note is a judgement about where to
    look FIRST, so it names several places in one paragraph. Repeated on one child's page it
    would read as a statement about that child — «на месте 5 руку поднимали часто» sitting
    on the page of place 3 is a sentence about somebody else, and the reader has no way to
    know. One paragraph, one location, beside the lesson list it is about.

    **Placed above the tables on purpose.** The block says the numbers are below it, and
    that sentence has to be true of the layout, not just of the intention.

    Every state prints. A run with no note says so and names the command that would make
    one; a note the guard rejected says that it was rejected, and what it tried to say that
    disqualified it. Rule 4 of this project applies to the note itself: an absence is
    reported, never rendered as a page that simply has one fewer section.
    """
    if stored is None:
        return ("<h2>Записка-ориентир</h2>"
                '<div class="machine"><span class="tag">машинная записка</span>'
                '<p class="who">Эта страница собрана без обращения к таблице записок, '
                "поэтому здесь не сказано ни что они есть, ни что их нет. Пересоберите "
                "кабинет: <code>classvision cabinet report</code>.</p></div>")

    blocks: list[str] = []
    for row in view.lessons:
        date = str(row["date_local"] or "без даты")
        run_id = row["selected_run_id"]
        if not run_id:
            blocks.append(_note_shell(
                date, "",
                '<p class="who">Для этой записи не выбран расчётный прогон, поэтому и '
                "записки нет. Выберите прогон: <code>classvision cabinet select-run "
                "&lt;run_id&gt;</code>.</p>"))
            continue
        blocks.append(_one_note(date, str(run_id), stored.get(str(run_id))))

    return (f"<h2>Записка-ориентир</h2>"
            f'<p class="muted">Короткий машинный текст о том, с чего начать чтение. '
            f"Пишется отдельно по каждому уроку и хранится вместе с тем измерением, о "
            f"котором написан: разбор урока заново с другими порогами — это другое "
            f"измерение, и старая записка к нему не переносится.</p>"
            f"{''.join(blocks)}"
            f'<div class="machine"><span class="tag">по классу целиком — нет</span>'
            f'<p class="who">{_e(notes.CLASS_NOTE_REFUSAL_RU)}</p></div>')


def _one_note(date: str, run_id: str, note: Any) -> str:
    if note is None:
        return _note_shell(
            date, "",
            f'<p class="who">Записка для этого урока не запрашивалась. Это не сбой: '
            f"весь отчёт ниже полон и без неё. Получить: <code>classvision cabinet note "
            f"{_e(run_id)}</code> (нужен ключ модели; без ключа команда так и скажет и "
            f"ничего не сломает).</p>")

    # `bundle_was_sent`, not `source`: a guard rejection reports `source = "none"` and yet
    # a request WAS made, and the traffic sentence is the one sentence on this page a school
    # will check against its own answer to «что уходит наружу».
    traffic = (f"Отправлено {note.bundle_bytes} байт счётчиков — без кадров, без имён, "
               f"без дат." if note.bundle_was_sent else "Наружу ничего не отправлялось.")
    who = (f"Написала модель <strong>{_e(note.model or 'неизвестна')}</strong> "
           f"(промпт {_e(note.prompt_version)}), {_e(note.generated_at)}. {traffic}")

    if note.state == "stale":
        # The note was written about numbers that are no longer the numbers under this run.
        # It is NOT shown: a paragraph about one set of figures printed above another set is
        # the misattribution this whole feature was designed around, only slower.
        return _note_shell(
            date, "",
            f'<p class="who">Сохранённая записка описывает не те числа, которые сейчас '
            f"лежат в базе по этому прогону, поэтому она не показана. {who} Обновить: "
            f"<code>classvision cabinet note {_e(run_id)}</code>.</p>")

    if note.state == "refused_by_guard":
        offending = ", ".join(_e(o) for o in note.guard_offending)
        return _note_shell(
            date, "",
            f'<p class="who"><strong>Записка отклонена проверкой и не показывается.</strong> '
            f"В ней оказались числа, которые не являются номерами мест"
            f"{f' ({offending})' if offending else ''} — а величины в записке запрещены: "
            f"их место в таблицах, где рядом с каждой стоит её покрытие. Отчёт ниже от "
            f"этого не пострадал. {who}</p>")

    if note.state == "absent_reason":
        # A refusal has to say what would change it — except after `--backend
        # deterministic`, where nothing is broken and the stored sentence already explains
        # itself. «Ключа нет» printed with no next step is the kind of refusal that gets
        # read as a fault in the page.
        fix = "" if note.backend_requested == "deterministic" else (
            " Это не сбой: записка — необязательная подсказка поверх отчёта, и все числа "
            "ниже посчитаны без неё. Чтобы она появилась, на машине должен быть ключ "
            f"модели (GEMINI_API_KEY), после чего: classvision cabinet note {_e(run_id)}.")
        return _note_shell(
            date, "",
            f'<p class="who"><strong>Записки нет.</strong> Причина, как её записала '
            f"система: «{_e(note.reason_ru or 'не записана')}».{fix}</p>")

    return _note_shell(
        date, "",
        f"<blockquote>{_e(note.text)}</blockquote>"
        f'<p class="who">{who}</p>'
        f'<p class="who">{_e(NOTE_DISCLAIMER_RU)}</p>')


# ---------------------------------------------------------------------------
# The weekly trend: the refusal, its gates, and its purchase order.
# ---------------------------------------------------------------------------


def _sparkline(values: list[float], labels: list[str], *, connect: bool) -> str:
    """Index per lesson, oldest left. No axis, because there is a table underneath.

    `connect` is False whenever the page has just refused to state a trend, and then the
    points are drawn WITHOUT the joining line. A line through two dots is a direction, and
    a direction is the exact claim the sentence above it says cannot be made yet — a reader
    who takes the picture and leaves the words behind would leave with the opposite of what
    the page said.

    **Below `SPARK_MIN_POINTS` nothing is drawn at all** and the caption stands alone. One
    dot in a framed box reads as a chart with one reading in it, which is a claim about
    precision this cabinet does not have; the sentence says the same thing and cannot be
    misread. `DESIGN.md` rule 6.
    """
    if not values:
        return ""
    word = weekly.lessons_word(len(values))
    if len(values) < SPARK_MIN_POINTS:
        return ('<p class="muted">одно наблюдение — графика нет: линия через одну точку '
                "читается как «стабильно», а одно наблюдение этого не поддерживает.</p>")

    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 5.0, hi + 5.0
    span = hi - lo
    inner_w, inner_h = SPARK_W - 2 * SPARK_PAD, SPARK_H - 2 * SPARK_PAD
    step = inner_w / max(len(values) - 1, 1)
    points = [(SPARK_PAD + i * step,
               SPARK_PAD + inner_h - (v - lo) / span * inner_h)
              for i, v in enumerate(values)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(points))
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='currentColor'>"
        f"<title>{_e(labels[i] if i < len(labels) else '')}: {values[i]:.1f}</title>"
        f"</circle>" for i, (x, y) in enumerate(points))
    line = (f"<path d='{path}' fill='none' stroke='currentColor' stroke-width='1.6' "
            f"stroke-opacity='.55'/>") if connect else ""
    if connect:
        caption = f"{len(values)} {word}, шкала {lo:.0f}–{hi:.0f}"
    else:
        caption = (f"{len(values)} {word}, шкала {lo:.0f}–{hi:.0f}; точки не соединены — "
                   f"направление по такому числу уроков не определяется")
    return (f"<svg class='spark' width='{SPARK_W}' height='{SPARK_H}' "
            f"viewBox='0 0 {SPARK_W} {SPARK_H}' role='img'>{line}{dots}</svg>"
            f"<p class='muted'>{caption}</p>")


def _gates(view_gates: list[dict[str, Any]]) -> str:
    """What must be true before a weekly trend exists, and which parts already are."""
    if not view_gates:
        return ""
    items = "".join(
        f'<li class="{"ok" if g["passed"] else "no"}">{_e(g["gate"])}: '
        f"<strong>{_e(g['measured'])}</strong> "
        f'<span class="none">(требуется: {_e(g["required"])})</span>'
        f'<span class="why">{_e(g["detail_ru"])}</span></li>'
        for g in view_gates)
    passed = sum(1 for g in view_gates if g["passed"])
    return _details(
        f"Что должно совпасть, чтобы динамика строилась — выполнено {passed} из "
        f"{len(view_gates)}",
        f'<ul class="gates">{items}</ul>')


def _requests_block(history: PlaceHistory) -> str:
    """«Что запросить у школы» — the refusal turned into a list somebody can send.

    Placed inside the refusal rather than beside it, because the two are one statement: a
    page that says «динамика не строится» and leaves the next step to the reader has
    described its own state instead of answering the question the reader arrived with. Every
    item carries its number (`weekly.Request`), so «ещё 4 урока» is on the page and not in
    the reader's arithmetic. Collapsed, because it is a form to fill in and not a finding —
    but its summary names the count, so nobody has to open it to learn there is one.
    """
    if not history.requests:
        return f'<p class="muted">{_e(weekly.NOTHING_TO_REQUEST_RU)}</p>'
    items = "".join(
        f'<div class="req"><span class="what">{i}. {_e(request.what_ru)}</span>'
        f'<span class="why">Зачем именно это: {_e(request.why_ru)}</span>'
        f'<span class="how">Как: {_e(request.how_ru)}</span></div>'
        for i, request in enumerate(history.requests, start=1))
    n = len(history.requests)
    return _details(
        f"Что запросить у школы, чтобы динамика стала возможной — {n} "
        f"{_plural(n, 'пункт', 'пункта', 'пунктов')}",
        f'<p class="muted">Список полный: если придёт всё перечисленное, динамика по '
        f"этому месту будет построена; если не хватит одного пункта — не будет.</p>"
        f"{items}")


def _trend_block(history: PlaceHistory, view: ClassView) -> str:
    """The weekly trend, or the reason there is none — as words, in no colour at all.

    The refusal used to be an amber-bordered box. It is not amber now, for the reason
    `DESIGN.md` rule 3 gives: `--alert` belongs to caveats, and a cabinet with no signed
    seating plan is the intended, fully useful configuration rather than a fault. A warning
    colour on it would push a school toward attaching children's names to get rid of the
    yellow, which is the opposite of what this refusal is for.
    """
    trend_view = history.trend_view
    body = [f"<p><strong>{_e(trend_view.headline_ru)}</strong></p>"
            f"<p>{_e(trend_view.detail_ru)}</p>"]

    if trend_view.trend and trend_view.trend.get("available"):
        t = trend_view.trend
        body.append(
            f"<p>Последний урок — <strong>{t['latest']}</strong> при собственной норме "
            f"<strong>{t['baseline_median']}</strong> (MAD {t['baseline_mad']}), это "
            f"{t['deviation_mads']} MAD. Подряд ниже нормы: {t['sustained_lessons']} "
            f"{weekly.lessons_word(t['sustained_lessons'])}. Норма посчитана по "
            f"{t['history_used']} собственным урокам этого ученика.</p>")
        chance = view.chance_note
        body.append(
            f'<p class="muted">Для сравнения: в классе из {chance["class_size"]} '
            f"учеников примерно {chance['expected_pupils_by_chance']} попало бы в такой "
            f"список просто случайно. {_e(chance['note'])}</p>")

    if trend_view.series:
        body.append(_sparkline(trend_view.series, trend_view.series_dates,
                               connect=trend_view.available))
    body.append(_gates(trend_view.gates))
    body.append(_requests_block(history))
    return "".join(body)


# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------


def _weeks_table(history: PlaceHistory,
                 voided: dict[str, dict[str, Any]] | None = None) -> str:
    voided = voided or {}
    heads = _counter_heads(voided)
    rows = []
    for week in history.weeks:
        if not week.observed:
            rows.append(
                f'<tr><td>{_e(week.label)}</td>'
                f'<td class="none" colspan="{4 + len(COUNTER_KEYS)}">уроков не загружено — '
                f"это не ноль, это отсутствие наблюдения</td></tr>")
            continue
        counters = "".join(
            _counter_cell(week.counts.get(key), voided.get(key), no_lesson=True)
            for key, _ in COUNTER_KEYS)
        index = ('<td class="none">—</td>' if week.index_median is None
                 else f"<td>{week.index_median:.1f}</td>")
        rows.append(
            f"<tr><td>{_e(week.label)}</td><td>{week.sessions}</td>"
            f"<td>{week.observed_minutes:.0f}</td>"
            f"{_coverage_cell(week.coverage_median)}{index}{counters}</tr>")
    return (f'<div class="wrap"><table><thead><tr><th>неделя</th><th>занятий</th>'
            f"<th>минут под наблюдением</th><th>видимость (медиана)</th>"
            f"<th>индекс активности</th>{heads}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>{_voided_note(voided)}")


def _sessions_table(history: PlaceHistory, lesson_links: dict[int, str] | None = None,
                    voided: dict[str, dict[str, Any]] | None = None) -> str:
    voided = voided or {}
    lesson_links = lesson_links or {}
    heads = _counter_heads(voided)
    rows = []
    for session in history.sessions:
        counters = "".join(_counter_cell(session.counts.get(key), voided.get(key))
                           for key, _ in COUNTER_KEYS)
        if session.activity.available and session.activity.index is not None:
            index = f"<td>{session.activity.index:.1f}</td>"
        else:
            index = (f'<td class="none" title="{_e(session.activity.reason)}">'
                     f"не считался</td>")
        note = " ".join(session.match_notes_ru + session.lesson_notes_ru)
        note_html = f'<br><span class="none">{_e(note)}</span>' if note else ""
        link = lesson_links.get(session.session_id)
        date = (f'<a href="{_e(link)}">{_e(session.date_local)}</a>' if link
                else _e(session.date_local))
        rows.append(
            f"<tr><td>{date}{note_html}</td>"
            f"<td>{len(session.parts)}</td>"
            f"<td>{session.observed_seconds / 60:.0f}</td>"
            f"{_coverage_cell(session.coverage)}{index}{counters}</tr>")
    return (f'<div class="wrap"><table><thead><tr><th>дата</th><th>файлов</th>'
            f"<th>минут</th><th>видимость</th><th>индекс</th>{heads}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# What each raw counter inside an index component is, in Russian. The component table used
# to print `json.dumps(part.raw)` into a cell — `{"head_down_observations": 1440,
# "measured_observations": 1500}` at a psychologist — which is a developer's debug output
# wearing the costume of a measurement: unreadable, and therefore unchecked, which is worse
# than absent because it looks like provenance. An unknown key falls through to its own
# name rather than being dropped: a component that grew a field must show up as an odd word,
# not vanish.
RAW_LABELS_RU: dict[str, str] = {
    "measured_observations": "читаемых наблюдений",
    "head_down_observations": "из них голова на парте",
    "turned_away_observations": "из них отвернулся назад",
    "away_observations": "из них не на своём месте",
    "hand_raises": "поднятая рука",
    "stands": "вставания",
    "board_visits": "выходы к доске",
    "normaliser": "делитель (сколько действий даёт полный балл)",
}


def _raw_phrase(raw: Any) -> str:
    """«читаемых наблюдений — 977; из них голова на парте — 53». A sentence, not a dump."""
    if not isinstance(raw, dict) or not raw:
        return "—"
    # `measured_observations` first when present: it is the DENOMINATOR, and every other
    # number in the cell is «из них», which is a false phrase if the total comes second.
    keys = ([k for k in ("measured_observations",) if k in raw]
            + [k for k in raw if k != "measured_observations"])
    return "; ".join(f"{RAW_LABELS_RU.get(key, key)} — {_e(raw[key])}" for key in keys)


def _components(history: PlaceHistory,
                voided: dict[str, dict[str, Any]] | None = None) -> str:
    """The index's parts, per lesson. The index is NEVER shown without them.

    `metrics/activity.py` publishes `parts` precisely so that no surface can render the
    total alone, and this is that promise on the cabinet's surface: every column here is a
    component with its own weight, and the last column is what it contributed.
    """
    latest = next((s for s in reversed(history.sessions)
                   if s.activity.available and s.activity.parts), None)
    if latest is None:
        return ('<p class="muted">Индекс ни за один урок не посчитан, поэтому и '
                "составляющих нет. Причина указана в таблице занятий.</p>")
    rows = "".join(
        f"<tr><td>{_e(part.label_ru)}</td><td>{part.value * 100:.0f}%</td>"
        f"<td>{part.weight:.2f}</td><td>{part.contribution * 100:.1f}</td>"
        f'<td class="none">{_raw_phrase(part.raw)}</td></tr>'
        for part in latest.activity.parts)
    return (f"<p>Разбор индекса за {_e(latest.date_local)} "
            f"(итого {latest.activity.index:.1f}):</p>"
            f'<div class="wrap"><table><thead><tr><th>составляющая</th><th>значение</th>'
            f"<th>вес</th><th>вклад</th><th>из чего посчитано</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            f"{_index_lower_bound_note(voided or {})}")


# ---------------------------------------------------------------------------
# The only trend this cabinet can draw: inside one lesson.
# ---------------------------------------------------------------------------


def _within_lesson_block(rows: list[dict[str, Any]]) -> str:
    """How this place's index moved ACROSS ITS OWN LESSON, one strip per lesson.

    **This is the headline of a place page, and it is drawn as tiles for that reason.** It is
    the only trend the cabinet can show today: a week-over-week statement about a child needs
    an attested seating plan, while within one lesson the place is the same place for the
    whole hour, so the comparison needs nothing extra. Six segment indices in six `.tile`s
    with their coverage underneath is the platform's own headline row — readable across a
    room, which a two-row table of 11px cells was not.

    **The coverage number sits inside the tile with the index, never in a separate row.** An
    index read without its coverage is the one misreading this block exists to prevent: an
    index that fell while coverage fell is an observation about the camera. The verdict
    already carries `visibility_bound` (how much of the fall the coverage change alone could
    explain) and that figure is rendered either way, because a reader who does not see it
    cannot tell the two cases apart.

    **The verdict is a word and takes no colour.** `DESIGN.md` rule 4: a direction is «ниже»
    first, and a copy of this page gets printed in grey.
    """
    if not rows:
        return ""

    blocks = []
    for row in rows:
        segments = row.get("segments") or []
        cells = []
        for segment in segments:
            activity = segment.get("activity") or {}
            coverage = segment.get("coverage")
            shown = ("" if coverage is None else f" · обзор {coverage:.0%}")
            label = f"отрезок {segment.get('ordinal')}{shown}"
            if activity.get("available"):
                cells.append(_tile(f"{activity['index']:.0f}", label))
            else:
                cells.append(_tile("не считался", label, css="void",
                                   title=str(activity.get("reason") or "")))

        bound = row.get("visibility_bound")
        bound_line = ""
        if bound is not None and bound > 0.05:
            bound_line = (
                f"<p>Из этого изменения падение обзора само по себе могло дать до "
                f"{bound:.1f} балла — остаток на поведение.</p>")
        elif bound is not None:
            bound_line = ("<p>Обзор между началом и концом урока не падал: изменение "
                          "<strong>не объясняется видимостью</strong>.</p>")

        date = row.get("date_local") or "дата не прочитана"
        minutes = None
        if segments:
            minutes = ((segments[0].get("end_minute") or 0)
                       - (segments[0].get("start_minute") or 0))
        header = _e(str(date))
        if minutes:
            header += f" · {len(segments)} отрезков по ≈{minutes:.0f} мин"

        blocks.append(
            f"<h3>{header}</h3>"
            f'<div class="tiles">{"".join(cells)}</div>'
            f'<div class="panel"><p><strong>'
            f'{_e(str(row.get("direction_ru") or ""))}</strong></p>'
            f"<p>{_e(str(row.get('reason_ru') or ''))}</p>{bound_line}</div>")

    return (
        f"<h2>Как менялся показатель ВНУТРИ урока</h2>"
        f'<p class="muted">Сравнение — этого места с самим собой в пределах одного урока, '
        f"и больше ни с чем.</p>"
        f"{''.join(blocks)}"
        f"{_details('Почему это не динамика по неделям и с чем эти отрезки несравнимы', '<p>Урок нарезан на равные отрезки, и показатель каждого отрезка посчитан только по его наблюдениям. <strong>Это не динамика по неделям</strong> и не утверждение о ребёнке между уроками: для такого утверждения нужен подписанный план рассадки, которого здесь нет.</p><p>Показатели отрезков сравнимы только друг с другом. С показателем за весь урок их сравнивать нельзя: событийная составляющая нормирована на урок, поэтому у отрезка она систематически ниже.</p>')}")


# ---------------------------------------------------------------------------
# Pages.
# ---------------------------------------------------------------------------


def _empty_page(summary: dict[str, Any]) -> str:
    return _page("Кабинет психолога: пока пусто", f"""
<h1>Кабинет психолога</h1>
{_tiles(_tile(str(summary['lessons']), "уроков в базе"),
        _tile(str(summary['runs']), "измерений"),
        _tile(str(summary['places']), "мест"))}
<div class="panel">
  <h3>Пока пусто</h3>
  <p>В базу кабинета не загружено ни одного разбора урока. Это не сбой отчёта: страница
     выглядит так до первого импорта.</p>
  <p>Чтобы здесь что-то появилось:</p>
  <ol>
    <li><code>classvision analyse &lt;запись.mp4&gt;</code> — разобрать запись;</li>
    <li><code>classvision cabinet import out/&lt;имя&gt;.analysis.json --room-key D14
        --class 3-Б</code> — положить разбор в кабинет.</li>
  </ol>
  <p>Недельная динамика по ученику появляется не сразу: нужно
     {weekly.REQUIRED_LESSONS} разобранных
     {_plural(weekly.REQUIRED_LESSONS, 'урока', 'урока', 'уроков')} этого ученика и
     подписанный план рассадки. До тех пор страница честно показывает счётчики по
     МЕСТАМ.</p>
</div>
""", active="index", comparable=False)


def _index_page(entries: list[tuple[str, str, str, ClassView]],
                summary: dict[str, Any], comparable: ComparableView,
                comparable_link: str) -> str:
    """The cabinet root, organised BY CLASS and only then by room.

    The room is where the camera hangs; the class is what a school has and what a
    psychologist was asked about. The previous version emitted one flat card per (room,
    class) pair headed «Класс X · комната Y», so two rooms sharing a class key produced two
    cards a reader naturally read as two weeks of one class — which, with the two recordings
    this cabinet actually holds, is the exact false sentence the whole surface is organised
    against. Grouping by class puts the dangerous case where it cannot be missed: the
    cross-room line sits at the top of its own group, above the links into it, rather than
    in a caveat further down one of them.
    """
    by_class: dict[str, list[tuple[str, str, str, ClassView]]] = {}
    for room, klass, link, view in entries:
        by_class.setdefault(str(klass), []).append((room, klass, link, view))

    across = {entry.class_key: entry for entry in comparable.classes}
    groups = []
    for klass, items in sorted(by_class.items()):
        entry = across.get(klass)
        display = entry.display_ru if entry else f"класс {klass}"
        warning = (_cross_room_line(entry.rooms, link=comparable_link) if entry else "")
        unstated = ('<p class="muted">Класс при загрузке не указывали: «-» — это не имя '
                    "класса, а его отсутствие. Две записи под этим ключом не становятся от "
                    "этого одним классом.</p>" if entry and not entry.stated else "")
        cards_of_class = "".join(
            f'<a class="tile" href="{_e(link)}"><strong>комната {_e(room)}</strong>'
            f'<span class="sub">занятий '
            f"{sum(1 for r in view.lessons if r['continues_lesson_id'] is None)} "
            f"(записей {len(view.lessons)}) · мест {len(view.histories)}, из них учеников "
            f"{len(view.pupil_histories)}"
            + (" · уроков с датой нет" if not view.span else
               f" · {view.span[0][0]}-W{view.span[0][1]:02d}" if len(view.span) == 1 else
               f" · {view.span[0][0]}-W{view.span[0][1]:02d}…"
               f"{view.span[-1][0]}-W{view.span[-1][1]:02d}")
            + "</span></a>"
            for room, _klass, link, view in items)
        groups.append(f"<h2>{_e(display)}</h2>{unstated}{warning}"
                      f'<div class="grid">{cards_of_class}</div>')
    cards = "".join(groups)

    # `attested_places` is rendered as a plain number even at zero, and NOT as «не
    # измерялось»: it is a genuine COUNT(*), so 0 is a finding — nobody has signed a seating
    # plan — and not an absence of measurement. The italic refusal style would say the
    # opposite of what the query knows. Same rule as the canteen's two unrecognised-session
    # tiles. Assembled here rather than inside the page's f-string because a comment inside
    # one is Python 3.12 syntax and this package supports 3.11.
    headline = _tiles(
        _tile(str(summary["lessons"]), "записей загружено"),
        _tile(str(summary["sessions"]), "занятий"),
        _tile(str(summary["runs"]), "измерений"),
        _tile(str(summary["places"]), "мест под наблюдением"),
        _tile(str(summary["attested_places"]), "мест с подписанным планом рассадки"))

    # The permanent sentence, ABOVE the numbers — the one place on this surface where prose
    # precedes a tile, and for the reason `web/templates/psychologist.html` gives for doing
    # the same: a reader who scans a list and leaves has already formed an impression, and
    # this is the line that must not come after the claim.
    return _page("Кабинет психолога", f"""
<h1>Кабинет психолога</h1>
{_reading_line()}
{headline}
{_details("Что в этих числах, а чего в кабинете нет",
          f"<ul><li>Записей с прочитанной датой — {summary['lessons_dated']} из "
          f"{summary['lessons']}. Запись без даты не попадает ни в одну недельную сводку: "
          f"это отсутствие наблюдения, а не ноль.</li>"
          f"<li>Подписанных мест — {summary['attested_places']}. Пока их нет, каждое место "
          f"называется «место N», а не именем: ни лицо, ни трекер имени здесь не "
          f"создают.</li>"
          f"<li>Машинных записок-ориентиров: показывается {summary['notes_available']} из "
          f"{summary['notes']} запрошенных. Записка — не измерение, и отчёт полон без "
          f"неё.</li>"
          f"<li>«Вовлечённость» здесь не измеряется и не выводится, а ни один порог в этом "
          f"отчёте не проверен на размеченном вручную уроке.</li></ul>")}
<div class="pages"><a href="{_e(comparable_link)}">Что сравнимо между уроками →</a></div>
<p class="muted">Свойства самих записей: видимость, число найденных мест, события по
   комнате, положение взрослого. Единственная страница, где записи из разных комнат стоят
   рядом — и там же написано, почему это не динамика.</p>
{cards}
""", active="index")


def _comparable_page(comparable: ComparableView, links: dict[int, str],
                     class_links: dict[tuple[str, str], str]) -> str:
    """The only table in this cabinet where two rooms are allowed to sit side by side.

    **The cross-room warning is said HERE, in full, and only here.** Every other page carries
    one line and a link back. That is not a saving of space: fifteen verbatim copies of a
    six-bullet block is the volume at which a reader learns the shape of the block and stops
    reading it, and the readers it stops being read by are exactly the ones it was written
    for.
    """
    if not comparable.lessons:
        return _page("Что сравнимо между уроками", """
<h1>Что сравнимо между уроками</h1>
<div class="panel"><h3>Уроков нет</h3>
<p>В кабинете нет ни одного разобранного урока, поэтому сравнивать нечего. Это отсутствие
   данных, а не пустая таблица.</p></div>
""", active="lessons")

    warnings = "".join(_cross_room_block(entry.rooms, entry.class_key,
                                         link=None)
                       for entry in comparable.cross_room)

    counter_rows = []
    for lesson in comparable.lessons:
        voided = {key: {"full": True, "why": why}
                  for key, why in lessons_mod.voided_counters(lesson.unmeasured).items()}
        counters = "".join(_counter_cell(lesson.counts.get(key), voided.get(key))
                           for key, _ in COUNTER_KEYS)
        unassigned = lesson.unassigned_share
        href = links.get(lesson.session_id)
        title = _e(lesson.date_local or "без даты")
        cell = f'<a href="{_e(href)}">{title}</a>' if href else title
        if not lesson.run_ids:
            counter_rows.append(
                f'<tr><td>{cell}<br><span class="none">'
                f"{_e(lesson.room_key)} · {_e(lesson.week_label)}</span></td>"
                f'<td class="none" colspan="{5 + len(COUNTER_KEYS)}">для этого '
                f"урока не выбран расчётный прогон — сравнивать нечего. Это не ноль.</td>"
                f"</tr>")
            continue
        counter_rows.append(
            f'<tr><td>{cell}<br><span class="none">'
            f"{_e(lesson.room_key)} · {_e(lesson.week_label)}</span></td>"
            f"<td>{'—' if lesson.window_minutes is None else f'{lesson.window_minutes:.0f}'}"
            f"</td><td>{len(lesson.pupil_places)}</td>"
            f"{_coverage_cell(lesson.coverage_median)}{_coverage_cell(lesson.coverage_min)}"
            f"<td>{'—' if unassigned is None else f'{unassigned * 100:.1f}'}</td>"
            f"{counters}</tr>")

    teacher_rows = []
    for lesson in comparable.lessons:
        teacher = lesson.teacher
        title = _e(lesson.date_local or "без даты")
        if teacher is None or not teacher.get("available"):
            why = ((teacher or {}).get("reason_ru")
                   or lesson.adult_place_missing_reason_ru
                   or "разбора взрослого в этом измерении нет")
            teacher_rows.append(
                f'<tr><td>{title}<br><span class="none">{_e(lesson.room_key)}</span></td>'
                f'<td class="none" colspan="{len(lessons_mod.TEACHER_STATES) + 1}">'
                f"взрослый в этой записи не измерен: {_e(why)}. Это не ноль.</td></tr>")
            continue
        cells = "".join(
            (f'<td class="void" title="{_e(state["voided_why_ru"])}">не изм.</td>'
             if state["voided_why_ru"] else
             _percent_cell(state["share_of_lesson_percent"]))
            for state in teacher["states"])
        teacher_rows.append(
            f'<tr><td>{title}<br><span class="none">{_e(lesson.room_key)}</span></td>'
            f"{_percent_cell(teacher['attributed_percent'])}{cells}</tr>")

    teacher_heads = "".join(f"<th>{_e(label)}</th>"
                            for _key, label in lessons_mod.TEACHER_STATES)
    definitions = "".join(
        f"<li><strong>{_e(column['label'])}</strong> — {_e(column['is'])}"
        f'<br><span class="none">Это НЕ: {_e(column["is_not"])}</span></li>'
        for column in comparable.columns)
    counter_heads = "".join(f"<th>{_e(label)}</th>" for _key, label in COUNTER_KEYS)

    single = ""
    if len(comparable.lessons) == 1:
        single = ('<div class="panel"><h3>В кабинете одна запись</h3>'
                  "<p>Сравнивать её не с чем. Таблица ниже — одна строка, и она описывает "
                  "саму запись; второй строки нет, и это не ноль.</p></div>")

    cards = "".join(
        f'<a class="tile" '
        f'href="{_e(class_links[(str(room["room_key"]), str(entry.class_key))])}">'
        f"<strong>{_e(entry.display_ru)} · комната {_e(room['room_key'])}</strong>"
        f'<span class="sub">записей {room["lessons"]}, ученических мест '
        f"{room['pupil_places']}, {_e(room['first_date'] or 'без даты')}</span></a>"
        for entry in comparable.classes for room in entry.rooms
        if (str(room["room_key"]), str(entry.class_key)) in class_links)

    return _page("Что сравнимо между уроками", f"""
<h1>Что сравнимо между уроками</h1>
{_tiles(_tile(str(len(comparable.lessons)), "записей в кабинете"),
        _tile(str(len(comparable.rooms)),
              _plural(len(comparable.rooms), "комната", "комнаты", "комнат")
              + ": " + ", ".join(comparable.rooms)))}
<div class="warning"><strong>Чего здесь нет.</strong>
<p>{_e(lessons_mod.WHAT_THIS_IS_NOT_RU)}</p></div>
<p>{_e(lessons_mod.WHAT_THIS_IS_RU)}</p>

{warnings}
{single}

<h2>Записи</h2>
<div class="wrap"><table><thead><tr><th>урок</th><th>минут разбора</th>
<th>ученических мест</th><th>видимость (медиана)</th><th>видимость (минимум)</th>
<th>вне мест, %</th>{counter_heads}</tr></thead>
<tbody>{''.join(counter_rows)}</tbody></table></div>
<p class="muted">Счётчики — суммы по всем ученическим местам комнаты за урок. «Не
   измерялось» означает, что на этой камере событие не может быть зафиксировано вовсе, и
   сравнивать такой столбец между камерами нельзя даже как свойство записи.</p>
{''.join(f'<div class="warning"><strong>Ноль, которому противоречит эта же запись '
         f'({_e(lesson.date_local or "без даты")}, {_e(lesson.room_key)})</strong>'
         f'<p>{_e(lesson.board_conflict_ru)}</p></div>'
         for lesson in comparable.lessons if lesson.board_conflict_ru)}

<h2>Где был взрослый, по записям</h2>
<div class="wrap"><table><thead><tr><th>урок</th><th>отнесено к положению, %</th>
{teacher_heads}</tr></thead><tbody>{''.join(teacher_rows)}</tbody></table></div>
<p class="muted">Все доли — от ВСЕГО урока, включая время, когда взрослого не нашли в кадре:
   это отдельный столбец, а не вычет.</p>
{_details("Что означает «не изм.» в этой таблице",
          "<p>Зона для этого состояния на камере не размечена, состояние не может "
          "возникнуть, и его время попало в соседние столбцы, которые на эту величину "
          "завышены. Это свойство разметки камеры, а не утверждение о работе учителя ни в "
          "одну сторону.</p>", css="warn")}

{_details(f"Что означает каждый столбец — и чего он не означает ({len(comparable.columns)} "
          f"столбца)", f"<ul>{definitions}</ul>")}

<h2>Классы и комнаты</h2>
<div class="grid">{cards}</div>
""", active="lessons")


def _overview(view: ClassView, links: dict[int, str],
              stored_notes: dict[str, Any] | None = None,
              lesson_links: dict[int, str] | None = None,
              lesson_views: list[LessonView] | None = None,
              comparable_link: str = "lessons.html") -> str:
    lesson_links = lesson_links or {}
    lesson_views = lesson_views or []
    voided = _voided_for_class(view)

    lesson_rows = []
    for row in view.lessons:
        marks = ""
        if row["continues_lesson_id"]:
            marks += (f'<span class="none"> · продолжение записи №'
                      f'{row["continues_lesson_id"]}</span>')
        if row["overlap_note"]:
            marks += f'<span class="none"> · {_e(row["overlap_note"])}</span>'
        date = _e(row["date_local"] or "без даты")
        href = lesson_links.get(int(row["lesson_id"]))
        cell = f'<a href="{_e(href)}">{date}</a>' if href else date
        lesson_rows.append(
            f"<tr><td>{cell}{marks}</td>"
            f"<td>{_e(row['started_at'] or '—')}</td>"
            f"<td>{(row['duration_seconds'] or 0) / 60:.0f}</td>"
            f"<td>{_e(row['clock_source'])}</td>"
            f"<td>{_e(row['seats_found'])}</td>"
            f'<td class="none">{_e((row["selected_run_id"] or "—")[:12])}</td>'
            f'<td class="none">{_e(row["thresholds_sha"])}</td></tr>')
    lessons = "".join(lesson_rows)

    # What happened IN THE ROOM, lesson by lesson. This is the class-level half of «что
    # произошло»: the per-place figures are one click away and this table deliberately does
    # not repeat them, because a room total and a child's total look identical in a table
    # and mean entirely different things.
    class_rows = []
    for lesson in lesson_views:
        href = lesson_links.get(lesson.parts[0]["lesson_id"]) if lesson.parts else None
        title = _e(lesson.date_local or "без даты")
        cell = f'<a href="{_e(href)}">{title}</a>' if href else title
        head = f'<tr><td>{cell}<br><span class="none">{_e(lesson.week_label)}</span></td>'
        if not lesson.run_ids:
            # «мест 0, видимость —, поднимал руку 0» for a lesson nobody has chosen a
            # measurement for is a row of zeros about an hour that was recorded and analysed.
            # One cell saying so, exactly as `_adult_page` does for an unlocated adult.
            class_rows.append(
                f'{head}<td class="none" colspan="{4 + len(COUNTER_KEYS)}">'
                f"для этого урока не выбран расчётный прогон — его содержимое не входит "
                f"ни в одну сводку. Это не ноль: выберите прогон командой "
                f"<code>classvision cabinet select-run &lt;run_id&gt;</code>.</td></tr>")
            continue
        counters = "".join(_counter_cell(lesson.counts.get(key), voided.get(key))
                           for key, _ in COUNTER_KEYS)
        window = ('<td class="none">—</td>' if lesson.window_minutes is None
                  else f"<td>{lesson.window_minutes:.0f}</td>")
        class_rows.append(
            f"{head}<td>{len(lesson.pupil_places)}</td>{window}"
            f"{_coverage_cell(lesson.coverage_median)}{_coverage_cell(lesson.coverage_min)}"
            f"{counters}</tr>")
    class_table = (
        f'<div class="wrap"><table><thead><tr><th>урок</th><th>ученических мест</th>'
        f"<th>минут разбора</th><th>видимость (медиана)</th><th>видимость (минимум)</th>"
        f"{_counter_heads(voided)}</tr></thead>"
        f"<tbody>{''.join(class_rows)}</tbody></table></div>"
        f'<p class="muted">Счётчики здесь — суммы ПО ВСЕЙ КОМНАТЕ за урок, а не по '
        f"ребёнку: они меняются, когда меняется число детей в кадре и качество обзора. "
        f"Число мест — это то, сколько устойчивых мест нашла кластеризация, а не "
        f"численность класса.</p>{_board_conflicts(lesson_views)}"
        f"{_voided_note(voided)}") if class_rows else (
        '<p class="muted">Ни по одному уроку этого класса нет выбранного измерения, '
        "поэтому таблицы нет. Это не пустой класс: выберите прогон командой "
        "<code>classvision cabinet select-run &lt;run_id&gt;</code>.</p>")

    class_note = ""
    if not view.class_stated:
        class_note = _details(
            "Класс при загрузке не указан: «-» — это не имя класса, а его отсутствие",
            "<p>Ключ класса — «-», то есть его никто не вводил: артефакт разбирает ФАЙЛ и "
            "не знает ни комнаты, ни класса. Две записи под этим ключом не становятся от "
            "этого одним классом.</p>"
            '<p class="muted">Задаётся при загрузке: <code>classvision cabinet import … '
            f"--room-key {_e(view.room_key)} --class 8-А</code>.</p>", css="warn")

    places = []
    for history in view.histories:
        state = history.trend_view.state
        badge = {"available": "динамика построена",
                 "insufficient_lessons": history.trend_view.headline_ru,
                 "identity_not_established": "место не подписано",
                 "place_unstable": "место опознано не во всех уроках",
                 "no_dated_lessons": "уроки без даты — в неделю не попадают",
                 "not_applicable_adult": "индекс и динамика не считаются",
                 "no_data": "уроков нет"}.get(state, state)
        if history.place["role"] != "pupil":
            totals = ("счётчики ученика к взрослому не применяются — только положение по "
                      "каждому уроку")
            role = " · взрослый"
        elif not history.sessions:
            # NOT «поднимал руку — 0». A place whose only recordings are undated has been
            # watched, sometimes for a whole hour; what is missing is the date, not the
            # behaviour. The previous card printed a row of confident zeros directly under
            # a badge saying the lessons were not accumulated, which is rule 3 of this
            # project broken in the space of one line.
            totals = ("счётчики за период не накоплены — это не ноль; по каждому уроку "
                      "они есть в разборе самого урока")
            role = ""
        else:
            # «выходил к доске — 0» on a camera whose board is behind the lens is the same
            # confident zero this module now refuses everywhere else, and the card is where
            # a reader meets it first.
            totals = ", ".join(
                f"{label} — не измерялось" if voided.get(key, {}).get("full")
                else f"{label} — {history.totals[key]}"
                for key, label in COUNTER_KEYS[:3])
            role = ""
        # «видимость от 0 %» for a place with no accumulated session was the same defect in
        # a second place: `coverage_min or 0` turned «нечего усреднять» into a measurement.
        coverage = ("видимость не накоплена" if history.coverage_min is None
                    else f"видимость от {history.coverage_min * 100:.0f}%")
        places.append(
            f'<a class="tile" href="{_e(links[int(history.place["place_id"])])}">'
            f"<strong>{_e(history.display_name_ru)}{role}</strong>"
            f'<span class="sub">занятий {len(history.sessions)} · '
            f"{coverage} · {_e(badge)}</span>"
            f'<span class="sub">{_e(totals)}</span></a>')

    threshold_warning = ""
    # Counted over THIS class's runs. `store_summary["threshold_sets"]` is the whole
    # cabinet's count, and using it hung «уроки измерены разными порогами» on every class
    # in the school as soon as any one class was re-analysed once.
    if view.threshold_sets_in_class > 1:
        threshold_warning = (
            '<p class="warning">Уроки этого класса измерены <strong>разными наборами '
            "порогов</strong> — в столбце «пороги» ниже больше одного значения. «Ребёнок "
            "стал менее активен» имеет смысл только если обе недели мерили одинаково; "
            "сравнение уроков с разными порогами — это сравнение настроек.</p>")

    extra = ""
    if view.extra_runs:
        rows = "".join(
            f"<li>{_e(r['date_local'])}: прогон <code>{_e(r['run_id'][:12])}</code> "
            f"(пороги {_e(r['thresholds_sha'])}) — сохранён, но в сводки НЕ входит; "
            f"используется <code>{_e((r['selected_run_id'] or '—')[:12])}</code></li>"
            for r in view.extra_runs)
        extra = _details(
            f"Повторные измерения тех же уроков — {len(view.extra_runs)} "
            f"{_plural(len(view.extra_runs), 'прогон', 'прогона', 'прогонов')} вне сводок",
            f'<p class="muted">Разбор с другими порогами — это НОВОЕ измерение, а не '
            f"уточнение прежнего, поэтому он сохранён отдельно и ничего не "
            f"перезаписал.</p><ul>{rows}</ul>")

    unplaced = ""
    if view.unplaced_seat_lessons:
        rows = "".join(f"<li>{_e(r['date_local'])}, {_e(r['seat_label'])}: "
                       f"{_e(r['place_match_reason_ru'])}</li>"
                       for r in view.unplaced_seat_lessons)
        unplaced = _details(
            f"Места, не привязанные к истории — {len(view.unplaced_seat_lessons)}",
            f'<p class="muted">Эти наблюдения сохранены полностью, но не участвуют ни в '
            f"одной накопительной сводке: присоединить их к ближайшей истории значило бы "
            f"приписать урок не тому ребёнку.</p><ul>{rows}</ul>", css="warn")

    # Recordings and lessons are counted separately and both are shown. A DVR that splits
    # one hour into two files makes them differ, and a page that printed only the file
    # count would report two lessons in a week that had one.
    sessions = sum(1 for row in view.lessons if row["continues_lesson_id"] is None)
    return _page(
        f"{view.class_display_ru} · комната {view.room_key}",
        f"""
<h1>{_e(view.class_display_ru)} · комната {_e(view.room_key)}</h1>
{_tiles(_tile(str(sessions), "занятий"),
        _tile(str(len(view.lessons)), "записей загружено"),
        _tile(str(len(view.pupil_histories)), "ученических мест"),
        _tile(str(len(view.histories)), "мест всего, вместе со взрослым"),
        _tile(str(len(view.span)), "недель в периоде"))}
{_reading_line()}
{_cross_room_line(view.class_rooms, here=view.room_key, link=comparable_link)}
{threshold_warning}

{_note_block(view, stored_notes)}

<h2>Уроки</h2>
<div class="wrap"><table><thead><tr><th>дата</th><th>начало</th><th>минут</th>
<th>откуда время</th><th>мест найдено</th><th>прогон</th><th>пороги</th></tr></thead>
<tbody>{lessons}</tbody></table></div>
{class_note}
{extra}
{unplaced}

<h2>Что происходило в комнате, по урокам</h2>
{class_table}

<h2>Места</h2>
<p class="muted">Порядок — по расположению в комнате. Это не рейтинг: сортировка по
   показателю была бы рекомендацией с опущенным обоснованием, а сравнивать детей друг с
   другом этот показатель не позволяет — ученик у окна и ученик в глубине класса видны
   камере по-разному.</p>
<div class="grid">{''.join(places)}</div>

{_caveats_block(view)}
""", active="class", section=(_class_page_name(view.room_key, view.class_key),
                              f"{view.class_display_ru} · {view.room_key}"))


# ---------------------------------------------------------------------------
# One lesson: the page somebody actually opens first.
# ---------------------------------------------------------------------------


def _teacher_lesson_block(lesson: LessonView) -> str:
    """Where the adult was, this lesson. No index, no comparison, no assessment.

    The split shown is `presence.state_share_of_lesson_percent` — «где в комнате он был» —
    and never `metrics.at_desk_share_of_observed`, which answers «что делало его тело за
    его собственным местом» and reads 96.5 % on the very camera whose position split reports
    `at_desk: 0.0` because no teacher-desk zone was drawn. Both numbers are true, they have
    different denominators and different subjects, and putting them in one column would be
    `MEASUREMENTS.md` §8's name collision arriving one consumer further along. The adult's
    own page keeps the ledger half, under its own heading and with that difference stated.

    **The minutes lead, as tiles, and «у доски был ХОТЬ КТО-ТО» stands in the same row.**
    Those two numbers are the pair a reader needs together on the D14 lesson: the adult was
    at the board 4 minutes, somebody was there for 30, and the pupils' «выходил к доске» is
    nonetheless 0 — which is a limitation of what a place-based counter can see and not a
    fact about children. The tile says whose number it is; the contradiction block above
    says what the zero does not mean.
    """
    teacher = lesson.teacher
    if teacher is None:
        why = lesson.adult_place_missing_reason_ru or (
            "в этом прогоне нет разбора взрослого. Это отсутствие измерения, а не вывод о "
            "том, что взрослого в классе не было.")
        return (f'<div class="panel"><h3>Взрослый в этом уроке не измерен</h3>'
                f"<p>{_e(why)}</p></div>")
    if not teacher.get("available"):
        return (f'<div class="panel"><h3>Взрослый в этом уроке не измерен</h3>'
                f"<p>{_e(teacher.get('reason_ru') or 'причина в артефакте не указана')}. "
                f"Это не ноль: ни одна доля ниже не подставляется.</p></div>")

    cells = []
    for state in teacher["states"]:
        if state["voided_why_ru"]:
            cells.append(_tile("не измерялось", str(state["label_ru"]), css="void",
                               title=str(state["voided_why_ru"])))
            continue
        minutes = state["minutes_of_lesson"]
        share = state["share_of_lesson_percent"]
        value = "—" if minutes is None else f"{minutes} мин"
        label = str(state["label_ru"])
        if share is not None:
            label += f" · {float(share):.1f} % урока"
        cells.append(_tile(value, label, css="" if minutes is not None else "void"))

    occupancy = teacher.get("board_occupancy") or {}
    if occupancy.get("available"):
        cells.append(_tile(
            f"{occupancy['minutes']} мин",
            f"у доски был ХОТЬ КТО-ТО · {occupancy['share_of_lesson_percent']} % урока",
            title=str(occupancy.get("means_ru") or "")))

    rows = "".join(
        f"<tr><td>{_e(state['label_ru'])}</td>"
        + (f'<td class="void" colspan="3">не измерялось — '
           f"{_e(state['voided_why_ru'])}</td>"
           if state["voided_why_ru"] else
           f"{_percent_cell(state['share_of_lesson_percent'])}"
           f"<td>{state['minutes_of_lesson'] if state['minutes_of_lesson'] is not None else '—'}</td>"
           f"<td>{state['episodes'] if state['episodes'] is not None else '—'}</td>")
        + "</tr>"
        for state in teacher["states"])

    inflated = ""
    if teacher["voided"]:
        inflated = (
            "<p>Время, которое не удалось отнести к неразмеченным зонам, не исчезло: оно "
            "попало в «среди учеников» и «перемещается», поэтому эти две доли для этой "
            "записи ЗАВЫШЕНЫ на неизвестную величину. Это свойство разметки камеры, а не "
            "утверждение о работе учителя ни в одну сторону.</p>")

    board_note = teacher.get("board_direction_of_error_ru") or ""
    occupancy_means = _e(occupancy.get("means_ru") or "") if occupancy.get("available") else ""

    # The occupancy sentence goes directly UNDER the tiles, because it explains the last of
    # them («у доски был ХОТЬ КТО-ТО») and nothing else on the page. Printed after the table
    # it read as that table's caption — and that table is about where the ADULT was, so the
    # sentence «личность здесь не устанавливается» landed on the wrong subject.
    return f"""
{_tiles(*cells)}
{f'<p class="muted">{occupancy_means}</p>' if occupancy_means else ''}
<div class="warning"><strong>Это не оценка работы учителя.</strong>
<p>{_e(teacher.get('not_an_assessment_ru') or '')}</p></div>
<h3>Где взрослый был в комнате</h3>
<div class="wrap"><table><thead><tr><th>где он был</th><th>доля урока, %</th>
<th>минут</th><th>эпизодов</th></tr></thead><tbody>{rows}</tbody></table></div>
{_details("Как считаются эти доли, и что в них не входит",
          f"<p>Доля урока, минуты и число эпизодов — по положению взрослого В КОМНАТЕ. "
          f"Знаменатель — весь урок, а не время, когда его видели: «не найден в кадре» — "
          f"отдельная строка, а не вычет из остальных. Отнесено к какому-либо положению: "
          f"{teacher['attributed_percent']} % урока.</p>"
          f"<p>Границ между эпизодами положения: "
          f"{teacher['transitions_between_episodes']} (без учёта потерь из кадра — "
          f"{teacher['transitions_excluding_out_of_frame']}); это НЕ «сколько раз он менял "
          f"позу за своим столом» — то другое число, и оно на странице места "
          f"взрослого.</p>{inflated}"
          + (f"<p>{_e(board_note)}</p>" if board_note else ""))}
"""


def _reservations_block(lesson: LessonView) -> str:
    items = lessons_mod.reservations(lesson)
    if not items:
        # Carries its own heading in the sentence, because with nothing reserved there is no
        # `<details>` to hang one on — and «все места были видны» with no framing reads as
        # praise for the class rather than as a statement about the camera.
        return ('<p class="muted"><strong>Чего этот урок не подтверждает:</strong> все '
                "ученические места этой записи были видны больше половины разобранных "
                "кадров и установили базовую позу. Это утверждение об обзоре камеры, а не "
                "о детях.</p>")
    rows = "".join(
        f"<li><strong>{_e(r.label_ru)}</strong> — {_e(r.what_ru)}. "
        f'<span class="none">{_e(r.detail_ru)}</span></li>' for r in items)
    return _details(
        f"Чего этот урок не подтверждает — {len(items)} "
        f"{_plural(len(items), 'место', 'места', 'мест')} с ограничением измерения",
        f"<ul>{rows}</ul>"
        f'<p class="muted">Это список ограничений ИЗМЕРЕНИЯ, а не список детей, на '
        f"которых стоит обратить внимание. Второго списка здесь нет и не будет: он был "
        f"бы пометкой, а система не помечает.</p>", css="warn")


def _lesson_page(lesson: LessonView, view: ClassView, *, back: str,
                 place_links: dict[int, str], comparable_link: str) -> str:
    """One lesson: what was recorded, what could not be measured, and what was counted."""
    voided = {key: {"full": True, "why": why}
              for key, why in lessons_mod.voided_counters(lesson.unmeasured).items()}
    adult_void = {"full": True,
                  "why": "счётчики ученика к взрослому не применяются: поднятая рука "
                         "взрослого — это указание на доску, а не участие в уроке."}
    heads = _counter_heads(voided)

    place_rows = []
    for place in lesson.places:
        if place.role == "adult":
            # One cell, not six «не измерялось» in a row. The counters are not missing for
            # the adult, they are inapplicable to him, and six identical refusals read as a
            # broken export rather than as a rule.
            counters = (f'<td class="void" colspan="{len(COUNTER_KEYS)}">'
                        f"{_e(adult_void['why'])}</td>")
        else:
            counters = "".join(_counter_cell(place.counts.get(key), voided.get(key))
                               for key, _ in COUNTER_KEYS)
        if place.role == "adult":
            index = ('<td class="none" title="индекс для взрослого не считается">'
                     "не считается</td>")
        elif place.index_available and place.index is not None:
            index = f"<td>{place.index:.1f}</td>"
        else:
            index = (f'<td class="none" title="{_e(place.index_reason_ru)}">'
                     f"не считался</td>")
        href = place_links.get(place.place_id) if place.place_id is not None else None
        name = _e(place.label_ru)
        cell = f'<a href="{_e(href)}">{name}</a>' if href else name
        if place.place_id is None:
            cell += (f'<br><span class="none">не привязано к истории: '
                     f"{_e(place.place_match_reason_ru)}</span>")
        place_rows.append(
            f"<tr><td>{cell}</td>"
            f"<td>{'ученик' if place.role == 'pupil' else 'взрослый'}</td>"
            f"{_coverage_cell(place.coverage)}"
            f"<td>{place.observed_minutes:.0f}</td>{index}{counters}</tr>")

    unmeasured = "".join(
        f"<li><strong>{_e(item.get('what'))}</strong> — {_e(item.get('why'))}</li>"
        for item in lesson.unmeasured)
    adult_missing = (f"<p><strong>{_e(lesson.adult_place_missing_reason_ru)}</strong></p>"
                     if lesson.adult_place_missing_reason_ru else "")
    # The summary carries the heading verbatim, because it is the heading: this is the block
    # the lesson page leads with, and a test asserts it precedes the first table. Collapsed,
    # not removed -- and its count is on the closed summary, so a reader learns there are
    # four such items without opening anything.
    count = len(lesson.unmeasured)
    unmeasured_block = _details(
        f"Что этой записью НЕ измерено — {count} "
        f"{_plural(count, 'пункт', 'пункта', 'пунктов')}" if count else
        "Что этой записью НЕ измерено: артефакт не перечислил ничего",
        (f"<ul>{unmeasured}</ul>" if unmeasured else
         "<p>Артефакт не перечислил ничего.</p>") + adult_missing, css="warn")

    unassigned = lesson.unassigned_share
    unassigned_html = (
        f"{unassigned * 100:.1f} % наблюдений человека не попали ни на одно найденное место"
        if unassigned is not None else
        "доля наблюдений вне мест этим прогоном не сохранена — это не ноль")

    notes_html = "".join(f"<li>{_e(note)}</li>" for note in lesson.notes_ru)
    files = "".join(
        f"<li>запись №{part['lesson_id']}: {_e(part['started_at'] or 'без времени')}, "
        f"{(part['duration_seconds'] or 0) / 60:.0f} мин, часы — "
        f"{_e(part['clock_source'])}, измерение <code>"
        f"{_e((part['selected_run'] or '—')[:12])}</code>, пороги "
        f"{_e(part['thresholds_sha'])}</li>" for part in lesson.parts)

    counters_class = "".join(_counter_cell(lesson.counts.get(key), voided.get(key))
                             for key, _ in COUNTER_KEYS)
    window = ("не сохранено" if lesson.window_minutes is None
              else f"{lesson.window_minutes:.0f} мин")

    return _page(lesson.title_ru, f"""
<h1>{_e(lesson.title_ru)}</h1>
{_tiles(_tile(str(len(lesson.pupil_places)), "ученических мест"),
        _tile(window, "разобрано из записи"),
        _tile('—' if lesson.coverage_median is None
              else f'{lesson.coverage_median * 100:.0f}%', "видимость места, медиана"),
        _tile('—' if lesson.coverage_min is None
              else f'{lesson.coverage_min * 100:.0f}%', "видимость места, минимум"),
        _tile(f"{lesson.place_minutes:.0f}", "место-минут под наблюдением"))}
<p class="muted">{_e(lesson.week_label)} · запись
   {_e(lesson.started_at or 'время начала не прочитано')} … {_e(lesson.ended_at or '—')}
   (источник времени — {_e(lesson.clock_source)}), {lesson.analysed_frames} кадров
   разобрано. Разобранное окно короче записи: оно начинается тогда, когда класс появился
   в кадре.</p>
{_reading_line()}
{_cross_room_line(view.class_rooms, here=view.room_key, link=comparable_link)}

<p>Всё, чего этой записью не измерено, перечислено ниже: в таблицах на месте таких величин
   стоит <strong>«не измерялось»</strong>, а не ноль.</p>
{unmeasured_block}

<h2>Комната целиком в этом уроке</h2>
<div class="wrap"><table><thead><tr><th>по всей комнате за урок</th>{heads}</tr></thead>
<tbody><tr><td>суммарно по ученическим местам</td>{counters_class}</tr></tbody></table></div>
<p class="muted">Это сумма по комнате, а не показатель ребёнка: она меняется, когда меняется
   число детей в кадре и качество обзора.</p>
{f'<div class="warning"><strong>Ноль, которому противоречит эта же запись</strong>'
 f'<p>{_e(lesson.board_conflict_ru)}</p></div>' if lesson.board_conflict_ru else ''}
{_voided_note(voided)}
<p class="muted">{_e(unassigned_html)}; мест, не установивших базовую позу —
   {lesson.seats_never_settled}; кадров без единого человека —
   {lesson.frames_with_no_person}.</p>

<h2>По местам</h2>
<div class="wrap"><table><thead><tr><th>место</th><th>роль</th><th>видимость</th>
<th>минут под наблюдением</th><th>индекс</th>{heads}</tr></thead>
<tbody>{''.join(place_rows)}</tbody></table></div>
<p class="muted">Порядок — по расположению в комнате, а не по показателю. Индекс показан
   только там, где место было видно больше половины разобранных кадров; разбор индекса на
   составляющие — на странице места.</p>
{_index_lower_bound_note(voided)}
{_reservations_block(lesson)}

<h2>Взрослый</h2>
{_teacher_lesson_block(lesson)}

{_details(f"Из чего собран этот урок — {len(lesson.parts)} "
          f"{_plural(len(lesson.parts), 'файл', 'файла', 'файлов')}",
          f"<ul>{files}</ul>" + (f"<ul>{notes_html}</ul>" if notes_html else ""))}
{_caveats_block(view, include_unmeasured=False)}
""", section=(back, f"{view.class_display_ru} · {view.room_key}"))


# ---------------------------------------------------------------------------
# One place across the term.
# ---------------------------------------------------------------------------


def _place_page(history: PlaceHistory, view: ClassView, back: str,
                lesson_links: dict[int, str] | None = None,
                comparable_link: str = "lessons.html",
                lesson_views: list[LessonView] | None = None) -> str:
    place = history.place
    if place["role"] == "adult":
        return _adult_page(history, view, back, comparable_link)
    voided = _voided_for_class(view)
    conflicts = _board_conflicts(lesson_views or [])
    section = (back, f"{view.class_display_ru} · {view.room_key}")

    attestation = (
        f"<p>Имя проставлено по подписанному плану рассадки: "
        f"<strong>{_e(history.attested['full_name'] or history.attested['external_id'])}"
        f"</strong>, утвердил {_e(history.attested['attested_by'])} "
        f"{_e(history.attested['attested_at'])}, действует с "
        f"{_e(history.attested['valid_from'])}"
        f"{' по ' + _e(history.attested['valid_to']) if history.attested['valid_to'] else ''}. "
        f"Основание: {_e(history.attested['decision_ref'])}.</p>"
        if history.attested else
        '<p class="muted">Плана рассадки на это место нет, поэтому страница называет его '
        "<strong>местом, а не ребёнком</strong>. Так и должно быть: ни лицо, ни трекер "
        "имени здесь не создают. Система не ставит диагнозов и не даёт рекомендаций.</p>")

    if not history.sessions:
        # The empty page states WHICH emptiness. «Уроков нет» sends the reader to look for
        # a missing recording; «уроки есть, но без даты» sends them to the camera's clock.
        # `TrendView` already distinguishes them, so this page must not flatten them back.
        view_state = history.trend_view
        return _page(history.display_name_ru, f"""
<h1>{_e(history.display_name_ru)}</h1>
{attestation}
<div class="panel">
<h3>{_e(view_state.headline_ru)}</h3>
<p>{_e(view_state.detail_ru)}</p>
<p class="muted">Счётчиков здесь нет — и это отсутствие наблюдения, а не ноль.</p>
{_requests_block(history)}
</div>
{_caveats_block(view)}
""", section=section)

    within = (store.within_lesson_for_place(_CONNECTION[0], history.place["place_id"])
              if _CONNECTION[0] is not None else [])

    return _page(history.display_name_ru, f"""
<h1>{_e(history.display_name_ru)}</h1>
<p class="muted">{_e(view.class_display_ru)}, комната {_e(view.room_key)} ·
   занятий под наблюдением {len(history.sessions)} ·
   видимость от {(history.coverage_min or 0) * 100:.0f}% кадров</p>
{attestation}
{_cross_room_line(view.class_rooms, here=view.room_key, link=comparable_link)}

{_within_lesson_block(within)}

<h2>Динамика по неделям</h2>
{_trend_block(history, view)}

<h2>За всё время</h2>
{_counter_tiles(history.totals, voided)}
<p class="muted">Каждый итог относится только к тем занятиям, где место было видно —
   минимальная видимость по периоду {(history.coverage_min or 0) * 100:.0f}%.</p>
{_totals_qualifiers(history)}
{conflicts}

<h2>По занятиям</h2>
{_sessions_table(history, lesson_links, voided)}

<h2>Из чего складывается индекс</h2>
{_components(history, voided)}

{_details(f"По неделям: {len(history.weeks)} "
          f"{_plural(len(history.weeks), 'неделя', 'недели', 'недель')} в периоде, "
          f"{sum(1 for w in history.weeks if w.observed)} с уроками",
          '<p class="muted">Прочерк — недели, за которые уроков не загружено. Это НЕ ноль: '
          "ноль означает «наблюдали и не увидели», прочерк — «не наблюдали».</p>"
          + _weeks_table(history, voided))}
{_caveats_block(view)}
""", section=section)


# The adult's share fields as `classvision/1.1` names them. Each name states its own
# DENOMINATOR, and `metrics/teacher.py` renamed them from `seated_share` /
# `out_of_frame_share` precisely because the old names did not — the comment beside that
# rename records that the collision «seated» (four states at the desk) with «seated» (one of
# those four, 8.6 minutes against 46.3) had already produced a false sentence in a report.
#
# So a `classvision/1.0` artefact is NOT read through this column. The old keys hold a
# number that is probably the same quantity, and "probably" is not a denominator: putting it
# under a heading that names one is the exact substitution the rename exists to prevent, and
# the previous code did it silently with a `metrics.get(new, metrics.get(old))` fallback.
# Such rows are rendered as their own sentence, with the raw old-named values intact.
ADULT_SHARE_KEYS = ("at_desk_share_of_observed", "out_of_frame_share_of_lesson")
ADULT_LEGACY_KEYS = ("seated_share", "standing_or_away_share", "out_of_frame_share")


def _adult_page(history: PlaceHistory, view: ClassView, back: str,
                comparable_link: str = "lessons.html") -> str:
    """Position over time, per lesson, with no index and no accumulation. On purpose.

    Carries the cross-room line like every other page, and for a sharper reason than the
    pupil pages do: this one has a per-lesson table with a date column, so two rooms' rows
    would sit under one heading and read as one person's term. The adult of camera01 and the
    adult of D14 are no more the same person than the children are.
    """
    rows = []
    measured = 0
    for row in view.lessons:
        record = store.teacher_record(
            _CONNECTION[0], row["selected_run_id"]) if row["selected_run_id"] else None
        metrics = (record or {}).get("metrics") or {}
        if not metrics:
            continue
        date_cell = _e(row["date_local"] or "без даты")
        if not metrics.get("available"):
            # A lesson in which the adult was not located is ONE cell saying so, not a row
            # of numbers. Rendering it normally printed «видимость 0 %» and «смен положения
            # 0» — a confident measurement of a person nobody found — with the two share
            # columns silently blank beside them.
            rows.append(
                f'<tr><td>{date_cell}</td><td class="none" colspan="4">'
                f"взрослый на этой записи не измерен: "
                f"{_e(metrics.get('reason') or 'причина в артефакте не указана')}. "
                f"Это не ноль.</td></tr>")
            continue
        measured += 1
        if not any(key in metrics for key in ADULT_SHARE_KEYS):
            legacy = ", ".join(f"{key} {metrics[key]}"
                               for key in ADULT_LEGACY_KEYS if metrics.get(key) is not None)
            rows.append(
                f"<tr><td>{date_cell}</td>{_coverage_cell(metrics.get('coverage'))}"
                f'<td class="none" colspan="3">разбор схемы '
                f"{_e(row['run_schema'] or 'неизвестной версии')}: доли положения "
                f"взрослого назывались тогда иначе и не называли свой знаменатель, "
                f"поэтому под этот столбец они не подставляются. Как записано в "
                f"артефакте: {_e(legacy) or '—'}. Смены положения: "
                f"{_e(metrics.get('transitions'))}.</td></tr>")
            continue
        # The keys are the artefact's own, and they are PERCENTS already
        # (`metrics/teacher.py::_pct`), so they go through `_percent_cell` rather than
        # `_coverage_cell`, which multiplies a fraction by 100.
        rows.append(
            f"<tr><td>{date_cell}</td>"
            f"{_coverage_cell(metrics.get('coverage'))}"
            f"{_percent_cell(metrics.get('at_desk_share_of_observed'))}"
            f"{_percent_cell(metrics.get('out_of_frame_share_of_lesson'))}"
            + ('<td class="none">—</td>' if metrics.get("transitions") is None
               else f"<td>{_e(metrics['transitions'])}</td>") + "</tr>")
    disclaimer = next(
        (((store.teacher_record(_CONNECTION[0], r["selected_run_id"]) or {})
          .get("metrics") or {}).get("not_an_assessment_ru")
         for r in view.lessons if r["selected_run_id"]), None)

    return _page("Место взрослого", f"""
<h1>Место взрослого</h1>
{_tiles(_tile(str(len(rows)), "уроков с разбором взрослого"),
        _tile(str(measured), "из них с измеренным положением"),
        _tile("не считается", "индекс активности", css="void",
              title="счётчики и индекс ученика к взрослому не применяются"),
        _tile("не считается", "динамика по неделям", css="void",
              title="накопленный по неделям показатель по взрослому был бы поверхностью "
                    "для сравнения сотрудников"))}
<div class="warning"><strong>Это не оценка работы учителя.</strong>
<p>{_e(disclaimer or 'Камера не отличает «сидит» от «не ведёт урок».')}</p>
<p>Поэтому здесь нет ни индекса, ни динамики, ни сводки за неделю: накопленный по неделям
   показатель по взрослому был бы поверхностью для сравнения сотрудников, построенной на
   измерении, которое такого сравнения не выдерживает.</p></div>
{_cross_room_line(view.class_rooms, here=view.room_key, link=comparable_link)}

<h2>Положение по урокам</h2>
<div class="wrap"><table><thead><tr><th>дата</th><th>видимость</th><th>за столом, %</th>
<th>вне кадра, %</th><th>смен положения</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{_details("Почему «за столом, %» здесь — не то же, что «за своим столом» в карточке урока",
          '<p>«За столом, %» здесь — доля НАБЛЮДЕНИЙ на его собственном месте: что делало '
          "его тело за его столом. Это другой вопрос, чем «где в комнате он был», и другой "
          "знаменатель. Разбор по комнате — в карточке каждого урока, и на камере без "
          "размеченной зоны учительского стола он показывает «за своим столом — не "
          "измерялось» рядом с 96,5 % в этом столбце. Два верных числа о разном; здесь они "
          "не смешиваются.</p>")}
{_caveats_block(view)}
""", section=(back, f"{view.class_display_ru} · {view.room_key}"))


# The adult page needs the connection to read `runs.teacher_json`, which is per-run rather
# than per-place. A one-slot module global rather than threading a connection through six
# signatures for one page; `render()` sets and clears it.
_CONNECTION: list[Any] = [None]


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def render(connection: sqlite3.Connection, out_dir: str | Path, *,
           room_key: str | None = None, class_key: str | None = None) -> dict[str, Any]:
    """Write the whole cabinet as static HTML. Returns what it wrote and for whom."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _CONNECTION[0] = connection

    summary = store.summary(connection)
    written: list[Path] = []

    if not summary["lessons"]:
        index = out / "index.html"
        index.write_text(_empty_page(summary), encoding="utf-8")
        _CONNECTION[0] = None
        return {"index": str(index), "pages": [str(index)], "classes": 0,
                "state": "empty"}

    scopes = [(r["room_key"], r["class_key"]) for r in summary["rooms"]]
    if room_key or class_key:
        scopes = [(r, c) for r, c in scopes
                  if (room_key is None or r == room_key)
                  and (class_key is None or c == class_key)]

    comparable_name = "lessons.html"
    entries = []
    all_lessons: list[LessonView] = []
    lesson_page_by_session: dict[int, str] = {}
    class_page_by_scope: dict[tuple[str, str], str] = {}

    for room, klass in scopes:
        view = weekly.class_view(connection, room_key=room, class_key=klass)
        slug = _slug(f"{room}-{klass}")
        links = {int(h.place["place_id"]): f"place-{slug}-{h.place['ordinal']}.html"
                 for h in view.histories}
        views = lessons_mod.lesson_views(connection, room_key=room, class_key=klass)
        # Two maps into one page, because the two tables key differently: the place table
        # rows are SESSIONS, the lesson list rows are RECORDINGS, and a DVR split makes
        # those different numbers for the same hour.
        # The date is in the filename AFTER the id, and that order is not cosmetic: a place
        # page is `place-<slug>-<ordinal>.html`, so a lesson page ending in `-<number>.html`
        # would be indistinguishable from place number N to anything matching on the tail of
        # a filename — and something does. Ending on the date (or on «без-даты») makes the
        # two families of page unambiguous by construction rather than by luck.
        session_links = {
            v.session_id: (f"lesson-{slug}-{v.session_id}-"
                           f"{_slug(v.date_local or 'без даты')}.html") for v in views}
        recording_links = {int(part["lesson_id"]): session_links[v.session_id]
                           for v in views for part in v.parts}
        lesson_page_by_session.update(session_links)
        class_page_by_scope[(str(room), str(klass))] = _class_page_name(room, klass)
        # Read, never generated. Rendering the cabinet must not make a network call: this
        # command runs on a school machine that may have no internet at all, and a page
        # build that quietly spent money per class would be a surprise nobody asked for.
        # `cabinet note` is the only thing that talks to a provider.
        stored_notes = {run_id: notes.stored_note(connection, run_id)
                        for run_id in {str(row["selected_run_id"]) for row in view.lessons
                                       if row["selected_run_id"]}}
        overview = out / _class_page_name(room, klass)
        overview.write_text(
            _overview(view, links, stored_notes, recording_links, views, comparable_name),
            encoding="utf-8")
        written.append(overview)

        for lesson in views:
            page = out / session_links[lesson.session_id]
            page.write_text(
                _lesson_page(lesson, view, back=overview.name, place_links=links,
                             comparable_link=comparable_name),
                encoding="utf-8")
            written.append(page)

        for history in view.histories:
            page = out / links[int(history.place["place_id"])]
            page.write_text(
                _place_page(history, view, overview.name, session_links,
                            comparable_name, views),
                encoding="utf-8")
            written.append(page)
        entries.append((room, klass, overview.name, view))
        all_lessons.extend(views)

    all_lessons.sort(key=lambda v: (v.started_at is None, v.started_at or ""))
    # `classes_across_rooms` is asked of the WHOLE store even when `render` was scoped to
    # one class, because the question it answers is «делит ли этот ключ ещё какая-то
    # комната» and a scoped query can only ever answer «нет» — which is the one answer that
    # must not be produced by the shape of the query.
    comparable = lessons_mod.ComparableView(
        lessons=all_lessons, classes=lessons_mod.classes_across_rooms(connection))
    comparable_path = out / comparable_name
    comparable_path.write_text(
        _comparable_page(comparable, lesson_page_by_session, class_page_by_scope),
        encoding="utf-8")
    written.insert(0, comparable_path)

    index = out / "index.html"
    index.write_text(_index_page(entries, summary, comparable, comparable_name),
                     encoding="utf-8")
    written.insert(0, index)
    _CONNECTION[0] = None
    return {"index": str(index), "pages": [str(p) for p in written],
            "classes": len(entries), "state": "ok",
            "comparable": str(comparable_path), "lessons": len(all_lessons)}


def _slug(value: str) -> str:
    keep = [c if c.isalnum() else "-" for c in value.lower()]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "class"

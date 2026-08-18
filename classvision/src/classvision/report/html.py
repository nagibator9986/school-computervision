"""One self-contained HTML file per lesson, for a psychologist rather than a developer.

**Self-contained means self-contained.** Every image is inlined as a data URI, the CSS is
inline, there are no fonts, scripts or stylesheets from anywhere. A school's psychologist
opens this from a shared folder on a machine with no internet and it renders identically,
this year and in five years. It is also what makes the file safe to attach to an email
without leaking a request to a CDN that says who read it and when.

**The caveats are copied from `artefact.CAVEATS_RU`, never retyped.** Four hand-written
copies of a caveat is four chances for one of them to quietly stop being true — the
neighbouring codebase paid for that lesson and this file inherits it. The same applies to
the teacher's `not_an_assessment_ru` text, which comes out of the metrics object.

**Layout is an argument about what gets read.** Documents are read from the top, so the
order here is: what this recording is → what could NOT be measured → the numbers → how to
read them. Putting the limitations *above* the table is deliberate and slightly
uncomfortable: a reader who scrolls to the pretty chart and stops should already have
passed the sentence saying a seat is not a child.

**No number appears without its coverage.** Every count in the per-seat table sits in a
row whose first columns are how much of the lesson that place was actually visible for.
There is no view of this report that shows a bare total, because a place seen for 70 % of
the lesson and one seen for 100 % produce counts that are not comparable, and a table that
lets them be compared invites exactly that.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from classvision.report.artefact import CAVEATS_RU

# `charts` is imported inside `render_report`, not here. It pulls in matplotlib, and
# `cabinet/report.py` imports THIS module only for `CSS` — one look for both surfaces
# rather than a second copy of it. A module-level import would mean the cabinet's HTML
# writer, which draws its sparkline as inline SVG and needs no plotting library at all,
# loads ~90 matplotlib modules to obtain a string. A test asserts the cabinet stays clean
# of every model and plotting package, and it caught exactly this.

CSS = """
:root { color-scheme: light dark;
  --bg:#ffffff; --fg:#16181d; --muted:#5b6270; --line:#e3e6ec; --card:#f7f8fa;
  --accent:#2f5bd0; --warn:#8a5a00; --warnbg:#fff6e0; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#14161a; --fg:#e9ecf1; --muted:#9aa3b2; --line:#2a2e37; --card:#1b1e24;
  --accent:#7aa2f7; --warn:#e0b25a; --warnbg:#2a2214; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 5rem; background:var(--bg); color:var(--fg);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; }
h1 { font-size:1.7rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.4rem;
  border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:1.5rem 0 .5rem; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:1rem 1.15rem; margin:1rem 0; }
.warn { background:var(--warnbg); border-color:var(--warn); }
.warn strong { color:var(--warn); }
ul { margin:.4rem 0; padding-left:1.2rem; } li { margin:.3rem 0; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.75rem; }
.kv { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.6rem .8rem; }
.kv .k { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
.kv .v { font-size:1.25rem; font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; }
table { border-collapse:collapse; width:100%; font-size:.9rem; min-width:760px; }
th,td { text-align:right; padding:.45rem .6rem; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; white-space:nowrap; }
th:first-child,td:first-child { text-align:left; white-space:normal; }
thead th { color:var(--muted); font-weight:600; font-size:.76rem; text-transform:uppercase;
  letter-spacing:.03em; border-bottom:2px solid var(--line); }
tbody tr:hover { background:var(--card); }
.cov { color:var(--muted); }
.anon { color:var(--muted); font-style:italic; }
figure { margin:1.25rem 0; }
figure img { width:100%; height:auto; display:block; border:1px solid var(--line);
  border-radius:8px; }
figcaption { color:var(--muted); font-size:.85rem; margin-top:.5rem; }
img.dark-only { display:none; } img.light-only { display:block; }
@media (prefers-color-scheme: dark) {
  img.dark-only { display:block; } img.light-only { display:none; } }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:.8rem; overflow-x:auto; font-size:.8rem; }
.foot { color:var(--muted); font-size:.82rem; margin-top:3rem;
  border-top:1px solid var(--line); padding-top:1rem; }
"""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _figure(pair: dict[str, Path], caption: str) -> str:
    """One chart, in both themes, switched by CSS rather than by script.

    Two copies of the image is a few hundred kilobytes and no JavaScript; the alternative
    is a script that flips a `src`, in a document whose whole point is to still work when
    everything around it has changed.
    """
    light = pair.get("light")
    dark = pair.get("dark")
    parts = []
    if light:
        parts.append(f'<img class="light-only" alt="{_e(caption)}" '
                     f'src="data:image/png;base64,{_b64(light)}">')
    if dark:
        parts.append(f'<img class="dark-only" alt="{_e(caption)}" '
                     f'src="data:image/png;base64,{_b64(dark)}">')
    if not parts:
        return ""
    return f"<figure>{''.join(parts)}<figcaption>{_e(caption)}</figcaption></figure>"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.0f}%".replace(".", ",")


def _prose(text: str) -> str:
    """The summary's plain text as HTML, keeping its section structure.

    `summary.render` writes headings as bare upper-case lines because its primary target
    is a text file that a psychologist can paste into an email. Splitting on blank lines
    alone would glue each heading onto the paragraph beneath it, which is what the first
    version of this report did. Upper-case-only lines become headings here rather than the
    summary emitting HTML, so the text file stays the canonical artefact and this is a
    presentation of it.
    """
    out: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        head = lines[0].strip()
        # An upper-case line with no lower-case letters is a section heading. Cyrillic
        # `.isupper()` behaves as expected; the digit check keeps "15 МИНУТ" style lines
        # from qualifying only if they also carry lower case, which they do not.
        if head and head == head.upper() and any(ch.isalpha() for ch in head):
            out.append(f"<h3>{_e(head)}</h3>")
            lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                out.append(f"<p>{_e(line)}</p>")
    return "".join(out)


def _seat_rows(artefact: dict[str, Any]) -> str:
    rows = []
    for seat in sorted(artefact.get("seats", []), key=lambda s: s["seat_id"]):
        ledger = seat.get("ledger", {})
        counts = ledger.get("counts", {})
        pupil = seat.get("pupil") or {}
        act = (seat.get("metrics", {}) or {}).get("activity", {}) or {}

        if pupil.get("established"):
            who = _e(pupil.get("full_name"))
        else:
            who = (f'<span class="anon">место {seat["seat_id"]} — имя не установлено</span>'
                   f'<br><span class="cov" style="font-size:.8rem">'
                   f'{_e((pupil.get("reason_ru") or "")[:90])}</span>')

        index = act.get("index")
        rows.append(
            "<tr>"
            f"<td>{who}</td>"
            f'<td class="cov">{_pct(100 * (ledger.get("coverage") or 0))}</td>'
            f'<td class="cov">{_e(ledger.get("observations"))}</td>'
            f"<td>{_e(counts.get('hand_raises', 0))}</td>"
            f"<td>{_e(counts.get('stands', 0))}</td>"
            f"<td>{_e(counts.get('away_episodes', 0))}</td>"
            f"<td>{_e(counts.get('head_down_episodes', 0))}</td>"
            f"<td>{_e(counts.get('turned_away_episodes', 0))}</td>"
            f"<td><strong>{'—' if index is None else _e(index)}</strong></td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_report(artefact_path: str | Path, out_dir: str | Path,
                  summary_text: str | None = None) -> Path:
    """Charts + one HTML file. Returns the path to the HTML."""
    from classvision.report import charts

    artefact_path = Path(artefact_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artefact = json.loads(artefact_path.read_text(encoding="utf-8"))

    drawn = charts.render_all(artefact, out_dir)

    provenance = artefact.get("provenance", {})
    lesson = artefact.get("lesson", {})
    uncertainty = artefact.get("uncertainty", {})
    teacher = artefact.get("teacher") or {}
    discovery = (provenance.get("room", {}) or {}).get("seat_discovery", {}) or {}

    window = lesson.get("window_wall") or [None, None]
    started = (window[0] or "")[:19].replace("T", " ")
    ended = (window[1] or "")[:19].replace("T", " ")

    unmeasured = "".join(
        f"<li><strong>{_e(u.get('what'))}</strong> — {_e(u.get('why'))}</li>"
        for u in lesson.get("unmeasured", [])
    )
    notes = "".join(f"<li>{_e(n)}</li>" for n in uncertainty.get("notes", []) if n)
    caveats = "".join(f"<li>{_e(c)}</li>" for c in artefact.get("caveats", CAVEATS_RU))

    teacher_metrics = teacher.get("metrics", {}) or {}
    teacher_html = "<p class='cov'>Взрослый на записи не определён.</p>"
    if teacher_metrics.get("available"):
        teacher_html = _teacher_block(teacher, teacher_metrics)

    sweep = discovery.get("seat_count_by_eps", {})
    sweep_html = " · ".join(f"{k} → {v}" for k, v in sorted(sweep.items()))
    warning = discovery.get("warning")
    warning_html = (f"<div class='card warn'><strong>Внимание.</strong> {_e(warning)}</div>"
                    if warning else "")

    summary_html = ""
    if summary_text:
        summary_html = f"<h2>Сводка</h2><div class='card'>{_prose(summary_text)}</div>"

    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Урок {_e(started)} — наблюдение</title>
<style>{CSS}</style></head><body><main>

<h1>Наблюдение за уроком</h1>
<p class="sub">{_e(started)} — {_e(ended)} · {_e(lesson.get('duration_minutes'))} минут ·
камера {_e(((provenance.get('room', {}) or {}).get('layout', {}) and
           ((provenance.get('room', {}) or {}).get('layout') or {}).get('camera')) or 'camera01')} ·
расчёт <code>{_e(artefact.get('run_id'))}</code></p>

<div class="grid">
  <div class="kv"><div class="k">ученических мест</div><div class="v">{_e(lesson.get('pupil_seats'))}</div></div>
  <div class="kv"><div class="k">проанализировано кадров</div><div class="v">{_e(uncertainty.get('analysed_frames'))}</div></div>
  <div class="kv"><div class="k">наблюдений за людьми</div><div class="v">{_e(uncertainty.get('observations_total'))}</div></div>
  <div class="kv"><div class="k">частота разбора</div><div class="v">{_e(provenance.get('sample_fps'))}/с</div></div>
</div>

<h2>Что измерить не удалось</h2>
<div class="card warn"><ul>{unmeasured}{notes}
<li>Наблюдений, не отнесённых ни к одному месту: {_e(uncertainty.get('observations_unassigned'))}
из {_e(uncertainty.get('observations_total'))}.</li>
<li>Мест, для которых не удалось установить базовую позу: {_e(uncertainty.get('seats_never_settled'))}.</li>
</ul></div>

{summary_html}

<h2>По местам</h2>
<p class="sub">Счётчики нельзя сравнивать между местами, не посмотрев на покрытие:
место, видимое 70 % урока, и место, видимое 100 %, дают несопоставимые числа.</p>
<div class="scroll"><table>
<thead><tr><th>Место / ученик</th><th>Покрытие</th><th>Набл.</th><th>Рука</th><th>Встал</th>
<th>Вне места</th><th>Голова опущена</th><th>Повернулся</th><th>Индекс</th></tr></thead>
<tbody>{_seat_rows(artefact)}</tbody></table></div>

{_figure(drawn.get('timeline', {}), 'Состояния по местам на протяжении урока. Серым — «поза не читалась»: это не спокойный ученик, а отсутствие наблюдения.')}
{_figure(drawn.get('composition', {}), 'Из чего складывался класс в каждую минуту урока.')}
{_figure(drawn.get('activity', {}), 'Индекс наблюдаемой активности, разложенный на составляющие. Итог сам по себе не читается — только вместе с частями.')}

<h2>Динамика внутри урока</h2>
{_within_lesson_block(artefact)}
{_figure(drawn.get('within_lesson', {}), 'Тот же индекс по равным отрезкам урока, с покрытием под каждым столбиком. Отрезки одного места сравнимы только между собой.')}

<h2>Взрослый в классе</h2>
{teacher_html}
{_figure(drawn.get('teacher', {}), 'Положение взрослого на протяжении урока.')}

<h2>Как были найдены места</h2>
<p>Места не заданы вручную — они найдены по самой записи, кластеризацией положений людей.
Найдено мест: <strong>{_e(discovery.get('seats_found'))}</strong>;
детектор видит в кадре {_e(discovery.get('expected_people'))} человек.
Порог выбран автоматически: <code>{_e((discovery.get('thresholds') or {}).get('eps_shoulder_widths'))}</code>
({_e((discovery.get('thresholds') or {}).get('chosen_by'))}).</p>
<p class="cov">Сколько мест нашлось бы при других порогах: {_e(sweep_html)}</p>
{warning_html}

<h2>Как читать эти числа</h2>
<div class="card"><ul>{caveats}</ul></div>

<p class="foot">Модель {_e((provenance.get('model') or {}).get('weights'))},
вход {_e((provenance.get('model') or {}).get('imgsz'))} px ·
время записи прочитано с таймкода камеры ({_e(provenance.get('clock_source'))},
расхождение {_e(provenance.get('clock_drift_seconds'))} с) ·
расчёт выполнен {_e(provenance.get('analysed_at'))}.<br>
Повторный расчёт с другими порогами — это ДРУГОЙ отчёт с другим идентификатором,
а не исправление этого.</p>

</main></body></html>"""

    path = out_dir / f"{artefact_path.stem}.html"
    path.write_text(document, encoding="utf-8")
    return path


def _p_value(value: Any) -> str:
    """A p for a Russian page. Four decimals, and a COMMA.

    `_dec` rounds to one decimal, which turns every p in this table into «0,0» — and a
    column of «0,0» beside the word «ниже» reads as certainty. This is the same defect as
    the English decimal point in `pipeline._seam_notes`, one column further along: a table
    that punctuates its numbers two ways looks machine-generated in the one place whose job
    is to be believed.
    """
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.4f}".replace(".", ",")


_WITHIN_DIRECTION_RU = {
    "lower": "ниже к концу урока",
    "higher": "выше к концу урока",
    "steady": "без направления",
    "unknown": "не установлено",
}

_WITHIN_REFUSAL_RU = {
    "too_few_usable_segments": "слишком мало измеренных отрезков",
    "coverage_change_can_explain_it": "объясняется падением видимости",
    "ordering_not_beyond_chance": "менялось не в одну сторону",
}


def _within_lesson_block(artefact: dict[str, Any]) -> str:
    """The within-lesson table, and the three sentences without which it must not be read.

    **The order is the argument again, and it is the same argument as everywhere else in
    this file: the limits go above the numbers.** Three things have to reach the reader
    before the word «ниже» does, and each of them is a defect this project has already paid
    for once at a different layer:

    1. *A segment index is not the lesson index.* Camera 01 place 3 reads 97,7 for the
       lesson and 88 / 68 / 66 / 67 / 69 by segment, and nothing declined — the event term
       is normalised to a whole lesson. A reader who compares the two columns concludes
       that eight children collapsed at the bell. The artefact carries that sentence
       (`not_comparable_to_ru`) so this page cannot retype it into something weaker.
    2. *How many places would look like this anyway.* `lesson.within_lesson_direction_by_chance`
       — the same duty `metrics/trend.expected_by_chance` performs for the term.
    3. *Which minutes are missing.* A place whose last segment was refused has nothing
       measured about the end of the lesson, which is the part the question is about.

    Every row carries the size of the change, the p it was established at, and the two
    quantities the direction had to clear: what the segmentation itself can produce, and
    what the change in visibility alone could account for. A row that showed a direction
    without those is a verdict.
    """
    seats = [s for s in artefact.get("seats", [])
             if (s.get("metrics") or {}).get("within_lesson")]
    if not seats:
        return ("<p class='cov'>Динамика внутри урока не рассчитывалась: в этом артефакте "
                "нет блока <code>within_lesson</code> ни для одного места.</p>")

    block = (seats[0].get("metrics") or {})["within_lesson"]
    chance = ((artefact.get("lesson") or {}).get("within_lesson_direction_by_chance")
              or {})

    rows = []
    for seat in sorted(seats, key=lambda s: s["seat_id"]):
        within = (seat.get("metrics") or {})["within_lesson"]
        change = within.get("change") or {}
        direction = str(change.get("direction"))
        label = _WITHIN_DIRECTION_RU.get(direction, direction)
        if not change.get("available"):
            because = _WITHIN_REFUSAL_RU.get(str(change.get("refused_because")), "")
            label = f"<span class='anon'>не установлено — {_e(because)}</span>"

        # Every component that got a direction of its own, named. This is the column that
        # carries camera 01 place 7, whose total says «не установлено» while its head-up
        # share falls by 21 index points with p = 0,06.
        moved = [f"{_e((c.get('label_ru') or key).split(' (')[0])}: "
                 f"{_WITHIN_DIRECTION_RU.get(str(c.get('direction')), '')} "
                 f"({_dec(c.get('delta_index_points'))}, p={_p_value(c.get('kendall_p'))})"
                 for key, c in (change.get("components") or {}).items()
                 if str(c.get("direction")) in ("lower", "higher")]
        moved_html = "<br>".join(moved) or "<span class='cov'>ни одна</span>"

        segments = " ".join(
            (f"<span title='отрезок {s['ordinal']}'>"
             f"{_dec(s['activity']['index'])}</span>"
             if (s.get("activity") or {}).get("available")
             else "<span class='anon' title='мало наблюдений'>—</span>")
            for s in within.get("segments", [])
        )
        coverage = " ".join(_pct(100 * float(s.get("coverage") or 0.0))
                            for s in within.get("segments", []))

        rows.append(
            "<tr>"
            f"<td>место {_e(seat['seat_id'])}</td>"
            f"<td>{label}</td>"
            f"<td>{_dec(change.get('delta_index_points'))}</td>"
            f"<td class='cov'>{_p_value(change.get('kendall_p'))}</td>"
            f"<td class='cov'>{_dec(change.get('floor_index_points'))}</td>"
            f"<td class='cov'>{_dec(change.get('visibility_bound_index_points'))}</td>"
            f"<td class='cov'>{_dec(change.get('covered_minutes'))}</td>"
            f"<td style='text-align:left'>{segments}<br>"
            f"<span class='cov' style='font-size:.8rem'>{coverage}</span></td>"
            f"<td style='text-align:left;white-space:normal'>{moved_html}</td>"
            "</tr>"
        )

    return (
        f"<div class='card warn'><strong>Эти числа нельзя сравнивать с индексом за весь "
        f"урок.</strong><br>{_e(block.get('not_comparable_to_ru'))}</div>"
        f"<div class='card'><strong>Сколько мест показало бы направление просто "
        f"случайно.</strong><br>{_e(chance.get('note_ru'))}</div>"
        f"<p class='sub'>Урок разрезан на {_e(block.get('segments_requested'))} равных "
        f"отрезков по {_dec(block.get('segment_minutes'))} мин. Сравнивается только место "
        f"само с собой. «Порог нарезки» — изменение, которое способна дать сама разрезка "
        f"на отрезки; «даёт видимость» — сколько баллов падения могло получиться из одного "
        f"лишь ухудшения видимости к концу урока. Направление называется только тогда, "
        f"когда изменение больше обоих.</p>"
        "<div class='scroll'><table>"
        "<thead><tr><th>Место</th><th>Направление</th><th>Δ, баллов</th><th>p</th>"
        "<th>порог нарезки</th><th>даёт видимость</th><th>измерено, мин</th>"
        "<th>индекс по отрезкам / видно</th><th>какие составляющие сдвинулись</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"
    )


def _dec(value: Any) -> str:
    """A number for a Russian page: comma as the decimal separator, em dash for missing.

    The rest of the report goes through `summary.num`, which does this; the teacher table
    was assembling f-strings directly and printed «1.6 мин» beside «14,5 минуты» two
    paragraphs earlier. A page that punctuates its numbers two ways looks machine-generated
    in the one place it most needs to look checked.
    """
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.1f}".replace(".", ",")


def _teacher_block(teacher: dict, metrics: dict) -> str:
    """The adult's half of the page.

    **Two rules govern the order of everything below, and both were learned from defects
    in this project rather than chosen for taste.**

    *The coverage comes first, above every share.* On camera D14 the follower can attribute
    only 45 % of the lesson to the adult; the state shares are computed against the whole
    lesson so they are arithmetically honest, but a reader who meets «у доски — 3 %» before
    meeting «опознан в 45 % кадров» has already formed a wrong impression and will not
    revise it. So the first thing on the page is how much of him we saw.

    *Every number is rendered with the denominator in its label*, in words, not as a
    suffix on a field name. The artefact separates `state_share_of_lesson_percent` from
    `state_share_of_attributed_percent` because they answer different questions; a page
    that printed one of them as "у доски" would silently pick one for the reader.
    """
    presence = metrics.get("presence") or {}
    ident = teacher.get("identification", {}) or {}
    follower = presence.get("identification", {}) or {}

    if not presence:
        # The pose-only path: an adult with a settled seat and no position taxonomy. This
        # is what every camera before D14 produced, and it stays supported rather than
        # being upgraded to a blank space.
        return (
            f"<div class='grid'>"
            f"<div class='kv'><div class='k'>за своим столом<br>"
            f"<span class='cov'>из наблюдений, где поза прочиталась</span></div>"
            f"<div class='v'>{_pct(metrics.get('at_desk_share_of_observed'))}</div></div>"
            f"<div class='kv'><div class='k'>стоя / вне места<br>"
            f"<span class='cov'>из наблюдений, где поза прочиталась</span></div>"
            f"<div class='v'>{_pct(metrics.get('standing_or_away_share_of_observed'))}</div>"
            f"</div>"
            f"<div class='kv'><div class='k'>вне кадра<br>"
            f"<span class='cov'>из всех проанализированных кадров</span></div>"
            f"<div class='v'>{_pct(metrics.get('out_of_frame_share_of_lesson'))}</div></div>"
            f"<div class='kv'><div class='k'>смен положения</div>"
            f"<div class='v'>{_e(metrics.get('transitions'))}</div></div>"
            f"</div>"
            f"<div class='card warn'><strong>Это не оценка работы учителя.</strong><br>"
            f"{_e(metrics.get('not_an_assessment_ru'))}</div>"
        )

    seen = presence.get("attributed_share_of_lesson_percent")
    of_lesson = presence.get("state_share_of_lesson_percent") or {}
    of_seen = presence.get("state_share_of_attributed_percent") or {}
    minutes = presence.get("state_minutes_of_lesson") or {}
    episodes = presence.get("episode_counts") or {}
    longest = presence.get("longest_episode_minutes_by_state") or {}
    labels = presence.get("definitions_ru") or {}

    coverage = (
        f"<div class='card warn'><strong>Сначала — сколько взрослого мы вообще видели: "
        f"{_pct(seen)} кадров урока.</strong><br>"
        f"Остальное отнесено к «вне кадра». Это значит «не смогли определить», а не "
        f"«отсутствовал» и не «ничего не делал». Все доли ниже посчитаны от всего урока, "
        f"поэтому они складываются в 100 % вместе с «вне кадра» — читать любую из них в "
        f"отрыве от него нельзя.</div>"
    )

    rows = "".join(
        f"<tr><td>{_e((labels.get(state) or {}).get('label', state))}</td>"
        f"<td class='num'>{_pct(of_lesson.get(state))}</td>"
        f"<td class='num'>{_pct(of_seen.get(state)) if state != 'out_of_frame' else '—'}</td>"
        f"<td class='num'>{_dec(minutes.get(state))}</td>"
        f"<td class='num'>{_e(episodes.get(state))}</td>"
        f"<td class='num'>{_dec(longest.get(state))}</td>"
        f"<td class='cov'>{_e((labels.get(state) or {}).get('confusion'))}</td></tr>"
        for state in ("at_board", "at_desk", "among_pupils", "moving", "out_of_frame")
    )
    table = (
        "<table><thead><tr><th>где был</th><th>% урока</th>"
        "<th>% из опознанных кадров</th><th>минут</th><th>эпизодов</th>"
        "<th>самый долгий, мин</th><th>что этот показатель НЕ различает</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )

    board = presence.get("board") or {}
    if board.get("zone_configured"):
        board_html = (
            f"<div class='card'><strong>«У доски» — то, о чём спрашивал заказчик.</strong>"
            f"<br>{_dec(board.get('minutes_of_lesson'))} мин · "
            f"{_pct(board.get('share_of_lesson_percent'))} урока · "
            f"{_e(board.get('episodes'))} эпизодов · самый долгий "
            f"{_dec(board.get('longest_episode_minutes'))} мин.<br>"
            f"<span class='cov'>{_e(board.get('direction_of_error_ru'))}</span></div>"
        )
    else:
        # No zone, no numbers, and the card says the question was not answered rather than
        # answering it with «0,0 мин · 0 % урока · 0 эпизодов». The artefact carries `null`
        # in those four fields for the same reason -- see `metrics/teacher.presence_block`.
        board_html = (
            "<div class='card'><strong>«У доски» — то, о чём спрашивал заказчик — на этой "
            "записи не измерялось.</strong><br>"
            f"<span class='cov'>{_e(board.get('direction_of_error_ru'))}</span></div>"
        )

    floor = presence.get("floor_coverage") or {}
    floor_html = ""
    if floor.get("available"):
        floor_html = (
            f"<div class='card'><strong>Покрытие класса.</strong> Взрослый побывал в "
            f"{_e(floor.get('cells_visited_by_adult'))} клетках размером с человека из "
            f"{_e(floor.get('cells_visited_by_anyone'))}, в которых за урок побывал "
            f"хоть кто-нибудь — {_pct(floor.get('share_of_room_in_use_percent'))}.<br>"
            f"<span class='cov'>{_e(floor.get('what_this_is_not'))}</span></div>"
        )

    route = {"designated_zone": "по зоне учительского стола, размеченной человеком",
             "inferred_by_scale": "по размеру фигуры (предположение программы)",
             "none": "не определён"}.get(str(follower.get("route")), str(follower.get("route")))
    signed = follower.get("zones_confirmed_by") or ""
    who = (
        f"<p class='cov'>Взрослый выделен: {_e(route)}. "
        f"Разметку зон подтвердил: {_e(signed) + '.' if signed else '<strong>никто</strong>.'} "
        f"Способ прослеживания: якорь по зоне стола плюс передачи трека, которым доверяет "
        f"сам детектор ({_e(follower.get('tracklets_kept_after_conflict_resolution'))} "
        f"отрезков из {_e(follower.get('tracklets_total'))}).</p>"
    )
    confirm = ("<p><strong>Требует подтверждения человеком.</strong></p>"
               if ident.get("needs_confirmation") or follower.get("needs_confirmation")
               else "")

    unmeasured = "".join(
        f"<li><strong>{_e(u.get('what'))}</strong> — {_e(u.get('why'))}</li>"
        for u in presence.get("unmeasured_ru", [])
    )

    return (
        coverage
        + f"<div class='grid'>"
          f"<div class='kv'><div class='k'>у доски<br><span class='cov'>% урока</span></div>"
          f"<div class='v'>{_pct(of_lesson.get('at_board'))}</div></div>"
          f"<div class='kv'><div class='k'>за своим столом<br><span class='cov'>% урока"
          f"</span></div><div class='v'>{_pct(of_lesson.get('at_desk'))}</div></div>"
          f"<div class='kv'><div class='k'>среди учеников<br><span class='cov'>% урока"
          f"</span></div><div class='v'>{_pct(of_lesson.get('among_pupils'))}</div></div>"
          f"<div class='kv'><div class='k'>вне кадра<br><span class='cov'>% урока</span>"
          f"</div><div class='v'>{_pct(of_lesson.get('out_of_frame'))}</div></div>"
          f"<div class='kv'><div class='k'>смен положения<br><span class='cov'>между "
          f"эпизодами, без переходов «вне кадра»</span></div>"
          f"<div class='v'>{_e(presence.get('transitions_between_episodes_excluding_out_of_frame'))}"
          f"</div></div>"
          f"</div>"
        + table + board_html + floor_html + who + confirm
        + (f"<p><strong>Что здесь не измеряется:</strong></p><ul>{unmeasured}</ul>"
           if unmeasured else "")
        + f"<div class='card warn'><strong>Это не оценка работы учителя.</strong><br>"
          f"{_e(metrics.get('not_an_assessment_ru'))}</div>"
    )

"""The pictures. Five of them, and each one exists because a table of the same numbers
was tried first and lost something the psychologist needs.

--------------------------------------------------------------------------------
**WHY DRAW ANYTHING AT ALL.**

The artefact already contains every total. A table of totals is nevertheless the wrong
primary surface for this data, for one measured reason: on `clip_15min` the eight pupil
places produce eight rows that are between 97 % and 100 % «сидит на месте», and read as a
table they are indistinguishable from each other and from a room of statues. The
information in this lesson is almost entirely in WHEN things happened — the two
hand-raises at place 3 land inside one minute, the six standing episodes at place 4 are
one sustained excursion at 10:11–10:13, and the adult's twelve raised-hand episodes are
spread across the whole hour. None of that survives summing to a column.

So the charts are not decoration on the numbers, they are the axis the numbers were
integrated over, put back.

--------------------------------------------------------------------------------
**THE RULE THAT SHAPES EVERY CHART HERE: NOT-MEASURED MUST NEVER LOOK LIKE MEASURED-CALM.**

The failure mode of a Gantt strip in this domain is specific and severe. `UNKNOWN` is a
state, absence is not a state at all, and a strip that draws either of them in a soft
neutral tone produces a picture of a quiet, well-behaved pupil out of a pupil we could not
see. On this recording that is not hypothetical: place 4 is out of view for 24 % of the
lesson and the adult for 15 % (`ledger.absent_observations`), so a quarter of one row is
at stake.

Three separate visual channels are therefore reserved for "no measurement", and none of
them is a colour that any measured state also uses:

  * `unknown` — somebody was there and the pose did not resolve — is drawn as a
    cross-hatch on the page background. It reads as "nothing recorded here", because that
    is what it is.
  * `not_observed` — nobody was found at this place in these frames — is drawn as a
    sparse diagonal hatch in the muted foreground colour, with no fill.
  * a GAP, where the timeline covers nothing at all, is left as bare background. Gaps are
    not a rendering bug and they are large: `ledger.timeline()` emits only runs that
    survived their minimum hold and duration, so a pupil whose head dips for ten seconds
    (below the 20 s `head_down_min_seconds`) contributes a hole rather than an episode.
    At place 2 that residue is 24 % of the lesson. The legend names it, and the caption
    points at `discarded_short_runs`, which is where those runs are counted.

--------------------------------------------------------------------------------
**COLOUR IS NEVER THE ONLY CHANNEL.**

Requirement: legible in greyscale and under common colour-vision deficiencies. The
palette is Okabe–Ito (designed for deuteranopia/protanopia), which solves the second half
and not the first — printed grey, several of its entries collide:

| colour | state | greyscale luma (0.299R+0.587G+0.114B) |
|---|---|---|
| `#F0E442` | поднял руку | 213 |
| `#E69F00` | встал | 162 |
| `#56B4E9` | сидит на месте | 158 |
| `#CC79A7` | у доски | 151 |
| `#D55E00` | вне места | 119 |
| `#009E73` | отвернулся | 106 |
| `#0072B2` | голова опущена | 87 |

`встал` (162) and `сидит` (158) are the same grey, and so are `у доски` (151) and
`встал`. Every entry within 20 luma of another therefore also carries a hatch, so the
strip survives a photocopier. The pairing of hatch to state is CHOSEN, with only one
mnemonic honoured: the two "not at their place" states get the two diagonal hatches.

--------------------------------------------------------------------------------
**WHAT THE CHARTS DELIBERATELY DO NOT DO.**

No chart ranks the seats. The activity chart is drawn in seat order, never sorted by
index, because a sorted bar chart of children is a league table however it is captioned,
and the index is explicitly not that kind of number. For the same reason the index bar is
only ever drawn as its four components stacked: `metrics/activity.py` keeps the parts so
that a surface *cannot* show the total alone, and this module honours that by having no
code path that draws a single-segment bar. That rule is what shapes the within-lesson chart
too: the obvious drawing of «динамика внутри урока» is one line per place, and a line of
six index values is precisely the discarded-components case, arriving by a different route.
It is drawn as six stacks instead, and nothing is ever connected across a segment that was
refused for coverage -- see `within_lesson_segments`.

No chart carries a threshold line, a target, or a colour that means "bad". There is no
red-amber-green anywhere in this file. The one saturated warm colour, `#D55E00`, is
assigned to «вне места», which is the most *visible* state, not the worst one.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")   # No display on the school's server, and none needed here.

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

from classvision.metrics.teacher import RU_LABELS as TEACHER_LABELS
from classvision.metrics.teacher import TeacherState
from classvision.states import RU_LABELS, PupilState

# The two non-states the timeline can carry. `NOT_OBSERVED` is written by
# `ledger.SeatLedger.timeline()`; `GAP` is not in the data at all -- it is what is left of
# the lesson after every span has been drawn, and it is given a legend entry so that the
# reader is told the white space means something.
NOT_OBSERVED = "not_observed"
GAP = "__gap__"

RU_EXTRA: dict[str, str] = {
    NOT_OBSERVED: "на месте никого не видно",
    GAP: "эпизод короче порога длительности (отброшен)",
}

# 1400 px wide at 100 dpi -- the width the report is specified to be readable at. Height
# is computed per chart from its row count, never fixed, so eight seats and thirty seats
# both get the same row thickness.
FIG_WIDTH_INCHES = 14.0
DPI = 100

# Height of one seat row, in inches. CHOSEN: at 0.42 in / 42 px a row is thick enough for
# a 3 px hatch to read and thin enough that a full class of 30 still fits one screen.
ROW_INCHES = 0.42

# A span narrower than this is drawn this wide anyway. WHY THIS IS ACCEPTABLE AND WHERE IT
# IS NOT: at 900 s across ~1150 px of axes, one pixel is 0.78 s, so a genuine 1.5 s
# hand-raise (the minimum the ledger will emit -- `hand_min_hold` = 3 observations at
# 2 fps) is two pixels and vanishes next to a hatch. The strip is a chart of WHEN, and
# durations are read from the table; widening the rarest and most informative events is
# the right trade for that chart and would be the wrong one for a chart of totals. CHOSEN
# at 3 s, and stated in the caption rather than hidden here.
MIN_DRAWN_SECONDS = 3.0

# The composition chart's bin. One minute, because that is the unit a teacher recalls a
# lesson in («первые пять минут», «после двадцатой минуты»), and because at 2 fps a
# one-minute bin holds 120 observations per seat, which is enough for a share to mean
# something.
BIN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Theme:
    """One colour scheme. Two exist so the page can follow `prefers-color-scheme`.

    The charts are PNGs, so they cannot adapt in the browser -- both are rendered and the
    page shows one. Twice the bytes, but a white chart on a dark page is the single most
    common way a self-contained report ends up unreadable at night.
    """

    key: str
    background: str      # the figure, matched to the report card behind it
    panel: str           # the plotting area
    text: str
    muted: str
    grid: str

    @property
    def is_dark(self) -> bool:
        return self.key == "dark"


LIGHT = Theme(key="light", background="#ffffff", panel="#f7f7f5", text="#1a1a1a",
              muted="#6b6b6b", grid="#d8d8d4")
DARK = Theme(key="dark", background="#16181c", panel="#1e2126", text="#e8e8e6",
             muted="#9aa0a6", grid="#33383f")
THEMES: tuple[Theme, ...] = (LIGHT, DARK)


# facecolor, hatch. `None` as a facecolor means NO FILL, and that is the load-bearing part
# of this table: the three entries that are not measurements are the three entries with no
# fill, so on any chart in this file they show the page through themselves. See the module
# docstring for the greyscale-luma argument behind every colour/hatch pairing.
STATE_STYLE: dict[str, tuple[str | None, str | None]] = {
    PupilState.SEATED.value:          ("#56B4E9", None),
    PupilState.HEAD_DOWN.value:       ("#0072B2", None),
    PupilState.TURNED_AWAY.value:     ("#009E73", "///"),
    PupilState.HAND_RAISED.value:     ("#F0E442", None),
    PupilState.STOOD_UP.value:        ("#E69F00", "xxx"),
    PupilState.AWAY_FROM_PLACE.value: ("#D55E00", "\\\\\\"),
    PupilState.AT_BOARD.value:        ("#CC79A7", "..."),
    PupilState.UNKNOWN.value:         (None, "xxx"),      # no fill: nothing was measured
    NOT_OBSERVED:                     (None, "///"),      # no fill: nobody was there
    GAP:                              (None, None),       # no fill, no marks: nothing here

    # The adult's POSITION states (`metrics/teacher.TeacherState`). A separate block, and
    # deliberately a different palette from the pupil states above: the two strips are
    # stacked on the same page and a reader who saw the same blue in both would take them
    # for the same measurement. They are not -- one is what a body was doing at its place,
    # the other is where in the room a body was.
    TeacherState.AT_BOARD.value:      ("#CC79A7", "..."),
    TeacherState.AT_DESK.value:       ("#0072B2", None),
    TeacherState.AMONG_PUPILS.value:  ("#009E73", "///"),
    TeacherState.MOVING.value:        ("#E69F00", "xxx"),
    # NO FILL, like every other entry that is not a measurement. `out_of_frame` for the
    # adult is the largest single quantity in this half of the report on camera D14 (55 %
    # of the lesson), and drawing it as a colour would make "we could not see him" look
    # exactly like a place he was.
    TeacherState.OUT_OF_FRAME.value:  (None, "///"),
}


def _style(state: str, theme: Theme) -> dict[str, Any]:
    """Patch keywords for one state. One place, so a legend cannot drift from a chart.

    Filled states get a dark edge (the hatch takes its colour from the edge, and every
    fill in the palette is light enough for a dark hatch in both themes). Unfilled states
    get a muted outline instead, which is what makes an empty band visible as a band
    rather than as a hole the renderer forgot.
    """
    face, hatch = STATE_STYLE.get(state, (theme.muted, None))
    if face is None:
        return {"facecolor": "none", "edgecolor": theme.muted, "hatch": hatch,
                "linewidth": 0.7}
    return {"facecolor": face, "edgecolor": "#1a1a1a", "hatch": hatch, "linewidth": 0.0}


def _label_of(state: str) -> str:
    known = {p.value: p for p in PupilState}
    if state in known:
        return RU_LABELS[known[state]]
    adult = {t.value: t for t in TeacherState}
    if state in adult:
        return TEACHER_LABELS[adult[state]]
    return RU_EXTRA.get(state, state)


def _percent(value: Any) -> str:
    return f"{value * 100:.0f} %" if isinstance(value, (int, float)) else "—"

# Bottom-to-top order for the stacked composition chart. The measured states sit on the
# baseline in descending order of how much of a lesson they usually are, and everything
# that is NOT a measurement is stacked on top, so the reader's eye can find the boundary
# between "what we saw" and "what we did not" as a single horizontal line.
STACK_ORDER: tuple[str, ...] = (
    PupilState.SEATED.value,
    PupilState.HEAD_DOWN.value,
    PupilState.TURNED_AWAY.value,
    PupilState.HAND_RAISED.value,
    PupilState.STOOD_UP.value,
    PupilState.AWAY_FROM_PLACE.value,
    PupilState.AT_BOARD.value,
    PupilState.UNKNOWN.value,
    NOT_OBSERVED,
    GAP,
)

# The activity index's four components, in the order `metrics/activity.py` emits them.
# Luma: 87 / 158 / 106 / 162 -- the last two collide with the second, hence the hatches.
PART_STYLE: dict[str, tuple[str, str | None]] = {
    "head_up_share":      ("#0072B2", None),
    "facing_front_share": ("#56B4E9", None),
    "at_place_share":     ("#009E73", "///"),
    "participation":      ("#E69F00", "xxx"),
}


# ----------------------------------------------------------------------------------
# time
# ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LessonClock:
    """Video seconds -> what a human would have seen on the classroom wall.

    Every x axis in this file is wall clock when the overlay was readable and elapsed time
    when it was not, and the axis LABEL says which. A chart whose x axis silently means
    two different things depending on the recording is how «в 10:11 он встал» becomes
    «на одиннадцатой минуте он встал» in somebody's notes.
    """

    start_seconds: float
    end_seconds: float
    wall_start: datetime | None

    @classmethod
    def of(cls, artefact: dict[str, Any]) -> LessonClock:
        lesson = artefact.get("lesson", {})
        window = lesson.get("window_seconds") or [0.0, 0.0]
        wall = (lesson.get("window_wall") or [None, None])[0]
        parsed: datetime | None = None
        if wall:
            try:
                parsed = datetime.fromisoformat(wall)
            except ValueError:
                parsed = None
        return cls(float(window[0]), float(window[1]), parsed)

    @property
    def duration(self) -> float:
        return max(self.end_seconds - self.start_seconds, 1.0)

    def label(self, video_seconds: float) -> str:
        if self.wall_start is None:
            elapsed = video_seconds - self.start_seconds
            return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        return (self.wall_start
                + timedelta(seconds=video_seconds - self.start_seconds)).strftime("%H:%M")

    @property
    def axis_title(self) -> str:
        return ("время по настенным часам (из наложенной метки на видео)"
                if self.wall_start is not None
                else "время от начала урока, мм:сс (часы на записи не прочитаны)")

    def tick_step(self) -> float:
        """A tick every whole minute, or every 2/5/10 as the lesson gets longer.

        Chosen so a 1400 px chart carries between 8 and 18 labels: fewer and the reader
        counts gridlines, more and the Russian labels overlap.
        """
        for step in (60.0, 120.0, 300.0, 600.0, 900.0):
            if self.duration / step <= 18:
                return step
        return 1800.0


def _apply(theme: Theme) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",       # ships with matplotlib and has Cyrillic
        "font.size": 11,
        "figure.facecolor": theme.background,
        "axes.facecolor": theme.panel,
        "axes.edgecolor": theme.grid,
        "axes.labelcolor": theme.text,
        "text.color": theme.text,
        "xtick.color": theme.muted,
        "ytick.color": theme.text,
        "grid.color": theme.grid,
        "hatch.linewidth": 0.9,
        "savefig.facecolor": theme.background,
    })


def _time_axis(ax, clock: LessonClock, theme: Theme) -> None:
    ax.set_xlim(clock.start_seconds, clock.end_seconds)
    ax.xaxis.set_major_locator(MultipleLocator(clock.tick_step()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: clock.label(v)))
    ax.set_xlabel(clock.axis_title, color=theme.muted, fontsize=10)
    ax.grid(axis="x", color=theme.grid, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)


def _span_patch(ax, x0: float, x1: float, y: float, height: float, state: str,
                theme: Theme) -> None:
    """One span of one state, drawn so that "not measured" cannot be mistaken for calm."""
    ax.add_patch(Rectangle((x0, y), max(x1 - x0, MIN_DRAWN_SECONDS), height,
                           **_style(state, theme)))


def _legend(fig, states: Sequence[str], theme: Theme, *, ncol: int = 5,
            y: float = 0.01) -> None:
    """One legend builder for every chart, fed from `_style`, so the key cannot lie."""
    handles = [Patch(label=_label_of(s), **_style(s, theme)) for s in states]
    for handle in handles:
        if handle.get_facecolor()[3] == 0.0:      # unfilled: give the swatch an outline
            handle.set_linewidth(0.7)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=9.5, handlelength=1.9,
               labelcolor=theme.text)


# ----------------------------------------------------------------------------------
# 1. the per-seat strip
# ----------------------------------------------------------------------------------


def seat_state_timeline(artefact: dict[str, Any], path: Path, theme: Theme) -> Path:
    """One row per place, coloured by state, with the adult below a rule of his own.

    The adult is on the same time axis because half the questions about a lesson are about
    coincidence — «он встал, когда учитель отошёл?» — and two charts cannot answer those.
    He is separated by a gap, a horizontal rule and a row label in a different colour
    because his row is NOT comparable to the others: he is nearest the lens, his shoulder
    width is 2.4x a pupil's, and `MEASUREMENTS.md` §5 showed that the same predicates
    applied to him produce numbers that dominate any pooled statistic. Same axis, separate
    class of object, and the picture has to say so.
    """
    _apply(theme)
    clock = LessonClock.of(artefact)
    seats = [s for s in artefact.get("seats", []) if s.get("role") != "excluded"]
    teacher = artefact.get("teacher")
    has_teacher = bool(teacher and teacher.get("timeline"))

    rows = len(seats) + (1 if has_teacher else 0)
    # Margins are reasoned in INCHES and converted, so the caption and the legend keep the
    # same physical size whether the room has eight seats or thirty.
    top_inches, bottom_inches = 1.15, 1.35
    height = top_inches + ROW_INCHES * max(rows, 1) + bottom_inches
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, height), dpi=DPI)
    fig.subplots_adjust(left=0.19, right=0.985,
                        top=1 - top_inches / height, bottom=bottom_inches / height)

    bar = 0.62                       # fraction of a row occupied by the bar
    labels: list[str] = []
    used: set[str] = set()

    def draw(index: int, timeline: Iterable[dict[str, Any]]) -> None:
        y = index + (1 - bar) / 2
        # The bare background is the GAP: every span is drawn on top of it, so whatever
        # the timeline does not cover stays visible as uncovered.
        for span in timeline:
            state = str(span.get("state"))
            used.add(state)
            _span_patch(ax, float(span["start_s"]), float(span["end_s"]), y, bar,
                        state, theme)

    for index, seat in enumerate(seats):
        draw(index, seat.get("timeline") or [])
        seen = _percent((seat.get("ledger") or {}).get("coverage"))
        labels.append(f"место {seat['seat_id']}   ·   видно {seen}")

    if has_teacher:
        draw(len(seats), teacher.get("timeline") or [])
        # The follower's attribution share, NOT the seat ledger's coverage. They are
        # different quantities and on D14 they are wildly different: the adult's discovered
        # seat is occupied 35 % of the lesson, while the follower places him somewhere in
        # the room for 45 % of it, and the strip above is drawn from the second. Labelling
        # a strip with the other one's number is how a chart and its caption disagree.
        adult_seen = ((teacher.get("metrics") or {}).get("presence") or {}).get(
            "attributed_share_of_lesson_percent")
        seen = f"{adult_seen:.0f} %" if isinstance(adult_seen, (int, float)) else "—"
        where = (f"место {teacher.get('seat_id')}" if teacher.get("seat_id") is not None
                 else "постоянного места нет")
        labels.append(f"ВЗРОСЛЫЙ · {where}   ·   опознан {seen} урока")
        ax.axhline(len(seats), color=theme.text, linewidth=1.2, alpha=0.65)

    ax.set_ylim(rows, 0)             # first seat at the top, reading order
    ax.set_yticks([i + 0.5 for i in range(rows)])
    ax.set_yticklabels(labels, fontsize=10)
    if has_teacher:
        ax.get_yticklabels()[-1].set_color(theme.muted)
        ax.get_yticklabels()[-1].set_fontweight("bold")
    _time_axis(ax, clock, theme)

    fig.suptitle("Что происходило на каждом месте, минута за минутой",
                 x=0.008, y=1 - 0.34 / height, ha="left", fontsize=15, color=theme.text)
    fig.text(0.008, 1 - 0.60 / height,
             "Пустое место в строке — не ошибка: там эпизод оказался короче своего порога "
             "длительности и был отброшен\n"
             "(см. «отброшенные короткие эпизоды» в таблице ниже). Эпизоды короче "
             f"{MIN_DRAWN_SECONDS:.0f} с нарисованы шириной {MIN_DRAWN_SECONDS:.0f} с, "
             "иначе их не видно: длительности читайте в таблице, а не по ширине полосы.",
             ha="left", va="top", fontsize=9.5, color=theme.muted, linespacing=1.5)

    order = [s for s in STACK_ORDER if s in used or s == GAP]
    _legend(fig, order, theme, ncol=5, y=0.16 / height)
    fig.savefig(path, dpi=DPI, facecolor=theme.background)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
# 2. the class as a whole, over time
# ----------------------------------------------------------------------------------


def _overlap(span: dict[str, Any], start: float, stop: float) -> float:
    return max(0.0, min(float(span["end_s"]), stop) - max(float(span["start_s"]), start))


def class_composition(artefact: dict[str, Any], path: Path, theme: Theme, *,
                      bin_seconds: float = BIN_SECONDS) -> Path:
    """The share of PUPIL-place-seconds in each state, per one-minute bin.

    The denominator is deliberately (number of pupil places x bin length) rather than
    (seconds we actually observed), so the "not measured" band is inside the same 100 %
    as everything else. Normalising by what we saw would make a minute in which half the
    class was hidden look exactly like a minute in which everyone was visible and calm —
    the same error the ledger's three separate counters exist to prevent, committed at
    the level of a picture.

    The adult is excluded here and only here. He is one person against eight and his
    states are produced by predicates calibrated on pupils; mixing him into a class share
    would move it by 1/9 for reasons that are about the camera.
    """
    _apply(theme)
    clock = LessonClock.of(artefact)
    seats = [s for s in artefact.get("seats", []) if s.get("role") == "pupil"]

    edges: list[float] = []
    t = clock.start_seconds
    while t < clock.end_seconds:
        edges.append(t)
        t += bin_seconds

    shares: dict[str, list[float]] = {s: [] for s in STACK_ORDER}
    for left in edges:
        right = min(left + bin_seconds, clock.end_seconds)
        capacity = max((right - left) * max(len(seats), 1), 1e-6)
        totals = {s: 0.0 for s in STACK_ORDER}
        for seat in seats:
            for span in seat.get("timeline") or []:
                state = str(span.get("state"))
                if state in totals:
                    totals[state] += _overlap(span, left, right)
        covered = sum(totals.values())
        totals[GAP] = max(capacity - covered, 0.0)
        for state in STACK_ORDER:
            shares[state].append(100.0 * totals[state] / capacity)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, 5.4), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.26)

    centres = [e + min(bin_seconds, clock.end_seconds - e) / 2 for e in edges]
    widths = [min(bin_seconds, clock.end_seconds - e) * 0.92 for e in edges]
    bottom = [0.0] * len(edges)
    used: list[str] = []
    for state in STACK_ORDER:
        values = shares[state]
        if max(values, default=0.0) <= 0.0:
            continue
        used.append(state)
        face, hatch = STATE_STYLE.get(state, (theme.muted, None))
        ax.bar(centres, values, width=widths, bottom=bottom,
               color=face if face else "none",
               edgecolor=theme.muted if face is None else "none",
               hatch=hatch, linewidth=0.0)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylim(0, 100)
    ax.set_ylabel("доля мест, %", color=theme.text, fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.grid(axis="y", color=theme.grid, linewidth=0.6, alpha=0.7)
    _time_axis(ax, clock, theme)

    fig.suptitle("Класс целиком: из чего складывалась каждая минута урока",
                 x=0.008, y=0.975, ha="left", fontsize=15, color=theme.text)
    fig.text(0.008, 0.905,
             f"100 % — это {len(seats)} ученических мест × 1 минута. "
             "Верхние полосы — то, что измерить не удалось; они внутри тех же 100 %, "
             "а не убраны из знаменателя.",
             ha="left", va="top", fontsize=9.5, color=theme.muted)
    _legend(fig, used, theme, ncol=5, y=0.005)
    fig.savefig(path, dpi=DPI, facecolor=theme.background)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
# 3. the activity index, which is never drawn as one number
# ----------------------------------------------------------------------------------


def activity_components(artefact: dict[str, Any], path: Path, theme: Theme) -> Path:
    """The index as a stack of its four parts. There is no code path that draws it whole.

    Horizontal bars, in seat order, never sorted. Sorting would turn eight children into a
    ranking, and the caption under a ranking is never read. The bar's segments are the
    четыре components with their own weights, and the unfilled remainder to 100 is drawn
    as an outline so the reader can see the ceiling the index is measured against instead
    of guessing whether 62 is high.

    A place whose coverage fell below `metrics.activity.MIN_COVERAGE` gets no bar and a
    printed reason, because the alternative -- a short bar -- is a low score for a pupil
    we could not see.
    """
    _apply(theme)
    seats = [s for s in artefact.get("seats", []) if s.get("role") == "pupil"]
    rows = max(len(seats), 1)
    height = 1.9 + 0.52 * rows + 0.5
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, height), dpi=DPI)
    fig.subplots_adjust(left=0.155, right=0.90, top=0.85, bottom=0.24)

    labels: list[str] = []
    used: list[str] = []
    for index, seat in enumerate(seats):
        activity = (seat.get("metrics") or {}).get("activity") or {}
        coverage = activity.get("coverage")
        seen = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "—"
        labels.append(f"место {seat['seat_id']}   ·   видно {seen}")

        # The ceiling. Drawn first and behind, so a short bar reads as "little of 100"
        # rather than as a complete object.
        ax.add_patch(Rectangle((0, index - 0.3), 100, 0.6, facecolor="none",
                               edgecolor=theme.grid, linewidth=1.0, linestyle=(0, (3, 3))))

        if not activity.get("available"):
            ax.text(1.5, index, f"индекс не рассчитан — {activity.get('reason', 'нет данных')}",
                    va="center", ha="left", fontsize=10, color=theme.muted, style="italic")
            continue

        left = 0.0
        for part in activity.get("parts", []):
            key = str(part.get("key"))
            value = float(part.get("contribution") or 0.0)
            face, hatch = PART_STYLE.get(key, (theme.muted, None))
            if key not in used:
                used.append(key)
            if value <= 0:
                continue
            ax.add_patch(Rectangle((left, index - 0.3), value, 0.6, facecolor=face,
                                   edgecolor="#00000055", hatch=hatch, linewidth=0.0))
            left += value
        ax.text(101.5, index, f"{activity.get('index')}", va="center", ha="left",
                fontsize=11, color=theme.text, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("вклад в индекс, баллов из 100", color=theme.muted, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", color=theme.grid, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)

    fig.suptitle("Индекс наблюдаемой активности — только вместе с составляющими",
                 x=0.008, y=0.975, ha="left", fontsize=15, color=theme.text)
    fig.text(0.008, 0.905,
             "Это не «вовлечённость». Это взвешенная сумма четырёх наблюдаемых признаков; "
             "веса выбраны разработчиком, а не получены из размеченных уроков. "
             "Пунктирная рамка — потолок в 100 баллов. Места идут по номерам и не "
             "сортируются по величине индекса.",
             ha="left", va="top", fontsize=9.5, color=theme.muted)

    handles: list[Patch] = []
    weights = {}
    for seat in seats:
        for part in ((seat.get("metrics") or {}).get("activity") or {}).get("parts", []):
            weights[str(part.get("key"))] = (part.get("label_ru"), part.get("weight"))
    for key in PART_STYLE:
        if key not in used:
            continue
        label, weight = weights.get(key, (key, None))
        face, hatch = PART_STYLE[key]
        handles.append(Patch(facecolor=face, edgecolor=theme.muted, hatch=hatch,
                             linewidth=0.6,
                             label=f"{label}  (вес {weight})" if weight else str(label)))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=9.5, labelcolor=theme.text)
    fig.savefig(path, dpi=DPI, facecolor=theme.background)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
# 4. the same index, segment by segment, one small panel per place
# ----------------------------------------------------------------------------------


# The share of a cell's height given to the coverage strip under each seat's bars. CHOSEN
# at 1:3.4 against the index panel: large enough that a coverage of 0.2 and one of 0.9 are
# obviously different heights, small enough that the eye reads the row as "an index with a
# qualifier" rather than as two charts of equal standing. They are NOT of equal standing —
# the coverage strip is the sentence that says how much of the bar above it to believe.
COVERAGE_STRIP_RATIO = (3.4, 1.0)


def within_lesson_segments(artefact: dict[str, Any], path: Path, theme: Theme) -> Path:
    """One small panel per place: the index per segment, stacked, over its own coverage.

    **This chart exists because a line would have lied twice.** The obvious rendering of
    «динамика внутри урока» is one line per seat, and it fails in the two ways this package
    spends most of its effort on:

    * *It would draw the total as a total.* `metrics/activity.py` keeps `parts` so that no
      surface can show the index alone, and `activity_components` above states there is no
      code path here that draws a single-segment bar. A line of six index values is exactly
      that code path with the components thrown away — and on camera 01 place 7 the
      components are the whole story: the index looks like noise (96, 61, 96, 50, 53, 55)
      while its head-up component falls monotonically from 27.0 to 5.6 points. So each
      segment is a STACK, and the reader can see which layer moved.
    * *It would interpolate across a refusal.* Two places on camera D14 lose their last
      segment to occlusion (place 6 is seen in 14 % of the final ten minutes). A line drawn
      through that gap invents the most interesting minutes of the lesson; a line that stops
      leaves a reader wondering whether the child left. Refused segments are therefore drawn
      as a full-height unfilled column carrying the word «не измерено», and nothing is ever
      connected across them.

    The coverage strip sits directly beneath the bars, on its own axis, with the
    `metrics.activity.MIN_COVERAGE` line marked, so «индекс 58» and «видно 56 % кадров»
    cannot be read one without the other. Places are in seat order and never sorted.
    """
    from classvision.metrics.activity import MIN_COVERAGE

    _apply(theme)
    seats = [s for s in artefact.get("seats", []) if s.get("role") == "pupil"
             and ((s.get("metrics") or {}).get("within_lesson"))]
    if not seats:
        # Nothing to draw is a real outcome (a lesson of excluded places, or an artefact
        # from before this metric existed). An empty figure with a sentence beats a
        # traceback in the report writer and beats a blank image in the page.
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, 1.6), dpi=DPI)
        ax.axis("off")
        ax.text(0.0, 0.5, "Динамика внутри урока не рассчитана ни для одного места.",
                fontsize=12, color=theme.muted, va="center")
        fig.savefig(path, dpi=DPI, facecolor=theme.background)
        plt.close(fig)
        return path

    columns = 2 if len(seats) > 3 else 1
    rows = math.ceil(len(seats) / columns)
    cell_inches = 2.9
    top_inches, bottom_inches = 1.95, 0.95
    height = top_inches + cell_inches * rows + bottom_inches
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, height), dpi=DPI)
    outer = fig.add_gridspec(rows, columns, hspace=0.62, wspace=0.20,
                             left=0.06, right=0.985,
                             top=1 - top_inches / height, bottom=bottom_inches / height)

    used_parts: list[str] = []
    for position, seat in enumerate(seats):
        block = (seat.get("metrics") or {}).get("within_lesson") or {}
        segments = block.get("segments") or []
        change = block.get("change") or {}
        row, column = divmod(position, columns)
        # The two axes of one place touch (hspace 0) inside their own cell, so the coverage
        # strip reads as the footing of the bars above it rather than as a separate chart
        # that happens to be nearby, while whole cells stay well apart.
        cell = outer[row, column].subgridspec(2, 1, hspace=0.0,
                                              height_ratios=list(COVERAGE_STRIP_RATIO))
        bars = fig.add_subplot(cell[0])
        strip = fig.add_subplot(cell[1])

        widths = [max(float(s["end_minute"]) - float(s["start_minute"]), 0.1)
                  for s in segments]
        centres = [(float(s["start_minute"]) + float(s["end_minute"])) / 2
                   for s in segments]
        span_end = max((float(s["end_minute"]) for s in segments), default=1.0)

        for segment, centre, width in zip(segments, centres, widths, strict=True):
            activity = segment.get("activity") or {}
            bars.add_patch(Rectangle((centre - width * 0.46, 0), width * 0.92, 100,
                                     facecolor="none", edgecolor=theme.grid,
                                     linewidth=0.9, linestyle=(0, (3, 3))))
            if not activity.get("available"):
                bars.add_patch(Rectangle((centre - width * 0.46, 0), width * 0.92, 100,
                                         facecolor="none", edgecolor=theme.muted,
                                         hatch="xxx", linewidth=0.8))
                bars.text(centre, 50, "не\nизмерено", ha="center", va="center",
                          fontsize=8, color=theme.text, linespacing=1.2,
                          bbox={"facecolor": theme.panel, "edgecolor": "none",
                                "pad": 1.5})
                continue
            bottom = 0.0
            for part in activity.get("parts", []):
                key = str(part.get("key"))
                value = float(part.get("contribution") or 0.0)
                if key not in used_parts:
                    used_parts.append(key)
                if value <= 0:
                    continue
                face, hatch = PART_STYLE.get(key, (theme.muted, None))
                bars.add_patch(Rectangle((centre - width * 0.46, bottom), width * 0.92,
                                         value, facecolor=face, edgecolor="#00000055",
                                         hatch=hatch, linewidth=0.0))
                bottom += value
            bars.text(centre, bottom + 3, f"{activity.get('index')}", ha="center",
                      va="bottom", fontsize=8.5, color=theme.text)

        bars.set_xlim(0, span_end)
        bars.set_ylim(0, 118)
        bars.set_xticks([])
        bars.yaxis.set_major_locator(MultipleLocator(50))
        bars.set_ylabel("баллов", fontsize=8.5, color=theme.muted)
        bars.tick_params(labelsize=8)
        for side in ("top", "right"):
            bars.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            bars.spines[side].set_color(theme.grid)

        coverage = (seat.get("ledger") or {}).get("coverage")
        seen = _percent(coverage)
        index = ((seat.get("metrics") or {}).get("activity") or {}).get("index")
        # Wrapped, not shrunk. At two columns a cell is ~6.6 inches and the verdict line
        # runs to about 95 characters at its longest ("...менялось не в одну сторону
        # (наклон +18,3 балла); опирается на 48 мин урока"), which ran off the right edge of
        # the figure and cut the clause naming how much of the lesson was measured -- the
        # clause a reader most needs and the easiest to lose. Same failure and same fix as
        # the teacher caption above.
        verdict = "\n".join(textwrap.wrap(_change_line(change), width=62)) or "—"
        bars.set_title(
            f"место {seat['seat_id']}   ·   за весь урок индекс {index}, видно {seen}\n"
            f"{verdict}",
            fontsize=9.5, color=theme.text, loc="left", pad=6, linespacing=1.45)

        for segment, centre, width in zip(segments, centres, widths, strict=True):
            value = 100.0 * float(segment.get("coverage") or 0.0)
            strip.add_patch(Rectangle((centre - width * 0.46, 0), width * 0.92, value,
                                      facecolor=theme.muted, edgecolor="none", alpha=0.55))
        strip.axhline(100 * MIN_COVERAGE, color=theme.text, linewidth=0.8,
                      linestyle=(0, (2, 2)), alpha=0.7)
        strip.set_xlim(0, span_end)
        strip.set_ylim(0, 100)
        strip.set_yticks([100 * MIN_COVERAGE])
        strip.set_yticklabels([f"{100 * MIN_COVERAGE:.0f}"], fontsize=7.5)
        strip.set_ylabel("видно, %", fontsize=8.5, color=theme.muted)
        strip.xaxis.set_major_locator(MultipleLocator(10))
        strip.tick_params(labelsize=8)
        strip.set_xlabel("минута урока", fontsize=8.5, color=theme.muted)
        for side in ("top", "right"):
            strip.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            strip.spines[side].set_color(theme.grid)

    fig.suptitle("Динамика внутри урока: тот же индекс по отрезкам урока",
                 x=0.008, y=1 - 0.34 / height, ha="left", fontsize=15, color=theme.text)
    block = (seats[0].get("metrics") or {}).get("within_lesson") or {}
    fig.text(
        0.008, 1 - 0.60 / height,
        f"Урок разрезан на {block.get('segments_requested')} равных отрезков по "
        f"{str(block.get('segment_minutes')).replace('.', ',')} мин. Столбики одного места "
        "сравнимы ТОЛЬКО между собой: с индексом за весь урок их сравнивать нельзя —\n"
        "событийная составляющая нормирована на урок, а не на отрезок. Серая полоса под "
        "столбиками — доля кадров, в которых на этом месте кого-то видно;\n"
        "пунктир на ней — порог, ниже которого индекс не считается вовсе. Места идут по "
        "номерам и не сортируются.",
        ha="left", va="top", fontsize=9.5, color=theme.muted, linespacing=1.5)

    handles = [Patch(facecolor=PART_STYLE[key][0], edgecolor=theme.muted,
                     hatch=PART_STYLE[key][1], linewidth=0.6, label=label)
               for key, label in _part_labels(seats).items() if key in used_parts]
    handles.append(Patch(facecolor="none", edgecolor=theme.muted, hatch="xxx",
                         linewidth=0.7, label="отрезок не измерен (мало наблюдений)"))
    handles.append(Patch(facecolor=theme.muted, alpha=0.55, edgecolor="none",
                         label="доля кадров, где место видно"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.004), fontsize=9.5, labelcolor=theme.text)
    fig.savefig(path, dpi=DPI, facecolor=theme.background)
    plt.close(fig)
    return path


def _part_labels(seats: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Component key -> its Russian label, taken from the artefact rather than retyped."""
    labels: dict[str, str] = {}
    for seat in seats:
        for part in ((seat.get("metrics") or {}).get("activity") or {}).get("parts", []):
            labels.setdefault(str(part.get("key")), str(part.get("label_ru")))
    return {key: labels.get(key, key) for key in PART_STYLE}


def _change_line(change: dict[str, Any]) -> str:
    """The verdict for one place, in one line, with its uncertainty attached.

    Never just a direction: the direction word is always followed by the size, and by the
    number of minutes the statement rests on when that is less than the lesson. A panel that
    said «ниже» and nothing else would be the ranking this package refuses to produce,
    printed one place at a time.
    """
    direction = str(change.get("direction"))
    delta = change.get("delta_index_points")
    covered = change.get("covered_minutes")
    refused = change.get("refused_segments") or 0
    if not change.get("available") and direction == "unknown":
        because = str(change.get("refused_because") or "")
        head = {
            "too_few_usable_segments": "направление не строится: слишком мало измеренных отрезков",
            "coverage_change_can_explain_it": "направление не строится: это объясняется падением видимости",
            "ordering_not_beyond_chance": "направление не установлено: менялось не в одну сторону",
        }.get(because, "направление не установлено")
        if isinstance(delta, (int, float)):
            head += f" (наклон {delta:+.1f} балла)".replace(".", ",")
    elif direction == "steady":
        head = f"без направления: {delta:+.1f} балла — в пределах разрешения показателя"
        head = head.replace(".", ",")
    else:
        word = "ниже к концу урока" if direction == "lower" else "выше к концу урока"
        head = f"{word}: {delta:+.1f} балла, p = {change.get('kendall_p')}"
        head = head.replace(".", ",")
    if refused and isinstance(covered, (int, float)):
        head += f"; опирается на {covered:.0f} мин урока".replace(".", ",")
    return head


# ----------------------------------------------------------------------------------
# 5. the adult
# ----------------------------------------------------------------------------------


# Which states, for the adult, are "at the desk" as opposed to "up". The mapping is the
# same one `metrics/teacher.py` uses, restated here as a rendering order rather than
# re-derived: one definition of sitting in this codebase, not two that drift.
TEACHER_SEATED = (PupilState.SEATED.value, PupilState.HEAD_DOWN.value,
                  PupilState.TURNED_AWAY.value, PupilState.HAND_RAISED.value)
TEACHER_UP = (PupilState.STOOD_UP.value, PupilState.AWAY_FROM_PLACE.value,
              PupilState.AT_BOARD.value)


def teacher_strip(artefact: dict[str, Any], path: Path, theme: Theme) -> Path:
    """Where the adult was in the ROOM over the lesson, and the same thing as shares.

    **The two summary bars are two bars and not one stack, and that is the whole reason
    this function is longer than it looks.** They are the same five states counted against
    two different denominators — every analysed frame of the lesson, and only the frames
    the follower could attribute to the adult — and the second is smaller than the first by
    exactly however much `out_of_frame` is. Merging them into a single 100 % bar would be
    arithmetic nonsense that reads as a clean summary, so each bar carries its denominator
    in its own label.

    On camera D14 `вне кадра` is 55 % of the lesson, and the layout is built around not
    letting that be skipped: it is the largest band on the strip, the largest segment of the
    upper bar, and the only segment with no fill. **A reader who takes the lower bar alone
    will overstate every state**, which is precisely why it is never drawn alone.

    The caption carries, compressed, the sentence that must survive a screenshot: this is
    position, not an assessment. The artefact holds the full text
    (`metrics.not_an_assessment_ru`) and the HTML prints it whole.
    """
    _apply(theme)
    clock = LessonClock.of(artefact)
    teacher = artefact.get("teacher") or {}
    timeline = teacher.get("timeline") or []
    metrics = teacher.get("metrics") or {}

    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, 5.6), dpi=DPI)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.95,
                            left=0.155, right=0.985, top=0.80, bottom=0.20)
    strip = fig.add_subplot(grid[0])
    bars = fig.add_subplot(grid[1])

    used: set[str] = set()
    for span in timeline:
        state = str(span.get("state"))
        used.add(state)
        _span_patch(strip, float(span["start_s"]), float(span["end_s"]), 0.2, 0.6,
                    state, theme)
    strip.set_ylim(0, 1)
    strip.set_yticks([0.5])
    strip.set_yticklabels(["взрослый (учитель)"], fontsize=10)
    _time_axis(strip, clock, theme)

    presence = metrics.get("presence") or {}
    of_lesson = presence.get("state_share_of_lesson_percent") or {}
    of_seen = presence.get("state_share_of_attributed_percent") or {}
    # Bottom to top in the same order as the legend, with the unmeasured state last, so the
    # boundary between "where he was" and "we could not say" is a single vertical line.
    order = [TeacherState.AT_BOARD.value, TeacherState.AT_DESK.value,
             TeacherState.AMONG_PUPILS.value, TeacherState.MOVING.value,
             TeacherState.OUT_OF_FRAME.value]

    def stacked(y: int, shares: dict[str, Any], title: str) -> None:
        left = 0.0
        for state in order:
            value = shares.get(state)
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            style = _style(state, theme)
            bars.add_patch(Rectangle((left, y - 0.28), value, 0.56, **style))
            if value >= 8:
                filled = style["facecolor"] != "none"
                bars.text(left + value / 2, y, f"{_label_of(state)} {value:.0f} %",
                          va="center", ha="center", fontsize=9.0,
                          color="#111111" if filled else theme.text)
            left += value
        bars.text(-1.5, y, title, va="center", ha="right", fontsize=10, color=theme.text)

    stacked(0, of_lesson, "из всего урока\n(включая «вне кадра»)")
    stacked(1, of_seen, "только из кадров,\nгде взрослый опознан")
    bars.set_xlim(0, 100)
    bars.set_ylim(1.7, -0.7)
    bars.set_yticks([])
    bars.set_xlabel("%", color=theme.muted, fontsize=10)
    bars.xaxis.set_major_locator(MultipleLocator(10))
    bars.grid(axis="x", color=theme.grid, linewidth=0.6, alpha=0.7)
    bars.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.spines["bottom"].set_color(theme.grid)

    fig.suptitle("Взрослый: где он был в классе в течение урока",
                 x=0.008, y=0.975, ha="left", fontsize=15, color=theme.text)
    seen = presence.get("attributed_share_of_lesson_percent")
    # Two short lines rather than one long one: at 14 inches the single-line version ran
    # off the right edge of the figure and the clause that got cut was the one about
    # «вне кадра» -- the sentence a reader most needs and the easiest to lose.
    caption = ("Это описание положения, а не оценка работы учителя: «сидит» не означает "
               "«не ведёт урок».")
    if isinstance(seen, (int, float)):
        caption += (f"\nВзрослый опознан в {seen:.0f} % кадров урока; остальное — «вне "
                    "кадра»: «не смогли определить», а не «отсутствовал».")
        # The board clause is a claim about a measurement, so it is printed only where the
        # measurement exists. It used to be unconditional, and on a camera whose layout has
        # `board_zone: null` -- which is what `configs/camera_01.yaml` has -- the strip
        # showed no `у доски` band at all and then told the reader underneath that the time
        # at the board was understated. Nothing was understated: nothing was measured.
        if (presence.get("board") or {}).get("zone_configured"):
            caption += " Время у доски здесь скорее занижено."
        else:
            caption += (" Зона доски для этой камеры не размечена, поэтому «у доски» "
                        "не измерялось вовсе.")
    fig.text(0.008, 0.925, caption, ha="left", va="top", fontsize=9.5,
             color=theme.muted, linespacing=1.5)
    _legend(fig, [s for s in order if s in used or s in of_lesson], theme, ncol=5,
            y=0.005)
    fig.savefig(path, dpi=DPI, facecolor=theme.background)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------


CHARTS = {
    "timeline": (seat_state_timeline,
                 "Что происходило на каждом месте, минута за минутой"),
    "composition": (class_composition,
                    "Класс целиком: из чего складывалась каждая минута"),
    "activity": (activity_components,
                 "Индекс наблюдаемой активности и его составляющие"),
    "within_lesson": (within_lesson_segments,
                      "Динамика внутри урока: индекс по отрезкам, с покрытием"),
    "teacher": (teacher_strip, "Взрослый: положение в течение урока"),
}


def render_all(artefact: dict[str, Any], out_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Every chart, in every theme. Returns {chart key: {theme key: png path}}.

    Both themes are always rendered rather than on demand, because the consumer is a
    single self-contained HTML file that has to work offline in whichever mode the reader's
    machine is in, and it cannot go back and ask for the other one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, dict[str, Path]] = {}
    for key, (function, _title) in CHARTS.items():
        made[key] = {}
        for theme in THEMES:
            path = out_dir / f"{key}_{theme.key}.png"
            made[key][theme.key] = function(artefact, path, theme)
    return made

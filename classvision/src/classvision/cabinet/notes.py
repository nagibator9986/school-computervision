"""The orientation note inside the cabinet: generated once per run, stored, shown, or refused.

`report/orientation.py` decides WHAT a note may say and enforces it. This module decides
what the cabinet sends, where the answer is kept, and when a kept answer stops being valid.
There is exactly one LLM path in this package and it is that one; nothing here calls a
provider directly.

--------------------------------------------------------------------------------------
**WHAT LEAVES THIS MACHINE, EXACTLY. A SCHOOL WILL ASK, AND «ДАННЫЕ ДЕТЕЙ УХОДЯТ ЗА
ГРАНИЦУ» IS A REAL ЗРК 94-V QUESTION EVEN FOR COUNTS.**

One HTTPS request per run, to the configured provider, containing two strings:

  * the system prompt — `orientation.SYSTEM_RU`, a fixed 7 257 bytes of rules in Russian,
    byte-identical for every lesson and every school, containing nothing about anybody;
  * the bundle — `orientation.compact_for_note(bundle_for_run(...))` serialised as JSON.
    **Measured, not estimated, on the two artefacts in `out/` (UTF-8, exactly as sent):
    2 286 bytes for the 5-place D14 session and 3 261 bytes for the 8-place camera-01
    lesson** — the two differ by three places and by 975 bytes, so a place costs about a
    third of a kilobyte and the fixed head is small. That is the whole payload; the
    artefact itself is 240–350 KB and the summary bundle of `report/summary.py` is 32 KB,
    and neither goes anywhere near this call.

The bundle contains, per place: the cabinet's own place number, how much of the lesson that
place was visible (a percentage), whether that share is low enough to require a spoken
caveat, the activity index, six episode counts, and the list of which of those counts are
zero only because no episode ran long enough. Plus the lesson's length in minutes, the
number of places, and the artefact's own list of what could not be measured. That is the
complete list of fields — `bundle_for_run` is a whitelist and the assertion is checkable by
reading it.

**Three things a counter can be, and the bundle distinguishes all three**, because a zero
read by something fluent becomes a universal negative and a universal negative about a child
is the one output this feature must not produce:

  * `null` — this camera cannot witness the event at all. Two separate causes, both handled:
    the artefact says so in its own `unmeasured` (camera 01's board is behind the lens), or
    the artefact CONTRADICTS ITS OWN ZERO and `lessons.board_conflict_for_run` catches it
    (D14 counts no pupil at the board while reporting somebody at that board for over half
    the lesson).
  * `0` with the counter named in `zero_but_briefly_seen` — the state was observed, and no
    run of it lasted long enough to become an episode. «Не было» and «было, но коротко» are
    different facts and used to arrive as the same `0`.
  * `0` on its own — nothing was seen at all, which is the only one of the three a reader may
    hear as «этого не было».

**What is NOT sent, and cannot be, because it is never put in the bundle:**

  * **No imagery of any kind.** No frame, no crop, no face, no thumbnail, no video. The
    analysis stage is entirely local (`vision/pose.py`, YOLO11m-pose on this machine) and
    nothing downstream of it has a picture to send.
  * **No names, no `external_id`, no roster row.** `pupil_name` is `None` in every seat of
    every bundle this module builds, by construction and not by accident — see
    `bundle_for_run`. A signed seating plan changes what the CABINET PAGE prints; it does
    not change what is sent. Consequently the note can only ever say «место 3», which is
    what the page says anyway.
  * **No timestamps, no dates, no room key, no class key, no school, no file name, no
    `run_id`.** The bundle carries a duration in minutes and nothing that places it on a
    calendar or in a building. Two lessons of the same class produce two requests that
    cannot be linked to each other, or to a room, by their content.
  * **No geometry.** No seat coordinates, no shoulder widths, no zone polygons.
  * **No per-observation timeline.** Counts and shares only.

**Is that personal data?** The honest answer is: it is de-identified behavioural data about
identifiable individuals held elsewhere. Nobody at the provider can turn «место 3, видно
62 % времени, поднимал руку 15 раз» back into a child; the school, holding the seating
plan, can. So the re-identification key never leaves the building, and the bundle without it
is close to the weakest form of the data that still supports the task. It is not zero, and
this module does not claim it is zero — which is why the whole path is **opt-in per run, by
an explicit command**, never part of `analyse` and never part of `cabinet import`. A school
that answers «нет» simply never runs `cabinet note`, and every page still renders in full:
the note is an aid on top of a complete deterministic report, and its absence is a stated
absence rather than a hole.

**Cost.** One call per run, small tier: ~9–11 KB in (7 257 B of prompt plus a 2.3–3.3 KB
bundle) and under 1 KB out — on the order of three thousand tokens for a lesson. The prompt
grew by half when the three meanings of a counter above were separated, and that is the
right trade: the alternative was a cheaper request that produced a false universal negative
about a child. Notes are stored, so
a re-render of the cabinet costs nothing at all — the page reads SQLite, never the network.
Re-generating is an explicit command, so a nightly HTML rebuild cannot quietly bill anyone.

--------------------------------------------------------------------------------------
**THE PLACE NUMBER IN THE NOTE IS THE CABINET'S, NOT THE ARTEFACT'S. THIS IS THE ONE
NON-OBVIOUS THING IN THE FILE AND IT WOULD HAVE BEEN A SILENT DEFECT.**

The artefact numbers seats per run (`seat_2`, `seat_3`, …, plus the adult, from
`room/seats.py`), and the cabinet numbers PLACES per room (`places.ordinal`, «место 1» …).
They are not the same numbering and on the real data they differ by one at every seat: on
`full_lesson.analysis.json` the adult holds artefact `seat_id = 1`, so artefact `seat_4` is
the cabinet's «место 3». Feeding the model artefact ids would therefore have produced a note
that says «место 4» directly above a table in which «место 4» is a different child — every
sentence true of the bundle, every sentence pointing one desk to the left, and nothing
anywhere to reveal it. So the bundle carries `places.ordinal`, which is exactly the number
the page prints, and `orientation.check()` then validates the note's digits against that
same set.

Two consequences follow, and both are stated rather than smoothed over:

  * **A seat with no place is not in the bundle.** If geometry could not attach a seat to a
    known place (`place_id IS NULL`), there is no number the page could resolve, so the seat
    is left out and its absence is listed in `unmeasured` — the model is told that some of
    the room is missing from what it was given, and the page repeats it. Silently shipping
    an unresolvable seat would put a number in the note that exists nowhere else.
  * **The adult is not in the bundle at all.** `weekly.py` refuses an activity index for the
    adult by role, and INTEGRATION.md §7 refuses a staff-comparison surface; a prose remark
    about where the teacher was would be that surface in one sentence. The omission is
    listed too, for rule 4.

--------------------------------------------------------------------------------------
**THE NOTE IS BUILT FROM THE STORE, NOT FROM THE ARTEFACT FILE.**

`runs.artefact_path` is documented as a last known path and not an identity: the JSON may
have been moved, archived or deleted long before a psychologist asks for a note, while the
cabinet's own promise is that the `.sqlite3` file is complete on its own — copy it to a USB
stick and everything renders. A generator that needed the original JSON would work on the
analyst's laptop and fail in the office, which is the machine that matters. Every number the
note needs was copied into `seat_lessons` at import, so it is read from there.

--------------------------------------------------------------------------------------
**THERE IS NO CLASS-LEVEL NOTE, AND THE REFUSAL IS THE DECISION — see
`CLASS_NOTE_REFUSAL_RU`, which the page prints where such a note would sit.**

It was considered and it is refused, for one reason that is not about taste:

**the guard cannot see the error a class-level note would make.** `orientation.check()`
rejects digits. A note about ONE lesson can only go wrong by producing a quantity, and it
cannot produce one. A note about SEVERAL lessons goes wrong by producing a DIRECTION — «на
месте 3 стало заметно спокойнее», «в этот раз к доске выходили чаще» — and a direction
contains no digits at all. It would pass the guard untouched, read as measured, and be
unfalsifiable on the page.

That claim is also precisely the one `metrics/trend.py` refuses to make without five of a
pupil's own lessons and an attested identity, and `weekly.trend_for` restates the refusal
with its four gates. Generating the same claim in prose, from the same data, with the gates
removed and the statistics replaced by fluency, would be a way of getting the refused
sentence onto the page by another door. Where the gates DO pass, the trend block already
states the direction with its median, its MAD and its sustained-lesson count — a paragraph
saying the same thing again in softer words would add a second opinion nobody can check.

So the cabinet's cross-lesson layer stays entirely deterministic, and the model is used only
where its judgement is about salience within a single hour.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from classvision.cabinet import lessons as lessons_mod
from classvision.cabinet import store
from classvision.report import orientation

# Backends this module accepts from an operator. "auto" resolves through
# `summary.available_backend()`, exactly as `orientation.write_note` does — one rule for
# what this environment can reach, not a second copy of it here.
BACKENDS = ("auto", "gemini", "openai", "deterministic")

# The whole of `--backend deterministic`, handled here rather than in `orientation.py`.
# Passing "deterministic" through to `write_note` would store the sentence «ни GEMINI_API_KEY,
# ни OPENAI_API_KEY не заданы», which on a machine that HAS a key is simply false — the
# operator chose not to use it. A stored reason is read months later by somebody deciding
# whether to try again, so it has to say which of the two happened.
DETERMINISTIC_REASON_RU = (
    "записка не запрашивалась: выбран режим `--backend deterministic`. Это не сбой и не "
    "отсутствие ключа — оператор решил не отправлять сводку счётчиков внешней модели. "
    "Все числа отчёта на месте: записка-ориентир ничего к ним не добавляет, она только "
    "подсказывает, с чего начать чтение."
)

# Printed on the class page where a cross-lesson note would go. The reasoning is in this
# module's docstring; this is the sentence a psychologist reads, and it has to name the
# action that would change the answer, like every other refusal in this package.
CLASS_NOTE_REFUSAL_RU = (
    "Записки по классу целиком — то есть по нескольким урокам сразу — здесь нет, и это "
    "решение, а не пропуск. Записка про один урок может ошибиться только величиной, а "
    "величин в ней нет по построению: проверка отклоняет любую цифру, кроме номера места. "
    "Записка про несколько уроков ошибается иначе — она говорит «стало спокойнее», "
    "«выходили чаще», — и в такой фразе нет ни одной цифры, поэтому проверка её пропустит. "
    "А это ровно то утверждение, которое динамика строит только по пяти собственным урокам "
    "ученика и только при подписанном плане рассадки. Получить его в обход этих условий, "
    "пересказом, нельзя. Когда условия выполнены, направление показывает блок «Динамика» на "
    "странице места — с медианой, разбросом и числом уроков, по которым оно посчитано."
)


# ---------------------------------------------------------------------------
# The bundle.
# ---------------------------------------------------------------------------


def bundle_for_run(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    """Everything the model is allowed to see about one run, and nothing else.

    A whitelist in the same sense as `report/summary.py::compact`: what is not built here
    cannot be sent, so the privacy claim in the module docstring is checkable by reading
    this function rather than by trusting a policy elsewhere.

    Shaped to be the input `orientation.compact_for_note` already expects — the same key
    names the artefact-side path uses (`coverage_percent`, `counts`, `activity.index`) —
    because the alternative is a second bundle format whose fields drift out of agreement
    with the tested one. `seat_id` carries the CABINET's place ordinal; see the module
    docstring for why that is not a detail.
    """
    run = connection.execute(
        "SELECT r.*, l.duration_seconds AS recording_seconds FROM runs r JOIN lessons l "
        "ON l.lesson_id = r.lesson_id WHERE r.run_id = ?", (run_id,)).fetchone()
    if run is None:
        raise store.Refusal(
            "unknown_run",
            f"прогона {run_id} нет в базе. Список прогонов: `cabinet show`.")

    rows = [dict(row) for row in connection.execute(
        """SELECT s.*, p.ordinal, p.label_ru FROM seat_lessons s
           LEFT JOIN places p ON p.place_id = s.place_id
           WHERE s.run_id = ? ORDER BY p.ordinal, s.seat_id""", (run_id,))]

    # A counter this camera CANNOT measure is sent as null, never as 0. On camera 01 the
    # board hangs behind the lens (`board_zone: null`), so `board_visits` is zero for every
    # pupil in every lesson that camera will ever record — and a zero handed to a language
    # model is an invitation to write «к доске никто не выходил», which is a finding about
    # children assembled out of a fact about a lens. Rule 3 of this project, and the same
    # rule `cabinet/lessons.py` applies to the tables; its map is imported rather than
    # copied, because two lists of which sentence voids which counter is two chances for one
    # of them to fall behind the analyser's wording.
    unmeasured = [dict(item) for item in json.loads(run["unmeasured_json"] or "[]")]
    voided = dict(lessons_mod.voided_counters(unmeasured))

    # ...and the OTHER way a zero in this column lies, which `voided_counters` cannot see.
    # On D14 the board IS in frame, so nothing voids «выходил к доске» and it arrives as a
    # measured 0 for all five places — while the same artefact reports somebody at that
    # board for 30.4 of the 58.1 minutes, only 4.0 of them the adult. The zero is true of
    # the ledger (`AT_BOARD` is a state of a PLACE, and a child at the board has left their
    # place) and false of the room. `cabinet/lessons.py` already prints that number beside
    # its contradiction; a note cannot print numbers, so its only honest options are a
    # stated null or a confident «к доске никто не выходил» — the very sentence the block
    # above refuses to risk on camera 01. Same treatment, second cause.
    if "board_visits" not in voided and lessons_mod.board_conflict_for_run(
            connection, run_id):
        voided["board_visits"] = "board_conflict"
        # Digit-free on purpose, like the unresolved-seat sentence below: the page's own
        # version of this warning quotes 30.4 and 52.3, and quoting them here would put two
        # numbers in front of a model whose every digit the guard rejects — paid for by
        # losing the whole note.
        unmeasured.append({
            "what": "выход ученика к доске этой записью не подтверждается",
            "why": "счётчик «у доски» описывает состояние МЕСТА, а ребёнок, вышедший к "
                   "доске, со своего места уходит и попадает в «уходил с места» или в "
                   "наблюдения вне мест; у доски в этой записи кто-то стоял значительную "
                   "часть урока",
        })

    seats: list[dict[str, Any]] = []
    unresolved = 0
    adults = 0
    brief_only_seen = False
    for row in rows:
        if str(row["role"]) != "pupil":
            adults += 1
            continue
        if row["ordinal"] is None:
            unresolved += 1
            continue
        brief = _zero_but_briefly_seen(row, voided)
        brief_only_seen = brief_only_seen or bool(brief)
        seats.append({
            # The number the PAGE prints. Not `seat_lessons.seat_id`.
            "seat_id": int(row["ordinal"]),
            # Never a name, on any code path, whatever the seating plan says. The key is
            # here because `compact_for_note` reads it to emit a boolean; the value is
            # `None` for the reason the module docstring gives at length.
            "pupil_name": None,
            # `store` keeps coverage as a 0..1 fraction; the bundle the note prompt was
            # written against carries a percentage under this exact name, and the prompt's
            # rule 6 quotes it («по полю coverage_percent»). Converting here, once, keeps
            # the prompt's own words true of what it is handed.
            "coverage_percent": (None if row["coverage"] is None
                                 else round(100.0 * float(row["coverage"]), 1)),
            "activity": {"index": row["activity_index"]},
            "counts": {key: (None if key in voided else row[key])
                       for key in ("hand_raises", "stands", "away_episodes",
                                   "head_down_episodes", "turned_away_episodes",
                                   "board_visits")},
            # Which of those zeros mean «не набралось эпизода», not «не было». See
            # `_zero_but_briefly_seen`.
            "zero_but_briefly_seen": brief,
        })

    if brief_only_seen:
        unmeasured.append({
            "what": "часть нулей в счётчиках означает «не набралось ни одного эпизода "
                    "нужной длины», а не «такого не было»",
            "why": "счётчики считают эпизоды, а короткие включения состояния в эпизод не "
                   "складываются; у каких мест и каких счётчиков это так, перечислено в "
                   "поле zero_but_briefly_seen",
        })

    if unresolved:
        # Worded without a numeral on purpose. Everything in `unmeasured` is quoted to the
        # model, the model is forbidden to write digits, and the guard rejects the WHOLE
        # note over one — so a sentence here saying «2 места» is a trap this module would be
        # setting for itself, paid for by losing the note entirely.
        unmeasured.append({
            "what": "часть мест этого урока не привязана к местам комнаты и в эту сводку "
                    "не попала",
            "why": "геометрия не позволила однозначно сопоставить их с известными местами",
        })
    if adults:
        unmeasured.append({
            "what": "взрослый в эту сводку не включён",
            "why": "показатели ученика к взрослому не применяются",
        })

    minutes = _analysed_minutes(run)
    if minutes is None:
        # Rule 4: never a silently missing field. Every `coverage_percent` above is a share
        # of this window, so a reader told nothing would take the shares as shares of the
        # lesson — which is what they usually are, and here is exactly what nobody can say.
        unmeasured.append({
            "what": "длина разобранного окна урока",
            "why": "в разборе не записаны границы окна, поэтому доли видимости не с чем "
                   "соотнести",
        })

    return {
        "lesson": {"duration_minutes": minutes},
        "seats": seats,
        "unmeasured": unmeasured,
    }


# Counter key -> the ledger state whose seconds it counts episodes of. Inverted from
# `store.COUNT_KEYS` rather than retyped, so a state that gains or renames a counter cannot
# leave a second copy of the mapping behind — the failure `lessons.COUNTER_VOIDED_BY`
# documents one file away.
_STATE_OF_COUNTER: dict[str, str] = {
    counter: state for state, counter in store.COUNT_KEYS.items() if counter
}


def _zero_but_briefly_seen(row: dict[str, Any], voided: dict[str, str]) -> list[str]:
    """Which of this place's ZERO counters had the state observed anyway. Rule 3, one level down.

    **The counters are EPISODE counts, and `0` therefore has two meanings that the bundle
    used to send as one.** «Не было ни разу» and «было, но ни одно включение не дотянуло до
    эпизода» are different facts, and `ledger.py` keeps them apart in two different fields —
    `episode_seconds` against `observed_seconds_by_state`, plus `discarded_short_runs` which
    counts the runs thrown away for being too short. The bundle carried only the episode
    count, so both arrived as `0`.

    Measured on the two real artefacts, and it is not an edge case:

      * camera 01, «место 8» — every counter zero, and the note written about it said
        «отсутствие каких-либо движений … всё время удалось провести спокойно за партой».
        That place has **150 observations of head_down, 75.0 seconds, 16 runs discarded as
        too short**, plus three observations of a raised hand. The sentence is false, and no
        digit in it could have been checked against anything.
      * D14 — `head_down_episodes` is 0 at all five places, and the note said «ни на одном
        из мест … не было отметки о том, что ребёнок опускал голову на стол». Places 2, 3
        and 5 have **32.0 s, 3.5 s and 20.0 s** of observed head_down behind that zero.
      * D14, «место 1» — «отходов от парты и поднятых рук не было», against 21 observations
        (10.5 s) of each.

    A universal negative is exactly what a fluent model builds out of a zero, and it is the
    same failure as `board_visits` one layer deeper: the previous fix asked whether the
    camera could see the event at all, and this one asks whether the counter's threshold ate
    it. So the zeros stay — they are true of the ledger, and blanking them would lose the
    real ones — and each place carries the list of counters whose zero must not be read as
    «не было». `SYSTEM_RU` rule 7 is what turns that list into a forbidden phrasing.

    A counter already voided for the camera is left out: it is `null`, not `0`, and has no
    zero to qualify.
    """
    seconds = json.loads(row.get("observed_seconds_by_state_json") or "{}") or {}
    out = []
    for counter, state in _STATE_OF_COUNTER.items():
        if counter in voided or row.get(counter):
            continue
        if float(seconds.get(state) or 0.0) > 0.0:
            out.append(counter)
    return out


def _analysed_minutes(run: sqlite3.Row) -> float | None:
    """The ANALYSED window in minutes, not the length of the recording.

    They differ, and the difference is the one every share in this bundle is a share of:
    on `full_lesson.analysis.json` the file is 52.4 minutes and the analysed window is 50.0,
    because the pipeline starts where the clock became readable. `coverage_percent` is
    computed against the window, so publishing the file's length beside it would offer a
    reader — human or model — two lengths and no way to tell which divides which. The
    artefact's own `lesson.duration_minutes` is this same window, so the note bundle and the
    lesson report agree on the number by construction rather than by coincidence.

    **And when the window is unknown, the answer is `None` — not the recording's length.**
    `runs.window_start_seconds` is nullable, so an artefact that did not report a window used
    to fall through to the file's duration and publish it under `lesson_minutes`, the name
    that everywhere else means the analysed window. That is the paragraph above arguing
    against itself: the two lengths differ by 2.4 minutes on the very artefact it quotes, and
    the substitute arrives with nothing to mark it as a substitute. It is the same shape as
    `MEASUREMENTS.md` §8 — one name, two quantities, no way for the reader to tell which —
    and the fact that no digit escapes the guard is luck, not a defence. An unknown length is
    sent as unknown and listed in `unmeasured` by the caller.
    """
    start, end = run["window_start_seconds"], run["window_end_seconds"]
    if start is None or end is None:
        return None
    return round((float(end) - float(start)) / 60.0, 1)


def payload(bundle: dict[str, Any]) -> tuple[str, str, int]:
    """The exact JSON that will leave the machine -> (text, sha256, bytes).

    Serialised the way `orientation.write_note` serialises it, so that the byte count in
    this module's docstring is the number a network capture would show rather than an
    estimate of it. The hash is what `stored_note` compares against later: it identifies
    the numbers a stored note was written about, which is a different question from which
    run it belongs to, and both are answered separately.
    """
    text = json.dumps(orientation.compact_for_note(bundle), ensure_ascii=False, indent=1)
    encoded = text.encode("utf-8")
    return text, hashlib.sha256(encoded).hexdigest(), len(encoded)


# ---------------------------------------------------------------------------
# Generating.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NoteOutcome:
    """What one `cabinet note` attempt did, including everything it decided not to do."""

    run_id: str
    lesson_id: int
    date_local: str | None
    # "stored" | "kept_existing" | "refused_by_guard" | "no_provider" | "deterministic"
    # | "call_failed"
    outcome: str
    available: bool
    text: str = ""
    model: str | None = None
    source: str = "none"
    guard_passed: bool | None = None
    reason_ru: str = ""
    bundle_bytes: int = 0
    notes_ru: list[str] = field(default_factory=list)
    # Whether THIS attempt put the bundle on the wire. Stored rather than derived; see
    # `bundle_was_sent`.
    bundle_sent: bool = False

    @property
    def bundle_was_sent(self) -> bool:
        """Did the bundle leave this machine DURING THIS ATTEMPT?

        **Not `model is not None`, and the difference is a false sentence about children's
        data.** `StoredNote.bundle_was_sent` asks a different question — «была ли отправлена
        сводка, о которой написана ЭТА ХРАНИМАЯ записка» — and answers it correctly from
        `model`, because a stored record's model is the model that wrote it. Copying that
        rule here made the two names mean two things: on the `kept_existing` path `model` is
        the EXISTING note's model, so a run that sent nothing at all reported «отправлено
        3261 байт счётчиков».

        That path is not a corner. It is the one this class was designed for: a nightly
        `cabinet note --all` on a machine whose key has expired keeps yesterday's notes, and
        printed the traffic line of yesterday's requests as though it had made them today.
        `--backend deterministic` over an existing note does the same. Both are exactly the
        case where a school reading the log wants to know that nothing went out.

        So the answer is recorded at the point of the attempt, from the attempt's own model,
        and a guard rejection still counts as sent — the request happened, only its answer
        was thrown away.
        """
        return self.bundle_sent

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "lesson_id": self.lesson_id,
            "date_local": self.date_local, "outcome": self.outcome,
            "available": self.available, "text": self.text, "model": self.model,
            "source": self.source, "guard_passed": self.guard_passed,
            "reason_ru": self.reason_ru, "bundle_bytes": self.bundle_bytes,
            "bundle_was_sent": self.bundle_was_sent, "notes_ru": self.notes_ru,
        }


def generate_for_run(connection: sqlite3.Connection, run_id: str, *,
                     backend: str = "auto", timeout: float = 60.0) -> NoteOutcome:
    """Ask for one run's note and store the answer. Never raises on a provider failure.

    Four outcomes, all normal, none an error:

      * a clean note is stored and replaces whatever was there;
      * the guard rejected the text — the rejection is stored, with its offending strings,
        so the page can say «записка отклонена» instead of quietly having no note;
      * there is no provider, or the call failed — the reason is stored;
      * `--backend deterministic` — the operator's own decision is stored as such.

    **A failure never overwrites an existing note.** The numbers a stored note describes
    have not changed just because today's key is missing, and a nightly job that replaced
    every note in the cabinet with «ключа нет» would be a regression caused by nothing
    happening. The old note is kept, the attempt is reported, and the caller prints both.
    """
    row = connection.execute(
        "SELECT r.run_id, r.lesson_id, l.date_local FROM runs r "
        "JOIN lessons l ON l.lesson_id = r.lesson_id WHERE r.run_id = ?",
        (run_id,)).fetchone()
    if row is None:
        raise store.Refusal(
            "unknown_run",
            f"прогона {run_id} нет в базе — записку не к чему привязать. Прогоны этого "
            f"кабинета перечисляет `cabinet show`; загрузить новый — `cabinet import`.")

    bundle = bundle_for_run(connection, run_id)
    _text, digest, size = payload(bundle)

    if backend == "deterministic":
        note = orientation.Note("", False, "none", reason=DETERMINISTIC_REASON_RU)
    elif not bundle["seats"]:
        # No place the note could point at. Asking anyway would spend a call to be told so.
        note = orientation.Note(
            "", False, "none",
            reason="в этом прогоне нет ни одного места, привязанного к комнате, — "
                   "показывать не на что. Записка появится, когда места этого урока "
                   "будут сопоставлены с местами комнаты.")
    else:
        note = orientation.write_note(bundle, backend=backend, timeout=timeout)

    outcome = _outcome_code(note, backend)
    existing = store.note_row(connection, run_id)

    if not note.available and existing is not None and existing["available"]:
        return NoteOutcome(
            run_id=run_id, lesson_id=int(row["lesson_id"]), date_local=row["date_local"],
            outcome="kept_existing", available=True, text=existing["text"],
            model=existing["model"], source=existing["source"],
            guard_passed=(None if existing["guard_passed"] is None
                          else bool(existing["guard_passed"])),
            reason_ru=note.reason, bundle_bytes=size,
            # The ATTEMPT's model, not the kept note's: nothing was sent unless this run
            # reached a provider. See `NoteOutcome.bundle_was_sent`.
            bundle_sent=note.model is not None,
            notes_ru=[f"новая записка не получена ({note.reason}); прежняя от "
                      f"{existing['generated_at']} оставлена без изменений — числа, о "
                      f"которых она написана, не менялись."])

    store.save_note(
        connection, run_id, text=note.text, available=note.available, source=note.source,
        backend_requested=backend, model=note.model, prompt_version=note.prompt_version,
        guard_passed=None if note.check is None else note.check.passed,
        guard_reason_ru="" if note.check is None else note.check.reason_ru,
        guard_offending=() if note.check is None else note.check.offending,
        reason_ru=note.reason, bundle_sha256=digest, bundle_bytes=size)

    return NoteOutcome(
        run_id=run_id, lesson_id=int(row["lesson_id"]), date_local=row["date_local"],
        outcome=outcome, available=note.available, text=note.text, model=note.model,
        source=note.source,
        guard_passed=None if note.check is None else note.check.passed,
        reason_ru=note.reason, bundle_bytes=size,
        bundle_sent=note.model is not None,
        notes_ru=([f"записка заменена: прежняя была от {existing['generated_at']}"]
                  if existing is not None and note.available else []))


def _outcome_code(note: orientation.Note, backend: str) -> str:
    if note.available:
        return "stored"
    if backend == "deterministic":
        return "deterministic"
    if note.check is not None and not note.check.passed:
        return "refused_by_guard"
    if "не удалось" in note.reason:
        return "call_failed"
    return "no_provider"


def selected_run_ids(connection: sqlite3.Connection, *, room_key: str | None = None,
                     class_key: str | None = None) -> list[str]:
    """The runs `--all` means: the ones the cabinet's pages actually show.

    A lesson re-analysed under different thresholds keeps both runs, and only the selected
    one is rendered anywhere (`store.seat_rows`). Generating a note for the other would
    spend a call on a paragraph no page has a place for; a note for it can still be asked
    for by naming its run id, which is what `select-run` users will want.
    """
    where, args = store._scope(room_key, class_key, prefix="l.")
    return [row["selected_run_id"] for row in connection.execute(
        f"SELECT l.selected_run_id FROM lessons l{where}"
        f"{' AND' if where else ' WHERE'} l.selected_run_id IS NOT NULL "
        f"ORDER BY l.started_at IS NULL, l.started_at", args)]


def generate_all(connection: sqlite3.Connection, *, backend: str = "auto",
                 room_key: str | None = None, class_key: str | None = None,
                 timeout: float = 60.0) -> list[NoteOutcome]:
    """Every selected run, continuing past anything that fails. One call per run."""
    return [generate_for_run(connection, run_id, backend=backend, timeout=timeout)
            for run_id in selected_run_ids(connection, room_key=room_key,
                                           class_key=class_key)]


# ---------------------------------------------------------------------------
# Reading back, with freshness.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoredNote:
    """A stored note as the page needs it: text, provenance, verdict, and freshness."""

    run_id: str
    text: str
    available: bool
    stale: bool
    model: str | None
    source: str
    # What the operator ASKED for, kept beside what happened: «записки нет» after
    # `--backend deterministic` is a decision and needs no fixing, while the same absence
    # after `--backend auto` is a missing key and does. The page says different things.
    backend_requested: str
    prompt_version: str
    guard_passed: bool | None
    guard_reason_ru: str
    guard_offending: list[str]
    reason_ru: str
    generated_at: str
    bundle_bytes: int

    @property
    def bundle_was_sent(self) -> bool:
        """Did the bundle actually leave this machine for this record?

        `source` cannot answer it: `orientation.write_note` returns `source = "none"` for a
        guard rejection, which happens AFTER a real request. `model` is the honest witness —
        it is set once a provider has been chosen and a call attempted, and stays `None`
        when there was no key, no provider, or an operator's `--backend deterministic`.
        The question matters because «отправлено N байт» printed about a request that was
        never made is a false statement about children's data leaving a building, which is
        the one sentence in this feature that has to be exactly true.
        """
        return self.model is not None

    @property
    def state(self) -> str:
        """"shown" | "stale" | "refused_by_guard" | "absent_reason" — one word for the page.

        Staleness is asked ONLY of a record that has text. A stored «ключа не было» carries
        the hash of a bundle that was never sent, and testing it first made a record with no
        note report «записка написана о других числах» — an accusation about a paragraph
        that does not exist, complete with «написала модель неизвестна».
        """
        if self.available:
            return "stale" if self.stale else "shown"
        if self.guard_passed is False:
            return "refused_by_guard"
        return "absent_reason"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "state": self.state, "text": self.text,
            "available": self.available, "stale": self.stale, "model": self.model,
            "source": self.source, "backend_requested": self.backend_requested,
            "prompt_version": self.prompt_version,
            "guard_passed": self.guard_passed, "guard_reason_ru": self.guard_reason_ru,
            "guard_offending": self.guard_offending, "reason_ru": self.reason_ru,
            "generated_at": self.generated_at, "bundle_bytes": self.bundle_bytes,
        }


def stored_note(connection: sqlite3.Connection, run_id: str) -> StoredNote | None:
    """The note of one run, with staleness decided rather than assumed. `None` if never asked.

    Freshness is recomputed, not trusted: the bundle is rebuilt from the store and hashed,
    and a note whose hash does not match the numbers now in the store is marked `stale` and
    is NOT shown. The run id already prevents a note crossing to a re-analysis; this catches
    the remaining case, where the numbers under one run id are not the numbers that note was
    written about — a cabinet rebuilt from re-exported artefacts, a corrected import, a
    column added later. A note is a statement about specific figures, and the moment we
    cannot show that they are the figures beside it, the honest move is to say so and
    re-generate.
    """
    row = store.note_row(connection, run_id)
    if row is None:
        return None
    try:
        _text, digest, _size = payload(bundle_for_run(connection, run_id))
    except store.Refusal:
        digest = ""
    return StoredNote(
        run_id=run_id, text=row["text"], available=bool(row["available"]),
        stale=digest != row["bundle_sha256"],
        model=row["model"], source=row["source"],
        backend_requested=row["backend_requested"], prompt_version=row["prompt_version"],
        guard_passed=(None if row["guard_passed"] is None else bool(row["guard_passed"])),
        guard_reason_ru=row["guard_reason_ru"],
        guard_offending=json.loads(row["guard_offending_json"] or "[]"),
        reason_ru=row["reason_ru"], generated_at=row["generated_at"],
        bundle_bytes=int(row["bundle_bytes"] or 0))

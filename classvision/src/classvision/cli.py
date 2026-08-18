"""The command line: seven verbs, and every one of them prints what it could not do.

**Exit codes carry meaning**, because these commands end up in a scheduled job on a school
machine where nobody reads stdout: `0` the analysis is usable, `1` it ran and the result
must not be trusted (the seat discovery disagreed with the detector, the clock could not
be read), `2` it could not run at all. A cron that treats "ran but untrustworthy" as
success is how a term's data quietly fills with merged seats.

**`analyse-session` is a separate verb because it makes a separate claim.** `analyse FILE`
says "this recording"; `analyse-session A B ...` says "these recordings are one continuous
lesson", which is a statement about the world that `session.py` checks against the wall
clock and can refuse. Refusal exits `2`: the files were not merged and no artefact was
written, which is the correct outcome for a job that was handed two different lessons.

**`analyse` never overwrites.** Its output is keyed by a `run_id` derived from the video
and every setting that could change a number, so re-running with a different threshold
writes a different file. That is deliberate friction: a report is a statement about a past
hour, and silently restating last month's lesson under this month's thresholds is the
defect this whole package is organised against.

**`--room` means exactly one thing, and the other one was renamed.** On `analyse`,
`analyse-session` and `zones` it is a PATH to a camera profile (`configs/camera_d14.yaml`)
— a set of polygons that changes numbers. The `cabinet` verbs used to spell their own,
unrelated argument `--room` as well: a room KEY, a bare string the operator asserts so that
two lessons can be filed together, which the artefact does not know. Two flags with one
name and two meanings in one program is the defect that has already shipped once in this
project, so the string half is now `--room-key` (`_common()` states why the string half is
the cheaper one to move).

This paragraph used to say the opposite — that the collision was «stated rather than
renamed» and that «both keep the name» — while the code below it had already renamed one.
A comment that contradicts the code it sits on is worse than no comment, because it is the
one an operator reads instead of `--help`. Note that argparse still accepts `--room` on
`cabinet` as an unambiguous ABBREVIATION of `--room-key`; that is argparse's behaviour and
not a second spelling this CLI offers, and nothing here documents it as one.

**`zones` exists because a polygon cannot be reviewed by reading it.** It draws the profile
over a real frame, runs the same pose model the analysis runs, and reports which zone each
person's shoulder line fell into. A zone that is silently wrong produces an artefact in
which «у доски» is simply always zero, and zero is a number a reader believes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OK = 0
UNTRUSTWORTHY = 1
FAILED = 2


def _layout(args):
    """The room profile, or None. Refusals here are fatal, not a fallback to "no zones".

    A `--room` that does not load must stop the run, because the alternative — carrying
    on with `layout=None` — produces a complete, plausible artefact in which «у доски» is
    absent and the adult was guessed by scale. That artefact is indistinguishable from one
    produced by a camera nobody has configured, which is precisely the confusion the
    profile exists to remove.
    """
    from classvision.room import layout_io

    path = getattr(args, "room", None)
    return None if not path else layout_io.load(path)


def _settings(args):
    from classvision.pipeline import Settings

    return Settings(
        layout=_layout(args),
        sample_fps=args.sample_fps,
        imgsz=args.imgsz,
        weights=args.weights,
        device=args.device,
        start_seconds=args.start or 0.0,
        end_seconds=args.end,
        cache_dir=Path(args.cache),
        seat_map=Path(args.seat_map) if args.seat_map else None,
        roster_csv=Path(args.roster) if args.roster else None,
        photos_root=Path(args.photos) if args.photos else None,
        class_name=args.class_name,
        face_corroboration=args.faces,
    )


def cmd_zones(args) -> int:
    """Draw a room profile over one real frame and say where every person landed.

    Exit `1` — ran, do not trust it — when the frame produced **no** person in the board
    zone while a board zone IS configured. That is not a hard error: an operator may
    legitimately check a frame in which nobody is at the board. But it is the exact
    symptom of the commonest way to get a zone wrong, so it must not exit `0` silently;
    the operator picks a frame where somebody IS at the board and runs it again.

    **The `board_zone is None` case is not that, and used to be reported as if it were.**
    On `configs/camera_01.yaml` the board hangs behind the lens, `board_zone` is a measured
    `null`, and `in_board` is therefore zero in every frame that will ever exist. The first
    version of this verb read that zero as the warning above and told the operator
    «разметка неверна» about the one profile in the repository whose null was arrived at by
    measurement — a «не измеряется» printed as a finding, which is the failure this whole
    package is organised against. Zero-because-nothing-was-there and
    zero-because-there-is-nothing-to-measure now print different sentences and return
    different codes, and `--expect-in-board` against a camera with no board zone is refused
    outright rather than silently satisfied by the null.
    """
    from classvision.report import zonecheck

    result = zonecheck.check(args.video, args.room, at_seconds=args.at,
                             out_path=args.out, weights=args.weights,
                             device=args.device, imgsz=args.imgsz)
    print(f"камера {result['camera']}  кадр {args.at} с  {result['frame'][0]}x"
          f"{result['frame'][1]}  людей найдено {result['people']}")
    for person in result["detail"]:
        anchor = ("—" if person["anchor"] is None
                  else f"({person['anchor'][0]:.0f}, {person['anchor'][1]:.0f})")
        print(f"  плечи {anchor:>16s}  низ рамки "
              f"({person['foot'][0]:.0f}, {person['foot'][1]:.0f})  →  {person['verdict']}")
    if result["image"]:
        print(f"{result['image']}")
    if result["no_anchor"]:
        print(f"  БЕЗ ЛИНИИ ПЛЕЧ  {result['no_anchor']} чел. — эти НЕ проверены ни на "
              f"одну зону; это не «вне зон»", file=sys.stderr)

    if not result["board_zone_configured"]:
        if args.expect_in_board is not None:
            print("\n  ОТКАЗ: у этой камеры `board_zone: null` — доски в кадре нет, и "
                  "«у доски» не может быть измерено ни в одном кадре. Ожидание "
                  f"«у доски» = {args.expect_in_board} проверить не на чем.",
                  file=sys.stderr)
            return FAILED
        print("\n  Зона доски у этой камеры не задана (`board_zone: null`) — это "
              "записанное решение, а не пропуск: доска вне поля зрения объектива. "
              "«У доски» на этой камере не измеряется никогда, и ноль здесь означает "
              "«нечего измерять», а не «никого не было».", file=sys.stderr)
        return OK

    if args.expect_in_board is not None and result["in_board"] != args.expect_in_board:
        print(f"\n  ВНИМАНИЕ: ожидалось «у доски» {args.expect_in_board}, получено "
              f"{result['in_board']}", file=sys.stderr)
        return UNTRUSTWORTHY
    if result["in_board"] == 0 and args.expect_in_board is None:
        print("\n  ВНИМАНИЕ: в этом кадре в зону доски не попал никто. Если у доски "
              "кто-то стоял — разметка неверна.", file=sys.stderr)
        return UNTRUSTWORTHY
    return OK


def cmd_analyse(args) -> int:
    from classvision.pipeline import analyse
    from classvision.room import layout_io
    from classvision.video import decode

    settings = _settings(args)
    if settings.layout is not None:
        # Before the ten-minute detection pass, not after: a profile drawn for another
        # frame size lands every zone a quarter of the way up and to the left, which on
        # D14 puts the board zone on the front row of desks. Nothing downstream notices.
        info = decode.probe(args.video)
        layout_io.check_frame(settings.layout, info.width, info.height)

    artefact = analyse(args.video, settings)
    out = Path(args.out) if args.out else Path(f"out/{Path(args.video).stem}.analysis.json")
    artefact.write(out)

    discovery = artefact.provenance.room["seat_discovery"]
    lesson = artefact.lesson
    print(f"{out}  run_id={artefact.run_id}")
    print(f"  урок      {lesson['window_wall'][0]} .. {lesson['window_wall'][1]}"
          f"  ({lesson['duration_minutes']} мин)")
    print(f"  места     {discovery['seats_found']} найдено, из них учеников "
          f"{lesson['pupil_seats']}, взрослый — место {lesson['adult_seat']}")
    print(f"  метод     {discovery.get('method')}, eps="
          f"{discovery['thresholds']['eps_shoulder_widths']} "
          f"({discovery['thresholds']['chosen_by']})")
    for item in lesson.get("unmeasured", []):
        print(f"  НЕ ИЗМЕРЕНО  {item['what']}: {item['why']}")

    if discovery.get("plausible") is False:
        print(f"\n  ВНИМАНИЕ: {discovery.get('warning')}", file=sys.stderr)
        return UNTRUSTWORTHY
    if artefact.provenance.clock_source == "unknown":
        print("\n  ВНИМАНИЕ: время записи не определено — этот прогон нельзя "
              "использовать в недельной динамике.", file=sys.stderr)
        return UNTRUSTWORTHY
    return OK


def cmd_analyse_session(args) -> int:
    """Several DVR files that are one lesson -> ONE artefact.

    Separate from `analyse` rather than a flag on it, because the two commands make
    different claims. `analyse FILE` says "this recording"; `analyse-session A B` says
    "these recordings are one continuous lesson", and that claim is checked (`session.py`)
    and can be refused. A flag would let a typo turn the first claim into the second.
    """
    from classvision import session as session_mod

    tolerance = (session_mod.MAX_SEAM_GAP_SECONDS if args.tolerance is None
                 else args.tolerance)
    try:
        artefact = session_mod.analyse(args.videos, _settings(args),
                                       tolerance_seconds=tolerance)
    except session_mod.NotOneLesson as refusal:
        print(f"ОТКАЗ СКЛЕИВАТЬ: {refusal}", file=sys.stderr)
        return FAILED

    out = (Path(args.out) if args.out
           else Path(f"out/{Path(args.videos[0]).stem}.session.analysis.json"))
    artefact.write(out)

    block = artefact.provenance.session or {}
    discovery = artefact.provenance.room["seat_discovery"]
    lesson = artefact.lesson
    print(f"{out}  run_id={artefact.run_id}")
    print(f"  сессия    {len(block.get('parts', []))} файла, "
          f"{block.get('duration_seconds', 0) / 60.0:.1f} мин, "
          f"источник времени — {block.get('clock_source')}")
    for part in block.get("parts", []):
        print(f"    [{part['index']}] {Path(part['path']).name}  "
              f"{part['started_at']}  {part['duration_seconds'] / 60.0:.1f} мин  "
              f"сдвиг {part['offset_seconds']:.2f} с  sha256={part['sha256'][:12]}…")
    for seam in block.get("seams", []):
        print(f"    стык после файла {seam['after_part']} на "
              f"{seam['at_session_seconds']:.2f} с ({seam['at_wall']}): {seam['kind']} "
              f"{seam['gap_seconds']:+.2f} с, отброшено повторов: "
              f"{seam['trimmed_observations']} набл. / {seam['trimmed_frames']} кадр.")
    print(f"  урок      {lesson['window_wall'][0]} .. {lesson['window_wall'][1]}"
          f"  ({lesson['duration_minutes']} мин)")
    print(f"  места     {discovery['seats_found']} найдено, из них учеников "
          f"{lesson['pupil_seats']}, взрослый — место {lesson['adult_seat']}")
    for item in lesson.get("unmeasured", []):
        print(f"  НЕ ИЗМЕРЕНО  {item['what']}: {item['why']}")

    if discovery.get("plausible") is False:
        print(f"\n  ВНИМАНИЕ: {discovery.get('warning')}", file=sys.stderr)
        return UNTRUSTWORTHY
    if artefact.provenance.clock_source == "unknown":
        print("\n  ВНИМАНИЕ: время записи не определено — этот прогон нельзя "
              "использовать в недельной динамике.", file=sys.stderr)
        return UNTRUSTWORTHY
    return OK


def cmd_report(args) -> int:
    from classvision.report import html, summary

    artefact = json.loads(Path(args.artefact).read_text(encoding="utf-8"))
    result = summary.summarise(artefact, backend=args.backend)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / (Path(args.artefact).stem + ".summary.ru.txt")
    text_path.write_text(result.text, encoding="utf-8")
    page = html.render_report(args.artefact, out_dir, summary_text=result.text)

    print(f"{page}\n{text_path}")
    # Which generator produced the text is not a detail. A school comparing two lessons
    # needs to know whether the wording changed because the child did or because an API
    # key appeared on the machine.
    print(f"  сводка    {result.source}"
          + (f" (причина отката: {result.fallback_reason})" if result.fallback_reason else ""))
    if result.number_check is not None:
        print(f"  {result.number_check.report_ru().splitlines()[0]}")
    return OK


def cmd_verify(args) -> int:
    from classvision.report.overlay import Window, agrees_with_artefact, render

    artefact = json.loads(Path(args.artefact).read_text(encoding="utf-8"))
    video = args.video or artefact["provenance"]["video_path"]

    check = agrees_with_artefact(video, args.artefact)
    print(f"разметка совпадает с отчётом: {check['ok']} "
          f"(мест проверено {check['seats_checked']}, расхождений {check['mismatch_count']})")
    for mismatch in check["mismatches"][:10]:
        print("   ", mismatch)

    if args.out:
        window = Window(args.start_seconds, args.end_seconds)
        path = render(video, args.artefact, args.out, window=window, fps_out=args.fps)
        print(f"{path}")
    return OK if check["ok"] else UNTRUSTWORTHY


def cmd_seatmap_template(args) -> int:
    from classvision.identity import seatmap

    seats, _ = seatmap.seats_from_artefact(args.artefact)
    path = seatmap.write_template(seats, args.out, camera=args.camera,
                                  class_name=args.class_name)
    print(f"{path}\n  заполните external_id и attested_by, пустые места останутся "
          f"анонимными — это нормально")
    return OK


def cmd_roster_check(args) -> int:
    from classvision.identity import roster as roster_mod

    register = roster_mod.load(args.roster, args.photos, class_name=args.class_name)
    summary = register.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return OK if not summary.get("discrepancies") else UNTRUSTWORTHY


def cmd_cabinet_import(args) -> int:
    """Ingest artefacts into the cabinet. Every refusal is printed; none is fatal to the rest.

    Exit code follows the module's convention: `0` everything usable, `1` it ran but
    something must not be trusted or was refused, `2` it could not run. A back-fill script
    that treats a refused artefact as success is how a term's data quietly acquires a
    doubled lesson.
    """
    from classvision.cabinet import store

    connection = store.connect(args.db)
    outcomes = store.import_many(
        connection, args.artefacts, room_key=args.room, class_key=args.class_name,
        allow_unclocked=args.allow_unclocked, allow_overlap=args.allow_overlap,
        select="always" if args.select else "if_first")

    refused = 0
    for path, outcome in outcomes:
        name = Path(path).name
        if isinstance(outcome, store.Refusal):
            refused += 1
            print(f"ОТКАЗ     {name}  [{outcome.code}]\n            "
                  f"{outcome.message_ru}", file=sys.stderr)
            continue
        if isinstance(outcome, store.Unreadable):
            refused += 1
            print(f"НЕ ЧИТАЕТСЯ {name}: {outcome}", file=sys.stderr)
            continue
        if outcome.already_imported:
            print(f"уже есть  {name}  run={outcome.run_id}")
        else:
            print(f"загружен  {name}  run={outcome.run_id}  урок №{outcome.lesson_id}"
                  f"{' (новый)' if outcome.is_new_lesson else ''}  "
                  f"мест {outcome.seats_stored} "
                  f"(новых {outcome.places_created}, узнано {outcome.places_matched}, "
                  f"без привязки {outcome.seats_unmatched})"
                  f"{'  ← выбран для сводок' if outcome.selected else ''}")
        for note in outcome.notes_ru:
            print(f"            {note}")
        for dropped in outcome.dropped_seats:
            print(f"            место {dropped['seat_label']}: {dropped['why_ru']}")

    print()
    _print_store(store.summary(connection))
    return UNTRUSTWORTHY if refused else OK


def cmd_cabinet_weekly(args) -> int:
    from classvision.cabinet import lessons as lessons_mod
    from classvision.cabinet import store, weekly

    connection = store.connect(args.db)
    view = weekly.class_view(connection, room_key=args.room, class_key=args.class_name)
    if args.json:
        print(json.dumps(view.to_dict(), ensure_ascii=False, indent=1, default=str))
        return OK

    sessions = sum(1 for row in view.lessons if row["continues_lesson_id"] is None)
    print(f"{view.class_display_ru}, комната {view.room_key}: "
          f"записей {len(view.lessons)} (занятий {sessions}), "
          f"мест {len(view.histories)}")
    weeks = [f"{y}-W{w:02d}" for y, w in view.span]
    print(f"недели: {', '.join(weeks) if weeks else '— (нет уроков с датой)'}")

    # Printed BEFORE the per-place blocks, not after them. An operator reading a terminal
    # scrolls to the bottom and stops, and the sentence that has to arrive before any number
    # is the one saying that the number next door belongs to another room.
    if view.other_rooms:
        from classvision.cabinet.lessons import ClassAcrossRooms

        across = ClassAcrossRooms(class_key=view.class_key, rooms=view.class_rooms)
        print("\n  ВНИМАНИЕ: уроки под этим ключом класса сняты разными камерами.")
        for sentence in across.warning_ru():
            print(f"    · {sentence}")

    # Counters this camera cannot produce at all print «не измерялось», here as on the HTML
    # pages. `выходил к доске 0` for a camera whose board is behind the lens is a confident
    # zero standing in for «не измеряли», and a terminal is not exempt from rule 3.
    voided = lessons_mod.voided_counters(
        [item for item in view.unmeasured
         if int(item.get("runs") or 0) >= view.runs_in_class])
    if voided:
        print()
        for key, why in voided.items():
            label = dict(weekly.COUNTER_KEYS).get(key, key)
            print(f"  НЕ ИЗМЕРЯЕТСЯ НА ЭТОЙ КАМЕРЕ  «{label}»")
            print(f"    {why}")
            print("    в строках ниже там стоит «не измерялось», а не ноль")
    print()

    # Requests are printed as WHAT + HOW beside each place, and the WHY once at the end of
    # the class. The full three-part block repeated for nine places is 135 lines of terminal
    # in which the four gates — the thing an operator actually scans for — stop being
    # findable, and an unreadable refusal is not an actionable one.
    reasons: dict[str, str] = {}

    for history in view.histories:
        trend_view = history.trend_view
        print(f"  {history.display_name_ru}  [{history.place['role']}]")
        print(f"    занятий {len(history.sessions)}, видимость от "
              f"{(history.coverage_min or 0):.0%}")
        if history.place["role"] != "pupil":
            print(f"    {trend_view.headline_ru}")
            print(f"      {trend_view.detail_ru}\n")
            continue
        for week in history.weeks:
            if not week.observed:
                print(f"    {week.label}  — уроков не загружено (это не ноль)")
                continue
            counters = ", ".join(
                f"{label} " + ("не измерялось" if key in voided
                               else str(week.counts.get(key)))
                for key, label in weekly.COUNTER_KEYS)
            index = ("—" if week.index_median is None else f"{week.index_median:.1f}")
            print(f"    {week.label}  занятий {week.sessions}, "
                  f"видимость {week.coverage_median:.0%}, индекс {index}; {counters}")
        print(f"    ДИНАМИКА: {trend_view.headline_ru}")
        print(f"      {trend_view.detail_ru}")
        for gate in trend_view.gates:
            print(f"      [{'x' if gate['passed'] else ' '}] {gate['gate']}: "
                  f"{gate['measured']}")
        # The four gates say WHICH precondition is missing. That is not the same thing as
        # knowing what to ask the school for, and the gap between them is where a refusal
        # stops being actionable — so the request list, with its numbers, prints too.
        if history.requests:
            print("      ЧТО ЗАПРОСИТЬ У ШКОЛЫ:")
            for number, request in enumerate(history.requests, start=1):
                print(f"        {number}. {request.what_ru}")
                print(f"           как: {request.how_ru}")
                reasons.setdefault(request.key, request.why_ru)
        else:
            print(f"      {weekly.NOTHING_TO_REQUEST_RU}")
        print()

    if reasons:
        print("ПОЧЕМУ ИМЕННО ЭТО (по одному разу на причину, а не по разу на место):")
        for key, why in reasons.items():
            print(f"  [{key}] {why}")
    return OK


def cmd_cabinet_lessons(args) -> int:
    """What is comparable ACROSS recordings — and the sentence saying what that is not.

    A separate verb rather than a section of `weekly`, because it answers a different
    question with a different subject. `weekly` is about places over time; this is about
    RECORDINGS, and the two must not share a screen without a heading between them.
    """
    from classvision.cabinet import lessons as lessons_mod
    from classvision.cabinet import store, weekly

    connection = store.connect(args.db)
    view = lessons_mod.comparable(connection)
    if args.json:
        print(json.dumps(view.to_dict(), ensure_ascii=False, indent=1, default=str))
        return OK

    if not view.lessons:
        print("в кабинете нет ни одного разобранного урока — сравнивать нечего "
              "(это отсутствие данных, а не ноль)")
        return OK

    print(lessons_mod.WHAT_THIS_IS_RU)
    print()
    print("ЧЕГО ЗДЕСЬ НЕТ: " + lessons_mod.WHAT_THIS_IS_NOT_RU)
    for entry in view.cross_room:
        print(f"\n  ВНИМАНИЕ ({entry.display_ru}):")
        for sentence in entry.warning_ru():
            print(f"    · {sentence}")
    if len(view.lessons) == 1:
        print("\n  В кабинете одна запись — сравнивать её не с чем. Ниже одна строка, и "
              "она описывает саму запись.")
    print()

    for lesson in view.lessons:
        voided = lessons_mod.voided_counters(lesson.unmeasured)
        window = ("—" if lesson.window_minutes is None
                  else f"{lesson.window_minutes:.0f} мин")
        print(f"  {lesson.date_local or 'без даты'}  комната {lesson.room_key}  "
              f"{lesson.week_label}  {window}")
        print(f"    ученических мест {len(lesson.pupil_places)}, видимость медиана "
              f"{'—' if lesson.coverage_median is None else f'{lesson.coverage_median:.0%}'}"
              f", минимум "
              f"{'—' if lesson.coverage_min is None else f'{lesson.coverage_min:.0%}'}"
              f", вне мест "
              f"{'—' if lesson.unassigned_share is None else f'{lesson.unassigned_share:.1%}'}")
        counters = ", ".join(
            f"{label} " + ("не измерялось" if key in voided
                           else str(lesson.counts.get(key)))
            for key, label in weekly.COUNTER_KEYS)
        print(f"    по всей комнате: {counters}")
        if lesson.board_conflict_ru:
            print(f"    ВНИМАНИЕ: {lesson.board_conflict_ru}")
        teacher = lesson.teacher
        if teacher is None or not teacher.get("available"):
            why = ((teacher or {}).get("reason_ru")
                   or lesson.adult_place_missing_reason_ru or "разбора взрослого нет")
            print(f"    взрослый: не измерен — {why} (это не ноль)")
        else:
            where = ", ".join(
                f"{state['label_ru']} " + ("не изм." if state["voided_why_ru"]
                                           else f"{state['share_of_lesson_percent']}%")
                for state in teacher["states"])
            print(f"    взрослый (доли ВСЕГО урока): {where}")
        for item in lesson.unmeasured:
            print(f"    НЕ ИЗМЕРЕНО  {item.get('what')}: {item.get('why')}")
        print()
    return OK


def cmd_cabinet_report(args) -> int:
    from classvision.cabinet import report, store

    connection = store.connect(args.db)
    result = report.render(connection, args.out, room_key=args.room,
                           class_key=args.class_name)
    print(f"{result['index']}\n  страниц {len(result['pages'])}, "
          f"классов {result['classes']}, состояние: {result['state']}")
    return OK


def cmd_cabinet_note(args) -> int:
    """Generate and store the orientation note of one run, or of every shown run.

    **This verb never fails a build, and that is a deliberate exit-code decision rather
    than an oversight.** Everything that can go wrong with a note — no key on this machine,
    a provider that timed out, a text the guard rejected — is a *stated absence*, and the
    cabinet renders completely without any note at all. Returning `1` («ран, но доверять
    нельзя») for a missing key would put a red line in a school's cron log about an optional
    aid, and the predictable response is to stop reading the log. So all of those exit `0`
    with a sentence saying what happened.

    The one thing that does exit `2` is a run id that is not in the database: that is a
    typo or a missing import, nothing was attempted, and a script that carried on would
    silently never generate the note it was asked for.
    """
    from classvision.cabinet import notes, store

    if bool(args.run_id) == bool(args.all):
        # Neither, or both. Both is the dangerous one: `note <run_id> --all` reads as
        # «этот прогон, среди всех» and would quietly spend a call per lesson in the
        # cabinet. Neither of them is guessed.
        print("укажите РОВНО одно: либо run_id одного прогона, либо --all по всем "
              "расчётным прогонам. Список прогонов: `classvision cabinet show`.",
              file=sys.stderr)
        return FAILED

    connection = store.connect(args.db)
    if args.all:
        outcomes = notes.generate_all(connection, backend=args.backend,
                                      room_key=args.room, class_key=args.class_name)
        if not outcomes:
            print("в базе нет ни одного расчётного прогона — записывать записку не к чему. "
                  "Сначала `cabinet import`.")
            return OK
    else:
        try:
            outcomes = [notes.generate_for_run(connection, args.run_id,
                                               backend=args.backend)]
        except store.Refusal as refusal:
            print(f"ОТКАЗ [{refusal.code}] {refusal.message_ru}", file=sys.stderr)
            return FAILED

    for outcome in outcomes:
        # «Отправлено» only when something was. The first version printed the byte count
        # the same way in both branches, so a run with no key reported «отправлено 2232
        # байт счётчиков» about a request that was never made — a false sentence about
        # data leaving a school, which is the single sentence in this command that has to
        # be exactly true.
        traffic = (f"отправлено {outcome.bundle_bytes} байт счётчиков "
                   f"(без кадров, без имён)" if outcome.bundle_was_sent
                   else f"НИЧЕГО НЕ ОТПРАВЛЕНО; сводка, которая ушла бы, — "
                        f"{outcome.bundle_bytes} байт счётчиков")
        print(f"{outcome.date_local or 'без даты'}  прогон {outcome.run_id}  "
              f"[{outcome.outcome}]  {traffic}")
        if outcome.model:
            print(f"    модель {outcome.model}")
        if outcome.available and outcome.text:
            print(f"    проверка: {'пройдена' if outcome.guard_passed else 'не проводилась'}")
            for line in outcome.text.splitlines():
                print(f"    | {line}")
        else:
            print(f"    записки нет: {outcome.reason_ru}")
        for note in outcome.notes_ru:
            print(f"    {note}")
    print("\nЗаписка — не измерение. Отчёт кабинета полон и без неё; показать её на "
          "странице класса: `classvision cabinet report`.")
    return OK


def cmd_cabinet_select_run(args) -> int:
    from classvision.cabinet import store

    connection = store.connect(args.db)
    moved = store.select_run(connection, args.run_id)
    print(f"урок №{moved['lesson_id']}: сводки теперь считаются по прогону "
          f"{moved['run_id']} (было: {moved['previous_run_id']}). Прежний прогон "
          f"сохранён и не удалён.")
    return OK


def cmd_cabinet_attest(args) -> int:
    from classvision.cabinet import store

    connection = store.connect(args.db)
    store.attest(connection, place_id=args.place_id, external_id=args.external_id,
                 full_name=args.full_name, attested_by=args.by, attested_at=args.at,
                 valid_from=args.valid_from, valid_to=args.valid_to,
                 decision_ref=args.decision, note=args.note)
    print(f"место {args.place_id} -> {args.external_id}, утвердил {args.by} ({args.at}), "
          f"основание {args.decision}")
    return OK


def cmd_cabinet_show(args) -> int:
    from classvision.cabinet import store

    connection = store.connect(args.db)
    if args.json:
        print(json.dumps(store.summary(connection), ensure_ascii=False, indent=1))
        return OK
    _print_store(store.summary(connection))
    for lesson in store.lessons(connection, room_key=args.room, class_key=args.class_name):
        runs = store.runs_for_lesson(connection, int(lesson["lesson_id"]))
        print(f"  урок №{lesson['lesson_id']}  {lesson['date_local'] or 'без даты'}  "
              f"{lesson['started_at'] or ''}  {(lesson['duration_seconds'] or 0)/60:.0f} мин"
              f"  часы={lesson['clock_source']}  измерений {len(runs)}"
              f"{'  ПРОДОЛЖЕНИЕ №' + str(lesson['continues_lesson_id']) if lesson['continues_lesson_id'] else ''}")
        for run in runs:
            mark = "*" if run["run_id"] == lesson["selected_run_id"] else " "
            print(f"     {mark} {run['run_id']}  пороги {run['thresholds_sha']}  "
                  f"мест {run['seats_found']}  кадров {run['analysed_frames']}")
        if lesson["overlap_note"]:
            print(f"       ! {lesson['overlap_note']}")
    for place in store.places(connection, room_key=args.room, class_key=args.class_name):
        names = store.attestations_for(connection, int(place["place_id"]))
        who = (f" -> {names[-1]['external_id']} (утвердил {names[-1]['attested_by']})"
               if names else " -> без имени")
        print(f"  место #{place['place_id']} {place['label_ru']} [{place['role']}] "
              f"({place['anchor_x']:.0f},{place['anchor_y']:.0f}){who}")
    return OK


def _print_store(summary: dict) -> None:
    print(f"в базе: уроков {summary['lessons']} (с датой {summary['lessons_dated']}, "
          f"занятий {summary['sessions']}), измерений {summary['runs']} "
          f"(не выбрано в сводки {summary['runs_not_selected']}), "
          f"мест {summary['places']} (учеников {summary['places_pupil']}, "
          f"подписано {summary['attested_places']}), "
          f"наборов порогов {summary['threshold_sets']}, "
          f"записок-ориентиров {summary['notes_available']} из {summary['notes']} "
          f"запрошенных")
    if summary["seat_lessons_unplaced"]:
        print(f"        {summary['seat_lessons_unplaced']} наблюдений без привязки к "
              f"месту — они сохранены, но не участвуют в накоплении")
    for item in summary["unmeasured"]:
        print(f"        НЕ ИЗМЕРЕНО  {item['what']}: {item['why']} "
              f"(прогонов: {item['runs']})")


def notes_backends() -> tuple[str, ...]:
    """What `cabinet note --backend` accepts, asked of the module that implements it.

    Imported here rather than listed here: a hard-coded copy of the list is how a CLI ends
    up offering a mode the code no longer has, or hiding one it grew.
    """
    from classvision.cabinet.notes import BACKENDS

    return BACKENDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classvision",
        description="Разбор записи урока: наблюдаемые факты по местам, без диагнозов.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse", help="разобрать запись и записать артефакт")
    analyse.add_argument("video")
    analyse.add_argument("--out", default=None)
    analyse.add_argument("--sample-fps", type=float, default=2.0,
                         help="сколько кадров в секунду разбирать (по умолчанию 2)")
    analyse.add_argument("--imgsz", type=int, default=1280)
    analyse.add_argument("--weights", default="yolo11m-pose.pt")
    analyse.add_argument("--device", default="mps")
    analyse.add_argument("--start", type=float, default=None)
    analyse.add_argument("--end", type=float, default=None)
    analyse.add_argument("--cache", default="classvision/.cache")
    analyse.add_argument("--seat-map", default=None,
                         help="подписанный план рассадки — ЕДИНСТВЕННЫЙ источник имён")
    analyse.add_argument("--roster", default=None)
    analyse.add_argument("--photos", default=None)
    analyse.add_argument("--class", dest="class_name", default=None)
    analyse.add_argument("--faces", action="store_true",
                         help="подтверждение по лицам (не создаёт имён; см. MEASUREMENTS §4)")
    analyse.add_argument("--room", default=None, metavar="FILE",
                         help="ФАЙЛ разметки комнаты (configs/camera_*.yaml): зона доски, "
                              "стол взрослого. Без него «у доски» не измеряется, а "
                              "взрослый определяется по размеру и помечается как "
                              "предположение. Не путать с `cabinet --room-key`: там "
                              "КЛЮЧ комнаты, строка, а не файл.")
    analyse.set_defaults(func=cmd_analyse)

    combined = sub.add_parser(
        "analyse-session",
        help="несколько файлов ОДНОГО урока -> один артефакт (проверяет непрерывность)")
    combined.add_argument("videos", nargs="+")
    combined.add_argument("--out", default=None)
    combined.add_argument("--sample-fps", type=float, default=2.0)
    combined.add_argument("--imgsz", type=int, default=1280)
    combined.add_argument("--weights", default="yolo11m-pose.pt")
    combined.add_argument("--device", default="mps")
    combined.add_argument("--cache", default="classvision/.cache")
    combined.add_argument("--seat-map", default=None,
                          help="подписанный план рассадки — ЕДИНСТВЕННЫЙ источник имён")
    combined.add_argument("--roster", default=None)
    combined.add_argument("--photos", default=None)
    combined.add_argument("--class", dest="class_name", default=None)
    combined.add_argument("--faces", action="store_true",
                          help="не поддерживается для склеенной сессии — команда откажет")
    combined.add_argument("--room", default=None, metavar="FILE",
                          help="файл разметки комнаты — один на всю сессию, потому что "
                               "склеиваются части ОДНОГО урока с одной камеры")
    combined.add_argument("--tolerance", type=float, default=None,
                          help="допуск на стыке в секундах (по умолчанию "
                               "session.MAX_SEAM_GAP_SECONDS = 2.0)")
    # `--start/--end` are deliberately NOT offered here: they cut ONE file, and on a session
    # they would silently take the same minute out of every part. `_settings` still reads
    # them, so they are pinned to "no cut" rather than left undefined.
    combined.set_defaults(func=cmd_analyse_session, start=None, end=None)

    zones = sub.add_parser(
        "zones",
        help="нарисовать разметку комнаты на реальном кадре и проверить её глазами")
    zones.add_argument("video")
    zones.add_argument("--room", required=True, metavar="FILE",
                       help="файл разметки, configs/camera_*.yaml")
    zones.add_argument("--at", type=float, required=True, metavar="SECONDS",
                       help="секунда записи. Берите кадр, в котором кто-то СТОИТ у доски: "
                            "разметку проверяет случай, который она должна поймать.")
    zones.add_argument("--out", default=None, metavar="FILE",
                       help="куда сохранить картинку; без него только текстовый разбор")
    zones.add_argument("--expect-in-board", type=int, default=None, metavar="N",
                       help="сколько человек ДОЛЖНО оказаться у доски в этом кадре. "
                            "Задавайте — тогда команда годится для регрессии, а не только "
                            "для взгляда.")
    zones.add_argument("--imgsz", type=int, default=1280)
    zones.add_argument("--weights", default="yolo11m-pose.pt")
    zones.add_argument("--device", default="mps")
    zones.set_defaults(func=cmd_zones)

    report = sub.add_parser("report", help="HTML-отчёт и русская сводка")
    report.add_argument("artefact")
    report.add_argument("--out", default="out/report")
    report.add_argument("--backend", default="auto",
                        help="auto | offline | gemini | openai (по умолчанию auto: без "
                             "ключа используется полноценный офлайн-генератор)")
    report.set_defaults(func=cmd_report)

    verify = sub.add_parser("verify", help="проверочное видео + сверка с отчётом")
    verify.add_argument("artefact")
    verify.add_argument("--video", default=None)
    verify.add_argument("--from", dest="start_seconds", type=float, default=0.0)
    verify.add_argument("--to", dest="end_seconds", type=float, default=60.0)
    verify.add_argument("--fps", type=float, default=8.0)
    verify.add_argument("--out", default=None,
                        help="если не задан, только сверка без рендера видео")
    verify.set_defaults(func=cmd_verify)

    template = sub.add_parser("seatmap-template",
                              help="заготовка плана рассадки по найденным местам")
    template.add_argument("artefact")
    template.add_argument("--camera", required=True)
    template.add_argument("--class", dest="class_name", default=None)
    template.add_argument("--out", default="seatmap.yaml")
    template.set_defaults(func=cmd_seatmap_template)

    check = sub.add_parser("roster-check", help="сверить реестр и фотографии")
    check.add_argument("roster")
    check.add_argument("--photos", default=None)
    check.add_argument("--class", dest="class_name", default=None)
    check.set_defaults(func=cmd_roster_check)

    # -- the cabinet: many lessons, accumulated -------------------------------------
    cabinet = sub.add_parser(
        "cabinet",
        help="накопление разборов по неделям (кабинет психолога)")
    cab = cabinet.add_subparsers(dest="cabinet_command", required=True)

    def _common(target) -> None:
        target.add_argument("--db", default="out/cabinet.sqlite3",
                            help="файл базы кабинета (по умолчанию out/cabinet.sqlite3)")
        # `--room-key`, not `--room`: elsewhere in this CLI `--room` is a profile FILE
        # (`analyse --room d14.yaml`). Two flags with one name and two meanings in one
        # program is a defect waiting for a tired operator, and the cheaper half to rename
        # is the one that takes a string rather than a path.
        target.add_argument("--room-key", dest="room", default=None, metavar="KEY",
                            help="ключ комнаты/камеры, по которому копится история. "
                                 "Артефакт этого НЕ знает: он разбирает файл. Без флага "
                                 "берётся камера из профиля комнаты, а если профиля не "
                                 "было — имя файла; источник записывается в базу.")
        target.add_argument("--class", dest="class_name", default=None,
                            help="класс. Тоже утверждение оператора, а не свойство записи.")

    imp = cab.add_parser("import", help="загрузить артефакт(ы) в кабинет")
    imp.add_argument("artefacts", nargs="+")
    _common(imp)
    imp.add_argument("--allow-unclocked", action="store_true",
                     help="загрузить запись без читаемого времени; она будет исключена "
                          "из всех недельных сводок")
    imp.add_argument("--allow-overlap", action="store_true",
                     help="загрузить запись, пересекающуюся по времени с уже "
                          "загруженной. Пометка останется в базе и в отчёте.")
    imp.add_argument("--select", action="store_true",
                     help="сделать этот прогон расчётным для урока. По умолчанию "
                          "повторный разбор с другими порогами сохраняется ОТДЕЛЬНО и "
                          "ничего не заменяет.")
    imp.set_defaults(func=cmd_cabinet_import)

    week = cab.add_parser("weekly", help="понедельные счётчики и динамика")
    _common(week)
    week.add_argument("--json", action="store_true")
    week.set_defaults(func=cmd_cabinet_weekly)

    across = cab.add_parser(
        "lessons",
        help="что сравнимо МЕЖДУ уроками: свойства самих записей, а не детей")
    across.add_argument("--db", default="out/cabinet.sqlite3")
    across.add_argument("--json", action="store_true")
    across.set_defaults(func=cmd_cabinet_lessons)

    rep = cab.add_parser("report", help="HTML-кабинет: обзор класса и страница на место")
    _common(rep)
    rep.add_argument("--out", default="out/cabinet")
    rep.set_defaults(func=cmd_cabinet_report)

    note = cab.add_parser(
        "note",
        help="машинная записка-ориентир по уроку: с чего начать чтение. Без величин.")
    note.add_argument("run_id", nargs="?", default=None,
                      help="прогон, о котором писать. Список — `cabinet show`.")
    _common(note)
    note.add_argument("--all", action="store_true",
                      help="по всем расчётным прогонам (по одному обращению к модели на "
                           "прогон). Повторный запуск не плодит записок: у прогона она "
                           "одна и она заменяется.")
    note.add_argument("--backend", default="auto", choices=list(notes_backends()),
                      help="auto — использовать ключ, если он есть в окружении; "
                           "deterministic — не обращаться к модели вовсе и записать это "
                           "решение. Без ключа команда не падает, а говорит об этом.")
    note.set_defaults(func=cmd_cabinet_note)

    show = cab.add_parser("show", help="что лежит в базе кабинета")
    _common(show)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_cabinet_show)

    pick = cab.add_parser("select-run",
                          help="выбрать, каким прогоном считается урок (единственный "
                               "способ сдвинуть этот указатель)")
    pick.add_argument("run_id")
    pick.add_argument("--db", default="out/cabinet.sqlite3")
    pick.set_defaults(func=cmd_cabinet_select_run)

    att = cab.add_parser("attest",
                         help="записать подписанное утверждение «на этом месте сидит X» "
                              "— единственный путь, которым имя попадает в кабинет")
    att.add_argument("--db", default="out/cabinet.sqlite3")
    att.add_argument("--place-id", type=int, required=True)
    att.add_argument("--external-id", required=True)
    att.add_argument("--full-name", default=None)
    att.add_argument("--by", required=True, help="кто утверждает (ФИО, должность)")
    att.add_argument("--at", required=True, help="дата утверждения, YYYY-MM-DD")
    att.add_argument("--valid-from", required=True)
    att.add_argument("--valid-to", default=None,
                     help="дата окончания. Без неё план не истекает и продолжит называть "
                          "прошлогодних детей — задавайте её.")
    att.add_argument("--decision", required=True,
                     help="ссылка на решение школы, разрешающее пофамильное накопление")
    att.add_argument("--note", default=None)
    att.set_defaults(func=cmd_cabinet_attest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"нет файла: {error}", file=sys.stderr)
        return FAILED
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return FAILED


if __name__ == "__main__":
    raise SystemExit(main())

"""Room layouts on disk: one small YAML per camera, loaded into `zones.RoomLayout`.

--------------------------------------------------------------------------------
**WHY A FILE AND NOT A CONSTANT.**

A zone is the only thing in this package that a *human* asserts about a room. Everything
else is measured off the footage and can be re-derived by re-running; a polygon cannot,
because the fact it encodes — «вот здесь доска, а вот здесь стол учителя» — is not in the
pixels in any way a model can be trusted to read. Two consequences follow, and this module
exists for both:

  * The assertion has to be **per camera and on disk**, next to the code, in a form a
    school's technician can open and correct. `configs/camera_d14.yaml` and
    `configs/camera_01.yaml` are shipped together deliberately: one camera has the board
    in frame and one does not, and the pair is the documentation of that difference. A
    constant buried in Python would have been copied to the second camera unchanged.

  * The assertion has to be **versioned into the artefact**, which it is: `pipeline`
    writes the polygons into `provenance.room.layout`, `run_id` hashes them, so a lesson
    re-analysed after somebody nudged a zone is a NEW artefact rather than a quiet
    restatement of the old one.

--------------------------------------------------------------------------------
**WHAT THIS MODULE REFUSES, AND WHY EACH REFUSAL IS A REAL DEFECT AND NOT TIDINESS.**

Every check below exists because the failure it prevents is SILENT. A room layout is
loaded once, at the top of a ten-minute run, and then quietly produces numbers about
children for a term. There is no second chance to notice.

  1. **Unknown keys are refused.** `teacher_zone:` instead of `teacher_desk_zone:` does
     not raise anything by itself — it produces a layout with no teacher zone, so
     `identify_adult` falls back to guessing by scale and the artefact says
     `inferred_by_scale, needs_confirmation: true`. Which looks exactly like a camera
     nobody has configured. A typo must not be indistinguishable from a decision.

  2. **`board_zone` must be PRESENT, and may be `null`.** Omitting it and writing `null`
     are different statements: «мы про доску не думали» and «доски в кадре нет». The
     second is a measured fact about camera 01 and belongs in the file; the first is a
     gap. `null` is accepted and produces `AT_BOARD` never firing, with `unmeasured`
     carrying the reason — which is the correct behaviour and must be chosen on purpose.

  3. **Out-of-frame vertices are refused, and the frame size is declared in the file.**
     A polygon drawn on a 1920x1080 preview and pasted into a 2560x1440 camera's profile
     is entirely inside the frame, entirely wrong, and fires on nothing — or worse, fires
     on the wrong desk. Declaring `frame:` lets `check_frame()` compare it with the video
     the run actually opened, so the mismatch is caught at the top of the run.

  4. **Degenerate polygons are refused** (fewer than three vertices, or zero area). A
     three-point polygon whose points are collinear contains nothing, forever, without
     error.

  5. **`board_zone` and `teacher_desk_zone` may not contain each other's vertices.** They
     answer contradictory questions — "this person is at the board" and "this seat is the
     adult's own desk" — and a room where one polygon overlaps the other has one of them
     drawn wrong.

  6. **A non-null `board_zone` REQUIRES `board_surface`, and the two must overlap.** This
     one is a trap guard, and it is worth spelling out because the trap is easy and the
     symptom is nothing at all. `zones.py` tests a person's SHOULDER ANCHOR. A person
     standing at the D14 board has their shoulder line at y 202..340; the chalk surface is
     at y 137..313; the floor their feet are on is at y 470..620. All three are plausible
     rectangles to draw, and only the first is the zone. Drawing the FLOOR strip — the
     intuitive answer, and the one this module was first specified with, and the one
     `GROUND_TRUTH_D14.md` asserted in prose until this rule was tightened — yields a
     polygon no shoulder line ever enters, so «у доски» is zero for the life of the
     install and nothing in the artefact says why. `lesson.unmeasured` does not catch it
     either: a zone IS configured, so the artefact prints «у доски: 0.0 мин» with healthy
     coverage counters, which is this project's signature defect — «не смогли измерить»
     rendered as a confident zero.

     `board_surface` is therefore **mandatory whenever a board zone is drawn**, and not
     merely honoured when present. The first version of this module checked the overlap
     only `if board_surface is not None`, which made the guard opt-in — and the operator
     who draws the floor strip is exactly the operator who would not think to declare the
     chalk rectangle, so the guard was absent precisely in the case it exists for. Two
     lines of YAML is the price; a term of zeroes was the alternative.

     `board_surface` is never tested against a person; it exists to be the thing the zone
     is checked against, and to let the verification render draw the board.

--------------------------------------------------------------------------------
**THE YAML NAMES THE FURNITURE; THE DATACLASS NAMES THE ROLE.**

The file says `teacher_desk_zone` because that is what a technician standing in the room
is looking at — the desk with the laptop. `RoomLayout` says `teacher_zone` because the
code's question is "whose seat is the adult's". Exactly one spelling is accepted in YAML,
and the other is refused by rule 1 rather than accepted as an alias: an alias means two
files in the same school can disagree about which key is real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classvision.room.zones import Point, Polygon, RoomLayout

# Every key the file may carry. A frozenset rather than a list of `if`s so that rule 1
# reports ALL the unknown keys at once -- a technician fixing a config by trial and error,
# one refusal per attempt, is a technician who gives up and deletes the file.
KEYS = frozenset({
    "camera", "frame", "board_surface", "board_zone", "teacher_desk_zone",
    "door_zone", "excluded_seats", "zones_confirmed_by", "notes",
})

# Keys that must appear, even if their value is null. `board_zone` is here for rule 2:
# «доски в кадре нет» is a finding and has to be written down, not left out.
REQUIRED = ("camera", "frame", "board_zone")

ZONE_KEYS = ("board_surface", "board_zone", "teacher_desk_zone", "door_zone")

# A polygon whose |shoelace area| is under this many square pixels is treated as
# degenerate. CHOSEN and deliberately tiny: at 2560x1440 the smallest zone anyone would
# draw on purpose -- a doorway seen edge-on -- is still ~65 x 260 px, four orders of
# magnitude above this. The bar exists to catch collinear points and duplicated vertices,
# not to have an opinion about how small a real zone may be.
MIN_POLYGON_AREA_PX2 = 100.0


class LayoutError(ValueError):
    """A room layout that must not be used. Always names the file and the offending key.

    A subclass of ValueError rather than a bare one so a caller can catch exactly this,
    and so the CLI can exit `2` (could not run) rather than `1` (ran, do not trust it) --
    a layout that will not load has produced no numbers at all, which is a different
    situation from numbers that came out wrong.
    """


def load(path: str | Path) -> RoomLayout:
    """Read one camera profile. Every failure is a `LayoutError` naming the file."""
    import yaml

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LayoutError(f"нет файла разметки комнаты: {path}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise LayoutError(f"{path}: не читается как YAML: {error}") from None
    return from_mapping(data, source=str(path))


def from_mapping(data: Any, *, source: str = "<memory>") -> RoomLayout:
    """The validator proper, separated from the file so the tests need no temp files."""
    if not isinstance(data, dict):
        raise LayoutError(f"{source}: ожидался словарь на верхнем уровне, получено "
                          f"{type(data).__name__}")

    unknown = sorted(set(data) - KEYS)
    if unknown:
        raise LayoutError(
            f"{source}: неизвестные ключи {unknown}. Допустимы только "
            f"{sorted(KEYS)}. Ключ с опечаткой не ошибка сам по себе — он молча даёт "
            f"разметку без этой зоны, и отчёт нельзя отличить от отчёта по "
            f"ненастроенной камере.")

    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise LayoutError(
            f"{source}: обязательные ключи отсутствуют: {missing}. `board_zone` обязателен "
            f"и может быть null — «доски в кадре нет» это установленный факт о камере, а "
            f"не пропуск.")

    camera = data["camera"]
    if not isinstance(camera, str) or not camera.strip():
        raise LayoutError(f"{source}: `camera` должно быть непустой строкой")

    width, height = _frame(data["frame"], source)

    polygons: dict[str, Polygon | None] = {}
    for key in ZONE_KEYS:
        polygons[key] = _polygon(data.get(key), key, width, height, source)

    board_zone = polygons["board_zone"]
    surface = polygons["board_surface"]
    teacher = polygons["teacher_desk_zone"]

    if surface is not None and board_zone is None:
        raise LayoutError(
            f"{source}: задан `board_surface`, но `board_zone` — null. Доска в кадре есть, "
            f"а зона не нарисована: это либо незаконченная разметка, либо решение, которое "
            f"надо записать словами в `notes` и убрать `board_surface`.")

    # The other half of rule 6, and the half that was missing. Without it the overlap check
    # below is OPT-IN: whoever draws the floor strip in front of the board is exactly
    # whoever would not think to declare the chalk rectangle, so the guard was absent in
    # the one case it exists for, and the result is «у доски: 0.0 мин» for the life of the
    # install with `lesson.unmeasured` silent -- a zone IS configured, so the artefact has
    # nothing to complain about.
    if board_zone is not None and surface is None:
        raise LayoutError(
            f"{source}: нарисована `board_zone`, но не задан `board_surface` — сам "
            f"прямоугольник доски. Он обязателен, потому что это единственное, чем "
            f"проверяется, что зона нарисована ПО ДОСКЕ, а не по полосе пола перед ней. "
            f"Зона проверяется по ЛИНИИ ПЛЕЧ: у стоящего у доски плечи оказываются НА "
            f"доске, а у полосы пола не бывает ни одного попадания — и «у доски» тихо "
            f"остаётся нулём всё время работы, причём `lesson.unmeasured` про это молчит, "
            f"потому что зона формально задана. Впишите координаты доски в "
            f"`board_surface` (две строки), либо поставьте `board_zone: null`, если доски "
            f"в кадре нет.")

    if (surface is not None and board_zone is not None
            and not _boxes_overlap(surface, board_zone)):
        raise LayoutError(
            f"{source}: `board_zone` нигде не пересекается с `board_surface`. Зона "
            f"проверяется по ЛИНИИ ПЛЕЧ человека, а не по его ступням: у стоящего у "
            f"доски плечи оказываются НА доске, а не на полу перед ней. Полоса пола "
            f"перед доской — самая частая ошибка здесь, и она не даёт ни одного "
            f"события «у доски» за всё время работы, молча.")

    if board_zone is not None and teacher is not None and _touch(board_zone, teacher):
        raise LayoutError(
            f"{source}: `board_zone` и `teacher_desk_zone` пересекаются. Они отвечают на "
            f"противоречащие вопросы («человек у доски» и «это место взрослого»), и точка "
            f"внутри обеих делает разбор недетерминированным.")

    return RoomLayout(
        camera=camera.strip(),
        frame_width=width,
        frame_height=height,
        board_zone=board_zone,
        teacher_zone=teacher,
        door_zone=polygons["door_zone"],
        excluded_seats=_excluded(data.get("excluded_seats"), source),
        zones_confirmed_by=_text(data.get("zones_confirmed_by"), "zones_confirmed_by",
                                 source),
        notes=_text(data.get("notes"), "notes", source),
    )


def check_frame(layout: RoomLayout, width: int, height: int) -> None:
    """Refuse a layout drawn for a different frame size than the video actually has.

    Called with the numbers `decode.probe` read off the file, not with the numbers the
    layout claims. A profile drawn on a 1920x1080 preview and applied to a 2560x1440
    recording is entirely inside the frame and entirely wrong: every zone sits a quarter
    of the way up and to the left of where it was drawn, which on D14 moves the board zone
    onto the front row of desks. Nothing downstream can notice.
    """
    if (layout.frame_width, layout.frame_height) != (width, height):
        raise LayoutError(
            f"разметка «{layout.camera}» нарисована для кадра "
            f"{layout.frame_width}x{layout.frame_height}, а запись — {width}x{height}. "
            f"Координаты зон в пикселях, они не переносятся между размерами кадра.")


def to_mapping(layout: RoomLayout, *, board_surface: Polygon | None = None) -> dict:
    """`RoomLayout` back to the file's shape, for the round-trip tests.

    It used to say "and for `zones-template`". There is no `zones-template` verb and there
    never was; a docstring naming a command that does not exist sends the reader to a
    traceback. The round trip is the only consumer, and it earns its keep: it is what pins
    that everything the ANALYSIS reads survives a write and a read.

    `board_surface` is passed in rather than read off the layout because `RoomLayout` does
    not carry it: it is a load-time check and a drawing aid, not something any measurement
    consumes, and adding an unused field to the dataclass would invite somebody to consume
    it. The round-trip property that matters is that everything the ANALYSIS reads
    survives, and that is what `tests/test_room_layout.py` pins.

    It is REQUIRED here whenever the layout has a board zone, and this function refuses
    rather than emitting a mapping without it. Since rule 6 became mandatory, a mapping
    with a board zone and no surface is a mapping `from_mapping` will not accept, and a
    writer that can emit a file its own reader rejects is a writer that will one day
    silently produce an unusable profile on a school's disk.
    """
    if layout.board_zone is not None and board_surface is None:
        raise LayoutError(
            f"разметка «{layout.camera}»: `board_zone` задана, поэтому `board_surface` "
            f"обязателен и должен быть передан сюда — иначе получится отображение, "
            f"которое `from_mapping` откажется прочитать обратно (правило 6).")
    out: dict[str, Any] = {
        "camera": layout.camera,
        "frame": {"width": layout.frame_width, "height": layout.frame_height},
    }
    if board_surface is not None:
        out["board_surface"] = [list(p) for p in board_surface]
    out["board_zone"] = None if layout.board_zone is None else [
        list(p) for p in layout.board_zone]
    if layout.teacher_zone is not None:
        out["teacher_desk_zone"] = [list(p) for p in layout.teacher_zone]
    if layout.door_zone is not None:
        out["door_zone"] = [list(p) for p in layout.door_zone]
    if layout.excluded_seats:
        out["excluded_seats"] = list(layout.excluded_seats)
    if layout.zones_confirmed_by:
        out["zones_confirmed_by"] = layout.zones_confirmed_by
    if layout.notes:
        out["notes"] = layout.notes
    return out


def board_surface_of(path: str | Path) -> Polygon | None:
    """The chalk rectangle from a profile, for the verification render. Never a zone.

    Kept as a separate reader, and named so that autocompleting it in the pipeline looks
    wrong, because the whole point of rule 6 is that this polygon is not the one people
    are tested against.

    The frame size comes through `_frame()`, the same validator `load()` uses, and NOT
    through a default. The first version wrote `frame.get("width", 10 ** 6)`, which meant
    that a profile with a missing or malformed `frame:` got a notional million-pixel frame
    here and its surface passed the out-of-frame check that `load()` would have refused —
    the same class of defect as the rest of this module, in the one function that exists
    to support the check.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    if "frame" not in data:
        raise LayoutError(f"{path}: нет ключа `frame` — размер кадра, в котором рисовались "
                          f"координаты, обязателен")
    width, height = _frame(data["frame"], str(path))
    return _polygon(data.get("board_surface"), "board_surface", width, height, str(path))


# -- the pieces --------------------------------------------------------------------


def _frame(value: Any, source: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise LayoutError(f"{source}: `frame` должен быть {{width: …, height: …}}")
    try:
        width, height = int(value["width"]), int(value["height"])
    except (TypeError, ValueError):
        raise LayoutError(f"{source}: `frame.width` и `frame.height` должны быть целыми "
                          f"числами") from None
    if width <= 0 or height <= 0:
        raise LayoutError(f"{source}: размер кадра должен быть положительным, получено "
                          f"{width}x{height}")
    return width, height


def _polygon(value: Any, key: str, width: int, height: int,
             source: str) -> Polygon | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 3:
        raise LayoutError(f"{source}: `{key}` должен быть списком из трёх и более вершин "
                          f"[x, y] либо null; получено {value!r}")

    points: list[Point] = []
    for index, vertex in enumerate(value):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise LayoutError(f"{source}: `{key}` вершина {index} — не пара [x, y]: "
                              f"{vertex!r}")
        try:
            x, y = float(vertex[0]), float(vertex[1])
        except (TypeError, ValueError):
            raise LayoutError(f"{source}: `{key}` вершина {index} не число: "
                              f"{vertex!r}") from None
        # Refused, not clamped. Clamping would move the polygon somewhere nobody drew and
        # then run the whole lesson against it.
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise LayoutError(
                f"{source}: `{key}` вершина {index} = ({x:g}, {y:g}) вне кадра "
                f"{width}x{height}. Скорее всего разметка рисовалась по кадру другого "
                f"размера — координаты зон в пикселях и не переносятся.")
        points.append((x, y))

    area = abs(_shoelace(points))
    if area < MIN_POLYGON_AREA_PX2:
        raise LayoutError(
            f"{source}: `{key}` вырожден — площадь {area:.1f} px² при минимуме "
            f"{MIN_POLYGON_AREA_PX2:g}. Такой многоугольник не содержит ни одной точки "
            f"и молчит всё время работы.")
    return tuple(points)


def _shoelace(points: list[Point]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _boxes_overlap(a: Polygon, b: Polygon) -> bool:
    """Bounding boxes, on purpose: rule 6 is a sanity check, not a geometry engine.

    An exact polygon intersection would refuse a zone that legitimately hangs a little
    below the board for short pupils, which is the D14 case (shoulder lines run to y 331
    against a board bottom of y 313). The question being asked is «эта зона вообще про эту
    доску?», and bounding boxes answer it.
    """
    ax1, ay1, ax2, ay2 = _bounds(a)
    bx1, by1, bx2, by2 = _bounds(b)
    return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2


def _touch(a: Polygon, b: Polygon) -> bool:
    """Does either polygon hold a vertex of the other? Cheap overlap test for rule 5."""
    from classvision.room.zones import point_in_polygon

    return (any(point_in_polygon(p, b) for p in a)
            or any(point_in_polygon(p, a) for p in b))


def _excluded(value: Any, source: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise LayoutError(f"{source}: `excluded_seats` должен быть списком номеров мест")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise LayoutError(f"{source}: `excluded_seats` — номер места должен быть целым "
                              f"числом ≥ 1, получено {item!r}")
        out.append(item)
    return tuple(out)


def _text(value: Any, key: str, source: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LayoutError(f"{source}: `{key}` должен быть строкой")
    return value.strip()

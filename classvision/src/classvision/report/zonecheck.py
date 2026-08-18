"""Draw a room profile over a real frame and say, per person, which zone they fell in.

**Why this is a shipped verb and not a scratch script.** A zone is the one input to this
package that nobody can check by reading it: `[[1024, 140], [1645, 140], …]` is a list of
pixel pairs, and the difference between a correct profile and one that never fires is
invisible in the text. It is completely obvious in a picture. So the picture is part of
the product: `classvision zones --room … --video … --at …` renders the polygons over the
frame the operator names, runs the same pose model the analysis runs, and prints for every
person found whether their shoulder line landed in the board zone, the teacher's desk
zone, or nothing.

**The frame to check is the one where something is happening.** On D14 that is t = 1764 s
of `D14_20260815103136.mp4` (11:01:00): a pupil is standing at the board while five other
people are seated, one of whom is the adult who has moved to sit AMONG the pupils. A
profile that gets that frame right is right about the case that matters — «стоит у доски»
versus «сидит не на своём месте» — and a profile that colours all six the same is wrong in
a way no unit test would have caught.

**The render draws the foot point too, in a different colour, and that is deliberate.**
The zone test uses the SHOULDER ANCHOR (`room/zones.py` states why), but the foot point is
the intuitive choice and the one a reader will assume. Showing both, and showing that the
board zone sits nowhere near the feet, is how the picture teaches the convention instead
of merely obeying it.

**There are FIVE verdicts, not three, and the two extra ones are the whole point.** «У
доски», «стол взрослого» and «дверь» are the easy cases. The other two are both flavours of
*not measured*, and each one has already been printed as a measurement somewhere in this
project's history:

  * **«линия плеч не найдена — не проверен»** — the pose model gave no confident shoulders,
    so this person was tested against **no zone at all**. 15 observations in 43 275 on D14.
    Printing «вне зон» here would say we looked and found them nowhere.
  * **«у этой камеры не задано ни одной зоны»** — there was nothing to test against. On
    `camera_01` the board hangs behind the lens and `board_zone` is a measured `null`, so
    `in_board` is zero in every frame that will ever exist. The verb used to read that zero
    as an empty frame and warn «разметка неверна» about the one profile in the repository
    whose null was arrived at by measurement.

`check()` therefore returns `board_zone_configured` alongside `in_board`: a caller that sees
only the count cannot distinguish «никого у доски» from «доски нет», and the CLI's exit code
depends on which it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from classvision.geometry import Keypoints, anchor, shoulder_width
from classvision.room import layout_io
from classvision.room.zones import RoomLayout

# BGR, because OpenCV. Chosen for a printed page as much as a screen: the two zones must
# stay distinguishable in greyscale, so one is light and one is dark rather than two hues.
BOARD_COLOUR = (60, 220, 60)        # board zone — green, like the board
SURFACE_COLOUR = (200, 200, 200)    # the chalk surface — grey outline, never a zone
TEACHER_COLOUR = (40, 150, 255)     # teacher's desk — orange
DOOR_COLOUR = (200, 120, 200)       # door — violet
ANCHOR_COLOUR = (0, 0, 255)         # the shoulder line: what is actually tested
FOOT_COLOUR = (255, 200, 0)         # the foot point: what a reader assumes is tested
ZONE_ALPHA = 0.22                   # CHOSEN: dark enough to read, light enough to see the
                                    # furniture underneath, which is the whole point


@dataclass(slots=True)
class PersonCheck:
    """One person in the checked frame, and where the profile put them."""

    box: tuple[float, float, float, float]
    anchor: tuple[float, float] | None
    foot: tuple[float, float]
    shoulder_px: float | None
    score: float
    in_board: bool
    in_teacher: bool
    in_door: bool
    # Which zones this profile actually has. Carried per person rather than inferred from
    # the three booleans above, because all three being False has two causes -- «проверен
    # против зон и ни в одну не попал» and «проверять было не против чего» -- and printing
    # one word for both is the same defect as printing «вне зон» for a person with no
    # shoulder line.
    zones_configured: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        """For the terminal, in Russian, like every other user-facing string here."""
        if self.anchor is None:
            # Not "outside every zone". A person whose shoulders the model could not find
            # was not tested at all, and the two must never print the same word -- the
            # doorway on D14 produces exactly this, and «вне зон» there would read as
            # «никто не входил».
            return "линия плеч не найдена — не проверен"
        if not self.zones_configured:
            return "у этой камеры не задано ни одной зоны — проверять не против чего"
        for flag, name in ((self.in_board, "у доски"),
                           (self.in_teacher, "стол взрослого"),
                           (self.in_door, "дверь")):
            if flag:
                return name
        return "вне зон"

    @property
    def tag(self) -> str:
        """For the drawn image, in ASCII, and the reason is a limitation not a preference.

        OpenCV's built-in Hershey fonts carry no Cyrillic glyphs: `putText` renders
        «у доски» as a row of `?`. The alternatives are shipping a TTF (a dependency, and
        a licensing question, for a caption) or drawing through PIL against a system font
        path that differs on every machine a school owns. So the picture is labelled in
        ASCII and the terminal — where the operator actually reads the result — is in
        Russian. The two say the same thing in the same order.
        """
        if self.anchor is None:
            return "NO SHOULDER LINE - not tested"
        if not self.zones_configured:
            return "no zones in this profile"
        for flag, name in ((self.in_board, "AT BOARD"),
                           (self.in_teacher, "TEACHER DESK"),
                           (self.in_door, "DOOR")):
            if flag:
                return name
        return "no zone"


def check(video: str | Path, room: str | Path, *, at_seconds: float,
          out_path: str | Path | None = None, weights: str = "yolo11m-pose.pt",
          device: str = "mps", imgsz: int = 1280, conf: float = 0.30) -> dict[str, Any]:
    """Render one frame with the profile drawn on it, and report every person on it.

    Returns a plain dict so the CLI can print it and a test can assert on it without a
    picture. The picture is for the human; the dict is what a regression test pins.
    """
    layout = layout_io.load(room)
    surface = layout_io.board_surface_of(room)

    frame = _frame_at(video, at_seconds)
    height, width = frame.shape[:2]
    # The same refusal the analysis makes, made here too and made FIRST: a profile drawn
    # for another frame size is wrong in a way the picture would hide, because the
    # polygons would still land somewhere plausible.
    layout_io.check_frame(layout, width, height)

    people = _people(frame, weights=weights, device=device, imgsz=imgsz, conf=conf)
    checks = [_check_one(person, layout) for person in people]

    if out_path is not None:
        image = _draw(frame, layout, surface, checks)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return {
        "camera": layout.camera,
        "video": str(video),
        "at_seconds": at_seconds,
        "frame": [width, height],
        "image": None if out_path is None else str(out_path),
        "people": len(checks),
        # WITHOUT this flag `in_board: 0` says two different things and the caller cannot
        # tell them apart: «зона задана, у доски никого» and «зоны доски у этой камеры нет
        # вовсе». The first is a fact about the frame and the commonest symptom of a
        # mis-drawn polygon; the second is a measured property of where the camera hangs
        # (camera_01) and can never be anything but zero. The CLI read the zero as the
        # first and told the operator «разметка неверна» about a profile that is right.
        # That is this project's signature defect: «не измеряется» rendered as a finding.
        "board_zone_configured": layout.board_zone is not None,
        "teacher_zone_configured": layout.teacher_zone is not None,
        "in_board": sum(1 for c in checks if c.in_board),
        "in_teacher": sum(1 for c in checks if c.in_teacher),
        "in_door": sum(1 for c in checks if c.in_door),
        "no_anchor": sum(1 for c in checks if c.anchor is None),
        "detail": [
            {"box": [round(v, 1) for v in c.box],
             "anchor": None if c.anchor is None else [round(v, 1) for v in c.anchor],
             "foot": [round(v, 1) for v in c.foot],
             "shoulder_px": None if c.shoulder_px is None else round(c.shoulder_px, 1),
             "score": round(c.score, 2),
             "verdict": c.verdict}
            for c in checks
        ],
    }


def _configured(layout: RoomLayout) -> tuple[str, ...]:
    return tuple(name for name, polygon in (("board_zone", layout.board_zone),
                                            ("teacher_desk_zone", layout.teacher_zone),
                                            ("door_zone", layout.door_zone))
                 if polygon is not None)


def _check_one(person, layout: RoomLayout) -> PersonCheck:
    keypoints = Keypoints(xy=person.keypoints.xy, conf=person.keypoints.conf)
    position = anchor(keypoints)
    return PersonCheck(
        box=person.box,
        anchor=position,
        foot=person.foot_point,
        shoulder_px=shoulder_width(keypoints),
        score=person.score,
        # `layout.contains` returns False for a None zone, which is what camera_01 needs:
        # a camera with no board must report "вне зон", never crash and never claim.
        in_board=position is not None and layout.contains(layout.board_zone, position),
        in_teacher=position is not None and layout.contains(layout.teacher_zone, position),
        in_door=position is not None and layout.contains(layout.door_zone, position),
        zones_configured=_configured(layout),
    )


def _frame_at(video: str | Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"не открывается видео: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"нет кадра на {seconds} с в {video}")
    return frame


def _people(frame: np.ndarray, *, weights: str, device: str, imgsz: int, conf: float):
    from classvision.vision.pose import PoseModel

    model = PoseModel(weights=weights, device=device, imgsz=imgsz, conf=conf)
    # `track=False`: one frame has no history, and asking for a tracker id here would
    # start tracker state that nothing resets.
    return model.look(frame, 0, 0.0, track=False).people


def _draw(frame: np.ndarray, layout: RoomLayout, surface, checks: list[PersonCheck]):
    image = frame.copy()
    overlay = image.copy()

    for polygon, colour in ((layout.board_zone, BOARD_COLOUR),
                            (layout.teacher_zone, TEACHER_COLOUR),
                            (layout.door_zone, DOOR_COLOUR)):
        if polygon is None:
            continue
        points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [points], colour)
    cv2.addWeighted(overlay, ZONE_ALPHA, image, 1.0 - ZONE_ALPHA, 0.0, image)

    for polygon, colour, label in ((layout.board_zone, BOARD_COLOUR, "board_zone"),
                                   (layout.teacher_zone, TEACHER_COLOUR,
                                    "teacher_desk_zone"),
                                   (layout.door_zone, DOOR_COLOUR, "door_zone")):
        if polygon is None:
            continue
        points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
        cv2.polylines(image, [points], True, colour, 3)
        top = min(polygon, key=lambda p: (p[1], p[0]))
        _label(image, label, (int(top[0]) + 6, int(top[1]) - 10), colour)

    if surface is not None:
        points = np.array([[int(x), int(y)] for x, y in surface], dtype=np.int32)
        cv2.polylines(image, [points], True, SURFACE_COLOUR, 2, cv2.LINE_AA)
        _label(image, "board_surface (NOT a zone)",
               (int(points[:, 0].min()) + 6, int(points[:, 1].min()) + 26),
               SURFACE_COLOUR)

    for person in checks:
        x1, y1, x2, y2 = (int(v) for v in person.box)
        colour = BOARD_COLOUR if person.in_board else (
            TEACHER_COLOUR if person.in_teacher else (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        # The foot point first and small, so it is visibly NOT the thing being tested.
        fx, fy = (int(v) for v in person.foot)
        cv2.drawMarker(image, (fx, fy), FOOT_COLOUR, cv2.MARKER_TILTED_CROSS, 22, 2)
        if person.anchor is not None:
            ax, ay = (int(v) for v in person.anchor)
            cv2.circle(image, (ax, ay), 9, ANCHOR_COLOUR, -1)
            cv2.circle(image, (ax, ay), 9, (255, 255, 255), 2)
            cv2.line(image, (ax, ay), (fx, fy), (120, 120, 120), 1, cv2.LINE_AA)
        _label(image, person.tag, (x1, max(y1 - 12, 22)), colour)

    # Anchored to the bottom of whatever frame this is, not to 1440: a profile is checked
    # on the camera it was drawn for, and the next camera will not be this size.
    bottom = image.shape[0]
    _label(image, "DOT   = shoulder line: this is what the zone test uses",
           (30, bottom - 110), ANCHOR_COLOUR)
    _label(image, "CROSS = box bottom (foot point): NOT tested",
           (30, bottom - 70), FOOT_COLOUR)
    _label(image, f"camera {layout.camera}", (30, bottom - 30), (255, 255, 255))
    return image


def _label(image, text: str, origin, colour) -> None:
    # Drawn twice: black underneath, so the caption survives both the green board and the
    # pale floor. A caption that disappears on half the frames is a caption nobody trusts.
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 5,
                cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_COMPLEX, 0.8, colour, 2,
                cv2.LINE_AA)

"""An annotated video: the analysis drawn back onto the footage it came from.

**This is the only deliverable that can falsify the whole pipeline.** Every other output
is numbers, and numbers agree with each other whether or not they are right. A chart
showing «место 6 отвернулось 23 раза» is equally convincing if seat 6 is a pupil, if it is
two pupils merged, or if the head-direction rule is firing on a hair style. Watching the
labels move over the actual bodies is the one check that catches all three, and it is how
the two real defects in this project were found: the adult being counted as a pupil
raising his hand, and the seat clustering collapsing nine places into three.

So this is not a demo. It is a test instrument, and it is deliberately unflattering:

  * **`UNKNOWN` is drawn**, in grey, as loudly as any other state. A viewer must be able
    to see how much of the lesson the system could not read. An overlay that renders only
    confident states is a highlight reel of the frames that worked.
  * **The seat circle is drawn even when nobody is in it**, so an empty place is visible
    as an empty place rather than as an absence of evidence.
  * **Observations at no seat are drawn too**, in red, because those are the ones that
    entered no counter. On this footage they are 4 % of everything.
  * **The adult's seat is coloured differently and labelled as the adult**, since the
    single most consequential thing a viewer can check in ten seconds is whether the
    person excluded from every pupil statistic is in fact the teacher.

The frames are re-decoded rather than cached from the analysis pass: the artefact holds
numbers, not pixels, and a 52-minute file at 2 fps would be 6 000 frames of 2560x1440 in
memory. Re-decoding a 60-second window costs about a second.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from classvision.geometry import Keypoints, anchor, shoulder_width
from classvision.states import RU_LABELS, PupilState, Thresholds, classify, read

# BGR. Chosen to stay distinguishable in greyscale (differing lightness) and for the two
# commonest colour-vision deficiencies: the states that matter most are separated by
# lightness rather than by hue alone.
STATE_COLOUR: dict[PupilState, tuple[int, int, int]] = {
    PupilState.SEATED: (150, 200, 120),
    PupilState.HAND_RAISED: (60, 200, 255),
    PupilState.STOOD_UP: (255, 180, 60),
    PupilState.AWAY_FROM_PLACE: (230, 120, 220),
    PupilState.AT_BOARD: (255, 120, 120),
    PupilState.HEAD_DOWN: (90, 110, 240),
    PupilState.TURNED_AWAY: (200, 160, 255),
    PupilState.UNKNOWN: (140, 140, 140),
}
ADULT_COLOUR = (60, 170, 255)
NO_SEAT_COLOUR = (70, 70, 240)


@dataclass(frozen=True, slots=True)
class Window:
    start_seconds: float
    end_seconds: float


def _text(image, string: str, origin, colour, scale: float = 0.8, thickness: int = 2):
    """Text with a dark outline, because this footage is a bright classroom in daylight
    and thin coloured text on a white wall is unreadable in exactly the frames a reviewer
    most wants to read."""
    cv2.putText(image, string, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(image, string, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def _detections_for(artefact: dict[str, Any], video: str | Path, settings):
    """Pass one for whatever this artefact was measured from — one file, or a session.

    An assembled artefact's times are session seconds, so re-running pass one on a single
    file would compare the report against a timeline that is 13 minutes short and report
    every seat as mismatched. The merge is re-done here from the parts the artefact itself
    records, which also means this check re-proves the contiguity claim rather than trusting
    the block that asserts it.
    """
    from classvision.pipeline import detect_pass

    block = (artefact.get("provenance") or {}).get("session")
    if not block:
        return detect_pass(Path(video), settings)

    from classvision import session as session_mod

    rebuilt = session_mod.resolve([part["path"] for part in block["parts"]],
                                  tolerance_seconds=block["tolerance_seconds"])
    if rebuilt.digest != block["digest"]:
        raise ValueError(
            "the files this artefact names no longer assemble into the session it "
            f"describes (digest {rebuilt.digest[:12]}… against {block['digest'][:12]}…). "
            "Something was renamed, replaced or re-cut; verifying against it would be "
            "checking one lesson's report against another lesson's footage.")
    detections, _ = session_mod.detect_session(rebuilt, settings)
    return detections


def _part_offset(artefact: dict[str, Any], video: str | Path) -> float:
    """Where the file being DRAWN sits on the artefact's timeline.

    Zero for an ordinary artefact. For an assembled one the states are stamped in session
    seconds while the video decoder counts from its own first frame, and without this the
    overlay would draw part two's footage against part one's labels — an off-by-thirteen-
    minutes error that looks like a completely broken analysis rather than like a bug here.
    """
    block = (artefact.get("provenance") or {}).get("session")
    if not block:
        return 0.0
    name = Path(video).name
    for part in block["parts"]:
        if Path(part["path"]).name == name:
            return float(part["offset_seconds"])
    raise ValueError(
        f"{name} is not one of this session's files ("
        + ", ".join(Path(p["path"]).name for p in block["parts"])
        + "). Render one of the parts: the overlay draws one file at a time, and the "
          "labels are placed on the session's clock.")


def render(video: str | Path, artefact_path: str | Path, out_path: str | Path, *,
           window: Window, detections=None, settings=None,
           fps_out: float = 10.0, width_out: int = 1600) -> Path:
    """Draw the analysis over a window of the footage.

    `detections` and `settings` are the pipeline's own, so the overlay shows exactly what
    the analysis saw — not a second, subtly different inference. Passing a freshly-run
    model here would produce a video that agrees with itself and not with the report,
    which is the one thing a verification artefact must never do.
    """
    from classvision.pipeline import Settings
    from classvision.room.seats import Anchor, assign
    from classvision.states import Baselines

    video = Path(video)
    artefact = json.loads(Path(artefact_path).read_text(encoding="utf-8"))
    settings = settings or Settings(sample_fps=float(artefact["provenance"]["sample_fps"]))
    thresholds = Thresholds(**{k: v for k, v in artefact["provenance"]["thresholds"].items()
                               if k in Thresholds.__slots__})
    if detections is None:
        detections = _detections_for(artefact, video, settings)
    # Session seconds of this file's first frame; 0.0 unless the artefact was assembled.
    part_offset = _part_offset(artefact, video)

    seats = artefact["seats"]
    adult_seat = artefact["lesson"].get("adult_seat")
    teacher = artefact.get("teacher") or {}
    centres = {s["seat_id"]: tuple(s["centre"]) for s in seats}
    scales = {s["seat_id"]: s["scale_px"] for s in seats}
    names = {}
    for s in seats:
        pupil = s.get("pupil") or {}
        names[s["seat_id"]] = (pupil.get("full_name") if pupil.get("established")
                               else f"место {s['seat_id']}")
    if adult_seat is not None and teacher.get("centre"):
        centres[adult_seat] = tuple(teacher["centre"])
        scales[adult_seat] = teacher.get("scale_px") or 200.0
        names[adult_seat] = "ВЗРОСЛЫЙ"
    elif adult_seat is not None:
        # An artefact written before `TeacherRecord` carried its centre. Draw nothing
        # rather than a circle at the origin, which is what the earlier version did and
        # which reads as "the adult's place is the top-left corner of the room".
        names[adult_seat] = "ВЗРОСЛЫЙ"

    # Rebuild the seat objects the assigner needs, from the artefact rather than by
    # re-clustering: re-clustering here could disagree with the report.
    class _S:
        __slots__ = ("centre", "scale", "seat_id")

        def __init__(self, seat_id, centre, scale):
            self.seat_id, self.centre, self.scale = seat_id, centre, scale

    seat_objects = [_S(sid, centres[sid], scales.get(sid, 90.0)) for sid in sorted(centres)]

    # Baselines must be warmed from the START of the lesson, not from the start of the
    # window: a window beginning at minute 30 would otherwise show every pupil as UNKNOWN
    # for its first ten seconds while the baselines re-settle, which is an artefact of the
    # overlay and would be read as an artefact of the analysis.
    # THE WINDOW MUST BE THE ARTEFACT'S. The pipeline classifies only frames inside the
    # detected lesson window; feeding this loop the whole file gave every seat a different
    # baseline and therefore different labels, so the overlay showed «встал» for a seat the
    # report counts zero stands on. A verification artefact that disagrees with the thing
    # it verifies is worse than none — it manufactures doubt about correct numbers and
    # confidence about wrong ones. Same window, same order, same thresholds.
    lesson_from, lesson_to = artefact["lesson"]["window_seconds"]

    baselines: dict[int, Baselines] = {}
    by_frame: dict[float, list[int]] = {}
    for i in range(len(detections.times)):
        t = float(detections.times[i])
        if lesson_from <= t <= lesson_to:
            by_frame.setdefault(t, []).append(i)

    frame_times = sorted(by_frame)
    states: dict[float, dict[int, tuple[PupilState, int]]] = {}
    for t in frame_times:
        rows = by_frame[t]
        anchors, keep = [], []
        for i in rows:
            person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
            scale = shoulder_width(person)
            position = anchor(person)
            if scale is None or position is None:
                continue
            anchors.append(Anchor(t, position, scale, None))
            keep.append(i)
        seat_ids = assign(anchors, seat_objects) if anchors else []
        frame_states: dict[int, tuple[PupilState, int]] = {}
        for position, seat_id in enumerate(seat_ids):
            i = keep[position]
            person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
            reading = read(person, t, thresholds)
            if seat_id is None:
                frame_states[-1 - i] = (PupilState.UNKNOWN, i)
                continue
            base = baselines.setdefault(seat_id, Baselines())
            base.observe(reading, thresholds)
            frame_states[seat_id] = (classify(reading, base, thresholds), i)
        if window.start_seconds <= t <= window.end_seconds:
            states[t] = frame_states

    # -- draw -----------------------------------------------------------------------
    step = max(int(round(20.0 / fps_out)), 1)   # source is 20 fps
    tmp = Path(out_path).with_suffix(".frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()

    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    written = 0
    index = 0
    analysed = np.array(frame_times, dtype=float)
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            seconds = index / source_fps + part_offset
            if seconds > window.end_seconds:
                break
            if seconds >= window.start_seconds and index % step == 0:
                ok, image = capture.retrieve()
                if ok:
                    nearest = float(analysed[int(np.argmin(np.abs(analysed - seconds)))])
                    _draw(image, states.get(nearest, {}), detections, centres, scales,
                          names, adult_seat, seconds, artefact)
                    scale_out = width_out / image.shape[1]
                    small = cv2.resize(image, (width_out, int(image.shape[0] * scale_out)))
                    cv2.imwrite(str(tmp / f"f_{written:06d}.jpg"), small,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    written += 1
            index += 1
    finally:
        capture.release()

    out_path = Path(out_path)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps_out),
         "-i", str(tmp / "f_%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out_path)], check=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()
    tmp.rmdir()
    return out_path


def agrees_with_artefact(video: str | Path, artefact_path: str | Path,
                         detections=None, settings=None) -> dict[str, Any]:
    """Re-derive the per-seat state histogram the way the overlay does, and compare.

    **Without this, the overlay is a claim rather than a check.** It draws states it
    computes itself, and the first version computed them over a different frame window
    than the pipeline, so it confidently labelled a seat «встал» that the report counts
    zero stands on. Nothing in the picture said the two disagreed — a viewer would simply
    have concluded the report was wrong.

    So the agreement is asserted, not assumed: this replays the same classification and
    checks the per-state observation counts against `ledger.state_observations` in the
    artefact. Any drift between the overlay and the pipeline shows up here as a number,
    and `ok` is False.
    """
    from classvision.pipeline import Settings
    from classvision.room.seats import Anchor, assign
    from classvision.states import Baselines

    artefact = json.loads(Path(artefact_path).read_text(encoding="utf-8"))
    settings = settings or Settings(sample_fps=float(artefact["provenance"]["sample_fps"]))
    thresholds = Thresholds(**{k: v for k, v in artefact["provenance"]["thresholds"].items()
                               if k in Thresholds.__slots__})
    if detections is None:
        detections = _detections_for(artefact, video, settings)

    class _S:
        __slots__ = ("centre", "scale", "seat_id")

        def __init__(self, seat_id, centre, scale):
            self.seat_id, self.centre, self.scale = seat_id, centre, scale

    places = [_S(s["seat_id"], tuple(s["centre"]), s["scale_px"]) for s in artefact["seats"]]
    teacher = artefact.get("teacher") or {}
    if teacher.get("seat_id") is not None and teacher.get("centre"):
        places.append(_S(teacher["seat_id"], tuple(teacher["centre"]),
                         teacher.get("scale_px") or 200.0))

    lesson_from, lesson_to = artefact["lesson"]["window_seconds"]
    by_frame: dict[float, list[int]] = {}
    for i in range(len(detections.times)):
        t = float(detections.times[i])
        if lesson_from <= t <= lesson_to:
            by_frame.setdefault(t, []).append(i)

    baselines: dict[int, Baselines] = {}
    counts: dict[int, dict[str, int]] = {}
    for t in sorted(by_frame):
        anchors, keep = [], []
        for i in by_frame[t]:
            person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
            scale = shoulder_width(person)
            position = anchor(person)
            if scale is None or position is None:
                continue
            anchors.append(Anchor(t, position, scale, None))
            keep.append(i)
        for position, seat_id in enumerate(assign(anchors, places) if anchors else []):
            if seat_id is None:
                continue
            i = keep[position]
            reading = read(Keypoints(xy=detections.xy[i], conf=detections.conf[i]),
                           t, thresholds)
            base = baselines.setdefault(seat_id, Baselines())
            base.observe(reading, thresholds)
            state = classify(reading, base, thresholds)
            counts.setdefault(seat_id, {})[state.value] = (
                counts.setdefault(seat_id, {}).get(state.value, 0) + 1)

    mismatches = []
    for seat in artefact["seats"]:
        expected = seat["ledger"]["state_observations"]
        got = counts.get(seat["seat_id"], {})
        for state in set(expected) | set(got):
            if expected.get(state, 0) != got.get(state, 0):
                mismatches.append({"seat_id": seat["seat_id"], "state": state,
                                   "artefact": expected.get(state, 0),
                                   "overlay": got.get(state, 0)})
    return {"ok": not mismatches, "seats_checked": len(artefact["seats"]),
            "mismatches": mismatches[:20], "mismatch_count": len(mismatches)}


def _draw(image, frame_states, detections, centres, scales, names, adult_seat,
          seconds: float, artefact: dict[str, Any]) -> None:
    for seat_id, centre in centres.items():
        radius = int(1.4 * scales.get(seat_id, 90.0))
        colour = ADULT_COLOUR if seat_id == adult_seat else (120, 120, 120)
        cv2.circle(image, (int(centre[0]), int(centre[1])), radius, colour, 2)

    for key, (state, i) in frame_states.items():
        box = detections.box[i]
        x1, y1, x2, y2 = (int(v) for v in box)
        if key < 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), NO_SEAT_COLOUR, 3)
            _text(image, "вне мест", (x1, max(y1 - 10, 24)), NO_SEAT_COLOUR, 0.7, 2)
            continue
        is_adult = key == adult_seat
        colour = ADULT_COLOUR if is_adult else STATE_COLOUR.get(state, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
        label = names.get(key, f"место {key}")
        _text(image, label, (x1, max(y1 - 40, 30)), colour, 0.8, 2)
        _text(image, ("исключён из статистики" if is_adult else RU_LABELS.get(state, "")),
              (x1, max(y1 - 12, 54)), colour, 0.7, 2)

    started = (artefact.get("lesson", {}).get("window_wall") or [None])[0]
    stamp = ""
    if started:
        from datetime import datetime, timedelta
        base = datetime.fromisoformat(started)
        offset = seconds - float(artefact["lesson"]["window_seconds"][0])
        stamp = (base + timedelta(seconds=offset)).strftime("%H:%M:%S")
    _text(image, f"{stamp}   t={seconds:.1f}s", (30, image.shape[0] - 90), (255, 255, 255), 1.1, 3)
    _text(image, "проверочная разметка: серый = поза не читалась, красный = вне мест",
          (30, image.shape[0] - 40), (200, 200, 200), 0.75, 2)

"""What a clip is, according to its own name.

`eval run --camera` picked ONE camera config for an entire run. The corpus is 344
`hall_right` clips and 299 `hall_left` ones, and `hall_left` carries a `mirror_ignore`
zone over a reflective column that **does not exist in hall_right's field of view**.
Evaluating one camera's footage against the other's zones silently blanks out part of the
frame -- and a silently blanked frame produces a lower recall number that looks exactly
like a tuning result.

So the camera comes from the clip. The school's recorder names every machine-made clip:

    hall_left_main_1009_1019_20260702_144150_952947.mp4           <- the crop (ROI)
    hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4  <- the full frame
    stairs_floor2_196_322_burst101_20260518_141523_230173.mp4     <- no `_main` at all

    <camera>[_main]_<track_a>_<track_b>[_burst<n>]_<YYYYMMDD>_<HHMMSS>_<micros>.<ext>

That is also the join between the two views of one incident: same camera, same track-ID
pair, seconds apart. 342 of the 381 incident-keys in the corpus (90%) have a crop partner.

**The CONFIG says what a camera is, not a regex.** The `_main` segment is optional -- 17
clips have none -- so "the camera is everything before the digits" is the obvious rule and
it is WRONG. `stairs_floor2` is a PREFIX of `stairs_floor2_second`, which is not a camera
at all, and that rule would quietly file five `stairs_floor2_second` clips under
`stairs_floor2`: a different camera, a different staircase, different zones. So a name is
matched against the configured camera names, LONGEST first (`stairs_floor2_aux` before
`stairs_floor2`), and the segment right after the match must be `main` or a track id.
Anything else is a name whose camera the config cannot confirm, and it raises.

A name this does not fit is a HARD ERROR, never a default: 3 human-named clips, and the 5
`stairs_floor2_second` ones. One of the human-named three is the only confirmed fight in
the whole corpus, so "unparsable" cannot mean "unscoreable": `eval/labels.csv` may state
the camera explicitly for a clip, and that human-provided value wins over inference (see
`camera_for`'s `explicit` parameter and `evaluation/labels.py`'s schema comment). That is
how a human resolves every clip named here -- by naming its camera, not by guessing it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from qorgan.config.camera import BullyingCamera, CameraConfig

# What follows the camera name. `main` is optional: both shapes are in the corpus.
CLIP_TAIL = re.compile(
    r"^(?:main_)?"
    r"(?P<track_a>\d+)_(?P<track_b>\d+)"
    r"(?:_burst(?P<burst>\d+))?"
    r"_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micros>\d{6})"
    r"\.[A-Za-z0-9]+$"
)

EXPECTED = (
    "<camera>[_main]_<track_a>_<track_b>[_burstNNN]_<YYYYMMDD>_<HHMMSS>_<micros>.<ext>, "
    "where <camera> is one of the names in config/cameras/"
)


class ClipNameError(Exception):
    """This clip's camera cannot be proved from its name. An error, never a default."""


@dataclass(frozen=True, slots=True)
class ClipName:
    """One clip, taken apart."""

    filename: str
    camera: str
    track_a: int
    track_b: int
    recorded_at: datetime
    is_burst: bool

    @property
    def pair(self) -> tuple[int, int]:
        """The track-ID pair. Half of the crop <-> full-frame join key."""
        return (self.track_a, self.track_b)


def parse_clip_name(filename: str, cameras: Iterable[str]) -> ClipName:
    """Take a clip's name apart, or refuse.

    `cameras` is the configured camera names -- the source of truth for what a camera is.
    A name is never invented from the shape of the string alone.
    """
    camera, tail = _camera_prefix(filename, cameras)
    match = CLIP_TAIL.match(tail)
    if match is None:
        raise _uninferable(filename)
    return ClipName(
        filename=filename,
        camera=camera,
        track_a=int(match["track_a"]),
        track_b=int(match["track_b"]),
        recorded_at=datetime.strptime(
            f"{match['date']}{match['time']}{match['micros']}", "%Y%m%d%H%M%S%f"
        ),
        is_burst=match["burst"] is not None,
    )


def _camera_prefix(filename: str, cameras: Iterable[str]) -> tuple[str, str]:
    """The configured camera this name starts with, and everything after it.

    LONGEST name first, because one camera's name can be a prefix of another's:
    `stairs_floor2` prefixes the configured `stairs_floor2_aux` AND the un-configured
    `stairs_floor2_second`. The segment after the match settles it -- `main` or a track id
    means the camera name ended there; anything else means the name belongs to some other
    camera, and this parser does not know which.
    """
    for camera in sorted(cameras, key=len, reverse=True):
        if not filename.startswith(f"{camera}_"):
            continue
        tail = filename[len(camera) + 1 :]
        segment = tail.split("_", 1)[0]
        if segment != "main" and not segment.isdigit():
            raise ClipNameError(
                f"{filename}: the name starts with the configured camera {camera!r}, but "
                f"what follows is {segment!r} -- neither `main` nor a track id. "
                f"{camera}_{segment!r} is not a configured camera, and calling this clip "
                f"{camera!r} would score it against another camera's zones, which "
                "silently blanks part of the frame and returns a plausible, wrong number. "
                "Name its camera explicitly in eval/labels.csv's `camera` column."
            )
        return camera, tail
    raise _uninferable(filename)


def _uninferable(filename: str) -> ClipNameError:
    return ClipNameError(
        f"cannot infer the camera from {filename!r}. Expected {EXPECTED}. "
        "Scoring a clip against another camera's zones silently blanks part of the "
        "frame, so an un-inferable name is an error and not a default."
    )


def camera_for(
    filename: str,
    cameras: Mapping[str, CameraConfig],
    explicit: str | None = None,
) -> BullyingCamera:
    """The camera whose zones and thresholds this clip must be scored against.

    `explicit` is a human-provided camera name -- typically `labels.csv`'s optional
    `camera` column. It WINS over inference, unconditionally: a human stating which
    camera the footage came from is better evidence than a filename parser, and it is
    the only way to score the three human-named clips in the corpus, one of which is the
    only confirmed fight there is. When `explicit` is absent (None or empty), the camera
    is inferred from `filename` as before -- and an un-inferable name is still a hard
    error, never a default.
    """
    if explicit:
        return _resolve(explicit, filename, cameras, source="labels.csv's camera column")
    name = parse_clip_name(filename, cameras).camera
    return _resolve(name, filename, cameras, source="the filename")


def _resolve(
    name: str, filename: str, cameras: Mapping[str, CameraConfig], source: str
) -> BullyingCamera:
    camera = cameras.get(name)
    if camera is None:
        raise ClipNameError(
            f"{filename}: {source} names camera {name!r}, which is not in config/cameras/. "
            f"Known: {', '.join(sorted(cameras))}"
        )
    if not isinstance(camera, BullyingCamera):
        raise ClipNameError(f"{filename}: {name!r} is not a bullying camera")
    return camera

"""A clip knows which camera it came from. Ask the clip.

`eval run --camera` picked ONE camera config for a whole run, and the corpus is 344
hall_right clips and 299 hall_left ones. hall_left carries a mirror_ignore zone over a
reflective column that is not in hall_right's field of view, so a global flag blanks part
of the frame for whichever half of the corpus it is wrong about -- silently, and the only
symptom is a recall number that looks like a tuning result.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import ClipNameError, camera_for, parse_clip_name

CROP = "hall_left_main_1009_1019_20260702_144150_952947.mp4"
BURST = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
RIGHT = "hall_right_main_212_233_burst101_20260702_101530_101010.mp4"

# The recorder does not always write a `_main` segment. 17 of the 663 clips in the corpus
# go straight from the camera name to the track-ID pair.
STAIRS = "stairs_floor2_196_322_burst101_20260518_141523_230173.mp4"
# ...and `stairs_floor2` is a PREFIX of `stairs_floor2_second`, which is NOT a configured
# camera. This is the clip that must never come back as `stairs_floor2`.
PREFIX_TRAP = "stairs_floor2_second_8_9_burst101_20260515_131458_435514.mp4"
AUX = "stairs_floor2_aux_main_5_6_burst101_20260518_141523_230173.mp4"


def _names() -> list[str]:
    """The configured camera names. The CONFIG says what a camera is, not a regex."""
    return list(load_cameras())


def test_the_camera_comes_from_the_filename() -> None:
    assert parse_clip_name(CROP, _names()).camera == "hall_left"
    assert parse_clip_name(RIGHT, _names()).camera == "hall_right"


def test_a_clip_with_no_main_segment_is_still_attributed_to_its_camera() -> None:
    """17 clips go camera -> track pair with no `_main` between them. Requiring `_main`
    made `_scan_plan` raise on every one, which took the whole scan down with it: nothing
    could be calibrated at all."""
    stairs = parse_clip_name(STAIRS, _names())

    assert stairs.camera == "stairs_floor2"
    assert stairs.pair == (196, 322)
    assert stairs.is_burst


def test_a_camera_name_that_merely_PREFIXES_this_one_is_never_guessed() -> None:
    """`stairs_floor2` is a prefix of `stairs_floor2_second`. A parser that took
    "everything before the digits" would hand these five clips to stairs_floor2 -- a
    DIFFERENT camera, with different zones -- and score them against a frame they were
    never shot in. That is the silent-wrong-camera failure this module exists to prevent,
    so it raises and a human resolves it in labels.csv."""
    with pytest.raises(ClipNameError, match="not a configured camera") as excinfo:
        parse_clip_name(PREFIX_TRAP, _names())

    assert PREFIX_TRAP in str(excinfo.value), "the error must name the file"
    assert "stairs_floor2_second" in str(excinfo.value)


def test_the_prefix_trap_does_not_resolve_to_a_camera_through_camera_for_either() -> None:
    """The end that matters: no zone set is ever handed back for this clip."""
    with pytest.raises(ClipNameError):
        camera_for(PREFIX_TRAP, load_cameras())


def test_the_longest_configured_camera_name_wins() -> None:
    """`stairs_floor2` is a prefix of `stairs_floor2_aux`, and both are real cameras."""
    assert parse_clip_name(AUX, _names()).camera == "stairs_floor2_aux"
    assert camera_for(AUX, load_cameras()).name == "stairs_floor2_aux"


def test_a_crop_and_its_burst_name_the_same_incident() -> None:
    """The join that halves the labelling time: same camera, same track pair, seconds
    apart. The crop was cut out of the burst."""
    crop, burst = parse_clip_name(CROP, _names()), parse_clip_name(BURST, _names())

    assert crop.camera == burst.camera
    assert crop.pair == burst.pair == (1009, 1019)
    assert not crop.is_burst
    assert burst.is_burst
    assert abs((burst.recorded_at - crop.recorded_at).total_seconds()) < 30


@pytest.mark.parametrize(
    "filename",
    ["IMG_2201.mp4", "драка_в_коридоре.mp4", "hall_left.mp4", "hall_left_main_1009.mp4"],
)
def test_an_uninferable_name_is_a_HARD_ERROR(filename: str) -> None:
    """Not a default. A clip scored against the wrong camera's zones is a lie that looks
    like a measurement, and the three human-named clips are precisely the ones whose
    camera nobody can prove."""
    with pytest.raises(ClipNameError, match="cannot infer the camera"):
        parse_clip_name(filename, _names())


def test_a_filename_naming_no_configured_camera_at_all_is_a_hard_error() -> None:
    """`basement` is not in config/cameras/, so no clip can be attributed to it. The
    config is the source of truth for what a camera IS."""
    with pytest.raises(ClipNameError, match="cannot infer the camera"):
        camera_for("basement_main_1_2_20260702_144150_952947.mp4", load_cameras())


def test_an_explicit_camera_that_is_not_in_the_config_is_a_hard_error() -> None:
    """The other door in: a typo in labels.csv's `camera` column."""
    with pytest.raises(ClipNameError, match="not in config/cameras"):
        camera_for(CROP, load_cameras(), explicit="basement")


def test_an_explicit_camera_wins_over_the_filename() -> None:
    """A human stating the camera is better evidence than the parser. If precedence were
    reversed this would resolve to hall_left (from CROP's own name) and the assertion
    below would fail."""
    cameras = load_cameras()

    resolved = camera_for(CROP, cameras, explicit="hall_right")

    assert resolved.name == "hall_right"
    assert resolved.name != parse_clip_name(CROP, _names()).camera


def test_a_human_named_clip_resolves_when_the_camera_is_explicit() -> None:
    """The whole point: the three human-named clips -- one of them the only confirmed
    fight in the corpus -- cannot be attributed by filename, but an explicit camera lets
    them resolve instead of raising."""
    cyrillic = "1.2 - нет буллинга.mp4"
    cameras = load_cameras()

    with pytest.raises(ClipNameError):
        camera_for(cyrillic, cameras)  # no explicit camera: still a hard error

    resolved = camera_for(cyrillic, cameras, explicit="hall_left")
    assert resolved.name == "hall_left"


def test_an_unknown_explicit_camera_is_a_hard_error_naming_the_file_and_the_value() -> None:
    """A typo in an explicit camera must not degrade into a guess."""
    cyrillic = "1.2 - подозрение на буллинг.mp4"

    with pytest.raises(ClipNameError, match=re.escape(cyrillic)) as excinfo:
        camera_for(cyrillic, load_cameras(), explicit="basement")

    assert "basement" in str(excinfo.value)
    assert "not in config/cameras" in str(excinfo.value)


def test_the_real_hall_configs_resolve_and_do_not_share_their_zones() -> None:
    """WHY this exists. If the two halls had the same zones, a global --camera flag would
    have been harmless and none of this would be worth a module."""
    cameras = load_cameras()
    left = camera_for(CROP, cameras)
    right = camera_for(RIGHT, cameras)

    assert left.name == "hall_left"
    assert right.name == "hall_right"
    assert left.bullying.zones.mirror_ignore != right.bullying.zones.mirror_ignore
    assert left.bullying.zones.normal_flow != right.bullying.zones.normal_flow


class RecordingModel:
    """Stands in for a YOLO. Records what it was asked to do, and does nothing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def track(self, _frame, **kwargs):
        self.calls.append(kwargs)
        return []


def test_the_video_source_tracks_on_the_configured_device() -> None:
    """One line, and it matters when 663 clips go through it.

    Without `device=`, Ultralytics picks its own default -- so the harness could score the
    whole corpus on the CPU, slowly, while production runs on the GPU. Same weights, but
    not the same measurement of anything that depends on wall-clock throughput, and a
    silent CPU fallback is exactly the kind of difference between bench and field this
    whole section exists to close.
    """
    import numpy as np

    from qorgan.config.loader import load_cameras
    from qorgan.evaluation.video import VideoSource

    model = RecordingModel()
    source = VideoSource(
        Path(CROP), camera_for(CROP, load_cameras()), device="cuda:1", model=model
    )
    source._detect(0.0, np.zeros((1440, 2560, 3), dtype=np.uint8))

    assert model.calls, "the model was never asked to track anything"
    assert model.calls[0]["device"] == "cuda:1", "Ultralytics chose the device, not us"

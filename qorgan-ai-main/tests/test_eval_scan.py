"""`qorgan eval scan`: what did the detector fire on, and when?

Watching 97 minutes of corridor to find the few seconds that matter is not a good use of
a human. The detector runs at threshold 0 over every clip and writes down every moment it
would have raised anything at all; the human then watches only those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qorgan.config.bullying import BullyingConfig
from qorgan.config.loader import load_cameras
from qorgan.evaluation import run, scanning
from qorgan.evaluation.clips import ClipNameError, camera_for
from qorgan.evaluation.scan import (
    NotAFullFrameError,
    ScanRow,
    coverage_path,
    load_candidates,
    load_coverage,
    refuse_crops,
    rows_for,
    write_candidates,
    write_coverage,
)
from qorgan.evaluation.scanning import CLIP_SUFFIXES, _clips_in, _scan_plan, cmd_scan
from tests.test_detection_pipeline import scene_an_assault, scene_two_children_talking
from tests.test_evaluation import SpyPose, SyntheticClip

CLIP = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
RIGHT = "hall_right_main_212_233_burst101_20260702_101530_101010.mp4"
# One of the three human-named clips. The recorder never touched it, so its camera is not
# in its name -- and one of these three is the only confirmed fight in the whole corpus.
CYRILLIC = "1.2 - подозрение на буллинг.mp4"
# No `_main`. 17 of the 663 clips are named this way, and every one of them used to take
# the whole scan down.
STAIRS = "stairs_floor2_196_322_burst101_20260518_141523_230173.mp4"
# A full frame with NO burst marker. 17 of these exist too -- which is exactly why the
# marker cannot be what tells a full frame from a crop.
NO_BURST = "hall_left_main_223_246_20260514_140739_409208.mp4"

FULL_FRAME = (2560, 1440)  # every one of the 663 clips, measured
A_CROP = (320, 450)  # the crops run 76-320 wide


@pytest.fixture(autouse=True)
def _clips_are_full_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clip files in these tests are empty: nothing here decodes a frame, so the
    resolution probe is stubbed. Tests that are ABOUT the probe override it."""
    monkeypatch.setattr(scanning, "_frame_size", lambda path: FULL_FRAME)


def test_a_scan_writes_one_row_per_incident_the_detector_fired_on() -> None:
    result = run(SyntheticClip(CLIP, scene_an_assault()), BullyingConfig(), pose=SpyPose())
    rows = rows_for(result)

    assert rows, "the detector found the assault but scan reported nothing"
    assert all(row.clip == CLIP for row in rows)
    assert all(row.score > 0 and 0.0 <= row.probability <= 1.0 for row in rows)
    assert all(0.0 <= row.confidence <= 1.0 for row in rows)


def test_a_merged_alert_is_not_a_second_row() -> None:
    """One physical incident, one thing for a human to watch. Forty rows for one scuffle
    would waste the labeller's time on the same four seconds forty times."""
    result = run(
        SyntheticClip(CLIP, scene_an_assault(frames=60)), BullyingConfig(), pose=SpyPose()
    )

    assert len(rows_for(result)) == 1


def test_a_quiet_clip_produces_no_candidates() -> None:
    result = run(SyntheticClip(CLIP, scene_two_children_talking()), BullyingConfig())

    assert rows_for(result) == []


def test_the_row_carries_the_raw_score_the_detector_fired_on_not_just_the_confidence() -> None:
    """WHY `Alert` grew a `score`: `Verdict` does not carry one.

    A human reading candidates.csv needs to see what the detector actually measured, not
    only how sure it ended up after the skeleton tier had its say. If `score` were quietly
    filled from the confidence, this would catch it: the two are different numbers, and
    the score is not bounded by 1.0.
    """
    result = run(SyntheticClip(CLIP, scene_an_assault()), BullyingConfig(), pose=SpyPose())
    alert = next(a for a in result.alerts if not a.merged)
    row = rows_for(result)[0]

    assert row.score == pytest.approx(round(alert.score, 3))
    assert row.score != pytest.approx(row.confidence)
    assert row.probability == pytest.approx(round(alert.verdict.candidate_probability, 3))


def test_candidates_round_trip(tmp_path: Path) -> None:
    rows = [ScanRow(CLIP, 2.4, 1.83, 0.71, 0.66), ScanRow(CLIP, 9.1, 2.20, 0.90, 0.88)]
    path = tmp_path / "candidates.csv"

    assert write_candidates(rows, path) == 2
    assert load_candidates(path) == rows


def test_loading_a_candidates_file_that_does_not_exist_is_an_empty_list(tmp_path: Path) -> None:
    """`eval label` is resumable, and resuming from nothing is a normal state."""
    assert load_candidates(tmp_path / "nope.csv") == []


# -- the corpus: which camera's zones does each clip get scored against? ----


def _clips(tmp_path: Path, *names: str) -> Path:
    """A clips directory. The files are empty: nothing here decodes a frame."""
    directory = tmp_path / "clips"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def _labels(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(f"video_id,t_start,t_end,label,camera\n{body}", encoding="utf-8")
    return path


def test_the_clip_list_is_the_video_files_and_nothing_else(tmp_path: Path) -> None:
    directory = _clips(tmp_path, CLIP, RIGHT, "candidates.csv", "notes.txt", "thumb.jpg")

    assert [p.name for p in _clips_in(directory)] == sorted([CLIP, RIGHT])
    assert {p.suffix for p in _clips_in(directory)} <= CLIP_SUFFIXES


def test_an_empty_clips_directory_is_an_error_not_an_empty_scan(tmp_path: Path) -> None:
    """Zero candidates from zero clips reads exactly like zero candidates from 663."""
    with pytest.raises(SystemExit, match="Nothing to scan"):
        _clips_in(_clips(tmp_path))


def test_the_camera_comes_from_the_clip_the_same_way_it_does_for_eval_run(tmp_path: Path) -> None:
    plan = _scan_plan(_clips(tmp_path, CLIP, RIGHT), tmp_path / "no-labels.csv")

    assert [(clip.name, camera.name) for clip, camera in plan] == [
        (CLIP, "hall_left"),
        (RIGHT, "hall_right"),
    ]


def test_a_human_named_clip_is_scanned_as_the_camera_the_labels_file_names(
    tmp_path: Path,
) -> None:
    """The only confirmed fight in the corpus is one of the three clips whose camera is
    not in its name. If scan inferred the camera and nothing else, it could not scan that
    clip at all -- so it uses labels.csv's `camera` column, exactly as `eval run` does.
    """
    labels = _labels(tmp_path, f"{CYRILLIC},0.0,8.0,bullying,hall_right\n")

    with pytest.raises(ClipNameError):
        camera_for(CYRILLIC, load_cameras())  # the filename alone cannot prove it

    plan = _scan_plan(_clips(tmp_path, CYRILLIC), labels)

    assert [(clip.name, camera.name) for clip, camera in plan] == [(CYRILLIC, "hall_right")]


def test_an_explicit_camera_beats_the_filename_in_scan_too(tmp_path: Path) -> None:
    """Same precedence, or the two commands would disagree about what a clip IS."""
    labels = _labels(tmp_path, f"{CLIP},0.0,8.0,bullying,hall_right\n")

    (_, camera), = _scan_plan(_clips(tmp_path, CLIP), labels)

    assert camera.name == "hall_right", "inference overruled the human"


def test_a_clip_whose_camera_nobody_can_prove_is_a_hard_error_naming_it(tmp_path: Path) -> None:
    """Never a default. hall_left masks a reflective column hall_right cannot see; the
    wrong camera's zones blank a region of the frame and return a plausible, wrong number.
    """
    with pytest.raises(SystemExit, match="cannot infer the camera") as excinfo:
        _scan_plan(_clips(tmp_path, CYRILLIC), tmp_path / "no-labels.csv")

    assert CYRILLIC in str(excinfo.value)
    assert "camera" in str(excinfo.value)


def test_every_camera_is_resolved_before_a_single_frame_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """663 clips is a long time on a GPU. A clip whose camera cannot be proved must stop
    the run in the first second, not forty minutes in, having already burned the scan.
    """
    scanned: list[Path] = []
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(
        scanning, "_scan_one", lambda clip, camera, pose, device: scanned.append(clip) or []
    )
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, CYRILLIC),  # sorts FIRST -- the good clip is ahead of it
        labels=tmp_path / "no-labels.csv",
        out=tmp_path / "candidates.csv",
        device="cpu",
    )

    with pytest.raises(SystemExit):
        cmd_scan(args)

    assert scanned == [], "it started decoding video before it knew every clip's camera"
    assert not (tmp_path / "candidates.csv").exists()


def test_a_clip_with_no_main_segment_no_longer_takes_the_whole_scan_down(
    tmp_path: Path,
) -> None:
    """17 stairs clips have no `_main`. `_scan_plan` raised on the first one it reached,
    so `qorgan eval scan` could not run over the corpus AT ALL and nothing could be
    calibrated. They are ordinary clips with an ordinary camera."""
    plan = _scan_plan(_clips(tmp_path, CLIP, STAIRS), tmp_path / "no-labels.csv")

    assert dict((c.name, cam.name) for c, cam in plan) == {
        CLIP: "hall_left",
        STAIRS: "stairs_floor2",
    }


# -- eval/clips/ holds full frames. A crop in there is scored as a scene. ---


def test_a_crop_is_refused_by_resolution_naming_it() -> None:
    """A 320x450 ROI has no scene, no zones, and the scorer's box-diagonal scaling is
    meaningless inside it. Scoring one produces numbers, and every one of them is a lie."""
    with pytest.raises(NotAFullFrameError, match="not full frames") as excinfo:
        refuse_crops({CLIP: FULL_FRAME, "crop.mp4": A_CROP})

    assert "crop.mp4" in str(excinfo.value)
    assert CLIP not in str(excinfo.value), "it refused the full frame too"


def test_a_tall_narrow_crop_is_refused_on_its_WIDTH() -> None:
    """The crops run up to 720 tall -- taller than plenty of full frames. Height alone
    would wave this one through; both dimensions have to clear the bar."""
    with pytest.raises(NotAFullFrameError):
        refuse_crops({"tall.mp4": (320, 720)})


def test_a_directory_of_full_frames_is_refused_nothing() -> None:
    assert refuse_crops({CLIP: FULL_FRAME, RIGHT: FULL_FRAME}) is None


def test_scan_refuses_a_crop_in_the_clips_directory_before_it_decodes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must fire in the pre-flight, next to the camera check -- not after forty
    minutes of GPU has already gone into a corpus that was never scoreable."""
    crop = "hall_left_main_1009_1019_20260702_144150_952947.mp4"
    monkeypatch.setattr(
        scanning, "_frame_size", lambda path: A_CROP if path.name == crop else FULL_FRAME
    )
    scanned: list[Path] = []
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(
        scanning, "_scan_one", lambda clip, camera, pose, device: scanned.append(clip) or []
    )
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, crop),
        labels=tmp_path / "no-labels.csv",
        out=tmp_path / "candidates.csv",
        device="cpu",
    )

    with pytest.raises(SystemExit, match="not full frames") as excinfo:
        cmd_scan(args)

    assert crop in str(excinfo.value), "it refused, but did not say which clip"
    assert scanned == [], "it scored a crop as if it were a scene"
    assert not (tmp_path / "candidates.csv").exists()


def test_a_full_frame_with_no_burst_marker_is_still_scanned(tmp_path: Path) -> None:
    """17 full-frame clips carry no `burst` marker, so the marker cannot be the test. The
    RESOLUTION is: full frames are 2560x1440 and crops are at most 320 wide."""
    plan = _scan_plan(_clips(tmp_path, NO_BURST), tmp_path / "no-labels.csv")

    assert [(clip.name, camera.name) for clip, camera in plan] == [(NO_BURST, "hall_left")]


def test_a_scan_of_clips_it_can_place_writes_the_candidates_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ScanRow(CLIP, 2.4, 1.83, 0.71, 0.66)
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(scanning, "_scan_one", lambda clip, camera, pose, device: [row])
    out = tmp_path / "eval" / "candidates.csv"
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, RIGHT),
        labels=tmp_path / "no-labels.csv",
        out=out,
        device="cpu",
    )

    assert cmd_scan(args) == 0
    assert load_candidates(out) == [row, row]


# -- the coverage manifest: PROOF of which clips the detector actually ran on ----


def test_coverage_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "candidates.coverage.csv"

    assert write_coverage([RIGHT, CLIP], path) == 2
    assert load_coverage(path) == {CLIP, RIGHT}


def test_a_missing_coverage_manifest_reads_as_None_never_an_empty_set(tmp_path: Path) -> None:
    """None is not {}. An empty manifest means the scan covered nothing; a missing file
    means we do not KNOW what was covered -- an old candidates.csv from before coverage was
    recorded. `eval sample` must be able to tell those two apart, so the reader must too."""
    assert load_coverage(tmp_path / "nope.coverage.csv") is None
    assert write_coverage([], tmp_path / "empty.coverage.csv") == 0
    assert load_coverage(tmp_path / "empty.coverage.csv") == set()


def test_the_manifest_sits_beside_the_candidates_file_paired_by_name(tmp_path: Path) -> None:
    out = tmp_path / "eval" / "candidates.csv"

    manifest = coverage_path(out)

    assert manifest.parent == out.parent, "the manifest must sit next to candidates.csv"
    assert manifest.name == "candidates.coverage.csv"


def test_a_scan_records_every_clip_it_covered_including_the_silent_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the manifest: a clip the detector ran to completion but that
    produced NO candidate is still COVERED. Absence from candidates.csv then proves silence
    (the clip was seen and nothing fired) instead of merely guessing it. RIGHT here fires
    nothing, yet it must appear in the manifest."""
    fired = ScanRow(CLIP, 2.4, 1.83, 0.71, 0.66)
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(
        scanning,
        "_scan_one",
        lambda clip, camera, pose, device: [fired] if clip.name == CLIP else [],
    )
    out = tmp_path / "eval" / "candidates.csv"
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, RIGHT),
        labels=tmp_path / "no-labels.csv",
        out=out,
        device="cpu",
    )

    assert cmd_scan(args) == 0

    covered = load_coverage(coverage_path(out))
    assert covered == {CLIP, RIGHT}, "a clip that fired nothing was left out of the manifest"
    assert {row.clip for row in load_candidates(out)} == {CLIP}, "RIGHT wrote a candidate row"


def test_the_manifest_is_written_in_the_same_scan_run_as_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two artifacts are written together at the end of one `cmd_scan`, so they cannot
    drift within a run: every clip in candidates.csv is a clip in the manifest."""
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(
        scanning,
        "_scan_one",
        lambda clip, camera, pose, device: [ScanRow(clip.name, 1.0, 1.0, 0.5, 0.5)],
    )
    out = tmp_path / "eval" / "candidates.csv"
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, RIGHT),
        labels=tmp_path / "no-labels.csv",
        out=out,
        device="cpu",
    )

    assert cmd_scan(args) == 0

    candidate_clips = {row.clip for row in load_candidates(out)}
    covered = load_coverage(coverage_path(out))
    assert candidate_clips <= covered, "a candidate names a clip the manifest does not cover"


def test_a_clip_the_scan_errored_on_is_not_certified_as_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip whose scan raised must NOT land in the manifest: it certifies clips that were
    decoded and run to completion, and this one was not.

    This used to be all-or-nothing -- the exception aborted the run and neither artifact was
    written, so nothing was falsely certified and nothing was kept either. The scan now
    survives the clip and keeps the other 656, and the certification rule is unchanged,
    which is the half that carries the safety. The clip that failed is named in its own
    artifact instead; `test_eval_scan_resume` is where that contract lives."""
    def _boom(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == RIGHT:
            raise RuntimeError("decode failed on RIGHT")
        return [ScanRow(clip.name, 2.4, 1.83, 0.71, 0.66)]

    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(scanning, "_scan_one", _boom)
    out = tmp_path / "eval" / "candidates.csv"
    args = argparse.Namespace(
        clips=_clips(tmp_path, CLIP, RIGHT),
        labels=tmp_path / "no-labels.csv",
        out=out,
        device="cpu",
    )

    assert cmd_scan(args) == 1, "a corpus with an unread clip must not exit clean"

    assert load_coverage(coverage_path(out)) == {CLIP}, "it certified a clip that raised"
    assert {row.clip for row in load_candidates(out)} == {CLIP}, "RIGHT wrote a candidate row"

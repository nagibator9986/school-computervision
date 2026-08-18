"""Assembling several DVR files into one lesson — the checks, and the refusals.

Two kinds of test here, deliberately.

**Synthetic videos** (`ffmpeg`, 64x64, a few seconds, named the way the D14 recorder names
its exports) exercise the contiguity rules without needing a gigabyte of footage of
children. They are enough because the rule under test is arithmetic on file headers and
clocks: the overlay is unreadable at 64x64, so the clock resolves from the file name, which
is exactly the path the real camera takes (`MEASUREMENTS.md` §9).

**A fake `detect_pass`** exercises the merge itself — offsets, overlap trimming, track-id
blocks — with observations whose every value is known in advance. Testing that against real
detections would be testing the model.

The real-footage tests at the end skip cleanly when the D14 files are absent, and say so.
A test that passes without its data is worse than no test.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pytest

from classvision import session as session_mod

ROOT = Path(__file__).resolve().parents[2]
D14_A = ROOT / "D14_20260815101759.mp4"
D14_B = ROOT / "D14_20260815103136.mp4"

needs_d14 = pytest.mark.skipif(not (D14_A.exists() and D14_B.exists()),
                               reason="the D14 recordings are not present")


def _clip(directory: Path, name: str, seconds: float, size: str = "64x64",
          fps: int = 20) -> Path:
    """A tiny valid mp4 whose NAME carries the start time, like the DVR's exports.

    The colour is derived from the name, and that is not decoration: two clips of identical
    black are byte-identical, and `resolve` refuses a session containing the same content
    twice — a refusal these fixtures would otherwise trip on their way to testing something
    else. Distinct content here is what makes the duplicate check testable separately.
    """
    path = directory / name
    tint = f"0x{abs(hash(name)) % 0xFFFFFF:06x}"
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={tint}:s={size}:r={fps}", "-t", str(seconds),
         "-pix_fmt", "yuv420p", str(path)], capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg unavailable: {result.stderr.decode()[:200]}")
    return path


def _stamp(when: dt.datetime) -> str:
    return when.strftime("%Y%m%d%H%M%S")


# -- ordering and contiguity -----------------------------------------------------------

def test_the_order_of_the_arguments_is_not_trusted(tmp_path):
    """A session is ordered by the recordings' own clocks. A caller who lists the parts
    backwards has made a typing mistake, not a claim about time."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)
    second = _clip(tmp_path, f"D14_{_stamp(start + dt.timedelta(seconds=5))}.mp4", 5.0)

    session = session_mod.resolve([second, first])
    assert [p.path.name for p in session.parts] == [first.name, second.name]
    assert session.parts[0].offset_seconds == 0.0
    assert session.parts[1].offset_seconds == pytest.approx(5.0, abs=0.01)
    assert session.duration_seconds == pytest.approx(10.0, abs=0.05)


def test_a_real_gap_is_refused_rather_than_concatenated(tmp_path):
    """The failure this module exists to prevent: a break glued into the middle of a
    lesson, reported afterwards as a child sitting motionless for the length of it."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)
    late = _clip(tmp_path, f"D14_{_stamp(start + dt.timedelta(seconds=65))}.mp4", 5.0)

    with pytest.raises(session_mod.NotOneLesson) as refusal:
        session_mod.resolve([first, late])
    assert "60.00 s" in str(refusal.value)


def test_an_overlap_beyond_the_tolerance_is_refused_too(tmp_path):
    """Symmetric on purpose: files that share a minute of footage would count that minute
    twice, which inflates exactly the counters this package states carefully."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 30.0)
    overlapping = _clip(tmp_path, f"D14_{_stamp(start + dt.timedelta(seconds=10))}.mp4", 30.0)

    with pytest.raises(session_mod.NotOneLesson):
        session_mod.resolve([first, overlapping])


def test_a_one_second_handover_is_inside_the_tolerance(tmp_path):
    """What the D14 recorder actually does: the next file's name claims a start a second
    before the previous file's last frame, because both stamps are whole seconds."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)
    second = _clip(tmp_path, f"D14_{_stamp(start + dt.timedelta(seconds=4))}.mp4", 5.0)

    session = session_mod.resolve([first, second])
    assert len(session.seams) == 1
    assert session.seams[0].kind == "overlap"
    assert session.seams[0].gap_seconds == pytest.approx(-1.0, abs=0.05)


# -- same room, same camera ------------------------------------------------------------

def test_two_resolutions_are_two_coordinate_systems_and_are_refused(tmp_path):
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0, size="64x64")
    second = _clip(tmp_path, f"D14_{_stamp(start + dt.timedelta(seconds=5))}.mp4", 5.0,
                   size="96x96")

    with pytest.raises(session_mod.NotOneLesson) as refusal:
        session_mod.resolve([first, second])
    assert "pixel coordinates" in str(refusal.value)


def test_two_dvr_channels_are_two_cameras_and_are_refused(tmp_path):
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)
    other = _clip(tmp_path, f"D07_{_stamp(start + dt.timedelta(seconds=5))}.mp4", 5.0)

    with pytest.raises(session_mod.NotOneLesson) as refusal:
        session_mod.resolve([first, other])
    assert "channel" in str(refusal.value)


def test_the_same_file_twice_would_double_every_counter_and_is_refused(tmp_path):
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)

    with pytest.raises(session_mod.NotOneLesson) as refusal:
        session_mod.resolve([first, first])
    assert "byte-identical" in str(refusal.value)


def test_an_undated_part_cannot_join_a_session(tmp_path):
    """Contiguity is a statement about clocks. With no clock there is no evidence that
    these two files are one lesson, and the module says so instead of assuming it."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    first = _clip(tmp_path, f"D14_{_stamp(start)}.mp4", 5.0)
    nameless = _clip(tmp_path, "something.mp4", 5.0)

    with pytest.raises(session_mod.NotOneLesson) as refusal:
        session_mod.resolve([first, nameless])
    assert "cannot date" in str(refusal.value)


# -- the merge itself ------------------------------------------------------------------

def _part(index: int, *, start: dt.datetime, duration: float, offset: float) -> session_mod.Part:
    return session_mod.Part(
        index=index, path=Path(f"part_{index}.mp4"), sha256=f"{index:064d}",
        size_bytes=1, started_at=start, clock_source="filename", clock_drift_seconds=None,
        fps=20.0, frame_count=int(duration * 20), width=2560, height=1440,
        duration_seconds=duration, offset_seconds=offset,
    )


def _fake_detections(times: list[float], tracks: list[int]):
    from classvision.pipeline import Detections

    n = len(times)
    return Detections(
        times=np.array(times, dtype=np.float32),
        xy=np.zeros((n, 17, 2), np.float32), conf=np.zeros((n, 17), np.float32),
        box=np.zeros((n, 4), np.float32), track=np.array(tracks, dtype=np.int32),
        frame_times=np.array(sorted(set(times)), dtype=np.float32), seconds_spent=1.0,
    )


def _two_part_session(gap: float):
    """Two 10 s parts, the second starting `gap` seconds after the first ends."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    parts = (_part(0, start=start, duration=10.0, offset=0.0),
             _part(1, start=start + dt.timedelta(seconds=10 + gap), duration=10.0,
                   offset=10.0 + gap))
    seam = session_mod.Seam(after_part=0, at_session_seconds=10.0, at_wall=None,
                            gap_seconds=gap)
    return session_mod.Session(parts=parts, seams=(seam,),
                               tolerance_seconds=session_mod.MAX_SEAM_GAP_SECONDS)


def test_observations_land_on_one_timeline_in_session_seconds(monkeypatch):
    """The entire point: downstream code is unchanged because it still receives one
    `Detections` whose times run from the lesson's first frame to its last."""
    import classvision.pipeline as pipeline

    per_file = {0: _fake_detections([0.0, 0.5, 1.0], [1, 1, 2]),
                1: _fake_detections([0.0, 0.5, 1.0], [1, 1, 2])}
    calls = {"n": 0}

    def fake(path, settings, *, use_cache=True):
        detections = per_file[calls["n"]]
        calls["n"] += 1
        return detections

    monkeypatch.setattr(pipeline, "detect_pass", fake)
    session = _two_part_session(gap=1.0)
    merged, diagnostics = session_mod.detect_session(session, pipeline.Settings())

    assert list(np.round(merged.times, 2)) == [0.0, 0.5, 1.0, 11.0, 11.5, 12.0]
    assert diagnostics["parts"][1]["offset_seconds"] == 11.0


def test_track_ids_from_different_files_can_never_collide(monkeypatch):
    """Track continuity across a seam is impossible — the tracker is restarted per file —
    so the merge makes it impossible to assume by accident as well as in principle."""
    import classvision.pipeline as pipeline

    per_file = [_fake_detections([0.0, 0.5], [1, 2]), _fake_detections([0.0, 0.5], [1, -1])]
    monkeypatch.setattr(pipeline, "detect_pass",
                        lambda path, settings, use_cache=True: per_file.pop(0))

    merged, _ = session_mod.detect_session(_two_part_session(gap=1.0), pipeline.Settings())
    assert list(merged.track) == [1, 2, session_mod.TRACK_ID_PART_STRIDE + 1, -1]
    # -1 stays -1: "the tracker had not committed" must not become a per-file identity.
    assert (merged.track == -1).sum() == 1


def test_an_overlap_is_trimmed_once_rather_than_counted_twice(monkeypatch):
    """A negative seam means both files hold the same wall seconds. Keeping both would
    observe every pupil twice for that second."""
    import classvision.pipeline as pipeline

    per_file = [_fake_detections([9.0, 9.5], [1, 1]),
                _fake_detections([0.0, 0.5, 1.0, 1.5], [1, 1, 1, 1])]
    monkeypatch.setattr(pipeline, "detect_pass",
                        lambda path, settings, use_cache=True: per_file.pop(0))

    session = _two_part_session(gap=-1.0)     # part 1 starts at 9.0 s, part 0 ends at 10.0
    merged, _ = session_mod.detect_session(session, pipeline.Settings())

    # 9.0 and 9.5 arrive from part 1 as well; only part 0's copies survive.
    assert list(np.round(merged.times, 2)) == [9.0, 9.5, 10.0, 10.5]
    assert session.seams[0].trimmed_observations == 2
    assert session.seams[0].kind == "overlap"


def test_the_sampling_cadence_survives_the_seam(monkeypatch):
    """Each file is sampled from its own first frame, so the two phases are unrelated. On
    the real D14 pair that put two samples 52 ms apart across the join — and the ledger
    charges every observation a whole `sample_interval`, so that pair bought 1.0 s of
    «наблюдалось» for 0.05 s of footage at every seat."""
    import classvision.pipeline as pipeline

    per_file = [_fake_detections([9.4, 9.9], [1, 1]),
                _fake_detections([1.0, 1.1, 1.5], [1, 1, 1])]
    monkeypatch.setattr(pipeline, "detect_pass",
                        lambda path, settings, use_cache=True: per_file.pop(0))

    session = _two_part_session(gap=-1.0)      # part 1 starts at 9.0 s on the timeline
    merged, _ = session_mod.detect_session(session, pipeline.Settings(sample_fps=2.0))

    # 10.0 and 10.1 are inside a sampling interval of part 0's last frame (9.9) and go.
    assert list(np.round(merged.frame_times, 2)) == [9.4, 9.9, 10.5]
    assert float(np.diff(merged.frame_times).min()) >= 0.5 - 1e-6


def test_a_session_of_one_file_is_not_a_second_identity_for_that_file(monkeypatch):
    """Otherwise `analyse X` and `analyse-session X` would put one measurement into the
    school's database twice, under two `run_id`s that nothing relates."""
    start = dt.datetime(2026, 8, 15, 10, 0, 0)
    parts = (_part(0, start=start, duration=10.0, offset=0.0),)
    one = session_mod.Session(parts=parts, seams=(), tolerance_seconds=2.0)
    assert one.digest == parts[0].sha256


def test_a_cut_that_would_be_applied_to_every_part_is_refused(monkeypatch):
    import classvision.pipeline as pipeline

    settings = pipeline.Settings(start_seconds=60.0)
    with pytest.raises(session_mod.NotOneLesson):
        session_mod.detect_session(_two_part_session(gap=0.0), settings)


def test_the_digest_covers_the_placement_not_only_the_files():
    """Two assemblies of the same files are two measurements. If the digest ignored the
    placement, the second would overwrite the first in the school's database."""
    one = _two_part_session(gap=0.0)
    other = _two_part_session(gap=1.0)
    assert one.digest != other.digest


def test_recorded_seconds_and_duration_are_different_numbers():
    """A hole in the recording is not a hole in the lesson, and the artefact must not let
    the two look alike."""
    session = _two_part_session(gap=1.5)
    assert session.duration_seconds == pytest.approx(21.5)
    assert session.recorded_seconds == pytest.approx(20.0)


# -- the real footage ------------------------------------------------------------------

@needs_d14
def test_the_two_d14_files_are_one_lesson_and_the_seam_is_measured():
    """The pair this module was written for. Every number here is read off the files."""
    session = session_mod.resolve([D14_A, D14_B])

    assert [p.path.name for p in session.parts] == [D14_A.name, D14_B.name]
    assert session.started_at == dt.datetime(2026, 8, 15, 10, 17, 59)
    assert session.clock_source == "filename"      # the overlay sits over a window
    # 817 s from the first file's start to the second's, plus the second file's 2903.9 s.
    assert session.duration_seconds == pytest.approx(3720.87, abs=0.1)

    seam = session.seams[0]
    assert seam.kind == "overlap"
    # The second file's name claims a start 1 s before the first file's last frame: two
    # whole-second stamps around one handover, which is what the tolerance is sized for.
    assert seam.gap_seconds == pytest.approx(-1.0, abs=0.05)
    assert abs(seam.gap_seconds) < session_mod.MAX_SEAM_GAP_SECONDS


@needs_d14
def test_a_session_of_one_file_is_just_that_file():
    """A degenerate session must not be a special case: one part, no seams, and the same
    timeline the ordinary pipeline would have built."""
    session = session_mod.resolve([D14_A])
    assert session.seams == ()
    assert session.parts[0].offset_seconds == 0.0
    assert session.duration_seconds == pytest.approx(818.0, abs=0.1)

"""The burned-in clock reader, pinned against frames whose text a human has read.

These are the 13 frames from `MEASUREMENTS.md` §1. They are the only ground truth in this
repository that did not come out of a model, so they are worth a table-driven test: if a
future change to the glyph masks or the cell grid breaks the reader, this fails with the
exact frame and the exact wrong string, rather than a term's lessons landing in the wrong
week.

The video is not in version control (`.gitignore` — it is footage of children), so every
test here skips cleanly when it is absent. A skipped test says so; a test that silently
passes without its data is worse than no test.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from classvision.video import clock

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "test_camera.mp4"
CLIP = ROOT / "clip_15min.mp4"

needs_video = pytest.mark.skipif(not VIDEO.exists(), reason="test_camera.mp4 not present")
needs_clip = pytest.mark.skipif(not CLIP.exists(), reason="clip_15min.mp4 not present")

# The measured start. Every other expectation in this file derives from it.
STARTED_AT = dt.datetime(2026, 8, 7, 9, 54, 58)


def test_date_order_is_decided_by_the_weekday_not_by_a_default():
    """«08-07-2026» is 8 July or 7 August depending on the DVR's menu. Only one of them
    is the Friday the same overlay prints, and that is how the ambiguity is resolved."""
    assert clock.parse("08-07-2026 09:55:03", "MDY").strftime("%a") == "Fri"
    assert clock.parse("08-07-2026 09:55:03", "DMY").strftime("%a") == "Wed"
    assert clock.parse("08-07-2026 09:55:03", "MDY").date() == dt.date(2026, 8, 7)


def test_parse_has_no_default_date_order():
    """A default here would be a guess about a camera's settings menu, applied silently
    to every date in the system."""
    with pytest.raises(TypeError):
        clock.parse("08-07-2026 09:55:03")  # type: ignore[call-arg]  # noqa


@needs_video
@pytest.mark.parametrize("offset", [0, 42, 600, 1800, 2400, 2700, 3000])
def test_reads_the_overlay_at_known_offsets(offset: int):
    """Decoded sequentially rather than by seeking: OpenCV's HEVC seek lands on a keyframe
    and reports a position it did not land on.

    **The tolerance is one second, and it is a property of the DVR rather than slack.**
    The overlay's second boundary has an arbitrary sub-frame phase against frame 0 — on
    this recording it falls at t ≈ 2.35 s — so the frame at exactly t = 600.0 s may still
    be showing the previous second. Demanding equality asserts a phase alignment that no
    camera promises. What matters, and what `test_the_clock_is_linear...` pins, is that
    the clock never skips or repeats a second.
    """
    from classvision.video import decode

    wanted = STARTED_AT + dt.timedelta(seconds=offset)
    for sample in decode.samples(VIDEO, sample_fps=0.0, start_seconds=offset):
        reading = clock.read_overlay(sample.image)
        drift = abs((reading.when - wanted).total_seconds())
        assert drift <= 1.0, f"t={offset}s read {reading.text!r}, wanted ~{wanted}"
        assert reading.margin > clock.MIN_MARGIN
        return
    pytest.fail("no frame decoded")


@needs_video
def test_the_clock_is_linear_which_is_what_licenses_wallclock_at():
    """`WallClock.at()` is a multiplication. This is the measurement that permits that."""
    result = clock.verify_linear(VIDEO)
    assert result["steps_all_one_second"], result
    assert result["interior_gaps_within_one_frame"], result


@needs_video
def test_drift_over_the_whole_recording_is_below_the_overlays_own_resolution():
    wall = clock.from_video(VIDEO)
    assert wall.source is clock.ClockSource.OVERLAY
    assert wall.started_at == STARTED_AT
    assert abs(wall.drift_seconds) < 1.0, wall.drift_seconds


@needs_clip
def test_the_clip_carries_its_own_start_and_is_not_assumed_from_the_parent():
    """The 15-minute clip was cut with `-c copy`; its start is read from its own first
    frame, not computed from an offset somebody remembered."""
    wall = clock.from_video(CLIP)
    assert wall.started_at == dt.datetime(2026, 8, 7, 9, 59, 58)
    assert wall.at(900).hour == 10


def test_unknown_clock_refuses_to_pretend():
    wall = clock.WallClock.unknown()
    assert wall.source is clock.ClockSource.UNKNOWN


def test_a_frame_too_small_to_hold_the_strip_is_refused_rather_than_rescaled():
    """The glyph masks are pixel thresholds on a 2560x1440 stream. A resized frame needs
    its own profile; silently rescaling would return a confident wrong time.

    Note the size: the strip only runs to x=736, so 1280x720 is NOT too small — it is
    merely a frame with no clock in it, which is the next test's case.
    """
    import numpy as np

    with pytest.raises(clock.OverlayUnreadable, match="profile"):
        clock.read_overlay(np.zeros((100, 100, 3), dtype=np.uint8))


def test_a_frame_with_no_legible_clock_is_refused_on_the_margin():
    """Large enough to contain the strip, but nothing in it discriminates one glyph from
    another. The reader must decline rather than return whichever template won by noise."""
    import numpy as np

    with pytest.raises(clock.OverlayUnreadable, match="runner-up"):
        clock.read_overlay(np.zeros((720, 1280, 3), dtype=np.uint8))

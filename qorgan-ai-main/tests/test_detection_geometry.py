"""Geometry and tracking. Pure functions, so these tests are exact rather than fuzzy."""

from __future__ import annotations

import pytest

from qorgan.detection.geometry import (
    Box,
    approach_speed,
    direction_change_degrees,
    dynamic_threshold,
    iou,
)
from qorgan.detection.tracking import TrackStore, travelled


def test_box_basics() -> None:
    box = Box(10, 20, 110, 220)
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20_000
    assert box.center == (60, 120)
    assert box.diagonal == pytest.approx(223.607, abs=0.01)


def test_a_box_is_clamped_into_the_frame() -> None:
    clamped = Box(-30, -10, 700, 500).clamped(640, 480)
    assert (clamped.x1, clamped.y1) == (0, 0)
    assert (clamped.x2, clamped.y2) == (640, 480)


def test_a_clamped_box_is_never_degenerate() -> None:
    """A zero-width box crops to nothing and blows up the pose model downstream."""
    clamped = Box(700, 500, 800, 600).clamped(640, 480)
    assert clamped.width >= 1
    assert clamped.height >= 1


def test_iou_of_identical_boxes_is_one() -> None:
    box = Box(0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou(Box(0, 0, 10, 10), Box(50, 50, 60, 60)) == 0.0


def test_iou_of_half_overlapping_boxes() -> None:
    # Intersection 50, union 150.
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_the_proximity_threshold_scales_with_how_big_the_people_look() -> None:
    """Two children at the far end of a corridor are 40 px apart when touching; two in
    the foreground are 200 px apart at arm's length. A fixed pixel threshold cannot
    serve both ends of the same camera."""
    far = Box(0, 0, 10, 20)  # tiny: at the far end of the corridor
    near = Box(0, 0, 150, 300)  # large: right up against the lens

    threshold_far = dynamic_threshold(far, far, base=50, close_distance_ratio=0.9)
    threshold_near = dynamic_threshold(near, near, base=50, close_distance_ratio=0.9)

    assert threshold_near > threshold_far
    assert threshold_far == 50  # the base acts as a floor for distant, small boxes


def test_direction_change_is_zero_when_a_person_is_still() -> None:
    assert direction_change_degrees((0.0, 0.0), (5.0, 5.0)) == 0.0


def test_a_reversal_is_180_degrees() -> None:
    assert direction_change_degrees((10.0, 0.0), (-10.0, 0.0)) == pytest.approx(180.0)


def test_a_right_turn_is_90_degrees() -> None:
    assert direction_change_degrees((10.0, 0.0), (0.0, 10.0)) == pytest.approx(90.0)


def test_approach_speed_is_positive_when_closing() -> None:
    a, b = (0.0, 0.0), (100.0, 0.0)
    closing = approach_speed(a, b, a_velocity=(20.0, 0.0), b_velocity=(0.0, 0.0))
    assert closing == pytest.approx(20.0)


def test_approach_speed_is_negative_when_separating() -> None:
    a, b = (0.0, 0.0), (100.0, 0.0)
    parting = approach_speed(a, b, a_velocity=(-20.0, 0.0), b_velocity=(0.0, 0.0))
    assert parting == pytest.approx(-20.0)


def test_two_people_running_side_by_side_are_not_approaching() -> None:
    """A large relative speed with zero approach speed is a chase, not a charge.
    Confusing the two is a whole class of false positive."""
    a, b = (0.0, 0.0), (0.0, 100.0)
    velocity = (30.0, 0.0)
    assert approach_speed(a, b, velocity, velocity) == pytest.approx(0.0)


# -- tracking: the acceleration units fix ---------------------------------


def _walk(store: TrackStore, fps: float, seconds: float, px_per_second: float) -> None:
    """Walk one person in a straight line at a constant physical speed."""
    step = 1.0 / fps
    for index in range(int(seconds * fps)):
        t = index * step
        x = px_per_second * t
        store.update({1: Box(x, 0, x + 50, 100)}, timestamp=t)


def test_speed_is_measured_in_pixels_per_second_not_per_frame() -> None:
    """The bug: the legacy measured speed in px/FRAME and called the derivative
    acceleration, so the same physical motion read differently at 8 fps and 25 fps and
    no threshold transferred between cameras."""
    slow_camera = TrackStore(smoothing=0.0)
    fast_camera = TrackStore(smoothing=0.0)

    _walk(slow_camera, fps=8, seconds=2.0, px_per_second=100.0)
    _walk(fast_camera, fps=25, seconds=2.0, px_per_second=100.0)

    slow = slow_camera.tracks[1].speed
    fast = fast_camera.tracks[1].speed

    # Same person, same walk, two cameras. The numbers must agree.
    assert slow == pytest.approx(100.0, rel=0.05)
    assert fast == pytest.approx(100.0, rel=0.05)
    assert slow == pytest.approx(fast, rel=0.05)


def test_a_constant_speed_produces_no_acceleration() -> None:
    store = TrackStore(smoothing=0.0)
    _walk(store, fps=10, seconds=2.0, px_per_second=120.0)
    assert store.tracks[1].acceleration == pytest.approx(0.0, abs=1.0)


def test_acceleration_is_px_per_second_squared_and_transfers_between_cameras() -> None:
    """A lunge: standing still, then moving off at 200 px/s.

    This is THE test. The same physical event, filmed by an 8 fps camera and a 25 fps
    one, must produce the same acceleration -- otherwise no `acceleration_threshold`
    can be shared between cameras and no recalibration means anything. The legacy was
    off by the ratio of the frame rates.

    Peak acceleration, not the final instantaneous value: the gates ask "did this
    person lunge at any point", and by the end of the move the speed has plateaued.
    """

    def peak_acceleration(fps: float) -> float:
        store = TrackStore(smoothing=0.0)
        step = 1.0 / fps
        peak = 0.0

        for index in range(int(1.0 * fps)):  # stand still for a second
            store.update({1: Box(0, 0, 50, 100)}, timestamp=index * step)

        start = 1.0
        for index in range(int(1.0 * fps)):  # then move off at 200 px/s
            t = start + index * step
            x = 200.0 * (t - start)
            store.update({1: Box(x, 0, x + 50, 100)}, timestamp=t)
            peak = max(peak, store.tracks[1].acceleration)

        return peak

    slow, fast = peak_acceleration(8), peak_acceleration(25)

    assert slow > 100, "the lunge was not detected at all"
    assert fast > 100
    assert slow == pytest.approx(fast, rel=0.25), (
        f"acceleration did not transfer between frame rates: "
        f"8fps={slow:.0f} px/s^2, 25fps={fast:.0f} px/s^2"
    )


def test_a_track_is_not_confirmed_until_it_has_been_seen_enough() -> None:
    """A track seen once is usually a detector hiccup, not a child."""
    store = TrackStore()
    store.update({1: Box(0, 0, 50, 100)}, timestamp=0.0)
    assert not store.confirmed(min_hits=2)

    store.update({1: Box(5, 0, 55, 100)}, timestamp=0.1)
    assert len(store.confirmed(min_hits=2)) == 1


def test_tracks_that_vanish_are_evicted() -> None:
    """Rule R8. Track ids only ever increase, so a dict keyed on them grows without
    limit -- the legacy leaked several of these for as long as the process lived."""
    store = TrackStore(max_lost=3)
    store.update({1: Box(0, 0, 50, 100)}, timestamp=0.0)
    assert 1 in store.tracks

    for index in range(1, 6):
        store.update({2: Box(0, 0, 50, 100)}, timestamp=index * 0.1)

    assert 1 not in store.tracks, "the vanished track was never evicted"
    assert 2 in store.tracks


def test_travelled_measures_real_displacement_not_jitter() -> None:
    store = TrackStore(smoothing=0.0)
    # A person standing still, with the detector box wobbling a couple of pixels.
    for index, dx in enumerate([0, 2, -1, 1]):
        store.update({1: Box(dx, 0, dx + 50, 100)}, timestamp=index * 0.1)

    assert travelled(store.tracks[1]) < 5.0

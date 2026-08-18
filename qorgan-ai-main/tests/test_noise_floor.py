"""The noise floor: what the detector reports when nothing is happening.

**A threshold below the noise floor cannot be tuned. It can only be replaced.** These
tests are the guard rail that says so in a way that fails a build rather than sitting in
a comment nobody reads.

The legacy shipped `acceleration_threshold: 6` in the incoherent unit (px/frame)/s. v2
converted it honestly to 60 px/s^2 -- and an honest conversion of a badly chosen number
is still a badly chosen number. The conversion fixed the UNITS bug. It could not fix the
VALUE, and nothing but a measurement can.
"""

from __future__ import annotations

import random

import pytest

from qorgan.config.bullying import PairMetrics
from qorgan.config.camera import BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.detection.geometry import Box
from qorgan.detection.tracking import TrackStore
from qorgan.evaluation.noise_floor import (
    ASSUMED_JITTER_PX,
    SETTLED_HITS,
    analysis_fps,
    calm_corridor,
    measure,
    separation,
    synthetic_floor,
)

# A shove is worth ~800 px/s^2 through this pipeline (see `_shove`). A threshold has to
# clear the noise, but it also has to leave room for the signal -- putting it at 0.9x a
# shove would pass this file's noise checks and detect nothing. Both jaws of the vice.
MAX_FRACTION_OF_A_SHOVE = 0.5


def _bullying_cameras() -> list[tuple[str, BullyingCamera]]:
    return [
        (name, camera)
        for name, camera in sorted(load_cameras().items())
        if isinstance(camera, BullyingCamera)
    ]


def _fps(camera: BullyingCamera) -> float:
    """The rate the loop really runs this camera at.

    Was `display_fps / det_every`, a field production never read -- so every floor in this
    file was modelled at 10 fps (hall) or 8 (stairs, yard) for a loop running at 15. The
    guard rail was denominated in units the thing it guards does not use.
    """
    return analysis_fps(camera.capture.stream_fps, camera.capture.det_every)


def _shove(metrics: PairMetrics, fps: float, *, seed: int = 7) -> float:
    """A child standing still is shoved: 0 -> 250 px/s in 0.2 s, then stumbles to a halt.

    The signal, measured through the same tracker as the noise. Same box wobble, so the
    comparison is fair.
    """
    rng = random.Random(seed)  # noqa: S311 - a shaky bounding box, not a cipher
    store = TrackStore(smoothing=metrics.tracker_smoothing, max_lost=metrics.tracker_max_lost)
    dt = 1.0 / fps
    peak = 0.0
    x, velocity = 640.0, 0.0

    for index in range(int(4 * fps)):
        now = index * dt
        if 2.0 <= now < 2.2:
            velocity = 250.0
        elif now >= 2.2:
            velocity = max(0.0, velocity - 400.0 * dt)
        x += velocity * dt

        cx = x + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        cy = 400.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        store.update({1: Box(x1=cx - 30, y1=cy - 80, x2=cx + 30, y2=cy + 80)}, now)

        track = store.tracks[1]
        if track.hits > 5:
            peak = max(peak, track.acceleration)

    return peak


# -- the instrument itself ---------------------------------------------------


def test_a_calm_corridor_still_reports_acceleration() -> None:
    """The whole premise. This person's true acceleration is zero on every frame."""
    metrics = PairMetrics()
    floor = measure(calm_corridor(fps=10.0), metrics)

    assert floor.samples > 500
    assert floor.acceleration.p50 > 0, "a wobbling box must produce phantom acceleration"
    assert floor.acceleration.peak > 50, "a 3px wobble is worth more than one might guess"


def test_a_perfectly_steady_box_has_no_floor() -> None:
    """The floor is the DETECTOR's wobble, not the tracker inventing motion from nothing.
    With a box that does not shake, the reported acceleration of a constant-velocity walk
    must be essentially zero -- otherwise the bug is in `tracking`, not in the camera."""
    floor = measure(calm_corridor(fps=10.0, jitter_px=0.0), PairMetrics())

    assert floor.acceleration.p99 < 1.0


def test_more_wobble_means_a_higher_floor() -> None:
    metrics = PairMetrics()
    quiet = measure(calm_corridor(fps=10.0, jitter_px=1.0), metrics)
    noisy = measure(calm_corridor(fps=10.0, jitter_px=5.0), metrics)

    assert noisy.acceleration.p99 > quiet.acceleration.p99 * 2


# -- the guard rail: every shipped threshold must clear its own camera's floor ---


@pytest.mark.parametrize(("name", "camera"), _bullying_cameras())
def test_the_acceleration_threshold_sits_above_the_noise_floor(
    name: str, camera: BullyingCamera
) -> None:
    """**If this fails, the camera is detecting its own box, not the children.**

    At 60 px/s^2 -- the value v2 shipped, honestly converted from the legacy's 6 --
    roughly 12% of the frames of an ordinary walk cross the threshold.
    """
    metrics = camera.bullying.metrics
    floor = synthetic_floor(metrics, _fps(camera))

    assert metrics.acceleration_threshold > floor.acceleration.peak, (
        f"{name}: acceleration_threshold={metrics.acceleration_threshold:.0f} px/s^2 sits "
        f"inside the noise floor of a CALM corridor (p99={floor.acceleration.p99:.0f}, "
        f"peak={floor.acceleration.peak:.0f}). "
        f"{floor.crossings(metrics.acceleration_threshold):.1%} of the frames of an "
        f"ordinary walk cross it. "
        f"Measured recommendation: {floor.recommended_acceleration_threshold:.0f}."
    )


@pytest.mark.parametrize(("name", "camera"), _bullying_cameras())
def test_the_acceleration_threshold_still_hears_a_shove(
    name: str, camera: BullyingCamera
) -> None:
    """The other jaw of the vice. Clearing the noise floor is easy if you set the
    threshold to a million; then nothing is ever detected and every test above still
    passes. A real shove has to remain comfortably over the bar."""
    metrics = camera.bullying.metrics
    shove_peak = _shove(metrics, _fps(camera))

    assert metrics.acceleration_threshold < shove_peak * MAX_FRACTION_OF_A_SHOVE, (
        f"{name}: acceleration_threshold={metrics.acceleration_threshold:.0f} is too close "
        f"to what a real shove produces ({shove_peak:.0f} px/s^2). The detector has been "
        f"made deaf in the name of quiet."
    )


@pytest.mark.parametrize(("name", "camera"), _bullying_cameras())
def test_a_shove_is_far_louder_than_a_walk(name: str, camera: BullyingCamera) -> None:
    """The premise that makes this camera usable at all: there is somewhere to PUT a
    threshold. We measure ~6x. Below ~2x, no threshold exists and the honest answer would
    be that acceleration cannot be used on this camera."""
    metrics = camera.bullying.metrics
    fps = _fps(camera)
    floor = synthetic_floor(metrics, fps)

    assert separation(floor, _shove(metrics, fps)) > 3.0, f"{name}: nowhere to put a threshold"


# -- the same number, used the other way round -------------------------------


def test_a_motionless_person_is_still_seen_as_motionless() -> None:
    """`static_close` (gate 1) uses acceleration as a CEILING, not a floor: it suppresses
    a pair that is close but *not moving*. That is the opposite constraint on the same
    quantity -- it must sit ABOVE the noise of a person standing still, while
    `PairMetrics.acceleration_threshold` must sit above the noise of a person WALKING.

    Raising one does not mean raising the other, and confusing them would silently kill
    the suppression that kills the commonest false positive in the building.
    """
    metrics = PairMetrics()
    gate = _hall().bullying.gates.static_close

    store = TrackStore(smoothing=metrics.tracker_smoothing)
    rng = random.Random(3)  # noqa: S311 - a shaky bounding box, not a cipher
    seen = still = 0

    for index in range(1800):  # three minutes of standing perfectly still
        cx = 640.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        cy = 400.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        store.update({1: Box(x1=cx - 30, y1=cy - 80, x2=cx + 30, y2=cy + 80)}, index * 0.1)

        track = store.tracks[1]
        if track.hits <= 5:
            continue
        seen += 1
        if track.speed < gate.speed_threshold and track.acceleration < gate.acceleration_threshold:
            still += 1

    assert still / seen > 0.90, (
        f"a motionless person reads as moving on {(1 - still / seen) * 100:.0f}% of frames, "
        f"so `static_close` cannot suppress two children standing next to each other"
    )


def _hall() -> BullyingCamera:
    camera = load_cameras()["hall_left"]
    assert isinstance(camera, BullyingCamera)
    return camera


def test_a_standing_person_reads_as_static_on_the_stairs() -> None:
    """**Gate 8 was dead on the only camera type it exists for.**

    `staircase_pass` suppresses "one person standing, one walking past" -- the pattern
    that makes a staircase a staircase, because there is nowhere else to go. It asks
    `a_speed < static_speed_threshold`, and the stairs profile shipped `4.0`: a legacy
    px/FRAME value compared against Track.speed in px/SECOND.

    A person standing perfectly still reads well above 4 px/s from box jitter alone. So
    `a_static` was never true, the gate never fired, and the units migration's single
    remaining miss took out the one suppression the stairs have.

    If this test fails, the VALUE in stairs.yaml is wrong, not the test. The unit is
    px/second and this is what a motionless person costs in it.
    """
    camera = _stairs()
    gate = camera.bullying.gates.staircase_pass
    metrics = camera.bullying.metrics
    fps = _fps(camera)

    store = TrackStore(smoothing=metrics.tracker_smoothing, max_lost=metrics.tracker_max_lost)
    rng = random.Random(11)  # noqa: S311 - a shaky bounding box, not a cipher
    seen = static = 0

    for index in range(int(120 * fps)):  # two minutes of standing perfectly still
        cx = 640.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        cy = 400.0 + rng.uniform(-ASSUMED_JITTER_PX, ASSUMED_JITTER_PX)
        store.update({1: Box(x1=cx - 30, y1=cy - 80, x2=cx + 30, y2=cy + 80)}, index / fps)

        track = store.tracks[1]
        if track.hits <= SETTLED_HITS:
            continue
        seen += 1
        if track.speed < gate.static_speed_threshold:
            static += 1

    assert static / seen > 0.90, (
        f"a person standing still on the stairs reads as MOVING on "
        f"{(1 - static / seen) * 100:.0f}% of frames against "
        f"static_speed_threshold={gate.static_speed_threshold} px/s. `a_static` is then "
        f"never true and gate 8 (staircase_pass) cannot fire at all."
    )


def test_the_stairs_thresholds_are_px_per_second_not_px_per_frame() -> None:
    """The regression this pins: 4.0 / 8.0 are the legacy's px/FRAME numbers, and they
    are what shipped. Anything under ~20 px/s is a px/frame value wearing a px/s label."""
    gate = _stairs().bullying.gates.staircase_pass

    assert gate.static_speed_threshold >= 20.0, "px/frame leaked back into stairs.yaml"
    assert gate.moving_speed_threshold > gate.static_speed_threshold


def _stairs() -> BullyingCamera:
    camera = load_cameras()["stairs_floor1"]
    assert isinstance(camera, BullyingCamera)
    return camera

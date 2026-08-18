"""A clip is not a camera in two ways, and both of them are in `capture/clip.py`.

**Time.** `CameraStream` stamps each frame with `time.monotonic()`, `worker/bullying.py`
hands that stamp to the detector, and every speed the detector compares against a profile
threshold is px per SECOND. A file decodes as fast as the CPU allows, so an unpaced file
delivers frames a millisecond apart where the camera delivered them 67 ms apart -- and
every measured speed comes out ~60x too high while every threshold stays where it was. The
value is right where it is produced and silently wrong one layer on. That is this
codebase's signature defect, and these tests are the ones that stop it.

**The end.** A file has one; RTSP does not.

The clock and the sleep are injected, so these tests assert what the pacing DID rather
than timing it with a stopwatch -- a test that sleeps for real is slow AND is measuring
the machine.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from qorgan.capture.clip import PacedClip, declared_fps, open_paced_clip

FPS = 12.0
PERIOD = 1.0 / FPS


class _Clock:
    """A monotonic clock nobody has to wait for. `sleep` is the only thing that moves it,
    unless a decode is made to cost time explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Handle:
    """The part of `cv2.VideoCapture` that `PacedClip` drives, on a clock we control.

    `decode_cost` is what makes this more than a list: real decoding takes time, and the
    whole question the pacing answers is what the interval between HANDOVERS is once that
    time is accounted for.
    """

    def __init__(self, frames: int, clock: _Clock, *, decode_cost: float = 0.0) -> None:
        self._frames = frames
        self._clock = clock
        self.decode_cost = decode_cost
        self._index = 0
        self.rewinds = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._clock.now += self.decode_cost
        if self._index >= self._frames:
            return False, None
        image = np.full((8, 8, 3), self._index % 255, dtype=np.uint8)
        self._index += 1
        return True, image

    def set(self, _prop: int, _value: float) -> bool:
        self._index = 0
        self.rewinds += 1
        return True

    def get(self, _prop: int) -> float:
        return FPS

    def release(self) -> None:
        self.released = True


def _paced(handle: _Handle, clock: _Clock, *, loop: bool = False) -> PacedClip:
    return PacedClip(handle, FPS, camera="hall_left", loop=loop, sleep=clock.sleep, clock=clock)


def _handover_instants(clip: PacedClip, clock: _Clock, count: int) -> list[float]:
    """The wall-clock moment each frame was handed to the reader. This is the quantity
    that becomes `Frame.captured_at`, and therefore the quantity the detector divides by."""
    out = []
    for _ in range(count):
        ok, _image = clip.read()
        assert ok, "the clip ran out early; this test needs every frame it asked for"
        out.append(clock.now)
    return out


def _gaps(instants: list[float]) -> list[float]:
    return [b - a for a, b in pairwise(instants)]


def _periods(count: int):
    """`count` intervals of exactly one frame period, to floating-point tolerance."""
    return pytest.approx([PERIOD] * count, abs=1e-9)


# -- the frame rate ----------------------------------------------------------


def test_frames_are_handed_over_at_the_rate_the_clip_was_recorded_at() -> None:
    """One second of wall clock must equal one second of the scene that was filmed.

    Not "roughly": exactly one period between handovers, because the detector divides a
    pixel distance by this interval and compares the answer to a threshold nobody is going
    to re-derive.
    """
    clock = _Clock()
    clip = _paced(_Handle(10, clock), clock)

    assert _gaps(_handover_instants(clip, clock, 6)) == _periods(5)


def test_decoding_costs_time_and_the_interval_still_comes_out_right() -> None:
    """The schedule is absolute, not "sleep a period after each decode".

    Sleeping a fixed period AFTER a decode that itself cost 30 ms delivers every frame
    30 ms late and drifts further with each one -- so a clip played for a minute would end
    up denominating its last frames in a different second from its first. `_due` advances
    by exactly one period per frame, so the decode cost comes out of the SLEEP.
    """
    clock = _Clock()
    clip = _paced(_Handle(10, clock, decode_cost=PERIOD * 0.3), clock)

    assert _gaps(_handover_instants(clip, clock, 6)) == _periods(5)
    assert all(sleep < PERIOD for sleep in clock.sleeps), (
        "the decode cost was added to the interval instead of being taken out of the sleep"
    )


def test_a_decoder_that_cannot_keep_up_says_so_rather_than_limping_quietly() -> None:
    """Frames arriving LATER than they were recorded understate every speed by the same
    factor. The pacing cannot fix that -- but a run whose numbers are wrong must not look
    exactly like a run whose numbers are right."""
    clock = _Clock()
    clip = _paced(_Handle(10, clock, decode_cost=PERIOD * 2), clock)

    _handover_instants(clip, clock, 5)

    assert clock.sleeps == [], "it slept while already behind"
    assert clip.late_frames == 4, "falling behind was not counted, so nothing could report it"


def test_time_lost_to_a_slow_decode_is_forgiven_and_not_repaid_in_a_burst() -> None:
    """The failure mode of a naive catch-up, which would be worse than being late.

    Carrying the debt forward means that the moment the machine recovers it fires every
    owed frame back to back with no sleep between them -- frames whose timestamps say they
    are microseconds apart, which is precisely the corruption the pacing exists to prevent.
    So the debt is dropped: after a slow patch, the very next frame waits a full period.
    """
    clock = _Clock()
    handle = _Handle(10, clock, decode_cost=PERIOD * 3)
    clip = _paced(handle, clock)

    _handover_instants(clip, clock, 3)  # three frames the decoder could not keep up with
    handle.decode_cost = 0.0
    recovered = _handover_instants(clip, clock, 3)

    assert _gaps(recovered) == _periods(2), (
        f"the debt was repaid as a burst: {_gaps(recovered)}. Those frames' timestamps "
        "would tell the detector the children moved that distance in no time at all."
    )


def test_a_clip_that_declares_no_frame_rate_is_refused_rather_than_guessed_at() -> None:
    """There is no honest fallback. `capture.stream_fps` is a fact about the CAMERA; using
    it to pace a 25 fps recording would stretch the video's own time by 1.67x and scale
    every px/s with it -- the same defect, wearing the configuration's clothes."""

    class _Mute(_Handle):
        def get(self, _prop: int) -> float:
            return 0.0

    with pytest.raises(OSError, match="no usable frame rate"):
        declared_fps(_Mute(3, _Clock()), Path("hall.mp4"))


# -- the end of the file -----------------------------------------------------


def test_a_looping_clip_starts_over_instead_of_ending() -> None:
    clock = _Clock()
    handle = _Handle(3, clock)
    clip = _paced(handle, clock, loop=True)

    instants = _handover_instants(clip, clock, 7)

    assert handle.rewinds == 2, "the clip did not start over"
    assert clip.loops == 2
    assert _gaps(instants) == _periods(6), (
        "the wrap dropped or doubled a frame interval -- a loop boundary must be an "
        "ordinary frame as far as every timestamp downstream is concerned"
    )


def test_a_stopping_clip_ends_and_stays_ended() -> None:
    clock = _Clock()
    handle = _Handle(2, clock)
    clip = _paced(handle, clock, loop=False)

    assert [clip.read()[0] for _ in range(4)] == [True, True, False, False]
    assert handle.rewinds == 0, "a clip told to stop rewound anyway"


def test_an_empty_clip_ends_even_when_it_was_told_to_loop() -> None:
    """Otherwise the reader thread rewinds and re-reads at full speed, forever, on a file
    that has nothing in it -- a busy loop with no output and no error."""
    clock = _Clock()
    handle = _Handle(0, clock)
    clip = _paced(handle, clock, loop=True)

    assert clip.read() == (False, None)
    assert handle.rewinds == 1, "it did not even try once, or it tried more than once"


# -- opening -----------------------------------------------------------------


def test_a_missing_clip_fails_loudly_and_by_name(tmp_path: Path) -> None:
    missing = tmp_path / "not_here.mp4"

    with pytest.raises(OSError, match="not_here.mp4"):
        open_paced_clip(missing, camera="hall_left", loop=True)

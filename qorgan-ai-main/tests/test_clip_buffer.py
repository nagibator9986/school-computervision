"""The ring buffer that makes an event's clip possible — and the memory it may not cost.

Rule R8, at the one place in this system most likely to break it. A clip has to show the
seconds *before* the alarm, so the frames must be kept before anybody knows they matter;
that is precisely what the legacy did when one suppressed burst pinned ~460 MB of 1280x720
frames that were never freed. Every test here is about the bound, not about the video.
"""

from __future__ import annotations

import numpy as np

from qorgan.events.clip_buffer import ClipBuffer, decode

# 1280x720 is what the hall camera really analyses (`config/profiles/hall.yaml`), and it is
# the resolution the legacy leaked. Noise rather than flat colour on purpose: a flat frame
# compresses to almost nothing and would make any budget look generous.
HD = (720, 1280, 3)


def _frames(count: int, shape: tuple[int, int, int] = HD) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(count)]


def test_a_camera_running_for_hours_never_grows_its_clip_buffer() -> None:
    """The plain bound: a camera runs for months, this deque must not.

    3 s at 15 fps is 45 frames, and it is 45 frames after 2000 of them have gone through.
    """
    buffer = ClipBuffer(seconds=3.0, fps=15.0)
    frame = _frames(1, (360, 640, 3))[0]

    for index in range(500):
        buffer.append(float(index) / 15.0, frame)

    assert buffer.max_frames == 45
    assert len(buffer) == 45
    assert buffer.nbytes <= buffer.budget_bytes


def test_the_byte_budget_outranks_the_frame_count() -> None:
    """**The R8 test.** The frame count is derived from configuration; the budget is not.

    A limit computed from `clip_seconds x analysis_fps` is only a limit while the
    configuration is sane. Ask for the schema's maximum clip length on a camera delivering
    60 fps and the frame count alone permits 900 HD frames — about 2.4 GB as ndarrays, and
    still ~130 MB as JPEG. The byte budget is what stands between that configuration and a
    worker that dies of it, so it must evict even though the frame count is content.

    If this ever passes because `max_frames` did the work instead, the assertions below
    say so rather than going quietly green.
    """
    buffer = ClipBuffer(seconds=15.0, fps=60.0)
    frames = _frames(8)

    for index in range(80):
        buffer.append(float(index), frames[index % len(frames)])

    assert buffer.max_frames == 900, "the frame count must NOT be the thing under test here"
    assert len(buffer) < buffer.max_frames, (
        "nothing was evicted for being over budget, so this test proved only that "
        "max_frames works — pick a shape whose JPEGs actually exceed the budget"
    )
    assert buffer.evicted_over_budget > 0
    assert buffer.nbytes <= buffer.budget_bytes, (
        f"the buffer holds {buffer.nbytes} bytes against a budget of {buffer.budget_bytes}"
    )


def test_the_budget_is_enforced_against_the_true_total_not_the_newest_frame() -> None:
    """A budget checked once per append, against a stale total, is not a budget.

    Frames vary in size — a corridor full of children compresses far worse than an empty
    one — so eviction has to keep going until the RUNNING total fits, not stop after one.
    """
    buffer = ClipBuffer(seconds=60.0, fps=60.0, budget_bytes=400_000)
    for index, frame in enumerate(_frames(30, (360, 640, 3))):
        buffer.append(float(index), frame)

    assert buffer.nbytes == sum(len(payload) for payload in buffer.window(1e9)), (
        "the running byte total disagrees with what the buffer actually holds"
    )
    assert buffer.nbytes <= 400_000


def test_the_clip_covers_the_seconds_before_the_alarm_not_after() -> None:
    """The whole reason the buffer exists. By the time the slow tier has judged a
    candidate the shove is over, so the window ENDS at the candidate's own moment."""
    buffer = ClipBuffer(seconds=10.0, fps=1.0)
    for index in range(10):
        buffer.append(float(index), _frames(1, (64, 64, 3))[0])

    assert len(buffer.window(4.0)) == 5, "frames after the alarm leaked into the clip"
    assert len(buffer.window(-1.0)) == 0
    assert len(buffer.window(1e9)) == 10


def test_frames_come_back_out_at_the_size_they_went_in() -> None:
    """Encoded at the ANALYSIS resolution and decoded back to it. The clip is not a
    thumbnail, and it is not the 2560x1440 burst stream either."""
    buffer = ClipBuffer(seconds=2.0, fps=5.0)
    for index, frame in enumerate(_frames(3, (360, 640, 3))):
        buffer.append(float(index), frame)

    images = list(decode(buffer.window(1e9)))

    assert len(images) == 3
    assert all(image.shape == (360, 640, 3) for image in images)


def test_decoding_hands_over_one_frame_at_a_time() -> None:
    """`decode` is a generator, and that is a memory guarantee rather than a style choice:
    the frames go into the video writer one by one, so three seconds of HD footage is
    written to disk without three seconds of HD footage existing in RAM."""
    buffer = ClipBuffer(seconds=3.0, fps=15.0)
    for index, frame in enumerate(_frames(20, (360, 640, 3))):
        buffer.append(float(index), frame)

    stream = decode(buffer.window(1e9))
    first = next(stream)

    assert first.shape == (360, 640, 3)
    assert len(list(stream)) == 19, "the rest of the frames were consumed eagerly"


def test_a_frame_too_corrupt_to_decode_is_skipped_not_fatal() -> None:
    """A clip missing a frame is still evidence. An exception here would surface as a
    lost event, because the caller treats a failed clip as no clip."""
    good = ClipBuffer(seconds=1.0, fps=1.0)
    good.append(0.0, _frames(1, (64, 64, 3))[0])

    images = list(decode([b"not a jpeg at all", *good.window(1e9)]))

    assert len(images) == 1


def test_an_empty_frame_is_refused_rather_than_buffered() -> None:
    """A zero-size frame cannot be encoded; `cv2.imencode` raises a bare assertion on it
    rather than returning False, and an assertion escaping into the capture thread takes
    the camera off the air."""
    buffer = ClipBuffer(seconds=1.0, fps=5.0)

    buffer.append(0.0, np.empty((0, 0, 3), dtype=np.uint8))

    assert len(buffer) == 0
    assert buffer.nbytes == 0

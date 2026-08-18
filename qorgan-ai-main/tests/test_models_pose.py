"""`build_crops` must be tied to the frame the candidate actually came from.

**The defect.** Both the worker and the harness keep a rolling
`deque(maxlen=CROP_BUFFER)` of recent frames and cut crops out of "whatever is in the
buffer right now". That is correct only by accident: today, every candidate is judged
synchronously, in the same loop iteration that appended its frame, so "right now" always
happens to still be the candidate's own moment. Batch the candidates, judge them after the
loop, parallelise, add a queue that drains slowly -- and `deque(maxlen=...)` has silently
evicted the candidate's own frames and replaced them with someone else's. The pose model
still finds skeletons in those frames (they are real frames, just the wrong ones) and
`judge()` still produces a confident verdict. Nothing raises. You get a plausible, wrong
answer with no error.

This test simulates that future batch caller directly against the buffer/`build_crops`
mechanics both callers share, and proves that a stale request raises instead of silently
returning the last 24 frames.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from qorgan.detection.geometry import Box
from qorgan.models.pose import CROP_BUFFER, CropProvenanceError, build_crops

BOXES = (Box(0, 0, 10, 10), Box(5, 5, 15, 15))


def _frame(fill: int) -> np.ndarray:
    return np.full((20, 20, 3), fill, dtype=np.uint8)


def test_build_crops_cuts_one_crop_per_buffered_frame() -> None:
    buffer = [(float(i), _frame(i)) for i in range(3)]

    crops = build_crops(buffer, BOXES, candidate_timestamp=2.0)

    assert len(crops) == 3


def test_build_crops_accepts_a_timestamp_still_inside_the_window() -> None:
    buffer = [(float(i), _frame(i)) for i in range(CROP_BUFFER)]

    crops = build_crops(buffer, BOXES, candidate_timestamp=buffer[-1][0])

    assert len(crops) == CROP_BUFFER


def test_build_crops_raises_when_a_batched_caller_judges_after_the_buffer_moved_on() -> None:
    """**Simulates the future batch caller.**

    A batch caller would collect candidates while iterating the frame source, then build
    their crops afterwards. Play out exactly that: append frames to a real
    `deque(maxlen=CROP_BUFFER)` (as the worker and the harness both do), remember a
    candidate's timestamp from early in the run, keep the loop going well past
    `CROP_BUFFER` more frames, THEN ask for that candidate's crops -- after its own frames
    have been silently evicted.
    """
    recent: deque[tuple[float, np.ndarray]] = deque(maxlen=CROP_BUFFER)

    for i in range(CROP_BUFFER):
        recent.append((float(i), _frame(i)))
    candidate_timestamp = recent[-1][0]  # born on the last frame of this window

    # The loop moves on -- a full buffer's worth of newer frames arrive before this
    # candidate is judged, exactly as a batched/queued/parallel caller would allow.
    for i in range(CROP_BUFFER, CROP_BUFFER * 2):
        recent.append((float(i), _frame(i)))

    with pytest.raises(CropProvenanceError) as excinfo:
        build_crops(list(recent), BOXES, candidate_timestamp)

    message = str(excinfo.value)
    assert "0" in message or f"{candidate_timestamp:.3f}" in message
    assert "buffer" in message.lower()


def test_build_crops_error_names_what_was_asked_for_and_what_the_buffer_holds() -> None:
    recent = [(float(i), _frame(i)) for i in range(CROP_BUFFER, CROP_BUFFER * 2)]

    with pytest.raises(CropProvenanceError) as excinfo:
        build_crops(recent, BOXES, candidate_timestamp=0.0)

    message = str(excinfo.value)
    assert "0.000" in message  # what was asked for
    assert f"{float(CROP_BUFFER):.3f}" in message  # what the buffer actually holds


def test_build_crops_raises_on_an_empty_buffer_rather_than_returning_nothing_quietly() -> None:
    with pytest.raises(CropProvenanceError):
        build_crops([], BOXES, candidate_timestamp=1.0)

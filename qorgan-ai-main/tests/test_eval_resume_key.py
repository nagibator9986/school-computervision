"""One key, computed one way -- the labeller's and the sampler's dedup must not drift.

`eval label` and `eval sample` both ask "has a human already settled this candidate?" and
they must answer it identically. They answer it with `resume_key`.

The trap this file exists to hold shut: the LIVE path holds the candidate's raw timestamp,
while the RESUME path only ever sees what labels.csv wrote -- 2-dp text. `scan.rows_for`
stores timestamps at 3 dp, so the two paths meet across a decimal round-trip, and a key
built on either side of it independently can round the binary half opposite ways. 166 of
the 10 000 3-dp timestamps in [0, 10) do exactly that.

What it costs, when it happens: a settled candidate is not recognised on resume, so it is
re-offered, so labels.csv gets a SECOND interval for one scuffle, so `total_fights` counts
that scuffle twice, so RECALL is reported LOWER than it is. Nothing is lost silently -- the
count is corrupted, which is worse, because a corrupted count still looks like an answer.

Today's corpus is 10.0 / 8.0 / 5.0 fps, and none of those reach the seam. 29.97 fps does
(t=1.535 -> live key 1.53, stored key 1.54), 29.97 is the most common real-world video
rate, and the school will send more footage. So these tests guard a LATENT trap, deliberately.
"""

from __future__ import annotations

from pathlib import Path

from qorgan.config.camera import BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.evaluation.clips import camera_for
from qorgan.evaluation.labelling import (
    already_labelled,
    append_label,
    interval_for,
    is_done,
)
from qorgan.evaluation.labels import LabelKind, load_labels
from qorgan.evaluation.sampling import draw
from qorgan.evaluation.scan import ScanRow

# A 29.97-fps-style moment. `round(1.535, 2)` is 1.53 (1.535 is not exactly representable
# in binary and the nearest double sits just below the half), but the interval END that
# labels.csv carries is `f"{3.535:.2f}"` == "3.54", and 3.54 - 2.0 keys as 1.54. The two
# paths disagree here, and this timestamp is the whole point of the file.
SEAM_TIMESTAMP = 1.535

CLIP = "hall_left_main_1009_1019_burst101_20260702_144150_952947.mp4"


def _cameras(clips: list[str]) -> dict[str, BullyingCamera]:
    return {clip: camera_for(clip, load_cameras()) for clip in clips}


def _candidate(timestamp: float) -> ScanRow:
    return ScanRow(CLIP, timestamp, 1.9, 0.8, 0.91)


def _settle(path: Path, row: ScanRow, kind: LabelKind = LabelKind.NORMAL) -> None:
    """Answer the candidate exactly as `eval label` does -- through the real file."""
    append_label(path, interval_for(row, kind))


def test_a_candidate_settled_at_a_29_97_moment_is_recognised_on_resume(tmp_path: Path) -> None:
    """The labeller's own resume, across the CSV round-trip it really does at lunchtime.

    Without this, the candidate is re-offered, the human answers it twice, and labels.csv
    carries one scuffle as two intervals.
    """
    labels = tmp_path / "labels.csv"
    row = _candidate(SEAM_TIMESTAMP)
    _settle(labels, row)

    assert is_done(row, already_labelled(labels)), (
        "a candidate the human already settled was not recognised on resume -- it will be "
        "re-offered and written to labels.csv twice, double-counting one fight"
    )


def test_the_sampler_does_not_re_propose_a_candidate_settled_at_a_29_97_moment(
    tmp_path: Path,
) -> None:
    """The sampler's `_judged` reads the same file and must reach the same verdict.

    The LabelSet comes from `load_labels` on a real labels.csv, not from an in-memory
    `Interval`: the seam only exists across the 2-dp text, so a hand-built interval would
    pass while the real resume still failed.
    """
    labels = tmp_path / "labels.csv"
    row = _candidate(SEAM_TIMESTAMP)
    _settle(labels, row)

    rows = draw(
        clips=[CLIP],
        candidates=[row],
        cameras=_cameras([CLIP]),
        coverage={CLIP},
        labelled=load_labels(labels),
        count=0,
    )

    assert rows == [], (
        "the sampler re-proposed a candidate a human had already settled -- the worklist "
        "and labels.csv disagree about what 'unjudged' means"
    )

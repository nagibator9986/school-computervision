"""`qorgan eval sample`: silence must be PROVEN by the coverage manifest, never inferred.

The signature bug of this project, in the sampler: `_silent` called a clip silent purely
because it had no row in candidates.csv. But a stale candidates.csv, a truncated scan, or a
clip added to the clips dir without a re-scan produces the SAME zero rows -- a clip the
detector NEVER PROCESSED, filed as "detector saw nothing". A real fight in an unscanned clip
would be filed silent and never sampled. `eval scan` now writes a coverage manifest beside
candidates.csv, and `eval sample` cross-checks it: a clip is silent only when the manifest
PROVES it was scanned and it produced no candidate. Everything else is a hard error.

The shared corpus helpers live in `test_eval_sample`; this module is the coverage contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qorgan.evaluation import cli as eval_cli
from qorgan.evaluation import scanning
from qorgan.evaluation.sampling import SampleError, Stratum
from qorgan.evaluation.scan import (
    coverage_path,
    load_candidates,
    write_candidates,
    write_coverage,
)
from tests.test_eval_sample import ALERT_CONFIDENCE, _candidate, _clip, _clips, _draw


def test_a_covered_clip_with_no_candidate_is_proven_silent() -> None:
    """The ONLY thing that may be called silent: a clip the manifest proves was scanned and
    that produced no candidate. Proven silence still works -- with no invented timestamp."""
    clips = _clips(3)

    rows = _draw(clips, [], coverage=set(clips), count=10)

    assert {row.stratum for row in rows} == {Stratum.SILENT}
    assert sorted(row.row.clip for row in rows) == sorted(clips)
    assert all(row.row.timestamp is None for row in rows), "a silent clip got an invented moment"


def test_a_clip_absent_from_the_coverage_manifest_is_unscanned_not_silent() -> None:
    """A clip added to the clips dir without a re-scan. It has no candidate row, so the old
    code called it silent. It was never processed: its silence is UNKNOWN, and unknown is a
    hard error naming the clip -- never a plausible 'silent' default, which is the bug."""
    clips = _clips(3)  # clips[2] is in the dir but the manifest covers only [0, 1]

    with pytest.raises(SampleError) as excinfo:
        _draw(clips, [], coverage={clips[0], clips[1]}, count=10)

    message = str(excinfo.value)
    assert clips[2] in message, "the error must name the unscanned clip"
    assert "coverage manifest" in message
    assert clips[0] not in message and clips[1] not in message, "it faulted a covered clip"


def test_a_candidate_clip_missing_from_the_manifest_is_a_hard_error() -> None:
    """Every clip with a candidate row MUST appear in the manifest, or the two artifacts are
    inconsistent (a stale candidates.csv against a fresh manifest, or vice versa). That is a
    hard error naming the clip, never silently tolerated."""
    clips = _clips(2)
    candidate = _candidate(clips[0], ALERT_CONFIDENCE)

    with pytest.raises(SampleError, match="inconsistent") as excinfo:
        _draw(clips, [candidate], coverage={clips[1]}, count=0)

    assert clips[0] in str(excinfo.value), "the error must name the inconsistent clip"


def test_a_missing_coverage_manifest_is_a_hard_error_telling_the_user_to_re_scan() -> None:
    """An old candidates.csv from before coverage was recorded: the manifest is missing
    entirely (`coverage=None`). This must NOT fall back to the old absence==silent guess --
    a guessed silence is exactly the bug. It is a hard, actionable error: re-run the scan."""
    clips = _clips(3)

    with pytest.raises(SampleError, match="eval scan") as excinfo:
        _draw(clips, [], coverage=None, count=10)

    assert "manifest" in str(excinfo.value)


# -- the CLI wiring: cmd_sample loads the manifest beside --candidates ----------


def _placeable_clips(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "clips"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def _sample_args(tmp_path: Path, clips_dir: Path, candidates: Path) -> argparse.Namespace:
    return argparse.Namespace(
        clips=clips_dir,
        candidates=candidates,
        out=tmp_path / "sample.csv",
        labels=tmp_path / "no-labels.csv",
        count=10,
        seed=7,
    )


def test_cmd_sample_reads_the_manifest_beside_the_candidates_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: with candidates.csv AND its coverage manifest present, a covered clip
    with no candidate is proposed as proven-silent and the worklist is written."""
    monkeypatch.setattr(scanning, "_frame_size", lambda path: (2560, 1440))
    clip = _clip(0)
    clips_dir = _placeable_clips(tmp_path, clip)
    candidates = tmp_path / "candidates.csv"
    write_candidates([], candidates)
    write_coverage([clip], coverage_path(candidates))

    assert eval_cli.cmd_sample(_sample_args(tmp_path, clips_dir, candidates)) == 0

    loaded = load_candidates(tmp_path / "sample.csv")
    assert [row.clip for row in loaded] == [clip]


def test_cmd_sample_without_a_manifest_exits_with_an_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest missing beside candidates.csv is a clean SystemExit, not a traceback --
    the user is told to re-run the scan, never handed a silently guessed sample."""
    monkeypatch.setattr(scanning, "_frame_size", lambda path: (2560, 1440))
    clip = _clip(0)
    clips_dir = _placeable_clips(tmp_path, clip)
    candidates = tmp_path / "candidates.csv"
    write_candidates([], candidates)  # candidates.csv, but NO coverage manifest beside it

    with pytest.raises(SystemExit, match="eval scan") as excinfo:
        eval_cli.cmd_sample(_sample_args(tmp_path, clips_dir, candidates))

    assert not (tmp_path / "sample.csv").exists(), "it wrote a sample from unproven silence"
    assert "manifest" in str(excinfo.value)


def _fail_on(clip: Path, bad: str) -> list:
    if clip.name == bad:
        raise OSError(f"cannot open video: {bad}")
    return []


def test_a_clip_the_scan_could_not_read_stops_the_sampler_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the count of unreadable clips ENDS UP -- and why surviving a bad clip is safe.

    `eval scan` no longer dies on a clip it cannot read: it names it and carries on with the
    other 656. The danger in that is silence, because a skipped clip has no candidate row,
    and absence from candidates.csv is exactly what used to be read as "the detector saw
    nothing here". It is not: nothing decoded it.

    The manifest keeps the two apart with no extra machinery. The scan leaves the unread clip
    OUT of it, so `eval sample` finds a clip in the directory that nothing proves was
    scanned, and refuses by name. The count cannot be lost between the two commands, and it
    cannot be quietly folded into the silent stratum -- the one place a real fight could hide
    and never be sampled.
    """
    good, bad = _clip(0), _clip(1)
    clips_dir = _placeable_clips(tmp_path, good, bad)
    monkeypatch.setattr(scanning, "_frame_size", lambda path: (2560, 1440))
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)
    monkeypatch.setattr(
        scanning, "_scan_one", lambda clip, camera, pose, device: _fail_on(clip, bad)
    )
    candidates = tmp_path / "candidates.csv"
    scan_args = argparse.Namespace(
        clips=clips_dir, labels=tmp_path / "no-labels.csv", out=candidates, device="cpu"
    )

    assert scanning.cmd_scan(scan_args) == 1

    with pytest.raises(SystemExit) as excinfo:
        eval_cli.cmd_sample(_sample_args(tmp_path, clips_dir, candidates))

    assert bad in str(excinfo.value), "the sampler drew on without naming the unread clip"
    assert good not in str(excinfo.value), "it faulted the clip that scanned fine"
    assert not (tmp_path / "sample.csv").exists(), "it sampled a corpus it had not read"

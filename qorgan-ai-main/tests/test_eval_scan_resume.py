"""`qorgan eval scan` survives the corpus: it writes as it goes, resumes, and names what
it could not read.

The first real run over the school's 657 clips crashed on clip 170 -- the host could not
allocate the 10.5 MB for one decoded frame -- and produced NO output file at all. 169 clips
of GPU time vanished, because both artifacts were written once, at the end. The crash is one
bug; the shape that turned it into total loss is another, and it would have cost the same
for a closed laptop lid or an impatient Ctrl+C.

Three properties, and every one of them is here because a 657-clip job has no business
being all-or-nothing:

  * the result exists on disk after every clip, not after the last one;
  * a re-run continues from where the last one stopped, and never doubles a row;
  * a clip that cannot be read is NAMED, is not certified as covered, and is COUNTED in the
    summary -- because a corpus that quietly skipped 200 files and one that read them look
    exactly alike in candidates.csv, and telling those apart is the whole job.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from qorgan.evaluation import scan as scan_module
from qorgan.evaluation import scanning
from qorgan.evaluation.scan import (
    ScanRow,
    coverage_path,
    load_candidates,
    load_coverage,
    load_unreadable,
    unreadable_path,
    write_candidates,
)
from qorgan.evaluation.scanning import INTERRUPTED_EXIT, cmd_scan

CLIP = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
RIGHT = "hall_right_main_212_233_burst101_20260702_101530_101010.mp4"
THIRD = "hall_left_main_223_246_20260514_140739_409208.mp4"
FULL_FRAME = (2560, 1440)

# The scan visits clips in sorted-name order, so the three above run CLIP, THIRD, RIGHT.
# The tests below stop on RIGHT when they mean 'the last one'.


@pytest.fixture(autouse=True)
def _no_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clip files here are empty and no pose model is loaded: this module is about the
    bookkeeping around `_scan_one`, never about what `_scan_one` finds."""
    monkeypatch.setattr(scanning, "_frame_size", lambda path: FULL_FRAME)
    monkeypatch.setattr(scanning, "_pose", lambda camera, device: None)


def _clips(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "clips"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def _args(tmp_path: Path, *names: str) -> argparse.Namespace:
    return argparse.Namespace(
        clips=_clips(tmp_path, *names),
        labels=tmp_path / "no-labels.csv",
        out=tmp_path / "eval" / "candidates.csv",
        device="cpu",
    )


def _one_row(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
    return [ScanRow(clip.name, 2.4, 1.83, 0.71, 0.66)]


# -- written as it goes ----------------------------------------------------


def test_each_clip_is_on_disk_before_the_next_one_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the failed run did not have. Not "the file exists at the end" -- the
    file has to exist DURING, or an interruption at clip 170 keeps nothing.

    Measured from inside the scan: each clip records how many clips the manifest on disk
    already certified when its own scan began."""
    args = _args(tmp_path, CLIP, RIGHT, THIRD)
    seen: list[int] = []

    def _watching(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        seen.append(len(load_coverage(coverage_path(args.out)) or set()))
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _watching)

    assert cmd_scan(args) == 0
    assert seen == [0, 1, 2], "the artifacts were written once at the end, not as it went"


def test_a_completed_scan_still_writes_every_artifact(tmp_path: Path, monkeypatch) -> None:
    """Writing incrementally must not have cost the end state: all three artifacts exist,
    the candidate rows are all there, and the unreadable list is present and EMPTY -- an
    empty file is the recorded fact "nothing failed", which a missing file is not."""
    monkeypatch.setattr(scanning, "_scan_one", _one_row)
    args = _args(tmp_path, CLIP, RIGHT)

    assert cmd_scan(args) == 0

    assert {row.clip for row in load_candidates(args.out)} == {CLIP, RIGHT}
    assert load_coverage(coverage_path(args.out)) == {CLIP, RIGHT}
    assert unreadable_path(args.out).is_file()
    assert load_unreadable(unreadable_path(args.out)) == {}


# -- resumable -------------------------------------------------------------


def test_a_second_run_scans_only_what_the_first_one_did_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """169 clips of GPU time is not something to spend twice. The coverage manifest already
    records exactly which clips ran to completion, so it is what the resume skips -- the
    same shape `eval label` uses, where labels.csv is both the result and the progress."""
    args = _args(tmp_path, CLIP, RIGHT, THIRD)

    def _stop_after_two(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == RIGHT:
            raise KeyboardInterrupt
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _stop_after_two)
    assert cmd_scan(args) == INTERRUPTED_EXIT
    assert load_coverage(coverage_path(args.out)) == {CLIP, THIRD}

    rescanned: list[str] = []

    def _second(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        rescanned.append(clip.name)
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _second)
    assert cmd_scan(args) == 0

    assert rescanned == [RIGHT], "it re-decoded clips the manifest already proved scanned"
    assert load_coverage(coverage_path(args.out)) == {CLIP, RIGHT, THIRD}


def test_resuming_does_not_write_a_clips_candidates_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that appended blindly would double every row of the first run. The rows of
    the covered clips are carried forward from the file itself, and only the clips that
    were never finished are scanned again."""
    args = _args(tmp_path, CLIP, RIGHT)

    def _stop_after_one(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == RIGHT:
            raise KeyboardInterrupt
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _stop_after_one)
    cmd_scan(args)
    monkeypatch.setattr(scanning, "_scan_one", _one_row)
    cmd_scan(args)

    assert [row.clip for row in load_candidates(args.out)] == [CLIP, RIGHT]


def test_ctrl_c_leaves_a_result_that_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C is the operator, not a fault. It must leave a usable partial answer -- a
    candidates file and a manifest that agree and parse -- never a truncated one, which
    reads back cleanly and short and is this project's signature failure."""
    args = _args(tmp_path, CLIP, RIGHT, THIRD)

    def _stop_at_the_last(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == RIGHT:
            raise KeyboardInterrupt
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _stop_at_the_last)

    assert cmd_scan(args) == INTERRUPTED_EXIT, "an interrupted scan claimed it finished"

    covered = load_coverage(coverage_path(args.out))
    assert covered == {CLIP, THIRD}
    assert {row.clip for row in load_candidates(args.out)} == covered
    assert not args.out.with_name(f"{args.out.name}.partial").exists(), "left a half-written file"


def test_a_candidates_file_with_no_manifest_beside_it_is_not_resumed_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old candidates.csv from before coverage was recorded. `load_coverage` returns None
    exactly because nobody can know what it covered, so resuming from it is impossible --
    and keeping its rows while re-scanning every clip would double each one. It is
    discarded, and every clip is scanned afresh."""
    monkeypatch.setattr(scanning, "_scan_one", _one_row)
    args = _args(tmp_path, CLIP)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "clip,timestamp,score,probability,confidence\n" f"{CLIP},2.4,1.83,0.71,0.66\n",
        encoding="utf-8",
    )
    assert load_coverage(coverage_path(args.out)) is None

    assert cmd_scan(args) == 0

    assert len(load_candidates(args.out)) == 1, "the stale file was resumed from and doubled"
    assert load_coverage(coverage_path(args.out)) == {CLIP}


# -- writing 657 times means the rename fails sometimes --------------------


def test_a_write_survives_the_destination_being_held_open_for_a_moment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found by running it, not by reading it. The first real resumable scan died at clip
    104 with `PermissionError: [WinError 5]` out of `os.replace` -- because this session was
    reading the coverage manifest at that instant to see how far it had got, and Python's
    `open()` asks for no FILE_SHARE_DELETE, so an ordinary reader blocks a rename on Windows.

    Writing three files 657 times makes that a routine event, not a race worth ignoring:
    a virus scanner or a search indexer is enough. Momentary is momentary -- it retries."""
    attempts: list[int] = []
    real = os.replace

    def _busy_twice(src: object, dst: object) -> None:
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(scan_module.os, "replace", _busy_twice)
    monkeypatch.setattr(scan_module, "REPLACE_BACKOFF_SECONDS", 0.0)
    path = tmp_path / "candidates.csv"

    assert write_candidates([ScanRow(CLIP, 2.4, 1.83, 0.71, 0.66)], path) == 1

    assert len(attempts) == 3, "it gave up on a lock that cleared"
    assert [row.clip for row in load_candidates(path)] == [CLIP]


def test_a_destination_that_never_frees_up_is_raised_not_waited_out_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is bounded, and the give-up is loud. A scan that cannot write its result
    down must say so and stop -- retrying in silence for the rest of the corpus would burn
    657 clips of GPU and leave the same nothing the crash left."""
    attempts: list[int] = []

    def _always_busy(src: object, dst: object) -> None:
        attempts.append(1)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(scan_module.os, "replace", _always_busy)
    monkeypatch.setattr(scan_module, "REPLACE_BACKOFF_SECONDS", 0.0)

    with pytest.raises(PermissionError):
        write_candidates([], tmp_path / "candidates.csv")

    assert len(attempts) == scan_module.REPLACE_ATTEMPTS


# -- a clip that cannot be read --------------------------------------------


def _one_bad_clip(bad: str):
    def _scan(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == bad:
            raise OSError(f"cannot open video: {bad}")
        return _one_row(clip, camera, pose, device)

    return _scan


def test_an_unreadable_clip_is_named_and_the_rest_of_the_corpus_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file must not cost the other 656. It is recorded by name WITH the reason -- a
    `MemoryError` and an `OSError` on the same clip mean opposite things, one the machine
    and one the file, and a bare "failed" cannot tell an operator which to go and fix."""
    monkeypatch.setattr(scanning, "_scan_one", _one_bad_clip(RIGHT))
    args = _args(tmp_path, CLIP, RIGHT, THIRD)

    assert cmd_scan(args) == 1

    failures = load_unreadable(unreadable_path(args.out))
    assert list(failures) == [RIGHT]
    assert "OSError" in failures[RIGHT] and "cannot open video" in failures[RIGHT]
    assert {row.clip for row in load_candidates(args.out)} == {CLIP, THIRD}


@pytest.mark.parametrize(
    "failure",
    [
        SystemError("<method 'read' of 'cv2.VideoCapture' objects> returned a result "
                    "with an exception set"),
        MemoryError("Unable to allocate 10.5 MiB for an array with shape (1440, 2560, 3)"),
        OSError("cannot open video"),
        ValueError("some decoder's idea of a bad file"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_the_kinds_of_failure_that_actually_happen_are_all_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """The catch around a clip is broad ON PURPOSE, and this is the list that made it so.

    The corpus was lost to the FIRST of these -- a bare `SystemError`, which is only how
    CPython reports that a C function returned with a Python error already set; the real
    error, two lines up the log, was the second. A decoder can also fail as an `OSError`
    from a truncated container or a `cv2.error` from a codec, and a hand-written tuple of
    the classes seen so far is a tuple that is wrong the next time a driver is updated.

    Each one must be recorded with its TYPE, because they call for opposite responses: the
    MemoryError is the machine and the OSError is the file."""
    def _raise(clip: Path, camera: object, pose: object, device: str) -> list[ScanRow]:
        if clip.name == RIGHT:
            raise failure
        return _one_row(clip, camera, pose, device)

    monkeypatch.setattr(scanning, "_scan_one", _raise)
    args = _args(tmp_path, CLIP, RIGHT)

    assert cmd_scan(args) == 1, f"{type(failure).__name__} took the whole corpus down"

    failures = load_unreadable(unreadable_path(args.out))
    assert list(failures) == [RIGHT]
    assert failures[RIGHT].startswith(type(failure).__name__)
    assert load_coverage(coverage_path(args.out)) == {CLIP}


def test_an_unreadable_clip_is_not_certified_as_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing that must never happen: a clip nothing decoded, filed among the clips
    the detector watched and found nothing in. Absence from candidates.csv would then read
    as proven silence, and a real fight inside it would never be sampled."""
    monkeypatch.setattr(scanning, "_scan_one", _one_bad_clip(RIGHT))
    args = _args(tmp_path, CLIP, RIGHT)

    cmd_scan(args)

    assert load_coverage(coverage_path(args.out)) == {CLIP}, "an unread clip was certified"


def test_the_count_of_unreadable_clips_reaches_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The number has to appear where a human reads it. A scan that skipped 200 clips and a
    scan that read them produce candidate files that look identical, and the difference --
    which is the difference between a measured corpus and an imagined one -- exists only if
    somebody is told. The exit status carries it too, for whoever is not watching."""
    monkeypatch.setattr(scanning, "_scan_one", _one_bad_clip(RIGHT))
    args = _args(tmp_path, CLIP, RIGHT)

    assert cmd_scan(args) == 1, "a scan that could not read a clip reported success"

    printed = capsys.readouterr().out
    assert "1 clip(s) COULD NOT BE READ" in printed
    assert RIGHT in printed, "it counted the failure but did not name it"
    assert "1/2 clip(s) proven scanned" in printed, "the coverage was reported as complete"


def test_a_clip_that_reads_on_the_second_run_leaves_the_unreadable_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure that stopped the real run was the HOST running out of memory, not a bad
    file: the same clip opened fine on its own afterwards. So a clip listed unreadable is
    always retried -- it is not covered, so a resume reaches it -- and its entry is dropped
    when it reads, rather than left behind to accuse a file that was never at fault."""
    args = _args(tmp_path, CLIP, RIGHT)
    monkeypatch.setattr(scanning, "_scan_one", _one_bad_clip(RIGHT))
    assert cmd_scan(args) == 1
    assert list(load_unreadable(unreadable_path(args.out))) == [RIGHT]

    monkeypatch.setattr(scanning, "_scan_one", _one_row)
    assert cmd_scan(args) == 0

    assert load_unreadable(unreadable_path(args.out)) == {}
    assert load_coverage(coverage_path(args.out)) == {CLIP, RIGHT}

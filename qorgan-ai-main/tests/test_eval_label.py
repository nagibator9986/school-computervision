"""`qorgan eval label`: watch the crop, record against the full frame.

A human deciding "is this a fight?" does not need the scene. They need the two children,
close up -- which is what the old detector's crop ROI is, and it is far faster to watch
than a 1440p wide shot in which the pair occupies 3% of the pixels.

But the crop has no scene, no zones and no frame geometry, so it can never be detector
input. Two views of one incident, each used for what it is good at. The label is always
written against the FULL FRAME.

No test here opens a real video player: the launch is injected. A test suite that shells
out to the OS on a developer's desktop is not a test suite, it is a slideshow.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from qorgan.config.loader import load_cameras
from qorgan.evaluation.labelling import (
    already_labelled,
    append_label,
    cmd_label,
    crop_partner,
    interval_for,
    is_done,
)
from qorgan.evaluation.labels import Interval, LabelError, LabelKind, load_labels
from qorgan.evaluation.scan import ScanRow, write_candidates

BURST = "hall_left_main_1009_1019_burst101_20260702_144158_552815.mp4"
CROP = "hall_left_main_1009_1019_20260702_144150_952947.mp4"
CROP_LATER = "hall_left_main_1009_1019_20260702_150000_000000.mp4"
OTHER_PAIR = "hall_left_main_77_88_20260702_144151_000000.mp4"
OTHER_CAMERA = "hall_right_main_1009_1019_20260702_144151_000000.mp4"

LONELY = "hall_right_main_212_233_burst101_20260702_101530_101010.mp4"

# The recorder's OTHER naming shape: no `_main` between the camera and the track pair.
# Both shapes are in the corpus, on both sides of the join.
STAIRS_BURST = "stairs_floor2_196_322_burst101_20260518_141523_230173.mp4"
STAIRS_CROP = "stairs_floor2_196_322_20260518_141519_100000.mp4"


def _cameras() -> list[str]:
    """The configured camera names -- what the pair-key parser measures a name against."""
    return list(load_cameras())


# -- the join: (camera, track_a, track_b) + nearest timestamp ---------------


def test_the_crop_partner_is_found_on_camera_pair_and_nearest_time() -> None:
    partner = crop_partner(BURST, [OTHER_CAMERA, OTHER_PAIR, CROP_LATER, CROP], _cameras())

    assert partner == CROP, "the join is (camera, track_a, track_b) + NEAREST timestamp"


def test_a_clip_with_no_main_segment_finds_its_crop_partner_too() -> None:
    """Both naming shapes are real, so the pair-key parser must read a camera name with or
    without `_main` after it. Without this, every stairs clip fell out of the join."""
    partner = crop_partner(STAIRS_BURST, [CROP, STAIRS_CROP], _cameras())

    assert partner == STAIRS_CROP


def test_a_full_frame_with_no_crop_partner_falls_back_to_itself() -> None:
    """The labeller then watches the wide shot -- slower, but it is a label, and a missing
    label is worse than a slow one."""
    assert crop_partner(BURST, [OTHER_CAMERA, OTHER_PAIR], _cameras()) is None


def test_a_burst_is_never_its_own_crop_partner() -> None:
    """A crop is by definition the non-burst view. Matching a burst against a burst would
    hand the labeller the very wide shot the crop exists to spare them -- and would call it
    a crop while doing it."""
    assert crop_partner(BURST, [BURST], _cameras()) is None


def test_an_unparsable_crop_name_is_skipped_not_crashed_on() -> None:
    """Human-named clips cannot be joined; they must not take the whole run down."""
    assert crop_partner(BURST, ["драка.mp4", CROP], _cameras()) == CROP


def test_a_crop_whose_camera_prefixes_another_is_never_joined_to_the_wrong_one() -> None:
    """`stairs_floor2_second` is not a configured camera. A crop named for it must not be
    handed to a `stairs_floor2` clip: the human would label one camera's incident against
    another camera's video_id."""
    impostor = "stairs_floor2_second_196_322_20260515_131458_435514.mp4"

    assert crop_partner(STAIRS_BURST, [impostor], _cameras()) is None


def test_an_unparsable_full_frame_name_has_no_partner_and_does_not_raise() -> None:
    assert crop_partner("драка.mp4", [CROP], _cameras()) is None


# -- the record: always the full frame -------------------------------------


def test_the_label_is_written_against_the_FULL_FRAME_whichever_view_was_watched() -> None:
    """The crop is a lens, never the record. `eval run` scores the full frame, so a label
    against a crop's video_id would match nothing and read as a missed fight."""
    row = ScanRow(BURST, 4.25, 1.9, 0.8, 0.88)
    interval = interval_for(row, LabelKind.BULLYING)

    assert interval.video_id == BURST
    assert interval.start == 2.25
    assert interval.end == 6.25
    assert interval.kind is LabelKind.BULLYING


def test_a_label_at_time_zero_does_not_start_before_the_clip_does() -> None:
    assert interval_for(ScanRow(BURST, 0.5, 1.9, 0.8, 0.88), LabelKind.NORMAL).start == 0.0


def test_labels_are_appended_and_the_file_stays_readable(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    append_label(path, interval_for(ScanRow(BURST, 4.0, 1.9, 0.8, 0.88), LabelKind.BULLYING))
    append_label(path, interval_for(ScanRow(BURST, 30.0, 1.5, 0.6, 0.60), LabelKind.NORMAL))

    labels = load_labels(path)

    assert len(labels) == 2
    assert len(labels.positives(BURST)) == 1


def test_append_writes_the_camera_and_never_a_ragged_row(tmp_path: Path) -> None:
    """The live labels.csv has FIVE columns. `append_label` wrote four-field rows, so every
    append against it produced a ragged row and the camera silently became None -- and the
    human-named clips (one of which is the only confirmed fight) became un-scannable."""
    path = tmp_path / "labels.csv"
    path.write_text(
        "video_id,t_start,t_end,label,camera\nprior.mp4,0.0,9.0,normal,hall_right\n",
        encoding="utf-8",
    )

    append_label(path, Interval("fight.mp4", 2.0, 6.0, LabelKind.BULLYING, camera="hall_left"))

    rows = [
        line.split(",")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    assert all(len(r) == 5 for r in rows), "a ragged row: the camera column fell off"
    assert load_labels(path).camera_for("fight.mp4") == "hall_left"


def test_a_fresh_labels_file_gets_the_five_column_header(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    append_label(path, interval_for(ScanRow(BURST, 4.0, 1.9, 0.8, 0.88), LabelKind.BULLYING))

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["video_id", "t_start", "t_end", "label", "camera"]


def test_appending_to_a_file_with_different_columns_is_an_error(tmp_path: Path) -> None:
    """A four-column legacy file cannot take a five-field row without going ragged. That is
    an error, not something to paper over: `csv.DictReader` would tolerate the ragged row and
    the camera would slide silently to None."""
    path = tmp_path / "labels.csv"
    path.write_text("video_id,t_start,t_end,label\nprior.mp4,0.0,9.0,normal\n", encoding="utf-8")

    with pytest.raises(LabelError, match="header"):
        append_label(path, Interval("fight.mp4", 2.0, 6.0, LabelKind.BULLYING))


def test_labelling_is_resumable(tmp_path: Path) -> None:
    """97 minutes of footage is more than one sitting. A labeller who stops for lunch must
    not come back to the beginning."""
    path = tmp_path / "labels.csv"
    done_row = ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)
    todo_row = ScanRow(BURST, 30.0, 1.5, 0.6, 0.60)
    append_label(path, interval_for(done_row, LabelKind.BULLYING))

    done = already_labelled(path)

    assert is_done(done_row, done)
    assert not is_done(todo_row, done)


def test_resuming_from_nothing_is_not_an_error(tmp_path: Path) -> None:
    assert already_labelled(tmp_path / "nope.csv") == set()


def test_two_early_candidates_on_one_clip_do_not_collide_on_resume(tmp_path: Path) -> None:
    """The signature bug: both candidates sit at t<=2s, so the clamped interval start is
    0.00 for BOTH. Keyed on that start, labelling the first marks the second done and it is
    never asked -- a candidate, possibly a real fight, lost. The RAW timestamp is distinct."""
    first = ScanRow(BURST, 0.5, 1.9, 0.8, 0.88)
    second = ScanRow(BURST, 1.5, 1.5, 0.6, 0.60)

    append_label(path=tmp_path / "labels.csv", interval=interval_for(first, LabelKind.BULLYING))
    done = already_labelled(tmp_path / "labels.csv")

    assert is_done(first, done), "the labelled candidate must be recognised as done"
    assert not is_done(second, done), "the second early candidate was silently marked done"


def test_two_early_candidates_end_to_end_the_second_is_still_asked(tmp_path: Path) -> None:
    """End to end: label the first early candidate, quit, resume -- the second is asked."""
    first = ScanRow(BURST, 0.5, 1.9, 0.8, 0.88)
    second = ScanRow(BURST, 1.5, 1.5, 0.6, 0.60)
    args = _corpus(tmp_path, [first, second], crops=[CROP])

    cmd_label(args, ask=_asker("b", "q")[0], play=Player())  # label the first, quit

    ask_again, asked = _asker("n")
    assert cmd_label(args, ask=ask_again, play=Player()) == 0
    assert len(asked) == 1, "the second early candidate was silently marked done on resume"
    assert len(load_labels(args.labels)) == 2, "the second early candidate was never recorded"


def test_a_hand_written_labels_file_with_a_camera_column_still_resumes(tmp_path: Path) -> None:
    """`labels.csv` carries an optional `camera` column and `#` comments. `eval label`
    appends to the same file a human edits, so reading it back must not choke on either --
    a labeller whose resume silently reset to zero would re-watch the afternoon."""
    path = tmp_path / "labels.csv"
    path.write_text(
        "# camera: optional.\nvideo_id,t_start,t_end,label,camera\n"
        f"{BURST},2.00,6.00,bullying,hall_left\n",
        encoding="utf-8",
    )

    assert is_done(ScanRow(BURST, 4.0, 1.9, 0.8, 0.88), already_labelled(path))


# -- the loop: the player is injected, never launched ----------------------


class Player:
    """Stands in for the OS video player. Records; opens nothing."""

    def __init__(self) -> None:
        self.played: list[Path] = []

    def __call__(self, path: Path) -> None:
        self.played.append(path)


def _asker(*answers: str) -> tuple[object, list[str]]:
    """A human, scripted. Running out of answers is a test bug, not a hang."""
    supply: Iterator[str] = iter(answers)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return next(supply)

    return ask, asked


def _corpus(tmp_path: Path, rows: list[ScanRow], *, crops: list[str] = ()) -> argparse.Namespace:
    """A candidates file, a clips dir and a crops dir. The videos are empty: nothing here
    decodes a frame, and nothing here plays one."""
    clips = tmp_path / "clips"
    crops_dir = tmp_path / "crops"
    clips.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    for row in rows:
        (clips / row.clip).write_bytes(b"")
    for name in crops:
        (crops_dir / name).write_bytes(b"")

    candidates = tmp_path / "candidates.csv"
    write_candidates(rows, candidates)
    return argparse.Namespace(
        candidates=candidates,
        clips=clips,
        crops=crops_dir,
        labels=tmp_path / "labels.csv",
    )


def test_the_crop_is_what_the_human_watches_and_the_full_frame_is_what_is_recorded(
    tmp_path: Path,
) -> None:
    """The whole point, in one test. Watch 320x450 of two children; write the label against
    the 2560x1440 clip `eval run` will score."""
    args = _corpus(tmp_path, [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)], crops=[CROP])
    player = Player()
    ask, _ = _asker("b")

    assert cmd_label(args, ask=ask, play=player) == 0
    assert player.played == [args.crops / CROP], "it opened the wide shot, not the crop"

    labels = load_labels(args.labels)
    assert [i.video_id for i in labels.positives(BURST)] == [BURST]


def test_a_stairs_clip_watches_its_crop_and_records_against_its_own_full_frame(
    tmp_path: Path,
) -> None:
    """The same contract, for the naming shape that has no `_main`. The crops live in
    their own directory (`eval/crops/`); the label still lands on the full frame, which is
    the only view that HAS a scene for `eval run` to score."""
    args = _corpus(tmp_path, [ScanRow(STAIRS_BURST, 4.0, 1.9, 0.8, 0.88)], crops=[STAIRS_CROP])
    player = Player()
    ask, _ = _asker("b")

    assert cmd_label(args, ask=ask, play=player) == 0
    assert player.played == [args.crops / STAIRS_CROP], "it did not find the crop"

    labels = load_labels(args.labels)
    assert [i.video_id for i in labels.positives(STAIRS_BURST)] == [STAIRS_BURST]


def test_a_candidate_with_no_crop_partner_is_watched_as_the_full_frame(tmp_path: Path) -> None:
    args = _corpus(tmp_path, [ScanRow(LONELY, 4.0, 1.9, 0.8, 0.88)], crops=[CROP])
    player = Player()
    ask, _ = _asker("n")

    assert cmd_label(args, ask=ask, play=player) == 0
    assert player.played == [args.clips / LONELY]
    assert load_labels(args.labels).for_video(LONELY)[0].kind is LabelKind.NORMAL


def test_a_crop_named_in_the_join_but_missing_from_disk_falls_back_to_the_full_frame(
    tmp_path: Path,
) -> None:
    """The join is on names. A name is not a file."""
    args = _corpus(tmp_path, [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)])  # no crop on disk
    player = Player()
    ask, _ = _asker("i")

    assert cmd_label(args, ask=ask, play=player) == 0
    assert player.played == [args.clips / BURST]


def test_a_skip_writes_nothing_and_the_candidate_comes_back_next_run(tmp_path: Path) -> None:
    """A skip is a skip. Never a guessed label -- a `normal` nobody chose is exactly the
    legacy's silent-negative bug, wearing a human's face."""
    row = ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)
    args = _corpus(tmp_path, [row], crops=[CROP])
    ask, _ = _asker("s")

    assert cmd_label(args, ask=ask, play=Player()) == 0
    assert not args.labels.exists(), "a skip invented a label"

    ask_again, asked = _asker("b")
    assert cmd_label(args, ask=ask_again, play=Player()) == 0
    assert len(asked) == 1, "the skipped candidate was not offered again"
    assert len(load_labels(args.labels)) == 1


def test_pressing_p_records_a_pending_label(tmp_path: Path) -> None:
    """`pending` means "I cannot judge this yet". It keeps the clip scannable and asserts
    nothing -- distinct from `s` (skip, writes nothing) and `i` (ignore, a judgement)."""
    args = _corpus(tmp_path, [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)], crops=[CROP])
    ask, _ = _asker("p")

    assert cmd_label(args, ask=ask, play=Player()) == 0
    assert [i.kind for i in load_labels(args.labels).for_video(BURST)] == [LabelKind.PENDING]


def test_the_labeller_offers_pending_distinct_from_skip_and_ignore() -> None:
    from qorgan.evaluation.labelling import KEYS_HELP, PROMPT

    assert "[p]ending" in PROMPT
    lowered = KEYS_HELP.lower()
    assert "cannot judge" in lowered, "pending's meaning is not spelled out"
    assert "writes nothing" in lowered, "skip is not distinguished from pending"
    assert "judgement" in lowered, "ignore is not distinguished from pending"


def test_an_answer_nobody_understands_re_prompts_and_never_becomes_a_label(
    tmp_path: Path,
) -> None:
    """Free text is never coerced. An answer that is not an exact key is re-asked, not
    turned into a guessed label -- and `s` remains the ONLY way to skip."""
    args = _corpus(tmp_path, [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)], crops=[CROP])
    ask, asked = _asker("maybe?", "s")  # re-prompt, then an explicit skip

    assert cmd_label(args, ask=ask, play=Player()) == 0
    assert len(asked) == 2, "the unrecognised answer was not re-prompted"
    assert not args.labels.exists(), "an unrecognised answer invented a label"


def test_free_text_beginning_with_a_key_letter_is_re_prompted_not_coerced(
    tmp_path: Path,
) -> None:
    """The forbidden guess: "not sure" -> 'n' recorded NORMAL, "beats me" -> 'b' BULLYING.
    An ambiguous answer must RE-PROMPT; only the following exact key is recorded."""
    args = _corpus(tmp_path, [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)], crops=[CROP])
    ask, asked = _asker("not sure", "", "b")  # ambiguous, empty, then a real key

    assert cmd_label(args, ask=ask, play=Player()) == 0
    assert len(asked) == 3, "the ambiguous and empty answers were not re-prompted"
    kinds = [i.kind for i in load_labels(args.labels).for_video(BURST)]
    assert kinds == [LabelKind.BULLYING], "free text became a label instead of re-prompting"


def test_quit_stops_the_run_and_keeps_what_was_already_decided(tmp_path: Path) -> None:
    first = ScanRow(BURST, 4.0, 1.9, 0.8, 0.88)
    second = ScanRow(BURST, 30.0, 1.5, 0.6, 0.60)
    args = _corpus(tmp_path, [first, second], crops=[CROP])
    player = Player()
    ask, asked = _asker("b", "q")

    assert cmd_label(args, ask=ask, play=player) == 0
    assert len(asked) == 2
    assert len(player.played) == 2
    assert len(load_labels(args.labels)) == 1, "quitting lost the decision before it"


def test_a_second_run_re_asks_nothing_and_duplicates_no_row(tmp_path: Path) -> None:
    """Resumable, end to end. Re-running must not re-ask, and must not write the same
    interval twice -- a duplicated positive would inflate recall out of thin air."""
    rows = [ScanRow(BURST, 4.0, 1.9, 0.8, 0.88), ScanRow(BURST, 30.0, 1.5, 0.6, 0.60)]
    args = _corpus(tmp_path, rows, crops=[CROP])
    ask, _ = _asker("b", "n")
    cmd_label(args, ask=ask, play=Player())

    player = Player()
    ask_again, asked = _asker()  # any question at all raises StopIteration
    assert cmd_label(args, ask=ask_again, play=player) == 0

    assert asked == [], "it asked again about a candidate already labelled"
    assert player.played == [], "it opened a clip it had no question about"
    assert len(load_labels(args.labels)) == 2, "a resumed run duplicated a row"


# -- a row with NO timestamp: the detector named no moment ------------------
#
# `qorgan eval sample` draws the SILENT clips -- the ones the fast tier never fired on --
# and the detector said nothing about them, so there is no moment to centre on. Those rows
# arrive here with `timestamp=None`, and the human judges the WHOLE clip.


def test_a_row_with_no_timestamp_is_labelled_over_the_WHOLE_clip() -> None:
    """The detector proposed no moment, so the label may not invent one.

    The alternative -- the original plan's -- was to write the clip's midpoint and let the
    +/-2 s pad make an interval of it. On a 90-second clip that records the fight as being
    at t=45 and the other 86 seconds as negative. A detector that then finds the real fight
    at t=40 is scored a FALSE POSITIVE for finding it. That is not a coarser measurement,
    it is a wrong one.
    """
    interval = interval_for(ScanRow(BURST, None, 0.0, 0.0, 0.0), LabelKind.BULLYING, 93.5)

    assert interval.video_id == BURST
    assert interval.start == 0.0
    assert interval.end == 93.5
    assert interval.kind is LabelKind.BULLYING


def test_a_timestamped_row_ignores_the_duration_and_keeps_its_moment() -> None:
    """A candidate HAS a moment. The whole-clip interval is for the rows that do not."""
    interval = interval_for(ScanRow(BURST, 4.25, 1.9, 0.8, 0.88), LabelKind.BULLYING, 93.5)

    assert (interval.start, interval.end) == (2.25, 6.25)


def test_labelling_a_silent_clip_records_the_whole_clip_and_asks_for_its_length_once(
    tmp_path: Path,
) -> None:
    """End to end: a silent row in, a whole-clip interval out. The duration is read from
    the video ONCE, and only for the rows that actually need it -- a timestamped candidate
    never opens a decoder at all."""
    silent = ScanRow(BURST, None, 0.0, 0.0, 0.0)
    args = _corpus(tmp_path, [silent, ScanRow(LONELY, 4.0, 1.9, 0.8, 0.88)])
    ask, _ = _asker("b", "n")
    measured: list[Path] = []

    def duration(path: Path) -> float:
        measured.append(path)
        return 93.5

    assert cmd_label(args, ask=ask, play=Player(), duration=duration) == 0

    positives = load_labels(args.labels).positives(BURST)
    assert [(i.start, i.end) for i in positives] == [(0.0, 93.5)]
    assert [p.name for p in measured] == [BURST], "it decoded a clip it had a timestamp for"


def test_a_labelled_silent_clip_is_not_re_asked_on_the_next_run(tmp_path: Path) -> None:
    """Resume works for the whole-clip rows too, or the 80 silent clips come back every
    time the labeller stops for lunch."""
    silent = ScanRow(BURST, None, 0.0, 0.0, 0.0)
    args = _corpus(tmp_path, [silent])
    cmd_label(args, ask=_asker("n")[0], play=Player(), duration=lambda path: 93.5)

    ask_again, asked = _asker()  # any question at all raises StopIteration
    assert cmd_label(args, ask=ask_again, play=Player(), duration=lambda path: 93.5) == 0

    assert asked == [], "it re-asked a silent clip a human had already judged"
    assert len(load_labels(args.labels)) == 1


def test_labelling_with_no_candidates_says_run_scan_first(tmp_path: Path) -> None:
    """Zero candidates from a scan that never ran reads exactly like zero candidates from
    a detector that found nothing."""
    args = _corpus(tmp_path, [])

    with pytest.raises(SystemExit, match="eval scan"):
        cmd_label(args, ask=_asker()[0], play=Player())

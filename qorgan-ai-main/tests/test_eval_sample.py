"""`qorgan eval sample`: the clips a human must watch, so that MISSES can be measured.

The corpus scan ran over all 657 clips and produced three populations, not one:

    51 alerts              confidence >= notify_threshold  -- measures PRECISION
    72 skeleton-suppressed confidence held at the cap      -- did the SKELETON veto a fight?
   517 silent              the fast tier never fired       -- did the FAST TIER miss one?

The middle one is the dangerous one, and a precision-only analysis never looks at it. The
mandatory-skeleton rule is vetoing HALF of everything the fast tier proposes: that is the
system's central safety property doing measurable work, and it is also the one place a
real fight could be discarded with nobody ever seeing it. A false alarm annoys a teacher;
a suppressed fight is a child nobody helped.

So the sample is STRATIFIED, every row records WHICH stratum it came from, and the draw is
seeded -- a sample nobody can reproduce is not evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qorgan.config.camera import BullyingCamera
from qorgan.config.loader import load_cameras
from qorgan.evaluation import cli as eval_cli
from qorgan.evaluation.clips import camera_for
from qorgan.evaluation.labelling import already_labelled, append_label, interval_for, is_done
from qorgan.evaluation.labels import Interval, LabelKind, LabelSet
from qorgan.evaluation.metrics import Prediction, evaluate
from qorgan.evaluation.sampling import (
    SAMPLE_COLUMNS,
    SampleRow,
    Stratum,
    draw,
    write_sample,
)
from qorgan.evaluation.scan import SCAN_COLUMNS, ScanRow, load_candidates


def _labelset(*intervals: Interval) -> LabelSet:
    return LabelSet(intervals=tuple(intervals))


# The real thresholds, from the real config: notify 0.85, cap 0.72.
ALERT_CONFIDENCE = 0.91
CAP_CONFIDENCE = 0.72
BELOW_CAP_CONFIDENCE = 0.55


def _clip(index: int) -> str:
    """A clip the recorder named, and therefore one whose camera is provable."""
    return f"hall_left_main_{index}_{index + 1}_burst101_20260702_1200{index:02d}_000000.mp4"


def _clips(count: int) -> list[str]:
    return [_clip(i) for i in range(count)]


def _cameras(clips: list[str]) -> dict[str, BullyingCamera]:
    """Each clip paired with the camera whose thresholds it must be judged against.

    The strata boundaries are CONFIG, not constants: `notify_threshold` and
    `cap_without_skeleton` live on the camera. Two cameras may draw the line differently.
    """
    configured = load_cameras()
    return {clip: camera_for(clip, configured) for clip in clips}


def _candidate(clip: str, confidence: float, timestamp: float = 4.0) -> ScanRow:
    return ScanRow(clip, timestamp, 1.9, 0.8, confidence)


def _corpus() -> tuple[list[str], list[ScanRow]]:
    """Three alerts, four suppressed, one below the cap, and twenty silent clips."""
    clips = _clips(28)
    candidates = [
        *[_candidate(clips[i], ALERT_CONFIDENCE) for i in range(3)],
        *[_candidate(clips[i], CAP_CONFIDENCE) for i in range(3, 7)],
        _candidate(clips[7], BELOW_CAP_CONFIDENCE),
    ]
    return clips, candidates


_COVER_ALL = object()  # sentinel: the manifest proves every clip in `clips` was scanned


def _draw(
    clips: list[str],
    candidates: list[ScanRow],
    coverage: object = _COVER_ALL,
    **kwargs: object,
) -> list[SampleRow]:
    """Silence is PROVEN now, not inferred: `draw` needs the coverage manifest. By default
    every clip is covered (so the existing proven-silence cases read the same), but a test
    can pass an explicit `coverage=` (or `None`, a missing manifest) to exercise the guards.
    """
    return draw(
        clips=clips,
        candidates=candidates,
        cameras=_cameras(clips),
        coverage=set(clips) if coverage is _COVER_ALL else coverage,
        **kwargs,  # type: ignore[arg-type]
    )


# -- deterministic: a sample nobody can reproduce is not evidence -----------


def test_the_same_seed_draws_the_same_sample_and_a_different_seed_draws_another() -> None:
    """Reproducibility is the whole difference between evidence and an anecdote.

    50 silent clips choose 10: two seeds agreeing by accident is a 1-in-10-billion event,
    so an equality here is the code ignoring the seed, not luck.
    """
    clips = _clips(50)

    first = _draw(clips, [], count=10, seed=7)
    again = _draw(clips, [], count=10, seed=7)
    other = _draw(clips, [], count=10, seed=8)

    assert [row.row.clip for row in first] == [row.row.clip for row in again]
    assert [row.row.clip for row in first] != [row.row.clip for row in other]


# -- stratified, and the stratum is ON the row ------------------------------


def test_every_row_carries_its_stratum_and_all_three_strata_appear() -> None:
    """An unweighted stratified sample is a biased estimate wearing a lab coat.

    A row that does not say which stratum it came from cannot be weighted back to the
    population it was drawn from, and the analysis it feeds is arithmetic on sand.
    """
    clips, candidates = _corpus()

    rows = _draw(clips, candidates, count=5, seed=7)

    assert rows
    assert all(isinstance(row.stratum, Stratum) for row in rows)
    assert {row.stratum for row in rows} == {
        Stratum.ALERT,
        Stratum.SKELETON_SUPPRESSED,
        Stratum.BELOW_CAP,
        Stratum.SILENT,
    }


def test_all_of_the_alerts_and_all_of_the_suppressed_are_taken_never_sampled_down() -> None:
    """A and B are small and they are the two that decide the question. B especially:
    the skeleton is vetoing half of what the fast tier proposes, and whether any of those
    vetoes threw away a real fight is invisible to every precision-only measurement."""
    clips, candidates = _corpus()

    rows = _draw(clips, candidates, count=2, seed=7)

    alerts = [r for r in rows if r.stratum is Stratum.ALERT]
    suppressed = [r for r in rows if r.stratum is Stratum.SKELETON_SUPPRESSED]

    assert len(alerts) == 3, "an alert was sampled away -- precision is now an estimate"
    assert len(suppressed) == 4, "a skeleton veto was sampled away -- the veto is unmeasured"
    assert len([r for r in rows if r.stratum is Stratum.SILENT]) == 2


def test_a_near_miss_is_its_own_stratum_and_below_cap_is_only_below_the_cap() -> None:
    """`below_cap` was the residual bucket and it lied: a candidate with skeleton CONFIRMED
    but under the alert threshold is not below the cap at all. Split so it cannot -- and both
    are DRAWN now (count=0 leaves only what classify produced), the evidence nearest the
    boundary that an operating point is chosen with."""
    clips = _clips(2)
    conf = _cameras(clips)[clips[0]].bullying.confidence
    near = (conf.cap_without_skeleton + conf.notify_threshold) / 2  # strictly between

    rows = _draw(
        clips,
        [
            _candidate(clips[0], near),
            _candidate(clips[1], conf.cap_without_skeleton - 0.1),
        ],
        count=0,
    )

    by_clip = {row.row.clip: row.stratum for row in rows}
    assert by_clip[clips[0]] is Stratum.NEAR_MISS, "skeleton confirmed, below the alert = near miss"
    assert by_clip[clips[1]] is Stratum.BELOW_CAP, "genuinely below the cap"


def test_the_stratum_boundary_is_the_cameras_own_config_not_a_hardcoded_number() -> None:
    """`notify_threshold` and `cap_without_skeleton` are per-camera config. A constant
    baked in here would silently misclassify the day one camera is retuned."""
    clips = _clips(3)
    camera = _cameras(clips)[clips[0]]
    confidence = camera.bullying.confidence

    rows = _draw(
        clips,
        [
            _candidate(clips[0], confidence.notify_threshold),
            _candidate(clips[1], confidence.cap_without_skeleton),
        ],
        count=0,
        seed=7,
    )

    by_clip = {row.row.clip: row.stratum for row in rows}
    assert by_clip[clips[0]] is Stratum.ALERT, "confidence == notify_threshold IS an alert"
    assert by_clip[clips[1]] is Stratum.SKELETON_SUPPRESSED


# -- joinable with labels.csv: never re-ask what a human already answered ---


def test_labelling_one_candidate_on_a_clip_does_not_erase_the_others() -> None:
    """The sampler's dedup was clip-level: one label ANYWHERE on a clip erased every other
    candidate on it, forever, with no log line -- the 145-candidates-on-140-clips collapse.
    The key must be the labeller's own (`labelling.resume_key` / `settled_key`, on the
    interval END -- never the clamped start). Two candidates on one clip are two distinct
    questions, and answering one leaves the other standing."""
    clip = _clip(0)
    first = _candidate(clip, ALERT_CONFIDENCE, timestamp=4.0)
    second = _candidate(clip, ALERT_CONFIDENCE, timestamp=30.0)
    labelled = _labelset(interval_for(first, LabelKind.NORMAL))  # only the first was answered

    rows = _draw([clip], [first, second], labelled=labelled, count=0)

    assert [row.row.timestamp for row in rows] == [30.0], "a label on one erased the other"


def test_two_early_candidates_on_one_clip_are_not_collided_by_the_start_clamp() -> None:
    """The signature bug, in the sampler. Two candidates at t<=2s both clamp to interval
    start 0.00. Settling the FIRST must not suppress the SECOND -- they are two distinct
    questions, and the dedup key must be the un-clamped interval END, not that start."""
    clip = _clip(0)
    first = _candidate(clip, ALERT_CONFIDENCE, timestamp=0.5)
    second = _candidate(clip, ALERT_CONFIDENCE, timestamp=1.5)
    labelled = _labelset(interval_for(first, LabelKind.NORMAL))  # only the first was answered

    rows = _draw([clip], [first, second], labelled=labelled, count=0)

    assert [row.row.timestamp for row in rows] == [1.5], "the second early candidate was erased"


def test_a_settled_whole_clip_label_suppresses_the_clips_silent_row() -> None:
    """A non-pending label is a judgement of the clip as a whole, so its SILENT row -- the
    question "did the fast tier miss a fight here?" -- has been answered."""
    clip = _clip(0)  # no candidate on it, so it is a silent clip
    labelled = _labelset(Interval(clip, 0.0, 90.0, LabelKind.NORMAL, camera="hall_left"))

    rows = _draw([clip], [], labelled=labelled, count=10)

    assert rows == [], "the silent row survived a settled whole-clip judgement"


def test_a_pending_row_suppresses_nothing() -> None:
    """A pending interval asserts nothing, so it cannot answer a question. A clip carrying
    only a pending row must still be proposed -- at every candidate, and as its silent row --
    which is the only reason the ignore-marked fight can ever be re-proposed."""
    clip = _clip(0)
    candidate = _candidate(clip, ALERT_CONFIDENCE, timestamp=4.0)
    pending = _labelset(Interval(clip, 0.0, 17.6, LabelKind.PENDING, camera="hall_left"))

    proposed = _draw([clip], [candidate], labelled=pending, count=0)
    assert [row.row.timestamp for row in proposed] == [4.0], "a pending row suppressed a candidate"

    silent_clip = _clip(1)
    pending_silent = _labelset(
        Interval(silent_clip, 0.0, 90.0, LabelKind.PENDING, camera="hall_left")
    )
    silent = _draw([silent_clip], [], labelled=pending_silent, count=10)
    assert [row.stratum for row in silent] == [Stratum.SILENT], "a pending row hid a silent clip"
    assert silent[0].row.timestamp is None


# -- never guess a timestamp ------------------------------------------------


def test_a_silent_clip_is_proposed_with_NO_timestamp_never_an_invented_one() -> None:
    """The detector said NOTHING about this clip. There is no moment to centre on.

    The original plan wrote the clip's MIDPOINT here, which `interval_for` would then pad
    to [mid-2, mid+2] -- a four-second window invented out of nothing. If the fight is at
    t=40 of a 90-second clip, that row asserts the fight is at t=45 AND that t=40 is
    negative, so a detector that finds the real fight is scored a false positive for it.
    A fabricated interval is not a weaker measurement, it is a wrong one.

    So: no timestamp at all. The human judges the WHOLE clip.
    """
    clips = _clips(6)

    rows = _draw(clips, [], count=6, seed=7)

    silent = [row for row in rows if row.stratum is Stratum.SILENT]
    assert len(silent) == 6
    assert all(row.row.timestamp is None for row in silent)


def test_a_candidate_keeps_the_exact_timestamp_the_detector_measured() -> None:
    """A/B rows have a real moment, and it is never rounded away or re-derived."""
    clips, _ = _corpus()

    rows = _draw(clips, [_candidate(clips[0], ALERT_CONFIDENCE, timestamp=17.25)], count=0)

    assert [row.row.timestamp for row in rows] == [17.25]


# -- the worklist is what `qorgan eval label` reads -------------------------


def test_the_worklist_is_the_file_eval_label_consumes(tmp_path: Path) -> None:
    """`eval label` reads it with `load_candidates`. If the sample writes a file that
    reader cannot parse, the worklist is a document, not a tool."""
    clips, candidates = _corpus()
    rows = _draw(clips, candidates, count=4, seed=7)
    path = tmp_path / "eval" / "sample.csv"

    assert write_sample(rows, path) == len(rows)

    loaded = load_candidates(path)
    assert len(loaded) == len(rows)
    assert [row.clip for row in loaded] == [row.row.clip for row in rows]
    assert any(row.timestamp is None for row in loaded), "the silent rows lost their blank"
    assert all(row.timestamp == 4.0 for row in loaded if row.clip in {clips[0], clips[3]})


def test_the_written_row_names_its_stratum_in_a_column_of_its_own(tmp_path: Path) -> None:
    clips, candidates = _corpus()
    path = tmp_path / "sample.csv"

    write_sample(_draw(clips, candidates, count=3, seed=7), path)
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0].split(",") == list(SAMPLE_COLUMNS)
    assert SAMPLE_COLUMNS[: len(SCAN_COLUMNS)] == SCAN_COLUMNS, "eval label reads by name"
    assert {line.split(",")[-1] for line in lines[1:]} == {
        Stratum.ALERT.value,
        Stratum.SKELETON_SUPPRESSED.value,
        Stratum.BELOW_CAP.value,
        Stratum.SILENT.value,
    }


# -- nothing is dropped quietly --------------------------------------------


def test_what_the_sample_leaves_out_is_logged_and_never_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A silent cap reads as "we covered everything" when it did not.

    What is left out now is the un-drawn silent clips and anything a human already labelled --
    the below-cap candidates are no longer held back, they are drawn. Each remaining hole in
    the coverage is counted out loud.
    """
    clips, candidates = _corpus()
    labelled = _labelset(Interval(clips[0], 0.0, 90.0, LabelKind.NORMAL, camera="hall_left"))

    with caplog.at_level(logging.INFO, logger="qorgan.evaluation.sampling"):
        _draw(clips, candidates, labelled=labelled, count=2, seed=7)

    logged = caplog.text
    assert "already labelled" in logged
    assert "not drawn" in logged


def test_the_confirmed_fight_as_pending_is_proposed_scored_and_blocks_the_baseline() -> None:
    """The defect end to end, in one test. The only confirmed fight sits as a `pending` row
    with an explicit camera and an unknown interval.

      - `eval sample` still proposes every candidate on the clip (pending suppresses nothing),
        so the human can convert it into the corpus's first positive;
      - `eval run` reports the pending count rather than a clean recall, and does not punish
        the detector for firing on the un-judged fight;
      - `eval save-baseline` refuses, because a baseline whose recall is fiction must not be
        frozen.
    """
    clip = _clip(0)
    labels = _labelset(Interval(clip, 0.0, 17.6, LabelKind.PENDING, camera="hall_left"))
    first = _candidate(clip, ALERT_CONFIDENCE, timestamp=4.0)
    second = _candidate(clip, ALERT_CONFIDENCE, timestamp=30.0)

    worklist = _draw([clip], [first, second], labelled=labels, count=0)
    assert sorted(row.row.timestamp for row in worklist) == [4.0, 30.0], "a candidate was erased"

    metrics = evaluate(labels, [Prediction(clip, 8.0, 0.9)])
    assert metrics.pending_intervals == 1
    assert metrics.false_positives == 0, "the detector was punished for firing on the fight"
    assert "PENDING" in metrics.summary()

    with pytest.raises(SystemExit, match="pending"):
        eval_cli._refuse_when_pending(metrics, "save-baseline")


def test_asking_for_more_silent_clips_than_exist_takes_all_of_them() -> None:
    clips, candidates = _corpus()

    rows = _draw(clips, candidates, count=10_000, seed=7)

    assert len([r for r in rows if r.stratum is Stratum.SILENT]) == 20


# -- `eval label`'s dedup must agree with `eval sample`'s: pending settles nothing ------


def test_a_pending_labelled_candidate_is_still_offered_by_eval_label(tmp_path: Path) -> None:
    """The other half of the defect: `eval sample` re-proposes a per-candidate `pending` row
    (`_judged` excludes pending), but `eval label`'s own resume must agree -- a pending
    candidate is "I cannot judge this yet", not "done", so it must be RE-OFFERED on the next
    `eval label` run exactly as the sampler keeps proposing it."""
    path = tmp_path / "labels.csv"
    candidate = ScanRow(_clip(0), 10.0, 1.9, 0.8, ALERT_CONFIDENCE)
    append_label(path, interval_for(candidate, LabelKind.PENDING))

    done = already_labelled(path)

    assert not is_done(candidate, done), "a pending candidate was silently marked done"


def test_a_settled_labelled_candidate_is_still_recognised_as_done(tmp_path: Path) -> None:
    """A real judgement (not pending) must still dedup normally -- the fix must not turn
    `already_labelled` into a no-op for settled rows."""
    path = tmp_path / "labels.csv"
    candidate = ScanRow(_clip(0), 20.0, 1.9, 0.8, ALERT_CONFIDENCE)
    append_label(path, interval_for(candidate, LabelKind.BULLYING))

    done = already_labelled(path)

    assert is_done(candidate, done), "a settled label was not recognised as done"


def test_a_whole_clip_pending_row_does_not_block_that_clips_own_candidates(
    tmp_path: Path,
) -> None:
    """The one confirmed fight: a WHOLE-CLIP `pending` row (t_start 0, no timestamp) must
    not suppress that same clip's own timestamped candidates. This already worked before the
    per-candidate fix, and it must keep working after it."""
    path = tmp_path / "labels.csv"
    clip = _clip(0)
    whole_clip_row = ScanRow(clip, None, 0.0, 0.0, 0.0)
    append_label(path, interval_for(whole_clip_row, LabelKind.PENDING, duration=17.6))

    candidate = ScanRow(clip, 4.0, 1.9, 0.8, ALERT_CONFIDENCE)
    done = already_labelled(path)

    assert not is_done(candidate, done), "a whole-clip pending row blocked its own candidate"

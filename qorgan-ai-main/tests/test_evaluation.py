"""The eval harness: labels, metrics, the regression gate, and R2 itself."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pytest

from qorgan.config.bullying import BullyingConfig
from qorgan.config.camera import BullyingCamera
from qorgan.detection.validation import SkeletonResult, score_skeleton
from qorgan.evaluation import cli as eval_cli
from qorgan.evaluation import evaluate, load_labels, pr_curve, report, run
from qorgan.evaluation.metrics import Metrics, Prediction, best_threshold
from qorgan.evaluation.regression import Baseline, check
from qorgan.models.validate import NO_POSE, validate_candidate
from tests.test_detection_pipeline import (
    HEIGHT,
    STEP,
    WIDTH,
    scene_an_assault,
    scene_two_children_talking,
)
from tests.test_eval_labels import _labels_file

FIGHT_AT = 2.0  # the assault scene reaches a grapple around here


# -- metrics ---------------------------------------------------------------


def _one_fight(tmp_path: Path):
    return load_labels(_labels_file(tmp_path, "hall.mp4,10.0,15.0,bullying\n"))


def test_an_alert_during_the_fight_is_a_true_positive(tmp_path: Path) -> None:
    metrics = evaluate(_one_fight(tmp_path), [Prediction("hall.mp4", 12.0, 0.9)])

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.recall == 1.0


def test_an_alert_nowhere_near_the_fight_is_a_false_positive(tmp_path: Path) -> None:
    metrics = evaluate(_one_fight(tmp_path), [Prediction("hall.mp4", 90.0, 0.9)])

    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1


def test_an_alert_just_before_the_fight_still_counts(tmp_path: Path) -> None:
    """A human annotator's start time is itself approximate."""
    metrics = evaluate(_one_fight(tmp_path), [Prediction("hall.mp4", 9.0, 0.9)], tolerance=2.0)
    assert metrics.true_positives == 1


def test_firing_forty_times_during_one_fight_is_not_forty_detections(tmp_path: Path) -> None:
    """THE metric that stops spam from looking like recall. A detector that alerts on
    every frame of one scuffle has found ONE fight and raised thirty-nine false alarms."""
    spam = [Prediction("hall.mp4", 10.0 + i * 0.1, 0.9) for i in range(40)]
    metrics = evaluate(_one_fight(tmp_path), spam)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 39
    assert metrics.precision < 0.05


def test_a_missed_fight_is_a_false_negative(tmp_path: Path) -> None:
    metrics = evaluate(_one_fight(tmp_path), [])

    assert metrics.false_negatives == 1
    assert metrics.recall == 0.0


def test_predictions_inside_an_ignore_interval_are_silently_absorbed(tmp_path: Path) -> None:
    """Footage nobody is confident about should neither reward nor punish."""
    labels = load_labels(_labels_file(tmp_path, "hall.mp4,10,15,bullying\nhall.mp4,40,50,ignore\n"))
    metrics = evaluate(labels, [Prediction("hall.mp4", 45.0, 0.9)])

    assert metrics.false_positives == 0
    assert metrics.true_positives == 0
    assert metrics.pending_intervals == 0, "an `ignore` is a judgement, not a pending row"


def test_a_pending_interval_is_counted_and_never_scored(tmp_path: Path) -> None:
    """The only confirmed fight, timestamp unknown, sits as a `pending` row. It asserts
    nothing: no TP, no FP, no FN. Firing on it must NOT be a false positive -- that would
    punish the detector for firing on un-judged footage. But the pending interval MUST be
    counted, so a clean-looking recall cannot hide that a clip is unlabelled."""
    labels = load_labels(_labels_file(tmp_path, "hall.mp4,0.0,17.6,pending\n"))

    metrics = evaluate(labels, [Prediction("hall.mp4", 8.0, 0.9)])

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0, "punished for firing on un-judged footage"
    assert metrics.false_negatives == 0
    assert metrics.pending_intervals == 1


def test_no_label_kind_is_silently_dropped_by_evaluate() -> None:
    """Adding a fifth LabelKind later must not default into silence: every kind has an
    explicit role in the score, or evaluate() would drop its intervals with no trace -- the
    exact disease that let `ignore` stand in for `pending` unnoticed."""
    from qorgan.evaluation.labels import LabelKind
    from qorgan.evaluation.metrics import _Role, role_of

    for kind in LabelKind:
        assert isinstance(role_of(kind), _Role), f"{kind} has no explicit role in evaluate()"


def test_gate_and_baseline_refuse_while_a_pending_interval_exists() -> None:
    """A baseline whose recall is fiction must not be frozen into a committed artefact, and a
    gate cannot judge a score built on clips no human has looked at."""
    pending = Metrics(true_positives=1, false_positives=0, false_negatives=0, pending_intervals=1)

    with pytest.raises(SystemExit, match="pending"):
        eval_cli._refuse_when_pending(pending, "save-baseline")


def test_a_score_with_no_pending_intervals_is_allowed() -> None:
    settled = Metrics(true_positives=1, false_positives=0, false_negatives=0)
    eval_cli._refuse_when_pending(settled, "gate")  # does not raise


def test_a_zero_positive_corpus_is_warned_about_not_left_reading_a_clean_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus with no `bullying` interval scores precision/recall/f1 all 0.000 -- exactly
    what a flawless detector on an empty corpus scores. `eval run` must SAY there is nothing
    to measure recall against, the way it already flags pending intervals, so a broken
    detector and a perfect one do not print the same three clean zeros. (The live labels.csv
    on this branch can be all-`normal`, so the corpus is built here, never read from disk.)"""
    all_normal = load_labels(_labels_file(tmp_path, "hall.mp4,10,15,normal\n"))
    metrics = evaluate(all_normal, [Prediction("hall.mp4", 12.0, 0.9)])
    assert metrics.positives == 0  # the corpus really holds no fights

    report.warn_when_no_positives(metrics)

    assert "no positives to measure recall against" in capsys.readouterr().out


def test_a_corpus_with_even_one_fight_is_not_warned_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One labelled fight gives recall a real denominator, so the warning must stay silent
    -- otherwise it would cry wolf on every ordinary run and teach people to ignore it."""
    one_fight = load_labels(_labels_file(tmp_path, "hall.mp4,10,15,bullying\n"))
    metrics = evaluate(one_fight, [])
    assert metrics.positives == 1

    report.warn_when_no_positives(metrics)

    assert capsys.readouterr().out == ""


def test_every_stratum_has_a_measures_entry_so_the_worklist_cannot_keyerror() -> None:
    """cmd_sample does `_MEASURES[stratum]` for every drawn stratum. A dict lookup that can
    KeyError on a valid enum member is a latent crash -- and now that BELOW_CAP and NEAR_MISS
    are drawn, the missing keys would fire the instant one appears in a worklist."""
    from qorgan.evaluation.sampling import Stratum

    for stratum in Stratum:
        assert stratum in eval_cli._MEASURES, f"{stratum} would KeyError in cmd_sample"


def test_the_confidence_threshold_filters_predictions(tmp_path: Path) -> None:
    quiet = Prediction("hall.mp4", 12.0, 0.40)
    metrics = evaluate(_one_fight(tmp_path), [quiet], threshold=0.85)

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1


def test_a_perfect_detector_scores_one(tmp_path: Path) -> None:
    metrics = evaluate(_one_fight(tmp_path), [Prediction("hall.mp4", 12.0, 0.9)])
    assert metrics.f1 == 1.0


def test_the_pr_curve_turns_picking_a_threshold_into_a_decision(tmp_path: Path) -> None:
    predictions = [
        Prediction("hall.mp4", 12.0, 0.90),  # real
        Prediction("hall.mp4", 80.0, 0.40),  # noise, quieter
    ]
    curve = pr_curve(_one_fight(tmp_path), predictions, steps=10)
    best = best_threshold(curve)

    assert len(curve) == 11
    assert best.metrics.f1 == 1.0
    assert best.threshold > 0.4, "a threshold above the noise should have been found"


# -- the regression gate ---------------------------------------------------


def _metrics(tp: int, fp: int, fn: int) -> Metrics:
    return Metrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def test_a_change_that_lowers_f1_fails_the_gate() -> None:
    baseline = Baseline.from_metrics(_metrics(9, 1, 1), videos=5)
    worse = _metrics(5, 5, 5)

    result = check(baseline, worse)

    assert not result.passed
    assert "F1 fell" in result.reason


def test_a_change_that_misses_more_fights_fails_even_if_f1_holds() -> None:
    """A missed fight and a false alarm are not the same kind of mistake, and a gate
    watching only F1 would happily trade one away for two of the other."""
    baseline = Baseline.from_metrics(_metrics(10, 10, 0), videos=5)
    fewer_alarms_but_misses_a_fight = _metrics(9, 6, 1)

    result = check(baseline, fewer_alarms_but_misses_a_fight)

    assert not result.passed
    assert "child" in result.reason


def test_an_improvement_passes_and_asks_for_the_baseline_to_be_updated() -> None:
    baseline = Baseline.from_metrics(_metrics(5, 5, 5), videos=5)
    result = check(baseline, _metrics(9, 1, 1))

    assert result.passed
    assert "Update the baseline" in result.reason


def test_no_change_passes() -> None:
    baseline = Baseline.from_metrics(_metrics(9, 1, 1), videos=5)
    assert check(baseline, _metrics(9, 1, 1)).passed


def test_a_baseline_round_trips(tmp_path: Path) -> None:
    original = Baseline.from_metrics(_metrics(9, 1, 1), videos=5, note="after unit fix")
    path = tmp_path / "baseline.json"
    original.save(path)

    assert Baseline.load(path) == original


# -- R2: the harness runs the PRODUCTION detector --------------------------


class SyntheticClip:
    """A scripted scene, standing in for a video plus YOLO."""

    def __init__(self, video_id: str, scene, repeat_after: float = 0.0) -> None:
        self.video_id = video_id
        self.width = WIDTH
        self.height = HEIGHT
        self._frames = list(scene)
        self._offset = repeat_after

    def __iter__(self):
        for index, detections in enumerate(self._frames):
            yield self._offset + index * STEP, detections

    def crops(self, _boxes, _candidate_timestamp):
        """A scripted scene has no pixels. The pose model is then handed nothing, which
        is exactly what `PoseEstimator` calls `skipped` -- and the cap applies."""
        return []


FALL = ["body_fall_or_low_posture", "close_upper_body_contact", "rapid_hand_motion"]


class SpyPose:
    """A pose model that records that it was asked, and says it saw a fall.

    No GPU, no crops, no video: the DECISION is what is under test.
    """

    def __init__(self) -> None:
        self.calls = 0

    def validate(self, crops) -> SkeletonResult:
        self.calls += 1
        return SkeletonResult(tuple(FALL), score_skeleton(FALL), skipped=False, skip_reason="")


def test_the_harness_detects_the_assault(tmp_path: Path) -> None:
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig())

    assert result.candidates > 0, "the detector found nothing"
    assert result.frames == 40


def test_the_harness_does_not_alert_on_children_talking() -> None:
    result = run(SyntheticClip("calm.mp4", scene_two_children_talking()), BullyingConfig())

    assert result.candidates == 0
    assert "static_close" in result.suppressed_by


def test_without_a_pose_model_the_event_is_logged_but_nobody_is_notified() -> None:
    """The cap, end to end.

    If the pose model could not look, the confidence is capped below the notify bar and
    nobody is woken up -- however sure the heuristics were. But the incident is still
    RECORDED, because a suspicious event a human should review is not the same thing as
    no event at all.
    """
    config = BullyingConfig()
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), config, pose=NO_POSE)

    assert result.alerts, "the incident was not even logged"
    assert all(not alert.notified for alert in result.alerts), "an unconfirmed event notified"
    assert all(
        alert.verdict.confidence <= config.confidence.cap_without_skeleton
        for alert in result.alerts
    )


def test_the_harness_actually_runs_the_skeleton() -> None:
    """**The bug that made the PR curve a fiction.**

    `harness.run`'s skeleton defaulted to `no_skeleton` and the CLI never passed one, so
    validation_score was always 0.0, every verdict was capped at cap_without_skeleton
    (0.72), and Alert.notified was always False. The Telegram threshold is 0.85. The
    curve was identically empty above 0.72 -- empty exactly where the decision lives.
    """
    config = BullyingConfig()
    spy = SpyPose()

    result = run(SyntheticClip("fight.mp4", scene_an_assault()), config, pose=spy)

    assert spy.calls > 0, "the harness never asked the pose model anything"
    assert any(
        alert.verdict.confidence > config.confidence.cap_without_skeleton
        for alert in result.alerts
    ), "every verdict is still capped at 0.72; nothing above the notify bar is reachable"
    assert any(alert.notified for alert in result.alerts), "no alert production would send"


def test_with_a_confirming_skeleton_the_assault_is_notified() -> None:
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig(), pose=SpyPose())

    assert result.alerts, "no alert at all"
    assert any(alert.notified for alert in result.alerts), "a confirmed fall did not notify anyone"


def test_one_fight_produces_one_notification_not_forty() -> None:
    """Merging, end to end. Dozens of Telegram messages for one shove trains staff to
    ignore the alerts, which is worse than having none."""
    result = run(
        SyntheticClip("fight.mp4", scene_an_assault(frames=60)),
        BullyingConfig(),
        pose=SpyPose(),
    )

    notified = [a for a in result.alerts if a.notified]
    assert len(notified) == 1, f"one incident produced {len(notified)} notifications"


def test_the_harness_scores_itself_against_labels(tmp_path: Path) -> None:
    """The whole loop: run the production detector over a labelled clip, and get a
    precision/recall/F1 back. This is the thing the legacy project never had."""
    labels = load_labels(_labels_file(tmp_path, "fight.mp4,0.5,3.5,bullying\n"))
    result = run(SyntheticClip("fight.mp4", scene_an_assault()), BullyingConfig(), pose=SpyPose())

    metrics = evaluate(labels, result.predictions)

    assert metrics.true_positives == 1, f"the labelled fight was not found: {metrics.summary()}"
    assert metrics.recall == 1.0


def test_the_worker_and_the_harness_judge_through_the_same_function() -> None:
    """Rule R2, checked mechanically -- for the SLOW TIER this time.

    The legacy had `analyze_aggression` in three files that had already diverged. Fixing
    "the harness never runs the skeleton" by writing a second crop->pose->judge path
    inside the harness would have fixed that bug by re-creating the one that caused it.
    """
    import qorgan.worker.bullying as worker
    from qorgan.evaluation import harness

    assert worker.validate_candidate is validate_candidate
    assert harness.validate_candidate is validate_candidate


def test_the_harness_imports_the_production_detector_not_a_copy() -> None:
    """Rule R2, checked mechanically. The legacy had analyze_aggression in three files
    that had already diverged, so months of tuning were measured against code that did
    not run in production."""
    import qorgan.worker.camera_loop as worker  # noqa: F401
    from qorgan.detection.pipeline import BullyingDetector as Production
    from qorgan.evaluation import harness

    assert harness.BullyingDetector is Production


# -- labels.csv's `camera` column, end to end through cli._evaluate --------
#
# Three clips in the corpus are named by a human, not the recorder, and cannot be
# attributed by filename. One of them is the only confirmed fight in the whole 663-clip
# corpus. Today, listing them in labels.csv makes `_evaluate` raise on the first one and
# score nothing. These tests drive the real `_evaluate`, with only the video decoder and
# the pose model faked out -- neither is available in a unit test.


class _FakeVideoSource:
    """Enough shape to satisfy harness.FrameSource, with no pixels and no ultralytics."""

    def __init__(self, path: Path, camera: BullyingCamera, *, device: str = "cuda:0") -> None:
        self.video_id = path.name
        self.width = camera.capture.frame_width
        self.height = camera.capture.frame_height

    def __iter__(self):
        return iter(())

    def crops(self, boxes, candidate_timestamp):
        return []


class _FakePoseEstimator:
    """Never actually asked: the fake source yields no frames, so no candidate exists
    for the slow tier to judge."""

    def __init__(self, settings, device: str = "cuda:0") -> None:
        pass


def _eval_args(labels: Path, clips: Path) -> argparse.Namespace:
    return argparse.Namespace(labels=labels, clips=clips, device="cpu", threshold=0.0)


def _write_labels_csv(path: Path, header: tuple[str, ...], *rows: tuple[str, ...]) -> None:
    """Real `csv.writer` quoting -- some of the human-named clips have a comma in the
    filename itself, so naive string formatting would silently split a field in two."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _fake_out_video_and_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qorgan.evaluation.video.VideoSource", _FakeVideoSource)
    monkeypatch.setattr("qorgan.models.pose.PoseEstimator", _FakePoseEstimator)


CYRILLIC_FIGHT = (
    "1.2 - ученики стоят разговаривают и резко начинают толкаться, "
    "подозрение на буллинг.mp4"
)
CYRILLIC_NO_BULLYING = "1.2 - нет буллинга.mp4"


def test_an_explicit_camera_lets_the_only_confirmed_fight_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / CYRILLIC_FIGHT).write_bytes(b"")

    labels_path = tmp_path / "labels.csv"
    _write_labels_csv(
        labels_path,
        ("video_id", "t_start", "t_end", "label", "camera"),
        (CYRILLIC_FIGHT, "1.0", "2.0", "bullying", "hall_left"),
    )
    _fake_out_video_and_pose(monkeypatch)

    labels, results = eval_cli._evaluate(_eval_args(labels_path, clips_dir))

    assert len(results) == 1
    assert results[0].video_id == CYRILLIC_FIGHT


def test_an_uninferable_name_with_no_explicit_camera_still_raises_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / CYRILLIC_NO_BULLYING).write_bytes(b"")

    labels_path = tmp_path / "labels.csv"
    _write_labels_csv(
        labels_path,
        ("video_id", "t_start", "t_end", "label"),
        (CYRILLIC_NO_BULLYING, "1.0", "2.0", "normal"),
    )
    _fake_out_video_and_pose(monkeypatch)

    with pytest.raises(SystemExit, match=re.escape(CYRILLIC_NO_BULLYING)):
        eval_cli._evaluate(_eval_args(labels_path, clips_dir))

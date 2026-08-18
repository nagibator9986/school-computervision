"""Event-level precision, recall and F1 — with a time tolerance.

**Why not clip-level accuracy.** A clip either "contains bullying" or does not, and a
detector that fires once at second 3 of a fight that happens at second 40 scores a
perfect hit on that measure. The legacy's harness worked this way, so it could not tell
a detector that finds fights from one that fires constantly.

Here a prediction must land *near the event in time* to count. A fight is detected iff at
least one alert falls within its interval, widened by a tolerance; alerts that land
nowhere near a fight are false positives.

Two more rules that matter:

  * **Each labelled fight can be found only once.** A detector that fires forty times
    during one four-second scuffle has found one fight and raised thirty-nine false
    alarms — and if we did not say so, spamming would look like recall.

  * **`ignore` intervals absorb predictions silently.** Footage nobody is confident about
    should neither reward nor punish.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qorgan.evaluation.labels import Interval, LabelKind, LabelSet


class _Role(Enum):
    """How a labelled interval bears on the score. EVERY LabelKind maps to exactly one, so
    that adding a fifth kind later cannot default into silence -- the exact disease that let
    `ignore` stand in for `pending` and produce a confident, wrong record."""

    POSITIVE = "positive"  # a fight: scored for TP/FN, matched by predictions
    NEGATIVE = "negative"  # documentation; everything outside a fight is negative anyway
    IGNORED = "ignored"  # a human judged "neither" -- absorbs predictions, a settled call
    PENDING = "pending"  # no human has looked -- excluded from TP/FP/FN, but COUNTED


_ROLES: dict[LabelKind, _Role] = {
    LabelKind.BULLYING: _Role.POSITIVE,
    LabelKind.NORMAL: _Role.NEGATIVE,
    LabelKind.IGNORE: _Role.IGNORED,
    LabelKind.PENDING: _Role.PENDING,
}


def role_of(kind: LabelKind) -> _Role:
    """The role of a LabelKind, or a hard error if it has none. A kind with no role would be
    silently dropped by `evaluate` -- a true number implying a false conclusion."""
    try:
        return _ROLES[kind]
    except KeyError as exc:  # a new LabelKind nobody wired in
        raise NotImplementedError(
            f"LabelKind.{kind.name} has no role in evaluate(); every kind must be handled "
            "explicitly, or a labelled interval vanishes from the score with no trace."
        ) from exc

# How far from a labelled fight an alert may land and still count. Half a second of
# tolerance reflects that a human annotator's start time is itself approximate.
DEFAULT_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class Prediction:
    """One alert the detector raised."""

    video_id: str
    timestamp: float
    confidence: float


@dataclass(frozen=True, slots=True)
class Metrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    # Intervals no human has judged yet (`pending`). They score nothing, but a metric that
    # dropped them silently would report a clean recall over a corpus with unlabelled clips
    # in it -- a true number implying a false conclusion. Surfaced, never hidden.
    pending_intervals: int = 0

    @property
    def precision(self) -> float:
        """Of the alerts we raised, how many were real? Low precision = staff stop looking."""
        raised = self.true_positives + self.false_positives
        return self.true_positives / raised if raised else 0.0

    @property
    def positives(self) -> int:
        """How many labelled fights the corpus holds -- the denominator of recall. When it
        is zero, recall is 0.000 by construction rather than by measurement, and a broken
        detector is indistinguishable from a flawless one (see `cmd_run`'s warning)."""
        return self.true_positives + self.false_negatives

    @property
    def recall(self) -> float:
        """Of the fights that happened, how many did we catch? Low recall = children hurt."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        pending = f"  [{self.pending_intervals} PENDING]" if self.pending_intervals else ""
        return (
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  f1 {self.f1:.3f}  "
            f"(tp {self.true_positives}, fp {self.false_positives}, fn {self.false_negatives})"
            f"{pending}"
        )


def evaluate(
    labels: LabelSet,
    predictions: list[Prediction],
    *,
    threshold: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
) -> Metrics:
    """Score a set of predictions against the ground truth at one confidence threshold."""
    _require_every_kind_handled(labels)
    kept = [p for p in predictions if p.confidence >= threshold]

    true_positives = 0
    false_positives = 0
    found: set[tuple[str, float, float]] = set()

    for prediction in sorted(kept, key=lambda p: (p.video_id, p.timestamp)):
        if _inside_ignored(prediction, labels):
            continue
        if _inside_pending(prediction, labels):
            # Not punished (no FP), nothing to credit (no TP). Unlike `ignore`'s silent
            # absorption, the interval is counted below, so the exclusion is surfaced.
            continue

        match = _matching_fight(prediction, labels, tolerance)
        if match is None:
            false_positives += 1
            continue

        identity = (match.video_id, match.start, match.end)
        if identity in found:
            # This fight was already found. Firing again during the same scuffle is not
            # a second detection -- it is noise, and noise is what makes staff stop looking.
            false_positives += 1
            continue

        found.add(identity)
        true_positives += 1

    total_fights = sum(len(labels.positives(video)) for video in labels.videos)
    pending = sum(len(labels.pending(video)) for video in labels.videos)
    return Metrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=total_fights - true_positives,
        pending_intervals=pending,
    )


def _require_every_kind_handled(labels: LabelSet) -> None:
    """Every kind present must map to an explicit role; a new kind with none raises here
    rather than defaulting into silence -- a labelled interval must never vanish silently."""
    for interval in labels.intervals:
        role_of(interval.kind)


def _inside_ignored(prediction: Prediction, labels: LabelSet) -> bool:
    return any(
        interval.contains(prediction.timestamp) for interval in labels.ignored(prediction.video_id)
    )


def _inside_pending(prediction: Prediction, labels: LabelSet) -> bool:
    return any(
        interval.contains(prediction.timestamp) for interval in labels.pending(prediction.video_id)
    )


def _matching_fight(prediction: Prediction, labels: LabelSet, tolerance: float) -> Interval | None:
    for interval in labels.positives(prediction.video_id):
        if interval.start - tolerance <= prediction.timestamp <= interval.end + tolerance:
            return interval
    return None


@dataclass(frozen=True, slots=True)
class CurvePoint:
    threshold: float
    metrics: Metrics


def pr_curve(
    labels: LabelSet,
    predictions: list[Prediction],
    *,
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
    steps: int = 20,
) -> list[CurvePoint]:
    """Sweep the confidence threshold. This is what turns "pick 0.85" from a guess into
    a decision you can defend."""
    return [
        CurvePoint(
            threshold=index / steps,
            metrics=evaluate(labels, predictions, threshold=index / steps, tolerance=tolerance),
        )
        for index in range(steps + 1)
    ]


def best_threshold(curve: list[CurvePoint]) -> CurvePoint:
    """The threshold with the highest F1. A starting point for a human, not a verdict:
    in a school, recall and precision are not worth the same."""
    return max(curve, key=lambda point: point.metrics.f1)

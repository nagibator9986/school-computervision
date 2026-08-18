"""How `qorgan eval run` prints its results.

Pure presentation, split out of the CLI so `cli.py` stays under the file-line limit and
the orchestration is not buried in formatting. Nothing here decides anything; it renders
the PR curve and the per-gate suppression tally a run already computed.
"""

from __future__ import annotations

from qorgan.evaluation.harness import RunResult
from qorgan.evaluation.labels import LabelSet
from qorgan.evaluation.metrics import Metrics, Prediction, best_threshold, pr_curve


def print_curve(labels: LabelSet, predictions: list[Prediction]) -> None:
    curve = pr_curve(labels, predictions)
    best = best_threshold(curve)

    print("\nthreshold  precision  recall     f1")
    for point in curve:
        marker = "  <- best f1" if point is best else ""
        m: Metrics = point.metrics
        print(
            f"  {point.threshold:.2f}      {m.precision:.3f}      "
            f"{m.recall:.3f}   {m.f1:.3f}{marker}"
        )

    print(
        "\nThe best-F1 threshold is a starting point, not a verdict: in a school, a missed "
        "fight and a false alarm are not worth the same."
    )


def warn_when_no_positives(metrics: Metrics) -> None:
    """Sibling to `cmd_run`'s pending warning: say when there is nothing to measure recall
    against at all.

    A corpus with zero `bullying` intervals scores recall 0.000 BY CONSTRUCTION, identical
    to a flawless detector on an empty corpus -- a broken detector is indistinguishable from
    a perfect one. Say so plainly, rather than let three clean zeros read as "measured, fine".
    """
    if metrics.positives == 0:
        print(
            "\n!! no `bullying` intervals in this corpus: there are no positives to measure "
            "recall against. Recall reads 0.000 by construction, exactly as a flawless "
            "detector on an empty corpus would -- a broken one is indistinguishable here. "
            "Label at least one fight, or read precision alone."
        )


def print_suppressions(results: list[RunResult]) -> None:
    totals: dict[str, int] = {}
    for result in results:
        for gate, count in result.suppressed_by.items():
            totals[gate] = totals.get(gate, 0) + count

    if not totals:
        return

    print("\nsuppressed by gate:")
    for gate, count in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {gate:<32} {count}")

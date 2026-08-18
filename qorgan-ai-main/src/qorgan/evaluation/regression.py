"""The regression gate: a change that lowers F1 fails.

The point of a benchmark is not to produce a number. It is to stop somebody — very
possibly you, six months from now, at the end of a long day — from nudging a threshold to
silence one annoying false positive and quietly costing the detector a fight it used to
catch. Nobody notices that trade at the time. The gate notices.

The baseline is a committed file. Beating it means updating it, deliberately, in a commit
that says why.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from qorgan.evaluation.metrics import Metrics

# F1 may drop by this much without failing: the harness has some run-to-run variance
# and we do not want a gate that cries wolf either.
TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class Baseline:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    videos: int
    note: str = ""

    @classmethod
    def from_metrics(cls, metrics: Metrics, videos: int, note: str = "") -> Baseline:
        return cls(
            precision=round(metrics.precision, 4),
            recall=round(metrics.recall, 4),
            f1=round(metrics.f1, 4),
            true_positives=metrics.true_positives,
            false_positives=metrics.false_positives,
            false_negatives=metrics.false_negatives,
            videos=videos,
            note=note,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Baseline:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reason: str
    baseline: Baseline
    current: Metrics

    def report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.reason}\n"
            f"  baseline: precision {self.baseline.precision:.3f}  "
            f"recall {self.baseline.recall:.3f}  f1 {self.baseline.f1:.3f}\n"
            f"  current:  {self.current.summary()}"
        )


def check(baseline: Baseline, current: Metrics, *, tolerance: float = TOLERANCE) -> GateResult:
    """Did this change make the detector worse?"""
    drop = baseline.f1 - current.f1

    if drop > tolerance:
        reason = f"F1 fell from {baseline.f1:.3f} to {current.f1:.3f} (-{drop:.3f})"
        return GateResult(False, reason, baseline, current)

    # A missed fight is not the same kind of mistake as a false alarm, and a gate that
    # only watched F1 would happily accept trading one away for two of the other.
    if current.false_negatives > baseline.false_negatives:
        reason = (
            f"the detector now misses {current.false_negatives} fights, "
            f"up from {baseline.false_negatives}. A missed fight is a child nobody helped."
        )
        return GateResult(False, reason, baseline, current)

    if current.f1 > baseline.f1 + tolerance:
        reason = f"F1 improved from {baseline.f1:.3f} to {current.f1:.3f}. Update the baseline."
        return GateResult(True, reason, baseline, current)

    return GateResult(True, f"no regression (f1 {current.f1:.3f})", baseline, current)

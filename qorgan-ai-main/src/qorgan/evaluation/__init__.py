"""The evaluation harness.

It imports the production detector. That is the whole point (rule R2): the legacy kept
three diverged copies of the detection logic, so its harness measured code that never ran
in a hallway, and every threshold tuned against it was tuned against a fiction.
"""

from qorgan.evaluation.clips import ClipName, ClipNameError, camera_for, parse_clip_name
from qorgan.evaluation.harness import Alert, FrameSource, RunResult, run
from qorgan.evaluation.labels import Interval, LabelError, LabelSet, load_labels
from qorgan.evaluation.metrics import Metrics, Prediction, evaluate, pr_curve
from qorgan.evaluation.regression import Baseline, check

__all__ = [
    "Alert",
    "Baseline",
    "ClipName",
    "ClipNameError",
    "FrameSource",
    "Interval",
    "LabelError",
    "LabelSet",
    "Metrics",
    "Prediction",
    "RunResult",
    "camera_for",
    "check",
    "evaluate",
    "load_labels",
    "parse_clip_name",
    "pr_curve",
    "run",
]

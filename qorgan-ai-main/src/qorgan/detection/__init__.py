"""The detection core: pure functions, shared by the worker and the eval harness.

Rule R2 lives here. There is exactly one implementation of the geometry, the tracking,
the scoring and the gates, and both the production worker and the evaluation harness
import it. The legacy had three diverged copies, so the harness measured code that did
not run in production and the tuning was built on a fake benchmark.
"""

from qorgan.detection.geometry import Box, dynamic_threshold, iou
from qorgan.detection.tracking import Track, TrackStore

__all__ = ["Box", "Track", "TrackStore", "dynamic_threshold", "iou"]

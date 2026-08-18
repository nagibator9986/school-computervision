"""Deciding whether a decoded frame is real or garbage.

This is a small module with a large history. The legacy heuristic was:

    if gray.std() < 4:  ->  frame is broken
    3 broken frames in a row  ->  reconnect

A dark corridor at night has a standard deviation below 4. So did fog, and a blank
wall. The camera therefore reconnected forever, in a tight loop, *precisely when the
scene was calm and nothing was happening* — the moment you most want it watching
(audit H-13).

Two changes fix it:

1. A low-texture frame is only suspect if it is ALSO not changing. A genuinely dark
   corridor still has sensor noise, so consecutive frames differ. A frozen decoder,
   a dropped feed, or a grey "no signal" card does not change at all.
2. The heuristic is allowed to give up. If it forces reconnect after reconnect and
   the picture keeps coming back the same, it is wrong about this camera — so it
   switches itself off, loudly, rather than looping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    # Below this standard deviation, a frame carries almost no texture.
    min_std: float = 4.0
    # Below this mean absolute difference, two frames are the same picture.
    min_change: float = 0.5
    # Consecutive corrupt frames before we conclude the stream is bad.
    corrupt_frames_before_reconnect: int = 5
    # Consecutive reconnects driven by this heuristic before it disables itself.
    reconnects_before_giving_up: int = 3
    enabled: bool = True


GRAYSCALE_NDIM = 2


def _grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == GRAYSCALE_NDIM:
        return frame
    # Cheap luminance. A full cvtColor on an HD frame just to take a std() is waste.
    return frame[..., :3].mean(axis=2)


def texture(frame: np.ndarray) -> float:
    """Standard deviation of luminance. Low means flat: dark, fogged, or dead."""
    return float(_grayscale(frame).std())


def change(frame: np.ndarray, previous: np.ndarray | None) -> float:
    """Mean absolute difference from the previous frame. Zero means frozen."""
    if previous is None or previous.shape != frame.shape:
        return float("inf")
    return float(np.abs(_grayscale(frame) - _grayscale(previous)).mean())


def is_corrupt(frame: np.ndarray, previous: np.ndarray | None, policy: QualityPolicy) -> bool:
    """A frame is corrupt only if it is BOTH featureless AND identical to the last one.

    Either condition alone is normal: a dark room is featureless but noisy, and a
    static scene of a textured wall is unchanging but perfectly valid to analyse.
    """
    if frame.size == 0:
        return True
    if not policy.enabled:
        return False
    if texture(frame) >= policy.min_std:
        return False
    return change(frame, previous) < policy.min_change

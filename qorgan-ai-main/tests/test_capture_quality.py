"""The frame-quality check, which the legacy got backwards.

Its heuristic was `gray.std() < 4 => broken`, and three broken frames forced a
reconnect. A dark corridor at night scores below 4. So the camera reconnected in a
tight loop exactly when the hallway was quiet -- the state you most want it watching.
"""

from __future__ import annotations

import numpy as np

from qorgan.capture.quality import QualityPolicy, change, is_corrupt, texture
from tests.fakes import dark_frame, frozen_frame, noisy_frame

POLICY = QualityPolicy()


def test_a_normal_frame_is_not_corrupt() -> None:
    assert not is_corrupt(noisy_frame(1), noisy_frame(0), POLICY)


def test_a_dark_but_live_corridor_is_not_corrupt() -> None:
    """The whole point. This frame is featureless -- and perfectly good."""
    night, previous = dark_frame(2), dark_frame(1)
    assert texture(night) < POLICY.min_std  # the legacy would have called this broken
    assert not is_corrupt(night, previous, POLICY)  # ...because it is still changing


def test_a_frozen_grey_card_is_corrupt() -> None:
    """Featureless AND unchanging. This one really is a dead feed."""
    assert is_corrupt(frozen_frame(), frozen_frame(), POLICY)


def test_a_static_but_textured_scene_is_not_corrupt() -> None:
    """An empty hallway with nobody in it is not a broken stream."""
    wall = noisy_frame(7)
    assert change(wall, wall) == 0.0  # nothing moved...
    assert not is_corrupt(wall, wall, POLICY)  # ...but there is plenty to see


def test_an_empty_frame_is_corrupt() -> None:
    assert is_corrupt(np.empty((0, 0, 3), dtype=np.uint8), None, POLICY)


def test_the_first_frame_of_a_session_is_trusted() -> None:
    """There is no previous frame to compare against, so we cannot call it frozen."""
    assert not is_corrupt(dark_frame(1), None, POLICY)


def test_a_disabled_policy_never_reports_corruption() -> None:
    disabled = QualityPolicy(enabled=False)
    assert not is_corrupt(frozen_frame(), frozen_frame(), disabled)

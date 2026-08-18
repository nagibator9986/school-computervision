"""Named frame-count constants used across the detector.

These are small integers, but they are not arbitrary: each one is a claim about how many
frames of evidence it takes to believe something on a ~10 fps corridor camera. They were
bare literals scattered through the legacy's 940-line scoring function, which is exactly
why nobody could tell which `>= 2` meant what.

They are module constants rather than config because they are structural — changing them
changes what the counters *mean*, not merely how sensitive they are. The sensitivity
knobs live in the config.

**READ THIS BEFORE YOU DERIVE A THRESHOLD FROM AN EVAL RUN.** Every count below is a
number of FRAMES, and a frame is only a duration if you know the frame rate. These were
chosen for the ~10 fps the legacy analysed at. The fleet's sub-stream delivers 15 fps
(`capture.stream_fps`), so on a `det_every: 1` camera each counter buys **1.5x less time**
than the comment beside it claims: `BRIEF_ENCOUNTER_FRAMES = 4` is 0.27 s, not "under half
a second".

They are deliberately **not** rescaled here. They are read by `gates.py`, `pipeline.py`
and `scoring.py`; changing them changes what the detector does, and this detector's
skeleton tier has been measured to carry no signal at all — so a desk-side retune would be
choosing new wrong numbers with more arithmetic. The counters get fixed on site, against
real cameras and real labels, at the same time as everything else that depends on the rate.

What IS fixed is the silence. `ASSUMED_ANALYSIS_FPS` and `counter_drift()` below make the
assumption a value anybody can find and compare, and `qorgan config validate` prints the
drift for every camera. An assumption you have to already know to ask about is how this
divergence survived unnoticed in the first place.
"""

from __future__ import annotations

# The analysis rate every frame count in this module was chosen for. NOT a knob: changing
# it does not rescale anything, it only changes what `counter_drift` reports. It is here so
# the assumption has a name and a location instead of living in prose.
ASSUMED_ANALYSIS_FPS = 10.0


def analysis_fps(stream_fps: float, det_every: int) -> float:
    """The rate the DETECTOR sees: one frame in every `det_every` the stream hands over.

    Lives here, in the module with no imports, because both ends of the divergence this
    fixes need it: the production config (`CaptureSettings.analysis_fps`) and the eval
    harness (`evaluation.noise_floor`). Two copies of this formula is how the two layers
    disagreed in the first place.
    """
    return stream_fps / det_every


def counter_drift(analysis_fps: float) -> float:
    """How far a camera's real analysis rate has moved from what the counters assume.

    1.0 means the counters mean what they say. 1.5 -- the shipped fleet on `det_every: 1`
    -- means every counter buys 1.5x less time than its comment claims. Below 1.0 they buy
    more (`canteen_inside`, at `det_every: 2`, sits at 0.75).

    Returned rather than raised or logged: the right response depends on what you are
    doing, and the one place that must not decide quietly is the detector itself.
    """
    return analysis_fps / ASSUMED_ANALYSIS_FPS


# Two frames of contact is "sustained" rather than a single-frame detector artefact.
SUSTAINED_FRAMES = 2

# Three frames is a firm, deliberate engagement rather than a brush past.
FIRM_FRAMES = 3

# A pair that is close for four frames or fewer, at ~10 fps, was together for under half
# a second. That is a crossing, not an encounter.
BRIEF_ENCOUNTER_FRAMES = 4

# Cosine of the angle between two velocity vectors, above which they count as "the same
# direction". 0.82 is about 35 degrees -- generous, because people walking together weave.
SAME_DIRECTION_ALIGNMENT = 0.82

# Below this speed (px/s) a person is standing still, not moving slowly.
STANDING_SPEED = 1.0

# Two samples is the minimum from which any velocity can be derived at all.
MIN_SAMPLES_FOR_MOTION = 2

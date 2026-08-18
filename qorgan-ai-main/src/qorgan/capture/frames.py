"""The ONE preprocessing step: the frame as the analysis loop actually sees it.

A pixel measurement means nothing without the frame it was measured in. The worker
analyses `capture.frame_width x frame_height`, so **every** number denominated in pixels --
a face size, a speed, a gate -- is denominated in THOSE pixels, and a number measured on any
other stream is not a smaller or larger version of the truth. It is a different quantity
wearing the same units.

**And that frame is PER PROFILE.** `base.yaml` holds a DEFAULT; profiles override it. There
is no fleet-wide analysis resolution, so this function takes `capture` and never assumes one.

This docstring will not tell you *which* profiles override, and the omission is deliberate.
It used to. The list said two. There were three -- and the one it left out is the camera that
CLOSES meal sessions. A list of "the ones that override" is a second source of truth that
goes stale the moment somebody edits a YAML file, which is the very bug this module exists
to prevent. Read `capture.frame_width` off the camera you care about.

That is not hypothetical, and it has now bitten twice. "2.2% of hall faces clear the 60 px
gate" was measured on the 2560x1440 HD evidence burst, a stream the analysis loop never
touches. Re-measured on the stream production DOES analyse, the same 14 970 faces give
**zero recognitions** -- but the first re-measurement did it at 960x540, the base default,
when the hall really runs at 1280x720, and so it got every derived figure wrong in the same
direction (it reported a 37.5 px largest face; the truth is 50 px). One function, taking one
`capture`, so that a caller cannot accidentally measure the other stream -- or the right
stream at a resolution nobody runs.
"""

# **`cv2` IS IMPORTED LAZILY, INSIDE EACH FUNCTION THAT USES IT.** Importing this
# module must not require OpenCV: the dashboard process reaches these files through the
# CLI parser and through `qorgan.notify`, and it neither decodes nor encodes a frame.
# The same separation `INTEGRATION.md` §1 keeps between the web and the model stack, for
# the same reason — and it takes ~120 MB and two system GL libraries out of the
# dashboard's container. Workers import it on first use exactly as before.

from __future__ import annotations

import numpy as np

from qorgan.config.common import CaptureSettings


def prepare_frame(image: np.ndarray, capture: CaptureSettings) -> np.ndarray:
    """Scale a decoded frame to the resolution the worker analyses it at.

    A no-op when the stream already delivers that resolution, so the common case copies
    nothing.
    """
    import cv2  # lazy — see the note above the imports
    height, width = image.shape[:2]
    if (width, height) == (capture.frame_width, capture.frame_height):
        return image
    return cv2.resize(image, (capture.frame_width, capture.frame_height))

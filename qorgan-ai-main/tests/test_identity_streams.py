"""Per STREAM, never per camera -- and each stream at the resolution IT is analysed at.

This file is the guard on the bug the command exists to prevent. The superseded "2.2% of
hall faces clear the gate" was measured on the 2560x1440 HD evidence burst, which the
analysis loop never touches. Production analyses `capture.frame_width x frame_height`, and
the same faces re-measured there clear the strict 60 px gate **0 of 14 970** times.

**That resolution is PER PROFILE, and getting it from the wrong place is the SECOND bug this
file guards.** 960x540 is `base.yaml`'s DEFAULT; `hall.yaml` overrides it to **1280x720**
(so does `canteen_entry.yaml`). These tests therefore take the hall's capture settings from
the hall's own config -- never from `CaptureSettings()` -- because measuring the hall at the
base default is exactly how the corpus's largest face was once reported as 37.5 px when it
is really 50 px.

The conclusion is unchanged and the MECHANISM is not. It is not that no hall face reaches
the small-face gate: at 1280x720, 77 of 14 970 do. It is that **not one of them is ever
recognised** -- the best score among all 77 is 0.350, against a min_score of 0.45. Size is
necessary and nowhere near sufficient, which is why "drop min_width to 40 and recover some"
recovers exactly nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from qorgan.config.common import CaptureSettings
from qorgan.config.identity import FaceGate
from qorgan.identity.camera import CameraCannotRecognise, measure_faces, refuse_if_hopeless
from qorgan.identity.streams import StreamSpec, clip_streams, sample, streams_for

# The real hall face, and the real streams it was seen on. 100 px wide in the 2560 px
# burst; HALF that in the stream the hall worker actually analyses (1280x720, per hall.yaml
# -- NOT base.yaml's 960x540 default, which is what made this 2.67x once and wrong).
HD_WIDTH, HD_HEIGHT = 2560, 1440
FACE_WIDTH_AT_HD = 100
FACE_HEIGHT_AT_HD = 120

# The hall's REAL analysis frame, and the size that 100 px HD face is in it.
HALL_ANALYSIS = (1280, 720)
FACE_WIDTH_AT_HALL = 50  # 100 * (1280 / 2560) -- above the 38 px gate, under the 60 px one


class _Face:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class ProportionalDetector:
    """The same person, seen through a frame that has been scaled.

    A face is a fixed fraction of the frame: 100 px wide in the 2560 px HD burst, and
    therefore 50 px in the hall's real 1280 px analysis frame. Scale the frame and the face
    scales with it -- which is precisely why the resolution the measurement was taken at IS
    the measurement, and why taking it from the wrong config is the same bug as taking it
    from the wrong stream.
    """

    def detect_faces(self, frame: np.ndarray) -> list[_Face]:
        scale = frame.shape[1] / HD_WIDTH
        return [_Face(int(FACE_WIDTH_AT_HD * scale), int(FACE_HEIGHT_AT_HD * scale))]


class FakeCapture:
    """An HD clip. cv2.VideoCapture's read()/release(), and nothing else."""

    def __init__(self, frames: int = 40) -> None:
        self._left = frames
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._left <= 0:
            return False, None
        self._left -= 1
        return True, np.zeros((HD_HEIGHT, HD_WIDTH, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _hall_capture() -> CaptureSettings:
    """The HALL's analysis frame, read from the hall's own config -- never assumed.

    `CaptureSettings()` would give base.yaml's 960x540 default, which the hall does not use.
    Reading it from anywhere but the camera is the bug these tests exist to catch.
    """
    from qorgan.config.loader import load_cameras
    from tests.conftest import CONFIG_DIR

    return load_cameras(CONFIG_DIR)["hall_left"].capture

def _analysis() -> StreamSpec:
    return StreamSpec(name="analysis", burst=False, capture=_hall_capture())


def _burst() -> StreamSpec:
    return StreamSpec(name="burst", burst=True, capture=None)


def _measure(spec: StreamSpec, frames: int = 40) -> object:
    handle = FakeCapture(frames)
    return measure_faces(
        sample(handle, spec, frames=frames, stride=1),
        ProportionalDetector(),
        FaceGate(),
        source=f"hall_left/{spec.name}",
    )


def test_the_hall_really_analyses_1280x720_not_the_base_default() -> None:
    """The assumption that was wrong, pinned so it cannot go wrong silently again.

    960x540 is `base.yaml`'s DEFAULT. `hall.yaml` and `canteen_entry.yaml` override it. Every
    face-size figure derived from the default was scaled by 0.375 when the truth is 0.5 --
    which is how the corpus's largest hall face was reported as 37.5 px when it is 50 px.
    """
    from qorgan.config.loader import load_cameras
    from tests.conftest import CONFIG_DIR

    cameras = load_cameras(CONFIG_DIR)
    hall = cameras["hall_left"].capture

    assert (hall.frame_width, hall.frame_height) == HALL_ANALYSIS
    assert (hall.frame_width, hall.frame_height) != (960, 540), (
        "the hall is being measured at base.yaml's default -- that is the bug"
    )
    assert CaptureSettings().frame_width == 960  # ...and the default really is 960, for others


def test_the_analysis_stream_is_measured_at_the_resolution_the_worker_feeds_it() -> None:
    """NOT on the raw decode, and NOT at an assumed default. This assertion is the task."""
    report = _measure(_analysis())

    assert report.analysed_at == HALL_ANALYSIS  # the HALL's frame, from the hall's config
    assert report.widths[0] == FACE_WIDTH_AT_HALL  # a 100px HD face is 50px here
    assert report.fraction_clearing == 0.0  # ...and 50px is under the strict 60px gate


def test_the_burst_is_measured_as_decoded_because_that_is_how_it_is_analysed() -> None:
    report = _measure(_burst())

    assert report.analysed_at == (HD_WIDTH, HD_HEIGHT)
    assert report.widths[0] == FACE_WIDTH_AT_HD
    assert report.fraction_clearing == 1.0


def test_the_same_camera_gives_opposite_answers_on_its_two_streams() -> None:
    """One number per CAMERA would have to be one of these two, and one of them is a lie.

    The burst says every face clears the gate. The substream the worker actually analyses
    says none of them do. Only the second one is about production.
    """
    burst = _measure(_burst())
    analysis = _measure(_analysis())

    assert burst.fraction_clearing > analysis.fraction_clearing
    assert analysis.fraction_clearing == 0.0

    refuse_if_hopeless(burst)  # the HD stream looks fine, and it is irrelevant
    with pytest.raises(CameraCannotRecognise):
        refuse_if_hopeless(analysis)  # ...and this is the stream that decides


def test_lowering_the_gate_admits_faces_and_still_recovers_nothing() -> None:
    """The trap a plausible-but-wrong 2.2% sets: "low but non-zero, so tune it lower".

    **The mechanism here was itself wrong once, and this test now pins the right one.** When
    the hall was measured at base.yaml's 960x540, a 100 px HD face came out 37 px and the
    story was "not one face even REACHES the 38 px small-face gate". At the hall's real
    1280x720 that same face is **50 px** and it clears the gate comfortably -- in the real
    corpus, 77 of 14 970 do (0.51%).

    So lowering the gate is not refuted by size. It is refuted by SCORE: of those 77 faces,
    **zero** are accepted at min_score 0.45 or 0.50, and the best score among all of them is
    **0.350**. The faces that are big enough are still far too degraded to match anybody.

    Same conclusion -- the hall recognises nobody -- by a different and correct route. A
    size-only check would now happily report "0.51% clear the gate" and invite exactly the
    tuning it is meant to forbid, which is why `refuse_if_hopeless` keys off the strict gate.
    """
    handle = FakeCapture(20)
    small_face_gate = FaceGate(min_width=38, min_height=48, min_area=1800)

    report = measure_faces(
        sample(handle, _analysis(), frames=20, stride=1),
        ProportionalDetector(),
        small_face_gate,
        source="hall_left/analysis",
    )

    # It CLEARS the lowered gate -- 50px >= 38px. The old expectation of 0 was an artefact
    # of measuring the hall at a resolution the hall does not run at.
    assert report.clearing_gate == 20
    assert report.widths[0] == FACE_WIDTH_AT_HALL

    # ...and it clears NOTHING at the strict gate, which is the one that decides.
    strict = measure_faces(
        sample(FakeCapture(20), _analysis(), frames=20, stride=1),
        ProportionalDetector(),
        FaceGate(),
        source="hall_left/analysis",
    )
    assert strict.clearing_gate == 0


def test_a_camera_is_two_streams_and_only_the_analysis_stream_gates_identity() -> None:
    from qorgan.config.loader import load_cameras
    from tests.conftest import CONFIG_DIR

    camera = load_cameras(CONFIG_DIR)["hall_left"]
    streams = streams_for(camera)

    names = [spec.name for spec in streams]
    assert names == ["analysis", "burst"]

    analysis, burst = streams
    assert analysis.capture is camera.capture  # scaled to what the worker feeds YOLO
    assert analysis.gates_identity
    assert burst.capture is None  # the burst is analysed as decoded
    assert not burst.gates_identity


def test_a_clip_is_reported_both_as_recorded_and_as_production_would_analyse_it() -> None:
    """The 250 hall clips ARE the HD burst. Measuring them as-recorded is what produced
    2.2%; measuring them at THE CAMERA'S OWN capture settings is what production would see.

    Note it takes the capture settings it is GIVEN. Hand it the hall's and it models the
    hall; hand it base.yaml's default and it models a camera nobody is asking about.
    """
    hall = _hall_capture()
    streams = clip_streams(hall)

    assert [spec.name for spec in streams] == ["as-recorded", "analysis"]
    assert streams[0].capture is None
    assert streams[1].capture == hall
    assert (hall.frame_width, hall.frame_height) == HALL_ANALYSIS
    assert streams[1].gates_identity


def test_each_row_says_whether_it_is_the_stream_the_worker_analyses() -> None:
    """Two rows, one of which is about production. The reader must not have to guess."""
    assert "THIS is the stream the worker analyses" in _analysis().note()
    assert "NOT the stream the worker analyses" in _burst().note()


def test_sampling_strides_across_the_source_instead_of_one_second_of_it() -> None:
    handle = FakeCapture(10)

    taken = list(sample(handle, _burst(), frames=100, stride=3))

    assert len(taken) == 4  # frames 0, 3, 6, 9 of the ten available -- then the clip ends


def test_sampling_stops_at_the_frame_budget() -> None:
    handle = FakeCapture(100)

    taken = list(sample(handle, _burst(), frames=5, stride=2))

    assert len(taken) == 5

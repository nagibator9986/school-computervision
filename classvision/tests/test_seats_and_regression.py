"""Seat discovery, the perspective warp, and a regression pin on the real lesson.

The last test in this file is the important one. `overlay.agrees_with_artefact()` already
proves that the verification video and the report come from the same computation; making
it a *test* is what stops that agreement quietly lapsing. It is the only end-to-end check
in the suite and it runs from the cached detections, so it costs seconds rather than the
15 minutes the model costs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from classvision.room import perspective, seats

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "classvision" / ".cache"
FULL = ROOT / "classvision" / "out" / "full_lesson.analysis.json"
VIDEO = ROOT / "test_camera.mp4"

# What a human counted in a frame of this room. Every number below is checked against it.
PUPILS = 8
PEOPLE = 9   # ...plus the adult at the front desk


def _cached_detections():
    files = sorted(CACHE.glob("det_*.npz"), key=lambda p: -p.stat().st_size)
    if not files:
        pytest.skip("no cached detections; run `classvision analyse` first")
    from classvision.pipeline import Detections

    return Detections.load(files[0])


def _anchors(detections):
    from classvision.geometry import Keypoints, anchor, shoulder_width

    out = []
    for i in range(len(detections.times)):
        person = Keypoints(xy=detections.xy[i], conf=detections.conf[i])
        scale = shoulder_width(person)
        position = anchor(person)
        if scale is not None and position is not None:
            out.append(seats.Anchor(float(detections.times[i]), position, scale, None))
    return out


# -- the perspective warp --------------------------------------------------------------

def test_a_flat_room_degrades_to_the_affine_map_rather_than_a_logarithm():
    """When everyone is the same size there is no gradient to integrate, and `ln(a·y+b)/a`
    is undefined. The warp must notice rather than divide by something near zero."""
    positions = np.column_stack([np.linspace(0, 1000, 50), np.linspace(0, 800, 50)])
    model = perspective.fit(positions, np.full(50, 90.0))
    assert model.mode in ("affine", "degenerate")


def test_the_warp_equalises_scale_on_the_real_room():
    detections = _cached_detections()
    anchors = _anchors(detections)
    positions = np.array([a.position for a in anchors])
    scales = np.array([a.scale for a in anchors])
    model = perspective.fit(positions, scales)

    assert model.usable, model.to_dict()
    assert model.r_squared > 0.8, model.to_dict()

    # The point of the warp: the ratio of true to predicted scale should be far tighter
    # than the raw pixel spread it replaces.
    ratio = scales / model.scale_at(positions[:, 1])
    raw_spread = float(np.percentile(scales, 90) / np.percentile(scales, 10))
    warped_spread = float(np.percentile(ratio, 90) / np.percentile(ratio, 10))
    assert warped_spread < raw_spread / 1.5, (raw_spread, warped_spread)


# -- seat discovery --------------------------------------------------------------------

def test_discovery_finds_one_place_per_person_on_the_real_lesson():
    detections = _cached_detections()
    anchors = _anchors(detections)
    frames = len({float(t) for t in detections.frame_times})
    found, diagnostics = seats.discover(anchors, analysed_frames=frames,
                                        expected_people=float(PEOPLE))
    assert len(found) == PEOPLE, diagnostics
    assert diagnostics["plausible"] is True
    assert diagnostics["method"] == "dbscan_in_perspective_corrected_space"
    # Every place must be genuinely occupied and genuinely compact, or it is a corridor.
    for seat in found:
        assert seat.occupancy >= seats.MIN_OCCUPANCY
        assert seat.spread <= seats.MAX_SPREAD_SCALES


def test_discovery_refuses_when_it_has_nothing_to_check_itself_against():
    """THE REGRESSION THIS TEST EXISTS FOR. `_choose_threshold` used to fall back to a
    single-link threshold of 0.9, which is outside `EPS_SWEEP` entirely; handed to DBSCAN
    as `eps` it returned ZERO seats — an empty room, reported confidently, with every
    downstream count correctly summing to nothing."""
    anchors = [seats.Anchor(float(i), (100.0 + i % 3, 200.0), 50.0, None) for i in range(60)]
    with pytest.raises(ValueError, match="expected_people"):
        seats.discover(anchors, analysed_frames=60, expected_people=None)


def test_the_eps_sweep_is_reported_so_the_chosen_value_can_be_argued_with():
    detections = _cached_detections()
    frames = len({float(t) for t in detections.frame_times})
    _, diagnostics = seats.discover(_anchors(detections), analysed_frames=frames,
                                    expected_people=float(PEOPLE))
    sweep = diagnostics["seat_count_by_eps"]
    assert len(sweep) == len(seats.EPS_SWEEP)
    assert any(int(v) == PEOPLE for v in sweep.values())


# -- the end-to-end pin ----------------------------------------------------------------

@pytest.mark.skipif(not FULL.exists(), reason="run `classvision analyse` first")
def test_the_artefact_describes_the_room_a_human_counted():
    artefact = json.loads(FULL.read_text(encoding="utf-8"))
    assert artefact["lesson"]["pupil_seats"] == PUPILS
    assert artefact["lesson"]["adult_seat"] is not None
    assert artefact["provenance"]["clock_source"] == "overlay"
    # Everything the run could not do must be listed, not omitted. This recording has no
    # audio, so «ответил вслух» must appear.
    unmeasured = " ".join(u["what"] for u in artefact["lesson"]["unmeasured"])
    assert "ответил" in unmeasured
    # No count may be rendered without its coverage.
    for seat in artefact["seats"]:
        assert "coverage" in seat["ledger"]


@pytest.mark.skipif(not (FULL.exists() and VIDEO.exists()),
                    reason="needs the artefact and the video")
def test_the_verification_overlay_reproduces_the_report_exactly():
    """A verification artefact that disagrees with its subject is worse than none: it
    manufactures doubt about correct numbers and confidence about wrong ones. This once
    failed by one observation per seat, because the artefact stored the lesson window
    rounded to 0.1 s and replaying it selected one frame fewer."""
    from classvision.report.overlay import agrees_with_artefact

    result = agrees_with_artefact(VIDEO, FULL)
    assert result["ok"], result["mismatches"]
    assert result["seats_checked"] == PUPILS

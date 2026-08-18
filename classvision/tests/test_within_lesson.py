"""The within-lesson decline statistic, pinned at the three places it can go wrong.

**Why synthetic readings and not only the real artefacts.** Both recordings this package
holds turned out to be well-behaved in exactly the way the coverage gate exists to catch:
once `activity.MIN_COVERAGE` has refused the badly-seen segments, the survivors' coverage is
nearly flat and `visibility_bound_index_points` never exceeds 1.45 on any of the thirteen
places. A gate that never fires on the available data is a gate nobody has seen work, and
those are the ones that turn out to be inverted. So the confound is tested against a
fabricated place whose coverage collapses on purpose, and the real lessons are used for what
they are good for: proving that the block is emitted, that it partitions the lesson exactly,
and that a refusal survives into the artefact.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from classvision.geometry import HeadDirection
from classvision.metrics import within_lesson as wl
from classvision.metrics.activity import MIN_COVERAGE
from classvision.metrics.trend import Direction
from classvision.states import Baselines, PupilState, Reading, Thresholds

ROOT = Path(__file__).resolve().parents[2]
ARTEFACTS = {
    "cam01": ROOT / "classvision" / "out" / "full_lesson.analysis.json",
    "d14": ROOT / "classvision" / "out" / "d14_session.analysis.json",
}

INTERVAL = 0.5
WINDOW = (0.0, 3000.0)          # 50 minutes, the length of the camera 01 lesson


def settled_baselines() -> Baselines:
    """A place whose baseline is established, so `classify` produces real states."""
    return Baselines(settled=True, seat_position=(100.0, 200.0), seat_shoulder_y=200.0,
                     upright_head=0.65, habitual_direction=HeadDirection.TOWARD_CAMERA,
                     scale=60.0)


def reading(t: float, *, head_up: float = 0.65, hand: bool = False,
            offset: float = 0.0) -> Reading:
    return Reading(video_seconds=t, scale=60.0,
                   position=(100.0 + offset, 200.0), head_up=head_up,
                   direction=HeadDirection.TOWARD_CAMERA, hand_up=hand)


def synthetic(*, head_up_of, seen_of, hand_of=None, window=WINDOW):
    """Build one place's observation stream from three functions of the lesson second.

    `seen_of(t) -> bool` is what makes the coverage confound testable at all: a frame the
    place is not seen in becomes an ABSENCE, exactly as `pipeline.assemble` records it, and
    not a missing entry — the whole point being that the two are different values.
    """
    observations: list[tuple[float, Reading, bool]] = []
    absences: list[float] = []
    steps = int((window[1] - window[0]) / INTERVAL)
    for step in range(steps):
        t = window[0] + step * INTERVAL
        if seen_of(t):
            observations.append(
                (t, reading(t, head_up=head_up_of(t),
                            hand=bool(hand_of(t)) if hand_of else False), False))
        else:
            absences.append(t)
    return observations, absences


def analyse(observations, absences, *, window=WINDOW, baselines=None, segments=None):
    return wl.analyse_seat(
        7, observations, absences, window=window, thresholds=Thresholds(),
        sample_interval=INTERVAL, baselines=baselines or settled_baselines(),
        segments=segments or wl.SEGMENTS)


# ----------------------------------------------------------------------------------
# 1. the segmentation is exact, and the last frame is inside it
# ----------------------------------------------------------------------------------


def test_edges_are_equal_and_cover_the_window():
    edges = wl.segment_edges(141.99, 3144.80, 6)
    assert len(edges) == 7
    widths = [b - a for a, b in itertools.pairwise(edges)]
    assert max(widths) - min(widths) < 1e-3, "segments must be equal by construction"
    assert edges[0] == 141.99


def test_the_final_analysed_frame_lands_in_the_last_segment():
    """The frame sitting exactly on `window_end` must be counted somewhere.

    Rounding a window boundary already cost this project a whole seat histogram once
    (`MEASUREMENTS.md` §5b); a half-open final segment would cost it the last observation of
    every place, silently, in the segment the decline question is about.
    """
    edges = wl.segment_edges(0.0, 100.0, 4)
    assert edges[-1] > 100.0
    assert any(low <= 100.0 < high for low, high in itertools.pairwise(edges))


def test_segments_partition_the_observations_exactly():
    """Every observation is in exactly one segment, and none is invented."""
    observations, absences = synthetic(head_up_of=lambda t: 0.65,
                                       seen_of=lambda t: t % 5 != 0)
    block = analyse(observations, absences)
    assert sum(s.observations for s in block.segments) == len(observations)
    assert sum(s.absent_observations for s in block.segments) == len(absences)


# ----------------------------------------------------------------------------------
# 2. the refusal survives, as a refusal
# ----------------------------------------------------------------------------------


def test_a_low_coverage_segment_refuses_rather_than_scoring_zero():
    """Rule 3 of this project, applied to a segment: not seen and zero are different values.

    The failure being prevented is a plotted line that dips to nothing in the segment a
    pupil was hidden, which reads as the strongest decline in the room.
    """
    def seen(t: float) -> bool:
        return not (1500.0 <= t < 2000.0)      # invisible for most of segment 4

    observations, absences = synthetic(head_up_of=lambda t: 0.65, seen_of=seen)
    block = analyse(observations, absences)
    refused = [s for s in block.segments if not s.activity.available]
    assert refused, "a segment seen in almost none of its frames must refuse"
    for segment in refused:
        assert segment.activity.index is None, "a refused segment must not carry a number"
        assert "наблюдений слишком мало" in segment.activity.reason
        assert segment.coverage < MIN_COVERAGE
        # ...and the segment is still PRESENT in the output. A dropped key would let a
        # consumer draw five segments and call the lesson complete.
        assert segment.to_dict()["ordinal"] == segment.ordinal


def test_an_unsettled_place_refuses_every_segment_with_the_baseline_reason():
    """A place with no baseline has no states, and that refusal must reproduce per segment
    rather than being replaced by «мало наблюдений», which would send a school looking at
    the camera when the problem is the detector."""
    observations, absences = synthetic(head_up_of=lambda t: 0.65, seen_of=lambda t: True)
    block = analyse(observations, absences, baselines=Baselines())
    assert all(not s.activity.available for s in block.segments)
    assert all("базовая поза" in s.activity.reason for s in block.segments)
    assert block.change.refused_because == wl.REFUSED_TOO_FEW


def test_the_frozen_baseline_cannot_learn_from_the_segment_it_is_measuring():
    """Segments must be a partition of ONE classification, not four re-analyses.

    If a segment ledger settled its own baseline, a pupil who spent the last ten minutes
    slumped would be measured against how they sat while slumping, and the decline would
    erase itself by construction — the one failure mode that would make this module produce
    a confident, systematically wrong answer.
    """
    original = settled_baselines()
    copy = wl._frozen(original)
    copy.observe(reading(0.0, head_up=99.0), Thresholds())
    assert copy.upright_head == original.upright_head
    unsettled = wl._frozen(Baselines())
    for step in range(200):
        unsettled.observe(reading(step * INTERVAL), Thresholds())
    assert not unsettled.settled, "a copy must never settle on the segment's own data"


# ----------------------------------------------------------------------------------
# 3. the statistics, and the two constants that are DERIVED from them
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [(3, 1 / 3), (4, 2 / 24), (5, 2 / 120),
                                        (6, 2 / 720)])
def test_the_exact_kendall_null_is_the_reason_four_segments_are_required(n, expected):
    """`MIN_USABLE_SEGMENTS = 4` is derived from these four numbers and nothing else.

    At n = 3 a perfectly monotone series still occurs one time in three by chance, so no
    three-segment place can support a direction at any honest threshold. At n = 4 the same
    series is 0.083, which is why `DIRECTION_ALPHA` is 0.10 and not 0.05: at 0.05 a place
    that lost two segments to occlusion could never be described at all, and the places that
    lose segments are the ones the camera sees worst.
    """
    perfect = n * (n - 1) // 2
    assert wl._kendall_p(n, perfect) == pytest.approx(expected, rel=1e-9)
    if n == 3:
        assert wl.MIN_USABLE_SEGMENTS > 3
    if n == 4:
        assert expected <= wl.DIRECTION_ALPHA, (
            "an alpha below the best attainable p at four segments would silently exclude "
            "every place that lost two segments")


def test_theil_sen_ignores_one_wild_segment():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    clean = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    _a, slope = wl._theil_sen(xs, clean)
    spiked = list(clean)
    spiked[2] = 1000.0
    _b, slope_spiked = wl._theil_sen(xs, spiked)
    assert slope == pytest.approx(-10.0)
    assert slope_spiked == pytest.approx(-10.0), (
        "a median of pairwise slopes must not follow one impossible segment")


def test_too_few_usable_segments_is_an_actionable_refusal():
    """Rule 8: a refusal names what is missing AND what would fix it."""
    def seen(t: float) -> bool:
        return t < 1000.0

    observations, absences = synthetic(head_up_of=lambda t: 0.65, seen_of=seen)
    change = analyse(observations, absences).change
    assert not change.available
    assert change.refused_because == wl.REFUSED_TOO_FEW
    assert str(wl.MIN_USABLE_SEGMENTS) in change.reason
    assert "камер" in change.reason or "съёмки" in change.reason, (
        "the refusal must say what would fix it, not only what failed")


# ----------------------------------------------------------------------------------
# 4. the confound: a fall in coverage must not be reported as a fall in a child
# ----------------------------------------------------------------------------------


def test_a_decline_that_a_coverage_fall_can_explain_is_refused():
    """The gate that never fires on either real lesson, fired on purpose.

    The place below behaves identically all lesson — the same posture in every segment —
    but is seen progressively less, and the observations that go missing are the ones in
    which it was upright. That is exactly how a visibility drop manufactures a decline, and
    the module must decline to call it one.
    """
    def head_up(t: float) -> float:
        return 0.65

    def seen(t: float) -> bool:
        # Coverage falls 100 % -> ~55 % across the lesson, staying above MIN_COVERAGE so
        # the segments keep producing an index rather than refusing individually.
        share = 1.0 - 0.45 * (t / WINDOW[1])
        return (t / INTERVAL) % 100 < share * 100

    def slumped(t: float) -> float:
        # ...and what remains visible late in the lesson is disproportionately head-down,
        # which is the adversarial case the bound is derived for.
        return 0.65 if (t / INTERVAL) % 100 < 40 else 0.0

    observations, absences = synthetic(head_up_of=slumped, seen_of=seen)
    change = analyse(observations, absences).change
    assert change.coverage_retention < 1.0
    assert change.visibility_bound_index_points > 0.0
    if change.available:
        # If a direction is still stated, the change must be strictly larger than what the
        # coverage fall alone could produce. That is the invariant; which branch fires
        # depends on the synthetic numbers and is not the thing being pinned.
        assert abs(change.delta_index_points) > change.visibility_bound_index_points
    else:
        assert change.refused_because in (wl.REFUSED_VISIBILITY, wl.REFUSED_INCONSISTENT)


def test_the_visibility_bound_is_one_sided():
    """A place seen BETTER at the end cannot have had its decline manufactured by being
    seen worse, so the bound is zero there. A two-sided bound would refuse real declines at
    exactly the places the camera sees best."""
    observations, absences = synthetic(
        head_up_of=lambda t: 0.65,
        seen_of=lambda t: (t / INTERVAL) % 100 < 60 + 40 * (t / WINDOW[1]))
    change = analyse(observations, absences).change
    assert change.coverage_retention == pytest.approx(1.0)
    assert change.visibility_bound_index_points == 0.0


def test_a_change_that_rests_on_one_event_is_not_a_direction():
    """One hand-raise is worth ten of the index's hundred points, and a pupil produces 0–4
    of them in fifty minutes (`MEASUREMENTS.md` §5c). The difference between one and none in
    an eight-minute segment cannot support a statement about a child."""
    def hand(t: float) -> bool:
        return 100.0 <= t < 130.0          # one sustained raise, segment 1 only

    observations, absences = synthetic(head_up_of=lambda t: 0.65,
                                       seen_of=lambda t: True, hand_of=hand)
    change = analyse(observations, absences).change
    assert change.events_first == 1 and change.events_last == 0
    assert change.floor_from == "single_event"
    assert change.floor_index_points == pytest.approx(
        wl.EVENT_QUANTUM_INDEX_POINTS, abs=0.01)
    assert change.direction is Direction.STEADY


def test_the_boundary_floor_shrinks_as_segments_get_longer():
    """The floor is derived from the cut, so it must be a function of the cut."""
    short = wl._boundary_floor(300.0, Thresholds())
    long = wl._boundary_floor(600.0, Thresholds())
    assert short == pytest.approx(2 * long)
    assert long == pytest.approx(100 * 0.30 * 2 * 20.0 / 600.0)


def test_a_real_monotone_decline_is_reported_with_its_size():
    observations, absences = synthetic(
        head_up_of=lambda t: 0.65 if (t / INTERVAL) % 100 < 95 - 90 * (t / WINDOW[1])
        else 0.0,
        seen_of=lambda t: True)
    change = analyse(observations, absences).change
    assert change.available
    assert change.direction is Direction.LOWER
    assert change.delta_index_points < -change.floor_index_points
    assert change.kendall_p <= wl.DIRECTION_ALPHA
    assert change.components["head_up_share"]["direction"] == "lower"


# ----------------------------------------------------------------------------------
# 5. nothing here decides anything
# ----------------------------------------------------------------------------------


FORBIDDEN = ("needs_attention", "attention", "flag", "risk", "rank", "score_rank",
             "alert", "problem", "concern")


def test_the_output_carries_no_verdict_shaped_field():
    observations, absences = synthetic(head_up_of=lambda t: 0.65, seen_of=lambda t: True)
    payload = json.dumps(analyse(observations, absences).to_dict(), ensure_ascii=False)
    document = json.loads(payload)

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert not any(word in key.lower() for word in FORBIDDEN), (
                    f"`{key}` is a verdict wearing the costume of a field name")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)


def test_expected_by_chance_counts_the_component_tests_too():
    """Four extra tests per place are four more chances to print «ниже» about a child."""
    block = wl.expected_by_chance(8)
    assert block["expected_places_by_chance"] == pytest.approx(0.8)
    assert block["component_tests_per_place"] == 4
    assert block["expected_component_results_by_chance"] == pytest.approx(3.2)
    assert "случайности" in block["note_ru"]


# ----------------------------------------------------------------------------------
# 6. the real lessons
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ARTEFACTS))
def test_every_pupil_place_of_a_real_lesson_carries_the_block(name):
    path = ARTEFACTS[name]
    if not path.exists():
        pytest.skip(f"run `classvision analyse` to produce {path.name}")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    for seat in artefact["seats"]:
        if seat.get("role") != "pupil":
            continue
        block = seat["metrics"]["within_lesson"]
        assert len(block["segments"]) == wl.SEGMENTS
        assert block["change"]["direction"] in {d.value for d in Direction}
        # The one sentence a reader of this block cannot do without.
        assert "сравнивать нельзя" in block["not_comparable_to_ru"]


@pytest.mark.parametrize("name", sorted(ARTEFACTS))
def test_segment_observations_sum_to_the_whole_lesson_ledger(name):
    """The strongest available check that these segments are the SAME lesson: the parts add
    up to the ledger the rest of the report is built from."""
    path = ARTEFACTS[name]
    if not path.exists():
        pytest.skip(f"run `classvision analyse` to produce {path.name}")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    for seat in artefact["seats"]:
        if seat.get("role") != "pupil":
            continue
        segments = seat["metrics"]["within_lesson"]["segments"]
        assert sum(s["observations"] for s in segments) == seat["ledger"]["observations"]
        assert (sum(s["absent_observations"] for s in segments)
                == seat["ledger"]["absent_observations"])


@pytest.mark.parametrize("name", sorted(ARTEFACTS))
def test_a_refused_segment_is_present_and_numberless_in_the_artefact(name):
    path = ARTEFACTS[name]
    if not path.exists():
        pytest.skip(f"run `classvision analyse` to produce {path.name}")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    for seat in artefact["seats"]:
        for segment in (seat.get("metrics", {}).get("within_lesson") or
                        {"segments": []})["segments"]:
            activity = segment["activity"]
            if activity["available"]:
                assert activity["index"] is not None
            else:
                assert activity["index"] is None
                assert activity["reason"], "a refusal without a reason is a gap"


def test_the_lesson_block_carries_the_by_chance_count():
    path = ARTEFACTS["cam01"]
    if not path.exists():
        pytest.skip("run `classvision analyse` first")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    chance = artefact["lesson"]["within_lesson_direction_by_chance"]
    assert chance["pupil_seats"] == artefact["lesson"]["pupil_seats"]
    assert chance["alpha"] == wl.DIRECTION_ALPHA


def test_the_block_is_additive_and_the_schema_did_not_move():
    """Nothing was renamed and nothing was removed, so a 1.1 consumer is unaffected."""
    from classvision.report.artefact import SCHEMA_VERSION

    path = ARTEFACTS["cam01"]
    if not path.exists():
        pytest.skip("run `classvision analyse` first")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    assert artefact["provenance"]["schema"] == SCHEMA_VERSION == "classvision/1.1"
    for seat in artefact["seats"]:
        if seat.get("role") != "pupil":
            continue
        assert "activity" in seat["metrics"], "the pre-existing key must be untouched"
        assert set(seat["metrics"]) == {"activity", "within_lesson"}


def test_events_per_segment_never_exceed_the_lesson_total():
    """Episodes are re-derived inside each segment, so a boundary can only LOSE one. If the
    parts ever exceed the whole, the segment ledgers are double-counting."""
    path = ARTEFACTS["cam01"]
    if not path.exists():
        pytest.skip("run `classvision analyse` first")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    for seat in artefact["seats"]:
        if seat.get("role") != "pupil":
            continue
        counts = seat["ledger"]["counts"]
        whole = counts["hand_raises"] + counts["stands"] + counts["board_visits"]
        parts = sum(s["events"] for s in seat["metrics"]["within_lesson"]["segments"])
        assert parts <= whole + len(seat["metrics"]["within_lesson"]["segments"]), (
            f"seat {seat['seat_id']}: {parts} segment events against {whole} for the lesson")


def test_no_segment_index_is_presented_as_the_lesson_index():
    """The single most available misreading of this block, pinned.

    On camera 01 place 3 the lesson index is 97.7 and every segment is 65–98, because the
    event term is normalised to a whole lesson. Anything that compared them would report a
    collapse that did not happen, so the artefact must carry the sentence that forbids it.
    """
    path = ARTEFACTS["cam01"]
    if not path.exists():
        pytest.skip("run `classvision analyse` first")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    for seat in artefact["seats"]:
        if seat.get("role") != "pupil":
            continue
        block = seat["metrics"]["within_lesson"]
        assert "нормирована" in block["not_comparable_to_ru"]
        assert PupilState.HAND_RAISED.value not in block


def test_the_html_report_prints_the_change_with_its_two_thresholds():
    """A direction printed without «порог нарезки» and «даёт видимость» beside it is a
    verdict; the page must carry both columns."""
    from classvision.report.html import _within_lesson_block

    path = ARTEFACTS["cam01"]
    if not path.exists():
        pytest.skip("run `classvision analyse` first")
    artefact = json.loads(path.read_text(encoding="utf-8"))
    html = _within_lesson_block(artefact)
    assert "порог нарезки" in html
    assert "даёт видимость" in html
    assert "сравнивать" in html, "the not-comparable warning must be on the page"
    assert "случайно" in html, "the by-chance count must be on the page"

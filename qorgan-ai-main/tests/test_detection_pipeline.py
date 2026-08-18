"""The detector, driven with synthetic hallway scenes.

The gate tests check each rule in isolation. These check the thing that actually matters:
given a sequence of frames that looks like a real event, does the pipeline reach the
right verdict? An attack must alert. Corridor life must not.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from qorgan.config.bullying import BullyingConfig
from qorgan.detection.geometry import Box
from qorgan.detection.pipeline import BullyingDetector, FrameResult

WIDTH, HEIGHT = 1280, 720
FPS = 10.0
STEP = 1.0 / FPS

# Big enough to clear min_box_area, and centred away from the frame edge.
PERSON_W, PERSON_H = 90, 200


def _person(cx: float, cy: float = 400.0) -> Box:
    return Box(cx - PERSON_W / 2, cy - PERSON_H / 2, cx + PERSON_W / 2, cy + PERSON_H / 2)


def _detector(**overrides) -> BullyingDetector:
    config = BullyingConfig.model_validate(overrides)
    return BullyingDetector(config, WIDTH, HEIGHT)


def _run(detector: BullyingDetector, scene: Iterator[dict[int, Box]]) -> list[FrameResult]:
    return [
        detector.process(detections, timestamp=index * STEP)
        for index, detections in enumerate(scene)
    ]


def _alerted(results: list[FrameResult]) -> bool:
    return any(r.candidates for r in results)


def _gates_that_fired(results: list[FrameResult]) -> set[str]:
    return {s.gate for r in results for s in r.suppressions}


# -- scenes ----------------------------------------------------------------


def scene_two_children_talking(frames: int = 40) -> Iterator[dict[int, Box]]:
    """Standing close, chatting. The commonest non-event in a school.

    Centres ~260 px apart: with 90 px-wide boxes that is a body-width of air between
    them, which is what conversation actually looks like. (Any closer and they really
    are touching -- the proximity threshold scales with box size for exactly this
    reason, so "close" means the same thing at both ends of the corridor.)
    """
    for _ in range(frames):
        yield {1: _person(600), 2: _person(860)}


def scene_walking_past_each_other(frames: int = 40) -> Iterator[dict[int, Box]]:
    """Two people crossing in a corridor. The dominant false positive."""
    for index in range(frames):
        yield {1: _person(400 + index * 22), 2: _person(900 - index * 22)}


def scene_walking_together(frames: int = 40) -> Iterator[dict[int, Box]]:
    """Two friends walking side by side to a lesson."""
    for index in range(frames):
        yield {1: _person(400 + index * 18), 2: _person(500 + index * 18)}


def scene_an_assault(frames: int = 40) -> Iterator[dict[int, Box]]:
    """The real thing: a rush, then a sustained grapple with the victim displaced.

    A stands still. B charges in from the right, makes contact, and the two are then
    locked together, jerking about, with A being shoved backwards.
    """
    a_x, b_x = 640.0, 1000.0

    for index in range(frames):
        if index < 6:  # B charges
            b_x -= 55
        elif index < 24:  # the grapple: overlapping, chaotic, A driven back
            jitter = 18 if index % 2 else -18
            b_x = a_x + 30 + jitter
            a_x -= 9  # the victim is displaced -- this is what makes it an assault
        else:  # they break apart
            b_x += 40

        yield {1: _person(a_x), 2: _person(b_x)}


def scene_lingering_after_the_fight(frames: int = 25) -> Iterator[dict[int, Box]]:
    """Two children standing close and milling about after the fight is over.

    Centres ~225 px apart: inside contact distance (so the pair stays a live candidate)
    but never a fresh *hard* contact -- no new grab or shove. This is the situation the
    post-alert separation guard exists for: the fight is over, the two are still on
    camera, and nothing new is happening. It must not re-alert.
    """
    a_x = 560.0
    for index in range(frames):
        jostle = 8 if index % 2 else -8
        yield {1: _person(a_x), 2: _person(a_x + 225 + jostle)}


# -- what must NOT alert ---------------------------------------------------


def test_two_children_talking_do_not_alert() -> None:
    detector = _detector()
    results = _run(detector, scene_two_children_talking())

    assert not _alerted(results), "children chatting raised a bullying alert"
    assert "static_close" in _gates_that_fired(results)


def test_two_people_walking_past_each_other_do_not_alert() -> None:
    detector = _detector()
    results = _run(detector, scene_walking_past_each_other())

    assert not _alerted(results), "a corridor crossing raised a bullying alert"


def test_two_friends_walking_together_do_not_alert() -> None:
    detector = _detector()
    results = _run(detector, scene_walking_together())

    assert not _alerted(results), "two friends walking to class raised an alert"


def test_an_empty_corridor_produces_nothing() -> None:
    detector = _detector()
    results = _run(detector, iter([{}, {}, {}]))

    assert not _alerted(results)
    assert all(r.tracks == 0 for r in results)


def test_one_person_alone_produces_nothing() -> None:
    """It takes two. A lone child cannot be a pair."""
    detector = _detector()
    results = _run(detector, iter([{1: _person(600)} for _ in range(20)]))

    assert not _alerted(results)
    assert all(r.pairs == 0 for r in results)


# -- what MUST alert -------------------------------------------------------


def test_an_assault_alerts() -> None:
    """If this ever stops passing, the detector has stopped detecting."""
    detector = _detector()
    results = _run(detector, scene_an_assault())

    assert _alerted(results), (
        "a charge, a grapple, and a displaced victim did not raise an alert. "
        f"gates that fired: {sorted(_gates_that_fired(results))}"
    )


def test_the_alert_carries_the_evidence_needed_to_judge_it() -> None:
    detector = _detector()
    results = _run(detector, scene_an_assault())
    candidate = next(c for r in results for c in r.candidates)

    assert candidate.score >= candidate.threshold
    assert 0.0 <= candidate.probability <= 1.0
    assert candidate.probability > 0.5  # it cleared the bar, so it must read above half
    assert 0.0 <= candidate.center[0] <= 1.0
    assert candidate.key == (1, 2)


# -- the flow-zone probability fix, end to end -----------------------------


def test_a_flow_zone_event_is_not_handed_a_saturated_probability() -> None:
    """The legacy bug, from the outside.

    It damped the score in a corridor lane (x0.2) and raised the bar (to 5.0), then
    normalised the damped score against the UNdamped base threshold (1.4) -- so every
    flow-zone event came out at ~0.997. That number fed the burst trigger, the
    validation gate and 70% of the final confidence, which meant the zone built to
    suppress corridor false positives was rubber-stamping them instead.
    """
    flow_config = {
        "zones": {
            "normal_flow": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}],
            "normal_flow_score_multiplier": 0.2,
            "normal_flow_alert_threshold": 5.0,
        }
    }
    detector = _detector(**flow_config)
    results = _run(detector, scene_an_assault())

    for result in results:
        for candidate in result.candidates:
            assert candidate.threshold == 5.0, "the flow zone did not raise the bar"
            # Normalised against 5.0, an event scoring ~5 reads ~0.5 -- not ~0.997.
            assert candidate.probability < 0.95, (
                f"flow-zone probability saturated at {candidate.probability:.3f}; "
                "this is the legacy bug"
            )


# -- one incident is a bounded burst, not a per-frame siren ----------------


def test_one_incident_does_not_alert_on_every_frame() -> None:
    """A single assault must not alert on all 60 frames.

    NOTE: despite where it once sat, this does NOT exercise the post-alert separation
    guard. It never calls mark_alerted(), so `alerted` stays False and the guard never
    arms. What bounds the burst here is the assault's own dynamics -- the fighters break
    apart (scene_an_assault, index >= 24) and the escalation counters drain. The guard
    itself is armed and proven in
    test_after_an_alert_a_new_alert_needs_fresh_contact_not_lingering below.
    """
    detector = _detector()
    results = _run(detector, scene_an_assault(frames=60))

    total = sum(len(r.candidates) for r in results)
    assert total > 0, "the assault was missed entirely"
    # The alert is allowed to persist for a few frames while the fight is happening;
    # what it must NOT do is fire on all 60.
    assert total < 30, f"one incident produced {total} candidate frames"


# -- the post-alert separation guard (armed the way production arms it) -----


def test_after_an_alert_a_new_alert_needs_fresh_contact_not_lingering() -> None:
    """After a real alert, mere continued proximity must not raise a second alert.

    This is the test the guard never had. The sibling test bounds one incident without
    ever arming the guard; here we arm it as qorgan.worker.bullying does -- mark_alerted()
    after the event -- then show that once the pair separates and drifts back together
    with no fresh physical contact, post_alert_separation catches every frame. Only a new
    hard contact may re-open the alert; lingering is not enough.
    """
    detector = _detector()
    t = 0.0

    # A real assault, and a real alert.
    first_alert = False
    for detections in scene_an_assault():
        result = detector.process(detections, timestamp=t)
        t += STEP
        first_alert = first_alert or bool(result.candidates)
    assert first_alert, "the assault must alert before the post-alert guard can be tested"

    # The worker marks the pair alerted after recording the event; the pipeline never does.
    state = detector.pairs.get((1, 2), t)
    state.mark_alerted()
    assert state.alerted and not state.awaiting_new_contact

    # The victim is displaced, so the pair separates: the guard arms and forces the
    # fight's escalation counters back down -- fresh contact is now required to alert.
    for _ in range(4):
        detector.process({1: _person(400), 2: _person(1000)}, timestamp=t)
        t += STEP
    assert state.awaiting_new_contact, "separation after an alert did not arm the guard"
    assert state.aggression_frames == 0, "the guard did not force the escalation counters down"

    # They drift back to standing close and mill about -- close enough to touch, but with
    # no new grab or shove. The leftover counters must not re-confirm into a second alert.
    second_alerts = 0
    guard_engaged = False
    for detections in scene_lingering_after_the_fight(frames=25):
        result = detector.process(detections, timestamp=t)
        t += STEP
        second_alerts += len(result.candidates)
        guard_engaged = guard_engaged or any(
            s.gate == "post_alert_separation" for s in result.suppressions
        )

    assert guard_engaged, "the post-alert separation guard never fired on the lingering pair"
    assert second_alerts == 0, (
        f"lingering proximity re-alerted {second_alerts} time(s) with no fresh contact"
    )
    assert state.awaiting_new_contact, "the guard disarmed without any fresh hard contact"


# -- memory ----------------------------------------------------------------


def test_pair_state_is_bounded() -> None:
    """Rule R8. Track ids only ever increase, so a dict keyed on PAIRS of them grows
    quadratically and forever. The legacy leaked exactly this."""
    detector = _detector()

    for index in range(200):
        # A fresh pair of track ids every frame: the worst case for the pair store.
        detector.process(
            {index * 2: _person(600), index * 2 + 1: _person(700)},
            timestamp=index * STEP,
        )

    assert len(detector.pairs) < 50, f"pair store grew to {len(detector.pairs)}"
    assert len(detector.tracks.tracks) < 50


@pytest.mark.parametrize("fps", [8.0, 25.0])
def test_the_same_assault_is_detected_at_any_frame_rate(fps: float) -> None:
    """The acceleration fix, from the outside. The same physical event, filmed at 8 fps
    and at 25 fps, must reach the same verdict -- otherwise no threshold transfers
    between cameras and none of the tuning means anything."""
    detector = _detector()
    step = 1.0 / fps

    # Replay the assault in real time, sampled at this camera's rate.
    results = []
    scene = list(scene_an_assault())
    duration = len(scene) * STEP
    samples = int(duration * fps)

    for index in range(samples):
        t = index * step
        frame_index = min(int(t / STEP), len(scene) - 1)
        results.append(detector.process(scene[frame_index], timestamp=t))

    assert _alerted(results), f"the assault was missed at {fps} fps"

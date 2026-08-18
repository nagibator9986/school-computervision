"""§12.1's sequence, driven end to end against a synthetic detector.

    объект обнаружен -> отслеживается несколько кадров -> проверка confidence
    -> проверка нахождения рядом с человеком -> повторное подтверждение
    -> snapshot и clip -> критическое уведомление

The first five steps are `WeaponsDetector`, and they are what this file drives. The last
two touch disk and the network and are `worker/weapons.py`'s.

**Never on one frame.** §12.1 says so in those words, and it is the property most easily
lost to a well-meaning edit, so it is asserted from three directions here: the count gate,
the reconfirmation gate, and the schema's own refusal to be configured out of either
(`test_weapons_config.py`).

The detector under test is the same object the worker drives (rule R2). There is no
"test pipeline" and no copy of the decision.
"""

from __future__ import annotations

from qorgan.config.weapons import WeaponsConfig
from qorgan.weapons.pipeline import (
    NEAR_A_PERSON,
    RECONFIRMED,
    TRACKED_ACROSS_FRAMES,
    WeaponsDetector,
)
from tests.weapons_fixtures import person_box, sighting

WIDTH, HEIGHT = 960, 540


def _detector(**overrides) -> WeaponsDetector:
    base = {"model": {"model": "qorgan-weapons.pt"}, "target_classes": ["knife", "firearm"]}
    base.update(overrides)
    return WeaponsDetector(WeaponsConfig.model_validate(base), WIDTH, HEIGHT)


def _people() -> dict[int, object]:
    return {7: person_box()}


def _run(detector: WeaponsDetector, frames: int, *, confidence: float = 0.9, step: float = 0.2):
    """Show the same knife, in the same hand, for `frames` analysed frames."""
    outcomes = []
    for index in range(frames):
        outcomes.append(
            detector.process([sighting("knife", confidence)], _people(), index * step)
        )
    return outcomes


# -- «НЕ отправлять тревогу по одному кадру» -------------------------------


def test_one_frame_never_alerts() -> None:
    outcomes = _run(_detector(), 1)
    assert outcomes[0].alerts == []
    assert outcomes[0].tracked == 1, "it is being TRACKED; that is not the same as alarming"


def test_two_frames_still_do_not_alert() -> None:
    """The shipped `min_track_observations` is 3. Two is a flicker."""
    outcomes = _run(_detector(), 2)
    assert all(o.alerts == [] for o in outcomes)


def test_the_alert_arrives_on_the_frame_the_last_gate_closes() -> None:
    outcomes = _run(_detector(), 4)
    fired = [index for index, o in enumerate(outcomes) if o.alerts]
    assert fired == [2], "third analysed frame: observations 3 >= 3 and strong 2 >= 2"


def test_a_model_that_keeps_hedging_never_alerts() -> None:
    """The reconfirmation gate is a SECOND and stricter question, not a restatement.

    0.5 clears the entry bar (`model.conf` 0.35) so the track accumulates for as long as
    you like -- and never clears `reconfirm_confidence` (0.60). "The model kept saying
    maybe" and "the model was sure more than once" are the two cases §12.1 asks us to
    tell apart, and this is the one that must stay silent.
    """
    outcomes = _run(_detector(), 30, confidence=0.5)
    assert all(o.alerts == [] for o in outcomes)


def test_a_pen_at_the_edge_of_the_size_gate_can_persist_forever_and_never_alert() -> None:
    detector = _detector()
    for index in range(40):
        outcome = detector.process([sighting("knife", 0.99, size=20.0)], _people(), index * 0.2)
        assert outcome.alerts == []


# -- «проверка нахождения рядом с человеком», at the OBSERVATION -----------


def test_a_knife_on_a_poster_accumulates_nothing() -> None:
    """The person check happens where the observation is folded into a track, not at the
    alert gate. So a track's count cannot include frames the check would have refused --
    which is what has to be true for the number on the panel to mean what it says."""
    detector = _detector()
    for index in range(40):
        outcome = detector.process([sighting("knife", 0.99)], {}, index * 0.2)
        assert outcome.alerts == []
        assert outcome.tracked == 0, "it never became a track at all"


def test_frames_without_a_person_do_not_count_towards_the_gate() -> None:
    """Two good frames, a hundred with nobody in shot, then one more good frame. The
    third observation is the third one that was actually beside a person."""
    detector = _detector()
    detector.process([sighting("knife", 0.9)], _people(), 0.0)
    detector.process([sighting("knife", 0.9)], _people(), 0.2)
    for index in range(100):
        assert detector.process([sighting("knife", 0.9)], {}, 0.4 + index * 0.001).alerts == []

    outcome = detector.process([sighting("knife", 0.9)], _people(), 0.6)
    assert len(outcome.alerts) == 1


def test_the_alert_names_the_person_it_was_beside() -> None:
    """The operator opens the clip looking for this track id."""
    outcomes = _run(_detector(), 3)
    alert = outcomes[-1].alerts[0]
    assert alert.person_track_id == 7


# -- what the alert carries ------------------------------------------------


def test_the_alert_carries_all_three_gates_as_evidence() -> None:
    """Only because all three passed. The row can then be audited against the config that
    produced it rather than against today's YAML."""
    alert = _run(_detector(), 3)[-1].alerts[0]
    assert alert.reasons == (TRACKED_ACROSS_FRAMES, NEAR_A_PERSON, RECONFIRMED)


def test_the_confidence_on_the_alert_is_the_best_observation_never_a_mean() -> None:
    """The question is whether the model was ever sure. A mean over a track that begins
    as the object comes into view answers a different one."""
    detector = _detector()
    for index, confidence in enumerate([0.65, 0.95, 0.62]):
        outcome = detector.process([sighting("knife", confidence)], _people(), index * 0.2)
    assert outcome.alerts[0].confidence == 0.95


def test_the_alert_reports_both_counts_it_cleared() -> None:
    alert = _run(_detector(), 3)[-1].alerts[0]
    assert alert.observations == 3
    assert alert.strong_observations == 3


# -- one knife carried down one corridor is ONE question -------------------


def test_a_still_visible_weapon_does_not_re_alert_every_frame() -> None:
    """One human decision, not one a second. The track keeps accumulating; nobody is
    asked again yet."""
    detector = _detector()
    outcomes = _run(detector, 60)  # 60 frames * 0.2 s = 12 s, well inside realert 60 s
    assert sum(len(o.alerts) for o in outcomes) == 1


def test_the_quiet_period_ends_and_the_question_is_asked_again() -> None:
    """One knife still being carried, 2.6 s later, with the quiet period set to 2 s.

    The knife is visible on EVERY frame across the quiet period, which is what "still being
    carried" means. It used to be shown on four frames and then again 2 s later with nothing
    in between -- and it re-alerted on the same track, because expiry ran at the end of
    `process` and so could not fire during a gap in which `process` was never called. That
    is the same hole `test_a_weapon_seen_again_after_a_stream_outage_starts_a_new_track`
    now pins from the other side: a track idle for longer than `track_idle_seconds` (1.5)
    is gone, so the old scenario would produce a NEW track needing three fresh
    observations, and testing the quiet period through it was testing two things at once.
    """
    detector = _detector(realert_seconds=2.0)
    outcomes = _run(detector, 14)  # 14 frames * 0.2 s = 2.6 s, continuously visible
    assert sum(len(o.alerts) for o in outcomes) == 2


def test_a_weapon_seen_again_after_a_stream_outage_starts_a_new_track() -> None:
    """**«Отслеживание несколько кадров» must survive a reconnect.**

    Measured before this was fixed: two observations, then no `process` call at all for
    9.8 s (which is what an RTSP drop looks like from in here), then one sighting in the
    same place -- and the first frame back alerted, on a track holding three observations
    of which two predated the outage, with `track_idle_seconds` at 1.5.

    `_match` is spatial and has no clock, so the only thing that ever stopped that was
    expiry running on the frames that arrived DURING the absence. An outage is exactly the
    case where none arrive. One fresh frame plus two from before a disconnection is not
    three frames of tracking, and this module exists to refuse that kind of arithmetic.
    """
    detector = _detector()
    detector.process([sighting("knife", 0.9)], _people(), 0.0)
    detector.process([sighting("knife", 0.9)], _people(), 0.2)

    back = detector.process([sighting("knife", 0.9)], _people(), 10.0)
    assert back.alerts == [], "one frame after a 9.8 s outage is not three frames"
    assert back.tracked == 1, "it is being tracked afresh, which is not the same as alarming"
    assert detector.process([sighting("knife", 0.9)], _people(), 10.2).alerts == []
    assert len(detector.process([sighting("knife", 0.9)], _people(), 10.4).alerts) == 1


# -- refusals are part of the result, not debris ---------------------------


def test_refusals_are_returned_and_counted_by_reason() -> None:
    """Which screen a camera's sightings die on is the difference between "there are no
    weapons here" and "everything this camera sees is below the size gate"."""
    detector = _detector()
    outcome = detector.process(
        [
            sighting("knife", 0.9, x1=300, y1=300, size=10.0),  # too small
            sighting("toy", 0.9, x1=600, y1=400),  # not a target, and nowhere near the knife
            sighting("knife", 0.9),  # fine, but nobody is in the frame
        ],
        {},
        0.0,
    )
    assert outcome.refused_by == {
        "below_size_gate": 1,
        "not_a_target": 1,
        "not_near_a_person": 1,
    }


def test_a_frame_with_nothing_in_it_is_not_an_error() -> None:
    outcome = _detector().process([], {}, 0.0)
    assert outcome.alerts == [] and outcome.refusals == [] and outcome.tracked == 0


# -- the track store is bounded, and the pipeline expires it every frame ---


def test_a_track_that_goes_away_is_dropped() -> None:
    detector = _detector()
    detector.process([sighting("knife", 0.9)], _people(), 0.0)
    assert detector.process([], _people(), 5.0).tracked == 0


def test_a_returning_object_starts_over_rather_than_inheriting_a_dead_count() -> None:
    """Two observations, a long gap, then two more: that is not four."""
    detector = _detector()
    detector.process([sighting("knife", 0.9)], _people(), 0.0)
    detector.process([sighting("knife", 0.9)], _people(), 0.2)
    detector.process([], _people(), 10.0)  # expires it

    assert detector.process([sighting("knife", 0.9)], _people(), 10.2).alerts == []
    assert detector.process([sighting("knife", 0.9)], _people(), 10.4).alerts == []

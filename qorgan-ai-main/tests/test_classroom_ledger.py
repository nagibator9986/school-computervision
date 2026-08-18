"""One track's running totals: the debounce, the baseline, and zero vs unmeasured.

The counters here are what a headteacher ends up reading, so the tests are chiefly about
the ways a raw per-frame predicate turns into a WRONG count: sixty raised hands from one
child holding their arm up for four seconds, a dozen from a wrist hovering on the
threshold, a stand split in two by one dropped keypoint.
"""

from __future__ import annotations

from qorgan.classroom.ledger import TrackLedger
from qorgan.config.classroom import HandRaiseRules, PlaceRules
from tests.classroom_fakes import seated, with_hand_up

HAND = HandRaiseRules(above_shoulder_ratio=0.35, min_hold_observations=3, min_gap_observations=3)
PLACE = PlaceRules(
    settle_observations=4,
    rise_ratio=0.8,
    min_hold_observations=3,
    away_ratio=2.0,
    min_away_seconds=5.0,
)

WIDTH = 40.0
SEAT_Y = 300.0
SEAT_X = 200.0


def _ledger() -> TrackLedger:
    return TrackLedger(track_id=1, first_seen=0.0, last_seen=0.0)


def _down(**kwargs):
    return seated(centre_x=SEAT_X, shoulder_y=SEAT_Y, shoulder_width=WIDTH, **kwargs)


def _up():
    return with_hand_up(centre_x=SEAT_X, shoulder_y=SEAT_Y, shoulder_width=WIDTH, above=30.0)


def _feed(ledger: TrackLedger, person, count: int, start: float = 0.0, step: float = 0.1) -> float:
    at = start
    for _ in range(count):
        ledger.observe(person, at, HAND, PLACE)
        at += step
    return at


# -- hands -------------------------------------------------------------------


def test_a_held_hand_is_counted_once_not_once_per_frame() -> None:
    """**The whole reason the debounce exists.**

    At 15 fps a hand held up for four seconds is sixty frames. Summing the per-frame
    predicate would report sixty raised hands for one question asked, and the child who
    held their arm up longest would top the report.
    """
    ledger = _ledger()
    _feed(ledger, _up(), 60)

    assert ledger.hand_raises == 1


def test_a_flicker_shorter_than_the_hold_is_not_a_raised_hand() -> None:
    """One or two frames of a wrist above the line is pose noise, not a question."""
    ledger = _ledger()
    _feed(ledger, _up(), HAND.min_hold_observations - 1)

    assert ledger.hand_raises == 0


def test_two_separate_raises_are_two_only_after_the_hand_comes_down() -> None:
    ledger = _ledger()
    at = _feed(ledger, _up(), 5)
    at = _feed(ledger, _down(), 5, start=at)
    _feed(ledger, _up(), 5, start=at)

    assert ledger.hand_raises == 2


def test_a_wrist_wavering_on_the_threshold_is_not_counted_over_and_over() -> None:
    """Without `min_gap_observations` the child with the least steady arm tops the report.

    One frame down is not the hand coming down; it is the pose model losing the wrist for
    a moment. Here the arm alternates up-one-frame, down-one-frame for a long stretch and
    must still read as the single raise it is.
    """
    ledger = _ledger()
    at = _feed(ledger, _up(), 5)
    for _ in range(20):
        at = _feed(ledger, _down(), 1, start=at)
        at = _feed(ledger, _up(), 1, start=at)

    assert ledger.hand_raises == 1


def test_a_hand_needs_no_seated_baseline() -> None:
    """A wrist is measured against that person's own shoulders, so its zero is a REAL
    zero even on a track that never settled -- unlike `stands` and `away_seconds`."""
    # Three frames: enough to hold a raise (min_hold_observations=3), one short of the
    # four a baseline needs (settle_observations=4).
    ledger = _ledger()
    _feed(ledger, _up(), 3)

    assert not ledger.settled
    assert ledger.hand_raises == 1


def test_an_unmeasurable_frame_is_not_counted_as_hand_down() -> None:
    """**A regression test for a bug this suite found in the ledger.**

    The guard was written per TRACK -- "has this track ever shown a usable shoulder
    width" -- which is a different question with a wrong answer. Once a child had been
    measured once, every later frame in which they turned away or were occluded by the
    child in front was scored against the REMEMBERED scale, and with the shoulders
    unreadable `hand_raised` returned False: read as **hand down**. Ten of those satisfy
    `min_gap_observations`, the counter re-armed, and one continuously raised hand was
    counted twice.

    The hand here never comes down. Anything above 1 means an unreadable stretch was
    treated as evidence about an arm nobody could see.
    """
    ledger = _ledger()
    at = _feed(ledger, _up(), 5)
    assert ledger.hand_raises == 1

    turned_away = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y, shoulder_width=4.0)
    at = _feed(ledger, turned_away, 10, start=at)
    _feed(ledger, _up(), 5, start=at)

    assert ledger.hand_raises == 1, "an unmeasurable stretch re-armed the raise counter"


def test_an_unmeasurable_frame_advances_last_seen_but_not_observations() -> None:
    """The tracker saw the person; we could not measure them. Two different facts.

    `observed_seconds` must keep growing (the child is still there) while `observations`
    must not (we learned nothing) -- that gap is the only thing on the row that can tell
    a track readable throughout from one readable a tenth of the time.
    """
    ledger = _ledger()
    at = _feed(ledger, _down(), 3)
    measured = ledger.observations

    turned_away = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y, shoulder_width=4.0)
    end = _feed(ledger, turned_away, 10, start=at)

    assert ledger.observations == measured
    assert ledger.last_seen == end - 0.1
    assert ledger.observed_seconds > 0.0


# -- the seated baseline -----------------------------------------------------


def test_the_baseline_settles_after_the_configured_number_of_frames() -> None:
    ledger = _ledger()
    _feed(ledger, _down(), PLACE.settle_observations - 1)
    assert not ledger.settled

    _feed(ledger, _down(), 1, start=1.0)
    assert ledger.settled
    assert ledger.seat_y == SEAT_Y
    assert ledger.seat_xy == (SEAT_X, SEAT_Y)


def test_place_metrics_are_unmeasured_until_the_baseline_settles() -> None:
    """Not zero. `settled` is what carries the difference to the row and to the page."""
    ledger = _ledger()
    standing = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y - 100.0, shoulder_width=WIDTH)
    _feed(ledger, standing, PLACE.settle_observations - 1)

    assert not ledger.settled
    assert ledger.stands == 0
    assert ledger.away_seconds == 0.0


# -- standing ----------------------------------------------------------------


def test_standing_up_is_counted_once_per_occasion() -> None:
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)
    risen = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y - 40.0, shoulder_width=WIDTH)
    _feed(ledger, risen, 30, start=at)

    assert ledger.stands == 1


def test_a_single_dropped_frame_does_not_split_one_stand_into_two() -> None:
    """The END of a stand is debounced as well as the start, and it has to be: «сколько
    раз встал» is a count of occasions, which is exactly what a flicker inflates."""
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)
    risen = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y - 40.0, shoulder_width=WIDTH)

    at = _feed(ledger, risen, 10, start=at)
    at = _feed(ledger, _down(), 1, start=at)  # one bad frame
    _feed(ledger, risen, 10, start=at)

    assert ledger.stands == 1


def test_sitting_back_down_and_standing_again_is_two() -> None:
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)
    risen = seated(centre_x=SEAT_X, shoulder_y=SEAT_Y - 40.0, shoulder_width=WIDTH)

    at = _feed(ledger, risen, 10, start=at)
    at = _feed(ledger, _down(), 10, start=at)
    _feed(ledger, risen, 10, start=at)

    assert ledger.stands == 2


# -- time away ---------------------------------------------------------------


def test_time_away_is_accrued_in_whole_excursions() -> None:
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)

    away = seated(centre_x=SEAT_X + 200.0, shoulder_y=SEAT_Y, shoulder_width=WIDTH)
    at = _feed(ledger, away, 100, start=at, step=0.1)  # ~10 s away
    _feed(ledger, _down(), 1, start=at)

    assert ledger.away_seconds >= 9.0
    assert ledger.brief_excursions == 0


def test_an_excursion_shorter_than_the_minimum_adds_no_time() -> None:
    """Tracker jitter around the threshold must not add real seconds a few frames at a
    time -- the result would be indistinguishable from a child who genuinely wandered."""
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)

    away = seated(centre_x=SEAT_X + 200.0, shoulder_y=SEAT_Y, shoulder_width=WIDTH)
    at = _feed(ledger, away, 10, start=at, step=0.1)  # ~1 s, under the 5 s minimum
    _feed(ledger, _down(), 1, start=at)

    assert ledger.away_seconds == 0.0
    assert ledger.brief_excursions == 1, "the discarded excursion is counted, not forgotten"


def test_an_absence_that_never_ends_is_still_counted() -> None:
    """**The one excursion most worth reporting is the one that does not close.**

    A child who walks out and never comes back before the bell has no "returned" frame,
    so without `finalise` their entire absence -- the longest in the lesson -- is the only
    kind that never counts.
    """
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)

    away = seated(centre_x=SEAT_X + 200.0, shoulder_y=SEAT_Y, shoulder_width=WIDTH)
    at = _feed(ledger, away, 100, start=at, step=0.1)

    assert ledger.away_seconds == 0.0, "nothing has closed the excursion yet"
    ledger.finalise(at)
    assert ledger.away_seconds >= 9.0


def test_finalising_twice_does_not_count_the_absence_twice() -> None:
    """It is called when a track retires AND again when the lesson closes, with a flush
    possibly in between, so it has to be idempotent."""
    ledger = _ledger()
    at = _feed(ledger, _down(), PLACE.settle_observations)
    away = seated(centre_x=SEAT_X + 200.0, shoulder_y=SEAT_Y, shoulder_width=WIDTH)
    at = _feed(ledger, away, 100, start=at, step=0.1)

    ledger.finalise(at)
    once = ledger.away_seconds
    ledger.finalise(at)

    assert ledger.away_seconds == once


def test_the_scale_is_the_largest_shoulder_width_the_track_has_shown() -> None:
    """A child turning sideways foreshortens their shoulders, and a shrinking scale
    shrinks every threshold with it -- which would make a turned child EASIER to credit
    with a raised hand than a facing one."""
    ledger = _ledger()
    ledger.observe(_down(), 0.0, HAND, PLACE)
    ledger.observe(seated(shoulder_width=20.0), 0.1, HAND, PLACE)

    assert ledger.scale == WIDTH

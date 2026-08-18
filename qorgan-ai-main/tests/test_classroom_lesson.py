"""The live ledgers: bounded, retired on time, and honest about what was refused.

A lesson is 45 minutes of frames and ByteTrack's ids only ever go up, so the interesting
questions here are all about memory and about what happens at the edges -- rule R8, and
the legacy habit of leaking a dict keyed on track id for as long as the process lived.
"""

from __future__ import annotations

from qorgan.classroom.association import Assignment
from qorgan.classroom.lesson import LessonAccumulator
from qorgan.config.classroom import HandRaiseRules, LessonRules, PlaceRules
from tests.classroom_fakes import seated, with_hand_up

HAND = HandRaiseRules(min_hold_observations=2, min_gap_observations=2)
PLACE = PlaceRules(settle_observations=2, min_away_seconds=1.0)
RULES = LessonRules(track_idle_seconds=5.0, max_tracks=3)


def _accumulator(rules: LessonRules = RULES) -> LessonAccumulator:
    return LessonAccumulator(rules=rules)


def _frame(*track_ids: int, raised: bool = False) -> Assignment:
    person = with_hand_up() if raised else seated()
    return Assignment(people={t: person for t in track_ids}, ambiguous=0, unclaimed=0)


def test_a_ledger_is_opened_for_each_new_track() -> None:
    tally = _accumulator()
    tally.observe(_frame(1, 2), 0.0, HAND, PLACE)

    assert {ledger.track_id for ledger in tally.live()} == {1, 2}


def test_doubt_accumulates_across_frames() -> None:
    """The counters are running totals for the lesson, not per-frame values -- `store.flush`
    ASSIGNS them to the row, so anything else would either double or reset them."""
    tally = _accumulator()
    tally.observe(Assignment(people={}, ambiguous=2, unclaimed=1), 0.0, HAND, PLACE)
    tally.observe(Assignment(people={}, ambiguous=3, unclaimed=4), 0.1, HAND, PLACE)

    assert tally.doubt.ambiguous == 5
    assert tally.doubt.unclaimed == 5


def test_a_track_nobody_has_seen_is_retired_and_handed_back() -> None:
    """Retired ledgers are RETURNED, not dropped: the caller must persist them, and this
    object will not mention them again."""
    tally = _accumulator()
    tally.observe(_frame(1, 2), 0.0, HAND, PLACE)

    retired = tally.observe(_frame(2), RULES.track_idle_seconds + 1.0, HAND, PLACE)

    assert [ledger.track_id for ledger in retired] == [1]
    assert {ledger.track_id for ledger in tally.live()} == {2}


def test_retirement_finalises_before_removing() -> None:
    """A child who walked out and never came back has an OPEN excursion. Removing the
    ledger without closing it discards the longest absence of the lesson -- the one most
    worth reporting is the only kind that would never count."""
    tally = _accumulator()
    at = 0.0
    for _ in range(4):  # settle the baseline where the child is sitting
        tally.observe(_frame(1), at, HAND, PLACE)
        at += 0.1

    away = Assignment(people={1: seated(centre_x=800.0)}, ambiguous=0, unclaimed=0)
    for _ in range(30):  # ~3 s away, and the track then vanishes
        tally.observe(away, at, HAND, PLACE)
        at += 0.1

    retired = tally.observe(_frame(2), at + RULES.track_idle_seconds + 1.0, HAND, PLACE)

    assert len(retired) == 1
    assert retired[0].away_seconds > 0.0, "the unfinished absence was thrown away"


def test_a_track_seen_recently_is_not_retired() -> None:
    tally = _accumulator()
    tally.observe(_frame(1), 0.0, HAND, PLACE)
    retired = tally.observe(_frame(1), RULES.track_idle_seconds - 0.5, HAND, PLACE)

    assert retired == []
    assert len(tally.live()) == 1


def test_the_map_is_capped_and_the_refusals_are_counted() -> None:
    """Rule R8. Silently admitting the fourth track grows the dict for as long as the
    process lives; silently dropping it lets the report describe a room it stopped
    watching. Refused AND counted is the only combination that is neither."""
    tally = _accumulator()
    tally.observe(_frame(1, 2, 3, 4, 5), 0.0, HAND, PLACE)

    assert len(tally.live()) == RULES.max_tracks
    assert tally.doubt.dropped_tracks == 2


def test_the_cap_counts_every_refusal_not_every_track() -> None:
    """A refused track that keeps appearing keeps being refused, and each refusal counts:
    the number says how much observation was lost, not how many ids were unlucky."""
    tally = _accumulator()
    tally.observe(_frame(1, 2, 3), 0.0, HAND, PLACE)
    tally.observe(_frame(1, 2, 3, 9), 0.1, HAND, PLACE)
    tally.observe(_frame(1, 2, 3, 9), 0.2, HAND, PLACE)

    assert tally.doubt.dropped_tracks == 2


def test_a_retired_slot_lets_a_new_track_in() -> None:
    """The cap is on LIVE ledgers. A room that turns over must not be locked out for the
    rest of the lesson by tracks that ended twenty minutes ago."""
    tally = _accumulator(LessonRules(track_idle_seconds=1.0, max_tracks=2))
    tally.observe(_frame(1, 2), 0.0, HAND, PLACE)

    tally.observe(_frame(3), 5.0, HAND, PLACE)

    assert {ledger.track_id for ledger in tally.live()} == {3}
    assert tally.doubt.dropped_tracks == 0


def test_a_track_visible_this_frame_is_never_retired_however_long_the_gap() -> None:
    """A long occlusion that ByteTrack nevertheless re-associates returns under the SAME
    id. Retiring it would open a second ledger with zeroed counters, which then upserts
    over the first row -- erasing everything that track did before the gap."""
    tally = _accumulator()
    at = 0.0
    for _ in range(6):
        tally.observe(_frame(1, raised=True), at, HAND, PLACE)
        at += 0.1
    assert [ledger.hand_raises for ledger in tally.live()] == [1]

    retired = tally.observe(_frame(1), at + RULES.track_idle_seconds * 4, HAND, PLACE)

    assert retired == []
    assert [ledger.hand_raises for ledger in tally.live()] == [1], "the count was reset"


def test_finishing_empties_the_map_and_finalises_everything() -> None:
    tally = _accumulator()
    tally.observe(_frame(1, 2), 0.0, HAND, PLACE)

    remaining = tally.finish(10.0)

    assert {ledger.track_id for ledger in remaining} == {1, 2}
    assert tally.live() == []


def test_counts_survive_across_frames_on_the_same_track() -> None:
    """The ledger is the same object frame to frame -- a new one each time would reset
    every debounce and report a raised hand per frame."""
    tally = _accumulator()
    at = 0.0
    for _ in range(6):
        tally.observe(_frame(1, raised=True), at, HAND, PLACE)
        at += 0.1

    assert [ledger.hand_raises for ledger in tally.live()] == [1]

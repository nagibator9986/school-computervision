"""Rule R8 on the weapon tracks: bounded twice, and the second bound is the one that holds.

`events/clip_buffer.py` states the rule this module obeys -- *"a limit derived from
configuration is not a limit, it is a hope about configuration"* -- and the weapon track
store is the place in this module where an unbounded dict would be easiest to write. Track
ids only ever go up, and the legacy leaked several dicts keyed on them.

  * the TTL is the ordinary bound, and it comes from `track_idle_seconds`;
  * `max_tracks` is the HARD ceiling and holds whatever the TTL says -- a camera pointed
    at a display of cutlery, a model that fires on texture, a `track_idle_seconds` raised
    in YAML by somebody chasing a missed detection.

Evictions are COUNTED, because a camera that is evicting produces output that means less
than it appears to, and "the camera saw nothing" and "the camera saw more than it can
hold" look identical from the outside.
"""

from __future__ import annotations

from qorgan.weapons.tracking import WeaponTrackStore
from tests.weapons_fixtures import sighting


def _store(*, idle_seconds: float = 1.5, max_tracks: int = 64) -> WeaponTrackStore:
    return WeaponTrackStore(idle_seconds=idle_seconds, max_tracks=max_tracks)


def _observe(store: WeaponTrackStore, s, at: float, *, strong: bool = True, person: int | None = 1):
    return store.observe(s, at, strong=strong, person_track_id=person)


def _scattered(index: int):
    """A 5 px object on a 90 px grid: far enough apart to be a DIFFERENT object.

    The association radius is 3 of the object's own diagonals, so a 5 px box reaches
    ~21 px. Spacing these 15 px apart -- as an earlier draft of this file did -- made
    every one of them fold into a single track, and the ceiling tests passed while
    exercising nothing. The spacing is checked by
    `test_the_scattered_objects_really_are_separate_tracks`.
    """
    return sighting(x1=30.0 + (index % 10) * 90.0, y1=30.0 + (index // 10) * 90.0, size=5.0)


def test_the_scattered_objects_really_are_separate_tracks() -> None:
    """The premise every ceiling test below rests on, asserted rather than assumed."""
    store = _store(idle_seconds=10_000.0, max_tracks=1000)
    for index in range(20):
        _observe(store, _scattered(index), 0.0)
    assert len(store) == 20


# -- association -----------------------------------------------------------


def test_the_same_object_in_the_same_place_is_one_track() -> None:
    store = _store()
    first = _observe(store, sighting(), 0.0)
    second = _observe(store, sighting(), 0.2)
    assert first.track_id == second.track_id
    assert second.observations == 2
    assert len(store) == 1


def test_an_object_that_jumps_across_the_frame_is_a_different_track() -> None:
    store = _store()
    _observe(store, sighting(x1=50, y1=50, size=20), 0.0)
    _observe(store, sighting(x1=800, y1=450, size=20), 0.2)
    assert len(store) == 2


def test_a_knife_that_becomes_an_axe_does_not_inherit_the_count() -> None:
    """The model changing its mind. Merging them would let a track accumulate the
    observations of one object under the name of another -- which is how a bat and a
    ruler add up to an alert neither of them earned."""
    store = _store()
    _observe(store, sighting("knife"), 0.0)
    _observe(store, sighting("knife"), 0.2)
    axe = _observe(store, sighting("axe"), 0.4)
    assert axe.observations == 1
    assert len(store) == 2


def test_the_radius_scales_with_the_object_not_with_pixels() -> None:
    """A knife in the foreground is a bigger box AND moves through more pixels for the
    same real movement."""
    small = _store()
    _observe(small, sighting(x1=100, y1=100, size=10), 0.0)
    _observe(small, sighting(x1=180, y1=100, size=10), 0.2)
    assert len(small) == 2, "80 px is far outside 3 diagonals of a 10 px box"

    big = _store()
    _observe(big, sighting(x1=100, y1=100, size=100), 0.0)
    _observe(big, sighting(x1=180, y1=100, size=100), 0.2)
    assert len(big) == 1, "the same 80 px is well inside 3 diagonals of a 100 px box"


# -- bound one: the TTL ----------------------------------------------------


def test_a_track_nobody_has_seen_lately_is_dropped() -> None:
    store = _store(idle_seconds=1.5)
    _observe(store, sighting(), 0.0)
    assert store.expire(1.4) == []
    assert len(store) == 1

    dropped = store.expire(2.0)
    assert len(dropped) == 1 and len(store) == 0


def test_a_hand_passing_behind_a_body_does_not_start_a_new_track() -> None:
    """1.5 s of tolerance is two missed analyses at `analyse_every: 3` plus a lot of
    slack, which is what it is for."""
    store = _store(idle_seconds=1.5)
    first = _observe(store, sighting(), 0.0)
    store.expire(1.0)
    assert _observe(store, sighting(), 1.0).track_id == first.track_id


# -- bound two: the hard ceiling, which holds when bound one is wrong ------


def test_the_ceiling_holds_even_when_nothing_has_expired() -> None:
    """Every track is fresh, the TTL would keep all of them, and the ceiling still bites.

    This is the bound that survives a `track_idle_seconds` raised in YAML, which is the
    edit somebody makes when a detection was missed.
    """
    store = _store(idle_seconds=10_000.0, max_tracks=4)
    for index in range(50):
        _observe(store, _scattered(index), 0.0)
    assert len(store) <= 4


def test_a_ceiling_of_one_still_works() -> None:
    store = _store(max_tracks=1)
    for index in range(10):
        _observe(store, sighting(x1=index * 100.0, y1=10.0, size=5.0), 0.0)
    assert len(store) <= 1


def test_evictions_are_counted_not_silently_absorbed() -> None:
    """"The camera saw nothing" and "the camera saw more than it can hold" look identical
    from the outside unless this number exists."""
    store = _store(idle_seconds=10_000.0, max_tracks=3)
    assert store.evicted_over_ceiling == 0
    for index in range(20):
        _observe(store, _scattered(index), 0.0)
    # 20 objects went in, 3 are held, and the other 17 are ACCOUNTED FOR rather than
    # having quietly disappeared. That the two numbers add up is the assertion.
    assert len(store) == 3
    assert store.evicted_over_ceiling == 17
    assert len(store) + store.evicted_over_ceiling == 20


def test_the_least_recently_seen_track_is_the_one_evicted() -> None:
    """Least recently SEEN, not first created: the oldest track may be the one still
    being watched, and evicting it would restart the count on the only object that
    matters while keeping a stack of dead ones."""
    store = _store(idle_seconds=10_000.0, max_tracks=2)
    watched = _observe(store, sighting(x1=10, y1=10, size=5), 0.0)
    _observe(store, sighting(x1=500, y1=10, size=5), 0.1)
    _observe(store, sighting(x1=10, y1=10, size=5), 5.0)  # the watched one, seen again

    _observe(store, sighting(x1=900, y1=400, size=5), 5.1)  # forces an eviction
    assert store.evicted_over_ceiling == 1
    still_here = _observe(store, sighting(x1=10, y1=10, size=5), 5.2)
    assert still_here.track_id == watched.track_id


def test_ids_are_never_reused_after_an_eviction() -> None:
    """An id that comes round again is an id that means two objects in one log."""
    store = _store(idle_seconds=10_000.0, max_tracks=2)
    seen = set()
    for index in range(20):
        seen.add(_observe(store, _scattered(index), 0.0).track_id)
    assert len(seen) == 20


# -- what a track counts ---------------------------------------------------


def test_strong_observations_are_counted_separately() -> None:
    store = _store()
    _observe(store, sighting(confidence=0.4), 0.0, strong=False)
    _observe(store, sighting(confidence=0.9), 0.2, strong=True)
    track = _observe(store, sighting(confidence=0.4), 0.4, strong=False)
    assert track.observations == 3
    assert track.strong_observations == 1
    assert track.best_confidence == 0.9


def test_the_person_a_track_was_last_beside_is_remembered() -> None:
    store = _store()
    _observe(store, sighting(), 0.0, person=3)
    track = _observe(store, sighting(), 0.2, person=9)
    assert track.person_track_id == 9

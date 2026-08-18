"""The adult's half: the follower's refusals, and the denominators of its numbers.

**Why these tests are shaped the way they are.** Two earlier versions of `follow_adult`
shipped confident, plausible, wrong trajectories — a greedy nearest-neighbour tracker that
drifted onto seated children, and a Viterbi over a size cue that cannot separate a slight
young man from an eleven-year-old. Neither had any internal signal that it was wrong;
both were caught only by drawing the answer onto real frames. So there are two kinds of
test here and they do different jobs:

  * **Synthetic tests of the RULES**, below — that a tracklet the operator's zone never
    touched cannot be claimed, that a hand-off further than the bound cannot chain, that
    `out_of_frame` never enters a denominator it does not belong in. These are fast, they
    run without footage, and they pin the properties that make the output honest.
  * **A rendered check against the recording**, which lives outside pytest
    (`scratchpad/verify.py`, `scratchpad/eval.py`) because it needs a 1 GB file and a
    model cache. Its result is quoted in `metrics/teacher.follow_adult`: 45.0 % of the
    lesson attributed, 7 of 7 hand-labelled attributions correct, 1 of 1 refusal correct.

A test suite alone would have passed all three versions. That is worth saying out loud in
the file whose job is to give confidence.
"""

from __future__ import annotations

import pytest

from classvision.metrics import teacher as T
from classvision.room.zones import RoomLayout

DESK = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
BOARD = ((0.0, 200.0), (400.0, 200.0), (400.0, 300.0), (0.0, 300.0))
LAYOUT = RoomLayout(camera="test", frame_width=800, frame_height=600,
                    board_zone=BOARD, teacher_zone=DESK)


def stream(*runs):
    """Build observations from (track_id, t0, count, (x, y), scale) tuples at 2 fps."""
    out = []
    for index, (track_id, t0, count, position, scale) in enumerate(runs):
        for step in range(count):
            out.append(T.AdultObservation(t0 + 0.5 * step, position, scale,
                                          index * 10_000 + step, track_id))
    out.sort(key=lambda o: o.video_seconds)
    return out


def frames(seconds: float):
    return [0.5 * i for i in range(int(seconds * 2))]


# -- the refusals -----------------------------------------------------------------------

def test_no_zone_and_no_fallback_is_a_refusal_not_a_guess():
    """With nothing to anchor on, the follower must produce an empty track and say why.

    The alternative — following whoever is nearest the middle of the frame — is what a
    tracker does by default, and it produces a full set of shares about an arbitrary child.
    """
    track = T.follow_adult(stream((1, 0.0, 40, (500.0, 500.0), 60.0)), frames(20),
                           layout=None, sample_interval=0.5)
    assert not track.found
    assert track.source == "none"
    assert track.needs_confirmation
    assert track.diagnostics["why"]
    assert all(s is T.TeacherState.OUT_OF_FRAME for s in track.state)


def test_a_body_that_never_enters_the_zone_is_never_claimed():
    """The zone is the ONLY place a track may start. A large, long-lived, perfectly
    trackable person elsewhere in the room is not the adult and must not become him."""
    track = T.follow_adult(stream((1, 0.0, 100, (600.0, 500.0), 90.0)), frames(50),
                           layout=LAYOUT, sample_interval=0.5)
    assert not track.found
    assert track.diagnostics["tracklets_seeded_by_zone"] == 0


def test_a_brief_visit_to_the_zone_does_not_seed_the_whole_track():
    """A pupil walking past the teacher's desk touches the polygon for a second or two.
    `MIN_ZONE_OBSERVATIONS` is what stops that from capturing the entire lesson."""
    short = T.MIN_ZONE_OBSERVATIONS - 1
    track = T.follow_adult(stream((1, 0.0, short, (50.0, 50.0), 60.0)),
                           frames(30), layout=LAYOUT, sample_interval=0.5)
    assert not track.found


# -- the chaining -----------------------------------------------------------------------

def test_a_tight_handoff_chains_and_a_loose_one_does_not():
    """The distance bound is the constant that was measured by being wrong first.

    At four shoulder widths the chain jumped from the teacher to a pupil standing beside
    him at the board, twice in ten hand-checked frames. Two people at a chalkboard stand
    about two shoulder widths apart, which is exactly why that value failed there.
    """
    scale = 50.0
    near = T.HANDOFF_SCALES * scale * 0.5
    far = T.HANDOFF_SCALES * scale * 2.0

    seeded = (1, 0.0, T.MIN_ZONE_OBSERVATIONS + 10, (50.0, 50.0), scale)
    start = 0.5 * (T.MIN_ZONE_OBSERVATIONS + 10)

    close = T.follow_adult(stream(seeded, (2, start + 0.5, 40, (50.0 + near, 50.0), scale)),
                           frames(60), layout=LAYOUT, sample_interval=0.5)
    distant = T.follow_adult(stream(seeded, (3, start + 0.5, 40, (50.0 + far, 50.0), scale)),
                             frames(60), layout=LAYOUT, sample_interval=0.5)

    assert close.diagnostics["tracklets_kept_after_conflict_resolution"] == 2
    assert distant.diagnostics["tracklets_kept_after_conflict_resolution"] == 1
    assert close.attributed > distant.attributed


def test_the_adult_is_in_one_place_at_a_time():
    """Two claimed tracklets overlapping in time is a contradiction, and the resolution
    step must remove one rather than reporting the adult in two places."""
    scale = 50.0
    long_seed = (1, 0.0, 120, (50.0, 50.0), scale)
    overlapping = (2, 10.0, 60, (50.0 + T.HANDOFF_SCALES * scale * 0.4, 50.0), scale)
    track = T.follow_adult(stream(long_seed, overlapping), frames(70),
                           layout=LAYOUT, sample_interval=0.5)
    # Exactly one position per analysed frame, never two.
    assert len(track.position) == len(track.frames)
    assert track.attributed <= len(track.frames)


# -- the states -------------------------------------------------------------------------

def test_the_desk_is_tested_before_the_board():
    """Deliberate precedence: the desk polygon is small and specific, the board polygon is
    large and loose, and when a specific human claim disagrees with a loose one the
    specific one wins."""
    overlapping = RoomLayout(camera="t", frame_width=800, frame_height=600,
                             board_zone=((0.0, 0.0), (400.0, 0.0),
                                         (400.0, 300.0), (0.0, 300.0)),
                             teacher_zone=DESK)
    track = T.AdultTrack(frames=[0.0], position=[(50.0, 50.0)], scale=[60.0],
                         index=[0], speed=[0.0])
    T.classify_track(track, overlapping)
    assert track.state[0] is T.TeacherState.AT_DESK


def test_a_frame_with_no_attribution_is_out_of_frame_and_nothing_else():
    track = T.AdultTrack(frames=[0.0], position=[None], scale=[None], index=[None],
                         speed=[None])
    T.classify_track(track, LAYOUT)
    assert track.state[0] is T.TeacherState.OUT_OF_FRAME


def test_pacing_at_the_board_is_at_the_board_not_moving():
    track = T.AdultTrack(frames=[0.0], position=[(200.0, 250.0)], scale=[60.0],
                         index=[0], speed=[T.MOVING_SPEED * 5])
    T.classify_track(track, LAYOUT)
    assert track.state[0] is T.TeacherState.AT_BOARD


# -- the denominators -------------------------------------------------------------------

@pytest.fixture
def block() -> dict:
    scale = 50.0
    observations = stream(
        (1, 0.0, 60, (50.0, 50.0), scale),                        # at his desk
        (2, 30.5, 40, (50.0 + T.HANDOFF_SCALES * scale * 0.4, 250.0), scale),
    )
    track = T.follow_adult(observations, frames(120), layout=LAYOUT, sample_interval=0.5)
    T.classify_track(track, LAYOUT)
    return T.presence_block(track, sample_interval=0.5, layout=LAYOUT)


def test_shares_of_the_lesson_include_out_of_frame_and_sum_to_a_hundred(block: dict):
    """The family that answers «какую часть урока». `out_of_frame` is one of its members,
    which is the only way a reader can see how much of the lesson is missing."""
    shares = block["state_share_of_lesson_percent"]
    assert T.TeacherState.OUT_OF_FRAME.value in shares
    assert sum(v for v in shares.values() if v) == pytest.approx(100.0, abs=0.5)


def test_shares_of_the_attributed_frames_exclude_out_of_frame(block: dict):
    """The family that answers «из того, что мы видели». Putting `out_of_frame` in here
    would divide "we could not see him" by "frames where we could", which is not a
    quantity."""
    shares = block["state_share_of_attributed_percent"]
    assert T.TeacherState.OUT_OF_FRAME.value not in shares
    assert sum(v for v in shares.values() if v) == pytest.approx(100.0, abs=0.5)


def test_every_share_field_names_its_denominator(block: dict):
    """The rename that this project already paid for once: `seated_share` meant a share of
    observations while `seconds` meant time inside qualifying episodes, and an LLM joined
    them into a false sentence about a real person."""
    for key in block:
        if "share" in key and key != "state_share_of_attributed_percent":
            assert ("of_lesson" in key or "of_attributed" in key or "of_board" in key
                    or "of_room" in key or "of_minute" in key), key


def test_episode_time_is_never_larger_than_observation_time(block: dict):
    """Two different quantities that a careless reader joins. Episode time only counts runs
    that survived their minimum hold, so it is ALWAYS the smaller of the two — and the
    arithmetic must keep saying so, whatever the footage."""
    observed = block["state_minutes_of_lesson"]
    episodes = block["episode_minutes_by_state"]
    for state, minutes in episodes.items():
        assert minutes <= observed[state] + 0.05, state


def test_the_attributed_share_is_reported_beside_every_total(block: dict):
    assert block["attributed_share_of_lesson_percent"] is not None
    assert block["analysed_frames"] > 0
    assert "attributed_frames" in block


# -- the honesty layer ------------------------------------------------------------------

FORBIDDEN = ("оценка активности", "рейтинг", "балл", "требует внимания", "score",
             "rating", "needs_attention", "эффективност")


def test_nothing_in_the_teacher_block_is_a_judgement(block: dict):
    """No quality score, no activity rating, no flag, and no place to add one. The block is
    searched as text so that a future field cannot smuggle one in under a new name."""
    import json
    text = json.dumps(block, ensure_ascii=False).lower()
    for word in FORBIDDEN:
        assert word not in text, f"«{word}» appeared in the teacher block"


def test_every_state_carries_what_it_cannot_tell_apart(block: dict):
    definitions = block["definitions_ru"]
    for state in T.TeacherState:
        entry = definitions[state.value]
        assert entry["confusion"], state
        assert entry["basis"], state
        assert entry["label"], state


def test_the_refusal_sentence_travels_with_the_numbers():
    metrics = T.teacher_metrics(None, track=T.AdultTrack(), sample_interval=0.5,
                                layout=LAYOUT)
    text = metrics.not_an_assessment_ru
    assert "не оценка работы учителя" in text
    assert "«Сидит» не означает «не ведёт урок»" in text


def test_out_of_frame_is_declared_as_unmeasured_when_it_is_large(block: dict):
    """A reader must meet "we could not determine his position for N % of the lesson" as an
    explicit item, not have to derive it by subtracting shares."""
    seen = block["attributed_share_of_lesson_percent"]
    if seen is not None and seen < 80.0:
        assert any("не смогли определить" in item["why"]
                   for item in block["unmeasured_ru"])


def test_floor_coverage_refuses_without_a_perspective_model():
    """A grid in pixels would measure the camera, not the room: a step at the front of the
    frame covers twelve times the pixels of the same step at the back."""
    result = T.floor_coverage(T.AdultTrack(), perspective=None, everyone=())
    assert result["available"] is False
    assert result["why"]

"""The adult who never sits anywhere — the case `metrics/teacher.py` was written for, and
the one nothing covered.

`pipeline.assemble` has two ways of building a `TeacherRecord`. One is taken when
`room/zones.identify_adult` found a SEAT for the adult: it carries a pose ledger. The other
is taken when it did not, and it carries `ledger={}` and nothing but the follower's answer.
Camera D14 is the reason the second branch exists — a man who works at three places in one
lesson has no seat — and on the run reviewed here it was reached whenever his desk cluster's
median lands a few pixels outside the drawn polygon, which is a matter of where the operator
put a corner.

Every test in this file is a regression from a defect found by rendering that branch:

  * `report/summary.py` raised `TypeError: float() argument must be a string or a real
    number, not 'NoneType'` and took the whole text summary and the HTML report with it,
    because it asked whether `coverage_percent` was present (it is — the follower's) and
    then printed `observations` from the ledger (it is not).
  * `TeacherMetrics.transitions` was filled, on this branch only, from the count of POSITION
    episode boundaries, `out_of_frame` boundaries included. The summary prints that field
    inside a sentence that begins «что делало его тело за его собственным столом … где поза
    прочиталась», so the report stated «смен положения — 25» about a lesson holding no pose
    observation at all, and `cabinet/report.py` puts the same field in one column across
    lessons from different cameras.
  * `out_of_frame_share_of_lesson` was the SEAT's emptiness on the ledger branch and the
    FOLLOWER's silence on this one. On D14 those are 51,5 % and 55,0 % of the same lesson.
"""

from __future__ import annotations

import pytest

from classvision.metrics.teacher import (
    AdultTrack,
    TeacherState,
    classify_track,
    teacher_metrics,
)
from classvision.report import summary as summarymod
from classvision.room.zones import RoomLayout

BOARD = ((1024.0, 140.0), (1645.0, 140.0), (1645.0, 340.0), (1024.0, 340.0))
DESK = ((960.0, 300.0), (1085.0, 300.0), (1085.0, 395.0), (960.0, 395.0))


def _layout(*, board=BOARD, desk=DESK) -> RoomLayout:
    return RoomLayout(camera="T", frame_width=2560, frame_height=1440,
                      board_zone=board, teacher_zone=desk)


def _track(n: int = 360) -> AdultTrack:
    """A lesson in which the adult is at his desk, then at the board, then lost, repeatedly."""
    track = AdultTrack()
    track.frames = [i * 0.5 for i in range(n)]
    positions: list[tuple[float, float] | None] = []
    for i in range(n):
        block = (i // 40) % 3
        positions.append((1020.0, 350.0) if block == 0
                         else ((1300.0, 200.0) if block == 1 else None))
    track.position = positions
    track.scale = [100.0 if p else None for p in positions]
    track.index = [0 if p else None for p in positions]
    track.speed = [None] * n
    track.source = "designated_zone"
    track.diagnostics = {"route": "designated_zone",
                         "attributed_share_of_lesson_percent": 66.7,
                         "zones_confirmed_by": ""}
    return track


def _record(metrics: dict) -> dict:
    """What `pipeline.assemble` emits when the adult has no seat: an EMPTY ledger."""
    return {"seat_id": None, "ledger": {}, "metrics": metrics, "timeline": [],
            "identification": {"source": "none", "needs_confirmation": True,
                               "evidence": {"why": "тест"}}}


def test_the_text_summary_survives_an_adult_with_no_seat():
    """The whole report used to die here, and `classvision report` exited non-zero."""
    track = _track()
    layout = _layout()
    classify_track(track, layout)
    metrics = teacher_metrics(None, track=track, sample_interval=0.5, layout=layout).to_dict()

    bundle = summarymod._teacher_bundle(_record(metrics))
    assert bundle["coverage_percent"] is not None, (
        "the follower's coverage is present on this branch -- it is what made the crash "
        "reachable, and a test that did not assert it would pass for the wrong reason")
    assert bundle["observations"] is None, "no ledger means no seat observations"

    text = summarymod._teacher_section({"teacher": bundle})
    assert text, "the teacher section must still be produced"
    # The seat paragraph is ABSENT rather than zeroed. There is no seat; «место было занято
    # в 0 % кадров» would be a measurement of a place that was never identified.
    assert "Само место у учительского стола" not in text


def test_position_episode_boundaries_are_not_reported_as_pose_transitions():
    """`transitions` names ONE quantity: seated <-> upright, from the pose ledger."""
    track = _track()
    layout = _layout()
    classify_track(track, layout)
    metrics = teacher_metrics(None, track=track, sample_interval=0.5, layout=layout).to_dict()

    assert metrics["transitions"] is None, (
        "with no pose ledger there is no pose transition count; filling it from "
        "`presence.transitions_between_episodes` put a number that counts losing sight of "
        "the adult into a field the report renders as «смен положения»")
    # ...and the position counts are still there, under names that state what they count.
    presence = metrics["presence"]
    assert presence["transitions_between_episodes"] > 0
    assert presence["transitions_between_episodes_excluding_out_of_frame"] >= 0

    text = summarymod._teacher_section({"teacher": summarymod._teacher_bundle(_record(metrics))})
    assert "смен положения" not in text, (
        "the pose paragraph must not appear when no pose observation exists")
    assert "Смен места" in text, "the POSITION count keeps its own, differently worded line"


def test_out_of_frame_means_the_adult_was_not_located_on_both_branches(monkeypatch):
    """One name, one meaning. The seat's emptiness is a different fact and stays named so."""

    class FakeEpisode:
        state = None
        duration = 0.0

    class FakeLedger:
        """A seat that was occupied nearly all lesson while the adult was elsewhere."""

        observations = 900
        absent_observations = 100      # seat empty in 10 % of frames
        coverage = 0.9
        observed_seconds = 450.0
        motion_per_observation = 0.1
        episodes: list = []
        state_observations = {"seated": 900}

    track = _track()
    layout = _layout()
    classify_track(track, layout)
    metrics = teacher_metrics(FakeLedger(), track=track, sample_interval=0.5,
                              layout=layout).to_dict()

    presence_out = metrics["presence"]["state_share_of_lesson_percent"][
        TeacherState.OUT_OF_FRAME.value]
    assert metrics["out_of_frame_share_of_lesson"] == pytest.approx(presence_out), (
        "the ledger says 10 % (the DESK was empty) and the follower says a third of the "
        "lesson (the ADULT was not located). Both were shipped, under names a reader "
        "cannot tell apart; only the second answers the question the name asks")


def test_no_board_zone_refuses_instead_of_reporting_zero_minutes_at_the_board():
    """Zero and «не измеряли» are different values -- rule 3, in the block that answers
    the client's actual question. `configs/camera_01.yaml` has `board_zone: null`."""
    track = _track()
    layout = _layout(board=None)
    classify_track(track, layout)
    metrics = teacher_metrics(None, track=track, sample_interval=0.5, layout=layout).to_dict()

    board = metrics["presence"]["board"]
    assert board["zone_configured"] is False
    for field in ("minutes_of_lesson", "share_of_lesson_percent",
                  "share_of_attributed_percent", "episodes", "longest_episode_minutes"):
        assert board[field] is None, (
            f"`board.{field}` was 0 -- structurally, because `classify_track` cannot emit "
            "AT_BOARD without a polygon -- and was printed under «то, о чём спрашивал "
            "заказчик» with «Скорее занижено» beside it")
    assert "Не измерялось" in board["direction_of_error_ru"]
    assert "не размечена зона доски" in board["direction_of_error_ru"]

    # ...and the same run must not claim, from the mere absence of a polygon, that the
    # board is physically behind the lens. On D14 it is not; it is in the middle of frame.
    note = metrics["not_an_assessment_ru"]
    assert "не размечена зона доски" in note
    assert note.count("находится позади объектива") == 0, (
        "the sentence stated the camera's mounting as a fact whenever no zone was "
        "configured; the recording does not carry that fact")


def test_the_facing_coverage_is_not_pinned_to_one_hundred_percent():
    """`geometry.head_direction` cannot return UNKNOWN for an observation the follower
    placed, because `geometry.anchor` already required the same two shoulders at the same
    threshold. So the old «доля кадров, где направление прочиталось» was always 100."""
    from collections import Counter

    track = _track()
    layout = _layout()
    classify_track(track, layout)
    board_frames = sum(1 for s in track.state if s is TeacherState.AT_BOARD)
    assert board_frames > 0

    # Every board observation classified, but a head keypoint found in only a third of them.
    directions = Counter({"toward_camera": board_frames // 3,
                          "away": board_frames - board_frames // 3})
    metrics = teacher_metrics(None, track=track, sample_interval=0.5, layout=layout,
                              head_directions=directions).to_dict()
    facing = metrics["presence"]["board"]["facing"]
    assert facing["observations"] == board_frames
    assert facing["head_keypoint_visible_share_of_board_percent"] < 100.0, (
        "`away` is the residual bucket for «no ear, no nose, no eye, shoulders present», "
        "which at a 50 px shoulder width is also what an unresolvable head looks like; "
        "counting it as a successful reading made the coverage a tautology")

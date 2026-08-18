"""The camera profiles, and the loader that refuses a bad one.

**These tests carry MEASURED coordinates, not invented ones.** Every point below was read
off `D14_20260815101759.mp4` / `D14_20260815103136.mp4` by running the same pose model the
analysis runs over 128 sampled frames — 911 observations of people — and recording, for
each, the shoulder anchor, the box bottom and the ankle confidences. Ankle confidence is
the label: a person the model can see the ankles of is standing on visible floor; a person
seated behind a desk has ankle confidences of ~0.01 and a box bottom that is the DESK EDGE.
That gives a ground truth which does not come from the zone being tested, so these are not
a restatement of the polygons.

The two frames that matter are named in the file and re-checkable by eye with
`classvision zones --room configs/camera_d14.yaml --at 1764 --out …`:

  * **t = 1764 s (11:01:00)** — one pupil standing at the board, five people seated, one
    of whom is the ADULT who has moved to sit among the pupils. This is the frame that
    separates «стоит у доски» from «сидит не на своём месте», and the adult sitting there
    is the false positive a careless polygon produces.
  * **t = 1038 s (10:48:54)** — TWO people at the board at once, a pupil at the left panel
    and the adult at the right. The adult's legs are cropped by the front-right desk, so
    his box bottom is y = 449 — four pixels from where a person SEATED at that desk lands.
    He is the reason the zone is tested against the shoulder line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classvision.room import layout_io
from classvision.room.zones import RoomLayout, point_in_polygon

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
D14 = CONFIGS / "camera_d14.yaml"
CAM01 = CONFIGS / "camera_01.yaml"
FULL = ROOT / "out" / "full_lesson.analysis.json"


# -- the two shipped profiles load at all ----------------------------------------------

def test_both_shipped_profiles_load():
    """If either file stops loading, every run that passes `--room` stops dead. That is
    the correct behaviour and it must be found here rather than on a school's cron."""
    for path in (D14, CAM01):
        assert isinstance(layout_io.load(path), RoomLayout), path


def test_the_pair_documents_the_difference_between_the_two_cameras():
    """The whole point of shipping two profiles: one camera can measure «у доски» and one
    cannot, and the difference is a property of where the camera hangs. A `board_zone` of
    `null` with a written reason is a finding; a missing key would be an oversight, and
    `layout_io` refuses that separately (`test_board_zone_key_may_not_be_omitted`)."""
    d14 = layout_io.load(D14)
    cam01 = layout_io.load(CAM01)
    assert d14.board_zone is not None
    assert cam01.board_zone is None
    assert "доск" in cam01.notes.lower() or "доск" in CAM01.read_text(encoding="utf-8")


# -- containment, against measured observations ----------------------------------------

# t = 1764 s of D14_20260815103136.mp4. (shoulder anchor, box bottom, what they are).
# Read off the model, not placed by hand.
FRAME_1101 = [
    ((967.2, 594.7), (993.8, 850.8), "pupil seated sideways, left column"),
    ((1112.9, 315.0), (1119.2, 551.7), "PUPIL STANDING AT THE BOARD"),
    ((1420.2, 590.1), (1425.8, 658.7), "pupil seated, right column"),
    ((1502.7, 417.0), (1506.6, 450.8), "THE ADULT, seated among the pupils"),
    ((1608.8, 467.1), (1582.5, 516.6), "pupil seated, right column"),
    ((1646.3, 556.6), (1640.8, 625.1), "pupil seated, right column"),
]

# t = 1038 s. Two people at the board; the adult's box bottom (449.2) is four pixels from
# where the adult SEATED at that same desk lands at 11:01 (450.8), which is the whole
# argument for the shoulder line.
FRAME_1048_AT_BOARD = [
    ((1480.0, 214.4), (1480.0, 449.2), "THE ADULT WRITING AT THE RIGHT OF THE BOARD"),
    ((1135.4, 275.4), (1149.2, 504.9), "PUPIL WRITING AT THE LEFT OF THE BOARD"),
]

# The adult at his OWN desk — real anchors, including the extremes of the 96 observations
# the sampling caught. Seated (box ~140 px tall) and leaning against it (box ~290 px tall)
# both land here. This is the nearest thing in the room to a false «у доски».
ADULT_AT_HIS_DESK = [
    (992.2, 339.0),     # leftmost
    (1080.0, 336.0),    # rightmost
    (1040.4, 341.3),    # the median, and what a discovered seat centre would be
    (1035.7, 373.4),    # lowest shoulder line
    (1045.3, 334.5),    # highest shoulder line — 3.9 px from the at-board cluster's floor
]

# The SAME adult, standing at the LEFT END OF THE BOARD, in front of his own desk, at
# t = 540 s of the short file and t = 894/942/990 s of the long one. His legs are behind
# his own desk, so his box bottom lands at y ≈ 403 — which is where he lands SEATED at
# that desk too. Only the shoulder line tells the two apart, and this is the case the
# board zone's notch in the left-hand side exists for.
ADULT_AT_THE_LEFT_END_OF_THE_BOARD = [(1088.3, 216.4), (1083.9, 221.2),
                                      (1066.0, 222.1), (1071.5, 239.5)]


def test_the_standing_pupil_is_in_the_board_zone_and_nobody_else_is():
    layout = layout_io.load(D14)
    inside = [note for anchor, _foot, note in FRAME_1101
              if point_in_polygon(anchor, layout.board_zone)]
    assert inside == ["PUPIL STANDING AT THE BOARD"], (
        "exactly one person is at the board in the 11:01 frame; anything else means the "
        "polygon has caught a seated pupil or the adult sitting among them")


def test_the_adult_sitting_among_the_pupils_is_not_at_the_board():
    """The specific false positive this room produces. At 11:01 the adult has left his own
    desk and is sitting at a pupil's desk against the board wall, which makes him
    displaced from his seat baseline — so `states.classify` would call him `AT_BOARD` the
    moment the zone accepted him."""
    layout = layout_io.load(D14)
    adult = FRAME_1101[3][0]
    assert not point_in_polygon(adult, layout.board_zone)
    assert not point_in_polygon(adult, layout.teacher_zone)


def test_both_people_at_the_board_are_caught_even_when_the_desk_crops_their_legs():
    layout = layout_io.load(D14)
    for anchor, _foot, note in FRAME_1048_AT_BOARD:
        assert point_in_polygon(anchor, layout.board_zone), note


def test_the_foot_point_would_not_have_separated_them():
    """The measurement that decided the convention, pinned so it cannot quietly regress.

    The adult STANDING at the board (t = 1038) and the adult SEATED at that same desk
    (t = 1764) have box bottoms 1.6 px apart. Any zone tested against the foot point either
    accepts both or rejects both.
    """
    standing_foot = FRAME_1048_AT_BOARD[0][1]
    seated_foot = FRAME_1101[3][1]
    assert abs(standing_foot[1] - seated_foot[1]) < 5.0
    assert abs(standing_foot[0] - seated_foot[0]) < 40.0
    # and the shoulder lines are far apart, which is why the zone uses them
    assert FRAME_1101[3][0][1] - FRAME_1048_AT_BOARD[0][0][1] > 150.0


def test_the_adult_at_his_own_desk_lands_in_the_teacher_zone_and_not_the_board():
    layout = layout_io.load(D14)
    for anchor in ADULT_AT_HIS_DESK:
        assert point_in_polygon(anchor, layout.teacher_zone), anchor
        assert not point_in_polygon(anchor, layout.board_zone), anchor


def test_the_same_adult_standing_at_the_left_end_of_the_board_IS_at_the_board():
    """The notch in the board zone, and the reason it is not a rectangle.

    His desk hides his legs, so his box bottom at the board (y ≈ 403) is the same as his
    box bottom seated at that desk (y ≈ 403..425). The shoulder line moves 120 px.
    """
    layout = layout_io.load(D14)
    for anchor in ADULT_AT_THE_LEFT_END_OF_THE_BOARD:
        assert point_in_polygon(anchor, layout.board_zone), anchor
        assert not point_in_polygon(anchor, layout.teacher_zone), anchor


def test_the_notch_is_what_separates_them_and_a_plain_rectangle_would_not():
    """Pinned as an argument, not just as an outcome: over the left third of the board the
    two clusters are separated by the SHOULDER LINE (240 vs 334) and not by x, so a zone
    whose bottom edge is one height everywhere has to choose between losing the adult
    teaching at the left of the board and swallowing the adult at his own desk."""
    layout = layout_io.load(D14)
    flat = ((1024.0, 140.0), (1645.0, 140.0), (1645.0, 340.0), (1024.0, 340.0))
    swallowed = [a for a in ADULT_AT_HIS_DESK if point_in_polygon(a, flat)]
    assert swallowed, "a flat-bottomed zone catches the adult at his own desk"
    assert not any(point_in_polygon(a, layout.board_zone) for a in ADULT_AT_HIS_DESK)


def test_no_point_can_be_in_both_zones():
    """`point_in_polygon` is a half-open ray cast, so two polygons sharing an edge would
    put the edge in exactly one of them — a rule nobody remembers. The profile leaves a
    5 px gap instead, and this walks the seam to prove the gap is real."""
    layout = layout_io.load(D14)
    for x in range(950, 1160):
        for y in range(140, 420):
            point = (float(x), float(y))
            assert not (point_in_polygon(point, layout.board_zone)
                        and point_in_polygon(point, layout.teacher_zone)), point


def test_camera_01_teacher_zone_holds_the_adult_seat_that_was_actually_discovered():
    """Camera 01's zone is checked against the artefact the pipeline really produced, not
    against a coordinate somebody liked. Skipped when that artefact is not present."""
    if not FULL.exists():
        pytest.skip("run `classvision analyse test_camera.mp4` first")
    layout = layout_io.load(CAM01)
    artefact = json.loads(FULL.read_text(encoding="utf-8"))
    adult = tuple(artefact["teacher"]["centre"])
    assert point_in_polygon(adult, layout.teacher_zone)
    for seat in artefact["seats"]:
        assert not point_in_polygon(tuple(seat["centre"]), layout.teacher_zone), (
            f"pupil seat {seat['seat_id']} falls in camera 01's teacher zone — the adult "
            f"would be identified as a child")


# -- round trip ------------------------------------------------------------------------

def test_round_trip_preserves_everything_the_analysis_reads():
    for path in (D14, CAM01):
        original = layout_io.load(path)
        surface = layout_io.board_surface_of(path)
        again = layout_io.from_mapping(
            layout_io.to_mapping(original, board_surface=surface), source=str(path))
        assert again == original, path


def test_round_trip_survives_a_yaml_write_and_read(tmp_path: Path):
    import yaml

    original = layout_io.load(D14)
    surface = layout_io.board_surface_of(D14)
    written = tmp_path / "again.yaml"
    written.write_text(
        yaml.safe_dump(layout_io.to_mapping(original, board_surface=surface),
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    assert layout_io.load(written) == original
    assert layout_io.board_surface_of(written) == surface


# -- refusals --------------------------------------------------------------------------

def _base(**overrides) -> dict:
    """A minimal VALID profile. `board_surface` is in it because rule 6 requires it
    whenever a board zone is drawn — see
    `test_a_board_zone_without_a_board_surface_is_refused`, which is the test that stopped
    the guard from being optional."""
    data = {
        "camera": "test",
        "frame": {"width": 2560, "height": 1440},
        "board_surface": [[1024, 137], [1603, 137], [1603, 313], [1024, 313]],
        "board_zone": [[1024, 140], [1645, 140], [1645, 340],
                       [1090, 340], [1090, 265], [1024, 265]],
    }
    data.update(overrides)
    return data


def test_a_board_zone_without_a_board_surface_is_refused():
    """The hole that made rule 6 opt-in, and the third instance of this project's
    signature defect.

    Rule 6 refuses a `board_zone` drawn as the strip of FLOOR in front of the board — a
    polygon no shoulder line ever enters — by checking it against `board_surface`. But the
    check only ran `if board_surface is not None`, so omitting one key disabled it. The
    operator who draws the floor strip is exactly the operator who would not think to
    declare the chalk rectangle, so the guard was missing in the one case it exists for.

    What that costs: «у доски» is 0.0 minutes for the life of the install, and
    `lesson.unmeasured` says NOTHING, because a board zone *is* configured — so the
    artefact reports a confident zero where the truth is «мы это не измерили». That is
    exactly the failure that shipped twice before (a guessed head position, a default
    clustering threshold).
    """
    data = _base()
    del data["board_surface"]
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(data)
    assert "board_surface" in str(error.value)
    # and the message has to say what to do, in both directions
    assert "board_zone: null" in str(error.value)


def test_the_floor_strip_is_refused_even_though_it_is_a_correct_polygon():
    """Belt and braces: with `board_surface` now mandatory, the floor strip cannot reach
    the analysis by either route — omitting the surface is refused by the test above, and
    declaring it is refused by the overlap check."""
    with pytest.raises(layout_io.LayoutError):
        layout_io.from_mapping(_base(
            board_zone=[[1105, 470], [1330, 470], [1310, 620], [1100, 620]]))


def test_to_mapping_cannot_emit_a_profile_its_own_reader_would_refuse():
    """A writer that produces an unloadable file writes a broken profile onto a school's
    disk and nobody finds out until the next run."""
    layout = layout_io.load(D14)
    with pytest.raises(layout_io.LayoutError):
        layout_io.to_mapping(layout)          # board_surface not passed
    # camera_01 has no board zone, so it needs no surface and round-trips as it is
    layout_io.from_mapping(layout_io.to_mapping(layout_io.load(CAM01)))


def test_board_surface_of_does_not_invent_a_frame_size():
    """`board_surface_of` used to default a missing `frame:` to a million pixels, which
    let a surface through the out-of-frame check that `load()` refuses. A helper that is
    laxer than the loader it supports is a hole in the loader."""
    import yaml

    bad = _base()
    del bad["frame"]
    path = Path(__file__).parent / "__tmp_no_frame.yaml"
    path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    try:
        with pytest.raises(layout_io.LayoutError):
            layout_io.board_surface_of(path)
    finally:
        path.unlink()


def test_a_vertex_outside_the_frame_is_refused_not_clamped():
    """The failure this prevents: a polygon drawn on a 1920x1080 preview and pasted into a
    2560x1440 profile. Clamping would move it somewhere nobody drew and run a term on it."""
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(_base(board_zone=[[1090, 140], [2600, 140], [2600, 340]]))
    assert "вне кадра" in str(error.value)
    assert "2600" in str(error.value)


def test_a_negative_vertex_is_refused():
    with pytest.raises(layout_io.LayoutError):
        layout_io.from_mapping(_base(board_zone=[[-1, 140], [1645, 140], [1645, 340]]))


def test_an_unknown_key_is_refused_and_named():
    """`teacher_zone:` instead of `teacher_desk_zone:` is silently a layout with no teacher
    zone, which looks exactly like a camera nobody configured."""
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(_base(teacher_zone=[[0, 0], [10, 0], [10, 10]]))
    assert "teacher_zone" in str(error.value)


def test_board_zone_key_may_not_be_omitted():
    data = _base()
    del data["board_zone"]
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(data)
    assert "board_zone" in str(error.value)


def test_board_zone_may_be_explicitly_null():
    """`null` is a finding, not a gap — and a camera with no board has no `board_surface`
    to declare either, so both keys go together in both directions."""
    data = _base(board_zone=None)
    del data["board_surface"]
    layout = layout_io.from_mapping(data)
    assert layout.board_zone is None


def test_a_degenerate_polygon_is_refused():
    """Three collinear points contain nothing, forever, without raising anything."""
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(_base(board_zone=[[100, 100], [200, 100], [300, 100]]))
    assert "вырожден" in str(error.value)


def test_fewer_than_three_vertices_is_refused():
    with pytest.raises(layout_io.LayoutError):
        layout_io.from_mapping(_base(board_zone=[[100, 100], [200, 200]]))


def test_a_board_zone_drawn_as_the_strip_of_FLOOR_is_refused():
    """The trap this whole module exists for.

    The intuitive board zone is the floor a person stands on: on D14 that strip is
    [[1105,470],[1330,470],[1310,620],[1100,620]], and it is a real, correct polygon — 61
    of 61 people standing in the aisle put their feet in it. It is also a polygon no
    SHOULDER LINE ever enters, so «у доски» would be zero for the life of the install with
    nothing in the artefact to say why. Declaring `board_surface` lets the loader catch it.
    """
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(_base(
            board_surface=[[1024, 137], [1603, 137], [1603, 313], [1024, 313]],
            board_zone=[[1105, 470], [1330, 470], [1310, 620], [1100, 620]]))
    assert "ЛИНИИ ПЛЕЧ" in str(error.value)


def test_a_board_surface_without_a_board_zone_is_refused():
    with pytest.raises(layout_io.LayoutError):
        layout_io.from_mapping(_base(
            board_surface=[[1024, 137], [1603, 137], [1603, 313], [1024, 313]],
            board_zone=None))


def test_overlapping_board_and_teacher_zones_are_refused():
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.from_mapping(_base(
            teacher_desk_zone=[[1100, 200], [1300, 200], [1300, 320], [1100, 320]]))
    assert "пересекаются" in str(error.value)


def test_a_frame_size_mismatch_is_refused_before_the_run():
    layout = layout_io.load(D14)
    layout_io.check_frame(layout, 2560, 1440)          # the real recording
    with pytest.raises(layout_io.LayoutError) as error:
        layout_io.check_frame(layout, 1920, 1080)
    assert "1920x1080" in str(error.value)


def test_a_missing_file_is_a_layout_error_not_a_traceback():
    with pytest.raises(layout_io.LayoutError):
        layout_io.load("/nonexistent/camera_nowhere.yaml")


# -- «нечего измерять» must never print as «ничего не нашли» --------------------------

def test_a_person_is_never_reported_as_outside_zones_that_do_not_exist():
    """`zonecheck` verdicts, checked without the model.

    Three outcomes that must stay three sentences: a person the model gave no shoulder
    line for was NOT TESTED; a person in a profile with no zones at all had NOTHING to be
    tested against; a person tested against real zones and caught by none is «вне зон».
    Collapsing any two of them is how «мы не смогли увидеть» becomes «мы посмотрели и там
    ноль», which is the defect this package is organised against.
    """
    from classvision.report.zonecheck import PersonCheck

    def _p(**kw):
        base = dict(box=(0., 0., 10., 10.), anchor=(5., 5.), foot=(5., 10.),
                    shoulder_px=40.0, score=0.9, in_board=False, in_teacher=False,
                    in_door=False, zones_configured=("board_zone",))
        base.update(kw)
        return PersonCheck(**base)

    assert _p(anchor=None).verdict == "линия плеч не найдена — не проверен"
    assert "не против чего" in _p(zones_configured=()).verdict
    assert _p().verdict == "вне зон"
    assert _p(in_board=True).verdict == "у доски"
    # the three ASCII tags on the picture must be three tags too
    assert len({_p(anchor=None).tag, _p(zones_configured=()).tag, _p().tag}) == 3


def test_camera_01_has_no_board_zone_so_a_zero_there_is_not_a_finding():
    """The CLI's `zones` verb used to exit 1 with «разметка неверна» on camera_01, whose
    `board_zone` is a MEASURED null (the board hangs behind the lens). This pins the flag
    the verb branches on, so that the null and an empty frame can never share a code path
    again."""
    from classvision.report.zonecheck import _configured

    assert _configured(layout_io.load(CAM01)) == ("teacher_desk_zone",)
    assert _configured(layout_io.load(D14)) == ("board_zone", "teacher_desk_zone",
                                                "door_zone")


def test_excluded_seats_must_be_seat_numbers():
    assert layout_io.from_mapping(_base(excluded_seats=[3, 7])).excluded_seats == (3, 7)
    for bad in ([0], ["3"], [True], [-1]):
        with pytest.raises(layout_io.LayoutError):
            layout_io.from_mapping(_base(excluded_seats=bad))

"""The accumulation layer's tests: what it refuses, what it never overwrites, and the
one arithmetic identity that makes the whole session-folding trustworthy.

The load-bearing test here is `test_one_part_session_reproduces_the_artefacts_own_index`.
`weekly.py` recomputes the activity index from a SUMMED ledger rather than averaging the
parts' indices, and everything longitudinal rests on that recomputation being the same
computation the artefact did. If a one-recording session reproduces the artefact's number
exactly, the two-recording case is the same function on a bigger ledger. If it does not,
every weekly figure is a second, quietly different implementation of the index — the
defect `MEASUREMENTS.md` §8 is entirely about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classvision.cabinet import lessons as lessons_mod
from classvision.cabinet import report, store, weekly

ARTEFACTS = Path(__file__).resolve().parents[1] / "out"
FULL = ARTEFACTS / "full_lesson.analysis.json"
CLIP = ARTEFACTS / "clip_15min.analysis.json"
CLIP_AGAIN = ARTEFACTS / "cli_test.json"          # same video, different run_id

pytestmark = pytest.mark.skipif(
    not FULL.exists() or not CLIP.exists(),
    reason="needs the analysed artefacts in out/")


@pytest.fixture()
def db(tmp_path):
    connection = store.connect(tmp_path / "cabinet.sqlite3")
    yield connection
    connection.close()


def _load(db, path, **kwargs):
    return store.import_artefact(db, path, room_key="camera01", class_key="3-Б", **kwargs)


# ---------------------------------------------------------------------------
# Idempotency and non-overwriting.
# ---------------------------------------------------------------------------


def test_reimporting_the_same_artefact_is_a_no_op(db):
    first = _load(db, FULL)
    before = store.summary(db)
    second = _load(db, FULL)

    assert second.already_imported is True
    assert second.run_id == first.run_id
    assert store.summary(db) == before


def test_a_reanalysis_is_a_new_run_and_does_not_replace_the_old_one(db):
    original = _load(db, CLIP)
    rerun = _load(db, CLIP_AGAIN)

    assert rerun.run_id != original.run_id
    assert rerun.lesson_id == original.lesson_id      # same recording
    assert rerun.selected is False                     # ...and the aggregate does not move
    assert store.summary(db)["runs"] == 2
    assert store.summary(db)["lessons"] == 1

    lesson = store.lessons(db)[0]
    assert lesson["selected_run_id"] == original.run_id
    # The old measurement is still there in full.
    assert {r["run_id"] for r in store.runs_for_lesson(db, rerun.lesson_id)} == {
        original.run_id, rerun.run_id}


def test_moving_the_selected_run_is_a_separate_named_act(db):
    original = _load(db, CLIP)
    rerun = _load(db, CLIP_AGAIN)
    moved = store.select_run(db, rerun.run_id)

    assert moved["previous_run_id"] == original.run_id
    assert store.lessons(db)[0]["selected_run_id"] == rerun.run_id
    assert len(store.runs_for_lesson(db, original.lesson_id)) == 2   # nothing deleted


# ---------------------------------------------------------------------------
# Refusals. Each asserts on the CODE, because that is what a back-fill script branches on,
# and on the message being non-empty, because that is what a human acts on.
# ---------------------------------------------------------------------------


def test_two_recordings_of_the_same_hour_are_refused(db):
    """`clip_15min.mp4` is a slice of `test_camera.mp4`: different hashes, one hour."""
    _load(db, FULL)
    with pytest.raises(store.Refusal) as error:
        _load(db, CLIP)
    assert error.value.code == "overlapping_lesson"
    assert "пересекается" in error.value.message_ru

    forced = _load(db, CLIP, allow_overlap=True)
    assert forced.lesson_id != 0
    lesson = next(row for row in store.lessons(db)
              if row["lesson_id"] == forced.lesson_id)
    assert lesson["overlap_note"]          # ...and the override is recorded for ever


def test_an_unclocked_run_is_refused_and_then_excluded_from_everything_weekly(db, tmp_path):
    document = json.loads(FULL.read_text(encoding="utf-8"))
    document["provenance"]["started_at"] = None
    document["provenance"]["clock_source"] = "unknown"
    path = tmp_path / "unclocked.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(store.Refusal) as error:
        _load(db, path)
    assert error.value.code == "no_wall_clock"

    _load(db, path, allow_unclocked=True)
    assert store.summary(db)["lessons"] == 1
    assert store.summary(db)["lessons_dated"] == 0
    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    assert view.span == []
    for history in view.pupil_histories:
        assert history.sessions == []
        assert history.trend_view.state == "no_dated_lessons"


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d: d["provenance"].__setitem__("schema", "classvision/2.0"), "schema_version"),
    (lambda d: d.__setitem__("caveats", []), "no_caveats"),
    (lambda d: d.__setitem__("seats", []), "no_seats"),
    (lambda d: d["provenance"].__setitem__("video_sha256", ""), "no_video_hash"),
    (lambda d: d["provenance"]["room"]["seat_discovery"].__setitem__("plausible", False),
     "implausible_seats"),
    (lambda d: d["seats"][0]["metrics"]["activity"].__setitem__("parts", []),
     "index_without_parts"),
])
def test_the_refusals_fire_with_their_own_codes(db, tmp_path, mutate, code):
    document = json.loads(FULL.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / f"{code}.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(store.Refusal) as error:
        _load(db, path)
    assert error.value.code == code
    assert error.value.message_ru.strip()


def test_a_refused_import_leaves_nothing_behind_for_the_next_one_to_commit(db, tmp_path):
    """`import_many` continues past a failure; a half-written lesson must not survive it."""
    _load(db, FULL)
    before = store.summary(db)

    broken = json.loads(CLIP.read_text(encoding="utf-8"))
    broken["seats"][2].pop("seat_id")          # raises deep inside `_store_seats`
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(Exception):             # noqa: B017 - any failure must roll back
        _load(db, path, allow_overlap=True)

    assert store.summary(db) == before
    _load(db, CLIP, allow_overlap=True)        # the next import must not commit the debris
    assert store.summary(db)["lessons"] == 2
    assert store.summary(db)["runs"] == 2


def test_a_lesson_that_is_both_a_continuation_and_a_forced_overlap_keeps_both_marks(
        db, tmp_path):
    """The one row where losing the override would matter most, so it is asserted.

    Recording A runs 09:54:58–10:47:22. Recording D sits at 10:50–11:10. Recording B starts
    one second after A ends (a DVR seam) and runs long enough to collide with D. B is
    therefore a continuation AND an overlap at once, and both facts must survive.
    """
    def clone(source, started, seconds, name):
        document = json.loads(Path(source).read_text(encoding="utf-8"))
        document["provenance"]["started_at"] = started
        document["provenance"]["duration_seconds"] = seconds
        document["provenance"]["video_sha256"] = name * 8      # a different recording
        document["run_id"] = name * 4
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path

    _load(db, FULL)                                            # A, ends 10:47:22.85
    _load(db, clone(CLIP, "2026-08-07T10:50:00", 1200.0, "dddd"))
    b = _load(db, clone(CLIP, "2026-08-07T10:47:23", 1500.0, "bbbb"), allow_overlap=True)

    lesson = next(row for row in store.lessons(db)
                  if row["lesson_id"] == b.lesson_id)
    assert lesson["continues_lesson_id"] is not None, "the DVR seam was lost"
    assert lesson["overlap_note"], "the operator's override was lost"


def test_a_truncated_file_is_unreadable_not_a_refusal(db, tmp_path):
    path = tmp_path / "half.json"
    path.write_text(FULL.read_text(encoding="utf-8")[:2000], encoding="utf-8")
    with pytest.raises(store.Unreadable):
        _load(db, path)


# ---------------------------------------------------------------------------
# Places: the thing `seat_id` cannot be.
# ---------------------------------------------------------------------------


def test_seats_of_two_recordings_of_one_room_land_on_the_same_places(db):
    first = _load(db, FULL)
    second = _load(db, CLIP, allow_overlap=True)

    assert first.places_created == first.seats_stored
    assert second.places_created == 0            # every seat recognised
    assert second.places_matched == second.seats_stored
    assert second.seats_unmatched == 0


def test_a_moved_camera_creates_new_places_rather_than_inheriting_a_history(db, tmp_path):
    """Shifting every seat far enough must NOT silently continue the old histories."""
    _load(db, FULL)
    document = json.loads(CLIP.read_text(encoding="utf-8"))
    for seat in document["seats"]:
        seat["centre"] = [seat["centre"][0] + 600.0, seat["centre"][1] + 600.0]
    if document.get("teacher") and document["teacher"].get("centre"):
        document["teacher"]["centre"] = [document["teacher"]["centre"][0] + 600.0,
                                         document["teacher"]["centre"][1] + 600.0]
    path = tmp_path / "moved.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    moved = _load(db, path, allow_overlap=True)
    assert moved.places_matched == 0
    assert moved.places_created == moved.seats_stored


def test_two_classes_in_one_room_do_not_share_places(db):
    _load(db, FULL)
    store.import_artefact(db, CLIP, room_key="camera01", class_key="5-А",
                          allow_overlap=True)
    a = store.places(db, room_key="camera01", class_key="3-Б")
    b = store.places(db, room_key="camera01", class_key="5-А")
    assert a and b
    assert not ({p["place_id"] for p in a} & {p["place_id"] for p in b})


# ---------------------------------------------------------------------------
# The identity that makes session folding trustworthy.
# ---------------------------------------------------------------------------


def test_one_part_session_reproduces_the_artefacts_own_index(db):
    _load(db, FULL)
    document = json.loads(FULL.read_text(encoding="utf-8"))
    by_label = {s["label"]: s for s in document["seats"]}

    checked = 0
    for place in store.places(db, room_key="camera01", class_key="3-Б"):
        if place["role"] != "pupil":
            continue
        sessions = weekly.sessions_for_place(db, int(place["place_id"]))
        assert len(sessions) == 1
        session = sessions[0]
        row = db.execute(
            "SELECT seat_label FROM seat_lessons WHERE place_id = ?",
            (place["place_id"],)).fetchone()
        artefact_index = by_label[row["seat_label"]]["metrics"]["activity"]["index"]

        assert session.activity.available is True
        assert round(session.activity.index, 1) == artefact_index
        checked += 1
    assert checked >= 8


def test_the_adult_gets_no_index_and_no_weeks(db):
    """Rule 1: the teacher half carries no quality judgement, so it carries no index."""
    _load(db, FULL)
    adults = [p for p in store.places(db) if p["role"] == "adult"]
    assert adults, "the adult must be stored as a place, not dropped"
    history = weekly.place_history(db, adults[0])
    assert history.weeks == []
    assert history.totals == {}
    assert history.trend_view.state == "not_applicable_adult"
    for session in history.sessions:
        assert session.activity.available is False


def test_a_split_recording_is_one_session_and_its_counters_add(db, tmp_path):
    """The D14 case: one lesson, two files, 1 s apart. Two lessons, ONE session."""
    _load(db, FULL)
    document = json.loads(CLIP.read_text(encoding="utf-8"))
    # Start it 1 second after the first recording ends -- exactly the D14 geometry.
    document["provenance"]["started_at"] = "2026-08-07T10:47:23"
    path = tmp_path / "part2.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    second = _load(db, path)
    lesson = next(row for row in store.lessons(db)
              if row["lesson_id"] == second.lesson_id)
    assert lesson["continues_lesson_id"] is not None

    place = next(p for p in store.places(db) if p["role"] == "pupil")
    sessions = weekly.sessions_for_place(db, int(place["place_id"]))
    assert len(sessions) == 1                       # two recordings, one session
    assert len(sessions[0].parts) == 2

    rows = db.execute(
        "SELECT hand_raises, observations FROM seat_lessons WHERE place_id = ?",
        (place["place_id"],)).fetchall()
    assert sessions[0].counts["hand_raises"] == sum(r["hand_raises"] for r in rows)
    assert sessions[0].observations == sum(r["observations"] for r in rows)


def _dvr_pair(tmp_path):
    """The D14 geometry as two files: A ends 10:47:22.85, B starts 10:47:23."""
    second = json.loads(CLIP.read_text(encoding="utf-8"))
    second["provenance"]["started_at"] = "2026-08-07T10:47:23"
    path = tmp_path / "part2.json"
    path.write_text(json.dumps(second, ensure_ascii=False), encoding="utf-8")
    return FULL, path


@pytest.mark.parametrize("reverse", [False, True])
def test_a_split_recording_is_one_session_whichever_file_arrives_first(
        db, tmp_path, reverse):
    """A DVR seam must not depend on the order two filenames were typed in.

    Found by running it: `_immediately_before` only ever looked BACKWARDS, so importing the
    real D14 pair long-file-first stored two independent lessons — «занятий 2» for an hour
    that had one, every per-lesson figure of it halved. Unlike the overlap case nothing
    refused and nothing warned, because "a lesson followed by another lesson" is what a
    normal school day looks like. A directory back-fill takes whatever order the glob
    produced, so this is the ordinary path, not an exotic one.
    """
    first, second = _dvr_pair(tmp_path)
    for path in ([second, first] if reverse else [first, second]):
        _load(db, path)

    assert store.summary(db)["lessons"] == 2
    assert store.summary(db)["sessions"] == 1, "the DVR seam depends on import order"

    chain = {row["lesson_id"]: row["continues_lesson_id"] for row in store.lessons(db)}
    assert sorted(v is None for v in chain.values()) == [False, True]

    place = next(p for p in store.places(db) if p["role"] == "pupil")
    sessions = weekly.sessions_for_place(db, int(place["place_id"]))
    assert len(sessions) == 1
    assert len(sessions[0].parts) == 2


def test_a_seam_wide_enough_to_read_as_an_overlap_is_not_refused_in_either_order(
        db, tmp_path):
    """The forward check must feed the overlap rule too, or order decides the refusal.

    `CONTINUATION_TOLERANCE_SECONDS` (2 s) is deliberately wider than
    `OVERLAP_TOLERANCE_SECONDS` (1 s), so a seam of −2 s is a continuation looked at from
    one side and an overlap looked at from the other.
    """
    second = json.loads(CLIP.read_text(encoding="utf-8"))
    second["provenance"]["started_at"] = "2026-08-07T10:47:21"   # 1.85 s BEFORE A ends
    path = tmp_path / "early.json"
    path.write_text(json.dumps(second, ensure_ascii=False), encoding="utf-8")

    _load(db, path)                 # the later half first...
    _load(db, FULL)                 # ...and the earlier half must not be refused
    assert store.summary(db)["sessions"] == 1


def test_an_adult_without_a_position_is_reported_and_never_silently_dropped(db, tmp_path):
    """The third outing of this project's signature defect, caught on the real directory.

    `clip_15min.named.json` carries a fully measured adult whose top-level `centre` is null.
    The import stored eight places beside two other runs of the SAME lesson that stored
    nine, and printed «без привязки 0» — a room with a hole in it, reported as clean.
    "We do not know where he was" must not render as "there was no adult".
    """
    document = json.loads(FULL.read_text(encoding="utf-8"))
    document["teacher"]["centre"] = None
    document["teacher"]["scale_px"] = None
    path = tmp_path / "no_adult_position.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    result = _load(db, path)
    assert not [p for p in store.places(db) if p["role"] == "adult"]
    assert result.dropped_seats, "the adult vanished without a word"
    assert "ОТСУТСТВИЕ ИЗМЕРЕНИЯ" in result.dropped_seats[0]["why_ru"]

    # ...and it survives into the page, because an import message scrolls past once.
    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    assert len(view.adults_without_place) == 1
    html = report._caveats_block(view)
    assert "Место взрослого не заведено" in html
    assert "не вывод о том, что взрослого" in html


def test_an_adult_that_was_located_still_produces_a_place(db):
    """The guard above must not start swallowing the ordinary case."""
    _load(db, FULL)
    assert [p for p in store.places(db) if p["role"] == "adult"]
    assert not weekly.class_view(db, room_key="camera01",
                                 class_key="3-Б").adults_without_place


# ---------------------------------------------------------------------------
# The trend and the states that are not a trend.
# ---------------------------------------------------------------------------


def test_without_an_attestation_the_trend_refuses_on_identity_not_on_counting(db):
    _load(db, FULL)
    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    for history in view.pupil_histories:
        assert history.trend_view.state == "identity_not_established"
        assert history.trend_view.available is False
        gates = {g["gate"]: g["passed"] for g in history.trend_view.gates}
        assert gates["identity_attested_for_the_whole_period"] is False


def test_with_an_attestation_and_one_lesson_the_refusal_becomes_a_countdown(db):
    _load(db, FULL)
    place = next(p for p in store.places(db) if p["role"] == "pupil")
    store.attest(db, place_id=int(place["place_id"]), external_id="student_1",
                 full_name="Тестовый Т.", attested_by="ТЕСТ", attested_at="2026-08-01",
                 valid_from="2026-08-01", valid_to="2026-12-31", decision_ref="ТЕСТ/1")

    history = weekly.place_history(db, place)
    assert history.trend_view.state == "insufficient_lessons"
    assert history.trend_view.lessons_have == 1
    assert history.trend_view.lessons_needed == weekly.REQUIRED_LESSONS
    assert "1 урок из 5" in history.trend_view.headline_ru


def test_the_trend_is_delegated_and_not_reimplemented(db):
    """Five synthetic lessons drive `metrics/trend.py` through this layer, unchanged."""
    from classvision.metrics import trend as trend_mod

    series = [70.0, 72.0, 68.0, 71.0, 40.0]
    sessions = [
        weekly.SessionSeat(
            session_id=i, date_local=f"2026-09-{i + 1:02d}", iso_year=2026,
            iso_week=36 + i, parts=[i], observations=1000, absent_observations=0,
            unreadable_observations=0, hand_unmeasurable_observations=0,
            observed_seconds=500.0, coverage=1.0, settled=True, counts={},
            activity=weekly.Activity(available=True, index=value, coverage=1.0),
            stored_index=value, place_stability="stable")
        for i, value in enumerate(series)]

    view = weekly.trend_for(sessions, attested={"external_id": "x", "full_name": "X",
                                                "attested_by": "y", "attested_at": "z"},
                            attestation_covers_all=True)
    assert view.state == "available"
    assert view.trend == trend_mod.trend(series, identity_stable=True).to_dict()


def test_an_attestation_needs_every_field(db):
    _load(db, FULL)
    place = next(p for p in store.places(db) if p["role"] == "pupil")
    with pytest.raises(store.Refusal) as error:
        store.attest(db, place_id=int(place["place_id"]), external_id="student_1",
                     attested_by="", attested_at="2026-08-01", valid_from="2026-08-01",
                     decision_ref="ТЕСТ/1")
    assert error.value.code == "incomplete_attestation"


# ---------------------------------------------------------------------------
# Weeks: zero and "not observed" are different values.
# ---------------------------------------------------------------------------


def test_a_week_with_no_lesson_carries_None_and_not_zero(db):
    _load(db, FULL)
    place = next(p for p in store.places(db) if p["role"] == "pupil")
    sessions = weekly.sessions_for_place(db, int(place["place_id"]))
    span = [(2026, 31), (2026, 32), (2026, 33)]
    rows = weekly.weeks_for(sessions, span)

    observed = [r for r in rows if r.observed]
    empty = [r for r in rows if not r.observed]
    assert len(observed) == 1 and len(empty) == 2
    for row in empty:
        assert all(value is None for value in row.counts.values())
        assert row.index_median is None
    for row in observed:
        assert all(value is not None for value in row.counts.values())


# ---------------------------------------------------------------------------
# The report: three designed states, and no external asset in any of them.
# ---------------------------------------------------------------------------


def test_the_empty_cabinet_renders_a_designed_page(db, tmp_path):
    result = report.render(db, tmp_path / "html")
    assert result["state"] == "empty"
    page = Path(result["index"]).read_text(encoding="utf-8")
    assert "Пока пусто" in page
    assert "cabinet import" in page          # ...and says what to do about it


def _without_css_comments(page: str) -> str:
    """The page with `/* ... */` removed, so a URL inside a comment is not read as a fetch.

    The stylesheet is generated by Tailwind and opens with its MIT attribution banner,
    which contains `https://tailwindcss.com`. That is a licence notice — MIT requires it to
    survive in copies — and a browser never requests it. The rule this test enforces is
    «the page must not REACH OUT», and a comment reaches nowhere.
    """
    out, cursor = [], 0
    while (start := page.find("/*", cursor)) != -1:
        out.append(page[cursor:start])
        end = page.find("*/", start)
        if end == -1:
            break
        cursor = end + 2
    out.append(page[cursor:])
    return "".join(out)


def test_every_page_is_self_contained(db, tmp_path):
    """No page may fetch anything: this export is opened from a folder, offline, by a
    school psychologist. Checked against the page with CSS comments stripped — see
    `_without_css_comments` for why, and note that the check below is now on what the
    browser would actually REQUEST rather than on any occurrence of a scheme."""
    _load(db, FULL)
    result = report.render(db, tmp_path / "html")
    for path in result["pages"]:
        page = Path(path).read_text(encoding="utf-8")
        code = _without_css_comments(page)
        for forbidden in ("http://", "https://", "<script", "@import", "cdn", "url(http"):
            assert forbidden not in code, f"{Path(path).name} reaches outside itself"
        # And the comment-stripping must not become a loophole: whatever survives in the
        # comments may be a NOTICE, never a reference the browser could follow.
        for reference in ('src="http', "src='http", 'href="http', "href='http"):
            assert reference not in page, f"{Path(path).name} links out from a comment"


def test_no_index_is_ever_rendered_without_its_components(db, tmp_path):
    _load(db, FULL)
    result = report.render(db, tmp_path / "html")
    place_pages = [p for p in result["pages"] if "place-" in Path(p).name]
    for path in place_pages:
        page = Path(path).read_text(encoding="utf-8")
        if "индекс активности" in page.lower() and "Из чего складывается индекс" in page:
            assert "составляющая" in page


def test_the_caveats_are_copied_from_the_artefact_not_retyped(db, tmp_path):
    from classvision.report.artefact import CAVEATS_RU

    _load(db, FULL)
    assert set(store.caveats(db)) == set(CAVEATS_RU)
    result = report.render(db, tmp_path / "html")
    page = Path(next(p for p in result["pages"] if "class-" in Path(p).name)).read_text(
        encoding="utf-8")
    for sentence in CAVEATS_RU:
        assert sentence.split("«")[0][:40] in page


def test_an_undated_place_shows_no_counters_rather_than_a_row_of_zeros(db, tmp_path):
    """Rule 3 on the class overview, which is where it was being broken.

    A recording whose clock could not be read is excluded from everything longitudinal —
    correctly — and the card for each place then summed an EMPTY list of sessions to 0 and
    printed «поднимал руку — 0 · видимость от 0%» one line under a badge saying the lesson
    was not accumulated. The pupil in question had been watched for fifty minutes.
    """
    document = json.loads(FULL.read_text(encoding="utf-8"))
    document["provenance"]["started_at"] = None
    document["provenance"]["clock_source"] = "unknown"
    path = tmp_path / "unclocked.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    _load(db, path, allow_unclocked=True)

    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    for history in view.pupil_histories:
        assert history.sessions == []
        assert all(value is None for value in history.totals.values())
        assert history.coverage_min is None

    result = report.render(db, tmp_path / "html")
    page = Path(next(p for p in result["pages"]
                     if "class-" in Path(p).name)).read_text(encoding="utf-8")
    assert "видимость от 0%" not in page
    assert "поднимал руку — 0" not in page
    assert "не накоплены" in page


def test_reimporting_under_a_different_room_key_is_refused_not_quietly_ignored(db):
    """Idempotency is on `run_id` alone, so a corrected key used to write nothing at all.

    The operator reads «уже импортирован — ничего не записано» as «принято», and the lesson
    stays filed under the mistyped room for ever while two rooms that should accumulate
    together never do.
    """
    _load(db, FULL)
    with pytest.raises(store.Refusal) as error:
        store.import_artefact(db, FULL, room_key="camera02", class_key="3-Б")
    assert error.value.code == "already_imported_elsewhere"
    assert "camera01" in error.value.message_ru

    same = _load(db, FULL)                       # the honest no-op still works
    assert same.already_imported is True
    assert store.summary(db)["lessons"] == 1


def test_the_threshold_warning_belongs_to_the_class_it_is_printed_on(db, tmp_path):
    """A second class re-analysed once must not hang a warning on every other class."""
    document = json.loads(FULL.read_text(encoding="utf-8"))
    document["provenance"]["thresholds"] = {"invented_for_this_test": 1}
    document["run_id"] = "cccc" * 4
    document["provenance"]["video_sha256"] = "cccc" * 16
    document["provenance"]["started_at"] = "2026-08-06T09:00:00"
    path = tmp_path / "other_class.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    _load(db, FULL)                                                    # 3-Б: one set
    store.import_artefact(db, path, room_key="camera01", class_key="5-А")
    store.import_artefact(db, FULL.parent / "cli_test.json", room_key="camera01",
                          class_key="5-А", allow_overlap=True)         # 5-А: two sets

    assert store.summary(db)["threshold_sets"] == 2                    # cabinet-wide
    quiet = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    noisy = weekly.class_view(db, room_key="camera01", class_key="5-А")
    assert quiet.threshold_sets_in_class == 1
    assert noisy.threshold_sets_in_class == 2

    result = report.render(db, tmp_path / "html", room_key="camera01", class_key="3-Б")
    page = Path(next(p for p in result["pages"]
                     if "class-" in Path(p).name)).read_text(encoding="utf-8")
    assert "разными наборами" not in page


def test_the_unmeasured_line_names_both_numbers_and_agrees_with_them(db):
    """«во всех 2 прогонах этого класса» read as a claim about the class, and could be false.

    It is the number of runs carrying that note, not the number of runs the class has.
    """
    assert report._runs_phrase(1, 1) == "в единственном прогоне этого класса"
    assert report._runs_phrase(4, 4) == "во всех 4 прогонах этого класса"
    assert report._runs_phrase(2, 4) == "в 2 прогонах из 4"
    assert report._runs_phrase(1, 4) == "в 1 прогоне из 4"


@pytest.mark.parametrize(("n", "expected"), [
    (1, "1 урок"), (2, "2 урока"), (5, "5 уроков"), (11, "11 уроков"), (21, "21 урок"),
])
def test_every_numeral_on_the_page_agrees_with_its_noun(n, expected):
    """A refusal that cannot conjugate itself reads as a broken template, not a refusal."""
    assert f"{n} {weekly.lessons_word(n)}" == expected


def test_the_sparkline_caption_agrees_with_its_own_count():
    caption = report._sparkline([50.0, 60.0], ["a", "b"], connect=False)
    assert "2 урока" in caption and "2 уроков" not in caption


def test_a_forced_overlap_qualifies_the_totals_and_not_only_the_rows(db, tmp_path):
    """Rule 2: a total is rendered with what qualifies it, and the override qualifies it.

    `clip_15min` is a slice of `test_camera`, so with `--allow-overlap` place 3's «поднимал
    руку — 2» is one hand raised in the long recording plus the SAME hand raised again in
    the slice. The per-lesson rows said so; the total, which is what gets quoted, did not.
    """
    _load(db, FULL)
    _load(db, CLIP, allow_overlap=True)
    result = report.render(db, tmp_path / "html")
    pupil = next(p for p in store.places(db) if p["role"] == "pupil")
    page = Path(next(p for p in result["pages"]
                     if p.endswith(f"-{pupil['ordinal']}.html"))).read_text(encoding="utf-8")

    body = page[page.index("За всё время"):]
    assert "Что оговаривает эти итоги" in body
    assert "подтверждено оператором" in body


def test_a_1_0_artefacts_adult_shares_are_not_filed_under_a_1_1_column(db, tmp_path):
    """Two names for one column, and the older one never said what its denominator was.

    `clip_15min.analysis.json` is `classvision/1.0` and carries `seated_share` /
    `out_of_frame_share`; `full_lesson.analysis.json` is `1.1` and carries
    `at_desk_share_of_observed` / `out_of_frame_share_of_lesson`. `metrics/teacher.py`
    renamed them because «seated» meant both "the four states at his desk" (46.3 min) and
    "the one state called seated" (8.6 min), and that collision had already produced a
    false sentence in a report. The page used to bridge the two with a silent
    `metrics.get(new, metrics.get(old))` and print the 1.0 number under the 1.1 heading.

    The legacy share is READ OUT OF THE FIXTURE rather than written here as a literal. It
    was `seated_share 90.0`, and the literal turned a fixture swap into three failures that
    named the assertion instead of the cause. What this test claims is that the number
    appears **verbatim, under its own 1.0 name** — which is a statement about the renderer,
    not about any particular lesson's adult.
    """
    _load(db, FULL)                                  # 1.1
    _load(db, CLIP, allow_overlap=True)              # 1.0
    result = report.render(db, tmp_path / "html")
    adult_place = next(p for p in store.places(db) if p["role"] == "adult")
    page = Path(next(p for p in result["pages"]
                     if p.endswith(f"-{adult_place['ordinal']}.html"))).read_text(
        encoding="utf-8")

    legacy = json.loads(CLIP.read_text(encoding="utf-8"))["teacher"]["metrics"]

    assert "96.5" in page                            # the 1.1 row still renders normally
    assert "classvision/1.0" in page                 # ...and the older row names itself
    assert "не подставляются" in page
    # The number is shown, not thrown away -- and under `seated_share`, the name the 1.0
    # document actually used, never under the 1.1 heading `за столом, %`.
    assert f"seated_share {legacy['seated_share']}" in page


def test_the_1_0_fixture_is_frozen_and_must_never_be_regenerated(db):
    """`out/clip_15min.analysis.json` is EVIDENCE, not output, and it lives in a directory
    full of regenerable artefacts under a name that invites `analyse --out`.

    It was regenerated once, by a reviewer refreshing stale fixtures after a threshold
    change, and the analyser cannot emit `classvision/1.0` any more — so the legacy-schema
    coverage above simply evaporated, and reported itself as three unrelated assertion
    failures about adult shares. This test fails first and says what happened.
    """
    document = json.loads(CLIP.read_text(encoding="utf-8"))
    assert document["provenance"]["schema"] == "classvision/1.0", (
        "clip_15min.analysis.json is the only classvision/1.0 document in the tree and the "
        "only evidence that the cabinet still reads one correctly. It appears to have been "
        "overwritten by a fresh `classvision analyse` run. Restore it (out/clip_15min."
        "named.json is a 1.0 document of the same recording) rather than deleting this test.")
    metrics = (document.get("teacher") or {}).get("metrics") or {}
    assert "seated_share" in metrics and "at_desk_share_of_observed" not in metrics, (
        "the 1.0 adult share names are what this fixture exists to carry")


def test_a_session_of_new_places_does_not_claim_they_were_matched(db):
    """One name, two meanings: `seat_lessons.place_match` is per recording and has four
    values; the session-level field had two and reused the word "matched" for both."""
    _load(db, FULL)
    place = next(p for p in store.places(db) if p["role"] == "pupil")
    session = weekly.sessions_for_place(db, int(place["place_id"]))[0]

    rows = db.execute("SELECT place_match FROM seat_lessons WHERE place_id = ?",
                      (place["place_id"],)).fetchall()
    assert {r["place_match"] for r in rows} == {"new"}     # nothing was ever matched...
    assert session.place_stability == "stable"             # ...and the session says so
    assert "place_match" not in session.to_dict()


# ---------------------------------------------------------------------------
# A counter the camera cannot measure is never a zero. Rule 3, one level further out
# than `WeekRow` had it.
# ---------------------------------------------------------------------------


def test_the_unmeasurable_counter_map_matches_what_the_analyser_emits():
    """The map is a string join, so a reworded analyser must fail HERE and not silently.

    `cabinet/lessons.py::COUNTER_VOIDED_BY` is keyed on `lesson.unmeasured[].what` as
    `pipeline.py` spells it. Nothing in Python couples the two, so a tidy-up of that
    sentence would stop the voiding firing and bring the confident zero back with no test
    failing — which is exactly the class of defect this project keeps paying for. The
    fixture is the evidence: `full_lesson.analysis.json` is a real camera01 run and really
    does carry the string.
    """
    document = json.loads(FULL.read_text(encoding="utf-8"))
    emitted = {item["what"] for item in document["lesson"]["unmeasured"]}
    assert "стоял у доски" in emitted, (
        "the analyser no longer emits «стоял у доски» for a camera with no board zone; "
        "cabinet/lessons.py::COUNTER_VOIDED_BY is keyed on that exact string and has just "
        "stopped voiding «выходил к доске», so every page is printing a confident 0 for an "
        "event that camera cannot witness. Update both together.")
    assert lessons_mod.COUNTER_VOIDED_BY["стоял у доски"] == ("board_visits",)
    assert lessons_mod.voided_counters(document["lesson"]["unmeasured"]).keys() == {
        "board_visits"}


def test_a_counter_the_camera_cannot_measure_is_printed_as_a_refusal_not_as_zero(
        db, tmp_path):
    """camera01's board is behind the lens, so «выходил к доске» is 0 by construction.

    Measured, not supposed: all eight pupils of `full_lesson.analysis.json` carry
    `board_visits: 0`, the artefact says why in `lesson.unmeasured`, and the teacher half of
    the same document already reports `presence.board.minutes_of_lesson: null` beside
    `zone_configured: false`. The pupil half printed the zero.
    """
    _load(db, FULL)
    result = report.render(db, tmp_path / "html")
    pupil = next(p for p in store.places(db) if p["role"] == "pupil")
    page = Path(next(p for p in result["pages"]
                     if p.endswith(f"-{pupil['ordinal']}.html"))).read_text(encoding="utf-8")

    # The «за всё время» tile, which is the number a reader quotes.
    assert ("<strong>не измерялось</strong><br><span class=\"none\">выходил к доске"
            in page)
    assert "<strong>0</strong><br><span class=\"none\">выходил к доске" not in page
    assert "зона доски не задана для этой камеры" in page
    # ...and the index that counts it as zero says it is a lower bound.
    assert "Индекс на этой камере — нижняя граница" in page

    overview = Path(next(p for p in result["pages"]
                         if "class-" in Path(p).name)).read_text(encoding="utf-8")
    assert "выходил к доске — 0" not in overview
    assert "выходил к доске — не измерялось" in overview


def test_a_zero_the_same_artefact_contradicts_is_printed_with_the_contradiction(
        db, tmp_path):
    """D14 CAN measure «у доски», reports 0 pupil visits, and 30 minutes of board use.

    The counter is a state of a PLACE, so a child who walks to the board has left theirs and
    lands in «уходил с места» or in the observations belonging to no place at all. The zero
    is true of the ledger and false of the room, and the page prints both numbers together.
    """
    d14 = ARTEFACTS / "d14_session.analysis.json"
    if not d14.exists():
        pytest.skip("needs out/d14_session.analysis.json")
    store.import_artefact(db, d14, room_key="D14", class_key="8-А")
    views = lessons_mod.lesson_views(db, room_key="D14", class_key="8-А")
    assert len(views) == 1
    lesson = views[0]

    assert lesson.counts["board_visits"] == 0
    assert "board_visits" not in lesson.voided          # this camera CAN see the board
    assert lesson.board_conflict_ru, "the contradicting number was not surfaced"
    assert "у доски кто-то находился" in lesson.board_conflict_ru
    assert "не означает, что ученики к доске не выходили" in lesson.board_conflict_ru

    result = report.render(db, tmp_path / "html")
    # On the lesson card, on the class page and on every place page of the class: the zero
    # appears on all three, so the number contradicting it has to appear on all three.
    for prefix in ("lesson-", "class-", "place-"):
        page = Path(next(p for p in result["pages"]
                         if Path(p).name.startswith(prefix))).read_text(encoding="utf-8")
        assert "Ноль, которому противоречит эта же запись" in page, prefix
        assert "30.4 мин" in page, prefix


# ---------------------------------------------------------------------------
# Two rooms under one class key: the state this surface exists to shout about.
# ---------------------------------------------------------------------------


def _two_rooms(db):
    """One class KEY, two rooms — the real cabinet's actual state, in miniature.

    `camera01` and `camera02` here stand for camera01 and D14 in `out/`: different rooms,
    different pupil counts, no attested identity anywhere. The overlap rule is scoped to a
    room, so the same hour in a second room is not a duplicate and is not refused.
    """
    _load(db, FULL)
    store.import_artefact(db, CLIP, room_key="camera02", class_key="3-Б")


def test_a_class_key_spanning_two_rooms_is_warned_about_on_every_page(db, tmp_path):
    _two_rooms(db)
    across = {c.class_key: c for c in lessons_mod.classes_across_rooms(db)}
    assert across["3-Б"].multi_room is True
    assert {r["room_key"] for r in across["3-Б"].rooms} == {"camera01", "camera02"}

    result = report.render(db, tmp_path / "html")
    warned = 0
    for path in result["pages"]:
        page = Path(path).read_text(encoding="utf-8")
        name = Path(path).name
        if name.startswith(("class-", "place-", "lesson-")) or name in ("index.html",
                                                                        "lessons.html"):
            assert "сняты РАЗНЫМИ камерами" in page, f"{name} could be misread"
            assert "НЕ является динамикой" in page, f"{name} does not say what follows"
            warned += 1
    assert warned == len(result["pages"])


def test_the_cross_room_warning_is_built_from_the_store_and_not_frozen(db):
    """A warning that keeps warning after the keys are fixed is worse than none."""
    _two_rooms(db)
    across = {c.class_key: c for c in lessons_mod.classes_across_rooms(db)}
    sentences = " ".join(across["3-Б"].warning_ru())
    assert "camera01" in sentences and "camera02" in sentences

    # ...and one room produces no warning at all, from the same code path.
    single = store.connect(":memory:")
    store.import_artefact(single, FULL, room_key="camera01", class_key="3-Б")
    only = {c.class_key: c for c in lessons_mod.classes_across_rooms(single)}
    assert only["3-Б"].multi_room is False
    assert only["3-Б"].warning_ru() == []
    assert report._cross_room_block(only["3-Б"].rooms, "3-Б") == ""
    single.close()


def test_an_unstated_class_key_is_not_rendered_as_the_name_of_a_class(db, tmp_path):
    """`"-"` is the ABSENCE of a class, and two lessons under it are not one class."""
    store.import_artefact(db, FULL, room_key="camera01")     # no --class
    view = weekly.class_view(db, room_key="camera01", class_key="-")
    assert view.class_stated is False
    assert view.class_display_ru == "класс не указан"

    result = report.render(db, tmp_path / "html")
    index = Path(result["index"]).read_text(encoding="utf-8")
    assert "класс не указан" in index
    assert "это не имя класса, а его отсутствие" in index


def test_no_page_states_or_implies_a_trend_when_none_can_be_computed(db, tmp_path):
    """The whole point. Two rooms, one lesson each, nothing attested — so no direction.

    Asserted against `metrics/trend.RU_DIRECTION`'s own vocabulary rather than a hand-typed
    list, so a new direction word cannot be added without this test noticing.

    The banned list is the three DIRECTIONS and the overview's «динамика построена» badge —
    the strings that would only ever be printed as a claim. It deliberately does NOT ban
    «стало хуже»: that phrase appears on these pages inside a refusal explaining why such a
    sentence cannot be produced from three lessons, and banning a word rather than a claim
    would push the refusal into a vaguer wording to satisfy a test.
    """
    from classvision.metrics import trend as trend_mod

    _two_rooms(db)
    result = report.render(db, tmp_path / "html")
    forbidden = [trend_mod.RU_DIRECTION[d] for d in (trend_mod.Direction.LOWER,
                                                     trend_mod.Direction.HIGHER,
                                                     trend_mod.Direction.STEADY)]
    forbidden += ["динамика построена"]
    for path in result["pages"]:
        page = Path(path).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in page, f"{Path(path).name} implies a trend: {phrase!r}"
        # ...and the sparkline never joins its dots when the trend was refused.
        assert "<path d=" not in page, f"{Path(path).name} drew a direction"


# ---------------------------------------------------------------------------
# The refusal as a purchase order.
# ---------------------------------------------------------------------------


def test_the_refusal_says_how_many_more_lessons_and_asks_for_a_signed_plan(db, tmp_path):
    _load(db, FULL)
    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    history = view.pupil_histories[0]
    keys = [r.key for r in history.requests]
    assert keys[:2] == ["more_lessons_same_room", "signed_seating_plan"]

    more = history.requests[0]
    assert "Ещё 4" in more.what_ru                       # 5 required, 1 in the store
    assert "ЭТОЙ ЖЕ комнате (camera01)" in more.what_ru
    assert "cabinet import" in more.how_ru

    plan = history.requests[1]
    # The display label and the database row id are BOTH in the sentence, and the sentence
    # says which is which: «место 3 (место №12)» would be two numbers under one word, in a
    # request a school is meant to act on.
    assert f"«{history.place['label_ru']}» в комнате camera01" in plan.what_ru
    assert "не является номером места в комнате" in plan.what_ru
    assert "С ДАТОЙ ОКОНЧАНИЯ" in plan.how_ru            # the trap an open plan sets
    assert f"--place-id {history.place['place_id']}" in plan.how_ru
    assert "решение школы" in plan.how_ru

    result = report.render(db, tmp_path / "html")
    page = Path(next(p for p in result["pages"]
                     if p.endswith(f"-{history.place['ordinal']}.html"))).read_text(
        encoding="utf-8")
    assert "Что запросить у школы" in page
    assert "Ещё 4" in page and "cabinet attest" in page


def test_the_request_list_grows_a_room_item_only_when_a_second_room_shares_the_key(db):
    _load(db, FULL)
    one = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    assert "one_class_one_room" not in [r.key for r in one.pupil_histories[0].requests]

    store.import_artefact(db, CLIP, room_key="camera02", class_key="3-Б")
    two = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    room_request = next(r for r in two.pupil_histories[0].requests
                        if r.key == "one_class_one_room")
    assert "camera02" in room_request.what_ru
    assert "комнаты camera02" in room_request.what_ru     # singular, because one room


def test_the_adult_is_asked_for_nothing_because_no_trend_is_owed(db):
    """Rule 1: there is no teacher trend to unlock, so there is no purchase order either."""
    _load(db, FULL)
    view = weekly.class_view(db, room_key="camera01", class_key="3-Б")
    adult = next(h for h in view.histories if h.place["role"] == "adult")
    assert adult.requests == []
    assert adult.trend_view.state == "not_applicable_adult"


# ---------------------------------------------------------------------------
# The lesson page and the cross-lesson page.
# ---------------------------------------------------------------------------


def test_there_is_one_lesson_page_per_session_and_not_per_recording(db, tmp_path):
    """A DVR split is one lesson. Two files must not produce two lesson cards."""
    first, second = _dvr_pair(tmp_path)
    _load(db, first)
    _load(db, second)
    assert store.summary(db)["lessons"] == 2
    assert store.summary(db)["sessions"] == 1

    views = lessons_mod.lesson_views(db, room_key="camera01", class_key="3-Б")
    assert len(views) == 1
    assert len(views[0].parts) == 2

    result = report.render(db, tmp_path / "html")
    pages = [p for p in result["pages"] if Path(p).name.startswith("lesson-")]
    assert len(pages) == 1
    page = Path(pages[0]).read_text(encoding="utf-8")
    assert "занятие записано 2 файлами подряд" in page
    # ...and no per-part index is invented for a session the place page recomputes.
    assert "не складывается из индексов частей" in page


def test_the_lesson_page_leads_with_what_could_not_be_measured(db, tmp_path):
    _load(db, FULL)
    result = report.render(db, tmp_path / "html")
    page = Path(next(p for p in result["pages"]
                     if Path(p).name.startswith("lesson-"))).read_text(encoding="utf-8")

    assert page.index("Что этой записью НЕ измерено") < page.index("Комната целиком")
    assert "в файле нет аудиодорожки" in page
    # The class-level counters are named as a property of the room, not of a child.
    assert "Это сумма по комнате, а не показатель ребёнка" in page
    # ...and the «what to look at» half is a list of measurement limits, never of children.
    assert "не список детей" in page or "были видны больше половины" in page


def test_the_cross_lesson_page_says_it_is_about_recordings_and_not_about_children(
        db, tmp_path):
    _two_rooms(db)
    result = report.render(db, tmp_path / "html")
    page = Path(result["comparable"]).read_text(encoding="utf-8")

    assert "Ни одна строка здесь не является утверждением о ребёнке" in page
    assert "Ни одна пара строк здесь не является динамикой" in page
    # Every column carries what it is NOT, and the two lists cannot drift apart.
    for column in lessons_mod.COMPARABLE_COLUMNS:
        assert column["is_not"][:40] in page
    assert "не численность класса" in page


def test_one_lesson_in_the_cabinet_is_a_designed_state_and_not_a_one_row_table(
        db, tmp_path):
    _load(db, FULL)
    result = report.render(db, tmp_path / "html")
    page = Path(result["comparable"]).read_text(encoding="utf-8")
    assert "В кабинете одна запись" in page
    assert "Сравнивать её не с чем" in page


def test_a_lesson_with_no_selected_measurement_is_a_sentence_and_not_a_row_of_zeros(
        db, tmp_path):
    """`select="never"` is a real state — an imported run nobody has chosen yet.

    Every table on this surface keys off `lessons.selected_run_id`, so such a lesson has no
    places, no coverage and no counters, and rendering it normally printed «мест 0,
    видимость —, поднимал руку 0» about an hour that was recorded and analysed in full.
    Rule 3 in the one row where the join, rather than the camera, produced the emptiness.
    """
    store.import_artefact(db, FULL, room_key="camera01", class_key="3-Б", select="never")
    view = lessons_mod.lesson_views(db, room_key="camera01", class_key="3-Б")[0]
    assert view.run_ids == []
    assert view.places == []
    assert view.notes_ru and "нет выбранного измерения" in view.notes_ru[0]
    assert "Это не пустой урок" in view.notes_ru[0]

    result = report.render(db, tmp_path / "html")
    overview = Path(next(p for p in result["pages"]
                         if "class-" in Path(p).name)).read_text(encoding="utf-8")
    assert "не выбран расчётный прогон" in overview
    assert "cabinet select-run" in overview
    comparable = Path(result["comparable"]).read_text(encoding="utf-8")
    assert "не выбран расчётный прогон" in comparable


def test_an_empty_cabinet_comparable_view_is_an_absence_and_not_an_empty_table(db):
    view = lessons_mod.comparable(db)
    assert view.lessons == []
    assert view.cross_room == []


def test_the_lesson_view_never_reads_the_artefact_again(db, tmp_path):
    """The card is computed from the store, so it cannot disagree with the other pages."""
    _load(db, FULL)
    lesson = lessons_mod.lesson_views(db, room_key="camera01", class_key="3-Б")[0]
    rows = db.execute(
        "SELECT SUM(hand_raises) AS n FROM seat_lessons WHERE role = 'pupil'").fetchone()
    assert lesson.counts["hand_raises"] == int(rows["n"])
    assert len(lesson.pupil_places) == sum(1 for p in store.places(db)
                                           if p["role"] == "pupil")


def test_the_cabinet_never_imports_a_model_stack():
    """The whole point of the artefact seam, asserted where the web side would break it."""
    import subprocess
    import sys

    code = ("import classvision.cabinet.store, classvision.cabinet.weekly, "
            "classvision.cabinet.report, sys, json;"
            "print(json.dumps([m for m in sys.modules if m.split('.')[0] in "
            "{'torch','ultralytics','cv2','insightface','onnxruntime','matplotlib'}]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]),
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert json.loads(out.stdout) == []

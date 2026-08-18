"""The orientation note inside the cabinet: what is sent, what is kept, and what expires.

Nothing here needs an API key, and that is deliberate — a test that requires a paid provider
is a test that gets skipped for ever, and then the only thing anybody ever runs is the guard
in `test_orientation.py`. What is pinned here is everything AROUND the call: the numbering
that goes into the request, the fields that must never be in it, and the four states the
page has to be able to render.

The load-bearing test is
`test_the_bundle_is_numbered_by_place_not_by_the_artefacts_own_seat_ids`. On the real
artefact the two numberings differ by one at every seat, so the first version of
`bundle_for_run` — which passed the artefact's `seat_id` straight through, as the
artefact-side path does — produced a note whose every sentence was true of the bundle and
pointed one desk to the left of what the page printed under the same words. Nothing in the
guard, in the prompt or in the HTML could have caught that: «место 4» is a valid seat label
on both sides of the mistake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classvision.cabinet import lessons, notes, report, store
from classvision.report import orientation

ARTEFACTS = Path(__file__).resolve().parents[1] / "out"
FULL = ARTEFACTS / "full_lesson.analysis.json"
CLIP = ARTEFACTS / "clip_15min.analysis.json"
CLIP_AGAIN = ARTEFACTS / "cli_test.json"          # same video, different thresholds

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


def _no_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _store_a_note(db, run_id, *, text="Стоит посмотреть на место 3.", available=True,
                  guard_passed=True, offending=(), reason="", model="gemini-3.6-flash",
                  backend="auto", bundle_sha=None):
    """Put a note in the store without going near a provider."""
    _text, digest, size = notes.payload(notes.bundle_for_run(db, run_id))
    return store.save_note(
        db, run_id, text=text, available=available, source="gemini" if available else "none",
        backend_requested=backend, model=model, prompt_version="orientation/1.0",
        guard_passed=guard_passed, guard_reason_ru="проверка пройдена" if guard_passed
        else "записка отклонена", guard_offending=offending, reason_ru=reason,
        bundle_sha256=bundle_sha or digest, bundle_bytes=size)


# ---------------------------------------------------------------------------
# What goes into the request.
# ---------------------------------------------------------------------------


def test_the_bundle_is_numbered_by_place_not_by_the_artefacts_own_seat_ids(db):
    """THE REGRESSION. The artefact numbers seats per run and the cabinet numbers places per
    room, and on `full_lesson.analysis.json` they differ by one at every seat because the
    adult holds artefact `seat_id = 1`. A note written against artefact ids would say «место
    4» directly above a table whose «место 4» is a different child — every sentence true,
    every sentence off by one desk, and nothing anywhere to reveal it."""
    result = _load(db, FULL)
    bundle = notes.bundle_for_run(db, result.run_id)

    artefact_ids = {int(row["seat_id"]) for row in db.execute(
        "SELECT seat_id FROM seat_lessons WHERE run_id = ? AND role = 'pupil'",
        (result.run_id,))}
    ordinals = {int(row["ordinal"]) for row in db.execute(
        "SELECT ordinal FROM places WHERE role = 'pupil'")}
    assert artefact_ids != ordinals                       # the premise of the whole test
    assert {seat["seat_id"] for seat in bundle["seats"]} == ordinals

    # ...and each number carries ITS OWN place's numbers, not merely a renumbering.
    for seat in bundle["seats"]:
        row = db.execute(
            "SELECT s.coverage, s.hand_raises FROM seat_lessons s "
            "JOIN places p ON p.place_id = s.place_id "
            "WHERE s.run_id = ? AND p.ordinal = ?",
            (result.run_id, seat["seat_id"])).fetchone()
        assert seat["coverage_percent"] == round(100.0 * row["coverage"], 1)
        assert seat["counts"]["hand_raises"] == row["hand_raises"]


def test_the_bundle_carries_no_name_even_when_the_place_is_attested(db):
    """A signed seating plan changes what the PAGE prints. It must not change what leaves
    the machine: the whole privacy claim in `cabinet/notes.py` is that the re-identification
    key stays in the school, and a `full_name` in the request body would hand over both
    halves at once."""
    result = _load(db, FULL)
    place = store.places(db)[0]
    store.attest(db, place_id=int(place["place_id"]), external_id="student_17",
                 full_name="Иванов Иван Иванович", attested_by="Петрова А.А., классный рук.",
                 attested_at="2026-08-01", valid_from="2026-08-01",
                 decision_ref="приказ №12 от 2026-08-01")

    text, _sha, _size = notes.payload(notes.bundle_for_run(db, result.run_id))
    assert "Иванов" not in text
    assert "student_17" not in text
    assert "Петрова" not in text
    assert all(seat["identified"] is False for seat in json.loads(text)["seats"])


def test_the_bundle_carries_no_dates_no_geometry_and_no_run_id(db):
    """The whitelist, asserted from the outside. Two lessons of one class produce two
    requests that cannot be linked to each other, to a room or to a calendar by content."""
    result = _load(db, FULL)
    text, _sha, size = notes.payload(notes.bundle_for_run(db, result.run_id))

    for forbidden in ("2026", "camera01", "3-Б", result.run_id, "centre", "anchor",
                      "video", "sha256", "full_lesson"):
        assert forbidden not in text, forbidden
    # A bound, not a decoration: the docstring quotes measured byte counts, and a bundle
    # that silently grew a field would make that paragraph false without failing anything.
    assert size < 4096


def test_the_adult_is_not_in_the_bundle_and_the_omission_is_stated(db):
    """`weekly.py` refuses an activity index for the adult by role and INTEGRATION.md §7
    refuses a staff-comparison surface; one fluent sentence about where the teacher was
    would be that surface. Rule 4 all the same: the omission is listed, not silent."""
    result = _load(db, FULL)
    bundle = notes.bundle_for_run(db, result.run_id)

    adults = db.execute("SELECT COUNT(*) FROM seat_lessons WHERE run_id = ? AND "
                        "role = 'adult'", (result.run_id,)).fetchone()[0]
    assert adults == 1
    assert len(bundle["seats"]) == 8
    assert any("взрослый" in item["what"] for item in bundle["unmeasured"])


def test_a_seat_with_no_place_is_left_out_and_its_absence_is_stated(db):
    """A seat geometry could not attach to a known place has no number the page could
    resolve, so it cannot be pointed at. Dropping it quietly would hide part of the room
    from the reader of the note; the model is told instead."""
    result = _load(db, FULL)
    db.execute("UPDATE seat_lessons SET place_id = NULL WHERE run_id = ? AND seat_id = 4",
               (result.run_id,))

    bundle = notes.bundle_for_run(db, result.run_id)
    assert len(bundle["seats"]) == 7
    # ...and the sentence saying so carries no numeral: everything in `unmeasured` is quoted
    # to a model that is forbidden to write digits, and the guard throws away the WHOLE note
    # over one. A bundle that says «2 места» is a trap this module would be setting for
    # itself, paid for by losing the note entirely.
    stated = [item["what"] for item in bundle["unmeasured"] if "не привязана" in item["what"]]
    assert stated and not any(character.isdigit() for character in stated[0])


def test_a_counter_this_camera_cannot_measure_is_sent_as_null_and_not_as_zero(db):
    """Rule 3 of this project, at the one place where a zero is read by something fluent.

    On camera 01 the board hangs behind the lens (`board_zone: null`), so `board_visits` is
    zero for every pupil in every lesson that camera will ever record. Handing a language
    model that zero is handing it «к доске никто не выходил» — a finding about children
    assembled out of a fact about a lens.
    """
    camera01 = _load(db, FULL)
    seats = notes.bundle_for_run(db, camera01.run_id)["seats"]
    assert all(seat["counts"]["board_visits"] is None for seat in seats)
    assert all(seat["counts"]["hand_raises"] is not None for seat in seats)


def test_the_board_zero_d14_contradicts_itself_is_not_sent_as_a_measurement(db):
    """The SECOND cause of a false zero in the same column, which the first fix could not see.

    This test used to assert the opposite — `board_visits == 0` for all five D14 places —
    on the reasoning that D14 has the board in frame, so the column is measurable and the
    zero is a measurement. The artefact disagrees with itself: `cabinet/lessons.py` reports,
    from this very run, somebody standing at that board for 30.4 of the 58.1 minutes with
    only 4.0 of them the adult, and `GROUND_TRUTH_D14.md` records a human seeing a PUPIL at
    the board at t≈1800. `AT_BOARD` is a state of a PLACE, so a child who walks to the board
    has left their place and is counted somewhere else — the zero is true of the ledger and
    false of the room.

    The class page prints that zero with the contradicting minutes beside it. A note may not
    print a number, so its only two options are a stated null and a confident «к доске никто
    не выходил» — which is the exact sentence the camera-01 fix above exists to prevent,
    arriving by a route that fix does not cover.
    """
    d14_path = ARTEFACTS / "d14_session.analysis.json"
    if not d14_path.exists():
        pytest.skip("needs out/d14_session.analysis.json")
    d14 = store.import_artefact(db, d14_path, room_key="D14", class_key="8-А")

    assert lessons.board_conflict_for_run(db, d14.run_id), "the contradiction must be found"
    bundle = notes.bundle_for_run(db, d14.run_id)
    assert all(seat["counts"]["board_visits"] is None for seat in bundle["seats"])

    # ...and the reason is stated rather than the column silently going blank (rule 4), in a
    # sentence carrying no digit for the guard to reject the whole note over.
    stated = [item["what"] for item in bundle["unmeasured"] if "к доске" in item["what"]]
    assert stated, bundle["unmeasured"]
    assert not any(character.isdigit() for character in stated[0] + str(
        [item["why"] for item in bundle["unmeasured"] if "к доске" in item["what"]]))

    # Every other counter of this camera stays a real number: the fix is about one column.
    assert all(seat["counts"]["turned_away_episodes"] is not None
               for seat in bundle["seats"])


def test_a_zero_that_only_means_no_episode_was_long_enough_is_labelled_as_such(db):
    """«Не было» and «было, но коротко» are different facts and used to arrive as one `0`.

    The counters are EPISODE counts. `ledger.py` keeps the distinction in
    `observed_seconds_by_state` beside `episode_seconds`, and the bundle carried only the
    latter — so a state observed for a minute in runs too short to become an episode reached
    the model as a flat zero, and a fluent model turns a flat zero into a universal negative.

    Both real notes did exactly that, and both sentences are false:

      * camera 01 «место 8» — «отсутствие каких-либо движений … всё время удалось провести
        спокойно за партой», against 150 observations of head_down (75.0 s) and three of a
        raised hand;
      * D14 — «ни на одном из мест … не опускал голову на стол», against 32.0 s, 3.5 s and
        20.0 s of observed head_down at three of the five places.
    """
    camera01 = _load(db, FULL)
    seats = {seat["seat_id"]: seat for seat in notes.bundle_for_run(
        db, camera01.run_id)["seats"]}

    # The place the note called motionless. Its zeros are the qualified kind.
    assert seats[8]["counts"]["head_down_episodes"] == 0
    assert "head_down_episodes" in seats[8]["zero_but_briefly_seen"]
    assert "hand_raises" in seats[8]["zero_but_briefly_seen"]

    # A zero with genuinely nothing behind it stays unqualified, or the label means nothing.
    assert seats[3]["counts"]["stands"] == 0
    assert "stands" not in seats[3]["zero_but_briefly_seen"]

    # A counter that is null for this camera has no zero to qualify.
    assert all("board_visits" not in seat["zero_but_briefly_seen"]
               for seat in seats.values())

    # A non-zero counter is never qualified: the label is about zeros only.
    assert "away_episodes" not in seats[1]["zero_but_briefly_seen"]
    assert seats[1]["counts"]["away_episodes"] == 4

    # The model is told the field exists and what it means, digit-free.
    stated = [item["what"] for item in notes.bundle_for_run(db, camera01.run_id)["unmeasured"]
              if "эпизода" in item["what"]]
    assert stated and not any(character.isdigit() for character in stated[0])


def test_the_prompt_decides_which_places_need_a_visibility_caveat_not_the_model(db):
    """Rule 8 used to leave «плохо видно» to the model's judgement, and it picked the worst two.

    On D14 the model qualified 61.5 % and 65.4 % and said nothing about «место 4» at 78.8 %
    — nearly a fifth of the hour unseen, described as flatly as a place seen throughout.
    Nothing in the prompt named a line, so there was nothing to be wrong about.
    `VISIBILITY_CAVEAT_BELOW_PERCENT` names it, and the bundle carries the verdict rather
    than the arithmetic.
    """
    d14_path = ARTEFACTS / "d14_session.analysis.json"
    if not d14_path.exists():
        pytest.skip("needs out/d14_session.analysis.json")
    d14 = store.import_artefact(db, d14_path, room_key="D14", class_key="8-А")
    small = orientation.compact_for_note(notes.bundle_for_run(db, d14.run_id))
    flagged = {seat["seat_id"]: seat["needs_visibility_caveat"] for seat in small["seats"]}

    assert flagged[4] is True, "78.8 % — the place the first prompt described with no caveat"
    assert flagged[2] is True and flagged[5] is True      # 61.5 %, 65.4 %
    assert flagged[1] is False and flagged[3] is False    # 94.9 %, 96.7 %


def test_an_unknown_coverage_needs_the_caveat_rather_than_being_treated_as_good(db):
    """The one direction this flag must never fail in: no coverage is not full coverage."""
    small = orientation.compact_for_note(
        {"seats": [{"seat_id": 1, "coverage_percent": None, "counts": {}}]})
    assert small["seats"][0]["needs_visibility_caveat"] is True


def test_the_lesson_length_is_the_analysed_window_not_the_file(db):
    """`coverage_percent` is a share of the analysed window. Publishing the file's length
    beside it would offer two lengths and no way to tell which divides which."""
    result = _load(db, FULL)
    bundle = notes.bundle_for_run(db, result.run_id)
    artefact = json.loads(FULL.read_text(encoding="utf-8"))

    assert bundle["lesson"]["duration_minutes"] == artefact["lesson"]["duration_minutes"]
    recording = store.lessons(db)[0]["duration_seconds"] / 60.0
    assert bundle["lesson"]["duration_minutes"] != round(recording, 1)


# ---------------------------------------------------------------------------
# What is kept, and for how long.
# ---------------------------------------------------------------------------


def test_regenerating_replaces_the_note_and_never_duplicates_it(db):
    """A nightly `cabinet note --all` must leave one note per run for ever. The key is the
    constraint; this asserts the constraint is the one that is actually there."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id, text="Первая записка про место 3.")
    _store_a_note(db, result.run_id, text="Вторая записка про место 5.")

    assert db.execute("SELECT COUNT(*) FROM run_notes").fetchone()[0] == 1
    assert notes.stored_note(db, result.run_id).text == "Вторая записка про место 5."


def test_a_reanalysis_starts_with_no_note_and_the_old_run_keeps_its_own(db):
    """THE REQUIREMENT. A lesson re-analysed under different thresholds is a new run, and a
    note written about the old thresholds must not appear beside the new numbers. Keying the
    note on `lesson_id` would have moved it across; keying it on `run_id` cannot."""
    original = _load(db, CLIP)
    rerun = _load(db, CLIP_AGAIN)
    assert rerun.lesson_id == original.lesson_id and rerun.run_id != original.run_id

    _store_a_note(db, original.run_id, text="Записка про место 3.")

    assert notes.stored_note(db, rerun.run_id) is None
    assert notes.stored_note(db, original.run_id).text == "Записка про место 3."

    # ...and it does not reappear when the aggregate is switched to the new run either.
    store.select_run(db, rerun.run_id)
    assert notes.stored_note(db, rerun.run_id) is None


def test_a_note_about_numbers_that_have_since_changed_is_stale_and_not_shown(db):
    """The second lock. The run id covers everything the ANALYSER could change; the bundle
    hash covers anything the STORE could — a rebuilt cabinet, a corrected import, a column
    added later. A note is a statement about specific figures, and once we cannot show they
    are the figures beside it, the honest move is to withhold it and say why."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id)
    assert notes.stored_note(db, result.run_id).state == "shown"

    db.execute("UPDATE seat_lessons SET hand_raises = hand_raises + 1 "
               "WHERE run_id = ? AND seat_id = 4", (result.run_id,))
    stale = notes.stored_note(db, result.run_id)
    assert stale.stale is True and stale.state == "stale"


def test_deleting_a_run_takes_its_note_with_it(db):
    """`ON DELETE CASCADE`, asserted rather than assumed: SQLite enforces foreign keys only
    when the pragma is on, and `connect()` is the only place that turns it on."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id)
    db.execute("DELETE FROM seat_lessons WHERE run_id = ?", (result.run_id,))
    db.execute("UPDATE lessons SET selected_run_id = NULL")
    db.execute("DELETE FROM runs WHERE run_id = ?", (result.run_id,))

    assert store.note_row(db, result.run_id) is None


def test_a_note_cannot_be_attached_to_a_run_that_does_not_exist(db):
    with pytest.raises(store.Refusal) as error:
        store.save_note(db, "нет-такого", text="", available=False, source="none",
                        backend_requested="auto", model=None,
                        prompt_version="orientation/1.0", guard_passed=None,
                        guard_reason_ru="", guard_offending=(), reason_ru="",
                        bundle_sha256="", bundle_bytes=0)
    assert error.value.code == "unknown_run"


# ---------------------------------------------------------------------------
# Generating, offline.
# ---------------------------------------------------------------------------


def test_no_key_is_a_stored_absence_and_not_an_error(db, monkeypatch):
    """Rule: the note may never fail a build. No key is an ordinary outcome, it is written
    down with its reason, and the page then has something specific to say."""
    _no_keys(monkeypatch)
    result = _load(db, FULL)
    outcome = notes.generate_for_run(db, result.run_id)

    assert outcome.outcome == "no_provider"
    assert outcome.available is False
    assert "API_KEY" in outcome.reason_ru
    assert outcome.bundle_was_sent is False          # ...and nothing left the machine
    assert notes.stored_note(db, result.run_id).state == "absent_reason"


def test_deterministic_never_reaches_the_provider_even_with_a_key_present(db, monkeypatch):
    """`--backend deterministic` is a school's «нет». It has to mean no request, not a
    request whose answer is discarded, so the provider path is made to explode if touched."""
    monkeypatch.setenv("GEMINI_API_KEY", "не-настоящий-ключ")

    def explode(*_args, **_kwargs):
        raise AssertionError("deterministic must not call a provider")

    monkeypatch.setattr("classvision.report.orientation.write_note", explode)
    result = _load(db, FULL)
    outcome = notes.generate_for_run(db, result.run_id, backend="deterministic")

    assert outcome.outcome == "deterministic"
    assert outcome.bundle_was_sent is False
    assert "deterministic" in notes.stored_note(db, result.run_id).reason_ru


def test_a_failed_regeneration_never_erases_a_good_note(db, monkeypatch):
    """A cron on a machine whose key expired would otherwise replace every note in the
    cabinet with «ключа нет» — a regression caused by nothing changing. The numbers the old
    note describes have not moved, so the old note stands and the attempt is reported."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id, text="Хорошая записка про место 3.")

    _no_keys(monkeypatch)
    outcome = notes.generate_for_run(db, result.run_id)

    assert outcome.outcome == "kept_existing"
    assert outcome.available is True
    assert notes.stored_note(db, result.run_id).text == "Хорошая записка про место 3."
    assert any("прежняя" in note for note in outcome.notes_ru)


def test_all_means_the_runs_the_pages_actually_show(db, monkeypatch):
    """A re-analysis that is stored but not selected is rendered nowhere, so `--all` does not
    spend a call on it. Naming its run id still works."""
    _no_keys(monkeypatch)
    original = _load(db, CLIP)
    rerun = _load(db, CLIP_AGAIN)

    assert notes.selected_run_ids(db) == [original.run_id]
    assert [o.run_id for o in notes.generate_all(db)] == [original.run_id]
    assert notes.generate_for_run(db, rerun.run_id).run_id == rerun.run_id


# ---------------------------------------------------------------------------
# The page.
# ---------------------------------------------------------------------------


def _class_page(db, tmp_path) -> str:
    """The class overview, found through the render result rather than by file name.

    The page set is `cabinet/report.py`'s business and it grows; a test that hard-codes
    `class-camera01-3-б.html` fails the day a page is added next to it, and a reader then
    has to work out whether the notes broke or the file was renamed.
    """
    result = report.render(db, tmp_path / "html")
    page = next(p for p in result["pages"] if Path(p).name.startswith("class-"))
    return Path(page).read_text(encoding="utf-8")


def test_the_page_marks_the_note_as_machine_written_and_points_at_the_tables(db, tmp_path):
    """Three things must travel with a shown note, because without them a fluent Russian
    paragraph on a clinical page reads as a finding: who wrote it, that it contains no
    measurement by construction, and where the numbers are."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id, text="Стоит посмотреть на место 3.")
    page = _class_page(db, tmp_path)

    assert "Стоит посмотреть на место 3." in page
    assert "gemini-3.6-flash" in page
    assert "машинная записка" in page
    assert "Все числа — в таблицах ниже" in page
    # ...and the block really is above the tables it says are below it.
    assert page.index("Стоит посмотреть на место 3.") < page.index("<h2>Уроки</h2>")


def test_a_rejected_note_is_reported_as_rejected_rather_than_omitted(db, tmp_path):
    """Silently having no section is how a reader concludes the lesson was unremarkable. The
    page says the guard threw the text away, and shows what disqualified it."""
    result = _load(db, FULL)
    _store_a_note(db, result.run_id, text="", available=False, guard_passed=False,
                  offending=("42 минуты",), reason="записка отклонена")
    page = _class_page(db, tmp_path)

    assert "отклонена проверкой" in page
    assert "42 минуты" in page


def test_a_run_with_no_note_says_so_and_names_the_command(db, tmp_path):
    result = _load(db, FULL)
    page = _class_page(db, tmp_path)

    assert "не запрашивалась" in page
    assert result.run_id in page


def test_the_class_level_refusal_is_printed_where_such_a_note_would_be(db, tmp_path):
    """The decision not to write a cross-lesson note is on the page, with the reason and the
    place where a direction IS stated. A refusal nobody can read is a missing feature."""
    _load(db, FULL)
    page = _class_page(db, tmp_path)

    assert "по классу целиком" in page
    assert "Динамика" in page


def test_rendering_the_cabinet_never_calls_a_provider(db, tmp_path, monkeypatch):
    """The office machine may have no internet, and a page build that spent money per class
    would be a surprise nobody asked for. `cabinet note` is the only door to a provider."""
    def explode(*_args, **_kwargs):
        raise AssertionError("rendering must not call a provider")

    monkeypatch.setattr("classvision.report.orientation.write_note", explode)
    _load(db, FULL)
    assert report.render(db, tmp_path / "html")["state"] == "ok"


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def test_the_command_exits_zero_without_a_key(tmp_path, monkeypatch, capsys):
    """The exit code is the contract with a school's cron job: a missing key is not a
    failed build, and a red line in a log about an optional aid trains people to stop
    reading the log."""
    from classvision import cli

    _no_keys(monkeypatch)
    path = tmp_path / "cabinet.sqlite3"
    connection = store.connect(path)
    _load(connection, FULL)
    connection.close()

    assert cli.main(["cabinet", "note", "--all", "--db", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "НИЧЕГО НЕ ОТПРАВЛЕНО" in printed
    assert "API_KEY" in printed


def test_an_unknown_run_id_is_the_one_thing_that_fails(tmp_path, monkeypatch):
    """Nothing was attempted and nothing will be: a script that carried on would silently
    never write the note it was asked for."""
    from classvision import cli

    _no_keys(monkeypatch)
    path = tmp_path / "cabinet.sqlite3"
    store.connect(path).close()

    assert cli.main(["cabinet", "note", "нет-такого", "--db", str(path)]) == 2
    assert cli.main(["cabinet", "note", "--db", str(path)]) == 2          # neither
    assert cli.main(["cabinet", "note", "x", "--all", "--db", str(path)]) == 2   # both


# ---------------------------------------------------------------------------
# The prompt and the bundle are one contract, joined by two field names.
# ---------------------------------------------------------------------------


def test_every_bundle_field_the_rules_depend_on_is_named_in_the_prompt():
    """A string join, made reviewable the only cheap way there is.

    Rules 7 and 8 are instructions to read two specific fields by name. Rename a field in
    `compact_for_note` and the rules keep reading beautifully while pointing at nothing:
    the model gets a bundle with no `zero_but_briefly_seen`, quietly reverts to «не было»,
    and the guard — which only rejects digits — passes every word of it. That is the same
    class of failure as `COUNTER_VOIDED_BY` keying on the analyser's own sentence, and it
    gets the same kind of test.
    """
    small = orientation.compact_for_note(
        {"seats": [{"seat_id": 1, "coverage_percent": 80.0, "counts": {},
                    "zero_but_briefly_seen": ["stands"]}]})
    for field in ("zero_but_briefly_seen", "needs_visibility_caveat", "coverage_percent"):
        assert field in small["seats"][0], field
        assert field in orientation.SYSTEM_RU, f"rule text no longer names {field}"


def test_the_prompt_still_forbids_the_three_sentences_that_were_measured_false():
    """Each clause here was added because a real invocation produced the sentence it bans.

    Kept as a test rather than as a comment because a prompt is prose, and prose loses
    clauses to tidying. The three: a bare negation of a counter that is only zero for want
    of a long-enough episode; a spatial claim about the board on a camera that cannot see
    one; and a share of the lesson stated in words instead of read from the table.
    """
    assert "длительных" in orientation.SYSTEM_RU
    assert "доски" in orientation.SYSTEM_RU and "обычного положения" in orientation.SYSTEM_RU
    assert "Большую часть урока не видно" in orientation.SYSTEM_RU
    # ...and the whole point of the module: the rules themselves may not model bad output.
    assert "НИ ОДНОЙ ЦИФРЫ" in orientation.SYSTEM_RU


def test_the_visibility_threshold_is_a_named_constant_and_not_a_number_in_prose():
    """Rule 5 of this project. The prompt must not carry its own copy of the line."""
    assert orientation.VISIBILITY_CAVEAT_BELOW_PERCENT == 90.0
    seat = {"seat_id": 1, "coverage_percent": 90.0, "counts": {}}
    assert orientation.compact_for_note(
        {"seats": [seat]})["seats"][0]["needs_visibility_caveat"] is False
    seat["coverage_percent"] = 89.9
    assert orientation.compact_for_note(
        {"seats": [seat]})["seats"][0]["needs_visibility_caveat"] is True


def test_an_unknown_analysed_window_is_sent_as_unknown_not_as_the_files_length(db):
    """One name, two quantities — `MEASUREMENTS.md` §8's failure, in the length field.

    `runs.window_start_seconds` is nullable, and the old fallback published the RECORDING's
    duration under `lesson_minutes`, the name that everywhere else means the analysed window.
    They differ by 2.4 minutes on `full_lesson.analysis.json`, and every `coverage_percent`
    in the bundle is a share of the window — so the substitute silently changed what all of
    them were shares of, with nothing marking it as a substitute.
    """
    result = _load(db, FULL)
    db.execute("UPDATE runs SET window_start_seconds = NULL, window_end_seconds = NULL "
               "WHERE run_id = ?", (result.run_id,))

    bundle = notes.bundle_for_run(db, result.run_id)
    recording = store.lessons(db)[0]["duration_seconds"] / 60.0
    assert bundle["lesson"]["duration_minutes"] is None
    assert bundle["lesson"]["duration_minutes"] != round(recording, 1)
    assert any("окна урока" in item["what"] for item in bundle["unmeasured"])


def test_keeping_an_old_note_must_not_report_todays_traffic(db, monkeypatch):
    """«Отправлено N байт» printed by a run that sent nothing — the one sentence that must be true.

    `StoredNote.bundle_was_sent` reads `model`, correctly: a stored record's model is the
    model that wrote it. `NoteOutcome` copied that rule and asked a different question with
    it — on the `kept_existing` path `model` is the EXISTING note's model, so an attempt that
    never reached a provider reported the byte count as sent.

    The path is the one the class exists for: a nightly `cabinet note --all` on a machine
    whose key expired keeps yesterday's notes, and printed yesterday's traffic line as
    today's. This is the same defect as `MEASUREMENTS.md` §8 — one name, two questions.
    """
    result = _load(db, FULL)
    _store_a_note(db, result.run_id, text="Стоит посмотреть на место 3.")
    _no_keys(monkeypatch)

    outcome = notes.generate_for_run(db, result.run_id, backend="auto")
    assert outcome.outcome == "kept_existing"
    assert outcome.available and outcome.text                 # the good note survives
    assert outcome.model == "gemini-3.6-flash"                # ...and says who wrote it
    assert outcome.bundle_was_sent is False, "nothing left the machine on this attempt"

    # The operator's own refusal, over an existing note, is the same story.
    kept = notes.generate_for_run(db, result.run_id, backend="deterministic")
    assert kept.outcome == "kept_existing" and kept.bundle_was_sent is False


def test_a_guard_rejection_still_counts_as_sent(db, monkeypatch):
    """The other direction. `source` reads "none" after a rejection, but the request happened,
    and a school asking «что ушло наружу» must be told that it did."""
    result = _load(db, FULL)

    def rejected(bundle, *, backend="auto", timeout=60.0):
        from classvision.report import orientation as o
        return o.Note("", False, "none", model="gemini-3.6-flash",
                      check=o.Check(passed=False, offending=("42",)),
                      reason="записка отклонена")

    monkeypatch.setattr("classvision.report.orientation.write_note", rejected)
    outcome = notes.generate_for_run(db, result.run_id, backend="auto")
    assert outcome.outcome == "refused_by_guard"
    assert outcome.bundle_was_sent is True

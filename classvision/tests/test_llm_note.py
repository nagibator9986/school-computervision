"""The orientation note's guards, on hand-written model responses. No network.

Every test here feeds `validate()` a response a model might plausibly produce and asserts
exactly what happens to it. The attacks are the point: the module's whole claim is that a
number cannot reach the prose, and a claim like that is worth only as much as the attempts
made to break it.

The original defect these guard against is in `note.py`'s module docstring and in
`MEASUREMENTS.md` §8: a text that passed a numbers-exist check and was still false.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classvision.report import note as N

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "classvision" / "out" / "full_lesson.analysis.json"

needs_artefact = pytest.mark.skipif(
    not FULL.exists(), reason="run `classvision analyse` first")


@pytest.fixture(scope="module")
def bundle() -> dict:
    from classvision.report.summary import compact
    return compact(FULL)


def response(**overrides) -> dict:
    base = {
        "opening": "Урок прошёл спокойно, класс оставался на своих местах.",
        "highlights": [
            {"subject": "seat_4", "metric_key": "head_up_share",
             "why_ru": "У этого места голова была опущена значительную часть урока."},
        ],
        "closing": "Остальные показатели в отчёте ниже.",
    }
    base.update(overrides)
    return base


# -- the happy path --------------------------------------------------------------------

@needs_artefact
def test_a_clean_response_renders_with_our_own_number(bundle: dict):
    result = N.validate(response(), bundle)
    assert result.ok, result.reason
    assert len(result.highlights) == 1
    highlight = result.highlights[0]
    # The value came from the bundle, not from the model.
    expected = highlight.metric.value_of(
        next(s for s in bundle["seats"] if s["seat_id"] == 4))
    assert highlight.value == pytest.approx(expected)
    # ...and it is printed in the text the model never wrote.
    assert "Место 4" in result.text
    assert "доля времени с поднятой головой" in result.text


@needs_artefact
def test_the_number_is_printed_beside_its_own_description(bundle: dict):
    """The residual risk is a qualitatively wrong claim. It is bounded by keeping the value
    in the same sentence as the model's words about it, so a mismatch is visible."""
    result = N.validate(response(), bundle)
    line = next(line for line in result.text.splitlines() if "Место 4" in line)
    assert "%" in line or "из" in line
    assert "голова была опущена" in line


# -- digits, in every script a model might reach for -----------------------------------

@pytest.mark.parametrize("smuggle", [
    "Голова опущена 42 минуты.",              # plain ASCII
    "Опущена ٤٢ минуты.",                     # Arabic-Indic
    "Опущена ４２ минуты.",                    # fullwidth
    "Доля составила ½ урока.",                # vulgar fraction
])
@needs_artefact
def test_any_digit_in_prose_is_rejected(bundle: dict, smuggle: str):
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "head_up_share", "why_ru": smuggle}]), bundle)
    assert not result.ok
    assert any("цифра" in d.get("why", "") or "число словами" in d.get("why", "")
               for d in result.dropped), result.dropped


@pytest.mark.parametrize("smuggle", [
    "Голова опущена девяносто шесть процентов времени.",
    "Опущена почти всё время урока.",
    "Опущена половину урока.",
    "Вставал дважды за урок.",
    "Опущена сорок минут.",
])
@needs_artefact
def test_quantities_spelled_as_words_are_rejected(bundle: dict, smuggle: str):
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "head_up_share", "why_ru": smuggle}]), bundle)
    assert not result.ok, f"smuggled quantity survived: {smuggle!r}"


@needs_artefact
def test_a_digit_in_the_opening_is_fatal_not_droppable(bundle: dict):
    """`opening` frames the whole note. A number there poisons it, and there is nothing
    left worth salvaging — unlike one bad highlight among six."""
    result = N.validate(response(opening="В классе было 8 учеников."), bundle)
    assert not result.ok
    assert "вступлении" in result.reason


# -- unknown keys and subjects ---------------------------------------------------------

@needs_artefact
def test_an_unknown_metric_key_is_dropped_and_counted(bundle: dict):
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "sample_fps", "why_ru": "Служебное поле."},
        {"subject": "seat_4", "metric_key": "head_up_share", "why_ru": "Голова опущена."}]),
        bundle)
    assert result.ok
    assert len(result.highlights) == 1
    assert any(d.get("why") == "показатель не в списке" for d in result.dropped)


@needs_artefact
def test_an_unknown_seat_is_dropped(bundle: dict):
    result = N.validate(response(highlights=[
        {"subject": "seat_99", "metric_key": "head_up_share", "why_ru": "Нет такого места."},
        {"subject": "seat_4", "metric_key": "head_up_share", "why_ru": "Голова опущена."}]),
        bundle)
    assert result.ok
    assert any(d.get("why") == "неизвестный объект" for d in result.dropped)


@needs_artefact
def test_a_teacher_metric_asked_of_a_seat_is_dropped(bundle: dict):
    """`at_desk_percent` exists in the bundle, but only for the adult. Pointing it at a
    pupil's seat is exactly the cross-subject misattribution the old design allowed."""
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "at_desk_percent", "why_ru": "Много сидел."}]),
        bundle)
    assert not result.ok
    assert any(d.get("why") == "показатель не в списке" for d in result.dropped)


@needs_artefact
def test_a_pupil_metric_asked_of_the_teacher_is_dropped(bundle: dict):
    result = N.validate(response(highlights=[
        {"subject": "teacher", "metric_key": "head_up_share", "why_ru": "Голова опущена."}]),
        bundle)
    assert not result.ok


@needs_artefact
def test_the_value_always_comes_from_the_named_subject(bundle: dict):
    """Two seats, same metric, different values — proving the lookup is per subject and a
    model cannot make one seat's number appear under another's name."""
    pairs = []
    for seat_id in (4, 9):
        result = N.validate(response(highlights=[
            {"subject": f"seat_{seat_id}", "metric_key": "head_up_share",
             "why_ru": "Смотрим на долю времени с поднятой головой."}]), bundle)
        assert result.ok
        pairs.append((seat_id, result.highlights[0].value))
    assert pairs[0][1] != pairs[1][1], "both seats returned the same value"
    for seat_id, value in pairs:
        seat = next(s for s in bundle["seats"] if s["seat_id"] == seat_id)
        assert value == pytest.approx(N.SEAT_KEYS["head_up_share"].value_of(seat))


# -- bounds ----------------------------------------------------------------------------

@needs_artefact
def test_highlights_are_capped(bundle: dict):
    many = [{"subject": "seat_4", "metric_key": "head_up_share", "why_ru": "Причина."}
            for _ in range(N.MAX_HIGHLIGHTS + 4)]
    result = N.validate(response(highlights=many), bundle)
    assert result.ok
    assert len(result.highlights) == N.MAX_HIGHLIGHTS
    assert sum(1 for d in result.dropped if d.get("why") == "сверх лимита") == 4


@needs_artefact
def test_an_over_long_field_is_rejected(bundle: dict):
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "head_up_share",
         "why_ru": "о" * (N.MAX_FIELD_CHARS + 1)}]), bundle)
    assert not result.ok
    assert any(d.get("why") == "слишком длинно" for d in result.dropped)


@needs_artefact
def test_an_empty_response_refuses_rather_than_rendering_nothing(bundle: dict):
    result = N.validate({"opening": "", "highlights": [], "closing": ""}, bundle)
    assert not result.ok


# -- the codebase's own rules ----------------------------------------------------------

@needs_artefact
def test_forbidden_vocabulary_is_caught_by_the_shared_checker(bundle: dict):
    """`note.py` does not keep its own copy of the forbidden-word list — `summary.py` owns
    it. This asserts the shared checker sees note text, so the two cannot drift."""
    from classvision.report.summary import find_forbidden_words

    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "head_up_share",
         "why_ru": "Ученик не вовлечён в урок и не старается."}]), bundle)
    if result.ok:
        assert find_forbidden_words(result.text), (
            "«вовлечён» reached the rendered note and the shared checker did not see it")


def test_the_system_prompt_states_the_no_numbers_rule_unambiguously():
    assert "НИ ОДНОГО ЧИСЛА" in N.SYSTEM_RU
    assert "Ни цифрами, ни словами" in N.SYSTEM_RU
    # ...and it must not itself contain a digit, or the model reasonably infers digits are
    # acceptable. The one place a count appears, it is spelled out.
    assert not N.contains_digits(N.SYSTEM_RU)


def test_the_instruction_limit_is_spelled_not_written_as_a_digit():
    assert not N.contains_digits(N._spell(N.MAX_HIGHLIGHTS))


def test_the_allowlists_have_no_overlapping_keys_with_different_meanings():
    """A key meaning one thing for a pupil and another for the adult is the naming trap
    that produced the original defect. Overlap is allowed only if the label agrees."""
    for key in set(N.SEAT_KEYS) & set(N.TEACHER_KEYS):
        assert N.SEAT_KEYS[key].label_ru == N.TEACHER_KEYS[key].label_ru, key


@needs_artefact
def test_every_allowlisted_seat_metric_resolves_on_a_real_seat(bundle: dict):
    """An allowlist entry that never resolves is a key the model will be told it may use
    and then have dropped for. Every one must work on at least one real seat."""
    unresolved = [
        metric.key for metric in N.SEAT_METRICS
        if all(metric.value_of(seat) is None for seat in bundle["seats"])
    ]
    assert not unresolved, f"allowlisted but never resolvable: {unresolved}"


@needs_artefact
def test_every_allowlisted_teacher_metric_resolves(bundle: dict):
    teacher = bundle.get("teacher")
    if not teacher or not teacher.get("available"):
        pytest.skip("no adult in this artefact")
    unresolved = [m.key for m in N.TEACHER_METRICS if m.value_of(teacher) is None]
    assert not unresolved, f"allowlisted but never resolvable: {unresolved}"


def test_the_response_schema_is_valid_json():
    json.dumps(N.RESPONSE_SCHEMA)
    assert N.RESPONSE_SCHEMA["required"] == ["opening", "highlights", "closing"]


@needs_artefact
def test_a_missing_value_is_unknown_and_never_silently_zero(bundle: dict):
    """The defect this pins: `value_of` read the wrong key name with a `0.0` default, so
    every share resolved to zero and was rendered as a fact about a child. It passed the
    "does this metric ever resolve" test because 0.0 is not None."""
    seat = next(s for s in bundle["seats"] if s["seat_id"] == 4)
    shares = [N.SEAT_KEYS[key].value_of(seat)
              for key in ("head_up_share", "facing_front_share", "at_place_share")]
    assert not all(value == 0.0 for value in shares), (
        "every share came back as zero — the lookup is reading a key that does not exist")

    stripped = dict(seat)
    stripped["activity"] = {"available": True, "index": None, "parts": []}
    assert N.SEAT_KEYS["head_up_share"].value_of(stripped) is None, (
        "a missing part must be UNKNOWN, not zero")


@needs_artefact
def test_rendered_shares_match_the_artefact_exactly(bundle: dict):
    """End to end: the percentage printed in the note is the percentage in the artefact,
    not a rescaled or defaulted version of it."""
    seat = next(s for s in bundle["seats"] if s["seat_id"] == 4)
    part = next(p for p in seat["activity"]["parts"] if p["key"] == "head_up_share")
    result = N.validate(response(highlights=[
        {"subject": "seat_4", "metric_key": "head_up_share",
         "why_ru": "Голова была опущена заметную долю урока."}]), bundle)
    assert result.ok, result.reason
    assert result.highlights[0].value == pytest.approx(part["value_percent"])

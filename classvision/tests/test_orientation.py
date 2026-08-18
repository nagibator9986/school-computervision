"""The orientation note's guard, which is the whole reason the note is allowed to exist.

The guard is pure and needs no API key, so all of this runs offline. The LLM call itself is
not tested here — a test that needs a paid provider is a test that gets skipped forever.
What IS pinned is the contract the provider's output has to satisfy before a human sees it.
"""

from __future__ import annotations

import pytest

from classvision.report import orientation

BUNDLE = {"seats": [{"seat_id": n} for n in (2, 3, 4, 5, 7, 8, 9)]}


def passes(text: str) -> bool:
    return orientation.check(text, BUNDLE).passed


# -- what must get through -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Стоит посмотреть на место 4.",
    "Руку поднимали на месте 5 и месте 8.",
    "Внимание на местах 4, 7 и 9.",
    "По месту 4 и месту 2 картина иная.",
    "На одном из мест ребёнок часто лежал на парте.",
    "Место 9 — ребёнок просидел спокойно; обзор был полным.",
])
def test_a_clean_note_is_not_rejected(text: str):
    """Rejecting good notes is not a safe failure: it teaches the reader to ignore the
    guard, and then the guard protects nobody. The enumeration forms are here because the
    first version of the pattern matched only a single reference and threw away a
    perfectly clean note over the «и 3» in «места 8 и 3»."""
    assert passes(text)


# -- what must not -----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "На месте 4 голова опущена 42 минуты.",
    "Это 85% урока.",
    "Индекс 52,5 из 100.",
    "Посмотрите на место 42.",
    "Места 4 и 42 отличаются.",
    "Ребёнок вставал 7 раз.",
])
def test_any_quantity_is_refused(text: str):
    assert not passes(text)


def test_the_measured_failure_this_module_exists_for():
    """THE REGRESSION. Gemini, asked to summarise the full artefact, wrote «Сидячее
    положение зафиксировано в 96,5 % случаев (6,0 минуты / 357,5 секунды)» — joining a
    share of observations to a total of qualifying episodes as though they were one fact.
    Both numbers were real and present in the source, so the existence-checker passed them.
    A note that may contain no quantity at all cannot produce that sentence."""
    sentence = "Сидячее положение зафиксировано в 96,5 % случаев (6,0 минуты / 357,5 секунды)."
    assert not passes(sentence)


def test_an_unknown_seat_number_is_a_quantity_not_a_label():
    """`место 42` in a room with seven seats is not an identifier — it is a number the
    model produced from somewhere, which is precisely the thing being excluded."""
    check = orientation.check("Посмотрите на место 42.", BUNDLE)
    assert not check.passed
    assert "42" in " ".join(check.offending)


def test_a_mixed_reference_is_refused_whole():
    """«места 4 и 42» is not a seat list with one stray quantity; it is a sentence whose
    meaning we cannot vouch for. Salvaging the half that parses would be guessing."""
    assert not passes("Места 4 и 42 отличаются.")


# -- the compaction ----------------------------------------------------------------------

def test_the_note_bundle_carries_coverage_under_the_name_the_bundle_uses():
    """A missing key reads as `None`, and `None` reads to the model as "cannot tell" — so
    the first version of `compact_for_note`, which read `coverage` instead of
    `coverage_percent`, silently cost every note its most useful sentence. The model was
    right to hedge; the null was ours."""
    seat = {"seat_id": 4, "coverage_percent": 99.6, "counts": {"hand_raises": 1},
            "activity": {"index": 52.5}}
    small = orientation.compact_for_note({"seats": [seat]})
    assert small["seats"][0]["coverage_percent"] == 99.6
    assert small["seats"][0]["activity_index"] == 52.5


def test_seat_labels_come_from_the_bundle_not_from_the_text():
    assert orientation.seat_labels(BUNDLE) == (2, 3, 4, 5, 7, 8, 9)


def test_no_provider_is_a_stated_outcome_not_an_exception(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    note = orientation.write_note(BUNDLE, backend="auto")
    assert note.available is False
    assert note.source == "none"
    assert "API_KEY" in note.reason

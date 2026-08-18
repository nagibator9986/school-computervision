"""The artefact's field names, pinned — because an ambiguous name shipped a false sentence.

**The defect this file exists for.** The ledger used to emit one key per state called
`seconds`, holding time inside *qualifying episodes*, while the adult's metrics carried
`seated_share` holding a share of *observations*. Asked to summarise the artefact, an LLM
wrote «Сидячее положение зафиксировано в 96,5% случаев (6,0 минуты / 357,5 секунды)».
96.5 % of 48 observed minutes is 46.3 minutes, not 6.0 — it had joined two different
quantities that shared a name. Both numbers existed in the document, so the guard that
checks every number in generated text against the source passed them: **it verifies that a
number exists, not that it is attached to the right claim.**

The repair was to put the unit and the denominator into the names. These tests make that
repair permanent. They are deliberately about NAMES and ARITHMETIC rather than about
values, so they keep working when the footage changes.

A second instance of the same trap was found while verifying the first: the adult's
aggregate summed four states and was called `seated`, while one of those four states is
itself called `seated` (46.3 minutes against 8.6). It is now `at_desk`. Both are pinned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classvision.report.artefact import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "classvision" / "out" / "full_lesson.analysis.json"

needs_artefact = pytest.mark.skipif(
    not FULL.exists(), reason="run `classvision analyse` first")


@pytest.fixture(scope="module")
def artefact() -> dict:
    return json.loads(FULL.read_text(encoding="utf-8"))


def test_schema_version_was_bumped_for_the_rename():
    """A consumer written against 1.0 must break loudly, not read the wrong key quietly.
    That is the entire reason the old name was removed rather than kept as an alias."""
    assert SCHEMA_VERSION == "classvision/1.1"


# -- the ledger: two times per state, and they may never share a name -------------------

@needs_artefact
def test_the_ambiguous_seconds_key_is_gone_everywhere(artefact: dict):
    for seat in artefact["seats"]:
        assert "seconds" not in seat["ledger"], (
            f"seat {seat['seat_id']} still carries a bare `seconds` key — that name does "
            "not say whether it counts observations or qualifying episodes")
    if artefact.get("teacher") and artefact["teacher"].get("ledger"):
        assert "seconds" not in artefact["teacher"]["ledger"]


@needs_artefact
def test_both_per_state_times_are_present_and_distinctly_named(artefact: dict):
    for seat in artefact["seats"]:
        ledger = seat["ledger"]
        assert "observed_seconds_by_state" in ledger
        assert "episode_seconds" in ledger


@needs_artefact
def test_observed_seconds_by_state_sums_to_the_observed_total(artefact: dict):
    """This is the arithmetic that identifies which of the two quantities you are holding.
    The observation-based one accounts for ALL the time the pupil was seen; the
    episode-based one cannot, because short runs are deliberately discarded."""
    for seat in artefact["seats"]:
        ledger = seat["ledger"]
        total = sum(ledger["observed_seconds_by_state"].values())
        assert total == pytest.approx(ledger["observed_seconds"], abs=1.0), (
            f"seat {seat['seat_id']}: per-state observed seconds sum to {total} but the "
            f"ledger reports {ledger['observed_seconds']} observed seconds")


@needs_artefact
def test_episode_seconds_never_exceeds_observed_seconds_for_a_state(artefact: dict):
    """Episodes are a SUBSET of observations by construction. If this ever inverts, the
    two quantities have been swapped somewhere — which is the original defect, returning."""
    for seat in artefact["seats"]:
        ledger = seat["ledger"]
        for state, episode in ledger["episode_seconds"].items():
            observed = ledger["observed_seconds_by_state"].get(state, 0.0)
            assert episode <= observed + 1.0, (
                f"seat {seat['seat_id']} state {state}: episode_seconds={episode} exceeds "
                f"observed_seconds_by_state={observed}")


# -- the adult: every share names its denominator, and the minutes are pre-computed -----

@needs_artefact
def test_adult_shares_carry_their_denominator_in_the_name(artefact: dict):
    teacher = artefact.get("teacher")
    if not teacher or not teacher["metrics"].get("available"):
        pytest.skip("no adult identified in this artefact")
    metrics = teacher["metrics"]
    for old in ("seated_share", "standing_or_away_share", "out_of_frame_share",
                "seated_minutes", "longest_seated_minutes"):
        assert old not in metrics, (
            f"`{old}` is back. Either it does not say what its denominator is, or it "
            "reuses the name of a single state for a four-state aggregate.")
    for new in ("at_desk_share_of_observed", "standing_or_away_share_of_observed",
                "out_of_frame_share_of_lesson", "at_desk_minutes", "observed_minutes",
                "longest_at_desk_episode_minutes"):
        assert new in metrics, f"`{new}` is missing"


@needs_artefact
def test_the_exact_sentence_the_llm_got_wrong_is_now_arithmetically_checkable(artefact: dict):
    """«96,5 % of the observed time» and «46,3 minutes» must be the same fact, and the
    artefact must state both so that no consumer has to multiply — or, worse, reach for
    the episode figure because it was the only "seated seconds" on offer."""
    teacher = artefact.get("teacher")
    if not teacher or not teacher["metrics"].get("available"):
        pytest.skip("no adult identified in this artefact")
    metrics = teacher["metrics"]
    derived = metrics["observed_minutes"] * metrics["at_desk_share_of_observed"] / 100.0
    assert derived == pytest.approx(metrics["at_desk_minutes"], abs=0.15), (
        f"at_desk_minutes={metrics['at_desk_minutes']} does not equal "
        f"observed_minutes x at_desk_share_of_observed = {derived:.1f}")


@needs_artefact
def test_the_adults_four_state_aggregate_is_not_confusable_with_the_seated_state(
        artefact: dict):
    """The aggregate is larger than the single state it used to share a name with. This
    test would have failed on the old naming and is the point of the `at_desk` rename."""
    teacher = artefact.get("teacher")
    if not teacher or not teacher["metrics"].get("available"):
        pytest.skip("no adult identified in this artefact")
    aggregate_minutes = teacher["metrics"]["at_desk_minutes"]
    single_state_minutes = teacher["ledger"]["observed_seconds_by_state"]["seated"] / 60.0
    assert aggregate_minutes > single_state_minutes, (
        "the four-state aggregate should exceed the single SEATED state; if they are "
        "equal, one of them is being computed from the other's definition")


# -- no field anywhere may be a bare share or a bare duration --------------------------

@needs_artefact
def test_the_report_states_the_right_pair_so_nobody_has_to_derive_it(artefact: dict):
    """The deterministic text must print the share AND its duration in one phrase.

    This is the end of the chain that started with the false LLM sentence. Renaming the
    fields stopped the two quantities being confusable; printing the correct pair means
    no reader — human or model — has to multiply anything to get the number they want.

    It also guards a regression this rename caused and this test caught: the sentence was
    silently DROPPED for one run, because the `if` had been updated to the new key while
    the body still read the old one. A missing sentence is invisible in review; a failing
    test is not.
    """
    from classvision.report.summary import compact, render

    text = render(compact(FULL))
    teacher = artefact.get("teacher")
    if not teacher or not teacher["metrics"].get("available"):
        pytest.skip("no adult identified in this artefact")

    share = teacher["metrics"]["at_desk_share_of_observed"]
    minutes = teacher["metrics"]["at_desk_minutes"]
    share_ru = f"{share:.1f}".replace(".", ",")
    minutes_ru = f"{minutes:.1f}".replace(".", ",")

    assert "за столом" in text, "the adult's position sentence is missing entirely"
    assert share_ru in text, f"the share {share_ru} % is not printed"
    assert minutes_ru in text, (
        f"the duration {minutes_ru} minutes is not printed — a reader is being left to "
        "multiply, which is exactly how the wrong duration got picked last time")
    # ...and the two must appear in the same phrase, not paragraphs apart.
    window = text[max(text.find("за столом") - 20, 0): text.find("за столом") + 160]
    assert share_ru in window and minutes_ru in window, (
        "the share and its duration are printed but not together")


AMBIGUOUS = ("seconds", "minutes", "share", "percent")


@needs_artefact
def test_no_metric_field_is_a_bare_unit_word(artefact: dict):
    """A field called exactly `share` or `minutes` cannot say what of. Nested containers
    whose KEY supplies the unit (`observed_seconds_by_state`) are fine; a leaf is not."""
    offenders: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in AMBIGUOUS and not isinstance(value, dict):
                    offenders.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(artefact.get("teacher") or {}, "teacher")
    for seat in artefact["seats"]:
        walk(seat["metrics"], f"seat{seat['seat_id']}.metrics")
    assert not offenders, f"bare unit words as leaf fields: {offenders}"

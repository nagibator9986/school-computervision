"""No page may tell a person to do something this system cannot do.

**THIS FILE EXISTS BECAUSE THE SAME MISTAKE WAS MADE THREE TIMES IN THREE DAYS, TWICE WHILE
FIXING IT.** The sequence, in order:

  1. `schools.html` told the operator the second school "can be removed from this page".
     There is no delete anywhere. Caught while writing it, removed.
  2. The camera refusal told the reader to go and look at `/schools`. Every role that can
     see that refusal gets 403 there. Caught by review, fixed.
  3. **The fix for (2) said "убрать её может тоже только он"** -- promising removal again,
     in different words, inside the change that removed the previous promise.

Three instances is not bad luck, it is a class: prose about what the system can do, written
in a template, where nothing checks it against the system. Fixing the third instance would
leave the class open, so this is aimed at the class.

**HOW IT WORKS.** Removal may be *discussed* -- a page saying "удалить нельзя" is telling
the truth and is exactly what these pages should say. What may not happen is a removal
mentioned in the POSITIVE while no removal exists. So a claim that mentions a school and a
removal verb must carry a negation NEAR that verb.

**WHAT THIS IS AND IS NOT, STATED EXACTLY, BECAUSE THE FIRST VERSION OVERSOLD ITSELF.** It
said any positive rephrasing "lacks a negation and fails", and offered «снимет её» as an
example -- which it did not catch, because `снять` is not a substring of «снимет». Three
sabotages of the real template walked straight past it.

Honestly: **one family, a list by verb, a rule by polarity, and the rule has a known hole.**

  * ONE FAMILY. Only taking a school away. Any other false promise -- about cameras, about
    accounts -- is not covered by anything here.
  * A LIST BY VERB. `REMOVAL_WORDS` is stems, and a stem not in it is invisible. Three were
    missing. There is no rule making the fourth one caught.
  * A RULE BY POLARITY. This part does generalise: a promise phrased any way at all fails
    unless a negation sits near the verb, so it is not a list of phrasings.
  * A KNOWN HOLE. The polarity rule is a 40-character window, which is a judgement. A
    negation that lands inside the window while belonging to something else still disarms
    it -- «школу удалить может админ, но не сегодня» passes. Narrowed from "anywhere in the
    sentence", not closed.

The whole check switches off the day the capability arrives: if `qorgan.schools` grows a
delete, the templates are free to say so, and this file says which line to delete.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import qorgan.schools as schools_module
from tests.conftest import REPO_ROOT

TEMPLATES = REPO_ROOT / "src" / "qorgan" / "web" / "templates"

# Jinja comments carry the REASONING about these claims -- including sentences quoting the
# false promises on purpose -- so they are stripped before the prose is judged. A comment is
# not shown to anybody.
JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)

# Jinja statements and expressions are stripped too, and for a reason worth recording: the
# schools table contains `{% for school in schools %}`, so leaving it in put the word
# "школ" into a chunk of pure markup and made the scanner report the table as a promise.
# Only what a reader can see is prose.
JINJA_CODE = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)

# Block-level tags end a claim; the rest are unwrapped and dropped. Without this the whole
# table was one chunk, and a chunk that large matches every word list ever written.
BLOCK_TAG = re.compile(
    r"</?(?:li|p|h[1-6]|tr|td|th|div|section|ul|ol|table|thead|tbody|form|label|button)\b[^>]*>",
    re.IGNORECASE,
)
ANY_TAG = re.compile(r"<[^>]+>")

SCHOOL_WORDS = ("школ",)

# Russian stems for taking a school away. Stems, not words, so declensions are covered.
# **A LIST BY VERB. Not a rule by verb -- and the difference is worth being exact about,
# because an earlier version of this file claimed more than it delivered.** Polarity is a
# RULE (see `_negated_near`); the verbs are a LIST, and a list only covers the stems in it.
# Three real sabotages walked past the first version:
#
#   * «суперадминистратор снимет её» -- and this was the file's OWN docstring example of
#     something that "lacks a negation and fails". `снять`/`снима` are not substrings of
#     «снимет». Now `сним`/`сня`.
#   * «суперадминистратор уберёт её» -- the perfective of «убрать», the most natural next
#     phrasing. No stem covered it. Now `убер`.
#   * «убрать её может только суперадминистратор установки, а не вы.» -- covered by the
#     stems, but disarmed by an unrelated «не » elsewhere in the sentence. That is the
#     polarity hole, fixed in `_negated_near` rather than here.
#
# Stems, not whole words, so declensions come free -- but matched at a WORD START, which is
# not decoration. As bare substrings `сня` matched inside «объяснят» ("will explain") and
# flagged a true sentence on the schools page; the collision class is real and this is what
# removes it. `сним` still collides with «снимок» (a snapshot), which does begin with the
# stem: measured today, no template says both «снимок» and «школ», so it costs nothing. If
# one ever does, narrow that entry to verb forms rather than deleting it.
#
# `отключ` is deliberately absent. The schools table renders «отключена» as a STATUS --
# `School.is_active`, which the register displays and no route changes -- and a page showing
# a status is not a page offering an action. Including it made the status label read as a
# promise. If a page ever offers to disable a school, that is a new verb and a new entry.
REMOVAL_WORDS = (
    "удал",
    "удаля",
    "убрать",
    "убрал",
    "убира",
    "убер",
    "снять",
    "снима",
    "сним",
    "сня",
)

# How close a negation has to sit to the verb before it counts as negating it.
#
# The first version accepted a negation ANYWHERE in the claim, so «убрать её может только
# суперадминистратор установки, а не вы» read as negated: the «не» belongs to «вы», forty
# characters away, and the promise stood. Sentences on these pages are one clause of promise
# and one of contrast, and the whole rhetoric of the page is "not you, him" -- so a rule that
# any «не » disarms is a rule that disarms itself here.
#
# 40 characters is a JUDGEMENT, not a measurement: wide enough for «удалить её нельзя» and
# «но не удаляются», narrow enough to exclude the trailing «а не вы». It is the weakest part
# of this check and it is stated as such rather than presented as a boundary.
NEGATION_WINDOW = 40

# A removal may be mentioned only in the negative. These are the ways Russian says "no" in
# this position; a positive claim carries none of them.
NEGATIONS = ("не ", "нет ", "нельзя", "никто", "никак", "невозмож", "без ")

# What would have to exist in `qorgan.schools` for a removal promise to be TRUE. Checked by
# name against the module, so the check retires itself the day somebody adds one.
REMOVAL_FUNCTIONS = ("delete_school", "remove_school", "archive_school", "deactivate_school")


def _removal_exists() -> list[str]:
    return sorted(name for name in REMOVAL_FUNCTIONS if hasattr(schools_module, name))


# Where one claim ends and the next begins. **NOT the newline** -- and that distinction is
# the whole reason this constant exists. The first version of this scanner read the template
# line by line, so a sentence wrapped across two lines was two half-sentences to it, and
# neither half carried both the subject and the verb. The check written to close a class of
# defect had that very class in it, and it was found by sabotaging it and watching it stay
# green. Templates wrap at 100 characters; claims do not.
CLAIM_BOUNDARIES = (". ", "; ")

WHITESPACE = re.compile(r"\s+")


def _prose(path: Path) -> str:
    """Everything on the page a human could read, and nothing else.

    Comments go because nobody is shown one -- and they deliberately quote the false
    promises. Jinja code goes because `{% for school in schools %}` is not a sentence about
    schools. Block tags become claim boundaries; the remaining tags are unwrapped so that
    `<strong>удалить нельзя</strong>` reads as one phrase rather than three.
    """
    text = JINJA_COMMENT.sub(" ", path.read_text(encoding="utf-8"))
    text = JINJA_CODE.sub(" ", text)
    text = BLOCK_TAG.sub("\u2029", text)
    return ANY_TAG.sub("", text)


def _claims(path: Path) -> list[str]:
    """The page as whitespace-normalised claims, each one whole.

    Line numbers are gone deliberately: a claim spans lines, so reporting one line of it
    would point at half a sentence. The claim itself is quoted in the failure instead,
    which is what somebody needs in order to find and fix it.
    """
    text = _prose(path)
    for boundary in CLAIM_BOUNDARIES:
        text = text.replace(boundary, "\u2029")
    return [
        WHITESPACE.sub(" ", chunk).strip()
        for chunk in text.split("\u2029")
        if WHITESPACE.sub(" ", chunk).strip()
    ]


def _negated_near(text: str, at: int, length: int) -> bool:
    """Is there a negation close enough to the verb at `at` to be negating IT?

    Proximity rather than presence-anywhere. See `NEGATION_WINDOW` for what that fixed and
    for the honest limits of the number.
    """
    window = text[max(0, at - NEGATION_WINDOW) : at + length + NEGATION_WINDOW]
    return any(word in window for word in NEGATIONS)


def _verb_pattern(verbs: tuple[str, ...]) -> re.Pattern[str]:
    r"""Stems, anchored at a word start. `\b` is what keeps a stem out of another word."""
    return re.compile(r"\b(?:" + "|".join(re.escape(v) for v in verbs) + r")", re.IGNORECASE)


def _says_it_positively(claim: str, verbs: tuple[str, ...]) -> bool:
    """Does this claim use one of `verbs` about a school, un-negated?

    Every occurrence is judged separately, and ONE un-negated occurrence is enough: a claim
    that negates its second verb does not thereby excuse its first, which is the shape of
    «удалить можно, но не сегодня».
    """
    low = claim.lower()
    if not any(word in low for word in SCHOOL_WORDS):
        return False

    return any(
        not _negated_near(low, match.start(), len(match.group()))
        for match in _verb_pattern(verbs).finditer(low)
    )


def _sentences_promising_removal(path: Path) -> list[str]:
    """Claims that mention a school being taken away, without saying it cannot be."""
    return [
        f"{path.name}: {claim}"
        for claim in _claims(path)
        if _says_it_positively(claim, REMOVAL_WORDS)
    ]


def _templates() -> list[Path]:
    return sorted(TEMPLATES.rglob("*.html"))


def test_there_are_templates_to_check() -> None:
    """A scan that found no files would pass everything below by vacuity."""
    assert len(_templates()) > 5, f"only found {len(_templates())} templates; wrong path?"


@pytest.mark.parametrize("template", [p.name for p in _templates()])
def test_no_page_offers_to_remove_a_school(template: str) -> None:
    """The class-closer. One parameter per template, so the failure names the file.

    If this goes red and a delete really has been built, delete this test rather than
    softening it -- and `_removal_exists` is what will tell you which is the case.
    """
    if _removal_exists():
        pytest.skip(
            f"qorgan.schools now has {_removal_exists()}, so a page MAY offer removal. "
            "Delete this test and the two 'удалить нельзя' sentences it protects."
        )

    path = next(p for p in _templates() if p.name == template)
    promises = _sentences_promising_removal(path)

    assert not promises, (
        "this page tells a person a school can be taken away, and nothing in this system "
        "can take one away -- `qorgan.schools` has list_schools, create_school and "
        "rename_school, and no delete:\n\n" + "\n".join(promises) + "\n\n"
        "Say it in the negative if it needs saying at all. This is the THIRD time this "
        "promise has been written, twice while removing the previous one, which is why it "
        "is checked instead of remembered."
    )


def test_the_scanner_would_catch_the_sentence_that_got_through() -> None:
    """The check, checked. A scanner nobody has seen bite is a scanner nobody has tested.

    This is the exact sentence that reached `main`'s review inside the fix for the previous
    instance. It must be caught, and its truthful negation must not be.
    """
    caught = _sentences_promising_removal_in_text(
        "Если вторая школа заведена по ошибке, убрать её может тоже только он."
    )
    assert caught, "the scanner does not catch the wording that actually got through"

    allowed = _sentences_promising_removal_in_text(
        "Удалить её нельзя: школы создаются и переименовываются, но не удаляются."
    )
    assert not allowed, (
        "the scanner rejects a truthful sentence saying removal is impossible, which is "
        "exactly what these pages SHOULD say. It would push the next author into silence."
    )


def _sentences_promising_removal_in_text(line: str) -> list[str]:
    """The same rule, applied to one claim. Shares `_says_it_positively` with the scanner
    so the self-test cannot pass against a rule the scanner does not actually use."""
    return [line] if _says_it_positively(line, REMOVAL_WORDS) else []


# The other half of the same class: promising a SCOPE the system does not have. `/logs` is
# installation-wide by construction -- `diagnostics.logfiles.recent(category, page,
# page_size)` has no school dimension at all and its own docstring says "across every
# process's log stream". Only the undelivered-alerts panel on that page is per-school.
PER_SCHOOL_CLAIM = "только вашу школу"
JOURNAL_WORDS = ("журнал", "логи", "/logs")


def test_no_page_lists_the_log_journal_among_the_per_school_surfaces() -> None:
    """The refusal listed `Журналы` first among things that "show only your school".

    VIEW_DIAGNOSTICS is held by admin and developer -- the same people who read that
    refusal -- so the page was promising them a scope the journal has never had. The
    parallel list on `schools.html` omitted journals deliberately; the two pages had
    drifted, and the one that was rewritten over-promised.

    Retires itself: if `recent()` ever takes a school, the claim becomes true and this test
    says so instead of failing.
    """
    from inspect import signature

    from qorgan.diagnostics.logfiles import recent

    scoped = [p for p in signature(recent).parameters if "school" in p or "tenant" in p]
    if scoped:
        pytest.skip(
            f"logfiles.recent now takes {scoped}, so the journal IS per-school. The pages "
            "may say so, and this test should go."
        )

    offenders = [
        f"{path.name}: {claim}"
        for path in _templates()
        for claim in _claims(path)
        if PER_SCHOOL_CLAIM in claim.lower()
        and any(word in claim.lower() for word in JOURNAL_WORDS)
    ]

    assert not offenders, (
        "a page claims the log journal shows only the reader's school. It does not -- "
        "`logfiles.recent` has no school parameter and reads every process's stream. Only "
        "the undelivered-alerts panel on /logs is scoped:\n\n" + "\n".join(offenders)
    )


def test_the_scanner_reads_a_claim_that_wraps_across_lines(tmp_path: Path) -> None:
    """**The defect this scanner had, pinned so it cannot come back.**

    The first version read the template line by line. Templates wrap at 100 characters, so a
    claim spanning two lines was two half-claims -- neither carrying both the subject and
    the verb -- and the sabotage that put the promise back split over two lines stayed
    green. The check written to close a class of defect contained that class.

    Written to a real file rather than asserted against a string, because the bug was in the
    file reading and not in the matching.
    """
    page = tmp_path / "wrapped.html"
    page.write_text(
        "<ul>\n  <li>\n    Если вторая школа заведена по ошибке, то\n"
        "    убрать её тоже может суперадминистратор.\n  </li>\n</ul>\n",
        encoding="utf-8",
    )

    assert _sentences_promising_removal(page), (
        "a promise split across two lines was not seen. The scanner is line-based again, "
        "and every wrapped claim in every template is invisible to it."
    )


def test_the_scanner_does_not_read_markup_or_jinja_as_prose(tmp_path: Path) -> None:
    """The other false direction: `{% for school in schools %}` is not a sentence.

    Leaving Jinja in put "школ" into chunks of pure markup, and the schools table -- one
    enormous chunk with every word in it -- was reported as a promise. A scanner that cries
    wolf on its own templates gets its word list trimmed until it stops, which is how it
    ends up unable to see anything.
    """
    page = tmp_path / "markup.html"
    page.write_text(
        "<table>\n{% for school in schools %}\n"
        "  <tr><td>{{ school.slug }}</td><td>отключена</td></tr>\n"
        "{% endfor %}\n</table>\n",
        encoding="utf-8",
    )

    assert not _sentences_promising_removal(page), (
        "the scanner read template machinery as a claim about schools. It will be quietened "
        "by deleting verbs, and then it will not catch the real thing."
    )


# Every phrasing that has ever walked past this scanner, kept as a permanent regression set.
# Three of these were found by review AFTER the scanner was written and declared robust, and
# one of them was this file's own docstring example of something it claimed to catch. They
# are here so the next widening of the stem list cannot quietly narrow the coverage back.
BYPASSES_FOUND_BY_REVIEW = (
    # `снять`/`снима` are not substrings of «снимет». This exact phrasing was offered in the
    # docstring as an example of what fails, and it passed.
    "Если вторая школа заведена по ошибке, суперадминистратор снимет её.",
    # The perfective of «убрать» -- the most natural next wording, covered by no stem.
    "Если вторая школа заведена по ошибке, суперадминистратор уберёт её.",
    # Stems matched; an unrelated «не вы» forty characters away disarmed the polarity rule.
    "Если вторая школа заведена по ошибке, убрать её может только "
    "суперадминистратор установки, а не вы.",
    # The original third instance, for completeness.
    "Если вторая школа заведена по ошибке, убрать её может тоже только он.",
)

# Sentences that must stay ALLOWED. A scanner tightened until it catches everything catches
# the truth too, and then it gets deleted by whoever has to write the next honest page.
TRUTHFUL_SENTENCES = (
    "Если вторая школа заведена по ошибке — удалить её нельзя: на этой установке школы "
    "создаются и переименовываются, но не удаляются.",
    "Удалить школу нельзя.",
    # Real prose from schools.html. «объяснят» contains the letters of the `сня` stem, which
    # is why stems are anchored at word starts.
    "Как только школ станет больше одной, страницы камер объяснят почему.",
)


@pytest.mark.parametrize("phrasing", BYPASSES_FOUND_BY_REVIEW)
def test_every_phrasing_that_once_slipped_through_is_caught(phrasing: str) -> None:
    """The regression set. Each of these was green once, on the real template."""
    assert _sentences_promising_removal_in_text(phrasing), (
        "this phrasing promises a school can be taken away and the scanner did not see it. "
        "It is in the regression set because it slipped through once already:\n\n"
        f"  {phrasing}\n\n"
        "Check REMOVAL_WORDS for a missing stem and _negated_near for the polarity rule."
    )


@pytest.mark.parametrize("phrasing", TRUTHFUL_SENTENCES)
def test_a_truthful_sentence_is_not_flagged(phrasing: str) -> None:
    """The other direction, which matters as much.

    A scanner that flags «удалить нельзя» makes the honest sentence unwritable, and the next
    author deletes the scanner rather than the page. Both directions or neither.
    """
    assert not _sentences_promising_removal_in_text(phrasing), (
        f"the scanner flagged a TRUE sentence:\n\n  {phrasing}\n\n"
        "It says removal is impossible, which is exactly what these pages should say."
    )

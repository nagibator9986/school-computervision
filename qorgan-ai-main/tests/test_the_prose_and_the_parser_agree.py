"""Prose about the CLI, checked against the CLI that is actually built.

The scanner is `tests/prose_scan.py`; this file is the argument for it and the tests.

**THIS FILE EXISTS BECAUSE THE SAME MISTAKE WAS MADE FOUR TIMES ON ONE BRANCH, IN BOTH
DIRECTIONS.** In order, each one measured:

  1. `accounts.py` said the installation's own account was made by
     `qorgan user add --superadmin`. That flag existed nowhere in the tree; the command
     offered `--role superadmin`, took it from argparse, then refused it as "unknown role".
  2. `db/models/auth.py` named a test, `test_only_a_superadmin_belongs_to_no_school`, that
     existed nowhere either. (Not a CLI claim -- see WHAT THIS DOES NOT COVER.)
  3. There is no `qorgan school add` command, and `schools.py`'s own docstring called it a
     second front door onto the same table, in the present tense.
  4. **The inversion, and the reason this checks both directions.**
     `web/templates/schools.html` told the superadmin, in Russian, that the flag for naming
     a school did not exist yet -- on the page that role now lands on, about a flag the
     same commit had shipped and `qorgan user add --help` was advertising.

     **That sentence is quoted verbatim in `FALSE_DENIALS_THAT_SHIPPED` below and NOT here.**
     A constant is code; a docstring is prose, and this scanner reads prose -- written out
     above it would itself be a denial of a live flag, and this file reddened on its own
     docstring the first time it ran. Recorded, not exempted: an exemption for the checker is
     a hole in the check. The cost, named: prose here cannot reproduce a false claim word for
     word and must paraphrase, or move it into a constant -- which is where the sister check
     keeps its own corpus of bad wordings too.

Four instances is not bad luck, it is a class: **prose that names a command or a flag,
written where nothing compares it with the parser.** Fixing the fourth would leave the class
open, so this is aimed at the class.

**HOW IT WORKS.** A CLI claim in this tree is written in a code span -- backticks in Python
prose, `<code>` in a template. That anchor is what makes this narrow enough to be worth
having: a sentence beginning with a command name is not a claim about one. Every span is
resolved against `cli.build_parser()` itself, never a list written here, which would be one
more thing to drift. Then polarity decides, exactly as
`test_the_pages_promise_only_what_exists.py` decides whether a removal may be mentioned:

  |                 | named POSITIVELY          | named NEGATIVELY                |
  |-----------------|---------------------------|---------------------------------|
  | parser HAS it   | fine                      | RED -- instance 4               |
  | parser LACKS it | RED -- instances 1, 3     | fine (an honest "there is none")|

The bottom-right cell is not a courtesy. Instance 3's honest repair is a sentence saying
there is no `qorgan school add` command, so a check reddening on that would leave its own
finding unfixable: the negative case had to be allowed before the positive one could be
enforced.

**EVERY RED HAS AN ANSWER A MAINTAINER CAN GIVE**, which is what decides whether a rule
survives the next person: a phantom named positively -> build it, or say it does not exist;
a real thing denied -> delete the denial, one word or one sentence. Neither answer is
"reword something unrelated" and neither is "there is nothing you can do".

**WHAT THIS DOES NOT COVER, DELIBERATELY.** Every figure below is measured with the rules as
they stand, and the first version of this list got two of them wrong -- the paragraph that
justifies a check's blind spots is the last place to estimate.

  * **Markdown -- not `docs/`, `README.md`, `HANDOVER.md` or `HANDOFF.md`.** Measured:
    `docs/` yields **three** reds, all under `docs/superpowers/`, all phantoms -- two naming
    a namesake-checking `pupils` subcommand a design plan proposed and nobody built, one
    naming an `eval` shorthand written with a pipe. (An earlier draft said six. Six was the
    number of *occurrences* of the command name, not of reds, and the miscount is recorded
    because this paragraph is the justification for the blind spot.) A plan that names the
    command it proposes is not defective; that is what a plan IS. The only answers available
    there are "it is a plan" and "it is dated", and a rule answered that way gets deleted
    wholesale -- which is how the sister branch lost two of them. Telling a runbook step from
    a proposal means judging intent, and a check that judges intent argues instead of
    failing. **The cost, stated:** a phantom in `README.md` is invisible here.
  * **Python STRING LITERALS -- and this is the sharpest omission, so it is second.** Only
    docstrings and comments are read. Measured: **106 sites in `src/` mention a CLI token
    inside a string literal.** Most are `add_argument("--role", ...)` declarations, which is
    why literals are not simply switched on -- the same extraction would read the parser's
    own definition as a claim about itself. But some are neither: `cli.py`'s refusal text
    prints `qorgan user add … --school <slug>` to a person at a terminal, and its `--help`
    text explains what the superadmin role is. **That is the same user-facing genre as
    instance 4**, and nothing checks it. Closing this needs a rule that separates a
    declaration from a message, which is a design, not a switch.
  * **`migrations/**/*.py`.** Nine files, none under `SRC_DIR`, so none scanned. Migration
    0009 writes the default school and its prose is read by nobody here.
  * **Prose that is not a code span.** Written without the markup, a claim is invisible.
    Measured: dropping the anchor turned 1 red into 50+, every one ordinary English
    following a command name.
  * **Anything that is not a command or a flag.** Instance 2 was a phantom *test name*.
    Function names, file paths and module names are checked by nothing here.
  * **Positionals and flag VALUES.** `qorgan user add alice --role wizard` passes: `alice`
    is a positional, `wizard` a value, and neither is resolved.
    `test_every_role_the_command_offers_can_actually_be_created` covers `--role`'s values.
  * **Whether a true claim is USEFUL.** This asks only whether the parser agrees.
"""

from __future__ import annotations

import pytest

from tests.prose_scan import (
    EVERY_FLAG,
    LEG_FLOORS,
    NODES,
    SOURCES,
    code_spans,
    false_denials,
    flatten,
    is_real,
    phantoms,
    report,
    scan,
)

# -- Vacuity. "It scanned nothing" must FAIL, not pass. ------------------------


def test_the_parser_offers_something_to_check() -> None:
    """A parser walked down to nothing makes both directions below vacuous.

    Direction B in particular is silent when `EVERY_FLAG` is empty: no span is real, so
    nothing can be falsely denied, and the check goes green having inspected an empty set.
    That is this project's signature defect and a guard on the sister branch was found doing
    exactly it, so the parser's own size is asserted before anything is read.
    """
    assert len(NODES) > 20, f"the parser walk found {len(NODES)} command paths"
    assert len(EVERY_FLAG) > 10, f"the parser walk found {len(EVERY_FLAG)} flags"
    assert ("user", "add") in NODES, "qorgan user add is not in the walked tree any more"
    assert "--school" in NODES[("user", "add")][0], "`qorgan user add` lost --school"


def test_every_leg_of_the_scan_is_still_wired_up() -> None:
    """The legs must all still BE there; the test below asks whether they are read.

    Two different failures, and the second cannot catch the first: a leg dropped from
    `sources()` has no parameter left to fail.
    """
    assert {leg for leg, _, _ in SOURCES} == set(LEG_FLOORS), (
        f"the scan covers {sorted({leg for leg, _, _ in SOURCES})} and LEG_FLOORS names "
        f"{sorted(LEG_FLOORS)}. A leg was added or dropped from `sources()`; if dropped, "
        "say so under WHAT THIS DOES NOT COVER before deleting it here."
    )


@pytest.mark.parametrize("leg", sorted(LEG_FLOORS))
def test_each_leg_of_the_scan_is_actually_read(leg: str) -> None:
    """**Asserted PER LEG, never over the total, and that distinction IS the test.**

    The first version totalled the three legs. Templates contribute 2 of 95 `qorgan` spans
    and 3 of 125 real spans, so src and tests cleared every threshold on their own -- and a
    one-character typo in the templates path dropped 18 files with no error and all fifteen
    tests green. Measured end to end: **templates dark, plus instance 4 re-planted verbatim,
    was 15/15 green.** The check could go blind on the exact leg it was built for.

    A union is not a guard, it is a subsidy: the big legs pay for the small one's silence.
    """
    min_files, min_inv, min_real, must_have = LEG_FLOORS[leg]
    items = [(path, pieces) for tag, path, pieces in SOURCES if tag == leg]
    spans = [s for _, pieces in items for piece in pieces for s, _, _ in code_spans(piece)]
    invocations = [s for s in spans if s.split()[:1] == ["qorgan"]]
    real = [s for s in spans if is_real(s)]

    assert len(items) >= min_files, (
        f"the `{leg}` leg found {len(items)} files, under its floor of {min_files}. A wrong "
        "path rglobs to nothing and raises NOTHING; this is what says so."
    )
    assert any(path.name == must_have for path, _ in items), (
        f"`{must_have}` is not among the {len(items)} files of the `{leg}` leg. That file "
        "carries the claims this check was written for."
    )
    assert len(invocations) >= min_inv, (
        f"the `{leg}` leg yielded {len(invocations)} `qorgan` invocations (floor {min_inv}). "
        "The files are being found and their prose is not being read."
    )
    assert len(real) >= min_real, (
        f"the `{leg}` leg yielded {len(real)} spans naming a real command or flag "
        f"(floor {min_real}); nothing in it can be caught denying one."
    )


def test_the_named_anchors_are_still_found() -> None:
    """Claims known to be in the tree. If they stop being found, the reader broke."""
    spans = [s for _, _, pieces in SOURCES for piece in pieces for s, _, _ in code_spans(piece)]
    assert any(s.startswith("qorgan user add") for s in spans), (
        "no `qorgan user add ...` span was found anywhere. The whole branch is about that "
        "command; if it is not being read, nothing below is being checked either."
    )
    assert "--school" in [s for s in spans if is_real(s)], (
        "`--school` is never read as a real flag; extraction broke"
    )


# -- The class-closers ---------------------------------------------------------


def test_no_prose_names_a_command_or_flag_that_does_not_exist() -> None:
    """Direction 1: instances 1 and 3. The phantom.

    Whole-tree rather than one parameter per file: 300-odd files would be 600 ids to say
    what one list says, and one failure naming every offender is what somebody fixing this
    actually wants. Per-LEG coverage is asserted above, which is the part a total hid.
    """
    found = scan(phantoms)
    assert not found, (
        "prose names a command or flag the built parser does not offer. Either build it, "
        "or say it does not exist -- a phantom named in the NEGATIVE is allowed on purpose, "
        "so writing that there is no such command is a legitimate fix and needs no exemption "
        "here:\n\n" + report(found)
    )


def test_no_prose_denies_a_command_or_flag_that_exists() -> None:
    """Direction 2: instance 4. The inversion, which is the one nobody looks for.

    A user-facing page said a flag did not exist on the day the flag shipped, and it sat on
    the landing page of the only role that reads it. Nothing caught it because prose about
    an ABSENT thing reads as harmless -- the whole branch had been watching for the other
    direction.
    """
    found = scan(false_denials)
    assert not found, (
        "prose says a command or flag does not exist, and the built parser offers it. The "
        "remedy is to delete the denial -- one word or one sentence. Check "
        "`qorgan <command> --help` before deciding this test is wrong:\n\n" + report(found)
    )


# -- The check, checked. A scanner nobody has seen bite is one nobody has tested.

# The real wording of three of the four instances, as they were actually committed.
PHANTOMS_THAT_SHIPPED = (
    # Instance 1, `accounts.py`, verbatim.
    "The role that manages schools is created by the installation itself "
    "(`qorgan user add --superadmin`), never through a form served to a tenant.",
    # Instance 3, `schools.py`, verbatim.
    "`qorgan school add` is a second front door onto the same table, and a rule written "
    "in a route is a rule the CLI does not have.",
    # Not committed, but the shape a subcommand typo takes.
    "Run `qorgan user create alice` at a shell on the server.",
)

FALSE_DENIALS_THAT_SHIPPED = (
    # Instance 4, `schools.html`, verbatim.
    "Флага <code>--school</code> пока нет, поэтому ни одной камере ни одной школы "
    "нельзя будет назначить рабочий процесс.",
    # The `db/models/school.py` wording the same review found stale on arrival: the
    # worked example of a denial that PRECEDES what it denies.
    "Stable, human-typable, and what a future `--school` flag takes.",
)

# Sentences that must stay ALLOWED. A scanner tightened until it catches everything catches
# the truth too, and then it is deleted by whoever has to write the next honest page.
TRUTHFUL_CLAIMS = (
    # The honest repair for instance 3 -- and the reason the negative case is allowed at
    # all. If this were caught, this file's own finding could not be fixed.
    "The /schools page is the only front door: there is no `qorgan school add` command.",
    # Instance 1's epitaph, which the tree carries today.
    "The flag this comment used to name (`--superadmin`) existed nowhere in the tree.",
    # `rename_school`, verbatim: the forward-looking word belongs to the URL, not the flag.
    # This is the sentence a both-ways denial window reddened, with no answer available.
    "It is what a `--school` flag, a log line and a future URL carry.",
    # A denial of something that really is absent. `--all` is not in the parser.
    "Both ids, named in full, by a human. There is no --all and no winner-picking.",
    # Ordinary use, with a positional and a value the parser never resolves.
    "Run `qorgan user add alice --role admin --school gymnasium-4` at a shell.",
    # A command name followed by ordinary English, which is not a claim about a flag.
    "`qorgan supervisor` is what actually runs the fleet, and `qorgan web` serves it.",
    # The unicode ellipsis is a placeholder. It was not, and the scanner read it as a
    # subcommand and reported this tree's own prose as a phantom.
    "The `qorgan …` spans are counted per leg.",
)


@pytest.mark.parametrize("prose", PHANTOMS_THAT_SHIPPED)
def test_a_phantom_is_caught(prose: str) -> None:
    """Each of these was committed, reviewed, and green."""
    assert phantoms(flatten(prose)), (
        f"this names a command or flag that does not exist, and it was not seen:\n\n  "
        f"{prose}\n\nCheck `unmet_claim` and the code-span anchor."
    )


@pytest.mark.parametrize("prose", FALSE_DENIALS_THAT_SHIPPED)
def test_a_false_denial_is_caught(prose: str) -> None:
    """The inversion, pinned separately: this is the direction that shipped unnoticed."""
    assert false_denials(flatten(prose)), (
        f"this denies a flag the parser offers, and it was not seen:\n\n  {prose}\n\n"
        "Check DENIALS_BEFORE / DENIALS_AFTER and CLI_NOUNS."
    )


@pytest.mark.parametrize("prose", TRUTHFUL_CLAIMS)
def test_a_truthful_claim_is_not_flagged(prose: str) -> None:
    """Both directions or neither.

    A check that reddens on honest prose is one whose every red is answered by rewording
    something unrelated, and that is the check the next maintainer deletes rather than
    satisfies. Two rules on the sister branch had to be rescoped for exactly this.
    """
    flat = flatten(prose)
    complaints = phantoms(flat) + false_denials(flat)
    assert not complaints, f"the scanner flagged a TRUE sentence:\n\n  {prose}\n\n{complaints}"


def test_an_escape_does_not_blind_the_rest_of_the_text() -> None:
    """**The defect this scanner had, pinned so it cannot come back.**

    Backtick pairing is sequential, so a markdown escape -- a run of two or more ticks --
    shifted every pair after it. Flattened whole-file, that blinded the rest of the FILE.
    Measured when found: this module saw zero `--school` spans in itself, and six real reds
    across two files were sitting behind the desync, one of them introduced by the very
    round that went looking for it.

    Two independent fixes, and this asserts the one a future refactor is likeliest to undo:
    runs are collapsed. The other is that prose is scanned piece by piece, so a stray tick
    can at worst blind its own comment. Both, because either alone leaves a way back in.
    """
    escaped = flatten("An escape `` `qorgan web` `` and then a claim: `qorgan school add`.")
    assert phantoms(escaped), (
        "a markdown escape earlier in the text hid a phantom after it. Pairing has "
        "desynchronised again: check BACKTICK_RUN in tests/prose_scan.py."
    )

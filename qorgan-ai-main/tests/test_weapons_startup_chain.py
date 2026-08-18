"""Nothing on the way from "the weights refused" to "the worker keeps running" may catch it.

Split out of `tests/test_weapons_refusal.py`, which asserts BEHAVIOUR -- that each of the
five ways the model can be absent raises. These tests assert STRUCTURE, by reading the
source with `ast`, and they exist because every behavioural test in that file could pass
while something upstream swallowed the exception and started a motion detector instead.
That is not a hypothetical: it is what the client's previous system did for months.

**This file also guards the guard.** The structural check used to be able to pass while
looking at nothing, which is this project's signature defect -- an empty state satisfying
a negative assertion -- living inside the check written to prevent exactly that. It was
demonstrated, on a scratch copy of `weapons/model.py`, by running the guard's own
predicate over four versions of the same code:

    as it stands                                      1 handler inspected  -> []  (can go red)
    the try/except moved to a sibling METHOD          0 handlers inspected -> []   GREEN
      ... the same move, but the handler SWALLOWS
          instead of re-raising (the forbidden
          motion-analysis fallback)                   0 handlers inspected -> []   GREEN
      ... the same swallowing code, with the helper
          NAMED in STARTUP_CHAIN                      1 handler inspected  -> RED, named

So `test_no_step_of_worker_startup_swallows_the_refusal` alone is worth only as much as
`STARTUP_CHAIN` is complete, and nothing made it stay complete. Splitting a 50-line
function -- which R1 forces sooner or later -- was enough to blind it silently. The tests
beside it close that:

  * every name in `STARTUP_CHAIN` must still resolve to a `def` in its file, so a rename,
    a move, or a conversion to `async def` fails LOUDLY instead of quietly covering less;
  * every `except` in a `STARTUP_CHAIN` file must sit inside a step the guard inspects, so
    a handler cannot be parked one function to the left of where anybody is looking;
  * a call to a named step must itself be inside a named step, so a step of construction
    cannot quietly become a loader anybody can call twice;
  * and, because all four of those read ONE file against ONE entry and so cannot see a
    caller in another module at all, no module under `src/` may call a step whose name is
    private to a different one. That last shape was the widest hole here and the only one
    nothing watched -- disclosed in prose for three rounds, then claimed to be covered by a
    pin whose fixture is a single parsed string and cannot hold a second module.

They are deliberately stricter than "weapons": any handler in these three files is on the
path a refusal takes out of the worker, so if a new one appears the right outcome is a
human deciding whether it re-raises, not a guard that shrinks to fit. Three ways the
JUDGING half used to wave a handler through are closed with them, each documented where it
lives in `tests/weapons_chain_reader.py`: no exception-type whitelist (the note where
`CATCHES_IT` used to be), a `raise` that only sometimes runs (`_judge_handler`), and a
swallow written as `contextlib.suppress`, which has no `except` for any of this to read
(`_Site`).

**Every red raised here has a move that satisfies it.** That is a design constraint, not a
courtesy: a rule that fires and cannot be answered is removed by whoever it blocks, and it
takes the rule that catches the real fallback with it.

The READING is next door and asserts nothing: `tests/weapons_chain_reader.py` for the four
rules that read one chain file against one entry, and `tests/weapons_reach_in_reader.py` for
the fifth, which reads every module under `src/` at once and carries the source it is pinned
against. Both were split off when this file, and then that one, reached the 500-line cap; the
cap is not the thing that was wrong. The rules stayed here because they are the argument, the
parsing went there because it is the machinery.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import SRC_DIR
from tests.weapons_chain_reader import (
    MODULE_SCOPE,
    STARTUP_CHAIN,
    _chain_file_from,
    _read_chain_file,
)
from tests.weapons_reach_in_reader import (
    MODEL_PYS_CROSS_MODULE_CLAIM,
    REACH_INS_ACROSS_MODULES,
    THE_REACH_INS_REPORTED,
    _private_steps,
    _reaching_into_another_module,
    _read_src_trees,
)


@pytest.mark.parametrize("relative", sorted(STARTUP_CHAIN))
def test_no_step_of_worker_startup_swallows_the_refusal(relative: str) -> None:
    """A degraded mode cannot be added here without this going red.

    `run_group` DOES catch `Exception` around `_serve` -- and re-raises after recording a
    crashed heartbeat, which is the behaviour we want and why the rule is "does not
    re-raise" rather than "does not catch".

    Four things this rule does NOT do, each of them a way it used to pass over a live
    fallback (see `_Site`, `_judge_handler` and the no-whitelist note, all in
    `tests/weapons_chain_reader.py`):

      * it does not check the exception TYPE. A whitelist cannot be complete here, because
        `_load_or_refuse` re-raises whatever torch threw;
      * it does not accept a `raise` that only sometimes runs. `if strict: raise` is a
        fallback with a flag on it;
      * it does not accept a `raise` that some path never reaches. The mirror spelling --
        `if not strict: warn; return None` above a `raise` -- is the same fallback written
        as a guard clause, and the `continue` version of it is the client's own defect at
        per-camera granularity;
      * it does not look only for `except`. `contextlib.suppress` is a catch site the
        repo's own linter will offer to write for you, and it can never re-raise.
    """
    offenders = _read_chain_file(relative).swallowing
    assert not offenders, (
        f"{relative} catches the weights refusal and carries on: {list(offenders)}. "
        "That is the motion-analysis fallback the client already had. Every catch site in "
        "these three files must re-raise on EVERY path out of it, whatever type it names -- "
        "a `raise` under an `if` counts only when every branch raises; a `return`, "
        "`continue`, `break` or `yield` disqualifies it at any depth, though NOT inside a "
        "`def` or `class` written in the handler, where it does not exit the handler at all; "
        "and a `contextlib.suppress` never counts, because it cannot re-raise. Where a red "
        "names the statement that disqualified it, that statement is the one to move."
    )


@pytest.mark.parametrize("relative", sorted(STARTUP_CHAIN))
def test_every_step_named_in_the_startup_chain_still_exists(relative: str) -> None:
    """**The test above passes while inspecting zero functions. This is why it may not.**

    A `STARTUP_CHAIN` name that matches nothing is not an absence of danger, it is an
    absence of looking -- and the guard cannot tell those apart, which is the same defect
    this whole module exists to refuse. So a rename, a move to another file, or a
    conversion to `async def` fails HERE, naming the step, on the commit that does it,
    instead of leaving a green suite covering one function less than it did yesterday.

    An entry that names no FUNCTION is the same defect with no name to report, so it is
    refused outright, and it has two spellings. `()` satisfies `missing` vacuously, because
    `missing` is computed over the names in the entry. `(MODULE_SCOPE,)` is worse, because
    it looks deliberate: it is non-empty, `missing` skips the sentinel by design so it
    cannot go red, and on a file with no module-level catch site -- `builders.py`, which has
    no `except` at all -- all four rules then come back empty while not one name has
    resolved to anything.

    So the sentinel is an ADDITION to a chain, never a chain by itself. A file is on the
    startup path because a FUNCTION in it is; `MODULE_SCOPE` exists only to let an
    import-time handler in such a file be judged rather than be an unanswerable red. An
    entry that names only the scope is naming the exception without the rule.
    """
    functions = STARTUP_CHAIN[relative]
    assert [name for name in functions if name != MODULE_SCOPE], (
        f"the STARTUP_CHAIN entry for {relative} names no function at all (it is "
        f"{list(functions)}). An entry that inspects nothing is not an absence of danger, "
        f"it is an absence of looking, and {MODULE_SCOPE!r} on its own inspects nothing on "
        "a file with no module-level handler. Name the functions the refusal travels "
        "through, or drop the key."
    )
    missing = _read_chain_file(relative).missing
    assert not missing, (
        f"STARTUP_CHAIN names {list(missing)} in {relative} and there is no such `def` "
        "there. The swallow guard is not inspecting them -- it walks straight past and "
        "passes. Either they were renamed or moved (say so here), or one became `async "
        "def`, which is `ast.AsyncFunctionDef` and is not matched at all."
    )


@pytest.mark.parametrize("relative", sorted(STARTUP_CHAIN))
def test_no_handler_in_a_startup_chain_file_is_out_of_the_guards_reach(relative: str) -> None:
    """The other half: a handler moved one function to the left of where anybody is looking.

    R1 caps a function at 50 lines, so `YoloWeaponModel.__init__` gets split sooner or
    later and the `try` around ultralytics travels into a new method with it. Name the new
    method in `STARTUP_CHAIN` and the swallow guard follows it; forget, and the guard
    inspects zero handlers and stays green over one that swallows the refusal. Naming a
    method fixes one instance of that; this fixes the shape of it, for every split nobody
    has done yet.

    Stricter than "handlers about weapons" on purpose. These three files ARE the path a
    refusal takes out of the worker, so a new `except` in one of them is a decision
    somebody has to make out loud -- add the function to `STARTUP_CHAIN` and let the
    re-raise rule judge it -- rather than a line that quietly lands outside the guard.

    **Every red this raises has a move that satisfies it, and that is load-bearing.** A
    rule that fires occasionally and costs ten seconds to answer survives; a rule that
    fires and cannot be answered gets deleted, and the rule that catches the real fallback
    goes with it. So a handler outside every `def` -- an import-time `try` at module level
    or in a class body, which has no function name to add -- is answered by naming
    `MODULE_SCOPE` in the entry, after which the re-raise rule judges it like any other. If
    it is an import guard that cannot honestly re-raise (`except ImportError: _CUDA =
    False`), the other answer is that it does not belong in a startup-chain file at all;
    both exits are named in the message, because an undisclosed exit is the same as none.
    """
    unreached = _read_chain_file(relative).unreached
    assert not unreached, (
        f"{relative} has catch site(s) no scope in STARTUP_CHAIN contains: "
        f"{list(unreached)}. The guard never reads them, so a swallowed refusal there is "
        "invisible. Name the enclosing function in STARTUP_CHAIN; for one outside every "
        f"`def`, name {MODULE_SCOPE!r} -- or move it out of this file, which is the honest "
        "answer for an import guard that cannot re-raise. An `async def` cannot be named "
        "(it is not matched), so that handler must move or the guard must learn "
        "AsyncFunctionDef."
    )


@pytest.mark.parametrize("relative", sorted(STARTUP_CHAIN))
def test_a_named_step_is_only_ever_called_from_inside_the_chain(relative: str) -> None:
    """A step of startup must not quietly become a seam anybody can call.

    The other two rules watch what happens to the refusal on its way OUT. This one watches
    the doorway: splitting `__init__` for R1 produced `_load_or_refuse`, a `@staticmethod`
    that takes a path and returns a loaded model, and `model.py:86-89` promises it is a
    step of construction rather than a loader. Nothing checked that promise, and
    `builders.py:68-97` already ships the lazy-loader template for the pose model that
    somebody will copy.

    So: a call to a named step must itself sit inside a named step. The chain is closed
    under calling within its own file, which is what "there is no `load()` to forget"
    means when written as a rule instead of a sentence. A `reload()` that calls
    `_load_or_refuse` again fails here, naming `reload`, before it can leave `self.weights`
    describing a file that is no longer running.

    Known limit, stated rather than implied: this sees `ast.Call` nodes. Passing a step
    around as a bare reference -- `callback=self._load_or_refuse` -- is not a call and is
    not caught.
    """
    escaped = _read_chain_file(relative).escaped_calls
    assert not escaped, (
        f"{relative} calls a STARTUP_CHAIN step from outside the chain: {list(escaped)}. "
        "A step of startup called from anywhere else is a load step somebody can forget, "
        "and the refusal it performs is not re-performed by the caller. Either the caller "
        "belongs in STARTUP_CHAIN, or the step should not be reachable from it."
    )


# The clause `model.py`'s class docstring uses to describe the rule above, lifted verbatim.
# It is the MECHANISM, not the intention, and the two are not the same thing here: of the five
# calls to a step in the fixture below, four are green and exactly one -- `reload`, labelled
# SHAPE 1 there -- is reported. The SHAPE numbers are the fixture's own; they do NOT line up
# with the bullets in `model.py`, whose list starts with the cross-module shape that no
# single-module fixture can hold at all.
_MODEL_PYS_CLAIM = (
    "a call to either step that is not lexically inside a `def` this file names as a step"
)

# The whole answer, not the label. This used to compare `entry.split(":")[0]`, so a rule that
# reported the wrong callee or the wrong line under the right label passed -- and the same
# lesson was applied forward to `THE_REACH_INS_REPORTED` while this one kept the loose form.
_THE_ONE_IN_FILE_CALL_REPORTED = ("reload: line 19 -> _load_or_refuse()",)

# FIVE calls to `_load_or_refuse`: the legitimate one in `YoloWeaponModel.__init__`, plus one
# per shape. Exactly ONE is reported. Measured, so the assertion below is not vacuous -- and
# the edit that measures it is specifically deleting the in-step SKIP (`id(node) in in_steps`
# in `_calls_from_outside_the_chain`), after which the same fixture reports all five, so four
# are being actively EXCLUDED rather than absent. Emptying the entry instead reports NONE, and
# proves nothing: a name that is not a step matches nothing, so there is nothing to exclude.
_FOUR_IN_FILE_CALLERS = '''
class YoloWeaponModel:
    def __init__(self, path):
        self._model = self._load_or_refuse(path)          # the legitimate one

        def later():                                      # SHAPE 3: closure inside __init__
            return self._load_or_refuse(path)

        self._later = later

    @staticmethod
    def _load_or_refuse(path):
        return path

    def _say_what_loaded(self, path):
        return self._load_or_refuse(path)                 # SHAPE 4: step calls step

    def reload(self, path):
        self._model = self._load_or_refuse(path)          # SHAPE 1: reported

class WeaponModelPool:
    def __init__(self, path):                             # SHAPE 2: __init__ matches by NAME
        self.model = YoloWeaponModel._load_or_refuse(path)
'''


def _assert_yolo_weapon_models_docstring_states(clause: str) -> None:
    """Fail unless `YoloWeaponModel`'s CLASS DOCSTRING contains `clause` verbatim.

    The docstring, not the file. Reading `model.py` whole -- which this used to do, under a
    variable called `docstring` -- lets the pinned clause be moved out of the class docstring
    into any comment anywhere in the module while the class docstring goes on to say
    something else entirely, and the pin stays green over exactly the drift it exists to
    stop. Whitespace is normalised first, so the clause may be re-wrapped freely; it may not
    be re-worded.
    """
    tree = ast.parse((SRC_DIR / "qorgan" / "weapons" / "model.py").read_text("utf-8"))
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "YoloWeaponModel"
    ]
    assert len(classes) == 1, (
        f"expected exactly one `class YoloWeaponModel` in model.py, found {len(classes)}. "
        "This test pins a sentence in its docstring and cannot tell which one you meant."
    )
    docstring = " ".join((ast.get_docstring(classes[0]) or "").split())
    assert clause in docstring, (
        "YoloWeaponModel's class docstring no longer states this rule in the words pinned "
        f"here:\n\n    {clause}\n\n"
        "Either restore it, or change it AND the expectation in this test together, having "
        "first measured what the rule now does. Do not describe it from memory -- three "
        "drafts of that paragraph have been wrong, each claiming more reach than the code has."
    )


def test_model_pys_account_of_the_call_rule_is_still_true() -> None:
    """**`model.py`'s description of the rule above, pinned to the rule.**

    That paragraph has been the subject of a finding in three consecutive review rounds, and
    all three drafts failed in the same direction: they claimed reach the rule does not have.
    "from anywhere but inside the chain" was false because the guard reads one file; "from
    anywhere else in this file" was false because a caller whose own `def` is named -- a
    second class's `__init__`, or a closure inside `__init__` -- is inside the chain by the
    mechanism's own definition. A fourth rewrite unpinned by anything would be a guess.

    So both halves are checked here, and they fail for different reasons:

      * the docstring must still contain the clause verbatim, so it cannot be reworded into
        something confident and wrong without this going red;
      * the real rule, run over the four in-file caller shapes, must still report exactly the
        one the clause says it reports -- so the sentence cannot quietly stop describing the
        code either.

    This is the shape `tests/test_supervisor.py` already uses on the heartbeat cadence: read
    the prose, compare it to the literal, fail on disagreement. A comment asserting what the
    code does not do is this codebase's signature disease, and the cure is not a better
    comment, it is a comment something reads.
    """
    _assert_yolo_weapon_models_docstring_states(_MODEL_PYS_CLAIM)

    reported = _chain_file_from(
        ast.parse(_FOUR_IN_FILE_CALLERS), ("__init__", "_load_or_refuse", "_say_what_loaded")
    ).escaped_calls
    assert reported == _THE_ONE_IN_FILE_CALL_REPORTED, (
        f"the call rule now reports {list(reported)} over the four in-file caller shapes, "
        f"not {list(_THE_ONE_IN_FILE_CALL_REPORTED)}. That is a real change in what the guard "
        "covers, and model.py's docstring describes the old behaviour. Update both, in that "
        "order."
    )


def test_no_other_module_calls_a_private_step_of_startup() -> None:
    """The one caller shape a single file cannot show -- and the only one nothing watched.

    Every rule above reads one chain file against its own entry, so a caller in ANOTHER
    module is not judged and found harmless: it is not read. Three drafts of `model.py`'s
    docstring disclosed that as a narrowness and the fourth claimed the pin covered it -- but
    the pin's fixture is one `ast.parse` of one string and cannot hold a second module, so
    that bullet was asserted by construction and checked by nothing. It is checked here.

    It is the WIDEST of them, which is why it earns a rule and not a better sentence:
    `weapons/pool.py` calling `YoloWeaponModel._load_or_refuse` leaves `self.weights`
    describing the file that stopped running and never re-runs `refuse_unusable_weights`, so
    case 4 -- a classifier -- can be swapped in after startup while the object stays alive.

    The fixture is asserted BEFORE the tree, for the reason
    `test_every_step_named_in_the_startup_chain_still_exists` gives: prove the rule can SEE
    the shape before reporting the tree is clean of it. What the reader READ is the other
    half of that and is the test below, kept separate so the two reds cannot be confused.
    Which names the rule polices, and why only those, is argued where it lives in
    `tests/weapons_chain_reader.py`.
    """
    _assert_yolo_weapon_models_docstring_states(MODEL_PYS_CROSS_MODULE_CLAIM)

    owners = _private_steps(STARTUP_CHAIN)
    fixture = {rel: ast.parse(src) for rel, src in REACH_INS_ACROSS_MODULES.items()}
    seen = _reaching_into_another_module(fixture, owners)
    assert seen == THE_REACH_INS_REPORTED, (
        f"the cross-module rule reports {list(seen)} over the fixture, not the two reach-ins "
        "it is built to see and the one own-call it must leave alone. That fixture is the "
        "defect this rule exists for -- every other rule in this file is green over it -- and "
        "it spans two owners, two packages and two step names on purpose, so a rule that has "
        "quietly narrowed to one of them fails here. If the change was deliberate, change "
        "model.py's sentence with it."
    )

    escaped = _reaching_into_another_module(_read_src_trees(), owners)
    assert not escaped, (
        f"a module reaches into another module's private startup step: {list(escaped)}. That "
        "step performs the refusal, and the caller does not re-perform it -- so the weights "
        "can be swapped for a file nothing checked while the object goes on describing the "
        "old one. A single-underscore name is private to its module; do not call it from "
        "outside. If the name is a coincidence rather than the step, it is still a reach into "
        "another module's private function, and the answer is the same."
    )


def test_the_cross_module_rule_reads_every_module_under_src() -> None:
    """The other half of the test above: what the reader READ, not what the rule found.

    An empty `escaped` is worth something only if the reader opened everything it claims to.
    Narrow `_read_src_trees` back to the `STARTUP_CHAIN` keys and the rule above silently
    becomes the four before it -- its fixture assertion still passes, because that runs over
    trees the test supplies, and its scan still passes, because a reader that looked only at
    the chain files cannot find a caller outside them. Green, twice, over a rule that has
    stopped doing the one thing it was added to do.

    That is this project's signature defect wearing the costume of the fix for it, so it is
    checked separately and reds separately. Re-globbing here rather than trusting the reader
    is the point: two independent statements of the same set, and they must agree.
    """
    on_disk = {path.relative_to(SRC_DIR.parent).as_posix() for path in SRC_DIR.rglob("*.py")}
    read = set(_read_src_trees())
    assert read == on_disk, (
        f"`_read_src_trees` does not read every module under `src/`; the two sets differ by "
        f"{sorted(on_disk ^ read)}. A rule that reads less than it claims answers empty for "
        "the wrong reason, and the cross-module rule above is the one thing standing between "
        "`weapons/pool.py` and a hot-swap that never re-runs the refusal."
    )


def test_the_only_place_that_catches_it_is_the_diagnostic_command() -> None:
    """One handler for `WeaponWeightsUnusable` in all of `src/`, and it is `weapons
    weights` turning a refusal into an exit CODE. It never returns a working model."""
    catchers = sorted(
        path.relative_to(SRC_DIR).as_posix()
        for path in SRC_DIR.rglob("*.py")
        if _catches_by_name(path)
    )
    assert catchers == ["qorgan/weapons/cli.py"], (
        f"something new catches WeaponWeightsUnusable: {catchers}. If it is not a "
        "diagnostic printing an exit code, it is a fallback."
    )


def _catches_by_name(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        if handler.type is None:
            continue
        names = [n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)]
        if "WeaponWeightsUnusable" in names:
            return True
    return False

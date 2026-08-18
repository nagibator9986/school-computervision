"""How the startup-chain guard READS a chain file. The rules that judge it are next door.

`tests/test_weapons_startup_chain.py` holds the assertions and the whole argument for why
they exist; this module holds only the parsing they share. It was split out when that
file reached 483 lines against `test_code_limits.MAX_FILE_LINES = 500`, which applies to
`tests/` as much as to `src/`. Split, never loosen -- the cap is not the thing that was
wrong.

This is the ONE-FILE half: four rules, each reading a single chain file against that file's
own entry. The cross-module rule -- every module under `src/` at once -- is different
machinery for a different question and lives in `tests/weapons_reach_in_reader.py`, split off
in turn when this file reached 497 of the same cap.

Nothing here asserts. Every function returns what it found, and `_ChainFile` keeps the four
answers apart so that an empty one cannot be mistaken for a clean one. That separation is
the entire subject of the file next door, so it is worth saying once here: the reason this
module reports four tuples rather than one boolean is that "I found no swallowing handler"
and "I read nothing" must never arrive as the same value.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tests.conftest import SRC_DIR

# The functions on the path from "load the weights" to "the worker keeps running". A
# handler anywhere along here that does not re-raise is the fallback this module forbids,
# and it would be invisible to every behavioural test in `test_weapons_refusal.py` -- they
# would still raise, and something upstream would still swallow it.
#
# `_load_or_refuse` and `_say_what_loaded` are named because R1 forced
# `YoloWeaponModel.__init__` under 50 lines and those two steps came out of it. Naming them
# does two jobs now: the swallow rule judges any handler they grow, and the call rule holds
# them to `model.py:86-89`'s claim that they are steps of construction rather than a loader
# anybody can call later.
#
# `MODULE_SCOPE` may appear in any entry. It names the import-time scope -- statements
# outside every `def` -- which has no function name to add, and without it a module-level
# handler produces a red with no move that satisfies it. A rule that cannot be satisfied is
# a rule the next maintainer deletes wholesale, so the scope is nameable.
MODULE_SCOPE = "<module>"

STARTUP_CHAIN = {
    "src/qorgan/worker/builders.py": ("build_all", "_build_weapons", "_one_weapons_pipeline"),
    "src/qorgan/worker/entrypoint.py": ("_serve", "run_group"),
    "src/qorgan/weapons/model.py": ("__init__", "_load_or_refuse", "_say_what_loaded"),
}

# THERE IS NO WHITELIST OF EXCEPTION TYPES HERE, and its absence is the rule.
#
# This file used to carry `CATCHES_IT = {"WeaponWeightsUnusable", "RuntimeError",
# "Exception", "BaseException"}` and judge only handlers naming one of those. That
# whitelist cannot be complete BY CONSTRUCTION: `model.py::_load_or_refuse` re-raises
# `type(exc)(...)` -- whatever torch threw -- and for case 3 that is an `OSError`
# (`model.py:153`, `test_weapons_refusal.py:174` asserts it). So
#
#     except OSError as exc:                      # in _build_weapons, on a stalling share
#         logger.warning("weights unreachable, starting without")
#         return {}
#
# was INSPECTED, waved through unjudged, and green -- while the weapons camera came up
# watching nothing. Worse, the remedy the out-of-reach rule advertises ("name the enclosing
# function") was a route from red to unjudged-green. Inside these three files every catch
# site must re-raise whatever it names: a handler on the startup path that does not
# re-raise is a fallback regardless of the type on it.


@dataclass(frozen=True)
class _Site:
    """One place in a chain file where an exception can be caught and dropped.

    Two shapes count, and the second is the one the repo's own linter will ASK for. `ruff`
    runs with `SIM` and `S` selected (`pyproject.toml:90`); SIM105 and S110 both tell a
    maintainer to rewrite `try: ... except X: pass` as `with suppress(X): ...`. That
    rewrite deletes the `ast.ExceptHandler` node and with it every rule in this file --
    `missing`, `swallowing` and `unreached` all go empty over a live suppression. This
    codebase has already been bitten by taking a linter's advice on load-bearing code:
    `src/qorgan/gpu.py:84` records `ruff --fix` reordering two imports and silently killing
    GPU face recognition. A `suppress` never re-raises, so inside a step of startup it can
    only ever be a fallback, and `reraises` is False for it by construction rather than by
    inspection.

    **A `suppress` counts only INSIDE a named step, and that boundary was set by a real
    line rather than by taste.** `entrypoint.py:241` is
    `with contextlib.suppress(ValueError): signal.signal(sig, handle)`, in
    `_install_signal_handlers` -- correct, narrow, and about installing a signal handler off
    the main thread, which has nothing to do with a weapons refusal. A whole-file rule makes
    that line a permanent red that naming cannot answer (a `suppress` can never be made to
    re-raise), which is precisely the state that gets a rule deleted rather than satisfied.
    So `unreached` reports `except` sites only. The gap this leaves is stated rather than
    hidden: a `suppress` in an UNNAMED function that is nevertheless on the path is not
    caught. To be on the path that function must be called from a step -- and if it holds an
    `except` instead, `unreached` still catches it.
    """

    node: ast.AST
    label: str
    lineno: int
    kind: str
    reraises: bool
    why: tuple[str, ...] = ()
    """The statements that disqualified it, when that is what happened. Named in the red."""

    def __str__(self) -> str:
        because = f" via {', '.join(self.why)}" if self.why else ""
        return f"{self.label}: line {self.lineno} ({self.kind}){because}"


@dataclass(frozen=True)
class _ChainFile:
    """What the guard could actually SEE in one file, as opposed to what it found there.

    Four fields, four different reds, kept apart on purpose: an empty `swallowing` means
    "every catch site I read re-raises on every path" only when the other three are empty
    too. On its own it means "I read nothing", which is the defect this file exists to close.
    """

    missing: tuple[str, ...]
    """Names in the entry with no matching `def` in the file. Coverage lost."""

    swallowing: tuple[str, ...]
    """Catch sites the guard reads that do not re-raise on every path out of them."""

    unreached: tuple[str, ...]
    """Catch sites in the file that no named scope contains, so nothing judges them."""

    escaped_calls: tuple[str, ...]
    """Calls to a named step from outside the chain -- a step reused as a public seam."""


def _read_chain_file(relative: str) -> _ChainFile:
    """Parse one `STARTUP_CHAIN` file ONCE and answer all four questions from that tree.

    One parse, because three of the four answers are set differences over the SAME nodes,
    and comparing nodes from two separate parses by line number would be a guess where
    object identity is available.
    """
    path = SRC_DIR.parent / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _chain_file_from(tree, STARTUP_CHAIN[relative])


def _chain_file_from(tree: ast.AST, functions: tuple[str, ...]) -> _ChainFile:
    """The four answers, over a tree that is already parsed.

    Separated from the file reading so that a test can run the REAL rules over source it
    constructs, rather than over the three files that happen to be clean today. That is
    what `test_model_pys_account_of_the_call_rule_is_still_true` does: the account
    `model.py` gives of the call rule has been wrong three times, so it is now checked
    against the rule itself rather than re-read by hand.
    """
    # `ast.FunctionDef` only, so that a step turned into `async def`
    # (`ast.AsyncFunctionDef`) shows up as MISSING rather than as silently-not-inspected.
    # That asymmetry is a trap, so it is made to fail rather than quietly absorbed.
    steps = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in functions]
    in_steps = {id(n) for step in steps for n in ast.walk(step)}
    sites = _catch_sites(tree)

    inspected = set(in_steps)
    if MODULE_SCOPE in functions:
        inspected |= {id(s.node) for s in sites if s.label == MODULE_SCOPE}

    return _ChainFile(
        missing=tuple(
            name
            for name in functions
            if name != MODULE_SCOPE and name not in {step.name for step in steps}
        ),
        swallowing=tuple(
            sorted(str(s) for s in sites if id(s.node) in inspected and not s.reraises)
        ),
        # `except` only. A `suppress` OUTSIDE the named steps is not reported: see `_Site`
        # for why that boundary is where it is, and which real line moved it there.
        unreached=tuple(
            sorted(str(s) for s in sites if s.kind == "except" and id(s.node) not in inspected)
        ),
        escaped_calls=_calls_from_outside_the_chain(tree, functions, in_steps),
    )


def _catch_sites(tree: ast.AST) -> list[_Site]:
    """Every `except` and every `contextlib.suppress` in the file, each with its scope."""
    labels = _scope_labels(tree)
    aliases = _suppress_aliases(tree)
    sites = []
    for node in ast.walk(tree):
        label = labels.get(id(node), MODULE_SCOPE)
        if isinstance(node, ast.ExceptHandler):
            reraises, leaves = _judge_handler(node.body)
            sites.append(_Site(node, label, node.lineno, "except", reraises, leaves))
        elif _is_suppress(node, aliases):
            sites.append(_Site(node, label, node.lineno, "contextlib.suppress", False))
    return sites


def _scope_labels(tree: ast.AST) -> dict[int, str]:
    """Every node -> the innermost `def` around it. Absent from the map means module scope.

    Sorted by `lineno` and overwritten as we go: any function containing a node starts at or
    before it, so the innermost is the one that starts LAST. `async` is kept in the label
    because "the guard cannot see this one" and "this one is async" are the same fact.
    """
    labels: dict[int, str] = {}
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    for node in sorted(defs, key=lambda n: n.lineno):
        label = f"async {node.name}" if isinstance(node, ast.AsyncFunctionDef) else node.name
        for child in ast.walk(node):
            labels[id(child)] = label
    return labels


def _suppress_aliases(tree: ast.AST) -> set[str]:
    """Local names that mean `contextlib.suppress` in this file, `import ... as` included."""
    names = {"suppress"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            names |= {a.asname or a.name for a in node.names if a.name == "suppress"}
    return names


def _is_suppress(node: ast.AST, aliases: set[str]) -> bool:
    """A `with suppress(...)` / `with contextlib.suppress(...)`, in any spelling of it."""
    if not isinstance(node, ast.With | ast.AsyncWith):
        return False
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr == "suppress":
            return True
        if isinstance(call.func, ast.Name) and call.func.id in aliases:
            return True
    return False


# WHY THE RE-RAISE RULE IS TWO QUESTIONS AND NOT ONE. Kept here rather than in a docstring
# because every round of this task has ended by expanding that docstring, and the function it
# was attached to reached 49 of the 50-line cap.
#
# The original rule was `any(isinstance(n, ast.Raise) for n in ast.walk(handler))` -- a
# `raise` ANYWHERE in the subtree, including one that never runs. The commonest way a hard
# refusal gets softened walked straight through it (left, below). Requiring the `raise` to be
# a statement of the handler's OWN body, or an `if` whose every branch raises, closed that --
# and left the mirror image (right, below) green, because asking "is a `raise` reached" never
# asks "does something leave first":
#
#     except Exception as exc:                    except Exception as exc:
#         if settings.strict_weapons:                 if not settings.strict_weapons:
#             raise                                       logger.warning("motion only")
#         logger.warning("motion only")                   return None
#         self._model = None                          raise
#                 RED                                         GREEN, and identical
#
# The right-hand column is not a contrivance: an early-return guard clause is at least as
# idiomatic as a guarded `raise`, and its `continue` spelling -- "this camera's weights are
# bad, keep the group up" -- is the client's original defect at per-camera granularity.
#
# So both questions are asked, and the exits are asked FIRST: a handler that can leave
# without raising is a fallback even when a later line does raise, and that later line is
# then dead code which reads as protection.
#
# The rule is conservative in one direction only. A `raise` inside a `for`, `while`, `with`
# or nested `try` is not counted as reached, so `with self._lock: raise` is called a swallow
# though every path out of it does raise. That is a false alarm with an obvious answer --
# move the `raise` to the handler's top level -- never a miss.


def _judge_handler(body: list[ast.stmt]) -> tuple[bool, tuple[str, ...]]:
    """Does every path out of this handler end in a `raise`, and if not, what leaves first?

    Returns both, because the second half is the diagnostic: a red that says only "this
    handler is a fallback" leaves the reader hunting for which line made it one.
    """
    leaves = tuple(_early_exits(body))
    return (not leaves and _reaches_a_raise(body)), leaves


def _early_exits(body: list[ast.stmt]) -> list[str]:
    """Every way out of the handler that is not a `raise`, without entering a nested `def`.

    `return`, `continue`, `break` -- and `yield`, which is the one that reads as innocent.
    A handler that yields hands control back to whoever is iterating, and if they abandon
    or close the generator, `GeneratorExit` is thrown at the `yield` and the `raise` below
    it never runs. No named step is a generator today; it is here because the sentence this
    rule is meant to satisfy says "any exit that is not a `raise`", and a rule one node type
    short of its own sentence is how the last three findings started.

    Pruned walk rather than `ast.walk`: a `return` inside a function DEFINED in the handler
    belongs to that function, not to this handler, and counting it would make an honest
    handler that happens to define a callback look like a fallback. `ClassDef` is pruned for
    the same reason and it is load-bearing -- a `break` inside a `for` in a class body is
    legal and does not exit the handler. `Lambda` is pruned belt-and-braces only: a lambda
    body is an expression and cannot contain any of these, so it can never change the answer.

    Everything else counts at any depth -- inside an `if`, a `for`, a `while`, a `with`, a
    nested `try`, a nested `try`'s own handler -- because the question is whether a path
    exists, not how deeply it is spelled.
    """
    found: list[str] = []
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(node, ast.Return | ast.Continue | ast.Break | ast.Yield | ast.YieldFrom):
            found.append(f"{type(node).__name__.lower()} on line {node.lineno}")
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found)


def _reaches_a_raise(body: list[ast.stmt]) -> bool:
    """Is a `raise` reached on every path through these statements?"""
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
        if (
            isinstance(stmt, ast.If)
            and _reaches_a_raise(stmt.body)
            and _reaches_a_raise(stmt.orelse)
        ):
            return True
    return False


def _calls_from_outside_the_chain(
    tree: ast.AST, functions: tuple[str, ...], in_steps: set[int]
) -> tuple[str, ...]:
    """Calls to a named step that are not themselves inside a named step.

    `model.py:86-89` claims `_load_or_refuse` and `_say_what_loaded` are steps of `__init__`
    "called from the constructor and nowhere else". Before the split that was a tautology --
    the code was inline, there was nothing to call. The split turned it into a falsifiable
    claim about a `@staticmethod` that takes `(model_path, artefact, camera)` and returns a
    loaded model, which is precisely the signature of a reusable loader, and nothing checked
    it. Add

        def reload(self, path: str) -> None:
            self._model = self._load_or_refuse(path, inspect_weights_file(path), self.camera)

    and there IS a load step to forget again: `self.weights` still describes the OLD
    artefact, so the panel and the fingerprint log name a file that is not running, and
    `refuse_unusable_weights` is not re-run, so a classifier -- case 4, the failure this
    module was built for -- can be swapped in at runtime while the object stays alive. Every
    other rule in this file stays green through it. And it is not an invented shape:
    `builders.py:68-97` already ships `PoseLoader`, the blessed lazy-loader template for the
    OTHER model in this codebase, for the next person to copy.
    """
    labels = _scope_labels(tree)
    escaped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in in_steps:
            continue
        name = _called_name(node.func)
        if name is not None and name != MODULE_SCOPE and name in functions:
            escaped.append(f"{labels.get(id(node), MODULE_SCOPE)}: line {node.lineno} -> {name}()")
    return tuple(sorted(escaped))


def _called_name(func: ast.expr) -> str | None:
    """`f()` -> `f`; `self.f()` and `mod.f()` -> `f`; anything else -> None."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None

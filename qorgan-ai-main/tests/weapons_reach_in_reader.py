"""The cross-module half of the startup-chain guard: one module reaching into another's step.

`tests/weapons_chain_reader.py` reads ONE chain file against ONE entry, four times over. This
module reads every module under `src/` at once, which is different machinery for a different
question, and it was split out when that file reached 497 of
`test_code_limits.MAX_FILE_LINES = 500`. Split, never loosen -- the cap is not the thing that
was wrong.

Like its sibling, **nothing here asserts.** The assertions are in
`tests/test_weapons_startup_chain.py`. Unlike its sibling this module also carries the SOURCE
its rule is pinned against, for the same reason `STARTUP_CHAIN` lives next door rather than in
the test file: a fixture the rule runs over is data, not argument, and five modules of it is
too much to keep beside an assertion.

It reuses `_scope_labels` and `_called_name` from next door rather than copying them, so the
label a red carries is the same string in both halves of the guard. That is the reach across
modules its own rule forbids -- which is not a contradiction, because the rule reads `src/`
and these are two files in `tests/` that already ship as one unit.
"""

from __future__ import annotations

import ast

from tests.conftest import SRC_DIR
from tests.weapons_chain_reader import MODULE_SCOPE, _called_name, _scope_labels

# WHY THIS RULE POLICES PRIVATE NAMES ONLY.
#
# `weapons_chain_reader.py` reads one file against one entry, so a caller in a DIFFERENT module
# is never judged and found harmless -- it is never read. That was the widest hole in the guard
# and the only one nothing watched:
#
#     # src/qorgan/weapons/pool.py
#     model._model = YoloWeaponModel._load_or_refuse(path, inspect_weights_file(path), cam)
#
# is the hot-swap the weapons module exists to refuse. `model.weights` goes on describing the
# file that stopped running, and `refuse_unusable_weights` is not re-run, so a classifier --
# case 4 -- can be swapped in after startup while the object stays alive. That file was written
# and run during round 2 of this task and every rule next door was green over it.
#
# This rule speaks only for step names that begin with ONE underscore. The two exclusions have
# DIFFERENT standing, and saying so is the point -- an enumeration that presents a principle
# and a measurement as the same thing is how `model.py`'s docstring went wrong five times.
#
#   * `build_all` and `run_group` are PUBLIC, and this is the exclusion MEASUREMENT forced.
#     Include them and the rule reds on `entrypoint.py:96`, where `_serve` calls `build_all`:
#     the worker starting up, correctly. `entrypoint.py` does not define `build_all` itself, so
#     the own-call skip does not cover it, and the only remedies are renaming a public function
#     or carving out one caller by hand. `supervisor/managed.py:43` hands `run_group` to a
#     `Process` as its target and is one edit from the same red. A published name's callers are
#     not this guard's business.
#   * `__init__` is excluded because the PRINCIPLE says so: it is a protocol slot, not a name
#     anybody chose, and not private by the convention this rule reads. Measured rather than
#     assumed, and it did not go the way it was first written up: including `__init__` is GREEN
#     over `src/` today, because both modules that call it across a boundary
#     (`models/pose.py:207`, `redaction.py:69`) define their own `__init__` and the own-call
#     skip already covers them. So this exclusion is latent, not live -- a caller that did NOT
#     define its own `__init__` would be reported as reaching into `weapons/model.py`, which is
#     a nonsense attribution with no honest answer.
#
# What is left is the case whose only honest answer is "you are reaching into another module's
# private function; do not" -- the answer regardless of this guard, and what ruff's `SLF` would
# say if `select` had it. A rule that fires with no answer is deleted by whoever it blocks, and
# it takes the rule that catches the real fallback with it. This one always has an answer.
#
# THE OWN-CALL SKIP IS PER CALL, NOT PER MODULE, and that difference is the rule. Per module --
# "this file has a `def _load_or_refuse` in it somewhere, so none of its calls count" -- was the
# first version, and it was cheaper to DEFEAT than to obey: keep the hot-swap and add a decoy
# `def` of the same name beside it, and every call in the file goes quiet. Measured green under
# it in three spellings: an explicit `YoloWeaponModel._load_or_refuse(...)` with a module-level
# decoy, the decoy written as a method of the pool, and the reach-in taken through the instance.
# So only two spellings count as the module's own, and each must ALSO be defined here: a bare
# `name(...)`, and `self.name(...)` / `cls.name(...)`. A call qualified by ANY other expression
# is a reach into something else's private member, and is reported whatever this module defines.
#
# The alternative offered and rejected, with the measurement that decided it: assert instead
# that each private step name has exactly one `def` in `src/` -- true today -- and let the decoy
# red on THAT. It reds on the wrong thing. An unrelated module that legitimately defines its own
# private `_serve` breaks that invariant while reaching into nothing, and its only remedy is
# renaming code that is correct. That is the red this branch has already rescoped two rules to
# avoid, and the per-call skip closes the decoy without buying it.


def _private_steps(chain: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Each single-underscore step name -> the one module entitled to call it. See above."""
    return {
        name: relative
        for relative, entry in chain.items()
        for name in entry
        if name.startswith("_") and not name.startswith("__")
    }


def _read_src_trees() -> dict[str, ast.AST]:
    """Every module under `src/`, parsed once, keyed as `STARTUP_CHAIN` keys itself."""
    return {
        path.relative_to(SRC_DIR.parent).as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(SRC_DIR.rglob("*.py"))
    }


def _reaching_into_another_module(
    trees: dict[str, ast.AST], owners: dict[str, str]
) -> tuple[str, ...]:
    """Calls to a private startup step from a module that does not own it.

    Takes parsed trees rather than reading them, for the reason `_chain_file_from` does: so a
    test can run the REAL rule over source it constructs. The shape guarded here needs two
    modules to exist at all, and a fixture that could not hold it is exactly how `model.py`
    came to say four shapes were checked when three were.

    Narrower than "nothing outside may name these", in three ways, each stated because an
    undisclosed narrowness is how this guard has been wrong before:

      * only `src/` is read, so a test or a script reaching in is not seen;
      * a call the module makes to its OWN function of that name is skipped -- per CALL, not
        per module; the note above has the decoy that distinction closes and the two spellings
        that count. Still skipped, and genuinely ambiguous: a module that both imports the step
        under its bare name and defines its own `def` of it;
      * `ast.Call` only, as next door -- `cb = mod._load_or_refuse` then `cb(p)` is a
        reference, and `getattr(mod, "_load_or_refuse")(p)` names `getattr`.
    """
    found = []
    for relative, tree in trees.items():
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        labels = _scope_labels(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node.func)
            owner = owners.get(name) if name is not None else None
            if owner is None or owner == relative or _is_its_own_call(node, name, defined):
                continue
            scope = labels.get(id(node), MODULE_SCOPE)
            found.append(f"{relative}:{node.lineno} {scope} -> {name}() [owned by {owner}]")
    return tuple(sorted(found))


def _is_its_own_call(node: ast.Call, name: str, defined_here: set[str]) -> bool:
    """Is THIS call the calling module's own function, rather than a reach into the step?

    Two spellings can be, and each must also be defined in this file: a bare `name(...)`, and
    `self.name(...)` / `cls.name(...)`. `Anything.name(...)` qualified by any other expression
    -- `YoloWeaponModel._load_or_refuse(p)`, `model._load_or_refuse(p)` -- reaches into another
    object's private member and is never this module's own, however many `def`s of that name
    sit beside it. See the note above the rule for the decoy this refusal closes.
    """
    if isinstance(node.func, ast.Attribute):
        value = node.func.value
        if not (isinstance(value, ast.Name) and value.id in {"self", "cls"}):
            return False
    return name in defined_here


# The source the rule is pinned against: one module reaching into another's private step. Round
# 2 wrote `pool.py`, found every rule green over it, and reverted it, so the hole lived in a
# report rather than in the repo. It is committed now.
#
# It deliberately spans TWO owners, in TWO packages, naming TWO different steps, and carries a
# module that must stay green. Round 4's version held one owner, one name, one package -- so a
# rule narrowed to `weapons/` only, or to `_load_or_refuse` only, stayed green over it AND over
# the 173 real modules, and neither test noticed. A fixture that exercises exactly one of
# anything cannot tell you the rule still covers the rest. The LINE NUMBERS below are part of
# the pin: edit the fixture and re-measure, rather than adjust the expectation until it fits.
REACH_INS_ACROSS_MODULES = {
    # owner 1, in `weapons/`
    "src/qorgan/weapons/model.py": '''
class YoloWeaponModel:
    def __init__(self, path):
        self._model = self._load_or_refuse(path)

    @staticmethod
    def _load_or_refuse(path):
        return path
''',
    # owner 2, in `worker/` -- a different package, a different step name
    "src/qorgan/worker/entrypoint.py": '''
def _serve(name):
    return name
''',
    # REPORTED. The decoy `def` beside the hot-swap is why the own-call skip is judged per
    # CALL: under a per-MODULE skip this whole file went quiet, so a decoy was cheaper than
    # obedience, and a rule cheaper to defeat than to obey is one that gets defeated.
    "src/qorgan/weapons/pool.py": '''
from qorgan.weapons.model import YoloWeaponModel


def _load_or_refuse(path):
    return path


class WeaponModelPool:
    def reload(self, model, path):
        model._model = YoloWeaponModel._load_or_refuse(path)
''',
    # REPORTED: the second owner, reached from a third package
    "src/qorgan/supervisor/relay.py": '''
from qorgan.worker import entrypoint


def start(name):
    return entrypoint._serve(name)
''',
    # NOT REPORTED: an unrelated module calling its OWN `_serve`. This is the cell the skip
    # exists for, and a fixture without it cannot show the skip still fires at all.
    "src/qorgan/canteen/portioning.py": '''
class ServingHatch:
    def open_for_lunch(self):
        return self._serve()

    def _serve(self):
        return "lunch"
''',
}

# What the rule must report over that fixture, in full. Not the labels alone: a rule reporting
# `reload -> _say_what_loaded()` on some other line would be a different rule with the same
# shadow, and this is a pin, so it holds the whole answer including where it is.
THE_REACH_INS_REPORTED = (
    "src/qorgan/supervisor/relay.py:6 start -> _serve() "
    "[owned by src/qorgan/worker/entrypoint.py]",
    "src/qorgan/weapons/pool.py:11 reload -> _load_or_refuse() "
    "[owned by src/qorgan/weapons/model.py]",
)

# `model.py`'s second mechanism clause, lifted verbatim, as `_MODEL_PYS_CLAIM` lifts the first.
MODEL_PYS_CROSS_MODULE_CLAIM = (
    "a call to either step from another module under `src/`, except a bare or "
    "`self`/`cls`-qualified call in a module that defines that name itself"
)

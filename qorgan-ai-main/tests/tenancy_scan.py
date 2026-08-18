"""Every database query in `src/`, and whether it names a school.

This is the fact-gathering half of `test_tenancy_guard`. It answers one question per
query site -- *which tables does this statement touch, and does anything in it constrain
`school_id`* -- and it answers it by resolving the OWNER of every attribute access, the
same way `tests/config_scan.py` does for config keys, and for the same reason: a
name-based grep cannot tell `Person.school_id` from `some_row.school_id`, and a guard
that cannot tell those apart is a guard that can be satisfied by an unrelated line.

WHAT A "QUERY SITE" IS. Three kinds, and the third is the one people forget:

  * `select(...)`, `update(...)`, `delete(...)` -- the statement constructors, recognised
    only when the name was imported from `sqlalchemy` in that module.
  * `session.get(Model, pk)` -- **a query with no WHERE clause at all.** It is the classic
    cross-tenant hole: `/events/{id}/review` takes an id out of a URL and hands it
    straight to the database. There is no filter to look for, so this scan reports it as
    touching the table and carrying nothing, which is exactly true.
  * Relationship traversal (`person.photos`) is NOT a site here, and cannot be: it starts
    from a row somebody already loaded, so it is only as safe as the query that loaded it.
    Stated so nobody mistakes this scan for complete.

WHAT A "UNIT" IS -- why the scan is not per-expression. A statement is rarely built in
one expression:

    query = select(Event, Camera.display_name).join(Camera, ...)
    if status_filter:
        query = query.where(Event.status == wanted)

Both lines build ONE statement. So a site's unit is its whole method chain, plus every
later chain in the same scope rooted at the variable the chain was assigned to. That is
also why the unit is per-variable and not per-function: `web/routes/events.py` builds
`query` and `counter` side by side in one function, and under a per-function rule a
filter on `query` would vouch for `counter` -- which joins nothing and would leak a count
of another school's incidents while the guard stayed green.

HELPERS ARE FOLLOWED. `diagnostics/alerts.py` builds its statement inside `_joined(...)`,
and `identity/registry.py` puts its predicate in `_active_only()`. A scan that stopped at
the call would call both unfiltered forever. So a call to a function this scan can resolve
-- in the same module, or imported from `src/` -- contributes that function's own
references to the unit, transitively.

LOCAL PREDICATES ARE FOLLOWED TOO, and the reason is the health of the exemption list.
A filter is very often named before it is used:

    mine = User.school_id == school
    total = session.scalar(select(func.count(User.id)).where(mine))

`.where(mine)` mentions no model attribute at all, so a scan that stopped at the chain
would report that statement unfiltered -- while the code is perfectly correct. That is
worse than a miss: the only way to quieten a false positive is an entry in
UNSCOPED_QUERIES, and an exemption list holding correctly-filtered code is a list nobody
can read as meaning anything. So a `Name` loaded inside the unit is resolved back to what
it was assigned in the same scope, and that assignment's references join the unit.

This is deliberately narrower than "everything in the function": only names the statement
actually mentions are followed, so a filter built for one query still cannot vouch for the
unfiltered one built beside it.

THE PRICE OF THAT, STATED PLAINLY: the merge is by reference, not by SQL semantics. A
unit that calls a helper mentioning `Camera.school_id` counts as scoped even if the helper
joined the wrong table. This scan proves that a school was NAMED, never that the join was
right. Proving the join is `test_tenancy_isolation`'s job, and it does it by putting two
schools in one database and trying to read across.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

MODELS_MODULE = "qorgan.db.models"
SQLALCHEMY = "sqlalchemy"
STATEMENT_BUILDERS = frozenset({"select", "update", "delete"})
PK_FETCH = "get"
MAX_ROUNDS = 12


@dataclass(frozen=True)
class Site:
    """One query, and everything the scan could prove about it."""

    module: str  # "qorgan/identity/registry.py"
    scope: str  # enclosing function qualname, or "<module>"
    ordinal: int  # which query in that scope, in source order
    lineno: int
    kind: str  # "select" | "update" | "delete" | "get"
    models: frozenset[str]  # mapped classes this unit touches
    refs: frozenset[tuple[str, str]]  # every (Model, attribute) the unit names

    @property
    def name(self) -> str:
        """Stable across edits that add a filter, which is what exemptions are keyed on."""
        return f"{self.module}::{self.scope}::{self.ordinal}"

    def constrains(self, column: str, tenant: frozenset[str]) -> bool:
        return any(model in tenant and attr == column for model, attr in self.refs)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    table: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            table[child] = node
    return table


def _functions(tree: ast.AST) -> dict[str, ast.AST]:
    """Every function in the module, by qualified name. Classes prefix their methods."""
    found: dict[str, ast.AST] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                found[name] = child
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return found


class Module:
    """One file of `src/`: its model aliases, its functions, and its query sites."""

    def __init__(self, path: Path, src: Path) -> None:
        self.rel = path.relative_to(src).as_posix()
        self.key = self.rel[:-3].replace("/", ".").removesuffix(".__init__")
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.parents = _parents(self.tree)
        self.functions = _functions(self.tree)
        self.models: dict[str, str] = {}
        self.builders: set[str] = set()
        self.imported: dict[str, str] = {}
        self._read_imports()

    def _read_imports(self) -> None:
        """Which local names are models, which are statement constructors, which are ours.

        Every `ImportFrom` in the file, including the two that live inside functions
        (`identity/registry.py`, `identity/report.py`). A module-wide alias table is
        conservative in the safe direction: it can only ever make the scan attribute MORE
        accesses to a model, never fewer.
        """
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if node.module.startswith(MODELS_MODULE) and alias.name[:1].isupper():
                    self.models[local] = alias.name
                elif node.module == SQLALCHEMY and alias.name in STATEMENT_BUILDERS:
                    self.builders.add(local)
                elif node.module.startswith("qorgan."):
                    self.imported[local] = f"{node.module}:{alias.name}"

    def resolve_call(self, name: str) -> str | None:
        """`_joined` -> "qorgan.diagnostics.alerts:_joined", or None if not ours."""
        if name in self.functions:
            return f"{self.key}:{name}"
        return self.imported.get(name)

    def scope_of(self, node: ast.AST) -> tuple[str, ast.AST]:
        """The function a node sits in, as (qualname, body node). Module level if none."""
        by_node = {n: q for q, n in self.functions.items()}
        current: ast.AST | None = node
        while current is not None:
            if current in by_node:
                return by_node[current], current
            current = self.parents.get(current)
        return "<module>", self.tree


def direct_refs(module: Module, node: ast.AST) -> tuple[set[tuple[str, str]], set[str]]:
    """(Model, attribute) pairs and mapped classes named anywhere under `node`."""
    refs: set[tuple[str, str]] = set()
    touched: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            owner = module.models.get(child.value.id)
            if owner is not None:
                refs.add((owner, child.attr))
                touched.add(owner)
        elif isinstance(child, ast.Name) and child.id in module.models:
            touched.add(module.models[child.id])
    return refs, touched


def called_functions(module: Module, node: ast.AST) -> set[str]:
    """Resolvable functions invoked under `node`, by "module:qualname"."""
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            target = module.resolve_call(child.func.id)
            if target is not None:
                out.add(target)
    return out


def _index(modules: list[Module]) -> dict[str, tuple[frozenset, frozenset]]:
    """What each function in `src/` names, transitively through the calls it makes."""
    refs: dict[str, set] = {}
    touched: dict[str, set] = {}
    calls: dict[str, set[str]] = {}
    for module in modules:
        for qualname, node in module.functions.items():
            key = f"{module.key}:{qualname}"
            refs[key], touched[key] = direct_refs(module, node)
            calls[key] = called_functions(module, node)

    for _ in range(MAX_ROUNDS):
        changed = False
        for key, targets in calls.items():
            for target in targets:
                if target not in refs or target == key:
                    continue
                if not refs[target] <= refs[key] or not touched[target] <= touched[key]:
                    refs[key] |= refs[target]
                    touched[key] |= touched[target]
                    changed = True
        if not changed:
            break
    return {k: (frozenset(refs[k]), frozenset(touched[k])) for k in refs}


def _extends(module: Module, parent: ast.AST, current: ast.AST) -> bool:
    """Is `parent` one more link in the statement `current` is part of?

    The first two cases are `.where(...)` written as an Attribute and then a Call. The
    third is a resolvable helper the half-built statement is handed to
    (`_joined(select(...), unsent)`): that call is part of building this same statement,
    and stopping at the argument would report the alerts panel unfiltered for ever.
    """
    if isinstance(parent, ast.Attribute) and parent.value is current:
        return True
    if isinstance(parent, ast.Call) and parent.func is current:
        return True
    return (
        isinstance(parent, ast.Call)
        and any(arg is current for arg in parent.args)
        and isinstance(parent.func, ast.Name)
        and module.resolve_call(parent.func.id) is not None
    )


def _chain_top(module: Module, node: ast.AST) -> ast.AST:
    """The whole method chain this constructor is the head of.

    `select(X).where(Y).limit(n)` is one statement written as three nested nodes, so the
    unit is the outermost of them.
    """
    current = node
    while True:
        parent = module.parents.get(current)
        if parent is None or not _extends(module, parent, current):
            return current
        current = parent


def _bound_name(module: Module, top: ast.AST) -> str | None:
    """`query = select(...)` -- the variable the rest of the function keeps building on."""
    parent = module.parents.get(top)
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        target = parent.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(parent, ast.AnnAssign | ast.NamedExpr):
        return parent.target.id if isinstance(parent.target, ast.Name) else None
    return None


def _assignments(body: ast.AST) -> dict[str, list[ast.AST]]:
    """Every `name = <expr>` in this scope, by name.

    A filter is regularly named before it is used (`mine = User.school_id == school`), and
    the statement that then says `.where(mine)` names no model attribute of its own.
    """
    found: dict[str, list[ast.AST]] = {}
    for node in ast.walk(body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.value)
        elif (
            isinstance(node, ast.AnnAssign | ast.NamedExpr)
            and node.value is not None
            and isinstance(node.target, ast.Name)
        ):
            found.setdefault(node.target.id, []).append(node.value)
    return found


def _named_by(
    module: Module,
    index: dict,
    node: ast.AST,
    scope_locals: dict[str, list[ast.AST]] | None = None,
    seen: frozenset[str] = frozenset(),
) -> tuple[set, set]:
    refs, touched = direct_refs(module, node)
    for target in called_functions(module, node):
        found = index.get(target)
        if found is not None:
            refs |= found[0]
            touched |= found[1]

    if scope_locals:
        # Only names this statement actually mentions, so a filter built for one query
        # still cannot vouch for the unfiltered one built beside it.
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            if child.id in seen or child.id not in scope_locals:
                continue
            for value in scope_locals[child.id]:
                more = _named_by(module, index, value, scope_locals, seen | {child.id})
                refs |= more[0]
                touched |= more[1]
    return refs, touched


def _query_roots(module: Module) -> list[tuple[ast.AST, str]]:
    roots: list[tuple[ast.AST, str]] = []
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in module.builders:
            roots.append((node, node.func.id))
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == PK_FETCH
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in module.models
        ):
            roots.append((node, PK_FETCH))
    roots.sort(key=lambda pair: (pair[0].lineno, pair[0].col_offset))
    return roots


def _unit(module: Module, index: dict, root: ast.AST) -> tuple[frozenset, frozenset]:
    """Everything the statement this root heads gets built out of, in its own scope."""
    top = _chain_top(module, root)
    _, body = module.scope_of(top)
    scope_locals = _assignments(body)
    refs, touched = _named_by(module, index, top, scope_locals)

    variable = _bound_name(module, top)
    if variable is not None:
        for node in ast.walk(body):
            if not isinstance(node, ast.Name) or node.id != variable:
                continue
            if not isinstance(node.ctx, ast.Load):
                continue
            more_refs, more_touched = _named_by(
                module, index, _chain_top(module, node), scope_locals, frozenset({variable})
            )
            refs |= more_refs
            touched |= more_touched
    return frozenset(refs), frozenset(touched)


def scan(src_dir: Path) -> tuple[Site, ...]:
    """Every query site in `src/`, in a stable order."""
    modules = [
        Module(path, src_dir)
        for path in sorted(src_dir.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    index = _index(modules)

    sites: list[Site] = []
    for module in modules:
        counters: dict[str, int] = {}
        for root, kind in _query_roots(module):
            scope, _ = module.scope_of(root)
            counters[scope] = counters.get(scope, 0) + 1
            refs, touched = _unit(module, index, root)
            sites.append(
                Site(
                    module=module.rel,
                    scope=scope,
                    ordinal=counters[scope],
                    lineno=root.lineno,
                    kind=kind,
                    models=touched,
                    refs=refs,
                )
            )
    return tuple(sites)

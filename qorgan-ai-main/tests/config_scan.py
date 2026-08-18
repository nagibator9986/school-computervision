"""Who reads which config key? An owner-resolving scan of `src/`.

A name-based scan cannot answer this question. `ProximityOnlyGate.max_speed` looks alive
because `PairData.max_speed` shares the name; `Counters.cooldown_seconds` looks alive
because `Burst.cooldown_seconds` exists. Both were dead. A scan that counts identifiers
counts the wrong thing, and would have to hand-wave the difference in a comment.

So this resolves the OWNER of every attribute access. `data.config.gates.proximity_only`
is followed link by link -- `data: GateInput` (a dataclass whose `config` field is a
`BullyingConfig`), `.gates` -> `Gates`, `.proximity_only` -> `ProximityOnlyGate` -- and
only then is `.max_speed` on that base counted as a read OF THAT KEY. `pair.max_speed`,
where `pair` resolves to the dataclass `PairData`, is not.

It answers in three parts, and the third is the point:

  * READS      -- a proven read site: the base resolved to this model.
  * UNRESOLVED -- no proven read, but there IS an access this scan could not attribute
    (an attribute on a base it could not type, or a dynamic `getattr`). The scan does
    not know, so it says so -- against the KEY, not the name, which is `config_report`'s
    job and its docstring -- and `test_config_deadkeys` makes a human answer.
  * DYNAMIC    -- an access so untyped that nothing can be concluded about any key.

Anything left over -- no read, and nothing unattributed either -- is DEAD: nothing in
`src/` can possibly read it. That subtraction is done in `test_config_deadkeys`.

REACHABILITY. A field is not alive merely because someone reads it. `ViolenceSettings.
model_path` IS read -- by `ViolenceSettings`' own validator, and by nothing else, because
`BullyingConfig.violence` is read nowhere, so no `ViolenceSettings` is ever consulted.
So a model counts only if a root camera model reaches it through fields that are
themselves read. An unreachable model's every field is dead, however much it reads itself.

LIMITS, stated plainly, because a scan that hid these would be the disease it treats:
  * Inference is annotation-driven. An unannotated parameter holding a config object
    yields UNRESOLVED -- never a false pass.
  * Reads inside a model's OWN methods count. Deliberate: `ZoneRect.contains` reads
    `self.x1`, and `contains` is called from live code. Reachability, not a special case
    here, is what catches the model that only reads itself.
  * A field written but never read (`Model(field=...)` and nothing else) reads as DEAD.
    That is correct for config: the YAML is the writer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tests.config_index import (
    BUILTINS,
    CFG,
    CLS,
    CONTAINERS,
    PLAIN_TY,
    STRS,
    UNKNOWN,
    Index,
    _cfg,
    _imports_of,
    _reachable,
    _string_set,
    build_index,
)
from tests.config_report import Note, Report, attribute_blind_spots


class _Resolver(ast.NodeVisitor):
    """Walks one scope, typing every expression it can, recording reads it can prove."""

    def __init__(self, path, env, imports, index: Index, report: Report | None, cls=None):
        self.path, self.env, self.imports = path, dict(env), imports
        self.index, self.report, self.cls = index, report, cls

    # -- typing ------------------------------------------------------------
    # An expression that can only ever produce a non-config value. Listing them lets the
    # scan say "definitely not a config" instead of "I don't know", which is the whole
    # difference between a key it can call dead and a key it must escalate to a human.
    _NEVER_CONFIG = (
        ast.JoinedStr | ast.List | ast.Dict | ast.Set | ast.Tuple
        | ast.Compare | ast.UnaryOp | ast.Lambda | ast.BinOp
    )
    # These pass their operand's type straight through.
    _TRANSPARENT = ast.Subscript | ast.Await | ast.Starred

    def resolve(self, node) -> Any:  # noqa: PLR0911 - one return per AST node kind
        """The type of an expression, as far as annotations can tell. None = unknown."""
        if isinstance(node, ast.Name):
            return self._name_ty(node)
        if isinstance(node, ast.Attribute):
            return self._attr_ty(node)
        if isinstance(node, ast.Call):
            return self._call_ty(node)
        if isinstance(node, ast.Constant):
            return (STRS, frozenset({node.value})) if isinstance(node.value, str) else PLAIN_TY
        if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            return self._comp_ty(node)
        if isinstance(node, self._TRANSPARENT):
            return self.resolve(node.value)
        if isinstance(node, ast.BoolOp):  # `a or b` is one of them, so: the union
            return self._join(self.resolve(v) for v in node.values)
        if isinstance(node, ast.IfExp):
            return self._join([self.resolve(node.body), self.resolve(node.orelse)])
        if isinstance(node, self._NEVER_CONFIG):
            return PLAIN_TY
        return UNKNOWN

    def _comp_ty(self, node) -> Any:
        """`{n: c for n, c in cameras.items() if ...}` is still a dict of cameras.

        The comprehension binds its own targets, so it needs its own scope. It gets no
        report: the ordinary walk visits the same nodes and would record every read twice.
        """
        inner = _Resolver(self.path, self.env, self.imports, self.index, None, self.cls)
        for generator in node.generators:
            inner.bind_iter(generator.target, generator.iter)
        return inner.resolve(node.value if isinstance(node, ast.DictComp) else node.elt)

    def _join(self, types) -> Any:
        """`a or b`, `a if c else b`: the value is one of them, so the type is the union."""
        seen = [t for t in types if t is not UNKNOWN and t is not PLAIN_TY]
        if not seen:
            return PLAIN_TY
        if all(t[0] == CFG for t in seen):
            return _cfg(frozenset().union(*(t[1] for t in seen)))
        return seen[0] if all(t == seen[0] for t in seen) else UNKNOWN

    def _name_ty(self, node) -> Any:
        if node.id == "self" and self.cls:
            return _cfg({self.cls}) if self.cls in self.index.models else (CLS, self.cls)
        name = self.imports.get(node.id, node.id)
        if name in self.index.models:
            return _cfg({name})
        if name in self.index.aliases:
            return _cfg(self.index.aliases[name])
        return self.env.get(node.id, UNKNOWN)

    def _attr_ty(self, node) -> Any:
        owner = self.resolve(node.value)
        if owner is UNKNOWN:
            return UNKNOWN
        kind, payload = owner
        if kind == CFG:
            return self._field_ty(payload, node.attr)
        if kind == CLS:
            return self.index.class_attrs.get(payload, {}).get(node.attr, UNKNOWN)
        return UNKNOWN

    def _field_ty(self, models, attr) -> Any:
        out, hit = set(), False
        for model in models:
            if attr in self.index.field_models.get(model, {}):
                hit = True
                out |= self.index.field_models[model][attr]
            elif attr in self.index.class_attrs.get(model, {}):
                return self.index.class_attrs[model][attr]
        return _cfg(out) if hit else UNKNOWN

    def _call_ty(self, node) -> Any:  # noqa: PLR0911 - one return per callable kind
        func = node.func
        if isinstance(func, ast.Name):
            name = self.imports.get(func.id, func.id)
            if name in self.index.models:
                return _cfg({name})
            if name in self.index.aliases:
                return _cfg(self.index.aliases[name])
            if name in self.index.other_classes:
                return (CLS, name)
            # `replace(policy, enabled=False)` is still a policy; `next(iter(d.values()))`
            # is still one of d's values. Containers and their elements are deliberately
            # not distinguished -- a config never holds a config of a DIFFERENT model in
            # a container, so collapsing them costs nothing and resolves a lot.
            if name in {"replace", "next", "iter", "deepcopy", "copy"} and node.args:
                return self.resolve(node.args[0])
            if name == "getattr":
                return self._getattr_ty(node)
            if name in BUILTINS or name in CONTAINERS:
                return PLAIN_TY
            return self.index.returns.get(name, UNKNOWN)
        if isinstance(func, ast.Attribute):
            return self._method_ty(func)
        return UNKNOWN

    def _method_ty(self, func) -> Any:
        base = self.resolve(func.value)
        if base is UNKNOWN:
            return UNKNOWN
        if func.attr in {"model_copy", "model_validate", "validate_python"} and base[0] == CFG:
            return base
        if func.attr in {"values", "copy"}:
            return base  # a dict of cameras yields cameras
        if base[0] == CLS:
            return self.index.method_returns.get((base[1], func.attr), UNKNOWN)
        if base[0] == CFG:
            for model in base[1]:
                got = self.index.method_returns.get((model, func.attr))
                if got is not None:
                    return got
        return UNKNOWN

    def _getattr_ty(self, node) -> Any:
        """`getattr(canteen, role, None)` where `role` is one of three literal strings."""
        if not node.args:
            return UNKNOWN
        base = self.resolve(node.args[0])
        names = self._attr_names(node)
        if base is UNKNOWN or base[0] != CFG or names is None:
            return UNKNOWN
        out, hit = set(), False
        for model in base[1]:
            for name in names:
                if name in self.index.field_models.get(model, {}):
                    hit = True
                    out |= self.index.field_models[model][name]
        return _cfg(out) if hit else UNKNOWN

    def _attr_names(self, node) -> frozenset[str] | None:
        """The set of attribute names a `getattr` call could be asking for."""
        if len(node.args) < 2:
            return None
        resolved = self.resolve(node.args[1])
        if resolved is not UNKNOWN and resolved[0] == STRS:
            return resolved[1]
        return None

    # -- scope binding -----------------------------------------------------
    def bind(self, target, ty) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = ty
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self.bind(element, UNKNOWN)

    def bind_assign(self, target, value) -> None:
        """`weights, metrics = config.weights, config.metrics` binds BOTH, positionally.

        Typing the tuple as one lump loses every element. This idiom is everywhere in the
        detector, so losing it means losing most of `ScoreWeights` and `PairMetrics` --
        which the scan would then have to report as blind spots it need not have had.
        """
        if (
            isinstance(target, ast.Tuple | ast.List)
            and isinstance(value, ast.Tuple | ast.List)
            and len(target.elts) == len(value.elts)
        ):
            for element, item in zip(target.elts, value.elts, strict=True):
                self.bind_assign(element, item)
            return
        self.bind(target, self.resolve(value))

    def bind_iter(self, target, iterable) -> None:
        """`for camera in cameras:`, `for name, camera in cameras.items():`, ..."""
        literal = _string_set(iterable)
        if literal is not None:
            return self.bind(target, (STRS, literal))
        unwrapped = self._unwrap(target, iterable)
        if unwrapped is None:
            return None
        base, kind = unwrapped
        ty = self.resolve(base)
        if kind == "items" and isinstance(target, ast.Tuple) and len(target.elts) == 2:
            self.bind(target.elts[0], PLAIN_TY)
            self.bind(target.elts[1], ty)
        elif kind == "keys":
            self.bind(target, PLAIN_TY)
        else:
            self.bind(target, ty)
        return None

    def _unwrap(self, target, iterable):
        """Peel `sorted(...)` / `enumerate(...)` / `.items()` off an iterable."""
        wrappers = {"sorted", "reversed", "list", "tuple", "set", "enumerate"}
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Name)
            and iterable.func.id in wrappers
            and iterable.args
        ):
            if (
                iterable.func.id == "enumerate"
                and isinstance(target, ast.Tuple)
                and len(target.elts) == 2
            ):
                self.bind(target.elts[0], PLAIN_TY)
                self.bind_iter(target.elts[1], iterable.args[0])
                return None
            self.bind_iter(target, iterable.args[0])
            return None
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Attribute)
            and iterable.func.attr in {"items", "values", "keys"}
        ):
            return iterable.func.value, iterable.func.attr
        return iterable, "iter"

    def seed(self, body) -> None:
        """Type every local name this scope binds. Repeated: later lines inform earlier."""
        module = ast.Module(body=list(body), type_ignores=[])
        for _ in range(3):
            for node in ast.walk(module):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    self.env[node.target.id] = self.index.ty_of_annotation(
                        node.annotation, self.imports
                    )
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.bind_assign(target, node.value)
                elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
                    self.bind_iter(node.target, node.iter)
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars is not None:
                            self.bind(item.optional_vars, self.resolve(item.context_expr))

    # -- walking -----------------------------------------------------------
    def visit_FunctionDef(self, node) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._function(node)

    def _function(self, node) -> None:
        env = dict(self.env)
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            env[arg.arg] = self.index.ty_of_annotation(arg.annotation, self.imports)
        for decorator in node.decorator_list:
            self.visit(decorator)
        inner = _Resolver(self.path, env, self.imports, self.index, self.report, self.cls)
        inner.seed(node.body)
        for stmt in node.body:
            inner.visit(stmt)

    def visit_ClassDef(self, node) -> None:
        inner = _Resolver(self.path, self.env, self.imports, self.index, self.report, node.name)
        for stmt in node.body:
            # `speed_threshold: float = Field(...)` in a class body DECLARES the key.
            # A declaration is not a use -- that is the whole distinction being drawn.
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is not None:
                    inner.visit(stmt.value)
                continue
            inner.visit(stmt)

    def visit_Attribute(self, node) -> None:
        owner = self.resolve(node.value)
        if owner is UNKNOWN:
            base = self._untyped(node.value, {node.attr})
            if node.attr in self.index.field_names:
                self._note(node.attr, node, "on a base the scan could not type", base=base)
        elif owner[0] == CFG:
            for model in owner[1]:
                if node.attr in self.index.fields.get(model, ()):
                    self._read(model, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "getattr":
            self._visit_getattr(node)
        if isinstance(func, ast.Attribute) and func.attr in {"model_dump", "model_dump_json"}:
            self._visit_dump(node, func)
        self.generic_visit(node)

    def _visit_getattr(self, node) -> None:
        """`getattr(x, name)` is an attribute access; how much it tells us varies.

        Base and name can each be known or not, and only "neither" is hopeless.
        """
        if not node.args:
            return
        base = self.resolve(node.args[0])
        names = self._attr_names(node)
        if names is not None:
            self._getattr_named(node, base, names)
        elif base is not UNKNOWN and base[0] == CFG:
            for model in base[1]:  # dynamic name, but the owner is known -- so name it
                for name in self.index.fields[model]:
                    self._note(name, node, f"getattr({model}, <dynamic name>)", owners={model})
        elif base is UNKNOWN:
            self._at(node, "getattr() on a base the scan could not type, with a dynamic name")

    def _getattr_named(self, node, base, names) -> None:
        """The attribute NAME is known, so this is an ordinary attribute access."""
        if base is UNKNOWN:
            key = self._untyped(node.args[0], names)
            for name in names & self.index.field_names:
                self._note(name, node, "getattr() on a base the scan could not type", base=key)
            return
        for model in base[1] if base[0] == CFG else ():
            for name in names:
                if name in self.index.fields.get(model, ()):
                    self._read(model, name)

    def _visit_dump(self, node, func) -> None:
        base = self.resolve(func.value)
        if base is not UNKNOWN and base[0] == CFG:
            for model in base[1]:
                for name in self.index.fields[model]:
                    why = f"{model}.{func.attr}() -- every field flows out"
                    self._note(name, node, why, owners={model})
        elif base is UNKNOWN:
            self._at(node, "model_dump() on a base the scan could not type")

    # -- recording ---------------------------------------------------------
    def _read(self, model: str, name: str) -> None:
        if self.report is not None:
            self.report.reads |= self.index.kin[(model, name)]

    def _untyped(self, expr, attrs: set[str]) -> str:
        """Record the SHAPE of a base the scan could not type: what is read off it.

        Every attribute, including ones no config declares -- `args.clips` is exactly the
        evidence that `args.device` is not a `WorkerGroup`. See `config_report`. Bases
        with the same source text in one file merge, which only widens the set, and a
        wider set matches fewer models: erring towards making a human look.
        """
        key = f"{self.path.as_posix()}::{ast.unparse(expr)}"
        if self.report is not None:
            self.report.untyped_attrs.setdefault(key, set()).update(attrs)
        return key

    def _note(self, name: str, node, why: str, *, base=None, owners=None) -> None:
        """An access this scan cannot attribute; WHOSE key it is, `config_report` decides.

        `owners`: the model is proven, only the NAME was in doubt; `base`: neither is.
        """
        if self.report is not None:
            site = f"{self._at_str(node)} {why}"
            self.report.notes.append(
                Note(name, site, base, frozenset(owners) if owners else None)
            )

    def _at(self, node, why: str) -> None:
        if self.report is not None:
            self.report.dynamic.add(f"{self._at_str(node)} {why}")

    def _at_str(self, node) -> str:
        return f"{self.path.as_posix()}:{getattr(node, 'lineno', 0)}"


def index_self_assignments(index: Index, trees: dict[Path, ast.Module]) -> None:
    """`self._policy = quality or QualityPolicy()` -- resolve the RHS to type the attr.

    Lives here rather than in `config_index` because it needs the resolver, and the
    resolver needs the index. Annotations first, then this.
    """
    for path, tree in trees.items():
        imports = _imports_of(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            attrs = index.class_attrs.setdefault(node.name, {})
            for stmt in node.body:
                if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                args = stmt.args
                env = {
                    a.arg: index.ty_of_annotation(a.annotation, imports)
                    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
                }
                probe = _Resolver(path, env, imports, index, None, node.name)
                _record_self_types(probe, stmt, attrs)


def _record_self_types(probe: _Resolver, stmt: ast.stmt, attrs: dict) -> None:
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and attrs.get(target.attr) is None
            ):
                resolved = probe.resolve(node.value)
                if resolved is not UNKNOWN:
                    attrs[target.attr] = resolved


def scan(src_dir: Path) -> tuple[Index, Report]:
    """Every config field in `src_dir`, sorted into read / unresolved / neither."""
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(src_dir.rglob("*.py"))
    }
    index = build_index(trees)
    index_self_assignments(index, trees)

    report = Report()
    for path, tree in trees.items():
        imports = _imports_of(tree)
        resolver = _Resolver(path.relative_to(src_dir), {}, imports, index, report)
        resolver.seed(tree.body)
        for stmt in tree.body:
            resolver.visit(stmt)
    report.reachable = _reachable(index, report.reads)
    # Only now, with every base's full shape known, can a blind spot say WHOSE key it is.
    report.unresolved = attribute_blind_spots(index, report.notes, report.untyped_attrs)
    return index, report

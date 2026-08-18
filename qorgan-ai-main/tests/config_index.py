"""The shape of the tree: which class owns which attribute, and of what type.

This is the lookup table `config_scan` walks over to answer "whose `max_speed` is this?".
It knows three kinds of thing:

  * every pydantic model under `qorgan.config`, its fields, and which of those fields
    hold OTHER config models (that parent->child link is what makes a dotted chain like
    `.gates.proximity_only.max_speed` resolvable);
  * every other class defined in `src/`, and the declared type of its attributes -- so
    `data.config` can be followed from `data: GateInput` to `BullyingConfig`, and so
    `pair.max_speed` can be recognised as NOT a config read;
  * module-level type aliases (`CameraConfig = Annotated[BullyingCamera | ...]`) and
    import aliases (`from ... import Counters as CounterConfig`).

Everything here is derived from annotations and pydantic's own `model_fields`. Nothing
is guessed: an annotation this module cannot place becomes UNKNOWN, which `config_scan`
reports rather than assumes.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import qorgan.config

# -- the type domain the resolver works in ---------------------------------
# A resolved type is one of:
#   ("cfg", frozenset[str])  -- one or more config models (a union, e.g. CameraConfig)
#   ("cls", str)             -- a non-config class defined in src/
#   ("strs", frozenset[str]) -- a value known to be one of these literal strings
#   ("plain", None)          -- definitely not a config object (a builtin, a literal)
#   None                     -- UNKNOWN. The scan does not know, and will say so.
CFG, CLS, STRS, PLAIN = "cfg", "cls", "strs", "plain"
UNKNOWN = None
PLAIN_TY = (PLAIN, None)

CONTAINERS = frozenset(
    """list List set Set dict Dict tuple Tuple Optional Annotated Sequence Iterable Iterator
    Mapping MutableMapping frozenset Final ClassVar Union Literal Field Callable Generator
    AsyncIterator deque Counter defaultdict type""".split()
)
BUILTINS = frozenset(
    """str int float bool bytes None Any object Path datetime date timedelta Decimal UUID
    ndarray Session Engine SecretStr""".split()
)


def _cfg(names) -> tuple:
    names = frozenset(names)
    return (CFG, names) if names else PLAIN_TY


def _config_models() -> dict[str, type[BaseModel]]:
    """Every pydantic model declared under `qorgan.config`."""
    out: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(qorgan.config.__path__):
        module = importlib.import_module(f"qorgan.config.{info.name}")
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__.startswith("qorgan.config.")
            ):
                out[obj.__name__] = obj
    return out


def _models_in(annotation: Any, known: dict[str, type[BaseModel]]) -> frozenset[str]:
    """The config models reachable inside an annotation: `X | None`, `list[X]`, ..."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if inspect.isclass(node) and issubclass(node, BaseModel) and node.__name__ in known:
            found.add(node.__name__)
        for arg in getattr(node, "__args__", ()) or ():
            walk(arg)

    walk(annotation)
    return frozenset(found)


def _leaves(node: ast.expr | None) -> list[str]:  # noqa: PLR0911 - one return per AST node kind
    """Every bare name inside an annotation expression."""
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            try:
                return _leaves(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return []
        return ["None"] if node.value is None else []
    if isinstance(node, ast.BinOp):
        return _leaves(node.left) + _leaves(node.right)
    if isinstance(node, ast.Subscript):
        return _leaves(node.value) + _leaves(node.slice)
    if isinstance(node, ast.Tuple | ast.List):
        return [leaf for element in node.elts for leaf in _leaves(element)]
    return []


def _string_set(node: ast.expr | None) -> frozenset[str] | None:
    """`("entry", "exit", "inside")` -> those three strings. Anything else -> None."""
    if not isinstance(node, ast.Tuple | ast.List | ast.Set):
        return None
    out: set[str] = set()
    for element in node.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            return None
        out.add(element.value)
    return frozenset(out)


def _imports_of(tree: ast.Module) -> dict[str, str]:
    """`from x import Counters as CounterConfig` -> {"CounterConfig": "Counters"}."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if alias.asname:
                    out[alias.asname] = alias.name.split(".")[-1]
    return out


@dataclass
class Index:
    """Everything the resolver needs to know about the shape of the tree."""

    models: dict[str, type[BaseModel]]
    fields: dict[str, set[str]]
    field_models: dict[str, dict[str, frozenset[str]]]
    kin: dict[tuple[str, str], set[tuple[str, str]]]
    aliases: dict[str, frozenset[str]]
    other_classes: set[str]
    class_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    returns: dict[str, Any] = field(default_factory=dict)
    method_returns: dict[tuple[str, str], Any] = field(default_factory=dict)

    @property
    def field_names(self) -> set[str]:
        return {name for names in self.fields.values() for name in names}

    def ty_of_annotation(self, node: ast.expr | None, imports: dict[str, str]) -> Any:
        if node is None:
            return UNKNOWN
        leaves = [imports.get(name, name) for name in _leaves(node)]
        if not leaves:
            return UNKNOWN
        found = {name for name in leaves if name in self.models}
        for name in leaves:
            found |= self.aliases.get(name, frozenset())
        if found:
            return _cfg(found)
        classes = [name for name in leaves if name in self.other_classes]
        if classes:
            return (CLS, classes[0])
        if all(name in BUILTINS or name in CONTAINERS for name in leaves):
            return PLAIN_TY
        return UNKNOWN


def _build_class_index(index: Index, trees: dict[Path, ast.Module]) -> None:
    """Type every attribute of every non-config class, so `data.config` resolves."""
    for tree in trees.values():
        imports = _imports_of(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                index.returns.setdefault(node.name, index.ty_of_annotation(node.returns, imports))
            if not isinstance(node, ast.ClassDef):
                continue
            attrs = index.class_attrs.setdefault(node.name, {})
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    attrs[stmt.target.id] = index.ty_of_annotation(stmt.annotation, imports)
                elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    _index_method(index, node.name, stmt, attrs, imports)


def _index_method(index: Index, cls: str, stmt: ast.stmt, attrs: dict, imports: dict) -> None:
    returns = index.ty_of_annotation(stmt.returns, imports)
    index.method_returns[(cls, stmt.name)] = returns
    if any(
        isinstance(d, ast.Name) and d.id in {"property", "cached_property"}
        for d in stmt.decorator_list
    ):
        attrs[stmt.name] = returns
    for node in ast.walk(stmt):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id == "self"
        ):
            attrs[node.target.attr] = index.ty_of_annotation(node.annotation, imports)


def build_index(trees: dict[Path, ast.Module]) -> Index:
    """The annotation-derived shape of the tree.

    `self.x = <expression>` needs the resolver to type its right-hand side, so that pass
    lives in `config_scan.index_self_assignments` and runs immediately after this.
    """
    models = _config_models()
    fields = {name: set(model.model_fields) for name, model in models.items()}
    field_models = {
        name: {f: _models_in(i.annotation, models) for f, i in model.model_fields.items()}
        for name, model in models.items()
    }
    # A read of `Sub.f` IS a read of `Base.f`: pydantic gives the subclass the same field.
    kin = {
        (name, f): {
            (other, f)
            for other, om in models.items()
            if f in fields[other] and (issubclass(om, model) or issubclass(model, om))
        }
        for name, model in models.items()
        for f in fields[name]
    }
    other_classes = {
        node.name
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name not in models
    }
    index = Index(models, fields, field_models, kin, {}, other_classes)
    _build_aliases(index, trees)
    _build_class_index(index, trees)
    return index


def _build_aliases(index: Index, trees: dict[Path, ast.Module]) -> None:
    """`CameraConfig = Annotated[BullyingCamera | CanteenCamera, Field(...)]`."""
    for tree in trees.values():
        for stmt in tree.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                continue
            leaves = _leaves(stmt.value)
            found = {n for n in leaves if n in index.models}
            if found and all(
                n in index.models or n in CONTAINERS or n in BUILTINS for n in leaves
            ):
                index.aliases[stmt.targets[0].id] = frozenset(found)


def _reachable(index: Index, reads: set[tuple[str, str]]) -> set[str]:
    """A model counts only if a root camera model reaches it through fields that are READ.

    `BullyingConfig.violence` is read nowhere, so no `ViolenceSettings` instance is ever
    consulted -- and its own validator reading its own fields does not change that.
    """
    parents: dict[str, set[tuple[str, str]]] = {}
    for parent, attrs in index.field_models.items():
        for name, children in attrs.items():
            for child in children:
                parents.setdefault(child, set()).add((parent, name))

    reachable = {name for name in index.models if name not in parents}
    growing = True
    while growing:
        growing = False
        for child, links in parents.items():
            if child in reachable:
                continue
            if any(parent in reachable and (parent, name) in reads for parent, name in links):
                reachable.add(child)
                growing = True
    return reachable

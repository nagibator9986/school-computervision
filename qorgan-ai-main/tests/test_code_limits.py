"""Rule R1, enforced.

The legacy `run_canteen()` is a single 6330-line function with 1074 local variables
and six nested closures. Nobody set out to write that; it grew one reasonable-looking
addition at a time, because nothing ever said no.

This test says no.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, SRC_DIR

MAX_FILE_LINES = 500
MAX_FUNCTION_LINES = 50

# Alembic writes these; their length is not a design choice we control.
EXCLUDED = {"migrations/versions", "migrations/env.py"}


def _python_files(root: Path) -> list[Path]:
    return [
        path for path in root.rglob("*.py") if not any(part in path.as_posix() for part in EXCLUDED)
    ]


ALL_FILES = _python_files(SRC_DIR) + _python_files(REPO_ROOT / "tests")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_file_is_under_the_line_limit(path: Path) -> None:
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_FILE_LINES, (
        f"{path.relative_to(REPO_ROOT)} is {lines} lines (limit {MAX_FILE_LINES}). "
        "Split it before it becomes a god-file."
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_every_function_is_under_the_line_limit(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        (node.name, node.end_lineno - node.lineno + 1)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.end_lineno is not None
        and node.end_lineno - node.lineno + 1 > MAX_FUNCTION_LINES
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)}: function(s) over {MAX_FUNCTION_LINES} lines: "
        + ", ".join(f"{name} ({length})" for name, length in offenders)
    )

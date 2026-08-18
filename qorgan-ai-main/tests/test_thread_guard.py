"""The guard that says no test may leave a worker running, and proof that it works.

A guard nobody has watched fail is a guard nobody knows the state of. Two things are
checked here, and they are different questions:

  * **Does it catch a leak?** Answered by running a real pytest, in a subprocess, over a
    test that deliberately leaks. Asserting on the fixture's internals would only prove
    the internals; what matters is that a leaking test comes back RED.
  * **Does it still cover everything?** Answered by PARSING `src/` for every thread the
    project constructs and checking the prefix list accounts for it. An allow-list that
    somebody must remember to update is one that silently stops covering things -- which
    is the shape of defect this whole guard exists to catch.

**Why the second check parses instead of grepping.** The first version matched
`threading.Thread(...name="x")` with a regex. Measured against the tree it found 7 names
for 7 constructor calls, so it was right -- but only because everybody had so far
remembered to pass `name=`. A thread started without one is not reported by such a regex;
it is simply not seen, and the allow-list stops being exhaustive with nothing going red.
Reading the syntax tree turns that around: every `Thread(...)` is found first, and then
each one is REQUIRED to carry a name. Forgetting the name is now the failure.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from tests.conftest import SRC_DIR, WORKER_THREAD_PREFIXES


def test_the_guard_fails_a_test_that_leaks_a_worker(tmp_path: Path) -> None:
    """**The guard, proven by watching it fail.**

    Run in a subprocess against a throwaway test file: a leaking test must come back red,
    and the message must name the thread, because "something leaked" sends the next person
    hunting while "notifier" sends them to the file that started it.
    """
    leaky = tmp_path / "test_leaky.py"
    leaky.write_text(
        "import threading, time\n"
        "def test_leaks():\n"
        "    t = threading.Thread(target=lambda: time.sleep(30), name='notifier', daemon=True)\n"
        "    t.start()\n",
        encoding="utf-8",
    )
    conftest = tmp_path / "conftest.py"
    conftest.write_text(
        f"import sys; sys.path.insert(0, {str(SRC_DIR.parent)!r})\n"
        "from tests.conftest import _no_worker_thread_outlives_its_test  # noqa: F401\n",
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603 - our own interpreter, literal argv
        [sys.executable, "-m", "pytest", str(leaky), "-p", "no:cacheprovider", "-q"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**_env(), "PYTHONPATH": str(SRC_DIR)},
        timeout=120, check=False,
    )

    assert result.returncode != 0, (
        "a test that started a thread and walked away was reported as passing:\n"
        f"{result.stdout[-2000:]}"
    )
    assert "notifier" in result.stdout, (
        f"the failure did not name the leaked thread:\n{result.stdout[-2000:]}"
    )


def test_the_guard_passes_a_test_that_cleans_up(tmp_path: Path) -> None:
    """The other half. A guard that failed everything would satisfy the test above and
    make the suite useless -- and the pressure would then be to delete the guard."""
    tidy = tmp_path / "test_tidy.py"
    tidy.write_text(
        "import threading\n"
        "def test_stops_what_it_starts():\n"
        "    stop = threading.Event()\n"
        "    t = threading.Thread(target=stop.wait, name='notifier', daemon=True)\n"
        "    t.start()\n"
        "    stop.set()\n"
        "    t.join(timeout=5)\n",
        encoding="utf-8",
    )
    conftest = tmp_path / "conftest.py"
    conftest.write_text(
        f"import sys; sys.path.insert(0, {str(SRC_DIR.parent)!r})\n"
        "from tests.conftest import _no_worker_thread_outlives_its_test  # noqa: F401\n",
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603 - our own interpreter, literal argv
        [sys.executable, "-m", "pytest", str(tidy), "-p", "no:cacheprovider", "-q"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**_env(), "PYTHONPATH": str(SRC_DIR)},
        timeout=120, check=False,
    )

    assert result.returncode == 0, (
        f"a test that stopped its own thread was failed by the guard:\n{result.stdout[-2000:]}"
    )


def _thread_constructions(tree: ast.AST) -> Iterator[tuple[int, str | None]]:
    """Every thread this file constructs, as (line, name prefix). `None` means no `name=`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_a_thread(node.func):
            yield node.lineno, _name_prefix(node)


def _is_a_thread(func: ast.expr) -> bool:
    """`threading.Thread(...)`, and also a bare `Thread(...)` from `from threading import`.

    The project only uses the first spelling today. Accepting both means the check does not
    quietly go blind the day somebody changes the import.
    """
    if isinstance(func, ast.Attribute):
        return func.attr == "Thread"
    return isinstance(func, ast.Name) and func.id == "Thread"


def _name_prefix(call: ast.Call) -> str | None:
    """The constant leading part of `name=`; `""` if it is computed; `None` if absent.

    Five of the seven kinds interpolate a camera or group -- `f"camera:{self.camera}"` --
    so what the guard can check is the literal head of the f-string. A name built any other
    way yields `""`, which matches no prefix and is therefore reported: a thread whose name
    cannot be read here is a thread nobody can promise the guard watches.
    """
    for keyword in call.keywords:
        if keyword.arg != "name":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.JoinedStr) and value.values:
            head = value.values[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                return head.value
        return ""
    return None


def test_every_thread_the_project_starts_is_covered_by_the_guard() -> None:
    """**The check that stops this going stale.**

    The guard matches on name prefixes, so a worker kind added next year with a new name
    would simply not be watched, and nothing would say so. This parses the source instead:
    every thread `src/` constructs must be named, and the name must begin with a prefix the
    guard knows.
    """
    named: dict[str, str] = {}
    unnamed: list[str] = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, prefix in _thread_constructions(tree):
            where = f"{path.relative_to(SRC_DIR)}:{lineno}"
            if prefix is None:
                unnamed.append(where)
            else:
                named[where] = prefix

    assert named or unnamed, "no threads found at all -- the scan has stopped working"

    # Strictly the worse case, so it is reported first: an unnamed thread gets a name like
    # "Thread-7" and NO prefix list can ever be written that watches it.
    assert not unnamed, (
        f"these threads are started without `name=`: {unnamed}. Give each a name beginning "
        f"with one of {WORKER_THREAD_PREFIXES}, or the guard cannot see it leak."
    )

    uncovered = {
        where: prefix
        for where, prefix in named.items()
        if not prefix.startswith(tuple(WORKER_THREAD_PREFIXES))
    }
    assert not uncovered, (
        "these threads are started by the project and NOT watched by the guard in "
        f"tests/conftest.py: {uncovered}. Add the prefix to WORKER_THREAD_PREFIXES, or the "
        "next leak of this kind goes unnoticed until it fails somebody else's test."
    )


def _env() -> dict[str, str]:
    import os

    # Inherit, minus anything that would point the subprocess at this run's database.
    return {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}

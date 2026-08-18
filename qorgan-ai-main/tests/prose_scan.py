"""The scanner behind `test_the_prose_and_the_parser_agree.py`. That file says WHY.

Split out under R1's 500-line limit, the way `tenancy_scan.py` and `config_scan.py` are
split from the tests that use them. Everything here is mechanism; every judgement, every
measurement and the scope argument live in the test module's docstring.

**One rule about this file's own prose, because the scanner reads it too:** a comment here
may not spell out a false CLI claim as an illustration. Both directions' examples live in
the test module's regression tuples, where they are string constants and therefore code.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path

from qorgan.cli import build_parser
from tests.conftest import REPO_ROOT, SRC_DIR

TEMPLATES = SRC_DIR / "qorgan" / "web" / "templates"

CommandPath = tuple[str, ...]

# -- The parser, as it is actually built. Never a list. ------------------------


def _parser_tree() -> dict[CommandPath, tuple[frozenset[str], dict[str, CommandPath]]]:
    """Every command path the real parser offers, with the flags each one accepts.

    Walked off `build_parser()` rather than written down, because a hand-written list is
    the second copy this whole check exists to stop. `_SubParsersAction` is private argparse
    API and the only way to enumerate subcommands; if argparse drops it this raises rather
    than quietly finding no commands, and the parser vacuity test catches the rest.
    """
    nodes: dict[CommandPath, tuple[frozenset[str], dict[str, CommandPath]]] = {}

    def walk(parser: argparse.ArgumentParser, path: CommandPath) -> None:
        subs: dict[str, CommandPath] = {}
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    subs[name] = (*path, name)
                    walk(sub, (*path, name))
        flags = frozenset(
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--")
        )
        nodes[path] = (flags, subs)

    walk(build_parser(), ())
    return nodes


NODES = _parser_tree()
EVERY_FLAG = frozenset(flag for flags, _ in NODES.values() for flag in flags)

# -- Prose, and the code spans in it -------------------------------------------

# Jinja statements and expressions are not prose. Jinja COMMENTS are kept, and that is a
# deliberate difference from `test_the_pages_promise_only_what_exists.py`, which strips them
# because nobody is shown one. This check is not about what a reader is told but about
# whether a claim matches the parser, and a developer comment naming a phantom command is
# the same defect wherever it sits -- two of the four instances were comments.
JINJA_CODE = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)

BACKTICK = re.compile(r"`([^`\n]+)`")
CODE_TAG = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

# Markdown emphasis is dropped before anything is matched: a correctness fix, not tidying.
# This tree bolds mid-sentence, and the first run of this scanner MISSED a denial because a
# `**` landed inside the phrase, so the literal it looks for was not there and a correctly
# worded epitaph read as a live claim. The remedy that offered a maintainer was "remove your
# bold" -- the reword-something-unrelated red this check must never produce.
EMPHASIS = re.compile(r"\*+")

# **A run of two or more backticks is a markdown ESCAPE, not a code span**, and collapsing
# it is the difference between a scanner that reads a file and one that reads its first
# paragraph. Backtick pairing is sequential, so one escape desynchronises every pair after
# it -- for the WHOLE file, once the file was flattened into a single string. Measured: the
# check's own module found **zero** `--school` spans in itself, because an escape sat in its
# docstring; six real reds across two files were invisible behind it, one of them introduced
# in the same round that looked for it. Fixed twice over -- runs are collapsed here, and
# prose is scanned PIECE BY PIECE below so a stray backtick can only blind its own comment.
BACKTICK_RUN = re.compile(r"``+")


def flatten(text: str) -> str:
    """One line, no emphasis, no escapes: a window around a span is then a fixed distance."""
    return EMPHASIS.sub("", WHITESPACE.sub(" ", BACKTICK_RUN.sub(" ", text)))


def python_prose(path: Path) -> list[str]:
    """Docstrings and comments, **each kept separate**. Code is not prose.

    Separate because pairing is sequential: one unbalanced backtick would otherwise blind
    the rest of the file. A piece is the largest unit inside which a stray tick may do harm.

    The docstring/comment split also matters on its own: without it the scanner reads
    `add_user.add_argument("--role", ...)` and the `cli.main(["user", "add", ...])` calls in
    the tests as claims, which is how a checker ends up agreeing with whatever the code says.
    """
    text = path.read_text(encoding="utf-8")
    pieces: list[str] = []
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                pieces.append(doc)
    pieces += [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT
    ]
    return [flatten(piece) for piece in pieces]


def template_prose(path: Path) -> list[str]:
    return [flatten(JINJA_CODE.sub(" ", path.read_text(encoding="utf-8")))]


def sources() -> list[tuple[str, Path, list[str]]]:
    """(leg, path, prose pieces). The test module's docstring says what is NOT here.

    The leg is carried rather than totalled away: see the per-leg vacuity test for the
    blindness that a union hid.
    """
    found = [("src", p, python_prose(p)) for p in sorted(SRC_DIR.rglob("*.py"))]
    found += [("tests", p, python_prose(p)) for p in sorted((REPO_ROOT / "tests").rglob("*.py"))]
    found += [("templates", p, template_prose(p)) for p in sorted(TEMPLATES.rglob("*.html"))]
    return found


SOURCES = sources()

# Per leg: fewest files, fewest `qorgan` invocations, fewest real-command spans, and one file
# that must be among them. Floors sit under the measured 165/131/18 files and 46/47/2
# invocations with headroom; the NAMED FILE is what makes a floor more than a number, since
# any floor is satisfied by a hundred files whose extraction returned "".
LEG_FLOORS: dict[str, tuple[int, int, int, str]] = {
    "src": (100, 20, 20, "cli.py"),
    "tests": (100, 20, 20, "test_the_superadmin_can_be_created.py"),
    "templates": (10, 1, 1, "schools.html"),
}


def code_spans(text: str) -> list[tuple[str, int, int]]:
    """Every code span, with where it sits, so polarity can be judged around it.

    Both markups are read in both kinds of file: templates carry `<code>` for the reader and
    backticks inside Jinja comments for the developer, and a phantom is a phantom in either.
    """
    spans = [
        (WHITESPACE.sub(" ", m.group(1)).strip(), m.start(1), m.end(1))
        for pattern in (BACKTICK, CODE_TAG)
        for m in pattern.finditer(text)
    ]
    return [s for s in spans if s[0]]


# -- Resolving a claim against the parser --------------------------------------

# `<name>`, `<slug>`, `...`, `…` -- somebody writing an example, not naming a subcommand.
# The unicode ellipsis is here because it was missing: this tree writes `qorgan …` and the
# scanner read the ellipsis as a subcommand and reported the tree's own prose as a phantom.
PLACEHOLDER = re.compile(r"^<.+>$|^\.\.\.$|^…$")


def unmet_claim(span: str) -> str | None:
    """What this span promises that the parser does not offer, or None.

    Resolution stops at the first flag, placeholder, or node with no subcommands -- so
    `qorgan user add alice --role admin` resolves to `user add`, treats `alice` as the
    positional it is, and then checks `--role`. Only two things are ever reported: a word
    where a subcommand was required and none matched, and a flag the resolved command does
    not accept.
    """
    tokens = span.split()
    if tokens[:1] != ["qorgan"]:
        return None

    tokens = tokens[1:]
    path: CommandPath = ()
    index = 0
    while index < len(tokens):
        word = tokens[index]
        if word.startswith("-") or PLACEHOLDER.match(word):
            break
        subcommands = NODES[path][1]
        if not subcommands:
            break  # positional territory: this command takes arguments, not subcommands
        if word not in subcommands:
            return (
                f"`qorgan {' '.join((*path, word))}` is not a command. "
                f"`qorgan {' '.join(path)}`".replace("`qorgan `", "`qorgan`")
                + f" offers: {', '.join(sorted(subcommands))}"
            )
        path = subcommands[word]
        index += 1

    flags = NODES[path][0]
    for word in tokens[index:]:
        if word.startswith("--") and word not in flags:
            where = " ".join(path) or "(no subcommand)"
            return (
                f"`qorgan {where}` has no flag `{word}`. It accepts: "
                f"{', '.join(sorted(flags))}"
            )
    return None


def is_real(span: str) -> bool:
    """Does the parser actually offer this exact flag or this exact command?"""
    if span in EVERY_FLAG:
        return True
    return span.split()[:1] == ["qorgan"] and unmet_claim(span) is None


# -- Polarity ------------------------------------------------------------------

# **Split by POSITION, and the split is load-bearing rather than tidy.** A word that denies
# a thing precedes it; a phrase that denies it follows it. Measured on the real tree: one
# docstring lists the flag, a log line and a forthcoming URL in a single sentence, where the
# forward-looking word belongs to the URL thirty characters downstream. A window that looked
# both ways reddened it, and its only answer would have been to reword a clause about
# something else -- precisely the red that gets a rule deleted. Worked examples of both
# positions are in the test module's regression tuples, as constants rather than as prose.
DENIALS_BEFORE = (
    "future", "planned", "no such", "there is no", "there was no",
    "нет такой", "не будет", "будущ",
)
DENIALS_AFTER = (
    "пока нет", "ещё нет", "еще нет", "не существует", "отсутствует",
    "does not exist", "doesn't exist", "never existed", "existed nowhere",
    "is not implemented", "does not yet exist",
)

# A denial has to be denying a COMMAND or a FLAG. Without this the scanner read "there was no
# way to say which school" and "for the person who has no terminal" as denials of the flags
# beside them: 27 reds, 25 of them nonsense. Requiring the noun is what took it to zero.
CLI_NOUNS = ("flag", "command", "option", "subcommand", "флаг", "команд", "ключ")

# How far a denial may sit from the thing it denies. A JUDGEMENT, not a measurement -- the
# same honest position `NEGATION_WINDOW` takes in the sister check.
DENIAL_WINDOW = 40
NOUN_WINDOW = 60


def denied_at(text: str, start: int, end: int) -> list[str]:
    """The denial phrases that are denying the span at [start, end), if any."""
    low = text.lower()
    hits = [d for d in DENIALS_BEFORE if d in low[max(0, start - DENIAL_WINDOW) : start]]
    hits += [d for d in DENIALS_AFTER if d in low[end : end + DENIAL_WINDOW]]
    if not hits:
        return []
    around = low[max(0, start - NOUN_WINDOW) : end + NOUN_WINDOW]
    return hits if any(noun in around for noun in CLI_NOUNS) else []


def _quote(text: str, start: int, end: int) -> str:
    return "..." + text[max(0, start - 90) : end + 90].strip() + "..."


# -- The two directions --------------------------------------------------------


def phantoms(text: str) -> list[tuple[str, str]]:
    """Commands and flags named POSITIVELY that the parser does not offer."""
    return [
        (problem, _quote(text, start, end))
        for span, start, end in code_spans(text)
        if (problem := unmet_claim(span)) and not denied_at(text, start, end)
    ]


def false_denials(text: str) -> list[tuple[str, str]]:
    """Commands and flags the parser DOES offer, written up as not existing."""
    return [
        (f"`{span}` exists, but is denied by {hits}", _quote(text, start, end))
        for span, start, end in code_spans(text)
        if is_real(span) and (hits := denied_at(text, start, end))
    ]


def scan(finder) -> list[tuple[Path, str, str]]:
    """Run one direction over every piece of every source. Pieces, never a whole file."""
    return [
        (path, problem, quoted)
        for _, path, pieces in SOURCES
        for piece in pieces
        for problem, quoted in finder(piece)
    ]


def report(found: list[tuple[Path, str, str]]) -> str:
    return "\n\n".join(
        f"  {path.relative_to(REPO_ROOT).as_posix()}\n    {problem}\n    {quoted}"
        for path, problem, quoted in found
    )

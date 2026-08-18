"""Media paths. Always relative in the database, always resolved against MEDIA_ROOT (rule R6).

The legacy project moved twice. Each move broke 100% of the photo and clip paths,
because the absolute path was stored in the row, and each move spawned another
one-off fix_*_paths.py script. Storing the relative path makes the third move free.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from qorgan.settings import get_settings, resolve


class PathOutsideRoot(ValueError):
    """A path escaped its root — a traversal attempt, or a bug."""


def media_root() -> Path:
    root = resolve(get_settings().media_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def to_relative(path: Path | str, root: Path | None = None) -> str:
    """Absolute filesystem path -> the POSIX-relative string we store in a row."""
    base = (root or media_root()).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise PathOutsideRoot(f"{resolved} is not inside {base}") from exc
    return relative.as_posix()


def to_absolute(relative: str, root: Path | None = None) -> Path:
    """Stored relative string -> absolute path, refusing anything that escapes the root."""
    base = (root or media_root()).resolve()
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PathOutsideRoot(f"unsafe relative path: {relative!r}")
    resolved = (base / Path(relative)).resolve()
    ensure_within(base, resolved)
    return resolved


def ensure_within(root: Path, candidate: Path) -> Path:
    """Guard for anything built from untrusted input (ZIP members, URL segments)."""
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if not candidate_resolved.is_relative_to(root_resolved):
        raise PathOutsideRoot(f"{candidate_resolved} escapes {root_resolved}")
    return candidate_resolved

"""Reading the school's roster off its own directory tree.

Five traps in this data, all found by looking rather than assuming.

**The filename prefix lies.** The `учитель` folder contains files named
`student_469_….jpg`. A teacher's photo is named "student". Person type comes from the
FOLDER, never from the filename. The obvious pattern is wrong, and trusting it would have
filed two teachers as pupils.

**Two filename forms, both live.** The 2025 delivery names a photo
`student_333_1778595343147.jpg`; the 2026 delivery names the same thing `1.jpg`. Both are
read here, because the school's staff photographs exist ONLY in the old form and have to
be carried across -- a cook who is not in the roster opens an "unknown" session at every
single lunch. The new form carries no prefix, so its id namespace comes from the folder:
`staff/` yields `staff_…`, a class folder yields `student_…`.

**The folder names arrive mangled.** In the 2026 delivery `1А` is on disk as `1╨Р` --
UTF-8 bytes written out as CP866. `decode_folder_name` undoes it, and does so safely for
names that were never mangled.

**The delivery is full of files that are not photographs.** It was zipped on a Mac, so it
carries a `__MACOSX/` tree with a 178-byte `._<n>.jpg` stub per photograph and a
`.DS_Store` per directory. The stubs have image extensions: any glob by extension returns
282 files where there are 141 children. `is_macos_metadata` names them exactly -- and
being exact, rather than a suffix filter, is what keeps it from becoming a silent skip.

**No silent fallback, ever.** A filename that does not match either pattern is a hard
error naming the file. It is never a guessed identity. This is the single most important
rule in the spec: the legacy's characteristic failure was not that it got identity wrong
-- it was that it INVENTED an identity and carried on. A refusal is recoverable. A quiet
guess is a child eating someone else's lunch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qorgan.enums import PersonType
from qorgan.identity.naming import is_class_folder

# student_333_1778595343147.jpg -- the 333 is a REAL school ID, unique across all 142.
FILENAME = re.compile(r"^(student|staff)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

# 1.jpg -- the 2026 delivery. The number IS the school's id; there is no prefix and no
# timestamp. Anchored on \d+ alone, so `1a.jpg`, `1 (2).jpg` and `._1.jpg` do NOT match.
PLAIN_FILENAME = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

STAFF_FOLDER = "staff"
TEACHER_FOLDER = "учитель"

# The two id namespaces. Kept separate from STAFF_FOLDER on purpose: that they are spelled
# the same today is a coincidence, and one of them is a directory on the school's disk
# while the other is a permanent database key.
STUDENT_PREFIX = "student"
STAFF_PREFIX = "staff"

# macOS leaves these beside the real files. Three EXACT artefacts -- a directory name, a
# basename prefix, and one exact filename. See `is_macos_metadata`.
APPLE_DOUBLE_DIR = "__MACOSX"
APPLE_DOUBLE_PREFIX = "._"
DS_STORE = ".DS_Store"


class BadFilename(ValueError):
    """A photo whose name we cannot read. Not an identity to guess at."""


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One person, as the school's own directory tree describes them.

    `full_name` is the exception: the tree does not carry names, so it is filled in
    afterwards from the names CSV (see `qorgan.identity.names`) and is None until then.
    """

    external_id: str  # "student_333" -- the school's id in its namespace
    person_type: PersonType
    class_name: str | None  # "5-А", the folder name as the school wrote it
    position: str | None  # "учитель", or None
    full_name: str | None = None  # from the CSV, never from a filename


def decode_folder_name(name: str) -> str:
    """`'1╨Р'` -> `'1А'`. UTF-8 bytes that were written out as CP866.

    Every class folder in the 2026 delivery arrived this way. Left alone, `1╨Р` is not a
    class name, so `folder_role` would refuse all thirteen folders and the entire delivery
    with them.

    **Safe to apply to a name that was never mangled**, which is why it can sit at the top
    of `folder_role` rather than at a call site that has to know which delivery it is
    holding. A correct name does not survive the round trip: `'1А'.encode('cp866')` is
    `b'1\\x80'`, and a lone `\\x80` is a UTF-8 continuation byte with nothing to continue,
    so the decode raises and the original is returned untouched. ASCII (`staff`) is a fixed
    point. This is asserted over every real folder name in both deliveries by
    `tests/test_identity_roster_2026.py`, not merely reasoned about here.
    """
    try:
        return name.encode("cp866").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def is_macos_metadata(path: Path) -> bool:
    """Bookkeeping macOS leaves beside the real files. Not a photograph, and not a person.

    Measured on the 2026 delivery, which was zipped on a Mac:

      * a `__MACOSX/` tree mirroring the class folders, holding a 178-byte `._<n>.jpg`
        stub for every one of the 141 photographs. Each stub carries a `.jpg` or `.jpeg`
        extension, so **any glob by extension returns 282 files where there are 141
        children**;
      * `.DS_Store`, once per directory -- 14 of them, one per class folder plus the root.

    Every one of these reaching `entry_for` is a hard error that refuses the entire
    delivery. None of them is a photograph of anybody.

    **This is not the extension filter the importer forbids, and the difference is the
    whole point.** A filter by extension decides what we ACCEPT and silently drops the
    rest -- which is a child who quietly does not exist. This names three EXACT artefacts:
    a path component equal to `__MACOSX`, a basename beginning `._`, and the exact
    basename `.DS_Store`. Anything else, however odd it looks, still goes to `entry_for`
    and still raises. `_1.jpg`, `.1.jpg`, `1..jpg` and `my.DS_Store` are none of these and
    are still refused; there is a test that says exactly that, and a second test that
    fails if this predicate is ever widened into a general "hidden file" rule.
    """
    return (
        APPLE_DOUBLE_DIR in path.parts
        or path.name.startswith(APPLE_DOUBLE_PREFIX)
        or path.name == DS_STORE
    )


def folder_role(folder: str) -> tuple[PersonType, str | None, str | None]:
    """(person_type, class_name, position). The folder decides, and only the folder."""
    folder = decode_folder_name(folder)
    if is_class_folder(folder):
        return PersonType.STUDENT, folder, None
    if folder == STAFF_FOLDER:
        return PersonType.STAFF, None, None
    if folder == TEACHER_FOLDER:
        return PersonType.STAFF, None, TEACHER_FOLDER
    raise ValueError(
        f"unknown roster folder {folder!r}: it is not a class (1-А .. 11-Б), "
        f"not {STAFF_FOLDER!r} and not {TEACHER_FOLDER!r}. Ask the school what it is "
        "rather than guessing who is inside it."
    )


def external_id_for(filename: str, person_type: PersonType | None = None) -> str:
    """`student_333_1778595343147.jpg` -> `student_333`; `1.jpg` -> `student_1`.

    The old form carries its own namespace and keeps it verbatim, `person_type` or not:
    `staff_334_….jpg` is `staff_334` wherever it is found. The new form carries no prefix
    at all, so the FOLDER supplies one -- the same rule that decides person type, applied
    to the id. Anything matching neither form raises.
    """
    prefixed = FILENAME.match(filename)
    if prefixed is not None:
        return f"{prefixed.group(1).lower()}_{prefixed.group(2)}"

    plain = PLAIN_FILENAME.match(filename)
    if plain is not None:
        return f"{_namespace(person_type, filename)}_{plain.group(1)}"

    raise BadFilename(
        f"{filename!r} matches neither {FILENAME.pattern} (the 2025 delivery) nor "
        f"{PLAIN_FILENAME.pattern} (the 2026 delivery). A photo whose name we cannot "
        "read is not an identity to guess at -- a quiet guess is a child eating "
        "someone else's lunch. Fix the name, or ask the school whose face this is."
    )


def _namespace(person_type: PersonType | None, filename: str) -> str:
    """Which id namespace an un-prefixed filename belongs to. The folder is the only source.

    Refusing when nobody said is the same rule as everywhere else in this module. Guessing
    `student` here would file the cook as a pupil, and a pupil is somebody the canteen
    expects to feed.
    """
    if person_type is None:
        raise BadFilename(
            f"{filename!r} carries no prefix of its own, so only the folder it sits in can "
            "say whose id namespace this is -- and no folder was given. Call "
            "entry_for(folder, filename) rather than external_id_for(filename)."
        )
    return STAFF_PREFIX if person_type is PersonType.STAFF else STUDENT_PREFIX


def entry_for(folder: str, filename: str) -> RosterEntry:
    """The folder says WHO they are. The filename says only WHICH one."""
    person_type, class_name, position = folder_role(folder)
    return RosterEntry(
        external_id=external_id_for(filename, person_type),
        person_type=person_type,
        class_name=class_name,
        position=position,
    )

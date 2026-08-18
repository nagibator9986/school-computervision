"""Carrying people forward from the 2025 delivery, by an explicit list of ids.

**Why a list and not "import the `staff` and `учитель` folders".**

The 2026 delivery contains no staff at all -- thirteen class folders, 141 pupils, nobody
else. Staff still have to exist in the roster, because **staff never open a meal session**:
a cook who is not on the roster is an "unknown" session at every single lunch, for ever.
So somebody must be carried across from the old tree.

Taking the whole folder is the obvious move and it is wrong, because the old folders are
known to contain records the new delivery deliberately fixed. Measured on the school's own
photographs (`docs/questions-for-school.md` §1, re-measured 2026-07-27 against all 141 new
photographs):

  * **`staff_464` and `student_477` are one human** -- face similarity 0.999, and
    `student_477` sits in the `учитель` folder. Neither is covered by the new delivery, so
    "take the folder" creates BOTH and reproduces exactly the duplicate the new delivery
    was sent to remove. A duplicate is not an untidy record: measured on the school's hall
    video, a person enrolled twice is compared against himself, the margin between first
    and second candidate collapses to 0.001 against a threshold of 0.05, and the system
    stops recognising him at all. **Only one of the two may be carried. Which one is the
    school's decision, not ours, and not this file's.**
  * **`staff_334` and `student_470` are one human** -- 0.984, and this pair crosses the
    pupil/staff line, which decides whether that person is fed at all.
  * **`staff_465`, `staff_466`, `staff_467`, `staff_468` have no detectable face** in any
    photograph the school has sent (`questions-for-school.md` §3). They can be created,
    but they can never be recognised, and they will sit on every "no meal record" report
    for ever. They need new photographs, not a carry-over.

So the carry-over is a **written-down decision**: a file naming each `external_id` to
bring across. "The whole folder" is a guess that happens to be spelled like a command;
a list is a choice somebody made, that a reviewer can read, and that this module checks.

An id in the list that is not found in the tree is a HARD ERROR. Without that check a
typo carries nobody across and looks exactly like success -- and the person it silently
dropped is discovered months later, one unknown lunch at a time.

File format -- one external_id per line, `#` starts a comment. **This is an illustration
of the format, not a decision.** No carry-over list is committed to this repository: the
464/477 choice belongs to the school and has not been made yet::

    # decided by <name>, <date>
    staff_334
    staff_464     # and NOT student_477 -- same human; the school picked this id
    student_469   # folder `учитель`; the filename says student and the filename lies
"""

from __future__ import annotations

from pathlib import Path

COMMENT = "#"


class CarryOverProblem(ValueError):
    """The carry-over list and the old tree do not agree. Never silently reconciled."""


def read_carry_over(path: Path) -> set[str]:
    """Read the decided list of external_ids. Blank lines and `#` comments are ignored."""
    chosen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = raw.split(COMMENT, 1)[0].strip()
        if not text:
            continue
        if len(text.split()) != 1:
            raise CarryOverProblem(
                f"{path}: line {line_no}: {raw.strip()!r} is not a single external_id. "
                "One id per line -- a line this file cannot read is a person it cannot "
                "carry, and that is not something to work out from context."
            )
        chosen.add(text)

    if not chosen:
        raise CarryOverProblem(
            f"{path}: names nobody. An empty carry-over list is not 'carry everybody' and "
            "it is not 'carry nobody' -- it is a file somebody meant to fill in."
        )
    return chosen


def check_all_found(chosen: set[str], found: set[str], root: Path) -> None:
    """Every id on the list must exist in the tree. A typo must not read as success.

    The reverse is NOT an error: the old tree holds 142 people and the list names three.
    Leaving the other 139 behind is the entire purpose -- they are in the 2026 delivery
    under new numbers, and re-importing them here would create the duplicates all over
    again.
    """
    missing = sorted(chosen - found)
    if not missing:
        return
    raise CarryOverProblem(
        f"{root}: the carry-over list names {len(missing)} id(s) that are not in this "
        f"tree: {', '.join(missing)}. Nothing was imported. Check the spelling against "
        "the folder -- a name this list gets wrong is a person who quietly does not "
        "arrive, and who is then missing one unrecognised lunch at a time."
    )

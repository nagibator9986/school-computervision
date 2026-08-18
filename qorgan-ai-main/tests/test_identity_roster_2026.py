"""The 2026 delivery: mangled folder names, bare-number filenames, AppleDouble stubs.

Every number here was measured on the school's own delivery before it was written down:
141 photographs in 13 class folders, beside a `__MACOSX` tree holding one 178-byte stub
per photograph. No photograph is opened by this suite, here or anywhere -- these are
pictures of children and the tests have no business inside them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.enums import PersonType
from qorgan.identity.roster import (
    BadFilename,
    RosterEntry,
    decode_folder_name,
    entry_for,
    external_id_for,
    folder_role,
    is_macos_metadata,
)

# The thirteen class folders of the 2026 delivery, as the school wrote them.
CLASSES_2026 = ("1А", "1Б", "1В", "2А", "2Б", "3А", "3Б", "4А", "5А", "6А", "7А", "8А", "9А")

# The 2025 delivery's folder names, which were never mangled.
FOLDERS_2025 = ("1-А", "5-А", "11-Б", "staff", "учитель")


def mangle(name: str) -> str:
    """What the school's zip did: UTF-8 bytes written out as CP866. The inverse of the fix."""
    return name.encode("utf-8").decode("cp866")


# -- CP866 -------------------------------------------------------------------


def test_the_mangled_folder_name_on_the_disk_is_the_one_from_the_delivery() -> None:
    """`1А` really is `1╨Р` on disk: `А` is UTF-8 D0 90, and CP866 reads D0 as `╨`, 90 as
    `Р`. Pinned as a literal so this test still means something if `mangle` is ever wrong."""
    assert mangle("1А") == "1╨Р"
    assert decode_folder_name("1╨Р") == "1А"


@pytest.mark.parametrize("class_name", CLASSES_2026)
def test_every_2026_class_folder_decodes_and_is_recognised_as_a_class(class_name: str) -> None:
    """Left mangled, `1╨Р` is not a class name -- so `folder_role` would refuse all
    thirteen folders, and the entire delivery with them."""
    on_disk = mangle(class_name)

    assert decode_folder_name(on_disk) == class_name

    person_type, folder_class, position = folder_role(on_disk)
    assert person_type is PersonType.STUDENT
    assert folder_class == class_name, "the class is stored as the school wrote it"
    assert position is None


@pytest.mark.parametrize("folder", (*CLASSES_2026, *FOLDERS_2025))
def test_a_name_that_was_never_mangled_survives_the_decode_untouched(folder: str) -> None:
    """This is what lets the decode sit at the top of `folder_role` unconditionally.

    A correct name does not survive the CP866 round trip -- `'1А'.encode('cp866')` is
    `b'1\\x80'`, and a lone `\\x80` is a UTF-8 continuation byte with nothing to continue --
    so it raises and the original comes back. ASCII is a fixed point. Asserted over every
    real folder name in BOTH deliveries rather than reasoned about in a docstring.
    """
    assert decode_folder_name(folder) == folder


def test_the_2025_folders_still_work_after_the_decode_was_added() -> None:
    """The staff carried over from 2025 come out of these folders. Breaking them would
    break the carry-over, which is the only reason staff exist in the roster at all."""
    assert folder_role("staff") == (PersonType.STAFF, None, None)
    assert folder_role("учитель") == (PersonType.STAFF, None, "учитель")
    assert folder_role("5-А") == (PersonType.STUDENT, "5-А", None)


# -- __MACOSX, ._* and .DS_Store ---------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "__MACOSX/1А/._1.jpg",
        "__MACOSX/1А/1.jpg",
        "__MACOSX/._1.jpg",
        "1А/._1.jpg",
        "1А/._141.jpeg",
        "1А/.DS_Store",  # 14 of these in the delivery, one per directory
        ".DS_Store",
    ],
)
def test_macos_metadata_is_not_a_photograph(path: str) -> None:
    """141 photographs arrived with 141 AppleDouble stubs of 178 bytes each, every one
    carrying a `.jpg` or `.jpeg` extension -- **any glob by extension returns 282 files
    where there are 141 children** -- plus 14 `.DS_Store` files. Every one of them reaching
    `entry_for` refuses the whole delivery, and none of them is a photograph of anybody."""
    assert is_macos_metadata(Path(path)) is True


@pytest.mark.parametrize(
    "path",
    [
        "1А/1.jpg",
        "1А/_1.jpg",  # ONE underscore -- not AppleDouble
        "1А/.1.jpg",  # a dot, no underscore -- not AppleDouble
        "1А/1..jpg",
        "1А/MACOSX/1.jpg",  # not the __MACOSX directory
        "1А/my.DS_Store",  # not the exact basename
        "1А/.DS_Store2",
        "1А/.hidden.jpg",  # a hidden file is NOT automatically metadata
        "5-А/student_333_1778595343147.jpg",
    ],
)
def test_nothing_else_is_quietly_treated_as_metadata(path: str) -> None:
    """**The narrowness is the whole point.**

    A filter by extension -- or a general "hidden file" rule -- decides what we ACCEPT and
    silently drops the rest, which is a child who quietly does not exist. This names three
    EXACT artefacts. Everything else, however odd it looks, still goes to `entry_for` and
    still raises. If this test ever goes green on a name it should not, the exclusion has
    become the silent fallback the whole spec exists to delete.
    """
    assert is_macos_metadata(Path(path)) is False


def test_an_apple_double_stub_would_otherwise_be_a_hard_error() -> None:
    """Proof that skipping them is load-bearing and not decoration: unskipped, `._1.jpg`
    refuses the delivery -- correctly, since it is not a name we can read."""
    with pytest.raises(BadFilename, match=r"\._1\.jpg"):
        external_id_for("._1.jpg", PersonType.STUDENT)


# -- the bare-number filename ------------------------------------------------


def test_a_bare_number_becomes_the_schools_id_under_the_folders_namespace() -> None:
    """`1.jpg` in `1А` is `student_1`. The new form carries no prefix, so the FOLDER
    supplies one -- the same rule that decides person type, applied to the id."""
    entry = entry_for(mangle("1А"), "1.jpg")

    assert entry == RosterEntry(
        external_id="student_1",
        person_type=PersonType.STUDENT,
        class_name="1А",
        position=None,
        full_name=None,
    )


@pytest.mark.parametrize(
    ("filename", "expected"), [("1.jpg", 1), ("99.jpg", 99), ("141.jpeg", 141)]
)
def test_the_number_is_carried_verbatim(filename: str, expected: int) -> None:
    assert external_id_for(filename, PersonType.STUDENT) == f"student_{expected}"


def test_the_bare_form_in_a_staff_folder_would_be_staff_not_a_pupil() -> None:
    """There is no such file today. If one ever arrives, filing the cook as a pupil would
    put them on the list of children the canteen expects to feed."""
    assert external_id_for("7.jpg", PersonType.STAFF) == "staff_7"


def test_a_bare_number_with_no_folder_to_name_it_is_refused() -> None:
    """The same rule as everywhere else: when nobody said, we do not guess. `student` is
    the likely answer and a likely answer is exactly what this module does not accept."""
    with pytest.raises(BadFilename, match="no folder was given"):
        external_id_for("1.jpg")


# -- the 2025 form still works -----------------------------------------------


def test_the_2025_filename_form_still_works() -> None:
    """It has to: the school's staff photographs exist ONLY in this form, and staff who
    are not on the roster open an "unknown" canteen session at every single lunch."""
    assert external_id_for("student_333_1778595343147.jpg") == "student_333"
    assert external_id_for("staff_334_1778595388766.jpg") == "staff_334"
    assert external_id_for("student_333_1778595343147.JPG") == "student_333"


def test_the_old_form_keeps_its_own_namespace_even_in_a_class_folder() -> None:
    """`staff_334_….jpg` is `staff_334` wherever it is found. Only the bare form asks the
    folder, because only the bare form has nothing of its own to go on."""
    assert external_id_for("staff_334_1778595388766.jpg", PersonType.STUDENT) == "staff_334"


def test_a_teacher_whose_photo_is_named_student_is_still_staff() -> None:
    """**The original trap, still caught.** `student_469` sits in the `учитель` folder.
    The folder decides the type; the filename decides only which one, and it lies about
    the first."""
    entry = entry_for("учитель", "student_469_1778954922.jpg")

    assert entry.person_type is PersonType.STAFF
    assert entry.position == "учитель"
    assert entry.external_id == "student_469", "the id is the id; only the TYPE was a lie"


# -- no silent fallback, ever ------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "photo.jpg",
        "Иванов Иван.jpg",  # the legacy's whole world
        "student_333.jpg",  # no timestamp
        "student__1778595343147.jpg",  # no id
        "teacher_469_1778954922.jpg",  # not a prefix we know
        "student_469_1778954922.bmp",  # not an image we take
        "student_469_1778954922",  # no suffix
        "1.bmp",  # right shape, wrong image
        "1a.jpg",  # not a number
        "1 (2).jpg",  # the copy Windows makes
        "1.jpg.jpg",
        "-1.jpg",
        "1.",
    ],
)
def test_a_filename_that_does_not_match_is_a_hard_error_naming_the_file(filename: str) -> None:
    """**The single most important rule in this spec, and widening the patterns did not
    weaken it.**

    The legacy's characteristic failure was not that it got identity wrong -- it was that
    it INVENTED an identity and carried on. A refusal is recoverable. A quiet guess is a
    child eating someone else's lunch (spec §1.2).
    """
    with pytest.raises(BadFilename) as caught:
        external_id_for(filename, PersonType.STUDENT)

    assert filename in str(caught.value), "the error must name the file, or nobody can fix it"


def test_an_unknown_folder_is_still_a_hard_error_naming_the_folder() -> None:
    with pytest.raises(ValueError, match="кухня"):
        folder_role("кухня")


def test_the_error_names_both_forms_so_a_human_can_see_which_one_to_fix() -> None:
    with pytest.raises(BadFilename) as caught:
        external_id_for("Иванов.jpg", PersonType.STUDENT)

    message = str(caught.value)
    assert "2025" in message and "2026" in message


# -- the two id namespaces do not collide ------------------------------------


def test_carried_over_staff_and_the_new_pupils_share_no_id() -> None:
    """`staff_464` (2025) and `student_1` (2026) live in one `persons.external_id` column,
    which is UNIQUE. The old ids run 333..477 with a `student_`/`staff_` prefix; the new
    run 1..141 with a prefix the folder supplies. Nothing in the new delivery can land on
    an old id, because the prefixes are per-namespace and the numbers do not overlap."""
    old = {
        external_id_for(name)
        for name in (
            "staff_334_1778595388766.jpg",
            "staff_464_1778595389916.jpg",
            "student_469_1778954922.jpg",
            "student_477_1782943018.jpg",
        )
    }
    new = {external_id_for(f"{n}.jpg", PersonType.STUDENT) for n in range(1, 142)}

    assert len(new) == 141, "141 photographs, 141 distinct ids"
    assert not (old & new), f"an id would be claimed by two different people: {old & new}"

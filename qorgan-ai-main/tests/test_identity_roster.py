"""The roster: the FOLDER decides who someone is. The filename only carries their id.

Two traps in this data, both found by looking rather than assuming.
"""

from __future__ import annotations

import pytest

from qorgan.enums import PersonType
from qorgan.identity.roster import BadFilename, RosterEntry, entry_for, external_id_for, folder_role

# -- the folder decides -------------------------------------------------------


def test_a_class_folder_is_a_pupil_and_keeps_its_name_verbatim() -> None:
    """The class is stored as the school wrote it -- `5-А`, with the hyphen -- because it
    is shown to a human: `Ученик 333, 5-А`."""
    person_type, class_name, position = folder_role("5-А")

    assert person_type is PersonType.STUDENT
    assert class_name == "5-А"
    assert position is None


def test_the_staff_folder_is_staff() -> None:
    person_type, class_name, position = folder_role("staff")

    assert person_type is PersonType.STAFF
    assert class_name is None
    assert position is None


def test_the_teacher_folder_is_staff_with_a_position() -> None:
    person_type, class_name, position = folder_role("учитель")

    assert person_type is PersonType.STAFF
    assert class_name is None
    assert position == "учитель"


def test_an_unknown_folder_is_a_hard_error_naming_the_folder() -> None:
    """Never a guess. A folder we do not understand is a question for the school."""
    with pytest.raises(ValueError, match="кухня"):
        folder_role("кухня")


# -- THE trap: the filename prefix lies ---------------------------------------


def test_a_teacher_whose_photo_is_named_student_is_still_staff() -> None:
    """**The `учитель` folder contains files named `student_469_….jpg`.**

    A teacher's photo is named "student". The obvious pattern is wrong, and trusting it
    would have filed two teachers as pupils. Person type comes from the FOLDER, never
    from the filename (spec §1.1).
    """
    entry = entry_for("учитель", "student_469_1778954922.jpg")

    assert entry == RosterEntry(
        external_id="student_469",
        person_type=PersonType.STAFF,
        class_name=None,
        position="учитель",
    )


def test_a_pupil_in_a_class_folder_carries_the_class() -> None:
    entry = entry_for("5-А", "student_333_1778595343147.jpg")

    assert entry == RosterEntry(
        external_id="student_333",
        person_type=PersonType.STUDENT,
        class_name="5-А",
        position=None,
    )


def test_staff_keep_the_staff_prefix_in_their_external_id() -> None:
    """The external_id is the matched prefix + id, verbatim. `staff_334` is not
    `student_334`, and the two are different people until a human says otherwise."""
    assert entry_for("staff", "staff_334_1778595388766.jpg").external_id == "staff_334"


# -- no silent fallback, ever -------------------------------------------------


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
    ],
)
def test_a_filename_that_does_not_match_is_a_hard_error_naming_the_file(filename: str) -> None:
    """**The single most important rule in this spec.**

    The legacy's characteristic failure was not that it got identity wrong -- it was that
    it INVENTED an identity and carried on. A refusal is recoverable. A quiet guess is a
    child eating someone else's lunch (spec §1.2).
    """
    with pytest.raises(BadFilename) as caught:
        external_id_for(filename)

    assert filename in str(caught.value), "the error must name the file, or nobody can fix it"


def test_the_case_of_the_suffix_does_not_matter() -> None:
    assert external_id_for("student_333_1778595343147.JPG") == "student_333"

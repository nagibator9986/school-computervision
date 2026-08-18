"""Class names, and the fact that the school's ID is the identity."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qorgan.enums import PersonType
from qorgan.identity.naming import display_name, is_class_folder, normalise_class


def test_class_spellings_are_unified() -> None:
    """`5А`, `5-А` and `5 а` are one class. The legacy stored all three."""
    assert normalise_class("5А") == normalise_class("5-А") == normalise_class("5 а")


def test_a_normalised_class_is_upper_case_and_has_no_separators() -> None:
    assert normalise_class(" 11 - б ") == "11Б"


@pytest.mark.parametrize("folder", ["1-А", "1-Б", "1-В", "5-А", "7-А", "11-А", "11-Б", "10-А"])
def test_every_class_folder_the_school_sent_is_recognised(folder: str) -> None:
    assert is_class_folder(folder)


@pytest.mark.parametrize("folder", ["staff", "учитель", "", "12-А", "0-А", "photos", "5"])
def test_a_folder_that_is_not_a_class_is_not_a_class(folder: str) -> None:
    assert not is_class_folder(folder)


# -- the display name: written once, used everywhere ------------------------


@dataclass(frozen=True, slots=True)
class _Person:
    external_id: str
    full_name: str | None = None
    person_type: PersonType = PersonType.STUDENT
    class_name: str | None = None
    position: str | None = None


def test_a_pupil_with_no_name_is_shown_by_their_id_and_class() -> None:
    """There is no roster of NAMES. There is a roster of IDS. Until the school sends the
    ID -> name table, this is the honest thing to put on a screen (spec §1)."""
    pupil = _Person(external_id="student_333", class_name="5-А")

    assert display_name(pupil) == "Ученик 333, 5-А"


def test_a_pupil_with_no_class_still_has_an_id() -> None:
    assert display_name(_Person(external_id="student_333")) == "Ученик 333"


def test_staff_are_not_called_pupils() -> None:
    cook = _Person(external_id="staff_334", person_type=PersonType.STAFF)

    assert display_name(cook) == "Сотрудник 334"


def test_a_teacher_is_called_a_teacher() -> None:
    """`учитель/student_469_….jpg` -- the filename says student, the FOLDER says teacher,
    and the folder is right (spec §1.1)."""
    teacher = _Person(external_id="student_469", person_type=PersonType.STAFF, position="учитель")

    assert display_name(teacher) == "Учитель 469"


def test_a_real_name_always_wins() -> None:
    """The day the school sends the ID -> name table, nothing else has to change."""
    pupil = _Person(external_id="student_333", full_name="Петрова Мария", class_name="5-А")

    assert display_name(pupil) == "Петрова Мария"

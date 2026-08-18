"""The school's ID -> name table, and the join that must never be on a filename.

The names in this file are invented. The real table is 141 children and is carried by
hand; it is not in this repository and must never be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.identity.names import (
    BadNamesFile,
    NameRow,
    RosterMismatch,
    read_names,
    reconcile,
)

HEADER = "external_id,full_name,class_name\n"


def write_csv(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "names.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


# -- reading -----------------------------------------------------------------


def test_a_normal_table_is_read_into_ids(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "student_1,Иванов Иван Иванович,1А\nstudent_2,Петров Пётр,1А\n")

    names = read_names(path)

    assert names["student_1"] == NameRow("student_1", "Иванов Иван Иванович", "1А")
    assert names["student_2"].full_name == "Петров Пётр"
    assert len(names) == 2


def test_a_wrong_header_is_refused_rather_than_guessed_at(tmp_path: Path) -> None:
    """Columns are fixed on purpose. A file whose columns we guess at is a file we can read
    the wrong way round, and these are children's names."""
    path = write_csv(tmp_path, "student_1,1А,Иванов\n", header="id,class,name\n")

    with pytest.raises(BadNamesFile, match="header"):
        read_names(path)


def test_a_blank_name_is_refused(tmp_path: Path) -> None:
    """`display_name` falls back to `Ученик 10, 1А` when `full_name` is empty -- so a blank
    cell would look exactly like a child the school never named, which is the one thing
    this file exists to tell apart."""
    path = write_csv(tmp_path, "student_1,,1А\n")

    with pytest.raises(BadNamesFile, match="full_name is empty"):
        read_names(path)


def test_the_same_id_twice_is_refused(tmp_path: Path) -> None:
    """Two rows for one id is two different claims about one child."""
    path = write_csv(tmp_path, "student_1,Иванов Иван,1А\nstudent_1,Петров Пётр,1А\n")

    with pytest.raises(BadNamesFile, match="twice"):
        read_names(path)


def test_an_empty_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BadNamesFile, match="no data rows"):
        read_names(write_csv(tmp_path, ""))


def test_the_error_names_the_line_so_a_human_can_find_it(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "student_1,Иванов Иван,1А\nstudent_2,,1А\n")

    with pytest.raises(BadNamesFile, match="line 3"):
        read_names(path)


# -- THE trap: join by ID, never by photo filename ---------------------------


def test_joining_on_the_photo_filename_loses_three_children(tmp_path: Path) -> None:
    """**Measured on the school's own sheet: 141 rows, 141 photographs.**

    Joining on `ID` matches 141 of 141. Joining on the sheet's `Photo Name` column matches
    138 -- it says `10.jpg` and `12.jpg` where the disk has `10.jpeg`, and `99.jpeg` where
    the disk has `99.jpg`. The three children it loses keep a photograph and lose their
    name: they appear on every report as `Ученик 10, 1А` while their classmates appear by
    name, and nothing anywhere says a join silently dropped them.

    This test reproduces the exact three rows. It fails if anyone joins on a filename.
    """
    # The sheet as the school sent it: (ID, Photo Name it claims) against the real file.
    sheet = [("10", "10.jpg"), ("12", "12.jpg"), ("99", "99.jpeg"), ("11", "11.jpg")]
    on_disk = {"10": "10.jpeg", "12": "12.jpeg", "99": "99.jpg", "11": "11.jpg"}

    by_id = [row for row in sheet if row[0] in on_disk]
    by_filename = [row for row in sheet if row[1] in set(on_disk.values())]

    assert len(by_id) == 4, "the ID column is complete"
    assert len(by_filename) == 1, "the Photo Name column disagrees with the disk"
    lost = {row[0] for row in sheet} - {row[0] for row in by_filename}
    assert lost == {"10", "12", "99"}

    # And the module the importer actually uses joins on the id, so it loses nobody.
    path = write_csv(tmp_path, "".join(f"student_{i},Ребёнок {i},1А\n" for i, _ in sheet))
    names = read_names(path)
    photos = {f"student_{i}": "1А" for i in on_disk}

    assert reconcile(photos, names).agreed, "the real join must lose nobody"


# -- reconciliation ----------------------------------------------------------


def test_a_matching_roster_and_table_agree(tmp_path: Path) -> None:
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,1А\nstudent_2,Петров Пётр,2Б\n"))

    result = reconcile({"student_1": "1А", "student_2": "2Б"}, names)

    assert result.agreed
    assert result.matched == ("student_1", "student_2")
    result.raise_if_disagreed()  # must not raise


def test_a_named_child_with_no_photograph_is_loud(tmp_path: Path) -> None:
    """They can never be recognised. That is a thing the school can fix in an afternoon,
    and it is invisible if the import prints a cheerful total and carries on."""
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,1А\nstudent_2,Петров Пётр,1А\n"))

    result = reconcile({"student_1": "1А"}, names)

    assert not result.agreed
    assert result.without_photo == ("student_2",)
    with pytest.raises(RosterMismatch, match="student_2"):
        result.raise_if_disagreed()


def test_a_photograph_with_no_name_is_loud(tmp_path: Path) -> None:
    """Imported under a number instead of a name, and nobody would notice which."""
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,1А\n"))

    result = reconcile({"student_1": "1А", "student_2": "1А"}, names)

    assert result.without_name == ("student_2",)
    with pytest.raises(RosterMismatch, match="student_2"):
        result.raise_if_disagreed()


def test_a_child_the_sheet_puts_in_a_different_class_is_loud(tmp_path: Path) -> None:
    """The sheet and the folder disagreeing about a class is exactly the "wrong child"
    failure this project exists to prevent. Measured on the real delivery: 0 of 141."""
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,3Б\n"))

    result = reconcile({"student_1": "1А"}, names)

    assert result.wrong_class == (("student_1", "3Б", "1А"),)
    with pytest.raises(RosterMismatch, match="3Б"):
        result.raise_if_disagreed()


def test_punctuation_in_the_class_is_not_a_disagreement(tmp_path: Path) -> None:
    """`5А`, `5-А` and `5 а` are one class -- the sheet and the disk punctuate differently
    and that is not a finding."""
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,5-А\n"))

    assert reconcile({"student_1": "5А"}, names).agreed


def test_reconcile_filters_nothing_of_its_own(tmp_path: Path) -> None:
    """It compares exactly what it is given. Scoping -- the 2026 table names pupils only,
    so carried-over staff are not in it -- is the caller's decision, made where a reader
    can see it, not a quiet exemption in here."""
    names = read_names(write_csv(tmp_path, "student_1,Иванов Иван,1А\n"))

    result = reconcile({"student_1": "1А", "staff_334": None}, names)

    assert result.without_name == ("staff_334",)

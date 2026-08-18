"""Importing the 2026 delivery: the walk, the names, and the carry-over by list.

**The photographs here are a 15-byte lie, on purpose.** The real ones are pictures of
children and this suite has no business inside them. `cv2` cannot decode these, which is
fine: every assertion below is about WHO was imported and under what id and name, and none
of it is about a face. The photos land as unenrollable, which is itself the reported,
non-silent outcome the importer promises.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.models import Person
from qorgan.enums import PersonType
from qorgan.faces.importer import import_directory
from qorgan.identity.carryover import CarryOverProblem, check_all_found, read_carry_over
from qorgan.identity.names import RosterMismatch, read_names
from qorgan.settings import Settings

SETTINGS = FaceModelSettings()
NOT_A_PHOTOGRAPH = b"not-a-real-face"  # 15 bytes, and deliberately not an image


class NoRecognizer:
    """Never asked: these bytes do not decode, so the importer never reaches detection."""

    def detect(self, _frame: object) -> list:
        raise AssertionError("a 15-byte lie must never reach face detection")


def mangle(name: str) -> str:
    """UTF-8 bytes written out as CP866, exactly as the delivery's zip did."""
    return name.encode("utf-8").decode("cp866")


def put(root: Path, folder: str, name: str) -> Path:
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(NOT_A_PHOTOGRAPH)
    return path


def names_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "names.csv"
    path.write_text("external_id,full_name,class_name\n" + body, encoding="utf-8")
    return path


def carry_list(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "carry-over.txt"
    path.write_text(body, encoding="utf-8")
    return path


# -- the walk ----------------------------------------------------------------


def test_the_macosx_stubs_are_not_imported_as_children(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**The delivery has 141 photographs and 141 AppleDouble stubs carrying .jpg names.**

    Unskipped, each stub is a hard error that refuses the whole delivery. Counted as a
    photograph, each is an extra child who does not exist. Both are wrong; only the real
    files are people.
    """
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    put(root, mangle("1А"), "2.jpg")
    put(root, mangle("1А"), "._1.jpg")
    put(root, mangle("1А"), "._2.jpg")
    put(root, mangle("1А"), ".DS_Store")
    put(root, "__MACOSX", "._whatever.jpg")
    put(root, f"__MACOSX/{mangle('1А')}", "._1.jpg")

    report = import_directory(root, NoRecognizer(), SETTINGS)

    session.expire_all()
    people = sorted(p.external_id for p in session.scalars(select(Person)))
    assert people == ["student_1", "student_2"], "a stub was imported as a child"
    assert report.people == 2


def test_a_mangled_class_folder_is_imported_under_the_name_the_school_wrote(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    root = tmp_path / "roster"
    put(root, mangle("9А"), "141.jpg")

    import_directory(root, NoRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.external_id == "student_141"
    assert person.class_name == "9А"
    assert person.person_type is PersonType.STUDENT


def test_a_name_we_cannot_read_still_refuses_the_whole_delivery(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Widening the patterns and skipping AppleDouble did not open a side door."""
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    put(root, mangle("1А"), "Иванов Иван.jpg")

    with pytest.raises(Exception, match="Иванов Иван.jpg"):
        import_directory(root, NoRecognizer(), SETTINGS)

    session.expire_all()
    assert session.scalars(select(Person)).all() == [], "nothing is written until the tree reads"


# -- the names ---------------------------------------------------------------


def test_the_full_name_is_filled_from_the_table(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The school sent the table; the owner decided the names are shown."""
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    put(root, mangle("1А"), "2.jpg")
    names = read_names(
        names_csv(tmp_path, "student_1,Иванов Иван Иванович,1А\nstudent_2,Петрова Анна,1А\n")
    )

    import_directory(root, NoRecognizer(), SETTINGS, names=names)

    session.expire_all()
    people = {p.external_id: p for p in session.scalars(select(Person))}
    assert people["student_1"].full_name == "Иванов Иван Иванович"
    assert people["student_1"].display == "Иванов Иван Иванович"
    assert people["student_2"].display == "Петрова Анна"


def test_a_child_with_no_row_refuses_the_import_before_anything_is_written(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Half-importing a roster and then refusing leaves somebody to work out which half."""
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    put(root, mangle("1А"), "2.jpg")
    names = read_names(names_csv(tmp_path, "student_1,Иванов Иван Иванович,1А\n"))

    with pytest.raises(RosterMismatch, match="student_2"):
        import_directory(root, NoRecognizer(), SETTINGS, names=names)

    session.expire_all()
    assert session.scalars(select(Person)).all() == []


def test_a_name_already_in_the_database_is_not_overwritten_by_a_reimport(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """A name may have been corrected by a human in the panel. Re-running an import
    silently reverting that is the legacy's `apply_runtime_migrations` behaviour, which
    re-derived `person_type` on every boot and undid manual fixes (audit H-02)."""
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    first = read_names(names_csv(tmp_path, "student_1,Иванов,1А\n"))
    import_directory(root, NoRecognizer(), SETTINGS, names=first)

    import_directory(
        root,
        NoRecognizer(),
        SETTINGS,
        names=read_names(names_csv(tmp_path, "student_1,ПЕРЕПИСАНО,1А\n")),
    )

    session.expire_all()
    assert session.scalars(select(Person)).one().full_name == "Иванов"


def test_staff_without_a_row_are_not_a_mismatch(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The 2026 table names PUPILS. The carried-over staff have no row and that is correct
    -- the scoping is written at the call site in `_check_names`, not hidden in reconcile."""
    root = tmp_path / "roster"
    put(root, mangle("1А"), "1.jpg")
    put(root, "staff", "staff_334_1778595388766.jpg")
    names = read_names(names_csv(tmp_path, "student_1,Иванов Иван,1А\n"))

    import_directory(root, NoRecognizer(), SETTINGS, names=names)

    session.expire_all()
    people = {p.external_id: p for p in session.scalars(select(Person))}
    assert people["student_1"].full_name == "Иванов Иван"
    assert people["staff_334"].full_name is None
    assert people["staff_334"].display == "Сотрудник 334"


# -- the carry-over, by explicit list ----------------------------------------


def test_the_carry_over_takes_the_listed_people_and_nobody_else(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**Taking the whole folder would recreate two people enrolled twice.**

    `staff_464` and `student_477` are one human (0.999); `staff_334` and `student_470` are
    one human (0.984). Both pairs are absent from the 2026 delivery, so "take the folder"
    reproduces exactly the duplicates the new delivery removed -- and a person enrolled
    twice is compared against himself and is not recognised at all.
    """
    old = tmp_path / "old"
    put(old, "staff", "staff_334_1778595388766.jpg")
    put(old, "staff", "staff_464_1778595389916.jpg")
    put(old, "учитель", "student_469_1778954922.jpg")
    put(old, "учитель", "student_477_1782943018.jpg")  # same human as staff_464
    decided = "# decided by the school\nstaff_334\nstaff_464  # NOT student_477\nstudent_469\n"
    chosen = read_carry_over(carry_list(tmp_path, decided))

    import_directory(old, NoRecognizer(), SETTINGS, only=chosen)

    session.expire_all()
    people = sorted(p.external_id for p in session.scalars(select(Person)))
    assert people == ["staff_334", "staff_464", "student_469"]
    assert "student_477" not in people, "the duplicate of staff_464 was carried across"


def test_a_carried_over_teacher_is_staff_even_though_the_file_says_student(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """`student_469` sits in `учитель`. The FOLDER decides who someone is; the filename
    lies about it -- and staff never open a meal session, so getting this wrong puts a
    teacher on the list of children the canteen expects to feed."""
    old = tmp_path / "old"
    put(old, "учитель", "student_469_1778954922.jpg")

    import_directory(old, NoRecognizer(), SETTINGS, only={"student_469"})

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.person_type is PersonType.STAFF
    assert person.position == "учитель"
    assert person.display == "Учитель 469"


def test_an_id_on_the_list_that_is_not_in_the_tree_refuses_the_import(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """A typo must not read as success. Without this the carry-over silently brings nobody
    across, and the missing person is found months later, one unknown lunch at a time."""
    old = tmp_path / "old"
    put(old, "staff", "staff_334_1778595388766.jpg")

    with pytest.raises(CarryOverProblem, match="staff_999"):
        import_directory(old, NoRecognizer(), SETTINGS, only={"staff_334", "staff_999"})

    session.expire_all()
    assert session.scalars(select(Person)).all() == []


def test_leaving_most_of_the_old_tree_behind_is_not_an_error() -> None:
    """The old tree holds 142 people and the list names three. Leaving the other 139 is
    the entire purpose: they are in the 2026 delivery under new numbers."""
    check_all_found({"staff_334"}, {"staff_334", "student_333", "student_470"}, Path("old"))


def test_an_empty_or_unreadable_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CarryOverProblem, match="names nobody"):
        read_carry_over(carry_list(tmp_path, "# nothing decided yet\n\n"))

    with pytest.raises(CarryOverProblem, match="single external_id"):
        read_carry_over(carry_list(tmp_path, "staff_334 staff_464\n"))


# -- the two deliveries in one namespace -------------------------------------


def test_carried_over_staff_and_the_new_pupils_coexist_without_collision(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """`persons.external_id` is UNIQUE and holds both deliveries. Old ids run 333..477 with
    their own prefix; new run 1..141 with a prefix the folder supplies. `staff_464` and
    `student_1` are different people and stay that way."""
    root = tmp_path / "roster"
    for number, class_name in ((1, "1А"), (72, "5А"), (141, "9А")):
        put(root, mangle(class_name), f"{number}.jpg")
    put(root, "staff", "staff_464_1778595389916.jpg")
    put(root, "учитель", "student_469_1778954922.jpg")

    report = import_directory(root, NoRecognizer(), SETTINGS)

    session.expire_all()
    people = {p.external_id: p for p in session.scalars(select(Person))}
    assert sorted(people) == ["staff_464", "student_1", "student_141", "student_469", "student_72"]
    assert report.people == 5, "five files, five distinct people, no id claimed twice"
    assert people["student_1"].person_type is PersonType.STUDENT
    assert people["staff_464"].person_type is PersonType.STAFF
    assert people["student_469"].person_type is PersonType.STAFF


def test_an_undecodable_photo_is_reported_by_name_never_silently_enrolled(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """A person the system can never recognise is something the school has to know about.

    The person IS created -- they are on the roster whether or not we can see their face --
    but the photograph is itemised by name so it reaches the school as a request for a new
    one. The four staff photographs with no detectable face are the real instance of this.
    """
    root = tmp_path / "roster"
    put(root, "staff", "staff_465_1778595393105.jpg")

    report = import_directory(root, NoRecognizer(), SETTINGS)

    assert report.embeddings == 0
    assert report.has_unenrollable
    assert [u.photo for u in report.unenrollable] == ["staff_465_1778595393105.jpg"]
    assert "staff_465_1778595393105.jpg" in report.summary()

    session.expire_all()
    assert session.scalars(select(Person)).one().external_id == "staff_465"

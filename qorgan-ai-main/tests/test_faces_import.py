"""Importing pupils: Zip Slip, and the photos that cannot be enrolled."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.detection.geometry import Box
from qorgan.enums import ExternalIdSource, PersonType
from qorgan.faces.importer import ImportReport, import_directory, safe_extract
from qorgan.faces.recognizer import DetectedFace
from qorgan.identity.roster import BadFilename
from qorgan.settings import Settings

SETTINGS = FaceModelSettings()

# -- Zip Slip ----------------------------------------------------------------


def test_a_normal_archive_extracts(tmp_path: Path) -> None:
    archive = tmp_path / "roster.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("5-А/student_333_1778595343147.jpg", b"pretend-jpeg")

    extracted = safe_extract(archive, tmp_path / "out")

    assert len(extracted) == 1
    assert extracted[0].read_bytes() == b"pretend-jpeg"


def test_an_archive_that_tries_to_escape_is_refused(tmp_path: Path) -> None:
    """The legacy called extractall() with NO validation of member names, on a directory
    chosen by an unauthenticated HTTP caller. A crafted archive could write anywhere on
    the server (audit H-03)."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../pwned.txt", b"gotcha")

    with pytest.raises(ValueError, match="Zip Slip"):
        safe_extract(archive, tmp_path / "out")

    assert not (tmp_path.parent / "pwned.txt").exists()


def test_an_absolute_member_path_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/passwd", b"gotcha")

    # Either it is rejected, or it lands harmlessly inside the destination -- never
    # outside it. What matters is that nothing escapes.
    try:
        extracted = safe_extract(archive, tmp_path / "out")
    except ValueError:
        return
    for path in extracted:
        assert (tmp_path / "out") in path.parents


# -- the report --------------------------------------------------------------


def test_a_clean_import_reports_only_what_it_did() -> None:
    report = ImportReport(people=138, photos=138, embeddings=138)

    assert "Imported 138 people" in report.summary()
    assert not report.has_unenrollable


def test_photos_that_cannot_be_enrolled_are_itemised_not_dropped() -> None:
    """Four of the school's staff photographs contain no detectable face at all --
    staff_465 through staff_468. They cannot be enrolled, and a person the system can
    never recognise is something the school has to know about (spec §1.1)."""
    report = ImportReport(people=138, photos=142, embeddings=138)
    report.cannot_enrol("staff_465_1778595393105.jpg", "no face found", faces=0)
    report.cannot_enrol("групповое.jpg", "3 faces found; a roster photo must show one", faces=3)

    summary = report.summary()

    assert report.has_unenrollable
    assert "2 photo(s) could NOT be enrolled" in summary
    assert "staff_465_1778595393105.jpg" in summary
    assert "3 faces found" in summary


# -- the roster walk ---------------------------------------------------------


class FakeRecognizer:
    """Returns a scripted number of faces per photo. No model, no GPU."""

    def __init__(self, faces_per_photo: int = 1, size: int = 120) -> None:
        self.faces_per_photo = faces_per_photo
        self.size = size
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[DetectedFace]:
        self.calls += 1
        return [
            DetectedFace(
                box=Box(0.0, 0.0, float(self.size), float(self.size)),
                embedding=_face(self.calls * 10 + index),
                detection_score=0.9,
            )
            for index in range(self.faces_per_photo)
        ]


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _roster(tmp_path: Path) -> Path:
    """The roster root is a directory of NOTHING BUT the roster.

    It cannot be `tmp_path` itself: the `settings` fixture puts `test.sqlite3`, `media/`
    and `logs/` in there, and the walk offers EVERY file it finds to `entry_for` -- so it
    would refuse `test.sqlite3` by name, correctly. (Correctly, because on the second run
    it would otherwise also re-walk the photos it had just copied into `media/people/5-А/`
    and import them a second time.) The old way of hiding all of that was a `.jpg`
    pre-filter in the walk, i.e. a second, silent definition of "a file we accept" sitting
    next to `roster.FILENAME`. There isn't one, so the input has to be honest instead.
    """
    return tmp_path / "roster"


def _photo(root: Path, folder: str, name: str) -> Path:
    """A real 200x200 JPEG on disk. cv2 has to be able to decode it."""
    import cv2

    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    buffer.tofile(str(path))
    return path


def test_a_pupil_is_imported_with_the_schools_own_id(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """`external_id = student_333`, `external_id_source = ROSTER`, `full_name = NULL`.
    Nothing is derived from anything (spec §4.3)."""
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")

    report = import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.external_id == "student_333"
    assert person.external_id_source is ExternalIdSource.ROSTER
    assert person.full_name is None
    assert person.person_type is PersonType.STUDENT
    assert person.class_name == "5-А"
    assert report.people == 1
    assert report.embeddings == 1


def test_person_type_comes_from_the_folder_never_from_the_filename(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**THE trap.** The `учитель` folder contains files named `student_469_….jpg`. A
    teacher's photo is named "student". Trusting the obvious pattern would have filed two
    teachers as pupils (spec §1.1)."""
    _photo(_roster(tmp_path), "учитель", "student_469_1778954922.jpg")

    import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.person_type is PersonType.STAFF, "a teacher was filed as a pupil"
    assert person.position == "учитель"
    assert person.class_name is None
    assert person.external_id == "student_469"  # the id is the id; only the TYPE was a lie
    assert person.display == "Учитель 469"


def test_staff_are_staff(settings: Settings, session: Session, tmp_path: Path) -> None:
    _photo(_roster(tmp_path), "staff", "staff_334_1778595388766.jpg")

    import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.person_type is PersonType.STAFF
    assert person.position is None
    assert person.display == "Сотрудник 334"


def test_a_filename_that_does_not_match_stops_the_import_dead(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """**The single most important rule in this spec.** A refusal is recoverable. A quiet
    guess is a child eating someone else's lunch (spec §1.2)."""
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")
    _photo(_roster(tmp_path), "5-А", "photo.jpg")

    with pytest.raises(BadFilename, match="photo.jpg"):
        import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)


def test_a_file_whose_extension_we_do_not_know_is_refused_not_skipped(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """No second definition of "a file we accept".

    If the walk pre-filtered on a suffix set of its own, a file whose extension is in that
    set but not in `roster.FILENAME` (or the other way round) would be SILENTLY SKIPPED
    instead of refused -- the exact silent fallback this spec exists to delete, rebuilt one
    task after deleting it. `entry_for` is the single gate, and it raises.
    """
    (_roster(tmp_path) / "5-А").mkdir(parents=True)
    (_roster(tmp_path) / "5-А" / "Thumbs.db").write_bytes(b"not a photo")

    with pytest.raises(BadFilename, match="Thumbs.db"):
        import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    assert session.scalars(select(Person)).all() == []


def test_an_unknown_folder_stops_the_import_dead(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _photo(_roster(tmp_path), "кухня", "staff_465_1778595393105.jpg")

    with pytest.raises(ValueError, match="кухня"):
        import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)


# -- the photos that cannot be enrolled --------------------------------------


def test_a_photo_with_no_face_is_itemised_and_the_person_still_exists(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Four staff photos contain no detectable face at all -- staff_465 through
    staff_468. They cannot be enrolled. They must be ITEMISED, not silently dropped: the
    school has four members of staff the system can never recognise, and it needs to know
    that (spec §1.1)."""
    _photo(_roster(tmp_path), "staff", "staff_465_1778595393105.jpg")

    report = import_directory(_roster(tmp_path), FakeRecognizer(faces_per_photo=0), SETTINGS)

    assert report.embeddings == 0
    assert [u.photo for u in report.unenrollable] == ["staff_465_1778595393105.jpg"]
    assert report.unenrollable[0].faces == 0

    session.expire_all()
    # The person exists -- they are on the roster -- and the DB records WHY they have no
    # face, so `gallery-report` can find them for ever, not just in this run's stdout.
    person = session.scalars(select(Person)).one()
    assert person.external_id == "staff_465"
    assert session.scalars(select(FaceEmbedding)).all() == []
    photo = session.scalars(select(PersonPhoto)).one()
    assert photo.person_id == person.id
    assert "no face" in photo.quality_note


def test_a_group_photo_is_itemised_not_guessed_at(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """Which one is the pupil? A group photograph cannot say, and guessing here poisons
    the gallery for everybody."""
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")

    report = import_directory(_roster(tmp_path), FakeRecognizer(faces_per_photo=3), SETTINGS)

    assert report.embeddings == 0
    assert report.unenrollable[0].faces == 3
    assert "3 faces" in report.unenrollable[0].reason

    session.expire_all()
    assert session.scalars(select(FaceEmbedding)).all() == []


def test_a_face_too_small_to_embed_is_itemised(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")

    report = import_directory(_roster(tmp_path), FakeRecognizer(size=30), SETTINGS)

    assert report.embeddings == 0
    assert "too small" in report.unenrollable[0].reason


# -- one import, not two ------------------------------------------------------


def test_importing_the_same_roster_twice_does_not_create_the_child_twice(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The id is the school's, and it is stable. Re-running the import is safe."""
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")

    import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)
    import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    assert len(session.scalars(select(Person)).all()) == 1


def test_reimporting_does_not_double_photos_or_embeddings(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """`ix_person_photos_sha256` exists to answer exactly one question: "is this content
    already stored for this person?" Before this fix nothing asked it, so `PersonPhoto`
    and `FaceEmbedding` were inserted unconditionally on every call -- re-running the
    import over an unchanged directory silently DOUBLED every photo and embedding row.
    `gallery-report` counts both, so a doubled gallery would report plausible-looking,
    wrong statistics. A second run must add nothing."""
    _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")

    first = import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)
    second = import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    assert len(session.scalars(select(Person)).all()) == 1
    assert len(session.scalars(select(PersonPhoto)).all()) == 1
    assert len(session.scalars(select(FaceEmbedding)).all()) == 1
    assert first.photos == 1
    assert first.embeddings == 1
    assert second.photos == 0
    assert second.embeddings == 0
    assert second.already_present == 1


def test_reimporting_an_unenrollable_photo_does_not_double_it_either(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The 0-face and >1-face cases still get a `Person` + `PersonPhoto(quality_note=…)`
    on the first run, and still no embedding -- but a second run over the same photo must
    not add a second `PersonPhoto` either."""
    _photo(_roster(tmp_path), "staff", "staff_465_1778595393105.jpg")

    first = import_directory(_roster(tmp_path), FakeRecognizer(faces_per_photo=0), SETTINGS)
    second = import_directory(_roster(tmp_path), FakeRecognizer(faces_per_photo=0), SETTINGS)

    session.expire_all()
    assert len(session.scalars(select(PersonPhoto)).all()) == 1
    assert session.scalars(select(FaceEmbedding)).all() == []
    assert first.photos == 1
    assert second.photos == 0
    assert second.already_present == 1


def test_the_same_bytes_under_two_different_people_warns_loudly(
    settings: Settings,
    session: Session,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two DIFFERENT children whose files happen to be byte-identical is a data problem
    (e.g. one photo copied under the wrong id), not a coincidence to merge silently. Both
    are still stored as their own person's photo -- but it must not pass without a trace."""
    import shutil

    src = _photo(_roster(tmp_path), "5-А", "student_333_1778595343147.jpg")
    twin = src.parent / "student_444_1778595343148.jpg"
    shutil.copy(src, twin)

    with caplog.at_level("WARNING"):
        report = import_directory(_roster(tmp_path), FakeRecognizer(), SETTINGS)

    session.expire_all()
    assert len(session.scalars(select(Person)).all()) == 2
    assert len(session.scalars(select(PersonPhoto)).all()) == 2
    assert report.photos == 2
    assert any("identical" in message or "already stored" in message for message in caplog.messages)


def test_the_zip_path_and_the_directory_path_are_the_same_import(
    settings: Settings, session: Session, tmp_path: Path
) -> None:
    """The web upload takes a ZIP. It must not be a second, divergent importer -- the
    legacy had `analyze_aggression` in three copies and they had already drifted apart."""
    from qorgan.faces.importer import import_archive

    photo = _photo(tmp_path / "src", "5-А", "student_333_1778595343147.jpg")
    archive = tmp_path / "roster.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(photo, arcname="5-А/student_333_1778595343147.jpg")

    report = import_archive(archive, FakeRecognizer(), SETTINGS)

    session.expire_all()
    person = session.scalars(select(Person)).one()
    assert person.external_id == "student_333"
    assert person.class_name == "5-А"
    assert report.embeddings == 1

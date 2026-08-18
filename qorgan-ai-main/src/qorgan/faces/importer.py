r"""Photographs in, embeddings out. Task 5 puts the roster walk on top of this.

Two things the legacy got wrong, both fixed here:

  * **Zip Slip.** It called `extractall()` with no validation of the member names at all,
    so an archive containing `..\..\Windows\System32\...` would write there — and the
    endpoint that accepted archives had no authentication (audit H-03). Every member is
    checked to land inside the extraction directory.

  * **Files were deleted before the transaction committed**, so a crash mid-import left
    the database rolled back and the photographs already gone (M-22). Nothing is deleted
    here, and the database work commits per person.

**AN IMPORT BELONGS TO ONE SCHOOL, AND `_get_or_create` IS WHERE AN UNSCOPED LOOKUP WOULD
DO THE MOST DAMAGE IN THIS CODEBASE.** `external_id` stopped being globally unique at
migration 0009, because two schools may each enrol a pupil numbered 7. Unscoped, importing
the SECOND school's roster would FIND the FIRST school's pupil 7 and attach this school's
photographs and face embeddings to a child in another school. That child would then be
recognised in the wrong building, fed under the wrong roster and named in the wrong canteen
report -- and nothing would report any of it, because from every layer's point of view the
person was simply "found". `school_id=None` means the only school there is and raises once
there are several, so an import that never said whose roster it was stops instead.

What is gone: the namesake machinery. It answered a question this data does not ask. The
school issues the ids; we read them (spec §1).
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorgan.config.identity import FaceModelSettings
from qorgan.db.engine import session_scope
from qorgan.db.models import FaceEmbedding, Person, PersonPhoto
from qorgan.db.tenancy import owned_by, resolve_school_id, scope
from qorgan.enums import ExternalIdSource, PersonType
from qorgan.faces.recognizer import DetectedFace, FaceRecognizer
from qorgan.identity.carryover import check_all_found
from qorgan.identity.names import NameRow, reconcile
from qorgan.identity.roster import RosterEntry, entry_for, is_macos_metadata
from qorgan.logging_setup import get_logger
from qorgan.paths import PathOutsideRoot, ensure_within
from qorgan.settings import get_settings, resolve

logger = get_logger(__name__)

# There is no IMAGE_SUFFIXES here, and there must never be one again. `roster.FILENAME` is
# the ONE definition of a file we accept, and it RAISES. A suffix set beside it would be a
# second definition: a file whose extension is in one list and not the other gets silently
# skipped instead of refused -- which is a child who quietly does not exist in the system.
# So every file under a roster folder goes to `entry_for`, and `BadFilename` is the gate.
#
# `is_macos_metadata` is the ONE exception, and it is deliberately not a suffix set: it
# names three EXACT artefacts macOS leaves behind (`__MACOSX/`, `._*`, `.DS_Store`), none
# of which is a photograph of anybody. The `._*` stubs carry .jpg extensions, so a suffix
# filter would have LET THEM IN as 141 extra children while dropping real files silently.
# See its docstring for why the distinction is not a technicality.

# A photo whose face is smaller than this is not worth embedding: it will produce a
# vector that matches everybody a little and nobody enough.
MIN_FACE_PX = 60


@dataclass(frozen=True, slots=True)
class Unenrollable:
    """A photograph we could not turn into a face. It is REPORTED, never dropped.

    Four of the school's staff photographs contain no detectable face at all --
    staff_465, staff_466, staff_467, staff_468. They cannot be enrolled, and a person the
    system can never recognise is something the school has to know about (spec §1.1).
    """

    photo: str
    reason: str
    faces: int


@dataclass
class ImportReport:
    people: int = 0
    photos: int = 0
    embeddings: int = 0
    # Photos whose content hash was already stored for that person -- a re-import that did
    # nothing. Distinct from `photos` so a second run over an unchanged directory is
    # legible as "nothing changed", not an identical-looking "138 photos imported" again.
    already_present: int = 0
    unenrollable: list[Unenrollable] = field(default_factory=list)

    def cannot_enrol(self, photo: str, reason: str, faces: int) -> None:
        self.unenrollable.append(Unenrollable(photo=photo, reason=reason, faces=faces))

    @property
    def has_unenrollable(self) -> bool:
        return bool(self.unenrollable)

    def summary(self) -> str:
        lines = [
            f"Imported {self.people} people, {self.photos} photos, {self.embeddings} embeddings.",
        ]
        if self.already_present:
            lines.append(
                f"{self.already_present} photo(s) were already present and were not "
                "re-imported (re-running an import is a no-op, not a duplicate)."
            )
        if not self.unenrollable:
            return "\n".join(lines)

        lines.append("")
        lines.append(
            f"{len(self.unenrollable)} photo(s) could NOT be enrolled. These people exist "
            "in the roster and the system can never recognise them. They will appear on "
            "every 'did not eat' report, for ever, until the school sends another photo:"
        )
        for item in sorted(self.unenrollable, key=lambda u: u.photo):
            lines.append(f"  {item.photo} — {item.reason}")
        return "\n".join(lines)


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Unzip, refusing any member that would land outside the destination.

    The legacy called `extractall()` with no checks whatsoever, on a directory chosen by
    an unauthenticated HTTP caller. A crafted archive could write anywhere on the server.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            target = destination / member.filename
            try:
                ensure_within(destination, target)
            except PathOutsideRoot:
                # Not a warning. Somebody built this archive on purpose.
                raise ValueError(
                    f"{archive.name}: refusing member {member.filename!r} — it would "
                    "escape the extraction directory (Zip Slip)"
                ) from None

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)

    return extracted


def import_directory(
    root: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
    names: Mapping[str, NameRow] | None = None,
    only: set[str] | None = None,
    school_id: int | None = None,
) -> ImportReport:
    """Walk the school's own directory tree. The FOLDER decides who someone is.

    A filename that does not match either pattern raises, naming the file. An unknown
    folder raises, naming the folder. Neither is guessed at, and neither is skipped: the
    legacy's characteristic failure was not getting identity wrong, it was INVENTING one
    and carrying on (spec §1.2).

    Every file is offered to `entry_for` -- no extension filter of our own, because that
    would be a second, silent definition of what we accept (see the note at the top).

    `names` is the school's ID -> name table. `only` is the carry-over list: the explicitly
    decided set of external_ids to take from an OLD tree, because taking a whole folder
    would recreate two known duplicate people and four who have no detectable face
    (`qorgan.identity.carryover`).

    The WHOLE tree is read, narrowed and reconciled BEFORE the first photograph is written,
    so a delivery that disagrees with the paperwork is refused while the database is still
    untouched -- rather than half-imported and then refused, which leaves somebody else to
    work out which half.
    """
    result = report or ImportReport()
    planned = _plan(root)

    if only is not None:
        check_all_found(only, {entry.external_id for _, entry in planned}, root)
        planned = [(photo, entry) for photo, entry in planned if entry.external_id in only]

    if names is not None:
        _check_names(planned, names)
        planned = [(photo, _with_name(entry, names)) for photo, entry in planned]

    for photo, entry in planned:
        _import_photo(photo, entry, recognizer, settings, result, school_id)

    return result


def _plan(root: Path) -> list[tuple[Path, RosterEntry]]:
    """Read the entire tree first. Nothing is written until every name has been understood."""
    planned = []
    for photo in sorted(p for p in root.rglob("*") if p.is_file()):
        if is_macos_metadata(photo.relative_to(root)):
            continue
        # The folder is the one directly containing the photo. Raises on both counts.
        planned.append((photo, entry_for(photo.parent.name, photo.name)))
    return planned


def _check_names(planned: list[tuple[Path, RosterEntry]], names: Mapping[str, NameRow]) -> None:
    """The names table covers PUPILS, and that exclusion is written here in the open.

    The 2026 delivery contains no staff at all; the school's six `staff` and two `учитель`
    photographs are carried across from the 2025 delivery under their old ids, and no row
    in the pupils' table names them. Scoping the comparison is therefore correct -- but it
    is a decision, so it is made at the call site where a reader can see it, not hidden
    inside `reconcile`, which compares exactly what it is given and filters nothing.
    """
    photos = {
        entry.external_id: entry.class_name
        for _, entry in planned
        if entry.person_type is PersonType.STUDENT
    }
    reconcile(photos, names).raise_if_disagreed()


def _with_name(entry: RosterEntry, names: Mapping[str, NameRow]) -> RosterEntry:
    """Joined on external_id. NEVER on the photo filename -- see `qorgan.identity.names`."""
    row = names.get(entry.external_id)
    return replace(entry, full_name=row.full_name) if row else entry


def import_archive(
    archive: Path,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    *,
    report: ImportReport | None = None,
    school_id: int | None = None,
) -> ImportReport:
    """The web upload. `safe_extract` + `import_directory` -- one import, not two."""
    staging = resolve(get_settings().media_root) / ".import" / archive.stem
    try:
        safe_extract(archive, staging)
        return import_directory(
            staging, recognizer, settings, report=report, school_id=school_id
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _import_photo(
    photo: Path,
    entry: RosterEntry,
    recognizer: FaceRecognizer,
    settings: FaceModelSettings,
    report: ImportReport,
    school_id: int | None,
) -> None:
    import cv2

    # NOT cv2.imread: it returns None for any non-ASCII path on Windows, and the class
    # folders are Cyrillic.
    image = cv2.imdecode(np.fromfile(str(photo), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        _store(
            entry, photo, None, settings, report, school_id,
            note="could not be decoded", faces=0,
        )
        return

    faces = recognizer.detect(image)
    face, note = _usable(faces)
    _store(entry, photo, face, settings, report, school_id, note=note, faces=len(faces))


def _usable(faces: list[DetectedFace]) -> tuple[DetectedFace | None, str | None]:
    """One face, big enough. Anything else is reported rather than guessed at."""
    if not faces:
        return None, "no face found"
    if len(faces) > 1:
        # Which one is the pupil? A group photograph cannot say, and guessing here poisons
        # the gallery for everybody.
        return None, f"{len(faces)} faces found; a roster photo must show one"

    face = faces[0]
    if min(face.width, face.height) < MIN_FACE_PX:
        return None, f"face too small ({face.width}x{face.height}, need {MIN_FACE_PX}px)"
    return face, None


def _store(
    entry: RosterEntry,
    photo: Path,
    face: DetectedFace | None,
    settings: FaceModelSettings,
    report: ImportReport,
    school_id: int | None,
    *,
    note: str | None,
    faces: int,
) -> None:
    """Write the person, the photo, and -- if the photo yielded a face -- the embedding.

    Idempotent on content, not just on the person: `ix_person_photos_sha256` was built to
    answer "is this exact photo already stored for this person?" and, before this, nothing
    ever asked it -- so re-importing an unchanged directory doubled every photo and every
    embedding. It is asked here, first, and a hit is a no-op.

    An unenrollable photo still gets a Person and a PersonPhoto carrying the reason. The
    person is on the roster whether or not we can recognise them, and recording WHY in the
    database is what lets `gallery-report` find them again next term rather than only in
    this run's stdout.

    Committed per person. The legacy did the whole import in one giant transaction while
    deleting files from disk BEFORE the commit, so a crash halfway through rolled the
    database back over photographs that were already gone (audit M-22).
    """
    from qorgan.paths import media_root, to_relative

    root = media_root()
    stored = _copy_into_media(photo, root, entry)
    digest = _digest(stored)

    with session_scope() as session:
        school = resolve_school_id(session, school_id)
        person = _get_or_create(session, entry, report, school)

        if _already_stored(session, person.id, digest, school) is not None:
            report.already_present += 1
            return

        _warn_if_shared_with_someone_else(session, digest, person)
        if note is not None:
            report.cannot_enrol(photo.name, note, faces)

        rel_path = to_relative(stored, root)
        _insert_photo(session, person.id, rel_path, digest, face, note, settings, report)


def _insert_photo(
    session: Session,
    person_id: int,
    rel_path: str,
    digest: str,
    face: DetectedFace | None,
    note: str | None,
    settings: FaceModelSettings,
    report: ImportReport,
) -> None:
    session.add(
        PersonPhoto(
            person_id=person_id,
            path=rel_path,
            sha256=digest,
            width=face.width if face else None,
            height=face.height if face else None,
            quality_note=note,
        )
    )
    report.photos += 1

    if face is None:
        return

    session.add(_embedding(person_id, face, settings))
    report.embeddings += 1


def _already_stored(
    session: Session, person_id: int, digest: str, school_id: int
) -> PersonPhoto | None:
    """"The same photo" means: same content hash, same person. That is what
    `ix_person_photos_sha256` is for -- nothing queried it before this."""
    return session.scalar(
        scope(select(PersonPhoto), PersonPhoto, school_id).where(
            PersonPhoto.person_id == person_id, PersonPhoto.sha256 == digest
        )
    )


def _warn_if_shared_with_someone_else(session: Session, digest: str, person: Person) -> None:
    # Scoped off `person`, whose school `_get_or_create` has just settled. Two schools
    # holding a byte-identical photo is not this school's data problem to be told about,
    # and the warning names a person_id -- another school's, if this were unscoped.
    """Two DIFFERENT children whose files are byte-identical is a data problem -- most
    likely one photo copied under the wrong id -- not a coincidence to merge silently. It
    is stored as its own photo for this person regardless; it is only ever LOUD, never
    dropped or merged."""
    other = session.scalar(
        scope(select(PersonPhoto), PersonPhoto, person.school_id).where(
            PersonPhoto.sha256 == digest, PersonPhoto.person_id != person.id
        )
    )
    if other is not None:
        logger.warning(
            "photo hash %s is already stored for person_id=%s (photo_id=%s) and is "
            "byte-identical to one now being imported for external_id=%s -- check these "
            "are not the same child's photo filed twice under two ids",
            digest[:12],
            other.person_id,
            other.id,
            person.external_id,
        )


def _embedding(person_id: int, face: DetectedFace, settings: FaceModelSettings) -> FaceEmbedding:
    """Which model produced this vector is recorded WITH it. The legacy shipped a rebuild
    script that wrote a different model's 512-d vectors into the same column."""
    return FaceEmbedding(
        person_id=person_id,
        model_name=settings.model_name,
        model_version=settings.model_version,
        dim=settings.embedding_dim,
        normalized=settings.normalized,
        vector=face.embedding.astype(np.float32).tobytes(),
    )


def _copy_into_media(photo: Path, root: Path, entry: RosterEntry) -> Path:
    folder = root / "people" / (entry.class_name or entry.position or "staff")
    folder.mkdir(parents=True, exist_ok=True)
    stored = folder / photo.name
    shutil.copy2(photo, stored)
    return stored


def _get_or_create(
    session: Session, entry: RosterEntry, report: ImportReport, school_id: int
) -> Person:
    """This school's person with that id, or a new one belonging to this school.

    The lookup is SCOPED; the module docstring records what an unscoped one would do.

    The name is FILLED IN when empty, and never overwritten once set.

    The school sent its ID -> name table in 2026, so `full_name` is no longer always NULL
    and `Ученик 333, 5-А` is now the fallback rather than the rule (see
    `qorgan.identity.naming.display_name`).

    An existing name is left alone on purpose. A name in the database may have been
    corrected by a human in the panel -- against this very table -- and a re-import
    silently reverting that correction is the legacy's `apply_runtime_migrations`
    behaviour, which re-derived `person_type` on every boot and undid manual fixes
    (audit H-02). Changing a name that is already recorded is a decision for a person,
    not a side effect of running an import twice.
    """
    person = session.scalar(
        select(Person).where(
            Person.external_id == entry.external_id, owned_by(Person, school_id)
        )
    )
    if person is not None:
        if entry.full_name and not person.full_name:
            person.full_name = entry.full_name
        return person

    person = Person(
        school_id=school_id,
        external_id=entry.external_id,
        # ROSTER. The school issued this id; we did not invent it.
        external_id_source=ExternalIdSource.ROSTER,
        # From the school's table, joined on the ID. NULL when it named nobody -- the
        # carried-over staff, for instance, whom the 2026 pupils' table does not cover.
        full_name=entry.full_name,
        person_type=entry.person_type,
        class_name=entry.class_name,
        position=entry.position,
    )
    session.add(person)
    session.flush()
    report.people += 1
    return person


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

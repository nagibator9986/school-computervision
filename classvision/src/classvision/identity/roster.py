"""The register of children this camera is allowed to name, and nothing more.

--------------------------------------------------------------------------------
**WHY A ROSTER MODULE EXISTS AT ALL, GIVEN THAT FACES DO NOT WORK HERE.**

It would be reasonable to ask why a 141-pupil register belongs in a module whose own
measurements (`MEASUREMENTS.md` §4) say face recognition cannot identify anyone on this
footage. The answer is that the register is not the *input* to recognition here, it is
the *bound* on naming. Three separate jobs, all of them refusals:

  1. **It closes the world.** `seatmap.py` may only write down a name that appears in
     this file. An operator who types «Ахметов» into a seat map for a child who is not on
     the register gets an error, not a new person. Without that, a seat map is a free-text
     field pointed at a child's psychological record.
  2. **It bounds the gallery.** The measured 0.30 best-cosine / 0.10 margin was obtained
     against **all 141 pupils across 13 classes**. The room holds nine. A gallery
     restricted to one `class_name` is a materially easier problem — 9 candidates instead
     of 141 — and `restrict_to()` is the whole reason `faces.py` has any chance of
     corroborating anything. This is not a performance optimisation; it changes the
     hypothesis space by a factor of ~15.
  3. **It states who is NOT in it.** The adult in this room appears in no roster row. Any
     matcher that always returns its best candidate will therefore return a *child's* name
     for the teacher. Open-set rejection is `assign.py`'s job, but it needs a register
     that is honest about being a closed list of pupils only.

--------------------------------------------------------------------------------
**DISCREPANCIES ARE RETURNED, NEVER RAISED.**

A missing photo is not a reason to refuse to analyse a lesson. The seat-based measurement
does not need photographs at all — it needs them only for the optional corroboration step
— so a school with 130 of 141 photos on file must still get its report, with the eleven
gaps listed. Raising here would convert a data-entry gap into a total outage, and the
person it would inconvenience is the psychologist, not whoever forgot to upload the photo.

So `load()` always returns a `Roster`; everything wrong with it is in `.discrepancies`,
each one naming its subject, and the caller decides what is fatal. `summary()` puts the
counts into the artefact so the report can say «фотографий не хватает у 11 из 141» rather
than quietly matching against a gallery with holes in it.

--------------------------------------------------------------------------------
**WHAT THIS FILE MEASURED ABOUT THE DATA IT WAS GIVEN.**

Run against the supplied `roster.csv` and `student_photos/` on 2026-08-12:

  * 141 rows, 13 classes (1-А … 9-А), 141 photos, **0 missing and 0 orphaned**.
  * The header carries a **UTF-8 BOM** (`ef bb bf`), so a plain `open(..., "utf-8")`
    yields a first column literally named `﻿external_id` and every lookup by
    `external_id` fails with a KeyError that reads like a missing column. `encoding=
    "utf-8-sig"` is not defensive coding here, it is the measured shape of this file.
  * Line endings are CRLF and names are Cyrillic with Kazakh patronymics
    (`…қызы` / `…ұлы`), so nothing may compare names case-folded against ASCII
    assumptions. Names are used for display only; every join goes through `external_id`.
  * `student_photos/` contains a `.DS_Store`. Photo discovery therefore globs known image
    extensions and skips dot-files, rather than listing the directory and hoping.
  * Photo file names are the numeric part of `external_id` only — `student_57` →
    `57.jpg` **or** `57.jpeg` (both extensions are present in the tree, in the same
    directories). Anything that assumed one extension would silently lose half the
    gallery, so the lookup tries all known extensions and reports the ambiguity if two
    files claim the same pupil.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The extensions actually present in the supplied tree, plus png because a school will
# eventually upload one. Order matters only for the ambiguity report below.
PHOTO_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# `external_id` is `student_<N>` and the photo is `<N>.<ext>`. The pattern is written out
# rather than assumed so that an id in another shape becomes a reported discrepancy
# instead of a silently photo-less pupil.
EXTERNAL_ID_PATTERN = re.compile(r"^student_(\d+)$")

REQUIRED_COLUMNS: tuple[str, ...] = ("external_id", "full_name", "class_name")


@dataclass(frozen=True, slots=True)
class Pupil:
    """One row of the register, plus the photograph that was found for it (or was not)."""

    external_id: str
    full_name: str
    class_name: str
    photo: Path | None
    row_number: int          # 1-based line in the CSV, for a discrepancy a human can find

    @property
    def has_photo(self) -> bool:
        return self.photo is not None

    def display(self) -> dict[str, str]:
        """The only three fields any surface is allowed to show for a named pupil."""
        return {"external_id": self.external_id, "full_name": self.full_name,
                "class_name": self.class_name}


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """Something wrong with the register, named precisely enough to be fixed.

    `kind` is a stable machine token and `detail_ru` is what a human reads. Both, because
    the web project needs to group these and the school office needs to act on them.
    """

    kind: str
    subject: str          # the external_id, class name or file path this is about
    detail_ru: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "subject": self.subject, "detail_ru": self.detail_ru}


# Every discrepancy kind this module can produce. Enumerated in one place because
# `assign.py` and the web importer both branch on them, and a token invented at the call
# site is a token nobody downstream handles.
KIND_PHOTO_MISSING = "photo_missing"
KIND_PHOTO_ORPHANED = "photo_orphaned"
KIND_PHOTO_AMBIGUOUS = "photo_ambiguous"
KIND_DUPLICATE_ID = "duplicate_external_id"
KIND_UNPARSABLE_ID = "unparsable_external_id"
KIND_MISSING_FIELD = "missing_field"
KIND_MISSING_COLUMN = "missing_column"
KIND_CLASS_DIR_UNKNOWN = "photo_directory_without_class"
KIND_CLASS_DIR_ABSENT = "class_without_photo_directory"


@dataclass(frozen=True, slots=True)
class Roster:
    """A closed list of pupils, the photographs found for them, and what did not add up."""

    pupils: tuple[Pupil, ...]
    discrepancies: tuple[Discrepancy, ...]
    csv_path: Path | None
    photos_root: Path | None
    class_name: str | None                      # None = the whole school
    _index: dict[str, Pupil] = field(default_factory=dict, repr=False)

    # -- lookup ---------------------------------------------------------------------

    def get(self, external_id: str) -> Pupil | None:
        """The pupil with this id, or None. Never a fuzzy match, never a nearest name."""
        return self._index.get(external_id)

    def __contains__(self, external_id: object) -> bool:
        return isinstance(external_id, str) and external_id in self._index

    def __len__(self) -> int:
        return len(self.pupils)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({p.class_name for p in self.pupils}))

    @property
    def with_photos(self) -> tuple[Pupil, ...]:
        return tuple(p for p in self.pupils if p.has_photo)

    def restrict_to(self, class_name: str) -> Roster:
        """The same register narrowed to one class — the single most useful call here.

        The measured matching problem was 141 candidates wide and returned a 0.10 margin.
        One class of this school is 1–23 pupils (median 12), so the same evidence is being
        asked a question an order of magnitude smaller. A caller that knows which class is
        in the room and does not call this is throwing away the only leverage available.

        An unknown `class_name` yields an EMPTY roster carrying a discrepancy, not an
        exception: a typo in a config must not take the analysis down, and an empty
        register simply means no seat may be named, which is the safe direction.
        """
        chosen = tuple(p for p in self.pupils if p.class_name == class_name)
        issues = list(self.discrepancies)
        if not chosen:
            issues.append(Discrepancy(
                kind=KIND_CLASS_DIR_ABSENT, subject=class_name,
                detail_ru=(f"В реестре нет класса «{class_name}»; известны: "
                           f"{', '.join(self.classes) or '—'}. Ни одно место не может "
                           f"получить имя."),
            ))
        return Roster(pupils=chosen, discrepancies=tuple(issues), csv_path=self.csv_path,
                      photos_root=self.photos_root, class_name=class_name,
                      _index={p.external_id: p for p in chosen})

    # -- provenance -----------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """What went into the artefact: the size and the holes, never the names.

        Deliberately carries no `full_name`. This block is written into every artefact,
        and an artefact that embeds 141 children's names for a lesson that named none of
        them has quietly become a personal-data export.
        """
        return {
            "csv": str(self.csv_path) if self.csv_path else None,
            "photos_root": str(self.photos_root) if self.photos_root else None,
            "restricted_to_class": self.class_name,
            "pupils": len(self.pupils),
            "pupils_with_photo": len(self.with_photos),
            "classes": list(self.classes),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }


def load(csv_path: str | Path, photos_root: str | Path | None = None, *,
         class_name: str | None = None) -> Roster:
    """Read the register and pair it with the photo tree. Never raises on bad data.

    The only exception that escapes is a missing/unreadable CSV, because "there is no
    register" is a different situation from "the register disagrees with the photos": the
    caller asked for a specific file and it is not there.
    """
    csv_path = Path(csv_path)
    photos_root = Path(photos_root) if photos_root is not None else None

    pupils: list[Pupil] = []
    issues: list[Discrepancy] = []
    seen: dict[str, int] = {}

    # utf-8-sig, not utf-8. See the module docstring: this file's header begins ef bb bf,
    # and the failure mode without this is a column named "﻿external_id".
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        for column in REQUIRED_COLUMNS:
            if column not in columns:
                issues.append(Discrepancy(
                    kind=KIND_MISSING_COLUMN, subject=column,
                    detail_ru=(f"В файле реестра нет колонки «{column}» "
                               f"(есть: {', '.join(columns) or '—'})."),
                ))
        if any(i.kind == KIND_MISSING_COLUMN for i in issues):
            return Roster(pupils=(), discrepancies=tuple(issues), csv_path=csv_path,
                          photos_root=photos_root, class_name=class_name, _index={})

        for row_number, row in enumerate(reader, start=2):
            external_id = (row.get("external_id") or "").strip()
            full_name = (row.get("full_name") or "").strip()
            row_class = (row.get("class_name") or "").strip()

            if not external_id or not full_name or not row_class:
                issues.append(Discrepancy(
                    kind=KIND_MISSING_FIELD, subject=external_id or f"строка {row_number}",
                    detail_ru=(f"Строка {row_number}: пустое поле "
                               f"(external_id={external_id!r}, full_name={full_name!r}, "
                               f"class_name={row_class!r}). Строка пропущена."),
                ))
                continue

            if external_id in seen:
                issues.append(Discrepancy(
                    kind=KIND_DUPLICATE_ID, subject=external_id,
                    detail_ru=(f"external_id «{external_id}» встречается в строках "
                               f"{seen[external_id]} и {row_number}. Взята первая; "
                               f"вторая пропущена."),
                ))
                continue
            seen[external_id] = row_number

            photo, photo_issue = _find_photo(photos_root, external_id, row_class)
            if photo_issue is not None:
                issues.append(photo_issue)

            pupils.append(Pupil(external_id=external_id, full_name=full_name,
                                class_name=row_class, photo=photo,
                                row_number=row_number))

    if photos_root is not None:
        issues.extend(_orphaned_photos(photos_root, pupils))

    roster = Roster(pupils=tuple(pupils), discrepancies=tuple(issues), csv_path=csv_path,
                    photos_root=photos_root, class_name=None,
                    _index={p.external_id: p for p in pupils})
    return roster.restrict_to(class_name) if class_name else roster


def _find_photo(photos_root: Path | None, external_id: str, class_name: str
                ) -> tuple[Path | None, Discrepancy | None]:
    """`student_57` in class `5-А` -> `student_photos/5-А/57.jpg` or `…/57.jpeg`.

    Both extensions occur in the same directories in the supplied tree, so all of them are
    tried. Two files for one pupil is reported rather than resolved by ordering: which of
    two photographs of a child is the right one is not a decision a glob should make.
    """
    if photos_root is None:
        return None, None

    match = EXTERNAL_ID_PATTERN.match(external_id)
    if match is None:
        return None, Discrepancy(
            kind=KIND_UNPARSABLE_ID, subject=external_id,
            detail_ru=(f"external_id «{external_id}» не имеет вида «student_<номер>», "
                       f"поэтому имя файла фотографии неизвестно."),
        )

    number = match.group(1)
    directory = photos_root / class_name
    found = [directory / f"{number}{ext}" for ext in PHOTO_EXTENSIONS
             if (directory / f"{number}{ext}").is_file()]

    if not found:
        return None, Discrepancy(
            kind=KIND_PHOTO_MISSING, subject=external_id,
            detail_ru=(f"Нет фотографии: ожидался файл {directory}/{number}"
                       f"{{{','.join(e.lstrip('.') for e in PHOTO_EXTENSIONS)}}}."),
        )
    if len(found) > 1:
        return found[0], Discrepancy(
            kind=KIND_PHOTO_AMBIGUOUS, subject=external_id,
            detail_ru=(f"Несколько файлов фотографии: "
                       f"{', '.join(p.name for p in found)}. Взят {found[0].name}; "
                       f"выбор должен сделать человек."),
        )
    return found[0], None


def _orphaned_photos(photos_root: Path, pupils: Iterable[Pupil]) -> list[Discrepancy]:
    """Photographs of children who are not on the register.

    Worth reporting for a reason that is not tidiness: an orphaned photo usually means a
    pupil left the school and their picture stayed. It is in the folder, so a matcher that
    built its gallery from the folder rather than from the register would happily return
    the name of a child who is no longer enrolled. This module builds galleries from the
    register; the check exists to prove the two agree.
    """
    issues: list[Discrepancy] = []
    if not photos_root.is_dir():
        return [Discrepancy(kind=KIND_PHOTO_MISSING, subject=str(photos_root),
                            detail_ru=f"Папки с фотографиями нет: {photos_root}.")]

    known_classes = {p.class_name for p in pupils}
    expected: set[Path] = {p.photo for p in pupils if p.photo is not None}  # type: ignore[misc]

    for directory in sorted(d for d in photos_root.iterdir() if d.is_dir()):
        if directory.name not in known_classes:
            issues.append(Discrepancy(
                kind=KIND_CLASS_DIR_UNKNOWN, subject=directory.name,
                detail_ru=(f"Папка «{directory.name}» есть в фотографиях, но такого "
                           f"класса нет в реестре."),
            ))
        for path in sorted(directory.iterdir()):
            # Skip dot-files: the supplied tree contains a .DS_Store, and reporting macOS
            # metadata as a missing pupil would train the reader to ignore this list.
            if path.name.startswith(".") or not path.is_file():
                continue
            if path.suffix.lower() not in PHOTO_EXTENSIONS:
                continue
            if path not in expected:
                issues.append(Discrepancy(
                    kind=KIND_PHOTO_ORPHANED, subject=str(path.relative_to(photos_root)),
                    detail_ru=(f"Фотография {path.relative_to(photos_root)} не "
                               f"соответствует ни одной строке реестра."),
                ))
    return issues


def main(argv: list[str] | None = None) -> int:
    """`python -m classvision.identity.roster roster.csv student_photos` — the audit.

    Prints the register's size and every discrepancy. Exists so that a school can check
    its own data before anybody runs an eleven-minute pose pass over a lesson.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Проверить реестр учеников и фотографии")
    parser.add_argument("csv", type=Path)
    parser.add_argument("photos", type=Path, nargs="?", default=None)
    parser.add_argument("--class-name", default=None)
    args = parser.parse_args(argv)

    roster = load(args.csv, args.photos, class_name=args.class_name)
    print(f"учеников: {len(roster)}  с фотографией: {len(roster.with_photos)}")
    print(f"классов:  {len(roster.classes)}  ({', '.join(roster.classes)})")
    if not roster.discrepancies:
        print("расхождений нет")
        return 0
    print(f"расхождений: {len(roster.discrepancies)}")
    for issue in roster.discrepancies:
        print(f"  [{issue.kind}] {issue.subject}: {issue.detail_ru}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

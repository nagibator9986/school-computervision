"""The school's ID -> name table, and the check that it agrees with the photographs.

**Why CSV and not the .xlsx the school actually sent.**

`openpyxl` is not a dependency of this project and is not being added for this. But the
deciding reason is not the dependency, it is that **a CSV can be read by a human and an
.xlsx cannot.** This file is the full names of 141 children. Whoever imports it should be
able to open it, in anything, and see exactly what they are about to put in a database --
without trusting a library, a spreadsheet's autocorrect, or this code. A format nobody can
inspect is a format nobody can check, and the thing being loaded here is a list of
children.

**Converting the .xlsx is a documented one-off, not a feature.** The CSV it produces is
carried in by hand and is NEVER committed -- it is personal data about children, and it
travels the way the photographs travel.

    external_id,full_name,class_name
    student_1,Иванов Иван Иванович,1А

An `.xlsx` is a zip of XML, so the standard library alone reads it -- no dependency, and
nothing to install on the school's machine. The recipe, which produced the 2026 CSV:

    import csv, zipfile
    from xml.etree import ElementTree

    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile("список.xlsx") as zf:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{NS}t"))
            for si in ElementTree.fromstring(zf.read("xl/sharedStrings.xml")).findall(f"{NS}si")
        ]
        sheet = ElementTree.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    # A cell with t="s" holds an INDEX into `shared`, not text. Read the header row to find
    # which column letter is `ID`, `ФИО` and `Класс` rather than assuming A/B/C -- the
    # school may reorder them, and reading names out of the wrong column is silent.

The sheet also has a `Photo Name` column. Do not write it to the CSV and do not join on
it; see below.

**The join is on `external_id`, and never on the photo filename.** The 2026 delivery's
sheet has a `Photo Name` column, and it is wrong for three of the 141 rows: it says
`10.jpg` and `12.jpg` where the disk has `10.jpeg`, and `99.jpeg` where the disk has
`99.jpg`. Joining on that column matches 138 of 141 and loses three children in silence --
they keep a photograph, lose their name, and appear on every report as `Ученик 10, 1А`
while their classmates appear by name. Joining on the id matches 141 of 141. There is a
test that fails if anyone changes it back.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("external_id", "full_name", "class_name")


class BadNamesFile(ValueError):
    """The names CSV cannot be read as written. Never a row quietly dropped."""


class RosterMismatch(ValueError):
    """The names table and the photographs do not describe the same set of people."""


@dataclass(frozen=True, slots=True)
class NameRow:
    """One line of the school's table. `class_name` is what the SHEET says, for checking."""

    external_id: str
    full_name: str
    class_name: str | None


def read_names(path: Path) -> dict[str, NameRow]:
    """Read the CSV into external_id -> row. Every defect is loud and names the line.

    A blank name is refused rather than stored: `display_name` falls back to `Ученик 10,
    1А` when `full_name` is empty, so a blank cell would look exactly like a child the
    school never sent a name for -- the one thing this file exists to tell apart.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _check_header(path, reader.fieldnames)
        rows: dict[str, NameRow] = {}
        for line_no, raw in enumerate(reader, start=2):
            row = _row(path, line_no, raw)
            if row.external_id in rows:
                raise BadNamesFile(
                    f"{path}: line {line_no}: external_id {row.external_id!r} appears "
                    "twice. Two rows for one id is two different claims about one child; "
                    "the school has to say which is right."
                )
            rows[row.external_id] = row

    if not rows:
        raise BadNamesFile(f"{path}: no data rows. An empty names table is not 'no names'.")
    return rows


def _check_header(path: Path, fieldnames: list[str] | None) -> None:
    if fieldnames is None or tuple(name.strip() for name in fieldnames) != COLUMNS:
        raise BadNamesFile(
            f"{path}: header is {fieldnames!r}, expected {list(COLUMNS)}. The columns are "
            "fixed on purpose -- a file whose columns we guess at is a file we can read "
            "the wrong way round, and these are children's names."
        )


def _row(path: Path, line_no: int, raw: Mapping[str, str | None]) -> NameRow:
    values = {key: (raw.get(key) or "").strip() for key in COLUMNS}
    for key in ("external_id", "full_name"):
        if not values[key]:
            raise BadNamesFile(
                f"{path}: line {line_no}: {key} is empty ({dict(raw)!r}). A blank name is "
                "indistinguishable from a child the school never named."
            )
    return NameRow(
        external_id=values["external_id"],
        full_name=values["full_name"],
        class_name=values["class_name"] or None,
    )


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Who the photographs and the names table agree about, and who they do not."""

    matched: tuple[str, ...]
    without_photo: tuple[str, ...]  # a name row nobody photographed
    without_name: tuple[str, ...]  # a photograph nobody named
    wrong_class: tuple[tuple[str, str, str], ...]  # (external_id, sheet, folder)

    @property
    def agreed(self) -> bool:
        return not (self.without_photo or self.without_name or self.wrong_class)

    def raise_if_disagreed(self) -> None:
        """Loud, before anything is written. Not a warning, and never a quiet skip.

        A child on the sheet with no photograph can never be recognised, and a photograph
        with no row gets imported under a number instead of a name. Both are things the
        school can fix in an afternoon and neither is visible if the import prints a
        cheerful total and carries on.
        """
        if self.agreed:
            return
        raise RosterMismatch("\n".join(self._complaints()))

    def _complaints(self) -> list[str]:
        lines = [
            f"the names table and the photographs disagree ({len(self.matched)} matched):"
        ]
        if self.without_photo:
            lines.append(
                f"  {len(self.without_photo)} named with no photograph: "
                f"{', '.join(self.without_photo)}"
            )
        if self.without_name:
            lines.append(
                f"  {len(self.without_name)} photographed with no name: "
                f"{', '.join(self.without_name)}"
            )
        for external_id, sheet, folder in self.wrong_class:
            lines.append(
                f"  {external_id}: the sheet says class {sheet!r}, the photograph is in "
                f"folder {folder!r}"
            )
        lines.append(
            "Ask the school. Importing part of a roster is how a child ends up in the "
            "system as somebody else."
        )
        return lines


def reconcile(
    photos: Mapping[str, str | None], names: Mapping[str, NameRow]
) -> Reconciliation:
    """Compare BY ID. `photos` maps external_id -> the class its folder says.

    Symmetric and total: it filters nothing on its own. Deciding WHICH ids are in scope --
    the 2026 delivery names pupils only, so the carried-over staff are not in this table --
    is the caller's decision, made where it can be seen, not a quiet exemption in here.
    """
    photo_ids, name_ids = set(photos), set(names)
    both = photo_ids & name_ids
    wrong = tuple(
        (external_id, names[external_id].class_name or "", photos[external_id] or "")
        for external_id in sorted(both)
        if names[external_id].class_name
        and not _same_class(names[external_id].class_name, photos[external_id])
    )
    return Reconciliation(
        matched=tuple(sorted(both)),
        without_photo=tuple(sorted(name_ids - photo_ids)),
        without_name=tuple(sorted(photo_ids - name_ids)),
        wrong_class=wrong,
    )


def _same_class(sheet: str, folder: str | None) -> bool:
    """`5А`, `5-А` and `5 а` are one class -- the sheet and the disk punctuate differently."""
    from qorgan.identity.naming import normalise_class

    if folder is None:
        return False
    return normalise_class(sheet) == normalise_class(folder)

"""Class names and display names. Pure: no I/O, no config, no database.

`display_name` is the one place the fallback name is written. The school sent its ID ->
name table with the 2026 delivery, so `full_name` is now usually set and `Ученик 333, 5-А`
is what a person WITHOUT one is called -- staff carried over from 2025, whom that table
does not cover, and anyone imported before it arrived. Written once, used by the web, the
reports and the CLI alike -- the legacy wrote its equivalent three times and the three
disagreed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from qorgan.enums import PersonType

# `5А`, `5-А` and `5 а` are one class. Hyphens (of every flavour) and spaces are noise.
_CLASS_NOISE = re.compile(r"[\s\-‐-―]+")
_CLASS = re.compile(r"^(1[01]|[1-9])[А-ЯЁ]$")

_NUMBER = re.compile(r"(\d+)")


def normalise_class(raw: str) -> str:
    """`5А`, `5-А` and `5 а` are one class. The legacy stored all three.

    Used to RECOGNISE a class folder. The class we STORE is the folder name verbatim,
    because it is shown to a human: `Ученик 333, 5-А`, hyphen and all.
    """
    text = unicodedata.normalize("NFKC", raw).strip()
    return _CLASS_NOISE.sub("", text).upper()


def is_class_folder(name: str) -> bool:
    """`1-А` .. `11-Б`. Anything else is staff, or a question for the school."""
    return bool(_CLASS.fullmatch(normalise_class(name)))


class Named(Protocol):
    """Anything with an identity. `Person`, `PersonInfo` and `Meal` all satisfy it."""

    external_id: str
    full_name: str | None
    person_type: PersonType
    class_name: str | None
    position: str | None


def display_name(person: Named) -> str:
    """What a human sees. **The name is a display field, not an identity.**

    The school's ID -> name table arrived with the 2026 delivery and is loaded by
    `qorgan.identity.names`, joined on the id. That changes what is on the screen and
    nothing else: the canteen record is still keyed on `external_id` -- `student_333` --
    and never on a name. The legacy keyed identity on "Surname Firstname" parsed out of a
    filename, so two children with the same name in the same class became one person.

    For anyone the table does not name, the id and the class remain the honest answer.

    Written once. The legacy wrote its equivalent in three places and the three disagreed.
    """
    if person.full_name:
        return person.full_name

    number = _number(person.external_id)

    if person.person_type is PersonType.STAFF:
        if person.position:
            # `учитель` -> `Учитель 469`. The FOLDER said teacher; the filename lied.
            return f"{person.position.capitalize()} {number}"
        return f"Сотрудник {number}"

    if person.class_name:
        return f"Ученик {number}, {person.class_name}"
    return f"Ученик {number}"


def _number(external_id: str) -> str:
    """`student_333` -> `333`. The school's own id, which is the whole point."""
    match = _NUMBER.search(external_id)
    return match.group(1) if match else external_id

"""The day `school_id` lands, every unfiltered query becomes a leak between schools.

This is the guard, and it exists BEFORE the column does, for the same reason
`test_web_auth.py` walks the real route table instead of listing the routes somebody
remembered: the dangerous query is never the one anybody thought about. It is the one
nobody did.

WHAT IT ENFORCES. `tests/tenancy_scan.py` finds every `select` / `update` / `delete` /
`session.get` in `src/` and reports which mapped classes each one touches. This module
says which of those classes belong to a school, and then requires -- of every single
query that touches one -- that something in the statement names `school_id`. Deny by
default: a query nobody thought about fails, it is not skipped. The exemption list starts
empty, and every entry costs a human a written sentence.

WHAT IS A ROOT TABLE, AND WHY THOSE. A root table is one that **nothing else in the
schema can answer for**: it has no NOT NULL foreign key into another school's data, so if
it does not say which school it belongs to, nobody can. `cameras`, `persons`, `users` and
`meal_windows` are those four, and they are the four that get the column.

Everything else reaches a school through a key that cannot be null -- an event through its
camera, a photograph through its person, a lesson track through its lesson -- so a second
`school_id` on those tables would be a second answer to a question that already has one,
and this codebase has been bitten twice by a value that was true in one layer and quietly
wrong in the next. `test_a_derived_table_reaches_a_school_through_a_key_that_cannot_be_
null` checks that premise against the metadata rather than trusting this paragraph, so a
new table cannot be quietly filed as derived while dangling.

**A derived table still has to be filtered.** It just has to be filtered by joining the
root that knows. That is deliberately more work at the call site, and it is the work that
keeps the two answers from disagreeing.

WHAT THIS GUARD CANNOT DO, said here rather than discovered later. It proves a school was
NAMED in the statement. It does not prove the join was the right one -- an event joined
to `persons` rather than to `cameras` would satisfy it and still leak. That half is
`tests/test_tenancy_isolation.py`, which puts two schools in one database and tries to
read across. Neither test is sufficient alone: one catches the query nobody wrote a filter
for, the other catches the filter that does not filter.
"""

from __future__ import annotations

import pytest

from qorgan.db.models import Base
from tests.conftest import SRC_DIR
from tests.tenancy_registry import (DERIVED_MODELS, INSTALLATION_MODELS, ROOT_MODELS,
                                    SCHOOL_COLUMN, TENANT_MODELS, UNATTRIBUTED_QUERIES,
                                    UNSCOPED_QUERIES)
from tests.tenancy_scan import scan

SITES = scan(SRC_DIR)


def _tenant_sites() -> list:
    return [site for site in SITES if site.models & TENANT_MODELS]


def _model_names() -> frozenset[str]:
    return frozenset(mapper.class_.__name__ for mapper in Base.registry.mappers)


def test_every_mapped_class_says_whose_it_is() -> None:
    """A new table is a school's or the installation's. There is no third answer.

    Deny by default at the schema level, one layer above the query guard: a table nobody
    classified would touch no tenant model, so every query against it would pass the guard
    below in silence. This is what makes that impossible.
    """
    classified = TENANT_MODELS | INSTALLATION_MODELS
    mapped = _model_names()
    assert mapped - classified == frozenset(), (
        f"these mapped classes belong to nobody: {sorted(mapped - classified)}. Put each "
        "in TENANT_MODELS (a school's data -- then decide root or derived) or in "
        "INSTALLATION_MODELS (the server's), with the reason. Until then every query "
        "against them is unguarded and this suite would not have said so."
    )
    assert classified - mapped == frozenset(), (
        f"{sorted(classified - mapped)} is classified here but no longer mapped. A "
        "classification that outlives its table is a line nobody rechecks."
    )


@pytest.mark.parametrize("model", sorted(ROOT_MODELS))
def test_a_root_table_carries_the_school_itself(model: str) -> None:
    """Nothing else in the schema can answer for these four, so they answer themselves."""
    mapper = next(m for m in Base.registry.mappers if m.class_.__name__ == model)
    columns = {column.name for column in mapper.local_table.columns}
    assert SCHOOL_COLUMN in columns, (
        f"{mapper.local_table.name} has no {SCHOOL_COLUMN}. It is a root table: no other "
        "table's foreign key can say which school its rows belong to, so without this "
        "column two schools' rows are indistinguishable and every query below is a leak."
    )


@pytest.mark.parametrize("model", sorted(DERIVED_MODELS))
def test_a_derived_table_reaches_a_school_through_a_key_that_cannot_be_null(
    model: str,
) -> None:
    """The premise of the root/derived split, checked against the metadata.

    A table filed as derived is claiming that some other tenant table answers for it. If
    the key it would answer through is nullable, that claim is false for every row where
    it IS null -- and those rows belong to no school while looking perfectly ordinary.
    """
    mapper = next(m for m in Base.registry.mappers if m.class_.__name__ == model)
    tables = {m.local_table.name: m.class_.__name__ for m in Base.registry.mappers}
    anchors = [
        key.column.table.name
        for column in mapper.local_table.columns
        if not column.nullable
        for key in column.foreign_keys
        if tables.get(key.column.table.name) in TENANT_MODELS
    ]
    assert anchors, (
        f"{mapper.local_table.name} is filed as derived -- meaning another school table "
        "answers for it -- but it has no NOT NULL foreign key into one. Either give it "
        f"{SCHOOL_COLUMN} and move it to ROOT_MODELS, or make the key non-nullable."
    )


@pytest.mark.parametrize("site", _tenant_sites(), ids=lambda s: s.name)
def test_every_query_against_a_school_table_names_a_school(site) -> None:
    """The guard. One parameter per query in `src/` that touches a school's data."""
    if site.constrains(SCHOOL_COLUMN, TENANT_MODELS):
        return
    if site.name in UNSCOPED_QUERIES:
        return

    touched = sorted(site.models & TENANT_MODELS)
    how = (
        "a `.where(Model.school_id == ...)`"
        if site.models & ROOT_MODELS
        else "a join to the root table that knows (see the module docstring)"
    )
    pytest.fail(
        f"{site.module}:{site.lineno} -- this {site.kind} touches {touched}, which is "
        f"one school's data, and nothing in the statement names {SCHOOL_COLUMN}. On a "
        "single-school installation that reads as harmless. On the day a second school "
        "exists it hands one school's children to another, and nothing anywhere says so: "
        "the page renders, the count is larger, the export is longer.\n\n"
        f"Add {how}. If this query is genuinely installation-wide, add\n"
        f'    "{site.name}": "<why>"\n'
        "to UNSCOPED_QUERIES -- a sentence a person can check, not a shrug."
    )


def test_no_query_is_too_dynamic_to_attribute() -> None:
    """A statement built on a variable class. Nothing can be concluded; say so out loud.

    `identity/merge.py::_repoint` is `update(model)` where `model` is a `type` parameter.
    The scan cannot know which table that is, so it cannot judge the query -- and a guard
    that silently skips what it cannot analyse is the disease this file treats. It is a
    hard failure with a named answer rather than a per-table exemption, because there is
    no table to hang one on.
    """
    opaque = {site.name for site in SITES if not site.models}
    unaccounted = sorted(opaque - set(UNATTRIBUTED_QUERIES))
    assert not unaccounted, (
        "these statements name no table this scan can resolve, so it cannot tell whether "
        "they are filtered:\n\n"
        + "\n".join(f"  {name}" for name in unaccounted)
        + "\n\nPrefer making the statement resolvable (name the class literally). If it "
        "genuinely cannot be -- a helper that repoints rows for several tables -- add it "
        "to UNATTRIBUTED_QUERIES with the invariant that keeps it inside one school."
    )


def test_no_exemption_outlives_its_reason() -> None:
    """An exemption nobody rechecks is how an allow-list becomes a rubber stamp."""
    known = {site.name for site in SITES}
    for listed, label in ((UNSCOPED_QUERIES, "UNSCOPED"), (UNATTRIBUTED_QUERIES, "UNATTR")):
        gone = sorted(set(listed) - known)
        assert not gone, (
            f"{label} names {gone}, which is no longer a query in src/. The exemption "
            "outlived the statement it excused. Delete the entry."
        )

    spare = sorted(
        name
        for name in UNSCOPED_QUERIES
        for site in SITES
        if site.name == name and site.constrains(SCHOOL_COLUMN, TENANT_MODELS)
    )
    assert not spare, (
        f"{spare} now names a school in the statement itself, so the excuse is spent. "
        "Delete the entry and let the guard hold it."
    )


def test_the_scan_found_the_queries_that_are_there() -> None:
    """A scan that resolved nothing would pass everything above by vacuity."""
    assert len(SITES) > 50, f"only {len(SITES)} query sites found in src/ -- scan broken?"
    assert len(_tenant_sites()) > 40, "almost no query touches a school table -- unlikely"
    assert any(site.kind == "get" for site in SITES), "no session.get() found"
    assert any(len(site.models) > 1 for site in SITES), "no joined query found"

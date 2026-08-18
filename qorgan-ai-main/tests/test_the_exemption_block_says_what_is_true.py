"""The exemption block makes claims about itself. These are the checks on those claims.

`tenancy_registry.py` excuses 26 queries from naming a school, each with a written
reason. Prose defends them; prose also gets things wrong, and three of its claims were
wrong at once until a review measured them:

  * it said **eleven** exemptions where the list holds **26** -- eleven is one sub-block --
    and that is the number its owner was given;
  * it said all eleven of that sub-block rest on the worker's refusal to run against two
    schools, when **two of them run in the supervisor**, which does not refuse at all;
  * it recorded the refusal itself as a NOTE, and a note is not a check.

So each claim gets a test here. The counts are asserted against the comment BY READING IT
AS TEXT, because a number in a comment is a number nobody re-measures -- `HANDOFF.md`
records a previous watchdog note in this repository that "asked you to notice, and twelve
commits later nobody had re-measured the number, because a guard that only asks you to
notice is not a check".

**WHAT A FAILURE HERE MEANS, AND WHAT IT DOES NOT.** Red does not mean somebody broke
something admirable. For the refusal tests it means an entry point that used to refuse now
answers -- which is exactly what plumbing a school through the worker looks like, and is a
perfectly reasonable change to make. It means: **go and re-read the nine entries that rest
on that refusal before deleting the test**, because the argument each of them makes has
just stopped being true, and every one of them touches a child's meal record or a
classroom. For the count test it means the comment and the dictionaries have drifted, and
the comment is what a human reads.

The one-school control is not decoration. Without it, every refusal assertion here would
also pass if the entry points raised for some unrelated reason -- a missing table, a broken
import -- and this file would be reporting a refusal that had nothing to do with a second
school.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from qorgan.canteen.reports import day_report
from qorgan.classroom.reports import recent_lessons
from qorgan.db.models import School
from qorgan.db.models.school import UndecidedSchool
from qorgan.events.store import camera_id_for, ensure_cameras
from qorgan.faces.gallery import load_gallery
from qorgan.identity.merge import resolve_external

# The entry points the worker and the CLI reach the database through, each called the way
# an un-plumbed caller calls it: naming no school. These are the six that NINE of the
# worker exemptions stand behind -- not an arbitrary sample. The other two of that
# sub-block run in the supervisor and are covered by `test_the_two_supervisor_sweeps...`.
ENTRY_POINTS: dict[str, Callable[[], object]] = {
    "events.store.ensure_cameras": lambda: ensure_cameras({}),
    "events.store.camera_id_for": lambda: camera_id_for("hall_left"),
    "faces.gallery.load_gallery": lambda: load_gallery("buffalo_l", "1"),
    "identity.merge.resolve_external": lambda: resolve_external("7"),
    "canteen.reports.day_report": lambda: day_report(date(2026, 3, 4)),
    "classroom.reports.recent_lessons": lambda: recent_lessons(10),
}


@pytest.fixture
def a_second_school(session: Session) -> Iterator[Session]:
    """`conftest.session` creates one school; this adds the second and nothing else.

    No cameras, no people, no lessons. The refusal being measured is about how many
    SCHOOLS exist, so any other row would only add ways for a call to fail for a reason
    that is not the one under test.
    """
    session.add(School(slug="gymnasium-4", name="Гимназия №4"))
    session.commit()
    yield session


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_an_un_plumbed_entry_point_refuses_rather_than_choosing(
    a_second_school: Session, name: str
) -> None:
    """Two schools, nothing plumbed: the call must stop, by name."""
    with pytest.raises(UndecidedSchool) as refused:
        ENTRY_POINTS[name]()

    assert "2 of them" in str(refused.value), (
        f"{name} raised UndecidedSchool, but not the several-schools refusal -- so this "
        f"parameter is not measuring what it claims. Message was: {refused.value}"
    )


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_the_same_entry_point_does_not_refuse_on_one_school(
    session: Session, name: str
) -> None:
    """The control, and the whole reason the test above means anything.

    One school, everything else identical. These calls may still fail -- `resolve_external`
    raises `LookupError` because no pupil holds id 7 on an empty database, which is correct
    -- but none of them may raise `UndecidedSchool`. If one did, the refusal above would be
    about something other than the second school, and nine exemptions would be resting on a
    fact this file had failed to measure.
    """
    try:
        ENTRY_POINTS[name]()
        raised: Exception | None = None
    except Exception as exc:  # the TYPE is precisely what is asserted below
        # Kept rather than swallowed. Any failure other than `UndecidedSchool` is this
        # call's own business on an empty database and not what this file is about --
        # pinning down the exact behaviour of six unrelated entry points against no data
        # would make this test fail for reasons that have nothing to do with tenancy.
        raised = exc

    assert not isinstance(raised, UndecidedSchool), (
        f"{name} refused to choose a school on an installation that has exactly ONE. That "
        "breaks every single-school deployment -- which is every deployment today -- and "
        "it makes the two-school assertion above vacuous, because the refusal it measures "
        f"would not be about the second school at all. Message was: {raised}"
    )


# The four numbers the exemption block states about itself, as they are WRITTEN there. The
# test below reads that file as text, so a comment that drifts from the dictionaries fails
# rather than quietly misinforming the next person to summarise this branch.
COUNT_CLAIMS = (
    "UNSCOPED_QUERIES holds {unscoped} entries",
    "UNATTRIBUTED_QUERIES holds {unattributed}",
    "total exempted {total}",
    "Of those, {worker} are the worker sub-block",
)

WORKER_MODULES = ("qorgan/canteen/sessions.py", "qorgan/classroom/store.py")

# The two worker-sub-block entries that run in the SUPERVISOR, not the worker, and so are
# NOT protected by the refusal the tests above measure. Both stand on their own
# installation-wide-janitor argument instead; see the block in `tenancy_registry.py`.
SUPERVISOR_SWEEPS = (
    "qorgan/canteen/sessions.py::close_sessions_nobody_exited._sweep::1",
    "qorgan/classroom/store.py::close_stale_lessons._sweep::1",
)


def test_the_block_states_its_own_size_correctly() -> None:
    """The comment is what a human reads, so the comment is what gets asserted.

    This exists because the number in it was wrong: the owner of this branch was told
    "eleven exemptions" when the list held 26. Eleven is the worker sub-block. Counting a
    sub-block as the whole is not a rounding error -- it understated the exempted queries
    by more than half, on a branch whose entire subject is which queries are allowed not to
    name a school.
    """
    from tests.tenancy_registry import UNATTRIBUTED_QUERIES, UNSCOPED_QUERIES

    source = (Path(__file__).parent / "tenancy_registry.py").read_text(encoding="utf-8")
    real = {
        "unscoped": len(UNSCOPED_QUERIES),
        "unattributed": len(UNATTRIBUTED_QUERIES),
        "total": len(UNSCOPED_QUERIES) + len(UNATTRIBUTED_QUERIES),
        "worker": sum(1 for key in UNSCOPED_QUERIES if key.startswith(WORKER_MODULES)),
    }
    missing = [claim.format(**real) for claim in COUNT_CLAIMS if claim.format(**real) not in source]

    assert not missing, (
        "the exemption block's own count comment no longer matches the dictionaries. It "
        f"should say, verbatim: {missing}. Real counts are {real}. Update the comment in "
        "tests/tenancy_registry.py -- and if you are about to summarise this branch for "
        "somebody, quote the TOTAL, not the worker sub-block."
    )


def test_the_two_supervisor_sweeps_do_not_rest_on_the_workers_refusal() -> None:
    """Nine of the eleven worker exemptions rest on the refusal. These two do not.

    They run in the supervisor process, which imports NONE of the six entry points the
    tests above measure and reads its sweep rules from YAML rather than from the database.
    On a two-school installation the supervisor therefore keeps sweeping the whole
    installation quite happily while the workers crash-loop -- so the refusal protects
    nothing here, and the block must not claim it does.

    It costs nothing, because both entries already carry a self-sufficient argument: an
    installation-wide janitor on an installation-wide rule, closing identical rows and
    reaching no user. This test pins the PREMISE of that split. If somebody makes the
    supervisor depend on a refusing entry point, the two arguments stop being independent
    and both entries need re-reading.
    """
    from tests.tenancy_registry import UNSCOPED_QUERIES

    for key in SUPERVISOR_SWEEPS:
        assert key in UNSCOPED_QUERIES, f"{key} is no longer exempted; delete it from here too"

    supervisor = (
        Path(__file__).parents[1] / "src" / "qorgan" / "supervisor" / "supervisor.py"
    ).read_text(encoding="utf-8")
    imported = sorted(
        name for name in ENTRY_POINTS if name.split(".")[-1] in supervisor
    )

    assert not imported, (
        f"the supervisor now reaches {imported}, which is one of the six entry points that "
        "REFUSE on a two-school database. That changes the argument for "
        f"{list(SUPERVISOR_SWEEPS)}: they were exempted as installation-wide janitors "
        "standing on their own, explicitly NOT on the refusal. Re-read both entries in "
        "tests/tenancy_registry.py."
    )

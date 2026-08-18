"""How a statement says which school it is about.

Two things live here and nothing else: the map from a table to the school that owns it,
and the one function that applies it. Both are in one place on purpose -- the alternative
is sixty hand-written joins, and sixty chances for one of them to join `persons` where it
meant `cameras` and filter perfectly while returning another school's children.

`tests/test_tenancy_guard.py` refuses to let any query in `src/` reach a school's table
without naming `school_id`; this module is what most of them name it through. The guard
proves a school was named. It cannot prove the join was right, and neither can this file
-- `tests/test_tenancy_isolation.py` proves that, by putting two schools in one database
and trying to read across.

WHY A TABLE AND NOT A COLUMN ON EVERY MODEL. Only four tables carry `school_id`: the ones
no other table can answer for. Everything else reaches a school through a foreign key
that cannot be NULL, and the route is written here once. Denormalising the column onto
`events` too would give the schema two answers to one question, and this project has been
bitten twice by a value that was true in one layer and quietly wrong in the next -- once
badly enough to make `person_type` unreliable for every pupil in the school (audit H-02).
"""

from __future__ import annotations

from typing import Any

from qorgan.db.models import (
    Camera,
    CanteenSession,
    Event,
    FaceEmbedding,
    Lesson,
    LessonTrack,
    MealWindow,
    Notification,
    Person,
    PersonPhoto,
    PsychologistNote,
    RecognitionAttempt,
    User,
    sole_school_id,
)


def school_column(root: type) -> Any:
    """The column that names the school, for each of the four tables that carries one.

    Written as literal attributes rather than `getattr(root, "school_id")` so that the
    guard's owner-resolving scan can see them: a filter it cannot attribute is a filter
    it must report as missing, and this is the module every filtered query goes through.
    """
    columns = {
        Camera: Camera.school_id,
        Person: Person.school_id,
        User: User.school_id,
        MealWindow: MealWindow.school_id,
    }
    if root not in columns:
        raise KeyError(f"{root.__name__} is not a root table; it carries no school_id")
    return columns[root]


def route(model: type) -> tuple[tuple[type, Any], ...]:
    """The joins that carry this table to the root that knows its school.

    An empty tuple means the table IS a root. A missing entry is a hard error and not a
    default: a table nobody routed is a table nobody can filter, and guessing a route is
    how a query ends up joined to the wrong parent while looking scoped.
    """
    routes: dict[type, tuple[tuple[type, Any], ...]] = {
        Camera: (),
        Person: (),
        User: (),
        MealWindow: (),
        PersonPhoto: ((Person, PersonPhoto.person_id),),
        FaceEmbedding: ((Person, FaceEmbedding.person_id),),
        Event: ((Camera, Event.camera_id),),
        Notification: ((Event, Notification.event_id), (Camera, Event.camera_id)),
        # Through the ENTRY camera, which cannot be null. `person_id` can be -- the entry
        # camera opens a session for a face it could not identify -- so routing through
        # the person would silently drop exactly the sessions the canteen worries about.
        CanteenSession: ((Camera, CanteenSession.entry_camera_id),),
        RecognitionAttempt: ((Camera, RecognitionAttempt.camera_id),),
        Lesson: ((Camera, Lesson.camera_id),),
        LessonTrack: ((Lesson, LessonTrack.lesson_id), (Camera, Lesson.camera_id)),
        # Through the PERSON the note is about, not the author. `person_id` is NOT NULL
        # and is the child the note concerns; `author_id` is ON DELETE SET NULL and is
        # staff, so routing through it would lose every note whose author has left and
        # would answer a different question anyway -- whose note it is, not whose child.
        PsychologistNote: ((Person, PsychologistNote.person_id),),
    }
    if model not in routes:
        raise KeyError(
            f"{model.__name__} has no route to a school. Add one here (and classify the "
            "table in tests/test_tenancy_guard.py) before querying it: a table with no "
            "route cannot be filtered, and a query nobody can filter is a leak waiting "
            "for the second school."
        )
    return routes[model]


def scope(statement: Any, model: type, school_id: int) -> Any:
    """Constrain a statement to one school, by whatever route that table takes to one."""
    hops = route(model)
    root = hops[-1][0] if hops else model
    for target, key in hops:
        statement = statement.join(target, target.id == key)
    return statement.where(school_column(root) == school_id)


def owned_by(root: type, school_id: int) -> Any:
    """The predicate alone, for a statement that already joins the root it needs.

    `/events` selects the camera's display name, so it joins `cameras` for its own
    reasons; `scope` would join it a second time and SQLAlchemy would refuse. The filter
    is the same filter either way, and it comes from the same table above.
    """
    return school_column(root) == school_id


def resolve_school_id(session: Any, school_id: int | None = None) -> int:
    """Which school this call is about: the one it named, or the only one there is.

    The fallback is not a convenience with a hidden cost -- see `db/models/school.py`.
    With one school it is the truth; with two it raises rather than choosing, which is
    the behaviour that makes every un-plumbed caller a loud failure on the day a second
    school arrives instead of a quiet cross-tenant read.
    """
    if school_id is not None:
        return int(school_id)
    return sole_school_id(session)

"""The four surfaces that read camera CONFIG, and why they close when a second school lands.

`/`, `/cameras`, `/api/cameras`, `/preview/{camera}.jpg` and `/settings` are built from the
camera YAML, through `app.state.cameras` or a fresh read of the files. **That configuration
has no school dimension at all** -- `BullyingCamera`, `CanteenCamera` and `ClassroomCamera`
carry no school field, and `load_cameras()` has no idea one exists. So on an installation
serving two schools, those pages cannot show a viewer only their own school's cameras.
Unscoped they show everybody's: the list, the location, the state, the RTSP host, and the
live JPEG frame of another school's corridor.

**The tenancy guard cannot see any of this, by construction.** It scans database queries
and reports the ones that do not name a school. There is no query here. The exposure is
real and the guard is silent, which is exactly the combination that needs a decision rather
than a filter.

**WHY REFUSE RATHER THAN SCOPE.** Scoping is a schema change, not a `.where()`:
`load_cameras()` has 41 call sites across ten modules -- the CLI, the worker entrypoint,
the evaluation harness, config provenance, the worker planner -- and a per-school camera
key entangles with `workers.yaml`, because a worker group spanning two schools is
incoherent now that `GalleryCache` holds ONE school's roster. That is a module of work, and
until it is done these pages cannot answer the question they are being asked.

**WHY REFUSE RATHER THAN WARN.** This branch is fail-closed everywhere else, and the
inconsistency was the argument: the system is willing to crash a detection worker into a
restart loop rather than guess which school a row belongs to, and was simultaneously
willing to hand over live video of another school's children rather than refuse. Both
positions cannot be right. This is the same refusal `sole_school_id` makes, moved up to the
one layer that has no database row to make it about.

**THE OPERATOR LOSES NOTHING.** On a two-school installation detection is ALREADY dead:
`ensure_cameras` refuses to choose a school, so no camera of any school can be registered
and the workers crash-loop. The camera wall is showing a fleet that is not running. This
turns a page that lies into a page that explains.

**IT UNDOES ITSELF, AND THE TRIGGER IS WIDER THAN IT LOOKS.** `several_schools()` counts
schools; it does not ask whether the configuration can name one. Those are the same
question today and stop being the same question on the likeliest scoping path -- splitting
the YAML per school (`config/schools/<slug>/cameras.yaml`, `load_cameras(school)`) scopes
the config completely while NO camera model ever grows a school field. On that path the
count is still greater than one, so these five pages would go on refusing for no reason: a
camera wall switched off on a system whose job is watching corridors.

So `tests/test_schools_page.py::test_the_camera_config_still_has_no_school...` triggers on
FOUR things, not one:

  1. a school field on any member of the `CameraConfig` union;
  2. a school parameter reaching `load_cameras` or `camera_views`;
  3. a school-IDENTIFYING field on `Settings` -- `school_timezone` is excluded by name,
     being the one school's wall clock rather than a tenancy dimension;
  4. a per-school config tree: `config/schools/` existing, or any subdirectory of `config/`,
     `config/cameras/` or `config/schools/` whose name is a real `School.slug`.

**Do not trust this list without reading that test; it has been wrong once already.** It
said "three things" and named a `config/schools/` check that a later change in the same
branch had replaced with slug matching -- so the paragraph written to make the prose match
the check outlived the check, and the layout named as likeliest three docstrings up was the
one no longer covered.

Whichever way the config layer learns about schools, that test goes red and hands over the
removal list: this module, its five call sites, its tests, and the warning bullet on the
schools page. Rewriting `several_schools()` into "can the config name a school?" was
considered and rejected: it would be an abstraction invented for one call site, and the
test already covers the case the abstraction would exist for.
"""

from __future__ import annotations

from fastapi import Request, status
from sqlalchemy import func, select
from starlette.responses import PlainTextResponse, Response

from qorgan.db.engine import session_scope
from qorgan.db.models import School
from qorgan.web.templating import render

# The number of schools from which the camera configuration stops being able to say whose
# it is. One school is not ambiguous: every camera in the YAML is that school's.
AMBIGUOUS_FROM = 2

# Russian plural agreement for `schools_phrase`. 1 -> школа, 2-4 -> школы, otherwise школ,
# except 11-14, which take the plural whatever their last digit is.
SINGULAR_TAIL = 1
FEW_FIRST, FEW_LAST = 2, 4
TEENS_FIRST, TEENS_LAST = 11, 14

# 409 rather than 403 or 503. It is not "you may not" -- the capability is genuinely held
# -- and it is not "come back later", because nothing retries into a fix. It is a conflict
# between what the page is being asked for and what this installation can currently answer.
REFUSED = status.HTTP_409_CONFLICT

# One sentence for the surfaces that cannot render a page. Kept beside the template so the
# two cannot drift into describing different situations.
PLAIN_REFUSAL = (
    "На этой установке больше одной школы, а список камер, превью и настройки читаются "
    "из конфигурации, у которой нет разделения по школам. Страница не может показать "
    "только вашу школу и не будет показывать чужую. Обратитесь к суперадминистратору "
    "установки: список школ доступен только ему."
)


def several_schools() -> int:
    """How many schools this installation serves.

    `School` is the tenancy register, not a school's data (`INSTALLATION_MODELS` in
    `tests/test_tenancy_guard.py`), so this query is not one the tenancy guard asks to be
    filtered -- a table whose rows ARE the schools cannot be scoped to one.
    """
    with session_scope() as session:
        return int(session.scalar(select(func.count(School.id))) or 0)


def ambiguous(count: int) -> bool:
    """Whether that many schools makes the camera configuration ambiguous.

    **THE ONLY COMPARISON AGAINST `AMBIGUOUS_FROM` IN THIS MODULE, AND THAT IS THE POINT.**
    The knob used to govern `refuse_page` while `config_cannot_name_a_school` carried its
    own hardcoded `> 1`, so three surfaces read it and two did not: setting it to 3 made the
    HTML pages serve while the preview kept refusing. One knob, two behaviours -- a value
    true in one layer and quietly inert in the next.

    That was fixed by giving both functions `>= AMBIGUOUS_FROM`, and the comment claiming
    "the only comparison in this module" was then written over two mirrored comparisons in
    two functions. Behaviourally identical, and the stated invariant was structurally false.
    A pure predicate on the number is what makes the sentence true instead of merely
    well-intentioned.
    """
    return count >= AMBIGUOUS_FROM


def config_cannot_name_a_school() -> bool:
    """True when the camera configuration is ambiguous about who it belongs to.

    Zero or one school is not ambiguous: with one school every camera in the YAML is that
    school's, which is the truth on every installation today.
    """
    return ambiguous(several_schools())


def schools_phrase(count: int) -> str:
    """`2` -> "2 школы", `5` -> "5 школ", `21` -> "21 школа".

    In Python rather than in the template, for the reason every other formatting decision
    in this codebase is: a template that decides something is a template nothing tests.
    "5 школы" is the kind of small wrongness that makes a reader trust the rest of the
    sentence less -- and this sentence is asking them to accept being refused.
    """
    tail = count % 10
    teens = TEENS_FIRST <= count % 100 <= TEENS_LAST
    if tail == SINGULAR_TAIL and not teens:
        return f"{count} школа"
    if FEW_FIRST <= tail <= FEW_LAST and not teens:
        return f"{count} школы"
    return f"{count} школ"


def refuse_page(request: Request) -> Response | None:
    """The rendered refusal for an HTML surface, or `None` when it is safe to serve.

    Returning `None` rather than raising keeps the check visible at the top of each route
    instead of hidden in a decorator -- there are only five, and each one is a place a
    reader should see that this page has a condition on it.
    """
    # Counted ONCE, and judged by the SAME predicate every other surface uses. Deciding
    # here with its own comparison is exactly how this module came to have one knob and two
    # behaviours; `ambiguous` owns the threshold now, and nothing else compares against it.
    count = several_schools()
    if not ambiguous(count):
        return None
    return render(
        request,
        "cameras_are_installation_wide.html",
        schools_phrase=schools_phrase(count),
        status_code=REFUSED,
    )


def refuse_plainly() -> Response | None:
    """The same refusal for `/api/cameras` and the JPEG, which cannot render a page.

    A human who opens either URL directly gets the sentence; the dashboard that would
    normally request the frames is itself refused, so nothing is left showing a broken
    image beside an explanation it never received.
    """
    if not config_cannot_name_a_school():
        return None
    return PlainTextResponse(PLAIN_REFUSAL, status_code=REFUSED)

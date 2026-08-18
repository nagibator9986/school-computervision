"""§12.1's weapon alerts: what was seen, WHICH WEIGHTS SAW IT, and who ruled on it.

Two things this page does that no other page here does, and both are §12.1's.

**It names the model.** «Панель обязана показывать, какие веса загружены и на чём они
проверены.» A module that cannot say what it is running is a module nobody can audit --
and this is not hypothetical tidiness: the client ran a weapons module for months whose
`best.pt` was a 0-byte file, and no screen anywhere would have said so. This one says the
file, its size, its fingerprint and what a human wrote about its evaluation, per camera,
read from the CONFIG at request time. Where a camera's weights cannot be read at all, the
page says that, in red, instead of drawing an empty table -- the state that must be
loudest is the state that was invisible.

Note what it deliberately does NOT do: it does not load the weights. Loading torch inside
a web request would break R3 (the web process knows nothing about the worker) and would
put a several-hundred-megabyte import behind a page load. So it reports the FILE, and says
plainly that only `qorgan weapons weights` can tell you what is inside it. Reporting the
file is enough to catch a 0-byte model, which is the failure that happened.

**Ruling on an alert is the product, not feedback.** On `/events`, marking a bullying
event confirmed is how the school tells a detector it was wrong. Here it is the output:
§12.1 says a weapon alert is never auto-actioned, so what the system produces is a
question and a person's name is the answer. `CONFIRM_WEAPON_ALERT` is therefore a separate
capability from `VIEW_WEAPONS` -- see `roles.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from sqlalchemy import func, select
from starlette.responses import RedirectResponse, Response

from qorgan.config.camera import WeaponsCamera
from qorgan.config.loader import ConfigError, load_cameras, load_workers
from qorgan.db.engine import session_scope
from qorgan.db.models import Camera, Event, User, WorkerHeartbeat
from qorgan.db.tenancy import owned_by
from qorgan.db.types import utcnow
from qorgan.enums import EventStatus, EventType, WorkerState
from qorgan.logging_setup import get_logger
from qorgan.notify.message import local_time
from qorgan.roles import Capability
from qorgan.weapons.feasibility import DEFAULT_OBJECT_CM, assess
from qorgan.weapons.store import WEAPON_VERDICTS, rule_on_weapon_alert
from qorgan.weapons.weights import inspect_weights_file
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()

viewer = Depends(require_capability(Capability.VIEW_WEAPONS))
ruler = Depends(require_capability(Capability.CONFIRM_WEAPON_ALERT))

PAGE_SIZE = 25


@dataclass(frozen=True, slots=True)
class WeightsRow:
    """One weapons camera's model, as the panel names it.

    `problem` is not an error state to hide. It is the single most important thing this
    page can say, because it is what nobody could see for months.
    """

    camera: str
    display_name: str
    path: str
    size: str
    fingerprint: str
    evaluated_on: str
    targets: str
    problem: str
    # Declared confusable classes. Shown because screen 3 of `weapons/rules.py` is the one
    # defence on this page that can be dead and silent: these slugs are a CONVENTION, and if
    # the weights cannot produce any of them the «нож или ручка» check never fires. The web
    # process does not load the model, so it cannot verify that — the row names the command
    # that can.
    confusables: str
    # **Why the weapons module is not running on this camera**, or "" when nothing knowable
    # from here is wrong. A file on disk is not a running module, and three separate states
    # used to render as an ordinary row with a size and a fingerprint in it: weights that are
    # present and do not LOAD (the worker crash-loops while this page looks healthy),
    # `enabled: false`, and a worker that is simply dead. See `_not_running`.
    not_running: str
    # True when the page KNOWS the module is not running. A non-empty `not_running` with
    # `certain=False` is the weaker claim -- "this cannot be confirmed from here" -- and is
    # rendered as a caveat rather than as an alarm. Collapsing the two would put the page
    # back in the business of implying things it has not checked.
    certain: bool


@dataclass(frozen=True, slots=True)
class ReachRow:
    """What ONE camera can physically see, in its own numbers. **Never a fleet average.**

    This exists because of what the client answered on 2026-07-29: a camera will be put
    at the entrance specifically so that the object is large -- **and the other cameras
    stay in play.** So the honest unit is one camera, and there is deliberately no
    summary row, no count of "cameras that work", and no green tick anywhere on this
    page. A module that reports itself by its best camera is the module that said face
    recognition worked, right up until somebody measured the corridor: 14 970 faces,
    median 11.5 px, zero recognised.

    Before this block existed the answer was real but invisible -- `qorgan weapons
    camera-report`, one camera at a time, in a terminal, behind a `--hfov-deg` flag the
    operator had to know. An answer nobody will see is the same as no answer.
    """

    camera: str
    display_name: str
    frame_width: int
    hfov: str
    # True when a human actually wrote the lens into this camera's YAML. False means the
    # schema default is doing the talking, and the row says ASSUMED rather than letting a
    # guessed lens be read as a measurement.
    hfov_stated: bool
    min_pixels: str
    max_distance: str
    # The object's size at a doorway and at the far end of a corridor, which are the two
    # ends of the argument, printed side by side so the fall-off is visible rather than
    # inferred from one number.
    near_px: str
    far_px: str
    near_m: str
    far_m: str
    reaches_far: bool


@dataclass(frozen=True, slots=True)
class AlertRow:
    id: int
    occurred_at: str
    camera: str
    summary: str
    confidence: float
    status: str
    snapshot: str | None
    clip: str | None
    # Who answered the question, and when. Empty while nobody has -- and the template
    # prints that as «ожидает подтверждения», never as anything resembling a finding.
    ruled_by: str
    ruled_at: str


@router.get("/weapons")
def weapons_page(request: Request, page: int = 1, user=viewer) -> Response:
    page = max(1, page)
    rows, total = _page(page, school_of(user))
    try:
        cameras = _weapons_cameras()
        weights, reach, config_error = _weights_rows(cameras), _reach_rows(cameras), ""
    except ConfigError as exc:
        # The one page that can explain a config error must not be the page a config error
        # takes down -- the same rule /settings follows. On site the laptop shows a
        # browser, not a terminal.
        weights, reach, config_error = [], [], str(exc)

    return render(
        request,
        "weapons.html",
        alerts=rows,
        total=total,
        page=page,
        pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
        weights=weights,
        reach=reach,
        object_cm=f"{DEFAULT_OBJECT_CM:g}",
        weights_error=config_error,
    )


@router.post("/weapons/{event_id}/rule")
def rule_on_alert(
    event_id: int, request: Request, verdict: str = Form(...), user=ruler
) -> Response:
    """A person's answer. **The only way a weapon alert ever leaves status NEW.**

    §12.1: «тревогу всегда подтверждает человек, и в записи остаётся, кто подтвердил».
    The user is taken from the session rather than from the form, so the recorded name is
    the one that authenticated and not one that was typed.
    """
    try:
        ruling = EventStatus(verdict)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unknown verdict") from None
    if ruling not in WEAPON_VERDICTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"a weapon alert is answered {[v.value for v in WEAPON_VERDICTS]}",
        )

    if not rule_on_weapon_alert(
        event_id, ruling, user.id, user.username, school_of(user)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such weapon alert")
    return RedirectResponse("/weapons", status_code=status.HTTP_303_SEE_OTHER)


def _weapons_cameras() -> list[tuple[str, WeaponsCamera]]:
    """Every weapons camera, by name. Read once per request and shared by both blocks.

    One `load_cameras()` for the two tables, so a config edited between them cannot make
    the page describe two different fleets in the same screenful.
    """
    return [
        (name, camera)
        for name, camera in sorted(load_cameras().items())
        if isinstance(camera, WeaponsCamera)
    ]


def _weights_rows(cameras: list[tuple[str, WeaponsCamera]]) -> list[WeightsRow]:
    """Every weapons camera's configured model, checked on disk. Never loaded here."""
    beats = _heartbeats()
    groups = _groups_by_camera()
    return [
        _weights_row(name, camera, _not_running(name, camera, groups.get(name), beats))
        for name, camera in cameras
    ]


def _heartbeats() -> dict[str, tuple[str, int, float | None]]:
    """Worker group -> (state, restarts, seconds since the last beat).

    Read from `worker_heartbeats`, which is how the web process learns about workers
    WITHOUT importing one (R3). `last_error` is deliberately not read: it is written from an
    arbitrary exception and can carry an `rtsp://user:password@host`, and redaction on this
    page is not worth the risk when the state and the restart count already say enough.
    """
    now = utcnow()
    with session_scope() as session:
        rows = session.scalars(select(WorkerHeartbeat)).all()
        return {
            row.group_name: (
                row.state.value,
                row.restart_count,
                (now - row.last_seen_at).total_seconds() if row.last_seen_at else None,
            )
            for row in rows
        }


def _groups_by_camera() -> dict[str, tuple[str, float]]:
    """Camera name -> (worker group, that group's heartbeat timeout).

    A separate `load_workers` read, and its failure is not allowed to take the page down:
    a broken `workers.yaml` must not remove the weights table, which is the one thing on
    this page that cannot be got any other way.
    """
    try:
        cameras = load_cameras()
        workers = load_workers(cameras)
    except ConfigError:
        return {}
    return {
        name: (group.name, group.heartbeat_timeout_seconds)
        for group in workers.groups
        for name in group.cameras
    }


def _not_running(
    name: str,
    camera: WeaponsCamera,
    group: tuple[str, float] | None,
    beats: dict[str, tuple[str, int, float | None]],
) -> tuple[str, bool]:
    """Why the weapons module is not running here, and whether that is CERTAIN.

    **The state this page could not describe.** A weights row with a size and a fingerprint
    in it says one thing only: a plausible file is on disk. It does not say the module is
    running, and the case where the difference matters most is the one the file checks
    cannot see — weights that clear the size gate and then fail to LOAD. The worker refuses
    to start, the supervisor restarts it, it refuses again, and this page used to draw a
    perfectly ordinary row throughout.
    """
    del name
    if not camera.enabled:
        return ("камера выключена в конфигурации (enabled: false) — её никто не открывает", True)
    if group is None:
        return ("камера не назначена ни одной группе воркеров в workers.yaml", True)
    group_name, timeout = group
    return _worker_not_running(group_name, timeout, beats.get(group_name))


def _worker_not_running(
    group_name: str, timeout: float, beat: tuple[str, int, float | None] | None
) -> tuple[str, bool]:
    """The heartbeat half of `_not_running`. Split out for ruff's return-count limit.

    The last two branches are the interesting ones and they say DIFFERENT things. A worker
    that is RUNNING and has restarted is the shape of a case-3 crash loop caught between
    cycles; a worker that is RUNNING and never restarted still proves nothing about the
    weights, because this process does not open them.
    """
    if beat is None:
        return (f"воркер группы «{group_name}» ни разу не отчитывался", True)

    state, restarts, age = beat
    if state != WorkerState.RUNNING.value:
        return (f"воркер группы «{group_name}»: состояние {state}, перезапусков {restarts}", True)
    if age is None or age > timeout:
        stale = "никогда" if age is None else f"{age:.0f} с"
        return (
            f"воркер группы «{group_name}» не отчитывался {stale} (порог {timeout:g} с)",
            True,
        )
    if restarts:
        return (
            f"воркер группы «{group_name}» уже перезапускался {restarts} раз(а) — "
            "если веса не загружаются, он будет падать по кругу",
            False,
        )
    return (
        "воркер жив, но загрузились ли эти веса — отсюда не видно: "
        "веб-процесс модель не открывает (правило R3)",
        False,
    )


def _reach_rows(cameras: list[tuple[str, WeaponsCamera]]) -> list[ReachRow]:
    """The optics answer for every weapons camera, one row each. See `ReachRow`."""
    return [_reach_row(name, camera) for name, camera in cameras]


def _reach_row(name: str, camera: WeaponsCamera) -> ReachRow:
    """One camera's own arithmetic, at its own resolution, through its own lens.

    `frame_width` is the width the WORKER analyses, not the camera's advertised
    resolution: the substream is what the model sees, and a report against the headline
    number would be flattering and wrong.
    """
    rules = camera.weapons
    near_m, far_m = 1.5, 15.0
    report = assess(
        camera=name,
        frame_width=camera.capture.frame_width,
        min_object_pixels=rules.min_object_pixels,
        hfov_deg=rules.lens_hfov_degrees,
        distances_m=(near_m, far_m),
    )
    near, far = report.samples
    return ReachRow(
        camera=name,
        display_name=camera.display_name,
        frame_width=report.frame_width,
        hfov=f"{rules.lens_hfov_degrees:g}°",
        # Pydantic knows whether the YAML said it or the schema did. A lens nobody has
        # checked is an assumption, and it is labelled as one rather than printed in the
        # same voice as a measurement.
        hfov_stated="lens_hfov_degrees" in rules.model_fields_set,
        min_pixels=f"{rules.min_object_pixels:g}",
        max_distance=f"{report.max_useful_distance_m:.1f}",
        near_px=f"{near.pixels:.0f}",
        far_px=f"{far.pixels:.0f}",
        near_m=f"{near_m:g}",
        far_m=f"{far_m:g}",
        reaches_far=far.clears_gate,
    )


def _weights_row(
    name: str, camera: WeaponsCamera, run: tuple[str, bool]
) -> WeightsRow:
    settings = camera.weapons.model
    targets = ", ".join(camera.weapons.target_classes)
    confusables = ", ".join(camera.weapons.confusable_classes)
    reason, certain = run
    try:
        artefact = inspect_weights_file(settings.model)
    except Exception as exc:
        # Missing, or empty. Rendered as the problem it is, with the file named -- an
        # unreadable model must be the loudest thing on this page, not a blank cell. And it
        # OVERRIDES whatever the worker looks like: a camera with no usable weights is not
        # running whatever its heartbeat says.
        return WeightsRow(
            camera=name,
            display_name=camera.display_name,
            path=settings.model,
            size="—",
            fingerprint="—",
            evaluated_on=settings.evaluated_on,
            targets=targets,
            confusables=confusables,
            problem=str(exc).splitlines()[0],
            not_running="веса непригодны — конвейер на этой камере не стартует вообще",
            certain=True,
        )
    return WeightsRow(
        camera=name,
        display_name=camera.display_name,
        path=artefact.path,
        size=f"{artefact.size_mb:.1f} МБ",
        fingerprint=artefact.fingerprint,
        evaluated_on=settings.evaluated_on,
        targets=targets,
        confusables=confusables,
        problem="",
        not_running=reason,
        certain=certain,
    )


def _page(page: int, school_id: int) -> tuple[list[AlertRow], int]:
    """This school's weapon events only. `/events` shows the bullying ones.

    Both statements carry both filters. `total` is built beside `query` out of the same
    table, so a filter on one that vouched for the other would leave the header counting
    alerts the list does not show -- another school's children, counted on this school's
    page, with the pager inventing pages that come back empty.
    """
    with session_scope() as session:
        mine = owned_by(Camera, school_id) & (Event.event_type == EventType.WEAPON)
        query = (
            select(Event, Camera.display_name, User.username)
            .join(Camera, Camera.id == Event.camera_id)
            .outerjoin(User, User.id == Event.reviewed_by_id)
            .where(mine)
        )
        total = int(
            session.scalar(
                select(func.count(Event.id))
                .join(Camera, Camera.id == Event.camera_id)
                .where(mine)
            )
            or 0
        )
        rows = session.execute(
            query.order_by(Event.occurred_at.desc())
            .limit(PAGE_SIZE)
            .offset((page - 1) * PAGE_SIZE)
        ).all()
        return [_row(event, camera, who) for event, camera, who in rows], total


def _row(event: Event, camera: str, ruled_by: str | None) -> AlertRow:
    return AlertRow(
        id=event.id,
        occurred_at=local_time(event.occurred_at),
        camera=camera,
        summary=event.summary_text,
        confidence=round(event.confidence, 2),
        status=event.status.value,
        snapshot=event.snapshot_path,
        clip=event.clip_path,
        ruled_by=ruled_by or "",
        ruled_at=local_time(event.reviewed_at) if event.reviewed_at else "",
    )

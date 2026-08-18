"""Live camera previews and worker health.

Every route here is behind the auth middleware, and the preview bytes themselves go
through an authenticated handler -- not StaticFiles. These are pictures of children in
a school corridor.

`/` is the wall: every camera at a glance. `/cameras` is the same fleet examined one row
at a time -- is it reachable, what rate is it really delivering, which worker owns it, what
went wrong -- and it is paginated because each row carries a live frame.

**Nothing here starts, stops, probes or reconfigures anything.** The legacy panel restarted
the AI workers when a tab was opened, with a five-second `thread.join()` inside the HTTP
handler, so which cameras were analysed depended on which tab an operator had open. Rule R3
puts coverage in `workers.yaml` and nowhere else; every route in this file is a reader.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import Response

from qorgan.config.camera import CameraConfig
from qorgan.config.common import fps_agrees
from qorgan.config.workers import WorkerGroup
from qorgan.diagnostics.workers import worker_rows
from qorgan.enums import WorkerState
from qorgan.preview.subscriber import Preview
from qorgan.roles import Capability
from qorgan.settings import get_settings
from qorgan.web.config_scope import refuse_page, refuse_plainly
from qorgan.web.security import require_capability
from qorgan.web.templating import render

router = APIRouter()

# Live video of a school corridor. Not every role that works here gets to watch children.
#
# VIEW_CAMERAS, and specifically NOT a new capability of its own: `/cameras` renders the
# same `/preview/{camera}.jpg` frames as the wall does, so a grant that opened one and not
# the other would be a difference the pictures do not have. §14 -- canteen staff work "БЕЗ
# доступа к буллингу" -- is about who may watch children in a corridor, not about which
# layout they watch them in. (The precedent for splitting a capability is `/media`, and it
# split because the TREE was mixed: incident evidence about a few named children versus a
# photographic register of every pupil. Those answer different questions for different
# people. A wall and a table of the same live frames do not.)
viewer = Depends(require_capability(Capability.VIEW_CAMERAS))

# Rows per page. Bounded by bandwidth, not taste: every row carries a live JPEG that the
# browser re-fetches once a second, so an unpaginated list costs (cameras x frame size) per
# second, forever, and gets worse every time the school adds a camera. The legacy re-sent
# the entire world every 2.5 s per client (audit M-19); this is that defect's cousin.
# The shipped fleet is ten cameras, so this page is genuinely paginated on day one rather
# than in theory.
PAGE_SIZE = 6


def _camera_rows(request: Request) -> list[dict[str, object]]:
    previews = request.app.state.previews
    now = time.time()
    rows = []
    for name, camera in sorted(
        request.app.state.cameras.items(), key=lambda item: item[1].priority
    ):
        preview = previews.latest(name, now=now)
        rows.append(
            {
                "name": name,
                "display_name": camera.display_name,
                "location": camera.location,
                "role": camera.role.value,
                "enabled": camera.enabled,
                "live": preview is not None,
                "status": preview.header.status if preview else "offline",
                "age_seconds": round(preview.age_seconds(now), 1) if preview else None,
            }
        )
    return rows


@router.get("/")
def dashboard(request: Request, _user=viewer) -> Response:
    # Refuses on a multi-school installation. `web/config_scope.py` argues why: this wall
    # is built from configuration that cannot name a school, so it can only show every
    # school's corridors or none.
    refused = refuse_page(request)
    if refused is not None:
        return refused
    return render(request, "dashboard.html", cameras=_camera_rows(request), workers=worker_rows())


# `response_model=None`: the handler can return either the camera payload or a refusal
# Response, and FastAPI would otherwise try to build a response model out of that union
# and refuse to start. The JSON shape on the success path is unchanged.
@router.get("/api/cameras", response_model=None)
def cameras_json(request: Request, _user=viewer) -> Response | dict[str, object]:
    """The same two rows the dashboard renders, as JSON. Deliberately not a second shape.

    **This endpoint is neither wider nor narrower than the page, and that is the decision.**
    It was audited on 2026-07-26 on the suspicion that it returned less than `/`; it does
    not. Both surfaces call `_camera_rows` and `diagnostics.workers.worker_rows` and
    nothing else, so there is no second projection of a camera anywhere in this process
    that could disagree with the
    first — which is the failure this project keeps meeting, a value true in one layer and
    quietly wrong in the next. Two tests hold the line
    (`tests/test_web_preview.py::test_the_json_api_carries_everything_the_dashboard_page_shows`).

    **It is nonetheless wider than its consumer, and that stays too.** `static/preview.js`
    reads exactly `name`, `status` and `live` off each camera and ignores `workers`
    entirely. The surplus is not dead weight to be trimmed: narrowing it to the three keys
    would make the page and the API two different shapes again, to save a few hundred bytes
    on a loopback request. Widening it is the thing to be careful about — every field here
    describes a school corridor and reaches any session holding VIEW_CAMERAS, so add a
    field when the PAGE needs it, never because an API "might as well" carry it.

    What is deliberately absent: any RTSP host, port, URL or credential. The legacy served
    `rtsp://admin:<password>@host` out of a table like this to anyone who asked (audit
    C-01, C-03). `_camera_rows` never reads those fields; do not add them here.
    """
    refused = refuse_plainly()
    if refused is not None:
        return refused
    return {"cameras": _camera_rows(request), "workers": worker_rows()}


@router.get("/preview/{camera}.jpg")
def preview_image(camera: str, request: Request, _user=viewer) -> Response:
    """The latest preview frame, as JPEG. Authenticated, and never cached.

    Refused before the camera name is even looked up on a multi-school installation: the
    name comes from a config file that cannot say whose camera it is, so resolving it at
    all would be answering the wrong question. This is the live-video surface, so it is the
    one where refusing rather than guessing matters most.
    """
    refused = refuse_plainly()
    if refused is not None:
        return refused
    if camera not in request.app.state.cameras:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown camera")

    preview = request.app.state.previews.latest(camera)
    if preview is None:
        # Honest 503 rather than a stale frame. A five-minute-old picture of an empty
        # corridor tells the operator everything is calm when nobody is watching at all.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no live preview"
        )

    return Response(
        content=preview.jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, private",
            "X-Preview-Seq": str(preview.header.seq),
        },
    )


def _heartbeats() -> dict[str, dict[str, object]]:
    """Every worker group's last heartbeat, keyed by group name.

    A shape adapter over ONE derivation, not a second one. `diagnostics.workers.worker_rows`
    is the single place a heartbeat row is built, and it REDACTS `last_error` on the way out
    -- that column holds a crashed worker's exception text, and a worker's exceptions quote
    the RTSP URL, which carries the camera password.

    This file used to build those rows itself. Keeping that copy through the merge would
    have restored the password to the dashboard while every test stayed green: the
    redaction tests would still have passed against `diagnostics.workers` while this page
    rendered the unredacted copy beside them. Two implementations of one row is exactly the
    defect this codebase keeps meeting.

    Keyed by group because `_health_rows` asks `heartbeats.get(group.name)`; the list that
    `worker_rows()` returns is the same data in the order the worker table wants it.
    """
    return {row["group"]: row for row in worker_rows()}


@dataclass(frozen=True, slots=True)
class CameraHealthRow:
    """One camera, and how much of what we say about it we actually know.

    The `_fps` fields are three different quantities and are kept apart on purpose.
    `configured_fps` is `capture.stream_fps` -- read off a SCREENSHOT of one camera's web
    interface, a claim about the fleet. `measured_fps` is what the camera loop counted the
    stream really delivering, or None while nobody has counted. `analysis_fps` is what the
    detector sees, derived from the CONFIGURED rate through the single derivation in
    `CaptureSettings`, and so wrong by exactly as much as the configuration is.

    Collapsing them into one "fps" is the `display_fps` defect (tests/test_analysis_rate.py):
    a number that was true as "what we configured" and false as "what the camera delivers",
    with nothing saying which one you were reading.
    """

    name: str
    display_name: str
    location: str
    role: str
    enabled: bool
    live: bool
    # Which QUESTION the absence of frames answers -- see `_signal`.
    signal: str
    status: str
    age_seconds: float | None
    configured_fps: int
    det_every: int
    analysis_fps: float
    measured_fps: float | None
    fps_disagrees: bool
    group: str | None
    device: str | None
    group_state: str | None
    group_error: str | None


def _signal(camera: CameraConfig, live: bool, heartbeat: dict[str, object] | None) -> str:
    """What "no preview arrived" is actually evidence of.

    ONE observation, several causes, and only one of them is the camera's fault. Reporting
    them all as "нет сигнала" sends an electrician up a ladder to a working camera while
    the supervisor sits crashed -- the observation is true at the layer that made it ("no
    frames reached us") and false one layer up ("this camera is broken").

      disabled    -- config says nobody watches it, so silence is the configuration working
      live        -- a frame arrived inside the staleness window
      worker_down -- its group is not running, so NOTHING is known about the camera itself
      no_frames   -- its group is running and still sends nothing: now the camera is the
                     suspect, and that is worth somebody walking to it
    """
    if not camera.enabled:
        return "disabled"
    if live:
        return "live"
    # A group that never reported is as unknown as one that reported CRASHED: in both cases
    # nobody is producing frames to be missing, so a missing heartbeat is not treated as
    # "running" by default. A permission-shaped decision, made the same way as everywhere
    # else here -- the failure of a lookup must not become a claim about a child's corridor.
    if heartbeat is None or heartbeat["state"] != WorkerState.RUNNING.value:
        return "worker_down"
    return "no_frames"


def _health_row(
    name: str,
    camera: CameraConfig,
    preview: Preview | None,
    now: float,
    group: WorkerGroup | None,
    heartbeat: dict[str, object] | None,
) -> CameraHealthRow:
    configured = camera.capture.stream_fps
    # Only from the frame we actually received. Never defaulted to the configured value:
    # "we have not measured" and "it delivers exactly what we configured" are different
    # answers, and only one of them is worth showing an operator.
    measured = preview.header.measured_fps if preview else None
    return CameraHealthRow(
        name=name,
        display_name=camera.display_name,
        location=camera.location,
        role=camera.role.value,
        enabled=camera.enabled,
        live=preview is not None,
        signal=_signal(camera, preview is not None, heartbeat),
        status=preview.header.status if preview else "offline",
        age_seconds=round(preview.age_seconds(now), 1) if preview else None,
        configured_fps=configured,
        det_every=camera.capture.det_every,
        analysis_fps=camera.capture.analysis_fps,
        measured_fps=round(measured, 1) if measured is not None else None,
        # The SAME tolerance the worker logs its warning by (`config.common.FPS_TOLERANCE`).
        # A badge here and a log line there disagreeing about one camera would be one more
        # value true in one layer and wrong in the next.
        fps_disagrees=measured is not None and not fps_agrees(float(configured), measured),
        group=group.name if group else None,
        device=group.device if group else None,
        group_state=str(heartbeat["state"]) if heartbeat else None,
        # The GROUP's error, and labelled as the group's in the template. `bullying_hall`
        # runs two cameras and reports one error for the process: printing it as this
        # camera's fault invents a fact, because the failure may be the other camera's.
        group_error=str(heartbeat["last_error"]) if heartbeat and heartbeat["last_error"] else None,
    )


def _health_rows(request: Request) -> list[CameraHealthRow]:
    """Every configured camera, in operator priority order.

    From the CONFIG, not from what happened to publish: a camera that has gone dark must
    still occupy a row saying so. A list built from live previews would quietly shorten
    itself exactly when something was wrong.
    """
    previews = request.app.state.previews
    workers = getattr(request.app.state, "workers", None)
    heartbeats = _heartbeats()
    now = time.time()

    rows = []
    for name, camera in sorted(
        request.app.state.cameras.items(), key=lambda item: item[1].priority
    ):
        group = workers.group_for(name) if workers else None
        heartbeat = heartbeats.get(group.name) if group else None
        preview = previews.latest(name, now=now)
        rows.append(_health_row(name, camera, preview, now, group, heartbeat))
    return rows


@router.get("/cameras")
def cameras_page(request: Request, page: int = 1, _user=viewer) -> Response:
    """The fleet, one row per camera. A GET renders and does nothing else (R3).

    Refused alongside the wall on a multi-school installation -- the same frames in a
    different layout, out of the same configuration that cannot name a school.
    """
    refused = refuse_page(request)
    if refused is not None:
        return refused
    rows = _health_rows(request)
    pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
    # Clamped at BOTH ends. `?page=99` is a bookmark somebody kept after the fleet shrank,
    # and an empty camera list on a safety dashboard reads as "nothing is being watched" --
    # a false alarm the school acts on.
    page = min(max(1, page), pages)
    start = (page - 1) * PAGE_SIZE

    return render(
        request,
        "cameras.html",
        cameras=rows[start : start + PAGE_SIZE],
        page=page,
        pages=pages,
        total=len(rows),
        # Everything above is one instant, and it stays that instant: nothing on this page
        # refreshes itself. `/` is the live wall; a page where the status badge updated
        # every second while the frame rate beside it was five minutes old would be two
        # truths in one row. An undated snapshot is read as "now", so it is dated.
        rendered_at=datetime.now(tz=get_settings().tz).strftime("%H:%M:%S"),
        # Not fatal to the page: the one thing worth saying loudest -- an enabled camera
        # assigned to no worker group -- is only ever seen on a page that opened.
        workers_error=getattr(request.app.state, "workers_error", None),
    )

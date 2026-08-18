"""The imported classroom analyses, as pages inside the platform. Five routes, one capability.

**One capability for all five**, `VIEW_CLASSROOM_ANALYSIS`, held by the PSYCHOLOGIST and by
ADMIN (`qorgan/roles.py`). Not the cabinet's own grant, and the difference is deliberate: the
psychologist's notes are a confidential record about a named child, while these pages are
observations of a PLACE that carry no name at all until somebody signs a seating plan. Two
different disclosures need two different grants, or the day one is widened the other moves with
it silently.

**Why the images are served here and not by `/media`.** `web/routes/media.py` classifies the
media tree by its top-level directory and refuses anything it has not been told about — and
`classvision/` is not in its map, so deny-by-default already answered. That is the right
answer: the route that knows what these pictures are is the route that must decide who may see
them. `frame_image` therefore looks the row up through its lesson and this school, and confines
the resolved path to MEDIA_ROOT, because a frame id is a number in a URL.

**No route here writes anything.** Not the reading, not the frames, not an attestation. The
reading is generated beside the recording where the model key lives, the stills are cut by
ffmpeg in a command, and a name may only arrive through a signed plan — so there is no button
on any of these pages, and there is nothing for CSRF to protect.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import FileResponse, Response

from qorgan.classvision.cabinet import lesson_view
from qorgan.classvision.classes_view import class_view, classes_index
from qorgan.classvision.lessons_index import lessons_index
from qorgan.classvision.frames_view import frame_image_path, frames_view
from qorgan.classvision.place_view import place_view
from qorgan.db.engine import session_scope
from qorgan.logging_setup import get_logger
from qorgan.paths import PathOutsideRoot, ensure_within
from qorgan.roles import Capability
from qorgan.settings import get_settings, resolve
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()

viewer = Depends(require_capability(Capability.VIEW_CLASSROOM_ANALYSIS))

# Only what the frame writer itself produces. Serving arbitrary types out of a media directory
# is how a stray .html file becomes stored XSS on your own origin -- the same rule, and the same
# reason, as `web/routes/media.py::ALLOWED_SUFFIXES`.
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@router.get("/psychologist/classes")
def classes_page(request: Request, user=viewer) -> Response:
    """Where the cabinet opens: the CLASSES, not a list of recordings and not a child.

    A psychologist thinks «8-А, у которого подписан план» before they think «место 7», and
    the old entry point — a flat list of lessons — made the first step of every walk a guess
    about which chair somebody sat in.
    """
    with session_scope() as session:
        rows = classes_index(session, school_id=school_of(user))
        return render(request, "cv_classes.html", rows=rows)


@router.get("/psychologist/classes/{class_key}")
def class_page(class_key: str, request: Request, user=viewer) -> Response:
    """One class: its rooms, the pupils in each, and what each room's seating plan says.

    404 rather than an empty page when the class has no imported lesson: a page that renders
    a class with nothing in it cannot be told from a class whose recordings failed to import.
    """
    with session_scope() as session:
        view = class_view(session, school_id=school_of(user), class_key=class_key)
        if view is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such class")
        return render(request, "cv_class.html", **view)


@router.get("/psychologist/lessons")
def lessons_page(request: Request, user=viewer) -> Response:
    """Every imported lesson: the measured ones and the demonstration, in two separate lists."""
    with session_scope() as session:
        index = lessons_index(session, school_id=school_of(user))
        return render(request, "cv_lessons.html", index=index)


@router.get("/psychologist/lessons/{lesson_id}")
def lesson_page(request: Request, lesson_id: int, user=viewer) -> Response:
    """One lesson: the reading, the per-place table, the adult, and what was not measured."""
    with session_scope() as session:
        view = lesson_view(session, school_id=school_of(user), lesson_id=lesson_id)
        if view is None:
            # 404 rather than an empty lesson: a page of zeros under an id this school does not
            # hold reads as «на этом уроке ничего не происходило», which is a statement about a
            # lesson that does not exist here.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such lesson")
        return render(request, "cv_lesson.html", **view)


@router.get("/psychologist/lessons/{lesson_id}/frames")
def frames_page(request: Request, lesson_id: int, user=viewer) -> Response:
    """The classification as it looked on the recording: stills, boxes, states, metrics."""
    with session_scope() as session:
        view = frames_view(session, school_id=school_of(user), lesson_id=lesson_id)
        if view is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such lesson")
        return render(request, "cv_frames.html", **view)


@router.get("/psychologist/lessons/{lesson_id}/frames/{frame_id}/image")
def frame_image(lesson_id: int, frame_id: int, request: Request, user=viewer) -> Response:
    """One still. Confined to MEDIA_ROOT whatever the database says the path is."""
    root = resolve(get_settings().media_root)
    with session_scope() as session:
        relative = frame_image_path(
            session, school_id=school_of(user), lesson_id=lesson_id, frame_id=frame_id)
    if relative is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such frame")

    target = (root / relative).resolve()
    try:
        # `RelPath` refuses an absolute path at bind time, so a stored path cannot escape --
        # but this is the check that is true whatever anybody stored later, and it costs one
        # call. Not a 404: a stored path pointing out of the tree is a fact somebody must see.
        ensure_within(root, target)
    except PathOutsideRoot:
        logger.warning("a stored frame path pointed outside MEDIA_ROOT",
                       extra={"frame": frame_id, "path": relative})
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden") from None
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="unsupported media type")
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return FileResponse(
        target,
        media_type=_media_type(target),
        headers={
            # A classroom still is a photograph of eight children. It does not belong in a
            # shared cache and it does not belong in the browser's disk cache on a staffroom
            # computer -- the same headers, for the same reason, as `/media`.
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/psychologist/places/{place_id}")
def place_page(request: Request, place_id: int, user=viewer) -> Response:
    """One place across its lessons: the weekly table, the segments, and the trend's refusal."""
    with session_scope() as session:
        view = place_view(session, school_id=school_of(user), place_id=place_id)
        if view is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such place")
        return render(request, "cv_place.html", **view)


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"

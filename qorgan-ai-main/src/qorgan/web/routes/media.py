"""Serving media. Through an authenticated handler, never StaticFiles.

Everything under MEDIA_ROOT is a photograph of a child or video of an incident involving
one. `app.mount("/media", StaticFiles(...))` would hand all of it to anyone who guessed a
URL — and the legacy did precisely that, on a server bound to 0.0.0.0 with no auth
anywhere (audit C-01).

Two things this handler does that StaticFiles cannot:

  * It requires a session, because the auth middleware sees it.
  * It refuses to serve anything outside MEDIA_ROOT, whatever the URL says. A path is a
    string an attacker controls, and `..` is a valid string.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from starlette.responses import FileResponse, Response

from qorgan.events.recorder import media_root
from qorgan.logging_setup import get_logger
from qorgan.paths import PathOutsideRoot, ensure_within
from qorgan.roles import Capability, capabilities_for
from qorgan.web.security import current_user

logger = get_logger(__name__)

router = APIRouter()

# WHAT EACH TOP-LEVEL DIRECTORY IS, and therefore who may read it. The tree is mixed, and
# it used to be one grant (`VIEW_MEDIA`) with that fact recorded in a comment instead of a
# check: whoever held it held the incident evidence AND a photographic register of every
# pupil in the school.
#
# Those answer different questions. The psychologist §13 describes is asked about one
# incident and needs its clip; nothing in that work touches the enrolment gallery. A model
# that cannot say "clips but not the gallery" fails at the moment someone is granted
# access, which is the worst moment to discover it.
MEDIA_CAPABILITIES: dict[str, Capability] = {
    "snapshots": Capability.VIEW_BULLYING_MEDIA,  # events/recorder.write_snapshot
    "clips": Capability.VIEW_BULLYING_MEDIA,  # events/recorder.write_clip
    "people": Capability.VIEW_PUPIL_PHOTOS,  # faces/importer, the enrolment gallery
}

# Only what we ourselves write. Serving arbitrary types out of a media directory is how a
# stray .html file becomes stored XSS on your own origin.
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}


@router.get("/media/{path:path}")
def serve_media(path: str, request: Request) -> Response:
    root = media_root()
    target = (root / path).resolve()

    try:
        ensure_within(root, target)
    except PathOutsideRoot:
        # Not a 404. Somebody typed `..` on purpose, and we should know about it.
        logger.warning(
            "blocked a path traversal attempt on /media",
            extra={
                "path": path,
                "user": getattr(request.state, "user", None) and request.state.user.username,
            },
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden") from None

    _require_the_capability_for(target, root, request)

    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unsupported media type")

    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")

    return FileResponse(
        target,
        media_type=_media_type(target),
        headers={
            # These are children. They do not belong in a shared cache, and they do not
            # belong in the browser's disk cache on a shared staffroom computer.
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _require_the_capability_for(target: Path, root: Path, request: Request) -> None:
    """Which half of the tree is this, and may this person read that half?

    **Decided from the RESOLVED path, never from the URL.** `..` is a valid string, so
    `people/../clips/x.mp4` starts with `people/` and is a clip. `ensure_within` does not
    catch it -- the path never leaves MEDIA_ROOT -- so a prefix test on what the caller
    typed would hand the incident evidence to a gallery-only grant. `target` has already
    been resolved and confined above; its first segment under the root is the truth.

    A directory the map does not name is refused to everyone, including an operator. New
    things appear under MEDIA_ROOT (`.import/` is where uploaded archives are unzipped --
    children's photographs, mid-import, which the single whole-tree capability served
    happily). Deny-by-default is the same rule `AuthMiddleware` applies to new routes and
    `capabilities_for` applies to new roles: it must be classified in writing to be read.
    """
    area = target.relative_to(root).parts[0] if target != root else ""
    needed = MEDIA_CAPABILITIES.get(area)

    if needed is None:
        logger.warning(
            "refused an unclassified area of the media tree",
            extra={"area": area, "user": current_user(request).username},
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    if needed not in capabilities_for(current_user(request).role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"requires: {needed.value}"
        )


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"

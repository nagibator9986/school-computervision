"""§13's cabinet: four routes, three capabilities, all added in this same change.

**The confidentiality boundary is here, in `require_capability`, and nowhere else.** §13:
«обычный оператор не должен видеть конфиденциальные записи психолога». The notes are a
SEPARATE PAGE rather than a section of the pupil page, and that is the whole design: a
section would be loaded by a route an operator or an administrator can open and then hidden
by an `{% if %}`, which means the bodies were already fetched and one careless template
edit publishes them. Here nothing that touches `psychologist_notes` is reachable without
`VIEW_PSYCHOLOGIST_NOTES`, and `tests/test_web_auth.py` walks the real route table.

**The pupil trend asks for a SUPERSET of what `/pupils/{id}/canteen` asks for.** It shows
the same child's meal record in a different shape, so it must not become a second door into
it: anybody who can open the trend can already open the sessions. That is why the
psychologist ROLE was widened (`roles.py`) rather than this PAGE narrowed.

Nothing here writes anything except a note and, on the events page, a referral. No page in
this package computes a recommendation; see `qorgan.psychologist`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.psychologist.attendance import attendance_trend
from qorgan.psychologist.cabinet import cabinet_view
from qorgan.psychologist.notes import NoteRejected, add_note, notes_for
from qorgan.roles import Capability
from qorgan.web.security import require_capability, school_of
from qorgan.web.templating import render

router = APIRouter()

cabinet_viewer = Depends(require_capability(Capability.VIEW_PSYCHOLOGIST_CABINET))

# All three, never any of them: one child's meal record read through the cabinet is the
# same disclosure as reading it through /pupils, so it takes the same grants plus the one
# that opens the cabinet at all.
trend_viewer = Depends(
    require_capability(
        Capability.VIEW_PSYCHOLOGIST_CABINET,
        Capability.VIEW_PUPILS,
        Capability.VIEW_CANTEEN,
    )
)

# VIEW_PUPILS as well as the notes grant: the page names the child it is about, and a
# confidential note under «Ученик 214» helps nobody.
notes_viewer = Depends(
    require_capability(Capability.VIEW_PSYCHOLOGIST_NOTES, Capability.VIEW_PUPILS)
)
# BOTH, and the reading half is not decoration: a refused note re-renders the notes page so
# the author reads the rule beside their own text, and that page carries every body. A
# write-only grant would therefore have been a door into the confidential history through a
# deliberately empty form. The split still says the thing it was made to say -- read
# without write is expressible, which is the direction a supervising role would need.
notes_author = Depends(
    require_capability(Capability.WRITE_PSYCHOLOGIST_NOTES, Capability.VIEW_PSYCHOLOGIST_NOTES)
)


@router.get("/psychologist")
def cabinet_page(request: Request, user=cabinet_viewer) -> Response:
    """What was handed over, and what is and is not accumulating behind it."""
    return render(request, "psychologist.html", cabinet=cabinet_view(school_id=school_of(user)))


@router.get("/psychologist/pupils/{person_id}")
def pupil_trend_page(person_id: int, request: Request, user=trend_viewer) -> Response:
    """One child's canteen attendance, week by week. Counts, never a conclusion."""
    trend = attendance_trend(person_id, school_id=school_of(user))
    if trend is None:
        # A 404, never an empty trend: eight empty weeks under an id nobody holds reads as
        # "this child stopped coming", which is a claim about a child who does not exist.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such person")
    return render(request, "psychologist_pupil.html", trend=trend)


@router.get("/psychologist/notes/{person_id}")
def notes_page(person_id: int, request: Request, user=notes_viewer) -> Response:
    """Confidential. The bodies are loaded HERE and on no other route."""
    history = notes_for(person_id, school_id=school_of(user))
    if history is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such person")
    return render(request, "psychologist_notes.html", history=history, error="")


@router.post("/psychologist/notes/{person_id}")
def add_note_route(
    person_id: int, request: Request, body: str = Form(...), user=notes_author
) -> Response:
    """One more note. Never an edit and never a delete -- see `psychologist.notes`."""
    school = school_of(user)
    try:
        add_note(person_id, author_id=user.id, body=body, school_id=school)
    except NoteRejected as refused:
        # Re-rendered rather than redirected, so the refusal names the rule on the page the
        # author is looking at. The rejected text is NOT quoted back: it is confidential.
        history = notes_for(person_id, school_id=school)
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="no such person"
            ) from None
        return render(
            request,
            "psychologist_notes.html",
            status_code=status.HTTP_400_BAD_REQUEST,
            history=history,
            error=str(refused),
        )

    return RedirectResponse(
        f"/psychologist/notes/{person_id}", status_code=status.HTTP_303_SEE_OTHER
    )

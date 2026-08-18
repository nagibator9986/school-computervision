"""Log in, log out."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request, status
from starlette.responses import RedirectResponse, Response

from qorgan.logging_setup import get_logger
from qorgan.web.security import SESSION_USER_KEY, authenticate, landing_for, load_user
from qorgan.web.templating import render

logger = get_logger(__name__)

router = APIRouter()


@router.get("/login")
def login_form(request: Request) -> Response:
    user_id = request.session.get(SESSION_USER_KEY)
    user = load_user(user_id) if user_id else None
    landing = landing_for(user) if user is not None else "/login"
    # Never bounce /login to /login. A role granted no capability has nowhere to land,
    # and a redirect loop reaches the school as ERR_TOO_MANY_REDIRECTS -- read as "the
    # system is down", not as the permissions mistake it is.
    if landing != "/login":
        return RedirectResponse(landing, status_code=status.HTTP_303_SEE_OTHER)
    return render(request, "login.html")


@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    user = authenticate(username, password)
    if user is None:
        # One message for "no such user" and for "wrong password": telling them apart
        # is a free list of who works here.
        logger.warning("failed login", extra={"username": username})
        return render(
            request,
            "login.html",
            error="Неверное имя пользователя или пароль",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.session.clear()  # no session fixation
    request.session[SESSION_USER_KEY] = user.id
    logger.info("login", extra={"username": user.username, "role": user.role.value})
    # Not always "/": that is the camera wall, and a canteen worker has no business there.
    return RedirectResponse(landing_for(user), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/logout")
def logout(request: Request) -> Response:
    request.session.clear()
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

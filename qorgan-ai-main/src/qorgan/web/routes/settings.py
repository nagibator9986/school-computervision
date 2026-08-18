"""The configuration, shown in full and editable nowhere.

Three legacy defects meet on this page, and each one shapes a decision here.

**It leaked the bot token.** `/settings?format=json` served the whole settings table --
plaintext Telegram token included -- to any anonymous caller (audit H-04). The token is
environment-only (R4) and is not rendered here in any form, not even masked: a mask is
still a statement about a secret, and there is nothing on this page that needs one.
Telegram appears as on or off.

**It had side effects on load.** `POST /page-activate/{page}` restarted the AI workers,
with a five-second `thread.join()` inside the HTTP handler, whenever somebody opened a tab
(audit H-18) -- so coverage depended on which browser tab was open. This route reads files
and reads settings. It starts nothing, stops nothing, writes nothing.

**It was a second source of truth.** A browser form that edits YAML makes the file and the
form disagree, and the supervisor goes on running whatever it loaded at startup. That is
this project's signature disease -- `min_score: 0.50`, documented in `config/identity.py`,
overridden by every profile, in force on zero cameras. So every value here is READ-ONLY and
travels with the file that supplies it (`qorgan.config.provenance`).

**Capability, not rank.** `VIEW_SETTINGS`, held by ADMIN and DEVELOPER only. Reading the
event log and reading the installation are different jobs: this page carries every camera's
RTSP host, every detector threshold and every ROI rectangle in the school, and §14's roles
overlap rather than nest, so an operator does not get it by being senior to anybody.

**There is deliberately no capability for CHANGING anything**, because there is nothing to
guard -- see `_mode_control` for what is missing and why. `roles.py`: "a permission
guarding nothing is a guess, and it would be enforced by nobody." When a control channel to
the supervisor exists, the right to use it must be a NEW capability and never this one: the
person who may read a threshold is not automatically the person who may switch a school's
cameras off.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from qorgan.config.loader import ConfigError
from qorgan.config.provenance import CameraView, camera_views, config_label
from qorgan.enums import SystemMode
from qorgan.roles import Capability
from qorgan.settings import get_settings
from qorgan.web.config_scope import refuse_page
from qorgan.web.security import require_capability
from qorgan.web.templating import render

router = APIRouter()

# Every camera's RTSP host, every threshold, every ROI. Not operator work.
viewer = Depends(require_capability(Capability.VIEW_SETTINGS))


def _workers_file() -> str:
    """`config/workers.yaml`, unless this installation keeps its config elsewhere.

    Built from `config_dir` rather than written out, because the whole point of the page is
    to name the file somebody must open. A hardcoded path that is right on the developer's
    machine and wrong on the school's would be this codebase's own disease, pointed at the
    reader instead of the detector.
    """
    return f"{config_label()}/workers.yaml"


def _camera_file() -> str:
    return f"{config_label()}/cameras/<камера>.yaml"


@dataclass(frozen=True, slots=True)
class Control:
    """Something the school might expect to change from here, and whether they can.

    `available` is false on both today. It is a field rather than an omission so the
    template renders the honest answer instead of silently drawing nothing -- a school
    that cannot see the control assumes it is elsewhere and goes looking.
    """

    title: str
    state: str
    available: bool
    reason: str


@dataclass(frozen=True, slots=True)
class Integration:
    title: str
    state: str
    note: str


@router.get("/settings")
def settings_page(request: Request, _user=viewer) -> Response:
    """Read the fleet config off disk and show it. A GET changes nothing at all.

    Refused on a multi-school installation: this page lists every camera's RTSP host and
    every threshold on the machine, out of files that cannot say which school they belong
    to. See `web/config_scope.py`.

    Read per request rather than off `app.state.cameras`, which the web process loaded once
    at startup: the files are what an engineer edits on site, and a page reporting a
    boot-time snapshot as the current configuration would be one more layer quietly
    disagreeing with the one below it.

    That choice has a cost, and the standing caveat in the template pays it out loud: this
    page is then FRESHER than the supervisor, the workers and the camera wall, all of which
    are still running what they read when they started. Naming all four is the point. A
    caveat that mentioned only the supervisor would leave an operator to discover on their
    own that `/` and `/settings` disagree about whether a camera is enabled.
    """
    refused = refuse_page(request)
    if refused is not None:
        return refused

    try:
        cameras = camera_views()
        config_error = ""
    except ConfigError as exc:
        # The one page that can explain a config error must not be the page a config error
        # takes down. On site the laptop is showing a browser, not a terminal.
        cameras, config_error = [], str(exc)

    return render(
        request,
        "settings.html",
        cameras=cameras,
        config_error=config_error,
        controls=(_mode_control(cameras), _camera_control(cameras)),
        integrations=(_telegram(),),
    )


def _mode_control(cameras: list[CameraView]) -> Control:
    """The operating mode: derived, never stored, and not switchable from a browser.

    Checked before writing this: `SystemMode` and the `mode_logs` table exist in the
    schema, and NOTHING in `src/` writes or reads either. There is also no control channel
    of any kind from the web process to the supervisor -- `supervisor/` has `run`, `tick`
    and `stop`, called in-process by `qorgan supervisor`, and the only thing crossing that
    boundary is the heartbeat row the supervisor writes and the dashboard reads.

    So the mode is reported as what it actually is: a description of the config, computed
    from which cameras are enabled. Offering a switch would mean inventing a second
    mechanism and then hoping the fleet noticed -- and an inert control that looks live is
    worse than an honest absence, because the school believes it.

    R3 makes this a feature rather than a gap: coverage is decided by config and the
    supervisor, never by which browser tab is open. In the legacy, nobody looking at the
    "Stairs" tab meant the stairs were not monitored at all.
    """
    return Control(
        title="Режим работы",
        state=(
            f"{_derived_mode(cameras).value} — вычислено по включённым камерам, "
            "нигде не хранится"
        ),
        available=False,
        reason=(
            "Переключение из веб-интерфейса недоступно. Покрытие задают "
            f"{_workers_file()} и ключ enabled: в {_camera_file()}; супервизор читает их при "
            "запуске. Канала управления от веб-процесса к супервизору не существует "
            "(src/qorgan/supervisor/), а кнопка, которая ничего не переключает, хуже, "
            "чем её отсутствие."
        ),
    )


def _camera_control(cameras: list[CameraView]) -> Control:
    """Enabling and disabling a camera. Same answer, and the same reason.

    `enabled:` is a YAML key: `config/loader.py::load_workers` refuses a config where an
    enabled camera belongs to no worker group ("a camera nobody runs is a camera nobody is
    watching"), and the supervisor spawns worker groups from that file at startup. Writing
    the flag from a browser would put the same fact in two places, and the `cameras` table
    in the database has an `enabled` column that nothing reads -- so a plausible-looking
    write path exists, and taking it would produce a switch that changes nothing.
    """
    enabled = sum(1 for camera in cameras if camera.enabled)
    return Control(
        title="Включение и отключение камеры",
        state=f"включено {enabled} из {len(cameras)}",
        available=False,
        reason=(
            f"Изменение из веб-интерфейса недоступно. Меняется ключом enabled: в "
            f"{_camera_file()}, после чего требуется перезапуск супервизора. Столбец "
            "cameras.enabled в базе не читает ни один процесс: запись туда переключила бы "
            "только надпись на этой странице."
        ),
    )


def _derived_mode(cameras: list[CameraView]) -> SystemMode:
    """Which kinds of camera are actually switched on.

    Sound because `load_workers` has already refused any config in which an enabled camera
    is assigned to no worker group -- so "enabled" and "covered" cannot drift apart at the
    file level. It says nothing about whether the supervisor is running, which is what the
    worker table on the camera wall is for.
    """
    kinds = {camera.camera_type for camera in cameras if camera.enabled}
    if not kinds:
        return SystemMode.IDLE
    if len(kinds) == 1:
        return SystemMode(next(iter(kinds)))
    return SystemMode.BOTH


def _telegram() -> Integration:
    """On or off. Never the token, and never the chat id.

    `telegram_enabled` is the whole of what this page may know. The legacy served the live
    bot token from its settings page to anyone who asked (audit H-04); a token in a browser
    is a token in a proxy log, a screenshot and a support chat. The chat id is withheld
    too: it names where a school's alerts about children are delivered.
    """
    enabled = get_settings().telegram_enabled
    return Integration(
        title="Telegram",
        state="включён" if enabled else "отключён",
        note=(
            "Токен бота и chat id хранятся только в переменных окружения (.env) и здесь "
            "не показываются никогда — легаси-панель отдавала токен любому анонимному "
            "посетителю (аудит H-04). Пустой токен означает, что тревоги видны только в "
            "журнале событий."
        ),
    )

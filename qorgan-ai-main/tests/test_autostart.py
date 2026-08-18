"""Windows autostart — client §11 item 15, which we never built.

**The launchers are RUN here, not read.** A regex asserting that a .bat contains the
string `cd /d "%~dp0.."` proves nothing about what cmd.exe does with it; these tests copy
the real files into a temporary install root, start them from a deliberately wrong
working directory, and read back where they actually went.

Why that is the thing worth testing: Task Scheduler starts a process with **no working
directory**, and `.env`, `yolov8n.pt` and every default path in this system resolve
against it. Each of them fails silently when it is wrong — a missing `.env` means the
published dev SECRET_KEY, a blank RTSP password and no Telegram, with nothing in the log
to say so; missing weights means Ultralytics quietly downloads them from the internet
into whatever folder the task happened to land in. A launcher that starts in the wrong
place produces a system that boots, looks healthy, and is wrong.

What these tests do NOT cover, and nothing here can: whether Task Scheduler on the
school's machine accepts the XML and honours it. That is first-contact work, like the
NVRs (HANDOVER §5) — the registration itself is `deploy/install-autostart.bat`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

DEPLOY = REPO_ROOT / "deploy"
LAUNCHERS = ("qorgan-supervisor.bat", "qorgan-web.bat")
TASK_XMLS = ("qorgan-supervisor.xml", "qorgan-web.xml")

# The token install-autostart.bat rewrites to the real install root.
ROOT_PLACEHOLDER = "__QORGAN_ROOT__"

TASK_NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the launchers are cmd.exe")


@pytest.fixture
def install_root(tmp_path: Path) -> Path:
    """A throwaway install root holding a copy of the real deploy/ directory.

    A copy, because the launchers anchor to their own location (`%~dp0..`) — which is
    exactly the property under test, and which cannot be observed by running the ones in
    this checkout, since those would find this checkout's venv and start a real
    supervisor against the school's config.
    """
    root = tmp_path / "qorgan-install"
    shutil.copytree(DEPLOY, root / "deploy")
    return root


def _run(script: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a launcher from `cwd` — deliberately NOT the install root."""
    return subprocess.run(
        ["cmd", "/c", str(script)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_launcher_starts_in_the_install_root_wherever_it_is_started_from(
    install_root: Path, tmp_path: Path, launcher: str
) -> None:
    """THE test. Task Scheduler sets no working directory, so the launcher must find the
    install root from its own location and go there before it does anything else."""
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()

    result = _run(install_root / "deploy" / launcher, cwd=elsewhere)

    assert str(install_root) in result.stdout, (
        f"{launcher} did not report the install root as its working directory.\n"
        f"started in: {elsewhere}\nstdout: {result.stdout!r}"
    )
    assert str(elsewhere) not in result.stdout, (
        f"{launcher} ran in the directory it was started from. On the school's machine "
        "that is wherever Task Scheduler felt like starting it, and .env is not there."
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_a_missing_virtualenv_is_a_loud_failure_not_a_completed_task(
    install_root: Path, tmp_path: Path, launcher: str
) -> None:
    """There is no venv in the temporary root, which is exactly the state of a machine
    where somebody moved the install or skipped HANDOVER §1.

    Task Scheduler shows a task that exits 0 as "completed successfully" and says nothing
    more, so a launcher that shrugs at a missing interpreter produces a school with no
    cameras being watched and a green tick in the scheduler UI.
    """
    result = _run(install_root / "deploy" / launcher, cwd=tmp_path)

    assert result.returncode != 0, (
        "a missing virtualenv exited 0; the scheduler will call that success"
    )
    assert ".venv" in result.stderr, f"the error does not name what is missing: {result.stderr!r}"


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_launcher_says_where_it_started_before_anything_can_go_wrong(
    install_root: Path, tmp_path: Path, launcher: str
) -> None:
    """The one question worth answering when a scheduled task misbehaves is "where did it
    start", and it must be answered even on the runs that then fail."""
    result = _run(install_root / "deploy" / launcher, cwd=tmp_path)

    assert result.returncode != 0  # no venv: this run failed
    assert str(install_root) in result.stdout, "a failing run did not say where it started"


# -- the task definitions ----------------------------------------------------


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_definition_is_well_formed_xml(name: str) -> None:
    ET.parse(DEPLOY / name)


@pytest.mark.parametrize(
    ("name", "launcher"),
    [("qorgan-supervisor.xml", "qorgan-supervisor.bat"), ("qorgan-web.xml", "qorgan-web.bat")],
)
def test_the_task_runs_the_launcher_and_not_python_directly(name: str, launcher: str) -> None:
    """Through the .bat, so the working-directory guarantee above applies to the real
    thing. A task calling python.exe directly would depend on WorkingDirectory being
    right in the XML, which is one edit away from being wrong forever."""
    root = ET.parse(DEPLOY / name).getroot()
    command = root.find(".//t:Actions/t:Exec/t:Command", TASK_NS)

    assert command is not None and command.text is not None
    assert command.text.endswith(f"deploy\\{launcher}")
    assert command.text.startswith(ROOT_PLACEHOLDER)


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_also_sets_the_working_directory(name: str) -> None:
    """Belt and braces: the .bat does not trust this, and this does not trust the .bat."""
    root = ET.parse(DEPLOY / name).getroot()
    working = root.find(".//t:Actions/t:Exec/t:WorkingDirectory", TASK_NS)

    assert working is not None
    assert working.text == ROOT_PLACEHOLDER


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_has_no_execution_time_limit(name: str) -> None:
    """Task Scheduler's DEFAULT is to kill a task after three days (PT72H).

    These two processes are meant to run until the machine is switched off. With the
    default, the cameras would go dark every third day — at a moment nobody could predict
    and nothing would report, since a killed task is not a crashed one and the supervisor
    is not there to restart itself.
    """
    root = ET.parse(DEPLOY / name).getroot()
    limit = root.find(".//t:Settings/t:ExecutionTimeLimit", TASK_NS)

    assert limit is not None
    assert limit.text == "PT0S", "PT0S means no limit; anything else kills a long-running service"


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_starts_at_boot_and_survives_a_restart(name: str) -> None:
    """"Autostart" means the machine can reboot at 3am after Windows Update and come back
    watching the corridors, with nobody logging in."""
    root = ET.parse(DEPLOY / name).getroot()

    assert root.find(".//t:Triggers/t:BootTrigger", TASK_NS) is not None
    assert root.find(".//t:Settings/t:RestartOnFailure", TASK_NS) is not None


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_does_not_stop_itself_for_power_or_idle_reasons(name: str) -> None:
    """Every one of these defaults to the wrong answer for a service that watches children
    in a corridor: Windows will happily stop it because the machine looks idle, or because
    somebody unplugged it."""
    root = ET.parse(DEPLOY / name).getroot()

    def _setting(path: str) -> str | None:
        node = root.find(f".//t:Settings/{path}", TASK_NS)
        return node.text if node is not None else None

    assert _setting("t:DisallowStartIfOnBatteries") == "false"
    assert _setting("t:StopIfGoingOnBatteries") == "false"
    assert _setting("t:RunOnlyIfIdle") == "false"
    assert _setting("t:IdleSettings/t:StopOnIdleEnd") == "false"


@pytest.mark.parametrize("name", TASK_XMLS)
def test_two_copies_of_a_worker_fleet_are_never_started(name: str) -> None:
    """Two supervisors means two processes per camera on one GPU, two writers to the
    database, and a memory plan (config/workers.yaml) that was measured for one."""
    root = ET.parse(DEPLOY / name).getroot()
    policy = root.find(".//t:Settings/t:MultipleInstancesPolicy", TASK_NS)

    assert policy is not None
    assert policy.text == "IgnoreNew"


# -- preparing a definition schtasks will accept -----------------------------


def _prepare(
    source: Path, destination: Path, root: str, user: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(DEPLOY / "prepare-task-xml.ps1"),
            "-Source", str(source), "-Destination", str(destination),
            "-Root", root, "-User", user,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_prepared_definition_is_the_utf16_schtasks_demands(tmp_path: Path, name: str) -> None:
    """MEASURED, not assumed: `schtasks /create /xml` rejects a UTF-8 task definition —
    with or without a BOM — as "ERROR: The task XML is malformed. (1,2)". The error names
    the XML declaration and says nothing about encoding, so getting this wrong costs an
    afternoon on a machine you flew to.

    The templates stay UTF-8 in the repo so they can be read and diffed; the conversion
    happens here, and the declaration is fixed to match the bytes.
    """
    out = tmp_path / "prepared.xml"

    result = _prepare(DEPLOY / name, out, r"C:\qorgan", "SCHOOL\\qorgan")

    assert result.returncode == 0, result.stderr
    raw = out.read_bytes()
    assert raw[:2] == b"\xff\xfe", "not UTF-16LE with a BOM; schtasks will call this malformed"
    assert 'encoding="UTF-16"' in raw.decode("utf-16"), "the declaration contradicts the bytes"


@pytest.mark.parametrize("name", TASK_XMLS)
def test_a_path_with_spaces_and_a_domain_account_survive_substitution(
    tmp_path: Path, name: str
) -> None:
    """Both values are full of backslashes, and PowerShell's `-replace` is a REGEX
    operator: `C:\\qorgan\\deploy` as a regex replacement eats its own separators and
    corrupts the path silently. A school install under "Program Files" with a domain
    account is the realistic case, not the exotic one."""
    out = tmp_path / "prepared.xml"
    root = r"C:\Program Files\qorgan ai"

    result = _prepare(DEPLOY / name, out, root, "SCHOOL\\qorgan-svc")

    assert result.returncode == 0, result.stderr
    text = out.read_bytes().decode("utf-16")
    launcher = name.replace(".xml", ".bat")
    assert f"{root}\\deploy\\{launcher}" in text
    assert f"<WorkingDirectory>{root}</WorkingDirectory>" in text
    assert "<UserId>SCHOOL\\qorgan-svc</UserId>" in text


def test_an_unsubstituted_placeholder_is_refused_rather_than_registered(tmp_path: Path) -> None:
    """A task pointing at a directory called __QORGAN_ROOT__ registers perfectly happily
    and never runs. Task Scheduler will not tell anyone; the school finds out when they
    need an alert that never came."""
    template = tmp_path / "broken.xml"
    template.write_text(
        (DEPLOY / "qorgan-supervisor.xml").read_text(encoding="utf-8").replace(
            "__QORGAN_USER__", "__QORGAN_SOMETHING_NOBODY_SUBSTITUTES__"
        ),
        encoding="utf-8",
    )

    result = _prepare(template, tmp_path / "out.xml", r"C:\qorgan", "SCHOOL\\qorgan")

    assert result.returncode != 0
    assert "placeholder" in result.stderr.lower()
    assert not (tmp_path / "out.xml").exists(), "a task definition that cannot run was written"


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_runs_as_a_real_user_and_not_as_LOCAL_SYSTEM(name: str) -> None:
    """S-1-5-18 (LOCAL SYSTEM) is the obvious choice for a service and it is wrong here.

    InsightFace resolves its buffalo_l pack out of `~/.insightface`. As SYSTEM that is
    `C:\\Windows\\System32\\config\\systemprofile\\.insightface`, where the school's
    341 MB pack is not — so face recognition would re-download it to a folder nobody
    knows about, or fail outright on a machine with no internet (HANDOVER §0).
    """
    root = ET.parse(DEPLOY / name).getroot()
    user = root.find(".//t:Principals/t:Principal/t:UserId", TASK_NS)

    assert user is not None
    assert user.text == "__QORGAN_USER__", "the task must run as the school's own account"
    assert "S-1-5-18" not in ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("name", TASK_XMLS)
def test_the_task_does_not_run_with_more_privilege_than_it_uses(name: str) -> None:
    """These processes need a GPU, some files and a network socket above port 1024. They
    need administrator for none of it — and they hold live video and photographs of
    children, which is the last thing that should be running elevated for no reason.

    Registering the task does need an admin prompt, because a BootTrigger does. That is
    the installer's problem once, not this process's privilege forever.
    """
    root = ET.parse(DEPLOY / name).getroot()
    level = root.find(".//t:Principals/t:Principal/t:RunLevel", TASK_NS)

    assert level is not None
    assert level.text == "LeastPrivilege"


def test_the_installer_rewrites_the_placeholder_the_xml_actually_uses() -> None:
    """The placeholder is a contract between three files. Renaming it in one of them
    would leave a task pointing at a directory literally called __QORGAN_ROOT__, which
    Task Scheduler accepts without complaint and which never runs."""
    installer = (DEPLOY / "install-autostart.bat").read_text(encoding="utf-8")

    assert ROOT_PLACEHOLDER in installer
    for name in TASK_XMLS:
        assert ROOT_PLACEHOLDER in (DEPLOY / name).read_text(encoding="utf-8")


def test_no_secret_is_baked_into_the_deploy_files() -> None:
    """R4. These files are committed; .env is not. The legacy shipped a live Telegram
    token and the fleet's RTSP password in its own repository.

    Assignments, not mentions: these files explain WHY the working directory matters by
    naming `.env` and `SECRET_KEY`, and prose about a secret is not a secret. What must
    never appear is a value — `SECRET_KEY=...`, `set RTSP_PASSWORD=...` — or a password
    handed to schtasks on a command line, where it lands in the console history and in
    every process listing on the machine while it runs.
    """
    import re

    assignment = re.compile(
        r"(secret_key|rtsp_password|telegram_bot_token|password)\s*=\s*\S", re.IGNORECASE
    )

    for path in DEPLOY.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not assignment.search(text), f"{path.name} assigns a secret a value"

        # /rp on an actual schtasks invocation — not the letters "/rp" appearing in a
        # comment explaining why it is not used.
        for line in text.splitlines():
            if "schtasks" in line.lower() and re.search(r"/rp\b", line, re.IGNORECASE):
                pytest.fail(
                    f"{path.name} passes a password to schtasks on the command line, where "
                    "it is visible in the process list of every user on the machine while "
                    f"it runs, and stays in the console history:\n    {line.strip()}"
                )

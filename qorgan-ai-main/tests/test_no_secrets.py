"""Rule R4, enforced.

The legacy shipped its live Telegram bot token in .env and one near-dictionary RTSP
password -- the same one for the whole fleet -- in all ten YAML files and in the
database. It then printed the full `rtsp://user:password@...` URL to stdout and drew
it as text onto debug JPEGs that the unauthenticated web UI served.

Every one of those is a grep away from being caught. So: grep for them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import CONFIG_DIR, REPO_ROOT

# A key whose NAME says it holds a secret has no business in a config file.
SECRET_KEY_PATTERN = re.compile(
    r"^\s*[\w-]*(password|passwd|secret|token|api[_-]?key|credential)[\w-]*\s*:", re.IGNORECASE
)

# Credentials embedded in a URL, e.g. rtsp://admin:hunter2@192.168.1.4
URL_CREDENTIALS = re.compile(r"\w+://[^:/\s]+:[^@/\s]+@")

# The legacy fleet password. Assembled at runtime so that this file is not itself
# the thing it is searching for -- and so that the string never appears in the repo.
LEGACY_PASSWORD = "12345678" + "_a"

CONFIG_FILES = sorted(CONFIG_DIR.rglob("*.yaml"))


def test_there_are_config_files_to_check() -> None:
    assert CONFIG_FILES, "no YAML found; this test would pass vacuously"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_declares_no_secret_shaped_key(path: Path) -> None:
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SECRET_KEY_PATTERN.match(line)
    ]
    assert not offenders, "secrets belong in the environment, not in config:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_embeds_no_credentials_in_a_url(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not URL_CREDENTIALS.search(text), f"{path.name} contains a URL with credentials in it"


def _tracked_text_files() -> list[Path]:
    roots = [REPO_ROOT / "src", REPO_ROOT / "config", REPO_ROOT / "tests"]
    suffixes = {".py", ".yaml", ".yml", ".toml", ".md", ".ini"}
    return [p for root in roots for p in root.rglob("*") if p.suffix in suffixes and p.is_file()]


def test_the_legacy_camera_password_appears_nowhere() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _tracked_text_files()
        if LEGACY_PASSWORD in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, f"the legacy fleet password was carried over into: {offenders}"


def test_env_example_is_tracked_but_env_is_not() -> None:
    assert (REPO_ROOT / ".env.example").is_file(), ".env.example must be shipped"
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore, ".env must be gitignored"

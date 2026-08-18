"""`qorgan weapons weights <camera>` — the command somebody runs on the school's machine.

It is the only thing in this project that will ever say what is actually inside a weapons
checkpoint, because the panel deliberately does not load one (rule R3: the web process
knows nothing about the worker). So it is the command that answers the client's question --
"is there a model in there?" -- and until this file existed **not one line of it was
covered**: `test_weapons_feasibility.py` drives `camera-report`, and nothing drove this.

That mattered more than an untested command usually does, for two reasons:

  * its case-3 handler (the file is present, is large enough, and does not load) is the
    ONLY handler for that case anywhere, and case 3 was the one failure mode
    `test_weapons_refusal.py` named in its docstring and did not assert;
  * the exit CODES are the product. A script has to tell "this camera cannot do this" (1)
    from "I could not answer" (2) from "yes" (0), and `identity`'s equivalent of a 1 was
    discovered in month four from an event log full of Unknown.

Every test here goes through `qorgan.cli.build_parser`, i.e. through the argv a person
types -- not by calling `cmd_weights` with a hand-built namespace. The subcommand being
wired into the real parser is half of what is being asserted; a command nobody can invoke
is not a diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from qorgan.cli import build_parser
from qorgan.settings import Settings, override_settings
from qorgan.weapons.cli import REFUSED, UNANSWERED
from qorgan.weapons.weights import MIN_PLAUSIBLE_WEIGHTS_BYTES
from tests.weapons_fixtures import config_dir_with, weapons_camera_dict

CAMERA = "entrance_weapons"

WeightsRun = Callable[..., int]


@pytest.fixture
def weights_command(settings: Settings, tmp_path: Path) -> WeightsRun:
    """Run `weapons weights` against a real config directory naming this model path.

    A copied config tree rather than a stub, so `load_cameras()` does the whole job it does
    in production -- base.yaml, the profile lookup, the three-layer merge and the
    discriminated union. A test that handed a `WeaponsCamera` straight to the command would
    skip every one of those and still look like coverage.
    """
    counter = {"n": 0}

    def run(model_path: str, camera: str = CAMERA) -> int:
        counter["n"] += 1
        directory = config_dir_with(
            tmp_path / f"cfg{counter['n']}",
            weapons_camera_dict(weapons={"model": {"model": model_path}}),
        )
        override_settings(settings.model_copy(update={"config_dir": directory}))
        args = build_parser().parse_args(["weapons", "weights", camera])
        return args.func(args)

    return run


# -- 1. the file is not there ----------------------------------------------


def test_a_missing_model_exits_refused_and_names_the_file(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "qorgan-weapons.pt"
    assert weights_command(str(missing)) == REFUSED
    stderr = capsys.readouterr().err
    assert str(missing) in stderr
    assert "not found" in stderr


def test_the_command_tells_the_reader_not_to_look_for_a_degraded_mode(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    """Looking for a fallback is how the previous system got its. The sentence is the
    product: whoever runs this on site reads it and stops looking."""
    weights_command(str(tmp_path / "absent.pt"))
    assert "fall back" in capsys.readouterr().err


# -- 2. the file is there and is empty -------------------------------------


def test_the_clients_own_zero_byte_artefact_exits_refused(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    """`models/best.pt`, 0 bytes, for months. `Path.is_file()` is True for it."""
    empty = tmp_path / "best.pt"
    empty.touch()
    assert empty.is_file(), "the premise: the check the old system made would pass"

    assert weights_command(str(empty)) == REFUSED
    stderr = capsys.readouterr().err
    assert "0 bytes" in stderr
    assert str(empty) in stderr


# -- 3. the file is there, is big enough, and does not load ----------------


def test_a_truncated_download_exits_refused_and_says_which_failure_it_is(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    """**The case that had no coverage anywhere.**

    These bytes are over the size gate, so the module's own two checks have already said
    yes and torch is the one that says no. The message must not collapse that into "could
    not load the model": it has to send the reader to a truncated download, a wrong format
    or an lfs pointer, because those are the three things it actually is, and it must say
    that the pipeline does not start.
    """
    broken = tmp_path / "truncated.pt"
    broken.write_bytes(b"PK\x03\x04" + b"\x17" * (MIN_PLAUSIBLE_WEIGHTS_BYTES * 2))

    assert weights_command(str(broken)) == REFUSED
    stderr = capsys.readouterr().err
    assert "did not load" in stderr
    assert str(broken) in stderr
    assert "truncated download" in stderr
    assert "does not start" in stderr


# -- "I could not answer" is a different fact from "the answer is no" ------


def test_an_unknown_camera_is_unanswered_and_not_refused(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    """2, not 1. A script that cannot tell these apart reports a typo as a broken camera."""
    assert weights_command(str(tmp_path / "x.pt"), camera="no_such_camera") == UNANSWERED
    assert "no camera called" in capsys.readouterr().err


def test_a_camera_that_is_not_a_weapons_camera_is_unanswered(
    weights_command: WeightsRun, tmp_path: Path, capsys
) -> None:
    """`hall_left` is a real camera in the school's own config and has no `weapons:` block.
    Reporting that as REFUSED would read as "this camera's weapons model is broken"."""
    assert weights_command(str(tmp_path / "x.pt"), camera="hall_left") == UNANSWERED
    assert "not a weapons camera" in capsys.readouterr().err

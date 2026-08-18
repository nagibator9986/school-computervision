"""No model ⇒ no weapons pipeline. The one refusal this module exists for.

The client ran a weapons module for months with **a `best.pt` of 0 bytes** in it. The code
checked that a path was configured, found one, failed to load it, logged a warning, fell
back to motion analysis, and reported healthy. Nobody was lied to on purpose; nothing ever
checked. The school watched a motion detector and called it weapons detection.

So these tests are not about an error message. They are about the four separate ways the
thing can be absent, that **each of them stops the worker**, and that no arrangement of
this code lets a weapons camera come up watching nothing:

  1. the file is not there;
  2. the file is there and is EMPTY, or far too small -- the case that actually happened,
     and the reason nothing here asks `Path.is_file()` and calls that a check;
  3. the file is there, is big enough, and does not load;
  4. it loads and is the wrong KIND of model -- a classifier, which has no boxes, so
     §12.1's «проверка нахождения рядом с человеком» can never be asked of it. It would
     have run at full GPU cost and been silent forever, looking healthy throughout.

And a fifth nobody thinks of: weights that load and detect but cannot produce any class
this camera alarms on. That is the 0-byte failure again with a working model in it.

`test_the_real_worker_startup_path_refuses` is the one that matters most, because it is
the only one here that is not about a function: it drives the worker's own builder with a
weapons camera and no weights, and asserts the exception comes out. The others could all
pass while something upstream caught it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from qorgan.config.weapons import KNOWN_CONFUSABLES, WeaponModelSettings
from qorgan.config.workers import WorkerGroup
from qorgan.weapons.model import YoloWeaponModel
from qorgan.weapons.weights import (
    MIN_PLAUSIBLE_WEIGHTS_BYTES,
    WeaponWeightsUnusable,
    inert_confusables,
    inspect_weights_file,
    refuse_unusable_weights,
)
from qorgan.worker import builders
from tests.weapons_fixtures import loaded_weights, plausible_weights, weapons_camera

# -- 1. the file is not there ----------------------------------------------


def test_a_missing_weights_file_refuses_and_names_it(tmp_path: Path) -> None:
    missing = tmp_path / "qorgan-weapons.pt"
    with pytest.raises(WeaponWeightsUnusable) as refused:
        inspect_weights_file(missing)

    message = str(refused.value)
    assert str(missing) in message, "the message must name the file somebody has to fetch"
    assert "not found" in message


def test_the_refusal_says_there_is_no_fallback(tmp_path: Path) -> None:
    """The sentence is the product here. Whoever reads it must not go looking for a
    degraded mode, because looking for one is how the previous system got its."""
    with pytest.raises(WeaponWeightsUnusable) as refused:
        inspect_weights_file(tmp_path / "nope.pt")
    assert "fall back" in str(refused.value)


# -- 2. the file is there and is empty -------------------------------------


def test_a_zero_byte_file_is_not_weights(tmp_path: Path) -> None:
    """**This is the client's actual state.** `models/best.pt`, 0 bytes, for months.

    `Path.is_file()` is True for it, which is exactly why "the path exists" is not a
    check and why this module measures the SIZE.
    """
    empty = tmp_path / "best.pt"
    empty.touch()
    assert empty.is_file(), "the premise: the path check the old system made would pass"

    with pytest.raises(WeaponWeightsUnusable) as refused:
        inspect_weights_file(empty)

    message = str(refused.value)
    assert "0 bytes" in message
    assert str(empty) in message


def test_a_git_lfs_pointer_committed_as_the_model_is_refused(tmp_path: Path) -> None:
    """~130 bytes of text that says where the artefact is. It is not the artefact."""
    pointer = tmp_path / "best.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        "size 6549796\n",
        encoding="utf-8",
    )
    with pytest.raises(WeaponWeightsUnusable):
        inspect_weights_file(pointer)


def test_the_size_gate_is_a_gate_and_not_a_rounding(tmp_path: Path) -> None:
    """One byte under refuses; one byte over is somebody else's problem (case 3)."""
    under = tmp_path / "under.pt"
    under.write_bytes(b"\x00" * (MIN_PLAUSIBLE_WEIGHTS_BYTES - 1))
    with pytest.raises(WeaponWeightsUnusable):
        inspect_weights_file(under)

    over = plausible_weights(tmp_path, "over.pt")
    artefact = inspect_weights_file(over)
    assert artefact.size_bytes == MIN_PLAUSIBLE_WEIGHTS_BYTES + 1


def test_a_readable_file_is_fingerprinted_so_two_best_pt_can_be_told_apart(
    tmp_path: Path,
) -> None:
    """Every Ultralytics run in the world names its output `best.pt`. The name is not an
    identity, so the panel shows a hash of the bytes."""
    one = tmp_path / "a.pt"
    one.write_bytes(b"\x01" * (MIN_PLAUSIBLE_WEIGHTS_BYTES + 1))
    two = tmp_path / "b.pt"
    two.write_bytes(b"\x02" * (MIN_PLAUSIBLE_WEIGHTS_BYTES + 1))

    assert inspect_weights_file(one).fingerprint != inspect_weights_file(two).fingerprint
    assert inspect_weights_file(one).fingerprint == inspect_weights_file(one).fingerprint


# -- 3. the file is there, is big enough, and does NOT load -----------------
#
# The case the four tests above hand on and nothing was catching. Cases 1, 2, 4 and 5 are
# all arithmetic or structure, so they can be asserted without torch; this one can only be
# asserted by letting torch try and fail, which is why it was the easy one to leave out.
#
# It has to be asserted, because it is the case where the module has ALREADY said yes
# twice -- the file is present and it is over the size gate -- and the only thing left
# between the school and a camera watching nothing is that the exception is allowed out.


def _unloadable(tmp_path: Path) -> Path:
    """Bytes that pass every check this module can make without torch, and are not a model.

    Over `MIN_PLAUSIBLE_WEIGHTS_BYTES`, so the size gate has already said yes: a truncated
    download, an lfs pointer somebody padded, or a `.pt` that is really something else.
    """
    path = tmp_path / "truncated.pt"
    path.write_bytes(b"PK\x03\x04" + b"\x17" * (MIN_PLAUSIBLE_WEIGHTS_BYTES * 2))
    return path


def test_a_file_that_clears_the_size_gate_and_does_not_load_leaves_no_object(
    tmp_path: Path,
) -> None:
    """Case 3, in the constructor: there is no half-built `YoloWeaponModel` to hold.

    The premise is asserted first, because it is the whole point: `inspect_weights_file`
    is HAPPY with these bytes. So this is not the size gate firing under another name --
    it is torch refusing, and the refusal being allowed out of `__init__`.

    `WeaponWeightsUnusable` is explicitly NOT expected. This module's own exception means
    "we checked and said no"; case 3 is the library saying no, and pretending otherwise
    would mean wrapping a torch failure in a sentence about a file we already validated.
    What matters is only that something is raised and nothing is returned.
    """
    path = _unloadable(tmp_path)
    assert inspect_weights_file(path).size_bytes > MIN_PLAUSIBLE_WEIGHTS_BYTES

    with pytest.raises(Exception) as broke:
        YoloWeaponModel(
            WeaponModelSettings(model=str(path)), "entrance_weapons", ("knife",), device="cpu"
        )
    assert not isinstance(broke.value, WeaponWeightsUnusable)

    # And it NAMES THE FILE. torch's own message for this is
    # `OSError: [Errno 22] Invalid argument`, which reaches the operator through the
    # heartbeat's `last_error` and names nothing at all -- while `worker/builders.py`
    # promises "the missing file named in every line". The promise was false for exactly
    # this case, which is the only one where the file is present.
    message = str(broke.value)
    assert str(path) in message, "the operator gets an errno unless the path is in here"
    assert "did not load" in message
    assert "truncated download" in message
    assert "entrance_weapons" in message, "and which camera it was"


def test_the_real_worker_startup_path_refuses_a_file_that_does_not_load(
    session: Session, tmp_path: Path
) -> None:
    """The same case at the only place it decides anything: `build_all`.

    The companion to `test_the_real_worker_startup_path_refuses`. That one proves a MISSING
    file stops the worker; this one proves a file that is present, is big enough, and is
    rubbish stops it too -- which is the state a half-finished `scp` leaves behind, and the
    one where "the weights are on the machine" is true and worthless.
    """
    del session  # `_build_weapons` calls ensure_cameras(); it needs the schema
    camera = weapons_camera(weapons={"model": {"model": str(_unloadable(tmp_path))}})

    with (
        patch.object(builders, "_require_gpu_now", lambda: None),
        pytest.raises(Exception) as broke,
    ):
        builders.build_all(_weapons_group(), {"entrance_weapons": camera})
    assert not isinstance(broke.value, SystemExit), "a refusal, not a clean exit"


# -- 4. it loaded, and it is the wrong kind of model ------------------------


def test_a_classifier_is_refused_even_though_it_loads() -> None:
    """The client's own defect, one layer deeper than they got to.

    A violence CLASSIFIER loads perfectly and emits no boxes, so it can never answer
    «рядом с человеком» -- it would run at full cost and be silent forever.
    """
    with pytest.raises(WeaponWeightsUnusable) as refused:
        refuse_unusable_weights(
            loaded_weights(task="classify", class_names=("violence", "normal")),
            ("knife",),
            "entrance_weapons",
        )
    assert "classify" in str(refused.value)


@pytest.mark.parametrize("task", ["classify", "pose", "segment", ""])
def test_only_a_detector_is_accepted(task: str) -> None:
    """`pose` and `segment` load fine and answer a different question."""
    with pytest.raises(WeaponWeightsUnusable):
        refuse_unusable_weights(loaded_weights(task=task), ("knife",), "cam")


def test_weights_that_declare_no_classes_are_refused() -> None:
    with pytest.raises(WeaponWeightsUnusable) as refused:
        refuse_unusable_weights(loaded_weights(class_names=()), ("knife",), "cam")
    assert "no classes" in str(refused.value)


# -- 5. it detects, and cannot produce anything this camera alarms on -------


def test_a_person_detector_configured_to_find_knives_is_refused() -> None:
    """The 0-byte failure with a working model in it.

    These weights load, detect, and have a perfectly good `person` class. The camera is
    configured for `knife`. The pipeline would run at full GPU cost and be silent BY
    CONSTRUCTION, and every screen would look healthy.
    """
    with pytest.raises(WeaponWeightsUnusable) as refused:
        refuse_unusable_weights(
            loaded_weights(class_names=("person", "car")), ("knife", "firearm"), "cam"
        )
    message = str(refused.value)
    assert "knife" in message and "firearm" in message, "say what was asked for"
    assert "person" in message, "and say what the weights can actually produce"


def test_a_partial_overlap_is_allowed_and_says_so() -> None:
    """Weights that do knives but not firearms are usable for the knife. Refusing the
    whole camera would be stricter than the question."""
    weights = loaded_weights(class_names=("knife", "scissors"))
    refuse_unusable_weights(weights, ("knife", "firearm"), "cam")
    assert weights.describes(("knife", "firearm")) == ("knife",)


def test_usable_weights_raise_nothing() -> None:
    assert refuse_unusable_weights(loaded_weights(), ("knife",), "cam") is None


# -- 6. it detects the targets, and screen 3 is DEAD ------------------------
#
# The asymmetry, named. `target_classes` are checked against the loaded model loudly, and a
# camera whose targets it cannot emit is refused outright. `confusable_classes` -- the whole
# input to screen 3, the «нож или ручка» withholding -- was checked against NOTHING, so the
# defence could be absent from the running system while present in the configuration.


def test_the_shipped_confusables_are_inert_against_a_realistic_class_map() -> None:
    """**The measurement, as a test, so it stops being a paragraph in a report.**

    Weights whose classes are (knife, person, cell phone) -- a perfectly plausible real
    detector -- against the shipped `confusable_classes`. The intersection is EMPTY, so
    screen 3 can never fire, at any confidence, for the life of the installation, and
    `refuse_unusable_weights` accepted exactly this silently.
    """
    weights = loaded_weights(class_names=("knife", "person", "cell phone"))
    inert = inert_confusables(weights, KNOWN_CONFUSABLES)

    assert set(inert) == set(KNOWN_CONFUSABLES), "not one of them is producible"
    assert "phone" in inert, "the schema says `phone`; COCO says `cell phone`. Not the same."
    assert refuse_unusable_weights(weights, ("knife",), "cam") is None, (
        "the premise: the targets are fine, so nothing else in this file complains"
    )


def test_an_inert_screen_three_is_not_a_refusal_and_that_is_deliberate() -> None:
    """A detector trained on knives has no reason to emit `pen`.

    Refusing here would block every plausible weapons model on the day one finally arrives,
    which is a worse failure than a defence that is inert. So the contract is: return the
    list, and make everything that CAN speak, speak. This pins the decision so that
    reversing it has to be meant.
    """
    weights = loaded_weights(class_names=("knife", "firearm"))
    assert refuse_unusable_weights(weights, ("knife",), "cam") is None
    assert inert_confusables(weights, KNOWN_CONFUSABLES)


def test_a_confusable_the_weights_do_produce_is_not_reported_inert() -> None:
    """The control. Without it the function could return everything and still pass above."""
    weights = loaded_weights(class_names=("knife", "scissors", "pen"))
    inert = inert_confusables(weights, KNOWN_CONFUSABLES)
    assert set(inert) == {"phone", "ruler", "kitchen_utensil", "toy", "clothing"}
    assert "scissors" not in inert
    assert "pen" not in inert


def test_declaring_no_confusables_leaves_nothing_to_be_inert_about() -> None:
    assert inert_confusables(loaded_weights(), ()) == ()


# -- the refusal reaches the worker, which is the point ---------------------


def _weapons_group() -> WorkerGroup:
    return WorkerGroup(name="weapons", device="cuda:0", cameras=["entrance_weapons"])


def test_the_real_worker_startup_path_refuses(session: Session, tmp_path: Path) -> None:
    """**The test this file exists for.** The worker's own builder, a real weapons camera,
    no weights: the exception must come out of `build_all`.

    Not a call to `inspect_weights_file` -- every other test here is that, and every one
    of them could pass while `_build_weapons` wrapped the whole thing in a `try`. This
    drives the function `_serve` actually calls.

    The GPU check is patched out and nothing else is: it runs BEFORE the weights are
    touched, so on a machine without a card this test would refuse for the wrong reason
    and prove nothing about weights.
    """
    del session  # `_build_weapons` calls ensure_cameras(); it needs the schema
    camera = weapons_camera(weapons={"model": {"model": str(tmp_path / "absent.pt")}})

    with (
        patch.object(builders, "_require_gpu_now", lambda: None),
        pytest.raises(WeaponWeightsUnusable) as refused,
    ):
        builders.build_all(_weapons_group(), {"entrance_weapons": camera})

    assert "absent.pt" in str(refused.value)


def test_the_real_worker_startup_path_refuses_a_zero_byte_file(
    session: Session, tmp_path: Path
) -> None:
    """The same path, with the client's exact artefact: a file that is there and empty."""
    del session
    empty = tmp_path / "best.pt"
    empty.touch()
    camera = weapons_camera(weapons={"model": {"model": str(empty)}})

    with (
        patch.object(builders, "_require_gpu_now", lambda: None),
        pytest.raises(WeaponWeightsUnusable) as refused,
    ):
        builders.build_all(_weapons_group(), {"entrance_weapons": camera})

    assert "0 bytes" in str(refused.value)


def test_a_group_with_no_weapons_camera_builds_nothing_and_asks_for_no_gpu() -> None:
    """The refusal must not become a tax on every other worker in the school."""
    assert builders._build_weapons(_weapons_group(), {}) == {}


# -- nothing anywhere may catch it and carry on ----------------------------
#
# In `tests/test_weapons_startup_chain.py`. Every test above asserts that something is
# RAISED, and every one of them would still pass if a step of worker startup caught it and
# started a motion detector -- so the structural half is a file of its own, reading `src/`
# with `ast` rather than driving it, and it also guards itself against inspecting nothing.

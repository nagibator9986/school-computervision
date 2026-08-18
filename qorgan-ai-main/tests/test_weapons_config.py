"""The weapons schema, and the arrangements it refuses to represent (rule R10).

Two kinds of refusal here and they are not the same kind of thing.

**Structural.** A weapons camera has no default weights and cannot carry a bullying block.
The legacy shipped ~25 bullying keys onto two canteen cameras that never read one of them,
because nothing stopped it; the discriminated union stops it. And
`WeaponModelSettings.model` has no default ON PURPOSE -- a default here would put a
plausible path in a YAML file that nothing on disk can satisfy, which is exactly how a
0-byte `best.pt` came to look like a configured model.

**Semantic.** §12.1 says «НЕ отправлять тревогу по одному кадру» in those words, and there
are two ways to configure your way back to one frame: ask for one observation, or ask for
a reconfirmation that cannot bite. Both are refused, and the second is the one that hides
-- a gate below the entry bar is a rename, not a check.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qorgan.config.camera import CAMERA_ADAPTER, WeaponsCamera
from qorgan.config.weapons import KNOWN_CONFUSABLES, KNOWN_TARGETS, WeaponsConfig
from tests.weapons_fixtures import weapons_camera_dict


def _config(**overrides) -> WeaponsConfig:
    base = {"model": {"model": "qorgan-weapons.pt"}}
    base.update(overrides)
    return WeaponsConfig.model_validate(base)


# -- there is no default that means "no model" -----------------------------


def test_a_weapons_camera_with_no_weapons_block_is_a_startup_error() -> None:
    """Before a process is spawned, before a frame is read."""
    camera = weapons_camera_dict()
    del camera["weapons"]
    with pytest.raises(ValidationError):
        CAMERA_ADAPTER.validate_python(camera)


def test_a_weapons_block_with_no_model_named_is_refused() -> None:
    with pytest.raises(ValidationError) as refused:
        WeaponsConfig.model_validate({"model": {}})
    assert "model" in str(refused.value)


def test_the_model_path_has_no_default_anybody_could_inherit() -> None:
    """Every other model in this schema defaults to a filename `fetch_models.py` fetches.
    There is no such file for weapons, and inventing one is the 0-byte defect in schema
    form."""
    from qorgan.config.weapons import WeaponModelSettings

    assert WeaponModelSettings.model_fields["model"].is_required()


# -- the union makes the legacy's mistake unrepresentable ------------------


def test_a_weapons_camera_cannot_carry_a_bullying_block() -> None:
    with pytest.raises(ValidationError):
        CAMERA_ADAPTER.validate_python(weapons_camera_dict(bullying={"zones": {}}))


def test_a_weapons_camera_must_have_the_weapons_role() -> None:
    with pytest.raises(ValidationError) as refused:
        CAMERA_ADAPTER.validate_python(weapons_camera_dict(role="main_hall"))
    assert "weapons role" in str(refused.value)


def test_a_valid_weapons_camera_resolves_to_the_weapons_type() -> None:
    camera = CAMERA_ADAPTER.validate_python(weapons_camera_dict())
    assert isinstance(camera, WeaponsCamera)
    assert camera.weapons.model.model == "qorgan-weapons.pt"


def test_an_unknown_key_anywhere_in_the_block_is_refused() -> None:
    """`extra="forbid"`. A misspelt knob that silently does nothing is how 225 keys got
    where they are."""
    with pytest.raises(ValidationError):
        _config(min_object_pixel=24.0)


# -- «НЕ отправлять тревогу по одному кадру» -------------------------------


def test_one_observation_is_refused_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        _config(min_track_observations=1)


def test_a_reconfirmation_that_cannot_bite_is_refused() -> None:
    """Asking for MORE reconfirmations than observations makes the alert unreachable
    rather than stricter: the reconfirmation is drawn from the observations the track
    already collected."""
    with pytest.raises(ValidationError) as refused:
        _config(min_track_observations=2, reconfirm_observations=3)
    assert "greater than min_track_observations" in str(refused.value)


def test_a_second_gate_under_the_entry_bar_is_refused() -> None:
    """The one that hides. Every observation that entered the track already satisfies it,
    so the "re-confirmation" checks nothing at all -- and every number in the YAML would
    look deliberate."""
    with pytest.raises(ValidationError) as refused:
        _config(model={"model": "w.pt", "conf": 0.5}, reconfirm_confidence=0.4)
    assert "checks nothing" in str(refused.value)


def test_the_shipped_defaults_need_three_frames_and_two_confident_ones() -> None:
    rules = _config()
    assert rules.min_track_observations == 3
    assert rules.reconfirm_observations == 2
    assert rules.reconfirm_confidence > rules.model.conf


# -- classes -------------------------------------------------------------


def test_a_target_the_pipeline_has_no_rule_for_is_refused() -> None:
    with pytest.raises(ValidationError) as refused:
        _config(target_classes=["chainsaw"])
    assert "chainsaw" in str(refused.value)


def test_a_class_cannot_be_both_the_alarm_and_the_thing_that_withholds_it() -> None:
    """Every sighting would suppress itself and the module would be silent while looking
    like it worked."""
    with pytest.raises(ValidationError) as refused:
        _config(target_classes=["knife"], confusable_classes=["knife", "pen"])
    assert "BOTH" in str(refused.value)


def test_the_client_asked_for_a_knife_and_a_firearm_and_both_are_targets() -> None:
    """Owner's answer, 2026-07-29: «нужны и нож, и огнестрел»."""
    assert "knife" in KNOWN_TARGETS
    assert "firearm" in KNOWN_TARGETS
    both = _config(target_classes=["knife", "firearm"])
    assert both.target_classes == ["knife", "firearm"]


def test_every_false_object_the_client_named_is_a_known_confusable() -> None:
    """§12.1: телефон, ручка, линейка, ножницы, кухонные предметы, игрушки, одежда."""
    assert set(KNOWN_CONFUSABLES) == {
        "phone",
        "pen",
        "ruler",
        "scissors",
        "kitchen_utensil",
        "toy",
        "clothing",
    }


def test_the_default_targets_and_confusables_do_not_overlap() -> None:
    assert not set(KNOWN_TARGETS) & set(KNOWN_CONFUSABLES)


# -- zones ---------------------------------------------------------------


def _zone(**overrides) -> dict:
    zone = {"area": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}, "classes": ["knife"]}
    zone.update(overrides)
    return zone


def test_a_zone_that_exempts_a_class_nothing_produces_is_refused() -> None:
    """A rule that does nothing is a rule somebody will trust."""
    with pytest.raises(ValidationError) as refused:
        _config(zones={"ordinary_tools": [_zone(classes=["spoon"])]})
    assert "spoon" in str(refused.value)


def test_a_zone_that_makes_nothing_ordinary_is_refused() -> None:
    with pytest.raises(ValidationError):
        _config(zones={"ordinary_tools": [_zone(classes=[])]})


def test_zones_are_fractions_of_the_frame_and_refuse_pixels() -> None:
    """Stored resolution-independently, like every other zone in this project."""
    with pytest.raises(ValidationError):
        _config(zones={"ordinary_tools": [_zone(area={"x1": 0, "y1": 0, "x2": 640, "y2": 480})]})


def test_no_zones_is_the_default_and_is_valid() -> None:
    """Owner's answer, 2026-07-29: the kitchen is OUTSIDE the controlled area, so no
    weapons camera in this school needs a zone. The mechanism stays because production
    reads it; the DEFAULT is that it changes nothing."""
    assert _config().zones.ordinary_tools == []


# -- the lens, which is the only optical input not already in this file ----


def test_the_lens_defaults_to_the_documented_assumption() -> None:
    assert _config().lens_hfov_degrees == 78.0


def test_the_lens_is_per_camera_so_two_cameras_can_disagree() -> None:
    """The whole point of the key: the client's answer of 2026-07-29 is that a camera
    goes at the entrance so the object is large AND the other cameras stay in play."""
    entrance = _config(lens_hfov_degrees=45.0)
    corridor = _config(lens_hfov_degrees=104.0)
    assert entrance.lens_hfov_degrees != corridor.lens_hfov_degrees


@pytest.mark.parametrize("degrees", [0.0, -10.0, 180.0, 361.0])
def test_an_impossible_lens_is_refused(degrees: float) -> None:
    with pytest.raises(ValidationError):
        _config(lens_hfov_degrees=degrees)


def test_the_schema_knows_whether_a_human_wrote_the_lens_down() -> None:
    """The panel prints ПРЕДПОЛОЖЕНИЕ when nobody did, and that distinction is read off
    pydantic rather than guessed by comparing to the default -- a camera whose datasheet
    really says 78° has been checked, and must not be labelled as an assumption."""
    assumed = _config()
    stated = _config(lens_hfov_degrees=78.0)
    assert "lens_hfov_degrees" not in assumed.model_fields_set
    assert "lens_hfov_degrees" in stated.model_fields_set

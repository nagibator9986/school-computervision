"""A declared config key with no consumer is a trap laid for whoever tunes next.

About thirty knobs were parsed, validated, and read by nothing -- `SeparationGuard`,
`ViolenceSettings`, `min_group_size`, six gate thresholds -- while the gates hardcoded
multipliers of `PairMetrics` values instead. Editing them in YAML did nothing at all.
That is exactly how the legacy's 225 keys got where they are: not by anyone deciding to
ship dead config, but by nothing ever saying no.

So: not a one-off cleanup. A test. `tests.config_scan` resolves the OWNER of every
attribute access in `src/` and this file fails if a declared field has no reader.

WHAT MAKES THIS DIFFERENT FROM A GREP. A name-based scan cannot tell a config field from
an identically-named attribute elsewhere: `ProximityOnlyGate.max_speed` reads as "alive"
because `PairData.max_speed` exists. The honest thing to do about a blind spot is not to
describe it in a comment -- a blind spot documented in a comment is a blind spot the next
person will not read. So the scan resolves owners, and where it CANNOT, it says so
mechanically, here, in `UNRESOLVED_KEYS`.

AND A BLIND SPOT IS ITSELF A KEY, NEVER A NAME. This is what makes the list safe, and it
was got wrong once already. `args.device` in `evaluation/cli.py` is an argparse Namespace
the scan cannot type. Keyed on the name, that one access excuses EVERY field called
`device` in the fleet -- so a genuinely dead `RtspSettings.device` could be exempted by a
single line here, on the strength of an access that had nothing to do with it, and the
suite would go green over a corpse. The check would have looked exactly like a check.
So `config_report` asks which models the base could have been, given every attribute read
off it: `args` is also read for `.clips` and `.out`, which no config declares, so it
could not have been an `RtspSettings` and it excuses nothing of `RtspSettings`'.

On that footing the list is enforced in three directions:

  * a key that is unresolved AND not otherwise proven read -- the only kind whose
    verdict the blind spot actually changes -- MUST be listed, with a justification and
    the read site a human actually found, otherwise `test_the_scan_declares_its_blind_
    spots` fails and names it. An unresolved key the scan already proves read some other
    way needs no entry: the unattributed access could not have flipped it to dead, so
    there is nothing to excuse. (Measured on this codebase: 24 keys are unresolved; 0
    need listing, because all 24 are also proven read.)
  * a listed key that no longer needs listing -- because the scan can now see it, or
    because the field is gone -- fails `test_no_exemption_outlives_its_reason`. A stale
    exemption is the same bug wearing the same hat;
  * a listed key that goes genuinely dead loses the access that excused it, so it stops
    being unresolved and starts being dead: it fails `test_no_exemption_outlives_its_
    reason` as a spent excuse AND the dead-key test by name. An entry here cannot make
    a key nothing reads pass, because the entry is only ever honoured for a key some
    access could really have been reading. The list cannot hide a corpse.

That is the point. An exemption costs a human a sentence, and the test collects the debt.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.config_index import UNKNOWN, Index
from tests.config_report import attribute_blind_spots, models_shaped_like
from tests.config_scan import scan
from tests.conftest import SRC_DIR

INDEX, REPORT = scan(SRC_DIR)

# Every key the scan cannot vouch for, and why it is nevertheless live. Each entry is a
# claim a human checked by hand and can be checked again. Keep it SHORT: it is an
# admission of a blind spot, not a parking space.
#
# It is empty -- but NOT because the scan attributed every access in `src/`. There are
# plenty it cannot type (`REPORT.notes` has them). It is empty because every KEY those
# accesses could have been reading is separately PROVEN read, so not one of them changes
# a verdict and not one needs excusing. The distinction is the point: "the scan has no
# blind spots" would be the comfortable claim, and a false one. What is true is that no
# blind spot currently decides anything -- and `test_the_scan_declares_its_blind_spots`
# is what will say so the moment that stops being true.
#
# Format: `"Model.field": "the read site a human found"`. If an entry ever appears, prefer
# making the access resolvable (annotate the parameter) over exempting it.
UNRESOLVED_KEYS: dict[str, str] = {}


def _declared() -> list[tuple[str, str]]:
    return sorted((model, name) for model, names in INDEX.fields.items() for name in names)


def _is_read(model: str, name: str) -> bool:
    """Read AND reachable. A model nothing ever consults does not read itself alive."""
    return (model, name) in REPORT.reads and model in REPORT.reachable


def _is_unresolved(model: str, name: str) -> bool:
    """Is there an access the scan cannot attribute that could have been THIS key?

    Not "is this name unattributable somewhere". `args.device` is an argparse Namespace;
    it is a blind spot about `device` and about nothing else. `config_report` qualifies
    every blind spot by the shape of the base it was on, so this asks about the key.
    """
    return (model, name) in REPORT.unresolved


def _would_be_reported_dead(model: str, name: str) -> bool:
    """The escape clause of the dead-key test, as a predicate: no read, and no excuse."""
    return not _is_read(model, name) and not _is_unresolved(model, name)


def _with_a_planted_dead_key(model: str, name: str) -> Index:
    """The index `src/` would build if someone declared `Model.name` and read it nowhere.

    Planted in the index rather than in `config/common.py` because the index is the whole
    of what a declaration contributes: `scan` takes the field off pydantic and asks no
    other question. The same plant was also made in `config/common.py` by hand, with the
    exemption below filled in to bury it: it failed here by name, and as a spent excuse.
    """
    fields = {owner: set(names) for owner, names in INDEX.fields.items()}
    fields[model].add(name)
    return replace(INDEX, fields=fields)


def test_a_dead_key_is_not_excused_by_a_name_collision_elsewhere() -> None:
    """`RtspSettings.device` is dead. `device` is a name the scan cannot always attribute.

    Those two facts must not add up to a pass. `args.device` in `evaluation/cli.py` is an
    argparse Namespace -- an access this scan cannot type, and so a blind spot about the
    NAME `device`. It is not a blind spot about `RtspSettings`, which that base could
    never have been. If the excuse is keyed on the bare name, one UNRESOLVED_KEYS line
    exempts a key nothing reads and the whole suite goes green over a corpse.
    """
    assert any(note.name == "device" and note.base for note in REPORT.notes), (
        "no unattributable access to the name `device` left in src/ -- this test's "
        "premise is gone; pick another name from REPORT.notes."
    )
    planted = _with_a_planted_dead_key("RtspSettings", "device")
    unresolved = attribute_blind_spots(planted, REPORT.notes, REPORT.untyped_attrs)

    assert ("RtspSettings", "device") not in REPORT.reads, "the plant is meant to be dead"
    assert ("RtspSettings", "device") not in unresolved, (
        "RtspSettings.device is read by nothing, yet the scan calls it unresolved -- so "
        "the dead-key test would skip it and one UNRESOLVED_KEYS line would bury it. It "
        "is excused by an access to the same NAME on a base that could never have been "
        "an RtspSettings. The excuse must name the key, not the name."
    )
    assert ("WorkerGroup", "name") in unresolved, (
        "the planted index resolves nothing at all -- this test would pass on any scan"
    )


def test_a_blind_spot_still_excuses_the_key_the_base_could_have_held() -> None:
    """The other direction, so the rule above is a rule and not just a refusal.

    Qualifying by shape must still let a real blind spot do its job: a base read only
    for `.device`, `.cameras` and `.name` could perfectly well be a `WorkerGroup`, and a
    human must still be allowed to answer for it. A mechanism that excused nothing would
    pass the test above by refusing everything -- and would force the next person to
    delete a key that IS read.
    """
    like_a_group = models_shaped_like(INDEX, {"device", "cameras", "name"})
    assert "WorkerGroup" in like_a_group

    like_an_argparse_namespace = models_shaped_like(INDEX, {"device", "clips", "out"})
    assert like_an_argparse_namespace == frozenset(), (
        f"a base read for .clips and .out is no config model, yet the scan thinks it "
        f"could be {sorted(like_an_argparse_namespace)}"
    )


def _needs_a_human() -> dict[str, list[str]]:
    """The keys whose verdict the scan's blind spot actually changes.

    An unattributed `.close()` on a database connection is a blind spot about the NAME
    `close` -- and `config_scan` will not even record it against `ScoreWeights.close`
    unless that base could have been a `ScoreWeights`. Of the keys it does record, only
    one kind needs an answer: the key that would otherwise be declared dead, where the
    scan cannot honestly say so. That, and only that, is what a human must answer for.
    """
    return {
        f"{model}.{name}": sorted(REPORT.unresolved[(model, name)])
        for model, name in _declared()
        if not _is_read(model, name) and _is_unresolved(model, name)
    }


@pytest.mark.parametrize("model,name", _declared(), ids=lambda v: v)
def test_every_declared_config_key_has_a_consumer(model: str, name: str) -> None:
    """Declared, parsed, validated -- and read by nothing. Wire it up or delete it."""
    if not _would_be_reported_dead(model, name):
        return

    reason = (
        "its model is never reached: no code reads the field that holds it"
        if model not in REPORT.reachable
        else "no code in src/ reads it"
    )
    pytest.fail(
        f"{model}.{name} is declared, parsed, validated, and read by NOTHING in src/ "
        f"-- {reason}. Editing it in YAML does nothing at all, and a tuner who does not "
        f"know that will spend a day on it. Wire it up or delete it (and delete the YAML "
        f"that sets it: extra='forbid' makes a leftover key a startup error). "
        f"If you believe it IS read, the scan could not see the read -- find the site, "
        f"then add it to UNRESOLVED_KEYS with that site as the justification."
    )


def test_the_scan_declares_its_blind_spots() -> None:
    """A key the scan cannot attribute must be accounted for BY A HUMAN, here.

    This is the check that stops the check from rotting. A scan that quietly skipped
    what it could not analyse would look like a dead-key test and be nothing at all --
    which is precisely the disease the whole file exists to treat.
    """
    blind = _needs_a_human()
    unaccounted = sorted(set(blind) - set(UNRESOLVED_KEYS))
    lines = [
        f"  {key}\n" + "\n".join(f"      {site}" for site in blind[key][:4])
        for key in unaccounted
    ]
    assert not unaccounted, (
        "The scan cannot tell whether these config keys are read. It will not guess, and "
        "it will not quietly skip them -- a blind spot nobody is forced to look at is not "
        "a blind spot anybody looks at:\n\n"
        + "\n".join(lines)
        + "\n\nFor each: find the read site by hand. If it IS read, prefer making the "
        "access resolvable (annotate the parameter or the attribute) -- that is a real "
        "fix. Only if you cannot, add it to UNRESOLVED_KEYS with the site you found. "
        "If it is NOT read, delete it."
    )


def test_the_empty_list_is_earned_not_vacuous() -> None:
    """`UNRESOLVED_KEYS` being empty is ambiguous by itself, so pin the other half.

    An empty list is what an honest scan with blind spots that excuse nothing would
    show -- but it is ALSO what a scan finding no blind spots at all would show, and
    those are very different claims. `test_the_scan_declares_its_blind_spots` only
    checks the first half: that no key currently *needs* an entry. It would pass just
    as happily if `_declared()` had zero unresolved keys, which is not what is actually
    true here and not what the module docstring's bullet 1 claims. So assert the
    premise directly: there exists a declared key that IS unresolved (the scan really
    cannot attribute some access to it) and IS ALSO separately proven read (some other
    access the scan could resolve reaches it) -- which is exactly why it needs no
    entry, and why the list can be legitimately empty while unresolved names exist.
    """
    unresolved_and_read = [
        (model, name)
        for model, name in _declared()
        if _is_unresolved(model, name) and _is_read(model, name)
    ]
    assert unresolved_and_read, (
        "no declared key is both unresolved and proven read in this scan -- the "
        "premise behind the empty UNRESOLVED_KEYS list (blind spots exist but excuse "
        "nothing because every one is proven read some other way) no longer holds. "
        "Before trusting that the list is still legitimately empty, re-check bullet 1 "
        "of the module docstring against what `_needs_a_human` actually returns."
    )


def test_no_exemption_outlives_its_reason() -> None:
    """A stale exemption is a dead key wearing a hat. Two ways to go stale, both fatal."""
    gone = sorted(k for k in UNRESOLVED_KEYS if not _exists(k))
    assert not gone, (
        f"UNRESOLVED_KEYS names {', '.join(gone)}, which no config model declares any "
        "more. The exemption outlived the key. Delete the entry."
    )
    needed = _needs_a_human()
    spare = sorted(set(UNRESOLVED_KEYS) - set(needed))
    assert not spare, (
        f"UNRESOLVED_KEYS still excuses {', '.join(spare)}, but the excuse is spent: the "
        "scan can now prove those keys read, or can now see they are not. Delete the "
        "entry, so the key goes back under the dead-key test where it belongs. An "
        "exemption nobody rechecks is how an allow-list becomes a rubber stamp."
    )


def _exists(key: str) -> bool:
    model, _, name = key.rpartition(".")
    return name in INDEX.fields.get(model, ())


def test_no_access_is_too_dynamic_to_analyse() -> None:
    """`getattr(x, name)` on an untyped base, or `**config` -- nothing can be concluded.

    One of these anywhere in `src/` and the scan is no longer sound about ANY key, so it
    is a hard failure rather than a per-key exemption: there is no key to hang it on.
    """
    assert not REPORT.dynamic, (
        "src/ contains attribute access the scan cannot reason about at all:\n\n"
        + "\n".join(f"  {site}" for site in sorted(REPORT.dynamic))
        + "\n\nWhile this exists, the dead-key test cannot promise a key is unread -- the "
        "dynamic access might read it. Type the base (annotate it), or use a literal "
        "attribute name."
    )


def test_the_scan_actually_resolved_something() -> None:
    """A scan that resolved nothing would pass every test above by vacuity.

    If `src/` were unparseable, or the model discovery silently returned nothing, every
    other test here goes green while checking exactly nothing. This is the tripwire.
    """
    assert len(INDEX.models) > 20, f"only {len(INDEX.models)} config models found"
    assert len(REPORT.reads) > 100, f"only {len(REPORT.reads)} reads resolved -- scan broken?"
    assert INDEX.ty_of_annotation(None, {}) is UNKNOWN

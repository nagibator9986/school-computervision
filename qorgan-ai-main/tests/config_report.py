"""What the scan found -- and, where it could not tell, WHOSE key it could not tell about.

`config_scan` walks `src/` and proves reads. This module holds its answer, and does the
one step that makes the answer safe to build an allow-list on.

That step is worth its own module because it was got wrong once, in the obvious way. An
access the scan cannot attribute -- `args.device`, where `args` is an argparse Namespace
it cannot type -- is a blind spot about the NAME `device`. Recorded against the name, it
excuses every field called `device` in the fleet: a genuinely dead `RtspSettings.device`
becomes exempt-able by one line in `UNRESOLVED_KEYS`, on the strength of an access that
had nothing to do with it, and the dead-key test goes green over a corpse. A check that
looks exactly like a check and does nothing is this codebase's signature bug; the
dead-key test is the thing built to catch it, so it does not get to have it.

So a blind spot is qualified before anyone may act on it: `models_shaped_like` asks which
config models the base could have been, given EVERY attribute anyone reads off it. `args`
is also read for `.clips` and `.out`, which no config declares -- so it is not a config,
and it excuses nothing. What survives is keyed on `(model, field)`: a blind spot about a
KEY, which an exemption can honestly name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.config_index import Index


@dataclass(frozen=True)
class Note:
    """One access the scan could not attribute. A blind spot, not yet qualified.

    `base` is the source text of the expression it was read off (`args`, `self._cfg`),
    which is all the scan knows about it -- and, it turns out, enough. `owners` is set
    only in the other case: the base WAS typed and the NAME was not (`getattr(cfg, x)`),
    where the owning model is already proven and no shape needs guessing.
    """

    name: str
    site: str
    base: str | None = None
    owners: frozenset[str] | None = None


@dataclass
class Report:
    """What the scan found. `unresolved` and `dynamic` are its own blind spots."""

    reads: set[tuple[str, str]] = field(default_factory=set)
    # (model, field) -> {"file:line why"}: accesses the scan could not attribute AND that
    # could, by the shape of the base, have been a read OF THIS KEY. Keyed on the key,
    # never on the bare name -- that distinction is this module's whole reason to exist.
    unresolved: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    # accesses so dynamic that nothing can be concluded about ANY key
    dynamic: set[str] = field(default_factory=set)
    reachable: set[str] = field(default_factory=set)
    # -- raw material for `attribute_blind_spots`, which fills `unresolved` --
    notes: list[Note] = field(default_factory=list)
    # untyped base expression -> every attribute name read off it anywhere in src/
    untyped_attrs: dict[str, set[str]] = field(default_factory=dict)


def models_shaped_like(index: Index, attrs: set[str]) -> frozenset[str]:
    """Which config models could a base be, given EVERY attribute read off it in src/?

    A model the base cannot be is a model whose key the base cannot excuse. `args` in
    `evaluation/cli.py` is read for `.device` -- but also for `.clips` and `.out`, which
    no config model declares. So it is an argparse Namespace, and it says nothing at all
    about anybody's `device`.

    Conservative in the safe direction: a base whose every attribute some model happens
    to have stays a candidate, and that key stays excusable by a human who found the read.
    Only a base that provably could NOT be the model loses the power to excuse its keys.
    """
    return frozenset(
        name
        for name, model in index.models.items()
        if attrs <= index.fields[name] | set(dir(model))
    )


def attribute_blind_spots(
    index: Index, notes: list[Note], untyped_attrs: dict[str, set[str]]
) -> dict[tuple[str, str], set[str]]:
    """Turn blind spots about a NAME into blind spots about a KEY.

    This is the step that stops the allow-list from being able to hide a corpse. An
    unattributable `.device` is not an excuse for every field called `device` in the
    fleet -- only for those on a model the base could actually have held.
    """
    out: dict[tuple[str, str], set[str]] = {}
    for note in notes:
        owners = note.owners
        if owners is None:
            owners = models_shaped_like(index, untyped_attrs.get(note.base or "", set()))
        for model in owners:
            if note.name in index.fields.get(model, ()):
                out.setdefault((model, note.name), set()).add(note.site)
    return out

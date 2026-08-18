"""§12.1: a second detection session that finds an object and then asks a person about it.

Read `qorgan/weapons/weights.py` first. It is the module this package exists for.

**The client has no weapons model.** Their `best.pt` is a 0-byte file and was meant to be
a violence *classifier* rather than a weapons *detector*, so the module they watched
working for months had no model in it on any day. The code fell back to motion analysis,
logged a warning, and reported healthy. Everything the school observed as "the bullying
module working" was a motion detector.

Nothing here has a fallback, and that is the design:

  * **No weights, no pipeline.** `model.YoloWeaponModel.__init__` refuses and names the
    file. An empty file is not weights and neither is a classifier -- both are checked,
    because "the path exists" is what the previous system checked. The refusal is in the
    CONSTRUCTOR rather than in a `load_weapon_model()` beside it, and that is the whole
    reason there is no such function: a separate loader is a thing a caller can decline to
    call, and the object it would return has to exist in an unloaded state for the caller
    to hold. Here there is no half-built object to hold. Either the weights loaded and you
    have a `YoloWeaponModel`, or `WeaponWeightsUnusable` was raised and you have nothing.
  * **The panel says what is running.** Which file, how big, its fingerprint, its classes,
    and what a human says it was evaluated on. A module that cannot say what it runs is a
    module nobody can audit.
  * **Nothing is auto-actioned.** An alert is a question; the answer is a person's name on
    the row (`weapons/store.py`).

The layering is the same as the bullying tier's and for the same reason (rule R2): the
DECISION is pure and lives in `pipeline.py`, `rules.py` and `tracking.py`, so it can be
tested without a GPU and without weights -- which is the only kind of test available to
us, since we have no weights. `model.py` is the only part that needs a card, and
`worker/weapons.py` is the only part that touches disk, the database or the network.
"""

from qorgan.weapons.pipeline import FrameOutcome, WeaponAlert, WeaponsDetector
from qorgan.weapons.weights import LoadedWeights, WeaponWeightsUnusable

__all__ = [
    "FrameOutcome",
    "LoadedWeights",
    "WeaponAlert",
    "WeaponWeightsUnusable",
    "WeaponsDetector",
]

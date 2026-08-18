"""What a block of the cabinet actually is TODAY, said in the block itself.

The old system's fatal property was not a missing feature: it was a page that looked like
it was working. `SignalState` exists so that no surface in this package can render a
number without also rendering what kind of number it is, and so that "there is nothing
here yet" and "there will never be anything here about a named child" cannot be shown the
same way — they are different answers and only one of them is waiting on a camera.

The labels live here and not in the template for the reason `web/routes/events.py` maps
its skip reasons in Python: a template that decides anything is a template nobody tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SignalState(StrEnum):
    """Three states, because there are three genuinely different situations.

    LIVE and EMPTY differ by whether rows have arrived. ANONYMOUS is not a third point on
    that line at all: it says the rows ARE arriving and still cannot become a statement
    about a named child, ever, because they carry no identity and `qorgan.classroom` may
    never give them one. Collapsing it into EMPTY would promise the school that waiting
    fixes it. Collapsing it into LIVE would be worse.
    """

    LIVE = "live"
    EMPTY = "empty"
    ANONYMOUS = "anonymous"


SIGNAL_LABELS: dict[SignalState, str] = {
    SignalState.LIVE: "сигнал живой",
    SignalState.EMPTY: "пока пусто",
    SignalState.ANONYMOUS: "накапливается, но без личности",
}


def signal_label(state: SignalState) -> str:
    """Missing entries degrade to the token rather than raising: a KeyError on the page
    whose job is to say what is and is not working would replace the answer with a 500."""
    return SIGNAL_LABELS.get(state, state.value)


@dataclass(frozen=True, slots=True)
class Block:
    """One section of the cabinet index: a count, a state, and the sentences it owes.

    `lines` is never empty. A block with a number and no explanation is the artefact this
    class was written to make impossible.
    """

    key: str
    title: str
    state: SignalState
    count: int
    lines: tuple[str, ...]

    @property
    def label(self) -> str:
        return signal_label(self.state)

"""The observable-activity index: a sum of things a camera saw, shown as its parts.

--------------------------------------------------------------------------------
**WHY IT IS NOT CALLED ENGAGEMENT.**

The client asked whether a pupil is «вовлечён в урок». That is not visible. A pupil
looking straight at the board may be elsewhere entirely and one staring at the desk may
be concentrating hard, and no camera resolves the difference — which is also why there is
no expression classifier anywhere in this package and no place to add one. What a camera
does witness is where a body is, which way a head points, and whether an arm goes up. This
index is a weighted sum of exactly those, and its name says so.

The distinction is not pedantry, it is what keeps the number safe to put in front of a
psychologist. «Индекс вовлечённости 42» invites a conclusion about a child. «Индекс
наблюдаемой активности 42, из них: голова поднята 71 %, лицом вперёд 88 %, на месте
96 %, рука поднята 1 раз» invites a question, which is the correct response.

--------------------------------------------------------------------------------
**THE INDEX IS ALWAYS RENDERED WITH ITS COMPONENTS. `Activity.parts` exists so that a
surface CANNOT show the total alone** — every component, its raw value, its weight and
its contribution travel together. A single number that cannot be taken apart is a verdict
wearing the costume of a measurement.

**Every weight below is CHOSEN, not fitted.** There is no labelled data to fit them to,
and pretending otherwise would be the worst failure available here. They are declared, they
are written into the artefact, and a change to them makes a new artefact rather than a new
truth about a child. The one thing they are chosen to satisfy is that no single component
can dominate: posture shares are bounded in [0, 1] and the event term is capped, so a
pupil cannot score highly by fidgeting or badly by sitting still.

**The index is refused, not estimated, when we did not see enough.** Below
`MIN_COVERAGE` the seat's totals are a sample of an unknown fraction of the lesson, and an
index computed on them would be a confident number about a pupil we mostly could not see.
`Activity.available` is False and every surface must say «недостаточно наблюдений».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from classvision.states import PupilState

# Below this fraction of analysed frames actually containing this pupil, no index is
# produced. CHOSEN. Half the lesson is the point at which "we watched and this is what
# happened" stops being a fair description of the number.
MIN_COVERAGE = 0.50

# How many participation events (hand up, stood, went to the board) count as a "full"
# lesson's worth for the event term. CHOSEN, and deliberately low, because the measured
# reality of this recording is that genuine hand-raises are RARE -- roughly ten pupil
# observations across 52 minutes once the adult was excluded (`MEASUREMENTS.md` §5). A
# normaliser of 10 would push every pupil's event term to nearly zero and make the term
# dead weight; a normaliser of 3 lets a single genuine raise register.
EVENTS_FOR_FULL_CREDIT = 3.0

WEIGHTS: dict[str, float] = {
    # Head up rather than down on the desk. The largest weight because it is the most
    # directly observable and the most robust of the four: it needs only that we can see
    # a head, and it is measured against this pupil's OWN upright posture.
    "head_up_share": 0.30,
    # Facing the way this pupil habitually faces, rather than turned away from the front
    # of the room. Measured against their own habitual direction, never an absolute one --
    # 61 % of all observations here read as one particular profile purely because of where
    # the camera hangs.
    "facing_front_share": 0.25,
    # At their own place rather than away from it. Weighted lowest of the three shares
    # because being out of one's place is often exactly what a lesson requires.
    "at_place_share": 0.15,
    # Participation events, capped. The only term that can be zero for a pupil who did
    # nothing visible, which is why it is a minority of the total: a quiet pupil who sits
    # up and faces the board is not a disengaged pupil, and must not score near zero.
    "participation": 0.30,
}


@dataclass(frozen=True, slots=True)
class Part:
    """One component of the index, with everything needed to argue with it."""

    key: str
    value: float          # 0..1
    weight: float
    raw: Any              # what it was computed from, in its own units
    label_ru: str

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass(frozen=True, slots=True)
class Activity:
    """One pupil-place's activity for one lesson, decomposed."""

    available: bool
    index: float | None            # 0..100
    coverage: float
    parts: tuple[Part, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "index": None if self.index is None else round(self.index, 1),
            "coverage": round(self.coverage, 3),
            "reason": self.reason,
            "parts": [
                {"key": p.key, "label_ru": p.label_ru, "value": round(p.value, 3),
                 "weight": p.weight, "raw": p.raw,
                 "contribution": round(p.contribution * 100, 1)}
                for p in self.parts
            ],
        }


def activity(ledger) -> Activity:
    """Compute the index for one `SeatLedger`, or refuse and say why.

    The shares are computed over MEASURED observations only — the denominator excludes
    frames in which this pupil was not visible. Including them would silently score a
    frequently-occluded pupil as though they had spent that time slumped, which is the
    single easiest way to manufacture a "decline" out of a pupil moving one seat further
    behind a taller neighbour.
    """
    coverage = ledger.coverage
    if coverage < MIN_COVERAGE:
        return Activity(available=False, index=None, coverage=coverage,
                        reason=f"наблюдений слишком мало: {coverage:.0%} кадров "
                               f"(требуется {MIN_COVERAGE:.0%})")
    if not ledger.baselines.settled:
        return Activity(available=False, index=None, coverage=coverage,
                        reason="базовая поза не установлена — не с чем сравнивать")

    measured = sum(count for state, count in ledger.state_observations.items()
                   if state != PupilState.UNKNOWN.value)
    if measured == 0:
        return Activity(available=False, index=None, coverage=coverage,
                        reason="ни одного читаемого наблюдения")

    def share_not(*states: PupilState) -> float:
        bad = sum(ledger.state_observations.get(s.value, 0) for s in states)
        return max(0.0, 1.0 - bad / measured)

    events = (ledger.count(PupilState.HAND_RAISED)
              + ledger.count(PupilState.STOOD_UP)
              + ledger.count(PupilState.AT_BOARD))

    parts = (
        Part("head_up_share", share_not(PupilState.HEAD_DOWN), WEIGHTS["head_up_share"],
             {"head_down_observations": ledger.state_observations.get(PupilState.HEAD_DOWN.value, 0),
              "measured_observations": measured},
             "голова поднята (не лежит на парте)"),
        Part("facing_front_share", share_not(PupilState.TURNED_AWAY),
             WEIGHTS["facing_front_share"],
             {"turned_away_observations": ledger.state_observations.get(PupilState.TURNED_AWAY.value, 0),
              "measured_observations": measured},
             "смотрит вперёд (не отвернулся назад)"),
        Part("at_place_share", share_not(PupilState.AWAY_FROM_PLACE),
             WEIGHTS["at_place_share"],
             {"away_observations": ledger.state_observations.get(PupilState.AWAY_FROM_PLACE.value, 0),
              "measured_observations": measured},
             "находится на своём месте"),
        Part("participation", min(1.0, events / EVENTS_FOR_FULL_CREDIT),
             WEIGHTS["participation"],
             {"hand_raises": ledger.count(PupilState.HAND_RAISED),
              "stands": ledger.count(PupilState.STOOD_UP),
              "board_visits": ledger.count(PupilState.AT_BOARD),
              "normaliser": EVENTS_FOR_FULL_CREDIT},
             "видимые действия: рука, вставание, выход к доске"),
    )

    index = 100.0 * sum(p.contribution for p in parts)
    return Activity(available=True, index=index, coverage=coverage, parts=parts)

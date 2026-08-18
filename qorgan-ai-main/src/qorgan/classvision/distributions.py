"""The empirical shapes a demonstration is built from, and the index recomputed from them.

Split out of `demo.py` because it is the half that must be argued about: every distribution
here is READ OFF the real recordings rather than chosen, and the one formula that is COPIED
from the analyser (`classvision/metrics/activity.py`) lives beside the measurement that keeps
the copy honest.

**Why the formula is copied at all.** `qorgan` may not import `classvision`: that separation
keeps torch out of the web process and the AGPL obligation off the network-served application
(`INTEGRATION.md` §1, §9). So four weights, one normaliser and one coverage floor exist twice
in this repository, which is precisely the "second, quietly different implementation" defect
this project keeps paying for (`MEASUREMENTS.md` §8).

**The mitigation is a measurement, not a promise.** `verify_against` recomputes every real
place's index from that artefact's OWN raw observation counts and compares it with the figure
the artefact published. `demo.generate` runs it over every source before it writes a single
row, and refuses to build if any place disagrees by more than one rounding step. A divergence
between the two implementations therefore stops a demonstration from existing, instead of
quietly producing numbers that look like the analyser's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DemoRefused(Exception):
    """Nothing was written, and this says what was missing or what disagreed."""


# The analyser's own constants (`classvision/metrics/activity.py`). Copied, not chosen — see
# the module docstring and `verify_against`, which is what keeps the copy honest.
WEIGHTS = {
    "head_up_share": 0.30,
    "facing_front_share": 0.25,
    "at_place_share": 0.15,
    "participation": 0.30,
}
EVENTS_FOR_FULL_CREDIT = 3.0
MIN_COVERAGE = 0.50
LABELS_RU = {
    "head_up_share": "голова поднята (не лежит на парте)",
    "facing_front_share": "смотрит вперёд (не отвернулся назад)",
    "at_place_share": "находится на своём месте",
    "participation": "видимые действия: рука, вставание, выход к доске",
}

@dataclass(slots=True)
class Profile:
    """The empirical shapes taken from the real artefacts. No constant in here was chosen."""

    coverages: list[float] = field(default_factory=list)
    head_down: list[float] = field(default_factory=list)
    turned_away: list[float] = field(default_factory=list)
    away: list[float] = field(default_factory=list)
    hand_raises: list[int] = field(default_factory=list)
    stands: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    sample_fps: list[float] = field(default_factory=list)
    geometry: list[tuple[float, float, float]] = field(default_factory=list)
    adult: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    unmeasured: list[dict] = field(default_factory=list)
    room_layout: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def read_profile(paths: list[Path]) -> Profile:
    """Pool the real recordings' distributions. Refuses rather than inventing a default."""
    if not paths:
        raise DemoRefused(
            "не указан ни один --from. Формы для демонстрации берутся из настоящих артефактов "
            "(распределения покрытия, поз и событий), а не из придуманных констант: демо, "
            "построенное на круглых числах, не похоже на эту школу и вводит в заблуждение."
        )
    profile = Profile()
    widest = 0
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        pupils = [s for s in document.get("seats") or [] if s.get("role") == "pupil"]
        if not pupils:
            raise DemoRefused(f"{path}: в артефакте нет ученических мест")
        _absorb(profile, document, pupils)
        profile.sources.append(f"{Path(path).name} ({len(pupils)} мест)")
        if len(pupils) > widest:
            widest = len(pupils)
            profile.geometry = [
                (float(s["centre"][0]), float(s["centre"][1]), float(s["scale_px"]))
                for s in sorted(pupils, key=lambda s: int(s["seat_id"]))
            ]
            profile.thresholds = document["provenance"].get("thresholds") or {}
            profile.room_layout = (document["provenance"].get("room") or {}).get("layout") or {}
            profile.caveats = list(document.get("caveats") or [])
            profile.unmeasured = list((document.get("lesson") or {}).get("unmeasured") or [])
            profile.adult = _adult_shape(document)
    return profile


def _absorb(profile: Profile, document: dict[str, Any], pupils: list[dict[str, Any]]) -> None:
    """One artefact's per-place component shares, straight off its own activity parts."""
    profile.durations.append(float((document.get("lesson") or {}).get("duration_minutes") or 0.0))
    profile.sample_fps.append(float(document["provenance"].get("sample_fps") or 2.0))
    for seat in pupils:
        ledger = seat.get("ledger") or {}
        counts = ledger.get("counts") or {}
        parts = {p["key"]: p for p in ((seat.get("metrics") or {}).get("activity") or {}).get(
            "parts", [])}
        profile.coverages.append(float(ledger.get("coverage") or 0.0))
        profile.hand_raises.append(int(counts.get("hand_raises") or 0))
        profile.stands.append(int(counts.get("stands") or 0))
        for key, bucket in (("head_up_share", profile.head_down),
                            ("facing_front_share", profile.turned_away),
                            ("at_place_share", profile.away)):
            if key in parts:
                bucket.append(round(1.0 - float(parts[key]["value"]), 4))


def _adult_shape(document: dict[str, Any]) -> dict[str, Any]:
    """The adult's POSE metrics only, and the geometry of his place. No follower here."""
    teacher = document.get("teacher") or {}
    metrics = {k: v for k, v in (teacher.get("metrics") or {}).items() if k != "presence"}
    centre = teacher.get("centre") or [0.0, 0.0]
    return {
        "metrics": metrics,
        "centre": (float(centre[0]), float(centre[1])),
        "scale_px": float(teacher.get("scale_px") or 0.0),
        "coverage": metrics.get("coverage"),
    }


def activity(histogram: dict[str, int], counts: dict[str, int], coverage: float) -> dict[str, Any]:
    """The analyser's index, recomputed here from an observation histogram.

    Same four components, same weights, same normaliser, same refusal below `MIN_COVERAGE`
    (where the index is None and the reason is the sentence, never a zero). Shares are over
    MEASURED observations, excluding `unknown`: including the unseen ones would score a
    frequently-occluded place as though it had spent that time slumped, which is the easiest
    possible way to manufacture a decline.
    """
    measured = sum(v for k, v in histogram.items() if k != "unknown")
    if coverage < MIN_COVERAGE or measured == 0:
        return {"available": False, "index": None, "coverage": round(coverage, 3),
                "reason": f"наблюдений слишком мало: {coverage:.0%} кадров "
                          f"(требуется {MIN_COVERAGE:.0%})", "parts": []}
    events = counts.get("hand_raises", 0) + counts.get("stands", 0) + counts.get("board_visits", 0)
    values = {
        "head_up_share": max(0.0, 1.0 - histogram.get("head_down", 0) / measured),
        "facing_front_share": max(0.0, 1.0 - histogram.get("turned_away", 0) / measured),
        "at_place_share": max(0.0, 1.0 - histogram.get("away_from_place", 0) / measured),
        "participation": min(1.0, events / EVENTS_FOR_FULL_CREDIT),
    }
    raw = {
        "head_up_share": {"head_down_observations": histogram.get("head_down", 0),
                          "measured_observations": measured},
        "facing_front_share": {"turned_away_observations": histogram.get("turned_away", 0),
                               "measured_observations": measured},
        "at_place_share": {"away_observations": histogram.get("away_from_place", 0),
                           "measured_observations": measured},
        "participation": {"hand_raises": counts.get("hand_raises", 0),
                          "stands": counts.get("stands", 0),
                          "board_visits": counts.get("board_visits", 0),
                          "normaliser": EVENTS_FOR_FULL_CREDIT},
    }
    parts = [{"key": key, "label_ru": LABELS_RU[key], "value": round(values[key], 3),
              "weight": WEIGHTS[key], "raw": raw[key],
              "contribution": round(values[key] * WEIGHTS[key] * 100, 1)} for key in WEIGHTS]
    index = 100.0 * sum(values[k] * WEIGHTS[k] for k in WEIGHTS)
    return {"available": True, "index": round(index, 1), "coverage": round(coverage, 3),
            "reason": "", "parts": parts}


def verify_against(path: Path, tolerance: float = 0.15) -> list[str]:
    """Recompute every real place's index from the artefact's OWN parts, and compare.

    This is the check that keeps a copied formula from becoming a second, quietly different
    definition of the index — the `MEASUREMENTS.md` §8 defect one layer up. It uses the
    artefact's own raw observation counts, so any disagreement is in the arithmetic and
    nowhere else. Tolerance is one rounding step on a 0.1-resolution figure.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    for seat in document.get("seats") or []:
        theirs = (seat.get("metrics") or {}).get("activity") or {}
        if not theirs.get("available"):
            continue
        parts = {p["key"]: p for p in theirs["parts"]}
        measured = int(parts["head_up_share"]["raw"]["measured_observations"])
        histogram = {
            "head_down": int(parts["head_up_share"]["raw"]["head_down_observations"]),
            "turned_away": int(parts["facing_front_share"]["raw"]["turned_away_observations"]),
            "away_from_place": int(parts["at_place_share"]["raw"]["away_observations"]),
            "seated": measured,
        }
        histogram["seated"] = measured - sum(
            v for k, v in histogram.items() if k != "seated")
        mine = activity(histogram, parts["participation"]["raw"],
                        float(seat["ledger"]["coverage"]))
        if abs(float(mine["index"]) - float(theirs["index"])) > tolerance:
            problems.append(
                f"{Path(path).name} {seat['label']}: артефакт {theirs['index']}, "
                f"пересчёт {mine['index']}")
    return problems

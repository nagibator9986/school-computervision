"""Rendering the gallery report for a human at a terminal.

Split out of `report.py` so the analysis stays pure and testable and the prose stays out
of its way. Every number printed here arrives with the measurement that produced it --
that is the whole rule, and guessing is what cost the legacy eighteen thresholds and
1 816 NULL canteen records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qorgan.config.identity import RecognitionPolicy
from qorgan.identity.report import DUPLICATE_SIMILARITY, HIST_LOW

if TYPE_CHECKING:
    from qorgan.identity.report import GalleryReport

BAR = 60  # widest bar in the histogram, in characters

# The measured band is empty from 0.48 to 0.77 (spec §2). These are the histogram buckets
# that fall wholly INSIDE it -- the ones whose emptiness is the finding rather than a
# coincidence of where a bucket edge happened to land. Annotate only those.
EMPTY_BAND_LOW, EMPTY_BAND_HIGH = 0.50, 0.75


def render(report: GalleryReport) -> str:
    return "\n".join(
        [
            f"{report.people} people, {report.pairs} distinct cross-person pair(s).",
            "",
            *_histogram_lines(report),
            "",
            *_impostor_lines(report),
            "",
            *_duplicate_lines(report),
            "",
            *_extrapolation_lines(report),
            "",
            *_unenrolled_lines(report),
        ]
    )


def _histogram_lines(report: GalleryReport) -> list[str]:
    peak = max((bucket.count for bucket in report.histogram), default=0)
    lines = [
        f"cross-person similarity — {report.people} people, {report.pairs} distinct pairs",
        "",
        f"  below {HIST_LOW:.2f}    {report.below_histogram:>5}",
    ]
    for bucket in report.histogram:
        bar = "#" * round(BAR * bucket.count / peak) if peak else ""
        # The HOLE is the evidence. Say so where it is, or a reader sees only blank rows.
        note = (
            "   <-- and not one pair lands in this band"
            if bucket.count == 0
            and bucket.low >= EMPTY_BAND_LOW
            and bucket.high <= EMPTY_BAND_HIGH
            else ""
        )
        lines.append(f"  [{bucket.low:.2f},{bucket.high:.2f})  {bucket.count:>5}  {bar}{note}")
    return lines


def _impostor_lines(report: GalleryReport) -> list[str]:
    policy = RecognitionPolicy()
    return [
        f"genuine impostors (pairs below {DUPLICATE_SIMILARITY:.2f}): {report.impostor_pairs}",
        f"  p50 {report.impostor_p50:.3f}   p90 {report.impostor_p90:.3f}   "
        f"p99 {report.impostor_p99:.3f}   max {report.impostor_max:.3f}",
        f"  pairs >= {report.gate:.2f}: {report.impostor_above_gate}",
        "",
        f"  MEASURED FLOOR under min_score: {report.impostor_max:.3f} — anything at or "
        "below that admits a known confusion.",
        f"  min_score is {policy.min_score:.2f}. The ceiling — whether a real camera face "
        "can reach it at all — is NOT measured, because every score above is a gallery "
        "photo against a gallery photo. This is a floor, not a tuned value. Get "
        "canteen-entry footage of a pupil we can name.",
    ]


def _duplicate_lines(report: GalleryReport) -> list[str]:
    if not report.duplicates:
        return ["duplicate enrolments: none. Every person holds one id."]

    lines = [
        f"DUPLICATE ENROLMENTS: {len(report.duplicates)} pair(s) of DIFFERENT ids whose "
        f"faces are the same person (>= {DUPLICATE_SIMILARITY:.2f}).",
        "",
        "  Their meals split across both ids, so the canteen record is already wrong for "
        "them — and the duplicate sits in top-2 and kills the gap, so the system is not "
        "inaccurate about these people. It is BLIND to them.",
        "",
        "  Detection is not resolution. Which id is canonical is a decision only the "
        "school can make, and adjacent ids in one class may be identical twins. Nothing "
        "is merged automatically. Run: qorgan pupils merge <keep_id> <drop_id>",
        "",
    ]
    lines.extend(
        f"  {pair.similarity:.3f}  {pair.external_a:<14} ({pair.display_a})"
        f"  <->  {pair.external_b:<14} ({pair.display_b})"
        for pair in report.duplicates
    )
    return lines


def _extrapolation_lines(report: GalleryReport) -> list[str]:
    lines = [
        f"P(two different children >= {report.gate:.2f}) = "
        f"{report.impostor_probability:.2e}. A school of S gives each child S-1 "
        "impostors: 1 - (1-p)^(S-1).",
        "",
        "  school   risk per child   children affected",
    ]
    lines.extend(
        f"  {risk.size:>6}   {risk.risk_per_child * 100:>13.1f} %   "
        f"{risk.children_affected:>17.0f}"
        for risk in report.extrapolation
    )
    lines.append("")
    lines.append(
        "  This is the argument for a SECOND photo per child. Put it in the questions to "
        "the school."
    )
    return lines


def _unenrolled_lines(report: GalleryReport) -> list[str]:
    if not report.unenrolled:
        return ["every roster photo produced exactly one face."]

    lines = [
        f"{len(report.unenrolled)} photo(s) could NOT be enrolled. These people are on the "
        "roster and the system can never recognise them:",
        "",
    ]
    lines.extend(
        f"  {item.external_id:<14} ({item.display:<20}) {item.reason} — {item.photo}"
        for item in report.unenrolled
    )
    return lines

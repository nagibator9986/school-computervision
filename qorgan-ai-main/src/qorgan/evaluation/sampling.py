"""Pass 2 of labelling: the clips that measure what the detector MISSED.

`eval scan` fires the detector at threshold 0 over the corpus and `eval label` lets a human
judge what it fired on. That yields **precision** -- how often we cry wolf -- and precision
is half a detector. It cannot tell you what you threw away.

The scan has now run over all 657 clips, and its output says the corpus is not one
population but three:

    51  alerts               confidence >= notify_threshold (0.85)
    72  skeleton-SUPPRESSED  confidence held at cap_without_skeleton (0.72)
    22  below the cap        the fast tier fired, but under the cap
   517  silent               the fast tier never fired at all

A single "random sample of the clips it did not fire on" collapses two completely different
failure modes into one bucket, and the one it hides is the one that matters:

  A. **alerts** -- a label here measures PRECISION. A false alarm annoys a teacher.

  B. **skeleton-suppressed** -- a label here asks: *did the skeleton veto a real fight?*
     The mandatory-skeleton rule is holding back HALF of everything the fast tier
     proposes. That rule is the system's central safety property -- an alert requires two
     independent tiers to agree -- and these 72 rows are it doing measurable work. But if
     it is also discarding real fights, that is a child nobody helped, and it is completely
     invisible to a precision-only analysis. Nobody would have sampled this stratum, and it
     is the most dangerous one in the corpus.

  C. **silent** -- a label here asks: *did the fast tier miss a fight?* But silence is
     PROVEN, never inferred from absence: a clip with no row in candidates.csv is silent only
     if `eval scan`'s coverage manifest proves the detector ran on it. A clip the scan never
     covered is UNKNOWN, not silent, and a hard error -- because a real fight in an unscanned
     clip, filed as silent, would never be sampled (`_prove_coverage`, `scan.load_coverage`).

A candidate above the cap but under the alert threshold is a NEAR_MISS -- skeleton confirmed,
just not loud enough -- and one at or below the cap is a BELOW_CAP. Both sit nearest the
decision boundary, which is exactly the evidence an operating point is chosen with, so both
are DRAWN whole rather than held back and merely logged.

So: all of the alerts, near-misses, skeleton-suppressed and below-cap candidates (they are
small and they are what decides the boundary), plus a seeded random sample of the silent
clips. Every row records WHICH stratum it came from, because an unweighted stratified sample
is a biased estimate wearing a lab coat -- without the stratum nobody can weight the answer
back to the 657 clips it was drawn from.

**This sample decides nothing.** It proposes clips for a human to watch. It never guesses a
label and it never guesses a timestamp: a silent clip has no timestamp, because the
detector named no moment in it, and `ScanRow.timestamp` is `None` there rather than a
midpoint somebody made up (see `scan.ScanRow` and `labelling.interval_for`).
"""

from __future__ import annotations

import csv
import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass
from enum import StrEnum
from pathlib import Path

from qorgan.config.bullying import Confidence
from qorgan.config.camera import BullyingCamera
from qorgan.evaluation.labelling import ResumeKey, is_done, settled_key
from qorgan.evaluation.labels import LabelKind, LabelSet
from qorgan.evaluation.scan import SCAN_COLUMNS, ScanRow, order_key
from qorgan.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_SILENT_SAMPLE = 80
DEFAULT_SEED = 7

# The worklist is a candidates file with one more column, so `eval label` reads it with
# `scan.load_candidates` unchanged -- that reader takes its columns by NAME and ignores
# the extra one.
SAMPLE_COLUMNS = (*SCAN_COLUMNS, "stratum")

# candidates.csv stores confidence rounded to 3 dp (`scan.rows_for`), so "held at exactly
# the cap" is compared at the precision the file actually carries.
CONFIDENCE_DECIMALS = 3


class Stratum(StrEnum):
    """Which population a row was drawn from, and therefore what a label on it measures."""

    ALERT = "alert"
    NEAR_MISS = "near_miss"
    SKELETON_SUPPRESSED = "skeleton_suppressed"
    BELOW_CAP = "below_cap"
    SILENT = "silent"


# Every stratum with a candidate on the clip is drawn; only SILENT is sampled down. NEAR_MISS
# and BELOW_CAP used to be held back and only logged -- but they are the candidates nearest
# the decision boundary, which is exactly the evidence an operating point is chosen with.
DRAWN_STRATA = (
    Stratum.ALERT,
    Stratum.NEAR_MISS,
    Stratum.SKELETON_SUPPRESSED,
    Stratum.BELOW_CAP,
    Stratum.SILENT,
)


class SampleError(Exception):
    """The sample cannot be drawn from this corpus. Always names the clip."""


@dataclass(frozen=True, slots=True)
class SampleRow:
    """One thing for a human to watch, and the stratum it stands for."""

    row: ScanRow
    stratum: Stratum


def classify(confidence: float, config: Confidence) -> Stratum:
    """Which stratum a candidate's confidence puts it in.

    The boundaries are the CAMERA's config, never a constant: `notify_threshold` and
    `cap_without_skeleton` are per-camera, and a number baked in here would quietly
    misclassify the whole corpus the day one camera is retuned.

    A caveat worth stating, because it is load-bearing: candidates.csv records no `capped`
    flag, so "the skeleton vetoed this" is inferred from the confidence sitting exactly on
    the cap. That is what the clamp does (`detection.validation`) and it is how the 72 were
    counted. A verdict that landed on the cap by arithmetic coincidence would be read as a
    veto -- an unlikely mislabel, and one that puts an extra clip in front of a human
    rather than hiding one from them.
    """
    if confidence >= config.notify_threshold:
        return Stratum.ALERT
    if round(confidence, CONFIDENCE_DECIMALS) == round(
        config.cap_without_skeleton, CONFIDENCE_DECIMALS
    ):
        return Stratum.SKELETON_SUPPRESSED
    if confidence > config.cap_without_skeleton:
        # Skeleton confirmed (above the cap), but under the alert threshold. NOT below the
        # cap at all -- a near miss, and filing it as `below_cap` was a residual bucket lying.
        return Stratum.NEAR_MISS
    return Stratum.BELOW_CAP


def draw(
    *,
    clips: Sequence[str],
    candidates: Sequence[ScanRow],
    cameras: Mapping[str, BullyingCamera],
    coverage: Collection[str] | None,
    labelled: LabelSet | None = None,
    count: int = DEFAULT_SILENT_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> list[SampleRow]:
    """The worklist: every candidate stratum whole, and `count` of the silent clips, `seed`.

    `clips` is every clip in eval/clips/; `candidates` is eval/candidates.csv; `cameras`
    pairs each clip with the camera whose thresholds it is judged against. `coverage` is the
    scan's coverage manifest -- the set of clips it PROVED it ran to completion, or `None`
    when no manifest exists at all (an old candidates.csv). It is cross-checked FIRST, before
    a single clip is called silent, because silence inferred from absence is this project's
    signature bug (`_prove_coverage`).

    Dedup is CANDIDATE-level, on the labeller's own key (`labelling.resume_key` live,
    `labelling.settled_key` off the file) -- the interval END, which is never clamped, so
    two early candidates that both clamp to start 0.00 stay
    distinct. A settled (non-pending) label suppresses the candidate whose moment matches
    it, AND that
    clip's silent row -- a human has judged the clip as a whole. A `pending` label suppresses
    NOTHING: it asserts nothing, so it cannot answer a question, and a clip carrying only a
    pending row must still be proposed at every one of its candidates. That is the only reason
    the ignore-marked fight can be re-proposed at all.

    Nothing is held back quietly. The un-drawn silent clips and the already-labelled ones are
    each counted into the log, because a silent omission reads afterwards as "we covered
    everything" when we did not.
    """
    _prove_coverage(clips, candidates, coverage)
    done_keys, judged_clips = _judged(labelled)
    proposed = [row for row in _stratify(candidates, cameras) if not is_done(row.row, done_keys)]
    silent = [clip for clip in _silent(clips, candidates) if clip not in judged_clips]

    # Every proposed row is a candidate, and `classify` only ever emits the four candidate
    # strata -- all of them drawn whole. SILENT is added below, never by `_stratify`.
    rng = random.Random(seed)  # noqa: S311 - choosing clips to watch, not keying a cipher
    drawn = sorted(rng.sample(silent, min(count, len(silent))))

    worklist = proposed + [SampleRow(_silent_row(clip), Stratum.SILENT) for clip in drawn]
    _log_what_was_left_out(
        silent=silent, drawn=len(drawn), skipped=_skipped(clips, judged_clips), seed=seed
    )
    return sorted(worklist, key=lambda row: (DRAWN_STRATA.index(row.stratum), order_key(row.row)))


def _judged(labelled: LabelSet | None) -> tuple[set[ResumeKey], set[str]]:
    """The candidate keys and clips a human has SETTLED. `pending` rows are excluded from
    both: a pending interval asserts nothing, so it suppresses neither a candidate nor a
    clip's silent row.

    The candidate keys are `labelling.settled_key(clip, end)` -- the interval END, taken
    straight off the settled row, which `interval_for` never clamps. `labelling.is_done`
    builds the same key from the live candidate row via `resume_key`, and both go through
    one `_moment_key`, so the sampler and the labeller cannot drift apart across the CSV's
    2-dp round-trip. Two early candidates whose starts both clamp to 0.00 stay two distinct
    questions, because the END is what is keyed on.
    """
    if labelled is None:
        return set(), set()
    settled = [i for i in labelled.intervals if i.kind is not LabelKind.PENDING]
    done_keys = {settled_key(i.video_id, i.end) for i in settled}
    judged_clips = {i.video_id for i in settled}
    return done_keys, judged_clips


def write_sample(rows: Iterable[SampleRow], path: Path) -> int:
    """Write the worklist. GITIGNORED: every row names a clip of children."""
    ordered = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLE_COLUMNS)
        writer.writerows((*astuple(row.row), row.stratum.value) for row in ordered)
    return len(ordered)


def counts(rows: Iterable[SampleRow]) -> dict[Stratum, int]:
    """How many of each stratum the worklist holds. The weights the analysis needs."""
    tally = dict.fromkeys(DRAWN_STRATA, 0)
    for row in rows:
        tally[row.stratum] = tally.get(row.stratum, 0) + 1
    return tally


def _stratify(
    candidates: Sequence[ScanRow], cameras: Mapping[str, BullyingCamera]
) -> list[SampleRow]:
    """Every candidate, tagged with the stratum its own camera's thresholds put it in."""
    rows = []
    for row in candidates:
        camera = cameras.get(row.clip)
        if camera is None:
            raise SampleError(
                f"{row.clip}: named in the candidates file but not in the clips directory. "
                "The scan and the clips have drifted apart; re-run `qorgan eval scan`."
            )
        rows.append(SampleRow(row, classify(row.confidence, camera.bullying.confidence)))
    return rows


def _prove_coverage(
    clips: Sequence[str], candidates: Sequence[ScanRow], coverage: Collection[str] | None
) -> None:
    """Silence must be PROVEN by the coverage manifest, never inferred from absence.

    Absence from candidates.csv cannot tell "covered but genuinely silent" from "never
    covered" -- both are zero rows. A stale candidates.csv, a truncated scan, or a clip
    added to the dir without a re-scan then becomes a false SILENT, a clip the detector
    never processed, and a real fight in it is filed as silent and never sampled. The
    manifest is the proof that distinguishes the two, and it is checked before any clip is
    called silent:

      - no manifest at all (`None`): an old candidates.csv from before coverage was recorded.
        Refuse -- do NOT fall back to the absence==silent guess, which is exactly the bug.
      - a clip with a candidate row but absent from the manifest: the two artifacts are
        inconsistent (one is stale). A hard error naming the clip.
      - a clip in the clips dir but absent from the manifest: UNSCANNED, its silence unknown.
        A hard error naming the clip -- never silently reclassified as silent.

    A clip that IS in the manifest with no candidate row is the only thing `_silent` may then
    call silent, and now it is proven.
    """
    if coverage is None:
        raise SampleError(
            "no coverage manifest beside the candidates file. This candidates.csv predates "
            "coverage recording, so a clip's silence cannot be proven: absence from the file "
            "is NOT proof the detector saw nothing. Re-run `qorgan eval scan` to write the "
            "manifest, then sample again."
        )
    covered = set(coverage)

    inconsistent = sorted({row.clip for row in candidates} - covered)
    if inconsistent:
        listed = "\n  ".join(inconsistent)
        raise SampleError(
            f"named in the candidates file but absent from the coverage manifest:\n  {listed}"
            "\n\nThe two artifacts are inconsistent -- one of them is stale. Re-run "
            "`qorgan eval scan` so candidates.csv and its manifest are written together."
        )

    unscanned = sorted(set(clips) - covered)
    if unscanned:
        listed = "\n  ".join(unscanned)
        raise SampleError(
            f"in the clips directory but NOT in the coverage manifest:\n  {listed}\n\n"
            "The scan never processed these clips, so their silence is unknown, not proven -- "
            "a real fight in one would be filed 'silent' and never sampled. Re-run "
            "`qorgan eval scan` over the clips directory, then sample again."
        )


def _silent(clips: Sequence[str], candidates: Sequence[ScanRow]) -> list[str]:
    """The clips the fast tier never fired on. Sorted, so the draw is reproducible: a set's
    iteration order is not stable across processes, and a seed cannot fix what it cannot
    see.

    Every clip here has already been PROVEN scanned by `_prove_coverage`, which `draw` runs
    first -- so "no candidate row" now means "covered and nothing fired", not "never seen"."""
    fired = {row.clip for row in candidates}
    return sorted(set(clips) - fired)


def _silent_row(clip: str) -> ScanRow:
    """A silent clip, as a row. No timestamp: the detector named no moment in it, and this
    tool does not invent one. The scores are 0.0 because nothing fired -- they are shown to
    the labeller and are never a label."""
    return ScanRow(clip=clip, timestamp=None, score=0.0, probability=0.0, confidence=0.0)


def _skipped(clips: Sequence[str], done: Collection[str]) -> int:
    return len([clip for clip in clips if clip in done])


def _log_what_was_left_out(
    *, silent: Sequence[str], drawn: int, skipped: int, seed: int
) -> None:
    """Every clip this sample does NOT put in front of a human, counted out loud. The
    below-cap and near-miss candidates are no longer among them -- they are drawn now."""
    logger.info("sample: seed %d, %d silent clip(s) available, %d drawn", seed, len(silent), drawn)
    if skipped:
        logger.info("sample: %d clip(s) skipped, already labelled in labels.csv", skipped)
    if len(silent) > drawn:
        logger.info(
            "sample: %d silent clip(s) not drawn -- this sample does NOT cover them, and a "
            "fight in one of them is a miss nobody will see",
            len(silent) - drawn,
        )

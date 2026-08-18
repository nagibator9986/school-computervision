# Task: fix the three labelling blind spots

**Branch:** `feat/detector-calibration` — worktree `c:\Users\tokmo\Downloads\qorgan ai\q.ai-calibration`
(All `qorgan eval scan|sample|label` code lives ONLY on this branch. `main` has none of it.)

## Why this is load-bearing

The corpus has **one** confirmed fight (`1.2 - ученики … подозрение на буллинг.mp4`). Nobody
knows *when* in the clip it happens, so it was written into `eval/labels.csv` as `ignore` —
because `ignore` was the only kind that could hold a row without asserting a fight.

But `ignore` already means something else: **"a human looked, and this is neither a fight nor
a false alarm — neither reward nor punish."** It is a *judgement*. Overloading it as a
placeholder for *"camera known, label pending"* produced a confident, plausible, wrong record:

- `metrics.evaluate()` counts `total_fights` from `bullying` rows only, so the fight
  contributes **no TP and no FN** — recall is `0/0` and *looks clean*.
- `metrics._inside_ignored()` **absorbs** any prediction inside the interval, so the detector
  firing on the real fight is silently swallowed.
- `eval sample` will **never propose the clip again** (defect 3), so the human can never
  convert it into the corpus's first positive.

**Until this is fixed, recall is unmeasurable.** This is the same disease as `newly_bound=False`
meaning both "already bound" and "failed to recognise": a real value pressed into service as a
placeholder. **Fix it in the TYPE, not by discipline.**

---

## Defect 1 — add a `pending` label kind

`src/qorgan/evaluation/labels.py:44-49` — `LabelKind` has exactly three members
(`bullying`, `normal`, `ignore`) and `_parse` (`labels.py:163-168`) hard-rejects anything else.

**Add a fourth member, `PENDING = "pending"`. Do NOT reuse or re-document an existing kind.**

Its contract, exactly:

1. **It carries the camera.** The `camera` column (`labels.py:170-171`) must survive on a
   pending row, and `LabelSet.camera_for()` must still resolve it. A pending row exists
   precisely so the clip stays scannable while the timestamp is unknown.
2. **It asserts nothing.** It is NOT a fight → it must never contribute to `total_fights`
   (`metrics.py:105`). It is NOT a judgement of "neither" → it must **NOT** absorb predictions
   the way `ignore` does (`metrics.py:87-88`, `_inside_ignored`). A pending interval means
   *"no human has looked yet"*, and a system that cannot represent that will invent an answer.
3. **It is invisible to every metric** — contributes no TP, no FP, no FN.
4. **It must not be silently invisible.** This is the crux. A metric that quietly drops data
   is exactly the bug this project keeps paying for: *a true number that implies a false
   conclusion.* Therefore `Metrics` must carry a **count of pending intervals**, and:
   - `eval run` prints it, prominently — a report that says "precision 0.94" while three
     clips are unlabelled must say so on the same screen.
   - `eval gate` and `eval save-baseline` must **REFUSE (hard error, non-zero exit)** while
     any `pending` interval exists. A baseline whose recall is fiction must not be frozen into
     a committed artefact.
5. `labelling.py:55-60` `CHOICES` — the interactive labeller must offer it (suggest key `p`),
   and its help text must state that `pending` means *"I cannot judge this yet"*, distinct from
   `s` (skip, writes nothing) and `i` (ignore, a judgement).

## Defect 1b — `eval label` never writes the camera column (found while mapping; fix it here)

`labelling.append_label` (`labelling.py:128-144`) writes a header of `REQUIRED_COLUMNS` — four
columns — and 4-field rows. **It can never record an explicit camera.** Appending to the live
5-column `eval/labels.csv` produces ragged rows; `csv.DictReader` tolerates them, so the camera
silently becomes `None` → "infer from filename" → and the three human-named clips (including
**the only confirmed fight**) become un-scannable on the next run.

Fix: `append_label` writes the `camera` column. Never emit a ragged row. If the file already
exists with a different column set, that is an error, not something to paper over.

---

## Defect 2 — the below-cap candidates are classified and then thrown away

`src/qorgan/evaluation/sampling.py:70-80`:

```python
class Stratum(StrEnum):
    ALERT = "alert"
    SKELETON_SUPPRESSED = "skeleton_suppressed"
    BELOW_CAP = "below_cap"
    SILENT = "silent"

DRAWN_STRATA = (Stratum.ALERT, Stratum.SKELETON_SUPPRESSED, Stratum.SILENT)
```

`draw()` (`sampling.py:141-142`) computes `held = [... BELOW_CAP]` and only **logs** it
(`_log_what_was_left_out`, `:221-228`). So the 22 below-cap candidates from the corpus scan are
never put in front of a human. They are the candidates nearest the decision boundary from below
— **exactly the evidence an operating point is chosen with.**

**Two things are wrong, fix both:**

**(a) `BELOW_CAP` is a misnomer — it is the residual bucket, and it lies.** `classify()`
(`sampling.py:109-115`) returns `ALERT` at `>= notify_threshold` (0.85), `SKELETON_SUPPRESSED`
at exactly `cap_without_skeleton` (0.72), and `BELOW_CAP` for *everything else*. A candidate at
**0.80** — skeleton CONFIRMED, but under the alert threshold — is not below the cap at all, and
it gets filed as `below_cap`. Today's corpus happens to contain none (the distribution is 22
below 0.72, 72 at exactly 0.72, 51 at ≥ 0.85), so the bucket has never yet lied out loud. Split
it so it cannot:

- `ALERT` — `confidence >= notify_threshold`
- `NEAR_MISS` — `cap_without_skeleton < confidence < notify_threshold` (skeleton confirmed,
  below the alert threshold)
- `SKELETON_SUPPRESSED` — `== cap_without_skeleton` (the skeleton looked and refused)
- `BELOW_CAP` — `< cap_without_skeleton`
- `SILENT` — no candidate on the clip at all (assigned in `draw`, not `classify`)

Keep the boundaries read from the camera's own config (`notify_threshold`,
`cap_without_skeleton`) — they are NOT constants. The existing test
`test_the_stratum_boundary_is_the_cameras_own_config_not_a_hardcoded_number`
(`test_eval_sample.py:144`) pins that; do not regress it.

**(b) Draw them.** `BELOW_CAP` and `NEAR_MISS` join `DRAWN_STRATA`. Then:
- `counts()` (`sampling.py:165-170`) seeds its tally from `DRAWN_STRATA` — it will now include them.
- **`cli.py:288-292` `_MEASURES` has no `BELOW_CAP` key, and `cli.py:281` does `_MEASURES[stratum]`
  — it will raise `KeyError` the instant one is drawn.** Add an entry for every stratum. A dict
  lookup that can `KeyError` on a valid enum member is a latent crash; make it total.

## Defect 3 — clip-level dedup is hiding candidates

The labeller and the sampler use **different dedup keys**, and the sampler's is the coarse one.

- `eval label` — candidate-level, **correct**: `labelling.py:147-166`, keyed on
  `(clip, round(start, 2))`. Two candidates on one clip are two distinct questions.
- `eval sample` — **clip-level**: `cli.py:270` passes `labelled=set(labels.videos)` (a set of
  *video_ids*, timestamps discarded), and `sampling.py:137-139` filters on `row.row.clip not in done`.

So **one label anywhere on a clip erases every other candidate on that clip from the worklist,
forever, with no log line.** That is the 145-candidates-on-140-clips collapse. And it is why the
`ignore`-marked fight clip can never be proposed again.

**Fix:** the sampler's dedup key must match the labeller's — `(clip, round(start, 2))`. Pass the
interval starts through, not just the video ids.

**And the rule that makes the whole thing cohere:**

> **A `pending` interval never suppresses anything.** It asserts nothing, so it cannot answer a
> question. A clip carrying only a pending row must still be proposed, at every one of its
> candidates.

A non-pending label suppresses (i) the candidate whose start matches its key, and (ii) the
`SILENT` row for that clip (a human has judged the clip as a whole).

`tests/test_eval_sample.py:169-181` (`test_a_clip_already_in_labels_csv_is_not_proposed_again`)
**asserts the current clip-level behaviour** — it passes bare clip names. It encodes the bug.
Rewrite it to the candidate-level contract.

---

## Acceptance — the test that proves the whole thing

Write a test that fails today and passes after, stating the real defect end to end:

> **Given** `labels.csv` carrying the only confirmed fight as a `pending` row with an explicit
> camera and an unknown interval, **when** the human runs `eval sample`, **then** every candidate
> on that clip is still proposed; **and** `eval run` reports the pending count rather than a clean
> recall; **and** `eval save-baseline` refuses.

Also pin the type-level guarantee: **no `LabelKind` member may be silently dropped by
`evaluate()`** — a test that enumerates `LabelKind` and asserts each is explicitly handled, so
that adding a fifth kind later cannot default into silence.

---

## Rules (each was paid for by a real bug on this project)

- **Never guess a value.** Enforce "unknown" in the TYPE, not by discipline.
- **A check nobody has watched FAIL is not a check.** Write the test first, RUN it, watch it
  fail for the right reason, then implement. No tautological asserts.
- **A corrected number is not corrected until you grep for it.** After changing the strata, grep
  the repo for `below_cap`, `DRAWN_STRATA`, `_MEASURES`, and `LabelKind` — including docs, CLI
  help text, and `eval/labels.csv`'s header comment — and fix every stale site.
- **If the brief contradicts the code, STOP and say so.** Do not quietly adapt.
- **DATA:** `eval/clips/`, `eval/crops/`, `student_photos/` are video and photographs of
  children. They are NOT in git and never will be. **NEVER `git add -A` or `git add .`.** Stage
  by explicit path; run `git status` before every commit.
- `eval/labels.csv` IS tracked. Updating the fight's row from `ignore` to `pending` is part of
  this task — it is the record that motivated it.

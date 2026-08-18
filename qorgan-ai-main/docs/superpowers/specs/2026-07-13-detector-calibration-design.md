# Spec B — Calibrating the bullying detector

**Status:** approved design, ready to plan.
**Date:** 2026-07-13.
**Depends on:** nothing. Runs in parallel with Spec A (they share no files).

---

## 1. Why this is the module that matters

Bullying detection is the main product, and **nobody knows how well it works.** Not
approximately — at all. There is no precision figure, no recall figure, and no
false-alarm rate. The thresholds in `config/profiles/*.yaml` are converted estimates.

Two things that made calibration *impossible* have been fixed: the acceleration unit bug,
and a threshold that sat below the noise floor of the YOLO box itself. Calibration is now
possible, and the data to do it has arrived.

---

## 2. What the data actually is

The school sent 1 958 clips. They are not 1 958 usable clips. Every one was probed:

| | count | footage | what it is |
|---|---|---|---|
| **Full-frame 2560×1440** | **663** | **97 min** | usable. 643 are `burst101` HD evidence bursts |
| Small crops (~320×450) | 1 293 | 113 min | the old detector's **cropped pair ROI** |
| Unreadable | 2 | — | |

**The crops cannot be detector input, and it is important to understand why**, because they
look like data. A 320×450 crop has no scene, no zones, and no frame geometry, and the
scorer's box-diagonal scaling is meaningless inside it. Running the detector on them would
produce numbers, and every one of them would be a lie.

**So the corpus for *evaluation* is 663 clips, ~97 minutes.** Frame rates vary (5 / 8 / 10
fps); the harness already reads fps from the file rather than assuming it.

### 2.1 But the crops are the review view — label from the crop, evaluate on the full frame

`hall_left_main_1009_1019_….mp4` (186×326) and
`hall_left_main_1009_1019_burst101_….mp4` (2560×1440) are the **same incident**: same
camera, same track-ID pair, seconds apart. The crop was cut out of the burst.

And a human deciding *"is this a fight?"* does not need the scene. They need to see **the
two children, close up** — which is exactly what the crop is, and it is far faster to watch
than a 1440p wide shot in which the pair occupies 3% of the pixels.

Measured across the whole corpus: joining on `(camera, track_a, track_b)` and nearest
timestamp, **621 of 660 full-frame clips (94%) have a crop partner.** The join is real, not
an artefact of one example. (The 3 unparsable filenames are the human-named clips.)

So: **`qorgan eval label` shows the crop; `qorgan eval run` scores the full frame.** Same
incident, two views, each used for what it is good at. This roughly halves the labelling
time and costs one join. Where no crop partner exists (6%), the labeller falls back to the
full frame.

### 2.2 What the corpus is good for, and what it is not

All 1 955 machine-made clips are the **old detector's trigger clips**, and they were mostly
false positives. So the corpus is, in effect, 97 minutes of ordinary school corridor —
**adversarially selected against a bullying detector.** That is the best possible negative
set, and it measures a false-positive rate directly.

What it cannot measure is **recall against the world**: there is no footage of a fight the
old detector *missed*, because a clip only exists if the old detector fired. Of the three
human-described clips, exactly one is a fight. One fight is not a recall number.

This gap is real, it is the school's to fill, and this spec names it rather than papering
over it. See `docs/questions-for-school.md`.

### 2.3 The corpus calibrates the hall, and only the hall

| camera | full-frame clips |
|---|---|
| `hall_right` | 344 |
| `hall_left` | 299 |
| stairs | 17 |
| yard | **0** |

**Stairs and yard remain uncalibrated after this work, and the report must say so.**
Seventeen clips with no fights in them is not a calibration.

---

## 3. B0 — make the bench match the field

Nothing measured on a harness that differs from production is worth measuring. These are
blocking, and they come first.

### 3.1 The harness never runs the skeleton — so the PR curve is a fiction

`harness.run`'s `skeleton` parameter defaults to `no_skeleton`, and
[`evaluation/cli.py:160`](../../../src/qorgan/evaluation/cli.py) never passes one. So
`validation_score` is always 0.0, **every verdict is capped at
`cap_without_skeleton = 0.72`**, and `Alert.notified` is always `False`.

The Telegram threshold is **0.85**. The PR curve is identically empty above 0.72. Today the
harness cannot produce a single alert that production would actually send, and calibrating
on it would tune a number that can never be reached.

**And the fix contains a trap.** Writing a fresh crop→pose→judge path inside the harness
would re-create the legacy's three-diverged-copies bug *while fixing a bug caused by it*.
So the crop→pose→judge step is extracted from
[`worker/bullying.py:151`](../../../src/qorgan/worker/bullying.py) into **one shared
function that both the worker and the harness call.** Rule R2, enforced rather than
restated.

### 3.2 Production does not resize; the harness does

[`evaluation/video.py:65`](../../../src/qorgan/evaluation/video.py) resizes every frame to
`capture.frame_width × frame_height`.
[`worker/camera_loop.py:99`](../../../src/qorgan/worker/camera_loop.py) hands YOLO whatever
the RTSP substream delivers, unresized.

Speeds are in px/s. **So a threshold tuned on the bench is in `capture.frame_width ×
frame_height` pixels and production is in some-other-resolution pixels**, and it does not
transfer. This is the same *class* of bug as the `(px/frame)/s` unit bug — a quantity that
silently changes meaning between the bench and the field.

**And that resolution is PER PROFILE, not one number.** 960×540 is only `base.yaml`'s
default: `hall.yaml` and `canteen_entry.yaml` override it to **1280×720**. Writing "the
analysis resolution is 960×540" as a blanket statement is how a whole family of derived
figures came out wrong once already (identity spec §2.4). Quote the key, not a number.

**Fix:** one shared `prepare_frame()`, called by both. Rule R2 was applied to *scoring* and
stopped there; preprocessing is the other half of "the same functions". Production then
analyses `capture.frame_width × frame_height` — **that camera's own** — regardless of what
the NVR sends, which also pins YOLO's input cost.

**This changes what production does today, and that must be written down where it will be
found.** Production currently analyses whatever the substream delivers; after this it
always analyses `capture.frame_width × frame_height`. The consequence is not small:

> **Every speed and acceleration threshold — and the measured noise floor — is now pinned
> to that resolution.** Changing `frame_width` silently invalidates every threshold in
> every profile, because px/s means something different in a different frame.

Acceptable, because we are recalibrating anyway and a pinned resolution is what *makes*
calibration transferable. But the next person to reach for `frame_width` must be told, in
the config file itself, that they have just invalidated the tuning. A comment at the key,
and a line in the profile headers.

### 3.3 The clip's camera must come from the clip

`eval run --camera` picks **one** camera config for an entire run. Our corpus is 344
`hall_right` + 299 `hall_left`, and `hall_left` carries a `mirror_ignore` zone over a
**reflective column that does not exist in hall_right's field of view**. Evaluating one
camera's footage against another's zones silently blanks out part of the frame.

The camera is inferable from the filename (`hall_right_main_…`). If it cannot be inferred,
that is a **hard error**, not a default.

### 3.4 Two smaller ones

- `VideoSource` calls `.track()` with no `device=`, taking the Ultralytics default rather
  than the configured GPU. One line — and it matters when 663 clips go through it.
- [`config/profiles/stairs.yaml:49-50`](../../../config/profiles/stairs.yaml) still carries
  `static_speed_threshold: 4.0` / `moving_speed_threshold: 8.0` — legacy **px/frame**
  values, compared against `Track.speed` in **px/s**. A person standing perfectly still
  reads well above 4 px/s from box jitter alone, so `a_static` is never true and **gate 8
  (`staircase_pass`) is dead on the one camera type it exists for.** The units migration
  missed this file.

### 3.5 A test that fails when a declared config key is read nowhere

About twelve config knobs are parsed and never consumed — `SeparationGuard`,
`ViolenceSettings`, `min_group_size`, several gate thresholds — while the gates hardcode
multipliers of `PairMetrics` values instead. Editing them in YAML does nothing at all.
That is a trap laid for whoever tunes next, and it is exactly how the legacy's 225 keys got
where they are.

So rather than a one-off cleanup: **a test that reflects over every Pydantic model in
`config/`, and fails if a declared field is referenced nowhere in `src/`.** Dead keys are
then either wired up or deleted to make it pass — and cannot rot back in.

The allowlist is empty. A "declared plug-in point with no consumer" is a dead key wearing a
hat; `ViolenceSettings` goes, and Spec C will design its own config when it needs one.

---

## 4. B1 — the corpus

The 663 full-frame clips go to `eval/clips/`, **gitignored**. They are footage of children.
`git status` is checked before every commit; `git add -A` is never used.

---

## 5. B2 — labelling, by exception and by sample

Watching 97 minutes of corridor to find the few seconds that matter is not a good use of a
human. Two passes:

**Pass 1 — by exception.** `qorgan eval scan` runs the detector over all 663 clips at
threshold 0 and emits `eval/candidates.csv` (clip, timestamp, score, probability,
confidence). A human then watches only what the detector **fired on** — perhaps 50–150
clips of 8 seconds each. This yields precision, and it concentrates the human's attention
exactly where the label changes the answer.

**Pass 2 — random sample.** Precision alone is half a detector. So we also label a random
sample of **~80 clips the detector did *not* fire on.**

This is the only recall signal this data can yield, and it is worth being precise about
what it is: not "fights we miss in the world" — that needs footage we do not have — but
**"fights we miss among clips the old detector flagged"**. That is a real number, and it is
the one that says whether we kept the true positives while dropping the false ones.

**The tool:** `qorgan eval label` — walks the candidates, opens **the crop partner** (§2.1)
in the default player, falling back to the full frame at the alert timestamp when no crop
exists; prompts `bullying / normal / ignore / skip`; appends to `eval/labels.csv`; and is
resumable. It is a dev tool; it gets no web route.

Labels are written against the **full-frame** `video_id`, whichever view the human watched.
The crop is a lens, never the record.

---

## 6. B3 — the curve, and the number the school actually feels

`qorgan eval run` reports precision, recall, F1 and a PR curve — **per camera**, now that
§3.3 makes per-camera evaluation correct.

But the figure that decides whether staff leave the system switched on is not F1. It is:

> **false alerts per hour.**

97 minutes of adversarially-selected corridor measures it directly, and it goes in the
report in bold. A detector with excellent F1 that wakes a teacher twice a night will be
unplugged within a week, and it will not matter what its F1 was.

Then: choose the operating point → set `notify_threshold` → `qorgan eval save-baseline` →
`qorgan eval gate` in CI, so a future threshold change that lowers F1, or raises the number
of missed fights, fails the build.

---

## 7. B4 — the honest report

What we know, with numbers. What we do not:

- **Recall against the world is unmeasured** (§2.2).
- **Stairs and yard are uncalibrated** (§2.3).
- Every number is from hall cameras at 10 fps in daylight.

Plus the written request to the school — see `docs/questions-for-school.md`.

---

## 8. Testing

| Unit | How it is tested |
|---|---|
| `prepare_frame` is shared | a test asserts the worker and the harness call the *same* function object |
| the harness runs the skeleton | a spy skeleton records that it was called; a verdict above 0.72 is reachable |
| camera inferred from filename | `hall_right_main_….mp4` → `hall_right`; an un-inferable name raises |
| stairs units | a standing person with ±3 px box jitter reads `static` — pins the px/s meaning |
| no dead config keys | reflection over `config/` vs references in `src/`; allowlist empty |
| `eval scan` / `eval label` | round-trip on a fixture clip; labels.csv append is resumable |

## 9. What this spec does not do

- It does not measure recall against the world. It cannot (§2.2).
- It does not calibrate the stairs or the yard (§2.3).
- It does not touch `faces/`, `identity/` or `canteen/` — those are Spec A's.

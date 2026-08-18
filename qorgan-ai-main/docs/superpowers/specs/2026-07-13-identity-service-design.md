# Spec A — The identity service

**Status:** approved design, ready to plan.
**Date:** 2026-07-13.
**Depends on:** nothing. Runs in parallel with Spec B (they share no files).

---

## 1. The insight that shrinks the problem

The school sent 142 photographs. Their filenames are `student_333_1778595343147.jpg`, and
`333` is a **real school ID** — issued by the school's own system, unique across all 142.

So identity is *given*, not inferred. Everything in `faces/identity.py` built to infer an
identity from a name — `generate_external_id`, the Cyrillic/Latin `CONFUSABLES` table,
`Identity`, `Suspect`, `CollisionReport`, `find_namesakes`, and the CLI command
`qorgan pupils check-namesakes` — is answering a question this data does not ask. It goes.

`external_id = "student_333"`, `external_id_source = ROSTER`. The **name is a nullable
display field**. Until the school sends an ID→name table the UI says `Ученик 333, 5-А`.

### 1.1 Two traps in the data, both found by looking rather than assuming

**The filename prefix lies.** The `учитель` folder contains files named
`student_469_….jpg`. A teacher's photo is named "student". **Person type comes from the
FOLDER, never from the filename.** The obvious pattern is wrong, and trusting it would
have filed two teachers as pupils.

**Four staff photos contain no detectable face at all** — `staff_465`, `staff_466`,
`staff_467`, `staff_468`. They cannot be enrolled. They must be itemised in the import
report, not silently dropped.

### 1.2 No silent fallback, ever

A filename that does not match `^(student|staff)_(\d+)_(\d+)\.(jpg|jpeg|png)$` is a **hard
error naming the file**. It is never a guessed identity.

This is the single most important rule in this spec. The legacy's characteristic failure
was not that it got identity wrong — it was that it *invented* an identity and carried on.
A refusal is recoverable. A quiet guess is a child eating someone else's lunch.

---

## 2. What we measured before designing anything

Run before a line of the module was written, because if one photo per child cannot separate
142 children it certainly cannot separate 800, and the module would be built on sand.

Every photo embedded with the production model (InsightFace `buffalo_l`, 512-d ArcFace),
then the full 138×138 cosine matrix. (138, not 142: four staff photos have no face.)

```
cross-person similarity — 138 people, 9 453 distinct pairs

  [0.30,0.35)   133  ############################################################
  [0.35,0.40)    47  ###############################################
  [0.40,0.45)     7  #######
  [0.45,0.50)     1  #
  [0.50,0.55)     0
  [0.55,0.60)     0        <-- 9 453 pairs, and not one lands in this band
  [0.60,0.65)     0
  [0.65,0.70)     0
  [0.70,0.75)     0
  [0.75,0.80)     3  ###
  [0.80,0.85)     1  #
  [0.95,1.01)     2  ##
```

The band is empty from 0.48 to 0.77. That emptiness is the measurement: **the six pairs at
the top are not a tail of the impostor distribution. They are a different population.**

### 2.1 Finding: six people hold two school IDs each

Two different children do not score 0.999.

| similarity | ID A | ID B | reading |
|---|---|---|---|
| 0.999 | `staff/464` | `учитель/477` | same person, two IDs |
| 0.984 | `11-А/470` | `staff/334` | same person, two IDs |
| 0.810 | `3-А/371` | `3-А/472` | same person, two IDs |
| 0.792 | `2-Б/369` | `2-Б/471` | same person, two IDs |
| 0.781 | `5-А/402` | `5-А/473` | same person, two IDs |
| 0.774 | `7-А/438` | `7-А/439` | **ambiguous — may be identical twins** |

IDs **470, 471, 472, 473, 477** all carry 10-digit timestamps; every other file carries a
13-digit one. A **second enrolment batch** re-registered five people who already existed.
`438`/`439` is the odd one out — adjacent IDs, same class, both in the *first* batch, and
the lowest score of the six. That one may be twins, and arithmetic cannot settle it.

**This is the exact mirror of the legacy's namesake bug.** The legacy collapsed two
children into one identity. This data does the reverse: one child holding two identities,
whose meals split across both — so **the school's existing canteen records are already
wrong for these six people.** The machinery we are deleting was aimed at the wrong failure;
its replacement is aimed at the one that is actually present, and it fires six times.

### 2.2 Finding: recognition can work — and `min_score` is currently too low

Duplicates removed, the genuine impostor distribution is:

| | value |
|---|---|
| p50 | 0.094 |
| p90 | 0.214 |
| p99 | 0.331 |
| **max** | **0.472** |
| pairs ≥ 0.45 | 1 of 9 447 |

One photo per child **does** separate these children. That was the open question; the
answer is yes.

But `RecognitionPolicy.min_score` is **0.45**, which is *below* the worst genuine impostor
at 0.472. Margin **−0.022**. It admits a known confusion today.

**Measured floor: `min_score ≥ 0.50`** — inside the empty band, above every impostor.

Note what guessing would have cost. `identity.py` carries `SAME_PERSON_SIMILARITY = 0.35`.
Against this data that constant would call **55 pairs** the same person. The measured band
says **0.60**. Every number this spec sets is recorded with the measurement beside it.

### 2.3 The ceiling, probed against real camera footage

The scores above are gallery-photo against gallery-photo. In production the query is a
*camera* face — blurred, off-angle, small. So the probe gives a hard **floor** under
`min_score` and, on its own, says nothing about the **ceiling**: whether a real camera face
can *reach* 0.50 at all. If it cannot, we set the gate above what any child can score and
recognise **nobody** — the 1816-NULL failure arriving from the opposite side.

That ceiling is not unmeasurable. Spec B has 663 clips of the hall these same children walk
through. So: **250 clips, 4 674 frames, 14 970 real faces, scored against the gallery.**

The hall is wider and further than the canteen entry, so this is a **pessimistic lower
bound** — if the hall clears 0.50, the canteen certainly does.

**It does not clear it, and the reason is not the threshold.**

```
face width in the 2560×1440 frame:   p50 = 23px    p90 = 45px    max = 100px
clearing the 60px enrolment gate:    332 / 14 970  (2.2%)

top-1 score against the gallery, BY FACE SIZE
  width        n       p50     p90     p99     max
  0–30px    10898    0.135   0.201   0.278   0.435
  30–60px    3740    0.159   0.312   0.466   0.604
  60–100px    331    0.198   0.266   0.374   0.513
```

**The binding constraint is face size, not `min_score`.** A 23-pixel face upscaled to
ArcFace's 112-pixel input is mush, and no value of any threshold recovers it.

So the result is **inconclusive for the canteen rather than fatal**: it fails to prove
success, it does not prove failure. What it *does* establish is that scores reach **0.604**
on real camera faces, and that the population above 0.45 lives at **40–64 px**. The canteen
entry camera is close-range, so its faces are far larger than the hall's. The ceiling
question stays open, but it is now bounded rather than blank.

**`min_score = 0.50` therefore still ships as a FLOOR with the ceiling not yet settled, and
the config says so in words.** It is not a settled value and must not be written as one.

**The ask to the school is unchanged and remains the highest-value item we can get:**
footage from the *canteen entry* camera of pupils we can name. One volunteer walking
through. That closes it.

### 2.4 The 2.2% is measured on a stream production never analyses — the real number of recognitions is ZERO

**Correction, and it inverts the conclusion.** Those 250 clips are 2560×1440 HD evidence
bursts. **Production does not analyse that stream.** Spec B §3.2 pins the analysis loop to
`capture.frame_width × frame_height` — so the gate must be re-expressed in the pixels the
worker actually sees.

**There is no single analysis resolution. It is PER PROFILE:**

| profile | analysis frame | scale from the 2560×1440 clip |
|---|---|---|
| `hall.yaml` | **1280×720** | **0.5** |
| `canteen_entry.yaml` | **1280×720** | 0.5 |
| everything else | 960×540 (inherited from `base.yaml`) | 0.375 |

960×540 is `base.yaml`'s **default**, not the fleet's resolution — and the hall, the camera
this whole section is about, **overrides it**. *That* is the argument for `camera-report`
measuring per stream **from the camera's own config** rather than assuming a number: a
blanket resolution is exactly the shape of mistake this section keeps making.

At the hall's real **1280×720** (scale 0.5, so 2× smaller in each dimension):

```
the strict 60px gate      ==  a 120px face in the 2560x1440 clip
the small-face 38px gate  ==  a  76px face in the 2560x1440 clip

largest face in the ENTIRE hall corpus (14 970 faces):   100px at HD  ->  50px at 1280x720

faces clearing the strict 60px gate:      0 of 14 970   (largest is 100px at HD; needs 120)
faces clearing the SMALL-FACE 38px gate:  77 of 14 970  (0.51%)

    of those 77, ACCEPTED at min_score 0.45:  0
    of those 77, ACCEPTED at min_score 0.50:  0
    the best score among all 77 is 0.350 -- far below either threshold.

=> ZERO recognitions in 14 970 faces.

hall face size at 1280x720, ALL 14 970 faces:
    p50 = 11.5px    p90 = 22.5px    max = 50px
    (HD:  p50 = 23px    p90 = 45px    max = 100px)
```

> **Two corrections have now landed on this table, and both erred in the same direction as
> the mistake they were fixing.**
>
> It first read `p50 = 18px, p90 = 24px` — the cached subset of faces ≥38 px at HD, **the
> largest 15%**, mislabelled as the median of all.
>
> It then read `p50 = 9px, max = 37.5px, 0 of 14 970 clear ANY gate` — computed at
> **960×540**, which is `base.yaml`'s default and **not the resolution the hall runs at**.
> The hall profile overrides to 1280×720, so every derived figure was scaled by 0.375 when
> the truth was 0.5.
>
> **The conclusion survived both times, but by a different mechanism each time, and the
> mechanism is the part that must be right.** It is not "no face reaches the gate" — 77 do.
> It is that **not one of them scores high enough to be recognised.**
>
> **A corrected number is not corrected until you have grepped for the VALUE, not the
> sentence — and until you have checked which config actually applies.**

**Seventy-seven faces in fifteen thousand clear the small-face gate, and not one of them is
recognised.** The best score any of them achieves is **0.350**, against a `min_score` of
0.45 on the small-face path and 0.50 on the strict one. The strict 60 px gate is cleared by
**0 of 14 970** — it needs a 120 px face in the clip, and the largest in the whole corpus is
100 px.

Face recognition on the hall analysis stream is not poor. **It is arithmetically
impossible** — the faces that are big enough are still too degraded to score, and the ones
that could score are not big enough to exist. The 2.2% was an optimistic upper bound
measured on a stream the analysis loop never touches.

### 2.5 So bullying events stay anonymous — and the HD burst does not rescue them

The stated point of lifting identity out of `config/canteen.py` was that *any* camera could
call it, and the headline use was **naming the children in a bullying event**.

The obvious rescue is to run identity on the **HD burst**: the burst already opens channel
101 on a suspicious pair, which is exactly the moment you want a face and the only moment
the pixels exist. It is the right instinct, so it was measured rather than assumed.

**On the burst itself:** 332 of 14 970 faces clear the 60 px gate (2.2%), and of those 332,
**exactly one** is accepted at `min_score = 0.50` (two at 0.45). That is ~1 recognition per
250 clips. Our gallery holds 138 of a 500–800-pupil school, so most faces belong to people
who *cannot* be matched; a full roster would raise it perhaps 4–5×.

**That still lands at roughly 2% of events producing a name.** Ninety-eight bullying events
in a hundred stay anonymous. That is not a capability — it is a coin that lands heads once
in fifty — and shipping it as "the system names the children involved" would be a lie.

**The burst does not fix it, because nothing in software can.** The children are 10–15 m
from the lens. That is optics.

**Decision:**

1. **Bullying events are ANONYMOUS at the current camera placement, and the system says so.**
   Identity is a **canteen** capability now, and a classroom one later (Spec D).
2. **`IdentityService` is still built so that any camera *can* call it** — that costs nothing
   beyond where the config lives, and it is the correct boundary regardless.
3. **The per-event burst binding is NOT built.** It would fire once in fifty and create an
   expectation the optics cannot honour. YAGNI. When a camera exists that could feed it, the
   architecture is already in place.
4. **Naming children in a bullying event requires a camera that can see a face** — a
   face-height camera at a chokepoint, or moving the existing ones. That is the school's
   decision, and it is optical, not a threshold. It goes in the questions.

### 2.6 `camera-report` — per STREAM, and it gates identity

This is now the most important small thing in the module, and it must answer the question
that was actually asked:

> **"Can this camera recognise anybody at the resolution the worker actually feeds it?"**

So it reports **per stream** — the analysis substream *and* the burst — each measured at the
resolution that stream is really analysed at, **read from that camera's own merged config**.
A number from the wrong stream is worse than no number, as §2.4 demonstrates at my own
expense — and so is a number from the right stream at the wrong resolution, which is the
*second* way §2.4 went wrong. There is no fleet-wide analysis resolution to assume: `hall`
and `canteen_entry` run at 1280×720, everything else inherits 960×540. A command that
assumes one number is a command that will one day print a confident wrong one.

And it **gates**: a camera whose faces essentially never clear the gate **cannot be
configured as identity-capable**. That turns "this camera recognises nobody" from a
discovery made after months of tuning into a fact asserted at startup. The legacy asked
this question eighteen times, in the form of a threshold. It is not a threshold question.

### 2.7 The duplicate identities collapse the gap — measured, and smaller than it first looked

The top hall matches are `id=334` and `id=470` — **the duplicate pair**. Both are the same
human, so top-1 and top-2 are *both him*, and the gap collapses:

```
40px  id=334  score 0.604   gap +0.001  ->  after merge  +0.413
42px  id=470  score 0.569   gap +0.003  ->  after merge  +0.418
43px  id=470  score 0.550   gap +0.006  ->  after merge  +0.420
```

`min_gap` is 0.05. **He is rejected as AMBIGUOUS every time.** This is the 1816-NULL
mechanism alive in this data — and note that ranking *by person* (last session's fix) cannot
help, because the two rows genuinely **are** different persons in the database.

**The gap collapse above is REAL and it stands.** It is a property of the **gallery** — one
human enrolled twice sits in his own top-2 — not a property of the hall. It would happen on
any camera, at any resolution, to those six people.

A first A/B claimed to test whether merging rescues recognition and came back **+0%**. That
refuted the claim as stated, for a reason that was my own error: those faces are 40–43 px,
**below** the 60 px gate, so they were never in the population being counted. Measured where
they actually live:

| gate (**face size at 2560×1440 HD**) | | accepted | killed by gap |
|---|---|---|---|
| ≥38px HD (small-face path) | before merge | 3 | **6** |
| ≥38px HD | after merge | **9** | 0 |
| ≥60px HD (strict gate) | before merge | 1 | 0 |
| ≥60px HD | after merge | 1 | 0 |

*(min_score = 0.50, n = 2 255 hall faces.)*

> **This accept-count does NOT describe production, and the caveat is load-bearing.** It was
> measured on faces ≥38 px **at HD**, which is ≥19 px at the hall's real 1280×720 —
> **below the 38 px production gate entirely.** So "3 → 9 accepts" is a statement about
> gallery faces at HD, not about what the hall worker would recognise. On the hall, per §2.4,
> **nothing is recognised regardless of merging**: 77 faces clear the gate and none of them
> scores above 0.350.
>
> The gap collapse (0.001 → 0.413) is unaffected by any of this. It is arithmetic on the
> gallery.

**The honest claim, correctly scoped:** merging is **not** a precondition for recognition in
general — it changes nothing for the 132 people enrolled once. It **rescues the six people
enrolled twice**, who are otherwise near-unrecognisable because their own duplicate sits in
top-2 and kills the gap. It matters **for the canteen**, where faces are large enough to be
recognised at all; it changes nothing on the hall, where nothing is recognised either way.

For those six children the system is not inaccurate. It is **blind**, and no threshold
anywhere would have shown why.

---

## 3. Harden the GPU guard — it is correct today only by an import-order accident

**First, a correction, because the first version of this section was wrong.**

While probing §2.3 I found InsightFace running on the CPU, and wrote here that the
project's anti-CPU guard was blind and that every VRAM measurement was a lie. **That was
false, and the bug was mine.** My probe script never imported torch. The repo's real path
does:

```
inspect_gpu: torch_cuda=True  onnx_cuda=True  ok=True
  detection:   CUDAExecutionProvider
  recognition: CUDAExecutionProvider     (all five models)
```

[`gpu.py:61-64`](../../../src/qorgan/gpu.py) imports **torch first**, which loads the CUDA
runtime DLLs into the process, and InsightFace then resolves them — exactly as its comment
claims. `scripts/vram_spike.py` imports torch at module level too, so
`config/workers.yaml`'s numbers were measured on the GPU and stand. **Production has been
on the GPU all along.**

**What is real, and smaller.** `ort.get_available_providers()` reports what onnxruntime was
*compiled* with — it returns `CUDAExecutionProvider` **even in a process where the provider
DLL cannot load and every session silently falls back to the CPU.** I demonstrated exactly
that. So the guard in `GpuReport.onnx_cuda`, and
[`recognizer.py:54`](../../../src/qorgan/faces/recognizer.py)'s refusal to run on the CPU,
are **load-bearing on an incidental import order** and cannot detect their own failure.

The order is protected by a single `# noqa: I001`. Remove it — or let a future refactor,
formatter, or import-sorter put `import onnxruntime` before `import torch` — and the guard
still says `ok=True` while the whole system runs 40× too slow. The comment says the order
matters. **Nothing enforces it.**

**The fix (hardening, not rescue):** `onnx_cuda` must **build a real ONNX session and read
back `session.get_providers()[0]`**. That is true regardless of import order, and it is the
only check that cannot be fooled. `qorgan doctor` gains it, and a test pins it.

Cheap, and it closes the gap between "the guard passes" and "the GPU is actually being
used" — which is the gap the whole module fell through once already.

---

## 4. Deliverables

### 4.1 `qorgan pupils gallery-report`

Not a diagnostic — a shipped command, because the failure it finds is live in the data.

Outputs:
- the cross-person similarity histogram (as above);
- **duplicate enrolments** at the measured threshold **0.60** — pairs of *different*
  `external_id`s whose faces are the same person;
- the impostor ceiling, and the implied floor under `min_score`;
- photos that cannot be enrolled (no face / >1 face), itemised;
- the extrapolation to a real school (below).

**Extrapolation.** 9 447 impostor pairs give P(two different children ≥ 0.45) = 1.06e-4.
A school of S gives each child S−1 impostors, so P(a child has ≥1 impostor above the gate)
= 1 − (1−p)^(S−1):

| school | risk per child | children affected |
|---|---|---|
| 142 | 1.5 % | ~2 |
| 500 | 5.2 % | ~26 |
| 800 | **8.1 %** | **~65** |
| 1200 | 11.9 % | ~143 |

At 800 pupils roughly one child in twelve has an impostor above a 0.45 gate. This is the
argument for a **second photo per child**, and it belongs in the questions to the school.

### 4.2 `qorgan pupils merge <keep_id> <drop_id>`

Detection is not resolution. Six people hold two IDs; **which ID is canonical is a decision
only the school can make**, and `7-А 438/439` may not be a duplicate at all.

So: `gallery-report` *detects* and *reports*; `merge` *executes a decision a human made*.
It re-points photos, embeddings and canteen sessions from `drop_id` onto `keep_id`,
deactivates `drop_id`, and records the merge. It never runs automatically.

Measured effect (§2.7): the gap collapse is real — 0.001 → **0.413** after merge — and it is
a property of the gallery, so it holds on any camera. The accompanying A/B (accepts 3 → 9,
gap-kills 6 → 0) was measured on **gallery faces ≥38 px at HD**, which is below the hall's
production gate; it does **not** describe the hall, where nothing is recognised regardless.
Merging matters where faces are big enough to be recognised at all — the **canteen**.

It does not improve the system; it makes six specific people visible to it.

### 4.3 `qorgan pupils import-roster <dir>`

Walks the directory tree. Folder decides class and person type; filename must match the
pattern or the import **fails loudly naming the file**.

| folder | person_type | class_name | position |
|---|---|---|---|
| `1-А` … `11-Б` | `STUDENT` | the folder name | — |
| `staff` | `STAFF` | — | — |
| `учитель` | `STAFF` | — | `учитель` |

`external_id_source = ROSTER`. `full_name = NULL`.

The existing ZIP path stays for the web upload, re-expressed as `safe_extract` +
`import_directory` so there is one import, not two.

### 4.4 `IdentityService` — recognise once per track, not once per frame

**The problem.** [`worker/canteen.py:82`](../../../src/qorgan/worker/canteen.py) calls
`recognizer.detect()` — which does face detection **and** the 512-d ArcFace embedding — on
every due frame, every 0.25 s, for every face in shot. The expensive half is the embedding.
A hall camera cannot call any of this, because the config lives in `config/canteen.py`.

**The design** (the client's doc §12.2: track → best face frame → recognise once → bind →
cache → re-run only when the track is lost):

- `FaceRecognizer` splits into `detect_faces()` (boxes + landmarks + det_score, cheap) and
  `embed(image, face)` (the vector, expensive).
- `PersonDetector` (YOLOv8n + ByteTrack — already run by the hall cameras) comes to the
  canteen cameras, giving real `track_id`s that survive a head-turn.
- Faces are assigned to person tracks by containment — a **pure function**, unit-testable
  with no GPU.
- Per track we keep **only the best face seen so far** (quality = area × det_score): one
  object, not a list (rule R8).
- After `min_face_frames` observations *or* `max_wait_seconds`, we **embed once**,
  `identify()` once, and bind `track_id → person_id`.
- Accepted ⇒ never recognised again. Rejected ⇒ retried up to `max_attempts` with backoff
  (this is where the small-face path lives). Track lost ⇒ binding evicted.

For five children queuing over ten seconds: **5 embeddings instead of ~200.**

### 4.5 A bug this design kills on the way past

`TrackAccumulator.hits` is keyed by **`person_id`**
([`faces/accumulator.py:41`](../../../src/qorgan/faces/accumulator.py)), and one
accumulator is shared across the whole camera. So weak top-1 hits from **different
children** currently corroborate each other: a crowd of unknowns can vote a stranger into
being pupil X — and that closes a meal session.

Keying by `(track_id, person_id)` falls straight out of per-track binding.

### 4.6 `qorgan identity camera-report <camera>`

See §2.6. Reports the face-size distribution PER STREAM and
the fraction clearing the recognition gate. It answers *"can this camera ever recognise
anybody?"* — a question that is about optics, not thresholds, and that the legacy never
asked in eighteen attempts at tuning.

### 4.7 `qorgan plan-workers`

`config/workers.yaml` was measured on a 4 GB RTX 3050 and under-uses the school's 4070. The
numbers themselves are sound — `vram_spike.py` imports torch at module level, so InsightFace
really was on the GPU when they were taken (§3). But they are numbers **for the wrong GPU**,
and we cannot measure a GPU we do not have, and we will not guess.

`scripts/vram_spike.py` becomes `qorgan plan-workers`: it runs **on the machine it will run
on**, measures the real per-process cost, and writes `config/workers.yaml`. The current
grouped config ships as the fallback.

It must now account for the YOLO the canteen cameras gained in §4.4.

---

## 5. Shape

```
src/qorgan/config/identity.py    FaceGate, RecognitionPolicy, SoftAccumulator,
                                 FaceModelSettings, BindingSettings   (moved out of canteen.py)
src/qorgan/identity/tracks.py    pure: assign_faces_to_tracks(faces, person_boxes)
src/qorgan/identity/binding.py   pure: the bind / retry / evict state machine.
                                 No GPU, no clock, no DB.
src/qorgan/identity/service.py   the impure shell: recognizer + gallery + bindings
src/qorgan/identity/report.py    gallery-report: the matrix, duplicates, extrapolation
```

`config/canteen.py` keeps only what is about **meals** — `SessionRules`,
`MealOutcomeRules`. A hall camera can then import `identity` without dragging in `canteen`,
which is the entire point of the move.

**Database:** `persons.full_name` becomes nullable (Alembic migration). One pure
`display_name(person)` → `"Ученик 333, 5-А"`, used by the web, the reports and Telegram
alike — so the fallback name is written once, not three times.

---

## 6. Testing

| Unit | How it is tested |
|---|---|
| `assign_faces_to_tracks` | pure — fixtures of boxes, no GPU |
| binding state machine | pure — a fake clock; bind, reject, retry, exhaust, evict |
| import refuses a bad filename | a file named `photo.jpg` raises, naming the file |
| person type comes from the folder | `учитель/student_469_….jpg` imports as STAFF |
| unenrollable photos are itemised | a faceless image appears in the report, is not silently dropped |
| duplicate detection | two embeddings of one person at 0.78 are flagged; two children at 0.47 are not |
| `merge` | photos, embeddings and sessions re-point; `drop_id` deactivates |
| accumulator is keyed by track | two different tracks weakly matching X do **not** corroborate |
| one embedding per track | a spy recognizer counts `embed` calls across 40 frames of one track: exactly 1 |
| the gap collapse is pinned | a gallery with one human under two ids yields `gap < min_gap` → `AMBIGUOUS`; after `merge`, the same face is `ACCEPTED`. This is §2.7 turned into a regression test, so the failure cannot return silently. |
| onnx really is on CUDA | the guard builds a session and asserts `session.get_providers()[0] == "CUDAExecutionProvider"` — `get_available_providers()` is not consulted, because it lies (§3) |

## 7. What this spec does not do

- It does not set `min_score` to a settled value. It sets a **floor** and names the ceiling
  as unmeasured (§2.3).
- It does not merge anybody. It reports six pairs and waits for the school (§4.2).
- It does not touch the bullying detector, the eval harness, or `camera_loop.py` — those
  are Spec B's, and the two specs share no files.

# What was measured on this footage, before anything was designed

Every number here came out of `test_camera.mp4` itself on 2026-08-12, on the machine the
module has to run on. Nothing below is an estimate, and where a measurement did **not**
support the thing we wanted to build, it says so rather than being rounded up.

The reason this file exists first: the `qorgan-ai-main` classroom package next door
carries a docstring saying *«We hold no recording of a lesson. Not one.»*, and every
threshold in its `config/classroom.py` is explicitly labelled a guess as a result. That
recording now exists. This is what it says.

---

## 1. The recording

| | |
|---|---|
| Container / codec | MP4 / HEVC |
| Resolution | 2560 × 1440 |
| Frame rate | 20.00 fps, constant |
| Frames | 62 901 |
| Duration | 3 144.85 s (52 min 25 s) |
| Size | 957 MB (≈2.44 Mbit/s) |
| Wall clock | **2026-08-07 09:54:58 → 10:47:22**, Friday |

**The wall clock is not a guess.** It is read out of the burned-in overlay
(`video/clock.py`). Two independent checks:

* **Linearity.** Decoding 200 consecutive frames, the clock ticks exactly once per 20
  frames and never skips or repeats a second. Over the whole 52 minutes the accumulated
  drift between the overlay and the file's own timeline is **0.20 s** — below the
  overlay's own 1 s quantisation. This is what licenses `WallClock.at()` to be a
  multiplication instead of an OCR call per frame.
* **Date order.** `08-07-2026` is ambiguous. The overlay also prints `Fri`, and only
  **2026-08-07** is a Friday (2026-07-08 is a Wednesday). The camera is therefore
  **MM-DD-YYYY**, established by evidence and stored in the camera profile — `parse()`
  has no default date order and will not guess one.

Reader accuracy: **13/13 frames** across the full 52 minutes, held-out frames included.
Weakest-glyph NCC margin ≈ 0.22 on every correct read; the module refuses any read below
0.07 rather than returning a plausible wrong time.

### The 15-minute working slice

`clip_15min.mp4` — `-ss 300 -t 900 -c copy`, landing exactly on a keyframe, so there is
no timestamp drift to correct.

| | |
|---|---|
| Wall clock | **09:59:58 → 10:14:57** |
| Frames | 18 002 |
| Chosen because | full class present (mean 9.0 people), the densest hand-raise period, and a sustained out-of-seat episode at 10:12–10:13 |

---

## 2. The scene

A ceiling-corner wide-angle camera in a **small** classroom. Six desk positions in two
columns; **9 pupils and 1 adult**.

Three properties of this geometry drive the whole design:

1. **The board is behind/under the camera.** Pupils face *toward* the lens, so faces are
   near-frontal — the opposite of the corridor case. Someone at the board appears at the
   bottom edge of the frame, large and foreshortened.
2. **The room is empty for the first ~110 s** (09:54:58 – 09:56:48). That is a free
   empty-room reference frame for desk/seat/zone calibration, with no people to subtract.
3. **The adult sits at a front desk with a laptop, back to the camera**, for most of the
   lesson. Whatever the teacher metric ends up being, on *this* recording its answer is
   largely "seated" — which is a finding, not a failure.

---

## 3. Detection and pose — YOLO11m-pose, Apple M2 Pro (MPS)

11 frames spread across the lesson.

| imgsz | s/frame | people found (median) |
|---|---|---|
| 960 | — (first run, includes model load) | 9 |
| **1280** | **0.109** | 9 |
| 1920 | 0.172 | 9 |

Person count is 9–10 in every populated frame, matching a manual count. Box confidences
0.85–0.94 at 1920, 0.56–0.93 at 960 — the low tail at 960 is the pupil furthest from the
camera. **1280 is the operating point**: same people found, 1.6× faster than 1920.

### The measurement that matters most: shoulder width

Every threshold in a classroom must be scaled by the person's own shoulder width, because
a front-row pupil is 2–3× the size of a back-row one in the same frame.

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| Shoulder width (px) | 19.2 | 70.2 | **89.2** | 119.7 | 232.2 |
| Person box height (px) | 110 | 196 | 270 | 366 | 533 |
| Mean keypoint confidence | 0.3 | 0.6 | 0.7 | 0.7 | 1.0 |

The neighbouring codebase declares `MIN_USABLE_SHOULDER_PX = 8.0`, below which no ratio
of that width means anything. **The p25 here is 70 px — nearly 9× that floor.** Shoulder-
normalised geometry is comfortably supported on this camera; it is not a marginal call.

### Throughput budget

Full sequential decode of all 62 901 frames: **107.9 s (583 fps)**. Decode is *not* the
bottleneck. At imgsz 1280 the pose model costs 0.109 s/frame, so:

| sampling | frames (52 min) | pose time |
|---|---|---|
| 2 fps | 6 290 | ≈ 11 min |
| 5 fps | 15 725 | ≈ 29 min |

---

## 4. Faces and identity — the result that changed the design

Detection (InsightFace `buffalo_l`, whole frame, det_size 1280):

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| Face height (px) | 30.8 | 53.5 | **64.0** | 73.7 | 95.2 |
| Detector score | 0.5 | 0.8 | 0.8 | 0.9 | 0.9 |

5–8 faces found per frame out of 9 pupils.

**64 px is not 11.5 px.** The neighbouring codebase's refusal to identify anyone in a
classroom rests on a corridor measurement — 14 970 faces, median 11.5 px, zero
recognised. That measurement is real and it does not transfer to this camera: these
faces are 5–6× larger and near-frontal.

**But bigger faces did not deliver recognition.** Every one of the **141/141** roster
photos produced a usable embedding (zero failures). Matching each face seen in the room
against that full 141-pupil gallery, single-frame:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| Best cosine | 0.1 | 0.3 | **0.3** | 0.4 | 0.6 |
| Gap to runner-up | 0.0 | 0.0 | **0.1** | 0.2 | 0.3 |

ArcFace embeddings normally want ≈0.4–0.5 to call two faces the same person. A median
best score of 0.3 with a **0.1** margin over the *next* candidate is not identification —
it is a coin toss with a preference.

Two caveats, both stated so nobody over- or under-reads this:

* The gallery was all **141 pupils across 13 classes**, not the ~9 actually in the room,
  and some observed faces belong to the adult, who is not in the roster at all. A gallery
  restricted to one class is a materially easier problem.
* This measured **single-frame** matching. The real pipeline gets hundreds of observations
  of the same seat and can aggregate before deciding, which is a different and much
  better-conditioned question.

**What this means for the design.** Face evidence is a *contribution*, not the mechanism.
The identity mechanism has to be **seat-based** — fixed desks in a fixed room, with a
one-time seat map — using aggregated face evidence to corroborate, and the module must be
able to return *"identity not established"* and fall back to anonymous seat labels rather
than attach a child's name to a 0.1 margin.

---

## 5. What is actually in this lesson

315 samples, one every 10 s, whole file.

| | |
|---|---|
| People per frame | min 0, median 9, max 10 |
| Empty frames | 11 (the first 110 s) |
| Frames with a wrist above a shoulder line | **80 of 315 (25 %)**, max 2 at once |
| Frames with a head near the desk line | **295 of 315 (94 %)** |
| Tallest person box | median 450 px, p95 594 px, max 826 px |

The last two rows are the important ones, and they are **negative results about the naive
predicates**, obtained before writing them properly:

* A raw "wrist above shoulder line by 0.35 shoulder widths" fires on **a quarter of all
  sampled frames**. Children sit with a hand propping their chin. The neighbouring
  config's `above_shoulder_ratio: 0.35` — self-labelled a guess — is far too permissive
  on this footage. A usable hand-raise predicate needs the wrist above the **head**, an
  elbow condition, and a hold, not a shoulder-line crossing.
* A raw "nose below shoulder line" head-down test fires on **94 %** of frames, because
  looking down at a desk is what pupils do all lesson. "Lying on the desk" cannot be a
  single-frame nose test; it needs the pupil's own upright baseline and a duration.

Out-of-seat episodes are present and localisable: tallest-box excursions above 650 px at
10:08:38 and a sustained run through 10:10:58–10:13:28 — both inside the 15-minute slice.

---

## 5a. Seat discovery: a method that got *worse* with more data, and the fix

The first implementation clustered anchors with hand-rolled single-link agglomerative
clustering under a scale-normalised distance. It gave the right answer on a sparse sample
and then failed as the data improved:

| anchors fed to it | seats found | truth |
|---|---|---|
| 10 s sampling, 52 min (2 699 anchors) | **9** ✓ | 9 |
| 2 fps, 15 min (15 556 anchors) | **3** ✗ | 9 |
| 2 fps, 52 min (53 244 anchors) | **5** ✗ | 9 |

**An algorithm that becomes less correct as you give it more evidence is the wrong
algorithm, not a badly tuned one.** The cause is single-link chaining: it joins two
clusters through *any* path of intermediate points, and a pupil leaning toward a
neighbour lays a continuous trail of anchors between two desks. Sparse sampling hid the
trail; dense sampling completed it.

This was caught by an automatic check, not by eye: `discover()` compares its own seat
count against the **detector's** median people-per-frame — an independent signal produced
by a different mechanism — and refuses to return a confident answer when they disagree.

### The fix: DBSCAN in perspective-corrected space

Two changes, both measured:

**1. Perspective correction (`room/perspective.py`).** Shoulder width across one frame
runs 19–232 px, so no single distance threshold means the same thing at the front and the
back. Fitting shoulder width as a function of image row gives **R² = 0.945** on this room,
and warping by `u = x/s(y)`, `v = ln(s(y))/a` turns pixel distance into shoulder widths
everywhere:

| | p10 | median | p90 |
|---|---|---|---|
| raw shoulder width (px) | 54.1 | — | 133.8 |
| scale ratio after warp | 0.71 | 1.00 | 1.26 |

A 2.5× spread becomes ±25 %, which is what makes one `eps` legitimate.

**2. DBSCAN instead of single-link.** A point only propagates a cluster if it has
`min_samples` neighbours within `eps`. A seat is occupied for most of the lesson; a trail
is traversed in a second. Trails become noise instead of bridges.

Result on the **full 52 minutes**: **9 places**, matching the 9 people the detector sees,
with `eps` selected automatically by that agreement. The eps sweep shows the plateau:

```
eps      0.20  0.25  0.30  0.35  0.40  0.50  0.60  0.75
seats       9     7     7     7     6     2     2     0
```

Discovered centres agree with the independent 10 s-sample run to within a few pixels
(e.g. adult seat (832, 1295) vs (832, 1287)).

## 5b. Two defects the verification overlay found

The annotated video (`report/overlay.py`) exists to falsify the numbers, and it earned its
place twice:

* **The adult's seat circle was drawn at (0, 0)** — `TeacherRecord` carried no centre, so
  every consumer of "all the places" silently lost one. Invisible in the JSON.
* **The overlay disagreed with the report it was verifying.** It labelled a seat «встал»
  where the artefact counts zero stands, because it warmed per-seat baselines over the
  whole file while the pipeline uses only the detected lesson window. A verification
  artefact that disagrees with its subject is worse than none.

The second fix exposed a genuine reproducibility bug: `window_seconds` was stored **rounded
to 0.1 s**, so replaying the analysis from the artefact selected one frame fewer and every
seat's histogram came out one observation short. The window is a reproducibility parameter,
not a display value, and is now stored at full precision.

`overlay.agrees_with_artefact()` now re-derives the classification and asserts the
per-seat state histograms match the artefact exactly — **0 mismatches across all 8 seats**
on both the 15-minute clip and the full lesson.

## 5c. Full-lesson result

50 minutes (09:57:19–10:47:22), 6 007 analysed frames, 53 170 person-observations.

| | |
|---|---|
| seats found | 9 (8 pupils + 1 adult), plausible ✓ |
| per-seat coverage | 0.85 – 1.00 |
| observations at no seat | 2 152 (4.0 %) |
| seats that never settled | 0 |
| hand-raise episodes per pupil | 0 – 4 |
| adult | 96.5 % seated, 5.1 % out of frame, 5 position changes |

Re-analysis from the detection cache takes **5–7 s**, so a threshold change costs seconds
rather than the ~15 minutes the model costs. That is what makes the thresholds arguable.

## 6. Environment

Apple M2 Pro, 16 GB, macOS 25.5. **No NVIDIA GPU, no CUDA.** Python 3.13 venv at
`../.venv`: torch 2.13 (MPS available), ultralytics 8.4.118, insightface 1.0.1,
onnxruntime 1.28 (CoreML + CPU execution providers), opencv 5.0.0.

No `GEMINI_API_KEY` or `OPENAI_API_KEY` is set, so the LLM summary path must have a
deterministic offline fallback that produces a real report, not a placeholder.

---

## 7. Roster and photos

`roster.csv` — 141 rows, `external_id,full_name,class_name`, UTF-8 with BOM on the header.
13 classes (1-А … 9-А). `student_photos/<class_name>/<N>.jpg|jpeg` where `N` is the
numeric part of `student_<N>`.

Cross-check: **141 roster entries, 141 photos, 0 missing, 0 orphaned.** One photo per
pupil — so there is no second image to validate an embedding against, which is part of
why §4 came out the way it did.

---

## 8. The LLM summary, tested against the live API — a negative result

Tested 2026-08-13 with a real Gemini key, `gemini-3.6-flash`, on the 50-minute artefact.
The bundle sent is **32 KB of JSON — numbers only, no frames, no faces**.

**It ran, and it passed the guard.** 350 numbers in the generated Russian text, every one of
them present in the input bundle. `NumberCheck` reported no unbacked values.

**The output was still wrong.** About the adult it wrote:

> «Сидячее положение зафиксировано в 96,5% случаев (6,0 минуты / 357,5 секунды)»

96.5 % of 48 observed minutes is **46.3** minutes, not 6.0. The model had joined two
different quantities that happened to share a name:

| field | value | what it actually was |
|---|---|---|
| `seated_share` | 96.5 % | share of observations at the desk |
| `seconds.seated` | 357.5 s = 6.0 min | only *qualifying episodes* of the SEATED state |

Both numbers were real and both were in the document, so a guard that checks whether a
number **exists** could not catch it. **The guard verifies existence, not attribution.**
That sentence is the single most useful thing this test produced.

### Two defects it exposed, both ours

1. **One name, two quantities.** The ledger emitted `seconds` per state (episode-based)
   while shares were observation-based. Fixed in schema **1.1**: `observed_seconds_by_state`
   (sums exactly to `observed_seconds`) and `episode_seconds` (always a subset). The old
   key was **removed, not aliased**, so a 1.0 consumer fails loudly.
2. **The same trap one level up**, found while verifying the first fix: the adult's
   aggregate summed four states (SEATED + HEAD_DOWN + TURNED_AWAY + HAND_RAISED = 46.3 min)
   and was called `seated`, while the SEATED state alone is 8.6 min. Now `at_desk_*`, and
   every share carries its denominator (`..._of_observed`, `..._of_lesson`).

A third defect appeared *during* the rename and was caught by a test rather than by review:
the teacher's position sentence was silently dropped for one run, because a condition had
been updated to the new key while its body still read the old one. A missing sentence is
invisible in review.

### What this changed about the design

The LLM added **no information** — it reformatted the same numbers more verbosely and lost
a section. Its only measured effect was to introduce an error. So its job was inverted:

> **The model selects and explains; the code supplies every number.** The model returns
> structured output whose free-text fields must contain no digits at all, and the rendering
> of every figure is done from the bundle by our own code.

Misattribution then becomes structurally impossible, because the model never types a
numeral. `NumberCheck` stays as a second line of defence over the rendered text.

The deterministic generator remains the record, and the report now prints the correct
pairing itself — «за столом — 96,5 % этого времени (46,3 минуты)» — so that no reader,
human or model, has to derive it.

**Cost, for the record:** one 32 KB bundle per lesson. Sending it means children's
behavioural data leaves the country → ЗРК 94-V applies even though the bundle contains no
imagery. That is a decision for the school, not a default.

---

## 9. The accumulation layer, measured on the artefacts that exist

`cabinet/store.py` was built against the real `out/` directory rather than against
fixtures, and three of its numbers came out of that directory rather than out of a design.

### 9.1 Cross-lesson place matching works, and the failure mode is visible

Two recordings of the same room (`test_camera.mp4`, 50 min, and `clip_15min.mp4`, a
15-minute slice), analysed independently — each discovering its own seats with its own
`seat_id` numbering:

| | |
|---|---|
| places created from the first recording | **9** (8 pupils + the adult) |
| places recognised in the second | **9 of 9** |
| unmatched / ambiguous | **0 / 0** |
| every seat shifted by 600 px (simulated camera move) | **0 matched, 9 new** |

The gate is 1.0 shoulder widths with a 2.0× margin to the runner-up
(`PLACE_MATCH_GATE_SCALES`, `PLACE_MATCH_MARGIN`). The last row is the important one: a
moved camera produces new places rather than a silently inherited history, so the defect
announces itself on a particular date instead of becoming a trend.

### 9.2 Two files of one lesson: folding agrees on observations, NOT on episodes

The D14 pair exists twice over: as one artefact of the concatenated recording, and as two
artefacts of the two files which the cabinet chains into one session
(`continues_lesson_id`). Two independent routes to the same hour, compared per place:

| place (x, y) | observations, concatenated | observations, folded | hand-raises |
|---|---|---|---|
| (838, 481) | 5 550 | 5 559 | 1 / 1 |
| (904, 583) | 4 644 | 4 524 | 23 / 28 |
| (1039, 341) | 3 722 | 3 718 | 0 / 0 |
| (1409, 567) | 7 613 | 7 614 | 0 / 0 |
| (1419, 452) | 4 327 | 4 327 | 15 / 15 |
| (1612, 472) | 9 304 | 9 658 | 6 / 6 |

**Observation counts and coverage agree to within 0.1–4 %; episode counts do not.**
«Вставал» came out 3 vs 0, 17 vs 14, 8 vs 3. That is not a bug and it is the reason
`_SummedLedger` refuses to invent episode objects: an episode is a *duration* with a hold
and a gap (`ledger.py`), and the seam of a DVR file roll cuts one of them in two while each
part re-establishes its own posture baseline from scratch. Observations add; episodes do
not. So the cabinet adds the artefacts' own episode COUNTS and recomputes the index from
the summed observation histogram with `metrics/activity.py` itself — and where the two
routes disagree, the concatenated analysis (`analyse-session`) is the better measurement
and is the one to prefer when it exists.

### 9.3 Two files of one hour are indistinguishable except by the clock

`clip_15min.mp4` is a `-c copy` slice of `test_camera.mp4`. Their content hashes differ, so
`run_id` differs, so nothing in the per-lesson pipeline can tell they are the same hour —
importing both doubles that hour in every weekly counter, and no later reader could detect
it. The wall clock can: 09:59:58–10:14:58 sits inside 09:54:58–10:47:22. **The overlap
refusal fired on the real `out/` directory the first time it was run**, which is how it
earned its place rather than being argued for.

The neighbouring case is the opposite and must not be refused: the D14 seam. File one
starts 10:17:59 and runs 818.00 s, ending at 10:31:37; file two starts 10:31:36. A
back-to-back pair therefore *looks like* a one-second overlap, because both clocks are
known only to the second. Hence `CONTINUATION_TOLERANCE_SECONDS = 2.0` and the rule that
continuation is asked **before** overlap — a window of `[0, gap]` misses the exact pair the
check exists for.

### 9.4 The seam was order-dependent, and only running it the other way showed it

Measured on the same two files, 2026-08-16, by importing them in each order into a fresh
database:

| import order | recordings | **sessions** | warning printed |
|---|---|---|---|
| `D14_short_only`, `D14_long_only` | 2 | **1** | «…считается ПРОДОЛЖЕНИЕМ…» |
| `D14_long_only`, `D14_short_only` | 2 | **2** ✗ | none |

The check looked only backwards — «какая запись кончается прямо перед этой» — so the
earlier half arriving second matched nothing. The result is «занятий 2» for an hour that
had one and every per-lesson figure of it halved, and unlike the overlap case **nothing
refuses and nothing warns**, because a lesson followed by another lesson is exactly what a
normal school day looks like. A directory back-fill imports in glob order and an operator
types two filenames in whichever order they read them, so this is the ordinary path.

Fixed with the mirror check (`_immediately_after`), which points an already-stored
successor back at the new lesson and is fed to the overlap rule alongside the backward
match — otherwise a seam of −2 s (inside `CONTINUATION_TOLERANCE_SECONDS`, outside
`OVERLAP_TOLERANCE_SECONDS`) would refuse in one order and pass in the other. Both orders
now yield one session and identical folded counters per place; the ordinals differ, because
which file first revealed a place is genuinely different, and ordinals are display numbers.

### 9.5 Three places where "not measured" was rendering as a confident number

All three found by reading the generated pages rather than the code.

| where | rendered | true value |
|---|---|---|
| class card for a place whose only lesson is undated | «поднимал руку — 0 · видимость от 0 %» | not accumulated; the pupil raised his hand twice in that lesson |
| import of an artefact whose `teacher.centre` is null | «мест 8 … без привязки 0» | 9 places in the other two runs of the same lesson; the adult was dropped in silence |
| adult row from a `classvision/1.0` artefact | `90.0` under the column «за столом, %» | a share whose denominator that schema never named — §6b of `INTEGRATION.md` is the record of why that name was abandoned |

The first two are rule 3 and rule 4 of this project respectively. The third is the same
name-collision class as §8 below, one consumer further along: the fix there was to rename
the field so the name carries the denominator, and a `get(new, get(old))` fallback in the
reader quietly undid it.

### 9.6 Two more confident zeros, both in the column «выходил к доске»

Found the same way as §9.5 — by reading the generated pages against the artefacts they were
generated from, on a cabinet rebuilt from the only two analysed lessons that exist.

**The first is structural and applies to every page of camera01.** That camera's board is
behind the lens, `configs/camera_01.yaml` records `board_zone: null` as a measured decision,
and `ledger.classify` therefore cannot emit `AT_BOARD` in any frame. All **8 of 8** pupil
places carry `counts.board_visits: 0`, and the artefact says why in `lesson.unmeasured`
(«стоял у доски: зона доски не задана для этой камеры»). The teacher half of the same
document already honoured that — `presence.board.minutes_of_lesson: null` beside
`zone_configured: false` — and §12.6 records what a confident zero there cost. The pupil half
printed `0`, in a column headed with an observable event, on the week table, the per-lesson
table, the totals tiles and the class card.

| where | printed | true value |
|---|---|---|
| class card, all 8 pupils | «выходил к доске — 0» | not measurable on this camera at all |
| «за всё время» tile | `0` | — |
| week / session tables | `0` | — |

`cabinet/lessons.py::COUNTER_VOIDED_BY` now maps the analyser's own `unmeasured` sentence
onto the counter it voids, and every counter cell on every cabinet page goes through
`report._counter_cell`. The mapping is a **string join** and is admitted as one:
`test_the_unmeasurable_counter_map_matches_what_the_analyser_emits` fails if `pipeline.py`
rewords that sentence, which is the cheapest way to make a join of this kind reviewable.

**The index inherits the zero and is now labelled a lower bound.** `metrics/activity.py`
weights «видимые действия» at **0.30** and computes it as
`min(1, (hand_raises + stands + board_visits) / 3)`. On camera01 one of the three summands is
structurally absent, so place 3's participation term of **33 %** (1 hand, 0 stands, 0 board
visits) and its index of **72.7** are floors rather than measurements of the same quantity
D14's indices measure. The weights were not changed — they are declared, not fitted, and
re-deriving them per camera would make two lessons incomparable in a way no reader could
see. What changed is that no page prints the index without saying so.

**The second zero is not structural, and it is worse.** D14 *can* see its board, so the
column is not voided — and it still comes out **0 for all 5 pupil places**, while the same
artefact's `board_occupancy` reports somebody at that board for **30.4 minutes, 52.3 % of
the lesson**, of which the adult accounts for **4.0**. Roughly twenty-six minutes of pupils
at a board, under a column saying no pupil went to one.

The cause is what the counter is *about*: `AT_BOARD` is a state of a PLACE, and a child who
walks to the board has left theirs — the observation lands in «уходил с места» or in the
**18.0 %** of that recording's observations that belong to no place at all. The zero is true
of the ledger and false of the room. It is **not repaired here**: repairing it means changing
`ledger.classify`, which by `run_id` construction is a new measurement of every lesson ever
analysed, not a display decision. What is repaired is the reader's exposure to it — the
contradicting number is printed beside the zero, out of the same artefact, on the lesson
card, the class page and every place page of that class.

### 9.7 Two lessons, two rooms, and the sentence the cabinet was one glance from producing

The cabinet was rebuilt from the only two analysed lessons in the tree, imported exactly as
an operator would import them:

| | camera01 | D14 |
|---|---|---|
| date / ISO week | 2026-08-07 / 2026-W32 | 2026-08-15 / 2026-W33 |
| pupil places found | **8** | **5** |
| class key stored | `-` (`not_stated`) | `-` (`not_stated`) |
| attestations | 0 | 0 |

**Neither operator typed `--class`, so both lessons are filed under the same key — and that
key is not a class, it is the absence of one.** The previous index page emitted one flat card
per (room, class) pair headed «Класс - · комната camera01», so the two cards sat adjacent,
one week apart, under one class name, with 8 places on one and 5 on the other and nothing on
the page saying they share no child. Every number was correct and nothing was ever summed;
the false sentence was available to the reader for free.

The fix is organisational rather than arithmetic — class first, room second, with
`lessons.ClassAcrossRooms.warning_ru()` assembled **from the store's own room keys, dates and
place counts** so that it cannot describe a configuration the database does not hold, and
disappears by itself once the keys are separated. It is rendered on all 21 generated pages,
and `test_a_class_key_spanning_two_rooms_is_warned_about_on_every_page` counts them.

What IS comparable across that pair is the pair of *recordings*, and it now has its own page
(`lessons.html`) on which every column carries what it is NOT — «ученических мест» is not a
class size, «видимость» is not attendance, «вне мест» is not children walking about. The one
sentence at the top of it is the whole design: a row changes when the number of children in
frame changes, not when one child's behaviour does.

Two facts that page surfaced, both of which had been sitting in the artefacts:

* The adult's position split is **not comparable across these two cameras either**. camera01
  has no `teacher_zone`, so `presence.state_share_of_lesson_percent["at_desk"]` is `0.0`
  while `metrics.at_desk_share_of_observed` for the same man in the same document is
  **96.5 %** — two true numbers, two denominators, two subjects (§8's collision one consumer
  further along). The two are now on different pages under different headings, and the
  unmeasurable states print «не измерялось» with the note that their time inflated
  «среди учеников» (**92.2 %** on that camera) by an unknown amount.
* The follower's coverage differs by a factor of two between the rooms — **93.3 %** of the
  lesson attributed on camera01 against **48.5 %** on D14 — so even «где был взрослый» is a
  comparison of two cameras before it is a comparison of two lessons.

**No page states or implies a trend**, and that is asserted rather than intended:
`test_no_page_states_or_implies_a_trend_when_none_can_be_computed` scans every generated page
for `metrics/trend.RU_DIRECTION`'s own vocabulary and for a joined sparkline path, reading
the banned strings out of `trend.py` so that a new direction word cannot be added without the
test noticing.

---

## 10. One lesson, two files: what merging them before analysis actually changed

The D14 recorder split this lesson in two. Measured on the pair (`session.py`,
2026-08-16), both files at 2 fps, imgsz 1280, from the detection cache:

| | file 1 | file 2 | session |
|---|---|---|---|
| wall start (from the file name; the overlay is unreadable on this camera) | 10:17:59 | 10:31:36 | 10:17:59 |
| duration (frames / fps) | 817.998 s | 2 903.865 s | **3 720.865 s** |
| seam | — | — | **−0.998 s (overlap)** |
| analysed frames in the lesson window | 1 424 | 5 556 | **6 977** |
| places found | 6 | 6 | 6 |
| lesson window | 11.9 min | 46.3 min | **58.1 min** |

**The seam is negative and that is the normal case, not an anomaly.** File one ends at
10:31:37.00 and file two's name claims 10:31:36 — both stamps are whole seconds, so a
clean handover reads as a one-second overlap. `MAX_SEAM_GAP_SECONDS = 2.0` admits it and
refuses anything larger in either direction; the same number, for the same reason, is
`cabinet/store.CONTINUATION_TOLERANCE_SECONDS`.

### 10.1 The thing this was built to prevent, demonstrated

Analysed separately, the two files number **the same six desks differently**:

| desk (centre in the session run, px) | file 1 alone | file 2 alone | session |
|---|---|---|---|
| (1039, 341) | seat 1 | seat 1 | seat 1 |
| (1409, 567) | seat 2 | seat 2 | seat 2 |
| (1419, 452) | **seat 5** | **seat 3** | seat 3 |
| (1612, 472) | **seat 3** | **seat 4** | seat 4 |
| ( 838, 481) | **seat 4** | **seat 5** | seat 5 |
| ( 904, 583) | seat 6 | seat 6 | seat 6 |

(The same desk's centre also moves a few pixels between runs — e.g. 1584 / 1617 / 1612 for
seat 4 — because each run clusters different evidence. Matching is by proximity; the ids
are what the reports would have printed.)

**Three of the six ids swap.** Adding file 1's «место 3 подняло руку» to file 2's would
add two different children, and nothing in either artefact says so — the numbering is a
reading order over whatever that file's clustering found, and the two files chose
different `eps` (0.6 and 0.4 against the session's 0.35) because they hold different
amounts of evidence. This is not a bug in the numbering; it is what "seat 3" means, and
it is why the merge has to happen **before** discovery rather than after analysis.

### 10.2 Comparing the session with the long file alone, on the seconds they share

Counting only session seconds ≥ 817 — exactly the long file's own window — so the extra
13 minutes cannot explain anything:

| place (сессия / один файл) | руки, эпизодов | seated, наблюдений | что разошлось |
|---|---|---|---|
| (1409, 566) | 0 / 0 | 5 108 / 5 112 | ничего заметного |
| (1419, 452) | 11 / 11 | 2 363 / 2 483 | ничего заметного |
| (1039, 341) | 0 / 0 | 2 584 / 2 495 | «вне места» 127 против 391 наблюдений |
| (1612, 472) | 4 / 4 | 5 873 / 6 025 | «встал» 3 против 8 эпизодов |
| ( 838, 481) | 0 / 0 | 112 / 3 573 | **head_down 3 506 против 0 наблюдений** |
| ( 904, 583) | 23 / 28 | 2 339 / 1 819 | «встал» 21 против 1, «вне места» 2 против 16 |

**Where the two runs disagree, the cause is the per-seat baseline, and it was isolated
rather than assumed.** Holding the session's seats fixed and moving only the start of the
classification window from 106 s to 817 s reproduces the whole difference:

| place | upright-head baseline (106 s → 817 s) | head_down observations |
|---|---|---|
| (838, 481) | **1.635 → 0.719** | **5 344 → 0** |
| (904, 583) | 0.814 → 0.930 | 40 → 80 |
| (1409, 566) | 0.713 → 0.602 | 0 → 0 |

`states.Baselines` settles on a seat's **first 20 observations and then freezes**, and the
merge moves where "first" is. At (838, 481) the first 20 observations of the session
include six at 153.5–156.0 s whose shoulder width is **11–26 px against that seat's usual
59 px** — a badly-resolved detection of somebody arriving — and because the upright datum
is the **p75** of head heights, those six drag it from ~0.72 to 1.635. Every later
observation then sits more than `head_drop = 0.55` below it, and the seat reads as
head-down for the whole lesson.

**That is a defect in the settling rule, exposed by the merge and not caused by it**: the
window had no quality gate on the scale. The merge is what made it visible — a single file
happened to start on clean observations.

#### 10.2a The settling rule now has a quality gate, and what it cost

The defect above was left documented rather than patched, on the ground that changing it
changes every artefact ever produced. That ground does not hold: `run_id` hashes
`provenance.thresholds`, so a changed settling rule *is* a new artefact by construction —
which is rule 6 of this project working exactly as designed, not a cost. What was actually
being preserved was a rendered sentence about a real child reading
**«22 эпизода с опущенной головой (суммарно 44,5 минуты)»** and an activity index of
**51 against 94.7–98.2** for every other place in the room, in the report a psychologist
receives, with a coverage figure of 78.8 % beside it saying the place was well seen.

`head_up` is head height **divided by** shoulder width, so a detection with a broken scale
does not add noise to a baseline — it multiplies it. Six readings of 11–26 px reported
`head_up` of 2.41–3.56 where the same child upright reads 0.6.

Two additions to `states.Thresholds`, both derived rather than tasted:

* `settle_scale_band = 0.6` — a settling observation whose shoulder width departs from the
  settling window's **own median** by more than this factor is discarded. Derived from the
  quantity it protects: an admitted reading a factor `r` too small inflates `head_up` by
  `head_up × (1/r − 1)`, and that must stay under `head_drop`, so with the measured upright
  `head_up` on this camera (median 0.63 across the six D14 places) and `head_drop = 0.55`
  the requirement is `1/r < 1 + 0.55/0.63`, i.e. **`r > 0.535`**. 0.6 is that bound with
  margin. Applied symmetrically, because a merged or doubled box gives a scale that is too
  *large*, which deflates `head_up` and corrupts `seat_shoulder_y` just as thoroughly.
  The band is **relative to the window's own median**, so it needs no per-camera
  calibration and a child genuinely detected small all lesson keeps every observation.
* `settle_window_limit = 5` — a place that cannot produce `settle_observations` mutually
  consistent readings within 100 observations (50 s at 2 fps) is **refused**, not settled:
  every state stays `UNKNOWN`, `uncertainty.seats_never_settled` counts it, and the new
  `ledger.settle_refusal` carries the reason in Russian. A place settled on rubbish is
  counted nowhere, which is the whole argument for preferring an unsettled one.

Measured effect on the D14 session (`--room configs/camera_d14.yaml`), same footage, same
seats, only the settling rule changed:

| place | upright-head baseline | head_down, наблюдений | эпизодов | индекс активности |
|---|---|---|---|---|
| ( 838, 481) | **1.635 → 0.693** | **5 343 → 0** | **22 → 0** | **51 → 96.9** |
| (1419, 452) | 0.651 → 0.648 | 65 → 64 | 0 → 0 | 97.9 → 97.9 |
| (1612, 472) | 0.594 → 0.594 | 7 → 7 | 0 → 0 | 94.9 → 94.9 |
| ( 904, 583) | 0.814 → 0.814 | 40 → 40 | 0 → 0 | 94.7 → 94.7 |
| (1409, 566) | 0.713 → 0.713 | 0 → 0 | 0 → 0 | 98.2 → 98.2 |

All six places still settle (`seats_never_settled: 0`); the cost is that (838, 481) settles
on its **26th** observation instead of its 20th — three seconds of that child's lesson
recorded as `UNKNOWN` instead of forty-four minutes of «лежит на парте» that never
happened. The corrected session now **agrees with the long-file-only run**, which had
`head_down = 0` at that place — the cross-check that exposed the defect in the first place.

Pinned by `tests/test_geometry_and_states.py`, which replays the measured twenty scales.

### 10.3 Two seam defects found by measuring the merged timeline

* **The sampling phase does not survive a seam.** Each file is sampled from its own first
  frame, so part one's last sample landed at 817.948 s and part two's first at
  817.99994 s — **52 ms apart** against a nominal 0.5 s. Both frames are real, but the
  ledger charges every observation `sample_interval` seconds, so that pair bought 1.0 s of
  «наблюдалось» for 0.05 s of footage at every seat. `detect_session` now requires a full
  sampling interval since the previous part's last kept frame; the minimum spacing over
  the whole merged timeline is 0.4998 s, and the seam costs 3 frames / 18 observations.
* **`scale_px` is stored rounded and the overlay uses it as a reproducibility parameter.**
  `overlay.agrees_with_artefact` reported 1 mismatch of 325 on the session artefact
  (`seat 4`, `away_from_place`, 325 vs 324) while all three single-file artefacts matched
  exactly. Re-running `assign` with the true scales and with the artefact's rounded ones
  differs on **exactly one observation, at t = 2184.91 s** — the assignment gate is
  `1.5 × max(anchor.scale, seat.scale)`, so 0.05 px of rounding moves one borderline
  observation out of its seat. Same class of defect as the `window_seconds` rounding in
  §5b: `scale_px` was a display value in the artefact and a reproducibility parameter in
  the overlay, and it cannot be both. **Fixed** — `pipeline.assemble` now writes
  `seat.scale` unrounded, as it already wrote `centre`; the two are used together by
  `seats.assign` and had no business being stored to different precisions. `run_id` does
  not hash the seat records, so no existing run identity moved. Verified:
  `agrees_with_artefact` on the re-run D14 session is `ok=True`, 0 mismatches of 325,
  matching the three single-file artefacts. The report and
  `identity/seatmap.write_template` already round it for display (`:.0f`).

## 11. The adult on camera D14 — three trackers, and only one of them was right

Camera D14 is the first recording in this project with **the board in frame**, so «стоит у
доски» stops being an absent measurement and becomes an answerable question. It is also the
first recording where the adult **moves**: he writes at the board, works at his own desk,
sits down among the pupils, and returns to the board, inside one lesson. Everything below
is from `D14_20260815103136.mp4` (2 904 s, 20 fps, 2560x1440, no audio), analysed at 2 fps
— 5 556 analysed frames in the detected lesson window of 46.3 min, 34 812 usable person
observations.

### 11.1 Why `identify_adult` cannot find this adult

The scale rule assumes the adult's desk is nearest the lens. On D14 the camera is at the
BACK of the room, so when he is at the board he is the FURTHEST person from it. Measured
ratio of observed shoulder width to what `room/perspective.py` predicts for a body at that
image row:

| | adult | pupils |
|---|---|---|
| depth-normalised shoulder width | **1.09 – 1.30** | **0.31 – 1.44** |

**The largest child in the room is bigger for her depth than he is.** Size cannot identify
this adult. It can only rule things out, which is all `RATIO_TOLERANCE` is used for.

### 11.2 Three trackers on ten hand-labelled frames

Ten frames were labelled by eye across the lesson (two at the board early, two at his desk,
one where he is genuinely not in frame, one at the board mid-lesson, two among the pupils,
one at the board late, one walking). Attribution was rendered onto the frames and checked.

| method | coverage | correct | wrong | silent |
|---|---|---|---|---|
| greedy nearest-neighbour, gated, size-weighted | 70 % | 2 | 6 | 2 |
| Viterbi over observations, size + travel cost | 57 % | 3 | 5 | 2 |
| **zone-anchored tracklets + detector hand-offs** | **45 %** | **7** | **0** | **2** (+1 correct refusal) |

The first two are the natural designs and both are wrong for the same underlying reason:
**a tracker that scores position prefers whatever is stationary**, and the adult is the one
person in the room who is not. Neither produced any internal signal of failure — both
emitted complete, plausible JSON. Only rendering the choice onto the frames found it.

What works instead uses two things that are not judgements at all: the operator's
`teacher_desk_zone` (the tracklet covering t = 74..647 s has **all 1 147** of its anchors
inside it), and the detector's own tracklets, which are long here — median 3.5 s but p90
**75.5 s**, with 40 of 559 exceeding two minutes. Chaining across hand-offs the detector
vouches for, and refusing everywhere else, buys correctness with coverage.

### 11.3 The hand-off distance, which was measured by being wrong

| hand-off distance | coverage | correct | wrong |
|---|---|---|---|
| 4.0 shoulder widths | 56.8 % | 6 | **2** |
| 2.0 shoulder widths | 45.0 % | 7 | 0 |
| 1.5 shoulder widths | 45.0 % | 7 | 0 |
| 1.0 shoulder widths | 45.0 % | 7 | 0 |

At 4.0 the chain jumps from the teacher to a pupil standing **beside him at the board** —
two people at a chalkboard stand about two shoulder widths apart, which is exactly why that
value fails there and nowhere else. Below 2.0 nothing further is bought. CHOSEN 2.0.

### 11.4 The result, and the shape of its error

45.0 % of the lesson attributed. Of that: **at his own desk 31.3 % of the lesson** (14.5
min), **among the pupils 10.1 %** (4.7 min), **at the board 3.4 %** (1.6 min), in transit
0.2 %. **Out of frame / not identified: 55.0 %** (25.5 min).

The 55 % is not uniformly distributed and the artefact says so: the follower loses him most
easily where he is smallest and most often occluded by pupils who have come out to the
board — which is the board. **Board time here is a lower bound.** The transitions between
places are lost for the same reason, which is why `moving` is 0.2 % and must not be read as
«почти не ходил по классу».

Two independent checks the numbers pass: at 10:41 he is at the front-left desk with a
laptop, and the strip shows `at_desk` there; at 11:01 he is sitting among the pupils, and
the strip shows `among_pupils` there. Both were facts supplied before the module was built.

### 11.4a What the review of this module found, and why none of it moved a number

Every figure in §11.4 reproduces exactly. What did not survive review was the *wording and
the typing* around them — four places where a quantity that had not been measured was
nonetheless rendered as a value:

| defect | what it printed | why it is wrong |
|---|---|---|
| `report/summary._teacher_section` read the seat ledger's `observations` after testing the follower's `coverage` | `TypeError`, non-zero exit from `classvision report` | the branch this module exists for (`teacher.seat_id: null`, `ledger: {}`) had no test and no report |
| `metrics.transitions` filled from `presence.transitions_between_episodes` on the no-seat branch | «смен положения — 25», inside a sentence beginning «что делало его тело за его собственным столом … где поза прочиталась» | no pose observation existed; 17 of the 25 were the follower losing him |
| `metrics.out_of_frame_share_of_lesson` taken from the seat ledger | 51,5 % beside the follower's 55,0 %, same lesson, same artefact | the ledger counts an EMPTY DESK; on D14 he is in frame at the board while it is empty |
| `presence.board.*` zeroed when no `board_zone` is configured | «0,0 мин · 0 % урока · 0 эпизодов» + «Скорее занижено…» | `classify_track` cannot emit `AT_BOARD` without a polygon; the zero is the shape of the config |

Two smaller ones, both of the same kind. `board.facing.share_of_board_observations_percent`
divided "board observations whose head direction is not `UNKNOWN`" by "board observations" —
but `geometry.head_direction` returns `UNKNOWN` only when it can see neither a head keypoint
nor the two shoulders, and `geometry.anchor` already refused to yield a position without
those same two shoulders at the same threshold. The field could not be anything but `100.0`,
and it was printed as the coverage counter for a reading the caveat beside it calls weak. It
now reports the share of board observations carrying an actual head keypoint, and names
`away` as the residual bucket it is. And `metrics/teacher._unmeasured` announced a refusal
gated on `diagnostics["ambiguous_frames_refused"]`, a key nothing has ever written — a
refusal that could not fire, reading as coverage of a case that is not covered.

The one genuine guess inside `classify_track` was measured rather than argued away: a frame
whose speed could not be computed (the first after any gap) is taken as settled and lands in
`AMONG_PUPILS`. On `D14_20260815103136.mp4` that is **2 observations of 560**, one second of
a 46-minute lesson. The count is now published per run and the state's stated rule says so.

### 11.5 Zones are tested against the shoulder line, not the feet

Measured on the full detection pass over both D14 files — 43 275 person observations, see
§12 and `configs/camera_d14.yaml`: a person standing at the board and a person seated at
the desk in front of it have box bottoms **two pixels apart** (449.2 at t = 1038 s against
450.8 at t = 1764 s, the same adult) because the desk crops the legs off both. Their
shoulder lines in those same two observations are **203 pixels apart** (214.4 vs 417.0). `pipeline.assemble` was testing the
foot point; it now tests `geometry.anchor`, and `room/zones.py` states the convention once
so the pupil path and the teacher path cannot drift apart.


## 12. The camera D14 room profile — where every polygon came from

**The sample is the FULL detection pass over both files: 43 275 person observations**
(8 174 from `D14_20260815101759.mp4` + 35 101 from `D14_20260815103136.mp4`), sampled every
0.5 s by the same model the analysis runs (yolo11m-pose, imgsz 1280, conf 0.30). Each
observation carries a shoulder anchor, a box bottom, a box height and two ankle
confidences. **The labels are the ankle confidence and the box height, neither of which the
zone can see** — a person whose ankles the model scores above 0.5, or whose box is over
150 px tall, is standing; a torso cropped by a desk is 81–146 px tall with ankle
confidences of ~0.01. So the numbers below are not a restatement of the polygons they
justify.

### 12.0 The first version of this section was measured on 2 % of that, and its ranges were wrong

The profile was originally justified on **911 observations from 128 sampled frames**, and
the ranges were written down as «за оба файла». The full pass was already sitting in
`.cache/` and cost nothing to score. The medians survived the check; the extremes did not,
and one claim was flatly false:

| claim, from the 911-observation subsample | full pass, 43 275 observations |
|---|---|
| adult at his own desk: shoulder line x 992–1080, y 334–373, **96 obs** | x 961–1080, y 300–395, **3 120 obs** — and clipped by the polygon at all four edges |
| «самая низкая линия плеч у доски — 331» | people standing on the floor in front of the board reach **y 409** |
| the notch's gap holds «единственный артефакт» at y 283 | y 280–300 alone holds **35 observations at 35 distinct times**, t = 995–2 502 s |
| board zone: 92 in, 1 exception | **5 560 in, 11 (0.20 %) that look seated**, all of them 116–146 px box fragments |
| door zone: one event, and it had no shoulder line | **25 events**, 24 with ankle confidence 0.85–1.00 — see 12.5 |
| right edge: «furthest at-board shoulder line is 1614» | **not confirmable at all** — see 12.7 |
| «pupils sit with their backs to the lens, faces barely visible» | a nose keypoint scores >0.5 in **44.1 %** of observations, an eye in **46.3 %** |

The polygons themselves were not changed by any of this: the medians they were centred on
were right. What changed is that the *reasons written beside the constants* are now true,
which is the only thing that makes a constant reviewable.

### 12.1 The floor strip was measured first, and it is now the independent check

The intuitive board zone is the floor a person stands on:
`[[1105,470],[1330,470],[1310,620],[1100,620]]`. On the full pass **3 083 observations put
their box bottom inside it, and not one of them looks seated**. It is a correct polygon,
and useless as a zone, because it covers only the open aisle in the middle of the board —
both ends of the board are hidden behind furniture:

* **Right end** — the front-right desk crops the legs off anyone standing there. Standing
  at the board and seated at that desk both put the box bottom on the desk edge, y ≈ 443–452.
* **Left end** — the teacher's own desk does the same to him: standing at the board he
  lands at y ≈ 404, seated at that desk at y ≈ 403–425.

The two cleanest single instances, both re-runnable with `classvision zones`:

| moment | what he is doing | box bottom | shoulder line |
|---|---|---|---|
| long file t = 1038 s | the adult **standing at the board** | 449.2 | **214.4** |
| long file t = 1764 s | the same adult **seated among the pupils** | 450.8 | **417.0** |

1.6 px apart in the foot point; 203 px apart in the shoulder line. No foot polygon
separates them; that is why zones are tested against `geometry.anchor`.

Because the floor strip's 3 083 hits are labelled by something the shoulder zone cannot
see, they are what the shoulder zone is now *scored against* rather than a rejected
alternative.

### 12.2 The board zone's bottom edge is a chosen trade-off, not a boundary of empty space

Of the 3 083 floor-strip observations, **3 030 (98.3 %) fall inside `board_zone`**; 30 are
rejected by the bottom edge at y = 340 and 23 by the notch. Their shoulder lines run
y 202–409, median 286, p95 323.

What the edge defends against: observations of people **seated at the front desks against
the same wall** (x 1090–1645, ankles below 0.2, box under 150 px) number **9 393**, and
their shoulder line reaches its 1st percentile only at **414**. So the edge sits at 340
against a seated population that starts at 414 — 74 px of margin — and pays 1.0 % of
standing observations for it. The asymmetry is deliberate and stated in the profile: the
error with a cost is calling a seated child «у доски».

### 12.3 Why `board_zone` has a notch

Over the left third of the board (x 1024–1090) the discriminator is **y**, not x. That band
holds 3 024 observations in two heaps:

| shoulder-anchor y | n | who |
|---|---|---|
| ≤ 239 | 254 | the adult teaching at the board's **left end**, legs behind his own desk |
| 240 – 319 | 69 (2.3 %) | in transit between the two |
| ≥ 320 | 2 701 | the adult **at his own desk** |

The bottom edge is therefore 265 for x < 1090 and 340 for x ≥ 1090. 265 splits the
transition band 6 / 63. **The band is sparse, not empty**, and which side of the line a
walking man falls on is arguable on the merits — it is not a precision problem, and the
profile says so rather than claiming a clean gap.

A flat-bottomed rectangle at 340 either loses the 254 at-board observations or swallows the
adult at his own desk; `tests/test_room_layout.py` pins that argument, not just its outcome.

### 12.4 `teacher_desk_zone`, and the fact that it clips its own cluster

The zone takes **3 120 observations** (660 short file, 2 460 long), median shoulder line
(1039, 341) — which is within 2 px of the seat centre the pipeline independently discovered
for the adult, (1039.2, 341.0), on a run that then identified him `designated,
needs_confirmation: false`. No pupil falls in it and nobody standing at the board does.

Its x/y *extents*, however, are the polygon's edges rather than the cluster's: 20
observations sit within 5 px of y = 300, six of y = 395, two of x = 960, one of x = 1085,
and a 20 px halo outside the zone holds a further 81. Some of the adult's time at his own
desk is outside his own zone. It is deliberately not widened — upward is `board_zone` and
its notch, and a point inside both zones makes the analysis non-deterministic; downward is
the aisle.

### 12.5 What the profile still cannot see — and one thing it can, contrary to the first version

* **Sound.** There is none in either file (`nb_streams=1`), so «ответил вслух» is not
  measurable at all. It is listed in `lesson.unmeasured`.
* **Observations with no shoulder line.** 15 of 43 275 (0.03 %) yield no confident
  shoulders and are therefore tested against **no zone at all**. `zonecheck` prints that as
  a distinct verdict from «вне зон», because a person who was not measured and a person who
  was measured and found nowhere are different facts.
* **Who is at the board.** The zone does not say whether the person is an adult or a pupil,
  and must not be used to decide: at t = 1764 s it is a pupil, at t = 1038 s it is one of
  each.
* **The doorway — this is the correction.** The first version of this section stated that
  the one doorway observation, at t = 2190 s, produced no shoulder line, and concluded that
  a zero at the door means «не видно». The full pass finds **25 observations inside
  `door_zone`** (2 short file, 23 long), 24 of them with ankle confidence 0.85–1.00 and box
  heights 280–474 px — people visible in full, not occluded to 10 %. There is an unbroken
  run of eight at t = 1207.4–1210.9 s, and there *are* shoulder lines at t = 2188.9 s and
  t = 2190.9 s, the exact moment cited as proof there were none. The subsample had simply
  landed on a frame where the model dropped the keypoints.

  Nothing consumes `door_zone`, so no number in any artefact moves. What moved is the
  sentence: «ноль у двери означает „не видно“» would have been a **false explanation of a
  zero**, which is worse than an unexplained one, and it was one measurement away from
  being caught.

### 12.6 `board_surface` is mandatory, because the guard that needs it was opt-in

`room/layout_io.py` rule 6 refuses a `board_zone` drawn as the strip of floor in front of
the board, by checking the zone against the declared chalk rectangle. Until this review the
check ran only `if board_surface is not None` — so omitting one optional key disabled it,
and the operator who draws the floor strip is precisely the operator who would not think to
declare the chalk rectangle.

The cost of that hole was measured rather than argued. The same recording,
`D14_20260815103136.mp4`, analysed twice — once with the floor strip as `board_zone` (the
profile that used to load), once with the shipped profile:

| `board_zone` | минут у доски | эпизодов | `zone_configured` | `lesson.unmeasured` |
|---|---|---|---|---|
| the floor strip | **0.0** | **0** | `true` | ответил вслух; подтверждение разметки зон |
| the shipped hexagon | 1.6 | 6 | `true` | ответил вслух; подтверждение разметки зон |

**The two artefacts are indistinguishable to a reader.** Same flag, same coverage counters,
same `unmeasured` list — and one of them says the teacher never went to the board. A
confident zero standing in for «мы это не измерили», which is the same failure as the
guessed head position and the default clustering threshold, arriving for the third time.
The loader now refuses the left-hand row by both routes (omitting `board_surface`, and
declaring one that the zone does not overlap), in 0.4 s, before the detection pass.

`board_surface` is now required whenever `board_zone` is non-null, `to_mapping` refuses to
emit a profile its own reader would reject, and `board_surface_of` no longer invents a
million-pixel frame when `frame:` is missing. `GROUND_TRUTH_D14.md`, which had asserted the
floor-strip convention in prose, now says why it is wrong instead of being quietly deleted.

### 12.7 The one edge that is not measured, said out loud

Every other constant in the profile is scored against a label the zone cannot see. The
board zone's **right edge, x = 1645, is not**, and the profile now says so in place of the
number that used to stand there.

The only independent «this person is standing at the board» label available on this camera
is the floor strip, and it spans x 1100–1330 — the centre panel. At the right end of the
board the front-right desk hides the feet, so there is no label at all. The subsample had
offered «the furthest at-board shoulder line across both files is 1614»; the full pass
cannot confirm it, because 377 standing observations sit right of the board's edge (x >
1603) in the board's y band and nothing in the data says whether they are at the board or
at the wall beside it.

So x = 1645 is chosen from the board's own geometry — 42 px of clearance past the chalk —
and is labelled as a choice rather than a measurement. That is the whole difference between
this line and the one it replaced: the polygon did not move.

### 12.8 Faces on this camera: an assumption that was never measured

The profile asserted that pupils sit with their backs to the lens, that faces are therefore
barely visible, and that face identity is consequently **weaker here** than the 0.30/0.10
measured in §4. The first clause is wrong and the third was never tested.

Measured on the full pass: the nose keypoint scores above 0.5 in **44.1 %** of the 43 260
observations with a shoulder line (above 0.3 in 52.7 %), an eye in **46.3 %** (54.4 %). The
verification frame at t = 1764 s shows four of six people turned toward the camera — the
desks stand in two columns along the walls, not in rows facing the board.

What is still unknown is whether those faces carry enough pixels to *recognise*: median
shoulder width across the lesson is 63 px (p05 21 px, p95 95 px), so a face at the far end
of the room is a few pixels across. No claim is now made in either direction. It changes
nothing operationally — `--faces` is off by default and can never create a name, only
corroborate or contradict one a signed seating plan already asserted — but «опознание по
лицу здесь слабее» was a measurement-shaped sentence with no measurement behind it, which
is the specific thing this file exists to prevent.

---

## 13. The orientation note in the cabinet, run against the live API on both lessons

Run 2026-08-16 with a real Gemini key, `gemini-3.6-flash`, prompt `orientation/1.0`, through
`classvision cabinet note --all` — **twice, half an hour apart, on the same two runs**.
**§13.8 supersedes this section's verdict**: re-checked against the ledger, both stored notes
carried false claims and the prompt is now `orientation/1.1`. The measurements below are kept
as they were taken, because the notes they certify are the evidence for what the digit guard
cannot see. Both
invocations passed the guard on both lessons («проверка пройдена: в записке нет величин»).
This section records what was sent, what came back, and — the part that matters — which of
the model's claims survive being checked against the numbers.

### 13.1 What left the machine, measured

| run | places | bytes sent | contents |
|---|---|---|---|
| `8acd73ced054611d` (camera 01, 2026-08-07) | 8 | **2 256** | place ordinal, coverage %, index, six episode counts |
| `81003e4ff5a8664c` (D14, 2026-08-15) | 5 | **1 484** | the same |

Plus the fixed 3 989-byte system prompt. The two bundles differ by three places and by 772
bytes, so a place costs about a quarter of a kilobyte. No imagery, no names, no
`external_id`, no dates, no room or class key, no `run_id`, no geometry, no timeline: the
whitelist is `cabinet/notes.py::bundle_for_run`, and `tests/test_cabinet_notes.py` asserts
the absences from the outside, including with an attested full name sitting in the store.

The two bundles differ by one field in a way worth recording: `board_visits` is **null** for
all eight places of camera 01 and **0** for all five of D14. Same counter, same zero in the
ledger, two different facts — camera 01 has `board_zone: null` and lists «стоял у доски» as
unmeasurable, so its zero is a property of the lens, while D14's board is in frame and its
zero is a measurement. The map from the analyser's own sentence to the counter it voids is
imported from `cabinet/lessons.py`, not copied. A zero sent instead of a null is an
invitation to write «к доске никто не выходил» — a finding about children assembled out of a
fact about a camera, and on D14 it would also be contradicted by `GROUND_TRUTH_D14.md`, which
records a pupil at the board at t ≈ 1800 s that the pupil-side counter never fired on.

### 13.2 The numbering defect this found before the model could exploit it

The artefact numbers seats per run; the cabinet numbers places per room. On
`full_lesson.analysis.json` the adult holds artefact `seat_id = 1`, so **artefact `seat_4` is
the cabinet's «место 3»** — the two numberings differ by one at every seat. A bundle built
from artefact ids (what the per-lesson path in `report/orientation.py` correctly uses for its
own surface) would have produced a note saying «место 4» directly above a table whose «место
4» is a different child: every sentence true of what was sent, every sentence pointing one
desk to the left, and nothing in the guard, the prompt or the HTML able to see it, because
«место 4» is a valid label on both sides of the mistake. The bundle now carries
`places.ordinal`, and `orientation.check()` validates the note's digits against that set.

### 13.3 Four runs of one command produced four different notes

`temperature=0`, identical bundles, four invocations within half an hour (16:29, 16:42, 16:48
and 16:59 UTC), and the texts differ in wording, in ordering and — see 13.5 — in whether they
contain a false sentence. This is a measured property of the tool, not a complaint about it,
and it decided two design points:

* the note is **stored** with the time and the model that produced it, and re-rendering the
  cabinet never regenerates it — otherwise the same lesson would say something slightly
  different every time a school rebuilt its pages;
* a note is never compared with another note. Two paragraphs about two lessons differ partly
  because the lessons differ and partly because the sampler did, and nothing on the page
  invites a reader to take a difference between them as a finding.

**The tables below check the 16:48 invocation, claim by claim.** The database in `out/`
holds whichever invocation was run last — regenerating is an operator's command, not a
rebuild — and every later one was checked the same way, against the same numbers. What
recurs across all four is in §13.6; what changed is the wording, and once, the truth of one
sentence.

### 13.4 Camera 01, the stored note, claim by claim

Coverage by place: 1 — 97.1 %, 2 — 85.1, 3 — 99.6, 4 — 91.5, 5 — 98.6, 6 — 96.3, 7 — 91.8,
8 — 89.9.

| claim | numbers | verdict |
|---|---|---|
| место 6 — частые вставания, отходы от парты и наклоны головы | stands 7, away 2, head_down 14 — the largest of the room on two of the three | holds, all three parts |
| места 3 и 5 — неоднократно опущенная на стол голова | head_down 7 and 5 | holds for **both** — the joint-claim rule survived |
| место 1 — регулярные отходы от рабочего места | away 4, the largest | holds |
| места 4 и 7 — многочисленные подъёмы руки | 4 and 4, the two largest | holds for both |
| место 8 — при хорошем обзоре ничего не зафиксировано, ребёнок спокойно сидел | all six counts 0 at coverage 89.9 % | holds; the sentence prompt rule 6 exists for |
| место 2 — видимость ниже, чем на остальных местах; повороты и вставания требуют проверки по видео | 85.1 % is the minimum, next is 89.9 %; turned_away 5, stands 2 | holds |

**Six claims, six hold.** No quantity appears in the text at all, and nothing is said about
«у доски» — on this camera `board_visits` was sent as `null`, not as `0`.

### 13.5 D14, the stored note, and the false sentence an earlier run produced

Coverage by place: 1 — 94.9 %, 2 — 61.5, 3 — 96.7, 4 — 78.8, 5 — 65.4.

| claim (stored) | numbers | verdict |
|---|---|---|
| места 2 и 5 видны лишь часть урока | 61.5 % and 65.4 %, the two lowest | holds |
| место 1 — регулярно вставал, ни разу не поднял руку, не отходил | stands 14, hand 0, away 0 | holds; «при полной видимости» is 94.9 %, see 13.6 |
| место 3 — постоянные отходы и частые повороты головы | away 7, turned_away 24 — the largest | holds (see 13.6 on «в сторону от доски») |
| место 4 — регулярные повороты при редких поднятиях руки | turned_away 19, hand 1 | holds |
| место 5 — частые вставания и поднятия рук, обзор ограничен | stands 21, hand 23 — both the largest; coverage 65.4 % | holds |
| место 2 — много поднятых рук при отсутствии перемещений, обзор неполный | hand 15, away 0, coverage 61.5 % | holds |

**The first invocation, twenty minutes earlier, wrote one sentence that does not hold:**

> «На месте 5 заметны постоянные подъёмы со стула и поднятия руки, но этот участок урока
> просматривался хуже других.»

Place 5 was seen 65.4 % of the lesson; **place 2 was seen 61.5 %** — worse. The comparative is
false, and it is false in a way the guard structurally cannot catch: **it contains no digit.**
This is §8's lesson one level up. The digit guard bounds the KIND of error a note can make
(never a misattributed quantity); it cannot bound the presence of error. That is exactly why
the note is rendered dashed rather than filled, labelled machine-written on both sides, and
placed **above** the table that contradicts it — and why there is no class-level note at all
(`notes.CLASS_NOTE_REFUSAL_RU`): a cross-lesson note's characteristic error is a direction,
and a direction has no digits either.

### 13.6 Three smaller findings, all of them ours to answer

1. **«В сторону от доски» / «повороты корпуса» (16:29 and 16:48 runs).**
   `states._turned_away` measures a departure from *this pupil's own habitual head direction*
   — head only, and with no knowledge of where the board is. On D14, where the board is in
   frame at the front of the room, «в сторону от доски» is a defensible reading of what
   `states.py`'s own docstring calls turning away from the front. On camera 01 it would not
   be: `board_zone: null`, «стоял у доски» is listed as unmeasurable, and the first run said
   it there anyway — a spatial claim about a board the analysis cannot see. The same run also
   wrote «повороты корпуса», a torso claim from a head-direction reading. The counter's own
   Russian label is «отворачивался назад», and the note is the only surface on which it gets
   re-worded; the tables print the label.
2. **«При полной видимости» about place 1 of D14 (94.9 %)** and **«всё это время» about place
   8 of camera 01 (89.9 %)** are both strictly «за всё наблюдавшееся время»: about three and
   about five minutes respectively were not observed. The claims they support are the right
   ones — quiet pupils rather than unseen ones — but «полная» is a word the coverage figure
   does not support, and the coverage figure is two lines below in the table.
3. **D14 place 4 (78.8 % — nearly a fifth of the hour unseen) is described in all three runs
   with no visibility caveat**, while 61.5 % and 65.4 % always get one. Prompt rule 7 asks for
   the caveat; the model drew its line somewhere near 70 % and nothing in the prompt names a
   threshold. The tables carry the figure for every place regardless.

Neither stored note asserts anything about «у доски», which is the trap on both cameras. On
camera 01 the counter is now sent as `null`, so there is nothing there to misread. On D14 the
board IS in frame, `board_visits` is a measured 0 for all five pupils — and
`GROUND_TRUTH_D14.md` records a pupil standing at the board at t ≈ 1800 s, which that counter
never fired on. A note saying «к доске никто не выходил» would have been the first false
sentence in this project traceable to a measurement gap rather than to a number. It was not
produced, because the prompt gives the model no vocabulary for asserting an absence — a rule
doing its job, not a proof.

### 13.7 The no-key path, run to the end

The same two artefacts imported into `out/cabinet_nokey.sqlite3` with `GEMINI_API_KEY`,
`GOOGLE_API_KEY` and `OPENAI_API_KEY` unset: `cabinet note --all` exits **0**, prints
«НИЧЕГО НЕ ОТПРАВЛЕНО; сводка, которая ушла бы, — 1 484 байт счётчиков», stores the absence
with its reason, and `cabinet report` then renders «Записки нет. Причина, как её записала
система: …» followed by the command that would produce one. No traceback, no empty block, and
no page that merely has one section fewer.

### 13.8 Review of §13: three false-claim classes found, fixed, and re-measured

§13 above certified the two stored notes as holding. **Re-checking each claim against the
ledger rather than against the bundle found three classes of false statement**, all of them
invisible to a guard that rejects digits, and one of them named in §13.1 and shipped anyway.
Prompt `orientation/1.1` and `bundle_for_run` now address all three.

**1. A zero the same artefact contradicts, sent as a measurement.** §13.1 records, in its own
words, that D14's pupil `board_visits = 0` "would also be contradicted by
`GROUND_TRUTH_D14.md`, which records a pupil at the board at t ≈ 1800 s" — and the bundle
sent the zero, with a test pinning it. `cabinet/lessons.py::_board_conflict` had already
quantified the contradiction from the same run: **somebody at that board for 30.4 of the 58.1
minutes, 4.0 of them the adult**, against a pupil column reading zero, because `AT_BOARD` is a
state of a PLACE and a child at the board has left theirs. `voided_counters` catches only the
other cause (camera 01's missing zone), so the new `lessons.board_conflict_for_run` is asked
as well and the counter goes out as `null` with a digit-free reason.

**2. `0` episodes read as «этого не было».** The counters are EPISODE counts; `ledger.py`
keeps `observed_seconds_by_state` beside `episode_seconds` precisely because a state can be
seen without any run of it lasting long enough to count. The bundle sent only the episode
count, so both meanings arrived as one `0` — and both stored notes turned it into a universal
negative:

| stored claim | reality behind the zero |
|---|---|
| camera 01, «место 8»: «отсутствие каких-либо движений … всё время удалось провести спокойно за партой» | 150 observations of `head_down` = **75.0 s**, 16 runs discarded as too short; plus 3 of a raised hand |
| D14: «ни на одном из мест … не опускал голову на стол» | places 2, 3, 5 → **32.0 s, 3.5 s, 20.0 s** of observed `head_down` |
| D14, «место 1»: «отходов от парты и поднятых рук не было» | **21 observations (10.5 s) of each** |

Across the two artefacts **17 place-counters read 0 while the state was observed**. Each
place now carries `zero_but_briefly_seen`, and rule 7 permits exactly one phrasing —
«длительных отходов от парты не отмечено» — banning the bare negation that reads as never.

**3. «Плохо видно» left to the model's judgement.** Rule 7 (old numbering) asked for a
visibility caveat and named no line, so the model qualified the two worst places and
described D14's «место 4» at **78.8 %** — a fifth of the hour unseen — as flatly as one seen
throughout. `VISIBILITY_CAVEAT_BELOW_PERCENT = 90.0` is now **CHOSEN** from the recordings:
39 counted episodes have a **median length of 7.3 s** (mean 14.2 s), and a place at exactly
that line is unseen for 5.0 minutes of the 50-minute lesson — room for about forty median
episodes. The bundle carries the verdict (`needs_visibility_caveat`), not the arithmetic.

Two further repairs in the same pass: a spatial claim about the board on the camera that
cannot see one («повороты головы в сторону от доски», and once «повороты корпуса» from a
head-only reading) is banned by rule 10, since `states._turned_away` measures departure from
a pupil's own habitual head direction and knows nothing about the room; and a share of the
lesson stated in words («места 2, 4 и 5 большую часть урока находились в зоне неполного
обзора» — they were *seen* 61.5–78.8 % of it) is banned by rule 8 as rule 1's own failure
mode, a quantity spoken rather than written.

**The mutation test — the failures are caused, not chanced.** The pre-fix prompt and pre-fix
bundle were replayed against the live key, 3 trials per lesson:

| configuration | trials | trials containing a false claim |
|---|---|---|
| `orientation/1.0` + old bundle | 6 | **6** |
| `orientation/1.1` + new bundle | 6+8+8 | **0** |

The old configuration produced «на месте 8 … ученик спокойно сидел на протяжении всего
занятия» in **3 of 3** camera-01 trials, «ни разу не поднимая руку» about D14's place 1, «в
сторону от доски» on the boardless camera, and no caveat for D14's «место 4» in 2 of 3
trials. The new one produced none of these in 22 trials, with the guard passing every time.

Sizes moved with the rules: system prompt **3 989 → 7 257 bytes**, bundles **2 256 → 3 261**
(camera 01) and **1 484 → 2 286** (D14). That is the price of the three distinctions, and it
buys the difference between «спокойно сидел весь урок» and a sentence that is true.

**Two things the fix does not claim.** The digit guard still cannot see any of this — it was
never able to, which is why the block stays dashed, labelled «машинная записка», and printed
above the table that would contradict it. And four invocations still produce four different
texts at `temperature=0`, so what is verified is the *class* of claim the prompt can make,
not one paragraph; that is why notes are stored with their timestamp and never regenerated
by a page build.

---

## 14. Within-lesson dynamics — the only «динамика снижения активности» this data supports

The client asked to analyse a child's «динамика снижения активности». Across lessons that
needs an identity: §4 measured that faces cannot supply one here (best cosine 0.30, margin
0.10), and the two analysed lessons are **two different rooms with different children** —
camera 01 on 2026-08-07 (8 pupil places) and camera D14 on 2026-08-15 (5). `metrics/trend.py`
refuses without ≥ 5 lessons of an attested identity and that refusal stands; the store holds
**0 attestations**. Comparing place 3 of one camera with place 3 of the other would be
arithmetic performed on strangers.

Inside ONE lesson the precondition is free — a place is a place for the whole 50–58 minutes —
so `metrics/within_lesson.py` answers the question there. Everything below was measured on
both real lessons on 2026-08-16, from the detection cache.

### 14.1 The segmentation was chosen by counting refusals, not by taste

Cutting the **detected lesson window** (never the file: camera 01 opens with 110 s of empty
room) into equal parts, and counting how many segments `activity.MIN_COVERAGE` refuses:

| scheme | segment length, cam01 / D14 | segments refused | places left with < 4 usable |
|---|---|---|---|
| thirds | 16.7 / 19.4 min | 0/24, 2/15 | **8 of 13** |
| quarters | 12.5 / 14.5 min | 0/32, 4/20 | 3 of 13 |
| fifths | 10.0 / 11.6 min | 0/40, 4/25 | 1 of 13 |
| **sixths** | **8.3 / 9.7 min** | **0/48, 4/30** | **0 of 13** |
| eighths | 6.3 / 7.3 min | 1/64, 6/40 | 0 of 13, but one segment at coverage **0.02** |

**CHOSEN: six.** The coarsest cut at which no place in either lesson drops below the four
usable segments the direction test needs, and the finest at which no segment degenerates.

**Fixed ten-minute bins were rejected by a measurement, not an argument.** Camera 01's window
is 3 002.8 s, so a 600 s grid leaves a **sixth bin of 2.8 seconds — six observations** — and
that bin produced an activity index of **40.0** with a coverage of **1.00** printed beside it.
A confident number with a clean coverage figure, computed on three seconds of a fifty-minute
lesson, in the bin the whole question is about. Equal fractions cannot produce it.

### 14.2 A segment index is NOT the lesson index, and the gap is 30 points

The index's participation term is normalised to `EVENTS_FOR_FULL_CREDIT = 3` events **per
lesson**. Camera 01 place 3: lesson index **97.7**, segment indices **97.6 / 68.1 / 68.6 /
65.1 / 67.0 / 69.6**. Nothing declined — the child's three qualifying episodes all fall in the
first eight minutes, which is full credit over the lesson and at most one segment's worth
anywhere. Segments are comparable to each other (equal length ⇒ equal exposure) and to nothing
else. The artefact carries that sentence (`not_comparable_to_ru`) and both surfaces print it.

### 14.3 `MIN_USABLE_SEGMENTS = 4` is derived, and the derivation is two numbers

Exact two-sided Mann–Kendall p for a **perfectly monotone** series, by enumeration
(`_kendall_p`, checked against brute force over all n! orderings):

| n | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| best attainable p | **0.3333** | 0.0833 | 0.0167 | 0.0028 | 0.0004 | <0.0001 |

At three segments the strongest possible evidence is one-in-three by chance. The same n = 3 is
degenerate for the fit's own scatter: over 5 000 simulated draws the MAD of the Theil–Sen
residuals is **zero in 100.0 % of three-point series** (it is zero by construction — two of
three residuals always land on the line), against a median of 0.313 at n = 4. Three points can
report neither a direction nor how noisy they were.

`DIRECTION_ALPHA = 0.10` is CHOSEN. Not 0.05, because 0.083 is the best a four-segment place
can do, and the places that lose segments are the ones the camera sees worst — a 0.05 bar
would silently restrict the statistic to well-seen children. The cost is published per lesson
in `lesson.within_lesson_direction_by_chance`: **0.8 of 8 places** on camera 01 would show a
direction from noise alone, and 3.2 of the 32 component tests.

### 14.4 The coverage confound: bounded, and then not found

The expected failure is that coverage falls as a lesson goes on, so a falling index is partly a
worsening picture. `_visibility_bound` bounds it: with relative retention `r = c_last/c_first`,
the observations missing at the end could adversarially have been the GOOD ones, so each share
may understate by `(1−s)(1−r)` and the index by `Σ w (1−s)(1−r)`. Measured across all thirteen
places of both lessons at sixths, that bound runs **0.00 – 1.45 index points**, against index
changes up to 27.9.

**The reason it is small is the finding.** The segments where visibility collapses are
*refused* by `activity.MIN_COVERAGE` before they reach the statistic — D14 place 6's final
segment is seen in **14 %** of its frames, place 3's third in **23 %** — so what survives has
near-flat coverage. The confound is handled by refusal, not by correction, and the artefact
says which minutes went missing (`refused_ordinals`, `covered_minutes`).

**And the premise is not true of this footage.** Coverage does not generally fall:

| | segment coverage, first → last |
|---|---|
| cam01 place 3 | 0.73 0.65 1.00 0.97 0.75 1.00 — **rises**, pupils are still arriving |
| cam01 place 9 | 0.83 1.00 0.96 0.61 1.00 1.00 — erratic, no trend |
| D14 place 6 | 0.56 0.89 0.95 0.74 0.64 **0.14** — rises, then collapses at the end |
| D14 place 4 | 0.90 1.00 0.93 1.00 0.99 0.95 — flat |

Because the bound never bound anything on real data, it is tested against a fabricated place
whose coverage falls 100 % → 55 % with unchanged behaviour
(`tests/test_within_lesson.py`). A gate nobody has seen fire is a gate that is probably
inverted.

### 14.5 What actually moved, on both lessons

Segments of 8.34 min (cam01) and 9.69 min (D14). Δ is the Theil–Sen change across the usable
span, in that place's own index points.

| | place | Δ | p | verdict |
|---|---|---|---|---|
| cam01 | 4 | **−25.3** | 0.017 | **ниже к концу урока** |
| cam01 | 8 | −5.4 | 0.056 | ниже к концу урока |
| cam01 | 9 | +0.8 | 0.136 | без направления (меньше порога нарезки 2.40) |
| cam01 | 2, 3, 5, 6, 7 | −11.5 … +14.2 | 0.27 – 0.72 | направление не установлено — менялось не в одну сторону |
| D14 | 2, 3 | −0.9, +0.4 | 1.00, 0.75 | без направления |
| D14 | 4, 5, 6 | +21.7, +18.6, +18.3 | 0.47 – 0.48 | направление не установлено |

**One place declines with the index behind it (cam01 place 4)**, and it is posture, not
events: its head-up share runs 0.946 → 0.660 → 0.863 → 0.986 → 0.778 → **0.320**.

**Of the six «declines» a naive reading would have counted, only one survived, and the reason
each of the others did not is different every time** — which is why the refusals are separate
values and not one flag:

* cam01 place 2 (−11.5): the whole difference is **one hand-raise** in segment 1 and none
  after. One episode is worth `100 × 0.30 / 3 = 10.0` index points, and pupils here produce
  0–4 per lesson (§5c), so the `single_event` floor absorbs it.
* cam01 place 3 (−4.0): **three episodes in segment 1**, 30 points of participation, and flat
  posture afterwards. Not a decline; a burst at the start.
* cam01 place 7 (−27.9): the largest movement in either lesson, and the total says
  «not established» because two hand-raise bursts in segments 1 and 3 put 30-point spikes into
  a falling series and destroyed its ordering (p = 0.27). **The component test finds it:
  head-up share falls 27.0 → 21.2 → 26.5 → 9.9 → 13.1 → 5.6 index points, Δ = −21.4,
  p = 0.056.** This is why every component carries its own direction — a composite can hide
  its own largest term.
* D14 places 4, 5, 6 all move UPWARD by 18–22 points and none is established: their event
  counts rise through the lesson (place 6: 0, 4, 10, 18, 11 episodes) but not monotonically.

### 14.6 Two defects found in this module by reading its own output

* **`covered_minutes` and `span_minutes` were one field.** The generated Russian said
  «описывает 39 минут урока из 58» beside a `covers_share_of_lesson` of 0.83 — 48 minutes.
  Same block, two answers to «сколько урока это описывает». Now three separately named
  durations: the fit's span (midpoint to midpoint), the measured minutes (usable segments
  only), and their share. Same failure class as §8.
* **The component directions had no magnitude floor**, and on D14 printed, about two real
  children, «место 2 · находится на своём месте: ниже к концу урока (−0,1 балла, p = 0,06)»
  and «место 4 · голова поднята: ниже (−0,0 балла, p = 0,06)». Mann–Kendall tests an ORDER,
  and an order can be perfectly monotone while the movement is below anything the measurement
  resolves. Each component now carries `_boundary_floor` computed with **its own weight**
  (2.40 / 2.00 / 1.20 index points on camera 01), and participation carries the 10.0-point
  event quantum. Both sentences are gone; camera 01 place 7's −21.4 survives.

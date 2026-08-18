# Handoff — Qorgan AI v2

Updated **2026-07-24**. Everything is committed; nothing is half-done.

Every number below was measured against the code at **`de5a3e4`**. If `git log --oneline de5a3e4..HEAD -- src`
returns anything, these numbers are older than the tree: **re-measure, do not trust this line.**

> **This guard has fired once already, silently.** The previous version of it pinned to `9d302b5`
> and said "anything after it on `main` is this documentation". Twelve `src/` commits later the
> count still read **1397** and nobody re-measured, because a guard that only asks you to notice is
> not a check. The command above is the check; run it, do not read this paragraph and move on.

**Measured on this tree, today — not remembered:**

```
pytest --junitxml → tests=1532  failures=0  errors=0  skipped=0  (exit 0)
ruff check .      → All checks passed!
```

And **green under the condition that used to break it**: the recorded reproducer is *two full suites
run concurrently* (the pair that went red almost immediately for both the zmq flake and
`test_det_every_is_honoured`). That reproducer was last run at 1397 and passed; it has **not** been
re-run at 1532. A timing test proven on an idle machine is proven of nothing — and this one is
currently proven only of an older tree.

**Both specs are on `main`.** Spec A (identity service) merged at `3fd7399`; Spec B (detector
calibration) at `c7d1c1d`. Both code follow-ups are merged: `55b5124` (plan-workers one-kind fleet)
and `9d302b5` (exit-cost visibility). The feature branches still exist but are **not where the work
lives** — `main` is.

What remains is not construction. It is the three things only the school can supply.

This file is *what is true now, what to run, what is blocked, what will bite you*. It is **not** a
changelog — `docs/superpowers/progress-ledger.md` is the append-only history. **Trust the ledger over
memory, and grep before you believe.**

---

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[ai]" --extra-index-url https://download.pytorch.org/whl/cu128

# `insightface` depends on plain `onnxruntime`, which SHADOWS `onnxruntime-gpu` and takes
# CUDAExecutionProvider with it. pip installs both without complaining.
pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0

qorgan doctor                       # torch AND onnxruntime must BOTH see the GPU
qorgan db upgrade
qorgan user add admin --role admin
# --role also takes psychologist and superadmin; superadmin is shell-only (schools
# register, §14). --school <slug> is required once a second school exists.
qorgan supervisor                   # the worker fleet
qorgan web                          # the dashboard on 127.0.0.1:8000
```

Read in this order: the two specs and two plans in `docs/superpowers/`, then
`docs/superpowers/progress-ledger.md`, then `docs/questions-for-school.md`.

## What you can actually run off-site

The block above is the *installed* order, and half of it needs the school's LAN. Off-site — on a
laptop, with no cameras and no NVR — this is the real division. It matters because the two commands
that look like the product are the two you cannot run.

**Runs anywhere, no cameras needed:**

```bash
qorgan doctor                      # GPU only; needs a CUDA card, not a network
qorgan config validate             # loads and validates all ten camera configs
qorgan db upgrade                  # creates/migrates the SQLite file
qorgan plan-workers --dry-run      # measures THIS GPU, prints the grouping, writes nothing
qorgan pupils import-roster <dir>  # needs the roster photos, which travel by hand
qorgan pupils gallery-report       # who is enrolled twice, can this gallery work at all
qorgan identity camera-report      # can a camera recognise anybody at the resolution it is fed
pytest --junitxml=<path>           # the suite: 1532, and it needs nothing external
```

**Needs the school's LAN — cannot be demonstrated here:**

- **`qorgan supervisor`** — it opens RTSP. **RTSP has never been reached from this machine** (see
  "Remaining risk"), and there is no `.env` here, so `RTSP_PASSWORD` has never been non-blank.
  It will start, fail to connect, and exercise the reconnect path. That is not a demo.
- **`qorgan web`** — starts and serves, but the dashboard has no live previews and no events
  without workers behind it. You can log in; you cannot show anything happening.

**The eval pipeline is a third case — it needs footage that is not in git:**

- `qorgan eval scan | sample | label` all read **`eval/clips/`**, which is **gitignored footage of
  children**. A fresh clone has an empty corpus and these commands have nothing to do. The corpus
  travels by hand, on a disk, like the roster photographs.
- **`qorgan eval gate` cannot run at all: `eval/baseline.json` does not exist.** Verified — there
  is no such file anywhere in the tree. The line further down calling it "the committed artefact"
  describes an intended artefact, not a present one: nothing has ever been frozen, because
  `save-baseline` is blocked on the same human labels `eval run` is. `gate` is a command with no
  baseline to gate against, and it will stay that way until the labelling in
  `docs/questions-for-school.md` comes back.

---

## What only the school can answer — the actual critical path

No code unblocks any of these. See **`docs/questions-for-school.md`** (it is written for the school,
in Russian, and carries the numbers). In order of what they unblock:

1. **Which of the six duplicate IDs is canonical** — and for the **two pairs that cross the
   pupil/staff line, is that person a pupil or staff?** Staff never open a meal session, so the
   answer decides whether that person is **fed**. (`7-А 438/439` may be twins; only the school can
   say.) These six are not misidentified — they are **invisible**: their own duplicate sits at top-2
   and collapses the recognition gap to +0.001.

2. **Canteen-entry footage of a pupil we can name.** One volunteer, one minute, one clip — and it
   closes **both** open recognition questions at once: the unmeasured **`min_score` ceiling** and the
   **`face_gate`** question. `min_score` is 0.50 in every profile: a **floor** measured above the
   worst impostor (`MEASURED_IMPOSTOR_CEILING = 0.472`), *not* a tuned value. **The ceiling is still
   unmeasured.** Set the gate above what a real child can score and we recognise nobody.

3. **The fight's start and end time.** `eval run` / the operating point (B10/B11) are blocked on
   **human labels**. The corpus has **one confirmed fight and its interval is unknown**. We measured
   *which camera* it came from (scene-matched against a validated 8/8 control — see
   `eval/README.md`), but **nothing here guesses a timestamp**. **One fight is not a recall number.**

---

## Where the detector actually is

`eval scan` **has run on the real footage** (this is the whole point of Spec B, and it is no longer
pending): **657 clips, 145 candidates, 51 alerts** at the shipped threshold — with the skeleton
**vetoing half the fast tier** (72 of 145 held at exactly 0.72). The pipeline is
`scan` → `sample` → `label` → `run` → `save-baseline`:

```bash
qorgan eval scan      # detector at threshold 0: every moment it would have fired on (resumable)
qorgan eval sample    # the stratified worklist: alerts, skeleton-vetoed, and the silent
qorgan eval label     # watch each candidate and label it (resumable)
qorgan eval run       # precision / recall / F1 + a PR curve   <-- BLOCKED on human labels
qorgan eval save-baseline
qorgan eval gate      # fails if a change made the detector worse
```

**`eval scan` is resumable, and it had to become so.** A later run over the same 657 clips died
on clip 170 — the host could not allocate the 10.5 MB for one decoded frame — and wrote **nothing
at all**, because both artifacts were written once, at the end. 169 clips of GPU went with it. It
now rewrites `candidates.csv`, `candidates.coverage.csv` and `candidates.unreadable.csv` after
**every** clip, and a re-run with the same `--out` skips whatever the coverage manifest already
proves was scanned — the same shape `eval label` uses, where the result file is also the progress.
A clip it cannot read is named in `candidates.unreadable.csv`, kept OUT of the manifest, counted in
the summary, and makes the exit status non-zero; `eval sample` then refuses to draw until it is
scanned or removed, so the count of unread clips cannot be lost between the two commands.

**Re-measured 2026-07-28**, whole corpus, resumable scan, `--device cuda:0`: **657/657 clips
covered, 148 candidates, 53 alerts, 72 held at exactly 0.72, 0 unreadable.** Three more candidates
and two more alerts than the 145/51 above — and that older pair cannot be checked, because **no
`eval/candidates.csv` exists on disk in either checkout**, so it names a run whose output is gone.
The 72-at-the-cap figure is identical, and the resumable rewrite is not what moved the other two:
over the 169 clips the crashed run did manage, its per-clip camera and candidate counts and this
branch's agree exactly, clip for clip. Treat 145/51 as a stale claim, not as a regression.

Also present: `qorgan eval template` (writes a `labels.csv` template) and `qorgan eval noise-floor`
(what a camera reports when nothing is happening). `template` still exists, but it is **not** the
labelling workflow any more — `scan` → `sample` → `label` is.

**`qorgan eval gate` cannot run "in CI":** there is no CI, and the corpus is gitignored footage no
runner can see. It is a local pre-merge command, with `eval/baseline.json` as the *intended*
committed artefact — **and that file does not exist.** Nothing has ever been frozen, because
`save-baseline` is blocked on the same human labels `eval run` is. Until they arrive, `gate` is a
command with no baseline to gate against. See "What you can actually run off-site" above.

### The thresholds are correct by construction and unvalidated in fact

Every speed and acceleration threshold in `config/profiles/*.yaml` is a **converted estimate**, not a
tuned value — and it is worth understanding why, because it is the reason `eval run` matters.

The legacy measured speed in px/**frame** and acceleration in the incoherent mixed unit
`(px/frame)/s`. A comment claimed the result was FPS-independent; it is not — 8 fps and 25 fps report
numbers for identical physical motion that differ by a factor of three. So every legacy threshold
meant something different on every camera and changed meaning whenever the frame rate or `det_every`
did, and **none of them transfer**. v2 measures **px/s and px/s²**, which are physical and do
transfer (`src/qorgan/detection/tracking.py`), and it smooths over a window of **time**, not of
samples — the subtle half of the same bug.

But **there is no exact conversion from a threshold tuned against an uncontrolled frame rate.** Most
shipped values are the legacy's, converted at its ~10 fps analysis rate. They are **starting points**.
Until labelled footage re-derives them, treat every speed and acceleration as provisional — the
detector is *correct by construction and unvalidated in fact*, and I would not tell a school
otherwise.

---

## What will bite you

- **`pytest -q` silently drops its summary line in this environment.** Use `--junitxml` and read the
  XML. A wrong path exits **4** having collected **nothing** and looks like success. Every count in
  this file came out of the XML.

- **The data is photographs and video of children. It is not in git and never will be.**
  `student_photos/`, `original_student_photos/`, `eval/clips/`, `eval/crops/` are gitignored. Copy
  them across by hand. **Never `git add -A` or `git add .`** — stage by explicit path, and run
  `git status` before you commit.

- **Worktrees have no `.venv`.** Use the main tree's interpreter with `PYTHONPATH` at *the worktree's*
  `src`. `pip install -e .` writes an absolute path into a `.pth` file that still points at the
  original clone, so without this two parallel worktrees silently grade each other's homework.
  `tests/conftest.py` hard-fails rather than let that happen — **trust it**; it is being loud on
  purpose, because an instruction is not a check.

- **Face recognition on the hall cameras is impossible, not merely poor.** At the hall's true analysis
  resolution (**1280×720** — `hall.yaml` overrides `config/base.yaml`'s 960×540; *there is no
  fleet-wide analysis resolution*), **0 of 14 970** real faces clear the strict gate. 77 clear the
  small-face gate and **not one is recognised** (best similarity among them: 0.35). **Zero
  recognitions in 14 970 faces.** Bullying events stay **anonymous**. The fix is a face-height camera at a chokepoint —
  **optics, not tuning**, and the school's call.

- **`test_det_every_is_honoured` is timing-dependent: it measures the machine as much as the code.**
  It is green today, including under the concurrent-suite reproducer. That is not a promise about a
  busier machine. If it goes red, suspect the load before the logic.

Three constraints that look like they could be simplified, and cannot:

| looks simplifiable | why it is not |
|---|---|
| `cv2.imencode` + manual write in `events/recorder.py` | `cv2.imwrite` fails **silently** on non-ASCII paths on Windows. Every path in this school contains Cyrillic; events would point at snapshots that were never written. |
| one YOLO per camera, never shared across a group | Ultralytics keeps **tracker state on the model**. A shared model gives children in two different corridors the same track IDs. |
| `onnxruntime-gpu==1.26.0` pinned and force-reinstalled | plain `onnxruntime` shadows it and takes CUDA with it. Face recognition then runs on the CPU, ~40× too slow, silently. `qorgan doctor` and `tests/test_gpu.py` both catch it. |

---

## The fleet: worker groups, not one process per camera

Measured on the 4 GB RTX 3050 by **`qorgan plan-workers`** — which measures the GPU you actually
have, rather than quoting the one we had:

**All four numbers below are GPU memory (VRAM):**

```
CUDA context + process   ~81 MB      YOLOv8n + ByteTrack   ~62 MB  PER CAMERA
pose (bullying group)    ~12 MB      InsightFace           ~708 MB PER CANTEEN GROUP
```

**The expensive thing is not the CUDA context — it is InsightFace, at ~708 MB per instance.** That is
why `config/workers.yaml` maps cameras to processes rather than running one process per camera. On a
bigger GPU, write one camera per group; no code changes.

**The InsightFace figure appears four ways across these documents; three are VRAM and one is not:**

| figure | what it is |
|---|---|
| **~708 MB** | **VRAM**, one InsightFace instance. The number `plan-workers` measured on the 3050. |
| **~700 MB** | the same 708, rounded (`config/workers.yaml`, `HANDOVER.md` §3). |
| **~850 MB** | **VRAM of the whole canteen *process*** — InsightFace *plus* that process's YOLO and CUDA context (708 + 62 + 81 ≈ 851). Not a second estimate of InsightFace. |
| **~341 MB** | **ON DISK.** The `buffalo_l` model pack in `~/.insightface/models/` (`HANDOVER.md` §0). A file to copy, not memory to fit. |

---

## The disease this codebase has

**A value that is true in one layer and quietly wrong in the next, with nothing in between that
complains.** Eight instances so far. It is the reason for the rules above, and the reason this file
exists as *one* file.

| artifact | looks like | actually was |
|---|---|---|
| `.gitignore: models/` | ignoring weights | swallowed **11 source files** — a fresh clone could not start, and the suite passed throughout, because tests run against the *disk*, not against *git* |
| `base.yaml: 960×540` | the analysis resolution | a **default**; hall and the canteen cameras override it — *a default is not a fleet* |
| `config/identity.py: 0.50` | the `min_score` floor | **overridden to 0.42–0.45 by every canteen profile** — in force on zero cameras |
| `_same_face(0.47)` | a 0.47 impostor | a **0.67** pair — a test whose *fixture lied* |

The `min_score` floor is now checked **at config-LOAD time, on the merged value** — the only layer
that is true — in `tests/test_config_files.py`, not on the model (a unit test may legitimately build
`RecognitionPolicy(min_score=0.1)` to exercise the matching logic; a field validator cannot tell the
difference). **A check aimed at the wrong target is not a check.**

The sharper version, learned the hard way: **a measurement inherits the causal model of whoever chose
what to measure.** Measuring the wrong quantity *precisely* still yields a wrong answer, and it feels
like rigour because there is a real number at the end of it.

---

## Remaining risk

- **The detector is unvalidated on real footage**, and the thresholds are estimates. Same cause; this
  is the big one.
- **Load has not been measured over 24 h.** The code is bounded by construction and there are tests
  for it, but "bounded in a test" and "flat RSS for a day on real cameras" are different claims and
  only the first has been made.
- **The RTSP cameras have never been reached from this machine.** `CameraStream` has run against
  fakes, not a real NVR. The reconnect logic is tested; the *credentials* are not, because they are
  not in the environment yet.

## Before this goes anywhere near production

0. **Confirm the database on the school's machine is a fresh one.** Until `fix/token-in-last-error`,
   `worker_heartbeats.last_error` stored a crashed worker's exception text verbatim — and a worker's
   exceptions quote the RTSP URL, which carries the camera password (`rtsp.build_url`, whose own
   docstring says never to log it). `notifications.last_error` had the same shape for the Telegram
   token, latent on the pinned httpx. Rows written before that fix still hold whatever leaked.
   **No scrub migration was written, deliberately**: there is no live data yet and the school's box
   starts from `qorgan db upgrade` on an empty file, so a scrub would be work against a database that
   should not exist. If a database is carried over from any earlier trial, it must be treated as
   holding secrets in the clear — rotate, or start it again.

0a. **Deploy the worker and the web process together — they now share a wire format.** The
   preview header carries `measured_fps` (the rate the camera loop counted the stream really
   delivering, as opposed to `capture.stream_fps`, which is a screenshot of one camera's web
   UI). A NEW publisher against an OLD subscriber raises `TypeError` on `PreviewHeader.decode`
   — an unknown keyword — so upgrading `qorgan supervisor` while leaving `qorgan web` on the
   previous build takes every preview off the air. The reverse order is safe. Both are started
   by `deploy\install-autostart.bat`, so in practice: stop both, update, start both.

1. **Rotate the Telegram bot token and every camera password.** The legacy committed a live token and
   a shared, near-dictionary RTSP password to its repo, so **the old credentials are burned** —
   rotating them is not optional, whatever else happens. v2 never stores them: `telegram_bot_token`
   and `rtsp_password` are read from the environment and default to blank
   (`src/qorgan/settings.py`; per-camera overrides `RTSP_PASSWORD__HALL_LEFT` &c.). *(The legacy tree
   is not in this repo, so the exact spread of the leak is taken on the record's word, not re-verified
   here. It does not change what you must do.)*
2. **Set `SECRET_KEY`.** The default is published in `src/qorgan/settings.py`, so anyone who has read
   the source can forge an operator session and reach live video. It is accepted only on a loopback
   bind in dev: a non-loopback `WEB_HOST` (what the client's §3 LAN access requires) or
   `QORGAN_ENV=prod` **refuses to start** until you set it. Generate one with
   `python -c "import secrets; print(secrets.token_urlsafe(48))"`. This is now a check, not an
   instruction — the previous version of this line was an instruction, and nothing enforced it.
3. **Put `.env` next to `pyproject.toml`.** It is read from the install root, not from the working
   directory, so a Task Scheduler entry (which has no working directory) still finds it.
4. **`QORGAN_ENV=prod` requires `WEB_HTTPS=true`.** prod marks the session cookie Secure, and a
   browser will not send a Secure cookie over plain `http://` — the login used to accept the correct
   password and then redirect to itself forever, logging nothing. If there is no TLS in front of the
   dashboard, run `QORGAN_ENV=dev` with a real `SECRET_KEY`; prod without TLS now refuses to start
   rather than presenting an unusable login screen.
5. **Decide the retention window** and put it on a schedule: `qorgan janitor --media-days N`
   (default 90; `--attempt-days` default 30). These are photographs of children.

## Open questions

- **"≥ 60 s in the canteen = ate"** is the legacy rule and the spec repeats it, so it is ported
  faithfully — but a child who finishes a meal in sixty seconds is a strange domain fact. It is config
  (`meal_outcome.ate_at_or_above_seconds`). Worth confirming.
- **`face_gate`** is **open by the human's decision — not closed by reasoning.** The small-face path
  accepts *lower scores* on the premise that "the same person comes top-1 repeatedly" corroborates a
  match. The premise is unsound in kind: **repetition defeats noise, not a systematic resemblance** —
  a persistent look-alike is corroborated exactly as eagerly as a true match, which is why weak top-1
  hits from *different* children could corroborate each other (see the identity-service plan in
  `docs/superpowers/plans/`). The path may accept smaller *faces* (its `face_gate` is real domain
  knowledge); it may not accept a lower *score*. Settled by the same canteen volunteer clip as the
  ceiling. *(A prior handoff put a "top-1 repeats 95% of the time" figure on this. I could not find
  that measurement anywhere in the repo — it is dropped as unverified, not restated. The argument
  above does not need it.)*
- The Telegram alert threshold is config, default **0.85**, with a validator enforcing
  `cap_without_skeleton (0.72) < notify_threshold` (`src/qorgan/config/bullying.py`). That inequality
  is the mechanism that makes skeleton confirmation **mandatory** for an alert. Confirm 0.85 with the
  school once there is a PR curve to justify it.

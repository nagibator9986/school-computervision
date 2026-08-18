# Qorgan AI — Rewrite Specification (v2)

**Status:** authoritative spec for the ground-up rewrite.
**Audience:** the engineer/agent building the new system.
**Reference material (read-only):**
- `C:\Users\tokmo\Downloads\qorgan ai\` — the legacy codebase. **Reference only. Never copy code from it.** Read it to recover *domain knowledge* (thresholds, gates, session rules), not structure.
- `C:\Users\tokmo\Downloads\qorgan ai\AUDIT.md` — full engineering audit of the legacy system. Every defect listed there must be absent in v2.

---

## 1. What the system is

A school video-analytics system running on one Windows machine with an NVIDIA GPU, watching ~10 RTSP cameras.

Three products in one:

1. **Bullying detection** — 6 cameras (hall ×2, stairs ×3, yard ×1). Detects physical aggression between two people, records a snapshot + clip, raises an alert, notifies staff via Telegram.
2. **Canteen attendance** — 4 cameras (entry corridor, exit corridor, 2 inside). Recognises pupils' faces, opens a meal session on entry, closes it on exit, records who ate.
3. **Web dashboard** — live camera previews, event log, pupil registry, canteen journal, analytics, settings.

Scale target: **one school, but the architecture must not block a move to many schools.** Concretely: no design decision may assume a single process, a single machine, or a single tenant's data living in module globals.

---

## 2. Non-negotiable rules

These exist because the legacy system violated every one of them. They are hard requirements, not preferences.

| # | Rule | Why (legacy failure) |
|---|---|---|
| R1 | **No file over ~500 lines. No function over ~50 lines.** | Legacy: `run_canteen()` is a single ~6 330-line function (`AUDIT.md` rounds this to ~6 300) with over 100 local variables (`AUDIT.md:213`; a precise count is not reproducible — estimates run 420–1 190 depending on whether nested scopes are included) and 5–6 nested closures. It cannot be tested, split, or reasoned about. |
| R2 | **One source of truth for detection logic.** The production worker and the eval harness call the *same* functions. | Legacy: `analyze_aggression` exists in 3 diverged copies. The eval harness measured code that does not run in production, so months of threshold tuning were built on a fake benchmark. |
| R3 | **AI workers run in their own processes and never depend on the UI.** | Legacy: which cameras get analysed depends on which browser tab the operator has open. Nobody looking at "Stairs" ⇒ stairs are not monitored at all. |
| R4 | **No secret in the repo, in a config file, in the database, in a log line, or burnt into a debug image.** All secrets come from environment variables. | Legacy: live Telegram bot token in `.env`, the RTSP password in plaintext in all 10 YAML files, in the DB, printed to stdout, and rendered as text onto debug JPEGs served by the unauthenticated web UI. (The literal used to be quoted here, which made this file the eleventh place it leaked. It must be rotated regardless — it is burned.) |
| R5 | **Every endpoint is authenticated by default (deny-by-default middleware).** | Legacy: ~50 endpoints, zero auth, bound to `0.0.0.0`. Anyone on the school network could view children's photos and live video, and delete pupils. |
| R6 | **No absolute path in the database or in code.** Media paths are relative to `MEDIA_ROOT`. | Legacy: the project moved twice; each move broke 100 % of photo/clip paths and spawned another one-off `fix_*_paths.py` script. |
| R7 | **A worker thread/process must never die silently.** Every loop body is wrapped, every exception logged, the supervisor restarts a dead worker. | Legacy: `try/finally` with no `except` in the event-writing loop. One `database is locked` kills the thread forever; the system then looks healthy but records zero events. |
| R8 | **Bounded memory.** No unbounded dict, no full-HD frame kept longer than needed, TTL eviction on every cache. | Legacy: one suppressed burst leaks ~460 MB of HD frames that are never freed. |
| R9 | **Tests exist.** Pure functions unit-tested; session state machine integration-tested; app startup smoke-tested. | Legacy: 0 tests for 18 000 lines of multi-threaded CV code. |
| R10 | **Config is a validated schema.** Unknown key ⇒ startup error. Defaults live in exactly one place. | Legacy: 225 config keys, no validation, defaults scattered across two god-files and contradicting the shipped YAML. A typo silently used a default. |

---

## 3. Data situation (read this before designing anything)

The legacy project was handed over second-hand. **Assume you are starting with no data:**

- The database file, pupil photos (`media/`), and the labelled video clips for detector evaluation are **not present** and **will be supplied later**.
- Therefore: **do not** write migration scripts against the old DB. **Do** write a clean import pipeline (pupil photos from ZIP archives per class) and a normal Alembic-style migration chain from an empty database.
- The system must run end-to-end with an empty database and zero pupils. Canteen simply records `unknown` sessions until a face registry is imported.

Known facts about the legacy data, for context only:
- Face recognition effectively did not work in production: **1 816 of 1 820 canteen records have `student_id = NULL`**. The 18 overlapping recognition thresholds in the configs are the fossil record of failed attempts to fix this by tuning.
- Pupil identity was keyed on `"Surname Firstname" + class` parsed out of photo *filenames*. Namesakes in the same class collapse into one person. v2 must key identity on an explicit `external_id`.

---

## 4. Architecture

### 4.1 Processes

```
supervisor (parent process)
├── web            — FastAPI + uvicorn, its own process
├── worker:hall_left        ─┐
├── worker:hall_right        │  one OS process per camera
├── worker:stairs_1          │  each: RTSP reader thread
├── worker:stairs_2          │        + inference loop
├── worker:yard              │        + bounded work queues
├── worker:canteen_entry     │
├── worker:canteen_exit      │
└── worker:canteen_inside_*  ─┘
```

- The **supervisor** owns the process table: spawn, health-check (heartbeat), restart with backoff, graceful shutdown. A crashed worker is restarted; a wedged worker (no heartbeat for N seconds) is killed and restarted. This is the single most important reliability feature and it does not exist in the legacy system at all.
- Workers **never** import from the web layer, and the web layer **never** imports a worker module. (Legacy: `web_dashboard.py` imports `CAMERA_REGISTRY` from `bullying_worker`, so loading a web route pulls in YOLO and torch as a side effect.)

### 4.2 IPC

- **Frames & live status:** each worker publishes JPEG previews + status over a ZeroMQ PUB socket on localhost; the web process SUBs. (`pyzmq`, pip-installable, no server to run.) Frames are encoded once, downscaled, rate-limited to ~3 FPS for preview. Never send raw frames.
- **Events:** workers write events directly to the database. Write rate is low (a few per minute), so this is safe under WAL.
- **Commands (mode switch, camera enable):** web → supervisor over a ZeroMQ REQ/REP control socket.

Rationale: this is the smallest thing that removes the GIL bottleneck and the process-global singletons, while keeping the door open to running workers on a second machine later (change the socket address for previews and commands). Note the door is only half-open: events go **directly to the database** (line above), and the default database is a SQLite *file* on the local disk — so a worker on a second machine also requires the move to Postgres (§4.3). "Change the socket address, nothing else" is true for previews and commands, and false for events.

### 4.3 Storage

- **SQLite in WAL mode** — `journal_mode=WAL`, `busy_timeout=10000`, `synchronous=NORMAL`, connections actually closed (`contextlib.closing`), retry on `OperationalError`.
- Access through **SQLAlchemy 2.0** with **Alembic** migrations, so moving to PostgreSQL later is a URL change, not a rewrite. Schema versioning is mandatory; runtime `ALTER TABLE` on startup is forbidden.
- **Never** run a data-mutating `UPDATE` inside a migration-on-startup path. (Legacy: `apply_runtime_migrations()` re-derived `person_type` for every pupil on every boot from 24 `LIKE` patterns, silently reverting manual corrections and turning any pupil whose surname contains `охран`/`повар` into staff.)

### 4.4 Configuration

- `config/base.yaml` — every default, once.
- `config/profiles/{hall,stairs,outdoor,canteen}.yaml` — the 3–4 real tuning profiles. The legacy 8 per-camera bullying configs differ in only ~29 of 56 shared keys, and most of those differences are identity/plumbing, not tuning.
- `config/cameras/{name}.yaml` — per-camera: RTSP host, role, and **zones**. Zones are genuinely unique per camera (mirror-ignore column, staircase area, normal-flow corridor lanes) — this is the one thing that must stay per-camera.
- **Pydantic models** validate the merged result. Unknown key ⇒ hard error at startup. `extra="forbid"`.
- RTSP credentials, Telegram token, and the dashboard password come **only** from environment variables. Ship `.env.example`, never `.env`.

---

## 5. Domain logic to preserve

This is the valuable part of the legacy system. The code is bad; the *knowledge encoded in it* is real and was earned in a live school hallway. Port the behaviour, not the code.

### 5.1 Bullying detection

**Pipeline:** RTSP substream (low-res, continuous) → YOLOv8n person detection + ByteTrack → per-pair kinematic scoring → suppression gates → confirmed pairs go to a slow validation queue (pose + skeleton) → confirmed events go to a recording queue (snapshot/clip + DB + Telegram). Cheap work on every frame; expensive work only for candidates. **Keep this two-tier structure — it is correct.**

**Pair metrics** (computed from bounding boxes, per frame): distance between centres; a `dynamic_threshold` scaled by the two boxes' diagonals; closing speed; a 6-frame `window_drop`; relative speed; strongest approach vector; max acceleration; max direction change; IoU overlap; `contact_like` and `hard_contact` (a tighter distance ratio or a higher IoU).

**Pair state counters** — `contact_frames`, `overlap_frames`, `aggression_frames`, `persistence_frames`, `still_frames`, `peak_close_frames`, `gap_frames`, `peak_score` (decays ×0.92). Critically: these **increment by 1 and decay by 1** per frame rather than resetting to zero. That hysteresis is deliberate; preserve it.

**Score:** additive, ~11 weighted contributions (proximity, approach, acceleration, direction change, contact, sustained overlap, "chaotic struggle" combo). Weights currently 0.7–1.2, hardcoded — in v2 they belong in the config schema.

**Zones:**
- `mirror_ignore` — tracks inside are excluded from pair analysis entirely (a reflective column produced phantom people).
- `normal_flow` — corridor traffic lanes. Score is multiplied by ~0.20–0.30 **and** the alert threshold is raised to 4.0–5.0. People walking past each other in a corridor are the dominant false-positive source.
- `staircase` — proximity threshold ×1.6, because people legitimately pass close on stairs.

**The 10 suppression gates** — this is the anti-false-positive core. Port each as a *named, individually testable predicate* in a declarative rule list, not as `if ...: continue` inside a loop:

1. `static_close` — two people standing close but still.
2. `standing_close_long` — same, sustained; with a "sudden action after static" bypass so a real attack after standing still is not swallowed.
3. `social_group` — a group in a flow zone moving the same direction.
4. `social_reapproach` — close → drifted apart → close again (friends talking).
5. `proximity_only` — close but no motion at all.
6. `normal_flow_motion_required` — in a flow zone, demand a strong action signal.
7. `crossing_pass` — two people walking past each other; with an "action evidence" bypass.
8. `staircase_pass` — one standing, one walking past on stairs.
9. `hall_final_confirmation` — hall cameras require sustained contact/overlap.
10. `benign_conversation` — confirmed pair but no victim displacement.

Plus a **post-alert separation guard**: after a real alert, if the pair separates, counters are forced down and a *new* physical contact is required before alerting again.

**Validation tier:** violence model (optional, currently absent — keep the plug-in point), pose analysis, and skeleton validation (YOLOv8n-pose, COCO-17 keypoints) producing 5 features: rapid hand motion, body fall / low posture, close upper-body contact, kick-like leg motion, sudden body displacement.

**Victim-evidence hierarchy** — the single most important anti-FP idea in the system:
- `body_fall_or_low_posture` is the only **clean** evidence.
- `sudden_body_displacement` is **motion-only** (could be perspective or box jitter).
- rapid hand motion / upper-body contact / kick-like motion are **weak** — never sufficient alone.

**Confidence:** `candidate_probability × 0.7 + validation_score × 0.3`, then adjusted by skeleton agreement. **Capped at 0.72 when the skeleton did not confirm.** Telegram fires at ≥ 0.85. The cap is therefore what makes skeleton confirmation *mandatory* for a notification. Preserve this mechanism; put all three numbers in one config block instead of hardcoding them in three files with three different values.

**Burst capture:** on a suspicious pair, briefly open the HD main stream (channel 101) for a good snapshot/clip while the low-res substream (102) continues to be the analysis source. Handle both orderings: burst finishes before the event is created, and event is created before the burst finishes (retro-attach media). **In v2, remove the legacy `burst_trigger_probability: 0.30` random gate** — evidence capture must be deterministic.

**Event merging:** deduplicate the same physical incident — same pair within 15 s (strict), same scene (overlapping track IDs) within 8 s, or spatially close pair centres. Notify once.

**Bugs to fix while porting** (do not carry these over):
- Acceleration is computed as `(px/frame)/sec`, not `px/s²`, despite a comment claiming FPS-independence. Every acceleration threshold therefore fails to transfer between cameras with different FPS. **Normalise speed to px/s first.** Until this is fixed, no recalibration is meaningful.
- `candidate_probability` is normalised against the *base* threshold, not the *effective* one, so in `normal_flow` zones (where score is ×0.2 and the threshold is 5.0) the probability is systematically wrong — and it feeds the burst trigger, the validation gate, and 70 % of the final confidence.
- `CameraStream.read()` returns the *same* frame again if the queue is empty and the frame is < 0.35 s old, so the main loop can process one physical moment 3× and inflate every temporal counter.
- `summary_text` is written to the DB before it is upgraded, so every event in the database says "Подозрение…" regardless of severity.

### 5.2 Canteen

**Session model** — the domain core. Port these rules exactly:
- Only the **entry** camera opens a session. Only the **exit** camera closes it. Inside cameras only *confirm presence* and may *late-bind an identity* to a session that was opened as Unknown.
- Staff never create meal sessions; they are tracked in a separate "staff inside" list with a TTL.
- A session older than 90 minutes is force-closed as `unknown` ("no exit recorded").
- Per-pupil cooldown of 60 s on entry and on exit (no double-counting).
- Exit will not close a session younger than 30 s (the exit camera sees the back of someone who just entered), except via an explicit quick-return path.

**"Ate / did not eat"** is decided purely by duration between entry and exit:
- `< 20 s` → "came in and left"
- `20–60 s` → "did not eat"
- `≥ 60 s` → "ate"

These thresholds are **hardcoded in the legacy service**, while the `not_eaten_seconds` / `eaten_minutes` keys in the YAML are dead — editing them changes nothing. In v2 they must be real, validated config.

**Small-face path — keep this, it is real domain knowledge.** Younger pupils' faces are systematically smaller than the size gate, so a separate "small face soft" path accumulates repeated top-1 hits at a lower score to recognise them. Do not delete it as a hack.

**Face recognition:**
- InsightFace `buffalo_l`, 512-d ArcFace embeddings, cosine similarity, decision = `score ≥ min_score AND gap ≥ min_gap` where `gap = top1 − top2`.
- **One** InsightFace instance per process (legacy created up to 5). **One** embedding matrix loaded into memory and matched with a single matmul (legacy re-read every embedding blob from SQLite for every face in every frame).
- Recognition must be a **pure function returning `(match, ranked, reason)`**. Legacy stored results on instance attributes of a shared singleton called from 4+ threads, so one thread could read another thread's top-5 — and that result closes meal sessions. This race can attribute a meal to the wrong child.
- Fix: when only one candidate exists, `gap` is set to a huge sentinel, disabling the ambiguity gate exactly when it matters most.
- **Collapse the 18 overlapping thresholds into one policy object per camera role** (entry / exit / inside), with an explicit small-face sub-policy. The legacy cascade has "strong" gates that are bypassed by "soft" gates 0.02 apart — i.e. decorative.

**Bugs to fix while porting:**
- `resolve_exit_session` will attach a recognised pupil to *someone else's* oldest Unknown session, giving them another child's dwell time and status.
- Losing a track on an inside camera calls `remove_session()`, which pops it from memory **without writing to the database** — the pupil vanishes from the record entirely.
- Sessions live only in RAM in a module-global singleton. **A process restart loses every open session silently.** In v2, sessions are persisted rows with an explicit state machine.

**Missing features to add** (the school needs them and they do not exist):
- Meal windows (breakfast / lunch) with per-meal deduplication.
- A "who did not eat today" report — requires joining against the full pupil roster. Currently unanswerable, because only pupils who were *seen* appear in the log.
- Per-class and per-period reports, with CSV export.

### 5.3 Web dashboard

Pages to keep: hall / stairs / extra-cameras live views, canteen (live sessions, journal, face-recognition debug panel), events log with filters + snapshot + clip + review/false-positive marking, pupil & staff registry with photos and re-index, analytics, settings (thresholds, mode schedule, camera registry).

Changes:
- **Authentication with roles**: `operator` (view, review events), `admin` (settings, pupils, cameras), `developer` (debug views, ROI calibration). Deny-by-default middleware. Bind to `127.0.0.1` by default.
- `/media` must be **served through an authenticated handler**, not `StaticFiles`. It contains children's photos and incident video.
- **Escape everything.** Legacy builds DOM with `innerHTML` from server JSON, so a pupil named `<img src=x onerror=…>` gives stored XSS in the operator's browser.
- **Paginate.** Legacy loads the entire events table and the entire canteen log on every render, every 2.5 s, per client.
- Event clips must actually be **playable** in the UI (a `<video>` element, not a link).
- No page-load side effects on the server. Legacy `POST /page-activate/{page}` restarts AI workers — with a 5 s `thread.join()` inside the HTTP handler — on every page load.
- Frontend: server-rendered Jinja + HTMX/Alpine. No build step, no SPA. But **zero business logic in templates** and no inline `<script>` blocks of hundreds of lines.

### 5.4 Notifications

One Telegram service. It must:
- Actually **upload** the snapshot and clip (`sendPhoto` / `sendVideo`). Legacy sends text with links to `http://127.0.0.1:8000/...`, which the recipient can never open — media delivery is broken by design.
- Have a **queue, rate limiting, retry with backoff, and 429 handling**. Legacy spawns a raw thread per alert with a bare `requests.post` and swallows every error.
- Log every send attempt to the database. Legacy logs only failures of a WhatsApp *placeholder* provider that always fails.
- Read its threshold from **one** place (legacy has 0.85 hardcoded in the worker, 0.85 again in the service, and 0.90 referenced in config comments).

### 5.5 Evaluation harness — build this early, not last

This is what makes the detector improvable, and it is the thing the legacy project most needs and least has.

- An explicit **labels file** (`video_id, t_start, t_end, label`). Filename-keyword labelling is not acceptable — legacy derives ground truth from Russian words in the filename, and silently labels anything unmatched as `normal`.
- The harness imports the **same** detection functions the worker uses. This is R2 and it is the whole point.
- **Event-level metrics** with a time tolerance, not clip-level binary. Precision, recall, F1, PR curve.
- A **regression gate**: a threshold change that lowers F1 fails the check.
- The harness must run the *full* pipeline including skeleton validation and the confidence cap — legacy's eval skips exactly the gates that decide whether an alert is sent.

Labelled clips will be supplied later. Build the harness and the label format now; a handful of clips is enough to prove it runs.

---

## 6. Database schema (v2)

Design fresh. Key corrections over legacy:

- `persons` table with an explicit `external_id UNIQUE` (from the school roster) as the identity key, and an explicit `person_type` **enum column set at import time** — not guessed by five different substring heuristics in five different files. `students` and `staff` are profiles, not a magic value in a `class` column that means "5-А" for a pupil and "Director" for a cook.
- `face_embeddings` in their own table with `model_name`, `model_version`, `dim`, and a `normalized` flag — so a model upgrade does not silently mix incompatible vectors (legacy has a dead DeepFace/Facenet512 rebuild script that would corrupt the InsightFace gallery if run).
- `canteen_sessions` **persisted**, with a state machine: `open → inside_confirmed → closed(ate|not_ate|left_immediately|unknown)`. Not a RAM dict.
- Media paths **relative** to `MEDIA_ROOT`.
- Timestamps stored as UTC, timezone-aware. Legacy stores naive local-time ISO strings and compares them by string prefix.
- Indexes on every hot column, including `face_embeddings.person_id` (legacy has no index on the join column of its hottest query).

---

## 7. Build order

Each phase ends with tests passing and something demonstrable. Do not start the next phase until the current one is green.

| Phase | Deliverable |
|---|---|
| **0 — Skeleton** | Repo, `git init` + first commit, `pyproject.toml` with **pinned** versions (incl. `onnxruntime-gpu`, CUDA `torch`, `Pillow` — all three missing from the legacy requirements), Pydantic config schema, SQLAlchemy models + Alembic migration to an empty DB, structured logging with rotation, pytest running. |
| **1 — Camera & web** | `CameraStream` (RTSP, reconnect, no stale-frame reuse), supervisor + one dummy worker process, ZeroMQ preview transport, FastAPI app with auth + roles, live preview page. Camera coverage is fixed by config and independent of the UI. |
| **2 — Bullying detector** | Pure detection core (`geometry`, `tracking`, `scoring`, `gates` as a rule list), the bullying worker, event recording, the **eval harness** on the same core. Port the 10 gates with unit tests for each. Fix the acceleration units and the flow-zone probability bug. |
| **3 — Faces & canteen** | Face recognition service (one model instance, in-memory matrix, pure function), pupil/staff import from ZIP archives, persisted canteen session state machine, canteen worker, journal + "who did not eat" report. |
| **4 — Notifications & hardening** | Telegram with queue/retry/media upload, media retention janitor, supervisor restart policy, load test with all 10 cameras, memory profile over 24 h. |

---

## 8. Definition of done

- All 10 cameras analysed continuously for 24 h with flat memory and no dead workers.
- No secret anywhere except environment variables.
- Every endpoint requires authentication; `/media` is access-controlled.
- The eval harness runs against the production detection functions and reports precision/recall/F1.
- A pupil can be imported, recognised at the canteen entry, and correctly recorded as having eaten.
- `pytest` green; no file over ~500 lines; no function over ~50 lines.

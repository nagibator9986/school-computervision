# Qorgan AI v2

School video analytics on one Windows machine with an NVIDIA GPU:

1. **Bullying detection** — 6 cameras (hall ×2, stairs ×3, yard ×1). YOLOv8 + pose + skeleton validation.
2. **Canteen attendance** — 4 cameras (entry, exit, 2 inside). InsightFace recognition, meal sessions.
3. **Web dashboard** — live previews, event log (each event with its clip, and with the reason an alert was or was not sent), canteen journal, the pupil register and duplicate merging, lesson reports, Telegram alerts, plus the installation pages: camera state, users, logs, settings, backups, notifications. Every page is behind a capability, and the capability table is `src/qorgan/roles.py` — `canteen_staff` reaches the canteen and nothing else. The roster is still **imported** from the CLI (`qorgan pupils import-roster` / `gallery-report` / `report` / `merge`); the dashboard reads it and merges duplicates.
4. **Classroom lesson metrics** — `/lessons`, admin only, counts per anonymous track with the caveat rendered on every page. **No classroom camera ships in `config/cameras/`** and no threshold in it has been validated against a real lesson; see `HANDOVER.md` §7 before promising it to anyone.

This is a ground-up rewrite. `REWRITE_SPEC.md` is the authoritative spec; the legacy
system in the parent directory is **reference only** — read it for domain knowledge
(thresholds, gates, session rules), never copy code from it.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# The AI stack needs the CUDA wheel index. A bare `pip install torch` silently gives
# you a CPU build that works and is ~40x too slow (the H-14 defect from the audit).
pip install -e ".[ai]" --extra-index-url https://download.pytorch.org/whl/cu128

# `insightface` depends on plain `onnxruntime`, which SHADOWS `onnxruntime-gpu` and
# takes CUDAExecutionProvider with it. pip installs both without complaining.
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0

qorgan doctor     # confirms torch AND onnxruntime can both see the GPU
```

Then copy `.env.example` to `.env` and fill it in. **No secret lives anywhere else** —
not in YAML, not in the database, not in a log line, not on a debug image.

## Run

```bash
qorgan doctor                           # torch AND onnxruntime must both see the GPU
qorgan db upgrade                       # empty database, zero pupils
qorgan user add admin --role admin      # prompts for a password
qorgan config validate                  # all 10 cameras, credentials redacted

# `--role` also takes `psychologist` and `superadmin`. The superadmin manages the
# schools register and is SHELL-ONLY -- the accounts page cannot assign it.
# `--school <slug>` is optional on a one-school install and required once a second exists.

qorgan supervisor                       # the worker fleet (one process per group)
qorgan web                              # the dashboard, on 127.0.0.1:8000
```

Pupils and the canteen:

```bash
qorgan pupils import-roster student_photos/student_photos   # the folder decides who is who
qorgan pupils gallery-report            # who is enrolled twice, and can this gallery work?
qorgan pupils report --csv today.csv    # who ate, and who has no meal record
```

Tuning the detector (needs labelled clips — see HANDOFF.md):

```bash
qorgan eval scan                        # detector first: every moment it would have fired on
qorgan eval sample                      # the stratified worklist a human must label
qorgan eval label                       # watch each candidate and label it
qorgan eval run                         # precision / recall / F1 + a PR curve
qorgan eval save-baseline               # then `eval gate` fails any regression
qorgan eval gate                        # fails if a change made the detector worse
```

`eval run` is blocked on human labels, not on code — see `HANDOFF.md`.

Housekeeping — put both of these on a schedule. Nothing in this system grows forever,
and one disk is not a backup:

```bash
qorgan janitor --media-days 90 --dry-run
qorgan backup                           # VACUUM INTO: safe on a live system, includes the WAL
```

`qorgan backup` writes `data/backups/qorgan-<date>.sqlite3`, opens it and integrity-checks
it before reporting success, and refuses to overwrite an existing file. Note that
`data/backups/` is the **same disk** as the database — copying it off the machine is a
decision for the school, not a default we can ship.

**Windows autostart** — `deploy\install-autostart.bat` registers `supervisor` and `web`
with Task Scheduler. See `docs/windows-autostart.md`; the launchers `cd` to the install
root themselves, because Task Scheduler sets no working directory and `.env`, the YOLO
weights and every default path resolve against it — each failing silently.

The supervisor and the web server are independent processes and neither needs the
other to start. That is the point: **which cameras get analysed is decided by
`config/workers.yaml`, never by which browser tab is open.**

## Layout

```
config/           base.yaml <- profiles/*.yaml <- cameras/*.yaml   (no secrets, ever)
src/qorgan/
  settings.py     env-only secrets, per-camera RTSP credential overrides
  redaction.py    scrubs secrets from every log line and JSON dump
  logging_setup.py
  paths.py        media paths are relative to MEDIA_ROOT; absolute paths are rejected
  enums.py        one definition of person_type, camera role, session state, ...
  roles.py        the capability table: what each role may open, named once
  config/         Pydantic schema; unknown key => startup error
  db/             SQLAlchemy 2.0 models + WAL engine
  capture/        RTSP and recorded-file frame sources
  detection/      the bullying detector — imported by BOTH the worker and the eval harness
  classroom/      lesson metrics: anonymous tracks, no faces, no person_id
  canteen/ faces/ identity/    recognition, the roster, meal sessions
  events/ notify/ preview/     event rows, media, Telegram, the preview bus
  worker/ supervisor/          one process per group; heartbeat and restart
  web/            FastAPI app, deny-by-default auth, routes/ and templates/
  evaluation/ planning/ maintenance/ diagnostics/
migrations/       Alembic
tests/
```

## Hard rules

These are enforced by tests, not by good intentions. Each exists because the legacy
system violated it — see `AUDIT.md` in the parent directory.

| Rule | Enforced by |
|---|---|
| No file > ~500 lines, no function > ~50 lines | `tests/test_code_limits.py` |
| One source of truth for detection logic (worker == eval harness) | Done: `worker/bullying.py` and `evaluation/harness.py` both import `BullyingDetector` from `qorgan.detection` |
| Workers are separate processes, never depend on the web UI | Phase 1: supervisor |
| No secret in code, config, DB, logs, or debug images | `tests/test_no_secrets.py`, `redaction.py` |
| Every endpoint authenticated by default | Phase 1: deny-by-default middleware |
| No absolute path in the database | `RelPath` SQLAlchemy type rejects them at bind time |
| No worker dies silently | Phase 1: supervisor heartbeat + restart |
| Tests exist | `pytest` |

## Deviations from the spec, agreed with the owner

- **Worker groups, not one process per camera.** Measured on the 4 GB RTX 3050 by
  `qorgan plan-workers` — which measures the GPU you have rather than quoting the one we
  had: CUDA context + process ~81 MB, YOLOv8n + ByteTrack ~62 MB *per camera*, pose ~12 MB
  *per bullying group*, InsightFace ~708 MB *per canteen group*. The expensive thing is not
  the CUDA context — it is InsightFace. So `config/workers.yaml` maps cameras to worker
  processes, and grouping the *canteen* cameras is what makes the fleet fit. On a larger
  GPU, write one camera per group; no code changes. Coverage still comes from config, never
  from the UI.
- **Identity is given, not inferred.** `qorgan pupils import-roster` reads the school's photo
  directory: the FOLDER decides class and person type, the FILENAME carries the school's own
  id. A filename that does not match `student|staff_<id>_<timestamp>.jpg` is a hard error
  naming the file — it is never a guessed identity. (The old name-derived `external_id`
  machinery is deleted.)
- **Telegram threshold is config, default 0.85**, with a validator enforcing
  `cap_without_skeleton (0.72) < notify_threshold (0.85)` — that inequality is what makes
  skeleton confirmation mandatory for an alert.

See `HANDOFF.md` for current build state.

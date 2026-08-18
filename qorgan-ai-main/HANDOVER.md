# HANDOVER — installing Qorgan AI v2 on the school's machine

For **the client's engineer**: the person who installs this on the school's box, points it at the
school's cameras, and runs it on site. It assumes you have never seen this repository.

> This is not the internal engineering handoff. That is **`HANDOFF.md`** — what our team knows,
> what is unfinished, and why. Read that one too if you intend to change the code. This file only
> gets it *running*.
>
> What the system is honestly capable of today — and what it must not yet be trusted with — is
> **`docs/client-note-2026-07-17.md`** (Russian). **Read it before you rely on a bullying alert.**
> It is dated, and it is not rewritten after the fact: **§7 below says which of the gaps it names
> have been built since**, and that section is the newer answer.

Every command below was executed against this tree before it was written down. Where a thing is
unverified or unmeasured, it says so.

---

## 0. What to send, and how

**Git carries none of this.** `.gitignore` excludes model weights (`*.pt`, `*.onnx`, `/models/`)
and every directory that holds photographs or video of children. A fresh clone does not run.

| what | size | where it is on the dev machine | how it travels |
|---|---|---|---|
| `yolov8n.pt` | 6.5 MB | repo root | copy into the repo root |
| `yolov8n-pose.pt` | 6.8 MB | repo root | copy into the repo root |
| InsightFace `buffalo_l` | ~341 MB **on disk** | `C:\Users\tokmo\.insightface\models\buffalo_l\` | copy to `<user>\.insightface\models\buffalo_l\`, **or** let it auto-download (below) |
| roster photos, **142 files / 17 folders** | ~small | `student_photos/student_photos/` | **hand-copied. Never emailed, never committed.** |
| `.env` | — | **does not exist here** | **written on their machine. Never sent.** |

**The two `.pt` files must sit in the directory you launch `qorgan` from.** `config/base.yaml` names
them as bare filenames (`model: "yolov8n.pt"`), which Ultralytics resolves against the *current
working directory* — and if it does not find them it silently downloads them from the internet into
that directory. Both behaviours are CWD-dependent, so **always start qorgan from the install root**.
(`.gitignore` mentions a `scripts/fetch_models.py`; **that script does not exist**. Copy by hand.)

**buffalo_l auto-download.** `faces/recognizer.py` builds `FaceAnalysis(name="buffalo_l")`, and
InsightFace's `FaceAnalysis.__init__` calls `ensure_available('models', name, root='~/.insightface')`
— which downloads the pack on first use if it is absent. So **if the school's machine has internet,
skip the 341 MB copy** and let the first run fetch it. If it does not, copy the folder. Either way the
files must be in place *before* the first canteen worker starts, not during.

**The 142 photos are the school's own children.** They are gitignored (`student_photos/`,
`original_student_photos/`, `*_student_photos/`) for that reason. Carry them on a disk, hand to hand.
The 17 folders are the classes (`1-А` … `11-Б`, `staff`, `учитель`) and **the folder decides who is
who** — do not flatten or rename them; `pupils import-roster` reads the tree.

---

## 1. Install order

**Python 3.11, and only 3.11.** `pyproject.toml` declares `requires-python = ">=3.11,<3.12"`. 3.12
will refuse to install; 3.10 is not supported.

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev]"

# The AI stack needs the CUDA wheel index. A bare `pip install torch` gives a CPU build
# that works fine and is ~40x too slow, silently.
pip install -e ".[ai]" --extra-index-url https://download.pytorch.org/whl/cu128

# `insightface` depends on plain `onnxruntime`, which SHADOWS `onnxruntime-gpu` and takes
# CUDAExecutionProvider with it. pip installs BOTH without complaining, and face recognition
# then runs on the CPU at ~40x too slow with only a warning nobody reads.
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0
```

Then, in this order:

```bash
qorgan doctor                          # torch AND onnxruntime must BOTH see the GPU
qorgan db upgrade                      # empty database, zero pupils
qorgan user add <name> --role admin    # prompts for the password; never takes it on argv
qorgan config validate                 # loads and validates every camera config
qorgan plan-workers --force            # MEASURES THIS GPU, rewrites config/workers.yaml
qorgan pupils import-roster student_photos/student_photos
qorgan supervisor                      # the worker fleet   (leave running)
qorgan web                             # the dashboard      (leave running, separate process)
```

`qorgan doctor` must print `onnx SESSION ran: CUDAExecutionProvider`. It builds a real ONNX session
rather than trusting the advertised provider list — advertising CUDA and failing to use it is exactly
the failure this catches. If it says CPU, the `onnxruntime` uninstall above did not take.

`--role` accepts `operator | admin | developer | canteen_staff | psychologist | superadmin`.

**`superadmin` is shell-only** — it manages the schools register (§14), belongs to no school, and
the accounts page deliberately cannot assign it, so one school's headteacher cannot mint an account
that reaches every other school. Create it with `qorgan user add <name> --role superadmin`.

**`--school <slug>`** names which school an account belongs to. It is optional while the
installation serves one school and **required once a second exists** — with several schools and no
`--school`, the command refuses rather than guessing. It is ignored for `superadmin`, which belongs
to no school by definition.

`qorgan --help` lists everything: `config`, `db`, `eval`, `pupils`, `identity`, `plan-workers`,
`doctor`, `supervisor`, `web`, `backup`, `janitor`, `user`.

Once it runs, make it start itself: **`docs/windows-autostart.md`** (`deploy\install-autostart.bat`).

---

## 2. `.env` — every key

Copy `.env.example` to `.env` **next to `pyproject.toml`, and nowhere else.** It is read from the
install root, *not* from the directory you happen to start in (`settings.py: ENV_FILE`), because
Windows Task Scheduler starts processes with no working directory. A `.env` anywhere else is not read
and the process starts on the defaults below **without saying so**.

| key | default | mandatory? | if missing / wrong |
|---|---|---|---|
| `SECRET_KEY` | `dev-only-insecure-key` | **YES**, in any real deployment | Signs session cookies. The default is **published in `settings.py`** — anyone who has read the source can forge an operator session and reach live video of children. **Enforced:** a non-loopback `WEB_HOST` **or** `QORGAN_ENV=prod` now **refuses to start** on the default key. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `QORGAN_ENV` | `dev` | no | `prod` marks the session cookie Secure. **`prod` with `WEB_HTTPS=false` refuses to start** — see below. |
| `WEB_HTTPS` | `false` | see below | The **only** honest source for the cookie's Secure flag. True while serving plain `http://` ⇒ the browser never returns the cookie and **login loops forever, logging nothing**. False while serving `https://` ⇒ you only lose a hardening flag. |
| `WEB_HOST` | `127.0.0.1` | for LAN access | `0.0.0.0` is what the client's §3 (panel reachable from other computers) requires — and it serves children's photographs and live video to the LAN, so it forces the `SECRET_KEY` check above. |
| `WEB_PORT` | `8000` | no | The client reported `0.0.0.0:8000` failing and `127.0.0.1:8010` working. That was almost certainly the login-loop above, not the port. |
| `RTSP_USER` | `admin` | effectively yes | Fleet-wide default camera user. |
| `RTSP_PASSWORD` | *(blank)* | effectively yes | Blank ⇒ every camera fails to authenticate. |
| `RTSP_USER__<CAMERA>` / `RTSP_PASSWORD__<CAMERA>` | — | no | Per-camera override; suffix is the camera name upper-cased, e.g. `RTSP_PASSWORD__HALL_LEFT`. Use these if the NVRs do not share a password — **they should not.** |
| `TELEGRAM_BOT_TOKEN` | *(blank)* | no | **Blank disables Telegram entirely, silently and by design.** The system runs without it. |
| `TELEGRAM_CHAT_ID` | *(blank)* | no | Both must be set or Telegram is off (`settings.telegram_enabled`). |
| `DATABASE_URL` | `sqlite+pysqlite:///./data/qorgan.sqlite3` | no | Relative paths resolve against the install root. |
| `MEDIA_ROOT` | `./media` | no | Snapshots **and event clips** live here (`snapshots/<date>/`, `clips/<date>/`) — so this is the directory that grows, and `qorgan janitor --media-days` is what bounds it (§5, §7). **Paths are stored relative in the database** — moving this directory does not break them (the legacy stored absolute paths and every move broke 100% of them). |
| `LOG_DIR` | `./logs` | no | |
| `CONFIG_DIR` | `./config` | no | |
| `SCHOOL_TIMEZONE` | `Asia/Almaty` | no | Validated at startup; an unknown zone fails immediately rather than at report time. All timestamps are stored UTC; this is only the local day boundary. |
| `PREVIEW_ADDRESS` | `tcp://127.0.0.1:5556` | no | Workers PUB, web SUBs. **Loopback only** — these are frames of children and they do not leave the machine. |
| `PREVIEW_STALE_AFTER_SECONDS` | `10.0` | no | |
| `LOG_LEVEL` | `INFO` | no | |
| `LOG_JSON` | `true` | no | |

**Rotate the Telegram bot token and every camera password before this goes live.** The legacy
committed a live token and a shared RTSP password to its repository — the old credentials are burned
regardless of anything else you do. v2 reads both from the environment only and defaults them to
blank; no secret is ever written to YAML, the database, a log line, or a debug image.

---

## 3. Their hardware

Reported by the client (§3): **RTX 4070**, driver 591.86, torch 2.11.0 with CUDA, ONNX Runtime with
TensorRT/CUDA/CPU providers, environment showing **CUDA 13.1**.

**CUDA 13.1 is a non-issue.** This development machine reports exactly the same
(`nvidia-smi` → `CUDA Version: 13.1`, driver 591.44) and runs `torch 2.11.0+cu128` with
`qorgan doctor` green. The driver's CUDA version is an upper bound, not a requirement. Do not go
chasing it.

**You must run `qorgan plan-workers --force` on their box.** The `config/workers.yaml` that ships was
measured on a **4 GB RTX 3050 Laptop** and is *wrong for a 4070* — it is correct, and it is for the
wrong card. It groups cameras to fit 4 GB:

```
bullying process (YOLOv8n + YOLOv8n-pose)   ~155 MB   VRAM, whole process
canteen  process (YOLOv8n + InsightFace)    ~850 MB   VRAM, whole process
```

**The expensive thing is not the CUDA context — it is InsightFace at ~700 MB per instance.** (That
~700 is the same measurement as the **~708 MB** in `HANDOFF.md`, rounded; the ~850 above is that
instance plus the canteen process's own YOLO and CUDA context. **Neither is the ~341 MB in §0**,
which is the pack's size **on disk**.) That is
the whole reason for grouping. On a 12 GB card, a **10-camera fleet at one OS process per camera fits
at roughly 4334 MB of 12282** — which is what the spec asks for and what the 3050 could not do.

**Treat that 4334 MB as a planning estimate, not a measurement you can rely on: it was not measured on
this machine** (there is no 4070 here). `plan-workers` measures the GPU you actually have — it loads
exactly what each kind of worker loads in production — rather than quoting the one we had. Run it,
and use what it writes. `--force` is required to overwrite an existing `workers.yaml`; it refuses by
default. `--dry-run` prints without writing.

Every enabled camera must appear in exactly one group or startup fails. A camera nobody runs is a
camera nobody is watching.

---

## 4. Their cameras are already configured

`config/cameras/*.yaml` ship the school's **real hosts**, read from their system. Verified in the
files:

| camera | host | profile |
|---|---|---|
| `hall_left` | **192.168.1.4** | `hall` |
| `hall_right` | **192.168.1.2** | `hall` |
| `canteen_entry` | **192.168.1.12** | `canteen_entry` |
| `canteen_exit` | **192.168.1.6** | `canteen_exit` |

Also configured: `stairs_floor1`, `stairs_floor2`, `stairs_floor2_aux`, `yard_entry`,
`canteen_inside_left`, `canteen_inside_service` — ten in total.

**You write nothing here unless an IP has moved.** The zones are configured too: `hall_left` carries a
`mirror_ignore` zone over the reflective column that produced phantom people for months, and both hall
cameras carry `normal_flow` lanes. Run `qorgan config validate` after any edit.

Note the analysis resolution is **per profile, not fleet-wide**: `hall.yaml` sets 1280×720 and
overrides `base.yaml`'s 960×540. **Every px/s threshold in a profile is expressed in pixels of that
profile's frame — change the resolution and every one of them is void.**

---

## 5. Things that have cost us real time

- **`pytest -q` silently drops its summary line in this environment.** Use
  `pytest --junitxml=out.xml` and read the XML. **A wrong path exits 4 having collected NOTHING and
  looks exactly like success** — always confirm the collected count is non-zero.
  Baseline on this tree, re-run **2026-07-24**: **`tests=1532 failures=0 errors=0 skipped=0`**
  (exit 0, 155 s). If your run collects a different number, the tree moved — trust your XML, not
  this line.

- **The data directories are photographs and video of children and are never in git.**
  `student_photos/`, `original_student_photos/`, `media/`, `logs/`, `eval/clips/`, `eval/crops/`,
  `Bandicut/`. **Never `git add -A` or `git add .`** — stage by explicit path, and run `git status`
  before every commit.

- **RTSP has never been reached from this machine. Not once.** `CameraStream` has run against fakes,
  never against a real NVR. The reconnect logic is tested; the *credentials and the network path are
  not*, because they have never existed here. **First contact with a real NVR happens on site, and it
  is the largest untested surface in the system.** Budget time for it, and expect the first failure to
  be authentication or a codec, not the code.

- **Start qorgan from the install root** (§0) — both the YOLO weights and `.env` resolve from there.
  This is why `deploy\qorgan-*.bat` `cd` to their own parent before doing anything: Task Scheduler
  sets no working directory, and all three failures above are silent.

- **Retention is a decision, not a default.** `qorgan janitor --media-days N` (default 90;
  `--attempt-days` default 30). Put it on a schedule. These are photographs of children.

- **A backup is a decision too.** `qorgan backup` copies the database correctly, including the WAL —
  but into `data/backups/`, which is the same disk. **Copy it off the machine.** And note that a
  backup is a copy of children's data: it belongs where the roster photographs are allowed to be,
  and nowhere else.

---

## 6. Verifying it works, in order

```bash
qorgan doctor                        # GPU: both stacks
qorgan config validate               # every camera config loads
qorgan db current                    # the migration revision
qorgan pupils gallery-report         # can this gallery work at all, and who is enrolled twice?
qorgan identity camera-report        # CAN THIS CAMERA RECOGNISE ANYBODY, at the resolution it is fed?
```

`identity camera-report` is the one to run at the school, on the **canteen entry** camera, on day one.
It answers "can this camera recognise anybody" honestly and refuses to let a camera that cannot be
assigned to recognition — so that this is discovered on the first day rather than after six months of
threshold tuning. **The hall cameras cannot** (measured: 0 recognitions in 14 970 faces; median face
11.5 px at 1280×720 — optics, not tuning). **The canteen entry camera is close-range and unmeasured.**
That is a genuinely different question and it needs one clip to answer.

---

## 6a. Autostart and backup — read `docs/windows-autostart.md`

Three things the client asked for (§7, §11 items 14 and 15) exist now:

- **The Telegram message carries the reasons, the time and the event id.** It used to be
  `"{severity} — {camera} ({confidence}%)"` and nothing else. The time is in the school's
  timezone, not UTC — the database is UTC and the school is UTC+5, and a teacher sent to
  the wrong five minutes of CCTV concludes the system is lying to them.
  Reasons are stored on `events.reasons` (migration `0004`), in Russian in the message.
- **`qorgan backup`** — `VACUUM INTO`, safe on a live system, includes the WAL, and reads
  the copy back before reporting success. **`data/backups/` is the same disk as the
  database; copying it off the machine is a decision the school has to make.**
- **`deploy\install-autostart.bat`** — registers both processes with Task Scheduler.

**Run `qorgan backup` and the janitor on a schedule** (`docs/windows-autostart.md` §4).
Neither of them needs the Administrator prompt the boot-triggered tasks do.

What is still true and matters more than any of the above: **an event that fires the
alert is only as trustworthy as the detector**, and the reasons now in the message are
the skeleton's, not a human's. They explain what the system thought it saw. Read
`docs/client-note-2026-07-17.md`.

## 6b. "Signal to operator" is the default, not a mode you build

The detector was measured to fire on ordinary walking (see the client note). So the honest
way to run the bullying module today is **signal, not alarm** — and that is already how a
fresh install behaves, by construction:

- **Every candidate that survives the ten gates is recorded as an event** (`_create` in
  `worker/bullying.py`), with a severity from its confidence, and appears on the `/events`
  page for a human to review and mark (`POST /events/{id}/review`, capability
  `REVIEW_BULLYING`). This happens regardless of Telegram.
- **Telegram is the separate, opt-in auto-alarm.** With `TELEGRAM_BOT_TOKEN` /
  `TELEGRAM_CHAT_ID` blank, `settings.telegram_enabled` is False and nothing is pushed —
  the operator simply reviews the dashboard. This is the shipping default.

To run signal-only: **leave the Telegram token blank.** To turn the auto-alarm on later
(once the detector earns it), set both env vars. Nothing else changes.

**The honest caveat, in writing so nobody is surprised:** because the detector cannot yet
rank walking below fighting, the `/events` dashboard will show many false candidates at
high severity. Signal-mode removes the 2 a.m. false Telegram; it does **not** reduce the
operator's review burden. That burden is the detector problem, fixed on-site with real
cameras and labels — not by a config flag. Do not read "signal to operator" as "solved".

---

## 7. What is built and what is not — do not promise the wrong half

`docs/client-note-2026-07-17.md` is the full disclosure with the evidence, and it is **dated**: it
was written on 2026-07-17 and sent to the client, so it is not edited afterwards. **Four of the gaps
it names have been built since.** Where that note and this section disagree, this one is newer.
Every line below was re-checked against the code on **2026-07-30**.

### Built since that note — do not promise their absence either

- **Video clips ARE written in production, by the bullying worker.** `write_clip()` lives in
  `events/recorder.py` and is called from `worker/bullying.py`: once when the event row is created,
  and again to back-fill an event whose first look had nothing to write yet. What governs the length
  is **`bullying.clip_seconds`** in the camera's profile — the footage leading up to the alarm,
  default **3.0 s**, and the schema accepts only `> 0` and `<= 15`. **There is no setting that turns
  clips off**: no zero, no `enabled: false`.
  A clip can still be **missing from an individual event**, and you will see that on site: the path
  column is left NULL when the ring buffer held nothing (the first seconds after a camera connects)
  or when the write failed. That event then carries a JPEG snapshot, or nothing — a path is **never**
  recorded for a file that is not on disk. `/events` renders a `<video>` only for rows that have one,
  and Telegram sends the video *after* the text, so a video that fails to send cannot undo an alert
  already delivered.
  **Clips are disk, and they are the school's disk**: `MEDIA_ROOT/clips/<date>/`, swept by
  `qorgan janitor --media-days` alongside the snapshots (§5). Nothing else in the system writes
  clips — the canteen and the lesson pages have none.
- **The reason an alert was not sent IS recorded.** `events.telegram_skip_reason` (migration `0007`),
  written by the bullying worker at the moment it decides and left NULL when the alert *was* sent.
  It is a fixed enumeration, not free text — pose not analysed, no skeleton confirmation, weak
  evidence only, not confident enough, already told — and `/events` and `/notifications` show it in
  Russian. Client §7's requirement is met. (The unrelated `skip_reason` on `SkeletonResult` still
  exists and is still a different thing.)
- **There IS a Students page.** `/pupils` (capability `VIEW_PUPILS` — operator, admin and developer)
  lists pupils and staff with class, type and recognition state; `/pupils/{id}/canteen` is one
  person's meal history; `/pupils/duplicates` is where two school ids are merged, and **merging alone
  needs `MERGE_PERSONS`, which only `admin` holds**. There are **16** page templates, not four. The
  CLI has not gone anywhere and is still how the roster arrives: `qorgan pupils import-roster` /
  `gallery-report` / `report` / `merge` (client §9 / §11 item 9).
- **Classroom lesson metrics exist** — `src/qorgan/classroom/`, its own worker, and the pages
  `/lessons` and `/lessons/{id}` behind `VIEW_LESSON_METRICS`, which **only `admin` holds**. Read the
  limits before you mention this to the school, because they are the feature as much as the counts
  are: it reports hands raised, times stood up, time away from a place, and presence — **per
  anonymous track, never per child**. It reads no faces, stores no person id, produces no score, no
  ranking and no flag, and says **nothing about the teacher**. **No threshold in it has been
  validated against a real lesson**; every number in `config/classroom.py` is an estimate, labelled
  as one. And it has **no camera**: `config/cameras/` holds ten cameras, six bullying and four
  canteen, and **not one classroom camera**, so on a fresh install `/lessons` is an empty page.
  Pointing a camera at a classroom is a decision the school makes, not something that ships.

### Still NOT built — do not promise these

- **No weapons detection.** Nothing under `src/`, `tests/`, `config/` or `migrations/` mentions
  weapons at all. It exists only on an unmerged branch (`feat/weapons-detection`), and an unmerged
  branch is not something the school has (client §12).
- **No teacher analytics.** Deliberate rather than pending: nothing in `qorgan/classroom` measures
  the adult in the room, and client §12.5 is not built.
- **There ARE now a psychologist role and a superadmin role.** This entry said there were neither,
  and it was true until `feat/psychologist-cabinet` and `feat/multi-school` were merged. `UserRole`
  is now `operator`, `admin`, `developer`, `canteen_staff`, `psychologist`, `superadmin`, and
  `parse_role` still rejects anything else. The psychologist's cabinet (`/psychologist`) and the
  schools register (`/schools`) both exist, and each arrived in the same change as the capabilities
  that guard it — which is the rule `roles.py` states and the reason neither role existed earlier.
  `superadmin` is the only role a form can never assign: see §2. Roles are still capabilities and
  not a rank: `CANTEEN_STAFF` holds exactly `VIEW_CANTEEN` and nothing else — no events, no review,
  no previews, neither half of the media tree (client §14, satisfied) — and `SUPERADMIN` holds no
  child-facing capability at all.
- **The bullying detector must not be trusted yet.** This is the headline of the client note, nothing
  above changes it, and it is the one item on this page that matters most. Read the note. §6b is how
  to run the system honestly in spite of it.

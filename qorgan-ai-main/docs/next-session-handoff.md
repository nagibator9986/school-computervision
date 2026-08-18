# Handoff for the next session — Qorgan AI v2

**Read this first, then `docs/client-spec-2026-07-16.md`, `REWRITE_SPEC.md`, `docs/client-note-2026-07-17.md`, `eval/FINDINGS.md`, `docs/questions-for-school.md`.**
Rewritten 2026-07-27 (previous edition 2026-07-24). Repo: private GitHub `MakazhanAlpamys/qorgan-ai`.

> **What changed since the 2026-07-24 edition.** That edition said "the found-defect backlog is
> exhausted, `main` is shippable, remaining work is on-site". Since then the **entire web control
> panel (client §9) was built and merged** — seven `feat/web-*` branches — plus CSRF, two
> secret-redaction fixes, a capability split, a timezone fix, and two layers of test isolation.
> `main` went from 1532 to 1884 tests and from 188 to 232 commits. §0, §3, §7, §9, §10 and §11 are
> new or rewritten. **§1, §2, §4, §5, §6, §8 are carried forward and still stand** — nothing since
> has touched the detector, the camera facts, or the on-site blockers.

> **Consolidated 2026-07-30 — read this before anything else.** Two days of work lived on three
> separate documentation branches and **none of it was on `main`.** A session starting from trunk saw
> §0 and §0a and nothing after them: not the login defect, not the lost corpus, not the three pilot
> modules, not the eight-table migration blast radius. All three branches are now merged here, on
> `docs/consolidated-2026-07-30`, together with `docs/next-session-prompt.md`.
>
> **Four dated sections, chronological and append-only: §0 (27th) → §0a (28th) → §0b (29th) → §0c
> (30th). Read all four, oldest first.** A later section corrects an earlier one **in words**, never
> by editing it, so where two disagree **the newer one wins** — and where an earlier section is fully
> restated by a later one, the earlier now carries a `Superseded by` note saying so.
>
> Every checkable claim was re-run as a command during consolidation. What had gone stale in a day is
> corrected in place and **labelled as a correction**. What was NOT re-run: the test suites — every
> suite total in this document is somebody else's run and is named as such.

---

## 0. State right now, measured 2026-07-27

- **`main` = `086eb96`, 232 commits, ruff clean.** Test suite: **1884 collected**.
- **The full run I took on `main` came back RED — one failure — and it is a load flake I caused.**
  `tests/test_analysis_rate.py::test_the_loop_measures_the_rate_it_is_actually_being_fed`,
  `AssertionError: the loop stalled at 62/90`. The assertion is `processed >= frames - 10`, i.e. a
  throughput claim, and I ran the suite **concurrently with four subagents**; it took 531 s against
  a normal ~378 s. **This is the isolation rule (§7) broken by the person who wrote it.**
  **Do not record `main` as green or as red on my word — re-run it alone and parse the XML.**
- **The same log carries dozens of `sqlite3.OperationalError: no such table: notifications`** from a
  leaked notifier thread polling after its test ended. They did not fail a test in that run. They are
  the defect §10 is about, and they are still present on `main`, because the fix is not merged.
- **Four branches are unmerged; two of them are this session's work and are the live decision:**

  | branch | tip | ahead | state |
  |---|---|---|---|
  | `fix/no-thread-outlives-its-test` | `55c3ee9` | 2 | guard, rebased on top of the fix below. **Not pushed.** |
  | `fix/leaked-worker-threads` | `ef0aad4` | 1 | the `lifespan` fix. Subsumed by the branch above. **Not pushed.** |
  | `fix/skeleton-scale-invariant` | `9e3835f` | 2, 29 behind | **parked on purpose** — the fix did not work, see §4 |
  | `fix/motion-fps-reference-scaling` | `23cf308` | 1, 84 behind | **held on the owner's decision**, see §4 |

  `ef0aad4` is an ancestor of `55c3ee9`, so **merging the guard branch brings both.**
- **Five commits in this repository exist nowhere but this disk** (`git rev-list --all --not
  --remotes`): the two above, **and the three commits behind the stash** (`af5b80b`, `8ada5f4`,
  `a49b085`). Push the two; decide the stash. Neither survives a disk failure today.
- **18 worktrees, 22 local branches.** 17 branches are fully merged into `main` and 13 of those still
  hold a worktree — stale, safe to remove, but ask before pruning: a worktree is also a record.
- **One stash, orphaned:** `stash@{0}` = `af5b80b`, 2026-07-14, `WIP on feat/detector-calibration`,
  holding `tests/test_eval_label.py` (+74). Its base branch no longer exists locally. `git status`
  will never mention it. **Decide whether it lives or dies; do not let it rot unseen.**
- **The data is video and photographs of children.** `student_photos/`, `original_student_photos/`,
  `class/`, `фотки учеников/`, `bullying_camera/`, `canteen/`, `Bandicut/`, `media/`, `logs/`,
  `eval/clips/`, `eval/crops/`. **NEVER open them — names and counts only, counted in Python, never
  in the shell.** **Never `git add -A` / `git add .`.** The check that answers the real question is
  **`git add -An`** — what would actually be staged. Zero media has ever been committed.
- **This nearly stopped being true on 2026-07-27.** The client's new photo folder arrived
  **unignored**, and `git add -An` listed **323 of its paths, 141 of them photographs of children**.
  A rule is now in `.gitignore` — and note *how* it was got right: the first attempt wrote the path
  **in quotes**, `.gitignore` has no quoting, and all 323 paths were still listed. **`git add -An` is
  the check that caught both the gap and the bad fix.** `git check-ignore -q <dir>/` would have said
  nothing useful, which is exactly what the comment block at `.gitignore:68-80` has been saying since
  the last four times.

### How to actually run this thing (nothing in the handoff used to say)

There is **no `.env` on this machine and `data/` is empty**, so any command touching the database or
the web fails on first contact. The install order lives in **`HANDOVER.md` §1–§2** — which §8 calls
"the CLIENT's runbook", but it is **yours too**: it is the only place recording the **Python 3.11-only**
constraint, the `[dev]`/`[ai]` extras and the CUDA `--extra-index-url`, the
`onnxruntime` → `onnxruntime-gpu` shadowing fix, and the two `.pt` weights that must sit in the launch
directory. To drive the panel in a browser yourself — which **no human has ever done** (§0):

```
.venv\Scripts\qorgan db upgrade
.venv\Scripts\qorgan user add <name> --role admin
.venv\Scripts\qorgan web            # then http://127.0.0.1:8000
```

Loopback defaults are fine; a non-loopback `WEB_HOST` refuses to start without `SECRET_KEY`.
**Telegram delivery runs inside `qorgan web`'s lifespan** (`web/app.py:97`), *not* in the supervisor —
so a `QUEUED` row with `sent_at` NULL is equally consistent with "the sender is broken" and "the web
process was simply never up". `docs/windows-autostart.md:9-11` says it plainly: if `qorgan web` is
down, no alerts are delivered and they queue as rows.

**There is no CI** (`HANDOFF.md:145`). "Push the branch" does not mean a runner will check it.
Pushing needs `-u`, because these two are the only branches in the repo with no upstream, and they
live in worktrees, not the main tree:

```
git -C "../q.ai-guard" push -u origin fix/no-thread-outlives-its-test
git -C "../q.ai-leak"  push -u origin fix/leaked-worker-threads
```

### What has actually been demonstrated — read this before quoting the test count

**1884 passing tests is not "it works".** Nothing in this system has met a camera. This table is
carried forward from 2026-07-24 and **none of it changed** — the web panel added pages, not evidence.

| claim | status | evidence |
|---|---|---|
| the detector raises candidates on the school's real footage | demonstrated on recorded clips | `eval scan`: 657 clips, 145 candidates, 51 alerts |
| the detector ranks walking above fighting | demonstrated on recorded clips | 33 human-described clips, §1 |
| faces on the hall cameras are too small to recognise | demonstrated on recorded clips | 0 recognitions in 14 970 faces |
| RTSP connects to a real NVR | **never — not once** | `CameraStream` has only ever run against fakes; no `.env` on this machine |
| a video clip is written for an event | **never — no production caller** | `write_clip` is referenced from tests and nowhere else in `src/` |
| a canteen session opens from a live camera | **never** | follows from RTSP |
| a Telegram message reaches a real chat | **never** | the one `notifications` row is `QUEUED`, `sent_at` NULL |
| memory/latency flat over 24 h | **never measured** | "bounded in a test" ≠ "flat RSS for a day on real cameras" |
| **the web panel is used by anyone** | **never** | every page is proven by `TestClient`, not by a browser a human drove |
| everything else | unit-tested only | 1884 tests |

## 0a. What changed on 2026-07-28

### `main` is green, and §0's red was the load flake it looked like

```
main @ 086eb96, alone:  1884 collected / 0 failures / 0 errors / 0 skipped / 388 s
```

The previous edition recorded a red on
`test_analysis_rate::test_the_loop_measures_the_rate_it_is_actually_being_fed` and refused
to call it either way. It did not reproduce. The run above was taken **while three
subagents were working**, which makes it a stronger claim than a green on an idle machine,
not a weaker one — the failing assertion is a throughput one.

### Eight branches, all pushed. Nothing exists in one copy any more

`ef0aad4` and `55c3ee9` — the two commits the previous edition flagged as existing nowhere
but this disk — are now on origin in four branches. Pushing was always permitted ("branch
and push yes, merge no"); the previous session simply had not done it.

| branch | what it is | own suite, run alone |
|---|---|---|
| `fix/leaked-worker-threads` | the `lifespan` fix | (ancestor of the guard) |
| `fix/no-thread-outlives-its-test` | the thread guard | proven, see below |
| `feat/clips-in-production` | `write_clip` finally has a production caller | 1903 / 0 |
| `feat/file-source-demo` | a camera may read a recording | 1925 / 0 |
| `feat/roster-2026-delivery` | the 2026 delivery is importable, with names | 1996 / 0 |
| `fix/rtsp-open-timeout` | a wedged camera no longer wedges shutdown | 1901 / 0 |
| `feat/classroom-lesson-metrics` | client §12.4, honest minimum | 2088 / 0 |
| `feat/telegram-skip-reason` | client §7, why no alert was sent | 1915 / 0 |
| `integration/demo-stack` | all eight together | see §0a below |

**`main` is untouched. No merge was performed into it.**

### The thread guard is proven, and it caught something written the same day

§10 of the previous edition said the guard's evidence "could not be found on disk, so treat
it as a claim to re-run". It was re-run. Merged into the integration branch, it turned the
wandering `no such table: users` / `database is locked` flake into **one deterministic red
naming the culprit test** — `test_file_source.py::test_a_camera_that_is_merely_off_the_air_
never_finishes_the_worker`, leaking `camera:hall_left`.

**Correction to a claim made during the session:** the disappearance of `database is locked`
is `ef0aad4`'s doing (the `lifespan` fix), *not* the guard's catch on that test. The leaked
thread was a `CameraStream` reader, which never calls `session_scope()`. Merging the guard
branch brings both commits; crediting one with the other's effect was sloppy.

### PRODUCTION DEFECT FOUND: a worker cannot shut down when its cameras do not answer

Measured against a real unreachable address, not reasoned about:

```
one camera, before:   stop() returned after 5.00 s;  thread actually died at 29.11 s
one camera, after:    stop() returned True at 3.03 s; thread died at 3.03 s
four cameras, before: 20.02 s, sequential, against a 10 s grace  -> OVER
four cameras, after:   5.00 s, parallel                          -> FITS
```

The reader thread sits **inside the `cv2.VideoCapture` constructor** and never reaches its
stop check. `CameraStream.stop()` joined for 5 s, returned as though it had succeeded, and
nulled the handle — so a failed shutdown left no trace anywhere. `bullying_stairs_yard` has
four cameras and `TERMINATE_GRACE_SECONDS` is 10.0, so the supervisor gives up, logs
`worker ignored terminate` and kills the process. **The log blames the worker; the cause is
a camera that did not answer.**

This is the state of day one on site: RTSP has never been reached, `RTSP_PASSWORD` is blank,
and a camera VLAN that is down during a service restart is ordinary, not exotic.

Why nothing caught it: grep-verified, **every pre-existing test injects a fake opener.**
Nothing in the suite had ever called the real `open_rtsp` against a real address.

Two things worth carrying forward:

- **OpenCV serialises the FFmpeg open.** Measured: four concurrent opens with a 4 s timeout
  return at 4.05 / 8.09 / 12.12 / 16.17 s. So on a wholly unreachable group **no wait short
  enough to fit the grace can collect every reader.** The design is therefore "bounded and
  honest" — report which readers did not stop — rather than "everything really stopped".
- `open_timeout = 4.0` and `LOOP_JOIN_SECONDS = 2.0` are **judgements, not measurements**.
  No camera on this network is reachable, so real LAN latency is unknown. Settle on site.

### DATA-LOSS DEFECT FOUND AND FIXED: a migration deleted every event and reported success

Migration `0006` widens `cameras.camera_type`. SQLite cannot ALTER a column, so alembic
batch mode rebuilds the table — create copy, move rows, **DROP the original**, rename. And
`events`, `canteen_sessions` and `recognition_attempts` all reference `cameras` with
`ondelete="CASCADE"`.

Verified by the controller, with the guard sabotaged and the revert confirmed byte-identical:

```
with the guard:     events before = 1   migration = success   events after = 1
guard neutered:     events before = 1   migration = success   events after = 0
```

**The obvious guard is a no-op.** `PRAGMA foreign_keys` is documented to do nothing inside a
transaction, so a migration that protects itself protects nothing. It is suspended in
`migrations/env.py` on a fresh connection *before* `begin_transaction()`, and it **refuses to
migrate** if the pragma did not take effect.

This is not specific to `0006`. Any future batch migration touching a table referenced with
CASCADE has the same shape, and now inherits the guard.

### The roster question is answered by measurement, not by asking the school

All 141 photographs of the new delivery embedded against all 142 of the existing roster
(`buffalo_l`, cosine). The split point is not invented: this project already measured its
worst impostor pair at **0.472** and ships `min_score = 0.50`.

```
126 matched at >= 0.90      1 ambiguous (0.685)      14 with no counterpart
class agreed on 127 of 127  |  nothing at all in the 0.472-0.60 band
```

**Conclusion: the same school, the same school year, the same roster deduplicated and
renumbered, plus 14 children who were not in the old set.** Not a new year — the classes
match one for one.

Three things fell out that nobody asked for:

- **The whole `47x` block** — `student_470` … `477` — is exactly the set with no counterpart.
  The measurement knew nothing of the duplicate hypothesis in `questions-for-school.md` §1;
  the block fell out on its own because each of those people's *first* registration is in
  the new delivery and claimed the match. **Independent confirmation of the six duplicates.**
- **`7-А 438` and `439` are two different children.** Both claimed distinct matches, and the
  new delivery contains no duplicates. The twins question is closed — do not re-ask it.
- `staff_465`–`468` have **no detectable face**, reproducing `questions-for-school.md` §3
  blind. That is also a control on the measurement itself.

**Consequence for the staff carry-over:** taking the `staff`/`учитель` folders wholesale
would re-import a known duplicate. `staff_464` and `student_477` are one person (0.999);
`staff_334` and `student_470` are one person (0.984). Enrollable staff is **three**, not
eight, and four need new photographs.

**Still not decided, and not decidable by measurement:** which of `staff_464` / `student_477`
is the record to keep. That is the school's, or the owner's, to say.

### A defect class this session added evidence for: a clean merge that breaks a feature

Two branches, **different files, not one overlapping line, no conflict marker**. One changed
`CaptureOpener` to take a second parameter (timeouts must reach `cv2.VideoCapture` as
constructor arguments or they do nothing); the other implemented an opener taking one. Git
merged them happily and the entire file-source feature raised `TypeError` on first connect.

Also silent: **two migrations both numbered `0006`**, on different filenames. `alembic heads`
answered `0006 (head)` **twice with only a UserWarning** — one migration would simply never
have run.

**"Merged cleanly" and "works" are two claims. Check both.** This belongs beside §7's
existing "'X clean' and 'Y passes' are two different claims".

The counter-measure that already existed and worked: `roles.py`'s comment about five branches
each rewriting two lines, where taking any single version would have silently revoked the
others' pages *with a green suite*. Resolving a conflict by choosing a side is how a feature
disappears without a failing test.

---

---

## 0b. What changed on 2026-07-29

### `main` = `6d19dd8`, 260 commits — re-run here, not quoted from a report

```
main @ 6d19dd8, alone:  2347 collected / 0 failures / 0 errors / 0 skipped / 484.8 s
ruff check .         :  All checks passed!   (exit 0, its own command)
```

That is my own run into my own XML, not the controller's — and the controller's independent run
earlier the same day agrees exactly (2347 / 0 / 0 / 0, 459.8 s). Two runs, two operators, same
number. `260` commits and the hash are `git rev-list --count main` / `git rev-parse main`.

Two branches merged since `a88cbf6`, and **nothing else**:

| merge | branch | what it is |
|---|---|---|
| `43ff6f1` | `fix/anonymous-request-must-not-eat-the-csrf-token` | logging in from a browser was impossible |
| `6d19dd8` | `fix/eval-scan-survives-the-corpus` | the scan that lost 169 clips of GPU and wrote a zero |

**None of the three pilot modules is in `main`.** They are three unmerged branches — read
"Three modules for the pilot" below before assuming anything they found is protected by a test
that runs on trunk.

### PRODUCTION DEFECT, FIXED AND MERGED: logging in from a browser was IMPOSSIBLE, and all 1884 tests passed

`AuthMiddleware` cleared the **whole** session whenever it saw a request from somebody not
logged in. The session carries two unrelated things: who is logged in, and this session's CSRF
token. So every login went:

1. `GET /login` mints the token and renders the form carrying it;
2. the browser asks for **`/favicon.ico`**, which is not a public path, so the branch runs and
   wipes the token it had no business touching;
3. `POST /login` presents a token the session no longer knows → **403**.

Every browser asks for a favicon. This was not an edge case — it was every login, every time,
and the first human ever to open this dashboard could not get past the form. Only the user key
is dropped now; session fixation is still defended where it belongs, in `login()`, which clears
everything on *successful* authentication.

**Why 1884 tests did not catch it: the defect lives in the GAP.** Every login test went
`GET /login` → `POST /login` with nothing in between, because `TestClient` fetches exactly the
paths a test names. The new tests walk that gap and are **parametrised over three ordinary
browser behaviours** rather than over the favicon alone — naming one of them would have let the
next one back in. (`97b6221`; `src/qorgan/web/security.py`, `tests/test_web_login_from_a_browser.py`.)

§0's table said "the web panel is used by anyone — **never**". This defect is what that row was
worth. The row still stands: a `TestClient` walking a browser's request sequence is closer, but
it is still not a browser a human drove.

### DATA-LOSS DEFECT, FIXED AND MERGED: `eval scan` lost 169 clips of GPU and wrote a zero

`qorgan eval scan` over the 657 clips died on **clip 170** and produced **no output file at
all** — `candidates.csv` and the coverage manifest were both written once, after the last clip.
A closed laptop lid or an impatient Ctrl+C would have cost exactly the same.

**What killed it was already on disk, two lines above the traceback in the crash log**, and it
is not what the exception said. `SystemError: <method 'read' of 'cv2.VideoCapture' objects>
returned a result with an exception set` is only CPython reporting that a C function returned
with a Python error already set; the real one was
`OpenCV alloc.cpp:73 (-4:Insufficient memory) Failed to allocate 11059200 bytes` /
`numpy _ArrayMemoryError: Unable to allocate 10.5 MiB for an array with shape (1440, 2560, 3)`.
**A host memory failure for one decoded frame, not a leak in the scan** — 234 clips of the
identical path, instrumented per clip, gave RSS oscillating 668–1580 MB with a *negative* net
slope and flat VMS/CUDA/handles, and clip 170 itself scanned clean at index 170. The machine
moved, not the code: commit charge 29.5 GB against a 33.1 GB limit. **Do not go hunting a leak
here.** Recorded as unreproduced rather than fitted with a story.

What changed, and why each part matters next time:

- **All three artifacts are rewritten after EVERY clip**, each through a temp file renamed over
  the target — a reader sees a complete old file or a complete new one, never a truncated one
  that would read back cleanly and short.
- **Resume uses the coverage manifest**, not a second checkpoint file that could disagree with
  the result. Same mechanism as `eval label`, deliberately not a new one.
- **A clip that cannot be read no longer takes the corpus down and cannot be lost either**: named
  with its reason in `candidates.unreadable.csv`, kept OUT of the manifest, counted in the
  summary, non-zero exit, and `eval sample` refuses to draw until it is scanned or removed.
- `_scan_into` catches `Exception`, not a hand-written tuple of decoder errors, **on purpose** —
  the corpus was lost to a bare `SystemError`. Narrowing it to `OSError` turns three of the four
  parametrised failure shapes red. Do not "tidy" that catch (`b3398bd`).
- **Found by running it, not by reading it:** the first real resumable run died at clip 104 with
  `PermissionError WinError 5` out of `os.replace`, because the session was *reading the manifest
  to check progress* and Python's `open()` asks for no `FILE_SHARE_DELETE`. If you tail a scan's
  output on Windows you are part of the experiment.

### The corpus is re-measured end to end, and 145/51 is a stale claim rather than a regression

```
657/657 clips covered · 148 candidates · 53 alerts · 72 held at exactly 0.72 · 0 unreadable
```

Two sittings with a real crash between them: run 1 died at clip 104, run 2 opened with
"657 clip(s), 103 already scanned" and finished the other 554. Exit 0, no clip unread, no
`.partial` left behind. `--device cuda:0`.

**The 145 candidates / 51 alerts that §0's evidence table and `HANDOFF.md` have been quoting
cannot be checked at all**: no `eval/candidates.csv` exists in any checkout, so that pair names a
run whose output is gone. The cap figure (72 at exactly 0.72) is identical, and the resumable
rewrite is not what moved the other two — over the 169 clips the crashed run did manage, its
per-clip camera and candidate counts and the new run's agree exactly, clip for clip.
**Treat 145/51 as stale, not as a regression.** `HANDOFF.md:151-157` now says so; §0's table in
this document still carries the old pair and is not being rewritten, because dated entries here
are append-only — this paragraph is the correction.

### Three modules for the pilot — the owner's decision, and where each actually stands

**Recorded as a decision, not as a plan.** The client asked for weapons (§12.1), the
psychologist's cabinet (§13) and several schools (§14) from the beginning and wants them present
at the pilot. The owner heard the argument for closing the basics first (detector accuracy,
canteen camera placement, 24 h run) and decided otherwise. **The basics run in parallel; they are
not a precondition here.** Do not re-litigate this by treating a module as blocked on them.

Verified against `git`, not against the ledger:

| module | branch | tip | state |
|---|---|---|---|
| psychologist's cabinet | `feat/psychologist-cabinet` | `9a59e54` | **CLOSED** — spec ✅, quality review clean after one fix round |
| several schools + tenancy | `feat/multi-school` | `7f53e6e` | in flight — fix round 2 of 5 dispatched (the five-route guard) |
| weapons | `feat/weapons-detection` | `fc171c0` | in flight — implementation + tests exist, not yet reviewed |

Sequential on purpose: three parallel agents on 2026-07-28 burned the shared session budget three
times as fast and all three stopped at ~70%. **Three branches at 70% is zero modules.**

**A merge-time collision git cannot show you:** `feat/psychologist-cabinet` and `feat/multi-school`
both add a migration numbered **0008** under different filenames. No textual conflict; `alembic
upgrade` breaks. Resolve when merging, whichever goes second. **Do not renumber on the branches** —
that makes one module depend on the other. This is the same shape as the two `0006` migrations §0a
records, second occurrence in two days.

### 17 commits exist nowhere but this disk — all three modules

```
git rev-list --all --not --remotes --count  ->  20
  feat/multi-school          11
  feat/psychologist-cabinet   3
  feat/weapons-detection      3
  stash@{0} and its two index/untracked commits   3
```

`origin/feat/psychologist-cabinet` is still at `123361d`, the **unverified WIP**; `origin/feat/
multi-school` is at `6683c79`. Anybody who clones this repo today gets three part-built modules
and none of the work described in this section. §0 made exactly this complaint about five
commits on 2026-07-27 and §0a closed it by pushing; it has re-opened, larger. **Push before you
do anything else.**

`stash@{0}` (`af5b80b`, 2026-07-14, `WIP on feat/detector-calibration`, `tests/test_eval_label.py`
+74) is **still undecided** — flagged on 2026-07-27, still there. `git status` will never mention it.

### What the three modules found that had nothing to do with the three modules

Each of these was exposed because a module walked a path nobody had walked. None was in a brief.

- **A class report for ANOTHER school's classroom, by typing a number in the address bar.**
  `/lessons/{lesson_id}` took the id straight from the URL. Found by refusing to accept a green
  sabotage: the tautology passed the tenancy guard AND the isolation test, because the isolation
  test walked only the **index** and never the page the index links to.
- **LIVE VIDEO of another school's cameras, and the tenancy guard cannot see it by construction.**
  `load_cameras()` reads YAML into `app.state.cameras` — **there is no database query at all**, so
  a scan over queries has nothing to scan. Any `VIEW_CAMERAS` account of any school gets another
  school's camera list, siting, health and a live JPEG frame; `VIEW_SETTINGS` gets the whole
  installation's RTSP hosts. **Still open by design** — see the owner decision below.
- **Merging duplicate pupils returned 500 even INSIDE your own school.** `POST /pupils/duplicates/
  merge` omitted `school_id`, so `UndecidedSchool` (a `RuntimeError`) escaped a
  `(LookupError, ValueError)` handler. Why it survived: removing `school_id=` reddens 3 tests and
  **183 others notice nothing**.
- **The superadmin page had zero behavioural tests.** Proved by sabotage, not by reading: swapping
  `MANAGE_SCHOOLS` for `VIEW_CANTEEN` let a **canteen worker create and rename schools**, and
  **565 tests still passed**.
- **`ensure_cameras` matched `cameras.name` across all schools** — two schools may both call a
  camera `hall_left`, and the second installation would have overwritten the first school's row.
- **The faces importer would have attached one school's photographs to another school's pupil 7.**
  An `external_id` is unique *within a school*; it never was globally.
- **`SUPERADMIN` held `VIEW_DIAGNOSTICS`, and `/logs` carries an undelivered-alerts panel** —
  camera name plus the minute an incident involving a child happened. R5 walks the route table and
  cannot see a wrong *capability* on a right route.

### Migrations: measured three times now, and the blast radius is EIGHT tables

> **Superseded by §0c's "Migrations: a fourth measurement".** §0c names all four table pairs
> individually and cites the guard by line. The eight-table list and the rule below are identical in
> both and were re-verified at consolidation; **the one fact only this copy carries** is that the
> test also asserts a *before*-count, so a silently skipped insert cannot read as a working guard.

This is the most expensive knowledge of the day. §0a recorded it for `cameras` → `events`. It has
since been re-measured on a different table pair (`notifications` → `events`) and then on
migration `0008` itself, by three different agents, twice adversarially:

> **A migration that rebuilds a table in SQLite cascade-deletes everything referencing it, at
> exit code 0, with the revision stamped. One `create_foreign_key` is enough to force the rebuild.**

Without the guard, the eight tables that lose **every row**:

```
events · notifications · canteen_sessions · recognition_attempts
person_photos · face_embeddings · lessons · lesson_tracks
```

The four tables `0008` rebuilds — `cameras`, `persons`, `users`, `meal_windows` — survive, and the
test asserts that too, so a broken migration cannot read as a working guard. There is also a
before-count assertion, so a silently skipped insert cannot read as one either.

The guard is `migrations/env.py::_suspend_foreign_keys`. **Two things about it are load-bearing and
look like details:** it runs on a fresh connection **before `begin_transaction()`** (inside a
transaction `PRAGMA foreign_keys` is a **documented no-op**, so a migration that protects itself
protects nothing — measured, the row was gone either way), and it is issued on the raw DBAPI
connection to stay out of SQLAlchemy's transaction bookkeeping. It **refuses to migrate** if the
pragma did not take.

The widened test lives on `feat/multi-school`
(`tests/test_migration_keeps_the_events.py`, 8 cascade + 4 rebuilt-root controls). **On `main`
today the equivalent test measures `events` only** — a future migration sparing `events` but not
`notifications` passes on trunk right now.

### The lesson that has now repeated FOUR TIMES in three days

> **Superseded by §0c's "SIX instances in three days".** The four instances below are the first four
> of §0c's six, and the counter-measures are the same three plus one. Read §0c's version; this one is
> kept because it is the dated record of what was known on the 29th, not because it adds a fact.

**A test that does not travel the path a human travels proves something other than what it
appears to prove.** Not a slogan — four measured instances, each found only because somebody
sabotaged a defence and watched the wrong thing stay green:

1. The `/media` directory-traversal test passed **with the defence sabotaged**, because httpx
   collapses `..` before the request is sent. The client was doing the defending.
2. Every login test passed while logging in from a browser was **impossible** — the tests never
   made the request the browser makes between the two they did make.
3. "An event can still be ruled on after it is referred to the psychologist" was green while the
   page had **stopped drawing that button** — the test POSTed straight to `/events/{id}/review`,
   a path the page no longer offered.
4. The tenancy isolation test walked the **list** of lessons and never opened the page the list
   links to, so a tautological filter passed both the guard and the isolation test while serving
   another school's classroom report.

The counter-measures that actually worked, each of them cheap: **assert what the PAGE draws, not
what the endpoint accepts**; **parametrise over the class of behaviours, never over the one
instance you found**; and **covering a list page is not covering the page it links to**.

One more, from the same family: **prose is not checked by a test suite unless a test reads it.**
The multi-school branch's own finding was "a number in prose nobody re-measured" (26 exemptions
reported as 11) — and the prose describing that *fix* then said a sabotage drops 12 tests when it
drops 18. What stuck was making prose testable: one test reads `tests/test_tenancy_guard.py` as
**text** and asserts the counts appear verbatim; another checks `docs/questions-for-school.md`
against the real permission table, so an undisclosed capability fails by name.

### Owner-facing: conditions and decisions nobody has made yet

These are not tasks. Each is a place where the code has taken a position by accident, or where the
owner has to choose before the pilot.

- **A second school must NOT be given alert-raising cameras until per-school notification routing
  exists.** `settings.telegram_bot_token` is one token per *installation*; the notifier has no idea
  which school a queued alert belongs to. Recorded as a condition on the multi-school module, not
  as a bug in it.
- **A referred event is not protected from expiry.** `janitor.PROTECTED = (CONFIRMED, REVIEWED)`
  and `DEFAULT_MEDIA_DAYS = 90`. An event a psychologist has been handed but nobody has ruled on
  is still `new`, so **its clip is deleted at 90 days**. Not a regression — the old status token
  was unprotected too — but "a referral does not protect the evidence" is a property **nobody
  chose**. It fell out.
- **The cross-school live-video exposure is a NOTICE, not a BOUNDARY.** The implementer put a
  warning block on `schools.html` and an expiry check that reddens the day any `CameraConfig`
  union member gains a school field — but nothing stops an operator reading the warning and
  creating the second school anyway, and the exposure is then real. The re-reviewer's argument,
  which is the one that decides it: *"This system is willing to CRASH-LOOP the detection worker
  rather than guess which school a row belongs to. It is simultaneously willing to SERVE ANOTHER
  SCHOOL'S LIVE VIDEO OF CHILDREN rather than refuse. Those two positions cannot both be right."*
  Its proposal — neither a hard gate nor a schema change — is a route-level refusal when
  `count(schools) > 1` on `/`, `/api/cameras`, `/preview/{camera}.jpg`, `/cameras`, `/settings`,
  and it costs the operator nothing they can do today anyway, **because detection is already dead
  on two schools**. Fix round 2 was dispatched for exactly this and had not landed when this was
  written.
- **`docs/questions-for-school.md` §10 was written by an agent and the owner has not read it.**
  It goes to a real school. It exists **only on `feat/psychologist-cabinet`** — it is not on
  `main`, so nobody reviewing trunk will meet it. §10.2 discloses four capability widenings the
  psychologist role grants beyond what §13 lists; a re-reviewer's opinion, deferred to the owner,
  is that two phrases make the school's non-decision *our* default, and that four yes/no lines
  would beat one open question. **Read it before it is sent.**
- **The client's weapons answers, recorded so they cannot be lost:** both **knife and firearm**;
  the **kitchen is out of scope**; they will hang a camera at the **entrance** to see well — **but
  the other cameras stay in scope**, so the module must be honest **per camera** rather than assume
  the good one.
- **Undecided:** whether the psychologist's cabinet shows an empty canteen-attendance block
  labelled "accumulating" before the canteen camera is moved, or hides it until there is data.
  Owner said "let's think, we'll decide". Default for now: show it, labelled. Revisit before the
  pilot.

### What in this section I checked with a command, and what I did not

**Checked:** `main`'s hash, commit count, suite and `ruff` (my own run, above); every branch tip and
which commits exist nowhere but this disk; that `97b6221` and `2f66ffb` are ancestors of `main`;
`_suspend_foreign_keys`, its placement before `begin_transaction()` and its refusal; the eight-table
`WATCHED` list and the four `REBUILT` controls; `janitor.PROTECTED` and `DEFAULT_MEDIA_DAYS = 90`;
that `ensure_cameras` now takes `school_id`; that `docs/questions-for-school.md` §10 exists on the
psychologist branch and **not** on `main`; the `24 / 2 / 26` exemption comment; that the
`schools.html` warning sends an operator to a `tests/` path.

**Not re-run by me, and named as somebody else's measurement:** the corpus scan itself
(657/148/53/72 comes from `6665315`'s own run, recorded in `HANDOFF.md:151`); every sabotage result
quoted above (565 tests passing with a canteen worker creating schools; +2 failures per removed
filter; 183 tests noticing nothing); the per-branch suite totals; and the "41 `load_cameras` call
sites across 10 modules" figure — a plain `grep -rn 'load_cameras(' src/` on `feat/multi-school`
counts **14 sites in 11 files**, so the 41 is counting something wider than call sites. **Ask what
it measured before quoting it.**

## 0c. What changed on 2026-07-30

### The three modules are still three branches, and none of them is finished

Checked with `git`, not read off the ledger: `git merge-base --is-ancestor <branch> main` answers
**NO for all three**, and `git rev-parse origin/<branch>` equals the local tip for all three, so the
unverified WIP `origin` was serving yesterday is gone.

| module | branch | tip | ahead | delta vs `main` | where it stopped |
|---|---|---|---|---|---|
| psychologist's cabinet | `feat/psychologist-cabinet` | `9a59e54` | 4 | 36 files, +3386/−62 | **closed** — spec ✅, quality clean after one fix round. Five deferred minors are parked for a whole-branch review, not discarded |
| several schools + tenancy | `feat/multi-school` | `9066ac2` | 20 | 50 files, +4829/−188 | fix round **4 of 5** landed. Re-review 4 was dispatched and **its verdict exists nowhere** — do not read `9066ac2` as reviewed |
| weapons | `feat/weapons-detection` | `86d1207` | 5 | 39 files, +6381/−274 | reviewed (spec ✅, quality **approved**, 3 Important + 8 Minor); fix round **1 of 5** dispatched and **nothing has landed** |

> **Two of those three tips moved later the same day.** `feat/multi-school` is at `ad90e2e` and
> `feat/weapons-detection` at `3aee202`. The row above is left as it was measured, because dated
> entries here are append-only — the correction is **"What moved after this section was written"**
> at the end of §0c. Do not check out `9066ac2` or `86d1207` expecting the current branch.

Suite totals per branch are **their own runs, not mine**: 2411 (psy), 2576 (multi-school), 2632
(weapons). `main` is still `6d19dd8` / 2347, which §0b measured twice with two operators. **Nothing
was merged into `main` today.**

The two `0008` migrations §0b warned about are confirmed by reading both files:
`0008_a_referral_is_an_act_by_a_named_person.py` and `0008_a_school_is_not_the_installation.py`,
both `revision = '0008'`, both `down_revision = '0007'`. `git` will merge them without a word.

### §0's stash section is now wrong, and the command that counts unpushed commits lies

- **The 2026-07-14 stash is gone.** `git stash list` is empty. Before dropping it, it was archived
  as the annotated tag **`archive/stash-2026-07-14`**, and the tag is on origin
  (`git ls-remote --tags origin` returns it, peeling to `af5b80b`). §0 describes it as a live
  `stash@{0}` awaiting a decision; that is no longer true, and §0 is not being rewritten because
  dated entries here are append-only. **This paragraph is the correction.**
- **It was never one file and +74.** `git diff --stat 9c9679c af5b80b` shows the tracked half only:
  `tests/test_eval_label.py`, +74. The stash commit has **three** parents, and the third
  (`a49b085`, the untracked half) carries `tests/test_eval_pending.py`, **+201**. Two files, **+275**.
  §0 quoted the number `git stash show --stat` prints, and that command cannot see the untracked
  parent. The tag message records why the content was not restored: it asserts a project decision
  that `main` has since **reversed** (a detector firing inside an un-judged interval counted as a
  false positive; `main` says the opposite in words), so restoring it restores a reversed decision
  as a red test.
- **§0's "five commits exist nowhere but this disk" is zero today — but you cannot check it the way
  §0 and §0b did.** `git rev-list --all --not --remotes` returns **3**, and names `af5b80b`,
  `8ada5f4`, `a49b085` — all three reachable from a tag that **is on origin**. `--remotes` covers
  `refs/remotes/*` only; it never looks at remote **tags**, so it reports commits as local-only when
  origin holds them. The honest count is `git rev-list --all --not --remotes --tags --count` → **0**.
  Use the tag-aware form, or you will push nothing and think you saved something.

### Defects the three modules exposed that had nothing to do with the three modules

Each was found because a module walked a path nobody had walked. None was in a brief. §0b lists
seven; these are the ones added or sharpened today.

- **LIVE VIDEO of another school, through a YAML file the tenancy guard cannot see by
  construction.** Verified by reading `config/loader.py:111`: `load_cameras()` resolves a directory,
  reads `base.yaml`, globs `cameras/*.yaml` — **there is no database query in it at all.** The guard
  is a static scan over queries, so there is nothing for it to scan; this is not a missed filter,
  it is a surface the whole mechanism is blind to. §0b left this open. It is now **mitigated on
  `9066ac2`**: `web/config_scope.py` refuses with **409** on `/`, `/api/cameras`,
  `/preview/{camera}.jpg`, `/cameras`, `/settings` when more than one school exists, called
  explicitly at each site rather than hidden in a decorator. The re-reviewer confirmed it by
  **publishing a real frame first** and only then hitting the route — asserting 409 alone passes
  with an empty subscriber, which is the same test-does-not-travel-the-path defect one level up.
  Read the owner list below before calling this closed: it is a page, not a boundary.
- **A false weapon alert after every RTSP reconnect.** `expire()` ran **after** association, and
  `WeaponTrackStore._match` is spatial with **no clock of its own**, so `track_idle_seconds` only
  held while frames kept arriving. Measured with the shipped gates: two observations, no `process()`
  call for **9.8 s**, then one sighting in the same place — the first frame back completed a
  three-observation track and **alerted**, two of the three observations from *before* the outage.
  `self.tracks.expire(now)` is now the first statement of `process()`
  (`src/qorgan/weapons/pipeline.py:134`, verified by reading it). **One statement moved, no
  threshold touched.** RTSP reconnects are routine in this school. The reviewer reproduced the
  defect by reverting the line and got the same alert, with a frames-arriving control at 0.
- **The same root cause one field over, still open:** `realert_seconds` does not survive a
  reconnect either — `alerted_at` lives on the track, and the track expired. Ledger measurement:
  a 3.0 s gap produced a second alert 0.6 s later against a **60 s** quarantine. Any state kept on
  an object whose lifetime is "seen recently" is a clock nobody wound.
- **A second, unguarded door to a weapons row.** `POST /events/{id}/review` is gated on
  `REVIEW_BULLYING`, **not** `CONFIRM_WEAPON_ALERT`, and — verified by reading the whole handler at
  `web/routes/events.py:121` — it looks up `session.get(Event, event_id)` and **never checks
  `event_type`**. Ledger measurement: POST on a weapons row returns 303 with `status=confirmed` and
  `reviewed_by_id` recorded; `verdict=reviewed` also passes, which leaves the alert in "waiting for
  a human" with nobody having answered — exactly the state `WEAPON_VERDICTS` excludes on purpose.
  `store.py:11` claims in prose that nothing in `src/` moves a weapons row off `NEW` except
  `rule_on_weapon_alert`. That claim is **false**, and no test anywhere hits that route with a
  weapons row.
- **`/weapons` cannot tell "the file is there" from "the module is watching this camera."** Three
  states draw as working: weights present and over the size gate but not loadable (a normal row
  with size and fingerprint, while the worker crash-loops), `enabled: false` (a clean row plus a
  full reach row, with no word that nobody is watching), and a plain dead worker (invisible). The
  empty-state text pushes the reader toward reading a green row as "watching".
- **`confusable_classes` is validated against nothing.** In the same file, `target_classes` is
  checked loudly against `KNOWN_TARGETS` (`config/weapons.py:323-327`) while `confusable_classes` is
  only checked for overlap with `target_classes` (`:330-333`) — never against the model's actual
  class names. Ledger measurement: weights emitting `(knife, person, cell phone)` against spiked
  confusables give an **empty intersection** and `refuse_unusable_weights` **accepts silently**.
  Screen 3 is inert and says nothing. The schema says `phone`; COCO says `cell phone`. Same shape as
  a permission that guards nothing.
- **`/media/{path:path}` is capability-gated and NOT school-gated** (`web/routes/media.py:53`) —
  recorded clips and snapshots are installation-wide. It needs an exact camera name plus a
  timestamp, and the five-route refusal removes the camera list as an enumerator, **but `/logs` can
  still name cameras.** Owner-facing, outside every diff so far.
- **A refusal page that promised what the system cannot do, in three places, twice inside the fix
  for itself.** The 409 body's only instruction sent the reader to `/schools`, which their role
  cannot open (measured: all three camera-capable roles get 409 on `/` and **403** on `/schools`;
  `MANAGE_SCHOOLS` is `SUPERADMIN` only). The text also promised a school could be *removed* —
  there is no delete anywhere in `src/`: `qorgan.schools` has list, create, rename. And
  at `02acc1d` the new page said «**Журналы**, события, уведомления, столовая, ученики и уроки …
  показывают **только вашу школу**» (`cameras_are_installation_wide.html:53`, read at that commit)
  while `diagnostics/logfiles.py:150` is `recent(category, page, page_size)` — **no school dimension
  at all**. That line was written by the diff that fixed the first two, and the parallel list on
  `schools.html` deliberately omits logs, so the two pages diverged and the rewritten one
  over-promised. All three are closed at `9066ac2`, and I checked rather than assumed: `Журналы` is
  gone from that item and `:56-59` now states the exception outright («он общий для всей установки …
  по школе разделена только панель недоставленных тревог»); the removal promise is recorded
  **positively** («удалить её нельзя … создаются и переименовываются, но не удаляются») so it cannot
  come back by inheritance. **Three one-line text defects in three consecutive rounds, each inside
  the fix for the previous one** — the review was working; the prose was where it kept failing.
- **`KNOWN_TARGETS` is a closed five-slug tuple in `src/`** (`config/weapons.py:55`:
  `knife, axe, bat, metal_object, firearm`), so the weapons module **cannot be switched on by a
  YAML edit alone** and nothing in the config says so.

### Migrations: a fourth measurement, same answer, and now the cheapest thing to break

§0b recorded three measurements. There are now **four**, on different table pairs and by different
agents, one of them adversarial twice over: `cameras → events`, `notifications → events`, migration
`0008` itself on the psychologist branch, and `0008` on the multi-school branch.

> **A migration that rebuilds a table in SQLite cascade-deletes everything referencing it, at exit
> code 0, with the revision stamped. One `create_foreign_key` is enough to force the rebuild.**

Without the guard, **eight** tables lose every row: `events`, `notifications`, `canteen_sessions`,
`recognition_attempts`, `person_photos`, `face_embeddings`, `lessons`, `lesson_tracks`. Verified by
reading `WATCHED` in `tests/test_migration_keeps_the_events.py` on `feat/multi-school` — eight
entries, plus four `REBUILT` root controls (`cameras`, `persons`, `users`, `meal_windows`) so a
broken migration cannot read as a working guard.

The guard is `migrations/env.py::_suspend_foreign_keys`, and **its placement is the whole thing**:
it runs on a fresh raw DBAPI connection **before `begin_transaction()`**, because inside a
transaction `PRAGMA foreign_keys` is a **documented no-op** — a migration that protects itself
protects nothing. It refuses to migrate if the pragma did not take (`env.py:72-77`, read directly).
**On `main` today the equivalent test still measures `events` alone.**

### The methodology below is worth more than any defect above

#### A test that does not travel the path a human travels — SIX instances in three days

§0b counted four. Six, all measured, each found only because somebody sabotaged a defence and
watched the wrong thing stay green:

1. The `/media` traversal test passed **with the defence sabotaged** — httpx collapsed `..` before
   the request went out. The client was doing the defending.
2. Every login test passed while logging in **from a browser was impossible**.
3. "An event can still be ruled on after referral" was green while the page had **stopped drawing
   the button** — the test POSTed straight to `/events/{id}/review`.
4. The tenancy isolation test walked the **list** of lessons and never opened the page the list
   links to, so a tautological filter served another school's classroom report.
5. The blast-radius assertion for `/schools` asserted `status in (200, 403)` with an `ADMIN`
   fixture, so **the loudest claim in the docstring passed structurally on a 403**. Measured after
   the fix: role swapped to `ADMIN` → 1 failed of 20, exactly the case the old form accepted.
6. `test_weapon_events_do_not_appear_among_the_bullying_ones` **never fetched `/events`**. With the
   `BULLYING` filter deleted, **74 tests across four files stayed green.**

The counter-measures that keep working, all cheap: **assert what the PAGE draws, not what the
endpoint accepts**; **covering a list page is not covering the page it links to**; and when a claim
is about a *body* — a JPEG, a note — **assert the body**, because a status code passes with an empty
publisher.

#### A check can contain, inside itself, the class of defect it was written to close

The promise scanner — written to stop any page claiming what the system cannot do — was
**line-based**. Templates wrap at 100 characters, so a claim spanning two lines was two half-claims
and neither carried both subject and verb; the sabotage that should have reddened it **passed**.
Its author's words:

> "the check written to close a class of defect had that class inside it."

Two more things came out of the same fix and both generalise. Leaving Jinja in the scanned text put
`школ` from `{% for school in schools %}` into what the scanner read as pure markup, which made the
entire schools table read as a promise — "**which is how a scanner gets its word list trimmed until
it sees nothing**". And the fix was deliberately a rule, not a list: a school may appear beside a
removal verb **only in the negative**, because "a phrase list would catch rewordings I can imagine;
the negation rule catches the ones I cannot."

**The headline of that round was not the five items it was sent to fix: THREE of its own checks were
silently inert, and it found each by sabotaging its own work.** The promise scanner above was the
first. The second was a config-directory trigger that was a plain **no-op** — its test took no
`session` fixture, so there was no database and no real `config_dir`, and the helper **swallowed the
error and returned `[]`**; a real `config/cameras/default/` left it green. It raises now instead of
guessing. The third is the next subsection.

#### A number in prose that nobody re-measured — three in one day

- "the I-3 sabotage drops **12** tests" — it drops **18**. Direction safe, number wrong, on a branch
  whose *own* finding that round was "a count in prose nobody re-measured" (26 exemptions reported
  as 11).
- "**41** `load_cameras` call sites across **10** modules". Three parties have now counted and got
  three answers: 41/10 (the original), 11 calls in 10 files (the controller), and 14 in 11 files
  (§0b). My own `git grep -n 'load_cameras(' feat/multi-school -- src/` gives **14 invocations in 11
  files**, plus **31** in `tests/`. So the figure is `src`+`tests` while its scope is `src` only —
  **the number and its reach do not agree inside one sentence**, which is the tell. It does not move
  the decision it was quoted for: the load-bearing argument is that **no member of the three-model
  `CameraConfig` union has a school-shaped field**, so scoping it is a schema change, and that
  stands on its own.
- "the stash holds one file, +74" — two files, **+275**. See above.

The counter-measure that stuck is not "be careful": **make the prose testable.** One test reads
`tests/test_tenancy_guard.py` as **text** and asserts the counts appear verbatim; another checks
`docs/questions-for-school.md` against the real permission table, so an undisclosed capability fails
**by name**.

#### Sabotage has to go in BOTH directions, or you are grading your own reasoning

An agent wrote a docstring claiming the status-only test was the weaker version of the live-frame
test. Measuring it both ways said otherwise:

```
gate moved BELOW the frame lookup  ->  status-only FAILED (via 503),  live-frame PASSED
409 returned WITH the JPEG body    ->  status-only PASSED,            live-frame FAILED
```

They catch **different** regressions, and only the second catches the one that actually **serves the
video**. In its own words:

> "I wrote the original from reasoning — and it would have survived if I hadn't sabotaged in both
> directions."

One-directional sabotage confirms a test is not dead. It says nothing about **which** regression the
test is for. The same round produced the smaller version of this lesson — "I fixed the instance and
not the class" — after a fix passed and the class stayed open.

### Owner decisions, collected — none of these is a task

Quoted from the controller's ledger, unedited in substance. Nobody has answered any of them.

- **A hard boundary instead of a notice, on the second school.** The five-route 409 is complete and
  honestly labelled, but "**the refusal is still a page, not a boundary**". Nothing stops an
  operator reading it and creating the second school anyway; the live-video exposure is then real.
  The implementer deliberately did **not** refuse the second school outright, because that removes a
  capability §14 grants and is a product decision on the owner's behalf. One-line change plus a test
  if the owner wants the gate.
- **Narrowing who may declare a pupil armed.** A `DEVELOPER` account — the supplier's — can do it
  today, inherited via `_OPERATOR_CAPABILITIES`. Deliberate and test-pinned. Corrected by the
  reviewer: "**narrowing DEVELOPER is needed AND is NOT a one-line change — it must land together
  with `review_event` refusing a non-`BULLYING` row, or it is a narrowing that looks done.**" The
  second door above is why.
- **`docs/questions-for-school.md` §10 goes to a real school and the owner has not read it.**
  Written by an agent; "customer-facing text nobody reviewed". Still only on
  `feat/psychologist-cabinet` — `git grep '^## 10'` finds it there and **not** on `main`, so nobody
  reviewing trunk will meet it.
- **A referred event is not protected from deletion by age.** `janitor.PROTECTED =
  (EventStatus.CONFIRMED, EventStatus.REVIEWED)` and `DEFAULT_MEDIA_DAYS = 90` (both read on `main`
  and on the branch, unchanged). "'A referral does not protect the evidence' is a property nobody
  has decided." It fell out.
- **A second school must not be connected to alert-raising cameras while Telegram is configured
  once per installation.** The notifier has no idea which school a queued alert belongs to.
  Recorded as a condition on the module, not a bug in it.
- **`/media` is installation-wide.** Same leak class one step over, outside every diff. Owner-facing.

### What the measurements do NOT say

- **"0 alerts on 657 clips" is weaker than it reads.** In the first pass **no real box got past
  screen 1** — 77 819 of 77 819 sightings refused as `not_a_target` — so that zero measures the
  screen, not the accept path. The weight is carried by the third pass: 4 092 **real** boxes
  relabelled as knives on real ByteTrack geometry, 3 602 refused as `not_near_a_person`, **8
  alerts**. A scheduled synthetic detector over the same frames gave 696 alerts on 656 clips, which
  is what makes the zero a measurement rather than a silence. All of this is the implementer's run,
  not mine.
- **The weapons module has never run against a camera the school owns.** Verified: `git grep -i
  weapons -- config/` returns **nothing** on `main` and nothing on `feat/weapons-detection`. All ten
  files in `config/cameras/` are bullying/canteen cameras. There is no `weapons` role anywhere in
  the configuration, so every feasibility number on `/weapons` describes a camera that does not
  exist yet.

### What in this section I checked with a command, and what I did not

**Checked here, this session:** all three branch tips, their ahead-counts and diffstats against
`main`, and that none is an ancestor of `main`; that all three match `origin`; the naive and
tag-aware unpushed counts (3 vs 0) and that all three "local" commits are reachable from a tag that
is on origin; `git stash list` empty and the tag's true content (two files, +275, via the third
parent); both `0008` revision/`down_revision` pairs; `load_cameras()` having no database access;
`_suspend_foreign_keys` sitting before `begin_transaction()` and refusing; the eight `WATCHED` and
four `REBUILT` tables; `expire(now)` as the first statement of `WeaponsPipeline.process`;
`review_event`'s `REVIEW_BULLYING` dependency and its missing `event_type` check; `logfiles.recent`
having no school parameter; `media.py:53`; `KNOWN_TARGETS`; the asymmetry between how
`target_classes` and `confusable_classes` are validated; `janitor.PROTECTED` and
`DEFAULT_MEDIA_DAYS`; §10's presence on one branch only; the absence of any `weapons` camera in
`config/`; my own `load_cameras` counts (14 in 11 `src/` files, 31 in `tests/`); and
`cameras_are_installation_wide.html` at **both** `02acc1d` and `9066ac2`, which is how I know the
logs claim was really there and is really gone rather than taking either from the ledger.

**Somebody else's measurement, named as such:** every suite total on every branch (2411 / 2576 /
2632 are the implementers' own runs); every sabotage outcome quoted above (74 tests green with the
filter deleted; 12-vs-18; 1 of 20 on the role swap; the 409-with-body pair); the 9.8 s outage
reproduction and the 3.0 s realert gap; the corpus passes (77 819 / 4 092 / 3 602 / 8 / 696); the
`refuse_unusable_weights` acceptance with spiked confusables; the three roles' 409/403 measurement.
**Do not promote any of these to "verified" by quoting them from here.**

### What moved after this section was written — re-checked at consolidation

Written a few hours later the same day, while the three documentation branches were being merged
into `docs/consolidated-2026-07-30` so that a session starting from `main` would see §0b and §0c at
all — until that merge, **none of the two days above existed on trunk in any form.** Same day, so
this stays inside §0c rather than opening a §0d. **Every line below is a command's answer, not a
reading of the ledger.**

**Two of the three branch tips moved. The psychologist's did not.**

| module | tip in the table above | tip now | ahead | delta vs `main` |
|---|---|---|---|---|
| psychologist's cabinet | `9a59e54` | `9a59e54` — **unchanged** | 4 | 36 files, +3386/−62 — unchanged |
| several schools | `9066ac2` | **`ad90e2e`** | 20 → **21** | 50 files, **+5004/−188** |
| weapons | `86d1207` | **`3aee202`** | 5 → **6** | **40** files, **+7106/−280** |

All three still equal `origin`, and `git merge-base --is-ancestor <branch> main` still answers **NO**
for all three. Nothing reached trunk: `git rev-parse main` is still `6d19dd8` and
`git rev-list --count main` is still **260**. The suite was not re-run; 2347 remains §0b's
two-operator measurement and is not being restated as fresh.

**`ad90e2e` lands multi-school's fix round 5 of 5,** and it closes the three items the prompt file
was still listing as outstanding. Verified by reading the branch, not the commit message:

- The literal `config/schools/` check is **restored** alongside slug matching
  (`tests/test_schools_page.py:404`), and `config/schools/` is back among the scanned roots. It had
  been *traded* for `config/cameras/<slug>/`, not extended — and the lost layout is the one three
  docstrings call likeliest.
- The prose is back in agreement with the code: `config_scope.py:44` now says the trigger fires on
  **FOUR** things and lists them, and `:54` states outright that it used to say "three".
- The negation rule's bypasses are fixed: stems widened (`убер`, `сним`, `сня`), matched at a **word
  start** — as bare substrings `сня` matched inside «объяснят» and flagged a true sentence — and
  polarity is now judged **per occurrence within a proximity window**, so a negation forty characters
  away belonging to a different verb no longer disarms the claim. Three sabotages that used to pass
  are pinned as a permanent regression set in both directions.
- **The hole is declared, not hidden:** the forty-character window is a judgement, and a negation
  landing inside it that belongs to something else still disarms the rule. Narrowed from "anywhere in
  the sentence", **not closed**, and the branch says so itself.

**What `ad90e2e` does not have is a re-review.** Round 5 was landed by the implementer; no reviewer
verdict for it exists anywhere on disk. §0c already recorded that re-review 4's verdict existed
nowhere. That is now true of round 5 as well — **do not read `ad90e2e` as reviewed.**

**`3aee202` is an unverified WIP, in its own words:**

> "INCOMPLETE and UNVERIFIED. The suite was not run, ruff was not run, no sabotage was done.
> Committed so that hours of work survive a disk, which is the lesson the 169 lost clips taught
> this week."

It *writes* closures for the three Important findings this section describes — the second door, the
three green states of `/weapons`, and the unvalidated `confusable_classes`. **Written is not
measured.** Two consequences for the text above:

- **"no test anywhere hits that route with a weapons row" is no longer true.**
  `tests/test_weapons_second_door.py` exists at `3aee202` (+247 lines) and is absent at `86d1207`,
  and `web/routes/events.py:179` is now
  `if event is None or event.event_type is not EventType.BULLYING:`. The defect as *described* above
  was real and was measured at `86d1207`; the fix has never been run.
- **`confusable_classes` is still validated against nothing.** At `3aee202` it appears only at
  `config/weapons.py:180`, `:330` and `:333` — the overlap check — while `target_classes` is still
  checked loudly against `KNOWN_TARGETS` at `:323-327`. This finding stands unchanged.

**Re-verified unchanged, each by its own command** — these are the claims above that a day could
have rotted and did not:

- `git stash list` is **empty**; the annotated tag `archive/stash-2026-07-14` exists locally and on
  `origin` (`git ls-remote --tags origin`), peeling to `af5b80b` in both places.
- The unpushed count still lies exactly as described: `git rev-list --all --not --remotes` returns
  the three tag-reachable commits `af5b80b`, `8ada5f4`, `a49b085`, and the tag-aware form
  `git rev-list --all --not --remotes --tags --count` returns **0**.
- `migrations/env.py::_suspend_foreign_keys` is at `env.py:41`, called at `:88`, and
  `context.begin_transaction()` is at `:97` — **the call really does precede the transaction**, and
  the refusal when the pragma does not take is there.
- `WATCHED` in `tests/test_migration_keeps_the_events.py` on the multi-school branch is still the
  **eight** tables named above, with the four `REBUILT` controls.
- Both `0008` migrations are still there under their two filenames, both `revision = '0008'`, both
  `down_revision = '0007'`.
- `config/loader.py:111` is `def load_cameras(...)` and reads YAML with **no database access**.
- `src/qorgan/weapons/pipeline.py:134` is `self.tracks.expire(now)`, and it really is the **first
  statement** of `process()` (the signature and docstring occupy `:113-133`).
- `web/routes/media.py:53`, `src/qorgan/config/weapons.py:55` (`KNOWN_TARGETS`, five slugs) and
  `diagnostics/logfiles.py:150` (`recent(category, page, page_size)` — no school dimension) are all
  where they are said to be.
- `git grep -i weapons -- config/` still returns **nothing** on the weapons branch; the module still
  has no camera the school owns. `docs/questions-for-school.md` §10 still exists **only** on
  `feat/psychologist-cabinet` and not on `main`.

**One claim in the prompt file was simply wrong and is corrected there: the weapons branch adds no
migration.** `git diff --stat main...3aee202 -- migrations/` is empty, and both `main` and the
weapons branch carry the same **seven** version files ending at `0007`. The migration collision is
exactly one — `0008` between the psychologist and multi-school — not two.

## 0d. What changed on 2026-07-31

A controlling session ran two SDD fix loops to their five-round cap, adjudicated everything left
open at the cap, and finished the documentation task. It kept a ledger as it went, and that ledger
is this section's **only** source. The ledger itself is **deliberately not in the repository**:
`<repo>/.superpowers/sdd/` is git-ignored and this project has already nearly lost 36 KB of analysis
there to a `git clean -fdx`. This section is the part of it that gets carried in.

**Three parties appear below and they are not interchangeable.**

- **The controller** — ran suites himself, sequentially and alone, into `--junitxml`, parsed the
  XML rather than reading a terminal, and adjudicated findings against the code rather than on a
  report's word. Where the ledger says "measured by me", that is him, and it is labelled so here.
- **Implementers and reviewers** — their suite totals, their sabotage outcomes, their predictions.
  Those stay theirs and are named as theirs. **A report is a claim, not a proof.**
- **This session**, writing the section — every `git` figure in the branch table, the merge notes
  and the commit ranges below was **re-run here on 2026-07-31**, not copied out of the ledger. The
  **suites were not re-run here.** No suite total in this section is mine.

Nothing below is promoted to a measurement by being restated.

### The branch table, re-run here — and §0c's two tables are both stale

Measured 2026-07-31 in the `docs/consolidated-2026-07-30` worktree with `git rev-parse`,
`git rev-list --count main..<branch>`, `git diff --shortstat main...<branch>`,
`git rev-parse origin/<branch>` and `git merge-base --is-ancestor`:

| module | branch | tip | ahead | delta vs `main` | where it stopped |
|---|---|---|---|---|---|
| psychologist's cabinet | `feat/psychologist-cabinet` | `9a59e54` | 4 | 36 files, +3386/−62 | **unchanged, and not examined at all this session** |
| several schools + tenancy | `feat/multi-school` | `ebec713` | 26 | 56 files, +6551/−199 | the independent review it had never had, then **5 fix rounds, capped**. 2 Important + 2 residual limits parked with written rulings |
| weapons | `feat/weapons-detection` | `ec2df29` | 20 | 43 files, +8167/−280 | **5 fix rounds, capped**, on top of a controller-verified Task 1. 3 Important + 2 Minor parked with written rulings |

All three still equal `origin`. `git merge-base --is-ancestor <branch> main` still answers **NO** for
all three. `main` is `6d19dd8`, `git rev-list --count main` is still **260**, and it equals
`origin/main`. **Nothing was merged into `main`, and nothing was pushed to it.**

> **§0c names `9066ac2`, `86d1207`, `ad90e2e` and `3aee202`. All four are behind now.** §0c already
> corrected its own table once, in "What moved after this section was written"; **that correction
> has itself gone stale**, and §0c is not being rewritten, because dated entries here are
> append-only. **This paragraph is the correction.** Do not check out any of those four expecting
> the current branch, and do not read §0c's ahead-counts or diffstats for multi-school and weapons
> as current — the table above is.

> **On the suite totals §0c carries.** 2576 (multi-school) and 2632 (weapons) are implementers' runs
> at tips that no longer exist; for weapons they are superseded below by the controller's own runs,
> each named with the commit it was taken on. 2411 (psy) is still that branch's own figure at an
> unchanged tip, and it is still **the implementer's run, not anybody else's**. **For multi-school
> there is no controller measurement at any tip** — see "Two figures the controller threw away".

Session commit ranges, counted here: `3aee202..ec2df29` is **14 commits** (5 files, +1164/−103);
`ad90e2e..ebec713` is **5 commits** (14 files, +1564/−28).

### Task 0 — the baseline stopped being a citation

Before anything was dispatched, the controller took the baseline himself. Two suites sequentially,
alone, nothing else on the machine, XML parsed rather than read off a terminal. **This is the
controller's own run:**

```
main-equivalent tree (code byte-identical to main @ 6d19dd8)
    pytest  2347 collected · 0 failed · 0 errors · 0 skipped · 616.4 s   exit 0
    ruff    All checks passed!                                          exit 0

feat/weapons-detection @ 3aee202
    pytest  2652 collected · 1 failed · 0 errors · 0 skipped · 550.2 s   exit 1
    ruff    All checks passed!                                          exit 0
    the ONE failure:
      tests.test_code_limits::test_every_function_is_under_the_line_limit[model.py]
```

Three things it settles:

- **2347 is now confirmed by a third operator on a third run.** §0b had two.
- **The weapons failure set is exactly the one the owner named** — nothing else was broken
  underneath it. The owner's claim was a claim; it is now a measurement.
- **`ruff` is clean on the weapons branch** — which `3aee202`'s own commit message said had never
  been run. So any ruff error appearing after Task 1 belongs to Task 1.

Id lists were kept (`base-main.ids` 2347, `base-weapons.ids` 2652) so that every later delta is
decomposed **by diffing ids, never by matching a total**. That decision is what caught two separate
disasters later in the session.

### Task 1 — weapons, `3aee202..ec2df29`: what the loop closed

Two things went in. `YoloWeaponModel.__init__` was **64 lines against a 50 cap**
(`model.py:86-149`, the controller's own AST measurement), which was the branch's only red. And the
structural guard over `STARTUP_CHAIN` **passed while inspecting zero handlers** — the controller had
demonstrated that on a scratch copy before dispatching.

Both closed, and the blind spot closed **by shape rather than by patching the instance**. The guard
now holds three rules it did not: every name in `STARTUP_CHAIN` must still resolve to a `def` in its
file, so a rename, a move, or a conversion to `async def` (which `ast.FunctionDef` does not match at
all) fails on the commit that does it; every `except` in a chain file must sit inside a step the
guard inspects; and a call to a named step from outside the chain is judged **per call**.

**The controller re-ran the sabotages himself against the committed code** — not on the
implementer's word — each reverted in a `finally`, with the tree and `HEAD` asserted afterwards:

```
guard untouched                                  10 pass / 0 fail
helper SWALLOWS instead of re-raising      -> RED  swallows_the_refusal[...model.py]
helper DROPPED from STARTUP_CHAIN          -> RED  no_handler..._out_of_the_guards_reach
a named step becomes `async def`           -> RED  every_step_named...still_exists
after all three: git status CLEAN · HEAD 8b38aae unchanged · guard 10 pass / 0 fail
```

**The middle one is the point.** Before the fix, dropping the helper from `STARTUP_CHAIN` left the
guard green while inspecting zero handlers. It is red now, and the controller watched it go red.

The round-by-round work is not reproduced here; three things from it generalise and are in "The
lessons this session earned" below. Two judgements are worth recording because they set precedent:

- **The implementer rejected the route the controller offered, with a measurement, and was right
  to.** The offered assertion ("each private step name has exactly one `def` in `src/`") would have
  reddened the round's own control — an unrelated module legitimately defining a private `_serve`
  while reaching into nothing — and its only remedy is renaming correct code. That is the
  unanswerable red the controller had himself ruled out twice on this branch. **It applied his rule
  against his suggestion.**
- **On one finding the reviewer was wrong and the implementer was right, and the controller
  measured before ruling.** The review's evidence was that `contextlib.suppress` does not appear in
  `src/`. It does — `supervisor/supervisor.py:118` and `worker/entrypoint.py:241`, and
  `entrypoint.py` **is itself a `STARTUP_CHAIN` file** where that use, around `signal.signal`, is
  legitimate. The proposed whole-file rule would have made a correct line permanently red with no
  way to answer it. The finding was real; the remedy was not. **A disputed point is settled by
  measurement, not by whoever repeats it last.**

### Task 2 — several schools, `ad90e2e..ebec713`: the review this branch had never had

`feat/multi-school` reached fix round 5 of 5 in a previous session **without an independent review
of the module at all**. This session commissioned one, read-only, no test execution. It returned
spec ❌.

**Finding A — CRITICAL. `/schools` was unreachable forever.** The reviewer said "SUPERADMIN
unassignable". The controller traced it end to end rather than accepting that, and it is worse and
more exact:

- `accounts.py:114-116` — `ASSIGNABLE_ROLES` excludes `SUPERADMIN`;
- `accounts.py:119-132` — `parse_role` raises `RoleRejected("unknown role")` for it;
- `cli.py:327` — `--role` offers `choices=[r.value for r in UserRole]`, superadmin **included**;
- `cli.py:212` — the CLI calls `parse_role`, so it hits the same refusal;
- `accounts.py:112` — a comment says a superadmin is made by `qorgan user add --superadmin`, and
  **`--superadmin` occurs exactly once in the entire tree: inside that comment.** No flag, no test.

So `qorgan user add x --role superadmin` was advertised by `--help`, accepted by argparse, then
refused as *"unknown role"* — for a role that is in the enum and in the help text. No path created a
`SUPERADMIN`, so `MANAGE_SCHOOLS` guarded nothing and `/schools` was dead. The module's own "Done
when" failed on **both** halves.

**Why the suite stayed green is the more valuable half:** `tests/test_schools_page.py:64` mints the
user as an ORM row directly, and **no test under `tests/` touches `parse_role` at all** (grepped by
the controller, zero hits). Every superadmin test proved the page works for a row it created itself.
Not one walked the path an installer walks. The suite total quoted around this finding — **2583** —
is an implementer's run at an older tip and is **on the controller's own list of totals he did not
take**; the grep is his, the number is not.

**Finding B — adjudicated as not this branch's.** The reviewer filed `web/routes/media.py:53` as
unscoped by school. `git diff main...feat/multi-school -- src/qorgan/web/routes/media.py` is
**empty** — the branch never touched it. It is pre-existing on `main`, already in §0c and already on
the owner's list. Real, but neither a regression nor a Task 2 defect. **Parked with a ruling; it
stays an owner item.**

What the five rounds then closed, each verified by the re-reviewer or the controller rather than
taken from the fix report:

- **The creation path.** `qorgan user add installer --role superadmin` measured rc 0, writes
  `school_id=NULL`, `/schools` → 200, with `ASSIGNABLE_ROLES` and `parse_role` byte-identical and
  still refusing `SUPERADMIN` for the web form. The new test binds to the **creation path** — empty
  DB, `cli.main()==0`, exactly one row, and a login verifying the hash the command wrote — so a
  hand-minted row cannot satisfy it.
- **The login dead end that the fix's own headline test passed against.** Measured by the
  re-reviewer: `POST /login → 303 Location='/login'`, and following the redirect the person lands on
  a blank login form — no nav, no `/schools` link, no username, **no error** — indistinguishable
  from a failed login. `/schools` opened only by typing the URL. **The new test asserted the
  redirect's status and not its `Location`**, so it was satisfied by exactly that dead end while its
  message said the account can log in. Closed in the next round and walked as a person walks it:
  `POST /login → 303 Location='/schools'` → `GET /schools → 200`, register drawn, nav drawn,
  username shown, no form, no error, `/login` while signed in redirects rather than loops. The
  replacement assertion follows `Location`, requires 200, and requires an `id` that exists in no
  other template.
- **Fixtures that mint a row production can no longer produce.** Probed against the real models:
  `None -> 1`, `null() -> None`, omitted `-> 1`, and **with two schools `None` raises
  `UndecidedSchool`** — one test survives only because `_accounts()` runs before the second school
  is added; reorder those two lines and it errors. **Blast radius bounded, and that is the good
  news:** every superadmin assertion on the branch is capability-based and no test reaches
  `school_of` as that role, so **no tenancy guarantee rested on the lying fixtures.**
- **`qorgan user add --role admin` crashed on a two-school install** — `UndecidedSchool` is a
  `RuntimeError`, not an `AccountError`, so nothing caught it and the installer got a traceback, on
  precisely the configuration this module exists to enable. Folded in deliberately as the module's
  own, and closed.
- **A false causal story in the code.** `web/security.py:182-183` said the dead end went unnoticed
  because no superadmin existed. Measured: a superadmin **did** log in on every suite run at
  `7f99333`, via a helper that asserted `status_code == 303` and sailed over it. **The status-only
  assertion hid it, not a missing account.**

**The best catch of the session was the implementer's own, in the final round.** Backtick pairing in
the branch's prose scanner was sequential, and prose was flattened **per file**, so a single
markdown escape desynchronised every pair after it **for the rest of the file**. Measured: the
check's own module found **zero** `--school` spans in itself — an escape sat at docstring line 34
and everything below it was invisible. Behind that silence were six real reds across two files,
**including one its own fix from earlier in the same round had introduced.** Its words: *a check
that cannot see the defect the same commit introduces is not a check.* Fixed twice over, because
either half alone leaves a way back in.

On the round's other finding the honest distinction it drew is the one worth preserving: with a leg
dark, the direction-B test **still cannot see** the re-planted instance — the check is genuinely
blind — but it now **refuses to report green while blind**. The controller's own sabotage scenario
went from 15/15 green to two reds.

### Task 3 — documentation, and a spot-check done by loading rather than reading

Complete. `docs/the-runbook-describes-a-system-that-moved` → `f69b5ad` (`HANDOVER.md` +
`README.md`); the session plan carried onto `docs/consolidated-2026-07-30` → `7f52eaf`. Re-measured
here: the runbook branch is **1 ahead of `main`, 2 files, +78/−20**; the consolidated branch is **8
ahead, 3 files, +1290/−109**. `main` untouched.

It found **two false claims beyond the two it was handed**, both in `HANDOVER.md` §7:
`_telegram_skip_reason` exists (`events.telegram_skip_reason`, migration `0007`), and classroom
analytics exists (`feat/classroom-lesson-metrics` is merged, `/lessons` is live, admin-only) — with
the honest limit that **zero classroom cameras are configured**. Weapons, teacher analytics,
psychologist and superadmin were re-checked and are **still absent**, and were kept with their
unmerged-branch provenance rather than quietly dropped.

**The agent's report is a claim, so the controller spot-checked it** — it is client-facing text.
Four of its concrete new assertions, verified against `main` by him:

```
bullying.clip_seconds   config/bullying.py:379  Field(default=3.0, gt=0, le=15.0)   OK
TelegramSkipReason      enums.py:178  StrEnum, named members, "not free text"       OK
migration 0007          adds events.telegram_skip_reason (and drops it on down)     OK
VIEW_LESSON_METRICS     ['admin']          MERGE_PERSONS ['admin']                  OK
VIEW_PUPILS             ['admin','developer','operator'] — and the text says so     OK
```

The permission table was checked **by LOADING it, not by reading it** (§7.2, `0b2f4c2`), which is
what makes "only `admin` holds it" a measurement rather than a reading of a source file — and one of
the three turned out **wider** than "admin only", which the corrected text already states correctly.

Two things carried forward: §8's line pointers for `HANDOVER.md` §6b and `README.md`'s deviations
table are now **274** and **132**; and **`HANDOFF.md` was out of scope and has NOT been audited.**
It is the next document to check.

**The gap this session opened with is closed:** all **14** documents named in §8's reading order and
in `docs/next-session-prompt.md` now resolve on `docs/consolidated-2026-07-30`, checked path by path
rather than trusted from a report. The plan blob was verified **by hash** — `d348cb2…` on both
`docs/consolidated-2026-07-30` and `docs/plan-modules-before-the-pilot`, so it was copied by
path-checkout, not re-typed, and the branch's stale `src/` did not come with it.

### The controller's own suite figures, each pinned to the commit it was taken on

Every line here is the controller's own run, sequential and alone, except where it says otherwise.
**None of them is mine, and none of them is current for a tip that has since moved.**

```
main-equivalent @ 6d19dd8   2347 / 0 / 0 / 0 · 616.4 s · ruff clean · exit 0
weapons @ 3aee202           2652 / 1 / 0 / 0 · 550.2 s · ruff clean · exit 1
weapons @ 8b38aae           2660 / 0 / 0 / 0 · 500.2 s · exit 0  — the IMPLEMENTER'S run,
                                 its XML re-parsed by the controller, not re-executed by him
weapons @ 6cab950           2663 / 0 / 0 / 0 · 518.5 s · ruff clean · exit 0
weapons @ c08c88c           2666 / 0 / 0 / 0 · 553.5 s · ruff clean · exit 0
multi-school @ 7f99333      2595 / 0 / 0 / 0 · 554.8 s · ruff clean · exit 0
```

Each delta was decomposed **by diffing ids**: `2652 → 2660` is +12/−4 net +8, four structural tests
re-homed plus six genuinely new cases plus the fixed `+2` cost of one new `.py`; `2660 → 2663` is
+3, nothing lost, exactly the three params of one new rule, and it makes fix round 1's *prediction*
of 2663 a measurement rather than a prediction; `2663 → 2666` is +3, nothing lost, a file split's
fixed cost plus one new test.

**The weapons tip, and the sentence the controller refused to round up:**

```
feat/weapons-detection @ ec2df29
   2670 collected · ruff clean · id delta vs his own c08c88c run: +4, NOTHING LOST
   ONE red: tests/test_camera_loop.py::test_det_every_is_honoured
```

That test is **named in this repository's documentation as a contention flake** — *"it measures the
machine as much as the code; if it goes red, suspect the load before the logic."* He applied the
prescribed remedy instead of filing a defect: **three runs alone, 6 collected each, 0 failures, 0
errors, exit 0 every time.** So it is load, not logic. Stated precisely, in his words:

> **I do not hold a clean `2670/0/0/0` on weapons.** I hold 2670 collected with one contention red,
> plus that test green 3/3 in isolation. That is the same standard §0a used to settle `main`'s
> original false red, and it is enough to call the branch green — but it is not the same sentence as
> "a clean full run", and I am not writing the stronger one.

**There is no controller run of the multi-school tip `ebec713` at all.** Both attempts were thrown
away; see below. The last controller figure for that branch is `7f99333` — **five commits behind the
tip** — and it may be cited only as an id baseline, never as the branch's number.

His own artefact chain, checked file by file rather than assumed:

```
base-main 2347 · base-weapons 2652 · t1-full 2660 · final-weapons 2663 · final-tenancy 2595
```

### Two figures the controller threw away, and the two process failures behind them

Both of these are the controller's own errors, recorded by him as such. They matter more than the
numbers they cost.

**(1) A shared scratchpad is not isolation.** He gave every agent the same scratchpad directory. Two
independently chose the prefix `t2-` — one meaning "task 2", the other "task 1, round 2" — and the
weapons agent's files **overwrote the tenancy agent's round-1 evidence** (`t2-final.xml`,
`t2-final.ids`, `t2-sab1/2.xml`, clobbered at 22:49). His note on it is the sharp part: *this is a
lesson I already had written down — parallel agents need a worktree each **and** a unique scratchpad
path each, or their numbers collide and look like success. I isolated the worktrees and did not
isolate the scratchpad.*

**What caught it was the discipline, not luck.** The tenancy agent decomposed its delta by diffing
ids and got "removed=136, added=2047" — obvious nonsense. **A count-only check would have compared
2601 against 690 and looked entirely plausible.** Third time on this project that id-diffing has
caught something a total would have hidden. Fixed going forward: **a subdirectory per task, not a
filename prefix.**

**(2) A suite run against a worktree that had a live writer in it — twice, the same error.**

- First: a tenancy figure came back `2602 collected / 1 failed`, the failure being
  `test_every_function_is_under_the_line_limit[cli.py0]`. Checked rather than filed: he had
  dispatched the round-3 implementer into that worktree and *then* let his own full suite run
  against it. The tree carried **8 modified/new files** (`cli.py` alone +81/−17) while `HEAD` was
  still `6602271`. **The suite measured a half-written working tree.** Discarded.
- Second: the same thing again, tree **dirty with 4 files** during round 5, so its two reds were the
  agent's in-progress edits — including its own new file over the 500-line cap. Discarded.

**The fix is mechanical rather than an intention, and that is the whole lesson.** After the first
contamination he added a refuses-to-start-dirty guard — but **it checks once, at the top of the
script**, and the second suite runs ten minutes later, by which time a writer he dispatched himself
has moved in. **The guard must run immediately before each suite, not once per script.** In his
words: *both bad numbers were caught by checking provenance before believing them, which is the
habit that keeps saving this session — but a habit that fires after the fact is a worse control than
a check that fires before.*

**One more thing that could have gone wrong and did not.** Two agents were killed mid-flight by an
API session limit (`You've hit your session limit`). The first action was the safety check this
project earned the hard way — **an agent that dies mid-sabotage cannot report that it left one.**
All four worktrees were checked at their heads, clean, and `grep SABOTAGE` over `src/` and `tests/`
in both working branches returned nothing. Nothing to undo, and one agent's 21 KB write-up had
already reached disk despite its last words being "let me write the full write-up".

### The lessons this session earned

**A pin detects CHANGE, not TRUTH — and this one is new to the project's record.**

> Pinning a sentence guarantees nobody edits it silently. **It does not guarantee the sentence was
> true when it was pinned.** Round 3 installed a pin to stop a paragraph going stale; rounds 4 and 5
> then pinned two more false clauses into place. The mechanism works exactly as designed and the
> description of the mechanism keeps drifting.

This is the single fact that makes sense of the three Important items parked on Task 1. It does not
argue against pinning — the pin is what turned a recurring silent falsehood into something that
fails loudly — it says what a pin is **for**.

**The obvious revert command can un-fix your own work.** This belongs beside the sabotage chain,
which reads: *a check nobody watched fail is not a check* → *TDD red is not enough, sabotage the
fix* → *a sabotage you did not verify APPLIED is a no-op* → *a sabotage you did not verify REVERTED
is a defect you shipped.* Add:

> An implementer's first instinct for reverting a sabotage on `schools.html` was
> `git checkout -- schools.html` — a file it had **also legitimately fixed in the same working
> tree**. That checkout would have discarded the real fix along with the sabotage and left the
> branch **green, unfixed, and looking finished** — the exact failure mode the whole sabotage
> discipline exists to prevent, arriving through the cleanup step. It was blocked, and it flagged
> it.

The controller's own `controller_sabotage.py` used the same command and was safe **only** because it
refuses to start unless `git status` is empty and asserts the restored bytes equal the originals.
His words: *that guard was luck dressed as rigour until now.* It is a rule from here on: **revert to
the bytes you saved, not to `HEAD`, whenever the file also carries your fix.** All the reverts in
the final rounds were done to saved bytes, verified by SHA-256.

**A proof is pinned to the tree it was taken on — including when the tree is your own diff.** A
paragraph opened *"every figure below is measured with the rules as they stand"* and said `docs/`
yields **three** reds; measured with **that same commit's own** flattener it yields **seven**,
because the backtick-run collapse the commit shipped opened markdown's fenced blocks and the figure
was not re-measured afterwards. The parenthetical added a round earlier to correct the round before
it is now false too. **A number correct when written, invalidated by the same commit that wrote it,
and its correction invalidated as well.** The controller applied the same rule to himself: two of
his own full-suite runs were declared **superseded id baselines only**, because the tips moved after
he took them.

**A rule cheaper to defeat than to obey is a rule that gets defeated.** A per-module skip meant a
module that reaches into a private step **and** contains any `def` of that name went green — so the
cheapest way to satisfy the red was to add a decoy `def`. Judged per call now, and all three decoy
spellings that were green are red.

**A rule with no way out is one the next maintainer deletes wholesale.** A module-level handler
produced a red **nobody could satisfy** — there is no function name to add. The ruling was to route
it to *judgement* rather than to *exemption*, and the residual limitation is **declared in the
code** in the style this project already uses: narrowed, not closed, and said so.

**A parked item that is actually done is a backlog entry somebody will pay to rediscover.** One
finding referred up as a deferral turned out to have shipped fixed in the same diff. Recorded
**CLOSED**, not parked.

**A test that does not travel the path a human travels — §0c counted six; it is eight.**

7. **Nothing under `tests/` touched `parse_role`.** Every superadmin test proved the `/schools` page
   works for an ORM row the test minted itself, while the only path an installer can walk was
   refused as "unknown role". The whole suite green — on an implementer's run, not the
   controller's.
8. **The fix's own test asserted a redirect's status and not its `Location`**, so it passed against
   a login that dead-ends on a blank form with no error. Its message said the account can log in.

The counter-measures are the ones §0c already names, plus one this session paid for: **bind the test
to the writer, not only to the invariant.** A replacement test that reddens under three production
sabotages was still satisfied by five hand-minted rows in the diff's own spelling — so "hand-minted
rows leave it red" was true only because one spelling cannot write NULL. Recorded rather than
overstated.

**A check contains, inside itself, the class of defect it was written to close — repeatedly, and
each time inside the fix for the previous one.** §0c recorded this shape once. This session produced
it five more times: five Important findings on Task 1's review were **all** about the completeness
of the new guard; a fix for one finding reproduced a different finding inside itself; narrowing a
reader back to its original scope produced **no red at all** — two greens over a dead rule, which
the implementer called *the signature defect wearing the costume of its fix*; the prose scanner
could not see the defect its own commit introduced; and a docstring paragraph has now been wrong
**five consecutive times**, each correction false in a new way. That last one is why round 4 was
dispatched to a **fresh implementer**: four consecutive false corrections to one sentence is the
textbook signal that an author cannot see their own problem.

**Prose that names something which does not exist — a class worth naming, two instances one file
apart.** `accounts.py:112` credited a flag `--superadmin` that occurs **exactly once in the tree,
inside that comment**. `db/models/auth.py:27` names `test_only_a_superadmin_belongs_to_no_school` as
the test enforcing an invariant, and **that test does not exist anywhere in the tree.** This is §0c's
"prose is not checked by a test suite unless a test reads it", in its most expensive form: a
reference to a guarantee nobody has to provide.

**A test asserting only a refusal, again.** An ordering guard asserts only `!= "/login"` and opens
nothing, and **no test pins operator, admin or developer to `/`** — so the silent-reorder failure
that its own sibling docstring names as the reason for its design is uncovered for three of five
roles. §7.1's "THE BIG ONE", one more time.

### What is parked, and with which ruling

Both loops hit their cap at round 5. **Anything still open at the cap was adjudicated in writing and
parked; there is no round 6.** The controller's reasoning for the cap is worth keeping: adjudicating
earlier would be pre-judging with a different name, and adjudicating never would be the loop that
does not converge. **Neither loop was declared BLOCKED** — nothing downstream builds on any parked
item, and every measured defeat of a guard is closed.

**Weapons (`ec2df29`) — 3 Important, 2 Minor:**

- **I3 — parked, real, not load-bearing.** `model.py`'s paragraph carries two false clauses: "over a
  two-module fixture" (it is five) and a "what it still passes over" list that omits the
  `cls = type(model)` shape. Sixth falsehood in that paragraph. Documentation precision inside a
  guard whose subject is documentation precision.
- **I4 — parked, and the sharpest.** A bullet says including `__init__` is "GREEN over `src/` today
  … latent, not live". **Measured RED under the rule shipped in the same commit** — both sites are
  `super().__init__()`, whose qualifier is a Call rather than `self`/`cls`. **First thing to fix if
  anyone reopens that file.**
- **I5 — parked, and the one to watch.** The per-call narrowing buys a *narrower* version of the red
  it was chosen to avoid: `other._serve()`, `self.hatch._serve()`, `hatch._serve()` in a module
  reaching into nothing are now reported, and **the red's message gives false advice for them**.
  Latent — no such code exists today. A rule that misadvises is a rule that gets deleted whole.
- **Minor — bare-name ambiguity: leave it skipped, correct the reason.** "Python cannot say" becomes
  "not decidable in general, and this guard declines to guess" — module-level import-versus-`def`
  *is* decidable by textual order; only the general case is not.
- **Minor — the sibling private reach: the docstring's self-accusation is FALSE.** The rule polices
  five `STARTUP_CHAIN` step names, not privacy in general, so those two helpers would not be flagged
  even if it read `tests/`. The module accuses itself of a violation it does not commit — **the same
  disease, in the apology.**
- **Carried, not parked:** `tests/test_weapons_startup_chain.py` is at **483 of the 500-line cap**
  and `weapons_chain_reader.py` at **464** — the next addition to either must **split first. Split,
  never loosen.** And `("<module>",)` as one file's entry leaves all four rules green while
  resolving no name.

**Several schools (`ebec713`) — 2 Important, 2 residual limits, 1 owner item:**

- **I1 — parked.** The "three reds in `docs/`" figure invalidated by its own commit, described under
  the lessons above. **Not load-bearing:** all seven are phantoms under `docs/superpowers/`, the
  argument the paragraph justifies is untouched, and markdown is out of scope by decision.
- **I2 — parked.** The `LEG_FLOORS` comment credits the **named file** with defeating an empty
  extraction; measured, the named-file assertion tests membership of the file *list* and is
  satisfied by an empty extraction exactly as a count is. What actually defeats it is the invocation
  and real-span floors named in the same sentence. **The mechanism is sound; the sentence
  misattributes which half does the work.**
- **Residual limit — file granularity is not covered.** Emptying one file's prose (`cli.py`,
  `schools.py`, `test_the_superadmin_can_be_created.py`) is 21/21 green, because those legs run 46
  and 53 invocations against floors of 20. **Leg granularity is covered; file granularity is not.**
- **Residual limit — a leg dropped from `sources()` AND `LEG_FLOORS`**, eleven lines apart in one
  file, is 20/20 green. Two coordinated edits, not one.
- **`web/routes/media.py:53` — parked with a ruling as an owner item**, not a branch defect.

**Left open by explicit decision at the weapons cap, and named as the sharpest omission in the
paragraph that justifies the blind spots: Python string literals are unscanned — 106 sites in
`src/`, including `cli.py`'s refusal text and `--help`.** Closing it needs a rule separating a
*declaration* (`add_argument("--role", …)`) from a *message*, and that is a design rather than a
switch. **Documented, not closed.**

### Owner decisions — six carried, two sharpened by measurement, one new

None of these is a coding task, and nobody has answered any of them. The six §0b and §0c recorded
all still stand; only the changes are written out here.

**1. A hard boundary instead of a notice on the second school — NEW EVIDENCE, measured by the
controller.** The five-route 409 is a *page*, not a *boundary*; nothing stops an operator creating
the second school. Until now the recorded consequence was cross-school live video. There is a second
and more mundane one:

```
resolve_school_id called from 12 sites: canteen/reports:186 · classroom/reports:183,230
  events/store:32,196 · faces/gallery:150,246 · faces/importer:334
  identity/duplicates:116 · identity/merge:141,182 · identity/report:258
```

and **`src/qorgan/faces/cli.py` contains no `school_id` anywhere** — no `--school` argument, and
`import_directory` is called without one. So on a two-school install
`qorgan pupils import-roster student_photos/student_photos` — **step 6 of `HANDOVER.md` §1's install
order** — reaches `resolve_school_id(session, None)` and dies with an uncaught `RuntimeError`. Same
for `gallery-report`, `report` and `merge`. **The documented day-one sequence crashes on precisely
the configuration this module exists to create, and the failure is a traceback rather than an
answer.** A hard gate would prevent both consequences. Deliberately not expanded into a fix round:
it is a coherent, separable piece of work across the `pupils`/`identity` CLI surface, and widening
an already-long loop is how a task stops converging.

**2. Narrowing who may declare a pupil armed — sharpened, and still not done.** Verified by reading
`roles.py`: `CONFIRM_WEAPON_ALERT` sits inside `_OPERATOR_CAPABILITIES` and `DEVELOPER` inherits
that set, so the supplier's own account can still do it. The second door (`review_event` on a
non-`BULLYING` row) is closed **in code** at `3aee202` and **was never measured**; it is measured now
only in the sense that the branch's suite is green, not that the narrowing was done. **The narrowing
itself is still not done anywhere.**

**7. NEW — `docs/questions-for-school.md` §4.6 asks the school a question the owner answered on
2026-07-23.** This one is the owner's because it is communication, not code.

The file still says both rows are `pending` and that «в вашей таблице нет ни одной строки, которую
мы имели бы право назвать подтверждённым буллингом», and §4.6 still asks «этот эпизод — буллинг, да
или нет?». **The tracked data stopped saying that a week ago.** The controller's own count of
`eval/labels.csv`, after correcting himself once:

> My own first count said "124 rows too short, prose leaking into the label column". **Wrong, and
> wrong in this project's signature way — I measured the wrong quantity precisely.** The file is 178
> physical lines of which **128 are `#` comments** that `labels.py:137` filters before csv ever sees
> them. The real content is **49 rows, every one 5 fields wide, every one carrying a camera**: 44
> `normal`, 2 `ignore`, 2 `bullying`, 1 `pending`.

The two `bullying` rows are the head-grab from both angles — `hall_right` 6.50–13.50 and `hall_left`
5.50–16.00 — and they are **the owner's own judgement, recorded properly**: `bad492d` (2026-07-23)
ruled it bullying, explicitly overruling the school's own hedge
(«окончательно между грубой игрой и издевательством не разграничить»), and `590316b` (2026-07-24)
caught that `labels.csv`'s header block still denied it and rewrote the block to record the
judgement, its date, its reversibility, and the rule that every recall figure must be quoted with
that paragraph. The mechanism is sound, and `save-baseline`/`gate` still refuse on the one remaining
`pending` (`cli.py:236-240`), which is correct.

**What is stale is the letter to the school.** The correction is one paragraph. **Telling a school
"we overruled your assessment of an incident involving your children" is the owner's call, not an
agent's** — exactly like §10. Recorded, **not actioned: no change set in this session touched
`docs/questions-for-school.md`, and none should until the owner rules.**

### Merge conditions, measured rather than assumed

- **`web/routes/events.py` conflicts in ALL THREE pairwise merges** of the pilot branches
  (`git merge-tree`), and all three edit the same `review_event` region — the file carrying the
  weapons second-door fix. **Resolve by union, never by choosing a side**, and **pin the survival of
  that fix with a test**: resolving by choosing a side loses a child-safety fix with a green suite.
  psy × multi-school conflicts in six files; psy × weapons and multi × weapons in `events.py` alone.
- **Two `0008` migrations**, both `down_revision = '0007'` — re-confirmed here today:
  `0008_a_referral_is_an_act_by_a_named_person.py` on the psychologist branch,
  `0008_a_school_is_not_the_installation.py` on multi-school. Git merges them silently; only
  `alembic upgrade` breaks. **Renumber at merge time, never on the branches.**
- **Weapons still adds no migration** — re-confirmed here today at the new tip:
  `git diff --stat main...feat/weapons-detection -- migrations/` is empty and both `main` and the
  weapons branch carry **seven** version files. **The collision is exactly one.**
- **`roles.py`: add to the union, never replace it.** Five branches once each rewrote those two
  lines and any single version silently revoked the others' pages **with a green suite**.
- **Merging `feat/multi-school` makes the runbook false again.** The documentation branch merges
  **CLEAN** against all three module branches (`git merge-tree`, the controller's measurement) — but
  **clean is not correct**: the moment multi-school lands, `HANDOVER.md` §7 and `README.md` will
  claim there is no superadmin role and no `--school` flag, and both will exist. Git will merge them
  silently and no test will notice, **because prose is not checked by a suite unless a test reads
  it.** Four one-line edits, already located: `README.md:40`, `HANDOVER.md:77`, `HANDOFF.md:52`,
  `docs/next-session-handoff.md:70`. **Deliberately not done on the module branch** — doing it there
  would create the conflict this paragraph reports does not exist.

### What is NOT claimed

- **Neither pilot branch is "ready".** This session fixed the one failing test on weapons and one
  Critical on multi-school, each through a review loop. **The weapons branch's own SDD process
  stopped at round 1 of 5 in a previous session**, and the ledger records its other findings as
  untouched by this session's work: `confusable_classes` validated against nothing, no weapons
  camera in `config/`, no model. Nobody re-measured those at `ec2df29`; §0c's account of them
  stands.
- **`feat/psychologist-cabinet` was not examined at all** this session.
- **There is no controller suite run of `ebec713`**, and no clean full run of `ec2df29` — see the
  contention-flake paragraph. Do not write either branch up as "green, full suite" on the strength
  of this section.
- **Every suite total not taken by the controller personally is still somebody else's run and is
  labelled so here.** Do not promote one to "verified" by quoting it from this section.

### What in this section was checked with a command, and by whom

**Re-run here on 2026-07-31, by the session writing this section, all `git` and nothing else:** all
three module branch tips, their ahead-counts and `--shortstat` diffstats against `main`, that each
equals `origin/<branch>` and that none is an ancestor of `main`; `main`'s hash, its 260-commit count
and its equality with `origin/main`; the two session commit ranges and their diffstats
(`3aee202..ec2df29` 14 commits, `ad90e2e..ebec713` 5); both documentation branches' ahead-counts and
diffstats; the two `0008` filenames on their two branches; that weapons adds no migration and that
`main` and the weapons branch each carry seven version files; `git stash list` empty; the naive
unpushed count (**3**) and the tag-aware one (**0**); and `git add -An` in this worktree.

**The controller's own measurements, named as his:** every suite figure in "The controller's own
suite figures" and its id deltas; the three sabotages re-run against the committed guard; the AST
function lengths in `model.py`; the focused `test_code_limits` run; the end-to-end trace of the
`SUPERADMIN` refusal; the `--superadmin` single-occurrence count; the four spot-checks of Task 3's
new assertions, with the permission table checked by loading it; the `resolve_school_id` call-site
list and `faces/cli.py` carrying no `school_id`; the `labels.csv` content count and both label
commits; `roles.py`'s `CONFIRM_WEAPON_ALERT` placement; the three pairwise `merge-tree` results and
the runbook branch's clean merge; the plan blob hash on two branches; and the two discarded tenancy
runs together with the dirty-tree evidence that discarded them.

**Somebody else's measurement, named as such and NOT promoted here:** the `2660` XML (the
implementer's run, the controller re-parsed it but did not execute it); every per-branch suite total
in §0c's table; the reviewers' findings before the controller adjudicated them; every sabotage
outcome quoted from a round's report rather than re-run by him; the re-reviewers' login walk-throughs
and the `None`/`null()`/omitted probe results; and the implementers' predictions — one of which
(2663) later became a measurement and says so, and the rest of which did not.

**Not done at all:** no suite was run by the session writing this section, `HANDOFF.md` was not
audited, `docs/questions-for-school.md` was not opened or edited, and nothing was merged, rebased or
pushed.

---

## 1. The single most important truth: the bullying detector carries no signal

Measured on the school's own clips (33 clips, one school, one day). **Read these as per-clip, not as
fleet properties** — only 4 of the 33 were instrumented (`eval/FINDINGS.md`, "Scope").

- Ordinary walking (clip20) confirms at **96.7%**. The one genuine head-grab (clip09) confirms at
  **26.7%**. **Walking confirms 3.6× more often than the incident.**
- Empirically the skeleton tier collapses to one bit: *did a hip-centre jump 28 crop-px between
  frames*. `body_fall_or_low_posture` is a genuine second path in the code and the two diverge on one
  clip, so this is an observation about this footage, not a property of the design.
- The weak features are **enablers**: `rapid_hand_motion` fires on 93–100% of anything,
  `kick_like_leg_motion` on 100% of the walk-past. They supply weight that
  `sudden_body_displacement`'s own 0.15 cannot reach. **They measure speed and crowd-count, not
  aggression** — clip09 confirms less than walking because it is slow and has 2 people versus 5.
- People are **140–160 px** at 1280×720, median keypoint confidence **0.80–0.87**. It is **not** "too
  small for pose" — that hypothesis was tested and killed. The features are structurally wrong
  regardless: thresholds live in a crop-pixel coordinate system that swings 0.55×–7.8× per pair.
- **This is not a calibration problem. No threshold fixes it.** It needs a different approach,
  decided on-site with real cameras and real labels.
- **Do NOT try to "improve detection" by tuning thresholds. That is the trap this whole project
  exists to avoid.**

## 2. The camera facts (this section supersedes every other fps number in the repo)

Source: the camera's own web UI (HiLook, 192.168.1.2), 2026-07-24 screenshots. **External to this
repo; nothing in the tree corroborates them.** What would settle it: an ISAPI
`Streaming/channels/102` query, or `ffprobe` on a live RTSP pull — both are one-minute, day-one jobs.

- **Main stream (101): 2560×1440 @ 20 fps.** The clips the school sent are this stream.
- **Sub-stream (102): 1280×720 @ 15 fps. This is what the detector is fed.**
- **There IS a self-downscale on 6 of the 10 cameras.** Only `hall`, `canteen_entry`, `canteen_exit`
  override to 1280×720. `canteen_inside`, `stairs`, `outdoor` run at `base.yaml`'s **960×540** —
  *below* what the sub-stream delivers, and every px/s threshold in those profiles is denominated in
  960×540 pixels.

**Fixed 2026-07-24 — the harness graded production at 10 fps for a loop running at 15.**
`display_fps` had three readers and none was the production loop. Now: `capture.stream_fps` is a fact
about the camera (fleet-wide 15, per-profile overrides deleted); the formula has **one** definition,
`detection/constants.py::analysis_fps`; the loop measures the delivered rate over its first 30 frames
and warns when config and reality differ by >25%; `qorgan config validate` prints the counter drift.

**Deliberately NOT done: the frame counters are unchanged.** `SUSTAINED_FRAMES=2`,
`FIRM_FRAMES=3`, `BRIEF_ENCOUNTER_FRAMES=4` were chosen for ~10 fps and at 15 fps each spans 0.67× the
time its comment claims. They are read by `gates.py`, `pipeline.py` and `scoring.py` — rescaling them
changes what the detector does, and on a detector measured to carry no signal that is choosing new
wrong numbers with more arithmetic. **On-site decision, with real labels.**

> **A check existed here and did not catch it, which is worth more than the fix.**
> `tests/test_config_deadkeys.py` fails any config key with no consumer in `src/`. `display_fps`
> passed it for the life of the project, because the eval harness read it. The test asks *"does
> anything read this key?"*; what was needed was *"is it read by the layer whose behaviour it claims
> to control?"* **Nothing currently detects that shape.**

## 3. What is built and merged

- **Canteen** — the strong module. Persisted session state machine, entry-opens/exit-closes, 90-min
  force-close wired into the supervisor tick, "who did not eat" joining the full roster. Face
  recognition ranks **by person, not by photo** — the fix for the legacy's 1816/1820 NULLs, where
  top-1 and top-2 were two photos of one child and the gap was 0 by construction.
- **Roles are capabilities, not a rank.** Deny-by-default, every role states its grants in writing,
  and `set(ROLE_CAPABILITIES) == set(UserRole)` is asserted in four separate test files.
  `CANTEEN_STAFF` holds one capability. **ADMIN and DEVELOPER differ by exactly `MANAGE_USERS` and
  `MERGE_PERSONS`** — both are claims the *school* makes about people, not things the vendor
  maintains; a developer login is the vendor's, and granting it `MANAGE_USERS` would let the supplier
  mint themselves an admin at any hour. **Psychologist and superadmin ARE now built**, each merged
  with the pages it guards — `/psychologist` and `/schools`. This line read "not built and must not
  be invented" and was correct while it stood: the rule is that a role without a page is a dead
  knob, not that these two roles were forbidden, and both arrived the only way that rule allows —
  in the same change as their routes. `SUPERADMIN` holds no child-facing capability and is the one
  role no form can assign.
- **The whole web panel (client §9) — merged this week, seven `feat/web-*` branches:**
  pupils · duplicates · cameras · notifications · logs · users · backup · settings. (That is eight
  route modules for seven branches — `duplicates.py` shipped on `feat/web-pupils`. And
  `routes/cameras.py` is not new: it was added 2026-07-12; only its template is this week's.)
  Server-rendered Jinja + HTMX/Alpine, no build step, no SPA, zero business logic in templates.
- **CSRF on every state-changing request**, deny-by-default keyed on HTTP **method**, including
  `POST /login` — login CSRF is a real attack and the login form is not an exception.
- **`/media` is two capabilities, not one**: `VIEW_BULLYING_MEDIA` and `VIEW_PUPIL_PHOTOS`. The old
  single grant would have handed a psychologist asked about one incident a photographic register of
  every child in the school. **The capability is decided by the RESOLVED path, never the URL string**,
  and anything unclassified — e.g. `.import/`, where uploaded archives land mid-import — is refused to
  everyone.
- **Secrets stop at the write site**: the RTSP password and the Telegram bot token were both landing
  in `last_error` database columns on their way to a screen. Redaction happens where the row is
  written, not where it is rendered. No scrub migration was written — there are no live rows.
- **`enabled: false` now actually stops a camera** (`worker/entrypoint.py::_switched_on`).
- **Settings page is read-only by construction**, enforced three ways: the router rejects any
  non-safe method, an AST scan of the route module and of `config/provenance.py` fails on any writer
  call (`dump`, `write_text`, `mkdir`, …), and a filesystem fingerprint proves a GET changes nothing.
  **The rule: never write through the web a value that lives in YAML — otherwise there are two
  sources of truth.** Editable through the web: runtime state only.
- **Test isolation, two layers, both merged**: `fix/test-teardown-race` (the suite could create a
  database where the developer keeps theirs) and `fix/test-isolation-leak` (the engine one test left
  behind decided which database the next test read).

## 4. On hold — do not merge without a decision

- **`fix/motion-fps-reference-scaling` (`23cf308`, worktree `q.ai-fps`).** The speed thresholds
  shipped at 2× the legacy's real value; the fix is arithmetically correct. **Held because, with the
  skeleton passing everything, no speed threshold can be chosen honestly** — and measurement showed
  it makes false alarms marginally *worse* (18→20 of 29 negatives notify). It is **84 commits behind
  `main`**: this is a rebase across a third of the project's history, not a pending merge. Do not read
  those figures — run them.
  Also on that branch: `window_drop_threshold` is a third instance of the same raw-vs-scaled bug —
  **and `main` says the opposite** (`config/profiles/hall.yaml:43`). Two tracked statements, flatly
  contradictory, both current. **Recorded, not resolved. Settle it against a live stream.**
- **`fix/skeleton-scale-invariant` (`9e3835f`, worktree `q.ai-det`).** Converts skeleton thresholds to
  torso-length units. **The branch's own record says the fix did not work:** false alerts 18/30 →
  17/30, and the walking-vs-incident ratio got *worse*, 3.63× → 8.74×. Those figures are in the
  earlier commit `3221802` and in `README.md` as that branch rewrote it; the tip `9e3835f` only says
  "a correct fix that made the thing it aimed at worse". Parked deliberately, with the
  map it drew rather than the code it carries. The constants were **not** tuned — lowering
  `DISPLACEMENT_TORSO` until the head-grab reconfirms is fitting to n=1.

**Do not touch either branch.**

## 5. Blocked on the school / on-site (not code)

> **Nothing in this repo names the school, the client, or any contact channel.** That is deliberate —
> it is a school full of children — and it is also a gap: a fresh session has no way to ask.
> **The channel is the owner, over Telegram.** Everything from the client has arrived that way.
> If you find yourself inferring what the school would want — a label, an ID, a threshold — **that is
> the moment to stop and write the question instead.** `docs/questions-for-school.md` collects them.

- **Detector redesign** — needs real cameras and real labels. The corpus has exactly **one** confirmed
  fight. One fight is not a recall number.
- **Camera placement** for canteen and hall. Recognition and pose both need faces bigger. **Optics,
  not code.**
- **Cooler zone coordinates** — deferred to on-site by the client. Zones are stored as fractions, so
  they are resolution-independent.
- **Resolution/CPU tradeoff** — measure on their i5-12400F + RTX 4070 with `qorgan plan-workers
  --force` before changing anything.

### The web→supervisor control socket was never built — mode switch and camera enable are unavailable on purpose

`REWRITE_SPEC.md:82` promises "web → supervisor over a ZeroMQ REQ/REP control socket". **It does not
exist.** The only ZeroMQ in `src/` is the preview bus (PUB/SUB, frames only). **This is the signature
defect appearing in the spec itself** — a promise true in the design document and absent below it.

Two things look like write paths and are traps:
- **`ModeLog` / `mode_logs`** — schema nothing reads or writes. A mode switch has a table waiting for
  it and no writer.
- **`cameras.enabled` is a column no process reads, but it IS written** — `ensure_cameras` rewrites it
  from config on every worker start. A web page that set it would look like it worked and be silently
  overwritten by the next worker launch.

**Do not build either against `mode_logs` or `cameras.enabled`; both would produce a page that reports
success and changes nothing.**

## 6. Roadmap for the new modules (client §12–14), cheapest → dearest

New modules **reuse** the built infrastructure, so the code is cheap; the expensive part is always
model + cameras + data, which is on-site.

1. **Identity beyond the canteen (§12.2)** — the mechanism in `src/qorgan/identity/` already
   implements every bullet; it is just instantiated only in the canteen worker. **But it only works
   where faces are big** — close range, not corridors.
2. **Multi-school / multi-tenant** — a migration + `school_id` on 4 root tables + composite uniques +
   a query filter. Mechanically small, **but the day `school_id` lands, every unfiltered query is a
   cross-tenant leak of children's data.** The client put it last; correct.
3. **Psychologist cabinet (§13)** — the role is trivial, but the page shows §12.3 signals that are not
   built. Shell is cheap; content depends on classroom analytics existing first.
4. **Weapons detection (§12.1)** — code ≈ half the canteen. **The wall is the model: the client has
   none** — their `best.pt` is 0 bytes and was a violence *classifier*, not a weapons *detector*. Most
   risk of all the new modules.
5. **Classroom + teacher analytics (§12.3–12.5)** — biggest build. Needs the one genuinely new data
   structure in the list: a **per-pupil longitudinal baseline** (compare a child to their own past
   norm, not to other children). No such table exists.

> **§18 gate lifted.** The client's own §18 said no new modules until the core is stable on-site. The
> owner has lifted that as a blocker: **phase 2 may be built off-site now, in parallel with
> stabilising the core on-site.**

## 7. Working discipline — every line was paid for by a real bug here

- **`pytest -q` silently drops its summary line.** Always `--junitxml=<path>` and parse the XML.
  **Confirm the collected count is non-zero** — a wrong path exits having collected NOTHING and looks
  exactly like success. This habit is what caught a merge that collected **8** tests instead of 1884.
- **"X clean" and "Y passes" are two different claims. Check both.** An agent reported "ruff clean"
  with 23 real errors and the merge inherited it (`f016048`, the only trunk-direct fix in recent
  history).
- **Interpreter:** `q.ai/.venv/Scripts/python.exe`. Worktrees have **no** `.venv`; run the main one
  with `PYTHONPATH=<worktree>/src`. `tests/conftest.py::_assert_testing_this_checkout` fails loudly
  if you test the wrong tree — **trust it**, it exists because a `.pth` file makes two worktrees
  silently grade each other's homework.
- **A subagent's report is a claim, not evidence.** Verify with your own hands: your own run into a
  unique XML, and **sabotage the fix** — break it and watch the *specific* test go red. A sabotage
  that stays green means you broke the wrong lever. A sabotage you did not `grep` to confirm APPLIED
  is a no-op; one you did not confirm REVERTED is a defect you shipped.
- **Before any N-run loop: one run, alone, parsed, green. Then multiply.** Twenty runs from an
  unverified baseline prove nothing — fifty minutes were spent this week proving a tree that was red
  deterministically from run 1, because of a 51-line function against a 50-line cap.
- **Do not run a suite concurrently with anything else.** It cost a false red on `main` this session
  (§0): four agents plus a suite, and a throughput assertion stalled at 62/90 frames. Known
  contention flakes: `test_analysis_rate` throughput, `test_det_every_is_honoured` timing,
  `test_web_pages.py` zmq "Address already in use".
- **Parallel writing subagents need isolation:** a git worktree each, branched off current `main`, not
  off each other, and a unique scratchpad subdir with unique filenames.
- **500-line file / 50-line function caps**, enforced by `tests/test_code_limits.py` over **`src/`
  and `tests/` both**. **Split, never loosen.** **The fixed cost of a new `.py` file is +2 collected
  items** — two `test_code_limits` parameters — in `src/` as well as in `tests/`. (This line used to
  say "+5, not 3", which is an observation about a typical new test file containing three tests, and
  it was handed to three agents as a rule on 2026-07-28 before one of them refused it and diffed the
  test ids instead. Measured on three independent branches, and again at scale on the integration
  run: 25 new files, +50 parameters, exactly.)
- **Decompose a delta by diffing test ids, never by matching a total to an expectation.** Doing so on
  the 2026-07-28 integration run explained four ids that *disappeared*: `test_code_limits` names its
  cases by basename, so adding `classroom/reports.py` beside an existing `reports.py` makes pytest
  disambiguate `[reports.py]` into `[reports.py0]`/`[reports.py1]`. Four out, eight in, nothing lost —
  and a total alone would have hidden it.
- **A test can assert the CLIENT's behaviour and look exactly like server coverage.** The
  `/media` traversal test passed — and kept passing with the defence sabotaged — because **httpx
  collapses `..` before the request is sent.** An attacker does not use httpx. `%2F` is how you make
  the client send the bytes an attacker would.
- **Write from memory only what no command can check. Everything else — check it.**

## 7.1 The defect classes this project keeps rediscovering

Mined from the commit messages, which are this project's real archive (§8). Each of these has bitten
**more than once**; the repetition is the reason they are listed rather than left in history.

- **THE BIG ONE — the empty or absent state satisfies a negative assertion.** Four separate
  instances in eight days. A test asserting only a *refusal* stays green when the whole area is
  deleted and everybody is refused (`de6aa59`). Two "passing" permission tests were passing because
  a missing route 404s (`4cfe24f`). A redaction test passed by rendering *nothing* — it invented a
  group name and `/cameras` maps heartbeats by group (`b0028b2`). `assert x is not None` passed with
  the worker deliberately stopped one line after starting (`ef0aad4`). **Assert in both directions,
  and assert the product still works, not only that the bad thing is refused.**
- **A permission table with one grant per line merges by SILENT REVOCATION.** `roles.py` conflicted
  **five times in eight days**. Each branch grants its own capability on its own line, so taking
  either side revokes the other's pages from two roles — **with a green suite**, because each
  branch's tests assert only their own capability (`3eede64`, `9f9dbe9`, `086eb96`). The grants are
  now **one union expression: add to it, never replace it.** Do not "tidy" that back into per-branch
  lines.
- **A moved symbol breaks callers that arrived AFTER the move, and git is textually silent.**
  `hash_password` moved to `qorgan.passwords`; six branches merged in between added seven new files
  importing the old path; new files never conflict, so the merge was clean and the suite **collected
  8 tests instead of 1884** (`086eb96`). **No re-export shim was added — it would reinstate the
  import cycle the move exists to remove.** Do not add one next time.
- **A guarantee is only as wide as the surfaces you assert it on.** Redaction lived in a helper only
  `/logs` was tested through; `/` and `/cameras` were never guarded. Reverting one query left **all
  1664 tests passing with the camera password rendered to a browser** (`b0028b2`).
- **Empty results render as good news.** A mistyped page number answered "did my alerts go out?"
  with a confident yes; an unrecognised category emptied the journal the same way (`a00d198`).
  **Report *unavailable*, never *empty*** — and never the configured value wearing a measurement's
  clothes ("не измерено", never the number from the config) (`26f0239`).
- **A number measured once, labelled as current.** The loop measures the delivered rate over its
  first 30 frames and never again — deliberately, it is a check on config, not a health metric. The
  page printed it under «Измерено в потоке», so a camera dropping frames all afternoon showed a
  healthy figure all day (`d6c3286`).
- **A config value's meaning lives in its consumer, not in its file.** `min_score: 0.50` was
  documented with a paragraph on why 0.50 is a measured floor — and every canteen profile quietly
  wrote 0.42–0.45 over it. **The documented value was in force on zero cameras.** The merge is three
  files deep (`base.yaml ← profiles/ ← cameras/`) and a number looks the same whichever supplied it.
  `test_config_provenance` now fails if any of 815 rows cannot say where it came from (`c7ccd76`).
- **Two implementations that are both correct still drift.** A second, *correct* timestamp converter
  kept the "local time appears" test green and only the one-function test caught it (`269c0a7`).
  **Pin reuse of the one function, not correctness of the output.**
- **Units are a security boundary.** bcrypt's maximum is in **bytes** because that is what bcrypt
  reads (40 Cyrillic letters is 80 bytes, and the truncated prefix would silently be a working
  password); the minimum is in **characters** because that is what a guesser guesses (a five-letter
  Russian word is ten bytes). **Normalising them to one unit looks like cleanup and lets weak
  passwords in** (`4cfe24f`).
- **Count what is ACTIVE, never rows.** An admin who cannot log in is not cover for anything; a row
  count sees two doors where there is one. *"That is this project's signature defect and it has been
  paid for three times already"* (`4cfe24f`).
- **Two CSS blocks that look additive are not.** Both pages defined the same selectors with different
  values off a status string; concatenating let the cascade decide, turning an undelivered alert from
  amber to inert grey — **and no test would notice**. Resolved by **scoping, not choosing**
  (`0190e73`).

## 7.2 Techniques worth reusing

- **Sabotage at the wrong DEPTH looks like a pass.** Deleting a filter reddens three tests; *moving
  it late* reddens exactly one — and that one is why it exists (`c2ad87e`). A sabotage that merely
  removes is weaker than one that displaces.
- **To reproduce an inter-test leak, use a fixture finaliser, not a test body.** Poisoning inside the
  test passes, because the test's own teardown tidies up after it; a real daemon does not poll
  politely inside a test body. Request the ghost fixture **before** `settings` so it finalises
  **after** it, and make it assert it really poisoned, so the pair cannot rot into a no-op
  (`d73614e`).
- **When something appears and nobody knows which test did it: a tracer, not a bisect.** A temporary
  `pytest_runtest_protocol` wrapper printing the nodeid the moment the file appeared named the
  culprit immediately — and it named the author's own new test (`19c6189`).
- **A test that asserts a place on disk must hardcode the path**, not read it back from the thing
  under test: reading it back follows the fix around and asserts a throwaway file was not created,
  which is vacuous (`19c6189`). **Do not "clean up" that literal.**
- **Pin the derivation, not the constant.** `1421 tests passed at both 220 and 110` — nothing pinned
  the calibration, only the plumbing. An `== 110` assertion written a day earlier would have passed
  at 220 (`23cf308`).
- **Two attractive fixes were written and disproved BY MEASUREMENT, and are recorded so nobody
  rewrites them:** reordering the settings/engine teardown ("the dangerous state is not *between*
  those two calls, it is *after both*" — and the first reproduction accidentally wrote the reordered
  version and passed, reproducing the fix rather than the bug), and a session-scoped autouse fixture
  for the settings cache, which is already too late because collection froze the LRU value
  (`19c6189`, `c1f77bb`).
- **When a guard test goes red in a merge, REPOINT it — do not adjust its assertion to match
  reality.** `/settings` began returning 403 where a pupils-section guard asserted 404; changing
  404→403 was refused, because a 403 comes from the capability layer and says nothing about the only
  question that test answers (`0b2f4c2`).
- **Verify a permission table by LOADING it, not by reading it** (`0b2f4c2`).

## 7.3 Landmines — things a plausible cleanup would break, several of them silently

This project writes its reasoning into docstrings, and most of it is safe to discover as you go.
These are the ones where **the tidy-up comes first and the discovery comes months later.**

- **`ruff --fix` can silently kill GPU face recognition.** `src/qorgan/gpu.py:84` imports `torch`
  **before** `onnxruntime`, held there by a lone `# noqa: I001`: *"importing torch first loads the
  CUDA runtime DLLs into the process, and on Windows that is what lets onnxruntime find them
  afterwards. Sorting these two lines breaks the GPU."* Mirrored at `planning/measure.py`. **§7 tells
  you to run ruff — so this is the one place to read the diff before accepting it.** `_probe_session()`
  is what notices, and it only notices if someone runs `qorgan doctor`.
- **`roles.py:186` — the grants are one union expression with a comment saying "add to the union;
  never replace it".** Splitting it back into readable per-role lines is how five branches silently
  revoked each other's pages (§7.1).
- **`models/pose.py:26` — `CROP_WIDTH = 320` is the unit of every skeleton threshold.** Raising it
  "for accuracy" retunes all five features at once. §1 forbids tuning thresholds; this is the knob
  that tunes them all without looking like one.
- **`detection/tracking.py:27` — the velocity window is `0.3` SECONDS, not N samples.** The natural
  deque refactor reintroduces exactly the fps-dependence that the whole §2 story exists to have
  killed.
- **`web/csrf.py` — two load-bearing details.** `compare_digest` runs only after both sides are
  proven non-empty, because *"an exception here would fail OPEN through the error handler"*; and the
  middleware **replays** the request body, because `BaseHTTPMiddleware` hands the route a different
  `Request` over the same ASGI channel. Remove the replay and every form 422s — and the Monday fix
  people reach for is deleting the CSRF layer.
- **Middleware order is reversed by Starlette.** `add_middleware(Auth); add_middleware(Csrf);
  add_middleware(Session)` yields runtime order **Session → Csrf → Auth**. Reordering the calls to
  "read better" changes the security order.
- **`web/security.py:120` — `required <= capabilities_for(role)` is SUBSET (all), never any.**
  Flipping it opens `/pupils/{id}/canteen` to canteen staff.
- **`canteen/reports.py` — the two Unknown counts are NOT disjoint and must never be summed**, and
  `did_not_eat` must never render without `caveat()`. The failure mode is telling a school that N
  children went hungry when the number is manufactured.
- **`identity/merge.py:36` — `RecognitionAttempt` is deliberately NOT re-pointed on a merge.**
  Dangling `top1_person_id` values pointing at inactive people are correct: *"A log you edit to match
  a later decision is not a log."* Those rows are the evidence FOR the merge.
- **`evaluation/sampling.py:248` — silence must be PROVEN by the coverage manifest.** A friendly
  `coverage or set()` fallback files an unscanned clip containing a real fight as "silent" — never
  sampled, never labelled. The corpus has exactly one confirmed fight.
- **Three orphan tables** — `app_settings`, `mode_logs`, `meal_windows` — plus
  `canteen_sessions.meal_window_id`, a declared FK no code path populates. Two are documented as
  deliberate (§5); the other two are simply undiscussed. **Do not build a feature on any of them
  without checking which case it is.**

### One found defect, verified, not fixed — a good first task

`src/qorgan/accounts.py:283-285` states: *"`web.routes.events` still formats `occurred_at` without
this conversion, so incident times on that page are shown in UTC … the two do not currently agree."*
**That is no longer true.** `web/routes/events.py:42` imports `local_time` and `:159` applies it; the
fix landed 2026-07-26 (`269c0a7`). The note outlived the defect it describes.

This is the project's signature disease in miniature — a comment asserting what the code does not do
— and it is exactly what `tests/test_supervisor.py` catches for the heartbeat cadence. Three lines,
zero risk. Left for you rather than folded into a documentation change set.

## 8. Reference materials, and the tracked docs

**Outside this repo:**
- **Legacy system: `C:\Users\tokmo\Downloads\Telegram Desktop\qorgan ai\`.** READ-ONLY reference —
  read it for domain knowledge, **never copy its code.** 21 756 Python lines of untested
  multi-threaded CV: `run_canteen()` is one ~6 330-line function, `analyze_aggression` existed in 3
  diverged copies, ~50 endpoints with zero auth, secrets in the repo/DB/logs. Files up to 500 KB —
  grep, never read whole.
- **A second, NOT byte-identical copy at `C:\Users\tokmo\Downloads\qorgan ai\`** (the folder that also
  contains this repo). They share the baseline commit, not the tree. `eval_harness.py` and the live
  `school_ai.db` exist **only** in the Telegram Desktop copy.
- **`C:\Users\tokmo\Downloads\qorgan ai\AUDIT.md`** (70 KB) — the full engineering audit of the
  legacy. Every H-xx/M-xx/L-xx it lists is what v2 was built to make absent. **This is the "why"
  behind almost every design choice.**

**Tracked docs, in reading order:**
1. **`docs/next-session-handoff.md`** (this file) — state, truths, roadmap, discipline. §0, §0a, §0b
   and §0c are dated and **append-only: read all four**, oldest first. A later section corrects an
   earlier one in words rather than by editing it, so the newest dated statement wins.
2. **`docs/next-session-prompt.md`** — the starting prompt, the four unbreakable rules, where every
   branch stands, and the traps. Short, and the only place the working rules are stated as rules.
   **It was missing from this list until 2026-07-30**, which is how a session could read the whole
   handoff and never meet them.
3. **`docs/client-spec-2026-07-16.md`** — the numbered spec §1–§18. **Every `§N` in this repo points
   here.** It is REQUIREMENTS **plus the client's claims about his own system**, several of which are
   false. Do not read a §N as a fact.
4. **`docs/client-request.md`** — the earlier informal three-point ask. Not the numbered spec.
5. **`REWRITE_SPEC.md`** — architecture and rules R1–R10. **If code contradicts it, the code is the
   fact.** Its `§4.3` is its own; it does not use the client's numbering.
6. **`docs/client-note-2026-07-17.md`** — the honest disclosure sent to the client. **STALE IN PART —
   read the dated correction block at its head first.** It is a record of what was sent.
7. **`eval/FINDINGS.md`** — the first detector measurement. Its "the skeleton is disagreeing, not
   abstaining" gloss is contradicted by §1 here. **History, not current truth.**
8. **`docs/questions-for-school.md`** — the blocked-on-the-school list. **§10 exists only on
   `feat/psychologist-cabinet`, not on `main`** — you will not meet it reviewing trunk, and the owner
   has not read it.
9. **`HANDOFF.md`** — engineering handoff. No numbered sections; cite by heading.
10. **`HANDOVER.md`** — the CLIENT's runbook: weights, roster photos, `.env`, the 3.11-only install
    order. §6b (operator-signal mode) is at line 272.
11. **`README.md`** — short; the deviations-from-spec table is at line 122.
12. **`docs/superpowers/`** — dated SDD notes; append-only, do not rewrite dated entries.

### CLI map

`doctor` · `db upgrade|current` · `user add --role` · `config validate` · `plan-workers --force` ·
`pupils import|import-roster|merge|gallery-report|report` · `identity camera-report` ·
`eval scan|sample|label|template|noise-floor|run|gate|save-baseline` ·
`supervisor` · `web` · `backup` · `janitor`.
**Run `qorgan <cmd> --help` rather than trusting this line** — it has been incomplete before.

## 9. The test suite as an instrument

84 test files, ~1064 test functions, 1884 collected items. **The suite polices the project's own
rules, not only its behaviour** — this is the most unusual thing about it and the easiest to break by
accident:

- `test_code_limits.py` — R1, 500/50 lines, over `src/` **and** `tests/`.
- `test_no_secrets.py` — R4, greps config and every tracked text file. Assembles the legacy password
  at runtime so the file is not itself the thing it searches for.
- `test_config_deadkeys.py` — R10, an **owner-resolving AST scan**, not a grep: a name-based scan
  cannot tell a config field from an identically-named attribute elsewhere. Its exemption list is
  **empty and proven earned**.
- `test_web_auth.py` — R5, walks the real route table; a new route is protected unless explicitly
  listed public.
- `test_web_settings.py` — the settings page has no write path, checked at the router **and** in the
  source AST.
- `test_evaluation.py` — R2, asserts the worker and the harness are the **same function object**.
- `test_supervisor.py` — parses a *comment* claiming a heartbeat cadence and fails if it disagrees
  with the literal. **A comment is a claim, so it is checked.**
- `test_engine_isolation.py` / `test_harness_isolation.py` — tests about the **instrument**, not the
  product. Deliberately ordered: the first plays the ghost, the second is the victim.

**Two import-time guards in `conftest.py`, neither a fixture, and both must stay that way:**
`_assert_testing_this_checkout()` and `_point_the_default_database_somewhere_harmless()`. The second
cannot be a fixture — `get_settings()` is LRU-cached and the first call anywhere freezes the defaults,
so a session-scoped autouse fixture is already too late. **That was tried.**

**Logging in requires a CSRF token, including `POST /login`.** Use `tests/web_login.py::with_token`.
Users are created as ORM rows with real bcrypt, never through an endpoint. Pupil photos in tests are a
15-byte lie, deliberately: *"these are photographs of children and the suite has no business opening
one."*

## 10. The live decision: how the thread-leak fix is proven

**The defect.** Every `client_for` fixture opens a `TestClient` against a shared `app`, so a test
wanting two roles runs `lifespan` twice on one app object. The second run overwrote
`app.state.notifier` while the first kept running — and shutdown can only stop what `app.state` still
points at. A leaked `NotificationWorker` polls `session_scope()` forever; `get_engine()` binds the
module-global engine lazily, so a poll at the wrong moment points the global at the wrong database and
**the next test to authenticate dies with `no such table: users`, in a file with nothing to do with
the cause.** It reads as flaky because it is.

**The fix** (`ef0aad4`) is in `lifespan`, not in the eleven fixtures that produce the shape, because
the twelfth written next month would reproduce it.

**The guard** (`55c3ee9`) is an autouse conftest fixture failing any test that leaves one of the
project's seven named worker threads alive, plus `test_thread_guard.py`, which **parses `src/` by AST**
and requires every `Thread(...)` to carry a known name prefix — so a worker kind added next year fails
until the guard learns it. The AST form replaced a regex that found 7 names for 7 constructors and was
still unsafe: **a thread started without `name=` is not reported by such a regex, it is simply not
seen.** Explicitly out of scope, in writing: two `multiprocessing` sites spawn *processes*, which the
guard does not watch.

**The methodological argument, which matters more than this bug.** Twenty repeated full-suite runs
were started as proof and then abandoned, because repetition is **sampling a symptom**: twenty green
runs bound the failure rate only at ~14% (95%), so "20/20" is consistent with a bug that appears once
in seven runs; ~300 runs would be needed for 1%, about 30 hours. And the pre-fix rate was never
measured, so there is nothing to compare against. **The guard checks the cause deterministically, in
one run.** That is why the twenty were stopped.

**What is reported as verified — by a session whose artefacts I could not find on disk, so treat it as
a claim to re-run, not as evidence:** full suite on the guard branch **1894 · 0 · 0 · 0, EXIT=0**, and
a sabotage (removing `_stop_any_previous_workers`) turning it red on
`test_web_settings.py::test_the_nav_never_draws_a_link_the_role_cannot_follow` naming both leaked
threads. **The arithmetic checks out** — 1884 on `main`, +5 for `test_web_lifespan.py`, +5 for
`test_thread_guard.py` = 1894 — which is consistency, not proof.

**So the first job of the next session is:** run the guard branch alone, parse the XML, redo the
sabotage yourself. Then push both branches and let the owner decide the merge. **The owner merges;
you do not.**

## 11. NEW — the client's phase-1 delivery arrived, and it is not what it looks like (2026-07-27)

Two things landed in the repo root, **measured 2026-07-27, nothing imported:**

- **`q.ai/фотки учеников/`** — an unpacked macOS zip with **two parallel trees**. The payload holds
  **141 real photographs** in 13 class folders. Alongside it, `__MACOSX/` holds **141 AppleDouble
  stubs of exactly 178 bytes each**, all named `._<n>.jpg`/`.jpeg`.
- **`q.ai/список.xlsx`** — one sheet `Лист1`, 141 data rows, columns **`ID | Класс | ФИО | Photo
  Name`**. This is the number → pupil mapping that was asked for.

### Three traps, in order of how much they cost

**1. A glob counts 282 photos. There are 141.** The `__MACOSX` stubs carry `.jpg`/`.jpeg`
extensions, so any extension glob doubles the count and half the results look destroyed at 178 bytes.
This is the same class of error as the 19-vs-28 in `bullying_camera/`. **Count in Python, exclude
`__MACOSX` and `._*`.**

**2. The folder names are CP866-mangled on disk**, not merely in the console: the payload directory
and all 13 class directories are UTF-8 bytes stored as CP866 (`'1╨Р'` → `1А`). **An importer that
does not `.encode('cp866').decode('utf-8')` will create garbage class labels.**

**3. Join `список.xlsx` on `ID`, never on `Photo Name`.** The sheet reconciles 141/141 on `ID` and on
`Класс`, but only **138/141** on `Photo Name` — three rows record the wrong extension (`10.jpg` and
`12.jpg` are `.jpeg` on disk; `99.jpeg` is `.jpg`). A filename join silently loses three children.

### What is good

- **141 photos, 141 numeric filenames, 141 unique values, contiguous 1…141, and no number appears in
  two class folders.** Each class occupies one unbroken block. The duplicate-ID failure this project
  fears is **not** present in this delivery.
- **No corruption found.** Zero 0-byte files, zero files under 2 KB, smallest real photo 20 881 B,
  median 132 KB, total 37.6 MB. **The owner's "slightly corrupted" is contradicted** — the impression
  is fully explained by the 178-byte `__MACOSX` stubs. One honest limit: a JPEG that is truncated but
  still large cannot be ruled out by names and sizes, and closing that needs a trailing-bytes check,
  which is not the same as opening an image. Not attempted; **ask before doing it.**

### The finding that decides what happens next

**The new numbers are 1…141. The existing roster's are `student_333`…`student_477`. The two sets are
disjoint — overlap is exactly zero.** (Verified twice, independently.)

The existing importable roster `student_photos/student_photos/` holds **142 photos in 17 folders**:
136 `student_NNN` plus 6 `staff_NNN`, and it carries **no names at all**.

**So importing this delivery as-is raises no error and creates 141 additional identities beside the
existing 136 — for a school of roughly 141 pupils. The failure mode is not a collision, it is a
silent doubling**, and it would be discovered the way the legacy's was: recognition quietly not
working. The obvious "fix" — offsetting the new numbers into the old range — is *precisely* the
one-child-becomes-two bug that `ExternalIdSource` refuses (`ROSTER` only; `GENERATED` was deliberately
removed).

Class composition also diverges in **both** directions (`1А` +4, `2А` +3, `3Б` +3, `5А` +3, but `2Б`
−1, `3А` −1, `6А` −1), and **`11А`, `11Б`, `staff` and `учитель` — 11 photos — have no counterpart in
the delivery at all.** This is not an increment to the existing roster. It is a fresh, self-contained
re-enumeration of the school.

**Whether the underlying children are the same people is not decidable from filenames and sizes.**
The only shared key that could exist is `ФИО`, and the existing roster has no names.

### Therefore, the question for the school — do not decide this yourself

> Which enumeration is the school's real one: the existing `student_333…477`, or the new `1…141`?
> And are the 141 new photos the same children as the existing 136, or a new capture of the roll?
> If they are the same children, we need the `ФИО` → `student_NNN` correspondence, because the two
> number systems share no key.

Add it to `docs/questions-for-school.md`. **Do not import until it is answered.** When it is:
throwaway database first, then `qorgan pupils import-roster`, then `qorgan pupils gallery-report`.

### The old duplicate-ID question is NOT closed — and the check that "closes" it answers a different question

An earlier draft of this section said it was closed, on the grounds that there are 142 photos with
142 distinct numbers and none in two folders. **That is true and it refutes nothing**, because the
defect is not *two files sharing a number* — it is **one face holding two different numbers**. Six
people are twelve distinct IDs, which is exactly what a distinct-number check reports as clean.

`docs/questions-for-school.md:17-22` rebuts that exact argument, pre-emptively and by name, and the
code settles it: `config/identity.py` records *"all 142 photographs embedded with the production
model, the full 138×138 cosine matrix, **the six duplicate enrolments removed** — 9 447 pairs of
DIFFERENT children"*. C(138,2) = 9 453, and 9 453 − 6 = 9 447. **The six pairs are a property of
this roster, measured with the production model — not of the legacy database.** They were re-verified
on the current roster on 2026-07-24 and matched to three decimals.

The only measurement that answers this question is a **face comparison** — `qorgan pupils
gallery-report` — never a filename check. Similarities run 0.774 to 0.999; one pair
(`11-А/470 ↔ staff/334`) crosses the pupil/staff line and **decides whether a child appears in the
meal record at all.**

**How this got into a handoff is the lesson.** It came in as a settled fact from the owner, and I
wrote it down instead of asking what quantity had actually been measured. A claim handed over as
finished still gets checked — especially when the repository already contains a written rebuttal of
it. The new §11 question (which enumeration is real) is genuinely new; **this old one is still open
and still blocking.**

## 12. Where to pick up

**Rewritten 2026-07-30 at consolidation.** The list here was the 2026-07-27 one, and three of its
five items had been finished by §0a and §0c without anybody striking them out — a reader with no
context would have spent the morning re-doing them. The old items are kept below with their
outcomes, because "this was done and here is how I know" is worth more than a deleted line.

**Settled — do not redo these:**

1. ~~Re-run `main` alone and settle whether §0's red is a load flake.~~ **Done.** §0a settled it;
   §0b re-measured `main` twice with two operators at 2347 / 0 / 0 / 0. `main` is green.
2. ~~Verify, push, and offer the merge of `fix/no-thread-outlives-its-test`; two commits exist only
   on this disk.~~ **Done.** `git merge-base --is-ancestor fix/no-thread-outlives-its-test main`
   answers yes — it is in trunk, and its tip `55c3ee9` equals `origin`.
4. ~~Decide the fate of `stash@{0}`.~~ **Done** — deleted, archived as the annotated tag
   `archive/stash-2026-07-14`, tag on `origin`, `git stash list` empty (§0c). **Do not restore it:**
   one of its tests asserts a project decision `main` has since reversed.

**Still standing:**

3. **Do NOT import the client's delivery** (§11). It is measured; it is clean; and its numbering
   shares no key with the existing roster, so importing it would silently double the school. **Send
   the question in §11 to the owner and wait.** Unchanged.
5. Everything else is on-site (§5) or phase 2 (§6), and phase 2 is no longer gated.

**What actually comes next, as of 2026-07-30 — read §0c and `docs/next-session-prompt.md` first:**

- **Nothing merges into `main` without the owner's word.** Branch and push, yes; merge, no.
- **The three pilot modules are three unmerged branches** — `feat/psychologist-cabinet` (`9a59e54`,
  closed), `feat/multi-school` (`ad90e2e`, round 5 of 5 landed but **never re-reviewed**),
  `feat/weapons-detection` (`3aee202`, **unverified WIP** — its own commit message says the suite,
  ruff and sabotage were all skipped). The weapons branch is where work stopped mid-sentence.
- **When these merge, two migrations are both numbered `0008`** and `git` will merge them silently;
  only `alembic upgrade` breaks. Resolve at merge time, never on the branches. Weapons adds no
  migration — that was checked, and an earlier draft of the prompt file said otherwise.
- **Six owner decisions are waiting and none of them is a coding task.** They are collected in §0c
  under "Owner decisions" and in the prompt file. The two with a child-safety edge: the second
  school's live video is refused by a *page*, not a *boundary*; and a supplier's `DEVELOPER` account
  can still declare a pupil armed.
- **`docs/questions-for-school.md` §10 goes to a real school, was written by an agent, and the owner
  has not read it.** It exists only on `feat/psychologist-cabinet`. Read it before it is sent.

**Not code, and still the real next step:** the school connecting the system to real cameras, which
produces the labels and the placement fixes that the detector and the canteen both need.

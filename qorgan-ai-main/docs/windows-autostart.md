# Windows autostart, and the database backup

For **the client's engineer**, on the school's machine. Read `HANDOVER.md` first — this
assumes the system already runs when you start it by hand.

Two processes must come back on their own after a reboot:

| | what it does | if it is not running |
|---|---|---|
| `qorgan supervisor` | the worker fleet — every camera | **nothing is watched.** No events, no canteen records. |
| `qorgan web` | the dashboard **and the Telegram queue** | no dashboard, and **no alerts are delivered** — they queue up as rows and sit there |

They are independent: neither needs the other to start, and the fleet keeps watching
whether or not the dashboard is up.

---

## 1. Install both tasks

From an **Administrator** command prompt, in the install root:

```bat
deploy\install-autostart.bat
```

It prints the install root and the account it will use, checks the virtualenv exists,
warns if `.env` is missing, and registers both tasks. **Windows prompts for your password
once per task** — that is Windows storing it in Credential Manager so the tasks can run
when nobody is logged in. It is never written to a file by us, and never passed on a
command line.

Administrator is required, and only for this: a task with a **boot trigger** cannot be
registered by a normal user. (Measured — everything else in these definitions registers
unelevated.)

Then, without rebooting:

```bat
schtasks /run /tn "Qorgan AI\qorgan-supervisor"
schtasks /run /tn "Qorgan AI\qorgan-web"

schtasks /query /tn "Qorgan AI\qorgan-supervisor" /v /fo list
```

To remove them:

```bat
schtasks /delete /tn "Qorgan AI\qorgan-supervisor" /f
schtasks /delete /tn "Qorgan AI\qorgan-web" /f
```

---

## 2. What is in `deploy\`, and why

| file | |
|---|---|
| `qorgan-supervisor.bat`, `qorgan-web.bat` | the launchers. **They `cd` to the install root before doing anything.** |
| `qorgan-supervisor.xml`, `qorgan-web.xml` | the task definitions, with `__QORGAN_ROOT__` / `__QORGAN_USER__` placeholders |
| `prepare-task-xml.ps1` | fills the placeholders in and converts to the UTF-16 `schtasks` demands |
| `install-autostart.bat` | runs the above, then `schtasks /create` |

### The working directory is the whole point

**Task Scheduler starts a process with no working directory of its own.** Three things
resolve against it, and every one of them fails *silently*:

- **`.env`** is read from the install root (`settings.py: ENV_FILE`). Not found ⇒ the
  system starts on the **published dev `SECRET_KEY`**, a **blank RTSP password** and **no
  Telegram** — and logs nothing to say so.
- **`yolov8n.pt` / `yolov8n-pose.pt`** are named as bare filenames in `config/base.yaml`,
  which Ultralytics resolves against the working directory — and **downloads from the
  internet** into it if they are absent.
- `data/`, `media/`, `logs/`, `config/` are all relative by default.

So the launchers anchor to their own location (`cd /d "%~dp0.."`) rather than trusting the
task definition. The XML sets `WorkingDirectory` as well; the `.bat` means it does not
matter if somebody edits that and gets it wrong. This is covered by tests that **run** the
launchers from a deliberately wrong directory (`tests/test_autostart.py`).

### Four Task Scheduler defaults that are wrong for this system

Each is set explicitly in both XMLs, and each is checked by a test:

| setting | default | why the default is wrong |
|---|---|---|
| `ExecutionTimeLimit` | **`PT72H`** | Task Scheduler would **kill both processes after three days**. A task Windows stopped has not crashed, so nothing restarts it: the cameras go dark every third day, silently. We set `PT0S` (no limit). |
| `MultipleInstancesPolicy` | queue/parallel | two supervisors on one GPU = two processes per camera and a memory plan measured for one. We set `IgnoreNew`. |
| `StopIfGoingOnBatteries`, `RunOnlyIfIdle`, `StopOnIdleEnd` | stop it | Windows stops the fleet because the machine "looks idle". Nobody is at the keyboard of a camera server — that is what it is for. |
| `RunLevel` | — | We set `LeastPrivilege`. These processes need a GPU, some files and a socket above port 1024; they hold live video of children and have no business running elevated. |

### It runs as a **user**, not as LOCAL SYSTEM

`S-1-5-18` is the obvious choice for a service and it is wrong here: **InsightFace loads
`buffalo_l` from `~/.insightface`**. As SYSTEM that is
`C:\Windows\System32\config\systemprofile\.insightface`, where the school's 341 MB pack is
not — so face recognition would re-download it to a folder nobody knows about, or fail
outright on a machine with no internet (`HANDOVER.md` §0).

### If you edit the XML by hand

`schtasks /create /xml` **rejects a UTF-8 file** — with or without a BOM — as:

```
ERROR: The task XML is malformed. (1,2)
```

which points at the XML declaration and says nothing about encoding. It wants **UTF-16**.
The templates are UTF-8 in the repo so they can be read and diffed; `prepare-task-xml.ps1`
converts on the way out. The definitions also declare `version="1.4"` — `1.2` is rejected
with "The task XML contains an unexpected node". Both measured on Windows 11, against
these files.

---

## 3. What autostart does **not** cover

- **It has never been run on the school's machine.** The task definitions were validated
  against a real Windows 11 Task Scheduler — it parsed them and stored every setting back
  unchanged — but the first registration on site is still a first. Budget ten minutes.
- **A task that starts is not a system that works.** If `.env` is missing or the NVR
  credentials are wrong, both tasks start, exit 0 or hang, and Task Scheduler reports
  success. **After installing, open the dashboard and confirm you see live previews.**
- Neither task waits for the network. There is a 30-second boot delay; the supervisor
  reconnects on its own past that.

---

## 4. Back up the database — also on a schedule

```bat
qorgan backup
```

Writes `<database dir>\backups\qorgan-<date>-<time>.sqlite3` — by default
`data\backups\`. It uses SQLite's `VACUUM INTO`, which means:

- **it is safe while everything is running** — no need to stop the workers;
- **it includes the WAL.** The database runs in WAL mode, so the newest committed events
  are in `qorgan.sqlite3-wal`, *not* in `qorgan.sqlite3`. **A scheduled `xcopy` of the
  database file — the obvious thing — silently restores a database missing exactly the
  most recent incidents, and looks successful doing it.**
- the copy is opened and integrity-checked before the command reports success.

It refuses to overwrite an existing file, so a run cannot destroy yesterday's.

Put it on a schedule next to the retention sweep:

```bat
schtasks /create /tn "Qorgan AI\qorgan-backup"  /tr "\"<ROOT>\.venv\Scripts\python.exe\" -m qorgan backup"                  /sc daily /st 03:00 /ru "%USERDOMAIN%\%USERNAME%"
schtasks /create /tn "Qorgan AI\qorgan-janitor" /tr "\"<ROOT>\.venv\Scripts\python.exe\" -m qorgan janitor --media-days 90" /sc daily /st 03:30 /ru "%USERDOMAIN%\%USERNAME%"
```

Replace `<ROOT>` with the install root. These two need no boot trigger, so they do not
need an Administrator prompt.

> **`data\backups\` is on the same disk as the database, which is not a backup of that
> disk.** Copy it off the machine — that part is a decision for the school, not a default
> we can ship. And **a backup is a copy of children's data**: it belongs wherever the
> roster photographs are allowed to be, and nowhere else.

The dashboard shows this same folder at **`/backups`** — what exists, when, how big — and
has a button that runs exactly this command on a background thread. It is there so that
somebody at the school can answer "did last night's backup run?" without a terminal: a
scheduled task nobody can see is a scheduled task that has been failing since March.
Looking and pressing are separate permissions (`view_backups`, `create_backup`; both held
by admin and developer, neither by an operator or canteen staff), and **the page does not
offer the file for download** — `src/qorgan/web/routes/backups.py` says why not.

To restore: stop both tasks, move the damaged `qorgan.sqlite3` (and its `-wal`/`-shm`
files) aside, copy the backup into place under that name, start the tasks. Then run
`qorgan db current` — a backup carries its own schema version, and a restore of an old
file may need `qorgan db upgrade`.

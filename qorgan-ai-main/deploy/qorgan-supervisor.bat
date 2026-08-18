@echo off
rem ===========================================================================
rem  Qorgan AI - the worker fleet. Started by Task Scheduler at boot.
rem
rem  THE FIRST EXECUTABLE LINE IS `cd /d "%~dp0.."` AND IT IS THE WHOLE POINT.
rem
rem  Task Scheduler starts a process with no working directory of its own. Three
rem  things in this system resolve against the working directory, and all three
rem  fail SILENTLY when it is wrong:
rem
rem    * .env  - read from the install root (settings.py: ENV_FILE). Not found
rem              means the dev SECRET_KEY, a blank RTSP password and no Telegram,
rem              with nothing logged to say so.
rem    * yolov8n.pt / yolov8n-pose.pt - config/base.yaml names them as bare
rem              filenames, which Ultralytics resolves against the CWD. Not found
rem              means it DOWNLOADS them from the internet into whatever directory
rem              the task happened to start in.
rem    * data/, media/, logs/, config/ - all relative by default.
rem
rem  %~dp0 is this file's own directory, so `%~dp0..` is the install root no
rem  matter who started this, from where, or with what. It does not depend on the
rem  task definition being right; the XML sets WorkingDirectory as well, and this
rem  line means it does not matter if somebody forgets.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

echo [qorgan] supervisor starting in "%CD%"

if not exist ".venv\Scripts\python.exe" (
    rem Loud, not cryptic. This is the failure an engineer will actually hit, and
    rem it is invisible in Task Scheduler's own UI: the task simply "completed".
    echo [qorgan] ERROR: no virtualenv found at "%CD%\.venv" >&2
    echo [qorgan] Install it first - see HANDOVER.md section 1. >&2
    exit /b 1
)

rem -m qorgan, not the qorgan.exe shim: one less thing to be missing or stale.
".venv\Scripts\python.exe" -m qorgan supervisor
exit /b %ERRORLEVEL%

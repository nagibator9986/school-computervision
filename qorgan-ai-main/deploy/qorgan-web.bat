@echo off
rem ===========================================================================
rem  Qorgan AI - the dashboard. Started by Task Scheduler at boot.
rem
rem  See qorgan-supervisor.bat for why the `cd /d "%~dp0.."` below is the whole
rem  point: Task Scheduler sets no working directory, and .env, the YOLO weights
rem  and every default path resolve against it -- each failing silently.
rem
rem  This process and the supervisor are independent and neither needs the other
rem  to start. That is deliberate: which cameras get analysed is decided by
rem  config/workers.yaml, never by whether the dashboard is up.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

echo [qorgan] web starting in "%CD%"

if not exist ".venv\Scripts\python.exe" (
    echo [qorgan] ERROR: no virtualenv found at "%CD%\.venv" >&2
    echo [qorgan] Install it first - see HANDOVER.md section 1. >&2
    exit /b 1
)

".venv\Scripts\python.exe" -m qorgan web
exit /b %ERRORLEVEL%

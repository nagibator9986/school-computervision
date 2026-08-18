@echo off
rem ===========================================================================
rem  Register both Qorgan AI tasks with Windows Task Scheduler.
rem
rem  Run this ONCE, from an ADMINISTRATOR command prompt, on the school's machine:
rem
rem      deploy\install-autostart.bat
rem
rem  It fills in this machine's real install root and user account, then hands the
rem  task definitions to schtasks. You will be prompted for your Windows password
rem  once per task -- that is Windows storing it in Credential Manager so the tasks
rem  can run when nobody is logged in. It is never written to disk by us.
rem
rem  Why an installer rather than "edit the XML and import it": the XML must name
rem  an absolute path, this repo does not know where it will live, and a task whose
rem  Command points at a directory called __QORGAN_ROOT__ is accepted by Task
rem  Scheduler without complaint and never runs. Nobody should have to notice that.
rem ===========================================================================
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
rem Collapse the trailing "\deploy\.." into a real absolute path: it ends up in a
rem task definition, and "C:\qorgan\deploy\..\deploy\x.bat" is not a good look in
rem an error message the school will be reading at 8am.
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "TASK_USER=%USERDOMAIN%\%USERNAME%"
set "TEMP_XML=%TEMP%\qorgan-task-%RANDOM%.xml"

echo.
echo   install root : %ROOT%
echo   run as user  : %TASK_USER%
echo.

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo ERROR: no virtualenv at "%ROOT%\.venv" - install first, see HANDOVER.md section 1. >&2
    exit /b 1
)
if not exist "%ROOT%\.env" (
    rem Not fatal, but it is the single most common way this system comes up wrong:
    rem no .env means the published dev SECRET_KEY, a blank RTSP password and no
    rem Telegram, and nothing in the log says so.
    echo WARNING: no .env at "%ROOT%\.env" - the tasks will start on defaults. >&2
    echo          See HANDOVER.md section 2. >&2
    echo.
)

call :register "Qorgan AI\qorgan-supervisor" "%~dp0qorgan-supervisor.xml" || exit /b 1
call :register "Qorgan AI\qorgan-web" "%~dp0qorgan-web.xml" || exit /b 1

echo.
echo Done. Both tasks start at boot. To start them now without rebooting:
echo     schtasks /run /tn "Qorgan AI\qorgan-supervisor"
echo     schtasks /run /tn "Qorgan AI\qorgan-web"
echo.
echo Check they are running:  schtasks /query /tn "Qorgan AI\qorgan-supervisor" /v /fo list
exit /b 0

rem ---------------------------------------------------------------------------
:register
set "TASK_NAME=%~1"
set "SOURCE_XML=%~2"

rem Substitute the placeholders and convert to the UTF-16 that schtasks demands.
rem See prepare-task-xml.ps1 -- both of those are measured requirements, and a UTF-8
rem definition is rejected as "malformed" with an error that explains nothing.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0prepare-task-xml.ps1" ^
  -Source "%SOURCE_XML%" -Destination "%TEMP_XML%" -Root "%ROOT%" -User "%TASK_USER%"
if errorlevel 1 (
    echo ERROR: could not prepare the task definition for %TASK_NAME%. >&2
    exit /b 1
)

echo Registering "%TASK_NAME%" -- Windows will now ask for your password.
rem /rp is deliberately NOT passed: a password on a command line is visible in the
rem process list and stays in the console history. schtasks prompts instead.
schtasks /create /tn "%TASK_NAME%" /xml "%TEMP_XML%" /ru "%TASK_USER%" /f
set "RC=%ERRORLEVEL%"

del "%TEMP_XML%" 2>nul

if not "%RC%"=="0" (
    echo ERROR: schtasks refused to create %TASK_NAME% ^(exit %RC%^). >&2
    echo        Are you running this from an Administrator prompt? >&2
    exit /b 1
)
exit /b 0

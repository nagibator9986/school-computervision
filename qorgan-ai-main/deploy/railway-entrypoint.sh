#!/usr/bin/env sh
# What happens between «Railway started the container» and «the dashboard answers».
#
# Four things, in this order, each of which fails LOUDLY rather than starting a dashboard
# that looks alive and is not:
#
#   1. bind the platform's $PORT and $HOST,
#   2. put the writable state on the mounted volume,
#   3. bring the schema to head,
#   4. optionally create the FIRST account, once, and only when asked.

set -eu

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
# 1. The port. Railway assigns one per deploy and passes it as $PORT; the application
#    reads WEB_PORT (pydantic-settings, `Settings.web_port`). Bridged here rather than in
#    settings.py so the only code that knows about a hosting platform is this file.
# ---------------------------------------------------------------------------
export WEB_PORT="${PORT:-${WEB_PORT:-8000}}"
# 127.0.0.1 is the application's default ON PURPOSE — exposing a wall of children's faces
# to a network is meant to be an explicit act. Inside a container the only thing that can
# reach it is the platform's own proxy, and binding to loopback would make it unreachable.
export WEB_HOST="${WEB_HOST:-0.0.0.0}"

# ---------------------------------------------------------------------------
# 2. State. A container filesystem is discarded on every deploy, so the database, the
#    media and the logs must live on the volume. Mount it at $QORGAN_STATE_DIR.
# ---------------------------------------------------------------------------
STATE="${QORGAN_STATE_DIR:-/state}"
mkdir -p "$STATE/data" "$STATE/media" "$STATE/logs"
export DATABASE_URL="${DATABASE_URL:-sqlite+pysqlite:///$STATE/data/qorgan.sqlite3}"
export MEDIA_ROOT="${MEDIA_ROOT:-$STATE/media}"
export LOG_DIR="${LOG_DIR:-$STATE/logs}"

case "$DATABASE_URL" in
  sqlite*)
    if [ ! -d "$STATE" ] || [ ! -w "$STATE" ]; then
      log "ОТКАЗ: $STATE не смонтирован или недоступен на запись."
      log "       База SQLite на файловой системе контейнера исчезнет при следующем"
      log "       развёртывании вместе со всеми учётными записями и разборами."
      log "       Подключите Volume на $STATE (Railway → Service → Data → Add Volume)."
      exit 1
    fi
    ;;
esac

# ---------------------------------------------------------------------------
# 3. Schema. Runs on every boot: `alembic upgrade head` is a no-op when there is nothing
#    to do, and the alternative is remembering to run it by hand exactly once.
# ---------------------------------------------------------------------------
log "миграции: alembic upgrade head"
alembic -c /app/alembic.ini upgrade head

# ---------------------------------------------------------------------------
# 4. The first account. Deliberately NOT automatic: an account created from environment
#    variables on every boot is a permanent back door. This runs only when both variables
#    are present AND the table is empty, prints what it did, and tells the operator to
#    remove the variables afterwards.
# ---------------------------------------------------------------------------
if [ -n "${BOOTSTRAP_ADMIN_USER:-}" ] && [ -n "${BOOTSTRAP_ADMIN_PASSWORD:-}" ]; then
  python - <<'PY'
import os
from sqlalchemy import func, select

from qorgan.accounts import create_account
from qorgan.db.engine import session_scope
from qorgan.db.models.auth import User
from qorgan.enums import UserRole

with session_scope() as session:
    existing = session.scalar(select(func.count(User.id))) or 0

if existing:
    print(f"первичная учётная запись пропущена: в базе уже {existing} учётных записей")
else:
    name = os.environ["BOOTSTRAP_ADMIN_USER"]
    create_account(name, os.environ["BOOTSTRAP_ADMIN_PASSWORD"], UserRole.ADMIN)
    print(f"создана первая учётная запись: {name} (admin)")
    print("ТЕПЕРЬ УБЕРИТЕ BOOTSTRAP_ADMIN_USER и BOOTSTRAP_ADMIN_PASSWORD из переменных")
    print("окружения: пока они там, пароль администратора лежит в настройках проекта.")
PY
fi

case "${1:-web}" in
  web)
    log "запуск панели на ${WEB_HOST}:${WEB_PORT} (QORGAN_ENV=${QORGAN_ENV:-dev})"
    exec qorgan web
    ;;
  *)
    # Any other CLI verb, so `railway run` can reach the same code the dashboard runs:
    #   railway run qorgan classvision status
    exec qorgan "$@"
    ;;
esac

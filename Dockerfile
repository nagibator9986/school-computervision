# The DASHBOARD image. Not the workers.
#
# What this container runs is the FastAPI dashboard and the CLI beside it. It does not
# capture video, does not run a detector, and has no GPU: `pyproject.toml [ai]` — torch,
# ultralytics, insightface — is deliberately absent, and so is `opencv-python` (see
# `deploy/requirements-web.txt` for why the web no longer needs it).
#
# Python 3.11 because `qorgan-ai-main/pyproject.toml` declares
# `requires-python = ">=3.11,<3.12"`. The development venv on the author's machine is 3.13
# and the suite passes on both; the container follows what the project PROMISES rather than
# what one laptop happens to have.
#
# **THE BUILD CONTEXT IS THE REPOSITORY ROOT, and every COPY below is written from there.**
# This repository holds two packages — `qorgan-ai-main/` (this) and `classvision/` (the
# offline analyser) — so a Dockerfile living inside one of them would be invisible to a
# platform that builds from the root, which is exactly what happened: Railway's autodetector
# saw `classvision/`, `qorgan-ai-main/`, a README and a .gitignore, and refused. Keeping the
# file here means the repository deploys with no per-service path settings to remember.
# `classvision/` never enters the image: the analyser runs offline, on a machine with a GPU.

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source, and
# a code edit should not reinstall SQLAlchemy.
COPY qorgan-ai-main/deploy/requirements-web.txt ./deploy/requirements-web.txt
RUN pip install -r deploy/requirements-web.txt

# The application. Installed with --no-deps: the line above is the complete and audited
# runtime, and letting pip re-resolve `pyproject.toml` here would quietly pull opencv back.
COPY qorgan-ai-main/pyproject.toml qorgan-ai-main/README.md ./
COPY qorgan-ai-main/src ./src
COPY qorgan-ai-main/migrations ./migrations
COPY qorgan-ai-main/alembic.ini ./alembic.ini
COPY qorgan-ai-main/config ./config
RUN pip install --no-deps -e .

COPY qorgan-ai-main/deploy/railway-entrypoint.sh /usr/local/bin/qorgan-entrypoint
RUN chmod +x /usr/local/bin/qorgan-entrypoint

# Writable state lives on a mounted volume, never in the image layer: a container
# filesystem is thrown away on every deploy, and this directory holds the database and
# the school's media.
ENV QORGAN_STATE_DIR=/state
RUN mkdir -p /state && chown -R nobody:nogroup /state

# **THE CONTAINER STARTS AS ROOT AND DOES NOT STAY ROOT.** There is no `USER` line here,
# and that is deliberate rather than an omission: Railway mounts the volume OWNED BY ROOT,
# over the top of the directory prepared above, so a process that was already `nobody`
# could not create a single directory inside it —
#
#     mkdir: cannot create directory '/state/data': Permission denied
#
# and the service restarted forever. The entrypoint therefore takes ownership of the mount
# and then drops to `nobody` with `setpriv` before anything else runs. The Python process
# is still unprivileged; only the four lines that prepare the mount are not.
#
# The alternative — leaving the whole dashboard running as root — would have been one word
# shorter and is not on offer: this process serves photographs of children.

# Документирующая пометка, и только: uvicorn слушает $PORT, который выдаёт платформа
# (Railway даёт 8080), а точка входа переносит его в WEB_PORT. Значение 8000 здесь
# расходилось с тем, что видит оператор в списке портов, и читалось как противоречие.
EXPOSE 8080
ENTRYPOINT ["qorgan-entrypoint"]
CMD ["web"]

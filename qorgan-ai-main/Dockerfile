# The DASHBOARD image. Not the workers.
#
# What this container runs is the FastAPI dashboard and the CLI beside it. It does not
# capture video, does not run a detector, and has no GPU: `pyproject.toml [ai]` — torch,
# ultralytics, insightface — is deliberately absent, and so is `opencv-python` (see
# `deploy/requirements-web.txt` for why the web no longer needs it).
#
# Python 3.11 because `pyproject.toml` declares `requires-python = ">=3.11,<3.12"`. The
# development venv on the author's machine is 3.13 and the suite passes on both; the
# container follows what the project PROMISES rather than what one laptop happens to have.

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source, and
# a code edit should not reinstall SQLAlchemy.
COPY deploy/requirements-web.txt ./deploy/requirements-web.txt
RUN pip install -r deploy/requirements-web.txt

# The application. Installed with --no-deps: the line above is the complete and audited
# runtime, and letting pip re-resolve `pyproject.toml` here would quietly pull opencv back.
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY config ./config
RUN pip install --no-deps -e .

COPY deploy/railway-entrypoint.sh /usr/local/bin/qorgan-entrypoint
RUN chmod +x /usr/local/bin/qorgan-entrypoint

# Writable state lives on a mounted volume, never in the image layer: a container
# filesystem is thrown away on every deploy, and this directory holds the database and
# the school's media.
ENV QORGAN_STATE_DIR=/state
RUN mkdir -p /state && chown -R nobody:nogroup /state

# Not root. The dashboard serves children's photographs; a process that does not need to
# write outside its state directory should not be able to.
USER nobody

EXPOSE 8000
ENTRYPOINT ["qorgan-entrypoint"]
CMD ["web"]

"""Alembic environment.

The database URL comes from DATABASE_URL, never from alembic.ini — a connection
string is a secret-shaped thing and does not belong in the repo.

Schema changes happen here and only here. The legacy ran ALTER TABLE statements on
every boot from application code, and ran a data-mutating UPDATE alongside them that
re-derived person_type for every pupil and reverted manual corrections (audit H-02).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import pool

from qorgan.db.engine import build_engine
from qorgan.db.models import Base
from qorgan.settings import get_settings

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _suspend_foreign_keys(connection) -> None:
    """Turn SQLite's foreign key enforcement OFF for the duration of the migration.

    **Without this, a batch migration that touches `cameras` DELETES THE SCHOOL'S
    EVENTS**, and does it quietly. SQLite cannot ALTER a column, so `render_as_batch`
    rebuilds the table: create a copy, move the rows, **DROP the original**, rename. That
    DROP is a delete of every row in `cameras` as far as the foreign keys are concerned,
    and `events`, `canteen_sessions` and `recognition_attempts` all reference it with
    `ondelete="CASCADE"`. The rebuilt table comes back with all its rows; the incident
    history does not. Nothing fails, and nothing says so.

    It has to be done HERE, on a fresh connection, before `begin_transaction()`. **Inside
    a transaction `PRAGMA foreign_keys` is a documented no-op**, so a migration that sets
    it itself would appear to be careful and protect nothing -- measured, on this
    schema: the events row was gone either way. It is issued on the raw DBAPI connection
    for the same reason, to stay out of SQLAlchemy's transaction bookkeeping.

    Nothing re-enables it. `build_engine`'s connect listener sets `foreign_keys=ON` on
    every connection it opens, and this one is closed and disposed when the migration
    finishes -- so the suspension cannot outlive the migration, and the application can
    never get a connection that inherited it.

    The trade: a migration can now insert a row that violates a foreign key without being
    stopped. That is the standard price for batch mode on SQLite, and it is much the
    smaller risk -- a bad row is visible and fixable, a silently emptied `events` table is
    discovered months later by a school asking where its incidents went.
    """
    if connection.dialect.name != "sqlite":
        return

    raw = connection.connection.driver_connection
    raw.execute("PRAGMA foreign_keys=OFF")
    still_on = raw.execute("PRAGMA foreign_keys").fetchone()[0]
    if still_on:
        raise RuntimeError(
            "refusing to migrate: PRAGMA foreign_keys=OFF did not take effect, so a "
            "batch table rebuild would cascade-delete every row that references the "
            "rebuilt table. This usually means a transaction was already open on this "
            "connection, where the pragma is a documented no-op."
        )


def run_migrations_online() -> None:
    engine = build_engine(_url())
    with engine.connect() as connection:
        # Before configure() and before any transaction. See the docstring: later is the
        # same as never, and it fails silently by deleting rows.
        _suspend_foreign_keys(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER a column; batch mode rebuilds the table instead.
            render_as_batch=True,
            compare_type=True,
            poolclass=pool.NullPool,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

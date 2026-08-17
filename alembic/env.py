from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from alembic import context
from mormi_api.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def synchronous_url(raw_url: str) -> str:
    url = make_url(raw_url)

    if url.drivername == "postgresql+asyncpg":
        url = url.set(drivername="postgresql+psycopg")

        query = dict(url.query)

        if "ssl" in query:
            query["sslmode"] = query.pop("ssl")

        url = url.set(query=query)

    elif url.drivername == "sqlite+aiosqlite":
        url = url.set(drivername="sqlite")

    return url.render_as_string(hide_password=False)


def configured_url() -> str:
    raw_url = os.getenv("MORMI_DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    return synchronous_url(raw_url)


def run_migrations_offline() -> None:
    context.configure(
        url=configured_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = configured_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

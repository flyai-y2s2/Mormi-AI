"""Apply additive AI database migrations without discarding legacy rows."""

from __future__ import annotations

import os
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url

from alembic import command
from mormi_api.db import Base
from mormi_api.settings import Settings


def synchronous_url(raw_url: str) -> str:
    url = make_url(raw_url)
    drivers = {
        "postgresql+asyncpg": "postgresql+psycopg",
        "sqlite+aiosqlite": "sqlite",
    }
    if url.drivername in drivers:
        url = url.set(drivername=drivers[url.drivername])
    return url.render_as_string(hide_password=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(root / "alembic.ini")
    settings = Settings()
    raw_url = os.getenv("MORMI_DATABASE_URL", settings.database_url)
    sync_url = synchronous_url(raw_url)
    config.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))

    engine = create_engine(sync_url)
    try:
        existing = set(inspect(engine).get_table_names())
        if "conversations" not in existing:
            # A brand-new database has no legacy rows to preserve.
            Base.metadata.create_all(engine)
            command.stamp(config, "head")
        else:
            # Existing pilot databases keep every row. The revision adds only
            # new tables and records the applied head in alembic_version.
            command.upgrade(config, "head")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

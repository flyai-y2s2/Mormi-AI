"""Additive database migration helpers shared by operations and tests."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url

from alembic import command

from .db import Base

OBSERVATION_TABLES = {
    "dialogue_turn_observations",
    "dialogue_claims",
    "dialogue_task_outcomes",
    "note_evidence_links",
    "ai_outbox_events",
}


def synchronous_url(raw_url: str) -> str:
    url = make_url(raw_url)
    drivers = {
        "postgresql+asyncpg": "postgresql+psycopg",
        "sqlite+aiosqlite": "sqlite",
    }
    if url.drivername in drivers:
        url = url.set(drivername=drivers[url.drivername])
    return url.render_as_string(hide_password=False)


def apply_database_migrations(raw_url: str, root: Path) -> None:
    """Apply or safely reconcile the v1 additive observation schema."""

    config = Config(root / "alembic.ini")
    sync_url = synchronous_url(raw_url)
    config.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))

    engine = create_engine(sync_url)
    try:
        existing = set(inspect(engine).get_table_names())
        if "conversations" not in existing:
            # A brand-new database has no legacy rows to preserve.
            Base.metadata.create_all(engine)
            command.stamp(config, "head")
        elif "alembic_version" in existing or not (existing & OBSERVATION_TABLES):
            # Existing pilot databases keep every row. The revision adds only
            # new tables and records the applied head in alembic_version.
            command.upgrade(config, "head")
        elif OBSERVATION_TABLES.issubset(existing):
            # Older deployments call Base.metadata.create_all() during app
            # startup. If that happened before this operational command, the
            # complete v1 schema already exists but Alembic has no version
            # row. Record the matching head instead of trying to recreate the
            # same tables.
            command.stamp(config, "head")
        else:
            partial = sorted(existing & OBSERVATION_TABLES)
            missing = sorted(OBSERVATION_TABLES - existing)
            raise RuntimeError(
                "Partial observation schema detected; refusing to guess. "
                f"present={partial}, missing={missing}"
            )
    finally:
        engine.dispose()

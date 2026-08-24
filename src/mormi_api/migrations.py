"""Additive database migration helpers shared by operations and tests."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import UniqueConstraint, create_engine, inspect
from sqlalchemy.engine import Connection, Engine, make_url

from alembic import command

from .db import Base

OBSERVATION_TABLES = {
    "dialogue_turn_observations",
    "dialogue_claims",
    "dialogue_task_outcomes",
    "note_evidence_links",
    "ai_outbox_events",
}

APPLICATION_TABLES = OBSERVATION_TABLES | {"ladder_analysis_jobs"}


def _observation_schema_issues(bind: Connection | Engine) -> list[str]:
    """Return structural differences that make the observation schema unsafe.

    Table names alone are not enough: an interrupted/manual deployment can
    leave a table present while a required column, FK, unique constraint or
    index is missing. Stamping that database as Alembic head would make later
    upgrades skip the repair and defer the failure to a child's live turn.
    """

    inspector = inspect(bind)
    existing = set(inspector.get_table_names())
    issues: list[str] = []
    for table_name in sorted(APPLICATION_TABLES):
        if table_name not in existing:
            issues.append(f"{table_name}:missing_table")
            continue
        expected = Base.metadata.tables[table_name]
        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        for column_name in sorted({column.name for column in expected.columns} - actual_columns):
            issues.append(f"{table_name}:missing_column:{column_name}")

        actual_foreign_keys = {
            (
                tuple(key.get("constrained_columns") or ()),
                str(key.get("referred_table")),
                tuple(key.get("referred_columns") or ()),
            )
            for key in inspector.get_foreign_keys(table_name)
        }
        expected_foreign_keys = {
            (
                tuple(element.parent.name for element in constraint.elements),
                next(iter(constraint.elements)).column.table.name,
                tuple(element.column.name for element in constraint.elements),
            )
            for constraint in expected.foreign_key_constraints
        }
        for columns, referred_table, referred_columns in sorted(
            expected_foreign_keys - actual_foreign_keys
        ):
            issues.append(
                f"{table_name}:missing_fk:{','.join(columns)}->"
                f"{referred_table}({','.join(referred_columns)})"
            )

        actual_unique_names = {
            str(constraint["name"])
            for constraint in inspector.get_unique_constraints(table_name)
            if constraint.get("name")
        }
        expected_unique_names = {
            str(constraint.name)
            for constraint in expected.constraints
            if isinstance(constraint, UniqueConstraint) and constraint.name
        }
        for name in sorted(expected_unique_names - actual_unique_names):
            issues.append(f"{table_name}:missing_unique:{name}")

        actual_index_names = {
            str(index["name"])
            for index in inspector.get_indexes(table_name)
            if index.get("name")
        }
        expected_index_names = {str(index.name) for index in expected.indexes if index.name}
        for name in sorted(expected_index_names - actual_index_names):
            issues.append(f"{table_name}:missing_index:{name}")
    return issues


def require_observation_schema(bind: Connection | Engine) -> None:
    issues = _observation_schema_issues(bind)
    if issues:
        raise RuntimeError(
            "Observation schema does not match the application contract; "
            "refusing to stamp or start from a silently partial migration. "
            f"issues={issues}"
        )


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
            require_observation_schema(engine)
            command.stamp(config, "head")
        elif "alembic_version" in existing or not (existing & OBSERVATION_TABLES):
            # Existing pilot databases keep every row. The revision adds only
            # new tables and records the applied head in alembic_version.
            command.upgrade(config, "head")
            require_observation_schema(engine)
        elif OBSERVATION_TABLES.issubset(existing):
            # Older deployments call Base.metadata.create_all() during app
            # startup. If that happened before this operational command, the
            # complete v1 schema already exists but Alembic has no version
            # row. Record the matching head instead of trying to recreate the
            # same tables.
            require_observation_schema(engine)
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

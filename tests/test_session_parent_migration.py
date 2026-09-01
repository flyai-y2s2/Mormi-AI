from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from test_observation_migration import _alembic_config

from alembic import command
from mormi_api.migrations import (
    apply_database_migrations,
    require_observation_schema,
    require_session_parent_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parent_migration_is_additive_and_old_reader_works(tmp_path: Path) -> None:
    path = tmp_path / "parent-migration.db"
    sync = f"sqlite:///{path}"
    url = f"sqlite+aiosqlite:///{path}"
    apply_database_migrations(url, ROOT, target_revision="20260826_06")
    engine = create_engine(sync)
    # Fresh ORM schema may contain optional tables. Exercise a real 06 DB.
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_session_parents")
    before = {
        table: [c["name"] for c in inspect(engine).get_columns(table)]
        for table in inspect(engine).get_table_names()
    }
    require_observation_schema(engine)
    with pytest.raises(RuntimeError, match="missing_table"):
        require_session_parent_schema(engine)
    apply_database_migrations(url, ROOT)
    require_session_parent_schema(engine)
    require_observation_schema(engine)
    for table, columns in before.items():
        assert [c["name"] for c in inspect(engine).get_columns(table)] == columns
    command.downgrade(_alembic_config(sync), "20260826_06")
    assert "dialogue_session_parents" not in inspect(engine).get_table_names()
    require_observation_schema(engine)
    apply_database_migrations(url, ROOT)
    require_session_parent_schema(engine)
    engine.dispose()


def test_parent_head_with_partial_cursor_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partial-parent.db"
    url = f"sqlite+aiosqlite:///{path}"
    apply_database_migrations(url, ROOT)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_session_parents")
        connection.exec_driver_sql(
            "CREATE TABLE dialogue_session_parents (conversation_id VARCHAR(100) PRIMARY KEY)"
        )
    # A turn-only rollback still reads all domain records. Parent activation fails.
    require_observation_schema(engine)
    with pytest.raises(RuntimeError, match="Session parent.*schema"):
        apply_database_migrations(url, ROOT)
    engine.dispose()

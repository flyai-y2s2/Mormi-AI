"""Add the PII-free generated dialogue copy cache.

Revision ID: 20260825_04
Revises: 20260825_03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "20260825_04"
down_revision: str | None = "20260825_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "dialogue_generated_copy_cache"
AVAILABLE_INDEX_NAME = "ix_dialogue_generated_copy_cache_available"


def _tables() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _columns() -> set[str]:
    return {
        str(column["name"])
        for column in inspect(op.get_bind()).get_columns(TABLE_NAME)
    }


def _indexes() -> set[str]:
    return {
        str(index["name"])
        for index in inspect(op.get_bind()).get_indexes(TABLE_NAME)
        if index.get("name")
    }


def upgrade() -> None:
    if TABLE_NAME not in _tables():
        op.create_table(
            TABLE_NAME,
            sa.Column("cache_key", sa.String(64), primary_key=True),
            sa.Column("key_version", sa.String(40), nullable=False),
            sa.Column(
                "status",
                sa.String(30),
                nullable=False,
                server_default="generating",
            ),
            sa.Column(
                "attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("lease_token", sa.String(64), nullable=True),
            sa.Column("artifact_json", sa.JSON(), nullable=True),
            sa.Column("artifact_sha256", sa.String(64), nullable=True),
            sa.Column("last_error_code", sa.String(80), nullable=True),
            sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            AVAILABLE_INDEX_NAME,
            TABLE_NAME,
            ["status", "available_at"],
        )
        return

    # Some unversioned deployments may already have created the ORM table.
    # Do not recreate or mutate its rows; only repair the additive index when
    # both indexed columns are present. The post-upgrade schema verifier will
    # reject any other partial table shape instead of guessing a repair.
    if (
        AVAILABLE_INDEX_NAME not in _indexes()
        and {"status", "available_at"}.issubset(_columns())
    ):
        op.create_index(
            AVAILABLE_INDEX_NAME,
            TABLE_NAME,
            ["status", "available_at"],
        )


def downgrade() -> None:
    if TABLE_NAME in _tables():
        op.drop_table(TABLE_NAME)

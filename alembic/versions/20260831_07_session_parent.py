"""Add optional raw-free session parent cursors (no domain schema changes).

Revision ID: 20260831_07
Revises: 20260826_06
"""

import sqlalchemy as sa

from alembic import op

revision = "20260831_07"
down_revision = "20260826_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "dialogue_session_parents" in sa.inspect(op.get_bind()).get_table_names():
        return  # The application-schema validator rejects partial existing tables.
    op.create_table(
        "dialogue_session_parents",
        sa.Column("conversation_id", sa.String(100), primary_key=True),
        sa.Column("graph_version", sa.String(60), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("turn_id", sa.String(100), nullable=False),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("checkpoint", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.conversation_id"], ondelete="CASCADE"
        ),
    )


def downgrade() -> None:
    # Domain state remains readable by the turn-only executor.
    op.drop_table("dialogue_session_parents")

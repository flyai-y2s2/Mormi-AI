"""Make home dialogue idempotency round-aware.

Revision ID: 20260825_03
Revises: 20260823_02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "20260825_03"
down_revision: str | None = "20260823_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT_NAME = "uq_conversation_learning_session_round"


def _conversation_columns() -> set[str]:
    return {
        str(column["name"])
        for column in inspect(op.get_bind()).get_columns("conversations")
    }


def _conversation_unique_constraints() -> set[str]:
    return {
        str(constraint["name"])
        for constraint in inspect(op.get_bind()).get_unique_constraints("conversations")
        if constraint.get("name")
    }


def upgrade() -> None:
    column_added = "conversation_round" not in _conversation_columns()
    if column_added:
        with op.batch_alter_table("conversations") as batch:
            batch.add_column(
                sa.Column(
                    "conversation_round",
                    sa.Integer(),
                    nullable=False,
                    server_default="1",
                )
            )

        # The old contract intended one dialogue per learning session, but a
        # historical race may have left more than one row. Preserve every row
        # and assign deterministic rounds before adding the unique constraint.
        op.execute(
            sa.text(
                """
                WITH ranked AS (
                    SELECT
                        conversation_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY learner_id, learning_session_id
                            ORDER BY created_at, conversation_id
                        ) AS inferred_round
                    FROM conversations
                    WHERE learning_session_id IS NOT NULL
                )
                UPDATE conversations
                SET conversation_round = (
                    SELECT inferred_round
                    FROM ranked
                    WHERE ranked.conversation_id = conversations.conversation_id
                )
                WHERE learning_session_id IS NOT NULL
                """
            )
        )

    if CONSTRAINT_NAME not in _conversation_unique_constraints():
        with op.batch_alter_table("conversations") as batch:
            batch.create_unique_constraint(
                CONSTRAINT_NAME,
                ["learner_id", "learning_session_id", "conversation_round"],
            )


def downgrade() -> None:
    if CONSTRAINT_NAME in _conversation_unique_constraints():
        with op.batch_alter_table("conversations") as batch:
            batch.drop_constraint(CONSTRAINT_NAME, type_="unique")

    if "conversation_round" in _conversation_columns():
        with op.batch_alter_table("conversations") as batch:
            batch.drop_column("conversation_round")

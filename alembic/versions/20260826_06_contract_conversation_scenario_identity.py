"""Contract the obsolete visit-wide conversation identity.

Revision ID: 20260826_06
Revises: 20260826_05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "20260826_06"
down_revision: str | None = "20260826_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_CONSTRAINT_NAME = "uq_conversation_learning_session_round"
NEW_CONSTRAINT_NAME = "uq_conversation_learning_session_scene_scenario_round"


def _conversation_unique_constraints() -> set[str]:
    return {
        str(constraint["name"])
        for constraint in inspect(op.get_bind()).get_unique_constraints("conversations")
        if constraint.get("name")
    }


def upgrade() -> None:
    constraints = _conversation_unique_constraints()
    if NEW_CONSTRAINT_NAME not in constraints:
        raise RuntimeError(
            "scenario identity contract cannot run before the transition unique exists"
        )
    if OLD_CONSTRAINT_NAME in constraints:
        with op.batch_alter_table("conversations") as batch:
            batch.drop_constraint(OLD_CONSTRAINT_NAME, type_="unique")


def downgrade() -> None:
    constraints = _conversation_unique_constraints()
    if OLD_CONSTRAINT_NAME not in constraints:
        # Final schema permits several scenarios to share one visit ID and
        # round.  Preserve every row while restoring the transition schema by
        # assigning deterministic visit-wide rounds first. NULL visit IDs stay
        # outside both idempotency constraints.
        conversations = sa.table(
            "conversations",
            sa.column("conversation_id", sa.String()),
            sa.column("learner_id", sa.Integer()),
            sa.column("learning_session_id", sa.String()),
            sa.column("conversation_round", sa.Integer()),
            sa.column("state_json", sa.JSON()),
            sa.column("created_at", sa.DateTime(timezone=True)),
        )
        bind = op.get_bind()
        rows = list(
            bind.execute(
                sa.select(
                    conversations.c.conversation_id,
                    conversations.c.learner_id,
                    conversations.c.learning_session_id,
                    conversations.c.conversation_round,
                    conversations.c.state_json,
                )
                .where(conversations.c.learning_session_id.is_not(None))
                .order_by(
                    conversations.c.learner_id,
                    conversations.c.learning_session_id,
                    conversations.c.conversation_round,
                    conversations.c.created_at,
                    conversations.c.conversation_id,
                )
            ).mappings()
        )
        visit_ranks: dict[tuple[int, str], int] = {}
        assignments: list[tuple[str, int, dict[str, object]]] = []
        for row in rows:
            visit_key = (int(row["learner_id"]), str(row["learning_session_id"]))
            rollback_round = visit_ranks.get(visit_key, 0) + 1
            visit_ranks[visit_key] = rollback_round
            raw_state = row["state_json"]
            state_json = dict(raw_state) if isinstance(raw_state, dict) else {}
            state_json["conversation_round"] = rollback_round
            assignments.append(
                (str(row["conversation_id"]), rollback_round, state_json)
            )

        # A row-by-row compression can temporarily collide with another row's
        # current scenario round. Move every row to a unique negative staging
        # round first, then write the final visit-wide rank and canonical JSON.
        for temporary_rank, (conversation_id, _, _) in enumerate(assignments, start=1):
            bind.execute(
                conversations.update()
                .where(conversations.c.conversation_id == conversation_id)
                .values(conversation_round=-(len(assignments) + temporary_rank))
            )
        for conversation_id, rollback_round, state_json in assignments:
            bind.execute(
                conversations.update()
                .where(conversations.c.conversation_id == conversation_id)
                .values(
                    conversation_round=rollback_round,
                    state_json=state_json,
                )
            )
        with op.batch_alter_table("conversations") as batch:
            batch.create_unique_constraint(
                OLD_CONSTRAINT_NAME,
                ["learner_id", "learning_session_id", "conversation_round"],
            )

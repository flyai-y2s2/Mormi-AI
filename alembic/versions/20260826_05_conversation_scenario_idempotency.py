"""Expand conversation idempotency for a rolling reader transition.

Revision ID: 20260826_05
Revises: 20260825_04
"""

from collections.abc import Sequence

from sqlalchemy import inspect

from alembic import op

revision: str = "20260826_05"
down_revision: str | None = "20260825_04"
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
    # This revision is the expand phase of an expand-contract rollout.  The
    # scenario-aware reader can be deployed while the previous image is still
    # live, so retain (or restore for a metadata-created fresh schema) the old
    # visit-wide key until revision 06 confirms every rollback image can read
    # the narrower five-part identity.
    if OLD_CONSTRAINT_NAME not in constraints:
        with op.batch_alter_table("conversations") as batch:
            batch.create_unique_constraint(
                OLD_CONSTRAINT_NAME,
                ["learner_id", "learning_session_id", "conversation_round"],
            )

    if NEW_CONSTRAINT_NAME not in constraints:
        with op.batch_alter_table("conversations") as batch:
            batch.create_unique_constraint(
                NEW_CONSTRAINT_NAME,
                [
                    "learner_id",
                    "learning_session_id",
                    "scene",
                    "scenario_id",
                    "conversation_round",
                ],
            )

def downgrade() -> None:
    if NEW_CONSTRAINT_NAME in _conversation_unique_constraints():
        with op.batch_alter_table("conversations") as batch:
            batch.drop_constraint(NEW_CONSTRAINT_NAME, type_="unique")

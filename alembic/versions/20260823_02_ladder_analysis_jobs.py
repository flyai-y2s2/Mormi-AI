"""Add durable ladder analysis jobs without storing child speech.

Revision ID: 20260823_02
Revises: 20260817_01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260823_02"
down_revision = "20260817_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ladder_analysis_jobs",
        sa.Column("analysis_id", sa.String(100), primary_key=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("learner_id", sa.Integer(), nullable=False),
        sa.Column("skill_id", sa.String(100), nullable=False),
        sa.Column("trigger_session_id", sa.String(100), nullable=False),
        sa.Column("session_ids_json", sa.JSON(), nullable=False),
        sa.Column("current_level", sa.String(10), nullable=False),
        sa.Column("performance_json", sa.JSON(), nullable=False),
        sa.Column("lower_rule_evidence_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_json", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(100), nullable=True),
        sa.Column("recommendation_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_ladder_analysis_idempotency"),
    )
    op.create_index("ix_ladder_analysis_claim", "ladder_analysis_jobs", ["status", "available_at"])
    op.create_index("ix_ladder_analysis_learner_id", "ladder_analysis_jobs", ["learner_id"])
    op.create_index("ix_ladder_analysis_skill_id", "ladder_analysis_jobs", ["skill_id"])
    op.create_index(
        "ix_ladder_analysis_trigger_session_id",
        "ladder_analysis_jobs",
        ["trigger_session_id"],
    )
    op.create_index(
        "ix_ladder_analysis_learner_skill",
        "ladder_analysis_jobs",
        ["learner_id", "skill_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("ladder_analysis_jobs")

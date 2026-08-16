"""Add evidence-linked dialogue observation tables.

Revision ID: 20260817_01
Revises: None
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "20260817_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tables() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    tables = _tables()
    if "dialogue_turn_observations" not in tables:
        op.create_table(
            "dialogue_turn_observations",
            sa.Column("observation_id", sa.String(100), primary_key=True),
            sa.Column("conversation_id", sa.String(100), nullable=False),
            sa.Column("learner_id", sa.Integer(), nullable=False),
            sa.Column("learning_session_id", sa.String(100), nullable=True),
            sa.Column("scene", sa.String(40), nullable=False),
            sa.Column("scenario_id", sa.String(100), nullable=False),
            sa.Column("task_id", sa.String(100), nullable=False),
            sa.Column("stage_id", sa.String(100), nullable=False),
            sa.Column("task_index", sa.Integer(), nullable=False),
            sa.Column("subgoal_id", sa.String(100), nullable=False),
            sa.Column("source_turn_id", sa.String(100), nullable=False),
            sa.Column("result_turn_id", sa.String(100), nullable=True),
            sa.Column("response_id", sa.String(100), nullable=True),
            sa.Column("response_type", sa.String(40), nullable=True),
            sa.Column("input_kind", sa.String(40), nullable=False),
            sa.Column("response_category", sa.String(50), nullable=False),
            sa.Column("difficulty_class", sa.String(40), nullable=False),
            sa.Column("concept_result", sa.String(40), nullable=False),
            sa.Column("safety_category", sa.String(40), nullable=False),
            sa.Column("misconception_tag", sa.String(100), nullable=True),
            sa.Column("bottleneck", sa.String(100), nullable=False),
            sa.Column("classifier_confidence", sa.Float(), nullable=True),
            sa.Column("expression_before", sa.String(10), nullable=False),
            sa.Column("expression_after", sa.String(10), nullable=False),
            sa.Column("hint_before", sa.String(10), nullable=False),
            sa.Column("hint_after", sa.String(10), nullable=False),
            sa.Column("transition_reason", sa.String(100), nullable=False),
            sa.Column("dialogue_act", sa.String(100), nullable=False),
            sa.Column("help_card_shown", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("help_card_level", sa.String(10), nullable=True),
            sa.Column(
                "help_card_auto_open",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("speaker_source", sa.String(60), nullable=False),
            sa.Column("verifier_status", sa.String(40), nullable=False),
            sa.Column("fallback_reason", sa.String(120), nullable=True),
            sa.Column("completion_outcome", sa.String(40), nullable=True),
            sa.Column(
                "adult_intervention_status",
                sa.String(40),
                nullable=False,
                server_default="not_collected",
            ),
            sa.Column(
                "record_origin",
                sa.String(40),
                nullable=False,
                server_default="live",
            ),
            sa.Column("analysis_json", sa.JSON(), nullable=False),
            sa.Column("runtime_json", sa.JSON(), nullable=False),
            sa.Column("versions_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.conversation_id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["source_turn_id"],
                ["turns.turn_id"],
                ondelete="CASCADE",
            ),
            sa.UniqueConstraint("source_turn_id", name="uq_observation_source_turn"),
            sa.UniqueConstraint(
                "conversation_id",
                "response_id",
                name="uq_observation_conversation_response",
            ),
        )
        op.create_index(
            "ix_observation_learner_created",
            "dialogue_turn_observations",
            ["learner_id", "created_at"],
        )
        for column in (
            "conversation_id",
            "learner_id",
            "learning_session_id",
            "scene",
            "scenario_id",
            "task_id",
            "source_turn_id",
            "result_turn_id",
            "response_category",
            "difficulty_class",
            "concept_result",
            "safety_category",
            "bottleneck",
            "transition_reason",
            "dialogue_act",
            "speaker_source",
        ):
            op.create_index(
                f"ix_dialogue_turn_observations_{column}",
                "dialogue_turn_observations",
                [column],
            )

    if "dialogue_claims" not in tables:
        op.create_table(
            "dialogue_claims",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("observation_id", sa.String(100), nullable=False),
            sa.Column("slot_id", sa.String(100), nullable=False),
            sa.Column("semantic_role", sa.String(40), nullable=False),
            sa.Column("value_json", sa.JSON(), nullable=True),
            sa.Column("factual", sa.Boolean(), nullable=False),
            sa.Column("validation_status", sa.String(40), nullable=False),
            sa.Column("evidence_span_encrypted", sa.Text(), nullable=True),
            sa.Column(
                "newly_verified",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["observation_id"],
                ["dialogue_turn_observations.observation_id"],
                ondelete="CASCADE",
            ),
        )
        op.create_index(
            "ix_dialogue_claims_observation_id",
            "dialogue_claims",
            ["observation_id"],
        )
        op.create_index(
            "ix_dialogue_claims_validation_status",
            "dialogue_claims",
            ["validation_status"],
        )
        op.create_index(
            "ix_claim_observation_slot",
            "dialogue_claims",
            ["observation_id", "slot_id"],
        )

    if "dialogue_task_outcomes" not in tables:
        op.create_table(
            "dialogue_task_outcomes",
            sa.Column("outcome_id", sa.String(100), primary_key=True),
            sa.Column("conversation_id", sa.String(100), nullable=False),
            sa.Column("learner_id", sa.Integer(), nullable=False),
            sa.Column("learning_session_id", sa.String(100), nullable=True),
            sa.Column("scene", sa.String(40), nullable=False),
            sa.Column("scenario_id", sa.String(100), nullable=False),
            sa.Column("task_id", sa.String(100), nullable=False),
            sa.Column("task_index", sa.Integer(), nullable=False),
            sa.Column("start_expression_level", sa.String(10), nullable=False),
            sa.Column("end_expression_level", sa.String(10), nullable=False),
            sa.Column("start_hint_level", sa.String(10), nullable=False),
            sa.Column("max_hint_level", sa.String(10), nullable=False),
            sa.Column("completion_outcome", sa.String(40), nullable=False),
            sa.Column("verified_slots_json", sa.JSON(), nullable=False),
            sa.Column("bottleneck_candidates_json", sa.JSON(), nullable=False),
            sa.Column("evidence_observation_ids_json", sa.JSON(), nullable=False),
            sa.Column("note_id", sa.String(100), nullable=True),
            sa.Column(
                "record_origin",
                sa.String(40),
                nullable=False,
                server_default="live",
            ),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.conversation_id"],
                ondelete="CASCADE",
            ),
            sa.UniqueConstraint(
                "conversation_id",
                "task_index",
                name="uq_task_outcome_conversation_index",
            ),
        )
        for column in (
            "conversation_id",
            "learner_id",
            "learning_session_id",
            "scenario_id",
            "task_id",
            "completion_outcome",
            "note_id",
        ):
            op.create_index(
                f"ix_dialogue_task_outcomes_{column}",
                "dialogue_task_outcomes",
                [column],
            )

    if "note_evidence_links" not in tables:
        op.create_table(
            "note_evidence_links",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("note_id", sa.String(100), nullable=False),
            sa.Column("observation_id", sa.String(100), nullable=False),
            sa.Column("source_slot_ids_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["note_id"],
                ["notes.note_id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["observation_id"],
                ["dialogue_turn_observations.observation_id"],
                ondelete="CASCADE",
            ),
            sa.UniqueConstraint(
                "note_id",
                "observation_id",
                name="uq_note_observation_evidence",
            ),
        )
        op.create_index(
            "ix_note_evidence_links_note_id",
            "note_evidence_links",
            ["note_id"],
        )
        op.create_index(
            "ix_note_evidence_links_observation_id",
            "note_evidence_links",
            ["observation_id"],
        )

    if "ai_outbox_events" not in tables:
        op.create_table(
            "ai_outbox_events",
            sa.Column("event_id", sa.String(100), primary_key=True),
            sa.Column("aggregate_type", sa.String(60), nullable=False),
            sa.Column("aggregate_id", sa.String(100), nullable=False),
            sa.Column("event_type", sa.String(100), nullable=False),
            sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.String(300), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "ix_ai_outbox_events_aggregate_id",
            "ai_outbox_events",
            ["aggregate_id"],
        )
        op.create_index(
            "ix_ai_outbox_events_event_type",
            "ai_outbox_events",
            ["event_type"],
        )
        op.create_index("ix_ai_outbox_events_status", "ai_outbox_events", ["status"])
        op.create_index(
            "ix_outbox_delivery",
            "ai_outbox_events",
            ["status", "available_at"],
        )


def downgrade() -> None:
    tables = _tables()
    for table in (
        "note_evidence_links",
        "dialogue_claims",
        "dialogue_task_outcomes",
        "ai_outbox_events",
        "dialogue_turn_observations",
    ):
        if table in tables:
            op.drop_table(table)

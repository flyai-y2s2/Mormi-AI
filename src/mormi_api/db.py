from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .schemas import utc_now


class Base(DeclarativeBase):
    pass


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: object) -> None:
    """Make local/test SQLite enforce the same parent-child contract as PostgreSQL."""

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DataMigrationRecord(Base):
    __tablename__ = "data_migrations"

    migration_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConversationRecord(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    learning_session_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    scene: Mapped[str] = mapped_column(String(40))
    scenario_id: Mapped[str] = mapped_column(String(100))
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(40), index=True)
    raw_retention_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TurnRecord(Base):
    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "response_id", name="uq_conversation_response"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        index=True,
    )
    task_id: Mapped[str] = mapped_column(String(100))
    state_version: Mapped[int] = mapped_column(Integer)
    # Legacy physical name retained for zero-downtime schema compatibility.
    # Current values are readable ``plain:<text>`` envelopes.
    mormi_question_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_contract: Mapped[dict[str, Any]] = mapped_column(JSON)
    response_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    response_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    result_turn_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    response_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # Legacy physical name; current values are plaintext envelopes.
    response_raw_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_structured: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    safety_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    response_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    expression_level: Mapped[str] = mapped_column(String(10))
    hint_level: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LearnerProfileRecord(Base):
    __tablename__ = "learner_profiles"

    learner_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PracticeResultRecord(Base):
    __tablename__ = "practice_results"

    practice_result_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    skill_id: Mapped[str] = mapped_column(String(100), index=True)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    success_rate: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class NoteRecord(Base):
    __tablename__ = "notes"

    note_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        index=True,
    )
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    skill_id: Mapped[str] = mapped_column(String(100), index=True)
    text: Mapped[str] = mapped_column(Text)
    attribution: Mapped[str] = mapped_column(String(30))
    evidence: Mapped[str] = mapped_column(String(40))
    attribution_label: Mapped[str] = mapped_column(String(80))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DialogueTurnObservationRecord(Base):
    """One evidence-linked pedagogical observation for an answered turn."""

    __tablename__ = "dialogue_turn_observations"
    __table_args__ = (
        UniqueConstraint("source_turn_id", name="uq_observation_source_turn"),
        UniqueConstraint(
            "conversation_id",
            "response_id",
            name="uq_observation_conversation_response",
        ),
        Index(
            "ix_observation_learner_created",
            "learner_id",
            "created_at",
        ),
    )

    observation_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        index=True,
    )
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    learning_session_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    scene: Mapped[str] = mapped_column(String(40), index=True)
    scenario_id: Mapped[str] = mapped_column(String(100), index=True)
    task_id: Mapped[str] = mapped_column(String(100), index=True)
    stage_id: Mapped[str] = mapped_column(String(100))
    task_index: Mapped[int] = mapped_column(Integer)
    subgoal_id: Mapped[str] = mapped_column(String(100))
    source_turn_id: Mapped[str] = mapped_column(
        ForeignKey("turns.turn_id", ondelete="CASCADE"),
        index=True,
    )
    result_turn_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    response_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    response_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    input_kind: Mapped[str] = mapped_column(String(40))
    response_category: Mapped[str] = mapped_column(String(50), index=True)
    difficulty_class: Mapped[str] = mapped_column(String(40), index=True)
    concept_result: Mapped[str] = mapped_column(String(40), index=True)
    safety_category: Mapped[str] = mapped_column(String(40), index=True)
    misconception_tag: Mapped[str | None] = mapped_column(String(100), nullable=True)
    bottleneck: Mapped[str] = mapped_column(String(100), index=True)
    classifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    expression_before: Mapped[str] = mapped_column(String(10))
    expression_after: Mapped[str] = mapped_column(String(10))
    hint_before: Mapped[str] = mapped_column(String(10))
    hint_after: Mapped[str] = mapped_column(String(10))
    transition_reason: Mapped[str] = mapped_column(String(100), index=True)
    dialogue_act: Mapped[str] = mapped_column(String(100), index=True)
    help_card_shown: Mapped[bool] = mapped_column(Boolean, default=False)
    help_card_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    help_card_auto_open: Mapped[bool] = mapped_column(Boolean, default=False)
    speaker_source: Mapped[str] = mapped_column(String(60), index=True)
    verifier_status: Mapped[str] = mapped_column(String(40))
    fallback_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    completion_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    record_origin: Mapped[str] = mapped_column(String(40), default="live")
    analysis_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    runtime_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    versions_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DialogueClaimRecord(Base):
    __tablename__ = "dialogue_claims"
    __table_args__ = (
        Index("ix_claim_observation_slot", "observation_id", "slot_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("dialogue_turn_observations.observation_id", ondelete="CASCADE"),
        index=True,
    )
    slot_id: Mapped[str] = mapped_column(String(100))
    semantic_role: Mapped[str] = mapped_column(String(40))
    value_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    factual: Mapped[bool] = mapped_column(Boolean)
    validation_status: Mapped[str] = mapped_column(String(40), index=True)
    # Legacy physical name; current values are plaintext evidence envelopes.
    evidence_span_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    newly_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DialogueTaskOutcomeRecord(Base):
    __tablename__ = "dialogue_task_outcomes"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "task_index",
            name="uq_task_outcome_conversation_index",
        ),
    )

    outcome_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        index=True,
    )
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    learning_session_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    scene: Mapped[str] = mapped_column(String(40))
    scenario_id: Mapped[str] = mapped_column(String(100), index=True)
    task_id: Mapped[str] = mapped_column(String(100), index=True)
    task_index: Mapped[int] = mapped_column(Integer)
    start_expression_level: Mapped[str] = mapped_column(String(10))
    end_expression_level: Mapped[str] = mapped_column(String(10))
    start_hint_level: Mapped[str] = mapped_column(String(10))
    max_hint_level: Mapped[str] = mapped_column(String(10))
    completion_outcome: Mapped[str] = mapped_column(String(40), index=True)
    verified_slots_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    bottleneck_candidates_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    evidence_observation_ids_json: Mapped[list[str]] = mapped_column(JSON)
    note_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    record_origin: Mapped[str] = mapped_column(String(40), default="live")
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class NoteEvidenceLinkRecord(Base):
    __tablename__ = "note_evidence_links"
    __table_args__ = (
        UniqueConstraint("note_id", "observation_id", name="uq_note_observation_evidence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    note_id: Mapped[str] = mapped_column(
        ForeignKey("notes.note_id", ondelete="CASCADE"),
        index=True,
    )
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("dialogue_turn_observations.observation_id", ondelete="CASCADE"),
        index=True,
    )
    source_slot_ids_json: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class OutboxEventRecord(Base):
    __tablename__ = "ai_outbox_events"
    __table_args__ = (
        Index("ix_outbox_delivery", "status", "available_at"),
    )

    event_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    aggregate_type: Mapped[str] = mapped_column(String(60))
    aggregate_id: Mapped[str] = mapped_column(String(100), index=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LadderAnalysisRecord(Base):
    """Durable subunit analysis metadata; child speech is deliberately excluded."""

    __tablename__ = "ladder_analysis_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ladder_analysis_idempotency"),
        Index("ix_ladder_analysis_claim", "status", "available_at"),
        Index("ix_ladder_analysis_learner_skill", "learner_id", "skill_id", "created_at"),
    )

    analysis_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(200))
    learner_id: Mapped[int] = mapped_column(Integer, index=True)
    skill_id: Mapped[str] = mapped_column(String(100), index=True)
    trigger_session_id: Mapped[str] = mapped_column(String(100), index=True)
    session_ids_json: Mapped[list[str]] = mapped_column(JSON)
    current_level: Mapped[str] = mapped_column(String(10))
    performance_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    lower_rule_evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    decision_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    recommendation_version: Mapped[int] = mapped_column(Integer, default=1)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        if self.engine.dialect.name == "sqlite":
            event.listen(
                self.engine.sync_engine,
                "connect",
                _enable_sqlite_foreign_keys,
            )
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

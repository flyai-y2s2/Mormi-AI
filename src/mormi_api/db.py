from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
    mormi_question_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_contract: Mapped[dict[str, Any]] = mapped_column(JSON)
    response_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    response_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    result_turn_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    response_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
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


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

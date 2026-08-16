from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from conftest import FakeGateway
from sqlalchemy import create_engine, delete, inspect, select

from alembic import command
from mormi_api.db import (
    Base,
    ConversationRecord,
    Database,
    DialogueTurnObservationRecord,
    OutboxEventRecord,
    TurnRecord,
)
from mormi_api.engine import ConversationEngine
from mormi_api.migrations import apply_database_migrations
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, SessionCreate, utc_now
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


def _alembic_config(database_url: str) -> Config:
    root = Path(__file__).resolve().parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_additive_migration_preserves_legacy_conversation_and_turn_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MORMI_DATABASE_URL", raising=False)
    database_path = tmp_path / "legacy.db"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url)
    ConversationRecord.__table__.create(engine)
    TurnRecord.__table__.create(engine)
    now = utc_now()
    with engine.begin() as connection:
        connection.execute(
            ConversationRecord.__table__.insert().values(
                conversation_id="conversation_legacy",
                learner_id=7,
                learning_session_id="lesson_legacy",
                scene="home_teach",
                scenario_id="home_teach",
                state_json={},
                state_version=1,
                status="active",
                raw_retention_until=None,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            TurnRecord.__table__.insert().values(
                turn_id="turn_legacy",
                conversation_id="conversation_legacy",
                task_id="home_teaching",
                state_version=1,
                mormi_question_encrypted="encrypted",
                turn_contract={},
                expression_level="L4",
                hint_level="H0",
                created_at=now,
            )
        )

    config = _alembic_config(database_url)
    command.upgrade(config, "head")

    with engine.connect() as connection:
        assert connection.execute(
            select(ConversationRecord.conversation_id)
        ).scalar_one() == "conversation_legacy"
        assert connection.execute(
            select(TurnRecord.turn_id)
        ).scalar_one() == "turn_legacy"
        tables = set(inspect(connection).get_table_names())
        assert "dialogue_turn_observations" in tables
        assert "dialogue_claims" in tables
        assert "dialogue_task_outcomes" in tables
        assert "note_evidence_links" in tables
        assert "ai_outbox_events" in tables

    command.downgrade(config, "base")
    with engine.connect() as connection:
        assert connection.execute(
            select(ConversationRecord.conversation_id)
        ).scalar_one() == "conversation_legacy"
        assert connection.execute(
            select(TurnRecord.turn_id)
        ).scalar_one() == "turn_legacy"
    engine.dispose()


def test_migration_stamps_complete_schema_created_by_app_startup(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-created.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    apply_database_migrations(
        f"sqlite+aiosqlite:///{database_path}",
        Path(__file__).resolve().parents[1],
    )

    with engine.connect() as connection:
        version = connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        assert version == "20260817_01"
    engine.dispose()


@pytest.mark.asyncio
async def test_historical_backfill_marks_missing_fields_without_reanalysis(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/backfill.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=9,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="0855ffef-3042-4ef4-85e0-22162b560c7e",
            type="no_response",
        ),
    )

    # Simulate a row created by the legacy application before observation
    # tables existed. The encrypted original turn remains untouched.
    async with database.sessions() as db:
        await db.execute(delete(OutboxEventRecord))
        await db.execute(delete(DialogueTurnObservationRecord))
        await db.commit()

    assert await repository.backfill_historical_observations() == 1
    assert await repository.backfill_historical_observations() == 0

    async with database.sessions() as db:
        observation = (
            await db.execute(select(DialogueTurnObservationRecord))
        ).scalar_one()
        outbox_count = len(list((await db.execute(select(OutboxEventRecord))).scalars()))
        turn = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()

    assert observation.record_origin == "historical_backfill"
    assert observation.difficulty_class == "not_collected"
    assert observation.transition_reason == "not_collected"
    assert observation.analysis_json["historical_backfill"] is True
    assert outbox_count == 0
    assert turn.response_raw_encrypted is not None
    await database.dispose()

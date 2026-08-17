from __future__ import annotations

import pytest
from conftest import FakeGateway
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from mormi_api.db import Database, DialogueClaimRecord, DialogueTurnObservationRecord
from mormi_api.engine import ConversationEngine
from mormi_api.repository import (
    PersistenceError,
    Repository,
    _is_duplicate_response_integrity_error,
)
from mormi_api.schemas import ChildResponse, SessionCreate
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


@pytest.mark.asyncio
async def test_sqlite_enforces_observation_claim_foreign_key(tmp_path: object) -> None:
    """Local tests must fail on the same missing-parent write as PostgreSQL."""

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/foreign-keys.db")
    await database.create_schema()

    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        async with database.sessions() as db:
            db.add(
                DialogueClaimRecord(
                    observation_id="missing-observation",
                    slot_id="answer",
                    semantic_role="conclusion",
                    value_json="3",
                    factual=True,
                    validation_status="verified",
                    evidence_span_encrypted="plain:3개",
                    newly_verified=True,
                )
            )
            await db.commit()

    await database.dispose()


@pytest.mark.parametrize(
    "message",
    [
        "UNIQUE constraint failed: turns.conversation_id, turns.response_id",
        (
            "duplicate key value violates unique constraint "
            '"uq_observation_conversation_response"'
        ),
    ],
)
def test_duplicate_response_constraints_are_idempotency_conflicts(message: str) -> None:
    error = IntegrityError("INSERT", {}, RuntimeError(message))

    assert _is_duplicate_response_integrity_error(error) is True


def test_foreign_key_failure_is_not_misclassified_as_duplicate_response() -> None:
    error = IntegrityError(
        "INSERT",
        {},
        RuntimeError(
            "insert on dialogue_claims violates foreign key constraint "
            '"dialogue_claims_observation_id_fkey"'
        ),
    )

    assert _is_duplicate_response_integrity_error(error) is False


@pytest.mark.asyncio
async def test_missing_observation_table_rolls_back_and_same_response_can_retry(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/missing-table.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    async with database.engine.begin() as connection:
        await connection.execute(text("DROP TABLE dialogue_turn_observations"))

    response = ChildResponse(
        turn_id=started.turn.turn_id,
        response_id="d2652b3b-f168-41d3-99e7-d898f4f215af",
        type="no_response",
    )
    with pytest.raises(PersistenceError, match="turn_persistence_failed"):
        await service.respond(started.conversation_id, response)

    # A failed observation write must not leave the source turn answered or
    # consume its response id. Once the schema is repaired, the child's exact
    # same submission is safe to retry and creates one observation.
    await database.create_schema()
    retried = await service.respond(started.conversation_id, response)
    assert retried.turn.status.value == "active"
    async with database.sessions() as db:
        observation_count = await db.scalar(
            select(func.count()).select_from(DialogueTurnObservationRecord)
        )
    assert observation_count == 1

    await database.dispose()

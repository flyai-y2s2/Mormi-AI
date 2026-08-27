from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from conftest import FakeGateway
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from mormi_api.db import (
    ConversationRecord,
    Database,
    DialogueClaimRecord,
    DialogueTurnObservationRecord,
    TurnRecord,
)
from mormi_api.engine import ConversationEngine
from mormi_api.repository import (
    PersistenceError,
    Repository,
    _is_duplicate_response_integrity_error,
)
from mormi_api.schemas import ChildResponse, SessionCreate, utc_now
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


@pytest.mark.asyncio
async def test_raw_retention_purge_clears_all_raw_stores_but_keeps_v2_markers(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/raw-retention.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    conversation_ids: list[str] = []
    for learner_id in (71, 72):
        started = await service.create_conversation(
            SessionCreate(
                learner_id=learner_id,
                scene="cafe",
                scenario_id="cafe_queue_demo",
                conversation_storage_consent=True,
                retention_policy="30_days",
            )
        )
        await service.respond(
            started.conversation_id,
            ChildResponse(
                turn_id=started.turn.turn_id,
                response_id=uuid4(),
                type="no_response",
            ),
        )
        conversation_ids.append(started.conversation_id)

    expired_at = utc_now() - timedelta(seconds=1)
    async with database.sessions() as db:
        observations = list(
            (
                await db.execute(
                    select(DialogueTurnObservationRecord).order_by(
                        DialogueTurnObservationRecord.learner_id
                    )
                )
            ).scalars()
        )
        assert len(observations) == 2
        for index, conversation_id in enumerate(conversation_ids):
            conversation = await db.get(ConversationRecord, conversation_id)
            assert conversation is not None
            state_json = dict(conversation.state_json)
            if index == 0:
                state_json["child_note_evidence"] = {"reason": "아이 원문"}
            else:
                state_json["runtime_contract_version"] = "verdict-v1"
                state_json["child_note_evidence"] = {
                    "relation": "verified_relation"
                }
            conversation.state_json = state_json
            conversation.raw_retention_until = expired_at
            db.add(
                DialogueClaimRecord(
                    observation_id=observations[index].observation_id,
                    slot_id="reason",
                    semantic_role="reason",
                    value_json=True,
                    factual=True,
                    validation_status="verified",
                    evidence_span_encrypted="plain:아이 원문",
                    newly_verified=True,
                )
            )
        await db.commit()

    await repository.purge_expired_raw_data()

    # A later active turn must inherit the disabled storage state instead of
    # recreating raw text after the indexed deadline was cleared.
    resumed = await service.snapshot(conversation_ids[0])
    await service.respond(
        conversation_ids[0],
        ChildResponse(
            turn_id=resumed.turn.turn_id,
            response_id=uuid4(),
            type="no_response",
        ),
    )

    async with database.sessions() as db:
        turns = list((await db.execute(select(TurnRecord))).scalars())
        claims = list((await db.execute(select(DialogueClaimRecord))).scalars())
        records = {
            record.conversation_id: record
            for record in (await db.execute(select(ConversationRecord))).scalars()
        }
    assert all(turn.response_raw_encrypted is None for turn in turns)
    assert all(turn.response_structured is None for turn in turns if turn.response_id)
    assert all(claim.evidence_span_encrypted is None for claim in claims)
    assert records[conversation_ids[0]].state_json["child_note_evidence"] == {}
    assert records[conversation_ids[0]].state_json["raw_storage_enabled"] is False
    assert records[conversation_ids[0]].state_json["raw_retention_until"] is None
    assert records[conversation_ids[1]].state_json["child_note_evidence"] == {
        "relation": "verified_relation"
    }
    assert records[conversation_ids[1]].state_json["raw_storage_enabled"] is False
    assert records[conversation_ids[1]].state_json["raw_retention_until"] is None
    assert all(record.raw_retention_until is None for record in records.values())

    await database.dispose()


@pytest.mark.asyncio
async def test_turn_commit_cannot_store_raw_after_deadline_before_periodic_purge(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/raw-deadline-race.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=81,
            scene="cafe",
            scenario_id="cafe_queue_demo",
            conversation_storage_consent=True,
            retention_policy="30_days",
        )
    )
    expired_at = utc_now() - timedelta(seconds=1)
    async with database.sessions() as db:
        conversation = await db.get(ConversationRecord, started.conversation_id)
        assert conversation is not None
        state_json = dict(conversation.state_json)
        state_json["raw_retention_until"] = expired_at.isoformat()
        state_json["child_note_evidence"] = {"reason": "만료된 아이 원문"}
        conversation.state_json = state_json
        conversation.raw_retention_until = expired_at
        await db.commit()

    effective_state = await repository.get_state(started.conversation_id)
    assert effective_state.raw_storage_enabled is False
    assert effective_state.raw_retention_until is None
    assert effective_state.child_note_evidence == {}

    await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id=uuid4(),
            type="no_response",
        ),
    )

    async with database.sessions() as db:
        conversation = await db.get(ConversationRecord, started.conversation_id)
        answered = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()
    assert conversation is not None
    assert conversation.raw_retention_until is None
    assert conversation.state_json["raw_storage_enabled"] is False
    assert conversation.state_json["raw_retention_until"] is None
    assert answered.response_raw_encrypted is None
    assert answered.response_structured is None

    await database.dispose()

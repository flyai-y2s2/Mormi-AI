from __future__ import annotations

import asyncio

import pytest
from conftest import FakeGateway
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from mormi_api.db import ConversationRecord, Database, TurnRecord
from mormi_api.engine import ConversationEngine
from mormi_api.repository import (
    Repository,
    _is_conversation_idempotency_integrity_error,
)
from mormi_api.schemas import SceneType, SessionCreate
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService

CAFE_CONTEXT = {
    "menu_items": [
        {"id": "juice", "name": "주스", "price": 3000},
        {"id": "cookie", "name": "쿠키", "price": 2000},
    ],
    "mormi_menu_id": "juice",
    "budget": 5000,
}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "duplicate key value violates unique constraint "
            '"uq_conversation_learning_session_scene_scenario_round"',
            True,
        ),
        (
            "duplicate key value violates unique constraint "
            '"uq_conversation_learning_session_round"',
            True,
        ),
        ("UNIQUE constraint failed: conversations.conversation_id", False),
    ],
)
def test_only_scenario_round_unique_key_is_a_create_idempotency_conflict(
    message: str,
    expected: bool,
) -> None:
    error = IntegrityError("INSERT", {}, RuntimeError(message))

    assert _is_conversation_idempotency_integrity_error(error) is expected


class ConcurrentLookupRepository(Repository):
    """Hold the initial idempotency reads until both creates saw no winner."""

    def __init__(self, database: Database, text_cipher: TextCipher) -> None:
        super().__init__(database, text_cipher)
        self._initial_lookups = 0
        self._both_initial_lookups_finished = asyncio.Event()

    async def conversation_id_for_learning_session(
        self,
        learner_id: int,
        learning_session_id: str,
        scene: SceneType,
        scenario_id: str,
        conversation_round: int,
    ) -> str | None:
        existing_id = await super().conversation_id_for_learning_session(
            learner_id,
            learning_session_id,
            scene,
            scenario_id,
            conversation_round,
        )
        self._initial_lookups += 1
        if self._initial_lookups == 2:
            self._both_initial_lookups_finished.set()
        await self._both_initial_lookups_finished.wait()
        return existing_id


@pytest.mark.asyncio
async def test_same_visit_round_is_idempotent_per_scene_and_scenario(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/scenario-idempotency.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    shared_identity = {
        "learner_id": 901,
        "scene": "cafe",
        "learning_session_id": "cafe-visit-901",
        "conversation_round": 1,
    }
    queue_request = SessionCreate(
        **shared_identity,
        scenario_id="cafe_queue",
        queue_context={"left_count": 2, "right_count": 4},
    )
    budget_request = SessionCreate(
        **shared_identity,
        scenario_id="cafe_budget_menu",
        cafe_context=CAFE_CONTEXT,
    )

    queue = await service.create_conversation(queue_request)
    budget = await service.create_conversation(budget_request)
    queue_retry = await service.create_conversation(queue_request)
    budget_retry = await service.create_conversation(budget_request)

    assert queue.conversation_id != budget.conversation_id
    assert queue_retry.conversation_id == queue.conversation_id
    assert budget_retry.conversation_id == budget.conversation_id
    assert (await repository.get_state(queue.conversation_id)).scenario_id == "cafe_queue"
    assert (await repository.get_state(budget.conversation_id)).scenario_id == "cafe_budget_menu"
    async with database.sessions() as session:
        rows = (
            await session.execute(
                select(ConversationRecord).where(
                    ConversationRecord.learner_id == 901,
                    ConversationRecord.learning_session_id == "cafe-visit-901",
                    ConversationRecord.conversation_round == 1,
                )
            )
        ).scalars()
        assert {record.scenario_id for record in rows} == {
            "cafe_queue",
            "cafe_budget_menu",
        }
    await database.dispose()


@pytest.mark.asyncio
async def test_concurrent_same_scenario_create_returns_the_committed_winner(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/concurrent-create.db")
    await database.create_schema()
    repository = ConcurrentLookupRepository(
        database,
        TextCipher("test-encryption-key"),
    )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    request = SessionCreate(
        learner_id=902,
        scene="cafe",
        scenario_id="cafe_budget_menu",
        learning_session_id="cafe-visit-902",
        conversation_round=1,
        cafe_context=CAFE_CONTEXT,
    )

    first, retry = await asyncio.wait_for(
        asyncio.gather(
            service.create_conversation(request),
            service.create_conversation(request),
        ),
        timeout=5,
    )

    assert first.conversation_id == retry.conversation_id
    assert first.turn.turn_id == retry.turn.turn_id
    async with database.sessions() as session:
        conversation_count = await session.scalar(
            select(func.count()).select_from(ConversationRecord)
        )
        turn_count = await session.scalar(select(func.count()).select_from(TurnRecord))
    assert conversation_count == 1
    assert turn_count == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_transition_schema_concurrent_same_scenario_still_returns_winner(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/transition-race.db")
    await database.create_schema()
    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE UNIQUE INDEX uq_conversation_learning_session_round "
                "ON conversations (learner_id, learning_session_id, conversation_round)"
            )
        )
    repository = ConcurrentLookupRepository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    request = SessionCreate(
        learner_id=903,
        scene="cafe",
        scenario_id="cafe_budget_menu",
        learning_session_id="cafe-visit-903",
        conversation_round=1,
        cafe_context=CAFE_CONTEXT,
    )

    first, retry = await asyncio.wait_for(
        asyncio.gather(
            service.create_conversation(request),
            service.create_conversation(request),
        ),
        timeout=5,
    )

    assert first.conversation_id == retry.conversation_id
    async with database.sessions() as session:
        assert await session.scalar(
            select(func.count()).select_from(ConversationRecord)
        ) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_transition_old_key_conflict_cannot_replay_a_different_scenario(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/transition-scope.db")
    await database.create_schema()
    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE UNIQUE INDEX uq_conversation_learning_session_round "
                "ON conversations (learner_id, learning_session_id, conversation_round)"
            )
        )
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    shared_identity = {
        "learner_id": 904,
        "scene": "cafe",
        "learning_session_id": "cafe-visit-904",
        "conversation_round": 1,
    }
    await service.create_conversation(
        SessionCreate(
            **shared_identity,
            scenario_id="cafe_queue",
            queue_context={"left_count": 2, "right_count": 4},
        )
    )

    with pytest.raises(IntegrityError):
        await service.create_conversation(
            SessionCreate(
                **shared_identity,
                scenario_id="cafe_budget_menu",
                cafe_context=CAFE_CONTEXT,
            )
        )

    await database.dispose()

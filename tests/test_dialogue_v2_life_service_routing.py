from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway

from mormi_api.content import representative_park_context
from mormi_api.db import Database
from mormi_api.dialogue_v2_life_runtime import DialogueV2LifeEngine
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DialogueRuntimeContractVersion,
    ResponseType,
    SessionCreate,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


class _UnusedV2Gateway:
    async def understand_v2(self, _: object) -> object:  # pragma: no cover
        raise AssertionError("initialization and snapshot must not call understanding")

    async def speak_v2(self, _: object) -> object:  # pragma: no cover
        raise AssertionError("initialization and snapshot must not call speaker")

    async def bridge_speak_v2(self, _: object) -> object:  # pragma: no cover
        raise AssertionError("initialization and snapshot must not call bridge speaker")


def _service(
    repository: Repository,
    *,
    with_life_engine: bool = True,
) -> ConversationService:
    gateway = _UnusedV2Gateway()
    return ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
        v2_engine=DialogueV2Engine(gateway),  # type: ignore[arg-type]
        life_v2_engine=(
            DialogueV2LifeEngine(gateway)  # type: ignore[arg-type]
            if with_life_engine
            else None
        ),
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        dialogue_v2_canary_percent=100,
        dialogue_v2_canary_salt="life-service-routing-test",
    )


@pytest.mark.asyncio
async def test_stable_cafe_session_pins_v3_and_retries_same_conversation(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/life-cafe-routing.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = _service(repository)
    request = SessionCreate(
        learner_id=801,
        scene="cafe",
        scenario_id="cafe_queue",
        learning_session_id="visit-cafe-queue-801",
        queue_context={"left_count": 2, "right_count": 4},
    )

    started = await service.create_conversation(request)
    state = await repository.get_state(started.conversation_id)
    retried = await service.create_conversation(request)

    assert retried.conversation_id == started.conversation_id
    assert state.runtime_contract_version is DialogueRuntimeContractVersion.VERDICT_V1
    assert state.pinned_dialogue_v2 is None
    assert state.pinned_dialogue_scenario_v3 is not None
    assert state.pinned_dialogue_scenario_v3.scenario_pack_id == "cafe.queue.v2"
    assert (
        state.pinned_dialogue_scenario_v3.selector_reason
        == "native_life_pack_canary_selected"
    )
    assert set(state.pinned_dialogue_scenario_v3.reasoning_ledgers) == set(
        state.task_ids
    )
    assert (await service.snapshot(started.conversation_id)).turn.task_anchor is not None
    advanced = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.NO_RESPONSE,
        ),
    )
    advanced_state = await repository.get_state(started.conversation_id)
    assert advanced.turn.state_version == 2
    assert advanced_state.pinned_dialogue_v2 is None
    assert advanced_state.pinned_dialogue_scenario_v3 is not None
    await database.dispose()


@pytest.mark.asyncio
async def test_life_scene_without_learning_session_identity_remains_legacy(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/life-legacy-routing.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = _service(repository)

    started = await service.create_conversation(
        SessionCreate(
            learner_id=802,
            scene="cafe",
            scenario_id="cafe_queue",
            queue_context={"left_count": 1, "right_count": 3},
        )
    )
    state = await repository.get_state(started.conversation_id)

    assert state.runtime_contract_version is DialogueRuntimeContractVersion.LEGACY_V1
    assert state.pinned_dialogue_v2 is None
    assert state.pinned_dialogue_scenario_v3 is None
    await database.dispose()


@pytest.mark.asyncio
async def test_park_request_context_reaches_materializer_and_pins_life_engine(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/life-park-routing.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = _service(repository)
    legacy_context = representative_park_context("amusement_ticket_multiply")
    expected_givens = {
        fact.key: fact.value
        for fact in legacy_context.facts
        if fact.key in {"ticket_price", "party_count"}
    }

    started = await service.create_conversation(
        SessionCreate(
            learner_id=803,
            scene="amusement_park",
            scenario_id="amusement_ticket_multiply",
            learning_session_id="visit-amusement-ticket-803",
            park_context=legacy_context,
        )
    )
    state = await repository.get_state(started.conversation_id)
    actual_facts = {
        item["key"]: item["value"]
        for item in state.scenario_data["park_context"]["facts"]
    }

    assert {key: actual_facts[key] for key in expected_givens} == expected_givens
    assert state.pinned_dialogue_scenario_v3 is not None
    assert state.pinned_dialogue_scenario_v3.scenario_pack_id.startswith(
        "amusement.amusement_ticket_multiply."
    )
    assert len(state.pinned_dialogue_scenario_v3.reasoning_ledgers) == 2
    assert started.turn.task_anchor is not None
    await database.dispose()


@pytest.mark.asyncio
async def test_persisted_life_snapshot_fails_closed_without_life_engine(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/life-fail-closed.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    selecting_service = _service(repository)
    started = await selecting_service.create_conversation(
        SessionCreate(
            learner_id=804,
            scene="cafe",
            scenario_id="cafe_queue",
            learning_session_id="visit-cafe-fail-closed-804",
            queue_context={"left_count": 2, "right_count": 5},
        )
    )

    unavailable_service = _service(repository, with_life_engine=False)
    with pytest.raises(RuntimeError, match="life V2 engine is unavailable"):
        await unavailable_service.snapshot(started.conversation_id)
    await database.dispose()


def test_verdict_state_with_both_snapshot_formats_is_rejected() -> None:
    # This invariant is exercised through the dispatcher rather than by adding
    # a cross-field validator that could make historical SessionState unreadable.
    service = object.__new__(ConversationService)
    service.engine = object()
    service.v2_engine = object()
    service.life_v2_engine = object()
    state = type("State", (), {})()
    state.runtime_contract_version = DialogueRuntimeContractVersion.VERDICT_V1
    state.pinned_dialogue_v2 = {"format": "v2"}
    state.pinned_dialogue_scenario_v3 = {"format": "v3"}

    with pytest.raises(RuntimeError, match="ambiguous pinned runtimes"):
        service._engine_for_state(state)  # type: ignore[arg-type]


def test_service_with_both_engines_advertises_v3_aggregate_reader() -> None:
    service = object.__new__(ConversationService)
    service.v2_engine = object()
    service.life_v2_engine = object()

    assert service.dialogue_snapshot_reader_capabilities == (
        "dialogue-v2-snapshot-reader-v2",
        "dialogue-v3-snapshot-reader-v1",
    )

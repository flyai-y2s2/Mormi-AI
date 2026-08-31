from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from conftest import FakeGateway
from test_dialogue_v2_graph_lifecycle import GatedGateway
from test_dialogue_v2_service_routing import _home_request

from mormi_api.db import Database
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import ConversationEngine
from mormi_api.main import respond_stream
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, DialogueRuntimeContractVersion, ResponseType
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


@pytest.mark.parametrize("http", [False, True])
async def test_closing_service_stream_closes_graph_without_model_or_commit(
    tmp_path: Any, http: bool
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/cancel.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("synthetic-key"))
    gateway = GatedGateway("none")
    engine = DialogueV2Engine(gateway)
    engine.run_turn_stream = engine._run_turn_graph
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),
        v2_engine=engine,
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        dialogue_v2_canary_percent=100,
    )
    try:
        first = await service.create_conversation(_home_request())
        response = ChildResponse(
            turn_id=first.turn.turn_id, response_id=uuid4(), type=ResponseType.TEXT, text="잠깐만"
        )
        if http:
            http_response = await respond_stream(first.conversation_id, response, None, service)
            events = http_response.body_iterator
            assert "response.accepted" in await anext(events)
            assert '"stage":"understanding"' in await anext(events)
        else:
            events = service.respond_stream(first.conversation_id, response)
            assert (await anext(events)).name == "accepted"
            assert (await anext(events)).stage == "understanding"
        await events.aclose()
        assert gateway.understanding_requests == []
        assert not [t for t in asyncio.all_tasks() if t.get_name() == "mormi-v2-turn"]
        assert (await service.snapshot(first.conversation_id)).turn == first.turn
        assert (
            await repository.response_exists(first.conversation_id, str(response.response_id))
            is None
        )
    finally:
        await database.dispose()

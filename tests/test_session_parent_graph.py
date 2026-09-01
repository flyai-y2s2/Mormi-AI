from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from session_parent_support import service
from test_dialogue_v2_life_runtime import LifeRuntimeGateway
from test_dialogue_v2_service_routing import _home_request

from mormi_api.db import Database
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, ResponseType
from mormi_api.security import TextCipher
from mormi_api.session_parent_graph import build_session_parent_graph
from mormi_api.session_parent_store import SessionParentStore


async def test_parent_wait_resume_across_fresh_service_instances(tmp_path: Path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/parent.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("synthetic"))
    gateway = LifeRuntimeGateway()
    app = service(repository, gateway)
    try:
        first = await app.create_conversation(_home_request())
        cursor = await SessionParentStore(database).load(first.conversation_id)
        assert cursor is not None and cursor.checkpoint is None
        sizes = []
        for i in range(40):
            response = ChildResponse(
                turn_id=first.turn.turn_id, response_id=uuid4(), type=ResponseType.NO_RESPONSE
            )
            app = service(repository, gateway, percent=0)
            first = await app.respond(first.conversation_id, response)
            cursor = await SessionParentStore(database).load(first.conversation_id)
            assert cursor is not None and cursor.checkpoint is not None
            assert cursor.phase == "waiting"
            assert cursor.state_version == first.turn.state_version == i + 2
            assert cursor.turn_id == first.turn.turn_id
            assert cursor.checkpoint["checkpoint"]["channel_values"]["response_id"] == ""
            assert "mormi_api" not in json.dumps(cursor.checkpoint)
            sizes.append(len(json.dumps(cursor.checkpoint)))
            assert await app.respond(first.conversation_id, response) == first
        assert gateway.understanding_requests == []
        # Wait/resume is bounded per invocation; neither the recursion budget nor
        # persisted checkpoint size grows with the complete conversation history.
        assert max(sizes) < 32768
        assert sizes[-1] < sizes[2] + 2000
    finally:
        await database.dispose()


def test_parent_graph_owns_human_input_loop_not_a_model_retry() -> None:
    graph = build_session_parent_graph()
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert edges == {
        ("__start__", "wait_for_input"),
        ("__start__", "__end__"),
        ("wait_for_input", "execute_turn"),
        ("execute_turn", "wait_for_input"),
        ("execute_turn", "__end__"),
    }

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from session_parent_support import service
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from test_dialogue_v2_fault_parity import complete_understanding
from test_dialogue_v2_graph_lifecycle import GatedGateway
from test_dialogue_v2_life_runtime import LifeRuntimeGateway
from test_dialogue_v2_runtime import RecordingV2Gateway
from test_dialogue_v2_service_routing import _home_request

from mormi_api.db import Database, DialogueTurnObservationRecord, SessionParentRecord
from mormi_api.main import respond_stream
from mormi_api.repository import PersistenceError, Repository, StaleConversationError
from mormi_api.schemas import ChildResponse, ResponseType, UnderstandingResponseV2
from mormi_api.security import TextCipher
from mormi_api.service import InvalidTurnResponseError
from mormi_api.session_parent_store import SessionParentStore


@pytest.fixture
async def repository(tmp_path: Path) -> AsyncIterator[Repository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/lifecycle.db")
    await database.create_schema()
    try:
        yield Repository(database, TextCipher("synthetic"))
    finally:
        await database.dispose()


def child(turn_id: str, *, text: str | None = None) -> ChildResponse:
    return ChildResponse(
        turn_id=turn_id,
        response_id=uuid4(),
        type=ResponseType.TEXT if text else ResponseType.NO_RESPONSE,
        text=text,
    )


@pytest.mark.parametrize(
    "stage", ["accepted", "understanding", "planning", "speaking", "validating"]
)
@pytest.mark.parametrize("http", [False, True])
async def test_parent_close_preserves_prework_backpressure(
    repository: Repository,
    stage: str,
    http: bool,
) -> None:
    gateway = GatedGateway("none")
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())
    response = child(first.turn.turn_id, text="합성 발화")
    if http:
        output = await respond_stream(first.conversation_id, response, None, app)
        events = output.body_iterator
    else:
        events = app.respond_stream(first.conversation_id, response)
    async for event in events:
        observed = (
            ("response.accepted" in event if stage == "accepted" else f'"stage":"{stage}"' in event)
            if http
            else event.stage == stage
        )
        if observed:
            break
    else:
        raise AssertionError("expected progress milestone missing")
    await events.aclose()
    assert await app.snapshot(first.conversation_id) == first
    assert (
        await repository.response_exists(first.conversation_id, str(response.response_id)) is None
    )
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("mormi-")]
    if stage in {"accepted", "understanding"}:
        assert gateway.understanding_requests == []
    if stage in {"accepted", "understanding", "planning", "speaking"}:
        assert gateway.speaker_plans == []


@pytest.mark.parametrize("stage", ["understanding", "speaking", "note"])
async def test_inflight_parent_cancellation_does_not_commit(
    repository: Repository,
    stage: str,
) -> None:
    gateway = GatedGateway(stage)
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())
    response = child(first.turn.turn_id, text="합성 학습 설명")

    async def consume() -> None:
        await app.respond(first.conversation_id, response)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(gateway.entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gateway.cancelled.is_set()
    assert await app.snapshot(first.conversation_id) == first
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("mormi-")]


async def test_checkpoint_write_failure_after_commit_replays_and_repairs(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = LifeRuntimeGateway()
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())
    original = app.session_parent.store.publish

    async def fail(*args: Any, **kwargs: Any) -> bool:
        raise SQLAlchemyError("synthetic private DB detail must not be logged")

    monkeypatch.setattr(app.session_parent.store, "publish", fail)
    response = child(first.turn.turn_id)
    result = await app.respond(first.conversation_id, response)
    assert result.turn.state_version == first.turn.state_version + 1
    count_before = len(gateway.speaker_plans)
    assert await app.respond(first.conversation_id, response) == result
    assert len(gateway.speaker_plans) == count_before
    cursor = await SessionParentStore(repository.database).load(first.conversation_id)
    assert cursor is not None and cursor.state_version == 1
    monkeypatch.setattr(app.session_parent.store, "publish", original)
    second = await app.respond(first.conversation_id, child(result.turn.turn_id))
    cursor = await SessionParentStore(repository.database).load(first.conversation_id)
    assert cursor is not None and cursor.state_version == second.turn.state_version == 3


async def test_domain_commit_failure_does_not_publish_or_retry(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = service(repository, LifeRuntimeGateway())
    first = await app.create_conversation(_home_request())
    calls = 0

    async def fail(**kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise PersistenceError("synthetic_failure")

    original = repository.commit_turn
    monkeypatch.setattr(repository, "commit_turn", fail)
    response = child(first.turn.turn_id)
    with pytest.raises(PersistenceError, match="synthetic_failure"):
        await app.respond(first.conversation_id, response)
    assert calls == 1
    assert await app.snapshot(first.conversation_id) == first
    cursor = await app.session_parent.store.load(first.conversation_id)
    assert cursor.checkpoint is None and cursor.generation == 1
    monkeypatch.setattr(repository, "commit_turn", original)
    result = await app.respond(first.conversation_id, response)
    assert result.turn.state_version == 2


@pytest.mark.parametrize("operation", ["load", "publish"])
async def test_slow_optional_cursor_io_cannot_hold_child_response(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    app = service(repository, LifeRuntimeGateway())
    app.session_parent.store_timeout_seconds = 0.01
    first = await app.create_conversation(_home_request())
    cancelled = asyncio.Event()

    async def hang(*args: Any, **kwargs: Any) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(app.session_parent.store, operation, hang)
    response = child(first.turn.turn_id)
    result = await asyncio.wait_for(app.respond(first.conversation_id, response), 2)
    assert cancelled.is_set()
    assert result.turn.state_version == 2
    assert await app.respond(first.conversation_id, response) == result


async def test_parent_bootstrap_failure_uses_existing_service_once(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = GatedGateway("none")
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic graph bootstrap error")

    monkeypatch.setattr(app.session_parent.graph, "ainvoke", fail)
    result = await app.respond(first.conversation_id, child(first.turn.turn_id, text="합성 발화"))
    assert result.turn.state_version == 2
    assert len(gateway.understanding_requests) == 1


async def test_parent_projection_error_after_commit_never_repeats_domain_work(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mormi_api.session_parent_graph as parent_module

    gateway = GatedGateway("none")
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic packet codec error")

    monkeypatch.setattr(parent_module, "export_boundary", fail)
    response = child(first.turn.turn_id, text="합성 발화")
    result = await app.respond(first.conversation_id, response)
    assert result.turn.state_version == 2
    assert len(gateway.understanding_requests) == 1
    assert await app.respond(first.conversation_id, response) == result


@pytest.mark.parametrize("corruption", ["missing", "malformed", "wrong_turn", "other_session"])
async def test_cursor_corruption_recovers_from_db(
    repository: Repository,
    corruption: str,
) -> None:
    app = service(repository, LifeRuntimeGateway())
    first = await app.create_conversation(_home_request())
    current = await app.respond(first.conversation_id, child(first.turn.turn_id))
    store = SessionParentStore(repository.database)
    cursor = await store.load(first.conversation_id)
    assert cursor is not None and cursor.checkpoint is not None
    packet = cursor.checkpoint
    if corruption == "missing":
        packet = None
    elif corruption == "malformed":
        packet = {"format": 1, "checkpoint": {}}
    elif corruption == "wrong_turn":
        packet["checkpoint"]["channel_values"]["turn_id"] = "old_turn"
    else:
        packet["checkpoint"]["channel_values"]["conversation_id"] = "another_session"
    async with repository.database.sessions() as db:
        await db.execute(update(SessionParentRecord).values(checkpoint=packet))
        await db.commit()
    result = await service(repository, LifeRuntimeGateway(), percent=0).respond(
        current.conversation_id, child(current.turn.turn_id)
    )
    repaired = await store.load(current.conversation_id)
    assert repaired is not None and repaired.turn_id == result.turn.turn_id
    assert repaired.state_version == 3


async def test_new_conversations_only_and_emergency_bypass(repository: Repository) -> None:
    gateway = LifeRuntimeGateway()
    old = service(repository, gateway, parent=False)
    first = await old.create_conversation(_home_request())
    enabled = service(repository, gateway)
    assert await enabled.create_conversation(_home_request()) == first
    assert await enabled.session_parent.store.load(first.conversation_id) is None
    second = await enabled.create_conversation(_home_request(conversation_round=2))
    assert second.conversation_id != first.conversation_id
    assert await enabled.session_parent.store.load(second.conversation_id) is not None
    result = await old.respond(second.conversation_id, child(second.turn.turn_id))
    # Re-enable without enrolling any other conversations, reconstruct from DB.
    resumed = await service(repository, gateway, percent=0).respond(
        result.conversation_id, child(result.turn.turn_id)
    )
    assert resumed.turn.state_version == 3


@pytest.mark.parametrize(
    ("consent", "retention"),
    [(False, "no_raw"), (True, "permanent"), (True, "30_days"), (True, "90_days")],
)
async def test_child_text_never_enters_parent_packets(
    repository: Repository,
    consent: bool,
    retention: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No live tracing even if deployment-wide tracing variables are enabled.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    sentinel = "원문검사표식_따로저장하면안되는아이의발화"
    gateway = RecordingV2Gateway([UnderstandingResponseV2(utterance_class="learning_response")])
    app = service(repository, gateway)
    request = _home_request()
    request = type(request).model_validate(
        {
            **request.model_dump(mode="json"),
            "conversation_storage_consent": consent,
            "retention_policy": retention,
        }
    )
    first = await app.create_conversation(request)
    await app.respond(first.conversation_id, child(first.turn.turn_id, text=sentinel))
    cursor = await app.session_parent.store.load(first.conversation_id)
    assert cursor is not None and cursor.checkpoint is not None
    packet = json.dumps(cursor.checkpoint, ensure_ascii=False)
    assert sentinel not in packet
    for forbidden in ("child_utterance", "speaker_plan", "recent_dialogue", "source_spans"):
        assert forbidden not in packet
    assert app.v2_engine._turn_graph.checkpointer is False
    assert app.v2_engine._understanding_graph.checkpointer is False
    assert app.v2_engine._speaker_graph.checkpointer is False


async def test_old_writer_cannot_overwrite_new_cursor(repository: Repository) -> None:
    app = service(repository, LifeRuntimeGateway())
    first = await app.create_conversation(_home_request())
    stale = await app.session_parent.store.load(first.conversation_id)
    second = await app.respond(first.conversation_id, child(first.turn.turn_id))
    current = await app.session_parent.store.load(first.conversation_id)
    assert not await app.session_parent.store.publish(
        stale,
        state_version=current.state_version,
        turn_id=current.turn_id,
        phase=current.phase,
        checkpoint=current.checkpoint,
    )
    third = await app.respond(second.conversation_id, child(second.turn.turn_id))
    assert not await app.session_parent.store.publish(
        current,
        state_version=current.state_version,
        turn_id=current.turn_id,
        phase=current.phase,
        checkpoint=current.checkpoint,
    )
    assert (
        await app.session_parent.store.load(first.conversation_id)
    ).turn_id == third.turn.turn_id


async def test_stale_input_does_not_advance_parent(repository: Repository) -> None:
    app = service(repository, LifeRuntimeGateway())
    first = await app.create_conversation(_home_request())
    second = await app.respond(first.conversation_id, child(first.turn.turn_id))
    before = await app.session_parent.store.load(first.conversation_id)
    with pytest.raises(InvalidTurnResponseError, match="stale"):
        await app.respond(first.conversation_id, child(first.turn.turn_id))
    assert await app.snapshot(first.conversation_id) == second
    assert await app.session_parent.store.load(first.conversation_id) == before


async def test_concurrent_request_local_savers_do_not_share_pending_input(
    repository: Repository,
) -> None:
    entered = 0
    both = asyncio.Event()
    commit_lock = asyncio.Lock()
    original_commit = repository.commit_turn

    async def serialized_commit(**kwargs: Any) -> None:
        # SQLite lacks PostgreSQL FOR UPDATE; model the existing commit lock,
        # NOT a parent lock, while letting both parent/model executions overlap.
        async with commit_lock:
            await original_commit(**kwargs)

    repository.commit_turn = serialized_commit  # type: ignore[method-assign]

    class Gateway(RecordingV2Gateway):
        async def understand_v2(self, request: Any) -> UnderstandingResponseV2:
            nonlocal entered
            entered += 1
            if entered == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 3)
            return complete_understanding(request)

    gateway = Gateway()
    app = service(repository, gateway)
    first = await app.create_conversation(_home_request())
    response = child(first.turn.turn_id, text="합성 설명")
    results = await asyncio.gather(
        app.respond(first.conversation_id, response),
        service(repository, gateway, percent=0).respond(first.conversation_id, response),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, BaseException) for r in results) == 1
    assert sum(isinstance(r, StaleConversationError) for r in results) == 1
    async with repository.database.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(DialogueTurnObservationRecord)) == 1
    assert (await repository.get_state(first.conversation_id)).state_version == 2

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any

import pytest
from langsmith.run_helpers import get_tracing_context
from test_dialogue_v2_fault_parity import complete_understanding
from test_dialogue_v2_runtime import RecordingV2Gateway, _initialize, _response, _state

from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import EngineProgress, EngineTurnResult
from mormi_api.schemas import ResponseType, UnderstandingResponseV2


class GatedGateway(RecordingV2Gateway):
    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage
        self.entered, self.release, self.cancelled = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

    async def gate(self, stage: str) -> None:
        if stage == self.stage:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def understand_v2(self, request: Any) -> UnderstandingResponseV2:
        self.understanding_requests.append(request)
        await self.gate("understanding")
        if self.stage == "note":
            return complete_understanding(request)
        return UnderstandingResponseV2(utterance_class="learning_response")

    async def speak_v2(self, plan: Any) -> Any:
        await self.gate("speaking")
        return await super().speak_v2(plan)

    async def contextualize_note(self, context: Any) -> Any:
        await self.gate("note")
        return await super().contextualize_note(context)


async def first_turn(engine: Any) -> tuple[Any, Any, Any]:
    state = _state("divide-share")
    turn = await _initialize(engine, state, "divide-share")
    response = _response(turn.turn_id, ResponseType.TEXT, text="10500을 3으로 나누면 3500원")
    return state, turn, response


async def test_graph_progress_preserves_demand_driven_execution() -> None:
    gateway = GatedGateway("none")
    engine = DialogueV2Engine(gateway)
    state, first, response = await first_turn(engine)
    events = engine.run_turn_stream(state, response, first.mormi.text)
    assert (await anext(events)).stage == "understanding"
    assert gateway.understanding_requests == []
    await events.aclose()
    assert gateway.understanding_requests == []
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mormi-v2-turn"]


@pytest.mark.parametrize("stage", ["understanding", "speaking", "note"])
async def test_cancelling_graph_cancels_inflight_call_without_advancing_state(stage: str) -> None:
    gateway = GatedGateway(stage)
    engine = DialogueV2Engine(gateway)
    state, first, response = await first_turn(engine)
    before = state.model_dump(mode="json")
    events = engine.run_turn_stream(state, response, first.mormi.text)

    async def consume() -> None:
        async for event in events:
            assert isinstance(event, EngineProgress), "cancelled turn must not finish"

    task = asyncio.create_task(consume())
    await asyncio.wait_for(gateway.entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await events.aclose()
    assert gateway.cancelled.is_set()
    assert state.model_dump(mode="json") == before
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mormi-v2-turn"]


@pytest.mark.parametrize("stop_at", ["planning", "speaking", "validating"])
async def test_closing_at_each_milestone_does_not_execute_next_step(
    stop_at: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = GatedGateway("none")
    engine = DialogueV2Engine(gateway)
    state, first, response = await first_turn(engine)
    called: list[str] = []
    method_name = {
        "planning": "_apply_execution",
        "speaking": "_render_execution_speech",
        "validating": "_compose_execution",
    }[stop_at]
    original = getattr(engine, method_name)
    if stop_at == "speaking":

        async def wrapped(turn: Any) -> None:
            called.append(stop_at)
            await original(turn)
    else:

        def wrapped(turn: Any) -> None:
            called.append(stop_at)
            original(turn)

    monkeypatch.setattr(engine, method_name, wrapped)
    events = engine.run_turn_stream(state, response, first.mormi.text)
    async for event in events:
        if isinstance(event, EngineProgress) and event.stage == stop_at:
            break
    assert called == []
    await events.aclose()
    assert called == []


async def test_shared_graph_keeps_concurrent_turns_and_context_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope: ContextVar[str] = ContextVar("parity_test_scope")
    entered: list[str] = []
    both_entered = asyncio.Event()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    class ConcurrentGateway(RecordingV2Gateway):
        async def understand_v2(self, request: Any) -> UnderstandingResponseV2:
            assert get_tracing_context()["enabled"] is False
            assert scope.get() == request.child_utterance
            entered.append(scope.get())
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 2)
            return UnderstandingResponseV2(utterance_class="learning_response")

    engine = DialogueV2Engine(ConcurrentGateway())

    async def run(label: str) -> EngineTurnResult:
        scope.set(label)
        state, first, _ = await first_turn(engine)
        response = _response(first.turn_id, ResponseType.TEXT, text=label)
        before = state.model_dump(mode="json")
        events = [e async for e in engine.run_turn_stream(state, response, first.mormi.text)]
        assert state.model_dump(mode="json") == before
        result = events[-1]
        assert isinstance(result, EngineTurnResult)
        return result

    left, right = await asyncio.gather(run("첫 번째 합성 발화"), run("두 번째 합성 발화"))
    assert left.state.conversation_id != right.state.conversation_id
    assert left.state.current_turn_id != right.state.current_turn_id
    assert left.state.state_version == right.state.state_version == 2
    assert engine._turn_graph.checkpointer is None
    assert engine._understanding_graph.checkpointer is None
    assert engine._speaker_graph.checkpointer is None


def test_retry_subgraphs_have_only_the_existing_bounded_loop() -> None:
    engine = DialogueV2Engine(RecordingV2Gateway())
    for graph in (engine._understanding_graph, engine._speaker_graph):
        edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
        assert ("attempt", "attempt") in edges
        assert ("attempt", "exhausted") in edges
        assert ("exhausted", "__end__") in edges


async def test_internal_diagnostics_never_include_frame_or_child_text(caplog: Any) -> None:
    import logging

    from mormi_api.observability import TurnScope, turn_scope

    caplog.set_level(logging.DEBUG, logger="mormi_api.orchestration")
    engine = DialogueV2Engine(GatedGateway("none"))
    state, first, response = await first_turn(engine)
    token = turn_scope.set(TurnScope(state.conversation_id, first.turn_id))
    try:
        _ = [event async for event in engine.run_turn_stream(state, response, first.mormi.text)]
    finally:
        turn_scope.reset(token)
    assert "graph_step" in caplog.text
    assert state.conversation_id in caplog.text
    assert response.text not in caplog.text
    assert "visible_facts" not in caplog.text
    assert "child_utterance" not in caplog.text

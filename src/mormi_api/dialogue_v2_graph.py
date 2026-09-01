"""Ephemeral V2 orchestration. Domain state is still committed only by the service.

No checkpoints, graph retries, parallel nodes, raw state logging or external traces.
The progress rendezvous preserves the old async generator's backpressure: work
starts only when the consumer resumes after receiving its pre-work milestone.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .dialogue_v2_graph_trace import trace_node
from .dialogue_v2_runtime import DialogueV2Engine, _TurnExecution
from .engine import EngineProgress, EngineTurnResult
from .schemas import ResponseType


class TurnGraphState(TypedDict):
    turn: _TurnExecution
    emit: Callable[[str], Awaitable[None]]


def build_turn_graph(engine: DialogueV2Engine) -> Any:
    builder = StateGraph(TurnGraphState)

    def node(method_name: str, milestone: str | None = None) -> Any:
        async def execute(state: TurnGraphState) -> dict[str, _TurnExecution]:
            turn = state["turn"]
            if milestone is not None:
                await state["emit"](milestone)
            with trace_node("turn", method_name):
                result = getattr(engine, method_name)(turn)
                if inspect.isawaitable(result):
                    await result
            return {"turn": turn}

        return execute

    async def interpret(state: TurnGraphState) -> dict[str, _TurnExecution]:
        if state["turn"].response.type is ResponseType.TEXT:
            await state["emit"]("understanding")
        with trace_node("turn", state["turn"].response.type.value):
            await engine._interpret_execution(state["turn"])
        return {"turn": state["turn"]}

    async def after_policy(state: TurnGraphState) -> str:
        semantics = state["turn"].semantics
        assert semantics is not None
        return "complete" if semantics.apply_result.completion.complete else "prepare_speech"

    async def speech_route(state: TurnGraphState) -> str:
        return engine._execution_speech_route(state["turn"])

    input_nodes = {
        "text": "understand_text",
        "no_response": "explicit_support",
        "choice": "structured_choice",
        "fill": "structured_fill",
        "action": "joint_action",
    }

    async def input_route(state: TurnGraphState) -> str:
        return input_nodes.get(state["turn"].response.type.value, "invalid_input")

    for input_node in [*input_nodes.values(), "invalid_input"]:
        builder.add_node(input_node, interpret)
        builder.add_edge(input_node, "apply_progress")
    builder.add_node("apply_progress", node("_apply_execution", "planning"))
    builder.add_node("complete", node("_complete_execution"))
    builder.add_node("prepare_speech", node("_prepare_execution_speech"))
    for route in ("main", "bridge", "stable", "safety", "understanding_fallback"):
        builder.add_node(
            route,
            node(
                "_render_execution_speech",
                "speaking" if route in {"main", "bridge"} else None,
            ),
        )
        builder.add_edge(route, "compose")
    builder.add_node("compose", node("_compose_execution", "validating"))
    builder.add_node("note", node("_note_execution"))
    builder.add_conditional_edges(
        START,
        input_route,
        {n: n for n in [*input_nodes.values(), "invalid_input"]},
    )
    builder.add_conditional_edges(
        "apply_progress",
        after_policy,
        {
            "complete": "complete",
            "prepare_speech": "prepare_speech",
        },
    )
    builder.add_conditional_edges(
        "prepare_speech",
        speech_route,
        {
            route: route
            for route in ("main", "bridge", "stable", "safety", "understanding_fallback")
        },
    )
    builder.add_edge("complete", "compose")
    builder.add_edge("compose", "note")
    builder.add_edge("note", END)
    # None inherits a parent's saver. False explicitly keeps child text and
    # transient execution frames out of the session parent's checkpoints.
    return builder.compile(checkpointer=False, name="mormi_v2_turn")


async def stream_turn_graph(
    graph: Any,
    turn: _TurnExecution,
) -> AsyncGenerator[EngineProgress | EngineTurnResult, None]:
    pending: asyncio.Queue[tuple[EngineProgress, asyncio.Future[None]]] = asyncio.Queue(1)

    async def emit(stage: str) -> None:
        acknowledged: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await pending.put((EngineProgress(stage), acknowledged))  # type: ignore[arg-type]
        await acknowledged

    async def invoke() -> None:
        # The frame contains transient child text. Even a tracing-enabled host
        # must not copy it to LangSmith or another external callback destination.
        with tracing_context(enabled=False):
            await graph.ainvoke({"turn": turn, "emit": emit}, config={"callbacks": []})

    running = asyncio.create_task(invoke(), name="mormi-v2-turn")
    receiving: asyncio.Task[tuple[EngineProgress, asyncio.Future[None]]] | None = None
    try:
        while True:
            receiving = asyncio.create_task(pending.get())
            done, _ = await asyncio.wait({running, receiving}, return_when=asyncio.FIRST_COMPLETED)
            if receiving in done:
                event, acknowledged = receiving.result()
                yield event
                acknowledged.set_result(None)
            else:
                receiving.cancel()
                await asyncio.gather(receiving, return_exceptions=True)
                await running  # Preserve the original exception, including cancellation.
                assert turn.result is not None
                yield turn.result
                return
    finally:
        # Closing a paused stream must not leave the graph running in the background.
        if receiving is not None and not receiving.done():
            receiving.cancel()
        if not running.done():
            running.cancel()
        await asyncio.gather(
            running, *([receiving] if receiving is not None else []), return_exceptions=True
        )

"""Session lifecycle graph; domain pedagogy stays in the existing turn service.

START -> wait_for_input -> execute_turn -> wait_for_input / END.
Only a committed SessionStatus.COMPLETED ends the parent. A life-task transition
remains ACTIVE. Human input is an interrupt, never an autonomous model loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langsmith import tracing_context
from sqlalchemy.exc import SQLAlchemyError

from .schemas import SessionState, SessionStatus
from .session_parent_checkpoint import (
    BoundarySaver,
    export_boundary,
    import_boundary,
    parent_config,
    staged_saver,
)
from .session_parent_store import PARENT_GRAPH_VERSION, ParentCursor, SessionParentStore

if TYPE_CHECKING:
    from .service import ConversationStreamEvent

logger = logging.getLogger(__name__)


class ParentState(TypedDict):
    conversation_id: str
    graph_version: str
    state_version: int
    turn_id: str
    phase: str
    response_id: str


@dataclass
class ParentRequest:
    """Transient invocation context, NEVER a graph channel or checkpoint value."""

    response_id: str
    run_turn: Callable[[], AsyncGenerator[ConversationStreamEvent, None]]
    emit: Callable[[ConversationStreamEvent], Awaitable[None]]
    result: ConversationStreamEvent | None = None
    started: bool = False


def build_session_parent_graph() -> Any:
    builder = StateGraph(ParentState, context_schema=ParentRequest)

    def wait_for_input(state: ParentState, runtime: Runtime[ParentRequest]) -> dict[str, str]:
        response = interrupt({"turn_id": state["turn_id"]})
        if response != {"response_id": runtime.context.response_id}:
            raise ValueError("parent resume token does not match request")
        return {"response_id": runtime.context.response_id}

    async def execute_turn(state: ParentState, runtime: Runtime[ParentRequest]) -> ParentState:
        request = runtime.context
        request.started = True
        async with aclosing(request.run_turn()) as events:
            async for event in events:
                if event.envelope is None:
                    await request.emit(event)
                else:
                    request.result = event
        if request.result is None or request.result.envelope is None:
            raise RuntimeError("turn service produced no result")
        turn = request.result.envelope.turn
        # No second completion decision, note emission, task advance or profile write.
        return {
            **state,
            "turn_id": turn.turn_id,
            "state_version": turn.state_version,
            "phase": "completed" if turn.status is SessionStatus.COMPLETED else "waiting",
            "response_id": "",
        }

    def route(state: ParentState) -> str:
        return END if state["phase"] == "completed" else "wait_for_input"

    builder.add_node("wait_for_input", wait_for_input)
    builder.add_node("execute_turn", execute_turn)
    builder.add_conditional_edges(START, route, {END: END, "wait_for_input": "wait_for_input"})
    builder.add_edge("wait_for_input", "execute_turn")
    builder.add_conditional_edges(
        "execute_turn", route, {END: END, "wait_for_input": "wait_for_input"}
    )
    return builder.compile(checkpointer=BoundarySaver(), name="mormi_session_parent_v1")


class SessionParentCoordinator:
    def __init__(self, store: SessionParentStore, *, store_timeout_seconds: float = 0.5) -> None:
        self.store = store
        self.store_timeout_seconds = store_timeout_seconds
        self.graph = build_session_parent_graph()

    async def stream(
        self,
        state: SessionState,
        cursor: ParentCursor,
        response_id: str,
        run_turn: Callable[[], AsyncGenerator[ConversationStreamEvent, None]],
    ) -> AsyncGenerator[ConversationStreamEvent, None]:
        started = time.perf_counter()
        assert state.current_turn_id is not None
        config = parent_config(state.conversation_id)
        pointer: ParentState = {
            "conversation_id": state.conversation_id,
            "graph_version": PARENT_GRAPH_VERSION,
            "state_version": state.state_version,
            "turn_id": state.current_turn_id,
            "phase": "completed" if state.status is SessionStatus.COMPLETED else "waiting",
            "response_id": "",
        }
        queue: asyncio.Queue[tuple[ConversationStreamEvent, asyncio.Future[None]]] = asyncio.Queue(
            1
        )

        async def emit(event: ConversationStreamEvent) -> None:
            acknowledged: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            await queue.put((event, acknowledged))
            await acknowledged

        request = ParentRequest(response_id=response_id, run_turn=run_turn, emit=emit)

        async def invoke() -> None:
            saver = InMemorySaver()
            restored = False
            publication = "not_published"
            if cursor.checkpoint is not None:
                try:
                    candidate = import_boundary(cursor.checkpoint, config)
                    saved = candidate.get_tuple(config)
                    assert saved is not None
                    values = saved.checkpoint["channel_values"]
                    if all(values.get(k) == v for k, v in pointer.items()):
                        saver, restored = candidate, True
                except (AttributeError, KeyError, TypeError, ValueError):
                    logger.warning("session_parent_checkpoint_rebuild reason=invalid_packet")
            token = staged_saver.set(saver)
            try:
                with tracing_context(enabled=False):
                    if restored:
                        snapshot = await self.graph.aget_state(config)
                        expected = () if pointer["phase"] == "completed" else ("wait_for_input",)
                        if snapshot.next != expected:
                            staged_saver.set(InMemorySaver())
                            restored = False
                    if not restored:
                        await self.graph.ainvoke(pointer, config, context=request)
                    await self.graph.ainvoke(
                        Command(resume={"response_id": response_id}), config, context=request
                    )
                    packet = export_boundary(staged_saver.get(), config)
                values = packet["checkpoint"]["channel_values"]
                try:
                    async with asyncio.timeout(self.store_timeout_seconds):
                        published = await self.store.publish(
                            cursor,
                            state_version=values["state_version"],
                            turn_id=values["turn_id"],
                            phase=values["phase"],
                            checkpoint=packet,
                        )
                    if not published:
                        publication = "superseded"
                        logger.info("session_parent_checkpoint_deferred reason=concurrent_update")
                    else:
                        publication = "saved"
                except (SQLAlchemyError, TimeoutError):
                    publication = "storage_unavailable"
                    # Domain commit already succeeded. Return its original result;
                    # the next request repairs the cursor from the canonical DB.
                    logger.warning("session_parent_checkpoint_deferred reason=storage_unavailable")
            except Exception:
                if not request.started:
                    # No domain work has begun: a broken parent bootstrap must
                    # not remove the already-working turn service. Execute ONCE.
                    logger.warning("session_parent_bypassed reason=bootstrap_failure")
                    async with aclosing(request.run_turn()) as events:
                        async for event in events:
                            if event.envelope is None:
                                await request.emit(event)
                            else:
                                request.result = event
                    return
                if request.result is None:
                    raise  # Preserve existing domain/model/storage errors before a final result.
                # A completed turn must not become a new HTTP error just because
                # the parent library/packet projection failed afterward. Do not
                # rerun the service; recover from DB on the next explicit request.
                logger.warning("session_parent_checkpoint_deferred reason=boundary_projection")
                publication = "projection_failed"
            finally:
                staged_saver.reset(token)
                if request.result is not None:
                    logger.info(
                        "session_parent_turn conversation_id=%s graph_version=%s "
                        "restored=%s checkpoint=%s duration_ms=%d",
                        state.conversation_id,
                        PARENT_GRAPH_VERSION,
                        restored,
                        publication,
                        int((time.perf_counter() - started) * 1000),
                    )

        worker = asyncio.create_task(invoke(), name="mormi-session-parent")
        receiving: asyncio.Task[tuple[ConversationStreamEvent, asyncio.Future[None]]] | None = None
        try:
            while True:
                receiving = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {worker, receiving}, return_when=asyncio.FIRST_COMPLETED
                )
                if receiving in done:
                    event, acknowledged = receiving.result()
                    yield event
                    acknowledged.set_result(None)
                else:
                    receiving.cancel()
                    await asyncio.gather(receiving, return_exceptions=True)
                    await worker
                    if request.result is None:
                        raise RuntimeError("parent produced no turn result")
                    yield request.result
                    return
        finally:
            if receiving is not None and not receiving.done():
                receiving.cancel()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(
                worker, *([receiving] if receiving else []), return_exceptions=True
            )

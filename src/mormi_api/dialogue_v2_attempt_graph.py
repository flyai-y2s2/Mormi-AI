"""Two-attempt subgraph, preserving the runtime's existing retry ownership.

The operation alone decides whether a retry is allowed (None) or a terminal
result is ready. No RetryPolicy, backoff, timeout, or model call is added here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .dialogue_v2_graph_trace import trace_node


@dataclass(slots=True)
class AttemptInvocation[C, R]:
    context: C
    operation: Callable[[C, int], Awaitable[R | None]]
    exhausted: Callable[[C], R]
    attempt: int = 0
    result: R | None = None


class AttemptGraphState(TypedDict):
    invocation: AttemptInvocation[Any, Any]


def build_two_attempt_graph(name: str) -> Any:
    builder = StateGraph(AttemptGraphState)

    async def attempt(state: AttemptGraphState) -> AttemptGraphState:
        invocation = state["invocation"]
        invocation.attempt += 1
        with trace_node(name, "attempt", invocation.attempt):
            invocation.result = await invocation.operation(invocation.context, invocation.attempt)
        return state

    async def route(state: AttemptGraphState) -> str:
        invocation = state["invocation"]
        if invocation.result is not None:
            return END
        return "attempt" if invocation.attempt < 2 else "exhausted"

    async def exhausted(state: AttemptGraphState) -> AttemptGraphState:
        invocation = state["invocation"]
        invocation.result = invocation.exhausted(invocation.context)
        return state

    builder.add_node("attempt", attempt)
    builder.add_node("exhausted", exhausted)
    builder.add_edge(START, "attempt")
    builder.add_conditional_edges(
        "attempt",
        route,
        {
            "attempt": "attempt",
            "exhausted": "exhausted",
            END: END,
        },
    )
    builder.add_edge("exhausted", END)
    return builder.compile(checkpointer=False, name=name)


async def run_attempt_graph[C, R](
    graph: Any,
    context: C,
    operation: Callable[[C, int], Awaitable[R | None]],
    exhausted: Callable[[C], R],
) -> R:
    invocation = AttemptInvocation(context, operation, exhausted)
    with tracing_context(enabled=False):
        await graph.ainvoke({"invocation": invocation}, config={"callbacks": []})
    assert invocation.result is not None
    return invocation.result

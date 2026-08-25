"""Request-scoped correlation for timing logs.

`turn_scope` is set once per conversation turn in the service layer and read
by the LLM gateway so every `llm_call` line can be joined back to its `turn`
line even when many turns run concurrently.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnScope:
    conversation_id: str
    turn_id: str


turn_scope: ContextVar[TurnScope | None] = ContextVar("turn_scope", default=None)


def scope_fields() -> tuple[str, str]:
    """Return (conversation_id, turn_id) for log lines; "-" outside a turn."""
    scope = turn_scope.get()
    if scope is None:
        return "-", "-"
    return scope.conversation_id, scope.turn_id

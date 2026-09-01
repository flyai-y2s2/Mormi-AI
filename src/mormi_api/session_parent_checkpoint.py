"""Request-local staging + durable, raw-free WAIT/END checkpoint packets.

LangGraph intermediate writes are deliberately ephemeral. Only the latest
committed boundary is exported by the coordinator. No pickle, model text,
request context, exception, or arbitrary object is accepted by the packet codec.
The DB stores one bounded packet, not an ever-growing conversation history.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt

staged_saver: ContextVar[InMemorySaver] = ContextVar("session_parent_staged_saver")


class BoundarySaver(BaseCheckpointSaver[str]):
    """Compiled once; each request gets an isolated staging saver via ContextVar."""

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await staged_saver.get().aget_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await staged_saver.get().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await staged_saver.get().aput_writes(config, writes, task_id, task_path)

    def get_next_version(self, current: str | None, channel: None) -> str:
        # Version generation is pure: LangGraph also calls it when drawing the
        # graph outside a live request. Per-request savers plus the DB generation
        # fence isolate concurrent runs; a process-local counter is not needed.
        version = 0 if current is None else int(str(current).split(".", 1)[0])
        return f"{version + 1:032}"


def parent_config(conversation_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": conversation_id, "checkpoint_ns": ""},
        "callbacks": [],
        "recursion_limit": 12,
    }


def _check_values(values: dict[str, Any]) -> None:
    allowed = {
        "conversation_id",
        "graph_version",
        "state_version",
        "turn_id",
        "phase",
        "response_id",
        "branch:to:wait_for_input",
        "branch:to:execute_turn",
    }
    if set(values) - allowed:
        raise ValueError("unexpected parent checkpoint channel")
    for key, value in values.items():
        if key.startswith("branch:"):
            if value is not None:
                raise ValueError("unexpected parent branch value")
        elif key == "state_version":
            if type(value) is not int or value < 1:
                raise ValueError("invalid parent state version")
        elif not isinstance(value, str) or len(value) > 100:
            raise ValueError("invalid parent pointer value")
    if values.get("response_id") != "":
        raise ValueError("cannot persist a processing parent")
    if values.get("phase") not in {"waiting", "completed"}:
        raise ValueError("invalid parent boundary")


def export_boundary(saver: InMemorySaver, config: RunnableConfig) -> dict[str, Any]:
    saved = saver.get_tuple(config)
    if saved is None:
        raise ValueError("missing parent checkpoint")
    values = saved.checkpoint["channel_values"]
    _check_values(values)
    pending = []
    for task_id, channel, value in saved.pending_writes or []:
        if channel != "__interrupt__" or not isinstance(value, (tuple, list)):
            raise ValueError("cannot persist non-boundary writes")
        interrupts = []
        for item in value:
            if not isinstance(item, Interrupt) or item.value != {"turn_id": values["turn_id"]}:
                raise ValueError("unexpected parent interrupt")
            interrupts.append({"id": item.id, "value": item.value})
        pending.append({"task_id": task_id, "interrupts": interrupts})
    if (values["phase"] == "waiting" and len(pending) != 1) or (
        values["phase"] == "completed" and pending
    ):
        raise ValueError("checkpoint is not at WAIT or END")
    # Only library control metadata, not configurable values or request context.
    metadata = {k: v for k, v in saved.metadata.items() if k in {"source", "step", "parents"}}
    packet = {"format": 1, "checkpoint": saved.checkpoint, "metadata": metadata, "pending": pending}
    encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 32768:
        raise ValueError("parent checkpoint exceeds boundary budget")
    return json.loads(encoded)  # type: ignore[no-any-return]


def import_boundary(packet: dict[str, Any], config: RunnableConfig) -> InMemorySaver:
    """Decode only our JSON boundary format; incompatible packets are rebuildable."""
    if not isinstance(packet, dict) or packet.get("format") != 1:
        raise ValueError("unsupported parent checkpoint format")
    checkpoint = packet["checkpoint"]
    _check_values(checkpoint["channel_values"])
    if checkpoint["channel_values"]["conversation_id"] != config["configurable"]["thread_id"]:
        raise ValueError("checkpoint belongs to another conversation")
    saver = InMemorySaver()
    saved_config = saver.put(config, checkpoint, packet["metadata"], checkpoint["channel_versions"])
    for pending in packet["pending"]:
        values = tuple(Interrupt(value=i["value"], id=i["id"]) for i in pending["interrupts"])
        saver.put_writes(saved_config, [("__interrupt__", values)], pending["task_id"])
    # Validate the reconstituted packet before allowing the graph to execute.
    export_boundary(saver, config)
    return saver

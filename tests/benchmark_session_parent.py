"""Synthetic SQLite API/service overhead only. Never invokes a live LLM.

Run: PYTHONPATH=src:tests .venv/bin/python tests/benchmark_session_parent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from session_parent_support import service
from test_dialogue_v2_life_runtime import LifeRuntimeGateway
from test_dialogue_v2_service_routing import _home_request

from mormi_api.db import Database
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, ResponseType
from mormi_api.security import TextCipher


async def measure(root: Path, parent: bool) -> dict[str, float | int]:
    database = Database(f"sqlite+aiosqlite:///{root}/{parent}.db")
    await database.create_schema()
    gateway = LifeRuntimeGateway()
    app = service(Repository(database, TextCipher("synthetic")), gateway, reference=not parent)
    durations: list[float] = []
    try:
        for i in range(45):
            current = await app.create_conversation(_home_request(learning_session_id=f"bench-{i}"))
            for _ in range(2):
                response = ChildResponse(
                    turn_id=current.turn.turn_id, response_id=uuid4(), type=ResponseType.NO_RESPONSE
                )
                start = time.perf_counter()
                current = await app.respond(current.conversation_id, response)
                if i >= 5:
                    durations.append((time.perf_counter() - start) * 1000)
        return {
            "samples": len(durations),
            "median_ms": round(statistics.median(durations), 3),
            "p95_ms": round(statistics.quantiles(durations, n=100)[94], 3),
            "model_calls": len(gateway.understanding_requests) + len(gateway.speaker_plans),
        }
    finally:
        await database.dispose()


async def main() -> None:
    logging.getLogger("mormi_api").setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory(prefix="mormi-parent-bench-") as temporary:
        root = Path(temporary)
        baseline = await measure(root, False)
        parent = await measure(root, True)
        print(
            json.dumps(
                {
                    "scope": "synthetic SQLite service, no live LLM; not production throughput",
                    "baseline": baseline,
                    "parent": parent,
                    "p95_added_ms": round(parent["p95_ms"] - baseline["p95_ms"], 3),
                    "parent_default_enabled": False,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

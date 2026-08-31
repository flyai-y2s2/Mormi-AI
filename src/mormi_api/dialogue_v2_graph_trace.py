"""Opt-in, raw-free orchestration diagnostics; not a new persisted audit schema."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from .observability import scope_fields

logger = logging.getLogger("mormi_api.orchestration")


@contextmanager
def trace_node(graph: str, node: str, attempt: int = 0) -> Iterator[None]:
    if not logger.isEnabledFor(logging.DEBUG):
        yield
        return
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        conversation_id, turn_id = scope_fields()
        logger.debug(
            "graph_step conversation_id=%s turn_id=%s graph=%s node=%s attempt=%d "
            "duration_ms=%.3f status=%s",
            conversation_id,
            turn_id,
            graph,
            node,
            attempt,
            (time.perf_counter() - started) * 1000,
            status,
        )

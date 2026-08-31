"""Local synthetic benchmark; no network/model/database calls.

Run: PYTHONPATH=src:tests .venv/bin/python tests/benchmark_dialogue_v2_orchestration.py
The zero-delay result is the CPU orchestration budget; an optional provider delay
is reported separately and must not be used to hide pure execution overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from parity_support import HOME
from test_dialogue_v2_runtime import RecordingV2Gateway, _initialize, _response, _state

from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.schemas import ResponseType, UnderstandingResponseV2


async def benchmark(samples: int = 100, delay_ms: float = 0) -> dict:
    class Gateway(RecordingV2Gateway):
        async def understand_v2(self, request):
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
            return UnderstandingResponseV2(utterance_class="learning_response")

        async def speak_v2(self, plan):
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
            return await super().speak_v2(plan)

    old = HOME.DialogueV2Engine(Gateway())
    started = time.perf_counter()
    new = DialogueV2Engine(Gateway())
    compile_ms = (time.perf_counter() - started) * 1000
    state = _state()
    first = await _initialize(new, state)
    response = _response(first.turn_id, ResponseType.TEXT, text="아직 헷갈려")
    timings: dict[str, list[float]] = {"old": [], "graph": []}
    for i in range(samples + 20):
        runs = [("old", old.run_turn_stream), ("graph", new._run_turn_graph)]
        for label, run in runs if i % 2 else reversed(runs):
            started = time.perf_counter()
            _ = [event async for event in run(state, response, first.mormi.text)]
            if i >= 20:
                timings[label].append((time.perf_counter() - started) * 1000)
    stats = {
        k: {
            "median_ms": statistics.median(v),
            "p95_ms": sorted(v)[int(samples * 0.95) - 1],
            "mean_ms": statistics.mean(v),
        }
        for k, v in timings.items()
    }
    loss = (1 - stats["old"]["mean_ms"] / stats["graph"]["mean_ms"]) * 100
    extra = stats["graph"]["p95_ms"] - stats["old"]["p95_ms"]
    return {
        "samples": samples,
        "provider_delay_ms_per_call": delay_ms,
        "timings": stats,
        "compile_ms": compile_ms,
        "extra_p95_ms": extra,
        "throughput_loss_pct": loss,
        "cpu_budget_passed": delay_ms == 0 and extra <= 20 and loss <= 10,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--model-delay-ms", type=float, default=0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.samples, args.model_delay_ms)), indent=2))

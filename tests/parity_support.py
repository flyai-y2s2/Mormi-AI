"""Independent a8c82ff executors; synthetic model responses only, never live shadowing."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

BASELINE = Path(__file__).parent / "fixtures/dialogue_v2_orchestration/baseline"


def _load(name: str, filename: str) -> Any:
    qualified = f"mormi_api._parity_{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, BASELINE / filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    source = (BASELINE / filename).read_text()
    if name == "life":
        source = source.replace("from .dialogue_v2_runtime import", "from ._parity_home import")
    exec(compile(source, str(BASELINE / filename), "exec"), module.__dict__)
    return module


HOME = _load("home", "home.py")
LIFE = _load("life", "life.py")


def payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return {f.name: payload(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [payload(v) for v in value]
    return copy.deepcopy(value)


def comparable(value: Any) -> Any:
    value = payload(value)
    if isinstance(value, dict):
        return {
            k: (0 if v is not None else None) if k.endswith("_latency_ms") else comparable(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [comparable(v) for v in value]
    return value


def assert_same(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_same(left, right, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


class RecordedCalls:
    def __init__(self, target: Any, trace: list[Any], label: str) -> None:
        self.target, self.trace, self.label = target, trace, label
        self.calls: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.target, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            request = payload((args, kwargs))
            self.trace.append((self.label, name, request))
            try:
                result = await method(*args, **kwargs)
            except BaseException as error:
                # A provider cancelled by its enclosing runtime timeout is replayed
                # as TimeoutError, not as cancellation of the reference test itself.
                recorded = TimeoutError() if isinstance(error, asyncio.CancelledError) else error
                self.calls.append((name, request, None, recorded))
                raise
            self.calls.append((name, request, copy.deepcopy(result), None))
            return result

        return call


class ReplayCalls:
    def __init__(self, recorded: RecordedCalls, trace: list[Any]) -> None:
        self.recorded, self.trace, self.index = recorded, trace, 0

    def __getattr__(self, name: str) -> Any:
        # Preserve optional protocol methods (e.g. contextualize_note).
        getattr(self.recorded.target, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            assert self.index < len(self.recorded.calls), f"extra baseline call: {name}"
            expected_name, expected_request, result, error = self.recorded.calls[self.index]
            self.index += 1
            request = payload((args, kwargs))
            assert (name, request) == (expected_name, expected_request)
            self.trace.append((self.recorded.label, name, request))
            if error is not None:
                raise error
            return copy.deepcopy(result)

        return call


def assert_replay_finished(proxy: ReplayCalls | None) -> None:
    if proxy is not None:
        assert proxy.index == len(proxy.recorded.calls)

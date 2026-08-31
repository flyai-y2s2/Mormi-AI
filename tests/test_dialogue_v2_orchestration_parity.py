from __future__ import annotations

import inspect
from itertools import product
from typing import Any

import pytest
from parity_support import (
    HOME,
    LIFE,
    RecordedCalls,
    ReplayCalls,
    assert_replay_finished,
    assert_same,
    comparable,
    payload,
)

import mormi_api.schemas as schemas
from mormi_api.dialogue_v2_life_runtime import DialogueV2LifeEngine
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import EngineProgress, EngineTurnResult


def cases() -> list[Any]:
    import test_dialogue_v2_life_runtime as life_tests
    import test_dialogue_v2_runtime as home_tests

    result = []
    for module in (home_tests, life_tests):
        for name, function in vars(module).items():
            if not name.startswith("test_") or not inspect.iscoroutinefunction(function):
                continue
            options = []
            for mark in getattr(function, "pytestmark", []):
                if mark.name != "parametrize":
                    continue
                names, values = mark.args[:2]
                names = [n.strip() for n in names.split(",")] if isinstance(names, str) else names
                options.append(
                    [dict(zip(names, [v] if len(names) == 1 else v, strict=True)) for v in values]
                )
            for i, combination in enumerate(product(*options)):
                parameters = {k: v for choice in combination for k, v in choice.items()}
                result.append(
                    pytest.param(function, parameters, id=f"{module.__name__}.{name}-{i}")
                )
    return result


@pytest.mark.parametrize(("case", "parameters"), cases())
async def test_existing_scenarios_match_frozen_executor(
    case: Any,
    parameters: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    original = DialogueV2Engine.run_turn_stream

    async def checked(
        self: Any, state: Any, response: Any, previous_question: str, *, recent_dialogue: Any = None
    ) -> Any:
        before = state.model_copy(deep=True)
        current_trace: list[Any] = []
        baseline_trace: list[Any] = []
        actual_gateway, actual_resolver = self.gateway, self.copy_resolver
        gateway = RecordedCalls(actual_gateway, current_trace, "model")
        resolver = (
            RecordedCalls(actual_resolver, current_trace, "cache")
            if actual_resolver is not None
            else None
        )
        self.gateway, self.copy_resolver = gateway, resolver
        events: list[Any] = []
        failure = None
        try:
            async for event in original(
                self, state, response, previous_question, recent_dialogue=recent_dialogue
            ):
                events.append(event)
                if isinstance(event, EngineProgress):
                    current_trace.append(("progress", event.stage))
                    yield event
        except Exception as error:
            failure = error
        finally:
            self.gateway, self.copy_resolver = actual_gateway, actual_resolver
        assert payload(state) == payload(before), "input state mutated"
        replay_gateway = ReplayCalls(gateway, baseline_trace)
        replay_resolver = ReplayCalls(resolver, baseline_trace) if resolver else None
        cls = (
            LIFE.DialogueV2LifeEngine
            if isinstance(self, DialogueV2LifeEngine)
            else HOME.DialogueV2Engine
        )
        baseline = cls(
            replay_gateway,
            copy_resolver=replay_resolver,
            show_internal_pedagogy=self.show_internal_pedagogy,
            classifier_timeout_seconds=self.classifier_timeout_seconds,
            speaker_timeout_seconds=self.speaker_timeout_seconds,
            bridge_timeout_seconds=self.bridge_timeout_seconds,
        )
        final = next((e for e in events if isinstance(e, EngineTurnResult)), None)
        if final is not None:
            monkeypatch.setattr(HOME, "new_id", lambda _: final.turn.turn_id)
            monkeypatch.setattr(LIFE, "new_id", lambda _: final.turn.turn_id)
        reference = []
        reference_failure = None
        schema_id = schemas.new_id
        if final is not None and final.turn.note_update is not None:
            note_id = final.turn.note_update.note_id
            schemas.new_id = lambda prefix: note_id if prefix == "note" else schema_id(prefix)
        try:
            async for event in baseline.run_turn_stream(
                before, response, previous_question, recent_dialogue=recent_dialogue
            ):
                reference.append(event)
                if isinstance(event, EngineProgress):
                    baseline_trace.append(("progress", event.stage))
        except Exception as error:
            reference_failure = error
        finally:
            schemas.new_id = schema_id
        assert (type(failure), str(failure)) == (type(reference_failure), str(reference_failure))
        assert_same(comparable(events), comparable(reference))
        assert current_trace == baseline_trace
        assert_replay_finished(replay_gateway)
        assert_replay_finished(replay_resolver)
        if failure is not None:
            raise failure
        assert final is not None
        yield final

    monkeypatch.setattr(DialogueV2Engine, "run_turn_stream", checked)
    kwargs = dict(parameters)
    for name in inspect.signature(case).parameters:
        if name not in kwargs:
            kwargs[name] = request.getfixturevalue(name)
    await case(**kwargs)

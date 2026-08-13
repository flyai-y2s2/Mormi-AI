from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from mormi_api.content import HOME_TEACH_TASK_ID, HOME_TEACHING_CATALOG, home_teaching_task
from mormi_api.llm import ClaudeGateway, structured_output_schema, validate_speaker_output
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SceneType,
    SessionState,
    SlotClaim,
    SpeakerContext,
    SpeakerOutput,
    UtteranceAnalysis,
)
from mormi_api.settings import Settings


def object_schemas(node: object) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(object_schemas(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(object_schemas(value))
    return found


def test_classifier_schema_is_strict_for_every_nested_object() -> None:
    schema = structured_output_schema(UtteranceAnalysis)
    objects = object_schemas(schema)

    assert len(objects) >= 2  # root UtteranceAnalysis and nested SlotClaim
    assert all(item.get("additionalProperties") is False for item in objects)
    assert all(item.get("required") == list(item.get("properties", {})) for item in objects)


def test_speaker_schema_is_strict() -> None:
    schema = structured_output_schema(SpeakerOutput)
    objects = object_schemas(schema)

    assert objects
    assert all(item.get("additionalProperties") is False for item in objects)
    assert all(item.get("required") == list(item.get("properties", {})) for item in objects)


def speaker_context() -> SpeakerContext:
    return SpeakerContext(
        dialogue_act="acknowledge_partial",
        required_question="왜 왼쪽 줄이 덜 기다릴까?",
        fallback_text="왼쪽이구나. 왜 왼쪽 줄이 덜 기다릴까?",
    )


def test_speaker_rejects_grading_synonyms() -> None:
    context = speaker_context()
    for text in (
        "맞아! 왜 왼쪽 줄이 덜 기다릴까?",
        "정확해! 왜 왼쪽 줄이 덜 기다릴까?",
        "잘했어! 왜 왼쪽 줄이 덜 기다릴까?",
        "옳아! 왜 왼쪽 줄이 덜 기다릴까?",
    ):
        assert validate_speaker_output(SpeakerOutput(text=text), context) is None


def test_speaker_rejects_system_status_voice() -> None:
    context = speaker_context()
    for text in (
        "그 부분은 기억했어. 왜 왼쪽 줄이 덜 기다릴까?",
        "네가 말한 데까지는 들었어. 왜 왼쪽 줄이 덜 기다릴까?",
        "그 부분은 확인했어. 왜 왼쪽 줄이 덜 기다릴까?",
    ):
        assert validate_speaker_output(SpeakerOutput(text=text), context) is None


def test_speaker_must_keep_the_orchestrator_question() -> None:
    context = speaker_context()
    assert (
        validate_speaker_output(
            SpeakerOutput(text="왼쪽이구나. 왜 왼쪽 줄이 덜 기다릴까?"),
            context,
        )
        is not None
    )
    assert (
        validate_speaker_output(
            SpeakerOutput(text="왼쪽이구나. 무슨 색을 좋아해?"),
            context,
        )
        is None
    )


@pytest.mark.asyncio
async def test_unrelated_classification_is_semantically_rechecked_once() -> None:
    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.7,
    )
    audited = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="count_sequence",
                value="one_by_one_order",
                factual=True,
                evidence_span="하나 둘 셋",
            )
        ],
        confidence=0.95,
    )

    class FakeMessages:
        def __init__(self) -> None:
            self.outputs = [first.model_dump_json(), audited.model_dump_json()]
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=self.outputs.pop(0))],
            )

    messages = FakeMessages()
    gateway = ClaudeGateway(Settings(anthropic_api_key=None))
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    task = home_teaching_task(HOME_TEACHING_CATALOG["number-count"], skill_id="number-count")
    state = SessionState(
        learner_id=1,
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=[HOME_TEACH_TASK_ID],
        expression_level=ExpressionLevel.L4,
    )

    result = await gateway.classify(
        state=state,
        task=task,
        previous_question="점이 두 개 있는 것 같은데, 너는 몇 개로 셌어?",
        response=ChildResponse(
            turn_id="turn_1",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="하나, 둘, 셋 하고 세면 돼",
        ),
    )

    assert result.response_category is ResponseCategory.CORRECT_PARTIAL
    assert result.claims[0].slot_id == "count_sequence"
    assert len(messages.prompts) == 2
    assert "semantic_relation_audit" in messages.prompts[1]

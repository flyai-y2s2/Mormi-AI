from __future__ import annotations

from typing import Any

from mormi_api.llm import structured_output_schema, validate_speaker_output
from mormi_api.schemas import SpeakerContext, SpeakerOutput, UtteranceAnalysis


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

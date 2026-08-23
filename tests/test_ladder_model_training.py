from __future__ import annotations

from mormi_api.ladder_model.dataset import (
    ConceptResult,
    LadderExample,
    LadderLevel,
    ResponseMode,
)
from mormi_api.ladder_model.training_data import LABEL_TO_ID, serialize_model_input


def test_model_input_keeps_response_semantics_but_not_target_label() -> None:
    example = LadderExample(
        sample_id="sample-1",
        learner_key="anon_a",
        learning_session_id="session-1",
        study_date="2026-08-17",
        skill_id="money-budget",
        current_level=LadderLevel.L3,
        response_mode=ResponseMode.NO_RESPONSE,
        response_text="[NO_RESPONSE]",
        concept_result=ConceptResult.NOT_ASSESSED,
        attempt_count=2,
        target_level=LadderLevel.L2,
        synthetic=False,
        source="validated_descent",
        rubric_version="ladder-label-v1",
    )

    text = serialize_model_input(example)

    assert "현재단계=L3" in text
    assert "응답방식=no_response" in text
    assert "정답여부=not_assessed" in text
    assert "[NO_RESPONSE]" in text
    assert "target" not in text.lower()
    assert LABEL_TO_ID[example.target_level] == 1


def test_label_order_follows_support_ladder() -> None:
    assert LABEL_TO_ID == {
        LadderLevel.L0: 0,
        LadderLevel.L2: 1,
        LadderLevel.L3: 2,
        LadderLevel.L4: 3,
    }


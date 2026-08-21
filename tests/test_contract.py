from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from mormi_api.main import (
    _request_validation_code,
    _service_error_code,
    _validation_issues,
    app,
)
from mormi_api.schemas import PracticeResult, PracticeSummary, SessionCreate


def test_openapi_exposes_frontend_agreed_paths() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/v1/conversations" in paths
    assert "/v1/conversations/{conversation_id}/responses" in paths
    assert "/v1/conversations/{conversation_id}/responses/stream" in paths
    assert "/v1/conversations/{conversation_id}" in paths
    assert "/v1/learners/{learner_id}/skill-profiles" in paths
    assert "/v1/learners/{learner_id}/star-notes" in paths
    assert "/v1/content/dictionary-cards/{curriculum_session_id}" in paths
    assert "/v1/conversations/{conversation_id}/dictionary-card" in paths

    child_response = schema["components"]["schemas"]["ChildResponse"]
    assert child_response["properties"]["response_id"]["format"] == "uuid"
    assert "response_id" in child_response["required"]
    assert (
        schema["components"]["schemas"]["SessionCreate"]["properties"]["learner_id"]["type"]
        == "integer"
    )
    assert "cafe_context" in schema["components"]["schemas"]["SessionCreate"]["properties"]
    cafe_context = schema["components"]["schemas"]["CafeSessionContext"]
    assert {"menu_items", "mormi_menu_id"}.issubset(cafe_context["required"])
    assert "child_menu_id" not in cafe_context["properties"]
    assert "paid_amount" not in cafe_context["properties"]
    queue_context = schema["components"]["schemas"]["QueueSessionContext"]
    assert queue_context["properties"]["left_count"]["minimum"] == 1
    assert queue_context["properties"]["left_count"]["maximum"] == 5
    assert queue_context["properties"]["right_count"]["minimum"] == 1
    assert queue_context["properties"]["right_count"]["maximum"] == 5
    assert "completion" in schema["components"]["schemas"]["TurnContract"]["properties"]
    assert "task_anchor" in schema["components"]["schemas"]["TurnContract"]["properties"]
    assert "dictionary_ref" in schema["components"]["schemas"]["TurnContract"]["properties"]
    conflict = paths["/v1/conversations/{conversation_id}/responses"]["post"]["responses"]["409"]
    assert conflict["content"]["application/json"]["schema"]["$ref"].endswith("/ConflictResponse")


def test_storage_consent_defaults_to_permanent_retention() -> None:
    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue_demo",
            conversation_storage_consent=True,
            retention_policy="no_raw",
        )

    request = SessionCreate(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_queue_demo",
    )
    assert request.conversation_storage_consent is True
    assert request.retention_policy.value == "permanent"
    assert request.retention_policy.expires_at(datetime.now(UTC)) is None


def test_compact_practice_summary_derives_success_rate() -> None:
    summary = PracticeSummary(
        skill_id="money_count",
        question_count=5,
        first_try_correct_count=3,
        wrong_attempt_count=2,
        earned_reward=850,
        misconception_tags=["coin_count_not_value"],
    )
    assert summary.success_rate == 0.6


def test_frontend_inline_practice_snapshot_does_not_repeat_ids() -> None:
    request = SessionCreate(
        learner_id=1,
        scene="home_teach",
        scenario_id="home_teach",
        learning_session_id="session_123",
        practice_result_id="practice_123",
        practice_summary={
            "curriculum_session_id": "add-pictures",
            "skill_id": "basic_addition",
            "question_count": 5,
            "first_try_correct_count": 3,
            "wrong_attempt_count": 2,
            "earned_reward": 850,
            "misconception_tags": ["count_all_error"],
        },
    )
    assert request.practice_summary is not None
    assert request.practice_summary.success_rate == 0.6

    stored = PracticeResult(
        **request.practice_summary.model_dump(),
        practice_result_id=request.practice_result_id,
        learner_id=request.learner_id,
    )
    assert stored.practice_result_id == "practice_123"
    assert stored.learner_id == 1


@pytest.mark.parametrize(
    ("learning_session_id", "practice_result_id", "message"),
    [
        (None, "practice_123", "learning_session_id is required"),
        ("   ", "practice_123", "learning_session_id is required"),
        ("session_123", None, "practice_result_id is required"),
        ("session_123", "   ", "practice_result_id is required"),
    ],
)
def test_home_teach_requires_non_empty_session_and_practice_ids(
    learning_session_id: str | None,
    practice_result_id: str | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SessionCreate(
            learner_id=1,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=learning_session_id,
            practice_result_id=practice_result_id,
        )


def test_practice_summary_rejects_free_text_child_utterance() -> None:
    with pytest.raises(ValidationError):
        PracticeSummary(
            skill_id="number_count",
            attempts=[
                {
                    "item_id": "number-count:0",
                    "correct": True,
                    "response": "사람을 한 명씩 세면 돼",
                }
            ],
        )


def test_validation_diagnostics_never_echo_rejected_input() -> None:
    secret_child_text = "저장하면 안 되는 아이 원문"
    with pytest.raises(ValidationError) as raised:
        PracticeSummary(
            skill_id="number_count",
            attempts=[
                {
                    "item_id": "number-count:0",
                    "correct": True,
                    "response": secret_child_text,
                }
            ],
        )

    issues = _validation_issues(raised.value)
    assert issues == [
        {
            "location": "attempts.0.response.int",
            "type": "int_parsing",
        },
        {
            "location": "attempts.0.response.float",
            "type": "float_parsing",
        },
        {
            "location": "attempts.0.response.list[str]",
            "type": "list_type",
        },
    ]
    assert secret_child_text not in str(issues)


def test_service_validation_error_gets_a_stable_safe_code() -> None:
    with pytest.raises(ValidationError) as raised:
        PracticeSummary(skill_id="number_count", question_count=1, first_try_correct_count=2)

    code, path, issues = _service_error_code(raised.value)
    assert code == "stored_state_validation_failed"
    assert path == "request"
    assert issues


def test_known_service_error_gets_a_stable_safe_code() -> None:
    code, path, issues = _service_error_code(
        ValueError("unsupported home curriculum_session_id: child-secret")
    )
    assert code == "home_curriculum_unsupported"
    assert path is None
    assert issues == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Value error, consented raw storage requires a retention_policy",
            "storage_consent_retention_mismatch",
        ),
        (
            "Value error, retention_policy must be no_raw without storage consent",
            "storage_retention_without_consent",
        ),
        (
            "Value error, practice_result_id is required for home_teach",
            "home_practice_result_missing",
        ),
    ],
)
def test_request_level_validation_gets_a_stable_safe_code(
    message: str,
    expected: str,
) -> None:
    error = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body",),
                "msg": message,
                "input": {"sensitive": "아이 원문"},
            }
        ]
    )
    assert _request_validation_code(error) == expected


def test_unknown_request_validation_exposes_only_the_rule_name() -> None:
    error = RequestValidationError(
        [
            {
                "type": "model_attributes_type",
                "loc": ("body",),
                "msg": "Input should be a valid dictionary",
                "input": "아이 원문",
            }
        ]
    )
    assert _request_validation_code(error) == "request_validation_failed.model_attributes_type"

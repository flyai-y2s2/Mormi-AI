from __future__ import annotations

import pytest
from pydantic import ValidationError

from mormi_api.main import app
from mormi_api.schemas import PracticeResult, PracticeSummary, SessionCreate


def test_openapi_exposes_frontend_agreed_paths() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/v1/conversations" in paths
    assert "/v1/conversations/{conversation_id}/responses" in paths
    assert "/v1/conversations/{conversation_id}" in paths
    assert "/v1/learners/{learner_id}/skill-profiles" in paths
    assert "/v1/learners/{learner_id}/star-notes" in paths

    child_response = schema["components"]["schemas"]["ChildResponse"]
    assert child_response["properties"]["response_id"]["format"] == "uuid"
    assert "response_id" in child_response["required"]
    assert (
        schema["components"]["schemas"]["SessionCreate"]["properties"]["learner_id"]["type"]
        == "integer"
    )
    assert "completion" in schema["components"]["schemas"]["TurnContract"]["properties"]
    conflict = paths["/v1/conversations/{conversation_id}/responses"]["post"]["responses"]["409"]
    assert conflict["content"]["application/json"]["schema"]["$ref"].endswith("/ConflictResponse")


def test_storage_consent_requires_finite_retention() -> None:
    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="home_teach",
            scenario_id="home_addition_teach",
            conversation_storage_consent=True,
            retention_policy="no_raw",
        )

    request = SessionCreate(
        learner_id=1,
        scene="home_teach",
        scenario_id="home_addition_teach",
        conversation_storage_consent=True,
        retention_policy="30_days",
    )
    assert request.retention_policy.value == "30_days"


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
        scenario_id="home_addition_teach",
        practice_result_id="practice_123",
        practice_summary={
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

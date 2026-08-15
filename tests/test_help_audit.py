from __future__ import annotations

import pytest
from pydantic import ValidationError

from mormi_api.content import TaskDefinition, queue_task
from mormi_api.help_audit import build_help_review_items, render_human_review
from mormi_api.schemas import ExpressionLevel, HintLevel


def test_review_pipeline_discovers_every_current_home_and_cafe_task() -> None:
    items = build_help_review_items()
    review_ids = [item.review_id for item in items]

    assert len(items) == 41
    assert len(review_ids) == len(set(review_ids))
    assert sum(review_id.startswith("home:") for review_id in review_ids) == 36
    assert any(review_id.startswith("cafe_queue:") for review_id in review_ids)
    assert any(review_id.startswith("cafe_budget_menu:") for review_id in review_ids)
    assert any(review_id.startswith("cafe_menu_total:") for review_id in review_ids)
    assert any(review_id.startswith("cafe_change:") for review_id in review_ids)


def test_review_items_expose_full_contract_for_offline_ai_and_humans() -> None:
    items = build_help_review_items()
    for item in items:
        assert set(item.help_plan) == {"H1", "H2", "H3"}, item.review_id
        assert item.help_skills, item.review_id
        assert item.accepted_methods, item.review_id
        assert item.method_policy in {"open_methods", "target_method"}, item.review_id
        assert set(item.required_slots) <= set(item.l0_joint_completion), item.review_id

    report = render_human_review(items)
    assert "질문·화면·H1·H2·H3" in report
    assert report.count("## home:") == 36
    assert "풀이 정책" in report
    assert "H3와 L0 공동 수행" in report


def test_task_registration_rejects_unreviewed_fact_references() -> None:
    task = queue_task(task_id="bad-ref", stage_id="queue", left=2, right=5)
    raw = task.model_dump(mode="json")
    raw["hints"][HintLevel.H1]["fact_refs"] = ["not_on_screen"]

    with pytest.raises(ValidationError, match="unknown fact refs"):
        TaskDefinition.model_validate(raw)


def test_task_registration_rejects_h3_without_complete_joint_performance() -> None:
    task = queue_task(task_id="bad-joint", stage_id="queue", left=2, right=5)
    raw = task.model_dump(mode="json")
    raw["steps"][ExpressionLevel.L0][0]["input"]["config"]["completion_values"].pop(
        "reason"
    )

    with pytest.raises(ValidationError, match="L0 cannot complete required slots"):
        TaskDefinition.model_validate(raw)

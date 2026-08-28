from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from mormi_api.content import (
    BUDGET_MENU_TASK_ID,
    CHANGE_TASK_ID,
    QUEUE_TASK_ID,
    TOTAL_CALC_TASK_ID,
)
from mormi_api.dialogue_v2_cafe_content import create_cafe_scenario_pack_v2
from mormi_api.dialogue_v2_life_content import (
    LifeScenarioPackV2,
    canonical_life_scenario_json_v2,
    life_scenario_hash_v2,
)
from mormi_api.schemas import (
    CafeMenuItem,
    CafeSessionContext,
    ExpressionLevel,
    QueueSessionContext,
)

FRONTEND_MENU = [
    CafeMenuItem(
        id="americano",
        name="아메리카노",
        price=3000,
        image_url="/figma/cafe/americano.png?v=2",
    ),
    CafeMenuItem(
        id="milk",
        name="우유",
        price=2000,
        image_url="/figma/cafe/milk.png?v=2",
    ),
    CafeMenuItem(
        id="strawberry-juice",
        name="딸기주스",
        price=4000,
        image_url="/figma/cafe/strawberry-juice.png?v=2",
    ),
    CafeMenuItem(
        id="cookie",
        name="쿠키",
        price=2000,
        image_url="/figma/cafe/cookie.png?v=2",
    ),
    CafeMenuItem(
        id="strawberry-cake",
        name="딸기케이크",
        price=4500,
        image_url="/figma/cafe/strawberry-cake.png?v=2",
    ),
    CafeMenuItem(
        id="sandwich",
        name="샌드위치",
        price=5000,
        image_url="/figma/cafe/sandwich.png?v=2",
    ),
]


def _cafe_context(
    mormi_menu_id: str,
    *,
    child_menu_id: str | None = None,
    budget: int | None = None,
    menu_items: list[CafeMenuItem] | None = None,
) -> CafeSessionContext:
    return CafeSessionContext(
        menu_items=menu_items or FRONTEND_MENU,
        mormi_menu_id=mormi_menu_id,
        child_menu_id=child_menu_id,
        budget=budget,
    )


def _only_variant(pack: LifeScenarioPackV2, task_id: str):
    stage = pack.stage_by_task_id(task_id)
    assert list(stage.variants) == [stage.default_variant_id]
    return stage.variants[stage.default_variant_id]


def _fact(pack: object, fact_id: str):
    graph = pack.reasoning_graph  # type: ignore[attr-defined]
    return next(fact for fact in graph.facts if fact.fact_id == fact_id)


def _projection(pack: LifeScenarioPackV2) -> list[tuple[str, str, str, object]]:
    return [
        (
            item.output_key,
            item.source_task_id,
            item.source_id,
            item.relation_value,
        )
        for item in pack.completion_projection
    ]


def test_queue_materializer_pins_graph_visual_and_exact_projection() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_queue",
        queue_context=QueueSessionContext(left_count=2, right_count=5),
    )
    task = _only_variant(scenario, QUEUE_TASK_ID)

    assert task.base_visual.type == "cafe_queues"
    assert task.base_visual.data == {
        "left_people": 2,
        "right_people": 5,
        "show_counts": False,
    }
    assert _fact(task, "left_count").value.value == 2
    assert _fact(task, "right_count").value.value == 5
    assert _fact(task, "final_choice").value.choice_id == "left"
    assert not _fact(task, "left_count").initially_visible
    relation = task.reasoning_graph.relations[0]
    assert relation.operation == "comparison"
    assert relation.comparison_goal == "minimum"
    assert relation.input_fact_ids == ["left_count", "right_count"]
    assert relation.output_fact_id == "final_choice"
    assert len(task.l2_plans) == 4
    assert {
        (completion.target_kind, completion.target_id)
        for completion in task.l0_joint_plan.completion_values
    } == {
        ("fact", "left_count"),
        ("fact", "right_count"),
        ("fact", "final_choice"),
        ("relation", "choose_shorter_queue"),
    }
    assert task.help_plan.H3.revealed_fact_ids == [
        "left_count",
        "right_count",
        "final_choice",
    ]
    assert task.help_plan.H3.revealed_relation_ids == ["choose_shorter_queue"]
    assert task.policies.note_context == "왼쪽 2명과 오른쪽 5명의 줄을 비교하는 방법"
    assert _projection(scenario) == [
        ("left_count", QUEUE_TASK_ID, "left_count", None),
        ("right_count", QUEUE_TASK_ID, "right_count", None),
        ("final_choice", QUEUE_TASK_ID, "final_choice", None),
        ("reason", QUEUE_TASK_ID, "choose_shorter_queue", "fewer_people"),
    ]


def test_budget_menu_preserves_multiple_answers_and_reviewed_choice_effects() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_budget_menu",
        cafe_context=_cafe_context("strawberry-juice", budget=6000),
    )
    task = _only_variant(scenario, BUDGET_MENU_TASK_ID)
    selected = _fact(task, "child_menu")

    assert {selected.value.choice_id, *(value.choice_id for value in selected.accepted_values)} == {
        "milk",
        "cookie",
    }
    assert task.base_visual.type == "cafe_menu"
    assert task.base_visual.data["auto_total"] is True
    assert task.base_visual.data["budget_status"] == "pending"
    assert task.policies.entry_expression_level.value == "L2"
    assert task.policies.note_relation_ids == ["choose_menu_within_budget"]
    assert _projection(scenario) == [("child_menu_id", BUDGET_MENU_TASK_ID, "child_menu", None)]

    choices = {choice.label.split()[0]: choice for choice in task.l2_plans[0].choices}
    assert choices["딸기주스"].disabled
    assert choices["딸기주스"].image_url == "/figma/cafe/strawberry-juice.png?v=2"
    assert choices["우유"].effect.verdict == "correct"
    assert choices["쿠키"].effect.verdict == "correct"
    assert choices["아메리카노"].effect.verdict == "incorrect"
    assert choices["아메리카노"].effect.visual_patch == {
        "child_pick": FRONTEND_MENU[0].model_dump(mode="json"),
        "total": 7000,
        "budget_status": "over",
    }
    assert task.l2_plans[0].input_config == {
        "component": "cafe_menu_picker",
        "budget": 6000,
        "mormi_menu_id": "strawberry-juice",
        "auto_total": True,
        "allow_same_menu": False,
    }
    assert task.l0_joint_plan.input_config["suggested_menu_id"] == "milk"
    assert task.help_plan.H3.visual_data["total"] == 6000


def test_menu_total_starts_directly_with_one_pinned_calculation_problem() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_menu_total",
        cafe_context=_cafe_context("americano", child_menu_id="milk"),
    )
    assert scenario.content_version == 2
    assert [stage.task_id for stage in scenario.task_stages] == [TOTAL_CALC_TASK_ID]
    calculation = _only_variant(scenario, TOTAL_CALC_TASK_ID)
    assert calculation.policies.entry_expression_level is ExpressionLevel.L4
    assert calculation.base_visual.type == "cafe_calculation"
    assert calculation.base_visual.data["mormi_menu"]["id"] == "americano"
    assert calculation.base_visual.data["child_menu"]["id"] == "milk"
    assert _fact(calculation, "mormi_menu_price").value.amount == 3000
    assert _fact(calculation, "child_menu_price").value.amount == 2000
    assert _fact(calculation, "total").value.amount == 5000
    relation = calculation.reasoning_graph.relations[0]
    assert relation.operation == "addition"
    assert relation.input_fact_ids == ["mormi_menu_price", "child_menu_price"]
    assert relation.output_fact_id == "total"
    assert calculation.help_plan.H3.visual_data["result"] == 5000
    assert len(calculation.l2_plans) == 2
    assert len(calculation.l0_joint_plan.completion_values) == 2
    assert calculation.policies.note_context == "3,000원과 2,000원을 더하기로 계산하는 방법"
    assert _projection(scenario) == [("result", TOTAL_CALC_TASK_ID, "total", None)]


def test_menu_total_old_request_without_child_pick_uses_first_different_menu() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_menu_total",
        cafe_context=_cafe_context("americano"),
    )
    calculation = _only_variant(scenario, TOTAL_CALC_TASK_ID)

    assert calculation.base_visual.data["mormi_menu"]["id"] == "americano"
    assert calculation.base_visual.data["child_menu"]["id"] == "milk"
    assert _fact(calculation, "mormi_menu_price").value.amount == 3000
    assert _fact(calculation, "child_menu_price").value.amount == 2000


def test_budget_menu_starts_with_mormi_help_seeking_voice() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_budget_menu",
        cafe_context=_cafe_context("strawberry-juice", budget=6000),
    )
    task = _only_variant(scenario, BUDGET_MENU_TASK_ID)

    assert task.initial_question.reviewed_fallback == (
        "나 6,000원 안에서 우리 둘의 메뉴를 고르려는데 헷갈려... "
        "나는 딸기주스를 골랐어. 너는 뭘 고르면 좋을지 알려줄래?"
    )


def test_change_uses_reviewed_subtraction_and_rejects_price_over_payment() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_change",
        cafe_context=_cafe_context("strawberry-cake"),
    )
    task = _only_variant(scenario, CHANGE_TASK_ID)

    assert _fact(task, "payment").value.amount == 10000
    assert _fact(task, "menu_price").value.amount == 4500
    assert _fact(task, "change").value.amount == 5500
    relation = task.reasoning_graph.relations[0]
    assert relation.operation == "subtraction"
    assert relation.input_fact_ids == ["payment", "menu_price"]
    assert task.base_visual.data["payment"] == 10000
    assert task.base_visual.data["menu_total"] == 4500
    assert task.help_plan.H3.revealed_fact_ids == ["change"]
    assert task.help_plan.H3.revealed_relation_ids == ["subtract_menu_price"]
    assert len(task.l2_plans) == 2
    assert len(task.l0_joint_plan.completion_values) == 2
    assert _projection(scenario) == [("result", CHANGE_TASK_ID, "change", None)]

    expensive_menu = [
        CafeMenuItem(id="expensive", name="비싼 메뉴", price=11000),
        CafeMenuItem(id="milk", name="우유", price=2000),
    ]
    with pytest.raises(ValueError, match="cannot exceed"):
        create_cafe_scenario_pack_v2(
            "cafe_change",
            cafe_context=_cafe_context("expensive", menu_items=expensive_menu),
        )


def test_cafe_materialization_hash_is_reproducible_and_detects_tampering() -> None:
    context = _cafe_context("americano", child_menu_id="milk")
    first = create_cafe_scenario_pack_v2("cafe_menu_total", cafe_context=context)
    second = create_cafe_scenario_pack_v2("cafe_menu_total", cafe_context=context)

    assert canonical_life_scenario_json_v2(first) == canonical_life_scenario_json_v2(second)
    assert life_scenario_hash_v2(first) == life_scenario_hash_v2(second)

    visual_payload = deepcopy(first.model_dump(mode="json"))
    visual_payload["task_stages"][0]["variants"]["default"]["base_visual"]["data"][
        "mormi_menu"
    ]["image_url"] = "/tampered.png"
    visual_tamper = LifeScenarioPackV2.model_validate(visual_payload)
    assert life_scenario_hash_v2(visual_tamper) != life_scenario_hash_v2(first)

    arithmetic_payload = deepcopy(first.model_dump(mode="json"))
    calculation_stage = arithmetic_payload["task_stages"][0]
    variant = calculation_stage["variants"][calculation_stage["default_variant_id"]]
    result_fact = next(
        fact for fact in variant["reasoning_graph"]["facts"] if fact["fact_id"] == "total"
    )
    result_fact["value"]["amount"] = 9999
    with pytest.raises(ValidationError, match="arithmetic contradicts"):
        LifeScenarioPackV2.model_validate(arithmetic_payload)

    task_payload = deepcopy(first.model_dump(mode="json"))
    task_payload["task_stages"][0]["task_id"] = "unknown"
    with pytest.raises(ValidationError, match="task ID"):
        LifeScenarioPackV2.model_validate(task_payload)

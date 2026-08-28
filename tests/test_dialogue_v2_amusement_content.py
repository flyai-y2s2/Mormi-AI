from __future__ import annotations

import random
from typing import Any

import pytest
from pydantic import ValidationError

from mormi_api.content import (
    PARK_PRIMARY_TASK_IDS,
    PARK_TRANSFER_TASK_IDS,
    generate_park_context,
    get_task,
)
from mormi_api.dialogue_v2_amusement_content import (
    AMUSEMENT_CONTENT_VERSION_V2,
    materialize_amusement_scenario_v2,
)
from mormi_api.dialogue_v2_life_content import (
    LifeScenarioPackV2,
    canonical_life_scenario_json_v2,
    life_scenario_hash_v2,
)
from mormi_api.schemas import (
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
)

SCENARIOS = (
    "amusement_ticket_multiply",
    "amusement_snack_divide",
    "amusement_pass_compare",
)

EXPECTED_PROJECTIONS = {
    "amusement_ticket_multiply": [
        "ticket_price",
        "party_count",
        "total_price",
    ],
    "amusement_snack_divide": [
        "snack_total",
        "payer_count",
        "per_person",
    ],
    "amusement_pass_compare": [
        "single_ride_price",
        "day_pass_price",
        "break_even_rides",
        "benefit_from_rides",
    ],
}


def _context(scenario_id: str):
    seed = {
        "amusement_ticket_multiply": 110,
        "amusement_snack_divide": 111,
        "amusement_pass_compare": 112,
    }[scenario_id]
    return generate_park_context(scenario_id, random.Random(seed))


def _pack_for_stage(scenario_pack: LifeScenarioPackV2, index: int):
    stage = scenario_pack.task_stages[index]
    return stage.variants[stage.default_variant_id]


def _scalar(value: MoneyValueV2 | NumberValueV2) -> int | float:
    return value.amount if isinstance(value, MoneyValueV2) else value.value


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_materializes_three_scenarios_as_six_ordered_tasks(scenario_id: str) -> None:
    context = _context(scenario_id)

    pack = materialize_amusement_scenario_v2(
        scenario_id,
        {"park_context": context.model_dump(mode="json")},
    )

    assert pack.scene == "amusement_park"
    assert pack.content_version == AMUSEMENT_CONTENT_VERSION_V2
    assert [stage.task_id for stage in pack.task_stages] == [
        PARK_PRIMARY_TASK_IDS[scenario_id],
        PARK_TRANSFER_TASK_IDS[scenario_id],
    ]
    primary = _pack_for_stage(pack, 0)
    transfer = _pack_for_stage(pack, 1)
    assert (primary.phase, transfer.phase) == ("primary", "transfer")
    assert primary.content_version == AMUSEMENT_CONTENT_VERSION_V2
    assert transfer.content_version == AMUSEMENT_CONTENT_VERSION_V2
    assert primary.policies.entry_expression_level is ExpressionLevel.L4
    assert transfer.policies.entry_expression_level is ExpressionLevel.L4
    assert primary.source_prompt == context.prompt
    assert transfer.source_prompt == context.transfer.prompt
    assert primary.policies.transition_text == (
        "아까 같이 살펴본 방법이 숫자가 바뀌어도 되는지 궁금해..."
    )


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_primary_and_transfer_graphs_use_distinct_pinned_arithmetic(
    scenario_id: str,
) -> None:
    context = _context(scenario_id)
    scenario_data = {"park_context": context.model_dump(mode="json")}
    pack = materialize_amusement_scenario_v2(scenario_id, scenario_data)
    primary = _pack_for_stage(pack, 0)
    transfer = _pack_for_stage(pack, 1)
    primary_legacy = get_task(PARK_PRIMARY_TASK_IDS[scenario_id], scenario_data)
    transfer_legacy = get_task(PARK_TRANSFER_TASK_IDS[scenario_id], scenario_data)

    for task_pack, legacy in ((primary, primary_legacy), (transfer, transfer_legacy)):
        relation = task_pack.reasoning_graph.relations[0]
        facts = {fact.fact_id: fact for fact in task_pack.reasoning_graph.facts}
        operands = [_scalar(facts[fact_id].value) for fact_id in relation.input_fact_ids]
        result = _scalar(facts[relation.output_fact_id].value)
        contract = legacy.arithmetic_contract
        assert contract is not None
        assert operands == [contract.left, contract.right]
        assert result == contract.result
        if relation.operation == "multiplication":
            assert operands[0] * operands[1] == result
        else:
            assert operands[0] / operands[1] == result

    primary_result = _scalar(
        next(
            fact
            for fact in primary.reasoning_graph.facts
            if fact.fact_id == primary.reasoning_graph.relations[0].output_fact_id
        ).value
    )
    transfer_result = _scalar(
        next(
            fact
            for fact in transfer.reasoning_graph.facts
            if fact.fact_id == transfer.reasoning_graph.relations[0].output_fact_id
        ).value
    )
    assert primary_result != transfer_result
    if scenario_id == "amusement_pass_compare":
        for task_pack in (primary, transfer):
            facts = {fact.fact_id: fact for fact in task_pack.reasoning_graph.facts}
            assert _scalar(facts["benefit_from_rides"].value) == (
                _scalar(facts["break_even_rides"].value) + 1
            )


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_preserves_reviewed_visual_help_and_dictionary_contracts(
    scenario_id: str,
) -> None:
    context = _context(scenario_id)
    scenario_data = {"park_context": context.model_dump(mode="json")}
    scenario_pack = materialize_amusement_scenario_v2(scenario_id, scenario_data)

    for index, task_id in enumerate(
        (PARK_PRIMARY_TASK_IDS[scenario_id], PARK_TRANSFER_TASK_IDS[scenario_id])
    ):
        life = _pack_for_stage(scenario_pack, index)
        legacy = get_task(task_id, scenario_data)
        assert life.dictionary_card_id == legacy.dictionary_card_id
        assert life.base_visual.model_dump(mode="json") == legacy.base_visual.model_dump(
            mode="json"
        )
        for level in (HintLevel.H1, HintLevel.H2, HintLevel.H3):
            actual = getattr(life.help_plan, level.value)
            expected = legacy.hints[level]
            assert actual.body == expected.body
            assert actual.action == expected.action
            assert actual.visual_type == expected.visual_type
            assert actual.visual_data == expected.visual_data

    primary = _pack_for_stage(scenario_pack, 0)
    transfer = _pack_for_stage(scenario_pack, 1)
    assert primary.base_visual.type == "amusement_park"
    assert transfer.base_visual.type == "amusement_park_transfer"
    if scenario_id == "amusement_ticket_multiply":
        assert primary.help_plan.H2.visual_type is None
        assert transfer.help_plan.H2.visual_type is None
    else:
        assert primary.help_plan.H2.visual_type == "amusement_equation"
        assert transfer.help_plan.H2.visual_type == "amusement_transfer_equation"
    assert primary.help_plan.H3.visual_type == "amusement_joint_solution"
    assert transfer.help_plan.H3.visual_type == "amusement_transfer_solution"


def test_pass_break_even_accepts_division_and_equivalent_inverse_methods() -> None:
    context = _context("amusement_pass_compare")
    scenario_pack = materialize_amusement_scenario_v2(
        "amusement_pass_compare",
        {"park_context": context.model_dump(mode="json")},
    )

    for task_pack in (
        _pack_for_stage(scenario_pack, 0),
        _pack_for_stage(scenario_pack, 1),
    ):
        relation = task_pack.reasoning_graph.relations[0]
        sufficient = "\n".join(relation.rubric.sufficient)
        assert relation.operation == "division"
        assert "자유이용권 가격을 1회 가격으로 나누어" in sufficient
        assert "1회 이용권 값에 횟수를 곱해서" in sufficient
        assert "반복해서 더해" in sufficient
        assert "곱하면" in sufficient


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_entry_l3_l2_l0_and_help_cover_every_required_target(
    scenario_id: str,
) -> None:
    context = _context(scenario_id)
    primary_task_id = PARK_PRIMARY_TASK_IDS[scenario_id]
    transfer_task_id = PARK_TRANSFER_TASK_IDS[scenario_id]
    scenario_pack = materialize_amusement_scenario_v2(
        scenario_id,
        {"park_context": context.model_dump(mode="json")},
        task_start_levels={
            primary_task_id: ExpressionLevel.L2,
            transfer_task_id: ExpressionLevel.L4,
        },
    )
    primary = _pack_for_stage(scenario_pack, 0)
    transfer = _pack_for_stage(scenario_pack, 1)
    assert primary.policies.entry_expression_level is ExpressionLevel.L2
    assert transfer.policies.entry_expression_level is ExpressionLevel.L4

    expected_target_count = 3 if scenario_id == "amusement_pass_compare" else 2
    expected_copy_count = 6 if scenario_id == "amusement_pass_compare" else 5
    for task_pack in (primary, transfer):
        required = {
            *(
                ("fact", fact_id)
                for fact_id in task_pack.reasoning_graph.completion.required_fact_ids
            ),
            *(
                ("relation", relation_id)
                for relation_id in task_pack.reasoning_graph.completion.required_relation_ids
            ),
        }
        assert len(required) == expected_target_count
        assert len(task_pack.initial_question.targets) == expected_target_count
        assert len(task_pack.l3_plans) == expected_target_count
        assert len(task_pack.l2_plans) == expected_target_count
        assert len(task_pack.copy_slots) == expected_copy_count
        assert len(task_pack.l0_joint_plan.completion_values) == expected_target_count
        assert all(
            any(choice.effect.verdict == "correct" for choice in plan.choices)
            and any(choice.effect.verdict == "incorrect" for choice in plan.choices)
            for plan in task_pack.l2_plans
        )
        assert not task_pack.help_plan.H1.revealed_fact_ids
        assert not task_pack.help_plan.H2.revealed_fact_ids
        h3_targets = {
            *(("fact", fact_id) for fact_id in task_pack.help_plan.H3.revealed_fact_ids),
            *(
                ("relation", relation_id)
                for relation_id in task_pack.help_plan.H3.revealed_relation_ids
            ),
        }
        assert required <= h3_targets

    with pytest.raises(ValueError, match="entry must be L4 or L2"):
        materialize_amusement_scenario_v2(
            scenario_id,
            {"park_context": context.model_dump(mode="json")},
            task_start_levels={primary_task_id: ExpressionLevel.L3},
        )


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_only_primary_task_can_create_a_star_note(scenario_id: str) -> None:
    context = _context(scenario_id)
    scenario_data = {"park_context": context.model_dump(mode="json")}
    scenario_pack = materialize_amusement_scenario_v2(scenario_id, scenario_data)
    primary = _pack_for_stage(scenario_pack, 0)
    transfer = _pack_for_stage(scenario_pack, 1)
    primary_legacy = get_task(PARK_PRIMARY_TASK_IDS[scenario_id], scenario_data)

    assert primary.policies.note_policy == "verified_child_or_coauthored"
    assert primary.policies.note_relation_ids == [primary.reasoning_graph.relations[0].relation_id]
    assert primary.policies.note_context == primary_legacy.note_context
    assert primary.policies.reviewed_direct_fallback is not None
    assert primary_legacy.note_direct_conclusion in (primary.policies.reviewed_direct_fallback)
    assert primary.policies.reviewed_coauthored_note == primary_legacy.coauthored_note

    assert transfer.policies.note_policy == "none"
    assert transfer.policies.note_relation_ids == []
    assert transfer.policies.note_skill_id is None
    assert transfer.policies.note_context is None
    assert transfer.policies.reviewed_direct_fallback is None
    assert transfer.policies.reviewed_coauthored_note is None


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_completion_projection_uses_exact_be_keys_from_primary_only(
    scenario_id: str,
) -> None:
    context = _context(scenario_id)
    pack = materialize_amusement_scenario_v2(
        scenario_id,
        {"park_context": context.model_dump(mode="json")},
    )

    assert [projection.output_key for projection in pack.completion_projection] == (
        EXPECTED_PROJECTIONS[scenario_id]
    )
    assert all(
        projection.source_task_id == PARK_PRIMARY_TASK_IDS[scenario_id]
        and projection.source_kind == "fact"
        and projection.source_id == projection.output_key
        and projection.relation_value is None
        for projection in pack.completion_projection
    )


def test_canonical_hash_is_stable_and_changes_with_pinned_context() -> None:
    scenario_id = "amusement_ticket_multiply"
    context = _context(scenario_id)
    scenario_data = {"park_context": context.model_dump(mode="json")}

    first = materialize_amusement_scenario_v2(scenario_id, scenario_data)
    second = materialize_amusement_scenario_v2(scenario_id, scenario_data)
    different_context = generate_park_context(scenario_id, random.Random(9981))
    different = materialize_amusement_scenario_v2(
        scenario_id,
        {"park_context": different_context.model_dump(mode="json")},
    )

    assert canonical_life_scenario_json_v2(first) == canonical_life_scenario_json_v2(second)
    assert life_scenario_hash_v2(first) == life_scenario_hash_v2(second)
    assert life_scenario_hash_v2(first) != life_scenario_hash_v2(different)


def test_rejects_tampered_arithmetic_help_and_projection_contracts() -> None:
    scenario_id = "amusement_snack_divide"
    context = _context(scenario_id)
    scenario_data = {"park_context": context.model_dump(mode="json")}
    pack = materialize_amusement_scenario_v2(scenario_id, scenario_data)

    arithmetic_tamper: dict[str, Any] = pack.model_dump(mode="json")
    primary_stage = arithmetic_tamper["task_stages"][0]
    primary_variant = primary_stage["default_variant_id"]
    facts = primary_stage["variants"][primary_variant]["reasoning_graph"]["facts"]
    result = next(fact for fact in facts if fact["fact_id"] == "per_person")
    result["value"]["amount"] += 1
    with pytest.raises(ValidationError, match="arithmetic contradicts"):
        LifeScenarioPackV2.model_validate(arithmetic_tamper)

    help_tamper: dict[str, Any] = pack.model_dump(mode="json")
    primary_stage = help_tamper["task_stages"][0]
    primary_variant = primary_stage["default_variant_id"]
    primary_stage["variants"][primary_variant]["help_plan"]["H3"]["revealed_fact_ids"] = []
    with pytest.raises(ValidationError, match="H3 must reveal"):
        LifeScenarioPackV2.model_validate(help_tamper)

    projection_tamper: dict[str, Any] = pack.model_dump(mode="json")
    projection_tamper["completion_projection"][0]["source_id"] = "invented_fact"
    with pytest.raises(ValidationError, match="projection fact is missing"):
        LifeScenarioPackV2.model_validate(projection_tamper)

    context_tamper = context.model_dump(mode="json")
    next(fact for fact in context_tamper["facts"] if fact["key"] == "per_person")["value"] += 1
    with pytest.raises(ValueError, match="inconsistent result"):
        materialize_amusement_scenario_v2(
            scenario_id,
            {"park_context": context_tamper},
        )


def test_requires_matching_pinned_park_context() -> None:
    ticket_context = _context("amusement_ticket_multiply")
    with pytest.raises(ValueError, match="park_context is required"):
        materialize_amusement_scenario_v2("amusement_ticket_multiply", {})
    with pytest.raises(ValueError, match="does not belong"):
        materialize_amusement_scenario_v2(
            "amusement_snack_divide",
            {"park_context": ticket_context.model_dump(mode="json")},
        )
    with pytest.raises(KeyError, match="unknown amusement"):
        materialize_amusement_scenario_v2(
            "not_a_scenario",
            {"park_context": ticket_context.model_dump(mode="json")},
        )

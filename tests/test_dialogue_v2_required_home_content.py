from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mormi_api.dialogue_v2_content import (
    AMUSEMENT_PREPARATION_HOME_SESSION_IDS,
    CAFE_REQUIRED_HOME_SESSION_IDS,
    REQUIRED_HOME_SESSION_IDS,
    RequiredHomeContentCatalogV2,
    RequiredHomeTeachingPackV2,
    canonical_catalog_json_v2,
    load_required_home_content_catalog_v2,
    required_home_content_pack_v2,
)
from mormi_api.schemas import MoneyValueV2


def _pack_payload(session_id: str) -> dict[str, Any]:
    return required_home_content_pack_v2(session_id).model_dump(mode="json")


def test_catalog_contains_the_nine_required_home_preparation_sessions() -> None:
    catalog = load_required_home_content_catalog_v2()

    assert set(catalog.by_session_id()) == REQUIRED_HOME_SESSION_IDS
    assert {
        "number-count",
        "number-compare",
        "money-count",
        "money-price",
        "money-budget",
    } == CAFE_REQUIRED_HOME_SESSION_IDS
    assert {
        "multiply-groups",
        "divide-share",
        "divide-group",
        "multiply-easy-tables",
    } == AMUSEMENT_PREPARATION_HOME_SESSION_IDS
    assert all(
        catalog.by_session_id()[session_id].journey.role == "cafe_required"
        for session_id in CAFE_REQUIRED_HOME_SESSION_IDS
    )
    assert all(
        catalog.by_session_id()[session_id].journey.role == "amusement_preparation"
        for session_id in AMUSEMENT_PREPARATION_HOME_SESSION_IDS
    )


@pytest.mark.parametrize("session_id", sorted(REQUIRED_HOME_SESSION_IDS))
def test_v2_pack_owns_a_reviewed_concept_example_without_fe_drill_identity(
    session_id: str,
) -> None:
    """The V2 source is self-contained and never guesses an FE item or seed."""

    pack = required_home_content_pack_v2(session_id)
    source_fields = type(pack.source_problem).model_fields
    assert "question_index" not in source_fields
    assert "item_id" not in source_fields
    assert "variant_seed" not in source_fields
    assert "equation" not in pack.source_problem.visual

    visible_fact_ids = {
        fact.fact_id for fact in pack.reasoning_graph.facts if fact.initially_visible
    }
    assert {
        binding.fact_id
        for binding in pack.source_problem.rendered_facts
        if binding.exposure == "explicit_given"
    } == visible_fact_ids


@pytest.mark.parametrize("session_id", sorted(REQUIRED_HOME_SESSION_IDS))
def test_each_pack_has_deterministic_l2_and_l0_contracts(session_id: str) -> None:
    pack = required_home_content_pack_v2(session_id)
    answer_plan = next(plan for plan in pack.l2_plans if plan.target.ask_kind == "answer")
    expected_choice_ids = [
        f"answer_{index}" for index in range(len(pack.source_problem.answers))
    ]

    assert [choice.choice_id for choice in answer_plan.choices] == expected_choice_ids
    assert [choice.label for choice in answer_plan.choices] == pack.source_problem.answers
    correct_choice = next(
        choice for choice in answer_plan.choices if choice.effect.verdict == "correct"
    )
    assert correct_choice.label == pack.source_problem.correct

    required_targets = {
        *(('fact', target_id) for target_id in pack.reasoning_graph.completion.required_fact_ids),
        *(
            ('relation', target_id)
            for target_id in pack.reasoning_graph.completion.required_relation_ids
        ),
    }
    joint_targets = {
        (completion.target_kind, completion.target_id)
        for completion in pack.l0_joint_plan.completion_values
    }
    assert joint_targets == required_targets
    assert pack.policies.taught_reward_requires_independent_evidence is True
    assert pack.policies.l0_completion_outcome == "supported"
    initial_help = next(
        slot for slot in pack.copy_slots if slot.purpose == "initial_help"
    )
    assert len(initial_help.targets) == 1
    assert initial_help.targets[0].ask_kind == "answer"


@pytest.mark.parametrize("session_id", sorted(REQUIRED_HOME_SESSION_IDS))
def test_help_ladder_reveals_final_truth_only_at_h3(session_id: str) -> None:
    pack = required_home_content_pack_v2(session_id)
    required_facts = set(pack.reasoning_graph.completion.required_fact_ids)
    required_relations = set(pack.reasoning_graph.completion.required_relation_ids)

    assert not required_facts.intersection(pack.help_plan.H1.revealed_fact_ids)
    assert not required_facts.intersection(pack.help_plan.H2.revealed_fact_ids)
    assert required_facts.issubset(pack.help_plan.H3.revealed_fact_ids)
    assert required_relations.issubset(pack.help_plan.H3.revealed_relation_ids)


def test_mixed_budget_pack_preserves_11000_as_correct_partial_progress() -> None:
    pack = required_home_content_pack_v2("multiply-easy-tables")
    facts = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
    relations = {
        relation.relation_id: relation for relation in pack.reasoning_graph.relations
    }

    purchase_total = facts["purchase_total"]
    assert isinstance(purchase_total.value, MoneyValueV2)
    assert purchase_total.value.amount == 11_000
    assert purchase_total.role == "intermediate_result"
    assert purchase_total.required_for_completion is False

    assert relations["sum_item_costs"].input_fact_ids == [
        "ticket_total",
        "drink_total",
        "sticker_total",
    ]
    assert relations["sum_item_costs"].output_fact_id == "purchase_total"
    assert relations["sum_item_costs"].required_for_completion is False

    shortage = facts["shortage"]
    assert isinstance(shortage.value, MoneyValueV2)
    assert shortage.value.amount == 1_000
    assert shortage.required_for_completion is True
    assert relations["calculate_shortage"].input_fact_ids == ["purchase_total", "budget"]
    assert relations["calculate_shortage"].output_fact_id == "shortage"
    assert relations["calculate_shortage"].required_for_completion is True
    assert pack.reasoning_graph.completion.required_fact_ids == ["shortage"]
    assert pack.reasoning_graph.completion.required_relation_ids == [
        "calculate_shortage"
    ]

    # The authored next-focus contract therefore never mistakes 11,000 won
    # for the final answer.  It can be acknowledged while these two targets
    # remain unresolved for the speaker.
    assert {
        (target.target_kind, target.target_id)
        for target in pack.initial_question.targets
    } == {("fact", "shortage"), ("relation", "calculate_shortage")}


def test_copy_slots_store_generation_contracts_but_not_generated_text() -> None:
    payload = _pack_payload("number-count")
    copy_slots = payload["copy_slots"]
    assert isinstance(copy_slots, list)
    copy_slots[0]["generated_text"] = "소스 팩에 쓰면 안 되는 캐시 결과"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RequiredHomeTeachingPackV2.model_validate(payload)


def test_pack_rejects_a_choice_effect_that_changes_canonical_truth() -> None:
    payload = _pack_payload("money-budget")
    l2_plans = payload["l2_plans"]
    assert isinstance(l2_plans, list)
    answer_plan = next(plan for plan in l2_plans if plan["plan_id"] == "l2.answer")
    answer_plan["choices"][0]["effect"]["interpreted_value"]["amount"] = 999

    with pytest.raises(ValidationError, match="canonical graph truth"):
        RequiredHomeTeachingPackV2.model_validate(payload)


def test_pack_rejects_teacher_voice_and_joint_wording_at_l2() -> None:
    teacher_payload = _pack_payload("number-count")
    teacher_payload["initial_question"]["reviewed_fallback"] = (
        "나 헷갈려... 왜 그렇게 생각했어?"
    )
    with pytest.raises(ValidationError, match="cannot evaluate or interrogate"):
        RequiredHomeTeachingPackV2.model_validate(teacher_payload)

    joint_payload = _pack_payload("number-count")
    l2_slot = next(
        slot
        for slot in joint_payload["copy_slots"]
        if slot["copy_slot"] == "number-count.l2.answer"
    )
    l2_slot["reviewed_fallback"] = (
        "나 점 개수가 헷갈려... 여기에서 같이 골라서 알려줄 수 있어?"
    )
    with pytest.raises(ValidationError, match="selection, not joint"):
        RequiredHomeTeachingPackV2.model_validate(joint_payload)


def test_pack_rejects_a_canonical_result_that_breaks_authored_arithmetic() -> None:
    payload = _pack_payload("money-count")
    total = next(
        fact for fact in payload["reasoning_graph"]["facts"] if fact["fact_id"] == "total_amount"
    )
    total["value"]["amount"] = 601
    total["accepted_surface_forms"] = ["601원", "601"]
    payload["source_problem"]["answers"][0] = "601원"
    payload["source_problem"]["correct"] = "601원"
    answer_plan = next(
        plan for plan in payload["l2_plans"] if plan["plan_id"] == "l2.answer"
    )
    answer_plan["choices"][0]["label"] = "601원"
    answer_plan["choices"][0]["effect"]["interpreted_value"]["amount"] = 601
    payload["l0_joint_plan"]["completion_values"][0]["value"]["amount"] = 601
    payload["help_plan"]["H3"]["body"] = "500+100=601이라서 모두 601원이야."
    payload["help_plan"]["H3"]["action"] = "500+100=601을 함께 확인하기"

    with pytest.raises(ValidationError, match="arithmetic result is invalid"):
        RequiredHomeTeachingPackV2.model_validate(payload)


def test_pack_rejects_visual_truth_drift_and_h0_operation_leak() -> None:
    drift_payload = _pack_payload("number-count")
    drift_payload["source_problem"]["visual"]["count"] = 9
    with pytest.raises(ValidationError, match="does not show its canonical value"):
        RequiredHomeTeachingPackV2.model_validate(drift_payload)

    leak_payload = _pack_payload("money-count")
    leak_payload["source_problem"]["visual"]["equation"] = "500+100"
    with pytest.raises(ValidationError, match="H0 visual cannot reveal"):
        RequiredHomeTeachingPackV2.model_validate(leak_payload)


def test_pack_rejects_dangling_copy_targets_and_duplicate_plan_routes() -> None:
    dangling_payload = _pack_payload("number-count")
    dangling_payload["copy_slots"][0]["targets"][0]["target_id"] = "ghost_fact"
    with pytest.raises(ValidationError, match="question target must reference"):
        RequiredHomeTeachingPackV2.model_validate(dangling_payload)

    duplicate_payload = _pack_payload("number-count")
    duplicate_payload["l3_plans"][1]["plan_id"] = duplicate_payload["l3_plans"][0][
        "plan_id"
    ]
    with pytest.raises(ValidationError, match="question plan ids must be unique"):
        RequiredHomeTeachingPackV2.model_validate(duplicate_payload)


def test_pack_rejects_h2_required_truth_and_non_neutral_help_copy() -> None:
    reveal_payload = _pack_payload("number-count")
    reveal_payload["help_plan"]["H2"]["revealed_relation_ids"] = ["count_each_once"]
    with pytest.raises(ValidationError, match="H2 cannot reveal a required relation"):
        RequiredHomeTeachingPackV2.model_validate(reveal_payload)

    voice_payload = _pack_payload("number-count")
    voice_payload["help_plan"]["H1"]["action"] = "나랑 같이 점을 살펴보자"
    with pytest.raises(ValidationError, match="neutral system scaffolds"):
        RequiredHomeTeachingPackV2.model_validate(voice_payload)


def test_pack_rejects_copy_or_alias_values_that_disagree_with_truth() -> None:
    help_payload = _pack_payload("money-count")
    help_payload["help_plan"]["H3"]["body"] = "500원에 100원을 더하면 700원이야."
    help_payload["help_plan"]["H3"]["action"] = "500+100=700을 함께 확인하기"
    with pytest.raises(ValidationError, match="canonical final numeric value"):
        RequiredHomeTeachingPackV2.model_validate(help_payload)

    alias_payload = _pack_payload("money-count")
    total = next(
        fact
        for fact in alias_payload["reasoning_graph"]["facts"]
        if fact["fact_id"] == "total_amount"
    )
    total["accepted_surface_forms"] = ["600원", "601"]
    with pytest.raises(ValidationError, match="surface form must equal canonical truth"):
        RequiredHomeTeachingPackV2.model_validate(alias_payload)


def test_pack_rejects_wrong_equation_roles_and_hidden_l3_answer_copy() -> None:
    equation_payload = _pack_payload("money-count")
    equation_payload["help_plan"]["H3"]["body"] = (
        "500+600=100이라서 모두 600원이야."
    )
    with pytest.raises(ValidationError, match="equation is mathematically invalid"):
        RequiredHomeTeachingPackV2.model_validate(equation_payload)

    leak_payload = _pack_payload("number-count")
    leak_payload["l3_plans"][0]["reviewed_fallback"] = (
        "나 점이 3개인지 헷갈려... 점이 3개인지 알려줄 수 있어?"
    )
    with pytest.raises(ValidationError, match="cannot reveal a hidden value"):
        RequiredHomeTeachingPackV2.model_validate(leak_payload)

    choice_leak_payload = _pack_payload("number-compare")
    choice_leak_payload["l3_plans"][0]["reviewed_fallback"] = (
        "나 오른쪽인지 헷갈려... 어느 쪽인지 알려줄 수 있어?"
    )
    with pytest.raises(ValidationError, match="cannot reveal a hidden value"):
        RequiredHomeTeachingPackV2.model_validate(choice_leak_payload)


def test_pack_rejects_visual_label_or_duplicate_count_drift() -> None:
    label_payload = _pack_payload("money-price")
    label_payload["source_problem"]["visual"]["labels"] = ["빵", "주스"]
    with pytest.raises(ValidationError, match="label does not match its role"):
        RequiredHomeTeachingPackV2.model_validate(label_payload)

    count_payload = _pack_payload("multiply-groups")
    count_payload["source_problem"]["visual"]["items"][0]["count"] = 3
    with pytest.raises(ValidationError, match="does not show its canonical value"):
        RequiredHomeTeachingPackV2.model_validate(count_payload)


def test_pack_rejects_prompt_or_source_choice_drift() -> None:
    prompt_payload = _pack_payload("multiply-groups")
    prompt_payload["source_problem"]["prompt"] = (
        "5,000원짜리 장난감 4개는 모두 얼마일까?"
    )
    with pytest.raises(ValidationError, match="only initially visible numeric facts"):
        RequiredHomeTeachingPackV2.model_validate(prompt_payload)

    answer_payload = _pack_payload("money-count")
    answer_payload["source_problem"]["answers"][1] = "999원"
    with pytest.raises(ValidationError, match="ordered L2 answer choices"):
        RequiredHomeTeachingPackV2.model_validate(answer_payload)


def test_pack_rejects_intermediate_alias_or_rubric_truth_drift() -> None:
    alias_payload = _pack_payload("multiply-easy-tables")
    purchase_total = next(
        fact
        for fact in alias_payload["reasoning_graph"]["facts"]
        if fact["fact_id"] == "purchase_total"
    )
    purchase_total["accepted_surface_forms"] = ["999원", "999"]
    with pytest.raises(ValidationError, match="surface form must equal canonical truth"):
        RequiredHomeTeachingPackV2.model_validate(alias_payload)

    rubric_payload = _pack_payload("multiply-easy-tables")
    sum_relation = next(
        relation
        for relation in rubric_payload["reasoning_graph"]["relations"]
        if relation["relation_id"] == "sum_item_costs"
    )
    sum_relation["rubric"]["sufficient"][0] = (
        "5,000원, 3,000원, 3,000원을 더해 12,000원을 구한다고 말한다"
    )
    with pytest.raises(ValidationError, match="rubric contains an undeclared value"):
        RequiredHomeTeachingPackV2.model_validate(rubric_payload)


def test_catalog_rejects_a_missing_required_session() -> None:
    payload = load_required_home_content_catalog_v2().model_dump(mode="json")
    payload["packs"] = payload["packs"][:-1]

    with pytest.raises(ValidationError):
        RequiredHomeContentCatalogV2.model_validate(payload)


def test_catalog_canonical_serialization_is_stable_and_unicode_preserving() -> None:
    first = canonical_catalog_json_v2()
    second = canonical_catalog_json_v2()

    assert first == second
    assert "수를 빠뜨리지 않고 세어요" in first
    assert "\\uC218" not in first

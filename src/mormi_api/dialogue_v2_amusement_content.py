"""Native V2 materializer for the three amusement-park teaching scenes.

``scenario_data.park_context`` is generated once by the AI service and pinned
to the conversation.  It is the arithmetic source of truth here.  The legacy
task builders remain the reviewed catalogue for Mormi copy, dictionary cards,
help cards, and FE visual payloads; this module translates that reviewed
content into the typed V2 life-content contract without regenerating numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from .content import (
    PARK_PRIMARY_TASK_IDS,
    PARK_SCENARIO_IDS,
    PARK_TRANSFER_TASK_IDS,
    TaskDefinition,
    get_task,
)
from .dialogue_v2_content import (
    CopySlotV2,
    QuestionPlanV2,
    RelationRubricV2,
    TargetRefV2,
)
from .dialogue_v2_life_content import (
    LifeChoiceEffectV2,
    LifeChoiceV2,
    LifeCompletionContractV2,
    LifeCompletionProjectionV2,
    LifeFactUpdateV2,
    LifeFactV2,
    LifeHelpCardV2,
    LifeHelpPlanV2,
    LifeJointFactCompletionV2,
    LifeJointRelationCompletionV2,
    LifeL0JointPlanV2,
    LifeL2ChoicePlanV2,
    LifeReasoningGraphV2,
    LifeRelationV2,
    LifeScenarioPackV2,
    LifeTaskPackV2,
    LifeTaskPoliciesV2,
    LifeTaskStageV2,
)
from .schemas import (
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
    ParkSessionContext,
    SceneType,
)

AMUSEMENT_NATIVE_V2_SCENARIO_IDS = frozenset(PARK_SCENARIO_IDS)
# Short alias retained for callers that adopted the first materializer draft.
AMUSEMENT_V2_SCENARIO_IDS = AMUSEMENT_NATIVE_V2_SCENARIO_IDS
AMUSEMENT_CONTENT_VERSION_V2 = 3


@dataclass(frozen=True)
class _ScenarioSpec:
    answer_fact_id: str
    given_fact_ids: tuple[str, str]
    relation_id: str
    operation: Literal["multiplication", "division"]
    projection_keys: tuple[str, ...]


_SPECS: dict[str, _ScenarioSpec] = {
    "amusement_ticket_multiply": _ScenarioSpec(
        answer_fact_id="total_price",
        given_fact_ids=("ticket_price", "party_count"),
        relation_id="calculate_ticket_total",
        operation="multiplication",
        projection_keys=("ticket_price", "party_count", "total_price"),
    ),
    "amusement_snack_divide": _ScenarioSpec(
        answer_fact_id="per_person",
        given_fact_ids=("snack_total", "payer_count"),
        relation_id="divide_snack_equally",
        operation="division",
        projection_keys=("snack_total", "payer_count", "per_person"),
    ),
    "amusement_pass_compare": _ScenarioSpec(
        answer_fact_id="break_even_rides",
        given_fact_ids=("day_pass_price", "single_ride_price"),
        relation_id="find_pass_break_even",
        operation="division",
        projection_keys=(
            "single_ride_price",
            "day_pass_price",
            "break_even_rides",
            "benefit_from_rides",
        ),
    ),
}


def _context_from_scenario_data(
    scenario_id: str,
    scenario_data: Mapping[str, Any],
) -> ParkSessionContext:
    raw = scenario_data.get("park_context")
    if raw is None:
        raise ValueError("scenario_data.park_context is required for amusement V2")
    context = raw if isinstance(raw, ParkSessionContext) else ParkSessionContext.model_validate(raw)
    expected_stage = {
        "amusement_ticket_multiply": "ticket",
        "amusement_snack_divide": "snack_split",
        "amusement_pass_compare": "pass_break_even",
    }[scenario_id]
    if context.stage_id != expected_stage:
        raise ValueError(
            "scenario_data.park_context does not belong to the requested scenario: "
            f"{context.stage_id!r} != {expected_stage!r}"
        )
    return context


def _context_token(context: ParkSessionContext) -> str:
    encoded = json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"park-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _entry_level(
    task_id: str,
    task_start_levels: Mapping[str, ExpressionLevel | str] | None,
) -> Literal[ExpressionLevel.L4, ExpressionLevel.L2]:
    level = ExpressionLevel((task_start_levels or {}).get(task_id, ExpressionLevel.L4)).canonical()
    if level not in {ExpressionLevel.L4, ExpressionLevel.L2}:
        raise ValueError(f"amusement V2 entry must be L4 or L2: {task_id}={level}")
    return cast("Literal[ExpressionLevel.L4, ExpressionLevel.L2]", level)


def _value(value: int, unit: str) -> MoneyValueV2 | NumberValueV2:
    return MoneyValueV2(amount=value) if unit == "원" else NumberValueV2(value=value, unit=unit)


def _surfaces(value: int, unit: str) -> list[str]:
    return list(dict.fromkeys([f"{value:,}{unit}", f"{value}{unit}", f"{value:,}", str(value)]))


_FACT_SPEAKER_LABELS = {
    "ticket_price": "입장권의 개당 값",
    "party_count": "함께 갈 사람 수",
    "total_price": "전체 입장권 값",
    "snack_total": "간식 전체 값",
    "payer_count": "함께 낼 사람 수",
    "per_person": "각자 낼 값",
    "day_pass_price": "자유이용권 값",
    "single_ride_price": "일반 이용권 값",
    "break_even_rides": "두 값이 같아지는 횟수",
    "benefit_from_rides": "자유이용권이 더 저렴한 시작 횟수",
}


def _fact(
    fact_id: str,
    value: int,
    unit: str,
    label: str,
    *,
    role: Literal[
        "visible_condition",
        "visible_value",
        "intermediate_result",
        "final_answer",
        "selection",
    ],
    visible: bool,
    required: bool,
) -> LifeFactV2:
    return LifeFactV2(
        fact_id=fact_id,
        role=role,
        value=_value(value, unit),
        # SpeakerAllowedFactV2 treats number words as semantic truth.  Catalog
        # labels therefore describe the role without counters such as "한 장"
        # or literals such as "1회"; the typed value carries the actual number.
        speaker_label=_FACT_SPEAKER_LABELS.get(fact_id, label),
        initially_visible=visible,
        required_for_completion=required,
        accepted_surface_forms=_surfaces(value, unit),
    )


def _facts_for_task(
    scenario_id: str,
    legacy: TaskDefinition,
) -> tuple[list[LifeFactV2], dict[str, int]]:
    arithmetic = legacy.arithmetic_contract
    if arithmetic is None:
        raise ValueError(f"amusement task has no arithmetic contract: {legacy.id}")
    spec = _SPECS[scenario_id]
    if arithmetic.operation != spec.operation:
        raise ValueError(f"amusement arithmetic operation drifted: {legacy.id}")

    left_id, right_id = spec.given_fact_ids
    units = {
        "amusement_ticket_multiply": ("원", "명", "원"),
        "amusement_snack_divide": ("원", "명", "원"),
        "amusement_pass_compare": ("원", "원", "번"),
    }[scenario_id]
    values = {
        left_id: arithmetic.left,
        right_id: arithmetic.right,
        spec.answer_fact_id: arithmetic.result,
    }
    facts = [
        _fact(
            left_id,
            arithmetic.left,
            units[0],
            arithmetic.left_label,
            role="visible_value",
            visible=True,
            required=False,
        ),
        _fact(
            right_id,
            arithmetic.right,
            units[1],
            arithmetic.right_label,
            role="visible_value",
            visible=True,
            required=False,
        ),
        _fact(
            spec.answer_fact_id,
            arithmetic.result,
            units[2],
            legacy.slots["answer"].description,
            role=(
                "intermediate_result" if scenario_id == "amusement_pass_compare" else "final_answer"
            ),
            visible=False,
            required=True,
        ),
    ]
    if scenario_id == "amusement_pass_compare":
        benefit = arithmetic.result + 1
        if not legacy.slots["benefit_from_rides"].accepts(benefit):
            raise ValueError("amusement pass benefit does not match its reviewed slot")
        values["benefit_from_rides"] = benefit
        facts.append(
            _fact(
                "benefit_from_rides",
                benefit,
                "번",
                legacy.slots["benefit_from_rides"].description,
                role="final_answer",
                visible=False,
                required=True,
            )
        )
    return facts, values


def _relation(
    scenario_id: str,
    legacy: TaskDefinition,
    values: Mapping[str, int],
) -> LifeRelationV2:
    spec = _SPECS[scenario_id]
    left_id, right_id = spec.given_fact_ids
    answer_id = spec.answer_fact_id
    equation = {
        "multiplication": "×",
        "division": "÷",
    }[spec.operation]
    sufficient = list(dict.fromkeys(legacy.accepted_methods))
    partial = [
        f"{values[left_id]:,}과 {values[right_id]:,}을 사용하지만 "
        f"{equation} 관계를 아직 설명하지 못함"
    ]
    incorrect = [
        "두 값을 더하거나 한 값만 사용해 전체 관계를 만들지 못함"
        if spec.operation == "multiplication"
        else "전체를 같은 크기로 나누거나 몇 묶음인지 찾는 관계가 아님"
    ]
    return LifeRelationV2(
        relation_id=spec.relation_id,
        role="procedure_step",
        operation=spec.operation,
        input_fact_ids=[left_id, right_id],
        output_fact_id=answer_id,
        evaluation_mode="open_semantic_support",
        speaker_label=legacy.slots["strategy"].description,
        rubric=RelationRubricV2(
            sufficient=sufficient,
            partial=partial,
            incorrect=incorrect,
        ),
        required_for_completion=True,
    )


def _target(scenario_id: str, slot_id: str) -> TargetRefV2:
    spec = _SPECS[scenario_id]
    if slot_id == "answer":
        return TargetRefV2(
            target_kind="fact",
            target_id=spec.answer_fact_id,
            ask_kind="answer",
        )
    if slot_id == "benefit_from_rides":
        return TargetRefV2(
            target_kind="fact",
            target_id="benefit_from_rides",
            ask_kind="answer",
        )
    if slot_id == "strategy":
        return TargetRefV2(
            target_kind="relation",
            target_id=spec.relation_id,
            ask_kind="reason_or_method",
        )
    raise ValueError(f"unknown amusement slot: {slot_id}")


def _question(plan_id: str, targets: Sequence[TargetRefV2], fallback: str) -> QuestionPlanV2:
    return QuestionPlanV2(
        plan_id=plan_id,
        targets=list(targets),
        reviewed_fallback=fallback,
    )


def _copy_slot(
    slot_id: str,
    purpose: Literal["initial_help", "l2_question", "l0_intro", "l0_action"],
    targets: Sequence[TargetRefV2],
    fallback: str,
    brief: str,
) -> CopySlotV2:
    return CopySlotV2(
        copy_slot=slot_id,
        purpose=purpose,
        targets=list(targets),
        generation_brief=brief,
        reviewed_fallback=fallback,
    )


def _l2_plan(
    scenario_id: str,
    phase: str,
    legacy: TaskDefinition,
    step: Any,
    target: TargetRefV2,
    fact_by_id: Mapping[str, LifeFactV2],
) -> LifeL2ChoicePlanV2:
    slot_id = step.target_slots[0]
    slot = legacy.slots[slot_id]
    choices: list[LifeChoiceV2] = []
    for option in step.input.choices:
        selected = step.choice_effects[option.id][slot_id]
        correct = slot.accepts(selected)
        if correct and target.target_kind == "fact":
            effect = LifeChoiceEffectV2(
                verdict="correct",
                fact_updates=[
                    LifeFactUpdateV2(
                        fact_id=target.target_id,
                        value=fact_by_id[target.target_id].value,
                    )
                ],
            )
        elif correct:
            effect = LifeChoiceEffectV2(
                verdict="correct",
                relation_ids=[target.target_id],
            )
        else:
            effect = LifeChoiceEffectV2(
                verdict="incorrect",
                misconception_code=(f"{scenario_id}.{phase}.{slot_id}.{option.id}")[:120],
            )
        choices.append(
            LifeChoiceV2(
                choice_id=option.id,
                label=option.label,
                image_url=option.image_url,
                disabled=bool(option.disabled),
                effect=effect,
            )
        )
    return LifeL2ChoicePlanV2(
        plan_id=f"l2.{step.id}",
        target=target,
        copy_slot=f"l2.{slot_id}",
        choices=choices,
        input_config=deepcopy(step.input.config),
        submit_label=step.input.submit_label or "알려주기",
    )


def _l0_plan(
    scenario_id: str,
    legacy: TaskDefinition,
    step: Any,
    fact_by_id: Mapping[str, LifeFactV2],
) -> LifeL0JointPlanV2:
    raw_values = step.input.config.get("completion_values")
    if not isinstance(raw_values, Mapping):
        raise ValueError(f"amusement joint action has no completion values: {step.id}")
    completions: list[LifeJointFactCompletionV2 | LifeJointRelationCompletionV2] = []
    for slot_id in legacy.required_slots:
        if slot_id not in raw_values or not legacy.slots[slot_id].accepts(raw_values[slot_id]):
            raise ValueError(f"amusement joint value drifted from reviewed truth: {slot_id}")
        target = _target(scenario_id, slot_id)
        if target.target_kind == "fact":
            completions.append(
                LifeJointFactCompletionV2(
                    target_id=target.target_id,
                    value=fact_by_id[target.target_id].value,
                )
            )
        else:
            completions.append(LifeJointRelationCompletionV2(target_id=target.target_id))
    input_config = {
        key: deepcopy(value)
        for key, value in step.input.config.items()
        if key != "completion_values"
    }
    return LifeL0JointPlanV2(
        action_id=step.id,
        intro_copy_slot="l0.intro",
        action_copy_slot="l0.action",
        completion_values=completions,
        button_label=step.input.submit_label or "같이 해보기",
        input_config=input_config,
    )


def _help_plan(
    legacy: TaskDefinition,
    required_targets: Sequence[TargetRefV2],
) -> LifeHelpPlanV2:
    required_fact_ids = [
        target.target_id for target in required_targets if target.target_kind == "fact"
    ]
    required_relation_ids = [
        target.target_id for target in required_targets if target.target_kind == "relation"
    ]

    def card(level: HintLevel) -> LifeHelpCardV2:
        hint = legacy.hints[level]
        is_h3 = level is HintLevel.H3
        return LifeHelpCardV2(
            level=level,
            support_type=hint.support_type,
            answer_policy=hint.answer_policy,
            body=hint.body,
            action=hint.action,
            visual_type=hint.visual_type,
            visual_data=deepcopy(hint.visual_data),
            revealed_fact_ids=required_fact_ids if is_h3 else [],
            revealed_relation_ids=required_relation_ids if is_h3 else [],
        )

    return LifeHelpPlanV2(
        H1=card(HintLevel.H1),
        H2=card(HintLevel.H2),
        H3=card(HintLevel.H3),
    )


def _materialize_task(
    scenario_id: str,
    context: ParkSessionContext,
    legacy: TaskDefinition,
    *,
    phase: Literal["primary", "transfer"],
    entry_level: Literal[ExpressionLevel.L4, ExpressionLevel.L2],
) -> LifeTaskPackV2:
    facts, values = _facts_for_task(scenario_id, legacy)
    relation = _relation(scenario_id, legacy, values)
    fact_by_id = {fact.fact_id: fact for fact in facts}
    required_fact_ids = [fact.fact_id for fact in facts if fact.required_for_completion]
    graph = LifeReasoningGraphV2(
        facts=facts,
        relations=[relation],
        completion=LifeCompletionContractV2(
            required_fact_ids=required_fact_ids,
            required_relation_ids=[relation.relation_id],
        ),
    )
    required_targets = [_target(scenario_id, slot_id) for slot_id in legacy.required_slots]
    l4_step = legacy.steps[ExpressionLevel.L4][0]
    l3_steps = legacy.steps[ExpressionLevel.L3]
    l2_steps = legacy.steps[ExpressionLevel.L2]
    l0_step = legacy.steps[ExpressionLevel.L0][0]

    copy_slots = [
        _copy_slot(
            "initial_help",
            "initial_help",
            required_targets,
            l3_steps[0].fallback_text,
            "아이가 첫 질문에서 막혔을 때 모르미가 작은 단서 하나를 부탁한다.",
        ),
        _copy_slot(
            "l0.intro",
            "l0_intro",
            required_targets,
            l0_step.fallback_text,
            "모르미가 도움 카드 순서대로 함께 해 보자고 부탁한다.",
        ),
        _copy_slot(
            "l0.action",
            "l0_action",
            required_targets,
            l0_step.fallback_text,
            "공동수행 버튼 앞에서 모르미가 짧게 함께 하자고 부탁한다.",
        ),
    ]
    l3_plans: list[QuestionPlanV2] = []
    l2_plans: list[LifeL2ChoicePlanV2] = []
    for step in l3_steps:
        target = _target(scenario_id, step.target_slots[0])
        l3_plans.append(_question(f"l3.{step.id}", [target], step.fallback_text))
    for step in l2_steps:
        slot_id = step.target_slots[0]
        target = _target(scenario_id, slot_id)
        l2_plans.append(_l2_plan(scenario_id, phase, legacy, step, target, fact_by_id))
        copy_slots.append(
            _copy_slot(
                f"l2.{slot_id}",
                "l2_question",
                [target],
                step.fallback_text,
                f"모르미가 {legacy.slots[slot_id].description} 선택지를 골라 달라고 부탁한다.",
            )
        )

    note_enabled = phase == "primary"
    policies = LifeTaskPoliciesV2(
        entry_expression_level=entry_level,
        note_policy="verified_child_or_coauthored" if note_enabled else "none",
        note_relation_ids=[relation.relation_id] if note_enabled else [],
        note_skill_id=legacy.skill_id if note_enabled else None,
        note_context=legacy.note_context if note_enabled else None,
        reviewed_direct_fallback=(
            legacy.note_direct_conclusion.strip() or legacy.coauthored_note
            if note_enabled
            else None
        ),
        reviewed_coauthored_note=legacy.coauthored_note if note_enabled else None,
        transition_text=(
            "아까 같이 살펴본 방법이 숫자가 바뀌어도 되는지 궁금해..."
            if note_enabled
            else None
        ),
    )

    return LifeTaskPackV2(
        pack_id=f"amusement.{scenario_id}.{_context_token(context)}.{phase}.v2",
        content_version=AMUSEMENT_CONTENT_VERSION_V2,
        scene=SceneType.AMUSEMENT_PARK,
        scenario_id=scenario_id,
        task_id=legacy.id,
        stage_id=legacy.stage_id,
        phase=phase,
        skill_id=legacy.skill_id,
        title=legacy.title,
        dictionary_card_id=legacy.dictionary_card_id,
        source_prompt=l4_step.prompt,
        base_visual=legacy.base_visual.model_copy(deep=True),
        reasoning_graph=graph,
        initial_question=_question("initial", required_targets, l4_step.fallback_text),
        l3_plans=l3_plans,
        copy_slots=copy_slots,
        l2_plans=l2_plans,
        l0_joint_plan=_l0_plan(scenario_id, legacy, l0_step, fact_by_id),
        help_plan=_help_plan(legacy, required_targets),
        policies=policies,
    )


def materialize_amusement_scenario_v2(
    scenario_id: str,
    scenario_data: Mapping[str, Any],
    *,
    task_start_levels: Mapping[str, ExpressionLevel | str] | None = None,
) -> LifeScenarioPackV2:
    """Materialize the primary and transfer task from one pinned park context."""

    if scenario_id not in AMUSEMENT_V2_SCENARIO_IDS:
        raise KeyError(f"unknown amusement V2 scenario: {scenario_id}")
    context = _context_from_scenario_data(scenario_id, scenario_data)
    frozen_data = {"park_context": context.model_dump(mode="json")}
    primary_task_id = PARK_PRIMARY_TASK_IDS[scenario_id]
    transfer_task_id = PARK_TRANSFER_TASK_IDS[scenario_id]
    primary = _materialize_task(
        scenario_id,
        context,
        get_task(primary_task_id, frozen_data),
        phase="primary",
        entry_level=_entry_level(primary_task_id, task_start_levels),
    )
    transfer = _materialize_task(
        scenario_id,
        context,
        get_task(transfer_task_id, frozen_data),
        phase="transfer",
        entry_level=_entry_level(transfer_task_id, task_start_levels),
    )
    token = _context_token(context)
    primary_variant = f"{token}.primary"
    transfer_variant = f"{token}.transfer"
    return LifeScenarioPackV2(
        pack_id=f"amusement.{scenario_id}.{token}.scenario.v2",
        content_version=AMUSEMENT_CONTENT_VERSION_V2,
        scene=SceneType.AMUSEMENT_PARK,
        scenario_id=scenario_id,
        task_stages=[
            LifeTaskStageV2(
                task_id=primary_task_id,
                default_variant_id=primary_variant,
                variants={primary_variant: primary},
            ),
            LifeTaskStageV2(
                task_id=transfer_task_id,
                default_variant_id=transfer_variant,
                variants={transfer_variant: transfer},
            ),
        ],
        completion_projection=[
            LifeCompletionProjectionV2(
                output_key=key,
                source_task_id=primary_task_id,
                source_kind="fact",
                source_id=key,
            )
            for key in _SPECS[scenario_id].projection_keys
        ],
    )


__all__ = [
    "AMUSEMENT_CONTENT_VERSION_V2",
    "AMUSEMENT_NATIVE_V2_SCENARIO_IDS",
    "AMUSEMENT_V2_SCENARIO_IDS",
    "materialize_amusement_scenario_v2",
]

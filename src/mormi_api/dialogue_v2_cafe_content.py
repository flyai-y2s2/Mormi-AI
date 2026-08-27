from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .content import (
    BUDGET_MENU_TASK_ID,
    CAFE_CHANGE_PAYMENT_AMOUNT,
    CHANGE_TASK_ID,
    QUEUE_TASK_ID,
    TOTAL_CALC_TASK_ID,
    TOTAL_MENU_PICK_TASK_ID,
    TaskDefinition,
    create_scenario_data,
    get_scenario,
    get_task,
)
from .dialogue_v2_content import CopySlotV2, QuestionPlanV2, RelationRubricV2, TargetRefV2
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
    LifePriorFactVariantSelectorV2,
    LifeReasoningGraphV2,
    LifeRelationV2,
    LifeScenarioPackV2,
    LifeTaskPackV2,
    LifeTaskPoliciesV2,
    LifeTaskStageV2,
)
from .schemas import (
    CafeMenuItem,
    CafeSessionContext,
    CanonicalValueV2,
    ChoiceValueV2,
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
    QueueSessionContext,
    SceneType,
)

CAFE_NATIVE_V2_SCENARIO_IDS = frozenset(
    {
        "cafe_queue",
        "cafe_budget_menu",
        "cafe_menu_total",
        "cafe_change",
    }
)


def _money(amount: int) -> MoneyValueV2:
    return MoneyValueV2(amount=amount)


def _count(value: int) -> NumberValueV2:
    return NumberValueV2(value=value, unit="명")


def _choice(value: str) -> ChoiceValueV2:
    return ChoiceValueV2(choice_id=value)


def _surface_forms(*values: str) -> list[str]:
    return list(dict.fromkeys(values))


def _fact_target(fact_id: str) -> TargetRefV2:
    return TargetRefV2(target_kind="fact", target_id=fact_id, ask_kind="answer")


def _relation_target(relation_id: str) -> TargetRefV2:
    return TargetRefV2(
        target_kind="relation",
        target_id=relation_id,
        ask_kind="reason_or_method",
    )


def _question(plan_id: str, targets: Sequence[TargetRefV2], fallback: str) -> QuestionPlanV2:
    return QuestionPlanV2(
        plan_id=plan_id,
        targets=list(targets),
        reviewed_fallback=fallback,
    )


def _copy_slot(
    copy_slot: str,
    purpose: str,
    targets: Sequence[TargetRefV2],
    fallback: str,
) -> CopySlotV2:
    return CopySlotV2.model_validate(
        {
            "copy_slot": copy_slot,
            "purpose": purpose,
            "targets": list(targets),
            "generation_brief": (
                "모르미가 아이를 평가하지 않고, 자신이 헷갈려 도움을 구하는 한 문장으로 말한다."
            ),
            "reviewed_fallback": fallback,
        }
    )


def _copy_slots(
    *,
    prefix: str,
    required_targets: Sequence[TargetRefV2],
    initial_help_target: TargetRefV2,
    initial_help_fallback: str,
    l2_items: Sequence[tuple[str, TargetRefV2, str]],
    l0_intro: str,
    l0_action: str,
) -> list[CopySlotV2]:
    slots = [
        _copy_slot(
            f"{prefix}.initial-help",
            "initial_help",
            [initial_help_target],
            initial_help_fallback,
        )
    ]
    slots.extend(
        _copy_slot(copy_id, "l2_question", [target], fallback)
        for copy_id, target, fallback in l2_items
    )
    slots.extend(
        [
            _copy_slot(
                f"{prefix}.l0-intro",
                "l0_intro",
                required_targets,
                l0_intro,
            ),
            _copy_slot(
                f"{prefix}.l0-action",
                "l0_action",
                required_targets,
                l0_action,
            ),
        ]
    )
    return slots


def _help_card(
    task: TaskDefinition,
    level: HintLevel,
    *,
    revealed_fact_ids: Sequence[str] = (),
    revealed_relation_ids: Sequence[str] = (),
) -> LifeHelpCardV2:
    source = task.hints[level]
    return LifeHelpCardV2(
        level=level,
        support_type=source.support_type,
        answer_policy=source.answer_policy,
        body=source.body,
        action=source.action,
        visual_type=source.visual_type,
        visual_data=dict(source.visual_data),
        revealed_fact_ids=list(revealed_fact_ids),
        revealed_relation_ids=list(revealed_relation_ids),
    )


def _help_plan(
    task: TaskDefinition,
    *,
    required_fact_ids: Sequence[str],
    required_relation_ids: Sequence[str],
) -> LifeHelpPlanV2:
    return LifeHelpPlanV2(
        H1=_help_card(task, HintLevel.H1),
        H2=_help_card(task, HintLevel.H2),
        H3=_help_card(
            task,
            HintLevel.H3,
            revealed_fact_ids=required_fact_ids,
            revealed_relation_ids=required_relation_ids,
        ),
    )


def _note_policy(
    task: TaskDefinition,
    relation_ids: Sequence[str],
    *,
    enabled: bool,
) -> LifeTaskPoliciesV2:
    if not enabled:
        return LifeTaskPoliciesV2(
            entry_expression_level=(
                ExpressionLevel.L2 if task.behavior == "menu_selection" else ExpressionLevel.L4
            ),
            note_policy="none",
            transition_text=task.transition_text,
        )
    direct_fallback = task.note_direct_conclusion.strip() or task.coauthored_note
    return LifeTaskPoliciesV2(
        entry_expression_level=(
            ExpressionLevel.L2
            if task.behavior in {"menu_selection", "budget_menu_selection"}
            else ExpressionLevel.L4
        ),
        note_policy="verified_child_or_coauthored",
        note_relation_ids=list(relation_ids),
        note_skill_id=task.skill_id,
        note_context=task.note_context,
        reviewed_direct_fallback=direct_fallback,
        reviewed_coauthored_note=task.coauthored_note,
        transition_text=task.transition_text,
    )


def _incorrect_effect(
    code: str,
    *,
    visual_patch: Mapping[str, Any] | None = None,
) -> LifeChoiceEffectV2:
    return LifeChoiceEffectV2(
        verdict="incorrect",
        misconception_code=code,
        visual_patch=dict(visual_patch or {}),
    )


def _queue_pack(scenario_data: Mapping[str, Any]) -> LifeTaskPackV2:
    task = get_task(QUEUE_TASK_ID, scenario_data)
    left = int(scenario_data["left_count"])
    right = int(scenario_data["right_count"])
    side = "left" if left < right else "right"

    left_target = _fact_target("left_count")
    right_target = _fact_target("right_count")
    side_target = _fact_target("final_choice")
    reason_target = _relation_target("choose_shorter_queue")
    required_targets = [left_target, right_target, side_target, reason_target]

    facts = [
        LifeFactV2(
            fact_id="left_count",
            role="intermediate_result",
            value=_count(left),
            speaker_label="왼쪽 줄 사람 수",
            initially_visible=False,
            required_for_completion=True,
            accepted_surface_forms=list(task.slots["left_count"].aliases),
        ),
        LifeFactV2(
            fact_id="right_count",
            role="intermediate_result",
            value=_count(right),
            speaker_label="오른쪽 줄 사람 수",
            initially_visible=False,
            required_for_completion=True,
            accepted_surface_forms=list(task.slots["right_count"].aliases),
        ),
        LifeFactV2(
            fact_id="final_choice",
            role="selection",
            value=_choice(side),
            speaker_label="사람이 적어 차례가 더 빨리 오는 줄",
            initially_visible=False,
            required_for_completion=True,
            accepted_surface_forms=list(task.slots["final_choice"].aliases),
        ),
    ]
    relation = LifeRelationV2(
        relation_id="choose_shorter_queue",
        role="selection_rule",
        operation="comparison",
        comparison_goal="minimum",
        input_fact_ids=["left_count", "right_count"],
        output_fact_id="final_choice",
        evaluation_mode="open_semantic_support",
        speaker_label="앞에 기다리는 사람이 적은 줄을 고르면 차례가 더 빨리 와",
        rubric=RelationRubricV2(
            sufficient=["앞에 기다리는 사람이 적어서", "사람이 적은 줄이 더 빨라서"],
            partial=["덜 기다려서", "더 빨라서"],
            incorrect=["사람이 많은 줄이 더 빨라서", "아무 줄이나 같아서"],
        ),
        required_for_completion=True,
    )

    def count_choices(step_index: int, fact_id: str, expected: int) -> list[LifeChoiceV2]:
        step = task.steps[ExpressionLevel.L2][step_index]
        return [
            LifeChoiceV2(
                choice_id=choice.id,
                label=choice.label,
                effect=(
                    LifeChoiceEffectV2(
                        verdict="correct",
                        fact_updates=[LifeFactUpdateV2(fact_id=fact_id, value=_count(expected))],
                    )
                    if int(choice.id) == expected
                    else _incorrect_effect("queue_count_error")
                ),
            )
            for choice in step.input.choices
        ]

    side_step = task.steps[ExpressionLevel.L2][2]
    side_choices = [
        LifeChoiceV2(
            choice_id=choice.id,
            label=choice.label,
            effect=(
                LifeChoiceEffectV2(
                    verdict="correct",
                    fact_updates=[LifeFactUpdateV2(fact_id="final_choice", value=_choice(side))],
                )
                if choice.id == side
                else _incorrect_effect("more_people_is_faster")
            ),
        )
        for choice in side_step.input.choices
    ]
    reason_step = task.steps[ExpressionLevel.L2][3]
    reason_choices = [
        LifeChoiceV2(
            choice_id=choice.id,
            label=choice.label,
            effect=(
                LifeChoiceEffectV2(
                    verdict="correct",
                    relation_ids=["choose_shorter_queue"],
                )
                if choice.id == "fewer"
                else _incorrect_effect("more_people_is_faster")
            ),
        )
        for choice in reason_step.input.choices
    ]

    l2_specs = [
        ("queue.l2-left", left_target, 0, count_choices(0, "left_count", left)),
        ("queue.l2-right", right_target, 1, count_choices(1, "right_count", right)),
        ("queue.l2-side", side_target, 2, side_choices),
        ("queue.l2-reason", reason_target, 3, reason_choices),
    ]
    l2_plans = [
        LifeL2ChoicePlanV2(
            plan_id=plan_id,
            target=target,
            copy_slot=f"{plan_id}.copy",
            choices=choices,
        )
        for plan_id, target, _, choices in l2_specs
    ]
    l2_copy_items = [
        (f"{plan_id}.copy", target, task.steps[ExpressionLevel.L2][index].prompt)
        for plan_id, target, index, _ in l2_specs
    ]

    initial_fallback = (
        "나 어느 줄에 서야 빨리 갈지 헷갈려... 양쪽 사람 수랑 줄, 이유를 알려줄 수 있어?"
    )
    return LifeTaskPackV2(
        pack_id="cafe.queue.v2",
        content_version=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_queue",
        task_id=QUEUE_TASK_ID,
        stage_id=task.stage_id,
        phase="single",
        skill_id=task.skill_id,
        title=task.title,
        dictionary_card_id=task.dictionary_card_id,
        source_prompt=initial_fallback,
        base_visual=task.base_visual.model_copy(deep=True),
        reasoning_graph=LifeReasoningGraphV2(
            facts=facts,
            relations=[relation],
            completion=LifeCompletionContractV2(
                required_fact_ids=["left_count", "right_count", "final_choice"],
                required_relation_ids=["choose_shorter_queue"],
            ),
        ),
        initial_question=_question("queue.initial", required_targets, initial_fallback),
        l3_plans=[
            _question(
                "queue.l3-left",
                [left_target],
                "왼쪽 줄에는 몇 명이 있는지 세어서 알려줄 수 있어?",
            ),
            _question(
                "queue.l3-right",
                [right_target],
                "오른쪽 줄에는 몇 명이 있는지 세어서 알려줄 수 있어?",
            ),
            _question("queue.l3-side", [side_target], task.steps[ExpressionLevel.L3][1].prompt),
            _question("queue.l3-reason", [reason_target], task.steps[ExpressionLevel.L3][2].prompt),
        ],
        copy_slots=_copy_slots(
            prefix="queue",
            required_targets=required_targets,
            initial_help_target=left_target,
            initial_help_fallback="왼쪽 줄부터 몇 명인지 세어서 알려줄 수 있어?",
            l2_items=l2_copy_items,
            l0_intro=task.steps[ExpressionLevel.L0][0].prompt,
            l0_action=task.hints[HintLevel.H3].action or task.hints[HintLevel.H3].body,
        ),
        l2_plans=l2_plans,
        l0_joint_plan=LifeL0JointPlanV2(
            action_id="queue.joint",
            intro_copy_slot="queue.l0-intro",
            action_copy_slot="queue.l0-action",
            completion_values=[
                LifeJointFactCompletionV2(target_id="left_count", value=_count(left)),
                LifeJointFactCompletionV2(target_id="right_count", value=_count(right)),
                LifeJointFactCompletionV2(target_id="final_choice", value=_choice(side)),
                LifeJointRelationCompletionV2(target_id="choose_shorter_queue"),
            ],
            button_label="같이 줄 고르기",
            input_config=dict(task.steps[ExpressionLevel.L0][0].input.config),
        ),
        help_plan=_help_plan(
            task,
            required_fact_ids=["left_count", "right_count", "final_choice"],
            required_relation_ids=["choose_shorter_queue"],
        ),
        policies=_note_policy(task, ["choose_shorter_queue"], enabled=True),
    )


def _menu_choices(
    *,
    menu_items: Sequence[CafeMenuItem],
    mormi_menu: CafeMenuItem,
    selected_fact: LifeFactV2,
    budget: int | None,
    relation_id: str | None,
    auto_total: bool,
) -> list[LifeChoiceV2]:
    choices: list[LifeChoiceV2] = []
    for index, item in enumerate(menu_items):
        choice_id = f"menu.{index:02d}"
        total = mormi_menu.price + item.price
        patch: dict[str, Any] = {"child_pick": item.model_dump(mode="json")}
        if auto_total:
            patch.update(
                total=total,
                budget_status=("within" if budget is not None and total <= budget else "over"),
            )
        if item.id == mormi_menu.id:
            effect = _incorrect_effect("same_menu_not_allowed")
            disabled = True
        elif budget is not None and total > budget:
            effect = _incorrect_effect("budget_exceeded", visual_patch=patch)
            disabled = False
        else:
            canonical = _choice(item.id)
            if not selected_fact.accepts_value(canonical):  # pragma: no cover - builder invariant
                raise ValueError("menu choice is outside the reviewed selection fact")
            effect = LifeChoiceEffectV2(
                verdict="correct",
                fact_updates=[LifeFactUpdateV2(fact_id=selected_fact.fact_id, value=canonical)],
                relation_ids=([relation_id] if relation_id else []),
                visual_patch=patch,
            )
            disabled = False
        choices.append(
            LifeChoiceV2(
                choice_id=choice_id,
                label=f"{item.name} {item.price:,}원",
                image_url=item.image_url,
                disabled=disabled,
                effect=effect,
            )
        )
    return choices


def _selection_fact(
    menu_items: Sequence[CafeMenuItem],
    mormi_menu: CafeMenuItem,
    *,
    budget: int | None,
) -> tuple[LifeFactV2, list[CafeMenuItem]]:
    selectable = [
        item
        for item in menu_items
        if item.id != mormi_menu.id and (budget is None or mormi_menu.price + item.price <= budget)
    ]
    if not selectable:
        raise ValueError("cafe selection needs one reviewed selectable menu")
    values: list[CanonicalValueV2] = [_choice(item.id) for item in selectable]
    surfaces = list(
        dict.fromkeys(surface for item in selectable for surface in (item.name, item.id))
    )
    return (
        LifeFactV2(
            fact_id="child_menu",
            role="selection",
            value=values[0],
            accepted_values=values[1:],
            speaker_label="아이가 고른 메뉴",
            initially_visible=False,
            required_for_completion=True,
            accepted_surface_forms=surfaces,
        ),
        selectable,
    )


def _menu_input_config(task: TaskDefinition) -> dict[str, Any]:
    return dict(task.steps[ExpressionLevel.L2][0].input.config)


def _menu_selection_pack(
    *,
    scenario_id: str,
    task: TaskDefinition,
    menu_items: Sequence[CafeMenuItem],
    mormi_menu: CafeMenuItem,
    budget: int | None,
    note_enabled: bool,
) -> LifeTaskPackV2:
    selected_fact, selectable = _selection_fact(menu_items, mormi_menu, budget=budget)
    relation_id = "choose_menu_within_budget" if budget is not None else None
    reviewed_selection_prompt = (
        f"나 {budget:,}원 안에서 우리 둘의 메뉴를 고르려는데 헷갈려... "
        f"나는 {mormi_menu.name}를 골랐어. 너는 뭘 고르면 좋을지 알려줄래?"
        if budget is not None
        else (
            f"나는 {mormi_menu.name}를 골랐어. 두 메뉴 값을 더해 보고 싶은데, "
            "너는 어떤 메뉴를 고를지 알려줄래?"
        )
    )
    facts = [selected_fact]
    relations: list[LifeRelationV2] = []
    targets = [_fact_target("child_menu")]
    required_relation_ids: list[str] = []
    if relation_id is not None:
        assert budget is not None
        facts.extend(
            [
                LifeFactV2(
                    fact_id="budget",
                    role="visible_condition",
                    value=_money(int(budget)),
                    speaker_label="쓸 수 있는 예산",
                    initially_visible=True,
                    accepted_surface_forms=_surface_forms(str(budget), f"{budget:,}원"),
                ),
                LifeFactV2(
                    fact_id="mormi_menu_price",
                    role="visible_value",
                    value=_money(mormi_menu.price),
                    speaker_label="모르미가 고른 메뉴 가격",
                    initially_visible=True,
                    accepted_surface_forms=_surface_forms(
                        str(mormi_menu.price),
                        f"{mormi_menu.price:,}원",
                    ),
                ),
            ]
        )
        relations.append(
            LifeRelationV2(
                relation_id=relation_id,
                role="selection_rule",
                operation="selection",
                input_fact_ids=["budget", "mormi_menu_price"],
                output_fact_id="child_menu",
                evaluation_mode="exact_semantic",
                speaker_label="두 메뉴의 전체 가격이 예산과 같거나 작도록 메뉴를 고른다",
                rubric=RelationRubricV2(
                    sufficient=["두 메뉴의 전체 가격이 예산 안이어서"],
                    partial=["예산 안이라서"],
                    incorrect=["가격을 비교하지 않아도 돼서"],
                ),
                required_for_completion=True,
            )
        )
        targets.append(_relation_target(relation_id))
        required_relation_ids.append(relation_id)

    choices = _menu_choices(
        menu_items=menu_items,
        mormi_menu=mormi_menu,
        selected_fact=selected_fact,
        budget=budget,
        relation_id=relation_id,
        auto_total=budget is not None,
    )
    input_config = _menu_input_config(task)
    l2_plans: list[LifeL2ChoicePlanV2] = []
    l2_copy_items: list[tuple[str, TargetRefV2, str]] = []
    for target in targets:
        suffix = "menu" if target.target_kind == "fact" else "budget-rule"
        plan_id = f"{scenario_id}.l2-{suffix}"
        copy_id = f"{plan_id}.copy"
        prompt = (
            reviewed_selection_prompt
            if target.target_kind == "fact"
            else "어떤 메뉴가 예산 안에 들어오는지 골라서 알려줄 수 있어?"
        )
        l2_plans.append(
            LifeL2ChoicePlanV2(
                plan_id=plan_id,
                target=target,
                copy_slot=copy_id,
                choices=[choice.model_copy(deep=True) for choice in choices],
                input_config=input_config,
            )
        )
        l2_copy_items.append((copy_id, target, prompt))

    l3_plans = [
        _question(
            f"{scenario_id}.l3-menu",
            [targets[0]],
            task.steps[ExpressionLevel.L3][0].prompt,
        )
    ]
    if relation_id is not None:
        l3_plans.append(
            _question(
                f"{scenario_id}.l3-budget-rule",
                [targets[1]],
                "나는 왜 그 메뉴를 예산 안에서 고를 수 있는지 헷갈려... 알려줄 수 있어?",
            )
        )

    suggested = selectable[0]
    joint_values: list[LifeJointFactCompletionV2 | LifeJointRelationCompletionV2] = [
        LifeJointFactCompletionV2(target_id="child_menu", value=_choice(suggested.id))
    ]
    if relation_id is not None:
        joint_values.append(LifeJointRelationCompletionV2(target_id=relation_id))

    joint_step = task.steps[ExpressionLevel.L0][0]
    return LifeTaskPackV2(
        pack_id=f"cafe.{scenario_id.replace('_', '-')}.{task.id.replace('_', '-')}.v2",
        content_version=1,
        scene=SceneType.CAFE,
        scenario_id=scenario_id,
        task_id=task.id,
        stage_id=task.stage_id,
        phase="selection" if scenario_id == "cafe_menu_total" else "single",
        skill_id=task.skill_id,
        title=task.title,
        dictionary_card_id=task.dictionary_card_id,
        source_prompt=reviewed_selection_prompt,
        base_visual=task.base_visual.model_copy(deep=True),
        reasoning_graph=LifeReasoningGraphV2(
            facts=facts,
            relations=relations,
            completion=LifeCompletionContractV2(
                required_fact_ids=["child_menu"],
                required_relation_ids=required_relation_ids,
            ),
        ),
        initial_question=_question(
            f"{scenario_id}.initial",
            targets,
            reviewed_selection_prompt,
        ),
        l3_plans=l3_plans,
        copy_slots=_copy_slots(
            prefix=scenario_id,
            required_targets=targets,
            initial_help_target=targets[0],
            initial_help_fallback=task.steps[ExpressionLevel.L3][0].fallback_text,
            l2_items=l2_copy_items,
            l0_intro=joint_step.prompt,
            l0_action=task.hints[HintLevel.H3].action or task.hints[HintLevel.H3].body,
        ),
        l2_plans=l2_plans,
        l0_joint_plan=LifeL0JointPlanV2(
            action_id=f"{scenario_id}.joint",
            intro_copy_slot=f"{scenario_id}.l0-intro",
            action_copy_slot=f"{scenario_id}.l0-action",
            completion_values=joint_values,
            button_label="같이 메뉴 고르기",
            input_config={
                **dict(joint_step.input.config),
                **input_config,
                "suggested_menu_id": suggested.id,
            },
        ),
        help_plan=_help_plan(
            task,
            required_fact_ids=["child_menu"],
            required_relation_ids=required_relation_ids,
        ),
        policies=_note_policy(task, required_relation_ids, enabled=note_enabled).model_copy(
            update={
                "transition_text": (
                    "네가 고른 메뉴의 값도 더해 보고 싶어..."
                    if scenario_id == "cafe_menu_total"
                    else task.transition_text
                )
            }
        ),
    )


def _calculation_pack(
    *,
    scenario_id: str,
    task: TaskDefinition,
    variant_id: str,
) -> LifeTaskPackV2:
    contract = task.arithmetic_contract
    if contract is None:  # pragma: no cover - legacy content validation guarantees it
        raise ValueError("cafe calculation task needs reviewed arithmetic")
    operation = contract.operation
    relation_id = "add_menu_prices" if operation == "addition" else "subtract_menu_price"
    result_fact_id = "total" if operation == "addition" else "change"
    left_fact_id = "mormi_menu_price" if operation == "addition" else "payment"
    right_fact_id = "child_menu_price" if operation == "addition" else "menu_price"
    symbol = "+" if operation == "addition" else "-"

    result_target = _fact_target(result_fact_id)
    relation_target = _relation_target(relation_id)
    required_targets = [result_target, relation_target]
    facts = [
        LifeFactV2(
            fact_id=left_fact_id,
            role="visible_value",
            value=_money(contract.left),
            speaker_label=contract.left_label,
            initially_visible=True,
            accepted_surface_forms=_surface_forms(
                str(contract.left),
                f"{contract.left:,}원",
            ),
        ),
        LifeFactV2(
            fact_id=right_fact_id,
            role="visible_value",
            value=_money(contract.right),
            speaker_label=contract.right_label,
            initially_visible=True,
            accepted_surface_forms=_surface_forms(
                str(contract.right),
                f"{contract.right:,}원",
            ),
        ),
        LifeFactV2(
            fact_id=result_fact_id,
            role="final_answer",
            value=_money(contract.result),
            speaker_label=contract.result_label,
            initially_visible=False,
            required_for_completion=True,
            accepted_surface_forms=_surface_forms(
                str(contract.result),
                f"{contract.result:,}",
                f"{contract.result}원",
                f"{contract.result:,}원",
            ),
        ),
    ]
    relation = LifeRelationV2(
        relation_id=relation_id,
        role="procedure_step",
        operation=operation,
        input_fact_ids=[left_fact_id, right_fact_id],
        output_fact_id=result_fact_id,
        evaluation_mode="open_semantic_support",
        speaker_label=(
            "두 메뉴 가격을 더해 전체 가격을 구한다"
            if operation == "addition"
            else "낸 돈에서 메뉴값을 빼 거스름돈을 구한다"
        ),
        rubric=RelationRubricV2(
            sufficient=[f"{contract.left:,}{symbol}{contract.right:,}으로 계산한다"],
            partial=["두 금액을 더한다" if operation == "addition" else "메뉴값을 뺀다"],
            incorrect=["두 금액을 뺀다" if operation == "addition" else "메뉴값을 더한다"],
        ),
        required_for_completion=True,
    )

    result_step = task.steps[ExpressionLevel.L2][1]
    result_choices = [
        LifeChoiceV2(
            choice_id=f"result.{index}",
            label=choice.label,
            effect=(
                LifeChoiceEffectV2(
                    verdict="correct",
                    fact_updates=[
                        LifeFactUpdateV2(
                            fact_id=result_fact_id,
                            value=_money(contract.result),
                        )
                    ],
                )
                if int(choice.id) == contract.result
                else _incorrect_effect("calculation_error")
            ),
        )
        for index, choice in enumerate(result_step.input.choices)
    ]
    operation_step = task.steps[ExpressionLevel.L2][0]
    correct_choice = "add" if operation == "addition" else "subtract"
    operation_choices = [
        LifeChoiceV2(
            choice_id=f"operation.{choice.id}",
            label=choice.label,
            effect=(
                LifeChoiceEffectV2(verdict="correct", relation_ids=[relation_id])
                if choice.id == correct_choice
                else _incorrect_effect("operation_confusion")
            ),
        )
        for choice in operation_step.input.choices
    ]
    l2_specs = [
        (
            f"{scenario_id}.{variant_id}.l2-result",
            result_target,
            result_step.prompt,
            result_choices,
        ),
        (
            f"{scenario_id}.{variant_id}.l2-relation",
            relation_target,
            operation_step.prompt,
            operation_choices,
        ),
    ]
    l2_plans = [
        LifeL2ChoicePlanV2(
            plan_id=plan_id,
            target=target,
            copy_slot=f"{plan_id}.copy",
            choices=choices,
        )
        for plan_id, target, _, choices in l2_specs
    ]
    l3_steps = task.steps[ExpressionLevel.L3]
    prefix = f"{scenario_id}.{variant_id}"
    return LifeTaskPackV2(
        pack_id=f"cafe.{scenario_id.replace('_', '-')}.{task.id.replace('_', '-')}.{variant_id}.v2",
        content_version=1,
        scene=SceneType.CAFE,
        scenario_id=scenario_id,
        task_id=task.id,
        stage_id=task.stage_id,
        phase="calculation" if scenario_id == "cafe_menu_total" else "single",
        skill_id=task.skill_id,
        title=task.title,
        dictionary_card_id=task.dictionary_card_id,
        source_prompt=task.steps[ExpressionLevel.L4][0].prompt,
        base_visual=task.base_visual.model_copy(deep=True),
        reasoning_graph=LifeReasoningGraphV2(
            facts=facts,
            relations=[relation],
            completion=LifeCompletionContractV2(
                required_fact_ids=[result_fact_id],
                required_relation_ids=[relation_id],
            ),
        ),
        initial_question=_question(
            f"{prefix}.initial",
            required_targets,
            task.steps[ExpressionLevel.L4][0].prompt,
        ),
        l3_plans=[
            _question(f"{prefix}.l3-result", [result_target], l3_steps[0].prompt),
            _question(f"{prefix}.l3-relation", [relation_target], l3_steps[1].prompt),
        ],
        copy_slots=_copy_slots(
            prefix=prefix,
            required_targets=required_targets,
            initial_help_target=result_target,
            initial_help_fallback=l3_steps[0].fallback_text,
            l2_items=[
                (f"{plan_id}.copy", target, prompt) for plan_id, target, prompt, _ in l2_specs
            ],
            l0_intro=task.steps[ExpressionLevel.L0][0].prompt,
            l0_action=task.hints[HintLevel.H3].action or task.hints[HintLevel.H3].body,
        ),
        l2_plans=l2_plans,
        l0_joint_plan=LifeL0JointPlanV2(
            action_id=f"{prefix}.joint",
            intro_copy_slot=f"{prefix}.l0-intro",
            action_copy_slot=f"{prefix}.l0-action",
            completion_values=[
                LifeJointFactCompletionV2(
                    target_id=result_fact_id,
                    value=_money(contract.result),
                ),
                LifeJointRelationCompletionV2(target_id=relation_id),
            ],
            button_label="같이 계산하기",
            input_config=dict(task.steps[ExpressionLevel.L0][0].input.config),
        ),
        help_plan=_help_plan(
            task,
            required_fact_ids=[result_fact_id],
            required_relation_ids=[relation_id],
        ),
        policies=_note_policy(task, [relation_id], enabled=True),
    )


def _validated_cafe_scenario_data(
    scenario_id: str,
    scenario_data: Mapping[str, Any],
) -> dict[str, Any]:
    if scenario_id not in CAFE_NATIVE_V2_SCENARIO_IDS:
        raise KeyError(f"unsupported native cafe V2 scenario: {scenario_id}")
    scenario = get_scenario(scenario_id)
    if scenario.scene is not SceneType.CAFE:
        raise ValueError("native cafe materializer received a non-cafe scenario")
    if scenario_id == "cafe_queue":
        queue_context = QueueSessionContext.model_validate(
            {
                "left_count": scenario_data.get("left_count"),
                "right_count": scenario_data.get("right_count"),
            }
        )
        return create_scenario_data(scenario_id, queue_context=queue_context)
    cafe_context = CafeSessionContext.model_validate(
        {
            "menu_items": scenario_data.get("menu_items"),
            "mormi_menu_id": scenario_data.get("mormi_menu_id"),
            "budget": scenario_data.get("budget"),
        }
    )
    return create_scenario_data(scenario_id, cafe_context)


def materialize_cafe_scenario_v2(
    scenario_id: str,
    scenario_data: Mapping[str, Any],
) -> LifeScenarioPackV2:
    """Build and validate one immutable cafe V2 scenario from trusted screen facts."""

    data = _validated_cafe_scenario_data(scenario_id, scenario_data)
    if scenario_id == "cafe_queue":
        queue_pack = _queue_pack(data)
        return LifeScenarioPackV2(
            pack_id="cafe.queue.v2",
            content_version=1,
            scene=SceneType.CAFE,
            scenario_id=scenario_id,
            task_stages=[
                LifeTaskStageV2(
                    task_id=QUEUE_TASK_ID,
                    default_variant_id="default",
                    variants={"default": queue_pack},
                )
            ],
            completion_projection=[
                LifeCompletionProjectionV2(
                    output_key="left_count",
                    source_task_id=QUEUE_TASK_ID,
                    source_kind="fact",
                    source_id="left_count",
                ),
                LifeCompletionProjectionV2(
                    output_key="right_count",
                    source_task_id=QUEUE_TASK_ID,
                    source_kind="fact",
                    source_id="right_count",
                ),
                LifeCompletionProjectionV2(
                    output_key="final_choice",
                    source_task_id=QUEUE_TASK_ID,
                    source_kind="fact",
                    source_id="final_choice",
                ),
                LifeCompletionProjectionV2(
                    output_key="reason",
                    source_task_id=QUEUE_TASK_ID,
                    source_kind="relation_constant",
                    source_id="choose_shorter_queue",
                    relation_value="fewer_people",
                ),
            ],
        )

    menu_items = tuple(CafeMenuItem.model_validate(item) for item in data["menu_items"])
    menu_by_id = {item.id: item for item in menu_items}
    mormi_menu = menu_by_id[str(data["mormi_menu_id"])]

    if scenario_id == "cafe_budget_menu":
        budget = int(data["budget"])
        task = get_task(BUDGET_MENU_TASK_ID, data)
        pack = _menu_selection_pack(
            scenario_id=scenario_id,
            task=task,
            menu_items=menu_items,
            mormi_menu=mormi_menu,
            budget=budget,
            note_enabled=True,
        )
        return LifeScenarioPackV2(
            pack_id="cafe.budget-menu.v2",
            content_version=1,
            scene=SceneType.CAFE,
            scenario_id=scenario_id,
            task_stages=[
                LifeTaskStageV2(
                    task_id=BUDGET_MENU_TASK_ID,
                    default_variant_id="default",
                    variants={"default": pack},
                )
            ],
            completion_projection=[
                LifeCompletionProjectionV2(
                    output_key="child_menu_id",
                    source_task_id=BUDGET_MENU_TASK_ID,
                    source_kind="fact",
                    source_id="child_menu",
                )
            ],
        )

    if scenario_id == "cafe_menu_total":
        selection_task = get_task(TOTAL_MENU_PICK_TASK_ID, data)
        selection_pack = _menu_selection_pack(
            scenario_id=scenario_id,
            task=selection_task,
            menu_items=menu_items,
            mormi_menu=mormi_menu,
            budget=None,
            note_enabled=False,
        )
        selectable_ids = [item.id for item in menu_items if item.id != mormi_menu.id]
        variants: dict[str, LifeTaskPackV2] = {}
        selector_values: dict[str, str] = {}
        for index, menu_id in enumerate(selectable_ids):
            variant_id = f"menu-{index:02d}"
            task_data = {**data, "child_menu_id": menu_id}
            calculation_task = get_task(TOTAL_CALC_TASK_ID, task_data)
            variants[variant_id] = _calculation_pack(
                scenario_id=scenario_id,
                task=calculation_task,
                variant_id=variant_id,
            )
            selector_values[menu_id] = variant_id
        default_variant_id = next(iter(variants))
        return LifeScenarioPackV2(
            pack_id="cafe.menu-total.v2",
            content_version=1,
            scene=SceneType.CAFE,
            scenario_id=scenario_id,
            task_stages=[
                LifeTaskStageV2(
                    task_id=TOTAL_MENU_PICK_TASK_ID,
                    default_variant_id="default",
                    variants={"default": selection_pack},
                ),
                LifeTaskStageV2(
                    task_id=TOTAL_CALC_TASK_ID,
                    default_variant_id=default_variant_id,
                    variants=variants,
                    selector=LifePriorFactVariantSelectorV2(
                        source_task_id=TOTAL_MENU_PICK_TASK_ID,
                        fact_id="child_menu",
                        value_to_variant_id=selector_values,
                    ),
                ),
            ],
            completion_projection=[
                LifeCompletionProjectionV2(
                    output_key="child_menu_id",
                    source_task_id=TOTAL_MENU_PICK_TASK_ID,
                    source_kind="fact",
                    source_id="child_menu",
                ),
                LifeCompletionProjectionV2(
                    output_key="result",
                    source_task_id=TOTAL_CALC_TASK_ID,
                    source_kind="fact",
                    source_id="total",
                ),
            ],
        )

    if mormi_menu.price > CAFE_CHANGE_PAYMENT_AMOUNT:
        raise ValueError("cafe_change menu price cannot exceed the 10,000 won payment")
    task = get_task(CHANGE_TASK_ID, data)
    pack = _calculation_pack(
        scenario_id=scenario_id,
        task=task,
        variant_id="default",
    )
    return LifeScenarioPackV2(
        pack_id="cafe.change.v2",
        content_version=1,
        scene=SceneType.CAFE,
        scenario_id=scenario_id,
        task_stages=[
            LifeTaskStageV2(
                task_id=CHANGE_TASK_ID,
                default_variant_id="default",
                variants={"default": pack},
            )
        ],
        completion_projection=[
            LifeCompletionProjectionV2(
                output_key="result",
                source_task_id=CHANGE_TASK_ID,
                source_kind="fact",
                source_id="change",
            )
        ],
    )


def create_cafe_scenario_pack_v2(
    scenario_id: str,
    *,
    cafe_context: CafeSessionContext | None = None,
    queue_context: QueueSessionContext | None = None,
) -> LifeScenarioPackV2:
    """Convenience entry point used by service creation and content tests."""

    scenario_data = create_scenario_data(
        scenario_id,
        cafe_context,
        queue_context=queue_context,
    )
    return materialize_cafe_scenario_v2(scenario_id, scenario_data)

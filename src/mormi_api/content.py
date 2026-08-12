from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from .schemas import (
    ChoiceOption,
    ExpressionLevel,
    HintLevel,
    InputContract,
    InputKind,
    SceneType,
    VisualContract,
)


class SlotDefinition(BaseModel):
    id: str
    description: str
    expected: str | int | float | bool
    aliases: list[str] = Field(default_factory=list)
    fact_sentence: str

    def accepts(self, value: object) -> bool:
        if value == self.expected:
            return True
        normalized = str(value).strip().lower().replace(" ", "")
        candidates = [str(self.expected), *self.aliases]
        return normalized in {item.strip().lower().replace(" ", "") for item in candidates}


class StepDefinition(BaseModel):
    id: str
    prompt: str = Field(max_length=50)
    target_slots: list[str]
    input: InputContract
    choice_effects: dict[str, dict[str, str | int | float | bool]] = Field(default_factory=dict)
    fallback_text: str = Field(max_length=50)


class HintDefinition(BaseModel):
    level: HintLevel
    body: str
    visual_type: str | None = None
    visual_data: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    id: str
    scene: SceneType
    stage_id: str
    skill_id: str
    title: str
    goal: str
    visible_facts: dict[str, Any]
    slots: dict[str, SlotDefinition]
    required_slots: list[str]
    steps: dict[ExpressionLevel, list[StepDefinition]]
    hints: dict[HintLevel, HintDefinition]
    base_visual: VisualContract
    misconception_tags: list[str]
    coauthored_note: str

    def step_for(
        self,
        level: ExpressionLevel,
        verified_slots: Mapping[str, object],
    ) -> StepDefinition:
        level_steps = self.steps[level]
        for step in level_steps:
            if any(slot not in verified_slots for slot in step.target_slots):
                return step
        return level_steps[-1]

    def missing_slots(self, verified_slots: Mapping[str, object]) -> list[str]:
        return [slot for slot in self.required_slots if slot not in verified_slots]

    def complete(self, verified_slots: Mapping[str, object]) -> bool:
        return not self.missing_slots(verified_slots)

    def validated_claims(
        self,
        claims: Iterable[tuple[str, object, bool]],
    ) -> dict[str, str | int | float | bool]:
        verified: dict[str, str | int | float | bool] = {}
        for slot_id, value, classifier_factual in claims:
            slot = self.slots.get(slot_id)
            if slot and classifier_factual and slot.accepts(value):
                verified[slot_id] = slot.expected
        return verified


class ScenarioDefinition(BaseModel):
    id: str
    scene: SceneType
    title: str
    task_ids: list[str]


def option(identifier: str, label: str, image_url: str | None = None) -> ChoiceOption:
    return ChoiceOption(id=identifier, label=label, image_url=image_url)


def text_input(*slots: str, placeholder: str = "모르미에게 알려줘") -> InputContract:
    return InputContract(kind=InputKind.TEXT, placeholder=placeholder, target_slots=list(slots))


def choice_input(slots: list[str], choices: list[ChoiceOption]) -> InputContract:
    return InputContract(kind=InputKind.CHOICES, target_slots=slots, choices=choices)


QUEUE_TASK = TaskDefinition(
    id="cafe_queue_3_vs_5",
    scene=SceneType.CAFE,
    stage_id="queue",
    skill_id="compare_quantity_in_context",
    title="줄 서기",
    goal="두 줄을 세고 사람이 적은 줄을 고른다.",
    visible_facts={"left_count": 3, "right_count": 5, "same_cashier_speed": True},
    slots={
        "left_count": SlotDefinition(
            id="left_count",
            description="왼쪽 줄 사람 수",
            expected=3,
            aliases=["3명", "세명", "세 명"],
            fact_sentence="왼쪽 줄에는 3명이 있어.",
        ),
        "right_count": SlotDefinition(
            id="right_count",
            description="오른쪽 줄 사람 수",
            expected=5,
            aliases=["5명", "다섯명", "다섯 명"],
            fact_sentence="오른쪽 줄에는 5명이 있어.",
        ),
        "smaller_number": SlotDefinition(
            id="smaller_number",
            description="3과 5 중 작은 수",
            expected=3,
            aliases=["3", "삼", "셋"],
            fact_sentence="3은 5보다 작아.",
        ),
        "final_choice": SlotDefinition(
            id="final_choice",
            description="사람이 적어서 덜 기다리는 줄",
            expected="left",
            aliases=["왼쪽", "왼쪽줄", "왼쪽 줄"],
            fact_sentence="왼쪽 줄에 서면 덜 기다려.",
        ),
        "reason": SlotDefinition(
            id="reason",
            description="사람 수가 적은 줄이 덜 기다린다는 이유",
            expected="fewer_people",
            aliases=["사람이적어서", "사람이 적어서", "3명이5명보다적어서"],
            fact_sentence="사람이 적은 줄이 덜 기다려.",
        ),
    },
    required_slots=["left_count", "right_count", "final_choice", "reason"],
    steps={
        ExpressionLevel.L4: [
            StepDefinition(
                id="free_explanation",
                prompt="어느 줄에 서면 덜 기다릴까? 어떻게 알았어?",
                target_slots=["left_count", "right_count", "final_choice", "reason"],
                input=text_input("left_count", "right_count", "final_choice", "reason"),
                fallback_text="어느 줄에 서면 덜 기다릴까? 어떻게 알았어?",
            )
        ],
        ExpressionLevel.L3: [
            StepDefinition(
                id="short_counts",
                prompt="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
                target_slots=["left_count", "right_count"],
                input=text_input("left_count", "right_count", placeholder="사람 수만 짧게 알려줘"),
                fallback_text="내가 한꺼번에 물어봤네. 사람 수만 알려줘.",
            ),
            StepDefinition(
                id="short_choice",
                prompt="어느 줄로 가면 좋을까?",
                target_slots=["final_choice"],
                input=text_input("final_choice", placeholder="왼쪽 또는 오른쪽"),
                fallback_text="어느 줄로 가면 좋을지만 알려줘.",
            ),
            StepDefinition(
                id="short_reason",
                prompt="왜 그 줄이 덜 기다리는 거야?",
                target_slots=["reason"],
                input=text_input("reason", placeholder="이유만 짧게 알려줘"),
                fallback_text="왜 그 줄이 덜 기다리는지만 알려줘.",
            ),
        ],
        ExpressionLevel.L2: [
            StepDefinition(
                id="choose_left_count",
                prompt="왼쪽 줄에는 몇 명이 있어?",
                target_slots=["left_count"],
                input=choice_input(
                    ["left_count"], [option("2", "2명"), option("3", "3명"), option("4", "4명")]
                ),
                choice_effects={
                    "2": {"left_count": 2},
                    "3": {"left_count": 3},
                    "4": {"left_count": 4},
                },
                fallback_text="말로 말하기 어려우면 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_right_count",
                prompt="오른쪽 줄에는 몇 명이 있어?",
                target_slots=["right_count"],
                input=choice_input(
                    ["right_count"], [option("4", "4명"), option("5", "5명"), option("6", "6명")]
                ),
                choice_effects={
                    "4": {"right_count": 4},
                    "5": {"right_count": 5},
                    "6": {"right_count": 6},
                },
                fallback_text="오른쪽 줄 사람 수도 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_side",
                prompt="사람이 더 적은 줄은 어느 쪽이야?",
                target_slots=["final_choice"],
                input=choice_input(
                    ["final_choice"], [option("left", "왼쪽 줄"), option("right", "오른쪽 줄")]
                ),
                choice_effects={
                    "left": {"final_choice": "left"},
                    "right": {"final_choice": "right"},
                },
                fallback_text="사람이 적은 줄을 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_reason",
                prompt="왜 그 줄이 덜 기다릴까?",
                target_slots=["reason"],
                input=choice_input(
                    ["reason"],
                    [option("fewer", "사람이 더 적어서"), option("more", "사람이 더 많아서")],
                ),
                choice_effects={
                    "fewer": {"reason": "fewer_people"},
                    "more": {"reason": "more_people"},
                },
                fallback_text="이유도 두 말 중에서 같이 골라 보자.",
            ),
        ],
        ExpressionLevel.L1: [
            StepDefinition(
                id="guided_count",
                prompt="사람을 한 명씩 눌러 두 줄을 같이 세어 볼까?",
                target_slots=["left_count", "right_count"],
                input=InputContract(
                    kind=InputKind.COUNT,
                    target_slots=["left_count", "right_count"],
                    config={
                        "left_person_ids": ["l1", "l2", "l3"],
                        "right_person_ids": ["r1", "r2", "r3", "r4", "r5"],
                    },
                ),
                fallback_text="내가 어디부터 볼지 몰랐네. 같이 세어 보자.",
            ),
            StepDefinition(
                id="guided_compare",
                prompt="3은 5보다 어떻게 돼?",
                target_slots=["smaller_number"],
                input=choice_input(
                    ["smaller_number"], [option("smaller", "작아"), option("larger", "커")]
                ),
                choice_effects={"smaller": {"smaller_number": 3}, "larger": {"smaller_number": 5}},
                fallback_text="3과 5를 놓고 관계부터 같이 보자.",
            ),
            StepDefinition(
                id="guided_map",
                prompt="3명이 있는 줄은 어느 쪽이야?",
                target_slots=["final_choice"],
                input=choice_input(
                    ["final_choice"], [option("left", "왼쪽"), option("right", "오른쪽")]
                ),
                choice_effects={
                    "left": {"final_choice": "left"},
                    "right": {"final_choice": "right"},
                },
                fallback_text="3명이 있는 줄을 장면에서 같이 찾아보자.",
            ),
            StepDefinition(
                id="guided_reason",
                prompt="사람이 적은 줄은 왜 덜 기다릴까?",
                target_slots=["reason"],
                input=choice_input(
                    ["reason"],
                    [option("fewer", "앞에 사람이 적어서"), option("more", "앞에 사람이 많아서")],
                ),
                choice_effects={
                    "fewer": {"reason": "fewer_people"},
                    "more": {"reason": "more_people"},
                },
                fallback_text="마지막 이유도 같이 이어 보자.",
            ),
        ],
        ExpressionLevel.L0: [
            StepDefinition(
                id="joint_performance",
                prompt="도움 카드 순서대로 나와 같이 해볼까?",
                target_slots=["left_count", "right_count", "final_choice", "reason"],
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=["left_count", "right_count", "final_choice", "reason"],
                    config={"steps": ["count_left", "count_right", "compare", "choose_queue"]},
                ),
                fallback_text="도움 카드 순서대로 나와 같이 해볼까?",
            )
        ],
    },
    hints={
        HintLevel.H1: HintDefinition(
            level=HintLevel.H1,
            body="왼쪽과 오른쪽에서 센 숫자를 나란히 놓아보세요.",
            visual_type=None,
        ),
        HintLevel.H2: HintDefinition(
            level=HintLevel.H2,
            body="숫자 카드 3과 5를 보고 더 작은 수를 찾아보세요.",
            visual_type="number_cards",
            visual_data={"cards": [3, 5], "neutral_style": True},
        ),
        HintLevel.H3: HintDefinition(
            level=HintLevel.H3,
            body="한 명씩 세고, 3과 5를 비교한 뒤 사람이 적은 줄을 찾아보세요.",
            visual_type="joint_steps",
            visual_data={"steps": ["한 명씩 세기", "3과 5 비교하기", "사람이 적은 줄 찾기"]},
        ),
    },
    base_visual=VisualContract(
        type="cafe_queues",
        data={"left_people": 3, "right_people": 5, "show_counts": False},
    ),
    misconception_tags=[
        "double_counting",
        "more_people_is_faster",
        "larger_is_smaller",
        "relation_mapping_error",
    ],
    coauthored_note="사람을 한 명씩 세고, 사람이 적은 줄을 고르면 덜 기다려.",
)


def calculation_task(
    *,
    task_id: str,
    title: str,
    skill_id: str,
    left: int,
    right: int,
    operation: str,
    result: int,
    scene: SceneType = SceneType.CAFE,
    stage_id: str | None = None,
) -> TaskDefinition:
    symbol = "+" if operation == "addition" else "-"
    method = "carry" if operation == "addition" else "regroup"
    method_label = "올림" if operation == "addition" else "받아내림"
    operation_phrase = "더해" if operation == "addition" else "빼서"
    place_action = "더해" if operation == "addition" else "빼"
    return TaskDefinition(
        id=task_id,
        scene=scene,
        stage_id=stage_id or ("home_teach" if scene is SceneType.HOME_TEACH else "calculation"),
        skill_id=skill_id,
        title=title,
        goal=f"{left:,}{symbol}{right:,}을 계산하고 {method_label} 방법을 설명한다.",
        visible_facts={"left": left, "right": right, "operation": operation},
        slots={
            "operation": SlotDefinition(
                id="operation",
                description="필요한 계산 종류",
                expected=operation,
                aliases=["더하기" if operation == "addition" else "빼기"],
                fact_sentence=(f"{left:,}원과 {right:,}원은 {operation_phrase} 계산해."),
            ),
            "result": SlotDefinition(
                id="result",
                description="계산 결과",
                expected=result,
                aliases=[str(result), f"{result:,}", f"{result:,}원"],
                fact_sentence=f"계산 결과는 {result:,}원이야.",
            ),
            "method": SlotDefinition(
                id="method",
                description=f"{method_label}이 필요한 자리 계산 방법",
                expected=method,
                aliases=[method_label, f"{method_label}해"],
                fact_sentence=f"자리값을 맞추고 {method_label}해서 계산해.",
            ),
        },
        required_slots=["operation", "result", "method"],
        steps={
            ExpressionLevel.L4: [
                StepDefinition(
                    id="free_explanation",
                    prompt="모두 얼마일까? 어떻게 계산했는지도 알려줘.",
                    target_slots=["operation", "result", "method"],
                    input=text_input("operation", "result", "method"),
                    fallback_text="결과와 계산 방법을 네 말로 알려줘.",
                )
            ],
            ExpressionLevel.L3: [
                StepDefinition(
                    id="short_result",
                    prompt="계산한 값은 얼마야?",
                    target_slots=["result"],
                    input=text_input("result", placeholder="금액만 알려줘"),
                    fallback_text="내가 많이 물어봤네. 금액부터 알려줘.",
                ),
                StepDefinition(
                    id="short_operation",
                    prompt="두 금액을 더해야 해, 빼야 해?",
                    target_slots=["operation"],
                    input=text_input("operation", placeholder="더하기 또는 빼기"),
                    fallback_text="어떤 계산인지부터 짧게 알려줘.",
                ),
                StepDefinition(
                    id="short_method",
                    prompt=f"자리 계산에서 {method_label}은 어떻게 했어?",
                    target_slots=["method"],
                    input=text_input("method", placeholder="방법만 짧게 알려줘"),
                    fallback_text=f"{method_label} 방법만 짧게 알려줘.",
                ),
            ],
            ExpressionLevel.L2: [
                StepDefinition(
                    id="choose_operation",
                    prompt="어떤 계산을 해야 할까?",
                    target_slots=["operation"],
                    input=choice_input(
                        ["operation"], [option("add", "더하기"), option("subtract", "빼기")]
                    ),
                    choice_effects={
                        "add": {"operation": "addition"},
                        "subtract": {"operation": "subtraction"},
                    },
                    fallback_text="말 대신 필요한 계산을 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_result",
                    prompt="계산한 값은 어느 쪽이야?",
                    target_slots=["result"],
                    input=choice_input(
                        ["result"],
                        [
                            option(str(result - 1000), f"{result - 1000:,}원"),
                            option(str(result), f"{result:,}원"),
                            option(str(result + 1000), f"{result + 1000:,}원"),
                        ],
                    ),
                    choice_effects={
                        str(result - 1000): {"result": result - 1000},
                        str(result): {"result": result},
                        str(result + 1000): {"result": result + 1000},
                    },
                    fallback_text="계산한 금액도 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_method",
                    prompt="자리 계산에서 무엇을 해야 할까?",
                    target_slots=["method"],
                    input=choice_input(
                        ["method"],
                        [option(method, method_label), option("ignore", "그대로 계산하기")],
                    ),
                    choice_effects={method: {"method": method}, "ignore": {"method": "ignore"}},
                    fallback_text="자리 계산 방법도 같이 골라 보자.",
                ),
            ],
            ExpressionLevel.L1: [
                StepDefinition(
                    id="guided_equation",
                    prompt="세로식 빈칸을 한 자리씩 같이 채워볼까?",
                    target_slots=["operation", "result", "method"],
                    input=InputContract(
                        kind=InputKind.EQUATION,
                        target_slots=["operation", "result", "method"],
                        config={
                            "left": left,
                            "right": right,
                            "operation": operation,
                            "places": ["만", "천", "백", "십", "일"],
                        },
                    ),
                    fallback_text="내가 어디부터 볼지 몰랐네. 한 자리씩 보자.",
                )
            ],
            ExpressionLevel.L0: [
                StepDefinition(
                    id="joint_equation",
                    prompt="도움 카드 순서대로 세로식을 같이 채울까?",
                    target_slots=["operation", "result", "method"],
                    input=InputContract(
                        kind=InputKind.JOINT,
                        target_slots=["operation", "result", "method"],
                        config={
                            "left": left,
                            "right": right,
                            "operation": operation,
                            "result": result,
                        },
                    ),
                    fallback_text="도움 카드 순서대로 세로식을 같이 채울까?",
                )
            ],
        },
        hints={
            HintLevel.H1: HintDefinition(
                level=HintLevel.H1,
                body=f"{left:,}원과 {right:,}원의 자리값을 맞춰 보세요.",
            ),
            HintLevel.H2: HintDefinition(
                level=HintLevel.H2,
                body=f"세로식에서 같은 자리끼리 {place_action} 보세요.",
                visual_type="place_value_equation",
                visual_data={"left": left, "right": right, "operation": operation},
            ),
            HintLevel.H3: HintDefinition(
                level=HintLevel.H3,
                body=f"같은 자리부터 계산하고 {method_label} 표시를 확인하세요.",
                visual_type="joint_equation_steps",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result": result,
                },
            ),
        },
        base_visual=VisualContract(
            type="vertical_equation",
            data={"left": left, "right": right, "operation": operation, "result_hidden": True},
        ),
        misconception_tags=[f"{method}_omission", "place_value_error", "operation_confusion"],
        coauthored_note=f"자리값을 맞추고 {method_label}해서 계산하면 {result:,}원이야.",
    )


MENU_TASK = calculation_task(
    task_id="cafe_menu_2800_plus_3200",
    title="메뉴값 계산하기",
    skill_id="addition_with_carry_in_context",
    left=2800,
    right=3200,
    operation="addition",
    result=6000,
    stage_id="menu",
)

CHANGE_TASK = calculation_task(
    task_id="cafe_change_10000_minus_6000",
    title="거스름돈 받기",
    skill_id="subtraction_with_regroup_in_context",
    left=10000,
    right=6000,
    operation="subtraction",
    result=4000,
    stage_id="change",
)

REMIX_QUEUE_TASK = QUEUE_TASK.model_copy(
    update={
        "id": "cafe_remix_queue_3_vs_5",
        "title": "처음부터 해보기: 줄 서기",
        "stage_id": "remix",
    },
    deep=True,
)

REMIX_MENU_TASK = calculation_task(
    task_id="cafe_remix_menu_3300_plus_2800",
    title="처음부터 해보기: 메뉴값",
    skill_id="addition_with_carry_in_context",
    left=3300,
    right=2800,
    operation="addition",
    result=6100,
    stage_id="remix",
)

REMIX_CHANGE_TASK = calculation_task(
    task_id="cafe_remix_change_10000_minus_6100",
    title="처음부터 해보기: 거스름돈",
    skill_id="subtraction_with_regroup_in_context",
    left=10000,
    right=6100,
    operation="subtraction",
    result=3900,
    stage_id="remix",
)

HOME_ADD_TASK = calculation_task(
    task_id="home_teach_3_plus_5",
    title="집에서 모르미 가르치기",
    skill_id="basic_addition",
    left=3,
    right=5,
    operation="addition",
    result=8,
    scene=SceneType.HOME_TEACH,
    stage_id="home_teach",
)


TASKS: dict[str, TaskDefinition] = {
    task.id: task
    for task in [
        QUEUE_TASK,
        MENU_TASK,
        CHANGE_TASK,
        REMIX_QUEUE_TASK,
        REMIX_MENU_TASK,
        REMIX_CHANGE_TASK,
        HOME_ADD_TASK,
    ]
}

SCENARIOS: dict[str, ScenarioDefinition] = {
    "cafe_outing": ScenarioDefinition(
        id="cafe_outing",
        scene=SceneType.CAFE,
        title="모르미와 카페 가기",
        task_ids=[
            QUEUE_TASK.id,
            MENU_TASK.id,
            CHANGE_TASK.id,
            REMIX_QUEUE_TASK.id,
            REMIX_MENU_TASK.id,
            REMIX_CHANGE_TASK.id,
        ],
    ),
    "cafe_queue_demo": ScenarioDefinition(
        id="cafe_queue_demo",
        scene=SceneType.CAFE,
        title="카페 줄 서기",
        task_ids=[QUEUE_TASK.id],
    ),
    "home_addition_teach": ScenarioDefinition(
        id="home_addition_teach",
        scene=SceneType.HOME_TEACH,
        title="덧셈을 모르미에게 알려주기",
        task_ids=[HOME_ADD_TASK.id],
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    try:
        return TASKS[task_id]
    except KeyError as error:
        raise KeyError(f"Unknown task: {task_id}") from error


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        raise KeyError(f"Unknown scenario: {scenario_id}") from error


def validate_content() -> None:
    for scenario in SCENARIOS.values():
        for task_id in scenario.task_ids:
            get_task(task_id)
    for task in TASKS.values():
        if set(task.required_slots) - set(task.slots):
            raise ValueError(f"{task.id}: required slot is undefined")
        for level in ExpressionLevel:
            if level not in task.steps or not task.steps[level]:
                raise ValueError(f"{task.id}: missing steps for {level}")
        for step in (item for steps in task.steps.values() for item in steps):
            if set(step.target_slots) - set(task.slots):
                raise ValueError(f"{task.id}/{step.id}: target slot is undefined")
        for hint_level in (HintLevel.H1, HintLevel.H2, HintLevel.H3):
            if hint_level not in task.hints:
                raise ValueError(f"{task.id}: missing {hint_level} hint")


validate_content()

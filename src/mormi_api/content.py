from __future__ import annotations

import random
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
    accepted_values: list[str | int | float | bool] = Field(default_factory=list)
    preserve_value: bool = False
    fact_sentence: str

    def accepts(self, value: object) -> bool:
        if value == self.expected:
            return True
        normalized = str(value).strip().lower().replace(" ", "")
        candidates = [
            str(self.expected),
            *self.aliases,
            *(str(item) for item in self.accepted_values),
        ]
        return normalized in {item.strip().lower().replace(" ", "") for item in candidates}

    def canonical(self, value: object) -> str | int | float | bool:
        if self.preserve_value and isinstance(value, (str, int, float, bool)):
            return value
        return self.expected


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
    behavior: str = "teaching"
    note_policy: str = "stage"
    transition_text: str | None = None

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
                verified[slot_id] = slot.canonical(value)
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


class MenuItem(BaseModel):
    id: str
    name: str
    price: int
    emoji: str


CAFE_MENU: tuple[MenuItem, ...] = (
    MenuItem(id="lemon", name="레몬 에이드", price=2800, emoji="🍋"),
    MenuItem(id="choco", name="핫초코", price=3200, emoji="☕"),
    MenuItem(id="sandwich", name="샌드위치", price=4300, emoji="🥪"),
    MenuItem(id="yogurt", name="딸기 요거트", price=5200, emoji="🍓"),
)
MENU_BY_ID = {item.id: item for item in CAFE_MENU}
CAFE_BUDGETS = (8000, 9000, 10000)


def menu_items_json() -> list[dict[str, str | int]]:
    return [item.model_dump() for item in CAFE_MENU]


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


KOREAN_COUNTS = {
    1: ["1명", "한명", "한 명"],
    2: ["2명", "두명", "두 명"],
    3: ["3명", "세명", "세 명"],
    4: ["4명", "네명", "네 명"],
    5: ["5명", "다섯명", "다섯 명"],
}


def _nearby_count_options(value: int) -> list[ChoiceOption]:
    values = sorted({max(1, value - 1), value, min(5, value + 1)})
    return [option(str(item), f"{item}명") for item in values]


def queue_task(
    *,
    task_id: str,
    stage_id: str,
    left: int,
    right: int,
    note_policy: str = "stage",
) -> TaskDefinition:
    task = QUEUE_TASK.model_copy(deep=True)
    smaller = min(left, right)
    side = "left" if left < right else "right"
    side_label = "왼쪽" if side == "left" else "오른쪽"
    task.id = task_id
    task.stage_id = stage_id
    task.title = "줄 서기" if stage_id == "queue" else "통합 실습: 줄 서기"
    task.visible_facts = {
        "left_count": left,
        "right_count": right,
        "same_cashier_speed": True,
    }
    task.slots["left_count"] = SlotDefinition(
        id="left_count",
        description="왼쪽 줄 사람 수",
        expected=left,
        aliases=KOREAN_COUNTS[left],
        fact_sentence=f"왼쪽 줄에는 {left}명이 있어.",
    )
    task.slots["right_count"] = SlotDefinition(
        id="right_count",
        description="오른쪽 줄 사람 수",
        expected=right,
        aliases=KOREAN_COUNTS[right],
        fact_sentence=f"오른쪽 줄에는 {right}명이 있어.",
    )
    task.slots["smaller_number"] = SlotDefinition(
        id="smaller_number",
        description=f"{left}과 {right} 중 작은 수",
        expected=smaller,
        aliases=[str(smaller)],
        fact_sentence=f"{smaller}이 더 작은 수야.",
    )
    task.slots["final_choice"] = SlotDefinition(
        id="final_choice",
        description="사람이 적어서 덜 기다리는 줄",
        expected=side,
        aliases=[side_label, f"{side_label}줄", f"{side_label} 줄"],
        fact_sentence=f"{side_label} 줄에 서면 덜 기다려.",
    )
    task.steps[ExpressionLevel.L2][0].input = choice_input(
        ["left_count"], _nearby_count_options(left)
    )
    task.steps[ExpressionLevel.L2][0].choice_effects = {
        choice.id: {"left_count": int(choice.id)}
        for choice in task.steps[ExpressionLevel.L2][0].input.choices
    }
    task.steps[ExpressionLevel.L2][1].input = choice_input(
        ["right_count"], _nearby_count_options(right)
    )
    task.steps[ExpressionLevel.L2][1].choice_effects = {
        choice.id: {"right_count": int(choice.id)}
        for choice in task.steps[ExpressionLevel.L2][1].input.choices
    }
    task.steps[ExpressionLevel.L1][0].input.config = {
        "left_person_ids": [f"l{index}" for index in range(1, left + 1)],
        "right_person_ids": [f"r{index}" for index in range(1, right + 1)],
    }
    task.steps[ExpressionLevel.L1][1].prompt = f"{left}과 {right} 중 더 작은 수는 뭐야?"
    task.steps[ExpressionLevel.L1][1].input = choice_input(
        ["smaller_number"],
        [option(str(left), str(left)), option(str(right), str(right))],
    )
    task.steps[ExpressionLevel.L1][1].choice_effects = {
        str(left): {"smaller_number": left},
        str(right): {"smaller_number": right},
    }
    task.steps[ExpressionLevel.L1][2].prompt = f"{smaller}명이 있는 줄은 어느 쪽이야?"
    task.hints[HintLevel.H2] = HintDefinition(
        level=HintLevel.H2,
        body=f"숫자 카드 {left}과 {right}를 보고 더 작은 수를 찾아보세요.",
        visual_type="number_cards",
        visual_data={"cards": [left, right], "neutral_style": True},
    )
    task.hints[HintLevel.H3] = HintDefinition(
        level=HintLevel.H3,
        body=f"한 명씩 세고, {left}과 {right}를 비교한 뒤 사람이 적은 줄을 찾아보세요.",
        visual_type="joint_steps",
        visual_data={"steps": ["한 명씩 세기", "두 수 비교하기", "사람이 적은 줄 찾기"]},
    )
    task.base_visual = VisualContract(
        type="cafe_queues",
        data={"left_people": left, "right_people": right, "show_counts": False},
    )
    task.note_policy = note_policy
    task.transition_text = "사람이 적은 줄을 찾았구나."
    return task


def menu_selection_task(
    *,
    task_id: str,
    stage_id: str,
    mormi_menu: MenuItem,
    budget: int | None,
    auto_total: bool,
    behavior: str,
    note_policy: str,
) -> TaskDefinition:
    valid_ids: list[str | int | float | bool] = [
        item.id
        for item in CAFE_MENU
        if budget is None or not auto_total or mormi_menu.price + item.price <= budget
    ]
    choices = [option(item.id, f"{item.emoji} {item.name} {item.price:,}원") for item in CAFE_MENU]
    input_contract = InputContract(
        kind=InputKind.CHOICES,
        target_slots=["child_menu"],
        choices=choices,
        config={
            "component": "cafe_menu_picker",
            "budget": budget,
            "mormi_menu_id": mormi_menu.id,
            "auto_total": auto_total,
            "allow_same_menu": True,
        },
    )
    prompt = f"나는 {mormi_menu.name}을 골랐어. 너는 뭘 고를래?"
    step = StepDefinition(
        id="pick_menu",
        prompt=prompt,
        target_slots=["child_menu"],
        input=input_contract,
        choice_effects={item.id: {"child_menu": item.id} for item in CAFE_MENU},
        fallback_text="네가 먹고 싶은 메뉴 하나를 골라 줄래?",
    )
    return TaskDefinition(
        id=task_id,
        scene=SceneType.CAFE,
        stage_id=stage_id,
        skill_id="choose_within_budget" if budget is not None else "choose_menu_for_calculation",
        title="예산 안에서 메뉴 고르기" if budget is not None else "계산할 메뉴 고르기",
        goal=(
            "두 메뉴가 예산 안에 들어오도록 고른다."
            if budget is not None
            else "계산할 메뉴를 하나 고른다."
        ),
        visible_facts={
            "budget": budget,
            "mormi_menu": mormi_menu.model_dump(),
            "menu_items": menu_items_json(),
            "auto_total": auto_total,
        },
        slots={
            "child_menu": SlotDefinition(
                id="child_menu",
                description="아이가 고른 메뉴",
                expected=valid_ids[0],
                accepted_values=valid_ids,
                preserve_value=True,
                fact_sentence="아이도 메뉴를 하나 골랐어.",
            )
        },
        required_slots=["child_menu"],
        steps={level: [step.model_copy(deep=True)] for level in ExpressionLevel},
        hints={
            HintLevel.H1: HintDefinition(
                level=HintLevel.H1,
                body="장바구니 합계와 예산을 나란히 확인해 보세요.",
            ),
            HintLevel.H2: HintDefinition(
                level=HintLevel.H2,
                body="모르미 메뉴 가격에 고른 메뉴 가격을 더해 보세요.",
                visual_type="budget_meter",
                visual_data={"budget": budget, "mormi_price": mormi_menu.price},
            ),
            HintLevel.H3: HintDefinition(
                level=HintLevel.H3,
                body="합계가 예산보다 크면 더 저렴한 메뉴로 바꿔 보세요.",
                visual_type="budget_menu_help",
                visual_data={"budget": budget, "mormi_menu": mormi_menu.model_dump()},
            ),
        },
        base_visual=VisualContract(
            type="cafe_menu",
            data={
                "menu_items": menu_items_json(),
                "budget": budget,
                "mormi_pick": mormi_menu.model_dump(),
                "child_pick": None,
                "auto_total": auto_total,
                "budget_status": "pending",
            },
        ),
        misconception_tags=["budget_exceeded", "price_comparison_error"],
        coauthored_note="메뉴 가격을 더한 금액이 예산보다 크면 다른 메뉴를 골라야 해.",
        behavior=behavior,
        note_policy=note_policy,
        transition_text="네 메뉴도 골랐구나.",
    )


def simple_calculation_task(
    *,
    task_id: str,
    stage_id: str,
    title: str,
    left: int,
    right: int,
    operation: str,
    left_label: str,
    right_label: str,
    behavior: str,
    note_policy: str,
    coauthored_note: str,
    context: Mapping[str, Any],
) -> TaskDefinition:
    result = left + right if operation == "addition" else left - right
    symbol = "+" if operation == "addition" else "-"
    operation_label = "더하기" if operation == "addition" else "빼기"
    distractors = sorted({max(0, result - 1000), result, result + 1000})
    result_choices = [option(str(value), f"{value:,}원") for value in distractors]
    operation_choices = [option("add", "더하기"), option("subtract", "빼기")]
    operation_effects: dict[str, dict[str, str | int | float | bool]] = {
        "add": {"operation": "addition"},
        "subtract": {"operation": "subtraction"},
    }
    result_effects: dict[str, dict[str, str | int | float | bool]] = {
        str(value): {"result": value} for value in distractors
    }

    l4 = StepDefinition(
        id="free_calculation",
        prompt=(
            "두 메뉴는 모두 얼마야? 어떻게 계산했어?"
            if operation == "addition"
            else "거스름돈은 얼마야? 어떻게 계산했어?"
        ),
        target_slots=["operation", "result"],
        input=text_input("operation", "result", placeholder="값과 계산 방법을 알려줘"),
        fallback_text="계산한 값과 어떤 계산인지 알려줘.",
    )
    l3 = [
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
            fallback_text="어떤 계산인지도 짧게 알려줘.",
        ),
    ]
    l2 = [
        StepDefinition(
            id="choose_operation",
            prompt="어떤 계산을 해야 할까?",
            target_slots=["operation"],
            input=choice_input(["operation"], operation_choices),
            choice_effects=operation_effects,
            fallback_text="필요한 계산을 같이 골라 보자.",
        ),
        StepDefinition(
            id="choose_result",
            prompt="계산한 금액은 어느 쪽이야?",
            target_slots=["result"],
            input=choice_input(["result"], result_choices),
            choice_effects=result_effects,
            fallback_text="계산한 금액도 같이 골라 보자.",
        ),
    ]
    fill_result = InputContract(
        kind=InputKind.FILL,
        target_slots=["result"],
        choices=result_choices,
        config={"expression": f"{left:,} {symbol} {right:,} = □"},
    )
    l1 = [
        StepDefinition(
            id="guided_operation",
            prompt=f"{left_label}에서 {right_label}을 어떻게 계산할까?",
            target_slots=["operation"],
            input=choice_input(["operation"], operation_choices),
            choice_effects=operation_effects,
            fallback_text="두 금액 사이 계산 기호부터 골라 보자.",
        ),
        StepDefinition(
            id="guided_result",
            prompt=f"{left:,} {symbol} {right:,}의 빈칸은 얼마야?",
            target_slots=["result"],
            input=fill_result,
            choice_effects=result_effects,
            fallback_text="가로식의 빈칸을 같이 채워 보자.",
        ),
    ]
    joint = StepDefinition(
        id="joint_calculation",
        prompt="도움 카드 순서대로 계산을 같이 해볼까?",
        target_slots=["operation", "result"],
        input=InputContract(
            kind=InputKind.JOINT,
            target_slots=["operation", "result"],
            config={"left": left, "right": right, "operation": operation, "result": result},
        ),
        fallback_text="도움 카드 순서대로 계산을 같이 해볼까?",
    )
    return TaskDefinition(
        id=task_id,
        scene=SceneType.CAFE,
        stage_id=stage_id,
        skill_id="add_menu_prices" if operation == "addition" else "calculate_change",
        title=title,
        goal=f"{left:,}{symbol}{right:,}을 생활 맥락에서 계산한다.",
        visible_facts={"left": left, "right": right, "operation": operation, **dict(context)},
        slots={
            "operation": SlotDefinition(
                id="operation",
                description="필요한 계산 종류",
                expected=operation,
                aliases=[operation_label],
                fact_sentence=f"{operation_label}로 계산해.",
            ),
            "result": SlotDefinition(
                id="result",
                description="계산 결과",
                expected=result,
                aliases=[str(result), f"{result:,}", f"{result:,}원"],
                fact_sentence=f"계산 결과는 {result:,}원이야.",
            ),
        },
        required_slots=["operation", "result"],
        steps={
            ExpressionLevel.L4: [l4],
            ExpressionLevel.L3: l3,
            ExpressionLevel.L2: l2,
            ExpressionLevel.L1: l1,
            ExpressionLevel.L0: [joint],
        },
        hints={
            HintLevel.H1: HintDefinition(
                level=HintLevel.H1,
                body=(
                    "두 메뉴 가격을 더해 보세요."
                    if operation == "addition"
                    else "10,000원에서 메뉴값을 빼 보세요."
                ),
            ),
            HintLevel.H2: HintDefinition(
                level=HintLevel.H2,
                body=f"{left:,} {symbol} {right:,} 가로식을 확인해 보세요.",
                visual_type="money_calculation",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result_hidden": True,
                },
            ),
            HintLevel.H3: HintDefinition(
                level=HintLevel.H3,
                body=f"도움 카드에서 {left:,} {symbol} {right:,}의 계산 순서를 따라가 보세요.",
                visual_type="joint_money_calculation",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result": result,
                },
            ),
        },
        base_visual=VisualContract(
            type="cafe_calculation",
            data={
                "left": left,
                "right": right,
                "operation": operation,
                "result_hidden": True,
                **dict(context),
            },
        ),
        misconception_tags=["operation_confusion", "calculation_error"],
        coauthored_note=coauthored_note,
        behavior=behavior,
        note_policy=note_policy,
        transition_text=f"계산하면 {result:,}원이구나.",
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

QUEUE_TASK_ID = "cafe_queue"
BUDGET_MENU_TASK_ID = "cafe_budget_menu_pick"
TOTAL_MENU_PICK_TASK_ID = "cafe_total_menu_pick"
TOTAL_CALC_TASK_ID = "cafe_total_calculation"
CHANGE_TASK_ID = "cafe_change"
INTEGRATED_QUEUE_TASK_ID = "cafe_integrated_queue"
INTEGRATED_MENU_TASK_ID = "cafe_integrated_menu_pick"
INTEGRATED_TOTAL_TASK_ID = "cafe_integrated_total"
INTEGRATED_CHANGE_TASK_ID = "cafe_integrated_change"

SCENARIOS: dict[str, ScenarioDefinition] = {
    "cafe_queue": ScenarioDefinition(
        id="cafe_queue",
        scene=SceneType.CAFE,
        title="1단계 줄 서기",
        task_ids=[QUEUE_TASK_ID],
    ),
    "cafe_queue_demo": ScenarioDefinition(
        id="cafe_queue_demo",
        scene=SceneType.CAFE,
        title="1단계 줄 서기(호환 ID)",
        task_ids=[QUEUE_TASK_ID],
    ),
    "cafe_budget_menu": ScenarioDefinition(
        id="cafe_budget_menu",
        scene=SceneType.CAFE,
        title="2단계 예산 안에서 메뉴 고르기",
        task_ids=[BUDGET_MENU_TASK_ID],
    ),
    "cafe_menu_total": ScenarioDefinition(
        id="cafe_menu_total",
        scene=SceneType.CAFE,
        title="3단계 메뉴값 계산하기",
        task_ids=[TOTAL_MENU_PICK_TASK_ID, TOTAL_CALC_TASK_ID],
    ),
    "cafe_change": ScenarioDefinition(
        id="cafe_change",
        scene=SceneType.CAFE,
        title="4단계 거스름돈 받기",
        task_ids=[CHANGE_TASK_ID],
    ),
    "cafe_integrated": ScenarioDefinition(
        id="cafe_integrated",
        scene=SceneType.CAFE,
        title="5단계 카페 통합 실습",
        task_ids=[
            INTEGRATED_QUEUE_TASK_ID,
            INTEGRATED_MENU_TASK_ID,
            INTEGRATED_TOTAL_TASK_ID,
            INTEGRATED_CHANGE_TASK_ID,
        ],
    ),
    "cafe_outing": ScenarioDefinition(
        id="cafe_outing",
        scene=SceneType.CAFE,
        title="5단계 카페 통합 실습(호환 ID)",
        task_ids=[
            INTEGRATED_QUEUE_TASK_ID,
            INTEGRATED_MENU_TASK_ID,
            INTEGRATED_TOTAL_TASK_ID,
            INTEGRATED_CHANGE_TASK_ID,
        ],
    ),
    "home_addition_teach": ScenarioDefinition(
        id="home_addition_teach",
        scene=SceneType.HOME_TEACH,
        title="덧셈을 모르미에게 알려주기",
        task_ids=[HOME_ADD_TASK.id],
    ),
}


def create_scenario_data(scenario_id: str, rng: Any | None = None) -> dict[str, Any]:
    chooser = rng or random.SystemRandom()
    data: dict[str, Any] = {"payment": 10000}
    if scenario_id in {"cafe_queue", "cafe_queue_demo", "cafe_integrated", "cafe_outing"}:
        left = chooser.choice(range(1, 6))
        right = chooser.choice([value for value in range(1, 6) if value != left])
        data.update(left_count=left, right_count=right)
    if scenario_id in {
        "cafe_budget_menu",
        "cafe_menu_total",
        "cafe_change",
        "cafe_integrated",
        "cafe_outing",
    }:
        data["mormi_menu_id"] = chooser.choice(CAFE_MENU).id
    if scenario_id in {"cafe_budget_menu", "cafe_integrated", "cafe_outing"}:
        data["budget"] = chooser.choice(CAFE_BUDGETS)
    return data


def _menu_from_data(data: Mapping[str, Any], key: str, default: str = "choco") -> MenuItem:
    return MENU_BY_ID.get(str(data.get(key, default)), MENU_BY_ID[default])


def get_task(task_id: str, scenario_data: Mapping[str, Any] | None = None) -> TaskDefinition:
    data = scenario_data or {}
    left_count = int(data.get("left_count", 3))
    right_count = int(data.get("right_count", 5))
    mormi_menu = _menu_from_data(data, "mormi_menu_id")
    child_menu = _menu_from_data(data, "child_menu_id", "lemon")
    budget = int(data.get("budget", 9000))
    if task_id == QUEUE_TASK_ID:
        return queue_task(task_id=task_id, stage_id="queue", left=left_count, right=right_count)
    if task_id == INTEGRATED_QUEUE_TASK_ID:
        return queue_task(
            task_id=task_id,
            stage_id="integrated",
            left=left_count,
            right=right_count,
            note_policy="none",
        )
    if task_id == BUDGET_MENU_TASK_ID:
        return menu_selection_task(
            task_id=task_id,
            stage_id="budget_menu",
            mormi_menu=mormi_menu,
            budget=budget,
            auto_total=True,
            behavior="budget_menu_selection",
            note_policy="stage",
        )
    if task_id == TOTAL_MENU_PICK_TASK_ID:
        return menu_selection_task(
            task_id=task_id,
            stage_id="menu_total",
            mormi_menu=mormi_menu,
            budget=None,
            auto_total=False,
            behavior="menu_selection",
            note_policy="none",
        )
    if task_id == INTEGRATED_MENU_TASK_ID:
        return menu_selection_task(
            task_id=task_id,
            stage_id="integrated",
            mormi_menu=mormi_menu,
            budget=budget,
            auto_total=False,
            behavior="integrated_menu_selection",
            note_policy="none",
        )
    if task_id in {TOTAL_CALC_TASK_ID, INTEGRATED_TOTAL_TASK_ID}:
        integrated = task_id == INTEGRATED_TOTAL_TASK_ID
        return simple_calculation_task(
            task_id=task_id,
            stage_id="integrated" if integrated else "menu_total",
            title="통합 실습: 메뉴값 계산" if integrated else "메뉴값 계산하기",
            left=mormi_menu.price,
            right=child_menu.price,
            operation="addition",
            left_label=mormi_menu.name,
            right_label=child_menu.name,
            behavior="integrated_total" if integrated else "menu_total",
            note_policy="none" if integrated else "stage",
            coauthored_note="두 메뉴의 전체 가격은 각 메뉴 가격을 더해서 구해.",
            context={
                "budget": budget if integrated else None,
                "mormi_menu": mormi_menu.model_dump(),
                "child_menu": child_menu.model_dump(),
            },
        )
    if task_id in {CHANGE_TASK_ID, INTEGRATED_CHANGE_TASK_ID}:
        integrated = task_id == INTEGRATED_CHANGE_TASK_ID
        menu_price = mormi_menu.price + child_menu.price if integrated else mormi_menu.price
        return simple_calculation_task(
            task_id=task_id,
            stage_id="integrated" if integrated else "change",
            title="통합 실습: 거스름돈" if integrated else "거스름돈 받기",
            left=10000,
            right=menu_price,
            operation="subtraction",
            left_label="낸 돈",
            right_label="전체 메뉴값" if integrated else mormi_menu.name,
            behavior="integrated_change" if integrated else "change",
            note_policy="stage",
            coauthored_note=(
                "사람이 적은 줄을 고르고 예산 안에서 메뉴를 고른 뒤, 전체 가격과 거스름돈을 계산해."
                if integrated
                else "거스름돈은 10,000원에서 메뉴 가격을 빼서 구해."
            ),
            context={
                "payment": 10000,
                "menu_total": menu_price,
                "mormi_menu": mormi_menu.model_dump(),
                "child_menu": child_menu.model_dump() if integrated else None,
            },
        )
    if task_id == HOME_ADD_TASK.id:
        return HOME_ADD_TASK
    raise KeyError(f"Unknown task: {task_id}")


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        raise KeyError(f"Unknown scenario: {scenario_id}") from error


def validate_content() -> None:
    for scenario in SCENARIOS.values():
        for task_id in scenario.task_ids:
            get_task(task_id)
    task_ids = {task_id for scenario in SCENARIOS.values() for task_id in scenario.task_ids}
    for task in [get_task(task_id) for task_id in task_ids]:
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

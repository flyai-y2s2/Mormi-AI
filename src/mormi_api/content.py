from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .schemas import (
    CafeMenuItem,
    CafeSessionContext,
    ChoiceOption,
    ExpressionLevel,
    HintLevel,
    InputContract,
    InputKind,
    QueueSessionContext,
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
    optional_slots: list[str] = Field(default_factory=list)
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


class HomeTeachingSpec(BaseModel):
    """Reviewed teaching content for one frontend curriculum session."""

    id: str
    subject: str
    unit: str
    title: str
    misconception_prompt: str = Field(max_length=50)
    learned_line: str = Field(max_length=120)
    fill_before: str
    fill_after: str
    fill_correct: str
    fill_options: list[str] = Field(min_length=2, max_length=6)
    short_prompt: str = Field(max_length=50)
    short_correct: str
    short_options: list[str] = Field(min_length=2, max_length=6)
    hint: str
    help_lines: list[str] = Field(min_length=1, max_length=4)
    misconception: str
    sample_problem: dict[str, Any]

    @model_validator(mode="after")
    def validate_turn_coherence_contract(self) -> HomeTeachingSpec:
        prompt = self.sample_problem.get("prompt")
        correct = self.sample_problem.get("correct")
        answers = self.sample_problem.get("answers")
        visual = self.sample_problem.get("visual")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("sample_problem.prompt is required")
        if not isinstance(answers, list) or len(answers) < 2:
            raise ValueError("sample_problem.answers needs at least two choices")
        if correct not in answers:
            raise ValueError("sample_problem.correct must be one of answers")
        if not isinstance(visual, Mapping) or not visual.get("type"):
            raise ValueError("sample_problem.visual.type is required")
        if self.short_prompt.strip() == "어떤 방법이 맞을까?":
            raise ValueError("short_prompt must name the current mathematical action")
        if self.misconception_prompt.strip() == self.short_prompt.strip():
            raise ValueError("L4 and L2 prompts must not collapse into the same request")
        return self


def _load_home_teaching_catalog() -> dict[str, HomeTeachingSpec]:
    path = Path(__file__).with_name("home_teaching_catalog.json")
    entries = [HomeTeachingSpec.model_validate(item) for item in json.loads(path.read_text())]
    catalog = {entry.id: entry for entry in entries}
    if len(catalog) != len(entries):
        raise ValueError("home teaching catalog contains duplicate ids")
    return catalog


HOME_TEACHING_CATALOG = _load_home_teaching_catalog()


def option(identifier: str, label: str, image_url: str | None = None) -> ChoiceOption:
    return ChoiceOption(id=identifier, label=label, image_url=image_url)


def text_input(*slots: str, placeholder: str = "모르미에게 알려줘") -> InputContract:
    return InputContract(kind=InputKind.TEXT, placeholder=placeholder, target_slots=list(slots))


def choice_input(slots: list[str], choices: list[ChoiceOption]) -> InputContract:
    return InputContract(kind=InputKind.CHOICES, target_slots=slots, choices=choices)


def menu_items_json(items: Sequence[CafeMenuItem]) -> list[dict[str, str | int | None]]:
    return [item.model_dump() for item in items]


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
                id="free_counts",
                prompt="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
                target_slots=["left_count", "right_count"],
                input=text_input("left_count", "right_count"),
                fallback_text="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
            ),
            StepDefinition(
                id="free_comparison",
                prompt="어느 줄에 서면 덜 기다릴까? 어떻게 알았어?",
                target_slots=["final_choice", "reason"],
                input=text_input("final_choice", "reason"),
                fallback_text="어느 줄에 서면 덜 기다릴까? 어떻게 알았어?",
            ),
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
                    config={
                        "steps": ["count_left", "count_right", "compare", "choose_queue"],
                        "completion_values": {
                            "left_count": 3,
                            "right_count": 5,
                            "final_choice": "left",
                            "reason": "fewer_people",
                        },
                    },
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
                            "completion_values": {
                                "operation": operation,
                                "result": result,
                                "method": method,
                            },
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
    task.title = "줄 서기"
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
    task.steps[ExpressionLevel.L0][0].input.config["completion_values"] = {
        "left_count": left,
        "right_count": right,
        "final_choice": side,
        "reason": "fewer_people",
    }
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
    menu_items: Sequence[CafeMenuItem],
    mormi_menu: CafeMenuItem,
    budget: int | None,
    auto_total: bool,
    behavior: str,
    note_policy: str,
) -> TaskDefinition:
    valid_ids: list[str | int | float | bool] = [
        item.id
        for item in menu_items
        if item.id != mormi_menu.id
        and (budget is None or not auto_total or mormi_menu.price + item.price <= budget)
    ]
    choices = [
        ChoiceOption(
            id=item.id,
            label=f"{item.name} {item.price:,}원",
            image_url=item.image_url,
            disabled=item.id == mormi_menu.id,
        )
        for item in menu_items
    ]
    input_contract = InputContract(
        kind=InputKind.CHOICES,
        target_slots=["child_menu"],
        choices=choices,
        config={
            "component": "cafe_menu_picker",
            "budget": budget,
            "mormi_menu_id": mormi_menu.id,
            "auto_total": auto_total,
            "allow_same_menu": False,
        },
    )
    if budget is not None:
        prompt = f"{budget:,}원 안에서 고르자. 나는 {mormi_menu.name}, 너는 뭘 고를래?"
        fallback = f"예산은 {budget:,}원이야. 네 메뉴 하나를 골라줄래?"
    else:
        prompt = f"나는 {mormi_menu.name}을 골랐어. 너는 뭘 고를래?"
        fallback = "계산할 메뉴를 하나 골라줄래?"
    if len(prompt) > 50:
        prompt = fallback
    step = StepDefinition(
        id="pick_menu",
        prompt=prompt,
        target_slots=["child_menu"],
        input=input_contract,
        choice_effects={item.id: {"child_menu": item.id} for item in menu_items},
        fallback_text=fallback,
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
            "menu_items": menu_items_json(menu_items),
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
                body=(
                    "장바구니 합계와 예산을 나란히 확인해 보세요."
                    if budget is not None
                    else "메뉴판에서 먹고 싶은 메뉴 하나를 골라 보세요."
                ),
            ),
            HintLevel.H2: HintDefinition(
                level=HintLevel.H2,
                body=(
                    "모르미 메뉴 가격에 고른 메뉴 가격을 더해 보세요."
                    if budget is not None
                    else "모르미가 고른 메뉴와 다른 메뉴를 하나 골라 보세요."
                ),
                visual_type="budget_meter" if budget is not None else "cafe_menu_focus",
                visual_data=(
                    {"budget": budget, "mormi_price": mormi_menu.price}
                    if budget is not None
                    else {"mormi_menu": mormi_menu.model_dump()}
                ),
            ),
            HintLevel.H3: HintDefinition(
                level=HintLevel.H3,
                body=(
                    "합계가 예산보다 크면 더 저렴한 메뉴로 바꿔 보세요."
                    if budget is not None
                    else "메뉴판에서 하나를 골라 장바구니에 담아 보세요."
                ),
                visual_type="budget_menu_help" if budget is not None else "cafe_menu_focus",
                visual_data=(
                    {"budget": budget, "mormi_menu": mormi_menu.model_dump()}
                    if budget is not None
                    else {"mormi_menu": mormi_menu.model_dump()}
                ),
            ),
        },
        base_visual=VisualContract(
            type="cafe_menu",
            data={
                "menu_items": menu_items_json(menu_items),
                "budget": budget,
                "mormi_pick": mormi_menu.model_dump(),
                "child_pick": None,
                "auto_total": auto_total,
                "budget_status": "pending",
            },
        ),
        misconception_tags=(
            ["budget_exceeded", "price_comparison_error"] if budget is not None else []
        ),
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
            prompt=(
                "두 메뉴 가격은 어떤 계산으로 합칠까?"
                if operation == "addition"
                else "거스름돈을 구하려면 어떤 계산을 할까?"
            ),
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
            config={
                "left": left,
                "right": right,
                "operation": operation,
                "result": result,
                "completion_values": {"operation": operation, "result": result},
            },
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


def home_teaching_task(
    spec: HomeTeachingSpec,
    *,
    skill_id: str,
) -> TaskDefinition:
    """Build a deterministic teaching task from reviewed curriculum content.

    The LLM may understand a child's paraphrase and phrase Mormi's reaction,
    but it never invents the target rule, choices, help cards, or note text.
    """

    expected_rule = spec.learned_line

    raw_sample = dict(spec.sample_problem)
    expected_answer = raw_sample.get("correct")
    sample_answers = raw_sample.get("answers")
    if not isinstance(expected_answer, (str, int, float, bool)):
        raise ValueError(f"{spec.id}: sample_problem.correct is required")
    if not isinstance(sample_answers, list) or len(sample_answers) < 2:
        raise ValueError(f"{spec.id}: sample_problem.answers needs at least two choices")

    answer_choices = [
        option(f"answer_{index}", str(label)) for index, label in enumerate(sample_answers)
    ]
    answer_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"answer_{index}": {"answer": expected_answer} if label == expected_answer else {}
        for index, label in enumerate(sample_answers)
    }

    short_choices = [
        option(f"short_{index}", label) for index, label in enumerate(spec.short_options)
    ]
    short_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"short_{index}": {"rule": expected_rule} if label == spec.short_correct else {}
        for index, label in enumerate(spec.short_options)
    }
    fill_choices = [option(f"fill_{index}", label) for index, label in enumerate(spec.fill_options)]
    fill_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"fill_{index}": {"rule": expected_rule} if label == spec.fill_correct else {}
        for index, label in enumerate(spec.fill_options)
    }
    sentence_frame = " ".join(
        part for part in (spec.fill_before.strip(), "□", spec.fill_after.strip()) if part
    )
    sample = raw_sample
    sample.pop("correct", None)

    return TaskDefinition(
        id=HOME_TEACH_TASK_ID,
        scene=SceneType.HOME_TEACH,
        stage_id="home_teach",
        skill_id=skill_id,
        title=spec.title,
        goal=f"반복한 {spec.title}의 핵심 방법을 모르미에게 가르친다.",
        visible_facts={
            "curriculum_session_id": spec.id,
            "target_rule": expected_rule,
            "sample_answer": expected_answer,
            "sample_problem": sample,
        },
        slots={
            # The concrete answer is useful partial evidence: when a child only
            # corrects Mormi's answer, preserve it and ask only for the method.
            # It is intentionally not required for completion because the
            # teaching goal and star-note evidence are the general rule.
            "answer": SlotDefinition(
                id="answer",
                description=f"화면의 {spec.title} 예시 문제 답",
                expected=expected_answer,
                aliases=[str(expected_answer).replace(",", "")],
                fact_sentence=f"이 문제의 답은 {expected_answer}이야.",
            ),
            "rule": SlotDefinition(
                id="rule",
                description=f"{spec.title}에서 사용하는 핵심 방법 전체",
                expected=expected_rule,
                aliases=[expected_rule.rstrip(".!?")],
                fact_sentence=expected_rule,
            )
        },
        required_slots=["rule"],
        steps={
            ExpressionLevel.L4: [
                StepDefinition(
                    id="free_explanation",
                    prompt=spec.misconception_prompt,
                    target_slots=["rule"],
                    optional_slots=["answer"],
                    input=text_input(
                        "answer",
                        "rule",
                        placeholder="답과 방법을 네 말로 알려줘",
                    ),
                    fallback_text=spec.misconception_prompt,
                )
            ],
            ExpressionLevel.L3: [
                StepDefinition(
                    id="short_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=text_input("answer", placeholder="답만 짧게 알려줘"),
                    fallback_text="내가 한꺼번에 물어봤네. 답부터 알려줘.",
                ),
                StepDefinition(
                    id="short_explanation",
                    prompt=spec.short_prompt,
                    target_slots=["rule"],
                    input=text_input("rule", placeholder="방법만 짧게 알려줘"),
                    fallback_text="내가 길게 물어봤네. 방법만 짧게 알려줘.",
                )
            ],
            ExpressionLevel.L2: [
                StepDefinition(
                    id="choose_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=choice_input(["answer"], answer_choices),
                    choice_effects=answer_effects,
                    fallback_text="말로 어렵다면 답부터 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_method",
                    prompt=spec.short_prompt,
                    target_slots=["rule"],
                    input=choice_input(["rule"], short_choices),
                    choice_effects=short_effects,
                    fallback_text="말로 어렵다면 필요한 방법을 같이 골라 보자.",
                )
            ],
            ExpressionLevel.L1: [
                StepDefinition(
                    id="guided_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=choice_input(["answer"], answer_choices),
                    choice_effects=answer_effects,
                    fallback_text="화면을 보며 답부터 하나 골라 보자.",
                ),
                StepDefinition(
                    id="complete_rule",
                    prompt=sentence_frame,
                    target_slots=["rule"],
                    input=InputContract(
                        kind=InputKind.FILL,
                        target_slots=["rule"],
                        choices=fill_choices,
                        config={"sentence": sentence_frame},
                    ),
                    choice_effects=fill_effects,
                    fallback_text="도움 카드 문장의 빈칸을 같이 채워 보자.",
                )
            ],
            ExpressionLevel.L0: [
                StepDefinition(
                    id="joint_reading",
                    prompt="도움 카드 문장을 나와 같이 읽어볼까?",
                    target_slots=["rule"],
                    input=InputContract(
                        kind=InputKind.JOINT,
                        target_slots=["rule"],
                        config={
                            "text": expected_rule,
                            "completion_values": {"rule": expected_rule},
                        },
                    ),
                    fallback_text="도움 카드 문장을 나와 같이 읽어볼까?",
                )
            ],
        },
        hints={
            HintLevel.H1: HintDefinition(level=HintLevel.H1, body=spec.hint),
            HintLevel.H2: HintDefinition(
                level=HintLevel.H2,
                body=spec.help_lines[-1],
                visual_type="home_practice_problem",
                visual_data=sample,
            ),
            HintLevel.H3: HintDefinition(
                level=HintLevel.H3,
                body=expected_rule,
                visual_type="joint_reading_card",
                visual_data={"text": expected_rule},
            ),
        },
        base_visual=VisualContract(
            type="home_teaching",
            data={
                "curriculum_session_id": spec.id,
                "subject": spec.subject,
                "unit": spec.unit,
                "title": spec.title,
                "problem": sample,
            },
        ),
        misconception_tags=[spec.misconception],
        coauthored_note=expected_rule,
    )

QUEUE_TASK_ID = "cafe_queue"
HOME_TEACH_TASK_ID = "home_teaching"
BUDGET_MENU_TASK_ID = "cafe_budget_menu_pick"
TOTAL_MENU_PICK_TASK_ID = "cafe_total_menu_pick"
TOTAL_CALC_TASK_ID = "cafe_total_calculation"
CHANGE_TASK_ID = "cafe_change"
CAFE_CHANGE_PAYMENT_AMOUNT = 10_000
MENU_SCENARIO_IDS = {"cafe_budget_menu", "cafe_menu_total", "cafe_change"}

SCENARIOS: dict[str, ScenarioDefinition] = {
    "home_teach": ScenarioDefinition(
        id="home_teach",
        scene=SceneType.HOME_TEACH,
        title="반복한 내용을 모르미에게 가르치기",
        task_ids=[HOME_TEACH_TASK_ID],
    ),
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
}


def create_scenario_data(
    scenario_id: str,
    cafe_context: CafeSessionContext | None = None,
    rng: Any | None = None,
    *,
    queue_context: QueueSessionContext | None = None,
    curriculum_session_id: str | None = None,
    skill_id: str | None = None,
    practice_result_id: str | None = None,
) -> dict[str, Any]:
    chooser = rng or random.SystemRandom()
    data: dict[str, Any] = {}
    if scenario_id in {"cafe_queue", "cafe_queue_demo"}:
        if queue_context is not None:
            # The screen already drew the lines. Mormi must count what the child sees.
            data.update(
                left_count=queue_context.left_count,
                right_count=queue_context.right_count,
            )
        else:
            left = chooser.choice(range(1, 6))
            right = chooser.choice([value for value in range(1, 6) if value != left])
            data.update(left_count=left, right_count=right)
    if scenario_id in MENU_SCENARIO_IDS:
        if cafe_context is None:
            raise ValueError("cafe_context is required for menu scenarios")
        data.update(cafe_context.model_dump(mode="json", exclude_none=True))
    if scenario_id == "home_teach":
        if not curriculum_session_id:
            raise ValueError("curriculum_session_id is required for home_teach")
        try:
            spec = HOME_TEACHING_CATALOG[curriculum_session_id]
        except KeyError as error:
            raise ValueError(
                f"unsupported home curriculum_session_id: {curriculum_session_id}"
            ) from error
        data.update(
            curriculum_session_id=spec.id,
            skill_id=skill_id or spec.id,
            home_teaching_spec=spec.model_dump(mode="json"),
        )
        if practice_result_id:
            data["practice_result_id"] = practice_result_id
    return data


def _menu_items_from_data(data: Mapping[str, Any]) -> tuple[CafeMenuItem, ...]:
    raw_items = data.get("menu_items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("scenario_data.menu_items is required")
    return tuple(CafeMenuItem.model_validate(item) for item in raw_items)


def _menu_from_data(
    data: Mapping[str, Any],
    key: str,
    menu_items: Sequence[CafeMenuItem],
    *,
    default_to_first: bool = False,
) -> CafeMenuItem:
    menu_by_id = {item.id: item for item in menu_items}
    selected_id = data.get(key)
    if selected_id is None and default_to_first:
        return menu_items[0]
    try:
        return menu_by_id[str(selected_id)]
    except KeyError as error:
        raise ValueError(f"scenario_data.{key} must reference menu_items") from error


def get_task(task_id: str, scenario_data: Mapping[str, Any] | None = None) -> TaskDefinition:
    data = scenario_data or {}
    left_count = int(data.get("left_count", 3))
    right_count = int(data.get("right_count", 5))
    if task_id == QUEUE_TASK_ID:
        return queue_task(task_id=task_id, stage_id="queue", left=left_count, right=right_count)
    if task_id == HOME_TEACH_TASK_ID:
        raw_spec = data.get("home_teaching_spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("scenario_data.home_teaching_spec is required")
        skill_id = data.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("scenario_data.skill_id is required")
        return home_teaching_task(HomeTeachingSpec.model_validate(raw_spec), skill_id=skill_id)
    menu_items = _menu_items_from_data(data)
    mormi_menu = _menu_from_data(data, "mormi_menu_id", menu_items)
    if task_id == BUDGET_MENU_TASK_ID:
        budget = int(data["budget"])
        return menu_selection_task(
            task_id=task_id,
            stage_id="budget_menu",
            menu_items=menu_items,
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
            menu_items=menu_items,
            mormi_menu=mormi_menu,
            budget=None,
            auto_total=False,
            behavior="menu_selection",
            note_policy="none",
        )
    if task_id == TOTAL_CALC_TASK_ID:
        # Conversation creation inspects every future task before the child has
        # chosen a menu. The placeholder is used only to derive static task
        # metadata; once this task is reached, the engine has stored the real
        # child_menu_id in scenario_data.
        child_menu = _menu_from_data(
            data,
            "child_menu_id",
            menu_items,
            default_to_first=True,
        )
        return simple_calculation_task(
            task_id=task_id,
            stage_id="menu_total",
            title="메뉴값 계산하기",
            left=mormi_menu.price,
            right=child_menu.price,
            operation="addition",
            left_label=mormi_menu.name,
            right_label=child_menu.name,
            behavior="menu_total",
            note_policy="stage",
            coauthored_note="두 메뉴의 전체 가격은 각 메뉴 가격을 더해서 구해.",
            context={
                "budget": None,
                "mormi_menu": mormi_menu.model_dump(),
                "child_menu": child_menu.model_dump(),
            },
        )
    if task_id == CHANGE_TASK_ID:
        return simple_calculation_task(
            task_id=task_id,
            stage_id="change",
            title="거스름돈 받기",
            left=CAFE_CHANGE_PAYMENT_AMOUNT,
            right=mormi_menu.price,
            operation="subtraction",
            left_label="낸 돈",
            right_label="메뉴 값",
            behavior="change",
            note_policy="stage",
            coauthored_note="거스름돈은 낸 돈에서 메뉴 값을 빼서 구해.",
            context={
                "payment": CAFE_CHANGE_PAYMENT_AMOUNT,
                "menu_total": mormi_menu.price,
                "mormi_menu": mormi_menu.model_dump(),
            },
        )
    raise KeyError(f"Unknown task: {task_id}")


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        raise KeyError(f"Unknown scenario: {scenario_id}") from error


def validate_content() -> None:
    validation_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="sample-a", name="메뉴 A", price=2000),
            CafeMenuItem(id="sample-b", name="메뉴 B", price=3000),
        ],
        mormi_menu_id="sample-a",
        budget=10000,
    )
    for scenario in SCENARIOS.values():
        if scenario.id == "home_teach":
            for spec in HOME_TEACHING_CATALOG.values():
                scenario_data = create_scenario_data(
                    scenario.id,
                    curriculum_session_id=spec.id,
                    skill_id=spec.id,
                )
                get_task(HOME_TEACH_TASK_ID, scenario_data)
            continue
        scenario_data = create_scenario_data(
            scenario.id,
            validation_context if scenario.id in MENU_SCENARIO_IDS else None,
        )
        for task_id in scenario.task_ids:
            task_data = scenario_data
            if task_id == TOTAL_CALC_TASK_ID:
                task_data = {**scenario_data, "child_menu_id": "sample-b"}
            get_task(task_id, task_data)
    task_ids = {
        task_id
        for scenario in SCENARIOS.values()
        if scenario.id != "home_teach"
        for task_id in scenario.task_ids
    }
    sample_data = {
        **create_scenario_data("cafe_menu_total", validation_context),
        "child_menu_id": "sample-b",
    }
    tasks_to_validate = [
        get_task(
            task_id,
            sample_data if task_id != QUEUE_TASK_ID else {},
        )
        for task_id in task_ids
    ]
    tasks_to_validate.extend(
        home_teaching_task(spec, skill_id=spec.id)
        for spec in HOME_TEACHING_CATALOG.values()
    )
    for task in tasks_to_validate:
        if set(task.required_slots) - set(task.slots):
            raise ValueError(f"{task.id}: required slot is undefined")
        for level in ExpressionLevel:
            if level not in task.steps or not task.steps[level]:
                raise ValueError(f"{task.id}: missing steps for {level}")
        for step in (item for steps in task.steps.values() for item in steps):
            if set(step.target_slots) - set(task.slots):
                raise ValueError(f"{task.id}/{step.id}: target slot is undefined")
            if set(step.optional_slots) - set(task.slots):
                raise ValueError(f"{task.id}/{step.id}: optional slot is undefined")
            if step.input.kind in {InputKind.CHOICES, InputKind.FILL}:
                visible_choice_ids = {choice.id for choice in step.input.choices}
                if set(step.choice_effects) != visible_choice_ids:
                    raise ValueError(
                        f"{task.id}/{step.id}: every visible choice needs one reviewed effect"
                    )
                correct_choice_ids = {
                    choice_id
                    for choice_id, effects in step.choice_effects.items()
                    if effects
                    and set(step.target_slots).issubset(effects)
                    and all(
                        task.slots[slot_id].accepts(value)
                        for slot_id, value in effects.items()
                    )
                }
                if not correct_choice_ids:
                    raise ValueError(
                        f"{task.id}/{step.id}: structured input needs a reviewed correct choice"
                    )
            if step.input.kind is InputKind.JOINT:
                completion_values = step.input.config.get("completion_values")
                if not isinstance(completion_values, Mapping):
                    raise ValueError(
                        f"{task.id}/{step.id}: joint input requires completion_values"
                    )
                if set(step.target_slots) - set(completion_values):
                    raise ValueError(
                        f"{task.id}/{step.id}: completion_values must cover target slots"
                    )
        for hint_level in (HintLevel.H1, HintLevel.H2, HintLevel.H3):
            if hint_level not in task.hints:
                raise ValueError(f"{task.id}: missing {hint_level} hint")


validate_content()

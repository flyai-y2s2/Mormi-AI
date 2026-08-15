from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .content import (
    HOME_TEACHING_CATALOG,
    MENU_SCENARIO_IDS,
    SCENARIOS,
    TOTAL_CALC_TASK_ID,
    TaskDefinition,
    create_scenario_data,
    get_task,
    home_teaching_task,
)
from .schemas import (
    CafeMenuItem,
    CafeSessionContext,
    ExpressionLevel,
    HintLevel,
    QueueSessionContext,
)


@dataclass(frozen=True)
class RegisteredHelpTask:
    review_id: str
    task: TaskDefinition


class HelpReviewItem(BaseModel):
    review_id: str
    title: str
    learning_goal: str
    first_question: str
    help_skills: list[str]
    method_policy: str
    accepted_methods: list[str]
    visible_facts: dict[str, Any]
    visual: dict[str, Any]
    required_slots: list[str]
    help_plan: dict[str, dict[str, Any]]
    l0_joint_completion: dict[str, Any]


class HelpAuditDecision(BaseModel):
    review_id: str
    approved: bool
    h1_directs_attention: bool
    h2_adds_concrete_support: bool
    h3_enables_success: bool
    same_problem_and_goal: bool
    grounded_in_visible_facts: bool
    method_policy_respected: bool
    child_can_tell_what_to_do: bool
    issues: list[str] = Field(default_factory=list, max_length=8)


class HelpAuditBatch(BaseModel):
    results: list[HelpAuditDecision]


OFFLINE_HELP_AUDIT_SYSTEM = """
너는 경계선지능 아동용 생활수학 콘텐츠를 출시 전에 검수하는 편집자다.
런타임 대화를 생성하지 않고, 제공된 도움카드 H1~H3의 품질만 엄격히 판정한다.

공통 계약:
- H1은 정답을 말하지 않고 현재 화면에서 볼 정보에 주의를 돌린다.
- H2는 같은 문제에서 수식, 강조, 조작, 순서, 선택 중 하나의 구체적 지원을 더한다.
- H3는 아이와 전체 과정을 함께 수행하고 현재 문제의 답으로 끝낼 수 있게 한다.
- H1→H2→H3로 갈수록 실제 지원 정보가 증가해야 한다.
- 화면이나 검수된 사실에 없는 수·대상·관계를 만들어내면 안 된다.
- open_methods에서는 도움카드의 한 방법을 유일한 정답처럼 강요하면 안 된다.
- target_method에서도 문구 암기가 아니라 명시된 수학 행동을 지원해야 한다.
- 짧다는 이유만으로 통과시키지 말고, 아동이 카드만 보고 다음 행동을 알 수 있는지 본다.

각 review_id를 정확히 한 번 판정하고, 문제를 발견하면 issues에 짧고 구체적으로 쓴다.
""".strip()


def registered_help_tasks() -> list[RegisteredHelpTask]:
    """Return every currently registered task with stable representative facts.

    This derives café tasks from ``SCENARIOS`` so a future registered stage is
    automatically pulled into the review pipeline instead of relying on a
    second hand-maintained audit list.
    """

    tasks = [
        RegisteredHelpTask(
            review_id=f"home:{spec.id}",
            task=home_teaching_task(spec, skill_id=spec.id),
        )
        for spec in HOME_TEACHING_CATALOG.values()
    ]
    cafe_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="sample-a", name="우유", price=2_000),
            CafeMenuItem(id="sample-b", name="주스", price=3_000),
            CafeMenuItem(id="sample-c", name="케이크", price=4_500),
        ],
        mormi_menu_id="sample-a",
        budget=9_000,
    )
    queue_context = QueueSessionContext(left_count=2, right_count=5)
    seen: set[tuple[str, str]] = set()
    for scenario in SCENARIOS.values():
        if scenario.id in {"home_teach", "cafe_queue_demo"}:
            continue
        scenario_data = create_scenario_data(
            scenario.id,
            cafe_context if scenario.id in MENU_SCENARIO_IDS else None,
            rng=random.Random(0),
            queue_context=queue_context if scenario.id == "cafe_queue" else None,
        )
        for task_id in scenario.task_ids:
            task_data = scenario_data
            if task_id == TOTAL_CALC_TASK_ID:
                task_data = {**scenario_data, "child_menu_id": "sample-b"}
            task = get_task(task_id, task_data)
            identity = (task.id, task.stage_id)
            if identity in seen:
                continue
            seen.add(identity)
            tasks.append(
                RegisteredHelpTask(
                    review_id=f"{scenario.id}:{task.stage_id}:{task.id}",
                    task=task,
                )
            )
    return tasks


def build_help_review_items() -> list[HelpReviewItem]:
    items: list[HelpReviewItem] = []
    for registered in registered_help_tasks():
        task = registered.task
        completion_values: dict[str, Any] = {}
        for step in task.steps[ExpressionLevel.L0]:
            values = step.input.config.get("completion_values")
            if isinstance(values, dict):
                completion_values.update(values)
        items.append(
            HelpReviewItem(
                review_id=registered.review_id,
                title=task.title,
                learning_goal=task.goal,
                first_question=task.steps[ExpressionLevel.L4][0].prompt,
                help_skills=list(task.help_skills),
                method_policy=task.help_method_policy,
                accepted_methods=task.accepted_methods,
                visible_facts=task.visible_facts,
                visual=task.base_visual.model_dump(mode="json"),
                required_slots=task.required_slots,
                help_plan={
                    level.value: task.hints[level].model_dump(mode="json")
                    for level in (HintLevel.H1, HintLevel.H2, HintLevel.H3)
                },
                l0_joint_completion=completion_values,
            )
        )
    return items


def render_human_review(items: Iterable[HelpReviewItem]) -> str:
    """Render compact blocks so question, visual and all help levels are co-visible."""

    lines = [
        "# 모르미 도움카드 출시 전 사람 검수표",
        "",
        "> 순진한 독자 기준으로 질문·화면·H1·H2·H3를 한 블록에서 읽습니다.",
        (
            "> 체크: 다음 행동이 분명한가 / 지원이 증가하는가 / "
            "사실에 근거하는가 / 방법을 강요하지 않는가."
        ),
        "",
    ]
    for item in items:
        lines.extend(
            [
                f"## {item.review_id} — {item.title}",
                "",
                f"- 목표: {item.learning_goal}",
                f"- 첫 질문: {item.first_question}",
                f"- 기능: {', '.join(item.help_skills)}",
                f"- 풀이 정책: {item.method_policy}",
                f"- 허용 방법: {' / '.join(item.accepted_methods)}",
                f"- 화면: `{json.dumps(item.visual, ensure_ascii=False, separators=(',', ':'))}`",
                "",
                "| 단계 | 역할 | 공개 수준 | 지원 수단 | 아이에게 보이는 문장 | 행동 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for level in ("H1", "H2", "H3"):
            step = item.help_plan[level]
            lines.append(
                f"| {level} | {step['support_type']} | {step['answer_policy']} | "
                f"{step['support_mode']} | {step['body']} | {step.get('action') or '-'} |"
            )
        lines.extend(
            [
                "",
                (
                    "- L0 공동 수행 완료값: `"
                    f"{json.dumps(item.l0_joint_completion, ensure_ascii=False)}`"
                ),
                "- [ ] 아동이 각 카드에서 무엇을 해야 하는지 안다.",
                "- [ ] H2는 H1보다, H3는 H2보다 실제 지원이 강하다.",
                "- [ ] 화면에 없는 사실이나 검수되지 않은 답을 만들지 않는다.",
                "- [ ] 허용된 다른 풀이를 막거나 한 방법만 강요하지 않는다.",
                "- [ ] H3와 L0 공동 수행으로 반드시 성공할 수 있다.",
                "",
            ]
        )
    return "\n".join(lines)


def help_audit_prompt(items: Iterable[HelpReviewItem]) -> str:
    payload = [item.model_dump(mode="json") for item in items]
    return json.dumps({"help_cards_to_review": payload}, ensure_ascii=False)

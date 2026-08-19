from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from .dictionary_catalog import DICTIONARY_CATALOG
from .help_audit import registered_help_tasks
from .schemas import ExpressionLevel, HintLevel


class DictionaryRelatedTask(BaseModel):
    review_id: str
    first_question: str
    help_plan: dict[str, str]


class DictionaryReviewItem(BaseModel):
    review_id: str
    card_id: str
    curriculum_session_id: str
    title: str
    learning_goal: str
    method_policy: str
    concept_lines: list[str]
    example_lines: list[str]
    facts: dict[str, Any]
    equation: dict[str, Any] | None
    visual: dict[str, Any]
    source_refs: list[str]
    related_tasks: list[DictionaryRelatedTask]


class DictionaryAuditDecision(BaseModel):
    review_id: str
    approved: bool
    standalone_and_clear: bool
    mathematically_correct: bool
    concept_and_example_separated: bool
    visual_matches_text: bool
    method_policy_respected: bool
    distinct_from_help_plan: bool
    child_appropriate_language: bool
    issues: list[str] = Field(default_factory=list, max_length=8)


class DictionaryAuditBatch(BaseModel):
    results: list[DictionaryAuditDecision]


OFFLINE_DICTIONARY_AUDIT_SYSTEM = """
너는 경계선지능 아동용 생활수학 '궁금해사전'을 출시 전에 검수하는 편집자다.
런타임 대사를 만들지 말고, 버전 관리되는 참고 카드의 품질만 엄격히 판정한다.

검수 계약:
- concept는 현재 문제의 답이 아니라 다른 예에도 적용되는 수학 관계를 설명한다.
- example은 구체적인 수를 사용하되 concept에 없는 새 전략을 정답처럼 끼워 넣지 않는다.
- 문장만 따로 읽어도 대명사나 이전 화면 없이 뜻을 이해할 수 있어야 한다.
- 수, 식, 단위와 수학 관계가 정확해야 한다.
- visual의 사실과 글의 사실이 같고, 그림에서 할 수 없는 조작을 요구하지 않는다.
- open_methods는 하나의 풀이만 유일한 정답처럼 강요하지 않는다.
- help_plan과 역할이 다르며, 도움카드 문구를 사전 설명으로 복사하지 않는다.
- 초등 아동이 읽을 수 있는 짧고 구체적인 한국어인지 확인한다.

각 review_id를 정확히 한 번 판정하고, 문제를 발견하면 issues에 짧고 구체적으로 쓴다.
""".strip()


def build_dictionary_review_items() -> list[DictionaryReviewItem]:
    related: dict[str, list[DictionaryRelatedTask]] = defaultdict(list)
    for registered in registered_help_tasks():
        task = registered.task
        related[task.dictionary_card_id].append(
            DictionaryRelatedTask(
                review_id=registered.review_id,
                first_question=task.steps[ExpressionLevel.L4][0].prompt,
                help_plan={
                    level.value: task.hints[level].body
                    for level in (HintLevel.H1, HintLevel.H2, HintLevel.H3)
                },
            )
        )

    return [
        DictionaryReviewItem(
            review_id=f"dictionary:{card.curriculum_session_id}",
            card_id=card.card_id,
            curriculum_session_id=card.curriculum_session_id,
            title=card.title,
            learning_goal=card.learning_goal,
            method_policy=card.method_policy,
            concept_lines=card.concept.lines,
            example_lines=card.example.lines,
            facts=card.example.facts,
            equation=(
                card.example.equation.model_dump(mode="json")
                if card.example.equation
                else None
            ),
            visual=card.visual.model_dump(mode="json"),
            source_refs=card.source_refs,
            related_tasks=related[card.card_id],
        )
        for card in DICTIONARY_CATALOG.cards
    ]


def render_dictionary_human_review(items: Iterable[DictionaryReviewItem]) -> str:
    lines = [
        "# 모르미 궁금해사전 출시 전 사람 검수표",
        "",
        "> 개념·예시·전용 그림·관련 질문·도움카드를 한 블록에서 함께 읽습니다.",
        "",
    ]
    for item in items:
        lines.extend(
            [
                f"## {item.review_id} — {item.title}",
                "",
                f"- 목표: {item.learning_goal}",
                f"- 개념: {' / '.join(item.concept_lines)}",
                f"- 예시: {' / '.join(item.example_lines)}",
                f"- 풀이 정책: {item.method_policy}",
                f"- 사실: `{json.dumps(item.facts, ensure_ascii=False)}`",
                f"- 식: `{json.dumps(item.equation, ensure_ascii=False)}`",
                f"- 전용 그림: `{json.dumps(item.visual, ensure_ascii=False)}`",
                f"- 출처: {' / '.join(item.source_refs)}",
                "",
                "| 관련 과제 | 첫 질문 | H1 | H2 | H3 |",
                "|---|---|---|---|---|",
            ]
        )
        for task in item.related_tasks:
            lines.append(
                f"| {task.review_id} | {task.first_question} | "
                f"{task.help_plan['H1']} | {task.help_plan['H2']} | {task.help_plan['H3']} |"
            )
        lines.extend(
            [
                "",
                "- [ ] 문장만 따로 읽어도 개념을 이해할 수 있다.",
                "- [ ] 수학적으로 정확하고 개념과 예시가 구분된다.",
                "- [ ] 그림의 수·대상·관계가 글과 일치한다.",
                "- [ ] 특정 풀이를 유일한 정답처럼 강요하지 않는다.",
                "- [ ] 도움카드와 역할·문구가 분리되어 있다.",
                "",
            ]
        )
    return "\n".join(lines)


def dictionary_audit_prompt(items: Iterable[DictionaryReviewItem]) -> str:
    payload = [item.model_dump(mode="json") for item in items]
    return json.dumps({"dictionary_cards_to_review": payload}, ensure_ascii=False)

from __future__ import annotations

import json
import re
from itertools import permutations
from pathlib import Path

import pytest
from pydantic import ValidationError

from mormi_api.content import (
    HOME_TEACHING_CATALOG,
    HintDefinition,
    home_teaching_task,
    queue_task,
    reviewed_help_card,
)
from mormi_api.copy_quality import validate_child_facing_math_copy
from mormi_api.schemas import ExpressionLevel, HintLevel, SafetyCategory
from mormi_api.security import deterministic_safety, safety_redirect

VAGUE_COPY = (
    "어떤 방법이 맞을까",
    "지금 상황",
    "지금 장면",
    "퍼진 넓이",
    "느낌으로",
    "눈대중",
    "한눈에 대충",
)

SYSTEM_STATUS_COPY = (
    "그 부분은 기억했어",
    "그 부분은 확인했어",
    "네가 말한 데까지",
)

WRONG_GUESS_OPENING = re.compile(r"(?:맞지|맞아|되는\s*거지|인\s*거지)\s*\?")

TEACHER_EVALUATION_COPY = re.compile(
    r"왜\s+.+(?:라고\s+)?생각했어|왜\s+그렇게\s+생각|어떻게\s+알았어|"
    r"어떻게\s+[^?]*(?:했어|셌어|찾았어|읽었어|비교했어)|까닭은\s+무엇|"
    r"이유를\s*(?:말|설명)|설명해\s*봐|말해\s*봐"
)


def test_home_catalog_has_one_live_explicit_help_plan_per_item() -> None:
    catalog_path = Path(__file__).parents[1] / "src/mormi_api/home_teaching_catalog.json"
    raw_catalog = json.loads(catalog_path.read_text())

    assert len(raw_catalog) == len(HOME_TEACHING_CATALOG)
    for raw in raw_catalog:
        assert "hint" not in raw, raw["id"]
        assert "help_lines" not in raw, raw["id"]
        assert set(raw["help_plan"]) == {"H1", "H2", "H3"}, raw["id"]
        assert raw["help_skills"], raw["id"]


@pytest.mark.parametrize(
    "copy",
    [
        "3십과 낱개 4개를 살펴보자.",
        "본 점마다 한 번씩 수를 말해 보자.",
        "10칸에서 찬 칸을 세어 보자.",
        "같은 수를 이어 더해 보자.",
        "앞의 곱에 3을 더해 보자.",
        "십 하나를 낱개 10개로 바꾸자.",
        "일이 모자라면 받아내림해.",
        "두 무리의 수를 비교해 보자.",
        "십과 일의 자리를 살펴보자.",
        "일의 자리를 12로 만든 뒤 5를 빼자.",
    ],
)
def test_child_math_copy_rejects_known_nonstandard_or_opaque_phrases(copy: str) -> None:
    with pytest.raises(ValueError):
        validate_child_facing_math_copy([copy])


def test_every_home_help_plan_increases_support_and_closes_with_current_answer() -> None:
    for spec in HOME_TEACHING_CATALOG.values():
        h1 = spec.help_plan.H1
        h2 = spec.help_plan.H2
        h3 = spec.help_plan.H3
        normalized_bodies = {re.sub(r"\s+", "", step.body) for step in (h1, h2, h3)}
        normalized_answer = re.sub(r"[\s,]", "", str(spec.sample_problem["correct"]))

        assert len(normalized_bodies) == 3, spec.id
        assert (h1.support_type, h1.answer_policy, h1.support_mode) == (
            "attention",
            "hidden",
            "attention",
        ), spec.id
        assert h2.support_type == "guided_action", spec.id
        assert h2.answer_policy == "partial", spec.id
        assert h2.support_mode.startswith("guided_"), spec.id
        assert h2.action, spec.id
        assert (h3.support_type, h3.answer_policy, h3.support_mode) == (
            "joint_model",
            "revealed",
            "joint_model",
        ), spec.id
        assert h3.action, spec.id
        assert normalized_answer in re.sub(r"[\s,]", "", h3.body), spec.id
        assert set(h1.fact_refs) <= {"sample_problem"}, spec.id
        assert set(h2.fact_refs) <= {"sample_problem"}, spec.id
        assert set(h3.fact_refs) == {"sample_problem", "sample_answer"}, spec.id


def test_help_contract_rejects_nonprogressive_support_and_wrong_answer_policy() -> None:
    with pytest.raises(ValidationError):
        reviewed_help_card(
            HintLevel.H2,
            "숫자를 확인해 보자.",
            support_mode="attention",
            fact_refs=["left"],
            action="숫자 확인하기",
        )

    with pytest.raises(ValidationError):
        # Constructing the model directly must not let a future task reveal an
        # answer at H1 by merely changing metadata around otherwise valid copy.
        HintDefinition(
            level=HintLevel.H1,
            body="두 값을 확인해 보자.",
            support_type="attention",
            answer_policy="revealed",
            support_mode="attention",
            fact_refs=["left", "right"],
        )


def test_h2_can_use_a_nonvisual_but_stronger_support_medium() -> None:
    hint = reviewed_help_card(
        HintLevel.H2,
        "500+100=□ 식의 빈칸을 채워 보자.",
        support_mode="guided_equation",
        fact_refs=["left", "right"],
        action="두 값을 식에 넣고 빈칸 채우기",
    )

    assert hint.visual_type is None
    assert hint.support_mode == "guided_equation"


def test_every_home_turn_uses_reviewed_specific_copy() -> None:
    for spec in HOME_TEACHING_CATALOG.values():
        copy = [
            spec.effective_l4_prompt,
            *([spec.entry_prompt] if spec.entry_prompt else []),
            spec.short_prompt,
            spec.learned_line,
            *spec.short_options,
            *spec.fill_options,
        ]
        assert spec.short_prompt.endswith("?"), spec.id
        assert spec.short_correct in spec.short_options, spec.id
        assert spec.fill_correct in spec.fill_options, spec.id
        assert len(spec.short_options) == len(set(spec.short_options)), spec.id
        assert len(spec.fill_options) == len(set(spec.fill_options)), spec.id
        assert all(len(option) <= 45 for option in spec.short_options), spec.id
        assert not any(term in text for term in VAGUE_COPY for text in copy), spec.id
        assert not any(term in text for term in SYSTEM_STATUS_COPY for text in copy), spec.id
        assert not any(TEACHER_EVALUATION_COPY.search(text) for text in copy), spec.id
        assert not WRONG_GUESS_OPENING.search(spec.effective_l4_prompt), spec.id


def test_korean_postpositions_attach_to_the_fill_blank() -> None:
    detached_particle = re.compile(r"□\s+(?:으로|로|을|를|이|가|은|는|의|와|과)(?:\s|$)")

    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        sentence = task.steps[ExpressionLevel.L1][1].input.config["sentence"]
        assert isinstance(sentence, str)
        assert not detached_particle.search(sentence), (spec.id, sentence)


def test_queue_reason_choices_explain_the_wait_instead_of_repeating_the_question() -> None:
    for left, right in permutations(range(1, 6), 2):
        task = queue_task(task_id="copy-test", stage_id="queue", left=left, right=right)
        smaller, larger = sorted((left, right))
        side = "왼쪽" if left < right else "오른쪽"
        expected_labels = [
            f"내 앞에 {smaller}명이 기다려서",
            f"내 앞에 {larger}명이 기다려서",
        ]

        for level, index in ((ExpressionLevel.L2, 3), (ExpressionLevel.L1, 3)):
            step = task.steps[level][index]
            assert step.prompt == f"나는 왜 {side} 줄이 더 빠른지 헷갈려... 같이 골라 볼까?"
            assert [choice.label for choice in step.input.choices] == expected_labels
            assert all("사람이 적어서" not in choice.label for choice in step.input.choices)


def test_every_dynamic_queue_choice_stays_inside_the_shared_count_contract() -> None:
    for left, right in permutations(range(1, 6), 2):
        task = queue_task(task_id="range-test", stage_id="queue", left=left, right=right)

        for step_index, expected in ((0, left), (1, right)):
            choices = task.steps[ExpressionLevel.L2][step_index].input.choices
            values = [int(choice.id) for choice in choices]
            assert expected in values
            assert all(1 <= value <= 5 for value in values)


@pytest.mark.parametrize("left,right", [(0, 2), (2, 0), (6, 2), (2, 6), (4, 4)])
def test_queue_task_rejects_counts_outside_the_shared_contract(left: int, right: int) -> None:
    with pytest.raises(ValueError):
        queue_task(task_id="invalid-range", stage_id="queue", left=left, right=right)


def test_every_mormi_task_question_asks_from_its_own_confusion() -> None:
    for left, right in permutations(range(1, 6), 2):
        task = queue_task(task_id="persona-test", stage_id="queue", left=left, right=right)
        for steps in task.steps.values():
            for step in steps:
                assert not TEACHER_EVALUATION_COPY.search(step.prompt), step.prompt
                assert not TEACHER_EVALUATION_COPY.search(step.fallback_text), step.fallback_text


def test_safety_fallbacks_set_a_clear_boundary_without_vague_scene_language() -> None:
    assert deterministic_safety("야 이 개새끼야") is SafetyCategory.ABUSIVE
    assert deterministic_safety("ㅅㅂ 닥쳐") is SafetyCategory.ABUSIVE
    assert safety_redirect(SafetyCategory.ABUSIVE) == "그 말은 듣기 싫어."

    for category in SafetyCategory:
        text = safety_redirect(category)
        assert len(text) <= 50
        assert "지금 상황" not in text
        assert "지금 장면" not in text


def test_ambiguous_korean_endings_do_not_trigger_the_sexual_gate() -> None:
    assert deterministic_safety("너 알면서 일부러 물어보지?") is SafetyCategory.NORMAL
    assert deterministic_safety("아직 자지 마") is SafetyCategory.NORMAL
    assert deterministic_safety("보지 보여줘") is SafetyCategory.SEXUAL

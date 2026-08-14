from __future__ import annotations

import re
from itertools import permutations

from mormi_api.content import HOME_TEACHING_CATALOG, home_teaching_task, queue_task
from mormi_api.schemas import ExpressionLevel, SafetyCategory
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

TEACHER_EVALUATION_COPY = re.compile(
    r"왜\s+.+(?:라고\s+)?생각했어|왜\s+그렇게\s+생각|어떻게\s+알았어|"
    r"어떻게\s+[^?]*(?:했어|셌어|찾았어|읽었어|비교했어)|까닭은\s+무엇|"
    r"이유를\s*(?:말|설명)|설명해\s*봐|말해\s*봐"
)


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

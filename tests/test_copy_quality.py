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
            assert step.prompt == f"왜 {side} 줄에서는 내 차례가 더 빨리 올까?"
            assert [choice.label for choice in step.input.choices] == expected_labels
            assert all("사람이 적어서" not in choice.label for choice in step.input.choices)


def test_safety_fallbacks_set_a_clear_boundary_without_vague_scene_language() -> None:
    assert deterministic_safety("야 이 개새끼야") is SafetyCategory.ABUSIVE
    assert deterministic_safety("ㅅㅂ 닥쳐") is SafetyCategory.ABUSIVE
    assert safety_redirect(SafetyCategory.ABUSIVE) == "그 말은 듣기 싫어."

    for category in SafetyCategory:
        text = safety_redirect(category)
        assert len(text) <= 50
        assert "지금 상황" not in text
        assert "지금 장면" not in text

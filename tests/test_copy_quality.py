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
    calculation_task,
    home_teaching_task,
    queue_task,
    reviewed_help_card,
    simple_calculation_task,
)
from mormi_api.copy_quality import validate_child_facing_math_copy
from mormi_api.schemas import (
    ExpressionLevel,
    HintLevel,
    InputKind,
    SafetyCategory,
    SceneType,
)
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

# L2는 아이가 선택해서 모르미에게 알려주는 단계다. 다만 수학 문장 속
# ``똑같이``까지 공동 수행으로 오인하지 않도록 실제 공동 행동 표현만 잡는다.
JOINT_L2_COPY = re.compile(
    r"(?:같이|함께)\s*(?:골라|찾아|선택|해\s*보|풀어|계산|채워|읽어)"
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


def test_every_home_task_uses_the_canonical_four_level_ladder() -> None:
    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        assert set(task.steps) == {
            ExpressionLevel.L4,
            ExpressionLevel.L3,
            ExpressionLevel.L2,
            ExpressionLevel.L0,
        }, spec.id
        assert all(
            step.input.kind is InputKind.CHOICES
            for step in task.steps[ExpressionLevel.L2]
        ), spec.id


def test_queue_reason_choices_explain_the_wait_instead_of_repeating_the_question() -> None:
    for left, right in permutations(range(1, 6), 2):
        task = queue_task(task_id="copy-test", stage_id="queue", left=left, right=right)
        smaller, larger = sorted((left, right))
        expected_labels = [
            f"내 앞에 {smaller}명이 기다려서",
            f"내 앞에 {larger}명이 기다려서",
        ]

        step = task.steps[ExpressionLevel.L2][3]
        assert step.prompt == "왜 그 줄에서 내 차례가 더 빨리 오는지 골라서 알려줄 수 있어?"
        assert "같이" not in step.prompt
        assert str(left) not in step.prompt
        assert str(right) not in step.prompt
        assert [choice.label for choice in step.input.choices] == expected_labels
        assert all("사람이 적어서" not in choice.label for choice in step.input.choices)


def test_queue_followups_do_not_reveal_unverified_side_or_counts() -> None:
    for left, right in permutations(range(1, 6), 2):
        task = queue_task(task_id="trust-test", stage_id="queue", left=left, right=right)
        correct_side = "왼쪽" if left < right else "오른쪽"

        l3_reason = task.steps[ExpressionLevel.L3][2]
        l2_reason = task.steps[ExpressionLevel.L2][3]
        for copy in (
            l3_reason.prompt,
            l3_reason.fallback_text,
            l2_reason.prompt,
            l2_reason.fallback_text,
        ):
            assert correct_side not in copy
            assert str(left) not in copy
            assert str(right) not in copy
        assert "같이" not in l2_reason.prompt
        assert "같이" not in l2_reason.fallback_text


def test_every_l2_prompt_keeps_the_child_as_the_teaching_subject() -> None:
    tasks = [
        *(home_teaching_task(spec, skill_id=spec.id) for spec in HOME_TEACHING_CATALOG.values()),
        calculation_task(
            task_id="legacy-addition-copy-test",
            title="덧셈",
            skill_id="addition",
            left=1200,
            right=500,
            operation="addition",
            result=1700,
        ),
        calculation_task(
            task_id="legacy-subtraction-copy-test",
            title="뺄셈",
            skill_id="subtraction",
            left=2000,
            right=1800,
            operation="subtraction",
            result=200,
        ),
        simple_calculation_task(
            task_id="cafe-addition-copy-test",
            stage_id="menu_total",
            title="메뉴 값 계산",
            left=700,
            right=500,
            operation="addition",
            left_label="주스",
            right_label="빵",
            behavior="menu_total",
            note_policy="stage",
            coauthored_note="두 메뉴의 값을 더하면 모두 1,200원이야.",
            context={},
        ),
        simple_calculation_task(
            task_id="cafe-subtraction-copy-test",
            stage_id="change",
            title="거스름돈 계산",
            left=2000,
            right=1800,
            operation="subtraction",
            left_label="낸 돈",
            right_label="메뉴 값",
            behavior="change",
            note_policy="stage",
            coauthored_note="낸 돈에서 메뉴 값을 빼면 거스름돈을 알 수 있어.",
            context={},
        ),
    ]

    for task in tasks:
        for step in task.steps[ExpressionLevel.L2]:
            assert step.input.kind is InputKind.CHOICES, (task.id, step.id)
            assert not JOINT_L2_COPY.search(step.prompt), (task.id, step.id, step.prompt)
            assert not JOINT_L2_COPY.search(step.fallback_text), (
                task.id,
                step.id,
                step.fallback_text,
            )


def test_calculation_followups_do_not_preteach_operation_or_column_method() -> None:
    for operation, left, right, result, hidden_terms in (
        ("addition", 1200, 500, 1700, ("더해야", "올림")),
        ("subtraction", 2000, 1800, 200, ("빼야", "받아내림")),
    ):
        task = calculation_task(
            task_id=f"calculation-trust-{operation}",
            title="계산",
            skill_id=operation,
            left=left,
            right=right,
            operation=operation,
            result=result,
            scene=SceneType.HOME_TEACH,
        )
        l3_operation = task.steps[ExpressionLevel.L3][1]
        l3_method = task.steps[ExpressionLevel.L3][2]
        child_copy = (
            l3_operation.prompt,
            l3_operation.fallback_text,
            l3_method.prompt,
            l3_method.fallback_text,
        )
        for hidden in hidden_terms:
            assert all(hidden not in copy for copy in child_copy), (operation, hidden, child_copy)


def _targeted_followup(task, verified_slots: dict[str, object]):
    return task.active_step(
        ExpressionLevel.L4,
        verified_slots,
        entry_active=False,
        targeted_followup=True,
    )


def test_home_count_and_compare_followups_use_only_child_verified_facts() -> None:
    count_task = home_teaching_task(
        HOME_TEACHING_CATALOG["number-count"],
        skill_id="number-count",
    )
    count_method = _targeted_followup(count_task, {"answer": "3"})
    assert count_method.target_slots == ["tracking"]
    assert "3" not in count_method.prompt

    compare_task = home_teaching_task(
        HOME_TEACHING_CATALOG["number-compare"],
        skill_id="number-compare",
    )
    compare_reason = _targeted_followup(compare_task, {"answer": "right"})
    assert compare_reason.target_slots == ["reason"]
    assert "오른쪽" in compare_reason.prompt
    assert "3" not in compare_reason.prompt
    assert "5" not in compare_reason.prompt

    compare_answer = _targeted_followup(
        compare_task,
        {"reason": "count_comparison"},
    )
    assert compare_answer.target_slots == ["answer"]
    assert "오른쪽이 더 많" not in compare_answer.prompt


def test_queue_followups_do_not_reveal_the_next_count_or_correct_side() -> None:
    task = queue_task(task_id="queue-state-trust", stage_id="queue", left=3, right=5)

    right_count = _targeted_followup(task, {"left_count": 3})
    assert "right_count" in right_count.target_slots
    assert "5" not in right_count.prompt

    reason = _targeted_followup(
        task,
        {"left_count": 3, "right_count": 5, "final_choice": "left"},
    )
    assert reason.target_slots == ["reason"]
    assert "왼쪽" not in reason.prompt
    assert "3" not in reason.prompt
    assert "5" not in reason.prompt

    choice = _targeted_followup(
        task,
        {"left_count": 3, "right_count": 5, "reason": "fewer_people"},
    )
    assert choice.target_slots == ["final_choice"]
    assert "왼쪽" not in choice.prompt


@pytest.mark.parametrize("operation,left,right,result", [
    ("addition", 700, 500, 1200),
    ("subtraction", 2000, 1800, 200),
])
def test_cafe_calculation_followups_do_not_reveal_missing_math_slots(
    operation: str,
    left: int,
    right: int,
    result: int,
) -> None:
    task = simple_calculation_task(
        task_id=f"cafe-state-trust-{operation}",
        stage_id="menu_total" if operation == "addition" else "change",
        title="금액 계산",
        left=left,
        right=right,
        operation=operation,
        left_label="첫 금액",
        right_label="둘째 금액",
        behavior="menu_total" if operation == "addition" else "change",
        note_policy="stage",
        coauthored_note="검사용 문장",
        context={},
    )

    operation_step = _targeted_followup(task, {"result": result})
    assert operation_step.target_slots == ["operation"]
    assert "더하" not in operation_step.prompt
    assert "빼" not in operation_step.prompt

    result_step = _targeted_followup(task, {"operation": operation})
    assert result_step.target_slots == ["result"]
    assert str(result) not in result_step.prompt


def test_all_home_catalog_followups_ask_only_for_the_missing_slot() -> None:
    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        if len(task.required_slots) != 2:
            continue
        first, second = task.required_slots

        after_first = _targeted_followup(task, {first: "verified"})
        after_second = _targeted_followup(task, {second: "verified"})

        assert second in after_first.target_slots, (spec.id, after_first.id)
        assert first not in after_first.target_slots, (spec.id, after_first.id)
        assert first in after_second.target_slots, (spec.id, after_second.id)
        assert second not in after_second.target_slots, (spec.id, after_second.id)


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

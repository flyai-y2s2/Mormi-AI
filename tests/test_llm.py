from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from mormi_api.content import (
    HOME_TEACH_TASK_ID,
    HOME_TEACHING_CATALOG,
    home_teaching_task,
    menu_selection_task,
    queue_task,
    simple_calculation_task,
)
from mormi_api.llm import (
    ClaudeGateway,
    structured_output_schema,
    validate_speaker_output,
    validate_speaker_verification,
)
from mormi_api.schemas import (
    CafeMenuItem,
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SceneType,
    SessionState,
    SlotClaim,
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
    SpeakerVerification,
    SpeakerVerificationPolicy,
    UtteranceAnalysis,
)
from mormi_api.settings import Settings


def object_schemas(node: object) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(object_schemas(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(object_schemas(value))
    return found


def test_classifier_schema_is_strict_for_every_nested_object() -> None:
    schema = structured_output_schema(UtteranceAnalysis)
    objects = object_schemas(schema)

    assert len(objects) >= 2  # root UtteranceAnalysis and nested SlotClaim
    assert all(item.get("additionalProperties") is False for item in objects)
    assert all(item.get("required") == list(item.get("properties", {})) for item in objects)


def test_speaker_schema_is_strict() -> None:
    schema = structured_output_schema(SpeakerOutput)
    objects = object_schemas(schema)

    assert objects
    assert all(item.get("additionalProperties") is False for item in objects)
    assert all(item.get("required") == list(item.get("properties", {})) for item in objects)

    verifier_objects = object_schemas(structured_output_schema(SpeakerVerification))
    assert verifier_objects
    assert all(item.get("additionalProperties") is False for item in verifier_objects)


def speaker_context() -> SpeakerContext:
    return SpeakerContext(
        dialogue_act="acknowledge_partial",
        required_question="나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        required_slot_ids=["reason"],
        required_slot_descriptions={"reason": "왼쪽 줄에서 덜 기다리는 이유"},
        fallback_text="아, 왼쪽이구나! 나는 왜 이 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
    )


def speaker_output(text: str, context: SpeakerContext) -> SpeakerOutput:
    return SpeakerOutput(
        text=text,
        dialogue_act=context.dialogue_act,
        asked_slot_ids=context.required_slot_ids,
    )


def speaker_guard() -> SpeakerGuardContract:
    return SpeakerGuardContract()


def test_speaker_rejects_grading_synonyms() -> None:
    context = speaker_context()
    for text in (
        "맞아! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "정확해! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "잘했어! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "옳아! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
    ):
        assert (
            validate_speaker_output(
                speaker_output(text, context), context, speaker_guard()
            )
            is None
        )


def test_speaker_rejects_system_status_voice() -> None:
    context = speaker_context()
    for text in (
        "그 부분은 기억했어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "네가 말한 데까지는 들었어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "그 부분은 확인했어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
    ):
        assert (
            validate_speaker_output(
                speaker_output(text, context), context, speaker_guard()
            )
            is None
        )


def test_speaker_must_keep_the_orchestrator_question() -> None:
    context = speaker_context()
    assert (
        validate_speaker_output(
            speaker_output(
                "아, 왼쪽이구나! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
                context,
            ),
            context,
            speaker_guard(),
        )
        is not None
    )
    assert (
        validate_speaker_output(
            speaker_output("왼쪽이구나. 무슨 색을 좋아해?", context),
            context,
            speaker_guard(),
        )
        is None
    )


def test_semantic_speaker_may_paraphrase_only_with_matching_contract() -> None:
    context = speaker_context().model_copy(
        update={"verification_policy": SpeakerVerificationPolicy.SEMANTIC}
    )
    output = speaker_output("나는 이 줄이 왜 더 빠른지 헷갈려... 알려줄 수 있어?", context)
    assert validate_speaker_output(output, context, speaker_guard()) is not None

    wrong_focus = output.model_copy(update={"asked_slot_ids": ["answer"]})
    assert validate_speaker_output(wrong_focus, context, speaker_guard()) is None


def test_speaker_can_ground_a_clarification_in_an_exact_child_phrase() -> None:
    context = SpeakerContext(
        dialogue_act="acknowledge_partial",
        required_question="어떻게 세는지 알려주면 안 될까?",
        verified_facts={"answer": "점은 세 개야."},
        required_slot_ids=["tracking"],
        required_slot_descriptions={"tracking": "점을 빠뜨리지 않고 세는 방법"},
        child_expression_mode="quote_safe",
        child_expression="차근차근 세어봐",
        allowed_numbers=["3"],
        verification_policy=SpeakerVerificationPolicy.SEMANTIC,
        fallback_text="아, 세 개구나! 어떻게 세는지 알려주면 안 될까?",
    )
    guard = SpeakerGuardContract(child_expression_source="차근차근 세어봐")
    output = SpeakerOutput(
        text="아, 세 개구나! ‘차근차근’은 어떻게 세는 거야?",
        dialogue_act=context.dialogue_act,
        asked_slot_ids=["tracking"],
        used_verified_slots=["answer"],
        used_child_expression=True,
        used_child_expression_spans=["차근차근"],
    )
    assert validate_speaker_output(output, context, guard) is not None

    verification = SpeakerVerification(
        approved=True,
        dialogue_act_preserved=True,
        required_focus_preserved=True,
        only_allowed_math_used=True,
        child_not_evaluated=True,
        character_consistent=True,
        detected_dialogue_act=context.dialogue_act,
        detected_asked_slot_ids=["tracking"],
        question_evidence_span="‘차근차근’은 어떻게 세는 거야?",
        child_expression_spans=["차근차근"],
        reason_code="approved",
    )
    assert validate_speaker_verification(verification, context, guard, output) is True

    invented_quote = output.model_copy(
        update={"used_child_expression_spans": ["천천히"]}
    )
    assert validate_speaker_output(invented_quote, context, guard) is None


def test_speaker_cannot_repeat_the_previous_line_verbatim() -> None:
    context = SpeakerContext(
        dialogue_act="acknowledge_unstructured_partial",
        previous_question="어떻게 세는지 알려주면 안 될까?",
        required_question="어떻게 세는지 알려주면 안 될까?",
        fallback_text="내가 또 똑같이 물었네... 같이 골라 볼까?",
    )
    assert (
        validate_speaker_output(
            speaker_output("어떻게 세는지 알려주면 안 될까?", context),
            context,
            speaker_guard(),
        )
        is None
    )


def test_classifier_receives_shared_semantic_roles_across_home_and_cafe_tasks() -> None:
    menu_items = (
        CafeMenuItem(id="tea", name="차", price=2_000),
        CafeMenuItem(id="cake", name="케이크", price=3_000),
    )
    task_cases = [
        (
            home_teaching_task(
                HOME_TEACHING_CATALOG["number-count"],
                skill_id="number-count",
            ),
            {"answer": "conclusion", "tracking": "method"},
        ),
        (
            home_teaching_task(
                HOME_TEACHING_CATALOG["number-compare"],
                skill_id="number-compare",
            ),
            {"answer": "conclusion", "reason": "reason"},
        ),
        (
            home_teaching_task(
                HOME_TEACHING_CATALOG["clock-basic"],
                skill_id="clock-basic",
            ),
            {"answer": "conclusion", "rule": "explanation"},
        ),
        (
            queue_task(task_id="queue_roles", stage_id="queue", left=2, right=5),
            {
                "left_count": "observation",
                "right_count": "observation",
                "smaller_number": "conclusion",
                "final_choice": "selection",
                "reason": "reason",
            },
        ),
        (
            simple_calculation_task(
                task_id="calculation_roles",
                stage_id="menu_total",
                title="메뉴값 계산하기",
                left=2_000,
                right=3_000,
                operation="addition",
                left_label="차",
                right_label="케이크",
                behavior="menu_total",
                note_policy="stage",
                coauthored_note="두 메뉴 가격을 더해서 전체 가격을 구해.",
                context={},
            ),
            {"operation": "operation", "result": "conclusion"},
        ),
        (
            menu_selection_task(
                task_id="menu_roles",
                stage_id="budget_menu",
                menu_items=menu_items,
                mormi_menu=menu_items[0],
                budget=10_000,
                auto_total=True,
                behavior="budget_menu_selection",
                note_policy="stage",
            ),
            {"child_menu": "selection"},
        ),
    ]

    for task, roles in task_cases:
        state = SessionState(
            learner_id=1,
            scene=task.scene,
            scenario_id="semantic_role_test",
            task_ids=[task.id],
            expression_level=ExpressionLevel.L4,
        )
        prompt = ClaudeGateway._classifier_prompt(
            state,
            task,
            previous_question=task.steps[ExpressionLevel.L4][0].prompt,
            response=ChildResponse(
                turn_id="turn_roles",
                response_id=uuid4(),
                type=ResponseType.TEXT,
                text="내가 본 걸 내 말로 설명했어",
            ),
        )
        payload = json.loads(prompt)

        assert set(payload["semantic_role_policy"]) == {
            "observation",
            "conclusion",
            "operation",
            "method",
            "reason",
            "explanation",
            "selection",
        }
        slots = payload["all_task_slot_contracts"]
        assert {slot_id: slots[slot_id]["semantic_role"] for slot_id in roles} == roles
        method_contract = payload["method_acceptance_contract"]
        assert method_contract["policy"] == task.help_method_policy
        assert method_contract["reviewed_examples"] == task.accepted_methods
        assert method_contract["help_card_route_is_not_the_only_correct_method"] is (
            task.help_method_policy == "open_methods"
        )


def test_speaker_rejects_teacher_style_probe() -> None:
    context = SpeakerContext(
        dialogue_act="acknowledge_partial",
        required_question="나 3이랑 5를 어떻게 비교해야 할지 헷갈려... 알려줄 수 있어?",
        required_slot_ids=["reason"],
        required_slot_descriptions={"reason": "3과 5를 비교하는 방법"},
        allowed_numbers=["3", "5"],
        fallback_text="나 3이랑 5를 어떻게 비교해야 할지 헷갈려... 알려줄 수 있어?",
    )
    for text in (
        "왜 오른쪽에 더 많다고 생각했어?",
        "오른쪽인 걸 어떻게 알았어?",
        "그렇게 생각한 근거가 뭐야?",
        "이유를 설명해 봐.",
    ):
        assert (
            validate_speaker_output(
                speaker_output(text, context), context, speaker_guard()
            )
            is None
        )

    assert (
        validate_speaker_output(
            speaker_output(
                "나 3이랑 5를 어떻게 비교해야 할지 헷갈려... 알려줄 수 있어?",
                context,
            ),
            context,
            speaker_guard(),
        )
        is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_category",
    [ResponseCategory.UNRELATED_RESPONSE, ResponseCategory.CONCEPTUAL_ERROR],
)
async def test_negative_free_text_classification_is_semantically_rechecked_once(
    initial_category: ResponseCategory,
) -> None:
    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=initial_category,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.7,
    )
    audited = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="tracking",
                value="count_each_once",
                factual=True,
                evidence_span="하나 둘 셋",
            )
        ],
        confidence=0.95,
    )

    class FakeMessages:
        def __init__(self) -> None:
            self.outputs = [first.model_dump_json(), audited.model_dump_json()]
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=self.outputs.pop(0))],
            )

    messages = FakeMessages()
    gateway = ClaudeGateway(Settings(anthropic_api_key=None))
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    task = home_teaching_task(HOME_TEACHING_CATALOG["number-count"], skill_id="number-count")
    state = SessionState(
        learner_id=1,
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=[HOME_TEACH_TASK_ID],
        expression_level=ExpressionLevel.L4,
    )

    result = await gateway.classify(
        state=state,
        task=task,
        previous_question="점이 두 개 있는 것 같은데, 너는 몇 개로 셌어?",
        response=ChildResponse(
            turn_id="turn_1",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="하나, 둘, 셋 하고 세면 돼",
        ),
    )

    assert result.response_category is ResponseCategory.CORRECT_PARTIAL
    assert result.claims[0].slot_id == "tracking"
    assert len(messages.prompts) == 2
    assert "semantic_relation_audit" in messages.prompts[1]
    assert "문구 일치가 아니라" in messages.prompts[0]
    assert "10개 중 색칠된 게 3개던데" not in messages.prompts[0]

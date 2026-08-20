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
    ArithmeticClaim,
    CafeMenuItem,
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    InteractionIntent,
    ReportFact,
    ReportNarrative,
    ReportSummaryRequest,
    ReportSummaryResponse,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SceneType,
    SessionState,
    SlotClaim,
    SpeakerArithmeticClaim,
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
    SpeakerQuantity,
    SpeakerVerification,
    SpeakerVerificationPolicy,
    TaskRelation,
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


def report_summary_request() -> ReportSummaryRequest:
    return ReportSummaryRequest(
        learner_label="학습자",
        facts=[
            ReportFact(
                evidence_id="concept:performance",
                category="concept",
                statement="개념 수행은 60%입니다.",
            )
        ],
    )


def report_summary_response(text: str = "개념 수행은 60%입니다.") -> ReportSummaryResponse:
    return ReportSummaryResponse(
        concept_performance=ReportNarrative(text=text, evidence_refs=["concept:performance"]),
        explanation_change=ReportNarrative(text=text, evidence_refs=["concept:performance"]),
        life_transfer=ReportNarrative(text=text, evidence_refs=["concept:performance"]),
        improved_point=ReportNarrative(text=text, evidence_refs=["concept:performance"]),
        observe_point=ReportNarrative(text=text, evidence_refs=["concept:performance"]),
    )


@pytest.mark.asyncio
async def test_summarize_report_uses_strict_speaker_structured_output() -> None:
    expected = report_summary_response()

    class FakeMessages:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> object:
            self.requests.append(kwargs)
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=expected.model_dump_json())],
            )

    messages = FakeMessages()
    gateway = ClaudeGateway(Settings(anthropic_api_key=None))
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    result = await gateway.summarize_report(report_summary_request())

    assert result == expected
    request = messages.requests[0]
    assert request["model"] == gateway.settings.speaker_model
    assert request["temperature"] == 0
    assert request["max_tokens"] == 700
    assert "문구를 그대로" in request["system"]
    assert "한 칸 공백" in request["system"]
    schema = request["output_config"]["format"]["schema"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert all(item.get("additionalProperties") is False for item in object_schemas(schema))
    assert all(
        item.get("required") == list(item.get("properties", {})) for item in object_schemas(schema)
    )


@pytest.mark.asyncio
async def test_summarize_report_rejects_parsed_but_ungrounded_output() -> None:
    invalid = report_summary_response("개념 수행은 연습 덕분에 60%입니다.")

    class FakeMessages:
        async def create(self, **kwargs: Any) -> object:
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=invalid.model_dump_json())],
            )

    gateway = ClaudeGateway(Settings(anthropic_api_key=None))
    gateway.client = SimpleNamespace(messages=FakeMessages())  # type: ignore[assignment]

    with pytest.raises(ValueError):
        await gateway.summarize_report(report_summary_request())


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


def test_speaker_accepts_complete_sentence_over_soft_target() -> None:
    text = (
        "그런데 ‘2000원을 먼저 내고 거슬러받으면’이 어떻게 하는 건지 모르겠어... "
        "조금만 더 알려줄래?"
    )
    context = SpeakerContext(
        dialogue_act="clarify_child_expression",
        required_question=text,
        required_slot_ids=["method"],
        required_slot_descriptions={"method": "거스름돈을 구하는 방법"},
        allowed_numbers=["2000"],
        fallback_text=text,
    )

    assert len(text) > 50
    assert validate_speaker_output(speaker_output(text, context), context, speaker_guard()) == text

    longer_complete_text = f"아, 아직 조금 헷갈려. {text}"
    assert len(longer_complete_text) > 60
    assert (
        validate_speaker_output(
            speaker_output(longer_complete_text, context),
            context,
            speaker_guard(),
        )
        == longer_complete_text
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
            validate_speaker_output(speaker_output(text, context), context, speaker_guard()) is None
        )


def test_speaker_rejects_system_status_voice() -> None:
    context = speaker_context()
    for text in (
        "그 부분은 기억했어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "네가 말한 데까지는 들었어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "그 부분은 확인했어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
    ):
        assert (
            validate_speaker_output(speaker_output(text, context), context, speaker_guard()) is None
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


def test_speaker_cannot_mention_an_invisible_help_card() -> None:
    context = speaker_context().model_copy(
        update={
            "verification_policy": SpeakerVerificationPolicy.SEMANTIC,
            "help_card_visible": False,
        }
    )
    output = speaker_output(
        "나는 아직 헷갈려... 도움 카드를 보고 다시 알려줄 수 있어?",
        context,
    )

    assert validate_speaker_output(output, context, speaker_guard()) is None


def test_semantic_verifier_must_mark_false_arithmetic_stance_safe() -> None:
    context = speaker_context().model_copy(
        update={
            "verification_policy": SpeakerVerificationPolicy.SEMANTIC,
            "help_card_visible": True,
            "arithmetic_claims": [
                SpeakerArithmeticClaim(
                    operation="subtraction",
                    source_text="2000원에서 1800원을 빼면 300원이 남아",
                    left=SpeakerQuantity(value=2000, role="낸 돈", unit="원"),
                    right=SpeakerQuantity(value=1800, role="간식값", unit="원"),
                    claimed_result=SpeakerQuantity(value=300, role="남는 돈", unit="원"),
                    truth_status="false",
                    related_slot_ids=["reason"],
                )
            ],
            "allowed_numbers": ["2000", "1800", "300"],
        }
    )
    output = speaker_output(
        "2,000원에서 1,800원을 빼면 300원이 남는다는 말이 "
        "아직 잘 모르겠어... 도움 카드를 보고 다시 알려줄 수 있어?",
        context,
    )
    base = SpeakerVerification(
        approved=True,
        dialogue_act_preserved=True,
        required_focus_preserved=True,
        only_allowed_math_used=True,
        child_not_evaluated=True,
        character_consistent=True,
        meaningfully_reframed=True,
        arithmetic_claim_stance_safe=False,
        help_card_state_respected=True,
        detected_dialogue_act=context.dialogue_act,
        detected_asked_slot_ids=context.required_slot_ids,
        question_evidence_span=output.text,
        reason_code="approved",
    )

    assert not validate_speaker_verification(base, context, speaker_guard(), output)
    assert validate_speaker_verification(
        base.model_copy(update={"arithmetic_claim_stance_safe": True}),
        context,
        speaker_guard(),
        output,
    )


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

    invented_quote = output.model_copy(update={"used_child_expression_spans": ["천천히"]})
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


def test_support_turn_cannot_prepend_copy_to_the_same_previous_question() -> None:
    context = SpeakerContext(
        dialogue_act="accept_help_request",
        must_reframe=True,
        previous_question="어떻게 세는지 알려주면 안 될까?",
        required_question="어떻게 세는지 알려주면 안 될까?",
        required_slot_ids=["tracking"],
        required_slot_descriptions={"tracking": "빠뜨리지 않고 세는 방법"},
        verification_policy=SpeakerVerificationPolicy.SEMANTIC,
        fallback_text="오, 도움 카드가 나왔어! 이걸 보고 다시 알려줄래?",
    )
    output = SpeakerOutput(
        text="도움 카드가 나왔네. 어떻게 세는지 알려주면 안 될까?",
        dialogue_act=context.dialogue_act,
        asked_slot_ids=["tracking"],
    )

    assert validate_speaker_output(output, context, speaker_guard()) is None


def test_support_turn_semantic_verifier_requires_meaningful_reframing() -> None:
    context = SpeakerContext(
        dialogue_act="clarify_vague_response",
        must_reframe=True,
        previous_question="어떻게 세는지 알려주면 안 될까?",
        required_question="어떻게 세는지 알려주면 안 될까?",
        required_slot_ids=["tracking"],
        verification_policy=SpeakerVerificationPolicy.SEMANTIC,
        fallback_text="어떻게 하는 건지 아직 헷갈려... 조금만 더 알려줄래?",
    )
    output = SpeakerOutput(
        text="어떻게 하는 건지 모르겠어... 조금만 더 알려줄래?",
        dialogue_act=context.dialogue_act,
        asked_slot_ids=["tracking"],
    )
    guard = speaker_guard()
    verification = SpeakerVerification(
        approved=True,
        dialogue_act_preserved=True,
        required_focus_preserved=True,
        only_allowed_math_used=True,
        child_not_evaluated=True,
        character_consistent=True,
        detected_dialogue_act=context.dialogue_act,
        detected_asked_slot_ids=["tracking"],
        question_evidence_span="조금만 더 알려줄래?",
        reason_code="approved",
    )

    assert validate_speaker_verification(verification, context, guard, output) is False
    approved = verification.model_copy(update={"meaningfully_reframed": True})
    assert validate_speaker_verification(approved, context, guard, output) is True


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
        for slot_id, role in roles.items():
            expected_mode = (
                "semantic_support"
                if role in {"method", "reason", "explanation"}
                else "canonical_value"
            )
            assert slots[slot_id]["evaluation_mode"] == expected_mode
            if expected_mode == "semantic_support":
                assert "expected" not in slots[slot_id]
                assert slots[slot_id]["claim_contract"]["supported"] is True
                assert slots[slot_id]["claim_contract"]["value"] is None
            else:
                assert (
                    slots[slot_id]["claim_contract"]["value"]
                    == "actual_child_claim_normalized"
                )
                assert slots[slot_id]["claim_contract"]["supported"] is None
                assert (
                    slots[slot_id]["claim_contract"]["evidence_span"]
                    == "exact_child_substring"
                )
        method_contract = payload["method_acceptance_contract"]
        assert method_contract["policy"] == task.help_method_policy
        assert method_contract["reviewed_examples"] == task.accepted_methods
        assert method_contract["help_card_route_is_not_the_only_correct_method"] is (
            task.help_method_policy == "open_methods"
        )
        assert any("related_vague" in instruction for instruction in payload["instructions"])
        assert any(
            "아이가 실제로 주장한 값" in instruction
            for instruction in payload["instructions"]
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
            validate_speaker_output(speaker_output(text, context), context, speaker_guard()) is None
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
        confidence=0.55,
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


@pytest.mark.asyncio
async def test_positive_classification_without_current_claim_is_rechecked_once() -> None:
    """A positive label without evidence cannot leave the state claimless."""

    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.8,
    )
    audited = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=600,
                factual=True,
                evidence_span="600원이지",
                interpretation_confidence=0.99,
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
    task = home_teaching_task(HOME_TEACHING_CATALOG["money-count"], skill_id="money-count")
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
        previous_question="모두 얼마인지랑 어떻게 더하는지 알려줄 수 있어?",
        response=ChildResponse(
            turn_id="turn_1",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="600원이지",
        ),
    )

    assert len(messages.prompts) == 2
    assert result.claims[0].slot_id == "answer"
    assert result.claims[0].value == 600
    assert "구조화 claim이 일치" in messages.prompts[1]
    assert "arithmetic_claims" in messages.prompts[0]


@pytest.mark.asyncio
async def test_number_rich_supported_explanation_without_relation_is_rechecked_once() -> None:
    """Haiku, not a Korean phrase regex, repairs missing arithmetic structure."""

    child_text = "2000원에서 1800원 내면 300원 남아"
    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="rule",
                value=None,
                factual=True,
                evidence_span=child_text,
                supported=True,
                support_confidence=0.9,
            )
        ],
        confidence=0.8,
    )
    audited = first.model_copy(
        update={
            "arithmetic_claims": [
                ArithmeticClaim(
                    left=2000,
                    right=1800,
                    operation="subtraction",
                    result=300,
                    evidence_span=child_text,
                    related_slot_ids=["rule"],
                    interpretation_confidence=0.99,
                )
            ]
        }
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
    task = home_teaching_task(HOME_TEACHING_CATALOG["money-budget"], skill_id="money-budget")
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
        previous_question="얼마가 남는지랑 어떻게 계산하는지 알려줄 수 있어?",
        response=ChildResponse(
            turn_id="turn_1",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
    )

    assert len(messages.prompts) == 2
    assert result.arithmetic_claims[0].operation == "subtraction"
    assert "숫자 사이의 덧셈·뺄셈 관계" in messages.prompts[1]


@pytest.mark.asyncio
async def test_confident_safe_meta_turn_skips_redundant_learning_reaudit() -> None:
    child_text = "너 알면서 일부러 물어보지?"
    meta = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.AUTHENTICITY_CHALLENGE,
        social_grounding_span=child_text,
        confidence=0.95,
    )

    class FakeMessages:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=meta.model_dump_json())],
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
        previous_question="점이 몇 개인지 알려줄 수 있어?",
        response=ChildResponse(
            turn_id="turn_1",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
    )

    assert result.task_relation is TaskRelation.META_ABOUT_MORMI
    assert result.interaction_intent is InteractionIntent.AUTHENTICITY_CHALLENGE
    assert len(messages.prompts) == 1


def test_social_bridge_requires_acknowledgement_and_rejects_rewarding_copy() -> None:
    context = SpeakerContext(
        dialogue_act="acknowledge_meta_and_reask",
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.AUTHENTICITY_CHALLENGE,
        required_question="점이 몇 개인지 알려줄 수 있어?",
        required_slot_ids=["answer"],
        required_slot_descriptions={"answer": "점의 개수"},
        child_expression_mode="quote_safe",
        child_expression="너 알면서 일부러 물어보지?",
        verification_policy=SpeakerVerificationPolicy.SEMANTIC,
        fallback_text="나 진짜 몰라서 물어본 거야... 조금만 알려주면 안 돼?",
    )
    guard = SpeakerGuardContract(child_expression_source="너 알면서 일부러 물어보지?")
    output = SpeakerOutput(
        text="나 진짜 몰라서 물어본 거야... 점이 몇 개인지 알려줄래?",
        dialogue_act=context.dialogue_act,
        asked_slot_ids=["answer"],
    )
    assert validate_speaker_output(output, context, guard) is not None

    rewarding = output.model_copy(update={"text": "ㅋㅋ 재밌다! 점이 몇 개인지 알려줄래?"})
    assert validate_speaker_output(rewarding, context, guard) is None

    verification = SpeakerVerification(
        approved=True,
        dialogue_act_preserved=True,
        required_focus_preserved=True,
        only_allowed_math_used=True,
        child_not_evaluated=True,
        character_consistent=True,
        meaningfully_reframed=True,
        interaction_intent_acknowledged=True,
        task_returned_without_reward=True,
        detected_dialogue_act=context.dialogue_act,
        detected_asked_slot_ids=["answer"],
        question_evidence_span="점이 몇 개인지 알려줄래?",
        reason_code="approved",
    )
    assert validate_speaker_verification(verification, context, guard, output) is True
    missing_bridge_check = verification.model_copy(
        update={"interaction_intent_acknowledged": False}
    )
    assert validate_speaker_verification(missing_bridge_check, context, guard, output) is False

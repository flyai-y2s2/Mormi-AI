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
    NOTE_CONTEXTUALIZER_SYSTEM,
    SPEAKER_SYSTEM,
    ClaudeGateway,
    structured_output_schema,
    validate_speaker_output,
)
from mormi_api.schemas import (
    CafeMenuItem,
    ChildResponse,
    DialogueHistoryTurn,
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
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
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


def test_l2_rejects_joint_performance_language_but_l0_allows_it() -> None:
    text = "그럼 나랑 같이 골라볼까?"
    l2_context = SpeakerContext(
        dialogue_act="reduce_expression_load",
        expression_level=ExpressionLevel.L2,
        required_question=text,
        required_slot_ids=["answer"],
        required_slot_descriptions={"answer": "선택할 답"},
        fallback_text="그럼 혹시 여기서 골라서 알려줄 수 있어?",
    )
    assert (
        validate_speaker_output(
            speaker_output(text, l2_context),
            l2_context,
            speaker_guard(),
        )
        is None
    )

    l0_context = l2_context.model_copy(
        update={
            "dialogue_act": "joint_mode",
            "expression_level": ExpressionLevel.L0,
            "required_question": text,
        }
    )
    l0_output = SpeakerOutput(
        text=text,
        dialogue_act="joint_mode",
        asked_slot_ids=["answer"],
    )
    assert validate_speaker_output(l0_output, l0_context, speaker_guard()) == text


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
        assert any(
            "related_vague" in instruction
            for instruction in payload["instructions"]
        )
        assert any(
            "아이가 실제로 주장한 값" in instruction
            for instruction in payload["instructions"]
        )


def test_classifier_prompt_carries_only_the_latest_six_dialogue_turns() -> None:
    task = home_teaching_task(
        HOME_TEACHING_CATALOG["number-count"],
        skill_id="number-count",
    )
    state = SessionState(
        learner_id=1,
        scene=SceneType.HOME_TEACH,
        scenario_id="history_prompt_test",
        task_ids=[task.id],
        expression_level=ExpressionLevel.L4,
    )
    history = [
        DialogueHistoryTurn(
            turn_id=f"history_{index}",
            mormi=f"이전 질문 {index}",
            child=f"이전 답 {index}",
            response_type=ResponseType.TEXT,
            response_category=ResponseCategory.CORRECT_PARTIAL,
        )
        for index in range(7)
    ]

    prompt = ClaudeGateway._classifier_prompt(
        state,
        task,
        previous_question=task.steps[ExpressionLevel.L4][0].prompt,
        response=ChildResponse(
            turn_id="turn_history",
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="아까 말한 방법으로 세면 돼",
        ),
        dialogue_history=history,
    )
    payload = json.loads(prompt)

    assert [turn["turn_id"] for turn in payload["recent_dialogue"]] == [
        f"history_{index}" for index in range(1, 7)
    ]
    assert payload["recent_dialogue"][-1]["child"] == "이전 답 6"
    assert any(
        "대명사·생략" in instruction for instruction in payload["instructions"]
    )


def test_reviewed_speaker_and_note_prompt_contracts_are_pinned() -> None:
    assert "아이보다 조금 서툴지만 지나치게 자기비하 하지 않는다." in SPEAKER_SYSTEM
    assert (
        "아이의 말이 모호하면 이해한 척하지 말고, 잘 모르겠다며 구체적인 정보를 "
        "자연스럽게 요청해라."
    ) in SPEAKER_SYSTEM
    for rule in (
        "L4, L3, L2에서는 아이가 모르미에게 알려주는 주체다.",
        'L2에서 "같이 고르자", "같이 찾아보자"라고 말하지 않는다.',
        '공동 수행을 뜻하는 "같이 해보자"는 L0에서만 사용한다.',
        "도움 카드가 실제로 화면에 표시된 경우에만 도움 카드를 언급한다.",
    ):
        assert rule in SPEAKER_SYSTEM

    assert "생략된 주어나 대상을 검증된 장면 사실로 명확히 한다." in (
        NOTE_CONTEXTUALIZER_SYSTEM
    )
    assert (
        "예: '둘이', '오른쪽', '그거'를 검증된 실제 대상으로 바꾼다."
        in NOTE_CONTEXTUALIZER_SYSTEM
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
async def test_negative_free_text_classification_is_left_for_graph_routing(
    initial_category: ResponseCategory,
) -> None:
    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=initial_category,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.7,
    )
    class FakeMessages:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=first.model_dump_json())],
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

    assert result.response_category is initial_category
    assert result.claims == []
    assert len(messages.prompts) == 1
    assert "문구 일치가 아니라" in messages.prompts[0]
    assert "10개 중 색칠된 게 3개던데" not in messages.prompts[0]


@pytest.mark.asyncio
async def test_positive_classification_without_current_claim_is_left_for_graph_routing() -> None:
    """A claim-free positive label stays evidence-free without a second judge."""

    first = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.8,
    )
    class FakeMessages:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=first.model_dump_json())],
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

    assert len(messages.prompts) == 1
    assert result.claims == []
    assert "arithmetic_claims" in messages.prompts[0]


@pytest.mark.asyncio
async def test_number_rich_explanation_without_relation_is_left_for_graph_routing() -> None:
    """The primary pass exposes ambiguity without paying for a second Haiku call."""

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
    class FakeMessages:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def create(self, **kwargs: Any) -> object:
            self.prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=first.model_dump_json())],
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

    assert len(messages.prompts) == 1
    assert result.arithmetic_claims == []
    assert "arithmetic_claims" in messages.prompts[0]


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

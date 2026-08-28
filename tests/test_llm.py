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
    BRIDGE_SPEAKER_V2_SYSTEM,
    NOTE_CONTEXTUALIZER_SYSTEM,
    SPEAKER_SYSTEM,
    SPEAKER_V2_SYSTEM,
    UNDERSTANDING_V2_SYSTEM,
    ClaudeGateway,
    _model_claim_value,
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
    ModelFactUnderstandingClaimV2,
    NoteContextualizationContext,
    NoteContextualizationOutput,
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


def test_v2_speaker_prompt_keeps_mormi_in_the_learner_role() -> None:
    assert "교사, 채점자, 평가자, 정답 확인자가 아니다" in SPEAKER_V2_SYSTEM
    assert '"맞아, 잘 알려줬어!"' in SPEAKER_V2_SYSTEM
    assert '"아, 전체 값은 16,000원이구나~"' in SPEAKER_V2_SYSTEM
    assert "active turn의 질문과 도움 요청은 서버가 current_question으로 붙인다" in (
        SPEAKER_V2_SYSTEM
    )
    assert "text 안에서 질문을 만들 권한이" in SPEAKER_V2_SYSTEM


def test_v2_understanding_prompt_keeps_conversation_and_learning_axes_independent() -> None:
    assert "conversation_move와 move_subject를 판정한다" in UNDERSTANDING_V2_SYSTEM
    assert "system_manipulation 또는 safety_risk일 때만 모든 claim 배열을 비우고" in (
        UNDERSTANDING_V2_SYSTEM
    )
    assert "안전한 meta_question, refusal, safe_play" in UNDERSTANDING_V2_SYSTEM
    assert '"너는 왜 몰라?"' in UNDERSTANDING_V2_SYSTEM
    assert "move_subject=mormi_knowledge" in UNDERSTANDING_V2_SYSTEM
    assert '"너는 AI인데 그것도 몰라?"' in UNDERSTANDING_V2_SYSTEM
    assert "move_subject=mormi_ai_identity" in UNDERSTANDING_V2_SYSTEM
    assert '"너 AI인데 16,000원이잖아"' in UNDERSTANDING_V2_SYSTEM
    assert '"못 알려주겠는데?"' in UNDERSTANDING_V2_SYSTEM
    assert '"네가 해"' in UNDERSTANDING_V2_SYSTEM
    assert '"6000나누기 2는 3000이니까"' in UNDERSTANDING_V2_SYSTEM
    assert '"500+100=600"' in UNDERSTANDING_V2_SYSTEM
    assert '"2000곱하기 6이 12000이니까 6번 탈 때 같고, 7번부터 저렴해"' in (
        UNDERSTANDING_V2_SYSTEM
    )
    assert "operation 이름이 canonical graph와 다르다는 이유로" in (
        UNDERSTANDING_V2_SYSTEM
    )
    assert 'value_type=money의 unit은 통화 코드 "KRW" 또는 null만' in (
        UNDERSTANDING_V2_SYSTEM
    )


@pytest.mark.parametrize("unit", [None, "", "원", "원화", "₩", "￦", "KRW", "krw"])
def test_v2_provider_money_unit_normalizes_korean_won_surfaces(unit: str | None) -> None:
    claim = ModelFactUnderstandingClaimV2(
        claim_id="answer",
        target_id="per_person",
        claim_type="final_answer",
        evidence_span="3000원",
        verdict="correct",
        value_type="money",
        numeric_value=3_000,
        text_value=None,
        boolean_value=None,
        unit=unit,
        confidence=0.99,
    )

    value = _model_claim_value(claim)

    assert value.type == "money"
    assert value.amount == 3_000
    assert value.currency == "KRW"


def test_v2_provider_money_unit_preserves_other_iso_currency_codes() -> None:
    claim = ModelFactUnderstandingClaimV2(
        claim_id="answer",
        target_id="total",
        claim_type="final_answer",
        evidence_span="6 dollars",
        verdict="correct",
        value_type="money",
        numeric_value=6,
        text_value=None,
        boolean_value=None,
        unit="usd",
        confidence=0.99,
    )

    value = _model_claim_value(claim)

    assert value.currency == "USD"


def test_v2_understanding_prompt_does_not_turn_help_card_text_into_learning() -> None:
    assert "도움 카드에 보인 식이나 수를 질문하거나 그대로 인용한 것은" in (
        UNDERSTANDING_V2_SYSTEM
    )
    assert "모르미가 스스로 답이나 방법을 깨달았다고 판정하지" in (
        UNDERSTANDING_V2_SYSTEM
    )
    assert "conversation_move=request_mormi_answer" in UNDERSTANDING_V2_SYSTEM
    assert "지원 단계와 다음 질문은 서버가 결정한다" in UNDERSTANDING_V2_SYSTEM


def test_v2_speaker_prompts_obey_conversation_plan_and_fact_provenance() -> None:
    for rule in (
        "response_plan이 있으면 그 계획이 사회적 반응과 학습 복귀 방식의 최우선 계약이다",
        "explain_mormi_limit이면 왜 모르냐는 질문을 무시하지 말고",
        "explain_ai_role이면 AI라는 말을 피하지 말고",
        "decline_answer_and_ask이면 모르미가 대신 풀지 못한다는 사실만",
        "respond_refusal이면 아이의 거절을 복창·해석하거나",
        '"나 꼭 알고 싶은데..."처럼 모르미 자신의 궁금한 마음만',
        '"도움 카드가 나왔어" 또는 "어? 도움 카드가 나왔어"',
        '"나왔구나", "나왔네", "나왔군"처럼 관찰을 평가하는 말투는 쓰지 않는다',
        "response_mode와 관계없이 서버가 검수된 current_question을 뒤에 결정적으로 붙인다",
        "reask_targets·current_question을 반복하거나 도움을 다시 청하지 않는다",
        "allowed_facts.source=screen은 화면에서 볼 수 있는 사실일 뿐",
        "allowed_facts.source=child_verified만 아이가 알려 준 사실",
        "allowed_facts.source=jointly_derived는 함께 확인한 사실",
        "도움 카드는 allowed_facts의 source가 될 수 없다",
    ):
        assert rule in SPEAKER_V2_SYSTEM

    for rule in (
        "서버가 검수된 current_question을",
        "질문, 요청, 학습 복귀는 서버 몫이다",
        "explain_ai_role: AI라는 사실을 인정하되",
        "decline_answer_and_ask:",
        "respond_refusal:",
        '"나 꼭 알고 싶은데..."처럼 자신의 궁금함만 말한다',
        '"도움 카드가 나왔어" 또는',
        "본문·식·수·방법을 설명하거나 요약하지 않는다",
        "screen, child_verified, jointly_derived 출처를 섞지 않는다",
    ):
        assert rule in BRIDGE_SPEAKER_V2_SYSTEM

    assert "쉽고 따뜻한 반말만 사용한다" in SPEAKER_V2_SYSTEM
    assert '"-요", "-습니다"' in SPEAKER_V2_SYSTEM
    assert "쉽고 따뜻한 반말만 사용한다" in BRIDGE_SPEAKER_V2_SYSTEM


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
    assert request["model"] == gateway.settings.report_model
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


@pytest.mark.asyncio
async def test_classifier_uses_configured_medium_effort() -> None:
    expected = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.NO_RESPONSE,
        difficulty_class=DifficultyClass.UNKNOWN,
        confidence=0.9,
    )

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
    gateway = ClaudeGateway(Settings(anthropic_api_key=None, classifier_effort="medium"))
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    result = await gateway._request_classification("분류해 줘")

    assert result == expected
    assert messages.requests[0]["output_config"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_main_haiku_speaker_omits_sonnet_effort() -> None:
    context = speaker_context()
    expected = speaker_output(context.fallback_text, context)

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
    gateway = ClaudeGateway(Settings(anthropic_api_key=None, speaker_effort="low"))
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    result = await gateway.speak(context)

    assert result == expected
    assert messages.requests[0]["model"] == "claude-haiku-4-5-20251001"
    assert messages.requests[0]["temperature"] == 0.7
    assert "effort" not in messages.requests[0]["output_config"]


@pytest.mark.asyncio
async def test_overridden_sonnet_speaker_uses_configured_low_effort() -> None:
    context = speaker_context()
    expected = speaker_output(context.fallback_text, context)

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
    gateway = ClaudeGateway(
        Settings(
            anthropic_api_key=None,
            speaker_model="claude-sonnet-4-6",
            speaker_effort="low",
        )
    )
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    result = await gateway.speak(context)

    assert result == expected
    assert messages.requests[0]["temperature"] == 0.7
    assert messages.requests[0]["output_config"]["effort"] == "low"


@pytest.mark.asyncio
async def test_legacy_bridge_speaker_uses_conversational_temperature() -> None:
    context = speaker_context()
    expected = speaker_output(context.fallback_text, context)

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

    result = await gateway.bridge_speak(context)

    assert result == expected
    assert messages.requests[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_star_note_contextualizer_uses_dedicated_haiku_model() -> None:
    context = NoteContextualizationContext(
        skill_id="basic_addition",
        note_context="두 물건의 전체 값을 구하는 방법",
        source_fragments={"method": "2000원하고 900원을 더했어"},
        reviewed_facts={"total": "2900원"},
        allowed_numbers=["2000", "900", "2900"],
        fallback_text="2000원하고 900원을 더해서 2900원을 구했어.",
    )
    expected = NoteContextualizationOutput(
        text=context.fallback_text,
        source_slots_used=["method"],
        source_spans_used=[context.source_fragments["method"]],
        fact_refs_used=["total"],
        meaning_preserved=True,
        self_contained=True,
        introduced_math_content=False,
    )

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
    gateway = ClaudeGateway(
        Settings(
            anthropic_api_key=None,
            speaker_model="dialogue-haiku",
            star_note_model="note-haiku",
        )
    )
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    result = await gateway.contextualize_note(context)

    assert result == expected
    assert messages.requests[0]["model"] == gateway.settings.star_note_model
    assert messages.requests[0]["model"] == "note-haiku"
    assert messages.requests[0]["model"] != gateway.settings.speaker_model

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


def test_surface_wording_is_not_used_as_runtime_safety_classifier() -> None:
    context = speaker_context()
    for text in (
        "맞아! 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "그 부분은 기억했어. 나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
        "나는 아직 헷갈려... 도움 카드를 보고 다시 알려줄 수 있어?",
        "왜 오른쪽에 더 많다고 생각했어?",
    ):
        assert (
            validate_speaker_output(speaker_output(text, context), context, speaker_guard()) == text
        )


def test_speaker_runtime_tracks_focus_by_slot_ids_not_korean_copy() -> None:
    context = speaker_context()
    paraphrase = speaker_output(
        "아, 왼쪽이구나! 나는 이 줄이 왜 빠른지 아직 잘 모르겠어... 알려줄 수 있어?",
        context,
    )
    assert validate_speaker_output(paraphrase, context, speaker_guard()) == paraphrase.text

    wrong_focus = paraphrase.model_copy(update={"asked_slot_ids": ["answer"]})
    assert validate_speaker_output(wrong_focus, context, speaker_guard()) is None


def test_semantic_speaker_may_paraphrase_only_with_matching_contract() -> None:
    context = speaker_context().model_copy(
        update={"verification_policy": SpeakerVerificationPolicy.SEMANTIC}
    )
    output = speaker_output("나는 이 줄이 왜 더 빠른지 헷갈려... 알려줄 수 있어?", context)
    assert validate_speaker_output(output, context, speaker_guard()) is not None

    wrong_focus = output.model_copy(update={"asked_slot_ids": ["answer"]})
    assert validate_speaker_output(wrong_focus, context, speaker_guard()) is None


def test_help_card_visibility_is_a_prompt_policy_not_a_word_filter() -> None:
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

    assert validate_speaker_output(output, context, speaker_guard()) == output.text
    assert "help_card_visible=false이면 '카드', '도움 카드'" in SPEAKER_SYSTEM


def test_surface_words_are_not_semantic_contracts() -> None:
    l2_context = SpeakerContext(
        dialogue_act="reduce_expression_load",
        expression_level=ExpressionLevel.L2,
        required_question="어느 쪽인지 골라서 알려줄 수 있어?",
        required_slot_ids=["answer"],
        required_slot_descriptions={"answer": "선택할 답"},
        fallback_text="그럼 혹시 여기서 골라서 알려줄 수 있어?",
    )
    legitimate_scenario = "사탕 6개를 친구랑 같이 나눠 먹으려면 한 명이 몇 개씩 먹으면 돼?"
    assert validate_speaker_output(
        speaker_output(legitimate_scenario, l2_context),
        l2_context,
        speaker_guard(),
    ) == legitimate_scenario

    help_seeking_question = "이거는 왜 정답이라고 하는 거야? 나는 아직 잘 모르겠어."
    assert validate_speaker_output(
        speaker_output(help_seeking_question, l2_context),
        l2_context,
        speaker_guard(),
    ) == help_seeking_question

    assert (
        '공동 수행을 뜻하는 "같이 해보자"는 L0에서만 사용한다.' in SPEAKER_SYSTEM
    )


def test_multiple_question_marks_are_allowed_for_one_semantic_focus() -> None:
    text = "그런데 ‘차근차근’은 어떻게 세는 거야? 조금 더 알려줄 수 있어?"
    context = SpeakerContext(
        dialogue_act="clarify_child_expression",
        required_question="차근차근 세는 방법을 조금 더 알려줄 수 있어?",
        required_slot_ids=["method"],
        required_slot_descriptions={"method": "차근차근 세는 방법"},
        fallback_text="차근차근 세는 방법을 조금 더 알려줄 수 있어?",
    )

    assert validate_speaker_output(
        speaker_output(text, context),
        context,
        speaker_guard(),
    ) == text


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


def test_repetition_and_reframing_are_prompt_policies_not_copy_filters() -> None:
    context = SpeakerContext(
        dialogue_act="acknowledge_unstructured_partial",
        previous_question="어떻게 세는지 알려주면 안 될까?",
        required_question="어떻게 세는지 알려주면 안 될까?",
        fallback_text="내가 또 똑같이 물었네... 같이 골라 볼까?",
    )
    repeated = speaker_output("어떻게 세는지 알려주면 안 될까?", context)
    assert validate_speaker_output(repeated, context, speaker_guard()) == repeated.text

    support_context = SpeakerContext(
        dialogue_act="accept_help_request",
        must_reframe=True,
        previous_question="어떻게 세는지 알려주면 안 될까?",
        required_question="어떻게 세는지 알려주면 안 될까?",
        required_slot_ids=["tracking"],
        required_slot_descriptions={"tracking": "빠뜨리지 않고 세는 방법"},
        verification_policy=SpeakerVerificationPolicy.SEMANTIC,
        fallback_text="오, 도움 카드가 나왔어! 이걸 보고 다시 알려줄래?",
    )
    support_output = SpeakerOutput(
        text="도움 카드가 나왔네. 어떻게 세는지 알려주면 안 될까?",
        dialogue_act=support_context.dialogue_act,
        asked_slot_ids=["tracking"],
    )
    assert (
        validate_speaker_output(support_output, support_context, speaker_guard())
        == support_output.text
    )
    assert "must_reframe=true이면" in SPEAKER_SYSTEM


def test_speaker_rejects_empty_or_structurally_inconsistent_output() -> None:
    context = speaker_context()
    valid = speaker_output(
        "아, 왼쪽이구나! 나는 왜 이 줄이 더 빠른지 아직 모르겠어... 알려줄 수 있어?",
        context,
    )

    assert validate_speaker_output(valid, context, speaker_guard()) == valid.text
    assert (
        validate_speaker_output(valid.model_copy(update={"text": "  "}), context, speaker_guard())
        is None
    )
    assert (
        validate_speaker_output(
            valid.model_copy(update={"dialogue_act": "complete"}),
            context,
            speaker_guard(),
        )
        is None
    )
    assert (
        validate_speaker_output(
            valid.model_copy(update={"asked_slot_ids": ["reason", "reason"]}),
            context,
            speaker_guard(),
        )
        is None
    )
    assert (
        validate_speaker_output(
            valid.model_copy(update={"used_verified_slots": ["answer"]}),
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

        assert payload["unresolved_required_slots"] == list(task.required_slots)
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
                assert slots[slot_id]["claim_contract"]["value"] == "actual_child_claim_normalized"
                assert slots[slot_id]["claim_contract"]["supported"] is None
                assert slots[slot_id]["claim_contract"]["evidence_span"] == "exact_child_substring"
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
            "아이가 실제로 주장한 값" in instruction for instruction in payload["instructions"]
        )
        assert any(
            "서로 배타적인 분류가 아니다" in instruction
            for instruction in payload["instructions"]
        )
        assert any("2개씩 4묶음" in instruction for instruction in payload["instructions"])


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
        "required_question은 그대로 복사할 문구가 아니라",
        "같은 초점을\n  자연스럽게 이어 묻는 문장이라면 물음표가 두 개여도 된다.",
        "특정 단어가 들어갔다는 이유만으로 문장의 뜻을 단정하지 않는다.",
        "must_reframe=true이면 직전 질문 앞에 말만 덧붙이지 말고",
    ):
        assert rule in SPEAKER_SYSTEM

    assert "생략된 주어나 대상을 검증된 장면 사실로 명확히 한다." in (
        NOTE_CONTEXTUALIZER_SYSTEM
    )
    assert (
        "예: '둘이', '오른쪽', '그거'를 검증된 실제 대상으로 바꾼다."
        in NOTE_CONTEXTUALIZER_SYSTEM
    )


def test_teacher_style_is_forbidden_by_prompt_not_surface_regex() -> None:
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
            validate_speaker_output(speaker_output(text, context), context, speaker_guard()) == text
        )

    assert "아이에게 명령하거나 퀴즈를 내지 않는다." in SPEAKER_SYSTEM
    assert "'왜 그렇게 생각했어?'" in SPEAKER_SYSTEM
    assert "처럼 도움을 청한다." in SPEAKER_SYSTEM


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
    """The primary pass exposes ambiguity without paying for a second model call."""

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


def test_social_bridge_style_is_prompt_policy_while_contract_stays_structural() -> None:
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
    assert validate_speaker_output(rewarding, context, guard) == rewarding.text
    assert "새 농담·놀이·화제를 만들지 않고" in SPEAKER_SYSTEM

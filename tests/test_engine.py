from __future__ import annotations

import pytest
from conftest import FakeGateway

from mormi_api.content import (
    CHANGE_TASK_ID,
    HOME_TEACH_TASK_ID,
    TOTAL_CALC_TASK_ID,
    calculation_task,
    create_scenario_data,
)
from mormi_api.engine import ConversationEngine
from mormi_api.schemas import (
    ArithmeticClaim,
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    HintLevel,
    InteractionIntent,
    NoteContextualizationContext,
    NoteContextualizationOutput,
    ResponseCategory,
    SafetyCategory,
    SceneType,
    SemanticAssessment,
    SessionState,
    SlotClaim,
    TaskRelation,
    UtteranceAnalysis,
)


class ContextualizedNoteGateway(FakeGateway):
    async def contextualize_note(
        self,
        context: NoteContextualizationContext,
    ) -> NoteContextualizationOutput:
        return NoteContextualizationOutput(
            text="5에서 3을 빼면 2 차이가 나니까 5가 훨씬 더 커.",
            source_slots_used=list(context.source_fragments),
            source_spans_used=list(context.source_fragments.values()),
            fact_refs_used=["note_context", "slot:answer"],
            meaning_preserved=True,
            self_contained=True,
            introduced_math_content=False,
        )


class InventedNoteGateway(FakeGateway):
    async def contextualize_note(
        self,
        context: NoteContextualizationContext,
    ) -> NoteContextualizationOutput:
        return NoteContextualizationOutput(
            # 9 is not present in either the child evidence or reviewed context.
            text="9에서 3을 빼면 6이니까 9가 더 커.",
            source_slots_used=list(context.source_fragments),
            source_spans_used=list(context.source_fragments.values()),
            fact_refs_used=["note_context"],
            meaning_preserved=True,
            self_contained=True,
            introduced_math_content=False,
        )


def test_complete_mormi_copy_keeps_a_natural_sentence_beyond_soft_target() -> None:
    text = (
        "그런데 ‘2000원을 먼저 내고 거슬러받으면’이 어떻게 하는 건지 모르겠어... "
        "조금만 더 알려줄래?"
    )

    rendered = ConversationEngine._complete_mormi_text(text)

    assert len(text) > 50
    assert rendered == text
    assert rendered.endswith("?")


def test_oversized_composition_is_preserved_without_slicing() -> None:
    question = "나는 두 금액을 어떤 계산으로 합치는지 아직 헷갈려... 알려줄 수 있어?"
    oversized = f"아, 네가 말한 내용은 잘 들었어. 그런데 아직 조금 더 알고 싶어. {question}"

    rendered = ConversationEngine._complete_mormi_text(oversized)

    assert rendered == oversized
    assert rendered.endswith("?")


def _number_comparison_state() -> SessionState:
    scenario_data = create_scenario_data(
        "home_teach",
        curriculum_session_id="number-compare",
        skill_id="number-compare",
    )
    return SessionState(
        learner_id=1,
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=[HOME_TEACH_TASK_ID],
        task_start_levels={HOME_TEACH_TASK_ID: ExpressionLevel.L4},
        scenario_data=scenario_data,
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )


@pytest.mark.asyncio
async def test_clear_current_task_answer_skips_sonnet_adjudication() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        task_relation=TaskRelation.CURRENT_TASK,
        answer_status=SemanticAssessment.COMPLETE,
        reason_status=SemanticAssessment.MISSING,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
            )
        ],
        confidence=0.95,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="45a91271-8be9-46e4-a7fd-f729766977f1",
            type="text",
            text="오른쪽",
        ),
        initial.mormi.text,
    )

    assert gateway.classify_calls == 1
    assert gateway.adjudicate_calls == 0
    assert gateway.bridge_speak_calls == 0


@pytest.mark.asyncio
async def test_high_impact_ambiguity_uses_sonnet_adjudication_once() -> None:
    primary = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.UNKNOWN,
        task_relation=TaskRelation.CURRENT_TASK,
        answer_status=SemanticAssessment.UNCERTAIN,
        reason_status=SemanticAssessment.MISSING,
        needs_adjudication=True,
        adjudication_reason="짧은 위치 답을 현재 장면과 대조해야 함",
        confidence=0.72,
    )
    adjudicated = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        task_relation=TaskRelation.CURRENT_TASK,
        answer_status=SemanticAssessment.COMPLETE,
        reason_status=SemanticAssessment.MISSING,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
            )
        ],
        confidence=0.96,
    )
    gateway = FakeGateway([primary], [adjudicated])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    _, effective, _ = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="9eb209cb-2419-43f3-8d25-f287476964f0",
            type="text",
            text="오른쪽",
        ),
        initial.mormi.text,
    )

    assert gateway.classify_calls == 1
    assert gateway.adjudicate_calls == 1
    assert gateway.bridge_speak_calls == 0
    assert effective.response_category is ResponseCategory.CORRECT_PARTIAL
    assert effective.claims[0].value == "오른쪽"


@pytest.mark.asyncio
async def test_safe_meta_utterance_uses_fast_dialogue_bridge() -> None:
    child_text = "너 알면서 일부러 물어보지?"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.AUTHENTICITY_CHALLENGE,
        social_grounding_span=child_text,
        confidence=0.94,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="90cbd679-5ec6-4742-af6e-2a761886d02c",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert gateway.classify_calls == 1
    assert gateway.adjudicate_calls == 0
    assert gateway.bridge_speak_calls == 1
    assert next_state.verified_slots == {}
    assert turn.status.value == "active"


@pytest.mark.asyncio
async def test_code_upgrades_classifier_partial_when_all_current_slots_are_verified() -> None:
    child_text = "오른쪽이 더 많구 둘을 빼면 2 차이가 나서 오른쪽이 더 커"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽이 더 많구",
            ),
            SlotClaim(
                slot_id="reason",
                factual=True,
                supported=True,
                support_confidence=0.98,
                evidence_span="둘을 빼면 2 차이가 나서 오른쪽이 더 커",
            ),
        ],
        confidence=0.92,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="4109e86e-98e9-4dcb-9964-a4990add6720",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert effective_analysis.response_category is ResponseCategory.CORRECT_FULL
    assert next_state.status.value == "completed"
    assert turn.status.value == "completed"
    assert turn.note_update is not None
    assert "둘을 빼면 2 차이가 나서 오른쪽이 더 커" in turn.note_update.text


@pytest.mark.asyncio
async def test_direct_note_resolves_scene_references_with_reviewed_context() -> None:
    child_text = "오른쪽이 더 많구 둘이 빼다보면 2 차이가 나기때문에 오른쪽이 훨씬 더 커"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽이 더 많구",
            ),
            SlotClaim(
                slot_id="reason",
                factual=True,
                supported=True,
                support_confidence=0.98,
                evidence_span="둘이 빼다보면 2 차이가 나기때문에 오른쪽이 훨씬 더 커",
            ),
        ],
        confidence=0.95,
    )
    engine = ConversationEngine(ContextualizedNoteGateway([analysis]))  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="6109e86e-98e9-4dcb-9964-a4990add6720",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert turn.note_update is not None
    assert turn.note_update.text == "5에서 3을 빼면 2 차이가 나니까 5가 훨씬 더 커."
    assert next_state.child_note_evidence == {
        "answer": "오른쪽이 더 많구",
        "reason": "둘이 빼다보면 2 차이가 나기때문에 오른쪽이 훨씬 더 커",
    }


@pytest.mark.asyncio
async def test_note_contextualizer_cannot_introduce_unknown_numbers() -> None:
    child_text = "오른쪽이 더 많고 둘을 빼면 2 차이가 나"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽이 더 많고",
            ),
            SlotClaim(
                slot_id="reason",
                factual=True,
                supported=True,
                support_confidence=0.98,
                evidence_span="둘을 빼면 2 차이가 나",
            ),
        ],
        confidence=0.95,
    )
    engine = ConversationEngine(InventedNoteGateway([analysis]))  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    _, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="7109e86e-98e9-4dcb-9964-a4990add6720",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert turn.note_update is not None
    assert "9" not in turn.note_update.text
    assert child_text in turn.note_update.text


@pytest.mark.asyncio
async def test_code_downgrades_classifier_full_when_only_one_current_slot_is_verified() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
            )
        ],
        confidence=0.99,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="5209e86e-98e9-4dcb-9964-a4990add6720",
            type="text",
            text="오른쪽",
        ),
        initial.mormi.text,
    )

    assert effective_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert next_state.verified_slots["answer"] == "오른쪽"
    assert turn.input.target_slots == ["reason"]


@pytest.mark.asyncio
async def test_verified_slots_recover_a_valid_answer_from_classifier_concept_error() -> None:
    child_text = "오른쪽이 더 많아. 왼쪽은 3개고 오른쪽은 5개니까"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        difficulty_class=DifficultyClass.CONCEPT,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽이 더 많아",
            ),
            SlotClaim(
                slot_id="reason",
                factual=True,
                supported=True,
                support_confidence=0.97,
                evidence_span="왼쪽은 3개고 오른쪽은 5개니까",
            ),
        ],
        misconception_tag="reversed_comparison",
        bottleneck="concept",
        confidence=0.85,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective_analysis, _ = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="6309e86e-98e9-4dcb-9964-a4990add6720",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert effective_analysis.response_category is ResponseCategory.CORRECT_FULL
    assert effective_analysis.difficulty_class is DifficultyClass.UNKNOWN
    assert effective_analysis.misconception_tag is None
    assert next_state.status.value == "completed"


@pytest.mark.asyncio
async def test_queue_keeps_l4_after_independent_counts_and_asks_the_comparison_next() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="left_count",
                value=3,
                factual=True,
                evidence_span="왼쪽 세 명",
            ),
            SlotClaim(
                slot_id="right_count",
                value=5,
                factual=True,
                evidence_span="오른쪽 다섯 명",
            ),
        ],
        confidence=1,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_queue_demo",
        task_ids=["cafe_queue"],
        task_start_levels={"cafe_queue": ExpressionLevel.L4},
        scenario_data={"left_count": 3, "right_count": 5},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="6156e551-25dc-4a46-b6d2-f78692c8c567",
            type="text",
            text="왼쪽 세 명, 오른쪽 다섯 명",
        ),
        initial.mormi.text,
    )

    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert turn.input.target_slots == ["final_choice", "reason"]
    assert "어느 줄" in turn.mormi.text
    assert turn.help_card is None


@pytest.mark.asyncio
async def test_partial_answer_preserves_fact_and_asks_only_missing_slot() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[SlotClaim(slot_id="final_choice", value="left", factual=True)],
        confidence=1,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_queue_demo",
        task_ids=["cafe_queue"],
        task_start_levels={"cafe_queue": ExpressionLevel.L4},
        scenario_data={"left_count": 3, "right_count": 5},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
        verified_slots={"left_count": 3, "right_count": 5},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="0e3fc94b-7cc7-4d1f-843c-8e0686543769",
            type="text",
            text="왼쪽",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots["final_choice"] == "left"
    assert next_state.expression_level is ExpressionLevel.L4
    assert turn.input.target_slots == ["reason"]
    assert "왜" in turn.mormi.text


@pytest.mark.asyncio
async def test_cafe_result_only_is_remembered_and_only_method_is_asked_next() -> None:
    """A natural uncommaed amount must not repeat the combined L4 question."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="result",
                value="6000원이야",
                factual=True,
                evidence_span="6000원이야",
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_menu_total",
        task_ids=[TOTAL_CALC_TASK_ID],
        task_start_levels={TOTAL_CALC_TASK_ID: ExpressionLevel.L4},
        scenario_data={
            "menu_items": [
                {"id": "milk", "name": "우유", "price": 2000},
                {"id": "juice", "name": "딸기주스", "price": 4000},
            ],
            "mormi_menu_id": "milk",
            "child_menu_id": "juice",
        },
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="f69ca63d-5e06-4e1a-8212-415eb858c8e0",
            type="text",
            text="6000원이야",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"result": 6000}
    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.entry_phase.value == "awaiting_targeted_followup"
    assert turn.input.target_slots == ["operation"]
    assert turn.mormi.text != initial.mormi.text
    assert "6,000원이구나" in turn.mormi.text
    assert "어떻게 계산한 건지" in turn.mormi.text


@pytest.mark.asyncio
async def test_numeric_claim_cannot_replace_child_wrong_amount_with_expected() -> None:
    """A model-authored expected value cannot overrule the child's 1,800 claim."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="operation",
                value="addition",
                factual=True,
                evidence_span="더하면",
            ),
            # Simulate the historical classifier bug: the child said 1,800,
            # but the model copied reviewed expected=1,700 into the claim.
            SlotClaim(
                slot_id="result",
                value=1700,
                factual=True,
                evidence_span="1800원이야",
                interpretation_confidence=1,
            ),
        ],
        confidence=1,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_menu_total",
        task_ids=[TOTAL_CALC_TASK_ID],
        task_start_levels={TOTAL_CALC_TASK_ID: ExpressionLevel.L4},
        scenario_data={
            "menu_items": [
                {"id": "first", "name": "첫 메뉴", "price": 1200},
                {"id": "second", "name": "둘째 메뉴", "price": 500},
            ],
            "mormi_menu_id": "first",
            "child_menu_id": "second",
        },
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="a86a1a11-dd76-49b0-a831-e746157cb356",
            type="text",
            text="1200원에 500원을 더하면 1800원이야",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"operation": "addition"}
    assert next_state.status.value == "active"
    assert turn.status.value == "active"
    assert effective_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert effective_analysis.difficulty_class is DifficultyClass.CONCEPT
    assert effective_analysis.claims[1].factual is False


def test_wrong_subtraction_cannot_satisfy_result_or_method_contract() -> None:
    task = calculation_task(
        task_id="subtraction_truth_gate",
        title="거스름돈 계산",
        skill_id="subtraction",
        left=5000,
        right=2400,
        operation="subtraction",
        result=2600,
    )
    child_text = "5000원에서 2400원을 빼면 2700원이야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        claims=[
            SlotClaim(
                slot_id="result",
                value=2700,
                factual=True,
                evidence_span="2700원이야",
                interpretation_confidence=1,
            ),
            SlotClaim(
                slot_id="method",
                factual=True,
                supported=True,
                support_confidence=1,
                evidence_span=child_text,
                ),
            ],
        arithmetic_claims=[
            ArithmeticClaim(
                left=5000,
                right=2400,
                operation="subtraction",
                result=2700,
                evidence_span=child_text,
                related_slot_ids=["method"],
                interpretation_confidence=1,
            )
        ],
    )

    ConversationEngine._invalidate_false_structured_arithmetic(task, child_text, analysis)
    ConversationEngine._ground_text_numeric_claims(task, child_text, analysis)
    verified = task.validated_slot_claims(analysis.claims)
    verified = ConversationEngine._filter_text_explanation_claims(
        task,
        child_text,
        analysis,
        verified,
    )

    assert "result" not in verified
    assert "method" not in verified
    assert analysis.claims[1].supported is False


@pytest.mark.parametrize(
    ("child_text", "confidence", "accepted"),
    [
        ("더하면 천칠백원이야", 0.7, True),
        ("더하면 천칠배건이야", 0.95, True),
        ("더하면 천칠배건이야", 0.4, False),
    ],
)
def test_korean_amounts_use_evidence_and_only_ambiguous_typos_need_high_confidence(
    child_text: str,
    confidence: float,
    accepted: bool,
) -> None:
    task = calculation_task(
        task_id="korean_amount_truth_gate",
        title="메뉴값 계산",
        skill_id="addition",
        left=1200,
        right=500,
        operation="addition",
        result=1700,
    )
    evidence = child_text.removeprefix("더하면 ")
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        claims=[
            SlotClaim(
                slot_id="result",
                value=1700,
                factual=True,
                evidence_span=evidence,
                interpretation_confidence=confidence,
            )
        ],
    )

    ConversationEngine._ground_text_numeric_claims(task, child_text, analysis)
    verified = task.validated_slot_claims(analysis.claims)

    assert (verified.get("result") == 1700) is accepted


@pytest.mark.asyncio
async def test_cafe_change_result_only_uses_the_same_targeted_followup_policy() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="result",
                value="5,500원이에요",
                factual=True,
                evidence_span="5500원이야",
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_change",
        task_ids=[CHANGE_TASK_ID],
        task_start_levels={CHANGE_TASK_ID: ExpressionLevel.L4},
        scenario_data={
            "menu_items": [
                {"id": "cake", "name": "딸기케이크", "price": 4500},
                {"id": "milk", "name": "우유", "price": 2000},
            ],
            "mormi_menu_id": "cake",
        },
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="045ec1d1-cc13-478a-995c-aa73c02bed01",
            type="text",
            text="5500원이야",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"result": 5500}
    assert next_state.expression_level is ExpressionLevel.L4
    assert turn.input.target_slots == ["operation"]
    assert "5,500원이구나" in turn.mormi.text
    assert "어떻게 계산한 건지" in turn.mormi.text


@pytest.mark.asyncio
async def test_cafe_operation_only_is_remembered_and_only_amount_is_asked_next() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="operation",
                value="addition",
                factual=True,
                evidence_span="더하면 돼",
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_menu_total",
        task_ids=[TOTAL_CALC_TASK_ID],
        task_start_levels={TOTAL_CALC_TASK_ID: ExpressionLevel.L4},
        scenario_data={
            "menu_items": [
                {"id": "milk", "name": "우유", "price": 2000},
                {"id": "juice", "name": "딸기주스", "price": 4000},
            ],
            "mormi_menu_id": "milk",
            "child_menu_id": "juice",
        },
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="e656002a-03e4-4900-8474-dc6d8297996e",
            type="text",
            text="더하면 돼",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"operation": "addition"}
    assert next_state.expression_level is ExpressionLevel.L4
    assert turn.input.target_slots == ["result"]
    assert "두 메뉴는 모두 얼마야" in turn.mormi.text


@pytest.mark.asyncio
async def test_help_request_lowers_expression_and_opens_help_card() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.HELP_REQUEST,
        difficulty_class=DifficultyClass.EXPRESSION,
        confidence=1,
    )
    engine = ConversationEngine(FakeGateway([analysis]), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_queue_demo",
        task_ids=["cafe_queue"],
        task_start_levels={"cafe_queue": ExpressionLevel.L4},
        scenario_data={"left_count": 3, "right_count": 5},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="48893134-202a-45ff-b1bb-e1e652fdb011",
            type="text",
            text="잘 모르겠어",
        ),
        initial.mormi.text,
    )

    assert next_state.expression_level is ExpressionLevel.L3
    assert next_state.hint_level is HintLevel.H1
    assert turn.help_card is not None
    assert turn.help_card.auto_open is True


@pytest.mark.asyncio
async def test_related_vague_policy_is_shared_by_cafe_reason_questions() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.RELATED_VAGUE,
        difficulty_class=DifficultyClass.EXPRESSION,
        grounding_span="그냥 골라봐",
        confidence=1,
    )
    engine = ConversationEngine(
        FakeGateway([analysis]),  # type: ignore[arg-type]
        show_internal_pedagogy=True,
    )
    state = SessionState(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_queue_demo",
        task_ids=["cafe_queue"],
        task_start_levels={"cafe_queue": ExpressionLevel.L4},
        scenario_data={"left_count": 3, "right_count": 5},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
        verified_slots={
            "left_count": 3,
            "right_count": 5,
            "final_choice": "left",
        },
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="72f3901d-f57a-461f-a932-2d16a11b4859",
            type="text",
            text="그냥 골라봐",
        ),
        initial.mormi.text,
    )

    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert next_state.vague_clarifications == 1
    assert turn.help_card is None
    assert "그냥 골라봐" in turn.mormi.text
    assert "조금만 더 알려줄래" in turn.mormi.text

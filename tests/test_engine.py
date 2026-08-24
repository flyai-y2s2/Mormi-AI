from __future__ import annotations

import pytest
from conftest import FakeGateway

from mormi_api.content import (
    CHANGE_TASK_ID,
    HOME_TEACH_TASK_ID,
    TOTAL_CALC_TASK_ID,
    TOTAL_MENU_PICK_TASK_ID,
    calculation_task,
    create_scenario_data,
    get_task,
)
from mormi_api.engine import ConversationEngine
from mormi_api.schemas import (
    ArithmeticClaim,
    CafeMenuItem,
    CafeSessionContext,
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
from mormi_api.settings import Settings


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


def test_natural_speaker_timeouts_are_bounded_but_not_overly_aggressive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MORMI_CLASSIFIER_MODEL", raising=False)
    monkeypatch.delenv("MORMI_BRIDGE_MODEL", raising=False)
    monkeypatch.delenv("MORMI_SPEAKER_MODEL", raising=False)
    monkeypatch.delenv("MORMI_SPEAKER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MORMI_BRIDGE_TIMEOUT_SECONDS", raising=False)

    settings = Settings(_env_file=None)
    engine = ConversationEngine(FakeGateway([]))  # type: ignore[arg-type]

    assert settings.classifier_model == "claude-sonnet-4-6"
    assert settings.bridge_model == "claude-haiku-4-5-20251001"
    assert settings.speaker_model == "claude-sonnet-4-6"
    assert settings.speaker_timeout_seconds == 10.0
    assert settings.bridge_timeout_seconds == 4.0
    assert engine.speaker_timeout_seconds == 10.0
    assert engine.bridge_timeout_seconds == 4.0


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


def _addition_state() -> SessionState:
    cafe_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="juice", name="주스", price=700),
            CafeMenuItem(id="bread", name="빵", price=500),
        ],
        mormi_menu_id="juice",
    )
    scenario_data = create_scenario_data("cafe_menu_total", cafe_context)
    scenario_data["child_menu_id"] = "bread"
    return SessionState(
        learner_id=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_menu_total",
        task_ids=[TOTAL_CALC_TASK_ID],
        task_start_levels={TOTAL_CALC_TASK_ID: ExpressionLevel.L4},
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
async def test_primary_sonnet_result_is_authoritative_without_second_adjudicator() -> None:
    primary = UtteranceAnalysis(
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
        needs_adjudication=True,
        adjudication_reason="구버전 필드가 잘못 채워짐",
        confidence=0.9,
    )
    gateway = FakeGateway([primary])
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
    assert gateway.adjudicate_calls == 0
    assert gateway.bridge_speak_calls == 0
    assert effective.response_category is ResponseCategory.CORRECT_PARTIAL
    assert effective.claims[0].value == "오른쪽"
    assert effective.needs_adjudication is False
    assert effective.adjudication_reason == ""


@pytest.mark.asyncio
async def test_safe_meta_utterance_uses_fast_dialogue_bridge() -> None:
    child_text = "너 알면서 일부러 물어보지?"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.AUTHENTICITY_CHALLENGE,
        conversation_only=True,
        conversation_summary="모르미가 정말 모르는지 의심함",
        bridge_reply="나 진짜 몰라서 그래... 어느 쪽이 더 많고 왜 그런지 알려줄 수 있어?",
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
    assert turn.mormi.text
    assert turn.mormi.text != analysis.bridge_reply


@pytest.mark.asyncio
async def test_open_set_task_comment_uses_one_call_bridge_without_enum_dependency() -> None:
    child_text = "그건 너무 쉽지"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_TASK,
        interaction_intent=InteractionIntent.NONE,
        conversation_only=True,
        conversation_summary="현재 과제가 쉽다고 말함",
        bridge_reply="오, 그렇게 느꼈구나. 그럼 어느 쪽이 더 많고 왜 그런지 알려줄 수 있어?",
        social_grounding_span=child_text,
        confidence=0.96,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="d97f5815-30e5-49f9-9076-a589c870d1e8",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert gateway.classify_calls == 1
    assert gateway.adjudicate_calls == 0
    assert gateway.bridge_speak_calls == 1
    assert effective.task_relation is TaskRelation.META_ABOUT_TASK
    assert next_state.verified_slots == {}
    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert turn.note_update is None
    assert "어떻게 하는 건지 모르겠어" not in turn.mormi.text
    assert turn.mormi.text
    assert turn.mormi.text != analysis.bridge_reply


@pytest.mark.asyncio
async def test_conversation_only_flag_cannot_discard_a_learning_claim() -> None:
    """A contradictory open-set flag must not bypass educational evidence."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        task_relation=TaskRelation.UNKNOWN,
        interaction_intent=InteractionIntent.NONE,
        conversation_only=True,
        conversation_summary="분류기가 대화 전용이라고 잘못 표시함",
        bridge_reply="그렇구나. 그럼 다시 알려줄 수 있어?",
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
                interpretation_confidence=1,
            )
        ],
        confidence=0.9,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, _ = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="a0ab4301-7d45-45be-99a2-f56505323bd2",
            type="text",
            text="오른쪽",
        ),
        initial.mormi.text,
    )

    assert gateway.bridge_speak_calls == 0
    assert next_state.verified_slots.get("answer") == "오른쪽"


@pytest.mark.asyncio
async def test_meta_turn_cannot_promote_model_invented_operation_or_result() -> None:
    """Reviewed answers in the prompt are not evidence that the child taught them."""

    child_text = "내가 왜 알려줘야 돼?"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.REFUSAL,
        conversation_only=False,
        claims=[
            SlotClaim(
                slot_id="operation",
                value="addition",
                factual=True,
                evidence_span=child_text,
            ),
            SlotClaim(
                slot_id="result",
                value=1200,
                factual=True,
                evidence_span=child_text,
            ),
        ],
        confidence=0.95,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _addition_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="cb2b32e5-aed4-4727-9df1-6f40fd6ecad8",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert effective.conversation_only is True
    assert effective.claims == []
    assert effective.arithmetic_claims == []
    assert next_state.verified_slots == {}
    assert gateway.bridge_speak_calls == 1
    assert gateway.speaker_contexts == []
    assert gateway.bridge_contexts[0].verified_facts == {}
    assert gateway.note_contexts == []
    assert turn.note_update is None


@pytest.mark.asyncio
async def test_mixed_meta_turn_preserves_only_child_grounded_learning_clause() -> None:
    """A mixed refusal may teach a fact, but only from its exact learning clause."""

    child_text = "왜 알려줘야 돼? 그래도 둘을 더하면 돼"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.REFUSAL,
        conversation_only=True,
        claims=[
            SlotClaim(
                slot_id="operation",
                value="addition",
                factual=True,
                evidence_span="둘을 더하면 돼",
            ),
            SlotClaim(
                slot_id="result",
                value=1200,
                factual=True,
                evidence_span=child_text,
            ),
        ],
        confidence=0.95,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _addition_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective, _ = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="dd17ca43-7cab-477e-9313-1655fe261250",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert effective.conversation_only is False
    assert [(claim.slot_id, claim.value) for claim in effective.claims] == [
        ("operation", "addition")
    ]
    assert next_state.verified_slots == {"operation": "addition"}
    assert gateway.bridge_speak_calls == 0
    assert len(gateway.speaker_contexts) == 1
    assert set(gateway.speaker_contexts[0].verified_facts) == {"operation"}
    assert "result" not in gateway.speaker_contexts[0].verified_facts


@pytest.mark.asyncio
async def test_off_topic_turn_cannot_promote_fabricated_closed_answer() -> None:
    child_text = "오늘 급식 뭐야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.OFF_TOPIC,
        interaction_intent=InteractionIntent.NONE,
        conversation_only=False,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            )
        ],
        confidence=0.95,
    )
    gateway = FakeGateway([analysis])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="e9358d36-fb82-4c4b-ab84-73813b5ddf63",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert effective.conversation_only is True
    assert effective.claims == []
    assert next_state.verified_slots == {}
    assert gateway.bridge_speak_calls == 1
    assert gateway.bridge_contexts[0].verified_facts == {}
    assert gateway.note_contexts == []
    assert turn.note_update is None


def test_menu_selection_transition_is_not_labelled_as_teaching() -> None:
    cafe_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="juice", name="주스", price=700),
            CafeMenuItem(id="bread", name="빵", price=500),
        ],
        mormi_menu_id="juice",
    )
    scenario_data = create_scenario_data(
        "cafe_menu_total",
        cafe_context,
    )
    scenario_data["child_menu_id"] = "bread"
    task = get_task(TOTAL_CALC_TASK_ID, scenario_data)
    # The task being completed is the preceding menu-selection task. Its
    # reviewed transition must describe a choice, not a teaching success.
    selection_task = get_task(TOTAL_MENU_PICK_TASK_ID, scenario_data)

    rendered = ConversationEngine._task_transition_then_question(
        selection_task,
        "",
        task.steps[ExpressionLevel.L4][0].prompt,
    )

    assert rendered.startswith("네 메뉴도 골랐구나.")
    assert "네가 알려줘서" not in rendered


def test_task_transition_without_reviewed_copy_does_not_add_generic_acknowledgement() -> None:
    cafe_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="juice", name="주스", price=700),
            CafeMenuItem(id="bread", name="빵", price=500),
        ],
        mormi_menu_id="juice",
    )
    scenario_data = create_scenario_data("cafe_menu_total", cafe_context)
    task = get_task(TOTAL_CALC_TASK_ID, scenario_data).model_copy(update={"transition_text": None})
    question = task.steps[ExpressionLevel.L4][0].prompt

    rendered = ConversationEngine._task_transition_then_question(
        task,
        "아이가 실제로 알려준 내용",
        question,
    )

    assert rendered == question
    assert "아, 그렇구나" not in rendered
    assert "응, 알겠어" not in rendered


def test_ladder_fallback_does_not_prepend_generic_excuse() -> None:
    scenario_data = create_scenario_data(
        "cafe_menu_total",
        CafeSessionContext(
            menu_items=[
                CafeMenuItem(id="juice", name="주스", price=700),
                CafeMenuItem(id="bread", name="빵", price=500),
            ],
            mormi_menu_id="juice",
        ),
    )
    task = get_task(TOTAL_CALC_TASK_ID, scenario_data)
    state = SessionState(
        learner_id=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_menu_total",
        task_ids=[TOTAL_CALC_TASK_ID],
        scenario_data=scenario_data,
        expression_level=ExpressionLevel.L3,
        task_start_level=ExpressionLevel.L4,
        task_start_levels={TOTAL_CALC_TASK_ID: ExpressionLevel.L4},
    )

    rendered = ConversationEngine._smooth_ladder_fallback(state, task)

    assert rendered == task.step_for(ExpressionLevel.L3, {}).prompt
    assert "한꺼번에 많이" not in rendered
    assert "말로만 들으려니" not in rendered


def test_deterministic_acknowledgement_uses_the_verified_count_not_example_copy() -> None:
    scenario_data = create_scenario_data(
        "cafe_menu_total",
        CafeSessionContext(
            menu_items=[
                CafeMenuItem(id="juice", name="주스", price=700),
                CafeMenuItem(id="bread", name="빵", price=500),
            ],
            mormi_menu_id="juice",
        ),
    )
    task = get_task(TOTAL_CALC_TASK_ID, scenario_data).model_copy(
        update={"skill_id": "number-count"}
    )

    rendered = ConversationEngine._younger_sibling_acknowledgement(
        task,
        {"answer": "4개"},
    )

    assert rendered == "아, 네 개구나!"
    assert "세 개" not in rendered


def test_deterministic_acknowledgement_omits_generic_glue_for_unknown_fact() -> None:
    scenario_data = create_scenario_data(
        "cafe_menu_total",
        CafeSessionContext(
            menu_items=[
                CafeMenuItem(id="juice", name="주스", price=700),
                CafeMenuItem(id="bread", name="빵", price=500),
            ],
            mormi_menu_id="juice",
        ),
    )
    task = get_task(TOTAL_CALC_TASK_ID, scenario_data).model_copy(
        update={"skill_id": "future-skill"}
    )

    rendered = ConversationEngine._younger_sibling_acknowledgement(
        task,
        {"future_slot": "아이의 새로운 설명"},
    )

    assert rendered is None


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
async def test_uncertain_open_set_utterance_uses_bridge_without_state_mutation() -> None:
    """Uncertain social text gets one cheap bridge call, never a second judge."""

    child_text = "그건 말한 것처럼 하면 되는 거잖아"
    primary = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.UNKNOWN,
        task_relation=TaskRelation.UNKNOWN,
        interaction_intent=InteractionIntent.NONE,
        conversation_only=True,
        conversation_summary="앞서 말한 방법을 가리키는 것 같지만 학습 답인지 불확실함",
        bridge_reply="그렇구나. 그럼 다시 알려줄 수 있어?",
        needs_adjudication=True,
        adjudication_reason="대명사가 최근 학습 설명을 가리킬 가능성이 있음",
        confidence=0.82,
    )
    gateway = FakeGateway([primary])
    engine = ConversationEngine(gateway)  # type: ignore[arg-type]
    state = _number_comparison_state()
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, effective, _ = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id="6a0a9858-1f5c-46eb-bf2a-9f044f435e70",
            type="text",
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert gateway.classify_calls == 1
    assert gateway.adjudicate_calls == 0
    assert gateway.bridge_speak_calls == 1
    assert effective.conversation_only is True
    assert effective.needs_adjudication is False
    assert effective.claims == []
    assert next_state.verified_slots == {}


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
        claims=[
            SlotClaim(
                slot_id="final_choice",
                value="left",
                factual=True,
                evidence_span="왼쪽",
            )
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
    assert all(claim.slot_id != "result" for claim in effective_analysis.claims)


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
    assert turn.mormi.text == "그럼 두 메뉴가 모두 얼마인지 알려줄 수 있어?"


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

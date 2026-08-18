from __future__ import annotations

import pytest
from conftest import FakeGateway

from mormi_api.content import (
    CHANGE_TASK_ID,
    HOME_TEACH_TASK_ID,
    TOTAL_CALC_TASK_ID,
    create_scenario_data,
)
from mormi_api.engine import ConversationEngine
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    HintLevel,
    ResponseCategory,
    SafetyCategory,
    SceneType,
    SessionState,
    SlotClaim,
    UtteranceAnalysis,
)


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
            SlotClaim(slot_id="left_count", value=3, factual=True),
            SlotClaim(slot_id="right_count", value=5, factual=True),
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

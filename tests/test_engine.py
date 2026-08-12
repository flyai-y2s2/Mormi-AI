from __future__ import annotations

import pytest
from conftest import FakeGateway

from mormi_api.engine import ConversationEngine
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    HintLevel,
    ResponseCategory,
    SafetyCategory,
    SessionState,
    SlotClaim,
    UtteranceAnalysis,
)


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
        task_ids=["cafe_queue_3_vs_5"],
        task_start_levels={"cafe_queue_3_vs_5": ExpressionLevel.L4},
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
    assert next_state.expression_level is ExpressionLevel.L3
    assert turn.input.target_slots == ["reason"]
    assert "왜" in turn.mormi.text


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
        task_ids=["cafe_queue_3_vs_5"],
        task_start_levels={"cafe_queue_3_vs_5": ExpressionLevel.L4},
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

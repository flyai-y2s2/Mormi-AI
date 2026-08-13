from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway

from mormi_api.content import (
    HOME_TEACH_TASK_ID,
    HOME_TEACHING_CATALOG,
    create_scenario_data,
    get_task,
    home_teaching_task,
)
from mormi_api.db import Database
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    ExpressionLevel,
    HintLevel,
    InputKind,
    LearnerProfile,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SceneType,
    SessionCreate,
    SessionState,
    SkillProfile,
    SlotClaim,
    UtteranceAnalysis,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


def home_state(
    curriculum_session_id: str,
    *,
    expression_level: ExpressionLevel,
    hint_level: HintLevel,
    verified_slots: dict[str, str | int | float | bool] | None = None,
) -> SessionState:
    """Build one reviewed home-teaching turn without the persistence layer."""

    scenario_data = create_scenario_data(
        "home_teach",
        curriculum_session_id=curriculum_session_id,
        skill_id=curriculum_session_id,
    )
    return SessionState(
        learner_id=1,
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=[HOME_TEACH_TASK_ID],
        task_start_levels={HOME_TEACH_TASK_ID: expression_level},
        scenario_data=scenario_data,
        expression_level=expression_level,
        task_start_level=expression_level,
        hint_level=hint_level,
        task_max_hint=hint_level,
        verified_slots=verified_slots or {},
    )


@pytest.mark.asyncio
async def test_wrong_fill_never_completes_number_compare_and_moves_to_joint_h3() -> None:
    """A wrong fill option must be a concept error, never a completed teaching turn."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L1,
        hint_level=HintLevel.H2,
        verified_slots={"answer": "오른쪽"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    assert state.subgoal_id == "complete_rule"
    assert initial.input.kind is InputKind.FILL

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.FILL,
            choice_ids=["fill_2"],  # "오른쪽만 세어"
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.CONCEPTUAL_ERROR
    assert next_state.status.value == "active"
    assert "rule" not in next_state.verified_slots
    assert turn.note_update is None
    assert next_state.expression_level is ExpressionLevel.L0
    assert next_state.hint_level is HintLevel.H3
    assert turn.input.kind is InputKind.JOINT
    assert turn.help_card is not None
    assert turn.help_card.level is HintLevel.H3


@pytest.mark.asyncio
async def test_correct_fill_completes_number_compare_as_supported_learning() -> None:
    """The reviewed correct fill option is the only supported completion path."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L1,
        hint_level=HintLevel.H2,
        verified_slots={"answer": "오른쪽"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.FILL,
            choice_ids=["fill_0"],  # "하나씩 짝지어"
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.CORRECT_FULL
    assert next_state.status.value == "completed"
    assert next_state.verified_slots["rule"] == (
        HOME_TEACHING_CATALOG["number-compare"].learned_line
    )
    assert turn.completion is not None
    assert turn.completion.outcome.value == "supported"
    assert turn.note_update is not None
    assert turn.note_update.attribution.value == "coauthored"


@pytest.mark.asyncio
async def test_error_analysis_cannot_complete_even_with_a_factual_claim() -> None:
    """The orchestrator requires both a success verdict and canonical facts."""

    expected_rule = HOME_TEACHING_CATALOG["number-compare"].learned_line
    contradictory_analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        difficulty_class=DifficultyClass.CONCEPT,
        claims=[SlotClaim(slot_id="rule", value=expected_rule, factual=True)],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([contradictory_analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="느낌으로 보면 돼",
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.CONCEPTUAL_ERROR
    assert "rule" not in next_state.verified_slots
    assert next_state.status.value == "active"
    assert turn.note_update is None
    assert turn.completion is None


@pytest.mark.asyncio
async def test_abusive_text_sets_a_clear_boundary_without_changing_learning_state() -> None:
    """Unsafe speech gets deterministic copy and cannot advance or lower the task."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    original_subgoal = state.subgoal_id
    state.current_turn_id = initial.turn_id

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="야 이 개새끼야",
        ),
        initial.mormi.text,
    )

    assert analysis.safety_category is SafetyCategory.ABUSIVE
    assert turn.mormi.text == (
        "그 말은 듣기 싫어. 점이 두 개 있는 것 같은데, 너는 몇 개로 셌어?"
    )
    assert next_state.status.value == "active"
    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert next_state.subgoal_id == original_subgoal
    assert next_state.verified_slots == {}
    assert turn.input == initial.input
    assert turn.note_update is None
    assert turn.completion is None


@pytest.mark.asyncio
async def test_claimless_correct_partial_never_falls_through_to_unrelated() -> None:
    """Related colloquial speech stays in the learning path without fake claims."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="하나, 둘, 셋 하고 세면 돼",
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert next_state.unrelated_count == 0
    assert next_state.expression_level is ExpressionLevel.L3
    assert next_state.hint_level is HintLevel.H0
    assert turn.mormi.text.endswith("점은 모두 몇 개일까?")
    assert "그 얘기는 이따" not in turn.mormi.text
    assert turn.input.kind is InputKind.TEXT
    assert turn.input.target_slots == ["answer", "count_sequence"]


@pytest.mark.asyncio
async def test_number_count_partial_meanings_are_remembered_separately() -> None:
    """A child's own counting words preserve useful pieces and ask only what is missing."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(slot_id="answer", value="3개", factual=True),
            SlotClaim(
                slot_id="count_sequence",
                value="하나 둘 셋 하고 세기",
                factual=True,
            ),
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="하나, 둘, 셋 하고 세면 돼",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {
        "answer": "3",
        "count_sequence": "one_by_one_order",
    }
    assert next_state.expression_level is ExpressionLevel.L3
    assert turn.mormi.text.endswith(
        "나는 점을 자꾸 하나 놓쳐. 셀 때 손가락은 어떻게 하면 돼?"
    )
    assert turn.input.target_slots == ["tracking", "count_sequence"]


@pytest.mark.asyncio
async def test_unrelated_recovery_repeats_the_actionable_current_question() -> None:
    """A recovery turn never asks the child to answer a meta 'return?' question."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.UNRELATED_RESPONSE,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        claims=[],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="오늘 급식 맛있었어",
        ),
        initial.mormi.text,
    )

    assert next_state.unrelated_count == 1
    assert turn.mormi.text.endswith(initial.mormi.text)
    assert "아까 질문으로 돌아갈까" not in turn.mormi.text
    assert turn.input == initial.input


def test_every_wrong_home_l2_l1_option_stays_unverified_and_incomplete() -> None:
    """Reviewed choices are a hard safety boundary across all 36 home lessons."""

    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]

    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        answer = task.slots["answer"].expected
        step_states = [
            (ExpressionLevel.L2, {}),
            (ExpressionLevel.L2, {"answer": answer}),
            (ExpressionLevel.L1, {}),
            (ExpressionLevel.L1, {"answer": answer}),
        ]

        for level, verified_slots in step_states:
            state = home_state(
                spec.id,
                expression_level=level,
                hint_level=HintLevel.H0,
                verified_slots=verified_slots,
            )
            step = task.step_for(level, state.verified_slots)
            assert step.input.kind in {InputKind.CHOICES, InputKind.FILL}

            for choice in step.input.choices:
                response = ChildResponse(
                    turn_id="turn_structured_option",
                    response_id=uuid4(),
                    type=(
                        ResponseType.FILL
                        if step.input.kind is InputKind.FILL
                        else ResponseType.CHOICE
                    ),
                    choice_ids=[choice.id],
                )
                analysis = engine._deterministic_analysis(state, task, response)
                factual_slots = {
                    claim.slot_id for claim in analysis.claims if claim.factual
                }
                is_correct = set(step.target_slots).issubset(factual_slots)
                if is_correct:
                    continue

                verified = task.validated_claims(
                    (claim.slot_id, claim.value, claim.factual) for claim in analysis.claims
                )
                merged = {**state.verified_slots, **verified}

                assert analysis.response_category not in {
                    ResponseCategory.CORRECT_FULL,
                    ResponseCategory.CORRECT_PARTIAL,
                }, f"{spec.id}/{step.id}/{choice.id}"
                assert not analysis.claims or all(
                    not claim.factual for claim in analysis.claims
                ), f"{spec.id}/{step.id}/{choice.id}"
                assert not task.complete(merged), f"{spec.id}/{step.id}/{choice.id}"


@pytest.mark.asyncio
async def test_no_response_lowers_expression_without_immediately_raising_concept_hint() -> None:
    """Silence means expression/no-response first, not a mathematical error."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.NO_RESPONSE,
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.NO_RESPONSE
    assert analysis.difficulty_class is DifficultyClass.EXPRESSION
    assert next_state.expression_level is ExpressionLevel.L3
    assert next_state.hint_level is HintLevel.H0
    assert next_state.task_max_hint is HintLevel.H0
    assert turn.help_card is None
    assert turn.input.kind is InputKind.TEXT
    assert turn.status.value == "active"


def test_unknown_or_multiple_choice_ids_are_input_errors_in_analysis() -> None:
    """Malformed structured input must never be interpreted as a math answer."""

    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L1,
        hint_level=HintLevel.H0,
        verified_slots={"answer": "오른쪽"},
    )
    task = get_task(HOME_TEACH_TASK_ID, state.scenario_data)

    for choice_ids in (["not-an-option"], ["fill_0", "fill_2"]):
        analysis = engine._deterministic_analysis(
            state,
            task,
            ChildResponse(
                turn_id="turn_bad_choice",
                response_id=uuid4(),
                type=ResponseType.FILL,
                choice_ids=choice_ids,
            ),
        )
        assert analysis.response_category is ResponseCategory.RECOGNITION_OR_INPUT_ERROR
        assert analysis.difficulty_class is DifficultyClass.INPUT
        assert analysis.claims == []


@pytest.mark.asyncio
async def test_service_rejects_unknown_or_multiple_choice_ids_before_analysis(
    tmp_path: object,
) -> None:
    """The API service rejects malformed IDs before they can reach LangGraph."""

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/structured-input.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    await repository.save_profile(
        LearnerProfile(
            learner_id=91,
            skills={
                "compare_quantity_in_context": SkillProfile(
                    skill_id="compare_quantity_in_context",
                    highest_stable_expression_level=ExpressionLevel.L2,
                )
            },
        )
    )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=91,
            scene=SceneType.CAFE,
            scenario_id="cafe_queue_demo",
        )
    )
    assert started.turn.input.kind is InputKind.CHOICES

    for choice_ids in (["not-an-option"], ["2", "3"]):
        with pytest.raises(ValueError, match="choice"):
            await service.respond(
                started.conversation_id,
                ChildResponse(
                    turn_id=started.turn.turn_id,
                    response_id=uuid4(),
                    type=ResponseType.CHOICE,
                    choice_ids=choice_ids,
                ),
            )

    untouched = await service.snapshot(started.conversation_id)
    assert untouched.turn.turn_id == started.turn.turn_id
    await database.dispose()

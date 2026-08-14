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
    state.supported_note_slots = ["answer"]
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    assert state.subgoal_id == "complete_comparison"
    assert initial.input.kind is InputKind.FILL

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.FILL,
            choice_ids=["left_more"],
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.CONCEPTUAL_ERROR
    assert next_state.status.value == "active"
    assert "reason" not in next_state.verified_slots
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
    state.supported_note_slots = ["answer"]
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.FILL,
            choice_ids=["right_more"],
        ),
        initial.mormi.text,
    )

    assert analysis.response_category is ResponseCategory.CORRECT_FULL
    assert next_state.status.value == "completed"
    assert next_state.verified_slots["reason"] == "count_comparison"
    assert turn.completion is not None
    assert turn.completion.outcome.value == "supported"
    assert turn.note_update is not None
    assert turn.note_update.attribution.value == "coauthored"


@pytest.mark.asyncio
async def test_error_analysis_cannot_complete_even_with_a_factual_claim() -> None:
    """The orchestrator requires both a success verdict and canonical facts."""

    contradictory_analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        difficulty_class=DifficultyClass.CONCEPT,
        claims=[SlotClaim(slot_id="reason", value="count_comparison", factual=True)],
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
    assert "reason" not in next_state.verified_slots
    assert next_state.status.value == "active"
    assert turn.note_update is None
    assert turn.completion is None


@pytest.mark.asyncio
async def test_bare_comparison_conclusion_is_kept_but_does_not_create_a_note() -> None:
    """'오른쪽이 커' answers which side, but it is not a general explanation."""

    child_text = "오른쪽이 커"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-compare",
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
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert next_state.status.value == "active"
    assert next_state.verified_slots == {"answer": "오른쪽"}
    assert "reason" not in next_state.child_note_evidence
    assert turn.note_update is None
    assert turn.input.target_slots == ["reason"]
    assert turn.mormi.text == (
        "아, 오른쪽이 더 많구나! 나 3이랑 5를 어떻게 비교할지 헷갈려... 알려줄 수 있어?"
    )


@pytest.mark.asyncio
async def test_even_an_overclaimed_short_conclusion_cannot_enter_the_star_note() -> None:
    """The code provenance gate stays safe even if the classifier overclaims reason."""

    child_text = "오른쪽이 더 많아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            ),
            SlotClaim(
                slot_id="reason",
                value="count_comparison",
                factual=True,
                evidence_span=child_text,
            ),
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-compare",
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
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert next_state.status.value == "active"
    assert next_state.verified_slots == {"answer": "오른쪽"}
    assert next_state.child_note_evidence == {}
    assert turn.note_update is None
    assert turn.input.target_slots == ["reason"]


@pytest.mark.asyncio
async def test_bare_amount_is_an_answer_not_a_star_note_explanation() -> None:
    """A result such as '600원이야' must never become the generalization note."""

    child_text = "600원이야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="600원",
                factual=True,
                evidence_span=child_text,
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "money-count",
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
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert next_state.status.value == "active"
    assert next_state.verified_slots == {"answer": "600원"}
    assert next_state.child_note_evidence == {}
    assert turn.note_update is None


@pytest.mark.asyncio
async def test_star_note_uses_only_the_exact_factual_clause_from_a_mixed_turn() -> None:
    """One wrong clause cannot contaminate a separately grounded method clause."""

    child_text = "손가락을 하나씩 펴면서 세면 돼. 점은 4개야"
    method_span = "손가락을 하나씩 펴면서 세면 돼"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.BOTH,
        claims=[
            SlotClaim(
                slot_id="tracking",
                value="point_each_dot",
                factual=True,
                evidence_span=method_span,
            ),
            SlotClaim(
                slot_id="answer",
                value="4",
                factual=False,
                evidence_span="점은 4개야",
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
        expression_level=ExpressionLevel.L3,
        hint_level=HintLevel.H0,
        verified_slots={"answer": "3"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert next_state.status.value == "completed"
    assert turn.note_update is not None
    assert method_span in turn.note_update.text
    assert "4개" not in turn.note_update.text
    assert HOME_TEACHING_CATALOG["number-count"].learned_line not in turn.note_update.text


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
        "그 말은 듣기 싫어. 점이 몇 개인지랑 어떻게 세는지 알려주면 안 될까?"
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
    assert turn.mormi.text.endswith("지금 점이 몇 개야?")
    assert "그 얘기는 이따" not in turn.mormi.text
    assert turn.input.kind is InputKind.TEXT
    assert turn.input.target_slots == ["answer"]


@pytest.mark.asyncio
async def test_number_count_accepts_childs_own_counting_method_without_forcing_pointing() -> None:
    """Saying the number words in order is a valid method, not a lesser fragment."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="tracking",
                value="하나 둘 셋 하면서 세기",
                factual=True,
                evidence_span="하나, 둘, 셋 하면서 세면 돼",
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
        verified_slots={"answer": "3"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="하나, 둘, 셋 하면서 세면 돼",
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"answer": "3", "tracking": "count_each_once"}
    assert next_state.status.value == "completed"
    assert turn.note_update is not None
    assert "하나, 둘, 셋 하면서 세면 돼" in turn.note_update.text
    assert "가리키며" not in turn.note_update.text


@pytest.mark.asyncio
async def test_repeated_known_fact_cannot_repeat_the_same_open_question_forever() -> None:
    """No-new-progress partials move down the expression ladder one step per turn."""

    duplicate_answer = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([duplicate_answer, duplicate_answer]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
        verified_slots={"answer": "오른쪽"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    after_first, _, first_turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="오른쪽",
        ),
        initial.mormi.text,
    )
    assert after_first.expression_level is ExpressionLevel.L3
    assert first_turn.mormi.text != initial.mormi.text
    assert first_turn.input.kind is InputKind.TEXT

    after_first.current_turn_id = first_turn.turn_id
    after_second, _, second_turn = await engine.run_turn(
        after_first,
        ChildResponse(
            turn_id=first_turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="오른쪽",
        ),
        first_turn.mormi.text,
    )
    assert after_second.expression_level is ExpressionLevel.L2
    assert second_turn.mormi.text != first_turn.mormi.text
    assert second_turn.input.kind is InputKind.CHOICES


@pytest.mark.asyncio
async def test_no_progress_at_lowest_level_enters_joint_help_instead_of_looping() -> None:
    duplicate_answer = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span="오른쪽",
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([duplicate_answer]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-compare",
        expression_level=ExpressionLevel.L0,
        hint_level=HintLevel.H2,
        verified_slots={"answer": "오른쪽"},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="오른쪽",
        ),
        initial.mormi.text,
    )

    assert next_state.expression_level is ExpressionLevel.L0
    assert next_state.hint_level is HintLevel.H3
    assert turn.input.kind is InputKind.JOINT
    assert turn.help_card is not None
    assert turn.help_card.level is HintLevel.H3


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
                factual_slots = {claim.slot_id for claim in analysis.claims if claim.factual}
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
                assert not analysis.claims or all(not claim.factual for claim in analysis.claims), (
                    f"{spec.id}/{step.id}/{choice.id}"
                )
                assert not task.complete(merged), f"{spec.id}/{step.id}/{choice.id}"


@pytest.mark.asyncio
async def test_no_response_is_a_help_request_and_opens_the_first_help_card() -> None:
    """The UI help action is expression support, never a mathematical error."""

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

    assert analysis.response_category is ResponseCategory.HELP_REQUEST
    assert analysis.difficulty_class is DifficultyClass.EXPRESSION
    assert next_state.expression_level is ExpressionLevel.L3
    assert next_state.hint_level is HintLevel.H1
    assert next_state.task_max_hint is HintLevel.H1
    assert turn.help_card is not None
    assert turn.help_card.auto_open is True
    assert turn.input.kind is InputKind.TEXT
    assert turn.status.value == "active"


@pytest.mark.asyncio
async def test_repeated_no_response_walks_every_ladder_step_without_changing_problem() -> None:
    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
    )
    turn = engine.initial_turn(state)
    state.current_turn_id = turn.turn_id
    original_visual = turn.visual

    expected = [
        (ExpressionLevel.L3, HintLevel.H1, InputKind.TEXT),
        (ExpressionLevel.L2, HintLevel.H2, InputKind.CHOICES),
        (ExpressionLevel.L1, HintLevel.H2, InputKind.CHOICES),
        (ExpressionLevel.L0, HintLevel.H3, InputKind.JOINT),
        (ExpressionLevel.L0, HintLevel.H3, InputKind.JOINT),
    ]

    for level, hint, input_kind in expected:
        previous_task_id = turn.task_id
        state, analysis, turn = await engine.run_turn(
            state,
            ChildResponse(
                turn_id=turn.turn_id,
                response_id=uuid4(),
                type=ResponseType.NO_RESPONSE,
            ),
            turn.mormi.text,
        )

        assert analysis.response_category is ResponseCategory.HELP_REQUEST
        assert analysis.difficulty_class is DifficultyClass.EXPRESSION
        assert state.expression_level is level
        assert state.hint_level is hint
        assert turn.input.kind is input_kind
        assert turn.task_id == previous_task_id
        assert turn.visual == original_visual
        assert turn.help_card is not None
        assert turn.help_card.auto_open is True
        assert turn.pedagogy is not None
        assert turn.pedagogy.expression_level is level
        assert turn.pedagogy.hint_level is hint
        task = get_task(state.current_task_id, state.scenario_data)
        current_question = task.step_for(level, state.verified_slots).prompt
        assert current_question in turn.mormi.text

    assert state.status.value == "active"


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

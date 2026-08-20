from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway

from mormi_api.content import (
    HOME_TEACH_TASK_ID,
    HOME_TEACHING_CATALOG,
    QUEUE_TASK_ID,
    create_scenario_data,
    get_task,
    home_teaching_task,
)
from mormi_api.db import Database
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ArithmeticClaim,
    ChildResponse,
    DifficultyClass,
    EntryPhase,
    ExpressionLevel,
    HelpCardEvent,
    HintLevel,
    InputKind,
    InteractionIntent,
    LearnerProfile,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SceneType,
    SessionCreate,
    SessionState,
    SkillProfile,
    SlotClaim,
    SupportTrigger,
    TaskRelation,
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
    assert next_state.verified_slots["reason"] is True
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
async def test_false_arithmetic_relation_cannot_verify_or_enter_the_star_note() -> None:
    """Haiku interprets wording; code rejects only the structured false equation."""

    child_text = "2000원에서 1800원 내면 300원 남아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=300,
                factual=True,
                evidence_span="300원",
                interpretation_confidence=1,
            ),
            SlotClaim(
                slot_id="rule",
                value=None,
                factual=True,
                evidence_span=child_text,
                supported=True,
                support_confidence=1,
            ),
        ],
        arithmetic_claims=[
            ArithmeticClaim(
                left=2000,
                right=1800,
                operation="subtraction",
                result=300,
                evidence_span=child_text,
                related_slot_ids=["rule"],
                interpretation_confidence=1,
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "money-budget",
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
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CONCEPTUAL_ERROR
    assert "answer" not in next_state.verified_slots
    assert "rule" not in next_state.verified_slots
    assert next_state.status.value == "active"
    assert turn.note_update is None
    assert turn.completion is None
    assert turn.help_card is not None
    assert turn.mormi.text == (
        "2,000원에서 1,800원을 빼면 300원이 남는다는 말이 "
        "아직 잘 모르겠어... 도움 카드를 보고 다시 알려줄 수 있어?"
    )
    assert "200원" not in turn.mormi.text


def test_false_arithmetic_claim_gets_reviewed_scene_roles_and_exact_evidence() -> None:
    child_text = "2000원에서 1800원을 빼면 300원이 남아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        arithmetic_claims=[
            ArithmeticClaim(
                left=2000,
                right=1800,
                operation="subtraction",
                result=300,
                evidence_span=child_text,
                related_slot_ids=["rule"],
                interpretation_confidence=1,
            )
        ],
        confidence=1,
    )
    task = home_teaching_task(
        HOME_TEACHING_CATALOG["money-budget"],
        skill_id="money-budget",
    )

    claims = ConversationEngine._speaker_arithmetic_claims(task, child_text, analysis)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.source_text == child_text
    assert claim.truth_status == "false"
    assert (claim.left.value, claim.left.role, claim.left.unit) == (2000, "낸 돈", "원")
    assert (claim.right.value, claim.right.role, claim.right.unit) == (1800, "간식값", "원")
    assert (claim.claimed_result.value, claim.claimed_result.role) == (300, "남는 돈")


def test_false_arithmetic_fallback_does_not_mention_an_invisible_help_card() -> None:
    child_text = "700원하고 500원을 더하면 1300원이야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        arithmetic_claims=[
            ArithmeticClaim(
                left=700,
                right=500,
                operation="addition",
                result=1300,
                evidence_span=child_text,
                related_slot_ids=["method"],
                interpretation_confidence=1,
            )
        ],
    )
    task = home_teaching_task(
        HOME_TEACHING_CATALOG["money-price"],
        skill_id="money-price",
    )
    claim = ConversationEngine._speaker_arithmetic_claims(task, child_text, analysis)[0]

    text = ConversationEngine._support_transition_fallback(
        SupportTrigger.CONCEPTUAL_CONFLICT,
        HelpCardEvent.NONE,
        arithmetic_claims=[claim],
        help_card_visible=False,
    )

    assert text == (
        "700원과 500원을 더하면 모두 1,300원이라는 말이 "
        "아직 잘 모르겠어... 어떻게 계산한 건지 다시 알려줄 수 있어?"
    )
    assert "카드" not in text
    assert "1,200" not in text


def test_reversed_addition_keeps_the_reviewed_scene_roles() -> None:
    child_text = "500원하고 700원을 더하면 1300원이야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        arithmetic_claims=[
            ArithmeticClaim(
                left=500,
                right=700,
                operation="addition",
                result=1300,
                evidence_span=child_text,
                related_slot_ids=["method"],
                interpretation_confidence=1,
            )
        ],
    )
    task = home_teaching_task(
        HOME_TEACHING_CATALOG["money-price"],
        skill_id="money-price",
    )

    claim = ConversationEngine._speaker_arithmetic_claims(task, child_text, analysis)[0]

    assert (claim.left.value, claim.left.role) == (500, "빵 가격")
    assert (claim.right.value, claim.right.role) == (700, "주스 가격")
    assert claim.claimed_result.role == "전체 금액"


@pytest.mark.asyncio
async def test_number_rich_explanation_without_structured_relation_fails_closed() -> None:
    """A positive label alone cannot bypass the structured arithmetic contract."""

    child_text = "2000원에서 1800원 내면 300원 남아"
    analysis = UtteranceAnalysis(
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
                support_confidence=1,
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        "money-budget",
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
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CONCEPTUAL_ERROR
    assert "rule" not in next_state.verified_slots
    assert next_state.status.value == "active"
    assert turn.note_update is None
    assert turn.completion is None


@pytest.mark.asyncio
async def test_true_arithmetic_relation_can_complete_without_a_phrase_allowlist() -> None:
    """A correct relation is accepted from structured meaning, not a Korean verb regex."""

    child_text = "500원이랑 100원을 모으면 600원이 되는 거야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=600,
                factual=True,
                evidence_span="600원",
                interpretation_confidence=1,
            ),
            SlotClaim(
                slot_id="rule",
                value=None,
                factual=True,
                evidence_span=child_text,
                supported=True,
                support_confidence=1,
            ),
        ],
        arithmetic_claims=[
            ArithmeticClaim(
                left=500,
                right=100,
                operation="addition",
                result=600,
                evidence_span=child_text,
                related_slot_ids=["rule"],
                interpretation_confidence=1,
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

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CORRECT_FULL
    assert next_state.status.value == "completed"
    assert next_state.verified_slots["answer"] == "600원"
    assert "rule" in next_state.verified_slots
    assert turn.note_update is not None
    assert turn.completion is not None


def test_reviewed_money_tasks_publish_arithmetic_truth_contracts() -> None:
    """Home money skills expose facts for validation without dictating child wording."""

    expected = {
        "money-count": ("addition", 500, 100, 600),
        "money-price": ("addition", 700, 500, 1200),
        "money-budget": ("subtraction", 2000, 1800, 200),
    }
    for curriculum_session_id, values in expected.items():
        spec = HOME_TEACHING_CATALOG[curriculum_session_id]
        task = home_teaching_task(spec, skill_id=spec.id)
        contract = task.arithmetic_contract

        assert contract is not None
        assert (contract.operation, contract.left, contract.right, contract.result) == values


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
@pytest.mark.parametrize(
    ("child_text", "evidence_span"),
    [
        ("10개중에 색칠된게 3개던데?", "색칠된게 3개"),
        ("빈 동그라미 말고 초록색은 셋이야", "초록색은 셋"),
        ("열 칸 중에서 채워진 건 세 칸이네", "채워진 건 세 칸"),
        ("동그라미를 보니까 색 있는 게 세 개였어", "색 있는 게 세 개"),
    ],
)
async def test_repeated_count_evidence_moves_from_targeted_text_to_choices(
    child_text: str,
    evidence_span: str,
) -> None:
    """A valid visual observation is preserved and never triggers a duplicate prompt."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="3",
                factual=True,
                evidence_span=evidence_span,
            )
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
    state.entry_phase = EntryPhase.AWAITING_TARGETED_FOLLOWUP
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert next_state.verified_slots == {"answer": "3"}
    assert next_state.concept_failures == 0
    assert next_state.expression_level is ExpressionLevel.L2
    assert next_state.hint_level is HintLevel.H0
    assert turn.input.kind is InputKind.CHOICES
    assert turn.input.target_slots == ["tracking"]
    assert turn.mormi.text != initial.mormi.text
    assert "같이 골라 볼까?" in turn.mormi.text
    assert turn.note_update is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("curriculum_session_id", "child_text", "evidence_span"),
    [
        ("money-count", "500원하고 100원이니까 600원이야", "600원이야"),
        ("money-budget", "돌려받는 돈은 200원이야", "200원이야"),
        ("pattern-number", "그다음 수는 10이야", "10이야"),
        ("clock-basic", "지금은 3시 30분이야", "3시 30분이야"),
    ],
)
async def test_repeated_factual_evidence_changes_support_across_home_domains(
    curriculum_session_id: str,
    child_text: str,
    evidence_span: str,
) -> None:
    """No-progress handling is task-contract based, not number-count specific."""

    expected_answer = HOME_TEACHING_CATALOG[curriculum_session_id].sample_problem["correct"]
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=expected_answer,
                factual=True,
                evidence_span=evidence_span,
            )
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = home_state(
        curriculum_session_id,
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
        verified_slots={"answer": expected_answer},
    )
    state.entry_phase = EntryPhase.AWAITING_TARGETED_FOLLOWUP
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert next_state.verified_slots == {"answer": expected_answer}
    assert next_state.concept_failures == 0
    assert next_state.expression_level is ExpressionLevel.L2
    assert turn.input.kind is InputKind.CHOICES
    assert turn.mormi.text != initial.mormi.text
    assert turn.note_update is None


@pytest.mark.asyncio
async def test_repeated_observations_change_support_in_cafe_queue_too() -> None:
    """The same no-progress policy applies to a cafe task, not just home lessons."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="left_count",
                value=3,
                factual=True,
                evidence_span="왼쪽은 세 명",
            ),
            SlotClaim(
                slot_id="right_count",
                value=5,
                factual=True,
                evidence_span="오른쪽은 다섯 명",
            ),
        ],
        confidence=1,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    state = SessionState(
        learner_id=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_queue",
        task_ids=[QUEUE_TASK_ID],
        task_start_levels={QUEUE_TASK_ID: ExpressionLevel.L4},
        scenario_data={"left_count": 3, "right_count": 5},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
        verified_slots={"left_count": 3, "right_count": 5},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id
    assert initial.input.target_slots == ["final_choice", "reason"]

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="왼쪽은 세 명이고 오른쪽은 다섯 명이야",
        ),
        initial.mormi.text,
    )

    assert returned_analysis.response_category is ResponseCategory.CORRECT_PARTIAL
    assert next_state.verified_slots == {"left_count": 3, "right_count": 5}
    assert next_state.concept_failures == 0
    assert next_state.expression_level is ExpressionLevel.L3
    assert turn.input.target_slots == ["final_choice"]
    assert turn.mormi.text != initial.mormi.text
    assert turn.note_update is None


@pytest.mark.asyncio
async def test_number_count_accepts_a_novel_grounded_method_without_a_method_code() -> None:
    """A valid new method satisfies meaning without expanding an alias enum."""

    child_method = "센 점마다 단추를 하나씩 옆으로 옮겨 놓으면 돼"

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="tracking",
                value=None,
                factual=True,
                evidence_span=child_method,
                supported=True,
                support_confidence=0.94,
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
            text=child_method,
        ),
        initial.mormi.text,
    )

    assert next_state.verified_slots == {"answer": "3", "tracking": True}
    assert next_state.status.value == "completed"
    assert turn.note_update is not None
    assert child_method in turn.note_update.text
    assert "가리키며" not in turn.note_update.text


def test_semantic_support_rejects_an_explicitly_unsupported_internal_code() -> None:
    task = home_teaching_task(
        HOME_TEACHING_CATALOG["number-count"],
        skill_id="number-count",
    )

    verified = task.validated_slot_claims(
        [
            SlotClaim(
                slot_id="tracking",
                value="count_each_once",
                factual=True,
                evidence_span="그냥 해",
                supported=False,
                support_confidence=0.99,
            )
        ]
    )

    assert verified == {}


def test_legacy_semantic_code_and_new_support_flag_are_the_same_completed_state() -> None:
    """A deployed conversation can resume after the storage contract changes."""

    task = home_teaching_task(
        HOME_TEACHING_CATALOG["number-count"],
        skill_id="number-count",
    )
    tracking = task.slots["tracking"]

    assert tracking.equivalent_state_value("count_each_once", True) is True
    assert tracking.equivalent_state_value(True, "one_by_one_order") is True
    assert tracking.equivalent_state_value(None, True) is False


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


@pytest.mark.asyncio
@pytest.mark.parametrize("scene", [SceneType.HOME_TEACH, SceneType.CAFE])
@pytest.mark.parametrize(
    "reported_category",
    [ResponseCategory.UNRELATED_RESPONSE, ResponseCategory.CONCEPTUAL_ERROR],
)
async def test_meta_challenge_gets_one_bounded_bridge_without_learning_mutation(
    scene: SceneType,
    reported_category: ResponseCategory,
) -> None:
    """A safe challenge is acknowledged, but never becomes learning evidence."""

    child_text = "너 알면서 일부러 물어보지?"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=reported_category,
        difficulty_class=DifficultyClass.ENGAGEMENT,
        task_relation=TaskRelation.META_ABOUT_MORMI,
        interaction_intent=InteractionIntent.AUTHENTICITY_CHALLENGE,
        social_grounding_span=child_text,
        confidence=0.96,
    )
    engine = ConversationEngine(  # type: ignore[arg-type]
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )
    if scene is SceneType.HOME_TEACH:
        state = home_state(
            "number-count",
            expression_level=ExpressionLevel.L4,
            hint_level=HintLevel.H0,
        )
    else:
        state = SessionState(
            learner_id=1,
            scene=SceneType.CAFE,
            scenario_id="cafe_queue",
            task_ids=[QUEUE_TASK_ID],
            task_start_levels={QUEUE_TASK_ID: ExpressionLevel.L4},
            scenario_data={"left_count": 3, "right_count": 5},
            expression_level=ExpressionLevel.L4,
            task_start_level=ExpressionLevel.L4,
            hint_level=HintLevel.H0,
        )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id
    original_slots = dict(state.verified_slots)
    original_subgoal = state.subgoal_id

    next_state, returned_analysis, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text=child_text,
        ),
        initial.mormi.text,
    )

    assert returned_analysis.task_relation is TaskRelation.META_ABOUT_MORMI
    assert next_state.unrelated_count == 1
    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert next_state.subgoal_id == original_subgoal
    assert next_state.verified_slots == original_slots
    assert turn.input == initial.input
    assert turn.visual == initial.visual
    assert turn.note_update is None
    assert turn.completion is None
    assert "진짜 몰라서" in turn.mormi.text
    assert turn.mormi.text != initial.mormi.text


@pytest.mark.asyncio
async def test_deterministic_playful_text_returns_neutrally_without_progress() -> None:
    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-count",
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
            text="메롱",
        ),
        initial.mormi.text,
    )

    assert analysis.safety_category is SafetyCategory.PLAYFUL_OFFTOPIC
    assert analysis.interaction_intent is InteractionIntent.PLAYFUL_TEASE
    assert next_state.expression_level is ExpressionLevel.L4
    assert next_state.hint_level is HintLevel.H0
    assert next_state.verified_slots == {}
    assert turn.input == initial.input
    assert "장난치는 거지" in turn.mormi.text
    assert turn.note_update is None


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

                verified = task.validated_slot_claims(analysis.claims)
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
    assert "도움 카드가 나왔어" in turn.mormi.text
    assert "도움 카드가 열렸네" not in turn.mormi.text
    assert initial.mormi.text not in turn.mormi.text


@pytest.mark.asyncio
async def test_related_vague_reply_gets_one_clarification_before_help_opens() -> None:
    """A vague on-topic phrase is not silently turned into a help request."""

    vague = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.RELATED_VAGUE,
        difficulty_class=DifficultyClass.EXPRESSION,
        grounding_span="잘 세봐",
        confidence=1,
    )
    engine = ConversationEngine(
        FakeGateway([vague, vague]),  # type: ignore[arg-type]
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
        verified_slots={"answer": 3},
    )
    turn = engine.initial_turn(state)
    state.current_turn_id = turn.turn_id

    first_state, first_analysis, first_turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="잘 세봐",
        ),
        turn.mormi.text,
    )

    assert first_analysis.response_category is ResponseCategory.RELATED_VAGUE
    assert first_state.expression_level is ExpressionLevel.L4
    assert first_state.hint_level is HintLevel.H0
    assert first_state.vague_clarifications == 1
    assert first_turn.help_card is None
    assert "잘 세봐" in first_turn.mormi.text
    assert "조금만 더 알려줄래" in first_turn.mormi.text
    assert turn.mormi.text not in first_turn.mormi.text

    second_state, _, second_turn = await engine.run_turn(
        first_state,
        ChildResponse(
            turn_id=first_turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="잘 세봐",
        ),
        first_turn.mormi.text,
    )

    assert second_state.expression_level.rank < ExpressionLevel.L4.rank
    assert second_state.hint_level is HintLevel.H1
    assert second_state.vague_clarifications == 0
    assert second_turn.help_card is not None
    assert "카드" in second_turn.mormi.text
    assert first_turn.mormi.text not in second_turn.mormi.text


@pytest.mark.asyncio
async def test_concept_conflict_uses_safe_grounding_without_evaluating_child() -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CONCEPTUAL_ERROR,
        difficulty_class=DifficultyClass.CONCEPT,
        grounding_span="빈칸도 같이 세면 돼",
        confidence=1,
    )
    engine = ConversationEngine(
        FakeGateway([analysis]),  # type: ignore[arg-type]
        show_internal_pedagogy=True,
    )
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L4,
        hint_level=HintLevel.H0,
        verified_slots={"answer": 3},
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, _, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="빈칸도 같이 세면 돼",
        ),
        initial.mormi.text,
    )

    assert next_state.hint_level is HintLevel.H1
    assert turn.help_card is not None
    assert "빈칸도 같이 세면 돼" in turn.mormi.text
    assert "아직 헷갈려" in turn.mormi.text
    assert "틀렸" not in turn.mormi.text


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
        assert "도움 카드가 열렸네" not in turn.mormi.text
        assert turn.mormi.text != ""

    assert state.status.value == "active"


@pytest.mark.parametrize(
    ("expression_level", "hint_level"),
    [
        (ExpressionLevel.L0, HintLevel.H0),
        (ExpressionLevel.L0, HintLevel.H2),
        (ExpressionLevel.L4, HintLevel.H3),
    ],
)
def test_initial_turn_normalizes_every_terminal_mismatch_to_joint_h3(
    expression_level: ExpressionLevel,
    hint_level: HintLevel,
) -> None:
    """Legacy/profile state may not expose only one half of the terminal contract."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-count",
        expression_level=expression_level,
        hint_level=hint_level,
    )

    turn = engine.initial_turn(state)

    assert state.expression_level is ExpressionLevel.L0
    assert state.hint_level is HintLevel.H3
    assert state.task_max_hint is HintLevel.H3
    assert turn.input.kind is InputKind.JOINT
    assert turn.help_card is not None
    assert turn.help_card.level is HintLevel.H3
    assert turn.pedagogy is not None
    assert turn.pedagogy.expression_level is ExpressionLevel.L0
    assert turn.pedagogy.hint_level is HintLevel.H3


@pytest.mark.parametrize(
    ("expression_level", "hint_level"),
    [
        (ExpressionLevel.L4, HintLevel.H2),
        (ExpressionLevel.L1, HintLevel.H0),
    ],
)
def test_nonterminal_expression_and_hint_levels_remain_independent(
    expression_level: ExpressionLevel,
    hint_level: HintLevel,
) -> None:
    """The terminal invariant must not collapse valid intermediate L/H pairs."""

    engine = ConversationEngine(FakeGateway(), show_internal_pedagogy=True)  # type: ignore[arg-type]
    state = home_state(
        "number-count",
        expression_level=expression_level,
        hint_level=hint_level,
    )

    turn = engine.initial_turn(state)

    assert state.expression_level is expression_level
    assert state.hint_level is hint_level
    assert turn.pedagogy is not None
    assert turn.pedagogy.expression_level is expression_level
    assert turn.pedagogy.hint_level is hint_level


@pytest.mark.asyncio
async def test_expression_block_at_l1_enters_complete_joint_contract() -> None:
    """The original asymmetric branch must not create L0-H2."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.EXPRESSION_BLOCK,
        difficulty_class=DifficultyClass.EXPRESSION,
        bottleneck="expression",
        confidence=1,
    )
    engine = ConversationEngine(
        FakeGateway([analysis]),
        show_internal_pedagogy=True,
    )  # type: ignore[arg-type]
    state = home_state(
        "number-count",
        expression_level=ExpressionLevel.L1,
        hint_level=HintLevel.H2,
    )
    initial = engine.initial_turn(state)
    state.current_turn_id = initial.turn_id

    next_state, classified, turn = await engine.run_turn(
        state,
        ChildResponse(
            turn_id=initial.turn_id,
            response_id=uuid4(),
            type=ResponseType.TEXT,
            text="말로 설명하기 어려워",
        ),
        initial.mormi.text,
    )

    assert classified.response_category is ResponseCategory.EXPRESSION_BLOCK
    assert next_state.expression_level is ExpressionLevel.L0
    assert next_state.hint_level is HintLevel.H3
    assert next_state.task_max_hint is HintLevel.H3
    assert turn.input.kind is InputKind.JOINT
    assert turn.help_card is not None
    assert turn.help_card.level is HintLevel.H3
    assert turn.help_card.auto_open is True
    assert "같이" in turn.mormi.text


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
